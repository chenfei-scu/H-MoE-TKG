from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import save_config
from .data import (
    SemanticCache,
    TemporalCollator,
    TemporalFactDataset,
    load_dataset_bundle,
)
from .losses import total_loss
from .metrics import evaluate_model
from .model import HMoETKG
from .utils import move_batch, resolve_device, save_json, set_seed


def _project_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = Path(config.get("config_path", Path.cwd() / "config.yaml"))
    project_root = config_path.parent.parent
    return (project_root / path).resolve()


def build_runtime(config: dict[str, Any], seed: int) -> dict[str, Any]:
    set_seed(seed)
    data_config = config["data"]
    training_config = config["training"]
    data_dir = _project_path(config, data_config["data_dir"])
    bundle = load_dataset_bundle(data_dir)
    semantic_path = data_config.get("semantic_cache")
    if semantic_path:
        semantic_path = _project_path(config, semantic_path)
    semantics = SemanticCache(
        semantic_path,
        semantic_dim=int(data_config["semantic_dim"]),
        allow_missing=bool(data_config["allow_missing_semantics"]),
    )
    collator = TemporalCollator(
        history=bundle["history"],
        semantics=semantics,
        graph_snapshots=int(data_config["graph_snapshots"]),
        neighbors_per_relation=int(data_config["neighbors_per_relation"]),
        sequence_length=int(data_config["sequence_length"]),
        recency_tau=float(data_config["recency_tau"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            TemporalFactDataset(bundle["train"]),
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            collate_fn=collator,
            num_workers=int(training_config["num_workers"]),
            generator=generator,
        ),
        "valid": DataLoader(
            TemporalFactDataset(bundle["valid"]),
            batch_size=int(training_config["eval_batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(training_config["num_workers"]),
        ),
        "test": DataLoader(
            TemporalFactDataset(bundle["test"]),
            batch_size=int(training_config["eval_batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(training_config["num_workers"]),
        ),
    }
    device = resolve_device(str(training_config["device"]))
    model = HMoETKG.from_config(bundle["metadata"], config).to(device)
    return {
        "bundle": bundle,
        "loaders": loaders,
        "model": model,
        "device": device,
    }


def train(config: dict[str, Any], seed: int) -> dict[str, Any]:
    runtime = build_runtime(config, seed)
    bundle = runtime["bundle"]
    loaders = runtime["loaders"]
    model = runtime["model"]
    device = runtime["device"]
    settings = config["training"]
    output_root = _project_path(config, settings["output_dir"])
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_mrr = -float("inf")
    best_loss = float("inf")
    stale_epochs = 0
    history_rows: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _ = total_loss(
                outputs,
                batch["targets"],
                label_smoothing=float(settings["label_smoothing"]),
                lambda_diversity=float(settings["lambda_diversity"]),
                lambda_routing=float(settings["lambda_routing"]),
                routing_gamma=float(settings["routing_gamma"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        validation, _ = evaluate_model(
            model, loaders["valid"], bundle["history"], device
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_mrr": float(validation["mrr"]),
        }
        history_rows.append(record)
        improved = validation["mrr"] > best_mrr + float(
            settings["minimum_improvement"]
        ) or (
            abs(validation["mrr"] - best_mrr)
            <= float(settings["minimum_improvement"])
            and train_loss < best_loss
        )
        if improved:
            best_mrr = float(validation["mrr"])
            best_loss = train_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "metadata": bundle["metadata"],
                    "validation": validation,
                    "config": config,
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        save_json(history_rows, output_dir / "training_history.json")
        if stale_epochs >= int(settings["patience"]):
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, routing = evaluate_model(
        model,
        loaders["test"],
        bundle["history"],
        device,
        collect_routing=True,
    )
    test_metrics.update(
        {
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "elapsed_seconds": time.time() - started,
        }
    )
    save_json(test_metrics, output_dir / "test_metrics.json")
    if routing is not None:
        np.save(output_dir / "routing_weights.npy", routing)
    return test_metrics


def load_checkpoint_runtime(
    config: dict[str, Any], checkpoint_path: str | Path, seed: int = 42
) -> dict[str, Any]:
    runtime = build_runtime(config, seed)
    checkpoint = torch.load(
        checkpoint_path, map_location=runtime["device"], weights_only=False
    )
    runtime["model"].load_state_dict(checkpoint["model_state"])
    runtime["checkpoint"] = checkpoint
    return runtime

