from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

from .data import HistoryIndex
from .utils import move_batch


def filtered_ranks(
    scores: torch.Tensor,
    subjects: torch.Tensor,
    relations: torch.Tensor,
    targets: torch.Tensor,
    timestamps: torch.Tensor,
    history: HistoryIndex,
) -> torch.Tensor:
    filtered = scores.clone()
    for row, (subject, relation, target, timestamp) in enumerate(
        zip(
            subjects.tolist(),
            relations.tolist(),
            targets.tolist(),
            timestamps.tolist(),
        )
    ):
        target_score = filtered[row, target].clone()
        other_true = history.filtered_objects(subject, relation, timestamp) - {target}
        if other_true:
            indices = torch.tensor(
                sorted(other_true), dtype=torch.long, device=filtered.device
            )
            filtered[row, indices] = -torch.inf
        filtered[row, target] = target_score
    target_scores = filtered.gather(1, targets.unsqueeze(1))
    return 1 + (filtered > target_scores).sum(dim=1)


def metrics_from_ranks(ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.float()
    return {
        "mrr": float((1.0 / ranks).mean()),
        "hits_at_1": float((ranks <= 1).float().mean()),
        "hits_at_3": float((ranks <= 3).float().mean()),
        "hits_at_10": float((ranks <= 10).float().mean()),
        "count": int(ranks.numel()),
    }


def history_category(history: HistoryIndex, subject: int, relation: int, timestamp: int) -> str:
    count = len(history.pair_history(subject, relation, timestamp, length=10**9))
    if count == 0:
        return "cold_start"
    if count <= 2:
        return "sparse_history"
    return "history_rich"


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    history: HistoryIndex,
    device: torch.device,
    collect_routing: bool = False,
) -> tuple[dict[str, object], np.ndarray | None]:
    model.eval()
    all_ranks: list[torch.Tensor] = []
    categories: dict[str, list[int]] = defaultdict(list)
    routing_rows: list[np.ndarray] = []
    for batch in loader:
        host_subjects = batch["subjects"]
        host_relations = batch["relations"]
        host_targets = batch["targets"]
        host_timestamps = batch["timestamps"]
        device_batch = move_batch(batch, device)
        outputs = model(device_batch)
        ranks = filtered_ranks(
            outputs["scores"],
            device_batch["subjects"],
            device_batch["relations"],
            device_batch["targets"],
            device_batch["timestamps"],
            history,
        ).cpu()
        all_ranks.append(ranks)
        for rank, subject, relation, timestamp in zip(
            ranks.tolist(),
            host_subjects.tolist(),
            host_relations.tolist(),
            host_timestamps.tolist(),
        ):
            categories[history_category(history, subject, relation, timestamp)].append(rank)
        if collect_routing:
            routing_rows.append(outputs["routing"].detach().cpu().numpy())

    ranks = torch.cat(all_ranks) if all_ranks else torch.empty(0, dtype=torch.long)
    overall = metrics_from_ranks(ranks)
    overall["by_history"] = {
        name: metrics_from_ranks(torch.tensor(values, dtype=torch.long))
        for name, values in categories.items()
    }
    routing = np.concatenate(routing_rows, axis=0) if routing_rows else None
    return overall, routing

