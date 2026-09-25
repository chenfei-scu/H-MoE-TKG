from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "data": {
        "data_dir": "data/ICEWS14/processed",
        "semantic_cache": "data/ICEWS14/semantic_cache.pt",
        "allow_missing_semantics": False,
        "semantic_dim": 768,
        "graph_snapshots": 4,
        "neighbors_per_relation": 10,
        "sequence_length": 20,
        "recency_tau": 30.0,
    },
    "model": {
        "embedding_dim": 200,
        "dropout": 0.2,
        "graph_layers": 2,
        "graph_bases": 8,
        "structural_backend": "dgl",
        "transformer_layers": 2,
        "transformer_heads": 4,
        "semantic_hidden_dim": 256,
        "router_hidden_dim": 128,
        "router_temperature": 1.0,
    },
    "training": {
        "batch_size": 512,
        "eval_batch_size": 256,
        "max_epochs": 200,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "label_smoothing": 0.1,
        "lambda_diversity": 0.05,
        "lambda_routing": 0.01,
        "routing_gamma": 1.0,
        "gradient_clip": 1.0,
        "patience": 10,
        "minimum_improvement": 0.0001,
        "num_workers": 0,
        "device": "auto",
        "output_dir": "outputs/icews14",
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = _merge(DEFAULTS, user_config)
    config["config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if key != "config_path"}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)

