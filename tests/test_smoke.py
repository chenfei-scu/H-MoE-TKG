from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hmoe_tkg.config import load_config
from hmoe_tkg.data import HistoryIndex
from hmoe_tkg.metrics import filtered_ranks
from hmoe_tkg.trainer import train


class SmokeTest(unittest.TestCase):
    def make_data(self, root: Path) -> Path:
        data_dir = root / "processed"
        data_dir.mkdir()
        splits = {
            "train": [
                (0, 0, 1, 0),
                (1, 1, 2, 1),
                (0, 0, 2, 2),
                (2, 1, 3, 3),
                (0, 1, 4, 4),
                (4, 0, 5, 5),
            ],
            "valid": [(0, 0, 3, 6), (1, 1, 4, 7)],
            "test": [(0, 0, 4, 8), (2, 1, 5, 9)],
        }
        for split, rows in splits.items():
            with (data_dir / f"{split}.tsv").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write("\t".join(map(str, row)) + "\n")
        (data_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "num_entities": 6,
                    "num_relations": 2,
                    "num_timestamps": 10,
                    "split_sizes": {key: len(value) for key, value in splits.items()},
                }
            ),
            encoding="utf-8",
        )
        return data_dir

    def test_training_and_filtered_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = self.make_data(root)
            config_path = root / "smoke.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "data_dir": str(data_dir),
                            "semantic_cache": str(root / "missing.pt"),
                            "allow_missing_semantics": True,
                            "semantic_dim": 16,
                            "graph_snapshots": 2,
                            "neighbors_per_relation": 2,
                            "sequence_length": 4,
                            "recency_tau": 3.0,
                        },
                        "model": {
                            "embedding_dim": 16,
                            "dropout": 0.0,
                            "graph_layers": 1,
                            "graph_bases": 2,
                            "structural_backend": "torch",
                            "transformer_layers": 1,
                            "transformer_heads": 4,
                            "semantic_hidden_dim": 16,
                            "router_hidden_dim": 8,
                            "router_temperature": 1.0,
                        },
                        "training": {
                            "batch_size": 3,
                            "eval_batch_size": 2,
                            "max_epochs": 2,
                            "learning_rate": 0.005,
                            "weight_decay": 0.0,
                            "label_smoothing": 0.1,
                            "lambda_diversity": 0.05,
                            "lambda_routing": 0.01,
                            "routing_gamma": 1.0,
                            "gradient_clip": 1.0,
                            "patience": 2,
                            "minimum_improvement": 0.0,
                            "num_workers": 0,
                            "device": "cpu",
                            "output_dir": str(root / "outputs"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = train(load_config(config_path), seed=7)
            self.assertEqual(metrics["count"], 2)
            self.assertTrue((root / "outputs" / "seed_7" / "best.pt").exists())
            self.assertGreaterEqual(metrics["mrr"], 0.0)
            self.assertLessEqual(metrics["mrr"], 1.0)

    def test_filter_removes_other_true_objects(self) -> None:
        facts = torch.tensor([(0, 0, 1, 2), (0, 0, 2, 2)]).numpy()
        history = HistoryIndex(facts, facts, num_relations=1)
        scores = torch.tensor([[0.0, 0.8, 0.9, 0.7]])
        rank = filtered_ranks(
            scores,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            history,
        )
        self.assertEqual(rank.item(), 1)


if __name__ == "__main__":
    unittest.main()

