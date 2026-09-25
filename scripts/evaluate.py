#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hmoe_tkg.config import load_config
from hmoe_tkg.metrics import evaluate_model
from hmoe_tkg.trainer import load_checkpoint_runtime
from hmoe_tkg.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = load_checkpoint_runtime(config, args.checkpoint)
    metrics, _ = evaluate_model(
        runtime["model"],
        runtime["loaders"][args.split],
        runtime["bundle"]["history"],
        runtime["device"],
    )
    if args.output:
        save_json(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

