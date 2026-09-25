#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hmoe_tkg.config import load_config
from hmoe_tkg.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train(load_config(args.config), args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

