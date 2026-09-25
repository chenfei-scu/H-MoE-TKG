#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 21, 42, 2026])
    args = parser.parse_args()
    train_script = Path(__file__).with_name("train.py")
    for seed in args.seeds:
        subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--config",
                args.config,
                "--seed",
                str(seed),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

