#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


METRICS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_10")


def expand(patterns: list[str]) -> list[Path]:
    return sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})


def load(paths: list[Path]) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--baseline", nargs="+")
    args = parser.parse_args()
    run_paths = expand(args.runs)
    if not run_paths:
        raise SystemExit("No run files matched")
    runs = load(run_paths)
    summary = {"seeds": [row.get("seed") for row in runs], "metrics": {}}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in runs], dtype=float)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    if args.baseline:
        baseline = load(expand(args.baseline))
        baseline_by_seed = {row.get("seed"): row for row in baseline}
        paired = [(row, baseline_by_seed[row.get("seed")]) for row in runs if row.get("seed") in baseline_by_seed]
        if len(paired) < 2:
            raise SystemExit("At least two seed-matched baseline runs are required")
        summary["paired_two_sided_p"] = {
            metric: float(
                ttest_rel(
                    [row[metric] for row, _ in paired],
                    [base[metric] for _, base in paired],
                ).pvalue
            )
            for metric in METRICS
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

