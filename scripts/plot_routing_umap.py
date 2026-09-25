#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = np.load(args.weights)
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError("Expected routing weights with shape (queries, 3)")
    projection = UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    ).fit_transform(weights)
    dominant = weights.argmax(axis=1)
    labels = ["structural", "temporal", "semantic"]
    colors = ["#3274A1", "#8C5DA6", "#E1812C"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        selected = dominant == index
        plt.scatter(projection[selected, 0], projection[selected, 1], s=9, alpha=0.7, label=label, color=color)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()

