#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def load_pairs(data_dir: Path) -> list[tuple[int, int]]:
    pairs = set()
    for split in ("train", "valid", "test"):
        with (data_dir / f"{split}.tsv").open("r", encoding="utf-8") as handle:
            for line in handle:
                subject, relation, _, _ = map(int, line.split())
                pairs.add((subject, relation))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    entities = json.loads((data_dir / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((data_dir / "relations.json").read_text(encoding="utf-8"))
    entity_labels = entities["id_to_label"]
    relation_labels = [label.replace("_", " ") for label in relations["id_to_label"]]
    pairs = load_pairs(data_dir)
    texts = [f"{entity_labels[subject]} [SEP] {relation_labels[relation]}" for subject, relation in pairs]

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModel.from_pretrained(
        args.model, local_files_only=args.local_files_only
    ).to(device)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[:, 0]
            vectors.append(hidden.float().cpu())
    payload = {
        "keys": torch.tensor(pairs, dtype=torch.long),
        "vectors": torch.cat(vectors, dim=0),
        "model": args.model,
        "max_length": args.max_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {len(pairs)} semantic vectors to {args.output}")


if __name__ == "__main__":
    main()

