#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected four fields")
            rows.append(tuple(fields))
    return rows


def load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier, label = line.rstrip("\n").split("\t", maxsplit=1)
            labels[identifier] = label
    return labels


def timestamp_key(value: str) -> tuple[int, object]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity-labels", type=Path)
    parser.add_argument("--relation-labels", type=Path)
    parser.add_argument("--allow-future-vocabulary", action="store_true")
    args = parser.parse_args()

    raw = {
        "train": read_rows(args.train),
        "valid": read_rows(args.valid),
        "test": read_rows(args.test),
    }
    train_entities = sorted(
        {row[0] for row in raw["train"]} | {row[2] for row in raw["train"]}
    )
    train_relations = sorted({row[1] for row in raw["train"]})
    if args.allow_future_vocabulary:
        train_entities = sorted(
            {field for rows in raw.values() for row in rows for field in (row[0], row[2])}
        )
        train_relations = sorted({row[1] for rows in raw.values() for row in rows})
    entity_to_id = {value: index for index, value in enumerate(train_entities)}
    relation_to_id = {value: index for index, value in enumerate(train_relations)}
    all_timestamps = sorted(
        {row[3] for rows in raw.values() for row in rows}, key=timestamp_key
    )
    timestamp_to_id = {value: index for index, value in enumerate(all_timestamps)}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_sizes = {}
    for split, rows in raw.items():
        converted = set()
        for subject, relation, obj, timestamp in rows:
            if subject not in entity_to_id or obj not in entity_to_id:
                raise ValueError(
                    f"{split} contains an entity absent from train: {subject!r} or {obj!r}. "
                    "Use --allow-future-vocabulary only if this is intentional."
                )
            if relation not in relation_to_id:
                raise ValueError(
                    f"{split} contains a relation absent from train: {relation!r}"
                )
            converted.add(
                (
                    entity_to_id[subject],
                    relation_to_id[relation],
                    entity_to_id[obj],
                    timestamp_to_id[timestamp],
                )
            )
        ordered = sorted(converted, key=lambda row: (row[3], row[0], row[1], row[2]))
        with (output / f"{split}.tsv").open("w", encoding="utf-8") as handle:
            for row in ordered:
                handle.write("\t".join(map(str, row)) + "\n")
        split_sizes[split] = len(ordered)

    entity_labels = load_labels(args.entity_labels)
    relation_labels = load_labels(args.relation_labels)
    entities_payload = {
        "label_to_id": entity_to_id,
        "id_to_label": [entity_labels.get(value, value) for value in train_entities],
        "source_identifiers": train_entities,
    }
    relations_payload = {
        "label_to_id": relation_to_id,
        "id_to_label": [relation_labels.get(value, value) for value in train_relations],
        "source_identifiers": train_relations,
    }
    metadata = {
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "num_timestamps": len(timestamp_to_id),
        "split_sizes": split_sizes,
        "timestamp_values": all_timestamps,
    }
    for name, payload in (
        ("entities.json", entities_payload),
        ("relations.json", relations_payload),
        ("metadata.json", metadata),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

