from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_json


def load_facts(path: str | Path) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing split file: {source}")
    facts = np.loadtxt(source, dtype=np.int64, delimiter="\t")
    if facts.size == 0:
        return np.empty((0, 4), dtype=np.int64)
    facts = np.atleast_2d(facts)
    if facts.shape[1] != 4:
        raise ValueError(f"Expected four columns in {source}, found {facts.shape[1]}")
    return facts


class TemporalFactDataset(Dataset):
    def __init__(self, facts: np.ndarray):
        self.facts = torch.as_tensor(facts, dtype=torch.long)

    def __len__(self) -> int:
        return self.facts.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.facts[index]


class SemanticCache:
    def __init__(
        self,
        path: str | Path | None,
        semantic_dim: int,
        allow_missing: bool = False,
    ) -> None:
        self.semantic_dim = semantic_dim
        self.lookup: dict[tuple[int, int], torch.Tensor] = {}
        if path is None or not Path(path).exists():
            if not allow_missing:
                raise FileNotFoundError(
                    f"Semantic cache not found at {path}. Run scripts/cache_text_features.py "
                    "or set data.allow_missing_semantics=true for diagnostics only."
                )
            return

        payload = torch.load(path, map_location="cpu", weights_only=False)
        keys = payload["keys"].long()
        vectors = payload["vectors"].float()
        if vectors.ndim != 2 or vectors.shape[1] != semantic_dim:
            raise ValueError(
                f"Semantic cache has shape {tuple(vectors.shape)}; expected (*, {semantic_dim})"
            )
        self.lookup = {
            (int(key[0]), int(key[1])): vector
            for key, vector in zip(keys.tolist(), vectors)
        }

    def get(self, subject: int, relation: int) -> torch.Tensor:
        vector = self.lookup.get((subject, relation))
        if vector is None:
            return torch.zeros(self.semantic_dim, dtype=torch.float32)
        return vector


class HistoryIndex:
    """Leakage-safe temporal histories and time-aware filter sets."""

    def __init__(self, facts: np.ndarray, train_facts: np.ndarray, num_relations: int):
        self.num_relations = num_relations
        self.pair_events: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self.outgoing_events: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        self.true_objects: dict[tuple[int, int, int], set[int]] = defaultdict(set)

        for subject, relation, obj, timestamp in facts.tolist():
            values = int(subject), int(relation), int(obj), int(timestamp)
            subject, relation, obj, timestamp = values
            self.pair_events[(subject, relation)].append((timestamp, obj))
            self.outgoing_events[subject].append((timestamp, relation, obj))
            self.outgoing_events[obj].append(
                (timestamp, relation + num_relations, subject)
            )
            self.true_objects[(subject, relation, timestamp)].add(obj)

        for events in self.pair_events.values():
            events.sort()
        for events in self.outgoing_events.values():
            events.sort()

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        degree_counts: dict[int, int] = defaultdict(int)
        for subject, relation, obj, _ in train_facts.tolist():
            pair_counts[(int(subject), int(relation))] += 1
            degree_counts[int(subject)] += 1
            degree_counts[int(obj)] += 1
        self.h_max = max(pair_counts.values(), default=1)
        self.d_max = max(degree_counts.values(), default=1)

    @staticmethod
    def _prefix(events: list[tuple], timestamp: int) -> list[tuple]:
        times = [event[0] for event in events]
        return events[: bisect_left(times, timestamp)]

    def pair_history(
        self, subject: int, relation: int, timestamp: int, length: int
    ) -> list[tuple[int, int]]:
        events = self.pair_events.get((subject, relation), [])
        return self._prefix(events, timestamp)[-length:]

    def structural_history(
        self,
        subject: int,
        timestamp: int,
        snapshots: int,
        per_relation: int,
    ) -> list[tuple[int, int, int]]:
        prefix = self._prefix(self.outgoing_events.get(subject, []), timestamp)
        if not prefix:
            return []
        selected_times = sorted({event[0] for event in prefix})[-snapshots:]
        selected_time_set = set(selected_times)
        grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for event_time, relation, neighbor in prefix:
            if event_time in selected_time_set:
                grouped[relation].append((event_time, relation, neighbor))
        selected: list[tuple[int, int, int]] = []
        for relation in sorted(grouped):
            selected.extend(grouped[relation][-per_relation:])
        selected.sort()
        return selected

    def query_statistics(
        self, subject: int, relation: int, timestamp: int, recency_tau: float
    ) -> tuple[float, float, float]:
        structural = self._prefix(self.outgoing_events.get(subject, []), timestamp)
        pair = self._prefix(self.pair_events.get((subject, relation), []), timestamp)
        degree = np.log1p(len(structural)) / np.log1p(max(self.d_max, 1))
        history = min(len(pair) / max(self.h_max, 1), 1.0)
        recency = 0.0
        if pair:
            recency = float(np.exp(-(timestamp - pair[-1][0]) / recency_tau))
        return float(min(degree, 1.0)), float(history), recency

    def filtered_objects(self, subject: int, relation: int, timestamp: int) -> set[int]:
        return self.true_objects.get((subject, relation, timestamp), set())


class TemporalCollator:
    def __init__(
        self,
        history: HistoryIndex,
        semantics: SemanticCache,
        graph_snapshots: int,
        neighbors_per_relation: int,
        sequence_length: int,
        recency_tau: float,
    ) -> None:
        self.history = history
        self.semantics = semantics
        self.graph_snapshots = graph_snapshots
        self.neighbors_per_relation = neighbors_per_relation
        self.sequence_length = sequence_length
        self.recency_tau = recency_tau

    def __call__(self, rows: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
        facts = torch.stack(list(rows), dim=0).long()
        subjects, relations, objects, timestamps = facts.unbind(dim=1)

        structural_rows: list[list[tuple[int, int, int]]] = []
        sequence_rows: list[list[tuple[int, int]]] = []
        statistics = []
        semantic_vectors = []
        for subject, relation, timestamp in zip(
            subjects.tolist(), relations.tolist(), timestamps.tolist()
        ):
            structural_rows.append(
                self.history.structural_history(
                    subject,
                    timestamp,
                    self.graph_snapshots,
                    self.neighbors_per_relation,
                )
            )
            sequence_rows.append(
                self.history.pair_history(
                    subject, relation, timestamp, self.sequence_length
                )
            )
            statistics.append(
                self.history.query_statistics(
                    subject, relation, timestamp, self.recency_tau
                )
            )
            semantic_vectors.append(self.semantics.get(subject, relation))

        max_neighbors = max(1, max(map(len, structural_rows), default=0))
        batch_size = facts.shape[0]
        neighbor_ids = torch.zeros((batch_size, max_neighbors), dtype=torch.long)
        neighbor_relations = torch.zeros((batch_size, max_neighbors), dtype=torch.long)
        neighbor_deltas = torch.zeros((batch_size, max_neighbors), dtype=torch.float32)
        neighbor_mask = torch.zeros((batch_size, max_neighbors), dtype=torch.bool)

        sequence_objects = torch.zeros(
            (batch_size, self.sequence_length), dtype=torch.long
        )
        sequence_deltas = torch.zeros(
            (batch_size, self.sequence_length), dtype=torch.float32
        )
        sequence_mask = torch.zeros(
            (batch_size, self.sequence_length), dtype=torch.bool
        )

        for row_index, (structural, sequence, timestamp) in enumerate(
            zip(structural_rows, sequence_rows, timestamps.tolist())
        ):
            for column, (event_time, relation, neighbor) in enumerate(structural):
                neighbor_ids[row_index, column] = neighbor
                neighbor_relations[row_index, column] = relation
                neighbor_deltas[row_index, column] = timestamp - event_time
                neighbor_mask[row_index, column] = True
            offset = self.sequence_length - len(sequence)
            for column, (event_time, obj) in enumerate(sequence, start=offset):
                sequence_objects[row_index, column] = obj
                sequence_deltas[row_index, column] = timestamp - event_time
                sequence_mask[row_index, column] = True

        return {
            "subjects": subjects,
            "relations": relations,
            "targets": objects,
            "timestamps": timestamps,
            "neighbor_ids": neighbor_ids,
            "neighbor_relations": neighbor_relations,
            "neighbor_deltas": neighbor_deltas,
            "neighbor_mask": neighbor_mask,
            "sequence_objects": sequence_objects,
            "sequence_deltas": sequence_deltas,
            "sequence_mask": sequence_mask,
            "query_statistics": torch.tensor(statistics, dtype=torch.float32),
            "semantic_vectors": torch.stack(semantic_vectors, dim=0),
        }


def load_dataset_bundle(data_dir: str | Path) -> dict[str, object]:
    root = Path(data_dir)
    metadata = load_json(root / "metadata.json")
    train = load_facts(root / "train.tsv")
    valid = load_facts(root / "valid.tsv")
    test = load_facts(root / "test.tsv")
    all_facts = np.concatenate([train, valid, test], axis=0)
    history = HistoryIndex(all_facts, train, int(metadata["num_relations"]))
    return {
        "metadata": metadata,
        "train": train,
        "valid": valid,
        "test": test,
        "history": history,
    }

