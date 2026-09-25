from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TimeEncoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.network(torch.log1p(delta.float()).unsqueeze(-1))


class RelationMessageLayer(nn.Module):
    def __init__(self, dimension: int, num_relations: int, num_bases: int):
        super().__init__()
        self.bases = nn.Parameter(torch.empty(num_bases, 2 * dimension, dimension))
        self.coefficients = nn.Parameter(torch.empty(num_relations, num_bases))
        self.self_projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def relation_weights(self) -> torch.Tensor:
        return torch.einsum("rb,bio->rio", self.coefficients, self.bases)

    def messages(
        self, node_states: torch.Tensor, time_states: torch.Tensor, relations: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([node_states, time_states], dim=-1)
        weights = self.relation_weights()[relations]
        return torch.einsum("...i,...io->...o", inputs, weights)


class TorchStructuralExpert(nn.Module):
    def __init__(
        self, dimension: int, num_relations: int, layers: int, num_bases: int, dropout: float
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(dimension)
        self.layers = nn.ModuleList(
            RelationMessageLayer(dimension, num_relations, num_bases)
            for _ in range(layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        subject_states: torch.Tensor,
        neighbor_states: torch.Tensor,
        relations: torch.Tensor,
        deltas: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        root = subject_states
        neighbors = neighbor_states
        time_states = self.time_encoder(deltas)
        float_mask = mask.unsqueeze(-1).to(root.dtype)
        denominator = float_mask.sum(dim=1).clamp_min(1.0)
        for layer in self.layers:
            messages = layer.messages(neighbors, time_states, relations) * float_mask
            aggregate = messages.sum(dim=1) / denominator
            root = F.relu(aggregate + layer.self_projection(root))
            neighbors = F.relu(layer.self_projection(neighbors))
            root = self.dropout(root)
        return root


class DGLStructuralExpert(TorchStructuralExpert):
    """Equivalent relation-aware message passing executed through DGL."""

    def forward(
        self,
        subject_states: torch.Tensor,
        neighbor_states: torch.Tensor,
        relations: torch.Tensor,
        deltas: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        try:
            import dgl
            import dgl.function as fn
        except ImportError as error:
            raise RuntimeError(
                "The selected structural backend is 'dgl', but DGL is not installed. "
                "Install the CUDA-compatible DGL wheel or set model.structural_backend=torch."
            ) from error

        batch_size, width, dimension = neighbor_states.shape
        flat_mask = mask.reshape(-1)
        edge_slots = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        if edge_slots.numel() == 0:
            root = subject_states
            for layer in self.layers:
                root = self.dropout(F.relu(layer.self_projection(root)))
            return root

        root_nodes = torch.div(edge_slots, width, rounding_mode="floor")
        source_nodes = batch_size + edge_slots
        graph = dgl.graph(
            (source_nodes, root_nodes),
            num_nodes=batch_size + batch_size * width,
            device=subject_states.device,
        )
        flat_neighbors = neighbor_states.reshape(-1, dimension)
        states = torch.cat([subject_states, flat_neighbors], dim=0)
        edge_relations = relations.reshape(-1)[edge_slots]
        edge_times = self.time_encoder(deltas.reshape(-1)[edge_slots])

        for layer in self.layers:
            messages = layer.messages(
                states[source_nodes], edge_times, edge_relations
            )
            graph.edata["message"] = messages
            graph.update_all(fn.copy_e("message", "message"), fn.mean("message", "aggregate"))
            aggregate = torch.zeros_like(states)
            aggregate[:batch_size] = graph.ndata["aggregate"][:batch_size]
            states = F.relu(aggregate + layer.self_projection(states))
            states = self.dropout(states)
        return states[:batch_size]


class TemporalSequenceExpert(nn.Module):
    def __init__(
        self,
        dimension: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_encoder = TimeEncoder(dimension)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=4 * dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.no_history = nn.Parameter(torch.empty(dimension))
        nn.init.normal_(self.no_history, std=0.02)

    def forward(
        self,
        object_states: torch.Tensor,
        relation_states: torch.Tensor,
        deltas: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence = object_states + relation_states.unsqueeze(1) + self.time_encoder(deltas)
        mask = mask.clone()
        empty = ~mask.any(dim=1)
        if empty.any():
            sequence[empty, -1] = self.no_history
            mask[empty, -1] = True
        length = sequence.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )
        encoded = self.transformer(
            sequence,
            mask=causal_mask,
            src_key_padding_mask=~mask,
        )
        return encoded[:, -1]


class SemanticExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.adapter(vectors)


class QueryRouter(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.network = nn.Sequential(
            nn.Linear(2 * dimension + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        subject_states: torch.Tensor,
        relation_states: torch.Tensor,
        statistics: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([subject_states, relation_states, statistics], dim=-1)
        return F.softmax(self.network(features) / self.temperature, dim=-1)


class HMoETKG(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        semantic_dim: int = 768,
        embedding_dim: int = 200,
        dropout: float = 0.2,
        graph_layers: int = 2,
        graph_bases: int = 8,
        structural_backend: str = "dgl",
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        semantic_hidden_dim: int = 256,
        router_hidden_dim: int = 128,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(2 * num_relations, embedding_dim)
        structural_class = (
            DGLStructuralExpert if structural_backend.lower() == "dgl" else TorchStructuralExpert
        )
        self.structural_expert = structural_class(
            embedding_dim,
            2 * num_relations,
            graph_layers,
            graph_bases,
            dropout,
        )
        self.temporal_expert = TemporalSequenceExpert(
            embedding_dim, transformer_layers, transformer_heads, dropout
        )
        self.semantic_expert = SemanticExpert(
            semantic_dim, semantic_hidden_dim, embedding_dim, dropout
        )
        self.router = QueryRouter(
            embedding_dim, router_hidden_dim, router_temperature
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        subjects = batch["subjects"]
        relations = batch["relations"]
        subject_states = self.entity_embeddings(subjects)
        relation_states = self.relation_embeddings(relations)
        neighbor_states = self.entity_embeddings(batch["neighbor_ids"])
        object_states = self.entity_embeddings(batch["sequence_objects"])

        structural = self.structural_expert(
            subject_states,
            neighbor_states,
            batch["neighbor_relations"],
            batch["neighbor_deltas"],
            batch["neighbor_mask"],
        )
        temporal = self.temporal_expert(
            object_states,
            relation_states,
            batch["sequence_deltas"],
            batch["sequence_mask"],
        )
        semantic = self.semantic_expert(batch["semantic_vectors"])
        expert_states = torch.stack([structural, temporal, semantic], dim=1)
        routing = self.router(
            subject_states, relation_states, batch["query_statistics"]
        )
        query = torch.sum(routing.unsqueeze(-1) * expert_states, dim=1)
        scores = torch.einsum(
            "bd,bd,nd->bn",
            query,
            relation_states,
            self.entity_embeddings.weight,
        )
        return {
            "scores": scores,
            "query": query,
            "routing": routing,
            "experts": expert_states,
        }

    @classmethod
    def from_config(
        cls, metadata: dict[str, Any], config: dict[str, Any]
    ) -> "HMoETKG":
        data = config["data"]
        model = config["model"]
        return cls(
            num_entities=int(metadata["num_entities"]),
            num_relations=int(metadata["num_relations"]),
            semantic_dim=int(data["semantic_dim"]),
            **model,
        )

