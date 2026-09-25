from __future__ import annotations

import torch
from torch.nn import functional as F


def diversity_loss(experts: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    normalized = experts / experts.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    similarities = torch.einsum("bid,bjd->bij", normalized, normalized)
    pairs = torch.stack(
        [similarities[:, 0, 1], similarities[:, 0, 2], similarities[:, 1, 2]],
        dim=-1,
    )
    return pairs.square().mean()


def routing_loss(
    routing: torch.Tensor, gamma: float = 1.0, epsilon: float = 1e-12
) -> torch.Tensor:
    query_entropy = -(routing * torch.log(routing + epsilon)).sum(dim=-1).mean()
    mean_utilization = routing.mean(dim=0)
    batch_entropy = -(
        mean_utilization * torch.log(mean_utilization + epsilon)
    ).sum()
    return query_entropy - gamma * batch_entropy


def total_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    label_smoothing: float,
    lambda_diversity: float,
    lambda_routing: float,
    routing_gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    forecast = F.cross_entropy(
        outputs["scores"], targets, label_smoothing=label_smoothing
    )
    diversity = diversity_loss(outputs["experts"])
    routing = routing_loss(outputs["routing"], gamma=routing_gamma)
    total = forecast + lambda_diversity * diversity + lambda_routing * routing
    return total, {
        "forecast": float(forecast.detach()),
        "diversity": float(diversity.detach()),
        "routing": float(routing.detach()),
        "total": float(total.detach()),
    }

