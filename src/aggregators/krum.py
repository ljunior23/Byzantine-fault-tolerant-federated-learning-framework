import torch
import numpy as np
from typing import List, Optional


def krum_aggregate(
    gradient_lists: List[List[torch.Tensor]],
    f: int = 0,
    weights: Optional[List[int]] = None,
    multi_krum_m: int = 1,
    **kwargs,
) -> List[torch.Tensor]:
    """
    Krum / Multi-Krum aggregation.

    Args:
        gradient_lists: List of per-client gradient lists.
        f: Number of suspected Byzantine clients.
        weights: Sample counts (used in Multi-Krum averaging).
        multi_krum_m: Number of top-k updates to average (1 = standard Krum).

    Returns:
        Aggregated gradient list.
    """
    n = len(gradient_lists)
    if n == 0:
        raise ValueError("No gradients provided to Krum.")

    f = min(f, (n - 2) // 2)  # Safety clamp
    k = n - f - 2  # Neighbors to consider

    # Flatten each client's gradients into a single vector
    flat = [_flatten(grads) for grads in gradient_lists]

    # Compute pairwise squared distances
    dist_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sum((flat[i] - flat[j]) ** 2).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Krum score = sum of k smallest distances for each client
    scores = []
    for i in range(n):
        dists = sorted([dist_matrix[i, j].item() for j in range(n) if j != i])
        scores.append(sum(dists[:k]))

    # Select top multi_krum_m clients with lowest scores
    selected_indices = sorted(range(n), key=lambda i: scores[i])[:multi_krum_m]

    if multi_krum_m == 1:
        return gradient_lists[selected_indices[0]]

    # Multi-Krum: weighted average of selected clients
    selected_weights = (
        [weights[i] for i in selected_indices] if weights else [1] * multi_krum_m
    )
    total_weight = sum(selected_weights)

    avg = [torch.zeros_like(g) for g in gradient_lists[0]]
    for idx, w in zip(selected_indices, selected_weights):
        for i, g in enumerate(gradient_lists[idx]):
            avg[i] += g * (w / total_weight)

    return avg


def _flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.view(-1) for g in grads])
