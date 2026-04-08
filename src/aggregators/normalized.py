import torch
from typing import List, Optional


def normalized_aggregate(
    gradient_lists: List[List[torch.Tensor]],
    f: int = 0,
    weights: Optional[List[int]] = None,
    clip_threshold: float = 1.0,
    **kwargs,
) -> List[torch.Tensor]:
    """
    Aggregate by clipping each client's global gradient norm, then averaging.

    Args:
        gradient_lists: Per-client gradient lists.
        f: Unused directly (clipping handles Byzantine influence).
        weights: Sample counts per client.
        clip_threshold: Max allowed gradient norm.

    Returns:
        Aggregated gradient list.
    """
    n = len(gradient_lists)
    if n == 0:
        raise ValueError("No gradients provided.")

    weights = weights or [1] * n
    total_weight = sum(weights)
    clipped = []

    for grads, w in zip(gradient_lists, weights):
        flat = torch.cat([g.view(-1) for g in grads])
        norm = torch.norm(flat).item()
        scale = min(1.0, clip_threshold / (norm + 1e-8))

        clipped.append(([g * scale for g in grads], w))

    # Weighted average of clipped gradients
    avg = [torch.zeros_like(g) for g in gradient_lists[0]]
    for clipped_grads, w in clipped:
        for i, g in enumerate(clipped_grads):
            avg[i] += g * (w / total_weight)

    return avg
