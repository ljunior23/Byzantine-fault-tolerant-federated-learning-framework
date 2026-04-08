import torch
import numpy as np
from typing import List, Optional


def geometric_median_aggregate(
    gradient_lists: List[List[torch.Tensor]],
    f: int = 0,
    weights: Optional[List[int]] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    **kwargs,
) -> List[torch.Tensor]:
    """
    Compute the geometric median of gradients using Weiszfeld's algorithm.

    Args:
        gradient_lists: Per-client gradient lists.
        f: Unused (geometric median is inherently robust for f < n/2).
        weights: Optional client weights.
        max_iter: Maximum Weiszfeld iterations.
        tol: Convergence tolerance.

    Returns:
        Aggregated gradient list (geometric median).
    """
    n = len(gradient_lists)
    if n == 0:
        raise ValueError("No gradients provided.")
    if n == 1:
        return gradient_lists[0]

    # Flatten all gradients
    flat = torch.stack([_flatten(grads) for grads in gradient_lists])  # [n, d]
    w = torch.ones(n, dtype=flat.dtype) / n if weights is None else _normalize(
        torch.tensor(weights, dtype=flat.dtype)
    )

    # Initialize at weighted mean
    median = (flat * w.unsqueeze(1)).sum(0)

    for iteration in range(max_iter):
        prev_median = median.clone()

        dists = torch.norm(flat - median.unsqueeze(0), dim=1)  # [n]
        dists = torch.clamp(dists, min=1e-8)

        # Weiszfeld weights
        inv_dists = w / dists
        new_median = (flat * inv_dists.unsqueeze(1)).sum(0) / inv_dists.sum()

        # Check convergence
        if torch.norm(new_median - prev_median) < tol:
            break
        median = new_median

    # Unflatten back to per-layer structure
    return _unflatten(median, gradient_lists[0])


def _flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.detach().view(-1) for g in grads])


def _unflatten(flat: torch.Tensor, template: List[torch.Tensor]) -> List[torch.Tensor]:
    result = []
    idx = 0
    for t in template:
        numel = t.numel()
        result.append(flat[idx: idx + numel].view(t.shape))
        idx += numel
    return result


def _normalize(w: torch.Tensor) -> torch.Tensor:
    return w / w.sum()
