import torch
from typing import List, Optional


def trimmed_mean_aggregate(
    gradient_lists: List[List[torch.Tensor]],
    f: int = 0,
    weights: Optional[List[int]] = None,
    **kwargs,
) -> List[torch.Tensor]:
    """
    Coordinate-wise trimmed mean: removes the top and bottom f values
    at each coordinate before averaging.

    Args:
        gradient_lists: Per-client gradient lists.
        f: Number of values to trim from each end per coordinate.
        weights: Ignored (trimmed mean doesn't use weighted averaging).

    Returns:
        Aggregated gradient list.
    """
    n = len(gradient_lists)
    if n == 0:
        raise ValueError("No gradients provided.")

    beta = min(f, (n - 1) // 2)  # Safety clamp; must trim < n/2

    result = []
    for layer_idx in range(len(gradient_lists[0])):
        # Stack this layer across all clients: shape [n, ...]
        stacked = torch.stack([gradient_lists[c][layer_idx] for c in range(n)], dim=0)
        shape = stacked.shape[1:]
        flat = stacked.view(n, -1)  # [n, d]

        if beta == 0:
            # No trimming - standard mean
            trimmed = flat.mean(dim=0)
        else:
            # Sort along client dimension and trim
            sorted_flat, _ = torch.sort(flat, dim=0)
            trimmed_flat = sorted_flat[beta: n - beta]  # [n - 2*beta, d]
            trimmed = trimmed_flat.mean(dim=0)

        result.append(trimmed.view(shape))

    return result
