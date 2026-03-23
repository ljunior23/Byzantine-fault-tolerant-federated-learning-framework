"""
Aggregation helper for combining accepted client updates in federated learning.
"""

from typing import Dict, List
import torch
import torch.nn as nn


def aggregate_updates(
    global_model: nn.Module,
    client_updates: Dict[int, List[torch.Tensor]],
    client_sizes: Dict[int, int],
) -> None:
    """
    Apply weighted aggregation to the client updates that passed detection.

    Each client update is stored as:
        global_params - local_params

    Because of that, the weighted average update is subtracted from the
    current global model.
    """
    if not client_updates:
        return

    total_samples = sum(client_sizes[cid] for cid in client_updates.keys())

    with torch.no_grad():
        params = list(global_model.parameters())

        for param_idx, param in enumerate(params):
            aggregated_delta = None

            for cid, update in client_updates.items():
                weight = client_sizes[cid] / total_samples
                contribution = update[param_idx].to(param.device) * weight

                if aggregated_delta is None:
                    aggregated_delta = contribution
                else:
                    aggregated_delta += contribution

            param.data -= aggregated_delta