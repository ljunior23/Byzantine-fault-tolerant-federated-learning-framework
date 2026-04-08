"""
Federated Learning Client with local training and gradient computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Standard FL client: receives global model, trains locally, returns gradients.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset,
        local_epochs: int = 3,
        batch_size: int = 64,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def set_global_params(self, global_params: List[torch.Tensor]):
        # Move incoming params to this client's device before copying
        self.model.to(self.device)
        with torch.no_grad():
            for param, gp in zip(self.model.parameters(), global_params):
                param.data.copy_(gp.to(self.device))

    def train(self, global_params: List[torch.Tensor]) -> Tuple[List[torch.Tensor], int]:
        """
        Run local training and compute pseudo-gradients (param delta).

        Returns:
            gradients: List of gradient tensors (global - local params).
            num_samples: Number of training samples used.
        """
        self.set_global_params(global_params)
        self.model.train()
        self.model.to(self.device)

        # Save initial (global) params
        initial_params = [p.data.clone() for p in self.model.parameters()]

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        num_samples = 0
        for epoch in range(self.local_epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                num_samples += len(inputs)

        # Compute pseudo-gradients (gradient = initial - updated params)
        # Return on CPU so aggregators and detection (NumPy) work on any setup
        with torch.no_grad():
            gradients = [
                (init - updated.data).cpu()
                for init, updated in zip(initial_params, self.model.parameters())
            ]

        logger.debug(
            f"Client {self.client_id}: trained on {num_samples} samples "
            f"over {self.local_epochs} epochs."
        )
        return gradients, len(self.dataset)


class NonIIDPartitioner:
    """
    Partitions a dataset into non-IID shards for simulating heterogeneous clients.
    Uses the Dirichlet distribution to control heterogeneity (lower alpha = more non-IID).
    """

    def __init__(self, dataset, num_clients: int, alpha: float = 0.5, seed: int = 42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.rng = torch.Generator().manual_seed(seed)

    def partition(self) -> List[Subset]:
        import numpy as np
        targets = torch.tensor([self.dataset[i][1] for i in range(len(self.dataset))])
        num_classes = int(targets.max().item()) + 1

        client_indices = [[] for _ in range(self.num_clients)]
        rng = np.random.default_rng(42)

        for cls in range(num_classes):
            cls_indices = (targets == cls).nonzero(as_tuple=True)[0].tolist()
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(cls_indices)).astype(int)

            idx = 0
            for cid, count in enumerate(proportions):
                client_indices[cid].extend(cls_indices[idx: idx + count])
                idx += count

        return [Subset(self.dataset, idxs) for idxs in client_indices]