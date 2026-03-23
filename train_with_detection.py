"""
Simple end-to-end federated learning run with attack injection, detection,
filtering, and aggregation.

This version uses synthetic MNIST-shaped data so the full pipeline can be
tested without downloading a dataset.
"""

import random
from typing import Dict, List

import torch
from torch.utils.data import TensorDataset

from cnn import get_model
from client import FederatedClient
from byzantine_simulator import (
    AttackOrchestrator,
    ByzantineAttackConfig,
    AttackType,
)
from detection_layers import DetectionLayer
from aggregation import aggregate_updates


def evaluate(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


def make_synthetic_mnist_dataset(num_samples=2000):
    # Random MNIST-shaped inputs and labels for pipeline testing
    x = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(x, y)


def split_dataset(dataset, num_clients=5):
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split_size = len(indices) // num_clients
    subsets = []

    for i in range(num_clients):
        start = i * split_size
        end = len(indices) if i == num_clients - 1 else (i + 1) * split_size
        subset_indices = indices[start:end]
        x = torch.stack([dataset[idx][0] for idx in subset_indices])
        y = torch.tensor([dataset[idx][1] for idx in subset_indices])
        subsets.append(TensorDataset(x, y))

    return subsets


def make_clients(global_model, train_dataset, num_clients=5, device="cpu"):
    client_datasets = split_dataset(train_dataset, num_clients=num_clients)

    clients = []
    for cid, subset in enumerate(client_datasets):
        client = FederatedClient(
            client_id=cid,
            model=global_model,
            dataset=subset,
            local_epochs=1,
            batch_size=64,
            lr=0.01,
            device=device,
        )
        clients.append(client)

    return clients


def run_federated_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = make_synthetic_mnist_dataset(2000)
    test_dataset = make_synthetic_mnist_dataset(500)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    global_model = get_model("mnist").to(device)
    clients = make_clients(global_model, train_dataset, num_clients=5, device=device)

    attack_config = ByzantineAttackConfig(
        attack_type=AttackType.SIGN_FLIP,
        malicious_ratio=0.2,
        scale_factor=10.0,
    )

    orchestrator = AttackOrchestrator(attack_config)

    # Slightly less aggressive settings work better for this noisy synthetic setup
    detector = DetectionLayer(
        z_thresh=2.0,
        cosine_thresh=0.0,
        reputation_thresh=0.2,
        fusion_thresh=0.8,
    )

    num_rounds = 3

    for round_num in range(1, num_rounds + 1):
        print(f"\n========== Round {round_num} ==========")

        global_params = [p.data.clone().detach() for p in global_model.parameters()]

        all_client_updates: Dict[int, List[torch.Tensor]] = {}
        client_sizes: Dict[int, int] = {}

        # Collect local updates from each client
        for client in clients:
            update, num_samples = client.train(global_params)
            all_client_updates[client.client_id] = update
            client_sizes[client.client_id] = num_samples

        poisoned_updates, malicious_ids = orchestrator.run_round(all_client_updates, round_num)
        detection_results = detector.detect(poisoned_updates)

        predicted_ids = [
            cid for cid, info in detection_results.items()
            if info["final_flag"]
        ]

        # Keep only the updates that passed detection
        clean_updates = {
            cid: update
            for cid, update in poisoned_updates.items()
            if not detection_results[cid]["final_flag"]
        }

        clean_client_sizes = {
            cid: client_sizes[cid]
            for cid in clean_updates.keys()
        }

        # Use all updates if every client gets filtered out
        if not clean_updates:
            print("All clients were flagged. Falling back to poisoned updates.")
            clean_updates = poisoned_updates
            clean_client_sizes = client_sizes

        print(f"Actual malicious clients:     {malicious_ids}")
        print(f"Predicted suspicious clients: {predicted_ids}")
        print(f"Accepted clients:             {list(clean_updates.keys())}")

        aggregate_updates(global_model, clean_updates, clean_client_sizes)

        acc = evaluate(global_model, test_loader, device=device)
        print(f"Test accuracy after round {round_num}: {acc:.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    run_federated_training()