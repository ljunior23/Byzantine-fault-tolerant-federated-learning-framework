import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../BFT/experiments
_PARENT   = os.path.dirname(_THIS_DIR)                   # .../BFT

for _candidate in [
    _THIS_DIR,                          
    _PARENT,                            
    os.path.join(_PARENT, "src"),       
]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple
import logging

import torchvision
import torchvision.transforms as transforms

from src.server import BFTFederatedServer, ServerConfig, ClientUpdate
from src.client import FederatedClient, NonIIDPartitioner
from src.models.cnn import get_model
from src.attacks.byzantine_simulator import AttackOrchestrator, ByzantineAttackConfig, AttackType

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


# Experiment Configuration 
ATTACK_TYPES = [
    AttackType.SIGN_FLIP,
    AttackType.GRADIENT_SCALING,
    AttackType.BACKDOOR,
    AttackType.ADAPTIVE,
    AttackType.RANDOM_NOISE,
]

MALICIOUS_RATIOS = [0.1, 0.2, 0.3, 0.4]

STRATEGIES = ["fedavg", "krum", "trimmed_mean", "geo_median", "normalized", "adaptive"]


def load_dataset(name: str, iid: bool = True):
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    return train, test


def evaluate(model, test_loader, device="cpu") -> Tuple[float, float]:
    """Returns (accuracy %, avg loss)."""
    model.eval()
    model.to(device)
    correct, total, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_sum += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total, loss_sum / len(test_loader)


def run_single_experiment(
    dataset_name: str,
    strategy: str,
    attack_type: AttackType,
    malicious_ratio: float,
    num_clients: int = 10,
    rounds: int = 20,
    iid: bool = False,
    device: str = "cpu",
) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: strategy={strategy} | attack={attack_type} | "
                f"ratio={malicious_ratio} | iid={iid}")
    logger.info('='*60)

    # Data
    train_data, test_data = load_dataset(dataset_name, iid)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)

    if iid:
        n = len(train_data)
        idxs = list(range(n))
        np.random.shuffle(idxs)
        splits = np.array_split(idxs, num_clients)
        client_datasets = [torch.utils.data.Subset(train_data, s) for s in splits]
    else:
        partitioner = NonIIDPartitioner(train_data, num_clients, alpha=0.5)
        client_datasets = partitioner.partition()

    # Model & Server
    model = get_model(dataset_name)
    server_config = ServerConfig(
        num_clients=num_clients,
        malicious_ratio=malicious_ratio,
        aggregation_strategy=strategy,
        rounds=rounds,
    )
    server = BFTFederatedServer(model, server_config)

    # Clients
    clients = [
        FederatedClient(cid, model, client_datasets[cid], device=device)
        for cid in range(num_clients)
    ]

    # Attack orchestrator
    attack_cfg = ByzantineAttackConfig(
        attack_type=attack_type,
        malicious_ratio=malicious_ratio,
    )
    orchestrator = AttackOrchestrator(attack_cfg)

    # Training loop
    accuracy_history = []
    loss_history = []

    for round_num in range(rounds):
        global_params = server.get_global_model_params()

        # Collecting client updates
        raw_gradients = {}
        sample_counts = {}
        for client in clients:
            grads, n_samples = client.train(global_params)
            raw_gradients[client.client_id] = grads
            sample_counts[client.client_id] = n_samples

        # Injecting Byzantine attacks
        poisoned_gradients, malicious_ids = orchestrator.run_round(raw_gradients, round_num)

        # Building ClientUpdate objects
        updates = [
            ClientUpdate(
                client_id=cid,
                gradients=poisoned_gradients[cid],
                num_samples=sample_counts[cid],
                round_num=round_num,
            )
            for cid in range(num_clients)
        ]

        # Aggregate
        _, round_metrics = server.aggregate(updates)

        # Evaluating every 5 rounds
        if round_num % 5 == 0 or round_num == rounds - 1:
            acc, loss = evaluate(server.model, test_loader, device)
            accuracy_history.append({"round": round_num, "accuracy": acc, "loss": loss})
            logger.info(f"Round {round_num:3d}: acc={acc:.2f}% | loss={loss:.4f} | "
                        f"strategy={round_metrics.get('strategy', strategy)}")

    final_acc, final_loss = evaluate(server.model, test_loader, device)
    summary = server.metrics.get_summary()

    result = {
        "config": {
            "dataset": dataset_name,
            "strategy": strategy,
            "attack_type": str(attack_type),
            "malicious_ratio": malicious_ratio,
            "iid": iid,
            "rounds": rounds,
            "num_clients": num_clients,
        },
        "final_accuracy": round(final_acc, 2),
        "final_loss": round(final_loss, 4),
        "accuracy_history": accuracy_history,
        "summary": summary,
    }

    logger.info(f"RESULT: {strategy}/{attack_type}/{malicious_ratio} → acc={final_acc:.2f}%")
    return result


def run_all_experiments(args):
    results = []
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    attack_list = ATTACK_TYPES if not args.quick else [AttackType.SIGN_FLIP, AttackType.ADAPTIVE]
    strategy_list = STRATEGIES if not args.quick else ["fedavg", "krum", "adaptive"]
    ratio_list = MALICIOUS_RATIOS if not args.quick else [0.2, 0.3]

    total = len(attack_list) * len(strategy_list) * len(ratio_list)
    logger.info(f"Running {total} experiments...")

    for i, (attack, strategy, ratio) in enumerate(product(attack_list, strategy_list, ratio_list)):
        logger.info(f"\nExperiment {i+1}/{total}")
        result = run_single_experiment(
            dataset_name=args.dataset,
            strategy=strategy,
            attack_type=attack,
            malicious_ratio=ratio,
            num_clients=args.num_clients,
            rounds=args.rounds,
            iid=args.iid,
            device=args.device,
        )
        results.append(result)

        # Save incrementally
        with open(out_dir / "all_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    logger.info(f"\nAll experiments complete. Results in results/all_results.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="BFT-FL Experiment Runner")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iid", action="store_true", help="Use IID data split")
    parser.add_argument("--quick", action="store_true", help="Run subset for quick testing")
    parser.add_argument("--strategy", default=None, help="Run single strategy")
    parser.add_argument("--attack", default=None, help="Run single attack type")
    parser.add_argument("--ratio", type=float, default=None, help="Single malicious ratio")
    args = parser.parse_args()

    if args.strategy and args.attack and args.ratio:
        # Single run
        result = run_single_experiment(
            dataset_name=args.dataset,
            strategy=args.strategy,
            attack_type=AttackType(args.attack),
            malicious_ratio=args.ratio,
            num_clients=args.num_clients,
            rounds=args.rounds,
            iid=args.iid,
            device=args.device,
        )
        print(json.dumps(result, indent=2, default=str))
    else:
        run_all_experiments(args)


if __name__ == "__main__":
    main()
