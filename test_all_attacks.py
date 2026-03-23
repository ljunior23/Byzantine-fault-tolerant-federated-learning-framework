"""
Quick test script for checking the detection layer against each supported
Byzantine attack type.
"""

import torch
from detection_layers import DetectionLayer
from byzantine_simulator import (
    AttackOrchestrator,
    ByzantineAttackConfig,
    AttackType,
)


def make_update(values):
    return [torch.tensor(values, dtype=torch.float32)]


def compute_metrics(predicted_ids, malicious_ids, all_ids):
    predicted_ids = set(predicted_ids)
    malicious_ids = set(malicious_ids)
    all_ids = set(all_ids)

    tp = len(predicted_ids & malicious_ids)
    fp = len(predicted_ids - malicious_ids)
    fn = len(malicious_ids - predicted_ids)
    tn = len((all_ids - malicious_ids) - predicted_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(all_ids) if len(all_ids) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
    }


def get_base_updates():
    # Small sample of normal looking client updates
    return {
        0: make_update([1.0, 1.1, 0.9, 1.0]),
        1: make_update([1.0, 0.95, 1.05, 1.0]),
        2: make_update([0.9, 1.0, 1.1, 1.0]),
        3: make_update([1.05, 1.0, 0.95, 1.0]),
        4: make_update([1.0, 1.0, 1.0, 0.9]),
    }


attack_types = [
    AttackType.SIGN_FLIP,
    AttackType.GRADIENT_SCALING,
    AttackType.RANDOM_NOISE,
    AttackType.BACKDOOR,
    AttackType.ADAPTIVE,
    AttackType.LABEL_FLIP,
]

for attack_type in attack_types:
    print(f"\n========== {attack_type.value.upper()} ==========")

    all_client_updates = get_base_updates()

    config = ByzantineAttackConfig(
        attack_type=attack_type,
        malicious_ratio=0.2,
        scale_factor=10.0,
        noise_std=1.0,
    )

    orchestrator = AttackOrchestrator(config)
    detector = DetectionLayer()

    poisoned_updates, malicious_ids = orchestrator.run_round(all_client_updates, round_num=1)
    results = detector.detect(poisoned_updates)

    predicted_ids = [cid for cid, info in results.items() if info["final_flag"]]
    metrics = compute_metrics(predicted_ids, malicious_ids, list(all_client_updates.keys()))

    print("Actual malicious_ids: ", malicious_ids)
    print("Predicted suspicious: ", predicted_ids)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")