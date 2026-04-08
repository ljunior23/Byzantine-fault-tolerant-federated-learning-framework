import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [_THIS_DIR, os.path.join(_THIS_DIR, "src")]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)


import torch
import numpy as np
import logging
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from aggregators.krum import krum_aggregate
from aggregators.trimmed_mean import trimmed_mean_aggregate
from aggregators.geometric_median import geometric_median_aggregate
from aggregators.normalized import normalized_aggregate
from detection.detector import GradientDetector
from utils.metrics import MetricsLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [SERVER] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    client_id: int
    gradients: List[torch.Tensor]
    num_samples: int
    round_num: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class ServerConfig:
    num_clients: int = 10
    malicious_ratio: float = 0.3
    aggregation_strategy: str = "adaptive"
    detection_enabled: bool = True
    reputation_enabled: bool = True
    min_clients_per_round: int = 5
    rounds: int = 50
    threat_threshold_high: float = 0.4
    threat_threshold_low: float = 0.1


class BFTFederatedServer:
    """Byzantine-Fault-Tolerant Federated Learning Server with adaptive aggregation."""

    AGGREGATORS = {
        "krum": krum_aggregate,
        "trimmed_mean": trimmed_mean_aggregate,
        "geo_median": geometric_median_aggregate,
        "normalized": normalized_aggregate,
        "fedavg": lambda updates, **kw: _fedavg(updates),
    }

    def __init__(self, model: torch.nn.Module, config: ServerConfig):
        self.model = model
        self.config = config
        self.global_params = [p.data.clone() for p in model.parameters()]
        self.reputation_scores: Dict[int, float] = defaultdict(lambda: 1.0)
        self.detector = GradientDetector()
        self.metrics = MetricsLogger()
        self.current_round = 0
        self.threat_level = 0.0
        self.strategy_history: List[str] = []

    def aggregate(self, updates: List[ClientUpdate]) -> Tuple[List[torch.Tensor], Dict]:
        """Run detection, select strategy, and aggregate gradients."""
        round_metrics = {"round": self.current_round, "num_updates": len(updates)}
        start_time = time.time()

        flagged_ids = set()
        detection_report = {}

        if self.config.detection_enabled:
            grads = [u.gradients for u in updates]
            detection_report = self.detector.analyze(grads, [u.client_id for u in updates])
            flagged_ids = detection_report.get("flagged_ids", set())

            if self.config.reputation_enabled:
                for u in updates:
                    if u.client_id in flagged_ids:
                        self.reputation_scores[u.client_id] *= 0.7
                    else:
                        self.reputation_scores[u.client_id] = min(
                            1.0, self.reputation_scores[u.client_id] * 1.05
                        )

        filtered_updates = [
            u for u in updates
            if u.client_id not in flagged_ids
            and self.reputation_scores[u.client_id] >= 0.3
        ]

        if len(filtered_updates) < self.config.min_clients_per_round:
            logger.warning("Too few trusted clients; using all updates with robust aggregator.")
            filtered_updates = updates

        strategy = self._select_strategy(detection_report, len(flagged_ids), len(updates))
        self.strategy_history.append(strategy)
        round_metrics["strategy"] = strategy
        round_metrics["flagged_clients"] = list(flagged_ids)

        aggregator = self.AGGREGATORS[strategy]
        n_malicious = max(0, int(len(updates) * self.config.malicious_ratio))
        agg_grads = aggregator(
            [u.gradients for u in filtered_updates],
            f=n_malicious,
            weights=[u.num_samples for u in filtered_updates],
        )

        # Moving aggregated gradients to the model's device before applying
        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), agg_grads):
                param.data.sub_(grad.to(model_device))
            self.global_params = [p.data.clone() for p in self.model.parameters()]

        round_metrics["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        round_metrics["threat_level"] = round(self.threat_level, 3)
        round_metrics["reputation_scores"] = {
            k: round(v, 3) for k, v in self.reputation_scores.items()
        }

        self.metrics.log(round_metrics)
        self.current_round += 1
        return agg_grads, round_metrics

    def _select_strategy(self, detection_report: Dict, n_flagged: int, n_total: int) -> str:
        if self.config.aggregation_strategy != "adaptive":
            return self.config.aggregation_strategy

        flagged_ratio = n_flagged / max(n_total, 1)
        cosine_anomaly = detection_report.get("cosine_anomaly_score", 0.0)
        zscore_anomaly = detection_report.get("zscore_anomaly_ratio", 0.0)
        self.threat_level = 0.4 * flagged_ratio + 0.3 * cosine_anomaly + 0.3 * zscore_anomaly

        if self.threat_level >= self.config.threat_threshold_high:
            logger.info(f"HIGH threat ({self.threat_level:.2f}): using Krum")
            return "krum"
        elif self.threat_level >= self.config.threat_threshold_low:
            logger.info(f"MEDIUM threat ({self.threat_level:.2f}): using Trimmed Mean")
            return "trimmed_mean"
        else:
            logger.info(f"LOW threat ({self.threat_level:.2f}): using Normalized Aggregation")
            return "normalized"

    def get_global_model_params(self) -> List[torch.Tensor]:
        return [p.data.clone().cpu() for p in self.model.parameters()]

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "reputation_scores": dict(self.reputation_scores),
            "round": self.current_round,
            "strategy_history": self.strategy_history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")


def _fedavg(gradient_lists: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    avg = [torch.zeros_like(g) for g in gradient_lists[0]]
    for grads in gradient_lists:
        for i, g in enumerate(grads):
            avg[i] += g
    return [g / len(gradient_lists) for g in avg]