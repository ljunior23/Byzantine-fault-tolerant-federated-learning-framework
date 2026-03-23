"""
Detection layers for spotting suspicious client updates in federated learning.

This includes z-score filtering, cosine similarity checks, reputation scoring,
and a magnitude check for unusually large updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import torch


def flatten_update(update: List[torch.Tensor]) -> torch.Tensor:
    # Merge one client's update into a single vector
    return torch.cat([tensor.detach().flatten().cpu() for tensor in update])


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = (torch.norm(a) * torch.norm(b)).item()
    if denom < 1e-12:
        return 0.0
    return torch.dot(a, b).item() / denom


def extract_features(flat_update: torch.Tensor, round_mean: torch.Tensor) -> Dict[str, float]:
    # Basic values used to compare client updates
    diff = flat_update - round_mean
    return {
        "l2_norm": torch.norm(flat_update).item(),
        "mean": flat_update.mean().item(),
        "std": flat_update.std(unbiased=False).item(),
        "max_abs": flat_update.abs().max().item(),
        "distance_to_mean": torch.norm(diff).item(),
        "cosine_to_mean": cosine_similarity(flat_update, round_mean),
    }


@dataclass
class ReputationTracker:
    scores: Dict[int, float] = field(default_factory=dict)
    init_score: float = 1.0
    min_score: float = 0.0
    max_score: float = 1.0
    penalty_z: float = 0.20
    penalty_cosine: float = 0.20
    reward_clean: float = 0.05

    def get(self, client_id: int) -> float:
        return self.scores.get(client_id, self.init_score)

    def update(self, client_id: int, z_flag: bool, cosine_flag: bool) -> float:
        # Lower trust when a client looks suspicious, slightly reward clean rounds 
        score = self.get(client_id)

        if z_flag:
            score -= self.penalty_z
        if cosine_flag:
            score -= self.penalty_cosine
        if not z_flag and not cosine_flag:
            score += self.reward_clean

        score = max(self.min_score, min(self.max_score, score))
        self.scores[client_id] = score
        return score


class DetectionLayer:
    def __init__(
        self,
        z_thresh: float = 1.5,
        cosine_thresh: float = 0.2,
        reputation_thresh: float = 0.4,
        fusion_thresh: float = 0.6,
    ):
        self.z_thresh = z_thresh
        self.cosine_thresh = cosine_thresh
        self.reputation_thresh = reputation_thresh
        self.fusion_thresh = fusion_thresh
        self.reputation = ReputationTracker()

    def detect(self, client_updates: Dict[int, List[torch.Tensor]]) -> Dict[int, Dict]:
        if not client_updates:
            return {}

        # Flatten updates & get the average update for the round
        flat_updates = {cid: flatten_update(update) for cid, update in client_updates.items()}
        client_ids = list(flat_updates.keys())
        stacked = torch.stack([flat_updates[cid] for cid in client_ids])
        round_mean = stacked.mean(dim=0)

        features = {
            cid: extract_features(flat_updates[cid], round_mean)
            for cid in client_ids
        }
        feature_names = list(next(iter(features.values())).keys())

        # Get round wide stats for each feature
        feature_stats = {}
        for name in feature_names:
            values = torch.tensor([features[cid][name] for cid in client_ids], dtype=torch.float32)
            mean = values.mean().item()
            std = values.std(unbiased=False).item()
            median = values.median().item()
            if std < 1e-8:
                std = 1e-8
            feature_stats[name] = {
                "mean": mean,
                "std": std,
                "median": median,
            }

        # Check for statistical outliers
        z_scores = {}
        z_flags = {}
        for cid in client_ids:
            per_feature_z = {}
            for name in feature_names:
                mu = feature_stats[name]["mean"]
                sigma = feature_stats[name]["std"]
                value = features[cid][name]
                per_feature_z[name] = abs((value - mu) / sigma)

            z_scores[cid] = max(per_feature_z.values())
            z_flags[cid] = z_scores[cid] > self.z_thresh

        # Catch updates that are much larger than the rest
        magnitude_flags = {}
        for cid in client_ids:
            l2_median = max(feature_stats["l2_norm"]["median"], 1e-8)
            max_abs_median = max(feature_stats["max_abs"]["median"], 1e-8)

            magnitude_flags[cid] = (
                features[cid]["l2_norm"] > 3.0 * l2_median or
                features[cid]["max_abs"] > 3.0 * max_abs_median
            )

        # Check whether an update points in a very different direction
        cosine_scores = {}
        cosine_flags = {}
        avg_cosines = {}

        for cid in client_ids:
            current = flat_updates[cid]
            others = [flat_updates[other_id] for other_id in client_ids if other_id != cid]

            if not others:
                avg_cos = 1.0
            else:
                avg_cos = sum(cosine_similarity(current, other) for other in others) / len(others)

            avg_cosines[cid] = avg_cos
            cosine_scores[cid] = 1.0 - max(-1.0, min(1.0, avg_cos))
            cosine_flags[cid] = avg_cos < self.cosine_thresh

        # Combine all signals into one final decision
        results = {}
        for cid in client_ids:
            rep = self.reputation.update(cid, z_flags[cid], cosine_flags[cid])

            normalized_z = min(z_scores[cid] / max(self.z_thresh, 1e-8), 1.5)
            normalized_cosine = min(cosine_scores[cid], 1.5)

            final_score = (
                0.4 * normalized_z +
                0.4 * normalized_cosine +
                0.2 * (1.0 - rep)
            )

            final_flag = (
                (final_score > self.fusion_thresh) or
                (rep < self.reputation_thresh) or
                magnitude_flags[cid]
            )

            results[cid] = {
                "features": features[cid],
                "z_score": z_scores[cid],
                "z_flag": z_flags[cid],
                "magnitude_flag": magnitude_flags[cid],
                "avg_cosine": avg_cosines[cid],
                "cosine_score": cosine_scores[cid],
                "cosine_flag": cosine_flags[cid],
                "reputation": rep,
                "final_score": final_score,
                "final_flag": final_flag,
            }

        return results