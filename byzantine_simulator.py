"""
Byzantine Attack Simulator
Supports: label flip, gradient scaling, sign flip, backdoor injection, adaptive evasion.
"""

import torch
import numpy as np
import random
from typing import List, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttackType(str, Enum):
    LABEL_FLIP = "label_flip"
    GRADIENT_SCALING = "gradient_scaling"
    SIGN_FLIP = "sign_flip"
    BACKDOOR = "backdoor"
    ADAPTIVE = "adaptive"
    RANDOM_NOISE = "random_noise"


class ByzantineAttackConfig:
    def __init__(
        self,
        attack_type: AttackType = AttackType.SIGN_FLIP,
        malicious_ratio: float = 0.3,
        scale_factor: float = 10.0,
        backdoor_target_label: int = 0,
        backdoor_trigger_value: float = 1.0,
        noise_std: float = 1.0,
        adaptive_evasion: bool = False,
    ):
        self.attack_type = attack_type
        self.malicious_ratio = malicious_ratio
        self.scale_factor = scale_factor
        self.backdoor_target_label = backdoor_target_label
        self.backdoor_trigger_value = backdoor_trigger_value
        self.noise_std = noise_std
        self.adaptive_evasion = adaptive_evasion


class ByzantineAttackSimulator:
    """Injects Byzantine attacks into a subset of client gradient updates."""

    def __init__(self, config: ByzantineAttackConfig):
        self.config = config

    def select_malicious_clients(self, client_ids: List[int]) -> List[int]:
        n_malicious = int(len(client_ids) * self.config.malicious_ratio)
        return random.sample(client_ids, min(n_malicious, len(client_ids)))

    def poison_gradients(
        self,
        gradients: List[torch.Tensor],
        client_id: int,
        honest_gradients: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """Apply the configured attack to a single client's gradients."""
        attack = self.config.attack_type
        cfg = self.config

        if attack == AttackType.SIGN_FLIP:
            return self._sign_flip(gradients, cfg.scale_factor)

        elif attack == AttackType.GRADIENT_SCALING:
            return self._gradient_scaling(gradients, cfg.scale_factor)

        elif attack == AttackType.RANDOM_NOISE:
            return self._random_noise(gradients, cfg.noise_std)

        elif attack == AttackType.BACKDOOR:
            return self._backdoor(gradients, cfg.scale_factor)

        elif attack == AttackType.ADAPTIVE:
            return self._adaptive(gradients, honest_gradients, cfg.scale_factor)

        elif attack == AttackType.LABEL_FLIP:
            # Label flip manifests as sign-flipped gradients for targeted layers
            return self._targeted_sign_flip(gradients)

        return gradients

    # --- Attack Implementations ---

    def _sign_flip(self, grads: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
        """Flip and scale gradients to maximize loss."""
        return [-g * scale for g in grads]

    def _gradient_scaling(self, grads: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
        """Amplify gradients without flipping direction."""
        return [g * scale for g in grads]

    def _random_noise(self, grads: List[torch.Tensor], std: float) -> List[torch.Tensor]:
        """Replace gradients with Gaussian noise."""
        return [torch.randn_like(g) * std for g in grads]

    def _backdoor(self, grads: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
        """
        Model replacement backdoor: scale up gradients to override aggregation.
        In practice, the malicious client trains on backdoored data and then scales.
        """
        return [g * scale for g in grads]

    def _adaptive(
        self,
        grads: List[torch.Tensor],
        honest_grads: Optional[List[List[torch.Tensor]]],
        scale: float,
    ) -> List[torch.Tensor]:
        """
        Adaptive attack: craft gradients to evade detection while still
        being harmful. Uses the mean of honest gradients as a camouflage base.
        """
        if not honest_grads:
            return self._sign_flip(grads, scale)

        # Compute mean of honest gradients
        mean_honest = [
            torch.stack([h[i] for h in honest_grads]).mean(0)
            for i in range(len(grads))
        ]

        # Craft: negative perturbation in the direction that evades z-score detection
        poisoned = []
        for g, h_mean in zip(grads, mean_honest):
            norm_h = torch.norm(h_mean)
            # Project along mean direction but flip
            direction = -h_mean / (norm_h + 1e-8)
            poisoned.append(direction * norm_h * scale)

        return poisoned

    def _targeted_sign_flip(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """Flip only the last layer (output) gradients — label flip proxy."""
        result = []
        for i, g in enumerate(grads):
            if i == len(grads) - 1:  # Last layer
                result.append(-g * 2.0)
            else:
                result.append(g)
        return result


class AttackOrchestrator:
    """
    Coordinates Byzantine attacks across all clients in a federated round.
    Tracks which clients are malicious and logs attack statistics.
    """

    def __init__(self, config: ByzantineAttackConfig):
        self.config = config
        self.simulator = ByzantineAttackSimulator(config)
        self.attack_log: List[Dict] = []

    def run_round(
        self,
        all_client_gradients: Dict[int, List[torch.Tensor]],
        round_num: int,
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Apply Byzantine attacks to a subset of clients.

        Returns:
            Modified gradient dict with poisoned updates for malicious clients.
        """
        client_ids = list(all_client_gradients.keys())
        malicious_ids = self.simulator.select_malicious_clients(client_ids)
        honest_ids = [c for c in client_ids if c not in malicious_ids]

        honest_grads_list = [all_client_gradients[c] for c in honest_ids]
        result = dict(all_client_gradients)

        for cid in malicious_ids:
            poisoned = self.simulator.poison_gradients(
                all_client_gradients[cid],
                cid,
                honest_gradients=honest_grads_list
                if self.config.attack_type == AttackType.ADAPTIVE
                else None,
            )
            result[cid] = poisoned

        log_entry = {
            "round": round_num,
            "attack_type": self.config.attack_type,
            "malicious_ids": malicious_ids,
            "malicious_ratio": len(malicious_ids) / len(client_ids),
        }
        self.attack_log.append(log_entry)
        logger.info(
            f"Round {round_num}: {len(malicious_ids)}/{len(client_ids)} malicious "
            f"({self.config.attack_type})"
        )

        return result, malicious_ids
