import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np

from aggregators.krum import krum_aggregate
from aggregators.trimmed_mean import trimmed_mean_aggregate
from aggregators.geometric_median import geometric_median_aggregate
from aggregators.normalized import normalized_aggregate
from detection.detector import GradientDetector
from attacks.byzantine_simulator import (
    ByzantineAttackSimulator, ByzantineAttackConfig, AttackType
)


# Fixtures

def make_gradients(n: int = 5, shape=(10,), val: float = 1.0):
    """Create n identical gradient lists."""
    return [[torch.full(shape, val)] for _ in range(n)]


def make_mixed_gradients(n_honest=7, n_malicious=3, shape=(10,)):
    """Mix of normal and adversarial gradients."""
    honest = [[torch.ones(shape)] for _ in range(n_honest)]
    malicious = [[torch.full(shape, -10.0)] for _ in range(n_malicious)]
    return honest + malicious


# Aggregator Tests 

class TestKrum:
    def test_identical_gradients(self):
        grads = make_gradients(5, val=2.0)
        result = krum_aggregate(grads, f=1)
        assert len(result) == 1
        assert torch.allclose(result[0], torch.full((10,), 2.0))

    def test_filters_outlier(self):
        grads = make_mixed_gradients(7, 3)
        result = krum_aggregate(grads, f=3)
        assert result[0].mean().item() > 0

    def test_multi_krum(self):
        grads = make_gradients(5, val=1.0)
        result = krum_aggregate(grads, f=1, multi_krum_m=3)
        assert torch.allclose(result[0], torch.ones(10))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            krum_aggregate([], f=0)


class TestTrimmedMean:
    def test_no_trim(self):
        grads = make_gradients(5, val=3.0)
        result = trimmed_mean_aggregate(grads, f=0)
        assert torch.allclose(result[0], torch.full((10,), 3.0))

    def test_trims_outliers(self):
        grads = make_mixed_gradients(7, 3)
        result = trimmed_mean_aggregate(grads, f=3)
        assert result[0].mean().item() > 0

    def test_clamps_f(self):
        grads = make_gradients(4, val=1.0)
        result = trimmed_mean_aggregate(grads, f=3)
        assert result is not None


class TestGeometricMedian:
    def test_single_gradient(self):
        grads = make_gradients(1, val=5.0)
        result = geometric_median_aggregate(grads, f=0)
        assert torch.allclose(result[0], torch.full((10,), 5.0))

    def test_robust_to_outliers(self):
        grads = make_mixed_gradients(7, 1)
        result = geometric_median_aggregate(grads, f=1)
        assert result[0].mean().item() > 0

    def test_convergence(self):
        grads = make_gradients(5, val=2.0)
        result = geometric_median_aggregate(grads, f=1, max_iter=50)
        assert torch.allclose(result[0], torch.full((10,), 2.0), atol=0.01)


class TestNormalizedAggregate:
    def test_clips_large_gradients(self):
        large = [[torch.full((10,), 100.0)]]
        normal = make_gradients(4, val=1.0)
        result = normalized_aggregate(large + normal, f=0, clip_threshold=1.0)
        assert result[0].mean().item() < 50.0

    def test_passes_normal_gradients(self):
        grads = make_gradients(5, val=0.1)
        result = normalized_aggregate(grads, f=0, clip_threshold=1.0)
        assert torch.allclose(result[0], torch.full((10,), 0.1), atol=0.01)


# Detector Tests 

class TestGradientDetector:
    def setup_method(self):
        self.detector = GradientDetector(zscore_threshold=2.0)

    def test_flags_sign_flip_attack(self):
        grads = make_mixed_gradients(7, 3)
        client_ids = list(range(10))
        report = self.detector.analyze(grads, client_ids)
        assert len(report["flagged_ids"]) > 0

    def test_clean_gradients_not_flagged(self):
        grads = make_gradients(8, val=1.0)
        # Adding slight noise
        noisy = [[g + torch.randn_like(g) * 0.01] for [g] in grads]
        report = self.detector.analyze(noisy, list(range(8)))
        assert len(report["flagged_ids"]) <= 1

    def test_empty_returns_safe(self):
        report = self.detector.analyze([], [])
        assert "flagged_ids" in report

    def test_returns_all_report_keys(self):
        grads = make_gradients(5)
        report = self.detector.analyze(grads, list(range(5)))
        for key in ["flagged_ids", "cosine_anomaly_score", "zscore_anomaly_ratio"]:
            assert key in report


# Attack Simulator Tests 

class TestByzantineAttacks:
    def _get_simulator(self, attack_type):
        cfg = ByzantineAttackConfig(attack_type=attack_type)
        return ByzantineAttackSimulator(cfg)

    def test_sign_flip_negates(self):
        sim = self._get_simulator(AttackType.SIGN_FLIP)
        grads = [torch.ones(10)]
        poisoned = sim.poison_gradients(grads, client_id=0)
        assert poisoned[0].mean().item() < 0

    def test_scaling_amplifies(self):
        cfg = ByzantineAttackConfig(attack_type=AttackType.GRADIENT_SCALING, scale_factor=5.0)
        sim = ByzantineAttackSimulator(cfg)
        grads = [torch.ones(10)]
        poisoned = sim.poison_gradients(grads, client_id=0)
        assert torch.allclose(poisoned[0], torch.full((10,), 5.0))

    def test_random_noise_changes_values(self):
        sim = self._get_simulator(AttackType.RANDOM_NOISE)
        grads = [torch.ones(10)]
        poisoned = sim.poison_gradients(grads, client_id=0)
        assert not torch.allclose(poisoned[0], torch.ones(10))

    def test_malicious_selection_ratio(self):
        cfg = ByzantineAttackConfig(malicious_ratio=0.3)
        sim = ByzantineAttackSimulator(cfg)
        ids = list(range(10))
        malicious = sim.select_malicious_clients(ids)
        assert len(malicious) == 3
