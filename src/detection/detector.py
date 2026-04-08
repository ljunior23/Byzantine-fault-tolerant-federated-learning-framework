import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class GradientDetector:
    """
    Layered anomaly detection for federated learning gradient updates
    """

    def __init__(
        self,
        zscore_threshold: float = 2.5,
        cosine_threshold: float = -0.5,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 2,
        pca_components: int = 50,
    ):
        self.zscore_threshold = zscore_threshold
        self.cosine_threshold = cosine_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.pca_components = pca_components

    def analyze(
        self,
        gradient_lists: List[List[torch.Tensor]],
        client_ids: List[int],
    ) -> Dict:
        """
        Running all detection layers and returning analysis report.
        """
        n = len(gradient_lists)
        if n < 2:
            return {"flagged_ids": set()}

        flat_grads = [self._flatten(grads) for grads in gradient_lists]
        norms = torch.tensor([g.norm().item() for g in flat_grads])

        zscore_flagged = self._zscore_filter(norms, client_ids)
        cosine_flagged, cosine_score = self._cosine_analysis(flat_grads, client_ids)
        cluster_flagged = self._dbscan_clustering(flat_grads, client_ids)

        all_flagged = zscore_flagged | cosine_flagged | cluster_flagged

        report = {
            "flagged_ids": all_flagged,
            "zscore_flagged": zscore_flagged,
            "cosine_flagged": cosine_flagged,
            "cluster_flagged": cluster_flagged,
            "cosine_anomaly_score": cosine_score,
            "zscore_anomaly_ratio": len(zscore_flagged) / n,
            "total_flagged": len(all_flagged),
            "total_clients": n,
        }

        if all_flagged:
            logger.info(f"Detection flagged {len(all_flagged)} clients: {all_flagged}")

        return report

    def _flatten(self, grads: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.detach().cpu().view(-1) for g in grads]).float()

    def _zscore_filter(self, norms: torch.Tensor, client_ids: List[int]) -> Set[int]:
        """Flag clients whose gradient norm is a z-score outlier."""
        if len(norms) < 3:
            return set()

        mean = norms.mean()
        std = norms.std()
        if std < 1e-8:
            return set()

        zscores = ((norms - mean) / std).abs()
        flagged = set()
        for i, (cid, z) in enumerate(zip(client_ids, zscores)):
            if z.item() > self.zscore_threshold:
                flagged.add(cid)
        return flagged

    def _cosine_analysis(
        self, flat_grads: List[torch.Tensor], client_ids: List[int]
    ) -> Tuple[Set[int], float]:
        """
        Flag clients with high negative cosine similarity to the majority.
        Returns flagged IDs and an overall anomaly score.
        """
        n = len(flat_grads)
        mean_grad = torch.stack(flat_grads).mean(0)
        mean_norm = mean_grad.norm()
        if mean_norm < 1e-8:
            return set(), 0.0

        flagged = set()
        cosine_scores = []
        for cid, g in zip(client_ids, flat_grads):
            cos = torch.nn.functional.cosine_similarity(
                g.unsqueeze(0), mean_grad.unsqueeze(0)
            ).item()
            cosine_scores.append(cos)
            if cos < self.cosine_threshold:
                flagged.add(cid)

        anomaly_score = max(0.0, -min(cosine_scores))  # Worst cosine → [0,1]
        return flagged, anomaly_score

    def _dbscan_clustering(
        self, flat_grads: List[torch.Tensor], client_ids: List[int]
    ) -> Set[int]:
        """
        Using DBSCAN to find gradient outliers. -1 label = noise/outlier.
        Applies simple PCA-like dimensionality reduction for speed.
        """
        n = len(flat_grads)
        if n < 4:
            return set()

        X = torch.stack(flat_grads).cpu().numpy()

        # Reducing dimensions via random projection for speed
        d = X.shape[1]
        target_dim = min(self.pca_components, d, n - 1)
        if target_dim < d:
            rng = np.random.default_rng(42)
            proj = rng.standard_normal((d, target_dim)) / np.sqrt(target_dim)
            X = X @ proj

        # Normalizing rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        X = X / norms

        try:
            labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(X)
        except Exception:
            return set()

        # Points labeled -1 are outliers
        flagged = {client_ids[i] for i, lbl in enumerate(labels) if lbl == -1}
        return flagged