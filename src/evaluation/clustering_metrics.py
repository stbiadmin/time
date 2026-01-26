"""
Module: clustering_metrics.py
=============================

Evaluation metrics for clustering models.

Provides internal metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
and external metrics (Adjusted Rand Index, NMI) for cluster quality assessment.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)


@dataclass
class ClusterQuality:
    """Container for cluster quality assessment."""
    cluster_id: int
    size: int
    silhouette_avg: float
    cohesion: float  # Average distance to centroid
    separation: float  # Average distance to nearest other centroid


class ClusteringEvaluator:
    """
    Evaluator for clustering model performance.

    Provides comprehensive clustering evaluation with internal metrics
    (no ground truth needed) and external metrics (when ground truth available).

    Attributes:
        features: Feature matrix used for clustering
        labels: Cluster assignments
        centroids: Cluster centers
        true_labels: Ground truth labels (if available)
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster labels (n_samples,)
            centroids: Cluster centers (n_clusters, n_features)
            true_labels: Ground truth labels for external evaluation
        """
        self.features = features
        self.labels = labels
        self.centroids = centroids
        self.true_labels = true_labels
        self.n_clusters = len(np.unique(labels))

    def compute_internal_metrics(self) -> Dict[str, float]:
        """
        Compute internal clustering quality metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Silhouette: how similar points are to own cluster vs others [-1, 1]
        metrics["silhouette_score"] = float(silhouette_score(self.features, self.labels))

        # Calinski-Harabasz: ratio of between-cluster to within-cluster dispersion
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(self.features, self.labels))

        # Davies-Bouldin: average similarity between clusters (lower is better)
        metrics["davies_bouldin"] = float(davies_bouldin_score(self.features, self.labels))

        if self.centroids is not None:
            inertia = self._compute_inertia()
            metrics["inertia"] = float(inertia)

        return metrics

    def compute_external_metrics(self) -> Dict[str, float]:
        """
        Compute external clustering quality metrics.

        Returns:
            Dictionary of metric names and values
        """
        if self.true_labels is None:
            return {"error": "No ground truth labels available"}

        metrics = {}

        # Adjusted Rand Index: agreement adjusted for chance [-1, 1]
        metrics["adjusted_rand_score"] = float(
            adjusted_rand_score(self.true_labels, self.labels)
        )

        # Normalized Mutual Information [0, 1]
        metrics["normalized_mutual_info"] = float(
            normalized_mutual_info_score(self.true_labels, self.labels)
        )

        # Homogeneity, Completeness, V-measure
        metrics["homogeneity"] = float(homogeneity_score(self.true_labels, self.labels))
        metrics["completeness"] = float(completeness_score(self.true_labels, self.labels))
        metrics["v_measure"] = float(v_measure_score(self.true_labels, self.labels))

        return metrics

    def analyze_cluster_quality(self) -> List[ClusterQuality]:
        """
        Analyze quality of each individual cluster.

        Returns:
            List of ClusterQuality objects
        """
        silhouette_vals = silhouette_samples(self.features, self.labels)

        cluster_qualities = []
        unique_labels = np.unique(self.labels)

        for cluster_id in unique_labels:
            mask = self.labels == cluster_id
            cluster_points = self.features[mask]
            cluster_silhouettes = silhouette_vals[mask]

            # Cohesion: average distance to centroid
            if self.centroids is not None:
                centroid = self.centroids[cluster_id]
                distances_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
                cohesion = float(np.mean(distances_to_centroid))
            else:
                centroid = cluster_points.mean(axis=0)
                distances_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
                cohesion = float(np.mean(distances_to_centroid))

            # Separation: average distance to nearest other centroid
            if self.centroids is not None and len(self.centroids) > 1:
                other_centroids = np.delete(self.centroids, cluster_id, axis=0)
                distances_to_others = np.linalg.norm(
                    centroid.reshape(1, -1) - other_centroids, axis=1
                )
                separation = float(np.min(distances_to_others))
            else:
                separation = float("inf")

            quality = ClusterQuality(
                cluster_id=int(cluster_id),
                size=int(mask.sum()),
                silhouette_avg=float(np.mean(cluster_silhouettes)),
                cohesion=cohesion,
                separation=separation,
            )
            cluster_qualities.append(quality)

        return cluster_qualities

    def _compute_inertia(self) -> float:
        """
        Compute total inertia (sum of squared distances to centroids).

        Returns:
            Total inertia value
        """
        inertia = 0.0
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            cluster_points = self.features[mask]
            centroid = self.centroids[cluster_id]
            inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia

    def get_summary_table(self) -> str:
        """
        Get a formatted summary of clustering metrics.

        Returns:
            Formatted string table
        """
        internal = self.compute_internal_metrics()
        external = self.compute_external_metrics() if self.true_labels is not None else {}

        lines = ["=" * 50, "CLUSTERING EVALUATION SUMMARY", "=" * 50, ""]

        lines.append("Internal Metrics:")
        lines.append("-" * 30)
        for name, value in internal.items():
            lines.append(f"  {name}: {value:.4f}")

        if external and "error" not in external:
            lines.append("")
            lines.append("External Metrics (vs Ground Truth):")
            lines.append("-" * 30)
            for name, value in external.items():
                lines.append(f"  {name}: {value:.4f}")

        qualities = self.analyze_cluster_quality()
        lines.append("")
        lines.append("Per-Cluster Quality:")
        lines.append("-" * 30)
        for q in qualities:
            lines.append(
                f"  Cluster {q.cluster_id}: "
                f"n={q.size:,}, "
                f"silhouette={q.silhouette_avg:.3f}, "
                f"cohesion={q.cohesion:.3f}"
            )

        lines.append("=" * 50)
        return "\n".join(lines)


def compute_cluster_purity(labels: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Compute cluster purity.

    Measures what fraction of samples in each cluster belong to the majority class.

    Args:
        labels: Predicted cluster labels
        true_labels: Ground truth labels

    Returns:
        Purity score [0, 1]
    """
    unique_clusters = np.unique(labels)
    correct = 0
    total = len(labels)

    for cluster in unique_clusters:
        mask = labels == cluster
        cluster_true_labels = true_labels[mask]
        if len(cluster_true_labels) > 0:
            values, counts = np.unique(cluster_true_labels, return_counts=True)
            correct += np.max(counts)

    return correct / total if total > 0 else 0.0
