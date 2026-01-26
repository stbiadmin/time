"""
Module: kmeans_clusterer.py
===========================

K-means clustering for network event classification with automatic K selection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


@dataclass
class KMeansConfig:
    """Configuration for K-means clustering."""
    n_clusters: Optional[int] = None  # If None, determined by elbow method
    k_range: Tuple[int, int] = (3, 10)  # Range for elbow analysis
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    random_state: int = 42
    use_minibatch: bool = False  # For very large datasets


class KMeansClusterer:
    """
    K-means clustering with automatic K selection.

    Wraps scikit-learn's K-means with elbow method for optimal K selection
    and multiple quality metrics for evaluation.

    Attributes:
        config: Clustering configuration
        model: Fitted KMeans model
        optimal_k: Selected number of clusters
        elbow_data: Data for elbow plot
    """

    def __init__(self, config: Optional[KMeansConfig] = None):
        """
        Initialize the clusterer.

        Args:
            config: Clustering configuration (uses defaults if None)
        """
        self.config = config or KMeansConfig()
        self.model: Optional[KMeans] = None
        self.optimal_k: Optional[int] = None
        self.elbow_data: Dict[int, float] = {}
        self.silhouette_data: Dict[int, float] = {}

    def fit_predict(
        self,
        X: np.ndarray,
        find_optimal_k: bool = True
    ) -> np.ndarray:
        """
        Fit the clustering model and return cluster labels.

        Args:
            X: Feature matrix (n_samples, n_features)
            find_optimal_k: Whether to search for optimal K

        Returns:
            Cluster labels for each sample
        """
        if self.config.n_clusters is not None:
            self.optimal_k = self.config.n_clusters
        elif find_optimal_k:
            self.optimal_k = self._find_optimal_k(X)
        else:
            self.optimal_k = (self.config.k_range[0] + self.config.k_range[1]) // 2

        ModelClass = MiniBatchKMeans if self.config.use_minibatch else KMeans

        self.model = ModelClass(
            n_clusters=self.optimal_k,
            init=self.config.init,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )

        labels = self.model.fit_predict(X)

        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Cluster labels

        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def _find_optimal_k(self, X: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method + silhouette.

        Args:
            X: Feature matrix

        Returns:
            Optimal number of clusters
        """
        k_min, k_max = self.config.k_range
        k_values = list(range(k_min, k_max + 1))

        inertias = []
        silhouettes = []

        print(f"Searching for optimal K in range [{k_min}, {k_max}]...")

        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                init=self.config.init,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))

            self.elbow_data[k] = kmeans.inertia_
            self.silhouette_data[k] = silhouettes[-1]

            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

        elbow_k = self._find_elbow_point(k_values, inertias)

        best_silhouette_k = k_values[np.argmax(silhouettes)]

        # Use silhouette if it's clearly better
        if silhouettes[k_values.index(best_silhouette_k)] > silhouettes[k_values.index(elbow_k)] + 0.05:
            optimal_k = best_silhouette_k
            print(f"Selected K={optimal_k} based on silhouette score")
        else:
            optimal_k = elbow_k
            print(f"Selected K={optimal_k} based on elbow method")

        return optimal_k

    def _find_elbow_point(
        self,
        k_values: List[int],
        inertias: List[float]
    ) -> int:
        """
        Find the elbow point in the inertia curve.

        Uses geometric approach: finds the point with maximum perpendicular
        distance from the line connecting the first and last points.

        Args:
            k_values: List of K values tested
            inertias: Corresponding inertia values

        Returns:
            K value at the elbow
        """
        # Normalize to [0, 1] range
        k_norm = np.array(k_values) - min(k_values)
        k_norm = k_norm / max(k_norm)

        inertia_norm = np.array(inertias) - min(inertias)
        if max(inertia_norm) > 0:
            inertia_norm = inertia_norm / max(inertia_norm)

        # Line from first to last point
        p1 = np.array([k_norm[0], inertia_norm[0]])
        p2 = np.array([k_norm[-1], inertia_norm[-1]])

        # Find point with maximum distance from line
        distances = []
        for i in range(len(k_values)):
            p = np.array([k_norm[i], inertia_norm[i]])
            d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            distances.append(d)

        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]

    def get_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Args:
            X: Feature matrix

        Returns:
            Dictionary of metric names and values
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        labels = self.model.labels_

        return {
            "n_clusters": self.optimal_k,
            "inertia": self.model.inertia_,
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels),
        }

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centroids.

        Returns:
            Array of cluster centers (n_clusters, n_features)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.cluster_centers_

    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get the number of samples in each cluster.

        Returns:
            Dictionary mapping cluster ID to count
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        labels = self.model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_distance_to_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate distance from each point to all centroids.

        Args:
            X: Feature matrix

        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.transform(X)

    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on distance to assigned centroid.

        Points far from their cluster center are more anomalous.

        Args:
            X: Feature matrix

        Returns:
            Anomaly scores for each sample
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        distances = self.get_distance_to_centroids(X)
        labels = self.model.labels_

        # Get distance to assigned centroid
        assigned_distances = distances[np.arange(len(labels)), labels]

        # Normalize to [0, 1] range
        min_dist = assigned_distances.min()
        max_dist = assigned_distances.max()
        if max_dist > min_dist:
            anomaly_scores = (assigned_distances - min_dist) / (max_dist - min_dist)
        else:
            anomaly_scores = np.zeros_like(assigned_distances)

        return anomaly_scores

    def get_state_dict(self) -> Dict:
        """
        Get state dictionary for saving the model.

        Returns:
            Dictionary containing model state
        """
        return {
            "config": {
                "n_clusters": self.optimal_k,
                "init": self.config.init,
                "n_init": self.config.n_init,
                "max_iter": self.config.max_iter,
            },
            "cluster_centers": self.model.cluster_centers_.tolist() if self.model else None,
            "elbow_data": self.elbow_data,
            "silhouette_data": self.silhouette_data,
        }
