"""
Module: clustering_trainer.py
=============================

Training pipeline for K-means clustering and LSA on network events.

Orchestrates the complete unsupervised learning pipeline: preprocessing,
feature combination, K-means clustering, and cluster interpretation.
"""

import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from ..models.kmeans_clusterer import KMeansClusterer, KMeansConfig
from ..models.lsa_analyzer import LSAAnalyzer, LSAConfig
from ..preprocessing.network_preprocessor import NetworkPreprocessor, NetworkDataConfig
from ..utils.logging_config import get_logger, log_section


class ClusteringTrainer:
    """
    Trainer for network event clustering pipeline.

    Coordinates the unsupervised learning pipeline: NetworkPreprocessor handles
    feature engineering, and KMeansClusterer finds natural groupings. Also
    handles cluster interpretation and evaluation.

    Attributes:
        preprocessor: Network event preprocessor
        clusterer: K-means clustering model
        cluster_labels: Discovered cluster labels
        cluster_interpretations: Human-readable cluster descriptions
    """

    def __init__(
        self,
        preprocessor_config: Optional[NetworkDataConfig] = None,
        kmeans_config: Optional[KMeansConfig] = None,
    ):
        """
        Initialize the clustering trainer.

        Args:
            preprocessor_config: Configuration for preprocessing
            kmeans_config: Configuration for K-means
        """
        self.logger = get_logger()

        self.preprocessor = NetworkPreprocessor(preprocessor_config)
        self.clusterer = KMeansClusterer(kmeans_config)

        self.features: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_interpretations: Dict[int, str] = {}
        self.df: Optional[pd.DataFrame] = None

    def fit(
        self,
        df: pd.DataFrame,
        find_optimal_k: bool = True
    ) -> np.ndarray:
        """
        Fit the complete clustering pipeline.

        Args:
            df: Network events DataFrame
            find_optimal_k: Whether to search for optimal K

        Returns:
            Cluster labels for each event
        """
        log_section(self.logger, "CLUSTERING PIPELINE")
        start_time = time.time()

        self.df = df.copy()

        self.logger.info("Step 1: Preprocessing features...")
        self.features, processed_df = self.preprocessor.fit_transform(df)
        self.logger.info(f"  Feature matrix shape: {self.features.shape}")

        self.logger.info("Step 2: Running K-means clustering...")
        self.cluster_labels = self.clusterer.fit_predict(
            self.features,
            find_optimal_k=find_optimal_k
        )
        self.logger.info(f"  Optimal K: {self.clusterer.optimal_k}")

        self.logger.info("Step 3: Interpreting clusters...")
        self._interpret_clusters()

        elapsed = time.time() - start_time
        self.logger.info(f"\nClustering complete in {elapsed:.2f}s")
        self.logger.info(f"Cluster distribution:")
        for cluster_id, count in self.clusterer.get_cluster_sizes().items():
            interpretation = self.cluster_interpretations.get(cluster_id, "Unknown")
            self.logger.info(f"  Cluster {cluster_id}: {count:,} events - {interpretation}")

        return self.cluster_labels

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            df: New events DataFrame

        Returns:
            Cluster labels for each event
        """
        if self.features is None:
            raise ValueError("Pipeline must be fitted before prediction")

        features = self.preprocessor.transform(df)
        return self.clusterer.predict(features)

    def _interpret_clusters(self) -> None:
        """
        Interpret clusters based on their characteristics.

        Analyzes each cluster to create human-readable labels based on
        dominant numerical feature values and severity distribution.
        """
        if self.df is None or self.cluster_labels is None:
            return

        df_with_clusters = self.df.copy()
        df_with_clusters["cluster"] = self.cluster_labels

        for cluster_id in range(self.clusterer.optimal_k):
            cluster_data = df_with_clusters[df_with_clusters["cluster"] == cluster_id]

            if len(cluster_data) == 0:
                self.cluster_interpretations[cluster_id] = "Empty cluster"
                continue

            interpretation_parts = []

            # Severity analysis
            severity_dist = cluster_data["severity"].value_counts(normalize=True)
            dominant_severity = severity_dist.index[0]
            if severity_dist.iloc[0] > 0.5:
                interpretation_parts.append(f"{dominant_severity} severity")

            # Duration analysis
            avg_duration = cluster_data["duration_ms"].mean()
            if avg_duration > 1000:
                interpretation_parts.append("long duration")
            elif avg_duration < 50:
                interpretation_parts.append("short duration")

            # Bytes analysis
            avg_bytes = cluster_data["bytes_transferred"].mean()
            if avg_bytes > 100000:
                interpretation_parts.append("high data transfer")
            elif avg_bytes < 1000:
                interpretation_parts.append("low data transfer")

            # Time analysis
            avg_hour = cluster_data["hour_of_day"].mean()
            if avg_hour < 6 or avg_hour > 22:
                interpretation_parts.append("off-hours activity")

            if interpretation_parts:
                self.cluster_interpretations[cluster_id] = ", ".join(interpretation_parts)
            else:
                self.cluster_interpretations[cluster_id] = "General traffic"

    def get_metrics(self) -> Dict[str, float]:
        """
        Get clustering quality metrics.

        Returns:
            Dictionary of metrics
        """
        if self.features is None:
            return {}
        return self.clusterer.get_metrics(self.features)

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of cluster characteristics.

        Returns:
            DataFrame with cluster statistics
        """
        if self.df is None or self.cluster_labels is None:
            return pd.DataFrame()

        df_with_clusters = self.df.copy()
        df_with_clusters["cluster"] = self.cluster_labels

        summary_data = []
        for cluster_id in range(self.clusterer.optimal_k):
            cluster_data = df_with_clusters[df_with_clusters["cluster"] == cluster_id]

            if len(cluster_data) == 0:
                continue

            summary = {
                "cluster_id": cluster_id,
                "count": len(cluster_data),
                "pct_of_total": len(cluster_data) / len(df_with_clusters) * 100,
                "avg_duration_ms": cluster_data["duration_ms"].mean(),
                "avg_bytes": cluster_data["bytes_transferred"].mean(),
                "dominant_severity": cluster_data["severity"].mode().iloc[0] if len(cluster_data) > 0 else "unknown",
                "dominant_protocol": cluster_data["protocol"].mode().iloc[0] if len(cluster_data) > 0 else "unknown",
                "interpretation": self.cluster_interpretations.get(cluster_id, "Unknown"),
            }
            summary_data.append(summary)

        return pd.DataFrame(summary_data)

    def evaluate_against_ground_truth(self) -> Dict[str, float]:
        """
        Evaluate clustering against ground truth labels.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.df is None or self.cluster_labels is None:
            return {}

        if "true_cluster" not in self.df.columns:
            self.logger.warning("No ground truth labels available")
            return {}

        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            homogeneity_score,
            completeness_score,
            v_measure_score,
        )

        true_labels = self.df["true_cluster"].values

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_labels_encoded = le.fit_transform(true_labels)

        metrics = {
            "adjusted_rand_score": adjusted_rand_score(true_labels_encoded, self.cluster_labels),
            "normalized_mutual_info": normalized_mutual_info_score(true_labels_encoded, self.cluster_labels),
            "homogeneity": homogeneity_score(true_labels_encoded, self.cluster_labels),
            "completeness": completeness_score(true_labels_encoded, self.cluster_labels),
            "v_measure": v_measure_score(true_labels_encoded, self.cluster_labels),
        }

        return metrics

    def get_anomaly_scores(self) -> np.ndarray:
        """
        Get anomaly scores for all events.

        Events far from their cluster center may be anomalous.

        Returns:
            Array of anomaly scores
        """
        if self.features is None:
            return np.array([])
        return self.clusterer.get_anomaly_scores(self.features)

    def save_results(self, output_dir: str) -> None:
        """
        Save clustering results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.df is not None and self.cluster_labels is not None:
            results_df = self.df.copy()
            results_df["predicted_cluster"] = self.cluster_labels
            results_df["anomaly_score"] = self.get_anomaly_scores()
            results_df.to_csv(output_path / "cluster_assignments.csv", index=False)

        summary_df = self.get_cluster_summary()
        summary_df.to_csv(output_path / "cluster_summary.csv", index=False)

        import json
        metrics = self.get_metrics()
        with open(output_path / "clustering_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Results saved to: {output_path}")

    def get_state_dict(self) -> Dict:
        """
        Get state dictionary for saving the pipeline.

        Returns:
            Dictionary containing pipeline state
        """
        return {
            "preprocessor": self.preprocessor.get_state_dict(),
            "clusterer": self.clusterer.get_state_dict(),
            "cluster_interpretations": self.cluster_interpretations,
        }
