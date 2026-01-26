"""
Module: inference.py
====================

Purpose: Inference wrapper classes for production model serving.

Business Context:
    Production inference requires:
    - Consistent interface regardless of model version
    - Preprocessing integration
    - Error handling
    - Batch and single-item inference

These inference engines wrap the trained models with:
    - Input validation
    - Preprocessing
    - Model inference
    - Postprocessing
    - Output formatting
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path

from .model_registry import ModelRegistry


class RMAInferenceEngine:
    """
    Inference engine for RMA shipping weight forecasting.

    This class provides a clean interface for making predictions:
        - Handles preprocessing automatically
        - Supports both single and batch predictions
        - Returns structured output with confidence information

    Attributes:
        model: Loaded PyTorch model
        preprocessor: Preprocessor state for encoding/scaling
        config: Model configuration
        device: Compute device

    Example:
        >>> engine = RMAInferenceEngine("outputs/models", version="v3")
        >>> forecast = engine.predict(historical_data)
    """

    def __init__(
        self,
        registry_path: str = "outputs/models",
        version: str = "v3"
    ):
        """
        Initialize the inference engine.

        Args:
            registry_path: Path to model registry
            version: Model version to load
        """
        self.registry = ModelRegistry(registry_path)
        self.version = version
        self.device = torch.device("cpu")  # Use CPU for inference

        # Load model artifacts
        artifacts = self.registry.load_rma_model(version)

        self.weights = artifacts["weights"]
        self.preprocessor_state = artifacts["preprocessor"]
        self.config = artifacts["config"]
        self.metadata = artifacts["metadata"]

        # Reconstruct model
        self.model = self._build_model()
        self.model.load_state_dict(self.weights)
        self.model.eval()
        self.model.to(self.device)

    def _build_model(self) -> torch.nn.Module:
        """
        Build the model from saved configuration.

        Returns:
            Instantiated model
        """
        from src.models.gru_forecaster import create_model

        # Get vocab sizes from preprocessor
        vocab_sizes = {}
        if "label_encoders" in self.preprocessor_state:
            for col, classes in self.preprocessor_state["label_encoders"].items():
                vocab_sizes[col] = len(classes)

        model = create_model(
            version=self.version,
            n_numerical_features=len(self.preprocessor_state.get("numerical_features", [])),
            vocab_sizes=vocab_sizes,
            config=self.config,
        )

        return model

    def predict(
        self,
        data: pd.DataFrame,
        return_confidence: bool = False
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Generate forecast predictions.

        The prediction flow:
            1. Validate input data
            2. Preprocess (encode categoricals, scale numericals)
            3. Create sequence tensor
            4. Run model inference
            5. Format output

        Args:
            data: DataFrame with historical data (last sequence_length days)
            return_confidence: Whether to include confidence intervals

        Returns:
            Dictionary with predictions and optional metadata
        """
        # Validate input
        required_cols = ["region", "sku_category", "request_urgency", "shipping_method",
                        "avg_repair_cycle_days", "failure_rate_pct", "day_of_week", "month"]

        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Preprocess
        processed = self._preprocess(data)

        # Run inference
        with torch.no_grad():
            predictions = self.model(
                processed["numerical"],
                processed["categorical"]
            )

        predictions = predictions.cpu().numpy()

        result = {
            "predictions": predictions,
            "prediction_horizon": predictions.shape[1],
            "model_version": self.version,
        }

        if return_confidence:
            # Simple confidence based on training metrics
            mae = self.metadata.get("training_metrics", {}).get("val_mae", 0)
            result["confidence_interval"] = {
                "lower": predictions - 1.96 * mae,
                "upper": predictions + 1.96 * mae,
            }

        return result

    def _preprocess(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Preprocess input data using saved preprocessor state.

        Args:
            data: Raw input DataFrame

        Returns:
            Dictionary with tensors for model input
        """
        data = data.copy()

        # Encode categoricals
        categorical_cols = self.preprocessor_state.get("categorical_features", [])
        encoded_categorical = []

        for col in categorical_cols:
            if col in data.columns and col in self.preprocessor_state.get("label_encoders", {}):
                classes = self.preprocessor_state["label_encoders"][col]
                # Map values to indices
                mapping = {c: i for i, c in enumerate(classes)}
                encoded = data[col].astype(str).map(mapping).fillna(0).astype(int)
                encoded_categorical.append(encoded.values)

        if encoded_categorical:
            categorical_tensor = torch.tensor(
                np.column_stack(encoded_categorical)
            ).unsqueeze(0).long()
        else:
            categorical_tensor = torch.zeros((1, len(data), 1)).long()

        # Scale numericals
        numerical_cols = self.preprocessor_state.get("numerical_features", [])
        numerical_values = data[numerical_cols].values

        scaler_mean = np.array(self.preprocessor_state.get("scaler_mean", [0] * len(numerical_cols)))
        scaler_std = np.array(self.preprocessor_state.get("scaler_std", [1] * len(numerical_cols)))

        scaled = (numerical_values - scaler_mean) / scaler_std
        numerical_tensor = torch.tensor(scaled).unsqueeze(0).float()

        return {
            "categorical": categorical_tensor,
            "numerical": numerical_tensor,
        }


class ClusteringInferenceEngine:
    """
    Inference engine for network event clustering.

    This class provides:
        - Event classification into discovered clusters
        - Anomaly scoring
        - Similar event retrieval

    Attributes:
        clusterer: K-means model state
        preprocessor: TF-IDF, LSA, scaler state
        interpretations: Human-readable cluster labels

    Example:
        >>> engine = ClusteringInferenceEngine("outputs/models", version="v1")
        >>> result = engine.classify(event_data)
    """

    def __init__(
        self,
        registry_path: str = "outputs/models",
        version: str = "v1"
    ):
        """
        Initialize the inference engine.

        Args:
            registry_path: Path to model registry
            version: Model version to load
        """
        self.registry = ModelRegistry(registry_path)
        self.version = version

        # Load artifacts
        artifacts = self.registry.load_clustering_model(version)

        self.clusterer_state = artifacts["clusterer"]
        self.preprocessor_state = artifacts["preprocessor"]
        self.interpretations = artifacts["interpretations"]
        self.metadata = artifacts["metadata"]

        # Reconstruct sklearn objects
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """
        Reconstruct sklearn pipeline from saved state.
        """
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler

        # Reconstruct KMeans
        centroids = np.array(self.clusterer_state.get("cluster_centers", []))
        n_clusters = len(centroids)

        self.kmeans = KMeans(n_clusters=n_clusters)
        # Fake fit by setting attributes
        self.kmeans.cluster_centers_ = centroids
        self.kmeans._n_threads = 1

        # Reconstructing TF-IDF and SVD would require fitting again
        # For demo purposes, we store the preprocessor state
        self.scaler_mean = np.array(self.preprocessor_state.get("scaler_mean", []))
        self.scaler_std = np.array(self.preprocessor_state.get("scaler_std", []))

    def classify(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Union[np.ndarray, List[str], Dict]]:
        """
        Classify network events into clusters.

        For new events:
            1. Preprocess (normalize, TF-IDF would need full pipeline)
            2. Assign to nearest cluster
            3. Compute anomaly score
            4. Return cluster label and interpretation

        Args:
            data: DataFrame with event features

        Returns:
            Dictionary with classification results
        """
        # Simplified inference using cluster centroids
        # In production, would need full preprocessing pipeline

        # Extract numerical features
        numerical_cols = self.preprocessor_state.get("numerical_features", [])

        if not all(col in data.columns for col in numerical_cols):
            raise ValueError(f"Missing numerical columns: {numerical_cols}")

        numerical_values = data[numerical_cols].values

        # Scale
        if len(self.scaler_mean) > 0:
            scaled = (numerical_values - self.scaler_mean) / self.scaler_std
        else:
            scaled = numerical_values

        # For demo, use only numerical features (simplified)
        # Would need full LSA pipeline for text features
        n_features = self.kmeans.cluster_centers_.shape[1]
        if scaled.shape[1] < n_features:
            # Pad with zeros for missing LSA features
            padding = np.zeros((len(scaled), n_features - scaled.shape[1]))
            scaled = np.hstack([scaled, padding])

        # Assign clusters
        labels = self.kmeans.predict(scaled[:, :n_features])

        # Compute distances
        distances = self.kmeans.transform(scaled[:, :n_features])
        min_distances = distances[np.arange(len(labels)), labels]

        # Anomaly scores (normalized distance)
        max_dist = min_distances.max() if min_distances.max() > 0 else 1
        anomaly_scores = min_distances / max_dist

        # Get interpretations
        cluster_labels = [
            self.interpretations.get(str(l), f"Cluster {l}")
            for l in labels
        ]

        return {
            "cluster_ids": labels,
            "cluster_labels": cluster_labels,
            "anomaly_scores": anomaly_scores,
            "model_version": self.version,
        }

    def get_cluster_info(self) -> Dict[int, Dict]:
        """
        Get information about each cluster.

        Returns:
            Dictionary of cluster ID to cluster info
        """
        n_clusters = len(self.kmeans.cluster_centers_)

        info = {}
        for i in range(n_clusters):
            info[i] = {
                "interpretation": self.interpretations.get(str(i), f"Cluster {i}"),
                "centroid": self.kmeans.cluster_centers_[i].tolist(),
            }

        return info
