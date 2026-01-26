"""
Module: model_registry.py
=========================

Purpose: Model serialization, versioning, and registry management.

Business Context:
    Production ML systems need robust model management:
    - Reproducible model artifacts
    - Version tracking
    - Metadata for deployment decisions
    - Easy loading for inference

This module handles the complete model lifecycle:
    - Saving models with all dependencies (weights, config, preprocessors)
    - Loading models for inference
    - Version management and metadata tracking
"""

import json
import torch
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    """Metadata for a saved model."""
    model_name: str
    model_version: str
    model_type: str  # 'rma_gru' or 'network_clustering'
    created_at: str
    training_metrics: Dict[str, float]
    config: Dict[str, Any]
    description: str = ""


class ModelRegistry:
    """
    Registry for managing ML model artifacts.

    The registry provides a standardized way to:
        - Save models with all dependencies
        - Load models for inference
        - Track model versions and metadata
        - Ensure reproducibility

    Attributes:
        registry_path: Base directory for model storage

    Example:
        >>> registry = ModelRegistry("outputs/models")
        >>> registry.save_rma_model(model, preprocessor, metrics, "v3")
        >>> loaded_model, preprocessor = registry.load_rma_model("v3")
    """

    def __init__(self, registry_path: str = "outputs/models"):
        """
        Initialize the registry.

        Args:
            registry_path: Base directory for storing models
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def save_rma_model(
        self,
        model: torch.nn.Module,
        preprocessor_state: Dict,
        training_history: Dict,
        metrics: Dict[str, float],
        config: Dict,
        version: str = "v1",
        description: str = ""
    ) -> str:
        """
        Save an RMA forecasting model with all dependencies.

        We save everything needed to recreate the model:
            - Model weights (state_dict)
            - Preprocessor state (encoders, scalers)
            - Training history
            - Configuration
            - Metadata

        Args:
            model: Trained PyTorch model
            preprocessor_state: Preprocessor state dictionary
            training_history: Training loss history
            metrics: Final evaluation metrics
            config: Model configuration
            version: Version string
            description: Model description

        Returns:
            Path to saved model directory
        """
        # Create version directory
        model_dir = self.registry_path / f"rma_gru_{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============ SAVE MODEL WEIGHTS ============
        torch.save(
            model.state_dict(),
            model_dir / "model_weights.pt"
        )

        # ============ SAVE PREPROCESSOR ============
        joblib.dump(preprocessor_state, model_dir / "preprocessor.joblib")

        # ============ SAVE TRAINING HISTORY ============
        with open(model_dir / "training_history.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {
                k: [float(v) for v in vals] if isinstance(vals, list) else vals
                for k, vals in training_history.items()
            }
            json.dump(history_serializable, f, indent=2)

        # ============ SAVE CONFIG ============
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # ============ SAVE METADATA ============
        metadata = ModelMetadata(
            model_name=f"rma_gru_{version}",
            model_version=version,
            model_type="rma_gru",
            created_at=datetime.now().isoformat(),
            training_metrics=metrics,
            config=config,
            description=description,
        )

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        print(f"Model saved to: {model_dir}")
        return str(model_dir)

    def load_rma_model(
        self,
        version: str = "v3"
    ) -> Dict[str, Any]:
        """
        Load an RMA forecasting model.

        Args:
            version: Model version to load

        Returns:
            Dictionary containing model artifacts
        """
        model_dir = self.registry_path / f"rma_gru_{version}"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Load all artifacts
        artifacts = {
            "weights": torch.load(model_dir / "model_weights.pt", map_location="cpu"),
            "preprocessor": joblib.load(model_dir / "preprocessor.joblib"),
            "config": json.load(open(model_dir / "config.json")),
            "metadata": json.load(open(model_dir / "metadata.json")),
        }

        if (model_dir / "training_history.json").exists():
            artifacts["history"] = json.load(open(model_dir / "training_history.json"))

        print(f"Model loaded from: {model_dir}")
        return artifacts

    def save_clustering_model(
        self,
        clusterer_state: Dict,
        preprocessor_state: Dict,
        metrics: Dict[str, float],
        cluster_interpretations: Dict[int, str],
        version: str = "v1",
        description: str = ""
    ) -> str:
        """
        Save a clustering pipeline with all dependencies.

        For clustering, we save:
            - K-means model state (centroids, config)
            - TF-IDF vectorizer and LSA model
            - Preprocessor state (scalers)
            - Cluster interpretations

        Args:
            clusterer_state: K-means state dictionary
            preprocessor_state: Preprocessor state dictionary
            metrics: Clustering metrics
            cluster_interpretations: Human-readable cluster labels
            version: Version string
            description: Model description

        Returns:
            Path to saved model directory
        """
        model_dir = self.registry_path / f"network_clustering_{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============ SAVE CLUSTERER STATE ============
        joblib.dump(clusterer_state, model_dir / "clusterer.joblib")

        # ============ SAVE PREPROCESSOR ============
        joblib.dump(preprocessor_state, model_dir / "preprocessor.joblib")

        # ============ SAVE INTERPRETATIONS ============
        with open(model_dir / "cluster_interpretations.json", "w") as f:
            json.dump(cluster_interpretations, f, indent=2)

        # ============ SAVE METADATA ============
        metadata = ModelMetadata(
            model_name=f"network_clustering_{version}",
            model_version=version,
            model_type="network_clustering",
            created_at=datetime.now().isoformat(),
            training_metrics=metrics,
            config={},
            description=description,
        )

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        print(f"Clustering model saved to: {model_dir}")
        return str(model_dir)

    def load_clustering_model(
        self,
        version: str = "v1"
    ) -> Dict[str, Any]:
        """
        Load a clustering model.

        Args:
            version: Model version to load

        Returns:
            Dictionary containing model artifacts
        """
        model_dir = self.registry_path / f"network_clustering_{version}"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        artifacts = {
            "clusterer": joblib.load(model_dir / "clusterer.joblib"),
            "preprocessor": joblib.load(model_dir / "preprocessor.joblib"),
            "interpretations": json.load(open(model_dir / "cluster_interpretations.json")),
            "metadata": json.load(open(model_dir / "metadata.json")),
        }

        print(f"Clustering model loaded from: {model_dir}")
        return artifacts

    def save_prophet_model(
        self,
        model: Any,
        preprocessor_state: Dict,
        training_history: Dict,
        metrics: Dict[str, float],
        config: Dict,
        version: str = "v1",
        description: str = ""
    ) -> str:
        """
        Save a Prophet forecasting model with all dependencies.

        Prophet models use joblib instead of torch.save since
        Prophet is scikit-learn compatible, not a PyTorch module.

        Args:
            model: Trained ProphetForecaster instance
            preprocessor_state: Preprocessor state dictionary
            training_history: Training results (fit time, cv metrics)
            metrics: Final evaluation metrics
            config: Model configuration
            version: Version string
            description: Model description

        Returns:
            Path to saved model directory
        """
        # Create version directory
        model_dir = self.registry_path / f"rma_prophet_{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============ SAVE PROPHET MODEL ============
        # Prophet models are serialized with joblib
        joblib.dump(model, model_dir / "model.joblib")

        # ============ SAVE PREPROCESSOR ============
        joblib.dump(preprocessor_state, model_dir / "preprocessor.joblib")

        # ============ SAVE TRAINING HISTORY ============
        with open(model_dir / "training_history.json", "w") as f:
            # Convert to JSON-serializable format
            history_serializable = {}
            for k, v in training_history.items():
                if isinstance(v, dict):
                    history_serializable[k] = {
                        str(k2): float(v2) if isinstance(v2, (int, float)) else v2
                        for k2, v2 in v.items()
                    }
                elif isinstance(v, (int, float)):
                    history_serializable[k] = float(v)
                else:
                    history_serializable[k] = v
            json.dump(history_serializable, f, indent=2)

        # ============ SAVE CONFIG ============
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # ============ SAVE METADATA ============
        metadata = ModelMetadata(
            model_name=f"rma_prophet_{version}",
            model_version=version,
            model_type="rma_prophet",
            created_at=datetime.now().isoformat(),
            training_metrics=metrics,
            config=config,
            description=description,
        )

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        print(f"Prophet model saved to: {model_dir}")
        return str(model_dir)

    def load_prophet_model(
        self,
        version: str = "v1"
    ) -> Dict[str, Any]:
        """
        Load a Prophet forecasting model.

        Args:
            version: Model version to load

        Returns:
            Dictionary containing model artifacts
        """
        model_dir = self.registry_path / f"rma_prophet_{version}"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Load all artifacts
        artifacts = {
            "model": joblib.load(model_dir / "model.joblib"),
            "preprocessor": joblib.load(model_dir / "preprocessor.joblib"),
            "config": json.load(open(model_dir / "config.json")),
            "metadata": json.load(open(model_dir / "metadata.json")),
        }

        if (model_dir / "training_history.json").exists():
            artifacts["history"] = json.load(open(model_dir / "training_history.json"))

        print(f"Prophet model loaded from: {model_dir}")
        return artifacts

    def list_models(self) -> Dict[str, list]:
        """
        List all available models in the registry.

        Returns:
            Dictionary of model types to versions
        """
        models = {"rma_gru": [], "rma_prophet": [], "network_clustering": []}

        for path in self.registry_path.iterdir():
            if path.is_dir():
                name = path.name
                if name.startswith("rma_gru_"):
                    version = name.replace("rma_gru_", "")
                    models["rma_gru"].append(version)
                elif name.startswith("rma_prophet_"):
                    version = name.replace("rma_prophet_", "")
                    models["rma_prophet"].append(version)
                elif name.startswith("network_clustering_"):
                    version = name.replace("network_clustering_", "")
                    models["network_clustering"].append(version)

        return models

    def get_model_metadata(
        self,
        model_type: str,
        version: str
    ) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model.

        Args:
            model_type: 'rma_gru' or 'network_clustering'
            version: Model version

        Returns:
            ModelMetadata object or None if not found
        """
        model_dir = self.registry_path / f"{model_type}_{version}"

        if not model_dir.exists():
            return None

        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        return ModelMetadata(**data)
