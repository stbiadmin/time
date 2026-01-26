"""
Module: gru_forecaster.py
=========================

GRU-based time series forecasting models for RMA shipping weight prediction.

This module contains three progressive model versions:
    - V1: Simple GRU with numerical features only
    - V2: GRU with categorical embeddings
    - V3: Full model with layer norm, residual connections, and regularization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class GRUForecasterV1(nn.Module):
    """
    Version 1: Simple GRU with numerical features only.

    Architecture:
        - 2-layer stacked GRU with uniform hidden size
        - Final hidden state passed through FC layer for prediction
        - Dropout between layers for regularization
    """

    def __init__(
        self,
        n_numerical_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 7,
    ):
        """
        Initialize the V1 model.

        Args:
            n_numerical_features: Number of numerical input features
            hidden_size: Size of GRU hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
            prediction_horizon: Number of future time steps to predict
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        self.gru = nn.GRU(
            input_size=n_numerical_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, prediction_horizon)

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            numerical: Numerical features (batch, seq_len, n_features)
            categorical: Ignored in V1 (for API consistency)

        Returns:
            Predictions (batch, prediction_horizon)
        """
        # output shape: (batch, seq_len, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        output, h_n = self.gru(numerical)

        # Use the final hidden state from the last layer
        final_hidden = h_n[-1]  # (batch, hidden_size)

        out = self.dropout(final_hidden)
        predictions = self.fc(out)  # (batch, prediction_horizon)

        return predictions


class GRUForecasterV2(nn.Module):
    """
    Version 2: GRU with categorical embeddings.

    Architecture:
        - Embedding layers for each categorical feature
        - Concatenate embeddings with numerical features
        - Same GRU architecture as V1
    """

    def __init__(
        self,
        n_numerical_features: int,
        vocab_sizes: Dict[str, int],
        embedding_dims: Dict[str, int],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 7,
    ):
        """
        Initialize the V2 model.

        Args:
            n_numerical_features: Number of numerical input features
            vocab_sizes: Dictionary of categorical feature vocabulary sizes
            embedding_dims: Dictionary of embedding dimensions per feature
            hidden_size: Size of GRU hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
            prediction_horizon: Number of future time steps to predict
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.categorical_features = list(vocab_sizes.keys())

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dims.get(name, 4),
            )
            for name, vocab_size in vocab_sizes.items()
        })

        total_embedding_dim = sum(embedding_dims.values())

        self.gru = nn.GRU(
            input_size=n_numerical_features + total_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, prediction_horizon)

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            numerical: Numerical features (batch, seq_len, n_numerical)
            categorical: Categorical indices (batch, seq_len, n_categorical)

        Returns:
            Predictions (batch, prediction_horizon)
        """
        batch_size, seq_len, _ = numerical.shape

        # Compute embeddings for each categorical feature
        embedded_features = []
        for i, name in enumerate(self.categorical_features):
            cat_indices = categorical[:, :, i]  # (batch, seq_len)
            embedded = self.embeddings[name](cat_indices)  # (batch, seq_len, embed_dim)
            embedded_features.append(embedded)

        all_embeddings = torch.cat(embedded_features, dim=-1)
        combined = torch.cat([numerical, all_embeddings], dim=-1)

        output, h_n = self.gru(combined)
        final_hidden = h_n[-1]

        out = self.dropout(final_hidden)
        predictions = self.fc(out)

        return predictions


class GRUForecasterV3(nn.Module):
    """
    Version 3: Full model with layer norm, residual connections, and regularization.

    Architecture:
        - Embedding layers for categorical features
        - Layer normalization after GRU
        - Residual connection from average embeddings to output
        - 2-layer MLP output head
    """

    def __init__(
        self,
        n_numerical_features: int,
        vocab_sizes: Dict[str, int],
        embedding_dims: Dict[str, int],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        prediction_horizon: int = 7,
    ):
        """
        Initialize the V3 model.

        Args:
            n_numerical_features: Number of numerical input features
            vocab_sizes: Dictionary of categorical feature vocabulary sizes
            embedding_dims: Dictionary of embedding dimensions per feature
            hidden_size: Size of GRU hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
            prediction_horizon: Number of future time steps to predict
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.categorical_features = list(vocab_sizes.keys())

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dims.get(name, 4),
            )
            for name, vocab_size in vocab_sizes.items()
        })

        total_embedding_dim = sum(embedding_dims.values())

        self.gru = nn.GRU(
            input_size=n_numerical_features + total_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size + total_embedding_dim, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, prediction_horizon)

        self.relu = nn.ReLU()

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            numerical: Numerical features (batch, seq_len, n_numerical)
            categorical: Categorical indices (batch, seq_len, n_categorical)

        Returns:
            Predictions (batch, prediction_horizon)
        """
        batch_size, seq_len, _ = numerical.shape

        # Compute embeddings
        embedded_features = []
        for i, name in enumerate(self.categorical_features):
            cat_indices = categorical[:, :, i]
            embedded = self.embeddings[name](cat_indices)
            embedded_features.append(embedded)

        all_embeddings = torch.cat(embedded_features, dim=-1)

        # Average embeddings for residual connection
        avg_embeddings = all_embeddings.mean(dim=1)  # (batch, total_embed_dim)

        # GRU forward pass
        combined = torch.cat([numerical, all_embeddings], dim=-1)
        output, h_n = self.gru(combined)
        final_hidden = h_n[-1]

        # Layer norm and residual connection
        normalized = self.layer_norm(final_hidden)
        combined_output = torch.cat([normalized, avg_embeddings], dim=-1)

        # Output MLP
        out = self.dropout(combined_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        predictions = self.fc2(out)

        return predictions


def create_model(
    version: str,
    n_numerical_features: int,
    vocab_sizes: Optional[Dict[str, int]] = None,
    config: Optional[Dict] = None,
) -> nn.Module:
    """
    Factory function to create a model by version name.

    Args:
        version: Model version ('v1', 'v2', or 'v3')
        n_numerical_features: Number of numerical input features
        vocab_sizes: Dictionary of categorical vocabulary sizes (for v2/v3)
        config: Model configuration dictionary

    Returns:
        Instantiated model
    """
    config = config or {}

    if version == "v1":
        return GRUForecasterV1(
            n_numerical_features=n_numerical_features,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
            prediction_horizon=config.get("prediction_horizon", 7),
        )
    elif version == "v2":
        if vocab_sizes is None:
            raise ValueError("vocab_sizes required for v2 model")
        embedding_dims = config.get("embedding_dims", {
            "region": 4,
            "sku_category": 8,
            "request_urgency": 2,
            "shipping_method": 2,
        })
        return GRUForecasterV2(
            n_numerical_features=n_numerical_features,
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
            prediction_horizon=config.get("prediction_horizon", 7),
        )
    elif version == "v3":
        if vocab_sizes is None:
            raise ValueError("vocab_sizes required for v3 model")
        embedding_dims = config.get("embedding_dims", {
            "region": 4,
            "sku_category": 8,
            "request_urgency": 2,
            "shipping_method": 2,
        })
        return GRUForecasterV3(
            n_numerical_features=n_numerical_features,
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
            prediction_horizon=config.get("prediction_horizon", 7),
        )
    else:
        raise ValueError(f"Unknown model version: {version}")


def get_model_summary(model: nn.Module) -> Dict:
    """
    Get a summary of model architecture and parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model summary information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_class": model.__class__.__name__,
    }
