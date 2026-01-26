"""
Module: rma_trainer.py
======================

Training pipeline for GRU-based RMA shipping weight forecasting models.

Implements:
    - Separate train/validation loops with proper mode switching
    - Learning rate scheduling for better convergence
    - Early stopping to prevent overfitting
    - Gradient clipping for training stability
"""

import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils.helpers import EarlyStopping, format_time, get_device
from ..utils.logging_config import get_logger, log_section, log_metrics


class RMATrainer:
    """
    Trainer for RMA forecasting models.

    Handles the complete training lifecycle including training loop,
    validation, learning rate scheduling, early stopping, and checkpointing.

    Attributes:
        model: PyTorch model to train
        device: Compute device (CPU/GPU/MPS)
        optimizer: Adam optimizer
        scheduler: Learning rate scheduler
        early_stopping: Early stopping utility
        history: Training history (losses, metrics)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Compute device (auto-detected if None)
        """
        self.logger = get_logger()
        self.device = device or get_device()
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Loss function
        loss_fn = config.get("loss_function", "mse")
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        elif loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.get("learning_rate", 0.001),
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.get("lr_scheduler", {}).get("factor", 0.5),
            patience=config.get("lr_scheduler", {}).get("patience", 5),
            verbose=True,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 10),
            mode="min",
        )

        # Training state
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "learning_rate": [],
        }
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train(self, max_epochs: int = 50) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Args:
            max_epochs: Maximum number of training epochs

        Returns:
            Training history dictionary
        """
        log_section(self.logger, "STARTING TRAINING")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max epochs: {max_epochs}")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss, val_mae = self._validate()

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["learning_rate"].append(current_lr)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # Log progress
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping check
            if self.early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Restored best model with val_loss: {self.best_val_loss:.4f}")

        total_time = time.time() - start_time
        log_section(self.logger, f"TRAINING COMPLETE ({format_time(total_time)})")

        return self.history

    def _train_epoch(self) -> float:
        """
        Run one training epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            numerical = batch["numerical"].to(self.device)
            categorical = batch["categorical"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(numerical, categorical)

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config.get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            # Weight update
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate(self) -> Tuple[float, float]:
        """
        Run validation.

        Returns:
            Tuple of (average loss, average MAE)
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                numerical = batch["numerical"].to(self.device)
                categorical = batch["categorical"].to(self.device)
                targets = batch["target"].to(self.device)

                predictions = self.model(numerical, categorical)

                loss = self.criterion(predictions, targets)
                mae = torch.abs(predictions - targets).mean()

                total_loss += loss.item()
                total_mae += mae.item()
                n_batches += 1

        return total_loss / n_batches, total_mae / n_batches

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for prediction data

        Returns:
            Tuple of (predictions, actual_targets) as numpy arrays
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                numerical = batch["numerical"].to(self.device)
                categorical = batch["categorical"].to(self.device)
                targets = batch["target"]

                predictions = self.model(numerical, categorical)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        return np.vstack(all_predictions), np.vstack(all_targets)

    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.

        Args:
            path: Path to save the checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load a training checkpoint.

        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Checkpoint loaded from: {path}")


def compute_baseline_metrics(
    train_loader: DataLoader,
    test_loader: DataLoader,
    prediction_horizon: int = 7
) -> Dict[str, float]:
    """
    Compute baseline metrics using naive persistence.

    The naive persistence baseline predicts that future values
    will equal the most recent observed values.

    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        prediction_horizon: Number of steps to predict

    Returns:
        Dictionary of baseline metrics
    """
    all_targets = []
    all_last_values = []

    for batch in test_loader:
        numerical = batch["numerical"]
        targets = batch["target"]

        # Use the last value repeated as baseline prediction
        last_values = numerical[:, -1, 0]
        baseline_pred = last_values.unsqueeze(1).repeat(1, prediction_horizon)

        all_targets.append(targets.numpy())
        all_last_values.append(baseline_pred.numpy())

    targets = np.vstack(all_targets)
    predictions = np.vstack(all_last_values)

    mae = np.abs(predictions - targets).mean()
    mse = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }
