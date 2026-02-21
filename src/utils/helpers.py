"""
Module: helpers.py
==================

Purpose: Common utility functions used across the ML demonstration pipeline.

These helper functions encapsulate common operations like setting
         random seeds for reproducibility, detecting compute devices, and
         managing file paths. Centralizing these reduces code duplication.

Business Context:
    Reproducibility is critical in production ML. These utilities ensure
    consistent behavior across different runs and environments.
"""

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    ML experiments should be reproducible. By setting seeds for
             Python's random module, NumPy, and PyTorch, we ensure that
             the same code produces the same results across runs.

    WHY THIS MATTERS:
        - Debugging: Same inputs produce same outputs
        - Validation: Results can be verified by others
        - Comparison: Fair comparison between model versions

    Args:
        seed: Random seed value (default: 42)

    Example:
        >>> set_seed(42)
        >>> np.random.rand()  # Will always produce the same value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ============ PYTORCH SPECIFIC SETTINGS ============
    # NOTE: These settings ensure deterministic behavior in PyTorch,
    #        though they may slightly reduce performance.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have the same seed mechanism, but manual_seed covers it
        pass

    # Ensure deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Get the best available compute device for PyTorch.

    This function checks for available hardware accelerators
             in order of preference: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The best available device

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Example:
        >>> ensure_dir("outputs/models")
        PosixPath('outputs/models')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Centralized configuration management ensures consistency
             across all pipeline stages and makes hyperparameter tuning
             easier to track and reproduce.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing all configuration settings

    Example:
        >>> config = load_config()
        >>> batch_size = config['training']['rma']['batch_size']
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary to save
        path: Output file path
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base, overlay wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_with_preset(
    config_path: str = "config/settings.yaml",
    preset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load base config, optionally deep-merge a preset from config/presets/<name>.yaml."""
    config = load_config(config_path)
    if preset_name:
        preset_path = Path("config/presets") / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_path}")
        with open(preset_path, "r") as f:
            preset = yaml.safe_load(f) or {}
        # Only merge data_generation keys from presets
        preset_data = {k: v for k, v in preset.items() if k != "description"}
        config = deep_merge(config, preset_data)
    return config


def add_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --dataset-name, --rma-data, --network-data args to any script's parser."""
    parser.add_argument(
        "--dataset-name", "-d",
        help="Dataset name: loads data/raw/rma_<name>.csv and data/raw/network_<name>.csv",
    )
    parser.add_argument(
        "--rma-data",
        help="Explicit path to RMA CSV (overrides --dataset-name)",
    )
    parser.add_argument(
        "--network-data",
        help="Explicit path to network events CSV (overrides --dataset-name)",
    )
    return parser


def resolve_data_paths(args, data_dir: str = "data/raw") -> Tuple[str, str]:
    """Resolve (rma_path, network_path) from parsed CLI args."""
    data_dir = Path(data_dir)
    if getattr(args, "dataset_name", None):
        rma_path = str(data_dir / f"rma_{args.dataset_name}.csv")
        network_path = str(data_dir / f"network_{args.dataset_name}.csv")
    else:
        rma_path = str(data_dir / "rma_shipping_data.csv")
        network_path = str(data_dir / "network_events.csv")

    if getattr(args, "rma_data", None):
        rma_path = args.rma_data
    if getattr(args, "network_data", None):
        network_path = args.network_data

    return rma_path, network_path


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2m 30s" or "1h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a PyTorch model.

    Understanding model complexity is important for:
             - Estimating training time
             - Assessing overfitting risk
             - Comparing model architectures

    Args:
        model: PyTorch model

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts

    Example:
        >>> params = count_parameters(model)
        >>> print(f"Trainable parameters: {params['trainable']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.

    Early stopping monitors a validation metric and stops training
             when the metric stops improving. This prevents overfitting and
             saves computation time.

    WHY THIS MATTERS:
        - Prevents overfitting by stopping before the model memorizes training data
        - Saves training time by not running unnecessary epochs
        - Automatically selects the best model checkpoint

    Attributes:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy

    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='min')
        >>> for epoch in range(max_epochs):
        ...     val_loss = train_epoch(model)
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

        # ============ SET COMPARISON FUNCTION ============
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current validation metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
