"""
Module: training_plots.py
=========================

Purpose: Visualizations for model training progress and comparison.

Business Context:
    Training visualizations help:
    - Monitor convergence during training
    - Identify overfitting (train vs validation divergence)
    - Compare model versions
    - Communicate results to stakeholders

Key training visualizations:
    - Loss curves: Show training convergence
    - Learning rate schedule: Show optimizer behavior
    - Model comparison charts: Demonstrate improvement
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (14, 6)
DPI = 150


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    output_path: Optional[str] = None
) -> None:
    """
    Plot training loss curves.

    This plot shows:
        - Training loss (should decrease)
        - Validation loss (should decrease but may plateau/increase)
        - Gap between them (indicates overfitting if too large)

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        title: Plot title
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # ============ LOSS CURVES ============
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)

    axes[0].set_title("Loss Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mark best epoch
    if "val_loss" in history:
        best_epoch = np.argmin(history["val_loss"]) + 1
        best_loss = min(history["val_loss"])
        axes[0].axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5)
        axes[0].annotate(
            f"Best: {best_loss:.4f}",
            xy=(best_epoch, best_loss),
            xytext=(best_epoch + 2, best_loss + 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10
        )

    # ============ VALIDATION MAE ============
    if "val_mae" in history:
        axes[1].plot(epochs, history["val_mae"], label="Val MAE", linewidth=2, color="green")
        axes[1].set_title("Validation MAE", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    elif "learning_rate" in history:
        axes[1].plot(epochs, history["learning_rate"], label="Learning Rate", linewidth=2, color="orange")
        axes[1].set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "mae",
    title: str = "Model Comparison",
    output_path: Optional[str] = None
) -> None:
    """
    Plot comparison of metrics across model versions.

    This bar chart clearly shows improvement across model versions,
             which is crucial for demonstrating the value of each enhancement.

    Args:
        metrics: Dictionary mapping model name to its metrics
        metric_name: Which metric to compare
        title: Plot title
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    model_names = list(metrics.keys())
    values = [metrics[name].get(metric_name, 0) for name in model_names]

    # Create bar chart
    bars = ax.bar(model_names, values, color='steelblue', edgecolor='black')

    # Color the best model differently
    best_idx = np.argmin(values)
    bars[best_idx].set_color('green')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(values),
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Add improvement annotations
    if len(values) > 1:
        baseline_val = values[0]
        for i, (name, val) in enumerate(zip(model_names[1:], values[1:]), 1):
            if baseline_val > 0:
                improvement = (baseline_val - val) / baseline_val * 100
                ax.annotate(
                    f'{improvement:+.1f}%',
                    xy=(i, val),
                    xytext=(i, val - 0.05 * max(values)),
                    ha='center',
                    fontsize=9,
                    color='green' if improvement > 0 else 'red'
                )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Version')
    ax.set_ylabel(metric_name.upper())
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_multi_metric_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    title: str = "Multi-Metric Model Comparison",
    output_path: Optional[str] = None
) -> None:
    """
    Plot multiple metrics across model versions as grouped bars.

    This allows comparing models across multiple dimensions
             simultaneously, giving a complete picture of performance.

    Args:
        metrics: Dictionary mapping model name to its metrics
        metric_names: List of metrics to compare
        title: Plot title
        output_path: Path to save figure
    """
    if metric_names is None:
        metric_names = ["mae", "rmse", "mape"]

    model_names = list(metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    colors = plt.cm.viridis(np.linspace(0, 0.8, n_models))

    for i, model_name in enumerate(model_names):
        values = [metrics[model_name].get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i], edgecolor='black')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.legend(title='Model Version')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_improvement_summary(
    metrics: Dict[str, Dict[str, float]],
    baseline_name: str = "baseline",
    output_path: Optional[str] = None
) -> None:
    """
    Plot percentage improvement over baseline for each model.

    This visualization clearly communicates the value delivered
             by each model improvement, which is key for business stakeholders.

    Args:
        metrics: Dictionary mapping model name to its metrics
        baseline_name: Name of the baseline model
        output_path: Path to save figure
    """
    if baseline_name not in metrics:
        print(f"Baseline '{baseline_name}' not found in metrics")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    baseline_mae = metrics[baseline_name]["mae"]
    improvements = {}

    for model_name, model_metrics in metrics.items():
        if model_name == baseline_name:
            improvements[model_name] = 0
        else:
            improvement = (baseline_mae - model_metrics["mae"]) / baseline_mae * 100
            improvements[model_name] = improvement

    model_names = list(improvements.keys())
    values = list(improvements.values())

    colors = ['gray' if v == 0 else ('green' if v > 0 else 'red') for v in values]
    bars = ax.bar(model_names, values, color=colors, edgecolor='black')

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{value:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_title(f'MAE Improvement vs {baseline_name.title()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Version')
    ax.set_ylabel('Improvement (%)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    # Add target line if applicable
    ax.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='35% Target')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()
