"""
Module: forecast_plots.py
=========================

Purpose: Visualizations for time series forecasting results.

Business Context:
    Forecast visualizations help stakeholders understand:
    - How well the model captures patterns
    - Where predictions deviate from reality
    - The reliability of forecasts at different horizons

Key forecast visualizations:
    - Predictions vs actuals: Shows fit quality
    - Error distribution: Reveals bias and variance
    - Per-horizon accuracy: Shows how accuracy degrades with time
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path


plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (14, 6)
FIGSIZE_LARGE = (14, 10)
DPI = 150


def plot_forecast_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model",
    output_path: Optional[str] = None
) -> None:
    """
    Create comprehensive forecast result visualizations.

    This function creates multiple views of forecast quality:
        1. Time series plot: Predictions overlaid on actuals
        2. Scatter plot: Predicted vs actual values
        3. Error distribution: Histogram of prediction errors
        4. Per-horizon analysis: How accuracy changes with horizon

    Args:
        predictions: Model predictions (n_samples, horizon)
        targets: Ground truth values (n_samples, horizon)
        dates: Optional date index for x-axis
        model_name: Name of the model for titles
        output_path: Base path for saving figures
    """
    output_dir = Path(output_path).parent if output_path else Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten for overall analysis
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # ============ FIGURE 1: PREDICTIONS VS ACTUALS SCATTER ============
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.scatter(targets_flat, preds_flat, alpha=0.3, s=10)

    # Add perfect prediction line
    min_val = min(targets_flat.min(), preds_flat.min())
    max_val = max(targets_flat.max(), preds_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Shipping Weight (kg)')
    ax.set_ylabel('Predicted Shipping Weight (kg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    ss_tot = np.sum((targets_flat - targets_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_dir / f"fig10_forecast_scatter_{model_name.lower().replace(' ', '_')}.png",
                   dpi=DPI, bbox_inches='tight')
    plt.close()

    # ============ FIGURE 2: ERROR DISTRIBUTION ============
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    errors = preds_flat - targets_flat

    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Prediction Error (kg)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # QQ plot approximation (residuals vs normal)
    sorted_errors = np.sort(errors)
    n = len(sorted_errors)
    theoretical_quantiles = np.linspace(0.001, 0.999, n)
    normal_quantiles = np.quantile(np.random.standard_normal(10000), theoretical_quantiles)

    axes[1].scatter(normal_quantiles, sorted_errors[:len(normal_quantiles)], alpha=0.3, s=5)
    axes[1].plot([-3, 3], [-3 * errors.std(), 3 * errors.std()], 'r--', linewidth=2)
    axes[1].set_title('Q-Q Plot (Errors vs Normal)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')

    plt.suptitle(f'{model_name} Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_dir / f"fig11_forecast_errors_{model_name.lower().replace(' ', '_')}.png",
                   dpi=DPI, bbox_inches='tight')
    plt.close()

    # ============ FIGURE 3: PER-HORIZON ANALYSIS ============
    if predictions.shape[1] > 1:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        horizon = predictions.shape[1]
        horizons = range(1, horizon + 1)

        # MAE per horizon
        mae_per_h = [np.mean(np.abs(predictions[:, h] - targets[:, h])) for h in range(horizon)]
        axes[0].bar(horizons, mae_per_h, color='steelblue', edgecolor='black')
        axes[0].set_title('MAE by Forecast Horizon', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Horizon (days ahead)')
        axes[0].set_ylabel('Mean Absolute Error (kg)')

        # RMSE per horizon
        rmse_per_h = [np.sqrt(np.mean((predictions[:, h] - targets[:, h]) ** 2)) for h in range(horizon)]
        axes[1].bar(horizons, rmse_per_h, color='coral', edgecolor='black')
        axes[1].set_title('RMSE by Forecast Horizon', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Horizon (days ahead)')
        axes[1].set_ylabel('Root Mean Squared Error (kg)')

        plt.suptitle(f'{model_name} Per-Horizon Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_dir / f"fig12_forecast_horizon_{model_name.lower().replace(' ', '_')}.png",
                       dpi=DPI, bbox_inches='tight')
        plt.close()

    print(f"Forecast figures saved to: {output_dir}")


def plot_forecast_sample(
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_indices: List[int] = None,
    n_samples: int = 4,
    output_path: Optional[str] = None
) -> None:
    """
    Plot sample forecasts to show individual prediction quality.

    This shows specific forecast examples, which helps
             visualize how the model performs on individual sequences.

    Args:
        predictions: Model predictions (n_samples, horizon)
        targets: Ground truth values (n_samples, horizon)
        sample_indices: Specific indices to plot
        n_samples: Number of samples to plot if indices not specified
        output_path: Path to save figure
    """
    if sample_indices is None:
        sample_indices = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)

    n_plots = len(sample_indices)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    horizon = predictions.shape[1]
    x = range(1, horizon + 1)

    for i, idx in enumerate(sample_indices[:4]):
        ax = axes[i]
        ax.plot(x, targets[idx], 'b-o', label='Actual', linewidth=2, markersize=8)
        ax.plot(x, predictions[idx], 'r--^', label='Predicted', linewidth=2, markersize=8)

        mae = np.mean(np.abs(predictions[idx] - targets[idx]))
        ax.set_title(f'Sample {idx} (MAE: {mae:.2f} kg)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Shipping Weight (kg)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sample Forecast Comparisons', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()
