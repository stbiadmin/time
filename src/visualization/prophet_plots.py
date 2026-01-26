"""
Module: prophet_plots.py
========================

Visualizations specific to Prophet forecasting models.

Prophet provides unique interpretability features that neural networks lack:
    - Decomposed trend showing growth/decline patterns
    - Seasonal components (weekly, yearly) as interpretable curves
    - Regressor effects showing impact of external variables
    - Uncertainty intervals for risk quantification
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (12, 6)
FIGSIZE_DOUBLE = (14, 8)
FIGSIZE_COMPONENTS = (12, 10)
DPI = 150


def plot_prophet_components(
    model: Any,
    train_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot Prophet's decomposed components.

    Shows the trend, weekly seasonality, yearly seasonality,
    and any regressor effects as separate subplots.

    Args:
        model: Fitted ProphetForecaster instance
        train_df: Training DataFrame used for fitting
        output_path: Path to save the figure
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before plotting components")

    # Get forecast with components
    forecast = model.model.predict(train_df)

    # Determine number of components to plot
    components = ['trend']
    if 'weekly' in forecast.columns:
        components.append('weekly')
    if 'yearly' in forecast.columns:
        components.append('yearly')

    # Add regressors
    for regressor in model.regressor_columns:
        if regressor in forecast.columns:
            components.append(regressor)

    n_components = len(components)
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3 * n_components))

    if n_components == 1:
        axes = [axes]

    for i, component in enumerate(components):
        ax = axes[i]

        if component == 'trend':
            ax.plot(forecast['ds'], forecast['trend'], 'b-', linewidth=2)
            ax.set_title('Trend', fontsize=12, fontweight='bold')
            ax.set_ylabel('Shipping Weight (kg)')

            # Add changepoints if available
            try:
                changepoints = model.get_changepoints()
                for cp_date in changepoints['ds']:
                    ax.axvline(x=cp_date, color='red', linestyle='--', alpha=0.3)
            except Exception:
                pass

        elif component == 'weekly':
            # For weekly, we need to aggregate by day of week
            weekly_data = forecast[['ds', 'weekly']].copy()
            weekly_data['dow'] = weekly_data['ds'].dt.dayofweek
            weekly_avg = weekly_data.groupby('dow')['weekly'].mean()

            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax.bar(range(7), weekly_avg.values, color='green', alpha=0.7)
            ax.set_xticks(range(7))
            ax.set_xticklabels(days)
            ax.set_title('Weekly Seasonality', fontsize=12, fontweight='bold')
            ax.set_ylabel('Effect (kg)')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        elif component == 'yearly':
            ax.plot(forecast['ds'], forecast['yearly'], 'purple', linewidth=2)
            ax.set_title('Yearly Seasonality', fontsize=12, fontweight='bold')
            ax.set_ylabel('Effect (kg)')

        else:
            # Regressor effect
            ax.plot(forecast['ds'], forecast[component], 'orange', linewidth=2)
            ax.set_title(f'Regressor: {component}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Effect (kg)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_prophet_forecast(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot Prophet forecast with uncertainty intervals.

    Shows:
    - Training data as points
    - Fitted values on training period
    - Forecast with confidence bands
    - Test actuals for comparison

    Args:
        model: Fitted ProphetForecaster instance
        train_df: Training DataFrame
        test_df: Test DataFrame with actuals
        output_path: Path to save the figure
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before plotting forecast")

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    # Get full forecast including history
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    forecast = model.model.predict(full_df)

    # Plot training data
    ax.scatter(
        train_df['ds'],
        train_df['y'],
        c='black',
        s=10,
        alpha=0.3,
        label='Training Data'
    )

    # Plot test data
    ax.scatter(
        test_df['ds'],
        test_df['y'],
        c='green',
        s=20,
        alpha=0.5,
        label='Test Actuals'
    )

    # Plot forecast
    ax.plot(
        forecast['ds'],
        forecast['yhat'],
        'b-',
        linewidth=2,
        label='Prophet Forecast'
    )

    # Plot uncertainty bands
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='blue',
        alpha=0.2,
        label=f'{int(model.config.interval_width * 100)}% Confidence Interval'
    )

    # Mark the train/test split
    split_date = train_df['ds'].max()
    ax.axvline(x=split_date, color='red', linestyle='--', linewidth=2, label='Train/Test Split')

    ax.set_title('Prophet Forecast with Uncertainty', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Shipping Weight (kg)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_prophet_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> None:
    """
    Compare Prophet and GRU models side by side.

    Creates a grouped bar chart showing MAE for each model version.

    Args:
        metrics: Dictionary of model_name -> {mae, mse, rmse, ...}
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Filter to models we want to compare
    model_order = ['baseline', 'gru_v3', 'prophet_v1', 'prophet_v2']
    model_names = [m for m in model_order if m in metrics]

    mae_values = [metrics[m].get('mae', 0) for m in model_names]

    # Color coding: baseline gray, GRU blue, Prophet green
    colors = []
    for m in model_names:
        if m == 'baseline':
            colors.append('#888888')
        elif m.startswith('gru'):
            colors.append('#1f77b4')
        else:
            colors.append('#2ca02c')

    # MAE comparison
    ax1 = axes[0]
    bars = ax1.bar(model_names, mae_values, color=colors, edgecolor='black')

    for bar, val in zip(bars, mae_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f'{val:.1f}',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    ax1.set_title('MAE Comparison (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (kg)')
    ax1.set_xticklabels(model_names, rotation=45, ha='right')

    # Improvement over baseline
    ax2 = axes[1]
    baseline_mae = metrics.get('baseline', {}).get('mae', 1)
    improvements = [(baseline_mae - metrics[m].get('mae', baseline_mae)) / baseline_mae * 100
                    for m in model_names]

    bars = ax2.bar(model_names, improvements, color=colors, edgecolor='black')

    for bar, val in zip(bars, improvements):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1 if val >= 0 else bar.get_height() - 3,
            f'{val:.1f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Improvement over Baseline', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE Reduction (%)')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_uncertainty_calibration(
    forecast: pd.DataFrame,
    actuals: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot calibration of Prophet's uncertainty intervals.

    Checks if the stated confidence interval actually contains
    the expected proportion of actuals.

    Args:
        forecast: Prophet forecast DataFrame with yhat_lower, yhat_upper
        actuals: Actual values
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Check coverage
    in_interval = (actuals >= forecast['yhat_lower'].values) & \
                  (actuals <= forecast['yhat_upper'].values)
    coverage = in_interval.mean() * 100

    # Residuals within/outside interval
    ax1 = axes[0]
    residuals = actuals - forecast['yhat'].values

    ax1.scatter(
        forecast['ds'][in_interval],
        residuals[in_interval],
        c='green',
        alpha=0.5,
        s=20,
        label=f'Within interval ({in_interval.sum()})'
    )
    ax1.scatter(
        forecast['ds'][~in_interval],
        residuals[~in_interval],
        c='red',
        alpha=0.7,
        s=30,
        label=f'Outside interval ({(~in_interval).sum()})'
    )

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Residuals by Interval Coverage', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual (kg)')
    ax1.legend()

    # Coverage summary
    ax2 = axes[1]
    categories = ['In Interval', 'Outside Interval']
    counts = [in_interval.sum(), (~in_interval).sum()]
    colors = ['green', 'red']

    bars = ax2.bar(categories, counts, color=colors, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha='center',
            fontsize=12,
            fontweight='bold'
        )

    ax2.set_title(f'Interval Coverage: {coverage:.1f}%', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_prophet_changepoints(
    model: Any,
    train_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize detected changepoints in the trend.

    Shows where Prophet detected significant shifts in the
    underlying growth trend.

    Args:
        model: Fitted ProphetForecaster instance
        train_df: Training DataFrame
        output_path: Path to save the figure
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before plotting changepoints")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Get forecast
    forecast = model.model.predict(train_df)

    # Plot trend
    ax.plot(forecast['ds'], forecast['trend'], 'b-', linewidth=2, label='Trend')

    # Get changepoints
    try:
        changepoints = model.get_changepoints()

        # Plot changepoints
        for idx, row in changepoints.iterrows():
            color = 'green' if row['delta'] > 0 else 'red'
            ax.axvline(x=row['ds'], color=color, linestyle='--', alpha=0.5)

        # Add legend entries
        ax.axvline(x=changepoints['ds'].iloc[0], color='green', linestyle='--',
                   alpha=0.5, label='Positive changepoint')
        ax.axvline(x=changepoints['ds'].iloc[0], color='red', linestyle='--',
                   alpha=0.5, label='Negative changepoint')

    except Exception as e:
        print(f"Could not extract changepoints: {e}")

    ax.set_title('Trend with Detected Changepoints', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Trend Component (kg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()
