"""
Module: regression_metrics.py
=============================

Purpose: Evaluation metrics for time series regression models.

Business Context:
    Forecasting accuracy directly impacts business decisions. Different
    metrics capture different aspects of model quality:
    - MAE: Average error in original units (kg in our case)
    - MAPE: Percentage error for business context
    - RMSE: Penalizes large errors more heavily

We compute multiple metrics because:
    - MAE is interpretable ("on average, we're off by X kg")
    - MAPE allows comparison across different scales
    - RMSE/MSE is what we optimize during training
    - R² shows how much variance we explain
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    description: str


class RegressionEvaluator:
    """
    Evaluator for regression model performance.

    This class computes and compares metrics across multiple
             model versions, enabling fair comparison and improvement
             tracking.

    Attributes:
        predictions: Dictionary of model name -> predictions
        targets: Ground truth targets
        metrics: Computed metrics for all models

    Example:
        >>> evaluator = RegressionEvaluator()
        >>> evaluator.add_predictions("baseline", baseline_preds, targets)
        >>> evaluator.add_predictions("model_v1", v1_preds, targets)
        >>> comparison = evaluator.compare_models()
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.predictions: Dict[str, np.ndarray] = {}
        self.targets: Optional[np.ndarray] = None
        self.metrics: Dict[str, Dict[str, float]] = {}

    def add_predictions(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Add predictions from a model and compute metrics.

        Args:
            model_name: Name/version of the model
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            Dictionary of computed metrics
        """
        self.predictions[model_name] = predictions

        if self.targets is None:
            self.targets = targets

        metrics = self._compute_metrics(predictions, targets)
        self.metrics[model_name] = metrics

        return metrics

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all regression metrics.

        Each metric provides different insight:
            - MAE: Mean Absolute Error - average magnitude of errors
            - MSE: Mean Squared Error - heavily penalizes large errors
            - RMSE: Root MSE - same units as target
            - MAPE: Mean Absolute Percentage Error - relative error
            - R²: Coefficient of determination - explained variance

        Args:
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            Dictionary of metric names and values
        """
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        # ============ BASIC METRICS ============
        errors = predictions - targets
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2

        mae = np.mean(abs_errors)
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)

        # ============ PERCENTAGE METRICS ============
        # NOTE: MAPE requires handling of zero/near-zero targets
        #        to avoid division by zero
        epsilon = 1e-8
        percentage_errors = abs_errors / (np.abs(targets) + epsilon)
        mape = np.mean(percentage_errors) * 100

        # ============ R-SQUARED ============
        # NOTE: R² = 1 - (SS_res / SS_tot)
        #        where SS_res is sum of squared residuals
        #        and SS_tot is total sum of squares
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))

        # ============ ADDITIONAL METRICS ============
        # Max error for worst-case analysis
        max_error = np.max(abs_errors)

        # Median absolute error (more robust to outliers)
        median_ae = np.median(abs_errors)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
            "max_error": float(max_error),
            "median_ae": float(median_ae),
        }

    def compare_models(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Compare all registered models.

        We compute relative improvements between models,
                 which is crucial for demonstrating model progression.

        Returns:
            Dictionary with comparison results
        """
        if len(self.metrics) < 2:
            return self.metrics

        # Find baseline (first model added or one named "baseline")
        baseline_name = "baseline" if "baseline" in self.metrics else list(self.metrics.keys())[0]
        baseline_metrics = self.metrics[baseline_name]

        comparison = {}
        for model_name, metrics in self.metrics.items():
            comparison[model_name] = {**metrics}

            # Add relative improvement vs baseline
            if model_name != baseline_name:
                for metric in ["mae", "mse", "rmse"]:
                    baseline_val = baseline_metrics[metric]
                    model_val = metrics[metric]
                    if baseline_val > 0:
                        improvement = (baseline_val - model_val) / baseline_val * 100
                        comparison[model_name][f"{metric}_improvement_%"] = round(improvement, 2)

        return comparison

    def get_summary_table(self) -> str:
        """
        Get a formatted summary table of all model metrics.

        Returns:
            Formatted string table
        """
        if not self.metrics:
            return "No models evaluated yet."

        # Header
        metrics_to_show = ["mae", "mse", "rmse", "mape", "r2"]
        header = f"{'Model':<20} | " + " | ".join(f"{m.upper():>10}" for m in metrics_to_show)
        separator = "-" * len(header)

        lines = [separator, header, separator]

        for model_name, metrics in self.metrics.items():
            values = [f"{metrics[m]:>10.4f}" for m in metrics_to_show]
            lines.append(f"{model_name:<20} | " + " | ".join(values))

        lines.append(separator)

        return "\n".join(lines)

    def get_improvement_summary(self, baseline_name: str = "baseline") -> str:
        """
        Get a summary of improvements over baseline.

        Args:
            baseline_name: Name of the baseline model

        Returns:
            Formatted improvement summary
        """
        if baseline_name not in self.metrics:
            return f"Baseline '{baseline_name}' not found."

        baseline = self.metrics[baseline_name]
        lines = [f"Improvements over {baseline_name}:", "-" * 40]

        for model_name, metrics in self.metrics.items():
            if model_name == baseline_name:
                continue

            mae_improvement = (baseline["mae"] - metrics["mae"]) / baseline["mae"] * 100
            lines.append(f"{model_name}: {mae_improvement:+.2f}% MAE reduction")

        return "\n".join(lines)


def compute_per_horizon_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics for each prediction horizon step.

    Forecast accuracy typically degrades with horizon length.
             This function shows how error increases for further-out
             predictions.

    Args:
        predictions: Predictions array (n_samples, horizon)
        targets: Targets array (n_samples, horizon)

    Returns:
        Dictionary mapping horizon step to metrics
    """
    n_horizons = predictions.shape[1]
    per_horizon = {}

    for h in range(n_horizons):
        preds_h = predictions[:, h]
        targets_h = targets[:, h]

        errors = preds_h - targets_h
        per_horizon[h + 1] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
        }

    return per_horizon
