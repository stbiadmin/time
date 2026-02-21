#!/usr/bin/env python3
"""
Script: 03b_train_rma_prophet.py
================================

Purpose: Train Prophet models for RMA shipping weight forecasting.

This script trains Prophet as an alternative to GRU for time series forecasting:
    - V1: Basic seasonality model (weekly + yearly patterns)
    - V2: With exogenous regressors (month-end, failure rate, urgency)

Prophet offers different trade-offs than GRU:
    + Interpretable components (trend, seasonality)
    + Built-in uncertainty quantification
    + Handles missing data gracefully
    + No GPU required
    - Less flexible for complex patterns
    - No embedding-based feature learning

Usage:
    python scripts/03b_train_rma_prophet.py

Expected Runtime: ~1-2 minutes (both versions)
"""

import argparse
import sys
import os
from pathlib import Path
import time

import pandas as pd
import numpy as np

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.helpers import set_seed, load_config, ensure_dir, add_dataset_args, resolve_data_paths
from src.utils.logging_config import setup_logging, log_section, log_metrics
from src.preprocessing.rma_preprocessor import RMAPreprocessor, RMADataConfig
from src.models.prophet_forecaster import create_prophet_model, get_prophet_summary
from src.training.prophet_trainer import ProphetTrainer, compute_prophet_baseline
from src.visualization.prophet_plots import (
    plot_prophet_components,
    plot_prophet_forecast,
    plot_prophet_comparison,
)
from src.visualization.training_plots import (
    plot_model_comparison,
    plot_improvement_summary,
)
from mlops.model_registry import ModelRegistry


def main():
    """Train Prophet model versions and compare with GRU results."""
    parser = argparse.ArgumentParser(description="Train Prophet RMA forecasting models")
    add_dataset_args(parser)
    args = parser.parse_args()
    rma_path, _ = resolve_data_paths(args)

    # ============ SETUP ============
    logger = setup_logging(log_level="INFO")
    config = load_config()
    set_seed(config["random_seed"])

    log_section(logger, "PROPHET FORECASTING MODEL TRAINING")

    # Ensure output directories
    ensure_dir("outputs/figures")
    ensure_dir("outputs/models")

    # ============ LOAD AND PREPROCESS DATA ============
    log_section(logger, "DATA PREPROCESSING FOR PROPHET")

    logger.info(f"Loading RMA data from {rma_path}...")
    rma_df = pd.read_csv(rma_path, parse_dates=["date"])
    logger.info(f"  Total records: {len(rma_df):,}")

    # Create preprocessor config
    preprocess_config = RMADataConfig(
        sequence_length=config["preprocessing"]["rma"]["sequence_length"],
        prediction_horizon=config["preprocessing"]["rma"]["prediction_horizon"],
        train_ratio=config["preprocessing"]["rma"]["train_ratio"],
        val_ratio=config["preprocessing"]["rma"]["val_ratio"],
        test_ratio=config["preprocessing"]["rma"]["test_ratio"],
    )

    preprocessor = RMAPreprocessor(preprocess_config)

    # Prepare data in Prophet format
    logger.info("Preparing data for Prophet...")
    train_df, val_df, test_df = preprocessor.prepare_for_prophet(
        rma_df,
        include_regressors=True
    )

    logger.info(f"  Train days: {len(train_df):,}")
    logger.info(f"  Val days: {len(val_df):,}")
    logger.info(f"  Test days: {len(test_df):,}")
    logger.info(f"  Date range: {train_df['ds'].min()} to {test_df['ds'].max()}")
    logger.info(f"  Regressors available: is_month_end, failure_rate_pct, avg_urgency")

    # ============ COMPUTE BASELINE ============
    log_section(logger, "BASELINE: NAIVE PERSISTENCE")

    baseline_metrics = compute_prophet_baseline(train_df, test_df)
    logger.info("Baseline (naive persistence):")
    log_metrics(logger, baseline_metrics)

    # Store all results
    all_metrics = {"baseline": baseline_metrics}
    all_histories = {}

    # Prophet configuration
    prophet_config = config.get("models", {}).get("prophet", {})
    cv_config = prophet_config.get("cross_validation", {})

    # ============ TRAIN VERSION 1 ============
    log_section(logger, "PROPHET V1: BASIC SEASONALITY")

    logger.info("Architecture:")
    logger.info("  - Multiplicative seasonality mode")
    logger.info("  - Weekly seasonality: enabled")
    logger.info("  - Yearly seasonality: enabled")
    logger.info("  - Changepoint detection: 25 potential changepoints")
    logger.info("  - No exogenous regressors")
    logger.info("\nProphet automatically detects:")
    logger.info("  - Trend changes (growth rate shifts)")
    logger.info("  - Weekly patterns (Mon-Sun variation)")
    logger.info("  - Yearly patterns (seasonal cycles)")

    v1_config = prophet_config.get("v1", {})
    model_v1 = create_prophet_model("v1", v1_config)

    summary = get_prophet_summary(model_v1)
    logger.info(f"\nSeasonality mode: {summary['seasonality_mode']}")

    trainer_v1 = ProphetTrainer(
        model=model_v1,
        train_df=train_df,
        val_df=val_df,
        config={"cross_validation": cv_config},
    )

    start_time = time.time()
    history_v1 = trainer_v1.train()
    v1_time = time.time() - start_time

    # Evaluate on test set
    preds_v1, targets_v1 = trainer_v1.predict(test_df)
    metrics_v1 = {
        "mae": float(np.abs(preds_v1 - targets_v1).mean()),
        "mse": float(((preds_v1 - targets_v1) ** 2).mean()),
        "rmse": float(np.sqrt(((preds_v1 - targets_v1) ** 2).mean())),
    }

    # Add MAPE
    mask = targets_v1 != 0
    if mask.sum() > 0:
        metrics_v1["mape"] = float(np.abs((targets_v1[mask] - preds_v1[mask]) / targets_v1[mask]).mean() * 100)

    all_metrics["prophet_v1"] = metrics_v1
    all_histories["prophet_v1"] = history_v1

    logger.info(f"\nV1 Results (trained in {v1_time:.1f}s):")
    log_metrics(logger, metrics_v1)

    if "cv_metrics" in history_v1 and history_v1["cv_metrics"]:
        logger.info("\nCross-validation metrics:")
        for k, v in history_v1["cv_metrics"].items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    # ============ TRAIN VERSION 2 ============
    log_section(logger, "PROPHET V2: WITH REGRESSORS")

    logger.info("CHANGES FROM V1:")
    logger.info("  [+] is_month_end regressor (captures end-of-month spikes)")
    logger.info("  [+] failure_rate_pct regressor (component failure correlation)")
    logger.info("  [+] avg_urgency regressor (urgency patterns)")
    logger.info("  [=] Same seasonality configuration")
    logger.info("\nWHY THIS MATTERS:")
    logger.info("  Regressors capture effects beyond seasonality:")
    logger.info("  - Month-end creates 1.3x volume spike (per config)")
    logger.info("  - Higher failure rates correlate with more RMAs")
    logger.info("  - Urgency distribution affects shipping patterns")

    v2_config = prophet_config.get("v2", {})
    model_v2 = create_prophet_model("v2", v2_config)

    summary = get_prophet_summary(model_v2)
    logger.info(f"\nRegressors: {summary['regressors']}")

    trainer_v2 = ProphetTrainer(
        model=model_v2,
        train_df=train_df,
        val_df=val_df,
        config={"cross_validation": cv_config},
    )

    start_time = time.time()
    history_v2 = trainer_v2.train()
    v2_time = time.time() - start_time

    # Evaluate on test set
    preds_v2, targets_v2 = trainer_v2.predict(test_df)
    metrics_v2 = {
        "mae": float(np.abs(preds_v2 - targets_v2).mean()),
        "mse": float(((preds_v2 - targets_v2) ** 2).mean()),
        "rmse": float(np.sqrt(((preds_v2 - targets_v2) ** 2).mean())),
    }

    if mask.sum() > 0:
        metrics_v2["mape"] = float(np.abs((targets_v2[mask] - preds_v2[mask]) / targets_v2[mask]).mean() * 100)

    all_metrics["prophet_v2"] = metrics_v2
    all_histories["prophet_v2"] = history_v2

    logger.info(f"\nV2 Results (trained in {v2_time:.1f}s):")
    log_metrics(logger, metrics_v2)

    if "cv_metrics" in history_v2 and history_v2["cv_metrics"]:
        logger.info("\nCross-validation metrics:")
        for k, v in history_v2["cv_metrics"].items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    # ============ GENERATE VISUALIZATIONS ============
    log_section(logger, "GENERATING VISUALIZATIONS")

    # Prophet component plots for V2
    try:
        plot_prophet_components(
            model=model_v2,
            train_df=train_df,
            output_path="outputs/figures/fig_prophet_components.png"
        )
        logger.info("  Created: fig_prophet_components.png")
    except Exception as e:
        logger.warning(f"  Could not create component plot: {e}")

    # Prophet forecast plot
    try:
        plot_prophet_forecast(
            model=model_v2,
            train_df=train_df,
            test_df=test_df,
            output_path="outputs/figures/fig_prophet_forecast.png"
        )
        logger.info("  Created: fig_prophet_forecast.png")
    except Exception as e:
        logger.warning(f"  Could not create forecast plot: {e}")

    # Model comparison across Prophet versions
    try:
        plot_model_comparison(
            all_metrics,
            metric_name="mae",
            title="MAE Comparison: Prophet Versions",
            output_path="outputs/figures/fig_prophet_comparison_mae.png"
        )
        logger.info("  Created: fig_prophet_comparison_mae.png")
    except Exception as e:
        logger.warning(f"  Could not create comparison plot: {e}")

    # ============ COMPARE WITH GRU ============
    log_section(logger, "COMPARISON WITH GRU")

    # Try to load GRU metrics for comparison
    try:
        registry = ModelRegistry("outputs/models")
        gru_artifacts = registry.load_rma_model("v3")
        gru_metrics = gru_artifacts["metadata"]["training_metrics"]

        logger.info("\nModel Comparison (Test MAE):")
        logger.info("-" * 40)
        logger.info(f"  Baseline (naive):     {baseline_metrics['mae']:.2f} kg")
        logger.info(f"  GRU V3:               {gru_metrics['mae']:.2f} kg")
        logger.info(f"  Prophet V1:           {metrics_v1['mae']:.2f} kg")
        logger.info(f"  Prophet V2:           {metrics_v2['mae']:.2f} kg")

        gru_improvement = (baseline_metrics["mae"] - gru_metrics["mae"]) / baseline_metrics["mae"] * 100
        p1_improvement = (baseline_metrics["mae"] - metrics_v1["mae"]) / baseline_metrics["mae"] * 100
        p2_improvement = (baseline_metrics["mae"] - metrics_v2["mae"]) / baseline_metrics["mae"] * 100

        logger.info("\nImprovement over baseline:")
        logger.info(f"  GRU V3:               {gru_improvement:.1f}%")
        logger.info(f"  Prophet V1:           {p1_improvement:.1f}%")
        logger.info(f"  Prophet V2:           {p2_improvement:.1f}%")

        # Create combined comparison plot
        combined_metrics = {
            "baseline": baseline_metrics,
            "gru_v3": gru_metrics,
            "prophet_v1": metrics_v1,
            "prophet_v2": metrics_v2,
        }

        try:
            plot_prophet_comparison(
                combined_metrics,
                output_path="outputs/figures/fig_gru_vs_prophet.png"
            )
            logger.info("\n  Created: fig_gru_vs_prophet.png")
        except Exception as e:
            logger.warning(f"  Could not create combined comparison: {e}")

    except FileNotFoundError:
        logger.info("\nGRU model not found. Run 03_train_rma_model.py first for comparison.")
        logger.info("\nProphet Results Only:")
        p1_improvement = (baseline_metrics["mae"] - metrics_v1["mae"]) / baseline_metrics["mae"] * 100
        p2_improvement = (baseline_metrics["mae"] - metrics_v2["mae"]) / baseline_metrics["mae"] * 100
        logger.info(f"  Prophet V1 improvement: {p1_improvement:.1f}%")
        logger.info(f"  Prophet V2 improvement: {p2_improvement:.1f}%")

    # ============ SAVE BEST MODEL ============
    log_section(logger, "SAVING MODELS")

    registry = ModelRegistry("outputs/models")

    # Save V1
    registry.save_prophet_model(
        model=model_v1,
        preprocessor_state=preprocessor.get_state_dict(),
        training_history=history_v1,
        metrics=metrics_v1,
        config=v1_config,
        version="v1",
        description="Prophet with weekly and yearly seasonality, no regressors"
    )

    # Save V2 as the best Prophet model
    registry.save_prophet_model(
        model=model_v2,
        preprocessor_state=preprocessor.get_state_dict(),
        training_history=history_v2,
        metrics=metrics_v2,
        config=v2_config,
        version="v2",
        description="Prophet with seasonality and exogenous regressors (month-end, failure rate, urgency)"
    )

    # ============ FINAL SUMMARY ============
    log_section(logger, "TRAINING COMPLETE")

    logger.info("\nProphet Training Summary:")
    logger.info("=" * 50)
    logger.info(f"  V1 MAE: {metrics_v1['mae']:.2f} kg (basic seasonality)")
    logger.info(f"  V2 MAE: {metrics_v2['mae']:.2f} kg (with regressors)")
    logger.info(f"  Total training time: {v1_time + v2_time:.1f}s")

    logger.info("\nProphet Advantages:")
    logger.info("  + Interpretable trend and seasonality components")
    logger.info("  + Built-in uncertainty intervals")
    logger.info("  + Handles missing data automatically")
    logger.info("  + Fast training (no GPU needed)")

    logger.info("\nProphet Trade-offs:")
    logger.info("  - Less flexible than neural networks")
    logger.info("  - No embedding-based feature learning")
    logger.info("  - Assumes additive/multiplicative decomposition")

    logger.info("\nModel artifacts saved to outputs/models/rma_prophet_v1/ and rma_prophet_v2/")
    logger.info("\nNext step: Run 05_evaluate_models.py for cross-model comparison")


if __name__ == "__main__":
    main()
