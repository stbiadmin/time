#!/usr/bin/env python3
"""
Script: 03_train_rma_model.py
=============================

Purpose: Train progressive GRU models for RMA shipping weight forecasting.

This script trains three model versions showing progressive improvement:
    - V1: Simple GRU with numerical features only
    - V2: GRU with categorical embeddings
    - V3: Full model with exogenous features and regularization

Each version builds on the previous, showing how architectural
improvements lead to better forecasting accuracy.

Usage:
    python scripts/03_train_rma_model.py

Expected Runtime: ~3-4 minutes (all three versions)
"""

import sys
import os
from pathlib import Path
import time

import pandas as pd
import torch

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.helpers import set_seed, load_config, ensure_dir, get_device
from src.utils.logging_config import setup_logging, log_section, log_metrics
from src.preprocessing.rma_preprocessor import RMAPreprocessor, RMADataConfig, create_data_loaders
from src.models.gru_forecaster import create_model, get_model_summary
from src.training.rma_trainer import RMATrainer, compute_baseline_metrics
from src.evaluation.regression_metrics import RegressionEvaluator
from src.visualization.training_plots import (
    plot_training_history,
    plot_model_comparison,
    plot_improvement_summary
)
from mlops.model_registry import ModelRegistry


def main():
    """Train all GRU model versions and compare results."""

    # ============ SETUP ============
    logger = setup_logging(log_level="INFO")
    config = load_config()
    set_seed(config["random_seed"])
    device = get_device()

    log_section(logger, "RMA FORECASTING MODEL TRAINING")

    # Ensure output directories
    ensure_dir("outputs/figures")
    ensure_dir("outputs/models")

    # ============ LOAD AND PREPROCESS DATA ============
    log_section(logger, "DATA PREPROCESSING")

    logger.info("Loading RMA data...")
    rma_df = pd.read_csv("data/raw/rma_shipping_data.csv", parse_dates=["date"])
    logger.info(f"  Total records: {len(rma_df):,}")

    # NOTE: We aggregate to daily regional totals for time series forecasting
    logger.info("Preprocessing data...")
    preprocess_config = RMADataConfig(
        sequence_length=config["preprocessing"]["rma"]["sequence_length"],
        prediction_horizon=config["preprocessing"]["rma"]["prediction_horizon"],
        train_ratio=config["preprocessing"]["rma"]["train_ratio"],
        val_ratio=config["preprocessing"]["rma"]["val_ratio"],
        test_ratio=config["preprocessing"]["rma"]["test_ratio"],
    )

    preprocessor = RMAPreprocessor(preprocess_config)
    train_dataset, val_dataset, test_dataset = preprocessor.fit_transform(
        rma_df,
        aggregation_level="region"
    )

    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Val samples: {len(val_dataset):,}")
    logger.info(f"  Test samples: {len(test_dataset):,}")
    logger.info(f"  Vocabulary sizes: {preprocessor.get_vocab_sizes()}")

    # Create data loaders
    batch_size = config["training"]["rma"]["batch_size"]
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    # Get feature dimensions
    sample = train_dataset[0]
    n_numerical = sample["numerical"].shape[1]
    vocab_sizes = preprocessor.get_vocab_sizes()

    logger.info(f"  Numerical features: {n_numerical}")
    logger.info(f"  Batch size: {batch_size}")

    # ============ COMPUTE BASELINE ============
    log_section(logger, "BASELINE: NAIVE PERSISTENCE")

    # NOTE: Baseline predicts that future values equal recent past values.
    #          Any good model should beat this simple heuristic.
    baseline_metrics = compute_baseline_metrics(
        train_loader,
        test_loader,
        prediction_horizon=preprocess_config.prediction_horizon
    )
    logger.info("Baseline (naive persistence):")
    log_metrics(logger, baseline_metrics)

    # Initialize evaluator
    evaluator = RegressionEvaluator()

    # Store all results
    all_histories = {}
    all_metrics = {"baseline": baseline_metrics}

    # Training configuration
    train_config = config["training"]["rma"]
    model_config = config["models"]["gru"]

    # ============ TRAIN VERSION 1 ============
    # CHANGES FROM BASELINE: First neural network approach
    log_section(logger, "MODEL V1: SIMPLE NUMERICAL GRU")

    logger.info("Architecture:")
    logger.info("  - 2-layer GRU (64 → 32 units)")
    logger.info("  - Numerical features only")
    logger.info("  - Dropout: 0.2")
    logger.info("\nExpected: ~15% MAE improvement over baseline")

    model_v1 = create_model(
        version="v1",
        n_numerical_features=n_numerical,
        config={**model_config["v1"], "prediction_horizon": preprocess_config.prediction_horizon}
    )

    summary = get_model_summary(model_v1)
    logger.info(f"Parameters: {summary['trainable_parameters']:,}")

    trainer_v1 = RMATrainer(model_v1, train_loader, val_loader, train_config, device)
    start_time = time.time()
    history_v1 = trainer_v1.train(max_epochs=train_config["max_epochs"])
    v1_time = time.time() - start_time

    # Evaluate
    preds_v1, targets_v1 = trainer_v1.predict(test_loader)
    metrics_v1 = evaluator.add_predictions("v1", preds_v1, targets_v1)
    all_metrics["v1"] = metrics_v1
    all_histories["v1"] = history_v1

    logger.info(f"\nV1 Results (trained in {v1_time:.1f}s):")
    log_metrics(logger, metrics_v1)

    # ============ TRAIN VERSION 2 ============
    # CHANGES FROM V1: Add categorical embeddings
    log_section(logger, "MODEL V2: GRU WITH EMBEDDINGS")

    logger.info("CHANGES FROM V1:")
    logger.info("  [+] Embedding layers for region, SKU, urgency, shipping method")
    logger.info("  [+] Embedding dims: region=4, sku=8, urgency=2, method=2")
    logger.info("  [=] GRU architecture unchanged")
    logger.info("\nWHY THIS MATTERS:")
    logger.info("  Embeddings capture semantic relationships between categories")
    logger.info("\nExpected: ~10% additional MAE reduction")

    model_v2 = create_model(
        version="v2",
        n_numerical_features=n_numerical,
        vocab_sizes=vocab_sizes,
        config={**model_config["v2"], "prediction_horizon": preprocess_config.prediction_horizon}
    )

    summary = get_model_summary(model_v2)
    logger.info(f"Parameters: {summary['trainable_parameters']:,}")

    trainer_v2 = RMATrainer(model_v2, train_loader, val_loader, train_config, device)
    start_time = time.time()
    history_v2 = trainer_v2.train(max_epochs=train_config["max_epochs"])
    v2_time = time.time() - start_time

    preds_v2, targets_v2 = trainer_v2.predict(test_loader)
    metrics_v2 = evaluator.add_predictions("v2", preds_v2, targets_v2)
    all_metrics["v2"] = metrics_v2
    all_histories["v2"] = history_v2

    logger.info(f"\nV2 Results (trained in {v2_time:.1f}s):")
    log_metrics(logger, metrics_v2)

    # ============ TRAIN VERSION 3 ============
    # CHANGES FROM V2: Full model with regularization
    log_section(logger, "MODEL V3: FULL MODEL WITH REGULARIZATION")

    logger.info("CHANGES FROM V2:")
    logger.info("  [+] Increased dropout (0.2 → 0.3)")
    logger.info("  [+] Layer normalization for stability")
    logger.info("  [+] Residual connection from embeddings")
    logger.info("  [+] 2-layer MLP output head")
    logger.info("\nWHY THIS MATTERS:")
    logger.info("  Better regularization prevents overfitting on richer features")
    logger.info("\nExpected: ~10% additional MAE reduction (35% total)")

    model_v3 = create_model(
        version="v3",
        n_numerical_features=n_numerical,
        vocab_sizes=vocab_sizes,
        config={**model_config["v3"], "prediction_horizon": preprocess_config.prediction_horizon}
    )

    summary = get_model_summary(model_v3)
    logger.info(f"Parameters: {summary['trainable_parameters']:,}")

    trainer_v3 = RMATrainer(model_v3, train_loader, val_loader, train_config, device)
    start_time = time.time()
    history_v3 = trainer_v3.train(max_epochs=train_config["max_epochs"])
    v3_time = time.time() - start_time

    preds_v3, targets_v3 = trainer_v3.predict(test_loader)
    metrics_v3 = evaluator.add_predictions("v3", preds_v3, targets_v3)
    all_metrics["v3"] = metrics_v3
    all_histories["v3"] = history_v3

    logger.info(f"\nV3 Results (trained in {v3_time:.1f}s):")
    log_metrics(logger, metrics_v3)

    # ============ GENERATE VISUALIZATIONS ============
    log_section(logger, "GENERATING VISUALIZATIONS")

    # Training history plots
    for version in ["v1", "v2", "v3"]:
        plot_training_history(
            all_histories[version],
            title=f"Model {version.upper()} Training History",
            output_path=f"outputs/figures/fig_training_{version}.png"
        )

    # Model comparison
    plot_model_comparison(
        all_metrics,
        metric_name="mae",
        title="MAE Comparison Across Model Versions",
        output_path="outputs/figures/fig_model_comparison_mae.png"
    )

    plot_improvement_summary(
        all_metrics,
        baseline_name="baseline",
        output_path="outputs/figures/fig_improvement_summary.png"
    )

    # ============ SAVE BEST MODEL ============
    log_section(logger, "SAVING MODELS")

    registry = ModelRegistry("outputs/models")

    # Save V3 as the production model
    registry.save_rma_model(
        model=model_v3,
        preprocessor_state=preprocessor.get_state_dict(),
        training_history=history_v3,
        metrics=metrics_v3,
        config={**model_config["v3"], "prediction_horizon": preprocess_config.prediction_horizon},
        version="v3",
        description="Full GRU model with embeddings, layer norm, and residual connections"
    )

    # ============ FINAL SUMMARY ============
    log_section(logger, "TRAINING COMPLETE")

    logger.info("\n" + evaluator.get_summary_table())
    logger.info("\n" + evaluator.get_improvement_summary("baseline"))

    total_improvement = (baseline_metrics["mae"] - metrics_v3["mae"]) / baseline_metrics["mae"] * 100
    logger.info(f"\nTotal MAE improvement: {total_improvement:.1f}%")
    logger.info(f"Total training time: {v1_time + v2_time + v3_time:.1f}s")

    logger.info("\nNext step: Run 04_train_clustering_model.py for network event clustering")


if __name__ == "__main__":
    main()
