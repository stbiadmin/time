#!/usr/bin/env python3
"""
Script: 05_evaluate_models.py
=============================

Purpose: Comprehensive evaluation of all trained models.

This script provides detailed evaluation:
    - RMA Forecasting: Per-horizon analysis, error distributions
    - Clustering: Cluster quality, confusion analysis
    - Cross-model comparisons and visualizations

Usage:
    python scripts/05_evaluate_models.py

Expected Runtime: ~20 seconds
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.helpers import set_seed, load_config, ensure_dir, get_device
from src.utils.logging_config import setup_logging, log_section, log_metrics
from src.preprocessing.rma_preprocessor import RMAPreprocessor, RMADataConfig, create_data_loaders
from src.models.gru_forecaster import create_model
from src.evaluation.regression_metrics import RegressionEvaluator, compute_per_horizon_metrics
from src.evaluation.clustering_metrics import ClusteringEvaluator, compute_cluster_purity
from src.visualization.forecast_plots import plot_forecast_results, plot_forecast_sample
from mlops.model_registry import ModelRegistry


def main():
    """Run comprehensive model evaluation."""

    # ============ SETUP ============
    logger = setup_logging(log_level="INFO")
    config = load_config()
    set_seed(config["random_seed"])
    device = get_device()

    log_section(logger, "COMPREHENSIVE MODEL EVALUATION")

    ensure_dir("outputs/figures")

    registry = ModelRegistry("outputs/models")

    # ============ EVALUATE RMA FORECASTING MODEL ============
    log_section(logger, "RMA FORECASTING EVALUATION")

    # Load model
    logger.info("Loading RMA model V3...")
    try:
        artifacts = registry.load_rma_model("v3")
        logger.info(f"  Model version: {artifacts['metadata']['model_version']}")
        logger.info(f"  Created: {artifacts['metadata']['created_at']}")
    except FileNotFoundError:
        logger.warning("RMA model not found. Run 03_train_rma_model.py first.")
        artifacts = None

    if artifacts:
        # Load and preprocess test data
        logger.info("\nPreparing test data...")
        rma_df = pd.read_csv("data/raw/rma_shipping_data.csv", parse_dates=["date"])

        preprocess_config = RMADataConfig(
            sequence_length=config["preprocessing"]["rma"]["sequence_length"],
            prediction_horizon=config["preprocessing"]["rma"]["prediction_horizon"],
            train_ratio=config["preprocessing"]["rma"]["train_ratio"],
            val_ratio=config["preprocessing"]["rma"]["val_ratio"],
            test_ratio=config["preprocessing"]["rma"]["test_ratio"],
        )

        preprocessor = RMAPreprocessor(preprocess_config)
        train_dataset, val_dataset, test_dataset = preprocessor.fit_transform(
            rma_df, aggregation_level="region"
        )

        _, _, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config["training"]["rma"]["batch_size"]
        )

        # Rebuild and load model
        vocab_sizes = preprocessor.get_vocab_sizes()
        n_numerical = train_dataset[0]["numerical"].shape[1]

        model = create_model(
            version="v3",
            n_numerical_features=n_numerical,
            vocab_sizes=vocab_sizes,
            config={**config["models"]["gru"]["v3"],
                    "prediction_horizon": preprocess_config.prediction_horizon}
        )
        model.load_state_dict(artifacts["weights"])
        model.to(device)
        model.eval()

        # Generate predictions
        logger.info("\nGenerating predictions on test set...")
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                numerical = batch["numerical"].to(device)
                categorical = batch["categorical"].to(device)
                targets = batch["target"]

                predictions = model(numerical, categorical)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        logger.info(f"  Test samples: {len(predictions):,}")
        logger.info(f"  Prediction horizon: {predictions.shape[1]} days")

        # Compute metrics
        logger.info("\nOverall Metrics:")
        evaluator = RegressionEvaluator()
        metrics = evaluator.add_predictions("v3", predictions, targets)
        log_metrics(logger, metrics)

        # Per-horizon analysis
        logger.info("\nPer-Horizon Analysis:")
        per_horizon = compute_per_horizon_metrics(predictions, targets)
        for h, h_metrics in per_horizon.items():
            logger.info(f"  Day {h}: MAE={h_metrics['mae']:.4f}, RMSE={h_metrics['rmse']:.4f}")

        # Generate forecast visualizations
        logger.info("\nGenerating forecast visualizations...")
        plot_forecast_results(
            predictions, targets,
            model_name="GRU_V3",
            output_path="outputs/figures/fig10_forecast.png"
        )

        plot_forecast_sample(
            predictions, targets,
            n_samples=4,
            output_path="outputs/figures/fig_forecast_samples.png"
        )

    # ============ EVALUATE CLUSTERING MODEL ============
    log_section(logger, "CLUSTERING EVALUATION")

    logger.info("Loading clustering model...")
    try:
        cluster_artifacts = registry.load_clustering_model("v1")
        logger.info(f"  Model version: {cluster_artifacts['metadata']['model_version']}")
        logger.info(f"  Created: {cluster_artifacts['metadata']['created_at']}")
    except FileNotFoundError:
        logger.warning("Clustering model not found. Run 04_train_clustering_model.py first.")
        cluster_artifacts = None

    if cluster_artifacts:
        # Load network events
        network_df = pd.read_csv("data/raw/network_events.csv", parse_dates=["timestamp"])

        # Reconstruct features (simplified - would need full pipeline in production)
        from src.preprocessing.network_preprocessor import NetworkPreprocessor, NetworkDataConfig

        preprocess_config = NetworkDataConfig(
            tfidf_max_df=config["preprocessing"]["network"]["tfidf"]["max_df"],
            tfidf_min_df=config["preprocessing"]["network"]["tfidf"]["min_df"],
            tfidf_ngram_range=tuple(config["preprocessing"]["network"]["tfidf"]["ngram_range"]),
            tfidf_max_features=config["preprocessing"]["network"]["tfidf"]["max_features"],
            lsa_n_components=config["preprocessing"]["network"]["lsa"]["n_components"],
        )

        preprocessor = NetworkPreprocessor(preprocess_config)
        features, processed_df = preprocessor.fit_transform(network_df)

        # Get cluster labels
        from sklearn.cluster import KMeans
        centroids = np.array(cluster_artifacts["clusterer"]["cluster_centers"])
        n_clusters = len(centroids)

        kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1, max_iter=1)
        kmeans.fit(features)  # This will use our provided centroids
        kmeans.cluster_centers_ = centroids  # Ensure centroids are exactly as loaded
        labels = kmeans.predict(features)

        # Evaluate
        logger.info("\nClustering Evaluation:")
        true_labels = network_df["true_cluster"].values if "true_cluster" in network_df.columns else None

        cluster_evaluator = ClusteringEvaluator(
            features=features,
            labels=labels,
            centroids=centroids,
            true_labels=true_labels
        )

        internal_metrics = cluster_evaluator.compute_internal_metrics()
        logger.info("\nInternal Metrics:")
        log_metrics(logger, internal_metrics)

        if true_labels is not None:
            external_metrics = cluster_evaluator.compute_external_metrics()
            logger.info("\nExternal Metrics:")
            log_metrics(logger, external_metrics)

            # Compute purity
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            true_encoded = le.fit_transform(true_labels)
            purity = compute_cluster_purity(labels, true_encoded)
            logger.info(f"\nCluster Purity: {purity:.4f}")

        # Per-cluster quality
        logger.info("\nPer-Cluster Quality:")
        qualities = cluster_evaluator.analyze_cluster_quality()
        for q in qualities:
            logger.info(
                f"  Cluster {q.cluster_id}: n={q.size:,}, "
                f"silhouette={q.silhouette_avg:.3f}"
            )

        # Print evaluation summary
        logger.info("\n" + cluster_evaluator.get_summary_table())

    # ============ SUMMARY ============
    log_section(logger, "EVALUATION COMPLETE")

    logger.info("\nFigures generated in outputs/figures/:")
    logger.info("  - Forecast scatter plots")
    logger.info("  - Error distribution plots")
    logger.info("  - Per-horizon analysis")
    logger.info("  - Sample forecast comparisons")

    logger.info("\nNext step: Run 06_export_for_serving.py to prepare models for deployment")


if __name__ == "__main__":
    main()
