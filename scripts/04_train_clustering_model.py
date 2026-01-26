#!/usr/bin/env python3
"""
Script: 04_train_clustering_model.py
====================================

Train K-means clustering with LSA for network event classification.

Usage:
    python scripts/04_train_clustering_model.py

Expected Runtime: ~30 seconds
"""

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

from src.utils.helpers import set_seed, load_config, ensure_dir
from src.utils.logging_config import setup_logging, log_section, log_metrics
from src.preprocessing.network_preprocessor import NetworkPreprocessor, NetworkDataConfig
from src.models.kmeans_clusterer import KMeansClusterer, KMeansConfig
from src.models.lsa_analyzer import LSAAnalyzer
from src.training.clustering_trainer import ClusteringTrainer
from src.evaluation.clustering_metrics import ClusteringEvaluator
from src.visualization.cluster_plots import plot_cluster_analysis, plot_silhouette_analysis
from mlops.model_registry import ModelRegistry


def main():
    """Train clustering pipeline and evaluate results."""

    # Setup
    logger = setup_logging(log_level="INFO")
    config = load_config()
    set_seed(config["random_seed"])

    log_section(logger, "NETWORK EVENT CLUSTERING PIPELINE")

    ensure_dir("outputs/figures")
    ensure_dir("outputs/models")

    # Load data
    log_section(logger, "LOADING DATA")

    logger.info("Loading network events data...")
    network_df = pd.read_csv("data/raw/network_events.csv", parse_dates=["timestamp"])
    logger.info(f"  Total events: {len(network_df):,}")
    logger.info(f"  Features: {list(network_df.columns)}")

    # Preprocessing
    log_section(logger, "PREPROCESSING PIPELINE")

    logger.info("Preprocessing configuration:")
    preprocess_config = NetworkDataConfig(
        tfidf_max_df=config["preprocessing"]["network"]["tfidf"]["max_df"],
        tfidf_min_df=config["preprocessing"]["network"]["tfidf"]["min_df"],
        tfidf_ngram_range=tuple(config["preprocessing"]["network"]["tfidf"]["ngram_range"]),
        tfidf_max_features=config["preprocessing"]["network"]["tfidf"]["max_features"],
        lsa_n_components=config["preprocessing"]["network"]["lsa"]["n_components"],
    )

    logger.info(f"  TF-IDF max_df: {preprocess_config.tfidf_max_df}")
    logger.info(f"  TF-IDF min_df: {preprocess_config.tfidf_min_df}")
    logger.info(f"  LSA components: {preprocess_config.lsa_n_components}")

    # Train clustering pipeline
    log_section(logger, "CLUSTERING PIPELINE")

    kmeans_config = KMeansConfig(
        k_range=tuple(config["models"]["kmeans"]["k_range"]),
        init=config["models"]["kmeans"]["init"],
        n_init=config["models"]["kmeans"]["n_init"],
        max_iter=config["models"]["kmeans"]["max_iter"],
    )

    logger.info(f"K-means configuration:")
    logger.info(f"  K search range: {kmeans_config.k_range}")
    logger.info(f"  Initialization: {kmeans_config.init}")
    logger.info(f"  Max iterations: {kmeans_config.max_iter}")

    trainer = ClusteringTrainer(
        preprocessor_config=preprocess_config,
        kmeans_config=kmeans_config,
    )

    start_time = time.time()
    cluster_labels = trainer.fit(network_df, find_optimal_k=True)
    training_time = time.time() - start_time

    logger.info(f"\nClustering complete in {training_time:.2f}s")

    # Analyze LSA topics
    log_section(logger, "LSA TOPIC ANALYSIS")

    lsa_analyzer = LSAAnalyzer()
    log_messages = network_df["log_message"].tolist()
    lsa_analyzer.fit_transform(log_messages)

    logger.info("\nTop 5 LSA Topics:")
    topic_terms = lsa_analyzer.get_topic_terms(n_terms=5)
    for topic_id in range(min(5, len(topic_terms))):
        terms = [t[0] for t in topic_terms[topic_id]]
        logger.info(f"  Topic {topic_id}: {', '.join(terms)}")

    # Evaluate clustering
    log_section(logger, "CLUSTERING EVALUATION")

    metrics = trainer.get_metrics()
    logger.info("Internal Metrics:")
    log_metrics(logger, metrics)

    if "true_cluster" in network_df.columns:
        logger.info("\nExternal Metrics (vs Ground Truth):")
        external_metrics = trainer.evaluate_against_ground_truth()
        log_metrics(logger, external_metrics)

    summary_df = trainer.get_cluster_summary()
    logger.info("\nCluster Summary:")
    print(summary_df.to_string())

    # Generate visualizations
    log_section(logger, "GENERATING VISUALIZATIONS")

    features = trainer.features
    elbow_data = trainer.clusterer.elbow_data
    silhouette_data = trainer.clusterer.silhouette_data
    centroids = trainer.clusterer.get_cluster_centers()
    true_labels = network_df["true_cluster"].values if "true_cluster" in network_df.columns else None

    plot_cluster_analysis(
        features=features,
        labels=cluster_labels,
        elbow_data=elbow_data,
        silhouette_data=silhouette_data,
        centroids=centroids,
        true_labels=true_labels,
        output_dir="outputs/figures"
    )

    plot_silhouette_analysis(
        features=features,
        labels=cluster_labels,
        output_path="outputs/figures/fig17_silhouette_analysis.png"
    )

    # Save model
    log_section(logger, "SAVING MODEL")

    registry = ModelRegistry("outputs/models")

    registry.save_clustering_model(
        clusterer_state=trainer.clusterer.get_state_dict(),
        preprocessor_state=trainer.preprocessor.get_state_dict(),
        metrics=metrics,
        cluster_interpretations=trainer.cluster_interpretations,
        version="v1",
        description="K-means clustering with LSA text features for network event classification"
    )

    # Summary
    log_section(logger, "CLUSTERING COMPLETE")

    logger.info(f"\nResults Summary:")
    logger.info(f"  Optimal K: {trainer.clusterer.optimal_k}")
    logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"  Training Time: {training_time:.2f}s")

    if "true_cluster" in network_df.columns:
        logger.info(f"\n  Adjusted Rand Index: {external_metrics['adjusted_rand_score']:.4f}")
        logger.info(f"  Normalized Mutual Info: {external_metrics['normalized_mutual_info']:.4f}")

    logger.info("\nCluster Interpretations:")
    for cluster_id, interpretation in trainer.cluster_interpretations.items():
        count = len(cluster_labels[cluster_labels == cluster_id])
        logger.info(f"  Cluster {cluster_id}: {interpretation} ({count:,} events)")

    logger.info("\nNext step: Run 05_evaluate_models.py for comprehensive evaluation")


if __name__ == "__main__":
    main()
