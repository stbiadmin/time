#!/usr/bin/env python3
"""
Script: 02_explore_data.py
==========================

Purpose: Exploratory Data Analysis for both datasets.

EDA helps us understand the data before modeling:
    - Identify distributions and outliers
    - Discover temporal patterns
    - Understand feature relationships
    - Guide preprocessing decisions

Usage:
    python scripts/02_explore_data.py

Expected Runtime: ~20-30 seconds
"""

import sys
import os
from pathlib import Path

import pandas as pd

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.helpers import set_seed, load_config, ensure_dir
from src.utils.logging_config import setup_logging, log_section
from src.visualization.eda_plots import plot_eda_rma, plot_eda_network


def main():
    """Run exploratory data analysis on both datasets."""

    # ============ SETUP ============
    logger = setup_logging(log_level="INFO")
    config = load_config()
    set_seed(config["random_seed"])

    log_section(logger, "EXPLORATORY DATA ANALYSIS")

    # Ensure output directory exists
    output_dir = ensure_dir("outputs/figures")

    # ============ LOAD DATA ============
    logger.info("Loading datasets...")

    rma_df = pd.read_csv("data/raw/rma_shipping_data.csv", parse_dates=["date"])
    network_df = pd.read_csv("data/raw/network_events.csv", parse_dates=["timestamp"])

    logger.info(f"  RMA data: {len(rma_df):,} records")
    logger.info(f"  Network events: {len(network_df):,} records")

    # ============ RMA DATA EDA ============
    log_section(logger, "RMA SHIPPING DATA EDA")

    # NOTE: We examine temporal patterns, distributions, and correlations
    #          to understand what the forecasting model needs to learn.

    logger.info("Key statistics:")
    logger.info(f"  Mean daily weight: {rma_df['shipping_weight_kg'].mean():.2f} kg")
    logger.info(f"  Std daily weight: {rma_df['shipping_weight_kg'].std():.2f} kg")
    logger.info(f"  Median daily weight: {rma_df['shipping_weight_kg'].median():.2f} kg")

    # Regional breakdown
    logger.info("\nRegional breakdown:")
    for region in rma_df['region'].unique():
        region_data = rma_df[rma_df['region'] == region]
        total_weight = region_data['shipping_weight_kg'].sum()
        pct = total_weight / rma_df['shipping_weight_kg'].sum() * 100
        logger.info(f"  {region}: {total_weight:,.0f} kg ({pct:.1f}%)")

    # Urgency distribution
    logger.info("\nUrgency distribution:")
    for urgency in sorted(rma_df['request_urgency'].unique()):
        count = len(rma_df[rma_df['request_urgency'] == urgency])
        pct = count / len(rma_df) * 100
        logger.info(f"  Level {urgency}: {count:,} ({pct:.1f}%)")

    # Generate visualizations
    logger.info("\nGenerating RMA visualizations...")
    plot_eda_rma(rma_df, output_dir=str(output_dir), save=True)

    # ============ NETWORK EVENTS EDA ============
    log_section(logger, "NETWORK EVENTS EDA")

    # NOTE: For clustering, we examine feature distributions and
    #          look for natural groupings that K-means might discover.

    logger.info("Key statistics:")
    logger.info(f"  Mean duration: {network_df['duration_ms'].mean():.2f} ms")
    logger.info(f"  Mean bytes: {network_df['bytes_transferred'].mean():,.0f}")
    logger.info(f"  Median duration: {network_df['duration_ms'].median():.2f} ms")
    logger.info(f"  Median bytes: {network_df['bytes_transferred'].median():,.0f}")

    # Protocol distribution
    logger.info("\nProtocol distribution:")
    for protocol, count in network_df['protocol'].value_counts().items():
        pct = count / len(network_df) * 100
        logger.info(f"  {protocol}: {count:,} ({pct:.1f}%)")

    # Severity distribution
    logger.info("\nSeverity distribution:")
    for severity, count in network_df['severity'].value_counts().items():
        pct = count / len(network_df) * 100
        logger.info(f"  {severity}: {count:,} ({pct:.1f}%)")

    # Ground truth clusters (since we generated the data)
    if 'true_cluster' in network_df.columns:
        logger.info("\nGround truth cluster distribution:")
        for cluster, count in network_df['true_cluster'].value_counts().items():
            pct = count / len(network_df) * 100
            logger.info(f"  {cluster}: {count:,} ({pct:.1f}%)")

    # Generate visualizations
    logger.info("\nGenerating Network Events visualizations...")
    plot_eda_network(network_df, output_dir=str(output_dir), save=True)

    # ============ SUMMARY ============
    log_section(logger, "EDA COMPLETE")

    logger.info("\nFigures generated:")
    logger.info("  RMA Analysis:")
    logger.info("    - fig01_rma_time_series.png")
    logger.info("    - fig02_rma_distributions.png")
    logger.info("    - fig03_rma_seasonality.png")
    logger.info("    - fig04_rma_correlation.png")
    logger.info("    - fig05_rma_sku_analysis.png")
    logger.info("\n  Network Events Analysis:")
    logger.info("    - fig06_network_distributions.png")
    logger.info("    - fig07_network_temporal.png")
    logger.info("    - fig08_network_scatter.png")
    logger.info("    - fig09_network_clusters.png")

    logger.info("\nNext step: Run 03_train_rma_model.py to train the GRU forecaster")


if __name__ == "__main__":
    main()
