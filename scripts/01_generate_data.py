#!/usr/bin/env python3
"""
Script: 01_generate_data.py
===========================

Purpose: Generate synthetic datasets for both ML scenarios.

This script creates realistic synthetic data with embedded patterns:
    1. RMA Shipping Data: 2 years of daily records with temporal patterns
    2. Network Events: 30,000 events with 6 distinct behavioral clusters

Usage:
    python scripts/01_generate_data.py

Expected Runtime: ~10-15 seconds
"""

import sys
import os
from pathlib import Path

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.helpers import set_seed, load_config
from src.utils.logging_config import setup_logging, log_section
from src.data_generation.rma_generator import generate_rma_data
from src.data_generation.network_events_generator import generate_network_events


def main():
    """Generate all synthetic datasets."""

    # ============ SETUP ============
    logger = setup_logging(log_level="INFO")
    config = load_config()
    set_seed(config["random_seed"])

    log_section(logger, "DATA GENERATION PIPELINE")

    # ============ STEP 1: GENERATE RMA DATA ============
    # NOTE: RMA data simulates 2 years of spare parts shipping records
    #          with weekly seasonality, regional trends, and SKU-specific patterns.
    log_section(logger, "STEP 1: RMA Shipping Data")

    logger.info("Generating RMA shipping data...")
    logger.info("Patterns embedded:")
    logger.info("  - Weekly seasonality (lower weekends)")
    logger.info("  - Monthly patterns (end-of-quarter spikes)")
    logger.info("  - Regional trends (APAC growing, NA stable)")
    logger.info("  - SKU-specific weight distributions")

    rma_df = generate_rma_data(
        config_path="config/settings.yaml",
        output_path="data/raw/rma_shipping_data.csv",
        seed=config["random_seed"]
    )

    logger.info(f"\nRMA Dataset Summary:")
    logger.info(f"  Total records: {len(rma_df):,}")
    logger.info(f"  Date range: {rma_df['date'].min()} to {rma_df['date'].max()}")
    logger.info(f"  Regions: {rma_df['region'].nunique()}")
    logger.info(f"  SKU categories: {rma_df['sku_category'].nunique()}")
    logger.info(f"  Weight range: {rma_df['shipping_weight_kg'].min():.2f} - {rma_df['shipping_weight_kg'].max():.2f} kg")

    # ============ STEP 2: GENERATE NETWORK EVENTS ============
    # NOTE: Network events simulate 30 days of log data with 6 distinct
    #          behavioral clusters that K-means should discover.
    log_section(logger, "STEP 2: Network Events Data")

    logger.info("Generating network events data...")
    logger.info("Clusters embedded:")
    logger.info("  - Normal web traffic (35%)")
    logger.info("  - Normal database queries (20%)")
    logger.info("  - Suspicious scanning (10%)")
    logger.info("  - Authentication failures (12%)")
    logger.info("  - Data exfiltration patterns (8%)")
    logger.info("  - Maintenance/backup events (15%)")

    network_df = generate_network_events(
        config_path="config/settings.yaml",
        output_path="data/raw/network_events.csv",
        seed=config["random_seed"]
    )

    logger.info(f"\nNetwork Events Summary:")
    logger.info(f"  Total events: {len(network_df):,}")
    logger.info(f"  Time range: {network_df['timestamp'].min()} to {network_df['timestamp'].max()}")
    logger.info(f"  Unique source IPs: {network_df['source_ip'].nunique()}")
    logger.info(f"  Severity distribution:")
    for severity, count in network_df['severity'].value_counts().items():
        logger.info(f"    {severity}: {count:,} ({count/len(network_df)*100:.1f}%)")

    # ============ SUMMARY ============
    log_section(logger, "DATA GENERATION COMPLETE")

    logger.info("\nFiles created:")
    logger.info("  data/raw/rma_shipping_data.csv")
    logger.info("  data/raw/network_events.csv")
    logger.info("\nNext step: Run 02_explore_data.py for EDA visualizations")


if __name__ == "__main__":
    main()
