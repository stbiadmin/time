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
    python scripts/01_generate_data.py -n small -p small --save-config
    python scripts/01_generate_data.py -n imbalanced -p imbalanced
    python scripts/01_generate_data.py -n custom --n-days 180 --n-events 10000
    python scripts/01_generate_data.py --list-presets

Expected Runtime: ~10-15 seconds
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import yaml
from src.utils.helpers import set_seed, load_config_with_preset, save_config
from src.utils.logging_config import setup_logging, log_section
from src.data_generation.rma_generator import generate_rma_data
from src.data_generation.network_events_generator import generate_network_events


def list_presets():
    """List available presets and exit."""
    presets_dir = Path("config/presets")
    if not presets_dir.exists():
        print("No presets directory found.")
        return

    preset_files = sorted(presets_dir.glob("*.yaml"))
    if not preset_files:
        print("No presets found.")
        return

    print("\nAvailable presets:")
    print("-" * 60)
    for p in preset_files:
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        desc = data.get("description", "(no description)")
        print(f"  {p.stem:20s} {desc}")
    print()


def apply_anomaly_ratio(config, ratio):
    """Redistribute cluster proportions so anomaly clusters sum to target ratio."""
    cluster_types = config["data_generation"]["network_events"]["cluster_types"]

    anomaly_keys = ["suspicious_scan", "auth_failure", "data_exfil"]
    normal_keys = [k for k in cluster_types if k not in anomaly_keys]

    # Current anomaly proportions (preserve relative weights within group)
    anomaly_total = sum(cluster_types[k] for k in anomaly_keys)
    normal_total = sum(cluster_types[k] for k in normal_keys)

    # Scale anomaly group to target ratio
    if anomaly_total > 0:
        anomaly_scale = ratio / anomaly_total
    else:
        anomaly_scale = 1.0

    # Scale normal group to fill remainder
    normal_target = 1.0 - ratio
    if normal_total > 0:
        normal_scale = normal_target / normal_total
    else:
        normal_scale = 1.0

    for k in anomaly_keys:
        cluster_types[k] = round(cluster_types[k] * anomaly_scale, 4)
    for k in normal_keys:
        cluster_types[k] = round(cluster_types[k] * normal_scale, 4)

    # Fix rounding to sum to 1.0
    total = sum(cluster_types.values())
    if total != 1.0:
        largest = max(normal_keys, key=lambda k: cluster_types[k])
        cluster_types[largest] = round(cluster_types[largest] + (1.0 - total), 4)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for ML scenarios"
    )
    parser.add_argument(
        "--dataset-name", "-n",
        help="Output name: produces data/raw/rma_<name>.csv, data/raw/network_<name>.csv",
    )
    parser.add_argument(
        "--preset", "-p",
        help="Load preset from config/presets/<name>.yaml",
    )
    parser.add_argument(
        "--seed", "-s", type=int,
        help="Override random seed",
    )
    parser.add_argument(
        "--rma-only", action="store_true",
        help="Generate only RMA data",
    )
    parser.add_argument(
        "--network-only", action="store_true",
        help="Generate only network events",
    )
    parser.add_argument(
        "--n-events", type=int,
        help="Override network_events.n_events",
    )
    parser.add_argument(
        "--n-days", type=int,
        help="Override rma.n_days",
    )
    parser.add_argument(
        "--anomaly-ratio", type=float,
        help="Set combined proportion for anomaly clusters (suspicious_scan + auth_failure + data_exfil)",
    )
    parser.add_argument(
        "--save-config", action="store_true",
        help="Save effective generation config as data/raw/<name>_generation_config.yaml",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List available presets and exit",
    )
    return parser.parse_args()


def main():
    """Generate all synthetic datasets."""
    args = parse_args()

    if args.list_presets:
        list_presets()
        return

    # ============ SETUP ============
    logger = setup_logging(log_level="INFO")
    config = load_config_with_preset(preset_name=args.preset)

    # Apply CLI overrides
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.n_days is not None:
        config["data_generation"]["rma"]["n_days"] = args.n_days
    if args.n_events is not None:
        config["data_generation"]["network_events"]["n_events"] = args.n_events
    if args.anomaly_ratio is not None:
        config = apply_anomaly_ratio(config, args.anomaly_ratio)

    seed = config["random_seed"]
    set_seed(seed)

    # Resolve output paths
    if args.dataset_name:
        rma_output = f"data/raw/rma_{args.dataset_name}.csv"
        network_output = f"data/raw/network_{args.dataset_name}.csv"
    else:
        rma_output = "data/raw/rma_shipping_data.csv"
        network_output = "data/raw/network_events.csv"

    log_section(logger, "DATA GENERATION PIPELINE")

    if args.preset:
        logger.info(f"Preset: {args.preset}")
    if args.dataset_name:
        logger.info(f"Dataset name: {args.dataset_name}")

    # ============ SAVE EFFECTIVE CONFIG ============
    if args.save_config:
        if args.dataset_name:
            config_output = f"data/raw/{args.dataset_name}_generation_config.yaml"
        else:
            config_output = "data/raw/generation_config.yaml"
        save_config(config["data_generation"], config_output)
        logger.info(f"Saved generation config to: {config_output}")

    # ============ STEP 1: GENERATE RMA DATA ============
    if not args.network_only:
        log_section(logger, "STEP 1: RMA Shipping Data")

        logger.info("Generating RMA shipping data...")
        logger.info("Patterns embedded:")
        logger.info("  - Weekly seasonality (lower weekends)")
        logger.info("  - Monthly patterns (end-of-quarter spikes)")
        logger.info("  - Regional trends (APAC growing, NA stable)")
        logger.info("  - SKU-specific weight distributions")

        rma_df = generate_rma_data(
            output_path=rma_output,
            seed=seed,
            config=config,
        )

        logger.info(f"\nRMA Dataset Summary:")
        logger.info(f"  Total records: {len(rma_df):,}")
        logger.info(f"  Date range: {rma_df['date'].min()} to {rma_df['date'].max()}")
        logger.info(f"  Regions: {rma_df['region'].nunique()}")
        logger.info(f"  SKU categories: {rma_df['sku_category'].nunique()}")
        logger.info(f"  Weight range: {rma_df['shipping_weight_kg'].min():.2f} - {rma_df['shipping_weight_kg'].max():.2f} kg")
        logger.info(f"  Output: {rma_output}")

    # ============ STEP 2: GENERATE NETWORK EVENTS ============
    if not args.rma_only:
        log_section(logger, "STEP 2: Network Events Data")

        logger.info("Generating network events data...")
        cluster_types = config["data_generation"]["network_events"]["cluster_types"]
        logger.info("Clusters embedded:")
        for name, proportion in cluster_types.items():
            logger.info(f"  - {name} ({proportion*100:.0f}%)")

        network_df = generate_network_events(
            output_path=network_output,
            seed=seed,
            config=config,
        )

        logger.info(f"\nNetwork Events Summary:")
        logger.info(f"  Total events: {len(network_df):,}")
        logger.info(f"  Time range: {network_df['timestamp'].min()} to {network_df['timestamp'].max()}")
        logger.info(f"  Unique source IPs: {network_df['source_ip'].nunique()}")
        logger.info(f"  Severity distribution:")
        for severity, count in network_df['severity'].value_counts().items():
            logger.info(f"    {severity}: {count:,} ({count/len(network_df)*100:.1f}%)")
        logger.info(f"  Output: {network_output}")

    # ============ SUMMARY ============
    log_section(logger, "DATA GENERATION COMPLETE")

    logger.info("\nFiles created:")
    if not args.network_only:
        logger.info(f"  {rma_output}")
    if not args.rma_only:
        logger.info(f"  {network_output}")
    logger.info("\nNext step: Run 02_explore_data.py for EDA visualizations")


if __name__ == "__main__":
    main()
