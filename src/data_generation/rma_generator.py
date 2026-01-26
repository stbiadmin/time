"""
Module: rma_generator.py
========================

Generate synthetic RMA (Return Merchandise Authorization) shipping data
with realistic temporal patterns for time series forecasting demonstration.

The generated data includes:
    - Weekly seasonality (lower weekend volumes)
    - Monthly patterns (end-of-quarter spikes)
    - Regional trends (different growth rates by geography)
    - SKU-specific weight distributions
    - Correlated features (urgency -> shipping method)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.helpers import set_seed, load_config
from ..utils.logging_config import get_logger


def generate_rma_data(
    config_path: str = "config/settings.yaml",
    output_path: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic RMA shipping data with embedded temporal patterns.

    Args:
        config_path: Path to configuration YAML file
        output_path: Optional path to save generated CSV
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic RMA shipping data
    """
    set_seed(seed)
    logger = get_logger()

    # Load configuration
    config = load_config(config_path)
    rma_config = config["data_generation"]["rma"]

    n_days = rma_config["n_days"]
    regions = rma_config["regions"]
    sku_categories = rma_config["sku_categories"]
    urgency_levels = rma_config["urgency_levels"]
    urgency_weights = rma_config["urgency_weights"]
    shipping_methods = rma_config["shipping_methods"]
    sku_base_weights = rma_config["sku_base_weights"]
    weekly_pattern = rma_config["weekly_pattern"]
    monthly_eom_spike = rma_config["monthly_eom_spike"]

    logger.info(f"Generating {n_days} days of RMA data...")

    # Generate date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Regional characteristics
    regional_profiles = {
        "NA": {"base_volume": 100, "growth_rate": 0.02, "volatility": 0.15},
        "EMEA": {"base_volume": 80, "growth_rate": 0.03, "volatility": 0.18},
        "APAC": {"base_volume": 120, "growth_rate": 0.08, "volatility": 0.20},
        "LATAM": {"base_volume": 40, "growth_rate": 0.05, "volatility": 0.25},
        "ANZ": {"base_volume": 25, "growth_rate": 0.01, "volatility": 0.12},
    }

    # Generate data records
    records = []

    for day_idx, date in enumerate(dates):
        # Calculate temporal factors
        day_of_week = date.weekday()
        weekly_factor = weekly_pattern[day_of_week]

        # Monthly pattern (spike at end of month/quarter)
        day_of_month = date.day
        days_in_month = (date.replace(month=date.month % 12 + 1, day=1) - timedelta(days=1)).day if date.month < 12 else 31
        is_eom = day_of_month >= days_in_month - 2
        is_eoq = date.month in [3, 6, 9, 12] and is_eom
        monthly_factor = monthly_eom_spike if is_eom else 1.0
        quarterly_factor = 1.2 if is_eoq else 1.0

        # Trend factor
        trend_position = day_idx / n_days

        for region in regions:
            profile = regional_profiles[region]

            # Calculate region-specific volume for this day
            base = profile["base_volume"]
            growth = 1 + profile["growth_rate"] * trend_position
            volatility = profile["volatility"]

            # Number of RMA requests (Poisson distributed)
            expected_requests = int(
                base * growth * weekly_factor * monthly_factor * quarterly_factor
            )
            n_requests = max(1, np.random.poisson(expected_requests))

            for _ in range(n_requests):
                # Generate SKU characteristics
                sku = np.random.choice(sku_categories)
                base_weight = sku_base_weights[sku]

                # Generate request urgency
                urgency = np.random.choice(urgency_levels, p=urgency_weights)

                # Determine shipping method (correlated with urgency)
                shipping_probs = _get_shipping_probs(urgency)
                shipping_method = np.random.choice(shipping_methods, p=shipping_probs)

                # Calculate shipping weight
                weight_multiplier = 1.0 + 0.1 * (urgency - 1)
                noise = np.random.lognormal(0, 0.3)
                shipping_weight = base_weight * weight_multiplier * noise

                # Generate exogenous features
                avg_repair_cycle = _get_repair_cycle(sku) + np.random.normal(0, 1)
                failure_rate = _get_failure_rate(sku, trend_position)

                # Create record
                record = {
                    "date": date,
                    "region": region,
                    "sku_category": sku,
                    "shipping_weight_kg": round(shipping_weight, 3),
                    "request_urgency": urgency,
                    "shipping_method": shipping_method,
                    "avg_repair_cycle_days": round(max(1, avg_repair_cycle), 1),
                    "failure_rate_pct": round(failure_rate, 2),
                    "day_of_week": day_of_week,
                    "month": date.month,
                    "is_quarter_end": int(is_eoq),
                }
                records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Generated {len(df):,} RMA records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Regions: {df['region'].unique().tolist()}")
    logger.info(f"SKU categories: {df['sku_category'].nunique()}")

    # Save to file
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")

    return df


def _get_shipping_probs(urgency: int) -> List[float]:
    """
    Get shipping method probabilities based on urgency level.

    Args:
        urgency: Urgency level (1=standard, 2=expedited, 3=critical)

    Returns:
        Probability distribution for [ground, express, air]
    """
    if urgency == 1:
        return [0.7, 0.25, 0.05]
    elif urgency == 2:
        return [0.3, 0.55, 0.15]
    else:
        return [0.1, 0.3, 0.6]


def _get_repair_cycle(sku: str) -> float:
    """
    Get average repair cycle time for a SKU category.

    Args:
        sku: SKU category name

    Returns:
        Average repair cycle in days
    """
    repair_cycles = {
        "CPU": 5.0,
        "GPU": 7.0,
        "RAM": 2.0,
        "SSD": 3.0,
        "HDD": 4.0,
        "PSU": 3.5,
        "MOBO": 8.0,
        "NIC": 2.5,
        "FAN": 1.5,
        "CABLE": 1.0,
        "DISPLAY": 6.0,
        "KEYBOARD": 2.0,
        "CHASSIS": 5.5,
        "COOLING": 3.0,
        "BATTERY": 2.5,
    }
    return repair_cycles.get(sku, 3.0)


def _get_failure_rate(sku: str, trend_position: float) -> float:
    """
    Get failure rate for a SKU category with trend component.

    Args:
        sku: SKU category name
        trend_position: Position in time series [0, 1]

    Returns:
        Failure rate as percentage
    """
    base_rates = {
        "CPU": 0.5,
        "GPU": 1.2,
        "RAM": 0.3,
        "SSD": 0.4,
        "HDD": 1.5,
        "PSU": 1.8,
        "MOBO": 0.8,
        "NIC": 0.4,
        "FAN": 2.0,
        "CABLE": 0.2,
        "DISPLAY": 1.0,
        "KEYBOARD": 0.6,
        "CHASSIS": 0.3,
        "COOLING": 1.2,
        "BATTERY": 1.5,
    }
    base = base_rates.get(sku, 1.0)

    # Add slight trend
    trend_factor = np.random.uniform(-0.2, 0.1) * trend_position
    noise = np.random.normal(0, 0.1)

    return max(0.1, base + trend_factor + noise)


def create_aggregated_timeseries(
    df: pd.DataFrame,
    agg_level: str = "region",
    target_col: str = "shipping_weight_kg"
) -> pd.DataFrame:
    """
    Aggregate RMA data to create time series at specified level.

    Args:
        df: Raw RMA data
        agg_level: Aggregation level ('region', 'sku', 'region_sku')
        target_col: Column to aggregate

    Returns:
        Aggregated time series DataFrame
    """
    if agg_level == "region":
        group_cols = ["date", "region"]
    elif agg_level == "sku":
        group_cols = ["date", "sku_category"]
    elif agg_level == "region_sku":
        group_cols = ["date", "region", "sku_category"]
    else:
        group_cols = ["date"]

    agg_df = df.groupby(group_cols).agg({
        target_col: "sum",
        "request_urgency": "mean",
        "avg_repair_cycle_days": "mean",
        "failure_rate_pct": "mean",
        "day_of_week": "first",
        "month": "first",
        "is_quarter_end": "first",
    }).reset_index()

    agg_df = agg_df.rename(columns={target_col: f"total_{target_col}"})

    return agg_df


if __name__ == "__main__":
    df = generate_rma_data(output_path="data/raw/rma_shipping_data.csv")
    print(f"\nGenerated {len(df):,} records")
    print(f"\nSample:\n{df.head()}")
    print(f"\nWeight statistics:\n{df['shipping_weight_kg'].describe()}")
