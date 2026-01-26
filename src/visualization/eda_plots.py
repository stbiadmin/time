"""
Module: eda_plots.py
====================

Purpose: Exploratory Data Analysis visualizations for both ML scenarios.

Business Context:
    EDA is the first step in any ML project. Visualizations help understand
    data distributions, patterns, and relationships before modeling.

These plots are designed for:
    - Presenting to stakeholders (clean, professional appearance)
    - Identifying data quality issues
    - Understanding temporal patterns and distributions
    - Guiding feature engineering decisions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path


# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (14, 6)
FIGSIZE_LARGE = (12, 10)
DPI = 150


def plot_eda_rma(
    df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    save: bool = True
) -> None:
    """
    Generate complete EDA visualizations for RMA data.

    This function creates a comprehensive set of EDA plots:
        1. Time series overview with trend
        2. Distribution of shipping weights
        3. Regional comparisons
        4. Seasonal patterns (weekly, monthly)
        5. Correlation heatmap
        6. SKU category analysis

    Args:
        df: RMA shipping DataFrame
        output_dir: Directory to save figures
        save: Whether to save figures to disk
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating RMA EDA visualizations...")

    # ============ FIGURE 1: TIME SERIES OVERVIEW ============
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Daily totals
    daily_totals = df.groupby("date")["shipping_weight_kg"].sum()

    # Plot raw data with rolling average
    axes[0].plot(daily_totals.index, daily_totals.values, alpha=0.3, label="Daily Total")
    axes[0].plot(
        daily_totals.index,
        daily_totals.rolling(window=7).mean(),
        color="red",
        linewidth=2,
        label="7-day Moving Average"
    )
    axes[0].plot(
        daily_totals.index,
        daily_totals.rolling(window=30).mean(),
        color="green",
        linewidth=2,
        label="30-day Moving Average"
    )
    axes[0].set_title("RMA Shipping Weight Over Time", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Total Shipping Weight (kg)")
    axes[0].legend()

    # Volume by region over time
    region_daily = df.groupby(["date", "region"])["shipping_weight_kg"].sum().unstack()
    region_daily.plot(ax=axes[1], alpha=0.7)
    axes[1].set_title("Shipping Weight by Region Over Time", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Shipping Weight (kg)")
    axes[1].legend(title="Region")

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig01_rma_time_series.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 2: WEIGHT DISTRIBUTIONS ============
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Overall distribution
    axes[0, 0].hist(df["shipping_weight_kg"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Shipping Weights", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Shipping Weight (kg)")
    axes[0, 0].set_ylabel("Frequency")

    # Log-transformed distribution
    axes[0, 1].hist(np.log1p(df["shipping_weight_kg"]), bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_title("Log-Transformed Weight Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Log(Shipping Weight + 1)")
    axes[0, 1].set_ylabel("Frequency")

    # Box plot by region
    df.boxplot(column="shipping_weight_kg", by="region", ax=axes[1, 0])
    axes[1, 0].set_title("Weight Distribution by Region", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Region")
    axes[1, 0].set_ylabel("Shipping Weight (kg)")
    plt.suptitle("")  # Remove automatic title

    # Box plot by urgency
    df.boxplot(column="shipping_weight_kg", by="request_urgency", ax=axes[1, 1])
    axes[1, 1].set_title("Weight Distribution by Urgency", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Urgency Level")
    axes[1, 1].set_ylabel("Shipping Weight (kg)")
    plt.suptitle("")

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig02_rma_distributions.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 3: SEASONAL PATTERNS ============
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Weekly pattern
    weekly_avg = df.groupby("day_of_week")["shipping_weight_kg"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[0].bar(day_names, weekly_avg.values, color="steelblue", edgecolor="black")
    axes[0].set_title("Average Weight by Day of Week", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Day of Week")
    axes[0].set_ylabel("Average Shipping Weight (kg)")
    axes[0].axhline(y=weekly_avg.mean(), color="red", linestyle="--", label="Overall Mean")
    axes[0].legend()

    # Monthly pattern
    monthly_avg = df.groupby("month")["shipping_weight_kg"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[1].bar(month_names, monthly_avg.values, color="coral", edgecolor="black")
    axes[1].set_title("Average Weight by Month", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Average Shipping Weight (kg)")
    axes[1].axhline(y=monthly_avg.mean(), color="red", linestyle="--", label="Overall Mean")
    axes[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig03_rma_seasonality.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 4: CORRELATION HEATMAPS (Individual vs Aggregated) ============
    # NOTE: This comparison shows a key insight - temporal features affect VOLUME
    #        not individual weights. Correlations look very different when aggregated.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Individual Record Level ---
    numeric_cols = ["shipping_weight_kg", "request_urgency", "avg_repair_cycle_days",
                    "failure_rate_pct", "day_of_week", "month", "is_quarter_end"]
    corr_matrix_individual = df[numeric_cols].corr()

    sns.heatmap(corr_matrix_individual, annot=True, cmap="RdBu_r", center=0,
                fmt=".2f", ax=axes[0], square=True)
    axes[0].set_title("Individual Record Correlations\n(Each row = one shipment)",
                      fontsize=12, fontweight="bold")

    # --- Right: Aggregated Daily Level ---
    # Aggregate to daily totals
    daily_agg = df.groupby("date").agg({
        "shipping_weight_kg": "sum",      # Total daily weight
        "request_urgency": "mean",         # Average urgency
        "avg_repair_cycle_days": "mean",
        "failure_rate_pct": "mean",
        "day_of_week": "first",
        "month": "first",
        "is_quarter_end": "first",
    }).reset_index()

    # Add request count as a feature
    daily_counts = df.groupby("date").size().reset_index(name="request_count")
    daily_agg = daily_agg.merge(daily_counts, on="date")

    # Correlation on aggregated data
    agg_cols = ["shipping_weight_kg", "request_count", "request_urgency",
                "day_of_week", "month", "is_quarter_end"]
    corr_matrix_agg = daily_agg[agg_cols].corr()

    sns.heatmap(corr_matrix_agg, annot=True, cmap="RdBu_r", center=0,
                fmt=".2f", ax=axes[1], square=True)
    axes[1].set_title("Aggregated Daily Correlations\n(Each row = one day)",
                      fontsize=12, fontweight="bold")

    plt.suptitle("Key Insight: Temporal features affect VOLUME, not individual weights",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig04_rma_correlation.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 5: SKU CATEGORY ANALYSIS ============
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Average weight by SKU
    sku_avg = df.groupby("sku_category")["shipping_weight_kg"].mean().sort_values(ascending=True)
    sku_avg.plot(kind="barh", ax=axes[0], color="teal", edgecolor="black")
    axes[0].set_title("Average Weight by SKU Category", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Average Shipping Weight (kg)")
    axes[0].set_ylabel("SKU Category")

    # Volume by SKU
    sku_count = df["sku_category"].value_counts().sort_values(ascending=True)
    sku_count.plot(kind="barh", ax=axes[1], color="purple", edgecolor="black")
    axes[1].set_title("Request Volume by SKU Category", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Requests")
    axes[1].set_ylabel("SKU Category")

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig05_rma_sku_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"RMA EDA figures saved to: {output_path}")


def plot_eda_network(
    df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    save: bool = True
) -> None:
    """
    Generate complete EDA visualizations for network events data.

    These plots help understand:
        - Event distribution patterns
        - Feature ranges and outliers
        - Temporal patterns in network activity
        - Relationship between features

    Args:
        df: Network events DataFrame
        output_dir: Directory to save figures
        save: Whether to save figures to disk
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating Network Events EDA visualizations...")

    # ============ FIGURE 1: EVENT DISTRIBUTIONS ============
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Duration distribution (log scale)
    axes[0, 0].hist(np.log1p(df["duration_ms"]), bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Log Duration Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Log(Duration + 1)")
    axes[0, 0].set_ylabel("Frequency")

    # Bytes distribution (log scale)
    axes[0, 1].hist(np.log1p(df["bytes_transferred"]), bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_title("Log Bytes Transferred Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Log(Bytes + 1)")
    axes[0, 1].set_ylabel("Frequency")

    # Port distribution
    port_counts = df["port"].value_counts().head(20)
    port_counts.plot(kind="bar", ax=axes[1, 0], color="green", edgecolor="black")
    axes[1, 0].set_title("Top 20 Ports by Frequency", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Port")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Severity distribution
    severity_counts = df["severity"].value_counts()
    colors = {"info": "green", "warning": "yellow", "error": "orange", "critical": "red"}
    severity_colors = [colors.get(s, "gray") for s in severity_counts.index]
    severity_counts.plot(kind="bar", ax=axes[1, 1], color=severity_colors, edgecolor="black")
    axes[1, 1].set_title("Event Severity Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Severity")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig06_network_distributions.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 2: TEMPORAL PATTERNS ============
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Hour of day distribution
    hourly_counts = df["hour_of_day"].value_counts().sort_index()
    axes[0].bar(hourly_counts.index, hourly_counts.values, color="steelblue", edgecolor="black")
    axes[0].set_title("Events by Hour of Day", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Number of Events")
    axes[0].set_xticks(range(0, 24, 2))

    # Day of week distribution
    daily_counts = df["day_of_week"].value_counts().sort_index()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(day_names, daily_counts.values, color="coral", edgecolor="black")
    axes[1].set_title("Events by Day of Week", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Number of Events")

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig07_network_temporal.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 3: DURATION VS BYTES SCATTER ============
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Sample for performance if large dataset
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)

    scatter = ax.scatter(
        np.log1p(sample_df["duration_ms"]),
        np.log1p(sample_df["bytes_transferred"]),
        c=sample_df["severity"].map({"info": 0, "warning": 1, "error": 2, "critical": 3}),
        cmap="RdYlGn_r",
        alpha=0.5,
        s=20
    )
    ax.set_title("Duration vs Bytes (colored by severity)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Log(Duration + 1)")
    ax.set_ylabel("Log(Bytes + 1)")
    plt.colorbar(scatter, ax=ax, label="Severity")

    plt.tight_layout()
    if save:
        plt.savefig(output_path / "fig08_network_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # ============ FIGURE 4: TRUE CLUSTER DISTRIBUTION (if available) ============
    if "true_cluster" in df.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        cluster_counts = df["true_cluster"].value_counts()
        cluster_counts.plot(kind="bar", ax=ax, color="purple", edgecolor="black")
        ax.set_title("Ground Truth Cluster Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Cluster Type")
        ax.set_ylabel("Number of Events")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save:
            plt.savefig(output_path / "fig09_network_clusters.png", dpi=DPI, bbox_inches="tight")
        plt.close()

    print(f"Network EDA figures saved to: {output_path}")
