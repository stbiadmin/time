"""
Module: rma_preprocessor.py
===========================

Preprocess RMA shipping data for GRU-based time series forecasting.

The preprocessing pipeline transforms raw RMA records into:
    1. Sequences of historical features (X) of length sequence_length
    2. Future target values (y) of length prediction_horizon
    3. Properly encoded categorical variables
    4. Scaled numerical features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


@dataclass
class RMADataConfig:
    """Configuration for RMA data preprocessing."""
    sequence_length: int = 30
    prediction_horizon: int = 7
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    scaler_type: str = "standard"


class RMAPreprocessor:
    """
    Preprocessor for RMA time series data.

    Handles all data transformations needed to convert raw RMA records
    into model-ready tensors. Maintains state (encoders, scalers) for inference.

    Attributes:
        config: Preprocessing configuration
        label_encoders: Dictionary of LabelEncoders for categorical features
        scaler: Scaler for numerical features
        fitted: Whether the preprocessor has been fitted
    """

    def __init__(self, config: Optional[RMADataConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or RMADataConfig()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.fitted = False

        self.categorical_features = [
            "region",
            "sku_category",
            "request_urgency",
            "shipping_method",
        ]
        self.numerical_features = [
            "avg_repair_cycle_days",
            "failure_rate_pct",
            "day_of_week",
            "month",
        ]
        self.target_column = "total_shipping_weight_kg"

    def fit_transform(
        self,
        df: pd.DataFrame,
        aggregation_level: str = "region"
    ) -> Tuple["RMATimeSeriesDataset", "RMATimeSeriesDataset", "RMATimeSeriesDataset"]:
        """
        Fit the preprocessor and transform data into train/val/test datasets.

        Args:
            df: Raw RMA DataFrame
            aggregation_level: How to aggregate ('region', 'sku', 'all')

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Aggregate data
        agg_df = self._aggregate_data(df, aggregation_level)

        # Temporal split (no future leakage)
        train_df, val_df, test_df = self._temporal_split(agg_df)

        # Fit encoders on training data only
        self._fit_encoders(train_df)
        self._fit_scaler(train_df)
        self.fitted = True

        # Transform all splits
        train_encoded = self._transform(train_df)
        val_encoded = self._transform(val_df)
        test_encoded = self._transform(test_df)

        # Create datasets
        train_dataset = self._create_dataset(train_encoded, "train")
        val_dataset = self._create_dataset(val_encoded, "val")
        test_dataset = self._create_dataset(test_encoded, "test")

        return train_dataset, val_dataset, test_dataset

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders and scalers.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If preprocessor hasn't been fitted
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        return self._transform(df)

    def _aggregate_data(
        self,
        df: pd.DataFrame,
        aggregation_level: str
    ) -> pd.DataFrame:
        """
        Aggregate raw records into time series.

        Args:
            df: Raw RMA records
            aggregation_level: Aggregation granularity

        Returns:
            Aggregated DataFrame
        """
        if aggregation_level == "region":
            group_cols = ["date", "region"]
        elif aggregation_level == "sku":
            group_cols = ["date", "sku_category"]
        else:
            group_cols = ["date"]

        agg_df = df.groupby(group_cols).agg({
            "shipping_weight_kg": "sum",
            "request_urgency": "mean",
            "avg_repair_cycle_days": "mean",
            "failure_rate_pct": "mean",
            "shipping_method": lambda x: x.mode().iloc[0] if len(x) > 0 else "ground",
            "day_of_week": "first",
            "month": "first",
        }).reset_index()

        agg_df = agg_df.rename(columns={"shipping_weight_kg": "total_shipping_weight_kg"})

        # If region is in group_cols, keep it; otherwise add a dummy
        if "region" not in agg_df.columns:
            agg_df["region"] = "ALL"
        if "sku_category" not in agg_df.columns:
            agg_df["sku_category"] = "ALL"

        # Convert urgency to int for consistency
        agg_df["request_urgency"] = agg_df["request_urgency"].round().astype(int).clip(1, 3)

        return agg_df.sort_values("date").reset_index(drop=True)

    def _temporal_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train/val/test sets.

        Args:
            df: Aggregated DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """
        Fit label encoders on categorical features.

        Args:
            df: Training DataFrame
        """
        for col in self.categorical_features:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].astype(str))

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """
        Fit scaler on numerical features.

        Args:
            df: Training DataFrame
        """
        if self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        numerical_data = df[self.numerical_features].values
        self.scaler.fit(numerical_data)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted encoders and scalers.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        df = df.copy()

        # Encode categoricals
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                df[f"{col}_encoded"] = self.label_encoders[col].transform(
                    df[col].astype(str)
                )

        # Scale numericals
        numerical_data = df[self.numerical_features].values
        scaled_data = self.scaler.transform(numerical_data)
        for i, col in enumerate(self.numerical_features):
            df[f"{col}_scaled"] = scaled_data[:, i]

        return df

    def _create_dataset(
        self,
        df: pd.DataFrame,
        split_name: str
    ) -> "RMATimeSeriesDataset":
        """
        Create a PyTorch dataset from transformed DataFrame.

        Args:
            df: Transformed DataFrame
            split_name: Name of the split ('train', 'val', 'test')

        Returns:
            RMATimeSeriesDataset instance
        """
        categorical_cols = [f"{col}_encoded" for col in self.categorical_features
                          if f"{col}_encoded" in df.columns]
        numerical_cols = [f"{col}_scaled" for col in self.numerical_features]

        return RMATimeSeriesDataset(
            df=df,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            target_col=self.target_column,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            split_name=split_name,
        )

    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for categorical features (for embeddings).

        Returns:
            Dictionary mapping feature names to vocabulary sizes
        """
        return {
            col: len(self.label_encoders[col].classes_)
            for col in self.categorical_features
            if col in self.label_encoders
        }

    def get_state_dict(self) -> Dict:
        """
        Get state dictionary for saving preprocessor.

        Returns:
            Dictionary containing all preprocessor state
        """
        return {
            "config": self.config.__dict__,
            "label_encoders": {
                col: encoder.classes_.tolist()
                for col, encoder in self.label_encoders.items()
            },
            "scaler_mean": self.scaler.mean_.tolist() if self.scaler else None,
            "scaler_std": self.scaler.scale_.tolist() if self.scaler else None,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "target_column": self.target_column,
        }

    def prepare_for_prophet(
        self,
        df: pd.DataFrame,
        include_regressors: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Prophet (different format than GRU sequences).

        Prophet expects:
        - 'ds': datetime column
        - 'y': target column
        - Additional columns for regressors

        Args:
            df: Raw RMA DataFrame
            include_regressors: Whether to include exogenous regressor columns

        Returns:
            Tuple of (train_df, val_df, test_df) in Prophet format
        """
        # Aggregate to daily level (global across all regions)
        daily_df = df.groupby("date").agg({
            "shipping_weight_kg": "sum",
            "request_urgency": "mean",
            "failure_rate_pct": "mean",
            "avg_repair_cycle_days": "mean",
            "is_quarter_end": "max",
        }).reset_index()

        # Rename to Prophet format
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(daily_df["date"]),
            "y": daily_df["shipping_weight_kg"],
        })

        if include_regressors:
            # Create is_month_end feature (last 3 days of month or quarter end)
            prophet_df["is_month_end"] = (
                (prophet_df["ds"].dt.day >= 28) |
                (daily_df["is_quarter_end"].values == 1)
            ).astype(float)

            # Add failure rate and urgency as regressors
            prophet_df["failure_rate_pct"] = daily_df["failure_rate_pct"].values
            prophet_df["avg_urgency"] = daily_df["request_urgency"].values

        # Sort by date
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

        # Apply same temporal split ratios
        n = len(prophet_df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = prophet_df.iloc[:train_end].copy()
        val_df = prophet_df.iloc[train_end:val_end].copy()
        test_df = prophet_df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def prepare_future_regressors(
        self,
        last_date: pd.Timestamp,
        periods: int = 7,
        train_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Prepare regressor values for future dates.

        For known regressors like is_month_end, we can compute exact values.
        For unknown regressors like failure_rate_pct, we use training averages.

        Args:
            last_date: Last date in training data
            periods: Number of future periods
            train_df: Training DataFrame to compute averages from

        Returns:
            DataFrame with future dates and regressor values
        """
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )

        future_df = pd.DataFrame({"ds": future_dates})

        # is_month_end is deterministic
        future_df["is_month_end"] = (future_df["ds"].dt.day >= 28).astype(float)

        # For other regressors, use recent averages
        if train_df is not None:
            recent_window = train_df.tail(30)  # Last 30 days average
            if "failure_rate_pct" in recent_window.columns:
                future_df["failure_rate_pct"] = recent_window["failure_rate_pct"].mean()
            if "avg_urgency" in recent_window.columns:
                future_df["avg_urgency"] = recent_window["avg_urgency"].mean()

        return future_df


class RMATimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for RMA time series sequences.

    Creates sliding windows over the time series where each sample
    is a sequence of `sequence_length` days and the target is the
    next `prediction_horizon` days of weights.

    Attributes:
        sequences: List of (features, target) tuples
        split_name: Name of the data split
    """

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str],
        target_col: str,
        sequence_length: int,
        prediction_horizon: int,
        split_name: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            df: Transformed DataFrame
            categorical_cols: List of encoded categorical column names
            numerical_cols: List of scaled numerical column names
            target_col: Name of target column
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Number of time steps to predict
            split_name: Name of the split
        """
        self.split_name = split_name
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.sequences = []

        n = len(df)
        min_length = sequence_length + prediction_horizon

        if n < min_length:
            print(f"Warning: {split_name} has only {n} samples, need {min_length}")
            return

        # Create sliding windows
        for i in range(n - min_length + 1):
            seq_df = df.iloc[i:i + sequence_length]
            target_df = df.iloc[i + sequence_length:i + sequence_length + prediction_horizon]

            cat_features = seq_df[categorical_cols].values.astype(np.int64)
            num_features = seq_df[numerical_cols].values.astype(np.float32)
            targets = target_df[target_col].values.astype(np.float32)

            self.sequences.append({
                "categorical": cat_features,
                "numerical": num_features,
                "target": targets,
            })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with 'categorical', 'numerical', and 'target' tensors
        """
        sample = self.sequences[idx]
        return {
            "categorical": torch.tensor(sample["categorical"]),
            "numerical": torch.tensor(sample["numerical"]),
            "target": torch.tensor(sample["target"]),
        }


def create_data_loaders(
    train_dataset: RMATimeSeriesDataset,
    val_dataset: RMATimeSeriesDataset,
    test_dataset: RMATimeSeriesDataset,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
