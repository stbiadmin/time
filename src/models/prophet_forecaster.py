"""
Module: prophet_forecaster.py
=============================

Prophet-based time series forecasting for RMA shipping weight prediction.

Prophet is a procedure for forecasting time series data based on an additive
model where non-linear trends are fit with yearly, weekly, and daily seasonality,
plus holiday effects. It works best with time series that have strong seasonal
effects and several seasons of historical data.

This module provides two versions:
    - V1: Basic seasonality model (weekly + yearly patterns)
    - V2: With exogenous regressors (month-end, failure rate, urgency)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not installed. Install with: pip install prophet")


@dataclass
class ProphetConfig:
    """Configuration for Prophet model."""
    seasonality_mode: str = 'multiplicative'
    changepoint_prior_scale: float = 0.05
    n_changepoints: int = 25
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    seasonality_prior_scale: float = 10.0
    uncertainty_samples: int = 1000
    interval_width: float = 0.80
    regressors: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, config: Dict) -> 'ProphetConfig':
        """Create config from dictionary."""
        return cls(
            seasonality_mode=config.get('seasonality_mode', 'multiplicative'),
            changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
            n_changepoints=config.get('n_changepoints', 25),
            yearly_seasonality=config.get('yearly_seasonality', True),
            weekly_seasonality=config.get('weekly_seasonality', True),
            daily_seasonality=config.get('daily_seasonality', False),
            seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
            uncertainty_samples=config.get('uncertainty_samples', 1000),
            interval_width=config.get('interval_width', 0.80),
            regressors=config.get('regressors', None),
        )


class ProphetForecaster:
    """
    Prophet-based time series forecaster.

    Provides a scikit-learn-like interface for Prophet with support for:
    - Multiple seasonality modes (additive/multiplicative)
    - Exogenous regressors
    - Uncertainty intervals
    - Cross-validation

    Prophet expects data in a specific format:
    - 'ds': datetime column
    - 'y': target column
    - Additional columns for regressors
    """

    def __init__(self, config: Optional[ProphetConfig] = None, **kwargs):
        """
        Initialize the Prophet forecaster.

        Args:
            config: ProphetConfig object or None for defaults
            **kwargs: Override config parameters
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        if config is None:
            config = ProphetConfig()
        elif isinstance(config, dict):
            config = ProphetConfig.from_dict(config)

        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.model = None
        self.is_fitted = False
        self.regressor_columns = []
        self.train_df = None

    def _create_model(self) -> Prophet:
        """Create a new Prophet model with current configuration."""
        model = Prophet(
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            n_changepoints=self.config.n_changepoints,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            uncertainty_samples=self.config.uncertainty_samples,
            interval_width=self.config.interval_width,
        )
        return model

    def fit(self, df: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit the Prophet model.

        Args:
            df: DataFrame with columns 'ds' (datetime), 'y' (target),
                and any regressor columns specified in config

        Returns:
            self for method chaining
        """
        # Validate required columns
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")

        self.model = self._create_model()

        # Add regressors if specified
        if self.config.regressors:
            for regressor in self.config.regressors:
                if regressor in df.columns:
                    self.model.add_regressor(regressor)
                    self.regressor_columns.append(regressor)

        # Store training data for component extraction
        self.train_df = df.copy()

        # Suppress Stan output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)

        self.is_fitted = True
        return self

    def predict(
        self,
        periods: int = 7,
        future_regressors: Optional[pd.DataFrame] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions for future periods.

        Args:
            periods: Number of future periods to predict
            future_regressors: DataFrame with regressor values for future dates
            include_history: If True, include predictions for training data

        Returns:
            DataFrame with predictions (yhat, yhat_lower, yhat_upper)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='D')

        # Add regressor columns
        if self.regressor_columns and future_regressors is not None:
            for col in self.regressor_columns:
                if col in future_regressors.columns:
                    # Merge on date
                    future = future.merge(
                        future_regressors[['ds', col]],
                        on='ds',
                        how='left'
                    )

        # Fill missing regressor values with training mean
        if self.regressor_columns:
            for col in self.regressor_columns:
                if col in future.columns:
                    future[col] = future[col].fillna(self.train_df[col].mean())

        # Generate predictions
        forecast = self.model.predict(future)

        if not include_history:
            # Return only future predictions
            forecast = forecast.tail(periods)

        return forecast

    def predict_with_actuals(
        self,
        test_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions aligned with test data for evaluation.

        This method creates sliding window predictions to match the evaluation
        format used by GRU models (n_samples, prediction_horizon).

        Args:
            test_df: Test DataFrame with 'ds', 'y', and regressor columns

        Returns:
            Tuple of (predictions, actuals) as numpy arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Generate predictions for the full test period
        forecast = self.model.predict(test_df)

        predictions = forecast['yhat'].values
        actuals = test_df['y'].values

        return predictions, actuals

    def cross_validate(
        self,
        horizon: str = '7 days',
        period: str = '30 days',
        initial: str = '365 days',
    ) -> pd.DataFrame:
        """
        Run Prophet cross-validation.

        Args:
            horizon: Forecast horizon for each fold
            period: Period between cutoff dates
            initial: Initial training period

        Returns:
            DataFrame with cross-validation results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validation(
                self.model,
                horizon=horizon,
                period=period,
                initial=initial,
            )

        return cv_results

    def get_cv_metrics(
        self,
        horizon: str = '7 days',
        period: str = '30 days',
        initial: str = '365 days',
    ) -> Dict[str, float]:
        """
        Get performance metrics from cross-validation.

        Returns:
            Dictionary with MAE, MSE, RMSE, MAPE metrics
        """
        cv_results = self.cross_validate(horizon, period, initial)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics_df = performance_metrics(cv_results)

        # Average across all horizons
        return {
            'cv_mae': float(metrics_df['mae'].mean()),
            'cv_mse': float(metrics_df['mse'].mean()),
            'cv_rmse': float(metrics_df['rmse'].mean()),
            'cv_mape': float(metrics_df['mape'].mean()),
        }

    def get_components(self) -> Dict[str, pd.DataFrame]:
        """
        Get trend and seasonality components.

        Returns:
            Dictionary with 'trend', 'weekly', 'yearly' DataFrames
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting components")

        forecast = self.model.predict(self.train_df)

        components = {
            'trend': forecast[['ds', 'trend']].copy(),
        }

        if 'weekly' in forecast.columns:
            components['weekly'] = forecast[['ds', 'weekly']].copy()

        if 'yearly' in forecast.columns:
            components['yearly'] = forecast[['ds', 'yearly']].copy()

        # Add regressor effects
        for regressor in self.regressor_columns:
            col_name = f'{regressor}'
            if col_name in forecast.columns:
                components[regressor] = forecast[['ds', col_name]].copy()

        return components

    def get_changepoints(self) -> pd.DataFrame:
        """
        Get detected trend changepoints.

        Returns:
            DataFrame with changepoint dates and magnitudes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting changepoints")

        changepoints = self.model.changepoints
        deltas = self.model.params['delta'].mean(axis=0)

        return pd.DataFrame({
            'ds': changepoints,
            'delta': deltas[:len(changepoints)],
        })

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters for serialization.

        Returns:
            Dictionary with model parameters
        """
        return {
            'config': {
                'seasonality_mode': self.config.seasonality_mode,
                'changepoint_prior_scale': self.config.changepoint_prior_scale,
                'n_changepoints': self.config.n_changepoints,
                'yearly_seasonality': self.config.yearly_seasonality,
                'weekly_seasonality': self.config.weekly_seasonality,
                'daily_seasonality': self.config.daily_seasonality,
                'seasonality_prior_scale': self.config.seasonality_prior_scale,
                'uncertainty_samples': self.config.uncertainty_samples,
                'interval_width': self.config.interval_width,
                'regressors': self.config.regressors,
            },
            'regressor_columns': self.regressor_columns,
            'is_fitted': self.is_fitted,
        }


def create_prophet_model(
    version: str,
    config: Optional[Dict] = None,
) -> ProphetForecaster:
    """
    Factory function to create a Prophet model by version name.

    Args:
        version: Model version ('v1' or 'v2')
        config: Model configuration dictionary

    Returns:
        Instantiated ProphetForecaster
    """
    config = config or {}

    if version == "v1":
        # V1: Basic seasonality, no regressors
        prophet_config = ProphetConfig(
            seasonality_mode=config.get('seasonality_mode', 'multiplicative'),
            changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
            n_changepoints=config.get('n_changepoints', 25),
            yearly_seasonality=config.get('yearly_seasonality', True),
            weekly_seasonality=config.get('weekly_seasonality', True),
            daily_seasonality=config.get('daily_seasonality', False),
            seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
            uncertainty_samples=config.get('uncertainty_samples', 1000),
            interval_width=config.get('interval_width', 0.80),
            regressors=None,  # V1 has no regressors
        )
        return ProphetForecaster(config=prophet_config)

    elif version == "v2":
        # V2: With exogenous regressors
        regressors = config.get('regressors', [
            'is_month_end',
            'failure_rate_pct',
            'avg_urgency',
        ])
        prophet_config = ProphetConfig(
            seasonality_mode=config.get('seasonality_mode', 'multiplicative'),
            changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
            n_changepoints=config.get('n_changepoints', 25),
            yearly_seasonality=config.get('yearly_seasonality', True),
            weekly_seasonality=config.get('weekly_seasonality', True),
            daily_seasonality=config.get('daily_seasonality', False),
            seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
            uncertainty_samples=config.get('uncertainty_samples', 1000),
            interval_width=config.get('interval_width', 0.80),
            regressors=regressors,
        )
        return ProphetForecaster(config=prophet_config)

    else:
        raise ValueError(f"Unknown Prophet version: {version}. Use 'v1' or 'v2'.")


def get_prophet_summary(model: ProphetForecaster) -> Dict:
    """
    Get a summary of Prophet model configuration.

    Args:
        model: ProphetForecaster instance

    Returns:
        Dictionary with model summary information
    """
    return {
        'model_class': 'ProphetForecaster',
        'seasonality_mode': model.config.seasonality_mode,
        'yearly_seasonality': model.config.yearly_seasonality,
        'weekly_seasonality': model.config.weekly_seasonality,
        'n_changepoints': model.config.n_changepoints,
        'regressors': model.config.regressors or [],
        'is_fitted': model.is_fitted,
    }
