"""
Module: prophet_trainer.py
==========================

Training pipeline for Prophet-based RMA shipping weight forecasting.

Unlike GRU training, Prophet:
    - Fits on the complete dataset in one call (no batches or epochs)
    - Uses built-in cross-validation for hyperparameter tuning
    - Provides uncertainty intervals automatically
    - Handles missing data gracefully
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.prophet_forecaster import ProphetForecaster, create_prophet_model
from ..utils.logging_config import get_logger, log_section


class ProphetTrainer:
    """
    Trainer for Prophet forecasting models.

    Provides a similar interface to RMATrainer for consistency,
    but handles Prophet's different training paradigm internally.

    Attributes:
        model: ProphetForecaster instance
        train_df: Training DataFrame in Prophet format
        val_df: Validation DataFrame in Prophet format
        history: Training history (fit time, cv metrics)
    """

    def __init__(
        self,
        model: ProphetForecaster,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: Dict,
    ):
        """
        Initialize the trainer.

        Args:
            model: ProphetForecaster instance
            train_df: Training data with 'ds', 'y', and regressor columns
            val_df: Validation data with same columns
            config: Training configuration dictionary
        """
        self.logger = get_logger()
        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.config = config

        # Training state
        self.history = {
            "fit_time": 0.0,
            "cv_time": 0.0,
            "cv_metrics": {},
        }
        self.is_trained = False

    def train(self) -> Dict[str, Any]:
        """
        Fit the Prophet model.

        Prophet fits on the entire training dataset at once,
        unlike neural networks which iterate over epochs.

        Returns:
            Training history dictionary
        """
        log_section(self.logger, "STARTING PROPHET TRAINING")
        self.logger.info(f"Training samples: {len(self.train_df)}")
        self.logger.info(f"Validation samples: {len(self.val_df)}")
        self.logger.info(f"Seasonality mode: {self.model.config.seasonality_mode}")
        self.logger.info(f"Regressors: {self.model.config.regressors or 'None'}")

        # Fit the model
        fit_start = time.time()
        self.model.fit(self.train_df)
        fit_time = time.time() - fit_start

        self.history["fit_time"] = fit_time
        self.logger.info(f"Model fit completed in {fit_time:.2f} seconds")

        # Run cross-validation if configured
        cv_config = self.config.get("cross_validation", {})
        if cv_config:
            cv_start = time.time()
            try:
                cv_metrics = self.model.get_cv_metrics(
                    horizon=cv_config.get("horizon", "7 days"),
                    period=cv_config.get("period", "30 days"),
                    initial=cv_config.get("initial", "365 days"),
                )
                cv_time = time.time() - cv_start

                self.history["cv_time"] = cv_time
                self.history["cv_metrics"] = cv_metrics

                self.logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
                self.logger.info(f"CV MAE: {cv_metrics.get('cv_mae', 'N/A'):.4f}")
                self.logger.info(f"CV RMSE: {cv_metrics.get('cv_rmse', 'N/A'):.4f}")
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}")
                self.history["cv_metrics"] = {"error": str(e)}

        self.is_trained = True
        log_section(self.logger, "PROPHET TRAINING COMPLETE")

        return self.history

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for test data.

        Args:
            test_df: Test DataFrame with 'ds', 'y', and regressor columns

        Returns:
            Tuple of (predictions, actuals) as numpy arrays
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Generate forecast for all test dates
        forecast = self.model.model.predict(test_df)

        predictions = forecast['yhat'].values
        actuals = test_df['y'].values

        return predictions, actuals

    def predict_future(
        self,
        periods: int = 7,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for future periods.

        Args:
            periods: Number of future periods to predict
            future_regressors: DataFrame with regressor values for future dates

        Returns:
            DataFrame with predictions including uncertainty intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(
            periods=periods,
            future_regressors=future_regressors,
            include_history=False,
        )

    def evaluate_on_validation(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.

        Returns:
            Dictionary with evaluation metrics
        """
        predictions, actuals = self.predict(self.val_df)

        mae = np.abs(predictions - actuals).mean()
        mse = ((predictions - actuals) ** 2).mean()
        rmse = np.sqrt(mse)

        # MAPE (avoiding division by zero)
        mask = actuals != 0
        if mask.sum() > 0:
            mape = np.abs((actuals[mask] - predictions[mask]) / actuals[mask]).mean() * 100
        else:
            mape = float('inf')

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
        }

    def get_components(self) -> Dict[str, pd.DataFrame]:
        """
        Get model components (trend, seasonality).

        Returns:
            Dictionary of component DataFrames
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to extract components")

        return self.model.get_components()

    def get_changepoints(self) -> pd.DataFrame:
        """
        Get detected changepoints.

        Returns:
            DataFrame with changepoint dates and magnitudes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get changepoints")

        return self.model.get_changepoints()


def compute_prophet_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute baseline metrics using naive persistence for Prophet data.

    Args:
        train_df: Training DataFrame with 'ds' and 'y' columns
        test_df: Test DataFrame with 'ds' and 'y' columns

    Returns:
        Dictionary of baseline metrics
    """
    # Naive persistence: predict last training value for all test points
    last_train_value = train_df['y'].iloc[-1]
    predictions = np.full(len(test_df), last_train_value)
    actuals = test_df['y'].values

    mae = np.abs(predictions - actuals).mean()
    mse = ((predictions - actuals) ** 2).mean()
    rmse = np.sqrt(mse)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
    }


def train_prophet_versions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
) -> Dict[str, Dict]:
    """
    Train both V1 and V2 Prophet models.

    Args:
        train_df: Training data in Prophet format
        val_df: Validation data in Prophet format
        test_df: Test data in Prophet format
        config: Configuration dictionary with 'prophet' section

    Returns:
        Dictionary with results for each version
    """
    logger = get_logger()
    results = {}

    prophet_config = config.get("models", {}).get("prophet", {})
    cv_config = prophet_config.get("cross_validation", {})

    # Version 1: Basic seasonality
    log_section(logger, "PROPHET V1: BASIC SEASONALITY")
    v1_config = prophet_config.get("v1", {})
    model_v1 = create_prophet_model("v1", v1_config)

    trainer_v1 = ProphetTrainer(
        model=model_v1,
        train_df=train_df,
        val_df=val_df,
        config={"cross_validation": cv_config},
    )
    history_v1 = trainer_v1.train()

    # Evaluate on test set
    predictions_v1, actuals_v1 = trainer_v1.predict(test_df)
    metrics_v1 = {
        "mae": float(np.abs(predictions_v1 - actuals_v1).mean()),
        "mse": float(((predictions_v1 - actuals_v1) ** 2).mean()),
        "rmse": float(np.sqrt(((predictions_v1 - actuals_v1) ** 2).mean())),
    }

    results["v1"] = {
        "model": model_v1,
        "trainer": trainer_v1,
        "history": history_v1,
        "predictions": predictions_v1,
        "actuals": actuals_v1,
        "metrics": metrics_v1,
    }

    logger.info(f"V1 Test MAE: {metrics_v1['mae']:.4f}")
    logger.info(f"V1 Test RMSE: {metrics_v1['rmse']:.4f}")

    # Version 2: With regressors
    log_section(logger, "PROPHET V2: WITH REGRESSORS")
    v2_config = prophet_config.get("v2", {})
    model_v2 = create_prophet_model("v2", v2_config)

    trainer_v2 = ProphetTrainer(
        model=model_v2,
        train_df=train_df,
        val_df=val_df,
        config={"cross_validation": cv_config},
    )
    history_v2 = trainer_v2.train()

    # Evaluate on test set
    predictions_v2, actuals_v2 = trainer_v2.predict(test_df)
    metrics_v2 = {
        "mae": float(np.abs(predictions_v2 - actuals_v2).mean()),
        "mse": float(((predictions_v2 - actuals_v2) ** 2).mean()),
        "rmse": float(np.sqrt(((predictions_v2 - actuals_v2) ** 2).mean())),
    }

    results["v2"] = {
        "model": model_v2,
        "trainer": trainer_v2,
        "history": history_v2,
        "predictions": predictions_v2,
        "actuals": actuals_v2,
        "metrics": metrics_v2,
    }

    logger.info(f"V2 Test MAE: {metrics_v2['mae']:.4f}")
    logger.info(f"V2 Test RMSE: {metrics_v2['rmse']:.4f}")

    return results
