"""
Module: app.py
==============

Purpose: FastAPI application for model serving.

Business Context:
    Production ML systems need REST APIs for:
    - Real-time inference
    - Integration with other services
    - Monitoring and health checks
    - Documentation and testing

This FastAPI application provides:
    - /api/v1/rma/forecast - RMA shipping weight forecasting
    - /api/v1/network/classify - Network event classification
    - /api/v1/health - Health check endpoint
    - Auto-generated OpenAPI documentation

Usage:
    uvicorn mlops.serving.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
import pandas as pd

from .schemas import (
    RMAForecastRequest,
    RMAForecastResponse,
    ForecastPrediction,
    NetworkEventRequest,
    NetworkClassifyResponse,
    ClassificationResult,
    HealthResponse,
)

# ============ APP CONFIGURATION ============

app = FastAPI(
    title="Time Series ML Demonstration API",
    description="""
    API for RMA shipping weight forecasting and network event classification.

    ## Models

    ### RMA Forecasting (GRU)
    Predicts aggregate shipping weights for RMA orders across regions.
    - Input: Last 30 days of historical data
    - Output: 7-day forecast with confidence intervals

    ### Network Event Classification (K-means + LSA)
    Classifies network events into behavioral clusters.
    - Input: Single network event features
    - Output: Cluster assignment and anomaly score
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============ CORS MIDDLEWARE ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ MODEL LOADING ============
# NOTE: Models are loaded lazily on first request to reduce startup time.
#       In production, you might load them at startup instead.

_rma_engine = None
_clustering_engine = None


def get_rma_engine():
    """Get or create RMA inference engine."""
    global _rma_engine
    if _rma_engine is None:
        try:
            from mlops.inference import RMAInferenceEngine
            _rma_engine = RMAInferenceEngine("outputs/models", version="v3")
        except Exception as e:
            print(f"Failed to load RMA model: {e}")
            return None
    return _rma_engine


def get_clustering_engine():
    """Get or create clustering inference engine."""
    global _clustering_engine
    if _clustering_engine is None:
        try:
            from mlops.inference import ClusteringInferenceEngine
            _clustering_engine = ClusteringInferenceEngine("outputs/models", version="v1")
        except Exception as e:
            print(f"Failed to load clustering model: {e}")
            return None
    return _clustering_engine


# ============ ENDPOINTS ============

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health and model loading status.

    Returns service status and which models are loaded.
    """
    rma_loaded = get_rma_engine() is not None
    clustering_loaded = get_clustering_engine() is not None

    status = "healthy" if (rma_loaded or clustering_loaded) else "degraded"

    return HealthResponse(
        status=status,
        models_loaded={
            "rma_gru": rma_loaded,
            "network_clustering": clustering_loaded,
        },
        version="1.0.0"
    )


@app.post("/api/v1/rma/forecast", response_model=RMAForecastResponse, tags=["RMA Forecasting"])
async def forecast_rma(request: RMAForecastRequest):
    """
    Generate 7-day shipping weight forecast.

    Provide the last 30 days of historical data to receive
    predictions for the next 7 days.

    The GRU model processes the sequence of historical data
    and outputs predictions for multiple horizons simultaneously.
    """
    engine = get_rma_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="RMA model not available")

    # Convert request to DataFrame
    data_dicts = [point.dict() for point in request.historical_data]
    df = pd.DataFrame(data_dicts)

    try:
        # Run inference
        result = engine.predict(df, return_confidence=True)

        # Format predictions
        predictions = []
        for i in range(result["prediction_horizon"]):
            pred = ForecastPrediction(
                day=i + 1,
                predicted_weight_kg=float(result["predictions"][0, i]),
                confidence_lower=float(result["confidence_interval"]["lower"][0, i])
                    if "confidence_interval" in result else None,
                confidence_upper=float(result["confidence_interval"]["upper"][0, i])
                    if "confidence_interval" in result else None,
            )
            predictions.append(pred)

        return RMAForecastResponse(
            predictions=predictions,
            model_version=result["model_version"],
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/network/classify", response_model=NetworkClassifyResponse, tags=["Network Classification"])
async def classify_network_event(request: NetworkEventRequest):
    """
    Classify a network event.

    Provide network event features to receive cluster assignment
    and anomaly score.

    The K-means model assigns the event to the nearest
    cluster centroid. Events far from their centroid have higher
    anomaly scores.
    """
    engine = get_clustering_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Clustering model not available")

    # Convert request to DataFrame
    df = pd.DataFrame([request.dict()])

    # Fill in missing time fields
    if df["hour_of_day"].isna().any():
        df["hour_of_day"] = datetime.now().hour
    if df["day_of_week"].isna().any():
        df["day_of_week"] = datetime.now().weekday()

    try:
        # Run classification
        result = engine.classify(df)

        # Define anomaly threshold
        anomaly_threshold = 0.7

        classification = ClassificationResult(
            cluster_id=int(result["cluster_ids"][0]),
            cluster_label=result["cluster_labels"][0],
            anomaly_score=float(result["anomaly_scores"][0]),
            is_anomalous=float(result["anomaly_scores"][0]) > anomaly_threshold
        )

        return NetworkClassifyResponse(
            classification=classification,
            model_version=result["model_version"],
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/api/v1/network/clusters", tags=["Network Classification"])
async def get_cluster_info():
    """
    Get information about discovered clusters.

    Returns the interpretation and characteristics of each cluster.
    """
    engine = get_clustering_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Clustering model not available")

    return engine.get_cluster_info()


# ============ STARTUP/SHUTDOWN ============

@app.on_event("startup")
async def startup_event():
    """Log startup message."""
    print("=" * 50)
    print("Time Series ML API Starting...")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("API shutting down...")


# ============ RUN DIRECTLY ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
