"""
Module: schemas.py
==================

Purpose: Pydantic models for API request/response validation.

Business Context:
    Strong API contracts ensure:
    - Clear documentation for consumers
    - Input validation
    - Consistent response formats
    - Type safety

Pydantic provides:
    - Automatic validation
    - JSON serialization
    - OpenAPI schema generation
    - IDE autocomplete support
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# ============ RMA FORECASTING SCHEMAS ============

class HistoricalDataPoint(BaseModel):
    """Single historical data point for RMA forecasting."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    region: str = Field(..., description="Geographic region (NA, EMEA, APAC, LATAM, ANZ)")
    sku_category: str = Field(..., description="SKU category (CPU, GPU, RAM, etc.)")
    shipping_weight_kg: float = Field(..., description="Total shipping weight in kg")
    request_urgency: int = Field(..., ge=1, le=3, description="Urgency level (1-3)")
    shipping_method: str = Field(..., description="Shipping method (ground, express, air)")
    avg_repair_cycle_days: float = Field(..., description="Average repair cycle time")
    failure_rate_pct: float = Field(..., description="Failure rate percentage")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")


class RMAForecastRequest(BaseModel):
    """Request body for RMA forecast endpoint."""
    historical_data: List[HistoricalDataPoint] = Field(
        ...,
        min_items=30,
        description="Last 30 days of historical data"
    )
    model_version: Optional[str] = Field("v3", description="Model version to use")

    class Config:
        schema_extra = {
            "example": {
                "historical_data": [
                    {
                        "date": "2024-01-01",
                        "region": "NA",
                        "sku_category": "CPU",
                        "shipping_weight_kg": 15.5,
                        "request_urgency": 2,
                        "shipping_method": "express",
                        "avg_repair_cycle_days": 5.0,
                        "failure_rate_pct": 0.5,
                        "day_of_week": 0,
                        "month": 1
                    }
                ],
                "model_version": "v3"
            }
        }


class ForecastPrediction(BaseModel):
    """Single day forecast prediction."""
    day: int = Field(..., description="Days ahead (1-7)")
    predicted_weight_kg: float = Field(..., description="Predicted shipping weight")
    confidence_lower: Optional[float] = Field(None, description="Lower confidence bound")
    confidence_upper: Optional[float] = Field(None, description="Upper confidence bound")


class RMAForecastResponse(BaseModel):
    """Response body for RMA forecast endpoint."""
    predictions: List[ForecastPrediction] = Field(..., description="7-day forecast")
    model_version: str = Field(..., description="Model version used")
    generated_at: datetime = Field(..., description="Prediction timestamp")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"day": 1, "predicted_weight_kg": 125.5, "confidence_lower": 110.0, "confidence_upper": 141.0},
                    {"day": 2, "predicted_weight_kg": 130.2, "confidence_lower": 114.7, "confidence_upper": 145.7},
                ],
                "model_version": "v3",
                "generated_at": "2024-01-15T10:30:00"
            }
        }


# ============ NETWORK CLUSTERING SCHEMAS ============

class NetworkEventRequest(BaseModel):
    """Request body for network event classification."""
    source_ip: str = Field(..., description="Source IP address")
    dest_ip: str = Field(..., description="Destination IP address")
    port: int = Field(..., ge=1, le=65535, description="Port number")
    duration_ms: float = Field(..., ge=0, description="Connection duration in ms")
    bytes_transferred: int = Field(..., ge=0, description="Bytes transferred")
    protocol: str = Field(..., description="Network protocol (TCP, UDP, ICMP)")
    log_message: str = Field(..., description="Log message text")
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, description="Hour of event")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week")

    class Config:
        schema_extra = {
            "example": {
                "source_ip": "10.0.0.1",
                "dest_ip": "192.168.1.100",
                "port": 443,
                "duration_ms": 150.5,
                "bytes_transferred": 8192,
                "protocol": "TCP",
                "log_message": "GET request to /api/v1/users completed successfully",
                "hour_of_day": 14,
                "day_of_week": 2
            }
        }


class ClassificationResult(BaseModel):
    """Classification result for a single event."""
    cluster_id: int = Field(..., description="Assigned cluster ID")
    cluster_label: str = Field(..., description="Human-readable cluster label")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly score (0-1)")
    is_anomalous: bool = Field(..., description="Whether event is considered anomalous")


class NetworkClassifyResponse(BaseModel):
    """Response body for network event classification."""
    classification: ClassificationResult = Field(..., description="Classification result")
    model_version: str = Field(..., description="Model version used")
    generated_at: datetime = Field(..., description="Classification timestamp")

    class Config:
        schema_extra = {
            "example": {
                "classification": {
                    "cluster_id": 0,
                    "cluster_label": "Normal web traffic",
                    "anomaly_score": 0.15,
                    "is_anomalous": False
                },
                "model_version": "v1",
                "generated_at": "2024-01-15T10:30:00"
            }
        }


# ============ HEALTH CHECK SCHEMAS ============

class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    version: str = Field(..., description="API version")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "rma_gru": True,
                    "network_clustering": True
                },
                "version": "1.0.0"
            }
        }
