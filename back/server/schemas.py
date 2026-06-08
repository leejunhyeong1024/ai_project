# back/server/schemas.py

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    server: str
    models_loaded: bool
    default_feature_count: int | None = None
    default_features_available: bool
    default_feature_error: str | None = None


class SimulationRequest(BaseModel):
    oil_type: str = Field(..., examples=["Dubai", "WTI", "Brent"])
    selected_features: dict[str, Any] = Field(default_factory=dict)


class SinglePrediction(BaseModel):
    oil_type: str
    model_type: str
    current_price: float
    predicted_return_pct: float
    predicted_price_10d: float
    horizon_trading_days: int
    feature_count: int
    feature_set: str | None = None
    model_name: str | None = None
    train_cutoff: str | None = None


class DefaultPredictionResponse(BaseModel):
    horizon_trading_days: int
    predictions: dict[str, SinglePrediction]


class SimulationPredictionResponse(SinglePrediction):
    selected_features: dict[str, Any]
    used_features: dict[str, Any] | None = None


class ModelSummaryResponse(BaseModel):
    models: dict[str, Any]


class FeatureListResponse(BaseModel):
    feature_count: int
    features: list[str]
