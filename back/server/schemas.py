# back/server/schemas.py

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    server: str
    models_loaded: bool
    default_feature_count: Optional[int] = None
    default_features_available: bool
    default_feature_error: Optional[str] = None


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
    feature_set: Optional[str] = None
    model_name: Optional[str] = None
    train_cutoff: Optional[str] = None


class DefaultPredictionResponse(BaseModel):
    horizon_trading_days: int
    predictions: dict[str, SinglePrediction]


class SimulationPredictionResponse(SinglePrediction):
    selected_features: dict[str, Any]
    used_feature_count: int
    applied_selected_features: Optional[dict[str, Any]] = None
    ignored_selected_features: Optional[dict[str, Any]] = None


class ModelSummaryResponse(BaseModel):
    models: dict[str, Any]


class FeatureListResponse(BaseModel):
    feature_count: int
    features: list[str]


class ModelFeatureListResponse(BaseModel):
    oil_type: str
    model_type: str
    feature_count: int
    features: list[str]


class SimulationOptionItem(BaseModel):
    key: str
    label: str
    description: Optional[str] = None
    default_value: Optional[Union[float, int]] = None
    unit: Optional[str] = None
    category: str


class SimulationOptionsResponse(BaseModel):
    categories: dict[str, list[SimulationOptionItem]]