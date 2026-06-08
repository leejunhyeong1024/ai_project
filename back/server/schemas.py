# back/server/schemas.py

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    oil_type: str = Field(default="Dubai", description="Dubai, Brent, WTI 중 하나")
    selected_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="사용자가 선택한 feature 값",
    )


class PredictionResponse(BaseModel):
    model_type: str
    oil_type: str
    current_price: float
    predicted_return_pct: float
    predicted_price_10d: float
    horizon_trading_days: int
    message: str


class FeatureInfo(BaseModel):
    key: str
    label: str
    model_feature: str
    type: str
    unit: str
    description: str


class FeatureListResponse(BaseModel):
    features: List[FeatureInfo]


class HealthResponse(BaseModel):
    status: str
    message: str


class ModelSummaryResponse(BaseModel):
    models: dict
