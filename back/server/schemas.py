# back/server/schemas.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# ==============================
# Health & Status
# ==============================

class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    server: str
    models_loaded: bool
    default_features_available: bool
    # 🚨 파이썬 3.9 호환을 위해 'int | None'을 'Optional[int]'로 교체!
    default_feature_count: Optional[int] = None
    default_feature_error: Optional[str] = None


# ==============================
# Model Summary
# ==============================

class ModelSummaryResponse(BaseModel):
    # 🚨 'dict[str, Any]' 대신 'Dict[str, Any]' 사용으로 호환성 확보
    models: Dict[str, Any]


# ==============================
# Feature info
# ==============================

class FeatureListResponse(BaseModel):
    feature_count: int
    # 🚨 'list[str]' 대신 'List[str]' 사용
    features: List[str]


# ==============================
# Prediction
# ==============================

class DefaultPredictionResponse(BaseModel):
    horizon_trading_days: int
    predictions: Dict[str, Any]


class SimulationRequest(BaseModel):
    # 프론트엔드 연동용 확장 필드 허용 (Optional)
    target_oil: Optional[str] = None
    oil_type: Optional[str] = "Dubai"
    feature_set_id: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    selected_features: Optional[Dict[str, Any]] = None
    shocks: Optional[Dict[str, Any]] = None


class SimulationPredictionResponse(BaseModel):
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
    
    # 프론트엔드 연동 데이터 규격 매핑용 필드 추가
    predicted: Optional[float] = None
    delta: Optional[float] = None
    activated_mode: Optional[str] = None
    factors: Optional[List[Dict[str, Any]]] = None
    
    selected_features: Dict[str, Any]
    used_features: Dict[str, Any]