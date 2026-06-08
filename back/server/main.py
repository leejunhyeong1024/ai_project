# back/server/main.py

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

try:
    from back.server.feature_builder import load_default_features
    from back.server.model_loader import (
        get_model_bundle,
        load_models,
        normalize_oil_type,
        summarize_models,
    )
    from back.server.model_router import route_model_type
    from back.server.schemas import (
        DefaultPredictionResponse,
        FeatureListResponse,
        HealthResponse,
        ModelSummaryResponse,
        SimulationPredictionResponse,
        SimulationRequest,
        StatusResponse,
    )
except ImportError:
    from feature_builder import load_default_features
    from model_loader import (
        get_model_bundle,
        load_models,
        normalize_oil_type,
        summarize_models,
    )
    from model_router import route_model_type
    from schemas import (
        DefaultPredictionResponse,
        FeatureListResponse,
        HealthResponse,
        ModelSummaryResponse,
        SimulationPredictionResponse,
        SimulationRequest,
        StatusResponse,
    )


app = FastAPI(
    title="Oil Price Prediction API",
    description="Dubai / WTI / Brent 10-trading-day oil price prediction API",
    version="1.0.0",
)

MODELS = None

CURRENT_PRICE_COLUMNS = {
    "dubai": "current_Dubai",
    "wti": "current_WTI",
    "brent": "current_Brent",
}

DISPLAY_OIL_NAMES = {
    "dubai": "Dubai",
    "wti": "WTI",
    "brent": "Brent",
}

HORIZON_TRADING_DAYS = 10


def get_models():
    global MODELS

    if MODELS is None:
        MODELS = load_models()

    return MODELS


def make_model_input(
    feature_values: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    row = {}

    for col in feature_cols:
        row[col] = feature_values.get(col, np.nan)

    X = pd.DataFrame([row], columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def predict_with_bundle(
    oil_type: str,
    model_type: str,
    feature_values: dict,
) -> dict:
    models = get_models()

    oil = normalize_oil_type(oil_type)
    bundle = get_model_bundle(models, oil, model_type)

    feature_cols = bundle["feature_cols"]
    model = bundle["model"]

    current_col = CURRENT_PRICE_COLUMNS[oil]

    if current_col not in feature_values:
        raise ValueError(f"default feature에 현재 가격 컬럼이 없습니다: {current_col}")

    current_price = float(feature_values[current_col])

    X = make_model_input(feature_values, feature_cols)
    pred_return = float(np.asarray(model.predict(X)).ravel()[0])

    predicted_price = current_price * (1 + pred_return / 100)

    return {
        "oil_type": DISPLAY_OIL_NAMES[oil],
        "model_type": model_type,
        "current_price": current_price,
        "predicted_return_pct": pred_return,
        "predicted_price_10d": float(predicted_price),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "feature_count": len(feature_cols),
        "feature_set": bundle.get("feature_set"),
        "model_name": bundle.get("model_name"),
        "train_cutoff": bundle.get("train_cutoff"),
    }


@app.get("/api/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
def status():
    status_result = {
        "server": "running",
        "models_loaded": MODELS is not None,
        "default_features_available": False,
    }

    try:
        defaults = load_default_features()
        status_result["default_feature_count"] = len(defaults)
        status_result["default_features_available"] = True
    except Exception as e:
        status_result["default_features_available"] = False
        status_result["default_feature_error"] = str(e)

    return status_result


@app.post("/api/reload-models", response_model=ModelSummaryResponse)
def reload_models():
    global MODELS

    MODELS = load_models()

    return {"models": summarize_models(MODELS)}


@app.get("/api/models", response_model=ModelSummaryResponse)
def models():
    loaded = get_models()

    return {"models": summarize_models(loaded)}


@app.get("/api/default-features", response_model=FeatureListResponse)
def default_features():
    try:
        defaults = load_default_features()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "feature_count": len(defaults),
        "features": sorted(defaults.keys()),
    }


@app.get("/api/predict/default", response_model=DefaultPredictionResponse)
def predict_default():
    try:
        defaults = load_default_features()

        predictions = {}

        for oil in ["dubai", "wti", "brent"]:
            result = predict_with_bundle(
                oil_type=oil,
                model_type="default",
                feature_values=defaults,
            )

            predictions[DISPLAY_OIL_NAMES[oil]] = result

        return {
            "horizon_trading_days": HORIZON_TRADING_DAYS,
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/predict/simulation",
    response_model=SimulationPredictionResponse,
)
def predict_simulation(request: SimulationRequest):
    try:
        oil = normalize_oil_type(request.oil_type)

        defaults = load_default_features()
        selected_features = request.selected_features or {}

        feature_values = defaults.copy()
        feature_values.update(selected_features)

        model_type = route_model_type(selected_features)

        result = predict_with_bundle(
            oil_type=oil,
            model_type=model_type,
            feature_values=feature_values,
        )

        return {
            **result,
            "selected_features": selected_features,
            "used_features": feature_values,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
