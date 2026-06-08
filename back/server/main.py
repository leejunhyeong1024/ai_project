# back/server/main.py

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

try:
    from back.server.feature_builder import (
        load_default_features,
        make_feature_vector,
        merge_user_features,
    )
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
        ModelFeatureListResponse,
        ModelSummaryResponse,
        SimulationOptionsResponse,
        SimulationPredictionResponse,
        SimulationRequest,
        StatusResponse,
    )
except ImportError:
    from feature_builder import (
        load_default_features,
        make_feature_vector,
        merge_user_features,
    )
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
        ModelFeatureListResponse,
        ModelSummaryResponse,
        SimulationOptionsResponse,
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

SIMULATION_OPTION_GROUPS = {
    "price": [
        {
            "key": "current_Dubai",
            "label": "Dubai 현재 가격",
            "description": "Dubai 원유의 현재 가격입니다.",
            "unit": "USD/barrel",
            "category": "price",
        },
        {
            "key": "current_WTI",
            "label": "WTI 현재 가격",
            "description": "WTI 원유의 현재 가격입니다.",
            "unit": "USD/barrel",
            "category": "price",
        },
        {
            "key": "current_Brent",
            "label": "Brent 현재 가격",
            "description": "Brent 원유의 현재 가격입니다.",
            "unit": "USD/barrel",
            "category": "price",
        },
    ],
    "market": [
        {
            "key": "DXY",
            "label": "달러 인덱스",
            "description": "미국 달러 강세를 나타내는 지표입니다.",
            "unit": "index",
            "category": "market",
        },
        {
            "key": "VIX",
            "label": "VIX 변동성 지수",
            "description": "시장 불확실성 또는 공포 지수로 사용됩니다.",
            "unit": "index",
            "category": "market",
        },
        {
            "key": "US10Y",
            "label": "미국 10년물 국채금리",
            "description": "미국 장기 금리 수준입니다.",
            "unit": "%",
            "category": "market",
        },
        {
            "key": "crude_inventory",
            "label": "미국 원유 재고",
            "description": "미국 원유 재고량입니다.",
            "unit": "thousand barrels",
            "category": "market",
        },
    ],
    "shock": [
        {
            "key": "gdelt_global_hormuz_risk_count",
            "label": "전세계 호르무즈 리스크",
            "description": "호르무즈 해협 관련 리스크 이벤트 수입니다. 입력 시 shock-aware 모델이 사용됩니다.",
            "unit": "count",
            "category": "shock",
        },
        {
            "key": "gdelt_MiddleEast_hormuz_risk_count",
            "label": "중동 호르무즈 리스크",
            "description": "중동 지역의 호르무즈 관련 리스크 이벤트 수입니다. 입력 시 shock-aware 모델이 사용됩니다.",
            "unit": "count",
            "category": "shock",
        },
        {
            "key": "gdelt_global_gulf_supply_disruption_count",
            "label": "전세계 공급 차질 리스크",
            "description": "원유 공급 차질 관련 이벤트 수입니다. 입력 시 shock-aware 모델이 사용됩니다.",
            "unit": "count",
            "category": "shock",
        },
        {
            "key": "gdelt_global_oil_infrastructure_attack_count",
            "label": "전세계 석유 인프라 공격",
            "description": "송유관, 정유시설, 항만 등 석유 인프라 공격 관련 이벤트 수입니다. 입력 시 shock-aware 모델이 사용됩니다.",
            "unit": "count",
            "category": "shock",
        },
    ],
    "conflict": [
        {
            "key": "MiddleEast_conflict_events",
            "label": "중동 분쟁 이벤트 수",
            "description": "중동 지역의 분쟁 이벤트 수입니다.",
            "unit": "count",
            "category": "conflict",
        },
        {
            "key": "MiddleEast_conflict_fatalities",
            "label": "중동 분쟁 사망자 수",
            "description": "중동 지역 분쟁의 사망자 수입니다.",
            "unit": "count",
            "category": "conflict",
        },
        {
            "key": "Global_conflict_events",
            "label": "전세계 분쟁 이벤트 수",
            "description": "전세계 분쟁 이벤트 수입니다.",
            "unit": "count",
            "category": "conflict",
        },
        {
            "key": "Global_conflict_fatalities",
            "label": "전세계 분쟁 사망자 수",
            "description": "전세계 분쟁 사망자 수입니다.",
            "unit": "count",
            "category": "conflict",
        },
    ],
}


def get_models():
    global MODELS

    if MODELS is None:
        MODELS = load_models()

    return MODELS


def split_selected_features_by_model(
    selected_features: dict,
    feature_cols: list[str],
) -> tuple[dict, dict]:
    applied = {}
    ignored = {}

    feature_col_set = set(feature_cols)

    for key, value in selected_features.items():
        if key in feature_col_set:
            applied[key] = value
        else:
            ignored[key] = value

    return applied, ignored


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
        raise ValueError(f"feature에 현재 가격 컬럼이 없습니다: {current_col}")

    current_price = float(feature_values[current_col])

    row = make_feature_vector(
        feature_values=feature_values,
        feature_cols=feature_cols,
    )

    X = pd.DataFrame([row], columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan)

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


@app.get(
    "/api/simulation-options",
    response_model=SimulationOptionsResponse,
)
def simulation_options():
    try:
        defaults = load_default_features()

        categories = {}

        for category, items in SIMULATION_OPTION_GROUPS.items():
            categories[category] = []

            for item in items:
                item_copy = item.copy()
                key = item_copy["key"]

                value = defaults.get(key)

                if value is not None:
                    try:
                        item_copy["default_value"] = float(value)
                    except Exception:
                        item_copy["default_value"] = None
                else:
                    item_copy["default_value"] = None

                categories[category].append(item_copy)

        return {"categories": categories}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/model-features/{oil_type}/{model_type}",
    response_model=ModelFeatureListResponse,
)
def model_features(oil_type: str, model_type: str):
    try:
        models = get_models()

        oil = normalize_oil_type(oil_type)
        model_type = str(model_type).strip().lower()

        bundle = get_model_bundle(
            models=models,
            oil_type=oil,
            model_type=model_type,
        )

        feature_cols = bundle["feature_cols"]

        return {
            "oil_type": DISPLAY_OIL_NAMES[oil],
            "model_type": model_type,
            "feature_count": len(feature_cols),
            "features": feature_cols,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        feature_values = merge_user_features(
            default_features=defaults,
            selected_features=selected_features,
        )

        model_type = route_model_type(selected_features)

        models = get_models()
        bundle = get_model_bundle(
            models=models,
            oil_type=oil,
            model_type=model_type,
        )

        feature_cols = bundle["feature_cols"]

        applied_selected_features, ignored_selected_features = (
            split_selected_features_by_model(
                selected_features=selected_features,
                feature_cols=feature_cols,
            )
        )

        result = predict_with_bundle(
            oil_type=oil,
            model_type=model_type,
            feature_values=feature_values,
        )

        return {
            **result,
            "selected_features": selected_features,
            "used_feature_count": len(feature_values),
            "applied_selected_features": applied_selected_features,
            "ignored_selected_features": ignored_selected_features,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
