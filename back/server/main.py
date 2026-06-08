# back/server/main.py

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from back.server.config import (
        PREDICTION_HORIZON_TRADING_DAYS,
        ensure_server_directories,
        print_server_paths,
    )
    from back.server.feature_builder import (
        build_feature_vector,
        get_current_price,
        get_default_feature_summary,
        load_default_features,
    )
    from back.server.model_loader import load_models, summarize_loaded_models
    from back.server.model_router import get_available_features, select_model_type
    from back.server.schemas import (
        FeatureListResponse,
        HealthResponse,
        ModelSummaryResponse,
        PredictionResponse,
        SimulationRequest,
    )
except ModuleNotFoundError:
    from config import (
        PREDICTION_HORIZON_TRADING_DAYS,
        ensure_server_directories,
        print_server_paths,
    )
    from feature_builder import (
        build_feature_vector,
        get_current_price,
        get_default_feature_summary,
        load_default_features,
    )
    from model_loader import load_models, summarize_loaded_models
    from model_router import get_available_features, select_model_type
    from schemas import (
        FeatureListResponse,
        HealthResponse,
        ModelSummaryResponse,
        PredictionResponse,
        SimulationRequest,
    )


# ==============================
# App setup
# ==============================

ensure_server_directories()

app = FastAPI(
    title="Oil Price Prediction API",
    description="Dubai oil price 10-trading-day prediction server",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pkl 파일은 조금 이따 만들 예정이라고 했으므로,
# 서버 import 자체가 죽지 않게 지연 로딩 처리.
MODELS: dict | None = None


# ==============================
# Internal utils
# ==============================


def get_models() -> dict:
    global MODELS

    if MODELS is None:
        MODELS = load_models()

    return MODELS


def predict_return_with_bundle(
    model_bundle: dict,
    user_features: dict | None = None,
) -> float:
    X = build_feature_vector(model_bundle, user_features)

    # 단일 회귀 모델 bundle
    if "model" in model_bundle:
        model = model_bundle["model"]
        pred = model.predict(X)
        return float(np.asarray(pred).ravel()[0])

    # 혹시 hybrid bundle이 들어와도 일단 대응
    if all(key in model_bundle for key in ["classifier", "normal_reg", "shock_reg"]):
        classifier = model_bundle["classifier"]
        normal_reg = model_bundle["normal_reg"]
        shock_reg = model_bundle["shock_reg"]
        threshold = float(model_bundle.get("threshold", 0.5))

        prob = classifier.predict_proba(X)

        if prob.shape[1] == 1:
            shock_prob = 0.0
        else:
            shock_prob = float(prob[:, 1][0])

        if shock_prob >= threshold:
            pred = shock_reg.predict(X)
        else:
            pred = normal_reg.predict(X)

        return float(np.asarray(pred).ravel()[0])

    raise ValueError("지원하지 않는 model_bundle 구조입니다.")


def build_prediction_response(
    model_type: str,
    oil_type: str,
    pred_return_pct: float,
) -> dict:
    current_price = get_current_price(oil_type)
    predicted_price = current_price * (1 + pred_return_pct / 100)

    return {
        "model_type": model_type,
        "oil_type": oil_type,
        "current_price": float(current_price),
        "predicted_return_pct": float(pred_return_pct),
        "predicted_price_10d": float(predicted_price),
        "horizon_trading_days": PREDICTION_HORIZON_TRADING_DAYS,
        "message": f"{model_type} model prediction completed",
    }


# ==============================
# Endpoints
# ==============================


@app.get("/", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "message": "Oil price prediction server is running",
    }


@app.get("/api/paths")
def show_paths():
    print_server_paths()

    return {
        "message": "Server paths printed to terminal",
    }


@app.get("/api/models", response_model=ModelSummaryResponse)
def model_summary():
    try:
        models = get_models()
        return {
            "models": summarize_loaded_models(models),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features", response_model=FeatureListResponse)
def feature_list():
    return {
        "features": get_available_features(),
    }


@app.get("/api/default-features")
def default_feature_summary():
    try:
        return {
            "default_features": get_default_feature_summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/default", response_model=PredictionResponse)
def predict_default():
    try:
        models = get_models()

        model_type = "default"
        model_bundle = models[model_type]

        pred_return_pct = predict_return_with_bundle(
            model_bundle=model_bundle,
            user_features={},
        )

        return build_prediction_response(
            model_type=model_type,
            oil_type="Dubai",
            pred_return_pct=pred_return_pct,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/simulation", response_model=PredictionResponse)
def predict_simulation(request: SimulationRequest):
    try:
        user_features = request.selected_features or {}

        model_type = select_model_type(user_features)

        models = get_models()
        model_bundle = models[model_type]

        pred_return_pct = predict_return_with_bundle(
            model_bundle=model_bundle,
            user_features=user_features,
        )

        return build_prediction_response(
            model_type=model_type,
            oil_type=request.oil_type,
            pred_return_pct=pred_return_pct,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload-models")
def reload_models():
    """
    모델 담당자가 final pkl 파일을 교체한 뒤 서버에서 모델 재로드할 때 사용.
    """

    global MODELS

    try:
        MODELS = load_models()

        return {
            "message": "Models reloaded successfully",
            "models": summarize_loaded_models(MODELS),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
def status():
    """
    pkl이 아직 없어도 서버 상태 확인 가능하게 만든 endpoint.
    """

    status_result = {
        "server": "running",
        "models_loaded": MODELS is not None,
    }

    try:
        defaults = load_default_features()
        status_result["default_feature_count"] = len(defaults)
        status_result["default_features_available"] = True
    except Exception as e:
        status_result["default_features_available"] = False
        status_result["default_feature_error"] = str(e)

    return status_result
