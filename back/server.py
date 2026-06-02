# back/server.py

from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================
# 경로 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent  # AI_PROJECT/back
PROJECT_DIR = BASE_DIR.parent  # AI_PROJECT

FINAL_MODEL_DIR = BASE_DIR / "models" / "final"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

MODEL_PATH = FINAL_MODEL_DIR / "dubai_model.pkl"
FEATURE_COLUMNS_PATH = FINAL_MODEL_DIR / "dubai_feature_columns.pkl"
DATASET_PATH = PROCESSED_DIR / "dubai_dataset.csv"


# ==============================
# FastAPI 앱 생성
# ==============================

app = FastAPI(
    title="Dubai Oil Price Prediction API",
    description="Predict Dubai oil price 10 trading days ahead using price and GPR features.",
    version="1.0.0",
)


# 프론트 연결용 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# 요청 / 응답 모델
# ==============================


class PredictRequest(BaseModel):
    features: Optional[Dict[str, float]] = None


# ==============================
# 모델 / 데이터 로드
# ==============================


def check_required_files():
    missing_files = []

    for path in [MODEL_PATH, FEATURE_COLUMNS_PATH, DATASET_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError("필수 파일이 없습니다:\n" + "\n".join(missing_files))


def load_model_resources():
    check_required_files()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    dataset = pd.read_csv(DATASET_PATH)

    if len(dataset) == 0:
        raise ValueError("dubai_dataset.csv가 비어 있습니다.")

    return model, feature_columns, dataset


model, feature_columns, dubai_dataset = load_model_resources()


# ==============================
# Feature 생성
# ==============================


def get_latest_default_features() -> pd.Series:
    """
    dubai_dataset.csv의 가장 최신 행을 기본 feature template으로 사용.
    """

    latest_row = dubai_dataset.iloc[-1].copy()

    missing_features = [col for col in feature_columns if col not in latest_row.index]

    if missing_features:
        raise ValueError(
            "dataset에 모델 feature가 없습니다: " + ", ".join(missing_features)
        )

    default_features = latest_row[feature_columns].copy()

    return default_features


def build_model_input(
    user_features: Optional[Dict[str, float]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    default feature에 사용자가 입력한 feature만 덮어쓰기.
    """

    feature_series = get_latest_default_features()

    overridden_features = []
    ignored_features = []

    if user_features:
        for key, value in user_features.items():
            if key in feature_columns:
                feature_series[key] = value
                overridden_features.append(key)
            else:
                ignored_features.append(key)

    # 숫자 변환 방어
    feature_series = pd.to_numeric(feature_series, errors="coerce")
    feature_series = feature_series.fillna(0)

    X = pd.DataFrame([feature_series], columns=feature_columns)

    return X, overridden_features, ignored_features


def get_current_price_from_input(
    X: pd.DataFrame,
    user_features: Optional[Dict[str, float]],
) -> float:
    """
    current_Dubai는 사용자가 입력했으면 그 값을 사용하고,
    아니면 default template의 current_Dubai를 사용.
    """

    if user_features and "current_Dubai" in user_features:
        return float(user_features["current_Dubai"])

    return float(X.iloc[0]["current_Dubai"])


# ==============================
# API
# ==============================


@app.get("/")
def root():
    return {
        "message": "Dubai Oil Price Prediction API is running.",
        "available_endpoints": [
            "GET /health",
            "GET /features",
            "GET /defaults",
            "POST /predict",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns),
        "dataset_rows": len(dubai_dataset),
    }


@app.get("/features")
def get_features():
    """
    모델이 실제로 사용하는 feature 목록 반환.
    프론트에서 입력 가능한 feature 목록 만들 때 사용.
    """

    return {
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "recommended_input_features": [
            "current_Dubai",
            "GPRD",
            "GPRD_ACT",
            "GPRD_THREAT",
            "GPRD_MA7",
            "GPRD_MA30",
        ],
    }


@app.get("/defaults")
def get_defaults():
    """
    최신 데이터 기준 default feature 값 반환.
    사용자가 입력하지 않으면 이 값들이 사용됨.
    """

    default_features = get_latest_default_features()

    recommended_features = [
        "current_Dubai",
        "GPRD",
        "GPRD_ACT",
        "GPRD_THREAT",
        "GPRD_MA7",
        "GPRD_MA30",
    ]

    default_dict = {col: float(default_features[col]) for col in feature_columns}

    recommended_default_dict = {
        col: float(default_features[col])
        for col in recommended_features
        if col in default_features.index
    }

    latest_date = None
    if "date" in dubai_dataset.columns:
        latest_date = str(dubai_dataset.iloc[-1]["date"])

    return {
        "latest_data_date": latest_date,
        "feature_count": len(feature_columns),
        "recommended_defaults": recommended_default_dict,
        "all_defaults": default_dict,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Dubai유 10거래일 뒤 가격 예측.

    입력 예:
    {
        "features": {
            "current_Dubai": 82.5,
            "GPRD": 140.2,
            "GPRD_ACT": 120.1
        }
    }
    """

    try:
        user_features = request.features or {}

        X, overridden_features, ignored_features = build_model_input(user_features)

        predicted_change = float(model.predict(X)[0])
        current_price = get_current_price_from_input(X, user_features)
        predicted_price = current_price + predicted_change

        used_default_count = len(feature_columns) - len(overridden_features)

        latest_date = None
        if "date" in dubai_dataset.columns:
            latest_date = str(dubai_dataset.iloc[-1]["date"])

        return {
            "oil_type": "dubai",
            "forecast_horizon": "10 trading days",
            "latest_default_data_date": latest_date,
            "current_price": round(current_price, 4),
            "predicted_change": round(predicted_change, 4),
            "predicted_price": round(predicted_price, 4),
            "model": "Lasso",
            "feature_set": "price_gpr",
            "feature_count": len(feature_columns),
            "overridden_features": overridden_features,
            "ignored_features": ignored_features,
            "used_default_count": used_default_count,
            "note": "입력하지 않은 feature는 최신 데이터 기준 default 값을 사용했습니다.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
