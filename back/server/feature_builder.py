# back/server/feature_builder.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from back.server.config import (
        DEFAULT_CURRENT_PRICE_KEY,
        DUBAI_DATASET_PATH,
        LATEST_FEATURE_DEFAULTS_PATH,
        LIVE_FEATURE_DEFAULTS_PATH,
    )
    from back.server.model_router import convert_user_features_to_model_features
except ModuleNotFoundError:
    from config import (
        DEFAULT_CURRENT_PRICE_KEY,
        DUBAI_DATASET_PATH,
        LATEST_FEATURE_DEFAULTS_PATH,
        LIVE_FEATURE_DEFAULTS_PATH,
    )
    from model_router import convert_user_features_to_model_features


# ==============================
# Default feature loading
# ==============================


def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_default_features() -> dict:
    """
    우선순위:
    1. live_feature_defaults.json 있으면 사용
    2. latest_feature_defaults.json 사용
    3. 둘 다 없으면 processed dubai_dataset.csv 마지막 행으로 자동 생성
    """

    if LIVE_FEATURE_DEFAULTS_PATH.exists():
        return load_json_file(LIVE_FEATURE_DEFAULTS_PATH)

    if LATEST_FEATURE_DEFAULTS_PATH.exists():
        return load_json_file(LATEST_FEATURE_DEFAULTS_PATH)

    return create_defaults_from_latest_dataset_row()


def create_defaults_from_latest_dataset_row() -> dict:
    if not DUBAI_DATASET_PATH.exists():
        raise FileNotFoundError(
            "default feature 파일도 없고 dubai_dataset.csv도 없습니다.\n"
            f"확인 경로: {DUBAI_DATASET_PATH}"
        )

    df = pd.read_csv(DUBAI_DATASET_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"dataset이 비어 있습니다: {DUBAI_DATASET_PATH}")

    latest = df.iloc[-1]

    drop_keywords = [
        "future_",
        "target",
        "target_date",
        "answer_",
        "actual_",
        "predicted_",
        "error",
    ]

    defaults = {}

    for col, value in latest.items():
        if col == "date":
            continue

        if any(keyword in col for keyword in drop_keywords):
            continue

        try:
            if pd.isna(value):
                continue

            defaults[col] = float(value)
        except Exception:
            continue

    save_json_file(defaults, LATEST_FEATURE_DEFAULTS_PATH)

    return defaults


# ==============================
# Feature vector
# ==============================


def apply_user_features_to_defaults(
    default_features: dict,
    user_features: dict | None,
) -> dict:
    feature_values = default_features.copy()

    converted_user_features = convert_user_features_to_model_features(user_features)

    for key, value in converted_user_features.items():
        try:
            feature_values[key] = float(value)
        except Exception:
            raise ValueError(f"feature 값은 숫자여야 합니다: {key}={value}")

    return feature_values


def build_feature_vector(
    model_bundle: dict,
    user_features: dict | None = None,
) -> pd.DataFrame:
    """
    모델이 학습한 feature_cols 순서에 맞춰 1행짜리 DataFrame 생성.
    """

    if "feature_cols" not in model_bundle:
        raise ValueError("model_bundle에 feature_cols가 없습니다.")

    feature_cols = model_bundle["feature_cols"]

    default_features = load_default_features()
    feature_values = apply_user_features_to_defaults(default_features, user_features)

    row = {}

    missing_cols = []

    for col in feature_cols:
        if col in feature_values:
            row[col] = feature_values[col]
        else:
            row[col] = 0.0
            missing_cols.append(col)

    X = pd.DataFrame([row], columns=feature_cols)

    return X


def get_current_price(oil_type: str = "Dubai") -> float:
    default_features = load_default_features()

    if oil_type.lower() == "dubai":
        key = "current_Dubai"
    elif oil_type.lower() == "brent":
        key = "current_Brent"
    elif oil_type.lower() == "wti":
        key = "current_WTI"
    else:
        key = DEFAULT_CURRENT_PRICE_KEY

    if key not in default_features:
        raise ValueError(f"현재 가격 key가 default feature에 없습니다: {key}")

    return float(default_features[key])


def get_default_feature_summary() -> dict:
    defaults = load_default_features()

    summary_keys = [
        "current_Dubai",
        "current_Brent",
        "current_WTI",
        "GPRD",
        "DXY",
        "VIX",
        "US10Y",
        "crude_inventory",
        "gdelt_hormuz_risk_count",
        "gdelt_gulf_supply_disruption_count",
        "gdelt_oil_infrastructure_attack_count",
        "gdelt_avg_tone",
    ]

    return {key: defaults.get(key) for key in summary_keys if key in defaults}
