# model/make_dummy_server_models.py

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dubai_dataset.csv"
FINAL_MODEL_DIR = PROJECT_DIR / "back" / "models" / "final"
DEFAULT_MODEL_PATH = FINAL_MODEL_DIR / "default_model.pkl"
SHOCK_AWARE_MODEL_PATH = FINAL_MODEL_DIR / "shock_aware_model.pkl"

PREDICTION_DIR = PROJECT_DIR / "data" / "prediction"
LATEST_FEATURE_DEFAULTS_PATH = PREDICTION_DIR / "latest_feature_defaults.json"

TARGET_COL = "target"


def clean_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    leak_keywords = [
        "future_",
        "target",
        "target_date",
        "answer_",
        "actual_",
        "predicted_",
        "error",
    ]

    result = []

    for col in numeric_cols:
        if col == "date":
            continue

        if any(keyword in col for keyword in leak_keywords):
            continue

        result.append(col)

    return result


def make_latest_defaults(df: pd.DataFrame, feature_cols: list[str]) -> None:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    latest = df.sort_values("date").iloc[-1]

    defaults = {}

    for col in feature_cols:
        value = latest.get(col)

        if pd.isna(value):
            defaults[col] = 0.0
        else:
            defaults[col] = float(value)

    with open(LATEST_FEATURE_DEFAULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=4, ensure_ascii=False)

    print("latest_feature_defaults 저장:", LATEST_FEATURE_DEFAULTS_PATH)
    print("default feature 수:", len(defaults))


def make_dummy_model(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].replace([float("inf"), float("-inf")], pd.NA)
    y = df[TARGET_COL]

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )

    model.fit(X, y)

    return model


def save_bundle(path: Path, model, feature_cols: list[str], model_type: str) -> None:
    bundle = {
        "oil_type": "dubai",
        "feature_set": "dummy_server_test",
        "model_name": f"{model_type}_dummy_regressor",
        "target_type": "return_pct",
        "target_horizon": "10_trading_days",
        "feature_cols": feature_cols,
        "model": model,
    }

    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print("저장:", path)


def main():
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"dataset이 없습니다: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {TARGET_COL}")

    feature_cols = clean_feature_cols(df)

    if not feature_cols:
        raise ValueError("사용 가능한 feature가 없습니다.")

    print("dataset:", DATASET_PATH)
    print("shape:", df.shape)
    print("feature 수:", len(feature_cols))

    model = make_dummy_model(df, feature_cols)

    save_bundle(DEFAULT_MODEL_PATH, model, feature_cols, "default")
    save_bundle(SHOCK_AWARE_MODEL_PATH, model, feature_cols, "shock_aware")

    make_latest_defaults(df, feature_cols)

    print("\n임시 서버 테스트용 pkl 생성 완료")


if __name__ == "__main__":
    main()
