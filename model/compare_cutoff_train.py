# model/compare_cutoff_train.py

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from config import (
    CURRENT_PRICE_COLUMNS,
    DATASET_PATHS,
    EXPERIMENT_DIR,
    PRIMARY_OIL_TYPE,
    SELECTED_FEATURES_PATH,
    TARGET_COL,
    ensure_directories,
)

# ==============================
# Experiment settings
# ==============================

TEST_START_DATE = "2026-04-01"
TEST_END_DATE = "2026-05-31"

CUTOFF_EXPERIMENTS = {
    "train_until_2024": "2024-12-31",
    "train_until_2026_03": "2026-03-31",
}


# ==============================
# Utility
# ==============================


def load_selected_features() -> dict:
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_feature_list(df: pd.DataFrame, features: list[str]) -> list[str]:
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)

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

    for col in features:
        if col not in numeric_cols:
            continue

        if col == "date":
            continue

        if any(keyword in col for keyword in leak_keywords):
            continue

        result.append(col)

    return result


def make_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    y = df[TARGET_COL].copy()

    return X, y


def make_models() -> dict:
    return {
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_prediction(
    df: pd.DataFrame,
    pred_return: np.ndarray,
    current_col: str,
) -> dict:
    """
    target은 10거래일 뒤 수익률(%).
    평가할 때는 가격으로 복원해서 MAE/RMSE 계산.
    """

    true_return = df[TARGET_COL].to_numpy()
    current_price = df[current_col].to_numpy()

    actual_future_price = current_price * (1 + true_return / 100)
    predicted_future_price = current_price * (1 + pred_return / 100)
    baseline_future_price = current_price.copy()

    abs_error = np.abs(actual_future_price - predicted_future_price)
    baseline_abs_error = np.abs(actual_future_price - baseline_future_price)

    if "target_shock" in df.columns:
        shock_mask = df["target_shock"].to_numpy().astype(bool)
    else:
        shock_mask = np.abs(true_return) >= 10.0

    normal_mask = ~shock_mask

    def mae_or_nan(mask, actual, pred):
        if mask.sum() == 0:
            return float("nan")
        return float(mean_absolute_error(actual[mask], pred[mask]))

    def rmse_or_nan(mask, actual, pred):
        if mask.sum() == 0:
            return float("nan")
        return float(mean_squared_error(actual[mask], pred[mask]) ** 0.5)

    return {
        "price_mae": float(
            mean_absolute_error(actual_future_price, predicted_future_price)
        ),
        "price_rmse": float(
            mean_squared_error(actual_future_price, predicted_future_price) ** 0.5
        ),
        "baseline_price_mae": float(
            mean_absolute_error(actual_future_price, baseline_future_price)
        ),
        "baseline_price_rmse": float(
            mean_squared_error(actual_future_price, baseline_future_price) ** 0.5
        ),
        "return_mae": float(mean_absolute_error(true_return, pred_return)),
        "return_rmse": float(mean_squared_error(true_return, pred_return) ** 0.5),
        "return_r2": float(r2_score(true_return, pred_return)),
        "normal_mae": mae_or_nan(
            normal_mask, actual_future_price, predicted_future_price
        ),
        "shock_mae": mae_or_nan(
            shock_mask, actual_future_price, predicted_future_price
        ),
        "baseline_normal_mae": mae_or_nan(
            normal_mask, actual_future_price, baseline_future_price
        ),
        "baseline_shock_mae": mae_or_nan(
            shock_mask, actual_future_price, baseline_future_price
        ),
        "normal_rmse": rmse_or_nan(
            normal_mask, actual_future_price, predicted_future_price
        ),
        "shock_rmse": rmse_or_nan(
            shock_mask, actual_future_price, predicted_future_price
        ),
        "median_abs_error": float(np.median(abs_error)),
        "max_abs_error": float(abs_error.max()),
        "within_3_pct": float((abs_error <= 3).mean() * 100),
        "within_5_pct": float((abs_error <= 5).mean() * 100),
        "within_10_pct": float((abs_error <= 10).mean() * 100),
        "actual_shock_rows": int(shock_mask.sum()),
        "actual_normal_rows": int(normal_mask.sum()),
        "baseline_median_abs_error": float(np.median(baseline_abs_error)),
        "baseline_max_abs_error": float(baseline_abs_error.max()),
    }


def save_model(
    oil_type: str,
    experiment_name: str,
    feature_set_name: str,
    model_name: str,
    feature_cols: list[str],
    model,
) -> str:
    model_path = (
        EXPERIMENT_DIR
        / f"{oil_type}_{experiment_name}_{feature_set_name}_{model_name}.pkl"
    )

    bundle = {
        "oil_type": oil_type,
        "experiment_name": experiment_name,
        "feature_set": feature_set_name,
        "model_name": model_name,
        "target_type": "return_pct",
        "feature_cols": feature_cols,
        "model": model,
    }

    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    return str(model_path)


# ==============================
# Main
# ==============================


def main():
    ensure_directories()

    oil_type = PRIMARY_OIL_TYPE
    current_col = CURRENT_PRICE_COLUMNS[oil_type]
    dataset_path = DATASET_PATHS[oil_type]

    print("=" * 80)
    print("cutoff 비교 실험 시작")
    print("=" * 80)
    print("oil_type:", oil_type)
    print("dataset:", dataset_path)
    print("current_col:", current_col)
    print("test period:", TEST_START_DATE, "~", TEST_END_DATE)

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {TARGET_COL}")

    if current_col not in df.columns:
        raise ValueError(f"현재 가격 컬럼이 없습니다: {current_col}")

    selected = load_selected_features()
    feature_sets = selected["feature_sets"]

    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)

    test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    if test_df.empty:
        raise ValueError(
            "test_df가 비었습니다. TEST_START_DATE / TEST_END_DATE를 확인하세요."
        )

    print("\n전체 데이터:")
    print("shape:", df.shape)
    print("date:", df["date"].min(), "~", df["date"].max())

    print("\n공통 Test 데이터:")
    print("shape:", test_df.shape)
    print("date:", test_df["date"].min(), "~", test_df["date"].max())

    if "target_shock" in test_df.columns:
        print("test shock rows:", int(test_df["target_shock"].sum()))

    results = []

    for experiment_name, cutoff_date in CUTOFF_EXPERIMENTS.items():
        cutoff = pd.to_datetime(cutoff_date)

        train_df = df[df["date"] <= cutoff].copy()

        # test 기간과 겹치면 안 됨
        train_df = train_df[train_df["date"] < test_start].copy()

        print("\n" + "=" * 80)
        print("Experiment:", experiment_name)
        print("=" * 80)
        print("cutoff:", cutoff_date)
        print("train shape:", train_df.shape)
        print("train date:", train_df["date"].min(), "~", train_df["date"].max())

        if "target_shock" in train_df.columns:
            print("train shock rows:", int(train_df["target_shock"].sum()))

        for feature_set_name, raw_features in feature_sets.items():
            feature_cols = clean_feature_list(df, raw_features)

            if not feature_cols:
                print("[스킵] feature 없음:", feature_set_name)
                continue

            X_train, y_train = make_xy(train_df, feature_cols)
            X_test, y_test = make_xy(test_df, feature_cols)

            models = make_models()

            print("\n" + "-" * 80)
            print("Feature set:", feature_set_name)
            print("feature_count:", len(feature_cols))
            print("-" * 80)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                pred_return = model.predict(X_test)

                metrics = evaluate_prediction(
                    test_df,
                    pred_return,
                    current_col,
                )

                model_path = save_model(
                    oil_type=oil_type,
                    experiment_name=experiment_name,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                    feature_cols=feature_cols,
                    model=model,
                )

                row = {
                    "oil_type": oil_type,
                    "experiment": experiment_name,
                    "train_cutoff": cutoff_date,
                    "test_start": TEST_START_DATE,
                    "test_end": TEST_END_DATE,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "feature_count": len(feature_cols),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "model_path": model_path,
                    **metrics,
                }

                results.append(row)

                print(
                    f"{model_name:14s} | "
                    f"MAE={metrics['price_mae']:.4f} | "
                    f"Baseline={metrics['baseline_price_mae']:.4f} | "
                    f"Normal={metrics['normal_mae']:.4f} | "
                    f"Shock={metrics['shock_mae']:.4f} | "
                    f"Median={metrics['median_abs_error']:.4f} | "
                    f"Max={metrics['max_abs_error']:.4f}"
                )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["price_mae", "shock_mae"],
        ascending=[True, True],
    ).reset_index(drop=True)

    output_csv = EXPERIMENT_DIR / "cutoff_compare_results.csv"
    output_json = EXPERIMENT_DIR / "cutoff_compare_results.json"

    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("cutoff 비교 결과 요약")
    print("=" * 80)

    show_cols = [
        "experiment",
        "train_cutoff",
        "feature_set",
        "model",
        "feature_count",
        "train_rows",
        "test_rows",
        "price_mae",
        "baseline_price_mae",
        "price_rmse",
        "normal_mae",
        "shock_mae",
        "baseline_shock_mae",
        "return_mae",
        "return_r2",
        "median_abs_error",
        "max_abs_error",
        "within_5_pct",
        "actual_shock_rows",
    ]

    existing_show_cols = [c for c in show_cols if c in result_df.columns]
    print(result_df[existing_show_cols].to_string(index=False))

    print("\n최고 성능:")
    best = result_df.iloc[0]
    for col in existing_show_cols:
        print(f"{col}: {best[col]}")

    print("\n저장 완료:")
    print("csv:", output_csv)
    print("json:", output_json)

    print("\ncutoff 비교 실험 완료")


if __name__ == "__main__":
    main()
