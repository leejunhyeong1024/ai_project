# model/train.py

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CURRENT_PRICE_COLUMNS,
    DATASET_PATHS,
    EXPERIMENT_DIR,
    PRIMARY_OIL_TYPE,
    SELECTED_FEATURES_PATH,
    TARGET_COL,
    TRAIN_RESULTS_JSON_PATH,
    TRAIN_RESULTS_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_directories,
)

# ==============================
# Utility
# ==============================


def load_selected_features() -> dict:
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"selected_features.json이 없습니다: {SELECTED_FEATURES_PATH}"
        )

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

        if col in ["date"]:
            continue

        if any(keyword in col for keyword in leak_keywords):
            continue

        if col not in result:
            result.append(col)

    return result


def make_models() -> dict:
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.01, max_iter=50000, random_state=42)),
            ]
        ),
        "huber": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(max_iter=1000)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


def time_series_split(df: pd.DataFrame):
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def make_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df[TARGET_COL].copy()

    return X, y


def evaluate_prediction(
    df: pd.DataFrame,
    pred_change: np.ndarray,
    current_col: str,
) -> dict:
    y_true_change = df[TARGET_COL].to_numpy()
    current_price = df[current_col].to_numpy()

    actual_future_price = current_price + y_true_change
    predicted_future_price = current_price + pred_change
    baseline_future_price = current_price

    abs_error = np.abs(actual_future_price - predicted_future_price)

    return {
        "change_mae": float(mean_absolute_error(y_true_change, pred_change)),
        "change_rmse": float(mean_squared_error(y_true_change, pred_change) ** 0.5),
        "change_r2": float(r2_score(y_true_change, pred_change)),
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
        "median_abs_error": float(np.median(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "within_3_pct": float((abs_error <= 3).mean() * 100),
        "within_5_pct": float((abs_error <= 5).mean() * 100),
        "within_10_pct": float((abs_error <= 10).mean() * 100),
        "actual_change_abs_gt_10_rows": int((np.abs(y_true_change) > 10).sum()),
        "actual_change_abs_gt_20_rows": int((np.abs(y_true_change) > 20).sum()),
    }


def save_experiment_model(
    model,
    oil_type: str,
    feature_set_name: str,
    model_name: str,
):
    model_path = EXPERIMENT_DIR / f"{oil_type}_{feature_set_name}_{model_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


# ==============================
# Main training
# ==============================


def main():
    ensure_directories()

    oil_type = PRIMARY_OIL_TYPE
    dataset_path = DATASET_PATHS[oil_type]
    current_col = CURRENT_PRICE_COLUMNS[oil_type]

    print("=" * 80)
    print("train 시작")
    print("=" * 80)
    print("oil_type:", oil_type)
    print("dataset:", dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset이 없습니다: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.sort_values("date").reset_index(drop=True)

    print("데이터 크기:", df.shape)
    print("날짜 범위:", df["date"].min(), "~", df["date"].max())

    selected = load_selected_features()
    feature_sets = selected["feature_sets"]

    train_df, val_df, test_df = time_series_split(df)

    print("\n데이터 분할:")
    print("Train:", train_df.shape, train_df["date"].min(), "~", train_df["date"].max())
    print("Val  :", val_df.shape, val_df["date"].min(), "~", val_df["date"].max())
    print("Test :", test_df.shape, test_df["date"].min(), "~", test_df["date"].max())

    models = make_models()
    results = []

    for feature_set_name, raw_feature_cols in feature_sets.items():
        feature_cols = clean_feature_list(df, raw_feature_cols)

        print("\n" + "-" * 80)
        print("Feature set:", feature_set_name)
        print("-" * 80)
        print("feature 개수:", len(feature_cols))

        X_train, y_train = make_xy(train_df, feature_cols)
        X_val, y_val = make_xy(val_df, feature_cols)
        X_test, y_test = make_xy(test_df, feature_cols)

        print("X_train:", X_train.shape)
        print("X_val  :", X_val.shape)
        print("X_test :", X_test.shape)

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

            val_metrics = evaluate_prediction(
                val_df,
                val_pred,
                current_col,
            )

            test_metrics = evaluate_prediction(
                test_df,
                test_pred,
                current_col,
            )

            model_path = save_experiment_model(
                model,
                oil_type,
                feature_set_name,
                model_name,
            )

            row = {
                "oil_type": oil_type,
                "feature_set": feature_set_name,
                "model": model_name,
                "feature_count": len(feature_cols),
                "model_path": str(model_path),
                "val_price_mae": val_metrics["price_mae"],
                "val_price_rmse": val_metrics["price_rmse"],
                "val_change_r2": val_metrics["change_r2"],
                "val_baseline_price_mae": val_metrics["baseline_price_mae"],
                "test_price_mae": test_metrics["price_mae"],
                "test_price_rmse": test_metrics["price_rmse"],
                "test_change_r2": test_metrics["change_r2"],
                "test_baseline_price_mae": test_metrics["baseline_price_mae"],
                "test_baseline_price_rmse": test_metrics["baseline_price_rmse"],
                "test_median_abs_error": test_metrics["median_abs_error"],
                "test_max_abs_error": test_metrics["max_abs_error"],
                "test_within_3_pct": test_metrics["within_3_pct"],
                "test_within_5_pct": test_metrics["within_5_pct"],
                "test_within_10_pct": test_metrics["within_10_pct"],
                "test_actual_change_abs_gt_10_rows": test_metrics[
                    "actual_change_abs_gt_10_rows"
                ],
                "test_actual_change_abs_gt_20_rows": test_metrics[
                    "actual_change_abs_gt_20_rows"
                ],
            }

            results.append(row)

            print(
                f"{model_name:14s} | "
                f"VAL MAE={val_metrics['price_mae']:.6f} | "
                f"TEST MAE={test_metrics['price_mae']:.6f} | "
                f"TEST RMSE={test_metrics['price_rmse']:.6f} | "
                f"TEST R2={test_metrics['change_r2']:.6f} | "
                f"TEST <=5$={test_metrics['within_5_pct']:.2f}%"
            )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["test_price_mae", "test_price_rmse"],
        ascending=[True, True],
    ).reset_index(drop=True)

    result_df.to_csv(TRAIN_RESULTS_PATH, index=False, encoding="utf-8-sig")

    with open(TRAIN_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("학습 결과 요약")
    print("=" * 80)

    display_cols = [
        "feature_set",
        "model",
        "feature_count",
        "test_price_mae",
        "test_price_rmse",
        "test_baseline_price_mae",
        "test_median_abs_error",
        "test_max_abs_error",
        "test_within_5_pct",
        "test_within_10_pct",
        "test_change_r2",
    ]

    print(result_df[display_cols].to_string(index=False))

    best = result_df.iloc[0]

    print("\n최고 성능:")
    print("feature_set:", best["feature_set"])
    print("model:", best["model"])
    print("feature_count:", best["feature_count"])
    print("test_price_mae:", best["test_price_mae"])
    print("test_price_rmse:", best["test_price_rmse"])
    print("baseline_price_mae:", best["test_baseline_price_mae"])
    print("model_path:", best["model_path"])

    print("\n저장 완료:")
    print("train results:", TRAIN_RESULTS_PATH)
    print("train results json:", TRAIN_RESULTS_JSON_PATH)

    print("\ntrain 완료")


if __name__ == "__main__":
    main()
