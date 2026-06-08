# model/train.py

from __future__ import annotations

import json
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from config import (
    CURRENT_PRICE_COLUMNS,
    DATASET_PATHS,
    EXPERIMENT_DIR,
    FORECAST_HORIZON_TRADING_DAYS,
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
# Constants
# ==============================

SHOCK_THRESHOLD_RETURN_PCT = 10.0


# ==============================
# Load / utility
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
    X = X.replace([np.inf, -np.inf], np.nan)

    y = df[TARGET_COL].copy()

    if "target_shock" in df.columns:
        y_shock = df["target_shock"].copy()
    else:
        y_shock = (df[TARGET_COL].abs() >= SHOCK_THRESHOLD_RETURN_PCT).astype(int)

    return X, y, y_shock


def calc_scale_pos_weight(y_shock: pd.Series) -> float:
    positive_count = int(y_shock.sum())
    negative_count = int(len(y_shock) - positive_count)

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


# ==============================
# Model definition
# ==============================


def make_hybrid_model(scale_pos_weight: float) -> dict:
    """
    Hybrid 구조:
    1. classifier: 현재 시점 feature로 shock 여부 예측
    2. normal_reg: normal 구간 target return 예측
    3. shock_reg: shock 구간 target return 예측

    target은 달러 변화량이 아니라 10거래일 뒤 수익률(%).
    """

    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    normal_reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    shock_reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    return {
        "classifier": classifier,
        "normal_reg": normal_reg,
        "shock_reg": shock_reg,
    }


# ==============================
# Prediction / evaluation
# ==============================


def predict_hybrid(
    X: pd.DataFrame,
    classifier,
    normal_reg,
    shock_reg,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    반환:
    - pred_return: 예측 수익률(%)
    - shock_probs: shock 확률
    - pred_shock_labels: threshold 기준 shock 예측 라벨
    """

    prob = classifier.predict_proba(X)

    if prob.shape[1] == 1:
        shock_probs = np.zeros(len(X))
    else:
        shock_probs = prob[:, 1]

    pred_shock_labels = (shock_probs >= threshold).astype(int)

    pred_return = np.zeros(len(X), dtype=float)

    normal_mask = pred_shock_labels == 0
    shock_mask = pred_shock_labels == 1

    if normal_mask.any():
        pred_return[normal_mask] = normal_reg.predict(X[normal_mask])

    if shock_mask.any():
        pred_return[shock_mask] = shock_reg.predict(X[shock_mask])

    return pred_return, shock_probs, pred_shock_labels


def evaluate_prediction(
    df: pd.DataFrame,
    pred_return: np.ndarray,
    current_col: str,
    pred_shock_labels: np.ndarray | None = None,
) -> dict:
    """
    target은 수익률(%).
    평가할 때는 반드시 가격으로 복원해서 Price MAE/RMSE 계산.

    actual_future_price = current_price * (1 + true_return / 100)
    predicted_future_price = current_price * (1 + pred_return / 100)
    baseline_future_price = current_price
    """

    y_true_return = df[TARGET_COL].to_numpy()
    current_price = df[current_col].to_numpy()

    actual_future_price = current_price * (1 + y_true_return / 100)
    predicted_future_price = current_price * (1 + pred_return / 100)
    baseline_future_price = current_price.copy()

    abs_error = np.abs(actual_future_price - predicted_future_price)

    if "target_shock" in df.columns:
        true_shock_mask = df["target_shock"].to_numpy().astype(bool)
    else:
        true_shock_mask = np.abs(y_true_return) >= SHOCK_THRESHOLD_RETURN_PCT

    true_normal_mask = ~true_shock_mask

    def mae_or_nan(mask, actual, pred):
        if mask.sum() == 0:
            return float("nan")
        return float(mean_absolute_error(actual[mask], pred[mask]))

    def rmse_or_nan(mask, actual, pred):
        if mask.sum() == 0:
            return float("nan")
        return float(mean_squared_error(actual[mask], pred[mask]) ** 0.5)

    result = {
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
        "return_mae": float(mean_absolute_error(y_true_return, pred_return)),
        "return_rmse": float(mean_squared_error(y_true_return, pred_return) ** 0.5),
        "return_r2": float(r2_score(y_true_return, pred_return)),
        "normal_mae": mae_or_nan(
            true_normal_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "shock_mae": mae_or_nan(
            true_shock_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "baseline_normal_mae": mae_or_nan(
            true_normal_mask,
            actual_future_price,
            baseline_future_price,
        ),
        "baseline_shock_mae": mae_or_nan(
            true_shock_mask,
            actual_future_price,
            baseline_future_price,
        ),
        "normal_rmse": rmse_or_nan(
            true_normal_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "shock_rmse": rmse_or_nan(
            true_shock_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "median_abs_error": float(np.median(abs_error)),
        "max_abs_error": float(abs_error.max()),
        "within_3_pct": float((abs_error <= 3).mean() * 100),
        "within_5_pct": float((abs_error <= 5).mean() * 100),
        "within_10_pct": float((abs_error <= 10).mean() * 100),
        "actual_shock_rows": int(true_shock_mask.sum()),
        "actual_normal_rows": int(true_normal_mask.sum()),
    }

    if pred_shock_labels is not None:
        pred_shock_labels = np.asarray(pred_shock_labels).astype(bool)

        tp = int((pred_shock_labels & true_shock_mask).sum())
        fp = int((pred_shock_labels & ~true_shock_mask).sum())
        fn = int((~pred_shock_labels & true_shock_mask).sum())
        tn = int((~pred_shock_labels & ~true_shock_mask).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        result.update(
            {
                "predicted_shock_rows": int(pred_shock_labels.sum()),
                "shock_true_positive": tp,
                "shock_false_positive": fp,
                "shock_false_negative": fn,
                "shock_true_negative": tn,
                "shock_precision": float(precision),
                "shock_recall": float(recall),
            }
        )

    return result


def tune_threshold(
    val_df: pd.DataFrame,
    X_val: pd.DataFrame,
    classifier,
    normal_reg,
    shock_reg,
    current_col: str,
) -> tuple[float, dict]:
    """
    test를 보지 않고 val 기준으로 threshold 선택.
    기준은 전체 Price MAE 최소.
    """

    thresholds = [
        0.02,
        0.03,
        0.05,
        0.07,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    best_threshold = thresholds[0]
    best_metrics = None

    for threshold in thresholds:
        val_pred, _, val_pred_shock = predict_hybrid(
            X_val,
            classifier,
            normal_reg,
            shock_reg,
            threshold,
        )

        metrics = evaluate_prediction(
            val_df,
            val_pred,
            current_col,
            pred_shock_labels=val_pred_shock,
        )

        if best_metrics is None or metrics["price_mae"] < best_metrics["price_mae"]:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def save_model_bundle(
    oil_type: str,
    feature_set_name: str,
    model_name: str,
    feature_cols: list[str],
    threshold: float,
    scale_pos_weight: float,
    classifier,
    normal_reg,
    shock_reg,
) -> str:
    model_path = EXPERIMENT_DIR / f"{oil_type}_{feature_set_name}_{model_name}.pkl"

    bundle = {
        "oil_type": oil_type,
        "feature_set": feature_set_name,
        "model_name": model_name,
        "target_type": "return_pct",
        "target_horizon": f"{FORECAST_HORIZON_TRADING_DAYS}_trading_days",
        "shock_definition": f"abs(target_return_pct) >= {SHOCK_THRESHOLD_RETURN_PCT}",
        "feature_cols": feature_cols,
        "threshold": threshold,
        "scale_pos_weight": scale_pos_weight,
        "classifier": classifier,
        "normal_reg": normal_reg,
        "shock_reg": shock_reg,
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
    print("hybrid train 시작")
    print("=" * 80)
    print("oil_type:", oil_type)
    print("dataset:", dataset_path)
    print("current_col:", current_col)
    print("target:", TARGET_COL, f"({FORECAST_HORIZON_TRADING_DAYS}거래일 뒤 수익률 %)")
    print("shock 기준:", f"abs(target) >= {SHOCK_THRESHOLD_RETURN_PCT}%")

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset이 없습니다: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.sort_values("date").reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {TARGET_COL}")

    if current_col not in df.columns:
        raise ValueError(f"현재 가격 컬럼이 없습니다: {current_col}")

    if "target_shock" not in df.columns:
        df["target_shock"] = (
            df[TARGET_COL].abs() >= SHOCK_THRESHOLD_RETURN_PCT
        ).astype(int)

    train_df, val_df, test_df = time_series_split(df)

    print("\n데이터 크기:", df.shape)
    print("날짜 범위:", df["date"].min(), "~", df["date"].max())

    print("\n데이터 분할:")
    print(
        "Train:",
        train_df.shape,
        train_df["date"].min(),
        "~",
        train_df["date"].max(),
        "| shock rows:",
        int(train_df["target_shock"].sum()),
    )
    print(
        "Val  :",
        val_df.shape,
        val_df["date"].min(),
        "~",
        val_df["date"].max(),
        "| shock rows:",
        int(val_df["target_shock"].sum()),
    )
    print(
        "Test :",
        test_df.shape,
        test_df["date"].min(),
        "~",
        test_df["date"].max(),
        "| shock rows:",
        int(test_df["target_shock"].sum()),
    )

    selected = load_selected_features()
    feature_sets = selected["feature_sets"]

    results = []

    for feature_set_name, raw_features in feature_sets.items():
        print("\n" + "-" * 80)
        print("Feature set:", feature_set_name)
        print("-" * 80)

        feature_cols = clean_feature_list(df, raw_features)

        if not feature_cols:
            print("[스킵] 사용할 feature가 없습니다.")
            continue

        X_train, y_train, y_train_shock = make_xy(train_df, feature_cols)
        X_val, y_val, y_val_shock = make_xy(val_df, feature_cols)
        X_test, y_test, y_test_shock = make_xy(test_df, feature_cols)

        print("feature 개수:", len(feature_cols))
        print("X_train:", X_train.shape)
        print("X_val  :", X_val.shape)
        print("X_test :", X_test.shape)

        positive_count = int(y_train_shock.sum())
        negative_count = int(len(y_train_shock) - positive_count)
        scale_pos_weight = calc_scale_pos_weight(y_train_shock)

        print(
            "shock classifier class balance:",
            f"positive={positive_count}",
            f"negative={negative_count}",
            f"scale_pos_weight={scale_pos_weight:.4f}",
        )

        model_name = "hybrid_xgb_rf_ridge"

        parts = make_hybrid_model(scale_pos_weight=scale_pos_weight)

        classifier = parts["classifier"]
        normal_reg = parts["normal_reg"]
        shock_reg = parts["shock_reg"]

        # 1. Shock classifier 학습
        classifier.fit(X_train, y_train_shock)

        # 2. Normal / shock regressor 분리 학습
        train_normal_mask = y_train_shock.to_numpy() == 0
        train_shock_mask = y_train_shock.to_numpy() == 1

        normal_reg.fit(X_train[train_normal_mask], y_train[train_normal_mask])

        if train_shock_mask.sum() > 0:
            shock_reg.fit(X_train[train_shock_mask], y_train[train_shock_mask])
        else:
            shock_reg = deepcopy(normal_reg)

        # 3. Val 기준 threshold 선택
        best_threshold, val_metrics = tune_threshold(
            val_df=val_df,
            X_val=X_val,
            classifier=classifier,
            normal_reg=normal_reg,
            shock_reg=shock_reg,
            current_col=current_col,
        )

        # 4. Test 최종 평가
        test_pred, test_shock_probs, test_pred_shock = predict_hybrid(
            X_test,
            classifier,
            normal_reg,
            shock_reg,
            best_threshold,
        )

        test_metrics = evaluate_prediction(
            test_df,
            test_pred,
            current_col,
            pred_shock_labels=test_pred_shock,
        )

        model_path = save_model_bundle(
            oil_type=oil_type,
            feature_set_name=feature_set_name,
            model_name=model_name,
            feature_cols=feature_cols,
            threshold=best_threshold,
            scale_pos_weight=scale_pos_weight,
            classifier=classifier,
            normal_reg=normal_reg,
            shock_reg=shock_reg,
        )

        row = {
            "oil_type": oil_type,
            "feature_set": feature_set_name,
            "model": model_name,
            "feature_count": len(feature_cols),
            "threshold": best_threshold,
            "scale_pos_weight": scale_pos_weight,
            "model_path": model_path,
        }

        for key, value in val_metrics.items():
            row[f"val_{key}"] = value

        for key, value in test_metrics.items():
            row[f"test_{key}"] = value

        results.append(row)

        print(f"model: {model_name}")
        print(f"best threshold from val: {best_threshold:.2f}")
        print(
            "VAL  | "
            f"Price MAE={val_metrics['price_mae']:.4f} | "
            f"Baseline={val_metrics['baseline_price_mae']:.4f} | "
            f"Normal MAE={val_metrics['normal_mae']:.4f} | "
            f"Shock MAE={val_metrics['shock_mae']:.4f} | "
            f"Pred shock={val_metrics.get('predicted_shock_rows', 0)} | "
            f"Recall={val_metrics.get('shock_recall', float('nan')):.4f}"
        )
        print(
            "TEST | "
            f"Price MAE={test_metrics['price_mae']:.4f} | "
            f"Baseline={test_metrics['baseline_price_mae']:.4f} | "
            f"Normal MAE={test_metrics['normal_mae']:.4f} | "
            f"Shock MAE={test_metrics['shock_mae']:.4f} | "
            f"Median={test_metrics['median_abs_error']:.4f} | "
            f"Max={test_metrics['max_abs_error']:.4f} | "
            f"Pred shock={test_metrics.get('predicted_shock_rows', 0)} | "
            f"Recall={test_metrics.get('shock_recall', float('nan')):.4f}"
        )
        print(
            "Shock clf | "
            f"TP={test_metrics.get('shock_true_positive', 0)} | "
            f"FP={test_metrics.get('shock_false_positive', 0)} | "
            f"FN={test_metrics.get('shock_false_negative', 0)} | "
            f"Precision={test_metrics.get('shock_precision', float('nan')):.4f} | "
            f"Recall={test_metrics.get('shock_recall', float('nan')):.4f}"
        )

    if not results:
        raise RuntimeError("학습 결과가 없습니다.")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        ["test_price_mae", "test_shock_mae"],
        ascending=[True, True],
    ).reset_index(drop=True)

    result_df.to_csv(TRAIN_RESULTS_PATH, index=False, encoding="utf-8-sig")

    with open(TRAIN_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("하이브리드 학습 결과 요약")
    print("=" * 80)

    show_cols = [
        "feature_set",
        "model",
        "feature_count",
        "threshold",
        "scale_pos_weight",
        "test_price_mae",
        "test_price_rmse",
        "test_baseline_price_mae",
        "test_return_mae",
        "test_return_r2",
        "test_normal_mae",
        "test_shock_mae",
        "test_baseline_shock_mae",
        "test_median_abs_error",
        "test_max_abs_error",
        "test_within_5_pct",
        "test_actual_shock_rows",
        "test_predicted_shock_rows",
        "test_shock_precision",
        "test_shock_recall",
    ]

    existing_show_cols = [col for col in show_cols if col in result_df.columns]
    print(result_df[existing_show_cols].to_string(index=False))

    best = result_df.iloc[0]

    print("\n최고 성능:")
    for col in existing_show_cols:
        print(f"{col}: {best[col]}")

    print("\n저장 완료:")
    print("train results:", TRAIN_RESULTS_PATH)
    print("train results json:", TRAIN_RESULTS_JSON_PATH)

    print("\nhybrid train 완료")


if __name__ == "__main__":
    main()
