# model/train_final_candidates.py

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from config import (
    CURRENT_PRICE_COLUMNS,
    DATASET_PATHS,
    EXPERIMENT_DIR,
    FORECAST_HORIZON_TRADING_DAYS,
    SELECTED_FEATURES_PATH,
    SELECTED_FEATURES_PATHS,
    TARGET_COL,
    ensure_directories,
)

# ==============================
# Experiment settings
# ==============================

OIL_TYPES = ["dubai", "wti", "brent"]

MODEL_MODES = {
    "default": {
        "train_cutoff": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
        "selection_metric": "price_mae",
    },
    "shock_aware": {
        "train_cutoff": "2026-03-31",
        "test_start": "2026-04-01",
        "test_end": None,
        "selection_metric": "shock_mae",
    },
}

FEATURE_SET_CANDIDATES = [
    "price_momentum_gpr",
    "price_momentum_gpr_selected_extra_10",
    "price_momentum_gpr_selected_extra_20",
    "price_momentum_gpr_selected_extra_30",
    "price_momentum_gpr_gdelt_event",
    "price_momentum_gpr_gdelt_event_tone",
    "price_momentum_gpr_region_conflict",
    "price_momentum_gpr_region_conflict_gdelt",
]

OUTPUT_CSV_PATH = EXPERIMENT_DIR / "final_candidate_results.csv"
OUTPUT_JSON_PATH = EXPERIMENT_DIR / "final_candidate_results.json"
MODEL_CANDIDATE_DIR = EXPERIMENT_DIR / "final_candidates"


# ==============================
# Utility
# ==============================


def load_selected_features(oil_type: str) -> dict:
    path = SELECTED_FEATURES_PATHS.get(oil_type, SELECTED_FEATURES_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"{oil_type} selected_features 파일이 없습니다: {path}\n"
            f"먼저 python3 model/select_features.py 를 실행하세요."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


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
        if col == "date":
            continue

        if col not in numeric_cols:
            continue

        if any(keyword in col for keyword in leak_keywords):
            continue

        result.append(col)

    return result


def split_by_cutoff(
    df: pd.DataFrame,
    train_cutoff: str,
    test_start: str,
    test_end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(train_cutoff)
    start = pd.to_datetime(test_start)

    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] >= start].copy()

    if test_end is not None:
        end = pd.to_datetime(test_end)
        test_df = test_df[test_df["date"] <= end].copy()

    return train_df, test_df


def make_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    y = df[TARGET_COL].copy()

    return X, y


def get_current_col(oil_type: str) -> str:
    if oil_type in CURRENT_PRICE_COLUMNS:
        return CURRENT_PRICE_COLUMNS[oil_type]

    fallback = {
        "dubai": "current_Dubai",
        "wti": "current_WTI",
        "brent": "current_Brent",
    }

    if oil_type not in fallback:
        raise ValueError(f"지원하지 않는 oil_type입니다: {oil_type}")

    return fallback[oil_type]


# ==============================
# Models
# ==============================


def make_models() -> dict:
    return {
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
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
                        n_estimators=300,
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
                        n_estimators=300,
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
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "lasso": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    Lasso(
                        alpha=0.001,
                        max_iter=50000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "huber": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    HuberRegressor(
                        epsilon=1.35,
                        alpha=0.001,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "extra_trees_shallow": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost_shallow": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        max_depth=2,
                        learning_rate=0.02,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=10.0,
                        reg_alpha=1.0,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ==============================
# Evaluation
# ==============================


def evaluate_prediction(
    df: pd.DataFrame,
    pred_return: np.ndarray,
    current_col: str,
) -> dict:
    true_return = df[TARGET_COL].to_numpy(dtype=float)
    current_price = df[current_col].to_numpy(dtype=float)

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
        if int(mask.sum()) == 0:
            return float("nan")

        return float(mean_absolute_error(actual[mask], pred[mask]))

    def rmse_or_nan(mask, actual, pred):
        if int(mask.sum()) == 0:
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
            normal_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "shock_mae": mae_or_nan(
            shock_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "baseline_normal_mae": mae_or_nan(
            normal_mask,
            actual_future_price,
            baseline_future_price,
        ),
        "baseline_shock_mae": mae_or_nan(
            shock_mask,
            actual_future_price,
            baseline_future_price,
        ),
        "normal_rmse": rmse_or_nan(
            normal_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "shock_rmse": rmse_or_nan(
            shock_mask,
            actual_future_price,
            predicted_future_price,
        ),
        "median_abs_error": float(np.median(abs_error)),
        "max_abs_error": float(abs_error.max()),
        "baseline_median_abs_error": float(np.median(baseline_abs_error)),
        "baseline_max_abs_error": float(baseline_abs_error.max()),
        "within_3_pct": float((abs_error <= 3).mean() * 100),
        "within_5_pct": float((abs_error <= 5).mean() * 100),
        "within_10_pct": float((abs_error <= 10).mean() * 100),
        "actual_shock_rows": int(shock_mask.sum()),
        "actual_normal_rows": int(normal_mask.sum()),
    }


def save_candidate_model(
    model,
    oil_type: str,
    model_mode: str,
    feature_set_name: str,
    regressor_name: str,
    feature_cols: list[str],
    train_cutoff: str,
) -> str:
    MODEL_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{oil_type}_{model_mode}_{feature_set_name}_{regressor_name}.pkl"
    path = MODEL_CANDIDATE_DIR / filename

    bundle = {
        "oil_type": oil_type,
        "model_mode": model_mode,
        "feature_set": feature_set_name,
        "model_name": regressor_name,
        "target_type": "return_pct",
        "target_horizon": f"{FORECAST_HORIZON_TRADING_DAYS}_trading_days",
        "train_cutoff": train_cutoff,
        "feature_cols": feature_cols,
        "model": model,
    }

    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    return str(path)


# ==============================
# Main
# ==============================


def main():
    ensure_directories()
    MODEL_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("최종 후보 모델 동시 테스트 시작")
    print("=" * 80)

    results = []

    for oil_type in OIL_TYPES:
        dataset_path = DATASET_PATHS[oil_type]
        current_col = get_current_col(oil_type)

        selected = load_selected_features(oil_type)
        feature_sets = selected["feature_sets"]

        available_feature_sets = [
            name for name in FEATURE_SET_CANDIDATES if name in feature_sets
        ]

        missing_feature_sets = [
            name for name in FEATURE_SET_CANDIDATES if name not in feature_sets
        ]

        print("\n" + "=" * 80)
        print(f"{oil_type.upper()} selected feature sets")
        print("=" * 80)

        for name in available_feature_sets:
            print("-", name, len(feature_sets[name]))

        if missing_feature_sets:
            print("\n[주의] selected_features에 없는 feature set:")
            for name in missing_feature_sets:
                print("-", name)

        if not available_feature_sets:
            print(f"[스킵] {oil_type}: 사용 가능한 feature set 없음")
            continue

        if not dataset_path.exists():
            print(f"[스킵] dataset 없음: {dataset_path}")
            continue

        df = pd.read_csv(dataset_path)
        df = clean_date_col(df)

        if TARGET_COL not in df.columns:
            raise ValueError(f"{oil_type}: target 컬럼이 없습니다: {TARGET_COL}")

        if current_col not in df.columns:
            raise ValueError(f"{oil_type}: 현재 가격 컬럼이 없습니다: {current_col}")

        print("\n" + "=" * 80)
        print(f"OIL TYPE: {oil_type.upper()}")
        print("=" * 80)
        print("dataset:", dataset_path)
        print("shape:", df.shape)
        print("date:", df["date"].min(), "~", df["date"].max())
        print("current_col:", current_col)

        if "target_shock" in df.columns:
            print("total shock rows:", int(df["target_shock"].sum()))

        for model_mode, mode_cfg in MODEL_MODES.items():
            train_cutoff = mode_cfg["train_cutoff"]
            test_start = mode_cfg["test_start"]
            test_end = mode_cfg["test_end"]

            train_df, test_df = split_by_cutoff(
                df=df,
                train_cutoff=train_cutoff,
                test_start=test_start,
                test_end=test_end,
            )

            if train_df.empty or test_df.empty:
                print(f"[스킵] {oil_type} {model_mode}: train/test 비어 있음")
                continue

            print("\n" + "-" * 80)
            print(f"MODE: {model_mode}")
            print("-" * 80)
            print("train cutoff:", train_cutoff)
            print("test period:", test_start, "~", test_end or "last")
            print(
                "train:",
                train_df.shape,
                train_df["date"].min(),
                "~",
                train_df["date"].max(),
            )
            print(
                "test :",
                test_df.shape,
                test_df["date"].min(),
                "~",
                test_df["date"].max(),
            )

            if "target_shock" in train_df.columns:
                print("train shock rows:", int(train_df["target_shock"].sum()))

            if "target_shock" in test_df.columns:
                print("test shock rows:", int(test_df["target_shock"].sum()))

            for feature_set_name in available_feature_sets:
                raw_features = feature_sets[feature_set_name]
                feature_cols = clean_feature_list(df, raw_features)

                if not feature_cols:
                    print(f"[스킵] {feature_set_name}: feature 없음")
                    continue

                X_train, y_train = make_xy(train_df, feature_cols)
                X_test, _ = make_xy(test_df, feature_cols)

                print(
                    "\nFeature set:",
                    feature_set_name,
                    "| feature count:",
                    len(feature_cols),
                )

                models = make_models()

                for regressor_name, model in models.items():
                    try:
                        model.fit(X_train, y_train)

                        pred_return = model.predict(X_test)
                        pred_return = np.asarray(pred_return).ravel()

                        metrics = evaluate_prediction(
                            df=test_df,
                            pred_return=pred_return,
                            current_col=current_col,
                        )

                        model_path = save_candidate_model(
                            model=model,
                            oil_type=oil_type,
                            model_mode=model_mode,
                            feature_set_name=feature_set_name,
                            regressor_name=regressor_name,
                            feature_cols=feature_cols,
                            train_cutoff=train_cutoff,
                        )

                        row = {
                            "oil_type": oil_type,
                            "model_mode": model_mode,
                            "train_cutoff": train_cutoff,
                            "test_start": test_start,
                            "test_end": test_end or str(test_df["date"].max().date()),
                            "feature_set": feature_set_name,
                            "regressor": regressor_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train_df),
                            "test_rows": len(test_df),
                            "model_path": model_path,
                            **metrics,
                        }

                        results.append(row)

                        print(
                            f"{regressor_name:14s} | "
                            f"MAE={metrics['price_mae']:.4f} | "
                            f"BASE={metrics['baseline_price_mae']:.4f} | "
                            f"NORMAL={metrics['normal_mae']:.4f} | "
                            f"SHOCK={metrics['shock_mae']:.4f} | "
                            f"MAX={metrics['max_abs_error']:.4f} | "
                            f"R2={metrics['return_r2']:.4f}"
                        )

                    except Exception as e:
                        print(
                            f"[실패] {oil_type} {model_mode} "
                            f"{feature_set_name} {regressor_name}: {e}"
                        )

                        row = {
                            "oil_type": oil_type,
                            "model_mode": model_mode,
                            "train_cutoff": train_cutoff,
                            "test_start": test_start,
                            "test_end": test_end,
                            "feature_set": feature_set_name,
                            "regressor": regressor_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train_df),
                            "test_rows": len(test_df),
                            "error": str(e),
                        }

                        results.append(row)

    if not results:
        raise RuntimeError("결과가 없습니다.")

    result_df = pd.DataFrame(results)

    if "price_mae" not in result_df.columns:
        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        raise RuntimeError(
            "성공한 실험 결과가 없습니다. price_mae 컬럼이 생성되지 않았습니다."
        )

    valid_df = result_df[result_df["price_mae"].notna()].copy()

    if valid_df.empty:
        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        raise RuntimeError("성공한 실험 결과가 없습니다.")

    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("전체 결과 저장 완료")
    print("=" * 80)
    print("csv:", OUTPUT_CSV_PATH)
    print("json:", OUTPUT_JSON_PATH)
    print("candidate model dir:", MODEL_CANDIDATE_DIR)

    print("\n" + "=" * 80)
    print("그룹별 추천 후보")
    print("=" * 80)

    for oil_type in OIL_TYPES:
        for model_mode in MODEL_MODES.keys():
            group = valid_df[
                (valid_df["oil_type"] == oil_type)
                & (valid_df["model_mode"] == model_mode)
            ].copy()

            if group.empty:
                continue

            if model_mode == "default":
                group = group.sort_values(
                    [
                        "price_mae",
                        "normal_mae",
                        "median_abs_error",
                        "max_abs_error",
                    ],
                    ascending=[True, True, True, True],
                )

                metric_name = "price_mae"

            else:
                group = group.sort_values(
                    [
                        "shock_mae",
                        "max_abs_error",
                        "price_mae",
                        "normal_mae",
                    ],
                    ascending=[True, True, True, True],
                )

                metric_name = "shock_mae"

            best = group.iloc[0]

            print("\n" + "-" * 80)
            print(f"{oil_type.upper()} | {model_mode} | 기준: {metric_name}")
            print("-" * 80)
            print("feature_set:", best["feature_set"])
            print("regressor:", best["regressor"])
            print("feature_count:", int(best["feature_count"]))
            print("price_mae:", best["price_mae"])
            print("baseline_price_mae:", best["baseline_price_mae"])
            print("normal_mae:", best["normal_mae"])
            print("shock_mae:", best["shock_mae"])
            print("baseline_shock_mae:", best["baseline_shock_mae"])
            print("max_abs_error:", best["max_abs_error"])
            print("return_r2:", best["return_r2"])
            print("model_path:", best["model_path"])

    print("\n최종 후보 테스트 완료")


if __name__ == "__main__":
    main()
