# model/select_features.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from config import (
    CURRENT_PRICE_COLUMNS,
    DATASET_PATHS,
    EXPERIMENT_DIR,
    FEATURE_SCORE_PATH,
    PRIMARY_OIL_TYPE,
    SELECTED_FEATURES_PATH,
    SELECTED_EXTRA_TOP_K_LIST,
    TARGET_COL,
    TRAIN_RATIO,
    ensure_directories,
)

# ==============================
# Utility
# ==============================


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()

    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - series.mean()) / std


def remove_leak_features(features: list[str]) -> list[str]:
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
        if any(keyword in col for keyword in leak_keywords):
            continue

        if col in ["date"]:
            continue

        result.append(col)

    return result


def unique_list(items: list[str]) -> list[str]:
    result = []

    for item in items:
        if item not in result:
            result.append(item)

    return result


def numeric_existing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)

    result = [col for col in features if col in numeric_cols]

    result = remove_leak_features(result)
    result = unique_list(result)

    return result


# ==============================
# Feature group rules
# ==============================


def is_price_momentum_feature(col: str, current_col: str) -> bool:
    if not col.startswith(current_col):
        return False

    keywords = [
        "_lag1",
        "_lag2",
        "_lag3",
        "_diff1",
        "_rolling_3_mean",
        "_diff5",
        "_diff10",
        "_diff20",
        "_return5",
        "_return10",
        "_return20",
        "_log_return5",
        "_log_return10",
        "_log_return20",
    ]

    return any(keyword in col for keyword in keywords) or col == current_col


def is_gpr_feature(col: str) -> bool:
    return col.startswith("GPRD")


def is_price_cross_feature(col: str) -> bool:
    patterns = [
        "current_Brent",
        "current_WTI",
        "Brent_minus_Dubai",
        "WTI_minus_Dubai",
        "Brent_minus_WTI",
        "Dubai_to_Brent_ratio",
        "Dubai_to_WTI_ratio",
        "Brent_to_WTI_ratio",
    ]

    return any(pattern in col for pattern in patterns)


def is_market_feature(col: str) -> bool:
    patterns = [
        "DXY",
        "VIX",
        "US10Y",
        "crude_inventory",
    ]

    return any(col == pattern or col.startswith(f"{pattern}_") for pattern in patterns)


def is_gdelt_feature(col: str) -> bool:
    return col.startswith("gdelt_")


def is_acled_feature(col: str) -> bool:
    patterns = [
        "MiddleEast_",
        "NorthAmerica_",
        "LatinAmerica_",
        "Russia_",
        "Other_",
        "global_total_events",
        "global_total_fatalities",
    ]

    return any(col.startswith(pattern) for pattern in patterns)


def get_base_features(df: pd.DataFrame, oil_type: str) -> list[str]:
    current_col = CURRENT_PRICE_COLUMNS[oil_type]

    features = []

    for col in df.columns:
        if is_price_momentum_feature(col, current_col):
            features.append(col)

        if is_gpr_feature(col):
            features.append(col)

    return numeric_existing_features(df, features)


def get_extra_candidate_features(df: pd.DataFrame, oil_type: str) -> list[str]:
    base_features = set(get_base_features(df, oil_type))

    candidates = []

    for col in df.columns:
        if col in base_features:
            continue

        if is_market_feature(col):
            candidates.append(col)
        elif is_gdelt_feature(col):
            candidates.append(col)
        elif is_acled_feature(col):
            candidates.append(col)
        elif is_price_cross_feature(col):
            candidates.append(col)

    return numeric_existing_features(df, candidates)


# ==============================
# Scoring
# ==============================


def prepare_xy(df: pd.DataFrame, features: list[str]):
    data = df[features + [TARGET_COL]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    X = data[features]
    # [수정] 0 대신 각 컬럼의 중앙값(median)으로 결측치를 채워 변수 랭킹 왜곡 방지
    X = X.fillna(X.median()).fillna(0) # 만약 컬럼 전체가 NaN이라 median이 없으면 0으로 최종 방어
    y = data[TARGET_COL]

    return X, y


def score_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        raise ValueError("점수화할 feature가 없습니다.")

    X, y = prepare_xy(df, features)

    # 1. abs corr
    abs_corr_values = []

    for col in features:
        corr = X[col].corr(y)

        if pd.isna(corr):
            corr = 0

        abs_corr_values.append(abs(corr))

    abs_corr = pd.Series(abs_corr_values, index=features)

    # 2. mutual information
    mutual_info_values = mutual_info_regression(
        X,
        y,
        random_state=42,
    )

    mutual_info = pd.Series(mutual_info_values, index=features)

    # 3. lasso coefficient
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(
        alpha=0.01,
        max_iter=50000,
        random_state=42,
    )
    lasso.fit(X_scaled, y)

    abs_lasso_coef = pd.Series(np.abs(lasso.coef_), index=features)

    # 4. random forest importance
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    rf_importance = pd.Series(rf.feature_importances_, index=features)

    score_df = pd.DataFrame(
        {
            "feature": features,
            "abs_corr": abs_corr.values,
            "mutual_info": mutual_info.values,
            "abs_lasso_coef": abs_lasso_coef.values,
            "rf_importance": rf_importance.values,
        }
    )

    score_df["total_score"] = (
        zscore(score_df["abs_corr"])
        + zscore(score_df["mutual_info"])
        + zscore(score_df["abs_lasso_coef"])
        + zscore(score_df["rf_importance"])
    )

    score_df = score_df.sort_values(
        "total_score",
        ascending=False,
    ).reset_index(drop=True)

    return score_df


# ==============================
# Main
# ==============================

def main():
    ensure_directories()

    oil_type = PRIMARY_OIL_TYPE
    dataset_path = DATASET_PATHS[oil_type]

    print("=" * 80)
    print("feature 선택 시작 (Data Leakage 방지 버전)")
    print("=" * 80)
    print("oil_type:", oil_type)
    print("dataset:", dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset이 없습니다: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.sort_values("date").reset_index(drop=True)

    # ----------------------------------------------------------------
    # [핵심 개선] 오직 Train 데이터 구간(70%)만 떼어내서 후속 연산 진행
    # ----------------------------------------------------------------
    train_size = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:train_size].copy()

    print("전체 데이터 크기:", df.shape)
    print(f"변수 선택에 사용할 Train 데이터 크기 (상위 {TRAIN_RATIO*100}%):", train_df.shape)
    print("Train 날짜 범위:", train_df["date"].min(), "~", train_df["date"].max())

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {TARGET_COL}")

    # 이후 함수들에는 전체 'df' 대신 'train_df'를 집어넣음
    base_features = get_base_features(train_df, oil_type)
    extra_candidates = get_extra_candidate_features(train_df, oil_type)

    print("\nfeature 구성:")
    print("base feature 개수:", len(base_features))
    print("extra 후보 개수:", len(extra_candidates))

    # 스코어링 함수에도 train_df를 투입
    score_df = score_features(train_df, extra_candidates)

    # (이하 json 저장 및 상위 30개 print 로직은 기존 코드와 동일하게 유지)
    score_df.to_csv(FEATURE_SCORE_PATH, index=False, encoding="utf-8-sig")
    
    selected = {
        "oil_type": oil_type,
        "base_feature_set": "price_momentum_gpr",
        "base_features": base_features,
        "extra_candidates_count": len(extra_candidates),
        "selected_extra": {},
        "feature_sets": {},
    }
    selected["feature_sets"]["price_momentum_gpr"] = base_features

    for k in SELECTED_EXTRA_TOP_K_LIST:
        top_features = score_df.head(k)["feature"].tolist()
        selected["selected_extra"][f"top{k}"] = top_features
        selected["feature_sets"][f"price_momentum_gpr_selected_extra_{k}"] = (
            base_features + top_features
        )

    with open(SELECTED_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("extra feature 상위 30개 (Train 데이터 기준)")
    print("=" * 80)
    print(score_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
