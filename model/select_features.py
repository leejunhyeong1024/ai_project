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

def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std): return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

def remove_leak_features(features: list[str]) -> list[str]:
    leak_keywords = ["future_", "target", "target_date", "answer_", "actual_", "predicted_", "error"]
    result = []
    for col in features:
        if any(keyword in col for keyword in leak_keywords) or col == "date": continue
        result.append(col)
    return result

def numeric_existing_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)
    return remove_leak_features(list(set([col for col in features if col in numeric_cols])))

def is_price_momentum_feature(col: str, current_col: str) -> bool:
    if not col.startswith(current_col): return False
    keywords = ["_lag", "_diff", "_rolling_", "_return"]
    return any(keyword in col for keyword in keywords) or col == current_col

def get_base_features(df: pd.DataFrame, oil_type: str) -> list[str]:
    current_col = CURRENT_PRICE_COLUMNS[oil_type]
    features = [col for col in df.columns if is_price_momentum_feature(col, current_col) or col.startswith("GPRD")]
    return numeric_existing_features(df, features)

def get_extra_candidate_features(df: pd.DataFrame, oil_type: str) -> list[str]:
    base = set(get_base_features(df, oil_type))
    cands = [col for col in df.columns if col not in base and any(col.startswith(p) for p in ["DXY", "VIX", "US10Y", "crude", "gdelt_", "ACLED", "current_"])]
    return numeric_existing_features(df, cands)

def prepare_xy(df: pd.DataFrame, features: list[str]):
    data = df[features + [TARGET_COL]].copy().replace([np.inf, -np.inf], np.nan)
    X = data[features].fillna(data[features].median()).fillna(0)
    y = data[TARGET_COL]
    return X, y

def score_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X, y = prepare_xy(df, features)
    
    abs_corr = pd.Series([abs(X[c].corr(y)) if not pd.isna(X[c].corr(y)) else 0 for c in features], index=features)
    mutual_info = pd.Series(mutual_info_regression(X, y, random_state=42), index=features)
    
    X_scaled = StandardScaler().fit_transform(X)
    lasso = Lasso(alpha=0.01, max_iter=50000, random_state=42).fit(X_scaled, y)
    abs_lasso_coef = pd.Series(np.abs(lasso.coef_), index=features)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1).fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=features)
    
    score_df = pd.DataFrame({"feature": features, "abs_corr": abs_corr, "mutual_info": mutual_info, "abs_lasso": abs_lasso_coef, "rf_imp": rf_importance})
    score_df["total_score"] = zscore(score_df["abs_corr"]) + zscore(score_df["mutual_info"]) + zscore(score_df["abs_lasso"]) + zscore(score_df["rf_imp"])
    return score_df.sort_values("total_score", ascending=False).reset_index(drop=True)

def main():
    ensure_directories()
    oil_type = PRIMARY_OIL_TYPE
    df = pd.read_csv(DATASET_PATHS[oil_type]).sort_values("date").reset_index(drop=True)
    
    train_df = df.iloc[:int(len(df) * TRAIN_RATIO)].copy()
    base_features = get_base_features(train_df, oil_type)
    extra_candidates = get_extra_candidate_features(train_df, oil_type)
    
    score_df = score_features(train_df, extra_candidates)
    score_df.to_csv(FEATURE_SCORE_PATH, index=False, encoding="utf-8-sig")
    
    selected = {"feature_sets": {"price_momentum_gpr": base_features}, "selected_extra": {}}
    for k in SELECTED_EXTRA_TOP_K_LIST:
        top = score_df.head(k)["feature"].tolist()
        selected["selected_extra"][f"top{k}"] = top
        selected["feature_sets"][f"price_momentum_gpr_selected_extra_{k}"] = base_features + top

    with open(SELECTED_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=4, ensure_ascii=False)
    print("피처 셀렉션 완료!")

if __name__ == "__main__":
    main()