# model/train.py
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

from config import (
    CURRENT_PRICE_COLUMNS, DATASET_PATHS, EXPERIMENT_DIR, PRIMARY_OIL_TYPE,
    SELECTED_FEATURES_PATH, TARGET_COL, TRAIN_RESULTS_JSON_PATH, 
    TRAIN_RESULTS_PATH, TRAIN_RATIO, VAL_RATIO, ensure_directories,
)

def load_selected_features() -> dict:
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_feature_list(df: pd.DataFrame, features: list[str]) -> list[str]:
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)
    leak_keywords = ["future_", "target", "target_date", "answer_", "actual_", "predicted_", "error"]
    return [c for c in features if c in numeric_cols and c != "date" and not any(k in c for k in leak_keywords)]

def make_hybrid_models(pos_weight: float) -> dict:
    return {
        "hybrid_rf_ridge": {
            "classifier": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(n_estimators=300, max_depth=4, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1))
            ]),
            "normal_reg": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1))
            ]),
            "shock_reg": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0))
            ])
        }
    }

def time_series_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def make_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COL].copy()
    y_shock = df['target_shock'].copy() if 'target_shock' in df.columns else np.zeros(len(df))
    return X, y, y_shock

def evaluate_prediction(test_df, test_pred, pred_labels, current_price_col):
    """모든 지표(MAE, 분류성능, 가격오차)를 한 번에 계산해서 반환하도록 통합 수정"""
    y_true = test_df[TARGET_COL]
    
    # 1. 가격 등락률(%) MAE
    mae = float(mean_absolute_error(y_true, test_pred))
    
    # 2. 방향성 적중률
    direction_acc = float(np.mean(np.sign(y_true) == np.sign(test_pred)))
    
    # 3. 쇼크 구간 세부 평가
    shock_mask = (y_true.abs() >= 15.0)
    shock_mae = float(mean_absolute_error(y_true[shock_mask], test_pred[shock_mask])) if shock_mask.any() else float("nan")
    
    # 4. 쇼크 감지 정확도 (분류기 성능)
    y_shock_true = test_df['target_shock']
    precision = float(precision_score(y_shock_true, pred_labels, zero_division=0))
    recall = float(recall_score(y_shock_true, pred_labels, zero_division=0))
    
    # 5. 예측 달러 가격 MAE 계산
    curr = test_df[current_price_col]
    true_price = curr * (1 + y_true / 100)
    pred_price = curr * (1 + test_pred / 100)
    price_mae = float(mean_absolute_error(true_price, pred_price))

    return {
        "target_mae_pct": mae,
        "direction_acc": direction_acc,
        "shock_mae_pct": shock_mae,
        "shock_precision": precision,
        "shock_recall": recall,
        "price_mae_dollar": price_mae
    }

def main():
    ensure_directories()
    oil_type = PRIMARY_OIL_TYPE
    df = pd.read_csv(DATASET_PATHS[oil_type]).sort_values("date").reset_index(drop=True)
    train_df, val_df, test_df = time_series_split(df)
    feature_sets = load_selected_features()["feature_sets"]
    
    results = []

    for f_name, raw_cols in feature_sets.items():
        f_cols = clean_feature_list(df, raw_cols)
        X_train, y_train, y_train_shock = make_xy(train_df, f_cols)
        X_test, y_test, y_test_shock = make_xy(test_df, f_cols)
        
        dynamic_pos_weight = (y_train_shock == 0).sum() / max(1, (y_train_shock == 1).sum())
        hybrid_dict = make_hybrid_models(pos_weight=dynamic_pos_weight)

        for model_name, parts in hybrid_dict.items():
            clf, norm_reg, shock_reg = parts["classifier"], parts["normal_reg"], parts["shock_reg"]
            
            # 1. 분류기 학습
            clf.fit(X_train, y_train_shock)
            
            # 2. 회귀 모델 학습 (정상/쇼크 분할)
            norm_mask = (y_train_shock == 0)
            shock_mask = (y_train_shock == 1)
            
            norm_reg.fit(X_train[norm_mask], y_train[norm_mask])
            if shock_mask.sum() > 0:
                shock_reg.fit(X_train[shock_mask], y_train[shock_mask])
            else:
                shock_reg = norm_reg

            # 3. Test 예측
            CUSTOM_THRESHOLD = 0.10
            shock_probs = clf.predict_proba(X_test)[:, 1]
            pred_shock_labels = (shock_probs >= CUSTOM_THRESHOLD).astype(int) 
            
            test_pred = np.zeros(len(X_test))
            
            mask_norm = (pred_shock_labels == 0)
            mask_shock = (pred_shock_labels == 1)
            
            if mask_norm.any():
                test_pred[mask_norm] = norm_reg.predict(X_test[mask_norm])
            if mask_shock.any():
                test_pred[mask_shock] = shock_reg.predict(X_test[mask_shock])
                
            # 평가 지표 산출
            metrics = evaluate_prediction(test_df, test_pred, pred_shock_labels, CURRENT_PRICE_COLUMNS[oil_type])
            
            row = {"feature_set": f_name, "model": model_name, **metrics}
            results.append(row)
            
            actual_shocks = int(y_test_shock.sum())
            total_alarms = int(pred_shock_labels.sum())
            true_hits = int(((y_test_shock == 1) & (pred_shock_labels == 1)).sum())
            
            print(f"[{f_name}] {model_name:15s}")
            print(f"   -> 🔍 팩트체크: 실제 쇼크 {actual_shocks}일 중 {true_hits}일 적중 (총 경보 횟수: {total_alarms}회)")
            print(f"   -> 📈 오차분석: 등락률 MAE={metrics['target_mae_pct']:.2f}%p (쇼크 시 {metrics['shock_mae_pct']:.2f}%p)")
            print(f"   -> 🎯 분류성능: 정밀도={metrics['shock_precision']:.2f}, 재현율={metrics['shock_recall']:.2f}, 방향적중률={metrics['direction_acc']:.2f}")
            print(f"   -> 💵 가격오차: 예측 달러 MAE=${metrics['price_mae_dollar']:.2f}")
            print("-" * 60)

    pd.DataFrame(results).to_csv(TRAIN_RESULTS_PATH, index=False, encoding="utf-8-sig")
    with open(TRAIN_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("하이브리드 파이프라인 학습 완료 및 저장 성공!")

if __name__ == "__main__":
    main()