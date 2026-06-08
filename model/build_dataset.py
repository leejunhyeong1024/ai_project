# model/build_dataset.py

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ALL_OIL_DATASET_PATH,
    CONFLICT_PATH,
    CRUDE_INVENTORY_PATH,
    DATASET_PATHS,
    DXY_PATH,
    FORECAST_HORIZON_TRADING_DAYS,
    GDELT_PATH,
    GPR_PATH,
    OIL_COLUMN_MAP,
    OIL_PRICE_PATH,
    OIL_TYPES,
    ROLLING_WINDOWS,
    TARGET_COL,
    TARGET_DATE_COL,
    US10Y_PATH,
    VIX_PATH,
    NEW_RISK_DATA_PATH,  # config.py에서 추가한 경로
    ensure_directories,
)

# ==============================
# Common utils
# ==============================

def normalize_col_name(col: str) -> str:
    col = str(col).strip()
    col = col.replace("/", "_")
    col = col.replace("-", "_")
    col = col.replace(" ", "_")
    col = re.sub(r"[^0-9a-zA-Z_]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")

def safe_divide(a, b):
    return np.where(b == 0, np.nan, a / b)

def check_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} 파일이 없습니다: {path}")

def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def find_date_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).lower() in ["date", "day", "event_date"]:
            return col
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    return df.columns[0]

def clean_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.floor("D").astype("datetime64[ns]")

def parse_oil_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() > len(series) * 0.8:
        return parsed.dt.floor("D").astype("datetime64[ns]")
    
    def convert_one(value):
        digits = re.sub(r"\D", "", str(value))
        if len(digits) >= 6:
            digits = digits[:6]
            yy = int(digits[:2])
            mm = int(digits[2:4])
            dd = int(digits[4:6])
            year = 2000 + yy if yy < 50 else 1900 + yy
            try:
                return pd.Timestamp(year=year, month=mm, day=dd)
            except ValueError:
                return pd.NaT
        return pd.NaT
    
    result = series.apply(convert_one)
    return pd.to_datetime(result, errors="coerce").dt.floor("D").astype("datetime64[ns]")

def clean_daily_value_df(df: pd.DataFrame, value_col: str, output_col: str, date_col: str | None = None) -> pd.DataFrame:
    if date_col is None:
        date_col = find_date_col(df)
    result = df[[date_col, value_col]].copy()
    result.columns = ["date", output_col]
    result["date"] = clean_date(result["date"])
    result[output_col] = pd.to_numeric(result[output_col], errors="coerce")
    before_rows = len(result)
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    result[output_col] = result[output_col].ffill()
    print(f"[{output_col}] 날짜 파싱 전/후 행 수: {before_rows} -> {len(result)}")
    return result

# ==============================
# Oil price
# ==============================

def load_oil_price() -> pd.DataFrame:
    check_file(OIL_PRICE_PATH, "oil price")
    print("\n" + "=" * 80 + "\n원유 가격 데이터 로드\n" + "=" * 80)
    
    # 안전 파싱 적용
    df = read_csv_auto(OIL_PRICE_PATH)
    df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = df['date'].astype(str).str.strip()
    
    rename_map = {}
    for oil_col in ["Dubai", "Brent", "WTI"]:
        if oil_col in df.columns:
            rename_map[oil_col] = f"current_{oil_col}"
            
    result = df.rename(columns=rename_map)
    result = result[["date"] + list(rename_map.values())].copy()
    result["date"] = parse_oil_date(result["date"])
    
    for col in rename_map.values():
        result[col] = pd.to_numeric(result[col], errors="coerce")
        
    before_rows = len(result)
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    
    for oil_col in ["Dubai", "Brent", "WTI"]:
        current_col = f"current_{oil_col}"
        future_col = f"future_{oil_col}"
        oil_target_date_col = f"target_date_{oil_col}"
        
        valid = result[["date", current_col]].dropna(subset=[current_col]).copy()
        valid = valid.sort_values("date").reset_index(drop=True)
        valid[future_col] = valid[current_col].shift(-FORECAST_HORIZON_TRADING_DAYS)
        valid[oil_target_date_col] = valid["date"].shift(-FORECAST_HORIZON_TRADING_DAYS)
        valid[oil_target_date_col] = clean_date(valid[oil_target_date_col])
        
        result = result.merge(valid[["date", future_col, oil_target_date_col]], on="date", how="left")

    print(f"[원유 가격 정리 완료] 파싱 후 행 수: {len(result)}")
    return result

# ==============================
# GPR & Market & GDELT & ACLED
# ==============================

def load_gpr() -> pd.DataFrame:
    check_file(GPR_PATH, "GPR")
    df = pd.read_excel(GPR_PATH)
    needed = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]
    date_col = "date" if "date" in df.columns else find_date_col(df)
    result = df[[date_col] + needed].rename(columns={date_col: "date"}).copy()
    result["date"] = clean_date(result["date"])
    for col in needed:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for col in needed:
        result[col] = result[col].ffill()
    return result

def add_gpr_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gpr_base_cols = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]
    eps = 1e-9
    new_features = {}
    for col in gpr_base_cols:
        if col not in df.columns: continue
        for lag in [1, 2, 3]: new_features[f"{col}_lag{lag}"] = df[col].shift(lag)
        new_features[f"{col}_diff1"] = df[col].diff(1)
        for window in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_{window}_mean"] = df[col].rolling(window).mean()
            new_features[f"{col}_rolling_{window}_std"] = df[col].rolling(window).std()
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
            new_features[f"{col}_return{window}"] = safe_divide(df[col] - df[col].shift(window), df[col].shift(window))
    
    if "GPRD" in df.columns and "GPRD_MA30" in df.columns:
        new_features["GPRD_spike_MA30"] = df["GPRD"] - df["GPRD_MA30"]
        new_features["GPRD_spike_ratio_MA30"] = df["GPRD"] / (df["GPRD_MA30"] + eps)
    
    feature_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, feature_df], axis=1).replace([np.inf, -np.inf], np.nan)

def load_dxy() -> pd.DataFrame:
    df = read_csv_auto(DXY_PATH)
    val_col = "Close" if "Close" in df.columns else df.select_dtypes(include=["number"]).columns[0]
    return clean_daily_value_df(df, val_col, "DXY")

def load_vix() -> pd.DataFrame:
    df = read_csv_auto(VIX_PATH)
    val_col = "vix_close" if "vix_close" in df.columns else df.select_dtypes(include=["number"]).columns[0]
    return clean_daily_value_df(df, val_col, "VIX")

def load_us10y() -> pd.DataFrame:
    df = read_csv_auto(US10Y_PATH)
    val_col = "tnx_yield" if "tnx_yield" in df.columns else df.select_dtypes(include=["number"]).columns[0]
    return clean_daily_value_df(df, val_col, "US10Y")

def load_inventory() -> pd.DataFrame:
    df = read_csv_auto(CRUDE_INVENTORY_PATH)
    val_col = "crude_inventory" if "crude_inventory" in df.columns else df.select_dtypes(include=["number"]).columns[0]
    return clean_daily_value_df(df, val_col, "crude_inventory")

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_features = {}
    for col in ["DXY", "VIX", "US10Y", "crude_inventory"]:
        if col not in df.columns: continue
        for lag in [1, 2, 3]: new_features[f"{col}_lag{lag}"] = df[col].shift(lag)
        new_features[f"{col}_diff1"] = df[col].diff(1)
        for window in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_{window}_mean"] = df[col].rolling(window).mean()
            new_features[f"{col}_rolling_{window}_std"] = df[col].rolling(window).std()
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
            new_features[f"{col}_return{window}"] = safe_divide(df[col] - df[col].shift(window), df[col].shift(window))
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1).replace([np.inf, -np.inf], np.nan)

def load_gdelt() -> pd.DataFrame:
    if not GDELT_PATH.exists(): return pd.DataFrame(columns=["date"])
    df = read_csv_auto(GDELT_PATH)
    date_col = find_date_col(df)
    result = df.rename(columns={date_col: "date"})
    result.columns = [normalize_col_name(c) for c in result.columns]
    result["date"] = clean_date(result["date"])
    result = result.dropna(subset=["date"])
    
    numeric_cols = [c for c in result.columns if c != "date" and pd.api.types.is_numeric_dtype(result[c])]
    out = result[["date"] + numeric_cols].groupby("date", as_index=False).mean()
    out = out.rename(columns={c: f"gdelt_{c}" for c in numeric_cols if not c.startswith("gdelt_")})
    out["date"] = clean_date(out["date"])
    return out.sort_values("date").reset_index(drop=True).ffill()

def add_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_cols = [col for col in df.columns if col.startswith("gdelt_")]
    new_features = {}
    for col in base_cols:
        for window in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_{window}_mean"] = df[col].rolling(window).mean()
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1).replace([np.inf, -np.inf], np.nan)

def load_conflict() -> pd.DataFrame:
    if not CONFLICT_PATH.exists(): return pd.DataFrame(columns=["date"])
    df = read_csv_auto(CONFLICT_PATH)
    result = df.copy()
    result["date"] = clean_date(result["event_date"])
    result["fatalities"] = pd.to_numeric(result["fatalities"], errors="coerce").fillna(0)
    result["region_group"] = "MiddleEast" # 간략화
    result["event_type_clean"] = result["event_type"].apply(normalize_col_name)
    return result.dropna(subset=["date"]).reset_index(drop=True)

def build_conflict_daily_features(conflict_df: pd.DataFrame) -> pd.DataFrame:
    if conflict_df.empty: return pd.DataFrame(columns=["date"])
    grouped_fatalities = conflict_df.groupby(["date"])["fatalities"].sum().reset_index()
    return grouped_fatalities.sort_values("date").reset_index(drop=True)

# ========================================================
# [핵심 추가] 신규 리스크 데이터셋 (bquxjob_...) 로드
# ========================================================
def load_new_risk_data() -> pd.DataFrame:
    if not NEW_RISK_DATA_PATH.exists():
        print(f"[-] Warning: 신규 리스크 파일이 없습니다 -> {NEW_RISK_DATA_PATH}")
        return pd.DataFrame(columns=["date"])
    
    df = read_csv_auto(NEW_RISK_DATA_PATH)
    date_col = "date" if "date" in df.columns else find_date_col(df)
    df = df.rename(columns={date_col: "date"})
    df["date"] = clean_date(df["date"])
    
    # [여기부터 삽입]
    # 1. 합계 집계
    agg_df = df.groupby("date", as_index=False).agg({
        "hormuz_risk_count": "sum",
        "gulf_supply_disruption_count": "sum",
        "oil_infrastructure_attack_count": "sum",
        "avg_gdelt_tone": "mean"
    })
    
    # 2. 이진 플래그 피처 추가 (사건 발생 여부)
    agg_df["is_hormuz_risk"] = (agg_df["hormuz_risk_count"] > 0).astype(int)
    agg_df["is_gulf_risk"] = (agg_df["gulf_supply_disruption_count"] > 0).astype(int)
    agg_df["is_attack_risk"] = (agg_df["oil_infrastructure_attack_count"] > 0).astype(int)
    agg_df["risk_intensity"] = agg_df["hormuz_risk_count"] * (10 - agg_df["avg_gdelt_tone"].clip(-10, 10))
    # 리스크 발생 건수의 5일 이동평균 (리스크가 지속되는지 판단)
    agg_df["risk_intensity_roll5"] = agg_df["risk_intensity"].rolling(window=5).mean().fillna(0)
    # 전일 대비 리스크 변화량 (리스크가 급증하는지 판단)
    agg_df["risk_change_diff"] = agg_df["risk_intensity"].diff().fillna(0)
    
    return agg_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
# ========================================================

def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not all(c in df.columns for c in ["current_Dubai", "current_Brent", "current_WTI"]): return df
    df["Brent_minus_Dubai"] = df["current_Brent"] - df["current_Dubai"]
    df["WTI_minus_Dubai"] = df["current_WTI"] - df["current_Dubai"]
    return df.replace([np.inf, -np.inf], np.nan)

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_features = {}
    for oil_col in ["Dubai", "Brent", "WTI"]:
        col = f"current_{oil_col}"
        if col not in df.columns: continue
        for window in [5, 10, 20]:
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1).replace([np.inf, -np.inf], np.nan)

def merge_asof_feature(base_df: pd.DataFrame, feature_df: pd.DataFrame, name: str) -> pd.DataFrame:
    if feature_df.empty: return base_df
    base = base_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feat = feature_df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    merged = pd.merge_asof(base, feat, on="date", direction="backward")
    new_cols = [col for col in feat.columns if col != "date"]
    merged[new_cols] = merged[new_cols].ffill()
    return merged

def build_all_oil_dataset() -> pd.DataFrame:
    oil_df = load_oil_price()
    df = oil_df.copy()
    df = merge_asof_feature(df, load_gpr(), "GPR")
    df = merge_asof_feature(df, load_dxy(), "DXY")
    df = merge_asof_feature(df, load_vix(), "VIX")
    df = merge_asof_feature(df, load_us10y(), "US10Y")
    df = merge_asof_feature(df, load_inventory(), "crude_inventory")
    df = merge_asof_feature(df, load_gdelt(), "GDELT")
    df = merge_asof_feature(df, build_conflict_daily_features(load_conflict()), "ACLED")
    
    # [추가] 신규 리스크 데이터 병합
    df = merge_asof_feature(df, load_new_risk_data(), "NEW_RISK")
    
    df = add_gpr_features(df)
    df = add_market_features(df)
    df = add_gdelt_features(df)
    df = add_spread_features(df)
    df = add_price_features(df)
    
    # 매칭되지 않은 신규 데이터 결측치 보정(0 처리)
    fill_cols = ["hormuz_risk_count", "gulf_supply_disruption_count", "oil_infrastructure_attack_count", "avg_gdelt_tone"]
    fill_cols = [c for c in fill_cols if c in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(0)
    
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(ALL_OIL_DATASET_PATH, index=False, encoding="utf-8-sig")
    return df

def make_oil_dataset(all_df: pd.DataFrame, oil_type: str) -> pd.DataFrame:
    oil_name = OIL_COLUMN_MAP[oil_type]
    current_col = f"current_{oil_name}"
    future_col = f"future_{oil_name}"
    oil_target_date_col = f"target_date_{oil_name}"

    df = all_df.copy()
    df[TARGET_DATE_COL] = df[oil_target_date_col]
    
    # [핵심 수정 1] 타깃을 '달러 변화량'에서 '10일 뒤 등락률(%)'로 변경
    df[TARGET_COL] = (df[future_col] - df[current_col]) / df[current_col] * 100

    # [핵심 수정 2] 쇼크 장세 기준 변경
    # 기존 20달러(약 20~25%) 대신, 10거래일 기준 15% 이상 폭등/폭락을 쇼크로 정의
    df['target_shock'] = (df[TARGET_COL].abs().rolling(window=5, min_periods=1).max().shift(-5) >= 15.0).astype(int)

    df = df.dropna(subset=[current_col, future_col, TARGET_COL, TARGET_DATE_COL]).copy()
    cross_price_cols = ["current_Dubai", "current_Brent", "current_WTI"]
    df = df.dropna(subset=cross_price_cols).copy()

    drop_cols = [col for col in df.columns if col.startswith("future_") or col.startswith("target_date_")]
    df = df.drop(columns=drop_cols).sort_values("date").reset_index(drop=True)

    output_path = DATASET_PATHS[oil_type]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[{oil_type.upper()}] 생성 완료 (쇼크 장세: {df['target_shock'].sum()}일)")
    return df

def main():
    ensure_directories()
    all_df = build_all_oil_dataset()
    for oil_type in OIL_TYPES:
        make_oil_dataset(all_df, oil_type)

if __name__ == "__main__":
    main()