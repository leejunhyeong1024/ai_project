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
    return (
        pd.to_datetime(series, errors="coerce").dt.floor("D").astype("datetime64[ns]")
    )


def parse_oil_date(series: pd.Series) -> pd.Series:
    """
    oil_price_daily.csv 날짜 복원용.
    정상 날짜면 그대로 파싱하고, 깨진 날짜면 숫자만 뽑아서 YYMMDD로 복원.
    예: 080102 -> 2008-01-02
    예: 180102 -> 2018-01-02
    """
    parsed = pd.to_datetime(series, errors="coerce")

    if parsed.notna().sum() > len(series) * 0.8:
        return parsed.dt.floor("D").astype("datetime64[ns]")

    def convert_one(value):
        digits = re.sub(r"\D", "", str(value))

        if len(digits) >= 8:
            digits8 = digits[:8]
            yyyy = int(digits8[:4])
            mm = int(digits8[4:6])
            dd = int(digits8[6:8])

            try:
                return pd.Timestamp(year=yyyy, month=mm, day=dd)
            except ValueError:
                pass

        if len(digits) >= 6:
            digits6 = digits[:6]
            yy = int(digits6[:2])
            mm = int(digits6[2:4])
            dd = int(digits6[4:6])

            year = 2000 + yy if yy < 50 else 1900 + yy

            try:
                return pd.Timestamp(year=year, month=mm, day=dd)
            except ValueError:
                return pd.NaT

        return pd.NaT

    result = series.apply(convert_one)
    return (
        pd.to_datetime(result, errors="coerce").dt.floor("D").astype("datetime64[ns]")
    )


def clean_daily_value_df(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    date_col: str | None = None,
) -> pd.DataFrame:
    if date_col is None:
        date_col = find_date_col(df)

    result = df[[date_col, value_col]].copy()
    result.columns = ["date", output_col]

    result["date"] = clean_date(result["date"])
    result[output_col] = pd.to_numeric(result[output_col], errors="coerce")

    before_rows = len(result)

    result = result.dropna(subset=["date"])
    result = result.sort_values("date")
    result = result.drop_duplicates(subset=["date"], keep="last")
    result = result.reset_index(drop=True)

    result[output_col] = result[output_col].ffill()

    print(f"[{output_col}] 날짜 파싱 전/후 행 수: {before_rows} -> {len(result)}")

    return result


# ==============================
# Oil price
# ==============================


def load_oil_price() -> pd.DataFrame:
    check_file(OIL_PRICE_PATH, "oil price")

    df = read_csv_auto(OIL_PRICE_PATH)

    print("\n" + "=" * 80)
    print("원유 가격 데이터 로드")
    print("=" * 80)
    print("경로:", OIL_PRICE_PATH)
    print("원본 크기:", df.shape)
    print("원본 컬럼:", df.columns.tolist())

    date_col = find_date_col(df)

    rename_map = {date_col: "date"}

    for oil_col in ["Dubai", "Brent", "WTI"]:
        if oil_col not in df.columns:
            raise ValueError(f"원유 가격 컬럼이 없습니다: {oil_col}")
        rename_map[oil_col] = f"current_{oil_col}"

    result = df.rename(columns=rename_map)
    result = result[["date", "current_Dubai", "current_Brent", "current_WTI"]].copy()

    result["date"] = parse_oil_date(result["date"])

    for col in ["current_Dubai", "current_Brent", "current_WTI"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    before_rows = len(result)

    result = result.dropna(subset=["date"])
    result = result.sort_values("date")
    result = result.drop_duplicates(subset=["date"], keep="last")
    result = result.reset_index(drop=True)

    print("날짜 파싱 전/후 행 수:", before_rows, "->", len(result))

    # 각 유종별로 자기 가격이 존재하는 거래일 기준 10거래일 뒤 future 생성
    for oil_col in ["Dubai", "Brent", "WTI"]:
        current_col = f"current_{oil_col}"
        future_col = f"future_{oil_col}"
        oil_target_date_col = f"target_date_{oil_col}"

        valid = result[["date", current_col]].dropna(subset=[current_col]).copy()
        valid = valid.sort_values("date").reset_index(drop=True)

        valid[future_col] = valid[current_col].shift(-FORECAST_HORIZON_TRADING_DAYS)
        valid[oil_target_date_col] = valid["date"].shift(-FORECAST_HORIZON_TRADING_DAYS)
        valid[oil_target_date_col] = clean_date(valid[oil_target_date_col])

        result = result.merge(
            valid[["date", future_col, oil_target_date_col]],
            on="date",
            how="left",
        )

        print(
            f"[{oil_col}] 가격 존재 행 수:",
            len(valid),
            "| future 결측:",
            int(valid[future_col].isna().sum()),
        )

    print("\n[원유 가격 정리 완료]")
    print(result.head().to_string(index=False))
    print(result.tail().to_string(index=False))
    print("크기:", result.shape)
    print("날짜 범위:", result["date"].min(), "~", result["date"].max())
    print("결측치 상위:")
    print(result.isna().sum().sort_values(ascending=False).head(20))

    return result


# ==============================
# GPR
# ==============================


def load_gpr() -> pd.DataFrame:
    check_file(GPR_PATH, "GPR")

    df = read_csv_auto(GPR_PATH)

    print("\n" + "=" * 80)
    print("GPR 데이터 로드")
    print("=" * 80)
    print("경로:", GPR_PATH)
    print("원본 크기:", df.shape)
    print("원본 컬럼:", df.columns.tolist())

    date_col = "date" if "date" in df.columns else find_date_col(df)

    if "GPRD" not in df.columns:
        raise ValueError(f"GPRD 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    result = df[[date_col, "GPRD"]].copy()
    result = result.rename(columns={date_col: "date"})

    result["date"] = clean_date(result["date"])
    result["GPRD"] = pd.to_numeric(result["GPRD"], errors="coerce")

    before_rows = len(result)

    result = result.dropna(subset=["date"])
    result = result.sort_values("date")
    result = result.drop_duplicates(subset=["date"], keep="last")
    result = result.reset_index(drop=True)

    result["GPRD"] = result["GPRD"].ffill()

    # gpr_daily.csv가 GPRD만 가진 경우를 기준으로 파생 생성
    if "GPRD_MA7" in df.columns:
        result["GPRD_MA7"] = pd.to_numeric(df["GPRD_MA7"], errors="coerce")
    else:
        result["GPRD_MA7"] = result["GPRD"].rolling(7, min_periods=1).mean()

    if "GPRD_MA30" in df.columns:
        result["GPRD_MA30"] = pd.to_numeric(df["GPRD_MA30"], errors="coerce")
    else:
        result["GPRD_MA30"] = result["GPRD"].rolling(30, min_periods=1).mean()

    if "GPRD_ACT" in df.columns:
        result["GPRD_ACT"] = pd.to_numeric(df["GPRD_ACT"], errors="coerce")
    else:
        result["GPRD_ACT"] = result["GPRD"]

    if "GPRD_THREAT" in df.columns:
        result["GPRD_THREAT"] = pd.to_numeric(df["GPRD_THREAT"], errors="coerce")
    else:
        result["GPRD_THREAT"] = result["GPRD"]

    needed = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]
    result = result[["date"] + needed]

    for col in needed:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        result[col] = result[col].ffill()

    print("\n[GPR 정리 완료]")
    print("날짜 파싱 전/후 행 수:", before_rows, "->", len(result))
    print(result.head().to_string(index=False))
    print(result.tail().to_string(index=False))
    print("크기:", result.shape)
    print("날짜 범위:", result["date"].min(), "~", result["date"].max())
    print("결측치:")
    print(result.isna().sum())

    return result


def add_gpr_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    gpr_base_cols = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30"]
    eps = 1e-9
    new_features = {}

    for col in gpr_base_cols:
        if col not in df.columns:
            continue

        for lag in [1, 2, 3]:
            new_features[f"{col}_lag{lag}"] = df[col].shift(lag)

        new_features[f"{col}_diff1"] = df[col].diff(1)

        for window in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_{window}_mean"] = (
                df[col].rolling(window).mean()
            )
            new_features[f"{col}_rolling_{window}_std"] = df[col].rolling(window).std()
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
            new_features[f"{col}_return{window}"] = safe_divide(
                df[col] - df[col].shift(window),
                df[col].shift(window),
            )

    if "GPRD" in df.columns and "GPRD_MA30" in df.columns:
        new_features["GPRD_spike_MA30"] = df["GPRD"] - df["GPRD_MA30"]
        new_features["GPRD_spike_ratio_MA30"] = df["GPRD"] / (df["GPRD_MA30"] + eps)

    if "GPRD_MA7" in df.columns and "GPRD_MA30" in df.columns:
        new_features["GPRD_MA7_MA30_gap"] = df["GPRD_MA7"] - df["GPRD_MA30"]
        new_features["GPRD_MA7_MA30_ratio"] = df["GPRD_MA7"] / (df["GPRD_MA30"] + eps)

    if "GPRD_ACT" in df.columns and "GPRD_THREAT" in df.columns:
        new_features["GPRD_THREAT_ACT_gap"] = df["GPRD_THREAT"] - df["GPRD_ACT"]
        new_features["GPRD_THREAT_ACT_ratio"] = df["GPRD_THREAT"] / (
            df["GPRD_ACT"] + eps
        )

    feature_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    return df.replace([np.inf, -np.inf], np.nan)


# ==============================
# Market data
# ==============================


def load_dxy() -> pd.DataFrame:
    check_file(DXY_PATH, "DXY")

    df = read_csv_auto(DXY_PATH)
    date_col = find_date_col(df)

    if "Close" in df.columns:
        value_col = "Close"
    elif "Adj Close" in df.columns:
        value_col = "Adj Close"
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("DXY 숫자 컬럼을 찾지 못했습니다.")
        value_col = numeric_cols[0]

    return clean_daily_value_df(df, value_col, "DXY", date_col)


def load_vix() -> pd.DataFrame:
    check_file(VIX_PATH, "VIX")

    df = read_csv_auto(VIX_PATH)
    date_col = find_date_col(df)

    if "vix_close" in df.columns:
        value_col = "vix_close"
    elif "Close" in df.columns:
        value_col = "Close"
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("VIX 숫자 컬럼을 찾지 못했습니다.")
        value_col = numeric_cols[0]

    return clean_daily_value_df(df, value_col, "VIX", date_col)


def load_us10y() -> pd.DataFrame:
    check_file(US10Y_PATH, "US10Y")

    df = read_csv_auto(US10Y_PATH)
    date_col = find_date_col(df)

    if "tnx_yield" in df.columns:
        value_col = "tnx_yield"
    elif "US10Y" in df.columns:
        value_col = "US10Y"
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("US10Y 숫자 컬럼을 찾지 못했습니다.")
        value_col = numeric_cols[0]

    return clean_daily_value_df(df, value_col, "US10Y", date_col)


def load_inventory() -> pd.DataFrame:
    check_file(CRUDE_INVENTORY_PATH, "US crude inventory")

    df = read_csv_auto(CRUDE_INVENTORY_PATH)
    date_col = find_date_col(df)

    if "crude_inventory" in df.columns:
        value_col = "crude_inventory"
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("crude_inventory 숫자 컬럼을 찾지 못했습니다.")
        value_col = numeric_cols[0]

    return clean_daily_value_df(df, value_col, "crude_inventory", date_col)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_cols = ["DXY", "VIX", "US10Y", "crude_inventory"]
    new_features = {}

    for col in base_cols:
        if col not in df.columns:
            continue

        for lag in [1, 2, 3]:
            new_features[f"{col}_lag{lag}"] = df[col].shift(lag)

        new_features[f"{col}_diff1"] = df[col].diff(1)

        for window in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_{window}_mean"] = (
                df[col].rolling(window).mean()
            )
            new_features[f"{col}_rolling_{window}_std"] = df[col].rolling(window).std()
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
            new_features[f"{col}_return{window}"] = safe_divide(
                df[col] - df[col].shift(window),
                df[col].shift(window),
            )

        if 5 in ROLLING_WINDOWS and 20 in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_5_20_gap"] = (
                df[col].rolling(5).mean() - df[col].rolling(20).mean()
            )

    feature_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    return df.replace([np.inf, -np.inf], np.nan)


# ==============================
# GDELT
# ==============================


def load_gdelt() -> pd.DataFrame:
    check_file(GDELT_PATH, "GDELT")

    df = read_csv_auto(GDELT_PATH)

    print("\n" + "=" * 80)
    print("GDELT 데이터 로드")
    print("=" * 80)
    print("경로:", GDELT_PATH)
    print("원본 크기:", df.shape)
    print("원본 컬럼:", df.columns.tolist())

    required = [
        "date",
        "country",
        "hormuz_risk_count",
        "gulf_supply_disruption_count",
        "oil_infrastructure_attack_count",
        "avg_gdelt_tone",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"GDELT 필수 컬럼이 없습니다: {missing}")

    result = df[required].copy()
    result["date"] = clean_date(result["date"])
    result["country"] = result["country"].astype(str).apply(normalize_col_name)

    count_cols = [
        "hormuz_risk_count",
        "gulf_supply_disruption_count",
        "oil_infrastructure_attack_count",
    ]

    for col in count_cols + ["avg_gdelt_tone"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    before_rows = len(result)

    result = result.dropna(subset=["date"])

    daily = result.groupby("date", as_index=False).agg(
        gdelt_hormuz_risk_count=("hormuz_risk_count", "sum"),
        gdelt_gulf_supply_disruption_count=("gulf_supply_disruption_count", "sum"),
        gdelt_oil_infrastructure_attack_count=(
            "oil_infrastructure_attack_count",
            "sum",
        ),
        gdelt_avg_tone=("avg_gdelt_tone", "mean"),
    )

    pivot_count_parts = []

    for value_col in count_cols:
        pivot = result.pivot_table(
            index="date",
            columns="country",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

        pivot.columns = [f"gdelt_{country}_{value_col}" for country in pivot.columns]
        pivot_count_parts.append(pivot)

    country_count = pd.concat(pivot_count_parts, axis=1).reset_index()

    tone_pivot = result.pivot_table(
        index="date",
        columns="country",
        values="avg_gdelt_tone",
        aggfunc="mean",
        fill_value=0,
    )

    tone_pivot.columns = [f"gdelt_{country}_avg_tone" for country in tone_pivot.columns]
    tone_pivot = tone_pivot.reset_index()

    daily = daily.merge(country_count, on="date", how="left")
    daily = daily.merge(tone_pivot, on="date", how="left")

    daily = daily.sort_values("date").reset_index(drop=True)
    daily = daily.replace([np.inf, -np.inf], np.nan)
    daily = daily.fillna(0)

    print("\n[GDELT 정리 완료]")
    print("원본 행 수:", before_rows)
    print("날짜 파싱 후 행 수:", len(result))
    print("크기:", daily.shape)
    print("날짜 범위:", daily["date"].min(), "~", daily["date"].max())
    print(daily.head().to_string(index=False))
    print(daily.tail().to_string(index=False))

    return daily


def add_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_cols = [col for col in df.columns if col.startswith("gdelt_")]
    new_features = {}

    for col in base_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if "_count" in col:
            for window in ROLLING_WINDOWS:
                new_features[f"{col}_rolling_{window}_sum"] = (
                    df[col].rolling(window).sum()
                )
                new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)

            rolling_5 = df[col].rolling(5).mean()
            rolling_20 = df[col].rolling(20).mean()

            new_features[f"{col}_rolling_5_20_gap"] = rolling_5 - rolling_20
            new_features[f"{col}_spike_ratio_5_20"] = rolling_5 / (rolling_20 + 1e-9)

        elif "tone" in col:
            for window in ROLLING_WINDOWS:
                new_features[f"{col}_rolling_{window}_mean"] = (
                    df[col].rolling(window).mean()
                )
                new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)

    feature_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    return df.replace([np.inf, -np.inf], np.nan)


# ==============================
# Conflict / ACLED
# ==============================


REGION_MAP = {
    "Saudi Arabia": "MiddleEast",
    "United Arab Emirates": "MiddleEast",
    "UAE": "MiddleEast",
    "Iran": "MiddleEast",
    "Iraq": "MiddleEast",
    "Yemen": "MiddleEast",
    "Oman": "MiddleEast",
    "Qatar": "MiddleEast",
    "Kuwait": "MiddleEast",
    "Syria": "MiddleEast",
    "Lebanon": "MiddleEast",
    "Israel": "MiddleEast",
    "Palestine": "MiddleEast",
    "United States": "NorthAmerica",
    "Canada": "NorthAmerica",
    "Mexico": "NorthAmerica",
    "Russia": "Russia",
    "Russian Federation": "Russia",
}


def map_region(country: str) -> str:
    country = str(country)

    if country in REGION_MAP:
        return REGION_MAP[country]

    latin_keywords = [
        "Brazil",
        "Argentina",
        "Venezuela",
        "Colombia",
        "Chile",
        "Peru",
        "Ecuador",
        "Bolivia",
        "Paraguay",
        "Uruguay",
    ]

    if any(keyword in country for keyword in latin_keywords):
        return "LatinAmerica"

    return "Other"


def load_conflict() -> pd.DataFrame:
    check_file(CONFLICT_PATH, "conflict")

    df = read_csv_auto(CONFLICT_PATH)

    print("\n" + "=" * 80)
    print("ACLED 분쟁 데이터 로드")
    print("=" * 80)
    print("경로:", CONFLICT_PATH)
    print("원본 크기:", df.shape)
    print("원본 컬럼:", df.columns.tolist())

    required = ["event_date", "country", "event_type", "fatalities"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"ACLED 필수 컬럼이 없습니다: {missing}")

    result = df.copy()

    result["date"] = clean_date(result["event_date"])
    result["fatalities"] = pd.to_numeric(result["fatalities"], errors="coerce").fillna(
        0
    )
    result["region_group"] = result["country"].apply(map_region)
    result["event_type_clean"] = result["event_type"].apply(normalize_col_name)

    result = result.dropna(subset=["date"])
    result = result.reset_index(drop=True)

    print("\n[ACLED 정리 완료]")
    print(result.head().to_string(index=False))
    print("크기:", result.shape)
    print("날짜 범위:", result["date"].min(), "~", result["date"].max())

    return result


def build_conflict_daily_features(conflict_df: pd.DataFrame) -> pd.DataFrame:
    df = conflict_df.copy()

    if df.empty:
        return pd.DataFrame(columns=["date"])

    grouped_count = (
        df.groupby(["date", "region_group", "event_type_clean"])
        .size()
        .reset_index(name="count")
    )

    grouped_fatalities = (
        df.groupby(["date", "region_group", "event_type_clean"])["fatalities"]
        .sum()
        .reset_index(name="fatalities")
    )

    count_pivot = grouped_count.pivot_table(
        index="date",
        columns=["region_group", "event_type_clean"],
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    count_pivot.columns = [
        f"{region}_{event}_count" for region, event in count_pivot.columns
    ]

    fatal_pivot = grouped_fatalities.pivot_table(
        index="date",
        columns=["region_group", "event_type_clean"],
        values="fatalities",
        aggfunc="sum",
        fill_value=0,
    )

    fatal_pivot.columns = [
        f"{region}_{event}_fatalities" for region, event in fatal_pivot.columns
    ]

    daily = pd.concat([count_pivot, fatal_pivot], axis=1).reset_index()

    global_daily = (
        df.groupby("date")
        .agg(
            global_total_events=("event_type", "size"),
            global_total_fatalities=("fatalities", "sum"),
        )
        .reset_index()
    )

    daily = daily.merge(global_daily, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)

    base_cols = [col for col in daily.columns if col != "date"]
    new_features = {}

    for col in base_cols:
        for window in ROLLING_WINDOWS:
            new_features[f"{col}_rolling_{window}_sum"] = (
                daily[col].rolling(window).sum()
            )

    feature_df = pd.DataFrame(new_features, index=daily.index)
    daily = pd.concat([daily, feature_df], axis=1)
    daily = daily.fillna(0)

    print("\n[ACLED daily feature 생성 완료]")
    print("크기:", daily.shape)
    print("날짜 범위:", daily["date"].min(), "~", daily["date"].max())

    return daily


# ==============================
# Spread / price momentum
# ==============================


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1e-9
    new_features = {}

    required = ["current_Dubai", "current_Brent", "current_WTI"]
    if not all(col in df.columns for col in required):
        return df

    new_features["Brent_minus_Dubai"] = df["current_Brent"] - df["current_Dubai"]
    new_features["WTI_minus_Dubai"] = df["current_WTI"] - df["current_Dubai"]
    new_features["Brent_minus_WTI"] = df["current_Brent"] - df["current_WTI"]

    new_features["Dubai_to_Brent_ratio"] = df["current_Dubai"] / (
        df["current_Brent"] + eps
    )
    new_features["Dubai_to_WTI_ratio"] = df["current_Dubai"] / (df["current_WTI"] + eps)
    new_features["Brent_to_WTI_ratio"] = df["current_Brent"] / (df["current_WTI"] + eps)

    spread_base = pd.DataFrame(new_features, index=df.index)

    spread_features = {}

    for col in spread_base.columns:
        for lag in [1, 2, 3]:
            spread_features[f"{col}_lag{lag}"] = spread_base[col].shift(lag)

        spread_features[f"{col}_diff1"] = spread_base[col].diff(1)

        for window in ROLLING_WINDOWS:
            spread_features[f"{col}_rolling_{window}_mean"] = (
                spread_base[col].rolling(window).mean()
            )
            spread_features[f"{col}_rolling_{window}_std"] = (
                spread_base[col].rolling(window).std()
            )
            spread_features[f"{col}_diff{window}"] = spread_base[col] - spread_base[
                col
            ].shift(window)

    df = pd.concat(
        [df, spread_base, pd.DataFrame(spread_features, index=df.index)],
        axis=1,
    )

    return df.replace([np.inf, -np.inf], np.nan)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    new_features = {}

    for oil_col in ["Dubai", "Brent", "WTI"]:
        col = f"current_{oil_col}"

        if col not in df.columns:
            continue

        for lag in [1, 2, 3]:
            new_features[f"{col}_lag{lag}"] = df[col].shift(lag)

        new_features[f"{col}_diff1"] = df[col].diff(1)
        new_features[f"{col}_rolling_3_mean"] = df[col].rolling(3).mean()

        for window in [5, 10, 20]:
            new_features[f"{col}_diff{window}"] = df[col] - df[col].shift(window)
            new_features[f"{col}_return{window}"] = safe_divide(
                df[col] - df[col].shift(window),
                df[col].shift(window),
            )
            new_features[f"{col}_log_return{window}"] = np.log(
                safe_divide(df[col], df[col].shift(window))
            )

    feature_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    return df.replace([np.inf, -np.inf], np.nan)


# ==============================
# Merge
# ==============================


def merge_asof_feature(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    if feature_df.empty:
        print(f"[병합 스킵] {name}: 빈 데이터")
        return base_df

    base = base_df.copy()
    feat = feature_df.copy()

    base["date"] = clean_date(base["date"])
    feat["date"] = clean_date(feat["date"])

    base = base.dropna(subset=["date"])
    feat = feat.dropna(subset=["date"])

    base = base.sort_values("date").reset_index(drop=True)
    feat = feat.sort_values("date").reset_index(drop=True)

    feat = feat.drop_duplicates(subset=["date"], keep="last")

    merged = pd.merge_asof(
        base,
        feat,
        on="date",
        direction="backward",
    )

    new_cols = [col for col in feat.columns if col != "date"]

    if new_cols:
        merged[new_cols] = merged[new_cols].ffill()

    print(f"[병합 완료] {name}: +{len(new_cols)} cols -> {merged.shape}")

    return merged


# ==============================
# Build dataset
# ==============================


def build_all_oil_dataset() -> pd.DataFrame:
    oil_df = load_oil_price()

    gpr_df = load_gpr()
    dxy_df = load_dxy()
    vix_df = load_vix()
    us10y_df = load_us10y()
    inventory_df = load_inventory()
    gdelt_df = load_gdelt()

    conflict_df = load_conflict()
    conflict_daily = build_conflict_daily_features(conflict_df)

    df = oil_df.copy()

    df = merge_asof_feature(df, gpr_df, "GPR")
    df = merge_asof_feature(df, dxy_df, "DXY")
    df = merge_asof_feature(df, vix_df, "VIX")
    df = merge_asof_feature(df, us10y_df, "US10Y")
    df = merge_asof_feature(df, inventory_df, "crude_inventory")
    df = merge_asof_feature(df, gdelt_df, "GDELT")
    df = merge_asof_feature(df, conflict_daily, "ACLED")

    df = add_gpr_features(df)
    df = add_market_features(df)
    df = add_gdelt_features(df)
    df = add_spread_features(df)
    df = add_price_features(df)

    df = df.sort_values("date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    df.to_csv(ALL_OIL_DATASET_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("통합 데이터 저장 완료")
    print("=" * 80)
    print("경로:", ALL_OIL_DATASET_PATH)
    print("크기:", df.shape)
    print("날짜 범위:", df["date"].min(), "~", df["date"].max())
    print("결측치 상위:")
    print(df.isna().sum().sort_values(ascending=False).head(30))

    return df


def make_oil_dataset(all_df: pd.DataFrame, oil_type: str) -> pd.DataFrame:
    oil_name = OIL_COLUMN_MAP[oil_type]

    current_col = f"current_{oil_name}"
    future_col = f"future_{oil_name}"
    oil_target_date_col = f"target_date_{oil_name}"

    required_cols = [current_col, future_col, oil_target_date_col]

    missing = [col for col in required_cols if col not in all_df.columns]
    if missing:
        raise ValueError(f"{oil_type}용 필수 컬럼이 없습니다: {missing}")

    df = all_df.copy()

    df[TARGET_DATE_COL] = df[oil_target_date_col]

    # target = 10거래일 뒤 수익률(%)
    df[TARGET_COL] = (df[future_col] - df[current_col]) / df[current_col] * 100

    # shock = 10거래일 수익률 절대값 10% 이상
    df["target_shock"] = (df[TARGET_COL].abs() >= 10.0).astype(int)

    before_rows = len(df)

    df = df.dropna(
        subset=[
            current_col,
            future_col,
            TARGET_COL,
            TARGET_DATE_COL,
        ]
    ).copy()

    after_target_rows = len(df)

    cross_price_cols = [
        "current_Dubai",
        "current_Brent",
        "current_WTI",
    ]

    before_cross_rows = len(df)
    df = df.dropna(subset=cross_price_cols).copy()
    after_cross_rows = len(df)

    drop_cols = [
        col
        for col in df.columns
        if col.startswith("future_") or col.startswith("target_date_")
    ]

    df = df.drop(columns=drop_cols)
    df = df.sort_values("date").reset_index(drop=True)

    output_path = DATASET_PATHS[oil_type]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"{oil_type.upper()} 데이터셋 저장 완료")
    print("=" * 80)
    print("경로:", output_path)
    print("크기:", df.shape)
    print("target:", f"({future_col} - {current_col}) / {current_col} * 100")
    print("target_date:", oil_target_date_col, "->", TARGET_DATE_COL)
    print("전체 행 수:", before_rows)
    print("target 필수값 제거 전/후:", before_rows, "->", after_target_rows)
    print(
        "cross-oil 현재 가격 결측 제거 전/후:",
        before_cross_rows,
        "->",
        after_cross_rows,
    )
    print("날짜 범위:", df["date"].min(), "~", df["date"].max())
    print("target 결측치:", int(df[TARGET_COL].isna().sum()))
    print("target_shock 기준: abs(target) >= 10.0")
    print("target_shock 개수:", int(df["target_shock"].sum()))
    print("target 통계:")
    print(df[TARGET_COL].describe())

    return df


def main():
    ensure_directories()

    print("=" * 80)
    print("build_dataset 시작")
    print("=" * 80)

    all_df = build_all_oil_dataset()

    for oil_type in OIL_TYPES:
        make_oil_dataset(all_df, oil_type)

    print("\n전체 데이터셋 생성 완료")


if __name__ == "__main__":
    main()
