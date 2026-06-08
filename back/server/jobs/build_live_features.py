# back/server/jobs/build_live_features.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =========================
# Path settings
# =========================
# file path:
# ai_project/back/server/jobs/build_live_features.py
#
# parents[0] = jobs
# parents[1] = server
# parents[2] = back
# parents[3] = ai_project
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

ALL_OIL_DATASET_PATH = PROCESSED_DIR / "all_oil_dataset.csv"
LIVE_RAW_PATH = PREDICTION_DIR / "live_raw.csv"

LIVE_PROCESSED_CSV_PATH = PREDICTION_DIR / "live_processed_features.csv"
LATEST_FEATURE_DEFAULTS_PATH = PREDICTION_DIR / "latest_feature_defaults.json"

# Dubai 가격과 spread 계열은 일부 기간에서 결측이 길게 이어질 수 있으므로
# live lag/rolling 계산 안정성을 위해 최근 60행보다 넉넉하게 가져온다.
HISTORY_TAIL_ROWS = 500


# =========================
# Utility
# =========================


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where(b == 0, np.nan, a / b)


def safe_log_return(series: pd.Series, periods: int) -> pd.Series:
    prev = series.shift(periods)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(series / prev)

    return pd.Series(result, index=series.index).replace([np.inf, -np.inf], np.nan)


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} 파일이 없습니다: {path}")

    return pd.read_csv(path)


def to_json_safe_value(value: Any):
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())

    return value


def clean_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("date 컬럼이 없습니다.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# Load data
# =========================


def load_history_tail() -> pd.DataFrame:
    df = read_csv_required(ALL_OIL_DATASET_PATH, "all_oil_dataset")
    df = clean_date_column(df)

    if len(df) < HISTORY_TAIL_ROWS:
        print(f"[WARN] history row가 {HISTORY_TAIL_ROWS}개보다 적습니다: {len(df)}")

    return df.tail(HISTORY_TAIL_ROWS).copy().reset_index(drop=True)


def load_live_raw_row() -> dict[str, Any]:
    df = read_csv_required(LIVE_RAW_PATH, "live_raw")

    if df.empty:
        raise ValueError(f"live_raw 파일이 비어 있습니다: {LIVE_RAW_PATH}")

    row = df.iloc[-1].to_dict()

    if "date" not in row:
        raise ValueError("live_raw에 date 컬럼이 없습니다.")

    return row


# =========================
# Live row construction
# =========================


def build_live_base_row(
    history_tail: pd.DataFrame, live_raw: dict[str, Any]
) -> pd.DataFrame:
    """
    all_oil_dataset의 마지막 행을 복사한 뒤,
    live_raw에서 가져온 최신 base 값만 덮어쓴다.

    이렇게 하면 GPR, GDELT tone, ACLED region feature처럼
    live API에서 못 가져온 값은 기존 마지막 값이 유지된다.
    """
    if history_tail.empty:
        raise ValueError("history_tail이 비어 있습니다.")

    live_row = history_tail.iloc[-1].copy()

    # date / metadata
    live_date = pd.to_datetime(live_raw.get("date"), errors="coerce")

    if pd.isna(live_date):
        raise ValueError(f"live_raw date 파싱 실패: {live_raw.get('date')}")

    live_row["date"] = live_date

    # Live raw에서 직접 덮어쓸 base 컬럼
    direct_cols = [
        "current_Dubai",
        "current_Brent",
        "current_WTI",
        "DXY",
        "VIX",
        "US10Y",
        "crude_inventory",
        "gdelt_global_hormuz_risk_count",
        "gdelt_MiddleEast_hormuz_risk_count",
        "gdelt_global_gulf_supply_disruption_count",
        "gdelt_MiddleEast_gulf_supply_disruption_count",
        "gdelt_global_oil_infrastructure_attack_count",
        "gdelt_MiddleEast_oil_infrastructure_attack_count",
    ]

    for col in direct_cols:
        if col in history_tail.columns and col in live_raw:
            value = pd.to_numeric(live_raw[col], errors="coerce")

            if pd.notna(value):
                live_row[col] = float(value)

    # Dubai proxy 보정:
    # fetch 단계에서는 Brent - 0.5로 임시 생성했지만,
    # 여기서는 최근 Brent-Dubai spread 평균으로 다시 보정한다.
    if (
        "current_Dubai" in history_tail.columns
        and "current_Brent" in history_tail.columns
        and "current_Brent" in live_raw
    ):
        brent_value = pd.to_numeric(live_raw.get("current_Brent"), errors="coerce")

        if pd.notna(brent_value):
            if "Brent_minus_Dubai" in history_tail.columns:
                spread = pd.to_numeric(
                    history_tail["Brent_minus_Dubai"],
                    errors="coerce",
                ).dropna()

                if not spread.empty:
                    recent_spread = float(spread.tail(20).mean())
                    live_row["current_Dubai"] = float(brent_value) - recent_spread

    # live_raw의 raw scenario 컬럼도 all_oil_dataset에 있으면 보존
    optional_raw_cols = [
        "hormuz_risk",
        "gulf_supply_disruption",
        "oil_infrastructure_attack",
    ]

    for col in optional_raw_cols:
        if col in history_tail.columns and col in live_raw:
            value = pd.to_numeric(live_raw[col], errors="coerce")

            if pd.notna(value):
                live_row[col] = float(value)

    live_df = pd.DataFrame([live_row], columns=history_tail.columns)

    return live_df


# =========================
# Feature recalculation
# =========================


def recompute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"current_Brent", "current_Dubai"}.issubset(df.columns):
        if "Brent_minus_Dubai" in df.columns:
            df["Brent_minus_Dubai"] = df["current_Brent"] - df["current_Dubai"]

        if "Dubai_to_Brent_ratio" in df.columns:
            df["Dubai_to_Brent_ratio"] = safe_divide(
                df["current_Dubai"],
                df["current_Brent"],
            )

    if {"current_WTI", "current_Dubai"}.issubset(df.columns):
        if "WTI_minus_Dubai" in df.columns:
            df["WTI_minus_Dubai"] = df["current_WTI"] - df["current_Dubai"]

        if "Dubai_to_WTI_ratio" in df.columns:
            df["Dubai_to_WTI_ratio"] = safe_divide(
                df["current_Dubai"],
                df["current_WTI"],
            )

    if {"current_Brent", "current_WTI"}.issubset(df.columns):
        if "Brent_minus_WTI" in df.columns:
            df["Brent_minus_WTI"] = df["current_Brent"] - df["current_WTI"]

        if "Brent_to_WTI_ratio" in df.columns:
            df["Brent_to_WTI_ratio"] = safe_divide(
                df["current_Brent"],
                df["current_WTI"],
            )

    return df.replace([np.inf, -np.inf], np.nan)


def recompute_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    all_oil_dataset에 이미 존재하는 feature 컬럼명을 기준으로
    lag / diff / return / log_return / rolling feature를 재계산한다.

    새 feature를 임의로 추가하지 않고,
    기존 all_oil_dataset 컬럼만 갱신한다.
    """
    df = df.copy()

    columns = list(df.columns)

    for col in columns:
        if col == "date":
            continue

        # lagN
        match = re.match(r"(.+)_lag(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = df[base_col].shift(periods)

            continue

        # diffN
        match = re.match(r"(.+)_diff(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = df[base_col] - df[base_col].shift(periods)

            continue

        # returnN
        match = re.match(r"(.+)_return(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = safe_divide(
                    df[base_col] - df[base_col].shift(periods),
                    df[base_col].shift(periods),
                )

            continue

        # log_returnN
        match = re.match(r"(.+)_log_return(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = safe_log_return(df[base_col], periods)

            continue

        # rolling_N_mean / rolling_N_std / rolling_N_sum
        match = re.match(r"(.+)_rolling_(\d+)_(mean|std|sum)$", col)
        if match:
            base_col = match.group(1)
            window = int(match.group(2))
            agg = match.group(3)

            if base_col in df.columns:
                if agg == "mean":
                    df[col] = df[base_col].rolling(window).mean()
                elif agg == "std":
                    df[col] = df[base_col].rolling(window).std()
                elif agg == "sum":
                    df[col] = df[base_col].rolling(window).sum()

            continue

    return df.replace([np.inf, -np.inf], np.nan)


def build_live_processed_features() -> pd.DataFrame:
    history_tail = load_history_tail()
    live_raw = load_live_raw_row()

    live_base_df = build_live_base_row(history_tail, live_raw)

    combined = pd.concat(
        [history_tail, live_base_df],
        ignore_index=True,
    )

    combined = clean_date_column(combined)

    # 같은 날짜가 이미 있으면 live row를 마지막 값으로 유지
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)

    # live feature 계산 안정화:
    # 가격 / market / GPR / GDELT base 값은 lag, rolling 계산 전에 ffill한다.
    # 과거 값만 이용하는 forward-fill이므로 미래 정보 누수는 아니다.
    fill_before_recompute_cols = [
        "current_Dubai",
        "current_Brent",
        "current_WTI",
        "DXY",
        "VIX",
        "US10Y",
        "crude_inventory",
        "GPRD",
        "GPRD_ACT",
        "GPRD_THREAT",
        "GPRD_MA7",
        "GPRD_MA30",
        "gdelt_global_hormuz_risk_count",
        "gdelt_MiddleEast_hormuz_risk_count",
        "gdelt_global_gulf_supply_disruption_count",
        "gdelt_MiddleEast_gulf_supply_disruption_count",
        "gdelt_global_oil_infrastructure_attack_count",
        "gdelt_MiddleEast_oil_infrastructure_attack_count",
    ]

    existing_fill_cols = [
        col for col in fill_before_recompute_cols if col in combined.columns
    ]

    if existing_fill_cols:
        combined[existing_fill_cols] = combined[existing_fill_cols].ffill()

        # history_tail 안에 특정 base 컬럼의 유효값이 전혀 없을 경우를 대비한다.
        # live row의 값으로 과거 결측을 채워 rolling/lag 계산이 NaN으로 무너지는 것을 방지한다.
        # 이 fallback은 live 서비스 안정성을 위한 보정이며, 학습 데이터 생성에는 사용하지 않는다.
        combined[existing_fill_cols] = combined[existing_fill_cols].bfill()

    combined = recompute_spread_features(combined)

    spread_cols = [
        "Brent_minus_Dubai",
        "WTI_minus_Dubai",
        "Brent_minus_WTI",
        "Dubai_to_Brent_ratio",
        "Dubai_to_WTI_ratio",
        "Brent_to_WTI_ratio",
    ]

    existing_spread_cols = [col for col in spread_cols if col in combined.columns]

    if existing_spread_cols:
        combined[existing_spread_cols] = combined[existing_spread_cols].ffill()
        combined[existing_spread_cols] = combined[existing_spread_cols].bfill()

    combined = recompute_pattern_features(combined)

    # 최종 live row에 들어가는 feature 중 target/future 계열이 아닌 값은
    # 서버 예측 안정성을 위해 남은 결측을 한 번 더 보정한다.
    protected_prefixes = ("future_", "target_date_")
    fillable_cols = [
        col
        for col in combined.columns
        if col != "date" and not col.startswith(protected_prefixes)
    ]

    if fillable_cols:
        combined[fillable_cols] = combined[fillable_cols].ffill()
        combined[fillable_cols] = combined[fillable_cols].bfill()
        combined[fillable_cols] = combined[fillable_cols].fillna(0)

    live_date = pd.to_datetime(live_raw["date"], errors="coerce")
    live_processed = combined[combined["date"] == live_date].copy()

    if live_processed.empty:
        raise RuntimeError("live_processed 행을 찾지 못했습니다.")

    # 하나만 유지
    live_processed = live_processed.tail(1).reset_index(drop=True)

    return live_processed


# =========================
# Save
# =========================


def save_live_processed_outputs(live_processed: pd.DataFrame):
    live_processed.to_csv(
        LIVE_PROCESSED_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    row = live_processed.iloc[0].to_dict()

    # 모델 입력용 defaults:
    # date도 같이 남겨두되, 서버에서 모델 feature만 골라 쓰면 된다.
    feature_defaults = {key: to_json_safe_value(value) for key, value in row.items()}

    with open(LATEST_FEATURE_DEFAULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            feature_defaults,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"[DONE] live processed CSV -> {LIVE_PROCESSED_CSV_PATH}")
    print(f"[DONE] latest feature defaults -> {LATEST_FEATURE_DEFAULTS_PATH}")

    print("\n[INFO] live processed row preview:")
    preview_cols = [
        "date",
        "current_Dubai",
        "current_Brent",
        "current_WTI",
        "DXY",
        "VIX",
        "US10Y",
        "crude_inventory",
        "Brent_minus_Dubai",
        "WTI_minus_Dubai",
        "Brent_minus_WTI",
    ]

    existing_preview_cols = [
        col for col in preview_cols if col in live_processed.columns
    ]

    print(live_processed[existing_preview_cols].to_string(index=False))

    row = live_processed.iloc[0]
    nan_cols = [col for col in live_processed.columns if pd.isna(row[col])]
    non_future_nan_cols = [
        col
        for col in nan_cols
        if not col.startswith("future_") and not col.startswith("target_date_")
    ]

    print("\n[INFO] NaN column count:", len(nan_cols))
    print("[INFO] non-future NaN column count:", len(non_future_nan_cols))

    if non_future_nan_cols:
        print("[WARN] non-future NaN columns:")
        for col in non_future_nan_cols[:50]:
            print("-", col)


# =========================
# Main
# =========================


def main():
    print("=" * 80)
    print("live feature 생성 시작")
    print("=" * 80)
    print("history:", ALL_OIL_DATASET_PATH)
    print("live raw:", LIVE_RAW_PATH)

    live_processed = build_live_processed_features()

    save_live_processed_outputs(live_processed)

    print("\nlive feature 생성 완료")


if __name__ == "__main__":
    main()
