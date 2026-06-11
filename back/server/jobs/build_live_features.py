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
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

ALL_OIL_DATASET_PATH = PROCESSED_DIR / "all_oil_dataset.csv"
ORIGINAL_DUBAI_DATASET_PATH = PROCESSED_DIR / "dubai_dataset.csv"
ORIGINAL_WTI_DATASET_PATH = PROCESSED_DIR / "wti_dataset.csv"
ORIGINAL_BRENT_DATASET_PATH = PROCESSED_DIR / "brent_dataset.csv"

LIVE_RAW_PATH = PREDICTION_DIR / "live_raw.csv"
RAW_HISTORY_CSV = RAW_DIR / "live_raw_history.csv"

LIVE_PROCESSED_CSV_PATH = PREDICTION_DIR / "live_processed_features.csv"
LATEST_FEATURE_DEFAULTS_PATH = PREDICTION_DIR / "latest_feature_defaults.json"

UPDATED_ALL_OIL_DATASET_PATH = PROCESSED_DIR / "all_oil_dataset.csv"
UPDATED_DUBAI_DATASET_PATH = PROCESSED_DIR / "dubai_dataset.csv"
UPDATED_WTI_DATASET_PATH = PROCESSED_DIR / "wti_dataset.csv"
UPDATED_BRENT_DATASET_PATH = PROCESSED_DIR / "brent_dataset.csv"

HISTORY_TAIL_ROWS = 500

LIVE_METADATA_COLS = {
    "date",
    "collected_at",
    "market_reference_date",
    "brent_reference_date",
    "wti_reference_date",
    "vix_reference_date",
    "us10y_reference_date",
    "dxy_reference_date",
    "gpr_reference_date",
    "acled_reference_date",
    "gdelt_reference_date",
}

PROTECTED_PREFIXES = ("future_", "target_date_")


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


def find_target_date_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("target_date_") or col == "target_date"
    ]


def sort_and_deduplicate_by_date(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_date_column(df)
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)
    return df


def remove_rows_after_last_known_target(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_date_column(df)
    target_date_cols = find_target_date_columns(df)

    if not target_date_cols:
        return df

    parsed_targets = []

    for col in target_date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parsed_targets.append(parsed)

    if not parsed_targets:
        return df

    all_target_dates = pd.concat(parsed_targets, ignore_index=True).dropna()

    if all_target_dates.empty:
        return df

    last_known_target_date = all_target_dates.max()

    if pd.isna(last_known_target_date):
        return df

    trimmed = df[df["date"] <= last_known_target_date].copy()

    if trimmed.empty:
        return df

    removed_count = len(df) - len(trimmed)

    if removed_count > 0:
        print(
            "[INFO] 기존 live patch 누적 row 제거:",
            removed_count,
            "rows after",
            last_known_target_date.date(),
        )

    return trimmed.reset_index(drop=True)


def drop_future_and_target_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = []

    for col in df.columns:
        if col in {"target", "target_date", "target_shock"}:
            drop_cols.append(col)
            continue

        if any(col.startswith(prefix) for prefix in PROTECTED_PREFIXES):
            drop_cols.append(col)
            continue

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df, drop_cols


def align_to_original_dataset_columns(
    updated_all: pd.DataFrame,
    original_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    if not original_path.exists():
        print(
            f"[WARN] {dataset_name} 원본 파일이 없어 updated_all 전체를 저장합니다: {original_path}"
        )
        return updated_all.copy()

    original = read_csv_required(original_path, dataset_name)
    original_columns = list(original.columns)

    aligned = updated_all.copy()

    for col in original_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan

    extra_columns = [col for col in aligned.columns if col not in original_columns]
    aligned = aligned[original_columns + extra_columns].copy()

    aligned, removed_future_cols = drop_future_and_target_columns(aligned)

    if "date" in aligned.columns:
        aligned = sort_and_deduplicate_by_date(aligned)

    print(
        f"[INFO] {dataset_name} 컬럼 정렬 완료:",
        f"original_cols={len(original_columns)}, saved_cols={len(aligned.columns)}",
    )
    print(
        f"[INFO] {dataset_name} 미래/타겟 관련 제거 컬럼 수:",
        len(removed_future_cols),
    )

    if removed_future_cols:
        print(
            f"[INFO] {dataset_name} 미래/타겟 관련 제거 컬럼 예시:",
            removed_future_cols[:20],
        )

    return aligned


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


def load_live_raw_history() -> pd.DataFrame:
    if RAW_HISTORY_CSV.exists():
        df = pd.read_csv(RAW_HISTORY_CSV)
        print(f"[INFO] live raw history 사용: {RAW_HISTORY_CSV}")
    else:
        df = read_csv_required(LIVE_RAW_PATH, "live_raw")
        print(f"[WARN] live raw history가 없어 live_raw 1행만 사용: {LIVE_RAW_PATH}")

    if df.empty:
        raise ValueError("live raw history가 비어 있습니다.")

    df = sort_and_deduplicate_by_date(df)

    return df


# =========================
# Live row construction
# =========================
def build_live_base_row(
    history_tail: pd.DataFrame,
    live_raw: dict[str, Any],
) -> pd.DataFrame:
    if history_tail.empty:
        raise ValueError("history_tail이 비어 있습니다.")

    live_row = history_tail.iloc[-1].copy()

    live_date = pd.to_datetime(live_raw.get("date"), errors="coerce")

    if pd.isna(live_date):
        raise ValueError(f"live_raw date 파싱 실패: {live_raw.get('date')}")

    live_row["date"] = live_date

    direct_cols = [
        col
        for col in live_raw.keys()
        if col in history_tail.columns
        and col not in LIVE_METADATA_COLS
        and not col.startswith(PROTECTED_PREFIXES)
    ]

    for col in direct_cols:
        value = pd.to_numeric(live_raw[col], errors="coerce")

        if pd.notna(value):
            live_row[col] = float(value)

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

    live_df = pd.DataFrame([live_row], columns=history_tail.columns)

    return live_df


def build_live_base_rows(
    history_tail: pd.DataFrame,
    live_raw_history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    working_tail = history_tail.copy().reset_index(drop=True)

    for _, raw_row in live_raw_history.iterrows():
        raw_dict = raw_row.to_dict()
        base_df = build_live_base_row(working_tail, raw_dict)

        rows.append(base_df.iloc[0])

        working_tail = pd.concat(
            [working_tail, base_df],
            ignore_index=True,
        )
        working_tail = sort_and_deduplicate_by_date(working_tail)
        working_tail = working_tail.tail(HISTORY_TAIL_ROWS).reset_index(drop=True)

    if not rows:
        raise ValueError("생성된 live base row가 없습니다.")

    return pd.DataFrame(rows, columns=history_tail.columns)


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


def recompute_gpr_special_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"GPRD", "GPRD_MA30"}.issubset(df.columns):
        if "GPRD_spike_MA30" in df.columns:
            df["GPRD_spike_MA30"] = (df["GPRD"] > df["GPRD_MA30"]).astype(int)

        if "GPRD_spike_ratio_MA30" in df.columns:
            df["GPRD_spike_ratio_MA30"] = safe_divide(df["GPRD"], df["GPRD_MA30"])

    if {"GPRD_MA7", "GPRD_MA30"}.issubset(df.columns):
        if "GPRD_MA7_MA30_gap" in df.columns:
            df["GPRD_MA7_MA30_gap"] = df["GPRD_MA7"] - df["GPRD_MA30"]

        if "GPRD_MA7_MA30_ratio" in df.columns:
            df["GPRD_MA7_MA30_ratio"] = safe_divide(df["GPRD_MA7"], df["GPRD_MA30"])

    if {"GPRD_THREAT", "GPRD_ACT"}.issubset(df.columns):
        if "GPRD_THREAT_ACT_gap" in df.columns:
            df["GPRD_THREAT_ACT_gap"] = df["GPRD_THREAT"] - df["GPRD_ACT"]

        if "GPRD_THREAT_ACT_ratio" in df.columns:
            df["GPRD_THREAT_ACT_ratio"] = safe_divide(
                df["GPRD_THREAT"],
                df["GPRD_ACT"],
            )

    return df.replace([np.inf, -np.inf], np.nan)


def recompute_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns = list(df.columns)

    for col in columns:
        if col == "date":
            continue

        if col.startswith(PROTECTED_PREFIXES):
            continue

        match = re.match(r"(.+)_lag(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = df[base_col].shift(periods)

            continue

        match = re.match(r"(.+)_diff(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = df[base_col] - df[base_col].shift(periods)

            continue

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

        match = re.match(r"(.+)_log_return(\d+)$", col)
        if match:
            base_col = match.group(1)
            periods = int(match.group(2))

            if base_col in df.columns:
                df[col] = safe_log_return(df[base_col], periods)

            continue

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


def stabilize_and_recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    combined = df.copy()

    base_fill_cols = [
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
    ]

    dynamic_fill_prefixes = [
        "gdelt_",
        "Europe_",
        "LatinAmerica_",
        "MiddleEast_",
        "NorthAmerica_",
        "Russia_Eurasia_",
        "Global_",
    ]

    fill_before_recompute_cols = [
        col
        for col in combined.columns
        if not col.startswith(PROTECTED_PREFIXES)
        and (
            col in base_fill_cols
            or any(col.startswith(prefix) for prefix in dynamic_fill_prefixes)
        )
    ]

    if fill_before_recompute_cols:
        combined[fill_before_recompute_cols] = combined[
            fill_before_recompute_cols
        ].ffill()
        combined[fill_before_recompute_cols] = combined[
            fill_before_recompute_cols
        ].bfill()

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
    combined = recompute_gpr_special_features(combined)

    fillable_cols = [
        col
        for col in combined.columns
        if col != "date" and not col.startswith(PROTECTED_PREFIXES)
    ]

    if fillable_cols:
        combined[fillable_cols] = combined[fillable_cols].ffill()
        combined[fillable_cols] = combined[fillable_cols].bfill()
        combined[fillable_cols] = combined[fillable_cols].fillna(0)

    return combined


def build_live_processed_features() -> pd.DataFrame:
    history_tail = load_history_tail()
    live_raw = load_live_raw_row()

    live_base_df = build_live_base_row(history_tail, live_raw)

    combined = pd.concat(
        [history_tail, live_base_df],
        ignore_index=True,
    )

    combined = sort_and_deduplicate_by_date(combined)
    combined = stabilize_and_recompute_features(combined)

    live_date = pd.to_datetime(live_raw["date"], errors="coerce")
    live_processed = combined[combined["date"] == live_date].copy()

    if live_processed.empty:
        raise RuntimeError("live_processed 행을 찾지 못했습니다.")

    live_processed = live_processed.tail(1).reset_index(drop=True)

    return live_processed


def build_updated_all_oil_dataset() -> pd.DataFrame:
    history = read_csv_required(ALL_OIL_DATASET_PATH, "all_oil_dataset")
    history = clean_date_column(history)
    history = remove_rows_after_last_known_target(history)
    history = sort_and_deduplicate_by_date(history)

    live_raw_history = load_live_raw_history()

    history_dates = set(history["date"].dt.strftime("%Y-%m-%d"))
    live_raw_history = live_raw_history.copy()
    live_raw_history["date_str"] = live_raw_history["date"].dt.strftime("%Y-%m-%d")

    live_rows_to_apply = live_raw_history.copy()

    if live_rows_to_apply.empty:
        raise ValueError("반영할 live raw row가 없습니다.")

    print(f"[INFO] live raw rows to apply: {len(live_rows_to_apply)}")
    print(
        "[INFO] live raw latest date:",
        live_rows_to_apply["date"].max().date(),
    )

    history_tail = history.tail(HISTORY_TAIL_ROWS).copy().reset_index(drop=True)
    live_base_rows = build_live_base_rows(
        history_tail,
        live_rows_to_apply.drop(columns=["date_str"], errors="ignore"),
    )

    combined = pd.concat(
        [history, live_base_rows],
        ignore_index=True,
    )

    combined = sort_and_deduplicate_by_date(combined)
    combined = stabilize_and_recompute_features(combined)

    added_dates = [
        date_str
        for date_str in live_raw_history["date_str"].tolist()
        if date_str not in history_dates
    ]

    if added_dates:
        print("[INFO] 새로 추가된 live 거래일:", added_dates[-10:])
    else:
        print("[INFO] 새로 추가된 거래일 없음. 기존 날짜를 최신 live 값으로 덮어씀.")

    return combined


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
        "GPRD",
        "GPRD_MA7",
        "GPRD_MA30",
        "Brent_minus_Dubai",
        "WTI_minus_Dubai",
        "Brent_minus_WTI",
        "gdelt_global_avg_tone",
        "gdelt_MiddleEast_avg_tone",
        "MiddleEast_conflict_events",
        "Global_conflict_events",
    ]

    existing_preview_cols = [
        col for col in preview_cols if col in live_processed.columns
    ]

    print(live_processed[existing_preview_cols].to_string(index=False))

    row = live_processed.iloc[0]
    nan_cols = [col for col in live_processed.columns if pd.isna(row[col])]
    non_future_nan_cols = [
        col for col in nan_cols if not col.startswith(PROTECTED_PREFIXES)
    ]

    print("\n[INFO] NaN column count:", len(nan_cols))
    print("[INFO] non-future NaN column count:", len(non_future_nan_cols))

    if non_future_nan_cols:
        print("[WARN] non-future NaN columns:")
        for col in non_future_nan_cols[:50]:
            print("-", col)


def save_updated_processed_datasets(updated_all: pd.DataFrame):
    updated_all = updated_all.copy()
    updated_all = sort_and_deduplicate_by_date(updated_all)

    updated_all.to_csv(
        UPDATED_ALL_OIL_DATASET_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    updated_dubai = align_to_original_dataset_columns(
        updated_all=updated_all,
        original_path=ORIGINAL_DUBAI_DATASET_PATH,
        dataset_name="dubai_dataset",
    )
    updated_wti = align_to_original_dataset_columns(
        updated_all=updated_all,
        original_path=ORIGINAL_WTI_DATASET_PATH,
        dataset_name="wti_dataset",
    )
    updated_brent = align_to_original_dataset_columns(
        updated_all=updated_all,
        original_path=ORIGINAL_BRENT_DATASET_PATH,
        dataset_name="brent_dataset",
    )

    updated_dubai.to_csv(
        UPDATED_DUBAI_DATASET_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    updated_wti.to_csv(
        UPDATED_WTI_DATASET_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    updated_brent.to_csv(
        UPDATED_BRENT_DATASET_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] updated all_oil_dataset -> {UPDATED_ALL_OIL_DATASET_PATH}")
    print(f"[DONE] updated dubai_dataset   -> {UPDATED_DUBAI_DATASET_PATH}")
    print(f"[DONE] updated wti_dataset     -> {UPDATED_WTI_DATASET_PATH}")
    print(f"[DONE] updated brent_dataset   -> {UPDATED_BRENT_DATASET_PATH}")
    print("[INFO] oil별 dataset은 다른 유종 컬럼을 제거해서 저장했습니다.")


def save_latest_prediction_outputs_from_updated_all(updated_all: pd.DataFrame):
    updated_all = sort_and_deduplicate_by_date(updated_all)

    live_processed = updated_all.tail(1).reset_index(drop=True)

    save_live_processed_outputs(live_processed)


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("daily processed dataset patch 시작")
    print("=" * 80)
    print("base all_oil_dataset:", ALL_OIL_DATASET_PATH)
    print("live raw:", LIVE_RAW_PATH)
    print("live raw history:", RAW_HISTORY_CSV)

    updated_all = build_updated_all_oil_dataset()

    save_updated_processed_datasets(updated_all)
    save_latest_prediction_outputs_from_updated_all(updated_all)

    latest_date = pd.to_datetime(updated_all["date"].max(), errors="coerce")

    print("\n[INFO] updated row count:", len(updated_all))

    if pd.notna(latest_date):
        print("[INFO] latest processed date:", latest_date.date())
    else:
        print("[INFO] latest processed date: unknown")

    print("\ndaily processed dataset patch 완료")


if __name__ == "__main__":
    main()
