# back/server/jobs/fetch_latest_raw_api_data.py

from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# =========================
# Path settings
# =========================
# file path:
# ai_project/back/server/jobs/fetch_latest_raw_api_data.py
#
# parents[0] = jobs
# parents[1] = server
# parents[2] = back
# parents[3] = ai_project
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "prediction"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "live_raw.csv"

ACLED_LOOKBACK_DAYS = 30


# =========================
# Environment variables
# =========================
load_dotenv(PROJECT_ROOT / ".env")

ACLED_EMAIL = os.getenv("ACLED_EMAIL")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD")
EIA_API_KEY = os.getenv("EIA_API_KEY")


# =========================
# ACLED target countries
# =========================
TARGET_OIL_COUNTRIES = [
    "Saudi Arabia",
    "Iran",
    "Iraq",
    "United Arab Emirates",
    "Kuwait",
    "Qatar",
    "Oman",
    "Bahrain",
    "Russia",
    "United States",
    "Norway",
    "Canada",
    "Kazakhstan",
    "Mexico",
    "Brazil",
]

HORMUZ_REGION_COUNTRIES = [
    "Saudi Arabia",
    "Iran",
    "Iraq",
    "United Arab Emirates",
    "Kuwait",
    "Qatar",
    "Oman",
    "Bahrain",
]

OIL_KEYWORDS = [
    "oil",
    "pipeline",
    "refinery",
    "terminal",
    "energy",
    "gas",
    "petroleum",
    "infrastructure",
    "port",
    "export",
    "offshore",
    "facility",
    "storage",
    "tank",
    "shipping",
    "crude",
    "lng",
    "drilling",
    "well",
    "tanker",
]


# =========================
# Common utilities
# =========================
def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if pd.isna(result):
            return default

        return result

    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default

        return int(float(value))

    except Exception:
        return default


# =========================
# Market data
# =========================
def get_latest_close(
    ticker: str,
    label: str,
    lookback_days: int = 10,
) -> float:
    """
    최근 lookback_days 동안의 일봉 중 가장 최근 유효 종가 반환.
    """
    try:
        hist = yf.Ticker(ticker).history(
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
        )

        if hist.empty or "Close" not in hist.columns:
            print(f"[WARN] No market data for {label} ({ticker})")
            return 0.0

        close_series = hist["Close"].dropna()

        if close_series.empty:
            print(f"[WARN] No valid close values for {label} ({ticker})")
            return 0.0

        value = float(close_series.iloc[-1])
        print(f"[INFO] {label}: {value}")

        return value

    except Exception as e:
        print(f"[WARN] Failed to fetch {label} ({ticker}): {e}")
        return 0.0


# =========================
# EIA data
# =========================
def get_latest_eia_crude_inventory(api_key: Optional[str]) -> float:
    """
    EIA API v2에서 최신 crude inventory 값 조회.
    API key가 없거나 실패하면 0.0 반환.
    """
    if not api_key:
        print("[WARN] EIA_API_KEY is not set")
        return 0.0

    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"

    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 10,
    }

    try:
        resp = requests.get(
            url,
            params=params,
            timeout=20,
        )
        resp.raise_for_status()

        payload = resp.json()
        data = payload.get("response", {}).get("data", [])

        if not data:
            print("[WARN] EIA returned empty data")
            return 0.0

        for row in data:
            val = row.get("value")

            if val is not None:
                value = safe_float(val, default=0.0)
                print(f"[INFO] crude_inventory: {value}")
                return value

        print("[WARN] No valid crude_inventory value found in EIA response")
        return 0.0

    except Exception as e:
        print(f"[WARN] Failed to fetch EIA crude inventory: {e}")
        return 0.0


# =========================
# ACLED login/session
# =========================
def acled_login_session() -> Optional[requests.Session]:
    """
    ACLED 웹 로그인 후 세션 반환.
    환경변수 없거나 로그인 실패 시 None 반환.
    """
    if not ACLED_EMAIL or not ACLED_PASSWORD:
        print("[WARN] ACLED_EMAIL or ACLED_PASSWORD is not set")
        return None

    login_url = "https://acleddata.com/user/login?_format=json"
    session = requests.Session()

    try:
        resp = session.post(
            login_url,
            json={
                "name": ACLED_EMAIL,
                "pass": ACLED_PASSWORD,
            },
            timeout=15,
        )

        if resp.status_code != 200:
            print(f"[WARN] ACLED login failed with status={resp.status_code}")
            print(f"[WARN] response text: {resp.text[:300]}")
            return None

        print("[INFO] ACLED login success")
        return session

    except Exception as e:
        print(f"[WARN] ACLED login error: {e}")
        return None


def fetch_acled_events_by_country_and_date(
    session: requests.Session,
    country: str,
    target_date: str,
    limit: int = 5000,
) -> List[dict]:
    """
    특정 국가 + 특정 날짜의 ACLED 이벤트 수집.
    """
    data_url = "https://acleddata.com/api/acled/read"

    all_rows: List[dict] = []
    page = 1

    while True:
        params = {
            "country": country,
            "limit": limit,
            "page": page,
            "event_date": target_date,
            "event_date_where": "=",
        }

        try:
            resp = session.get(
                data_url,
                params=params,
                timeout=20,
            )
            resp.raise_for_status()

            payload = resp.json()

            if not payload.get("success"):
                print(
                    "[WARN] ACLED rejected request: "
                    f"country={country}, page={page}, msg={payload.get('message')}"
                )
                break

            batch = payload.get("data", [])

            if not isinstance(batch, list) or len(batch) == 0:
                break

            all_rows.extend(batch)

            if len(batch) < limit:
                break

            page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"[WARN] ACLED fetch error country={country}, page={page}: {e}")
            break

    return all_rows


def fetch_acled_all_countries_for_date(
    session: requests.Session,
    target_date: str,
) -> List[dict]:
    """
    15개 핵심 산유국의 target_date 이벤트 통합 수집.
    """
    all_rows: List[dict] = []

    for country in TARGET_OIL_COUNTRIES:
        rows = fetch_acled_events_by_country_and_date(
            session=session,
            country=country,
            target_date=target_date,
        )

        all_rows.extend(rows)

        print(f"[INFO] ACLED {country}: {len(rows)} rows on {target_date}")

        time.sleep(0.5)

    return all_rows


def compute_acled_features(rows: List[dict]) -> Dict[str, int]:
    """
    ACLED 원시 이벤트 목록에서 live raw feature 계산.
    """
    if not rows:
        return {
            "hormuz_risk": 0,
            "gulf_supply_disruption": 0,
            "oil_infrastructure_attack": 0,
        }

    hormuz_risk = sum(
        1 for row in rows if row.get("country") in HORMUZ_REGION_COUNTRIES
    )

    gulf_supply_disruption = 0

    for row in rows:
        gulf_supply_disruption += safe_int(
            row.get("fatalities", 0),
            default=0,
        )

    oil_infrastructure_attack = 0

    for row in rows:
        text_parts = [
            str(row.get("event_type", "")),
            str(row.get("sub_event_type", "")),
            str(row.get("notes", "")),
            str(row.get("actor1", "")),
            str(row.get("actor2", "")),
            str(row.get("assoc_actor_1", "")),
            str(row.get("assoc_actor_2", "")),
            str(row.get("location", "")),
            str(row.get("admin1", "")),
        ]

        joined = " ".join(text_parts).lower()

        if any(keyword in joined for keyword in OIL_KEYWORDS):
            oil_infrastructure_attack += 1

    return {
        "hormuz_risk": int(hormuz_risk),
        "gulf_supply_disruption": int(gulf_supply_disruption),
        "oil_infrastructure_attack": int(oil_infrastructure_attack),
    }


def get_latest_acled_features(
    max_lookback_days: int = ACLED_LOOKBACK_DAYS,
) -> Tuple[str, Dict[str, int]]:
    """
    오늘부터 max_lookback_days만큼 뒤로 탐색하면서
    이벤트가 존재하는 가장 최근 날짜의 ACLED feature 반환.
    """
    session = acled_login_session()

    if session is None:
        return "", {
            "hormuz_risk": 0,
            "gulf_supply_disruption": 0,
            "oil_infrastructure_attack": 0,
        }

    today = datetime.now(timezone.utc).date()

    for i in range(max_lookback_days + 1):
        candidate_date = (today - timedelta(days=i)).strftime("%Y-%m-%d")

        print(f"[INFO] Trying ACLED date: {candidate_date}")

        rows = fetch_acled_all_countries_for_date(
            session=session,
            target_date=candidate_date,
        )

        if rows:
            features = compute_acled_features(rows)

            print(f"[INFO] ACLED latest usable date found: {candidate_date}")
            print(f"[INFO] ACLED features: {features}")

            return candidate_date, features

    print("[WARN] No ACLED events found in lookback window")

    return "", {
        "hormuz_risk": 0,
        "gulf_supply_disruption": 0,
        "oil_infrastructure_attack": 0,
    }


# =========================
# Build integrated live raw row
# =========================
def build_live_raw_row() -> pd.DataFrame:
    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reference_date = datetime.now().strftime("%Y-%m-%d")

    # Market data
    current_brent = get_latest_close("BZ=F", "Brent")
    current_wti = get_latest_close("CL=F", "WTI")
    vix = get_latest_close("^VIX", "VIX")
    us10y = get_latest_close("^TNX", "US10Y")
    dxy = get_latest_close("DX-Y.NYB", "DXY")

    # Dubai 실시간 공개 API가 제한적이므로 임시 proxy.
    # build_live_features.py에서 historical Brent-Dubai spread 기반으로 다시 보정 가능.
    current_dubai = current_brent - 0.5 if current_brent > 0 else 0.0

    # EIA
    crude_inventory = get_latest_eia_crude_inventory(EIA_API_KEY)

    # ACLED
    acled_date, acled_features = get_latest_acled_features(
        max_lookback_days=ACLED_LOOKBACK_DAYS,
    )

    hormuz_risk = int(acled_features["hormuz_risk"])
    gulf_supply_disruption = int(acled_features["gulf_supply_disruption"])
    oil_infrastructure_attack = int(acled_features["oil_infrastructure_attack"])

    row = {
        # Common metadata
        "date": reference_date,
        "collected_at": collected_at,
        "market_reference_date": reference_date,
        "acled_reference_date": acled_date if acled_date else "",
        # Price columns - processed/model compatible names
        "current_Dubai": float(current_dubai),
        "current_Brent": float(current_brent),
        "current_WTI": float(current_wti),
        # Market columns - processed/model compatible names
        "DXY": float(dxy),
        "VIX": float(vix),
        "US10Y": float(us10y),
        "crude_inventory": float(crude_inventory),
        # Raw scenario columns
        "hormuz_risk": hormuz_risk,
        "gulf_supply_disruption": gulf_supply_disruption,
        "oil_infrastructure_attack": oil_infrastructure_attack,
        # GDELT-compatible proxy columns
        "gdelt_global_hormuz_risk_count": hormuz_risk,
        "gdelt_MiddleEast_hormuz_risk_count": hormuz_risk,
        "gdelt_global_gulf_supply_disruption_count": gulf_supply_disruption,
        "gdelt_MiddleEast_gulf_supply_disruption_count": gulf_supply_disruption,
        "gdelt_global_oil_infrastructure_attack_count": oil_infrastructure_attack,
        "gdelt_MiddleEast_oil_infrastructure_attack_count": oil_infrastructure_attack,
    }

    return pd.DataFrame([row])


def main():
    df = build_live_raw_row()

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] Saved CSV -> {OUTPUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
