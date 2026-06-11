# back/server/jobs/fetch_latest_raw_api_data.py

from __future__ import annotations

import os
import re
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# =========================
# Path settings
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PREDICTION_DIR / "live_raw.csv"
RAW_HISTORY_CSV = RAW_DIR / "live_raw_history.csv"

ACLED_LOOKBACK_DAYS = 3
GDELT_LOOKBACK_DAYS = 3
GDELT_MAX_RECORDS = 250

GPR_PAGE_URLS = [
    "https://www.matteoiacoviello.com/gpr.htm",
    "https://www.policyuncertainty.com/gpr.html",
]

GPR_DIRECT_CANDIDATE_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xlsx",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily.xlsx",
]


# =========================
# Environment variables
# =========================
load_dotenv(PROJECT_ROOT / ".env")

ACLED_EMAIL = os.getenv("ACLED_EMAIL")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD")
EIA_API_KEY = os.getenv("EIA_API_KEY")


# =========================
# Region settings
# =========================
REGION_COUNTRIES = {
    "Europe": [
        "Norway",
        "United Kingdom",
        "Germany",
        "France",
        "Italy",
        "Spain",
        "Netherlands",
        "Poland",
        "Ukraine",
    ],
    "LatinAmerica": [
        "Brazil",
        "Mexico",
        "Venezuela",
        "Colombia",
        "Argentina",
        "Ecuador",
    ],
    "MiddleEast": [
        "Saudi Arabia",
        "Iran",
        "Iraq",
        "United Arab Emirates",
        "Kuwait",
        "Qatar",
        "Oman",
        "Bahrain",
        "Yemen",
        "Syria",
        "Israel",
        "Lebanon",
        "Jordan",
    ],
    "NorthAmerica": [
        "United States",
        "Canada",
    ],
    "Russia_Eurasia": [
        "Russia",
        "Kazakhstan",
        "Azerbaijan",
        "Turkmenistan",
        "Georgia",
        "Armenia",
    ],
}

ALL_ACLED_COUNTRIES = sorted(
    {country for countries in REGION_COUNTRIES.values() for country in countries}
)

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

ACLED_EVENT_TYPE_TO_COLUMN = {
    "Battles": "battles_count",
    "Explosions/Remote violence": "explosions_count",
    "Violence against civilians": "violence_civilians_count",
    "Protests": "protests_count",
    "Riots": "riots_count",
}

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

GDELT_REGIONS = {
    "global": "",
    "Europe": "(norway OR united kingdom OR germany OR france OR italy OR spain OR netherlands OR poland OR ukraine)",
    "LatinAmerica": "(brazil OR mexico OR venezuela OR colombia OR argentina OR ecuador)",
    "MiddleEast": "(saudi arabia OR iran OR iraq OR united arab emirates OR kuwait OR qatar OR oman OR bahrain OR yemen OR syria OR israel OR lebanon OR jordan)",
    "NorthAmerica": "(united states OR canada)",
    "Russia_Eurasia": "(russia OR kazakhstan OR azerbaijan OR turkmenistan OR georgia OR armenia)",
}

GDELT_RISK_QUERIES = {
    "hormuz_risk": '(hormuz OR "strait of hormuz")',
    "gulf_supply_disruption": '("oil supply disruption" OR "gulf supply disruption" OR "shipping disruption" OR "tanker disruption")',
    "oil_infrastructure_attack": '("oil infrastructure attack" OR "pipeline attack" OR "refinery attack" OR "oil terminal attack" OR "energy facility attack")',
}


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


def safe_date_string(value) -> str:
    try:
        if value is None or pd.isna(value):
            return ""

        dt = pd.to_datetime(value, errors="coerce")

        if pd.isna(dt):
            return ""

        return dt.strftime("%Y-%m-%d")

    except Exception:
        return ""


def load_latest_history_fallback() -> dict:
    """
    live_raw_history.csv의 마지막 행을 fallback 값으로 사용한다.
    최근 API 조회가 실패하거나 데이터가 없을 때 사용한다.
    """
    if not RAW_HISTORY_CSV.exists():
        print("[WARN] live_raw_history.csv가 없습니다. fallback 사용 불가.")
        return {}

    try:
        history = pd.read_csv(RAW_HISTORY_CSV)

        if history.empty:
            return {}

        if "date" in history.columns:
            history["date"] = pd.to_datetime(history["date"], errors="coerce")
            history = history.dropna(subset=["date"])
            history = history.sort_values("date")

        if history.empty:
            return {}

        fallback = history.iloc[-1].to_dict()
        print(
            f"[INFO] fallback row loaded from live_raw_history: {fallback.get('date')}"
        )

        return fallback

    except Exception as e:
        print(f"[WARN] failed to load raw history fallback: {e}")
        return {}


# =========================
# Market data
# =========================
def get_latest_close_with_date(
    ticker: str,
    label: str,
    lookback_days: int = 10,
) -> tuple[float, str]:
    """
    yfinance에서 최신 유효 종가와 실제 마지막 거래일 날짜를 함께 반환한다.

    중요:
    오늘이 주말/휴일이면 yfinance는 마지막 거래일 가격을 반환한다.
    따라서 live_raw의 date에는 오늘 날짜가 아니라 여기서 얻은 실제 거래일을 써야 한다.
    """
    try:
        hist = yf.Ticker(ticker).history(
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
        )

        if hist.empty or "Close" not in hist.columns:
            print(f"[WARN] No market data for {label} ({ticker})")
            return 0.0, ""

        valid = hist.dropna(subset=["Close"]).copy()

        if valid.empty:
            print(f"[WARN] No valid close values for {label} ({ticker})")
            return 0.0, ""

        latest_index = valid.index[-1]
        latest_date = pd.to_datetime(latest_index).strftime("%Y-%m-%d")
        latest_close = float(valid["Close"].iloc[-1])

        print(f"[INFO] {label}: {latest_close} on {latest_date}")

        return latest_close, latest_date

    except Exception as e:
        print(f"[WARN] Failed to fetch {label} ({ticker}): {e}")
        return 0.0, ""


def choose_market_reference_date(
    brent_date: str,
    wti_date: str,
    fallback: dict,
) -> str:
    """
    live_raw의 date로 사용할 실제 유가 거래일을 결정한다.

    우선순위:
    1. Brent/WTI 중 가장 최근 실제 거래일
    2. fallback의 market_reference_date
    3. fallback의 date
    4. 오늘 날짜
    """
    date_candidates = []

    for value in [brent_date, wti_date]:
        date_str = safe_date_string(value)

        if date_str:
            date_candidates.append(pd.to_datetime(date_str))

    if date_candidates:
        selected = max(date_candidates)
        return selected.strftime("%Y-%m-%d")

    fallback_market_date = safe_date_string(fallback.get("market_reference_date"))

    if fallback_market_date:
        return fallback_market_date

    fallback_date = safe_date_string(fallback.get("date"))

    if fallback_date:
        return fallback_date

    return datetime.now().strftime("%Y-%m-%d")


# =========================
# EIA data
# =========================
def get_latest_eia_crude_inventory(api_key: Optional[str]) -> float:
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
# GPR
# =========================
def discover_gpr_data_urls() -> list[str]:
    urls = list(GPR_DIRECT_CANDIDATE_URLS)

    for page_url in GPR_PAGE_URLS:
        try:
            resp = requests.get(page_url, timeout=20)
            resp.raise_for_status()

            html = resp.text

            links = re.findall(
                r'href=["\']([^"\']+\.(?:xls|xlsx|csv))["\']',
                html,
                flags=re.IGNORECASE,
            )

            for link in links:
                full_url = urljoin(page_url, link)
                lower_url = full_url.lower()

                if "gpr" in lower_url and full_url not in urls:
                    urls.append(full_url)

        except Exception as e:
            print(f"[WARN] GPR page scan failed: {page_url}, {e}")

    return urls


def read_gpr_file_from_url(url: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        content = resp.content

        if len(content) < 1000:
            print(f"[WARN] GPR file too small: {url}")
            return None

        lower_url = url.lower()

        if lower_url.endswith(".csv"):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))

        if df.empty:
            return None

        print(f"[INFO] GPR file loaded: {url}")
        print(f"[INFO] GPR columns: {list(df.columns)[:20]}")

        return df

    except Exception as e:
        print(f"[WARN] GPR file load failed: {url}, {e}")
        return None


def normalize_gpr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    lower_map = {str(col).strip().lower(): col for col in df.columns}

    date_col = None

    for candidate in ["date", "day", "daily", "observation_date"]:
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break

    if date_col is not None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif {"year", "month", "day"}.issubset(lower_map.keys()):
        year_col = lower_map["year"]
        month_col = lower_map["month"]
        day_col = lower_map["day"]

        df["date"] = pd.to_datetime(
            dict(
                year=pd.to_numeric(df[year_col], errors="coerce"),
                month=pd.to_numeric(df[month_col], errors="coerce"),
                day=pd.to_numeric(df[day_col], errors="coerce"),
            ),
            errors="coerce",
        )
    else:
        raise ValueError("GPR 데이터에서 날짜 컬럼을 찾지 못했습니다.")

    col_aliases = {
        "GPRD": ["GPRD", "GPR", "gprd", "gpr"],
        "GPRD_ACT": [
            "GPRD_ACT",
            "GPR_ACT",
            "GPRD_ACTS",
            "GPR_ACTS",
            "gprd_act",
            "gpr_act",
        ],
        "GPRD_THREAT": [
            "GPRD_THREAT",
            "GPR_THREAT",
            "GPRD_THREATS",
            "GPR_THREATS",
            "gprd_threat",
            "gpr_threat",
        ],
    }

    normalized = pd.DataFrame()
    normalized["date"] = df["date"]

    for target_col, aliases in col_aliases.items():
        found_col = None

        for alias in aliases:
            if alias in df.columns:
                found_col = alias
                break

        if found_col is None:
            alias_lower = [alias.lower() for alias in aliases]

            for col in df.columns:
                if str(col).strip().lower() in alias_lower:
                    found_col = col
                    break

        if found_col is not None:
            normalized[target_col] = pd.to_numeric(df[found_col], errors="coerce")
        else:
            normalized[target_col] = np.nan

    normalized = normalized.dropna(subset=["date"])
    normalized = normalized.sort_values("date").reset_index(drop=True)

    normalized[["GPRD", "GPRD_ACT", "GPRD_THREAT"]] = normalized[
        ["GPRD", "GPRD_ACT", "GPRD_THREAT"]
    ].ffill()

    normalized["GPRD_MA7"] = normalized["GPRD"].rolling(7, min_periods=1).mean()
    normalized["GPRD_MA30"] = normalized["GPRD"].rolling(30, min_periods=1).mean()

    return normalized


def get_latest_gpr_features_from_download() -> tuple[str, dict[str, float]]:
    urls = discover_gpr_data_urls()

    for url in urls:
        raw_df = read_gpr_file_from_url(url)

        if raw_df is None:
            continue

        try:
            gpr_df = normalize_gpr_columns(raw_df)
            gpr_df = gpr_df.dropna(subset=["GPRD"])

            if gpr_df.empty:
                print(f"[WARN] GPRD column empty in file: {url}")
                continue

            latest_row = gpr_df.iloc[-1]
            latest_date = latest_row["date"]

            features = {
                "GPRD": safe_float(latest_row.get("GPRD"), default=0.0),
                "GPRD_ACT": safe_float(latest_row.get("GPRD_ACT"), default=0.0),
                "GPRD_THREAT": safe_float(latest_row.get("GPRD_THREAT"), default=0.0),
                "GPRD_MA7": safe_float(latest_row.get("GPRD_MA7"), default=0.0),
                "GPRD_MA30": safe_float(latest_row.get("GPRD_MA30"), default=0.0),
            }

            print(f"[INFO] GPR latest usable date found: {latest_date.date()}")
            print(f"[INFO] GPR features: {features}")

            return str(latest_date.date()), features

        except Exception as e:
            print(f"[WARN] GPR normalize failed for {url}: {e}")
            continue

    return "", {}


def fill_gpr_features_from_fallback(fallback: dict) -> dict[str, float]:
    return {
        "GPRD": safe_float(fallback.get("GPRD"), default=0.0),
        "GPRD_ACT": safe_float(fallback.get("GPRD_ACT"), default=0.0),
        "GPRD_THREAT": safe_float(fallback.get("GPRD_THREAT"), default=0.0),
        "GPRD_MA7": safe_float(fallback.get("GPRD_MA7"), default=0.0),
        "GPRD_MA30": safe_float(fallback.get("GPRD_MA30"), default=0.0),
    }


def get_latest_gpr_features(fallback: dict) -> tuple[str, dict[str, float]]:
    gpr_date, gpr_features = get_latest_gpr_features_from_download()

    if gpr_date:
        return gpr_date, gpr_features

    print("[WARN] GPR download failed. Using live_raw_history fallback.")

    fallback_date = safe_date_string(fallback.get("gpr_reference_date"))
    return fallback_date, fill_gpr_features_from_fallback(fallback)


# =========================
# ACLED
# =========================
def acled_login_session() -> Optional[requests.Session]:
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
    all_rows: List[dict] = []

    for country in ALL_ACLED_COUNTRIES:
        rows = fetch_acled_events_by_country_and_date(
            session=session,
            country=country,
            target_date=target_date,
        )

        all_rows.extend(rows)

        print(f"[INFO] ACLED {country}: {len(rows)} rows on {target_date}")

        time.sleep(0.35)

    return all_rows


def compute_acled_features(rows: List[dict]) -> Dict[str, int]:
    features: Dict[str, int] = {}

    for region in REGION_COUNTRIES.keys():
        features[f"{region}_conflict_events"] = 0
        features[f"{region}_conflict_fatalities"] = 0

        for suffix in ACLED_EVENT_TYPE_TO_COLUMN.values():
            features[f"{region}_{suffix}"] = 0

    features["Global_conflict_events"] = 0
    features["Global_conflict_fatalities"] = 0

    for suffix in ACLED_EVENT_TYPE_TO_COLUMN.values():
        features[f"Global_{suffix}"] = 0

    features["hormuz_risk"] = 0
    features["gulf_supply_disruption"] = 0
    features["oil_infrastructure_attack"] = 0

    if not rows:
        return features

    for row in rows:
        country = row.get("country")
        event_type = row.get("event_type", "")
        fatalities = safe_int(row.get("fatalities", 0), default=0)

        matched_regions = [
            region
            for region, countries in REGION_COUNTRIES.items()
            if country in countries
        ]

        features["Global_conflict_events"] += 1
        features["Global_conflict_fatalities"] += fatalities

        if event_type in ACLED_EVENT_TYPE_TO_COLUMN:
            suffix = ACLED_EVENT_TYPE_TO_COLUMN[event_type]
            features[f"Global_{suffix}"] += 1

        for region in matched_regions:
            features[f"{region}_conflict_events"] += 1
            features[f"{region}_conflict_fatalities"] += fatalities

            if event_type in ACLED_EVENT_TYPE_TO_COLUMN:
                suffix = ACLED_EVENT_TYPE_TO_COLUMN[event_type]
                features[f"{region}_{suffix}"] += 1

        if country in HORMUZ_REGION_COUNTRIES:
            features["hormuz_risk"] += 1

        features["gulf_supply_disruption"] += fatalities

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
            features["oil_infrastructure_attack"] += 1

    return {key: int(value) for key, value in features.items()}


def empty_acled_features() -> Dict[str, int]:
    return compute_acled_features([])


def fill_acled_features_from_fallback(fallback: dict) -> Dict[str, int]:
    base = empty_acled_features()

    for key in base.keys():
        if key in fallback and pd.notna(fallback[key]):
            base[key] = safe_int(fallback[key], default=0)

    return base


def get_latest_acled_features(
    max_lookback_days: int = ACLED_LOOKBACK_DAYS,
) -> Tuple[str, Dict[str, int]]:
    session = acled_login_session()

    if session is None:
        return "", empty_acled_features()

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
            print(f"[INFO] ACLED feature count: {len(features)}")

            return candidate_date, features

    print(f"[WARN] No ACLED events found in last {max_lookback_days} days")

    return "", empty_acled_features()


# =========================
# GDELT
# =========================
def gdelt_datetime_window_for_date(target_date: str) -> tuple[str, str]:
    date_obj = pd.to_datetime(target_date).date()

    start = datetime(
        year=date_obj.year,
        month=date_obj.month,
        day=date_obj.day,
        hour=0,
        minute=0,
        second=0,
    )

    end = start + timedelta(days=1)

    return (
        start.strftime("%Y%m%d%H%M%S"),
        end.strftime("%Y%m%d%H%M%S"),
    )


def fetch_gdelt_articles(
    query: str,
    target_date: str,
    max_records: int = GDELT_MAX_RECORDS,
) -> list[dict]:
    startdatetime, enddatetime = gdelt_datetime_window_for_date(target_date)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
        "startdatetime": startdatetime,
        "enddatetime": enddatetime,
    }

    try:
        resp = requests.get(
            url,
            params=params,
            timeout=20,
        )

        if resp.status_code != 200:
            print(f"[WARN] GDELT status={resp.status_code}, query={query[:80]}")
            return []

        payload = resp.json()
        articles = payload.get("articles", [])

        if not isinstance(articles, list):
            return []

        return articles

    except Exception as e:
        print(f"[WARN] GDELT fetch failed: {e}, query={query[:80]}")
        return []


def extract_article_tone(article: dict) -> float | None:
    for key in ["tone", "avgTone", "averageTone"]:
        if key in article:
            try:
                value = float(article[key])

                if pd.notna(value):
                    return value
            except Exception:
                pass

    return None


def compute_gdelt_avg_tone(
    region_name: str,
    target_date: str,
) -> float:
    region_query = GDELT_REGIONS.get(region_name, "")
    base_query = "(oil OR crude OR petroleum OR energy OR refinery OR pipeline OR tanker OR opec)"

    if region_query:
        query = f"{base_query} {region_query}"
    else:
        query = base_query

    articles = fetch_gdelt_articles(
        query=query,
        target_date=target_date,
    )

    tones = [
        tone
        for tone in (extract_article_tone(article) for article in articles)
        if tone is not None
    ]

    if not tones:
        return 0.0

    return float(sum(tones) / len(tones))


def compute_gdelt_risk_count(
    risk_key: str,
    region_name: str,
    target_date: str,
) -> int:
    risk_query = GDELT_RISK_QUERIES[risk_key]
    region_query = GDELT_REGIONS.get(region_name, "")

    if region_query:
        query = f"{risk_query} {region_query}"
    else:
        query = risk_query

    articles = fetch_gdelt_articles(
        query=query,
        target_date=target_date,
    )

    return int(len(articles))


def get_gdelt_features_for_date(target_date: str) -> Dict[str, float | int]:
    features: Dict[str, float | int] = {}

    for region in GDELT_REGIONS.keys():
        tone_col = f"gdelt_{region}_avg_tone"
        tone_value = compute_gdelt_avg_tone(
            region_name=region,
            target_date=target_date,
        )

        features[tone_col] = float(tone_value)

        print(f"[INFO] GDELT {tone_col}: {tone_value}")

        time.sleep(0.25)

    for risk_key in GDELT_RISK_QUERIES.keys():
        for region in GDELT_REGIONS.keys():
            col = f"gdelt_{region}_{risk_key}_count"

            count_value = compute_gdelt_risk_count(
                risk_key=risk_key,
                region_name=region,
                target_date=target_date,
            )

            features[col] = int(count_value)

            print(f"[INFO] GDELT {col}: {count_value}")

            time.sleep(0.25)

    return features


def empty_gdelt_features() -> Dict[str, float | int]:
    features: Dict[str, float | int] = {}

    for region in GDELT_REGIONS.keys():
        features[f"gdelt_{region}_avg_tone"] = 0.0

    for risk_key in GDELT_RISK_QUERIES.keys():
        for region in GDELT_REGIONS.keys():
            features[f"gdelt_{region}_{risk_key}_count"] = 0

    return features


def fill_gdelt_features_from_fallback(fallback: dict) -> Dict[str, float | int]:
    base = empty_gdelt_features()

    for key in base.keys():
        if key not in fallback or pd.isna(fallback[key]):
            continue

        if key.endswith("_avg_tone"):
            base[key] = safe_float(fallback[key], default=0.0)
        else:
            base[key] = safe_int(fallback[key], default=0)

    return base


def get_latest_gdelt_features(
    fallback: dict,
    max_lookback_days: int = GDELT_LOOKBACK_DAYS,
) -> tuple[str, Dict[str, float | int]]:
    today = datetime.now(timezone.utc).date()

    for i in range(max_lookback_days + 1):
        candidate_date = (today - timedelta(days=i)).strftime("%Y-%m-%d")

        print(f"[INFO] Trying GDELT date: {candidate_date}")

        features = get_gdelt_features_for_date(candidate_date)

        total_signal = 0.0

        for key, value in features.items():
            if key.endswith("_avg_tone"):
                total_signal += abs(safe_float(value, default=0.0))
            else:
                total_signal += abs(safe_int(value, default=0))

        if total_signal > 0:
            print(f"[INFO] GDELT latest usable date found: {candidate_date}")
            return candidate_date, features

    print(f"[WARN] No GDELT signal found in last {max_lookback_days} days")

    return "", fill_gdelt_features_from_fallback(fallback)


# =========================
# Build integrated live raw row
# =========================
def build_live_raw_row() -> pd.DataFrame:
    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fallback = load_latest_history_fallback()

    current_brent, brent_date = get_latest_close_with_date("BZ=F", "Brent")
    current_wti, wti_date = get_latest_close_with_date("CL=F", "WTI")
    vix, vix_date = get_latest_close_with_date("^VIX", "VIX")
    us10y, us10y_date = get_latest_close_with_date("^TNX", "US10Y")
    dxy, dxy_date = get_latest_close_with_date("DX-Y.NYB", "DXY")

    market_reference_date = choose_market_reference_date(
        brent_date=brent_date,
        wti_date=wti_date,
        fallback=fallback,
    )

    if current_brent == 0:
        current_brent = safe_float(fallback.get("current_Brent"), default=0.0)

    if current_wti == 0:
        current_wti = safe_float(fallback.get("current_WTI"), default=0.0)

    if vix == 0:
        vix = safe_float(fallback.get("VIX"), default=0.0)

    if us10y == 0:
        us10y = safe_float(fallback.get("US10Y"), default=0.0)

    if dxy == 0:
        dxy = safe_float(fallback.get("DXY"), default=0.0)

    current_dubai = current_brent - 0.5 if current_brent > 0 else 0.0

    if current_dubai == 0:
        current_dubai = safe_float(fallback.get("current_Dubai"), default=0.0)

    crude_inventory = get_latest_eia_crude_inventory(EIA_API_KEY)

    if crude_inventory == 0:
        crude_inventory = safe_float(fallback.get("crude_inventory"), default=0.0)

    gpr_date, gpr_features = get_latest_gpr_features(fallback)

    acled_date, acled_features = get_latest_acled_features(
        max_lookback_days=ACLED_LOOKBACK_DAYS,
    )

    if not acled_date:
        print("[WARN] ACLED recent data not found. Using live_raw_history fallback.")
        acled_features = fill_acled_features_from_fallback(fallback)
        acled_date = safe_date_string(fallback.get("acled_reference_date"))

    gdelt_date, gdelt_features = get_latest_gdelt_features(
        fallback=fallback,
        max_lookback_days=GDELT_LOOKBACK_DAYS,
    )

    if not gdelt_date:
        print("[WARN] GDELT recent data not found. Using live_raw_history fallback.")
        gdelt_date = safe_date_string(fallback.get("gdelt_reference_date"))

    hormuz_risk = int(acled_features.get("hormuz_risk", 0))
    gulf_supply_disruption = int(acled_features.get("gulf_supply_disruption", 0))
    oil_infrastructure_attack = int(acled_features.get("oil_infrastructure_attack", 0))

    row = {
        # date는 오늘 날짜가 아니라 실제 유가 마지막 거래일
        "date": market_reference_date,
        "collected_at": collected_at,
        "market_reference_date": market_reference_date,
        "brent_reference_date": brent_date,
        "wti_reference_date": wti_date,
        "vix_reference_date": vix_date,
        "us10y_reference_date": us10y_date,
        "dxy_reference_date": dxy_date,
        "gpr_reference_date": gpr_date if gpr_date else "",
        "acled_reference_date": acled_date if acled_date else "",
        "gdelt_reference_date": gdelt_date if gdelt_date else "",
        "current_Dubai": float(current_dubai),
        "current_Brent": float(current_brent),
        "current_WTI": float(current_wti),
        "DXY": float(dxy),
        "VIX": float(vix),
        "US10Y": float(us10y),
        "crude_inventory": float(crude_inventory),
        "hormuz_risk": hormuz_risk,
        "gulf_supply_disruption": gulf_supply_disruption,
        "oil_infrastructure_attack": oil_infrastructure_attack,
    }

    row.update(gpr_features)
    row.update(acled_features)
    row.update(gdelt_features)

    row["gdelt_global_hormuz_risk_count"] = int(
        row.get("gdelt_global_hormuz_risk_count", hormuz_risk)
    )
    row["gdelt_MiddleEast_hormuz_risk_count"] = int(
        row.get("gdelt_MiddleEast_hormuz_risk_count", hormuz_risk)
    )
    row["gdelt_global_gulf_supply_disruption_count"] = int(
        row.get("gdelt_global_gulf_supply_disruption_count", gulf_supply_disruption)
    )
    row["gdelt_MiddleEast_gulf_supply_disruption_count"] = int(
        row.get("gdelt_MiddleEast_gulf_supply_disruption_count", gulf_supply_disruption)
    )
    row["gdelt_global_oil_infrastructure_attack_count"] = int(
        row.get(
            "gdelt_global_oil_infrastructure_attack_count", oil_infrastructure_attack
        )
    )
    row["gdelt_MiddleEast_oil_infrastructure_attack_count"] = int(
        row.get(
            "gdelt_MiddleEast_oil_infrastructure_attack_count",
            oil_infrastructure_attack,
        )
    )

    print(f"[INFO] selected market_reference_date: {market_reference_date}")

    return pd.DataFrame([row])


# =========================
# Save
# =========================
def save_live_raw(df: pd.DataFrame):
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] Saved latest live raw CSV -> {OUTPUT_CSV}")


def append_live_raw_history(df: pd.DataFrame):
    if RAW_HISTORY_CSV.exists():
        history = pd.read_csv(RAW_HISTORY_CSV)
        combined = pd.concat([history, df], ignore_index=True)
    else:
        combined = df.copy()

    if "date" not in combined.columns:
        raise ValueError("live raw history에 date 컬럼이 없습니다.")

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date")

    # 같은 실제 거래일은 마지막 수집값으로 덮어쓴다.
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    combined.to_csv(
        RAW_HISTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] Appended live raw history -> {RAW_HISTORY_CSV}")
    print(f"[INFO] live_raw_history rows: {len(combined)}")


# =========================
# Main
# =========================
def main():
    df = build_live_raw_row()

    save_live_raw(df)
    append_live_raw_history(df)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
