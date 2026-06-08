# back/server/config.py

from __future__ import annotations

from pathlib import Path

# ==============================
# Path settings
# ==============================

# 현재 파일: AI_PROJECT/back/server/config.py
# parents[0] = server
# parents[1] = back
# parents[2] = AI_PROJECT
PROJECT_DIR = Path(__file__).resolve().parents[2]

BACK_DIR = PROJECT_DIR / "back"
SERVER_DIR = BACK_DIR / "server"
MODEL_DIR = BACK_DIR / "models"

FINAL_MODEL_DIR = MODEL_DIR / "final"
EXPERIMENT_MODEL_DIR = MODEL_DIR / "experiments"

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTION_DIR = DATA_DIR / "prediction"

# ==============================
# Model paths
# ==============================

DEFAULT_MODEL_PATH = FINAL_MODEL_DIR / "default_model.pkl"
SHOCK_AWARE_MODEL_PATH = FINAL_MODEL_DIR / "shock_aware_model.pkl"

# ==============================
# Data paths
# ==============================

LATEST_FEATURE_DEFAULTS_PATH = PREDICTION_DIR / "latest_feature_defaults.json"

# 실시간 데이터 담당자가 갱신할 수 있는 파일
LIVE_FEATURE_DEFAULTS_PATH = PREDICTION_DIR / "live_feature_defaults.json"

# fallback용 processed dataset
DUBAI_DATASET_PATH = PROCESSED_DIR / "dubai_dataset.csv"
BRENT_DATASET_PATH = PROCESSED_DIR / "brent_dataset.csv"
WTI_DATASET_PATH = PROCESSED_DIR / "wti_dataset.csv"

# ==============================
# Service settings
# ==============================

DEFAULT_OIL_TYPE = "Dubai"
DEFAULT_CURRENT_PRICE_KEY = "current_Dubai"

# 모델 target은 10거래일 뒤 수익률(%)
PREDICTION_HORIZON_TRADING_DAYS = 10

# shock-aware model 라우팅 기준
SHOCK_USER_FEATURE_KEYS = {
    "hormuz_risk",
    "gulf_supply_disruption",
    "oil_infrastructure_attack",
    "news_tone",
    "gdelt_avg_tone",
}

# ==============================
# Utility
# ==============================


def ensure_server_directories() -> None:
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def print_server_paths() -> None:
    print("=" * 80)
    print("Server path settings")
    print("=" * 80)
    print("PROJECT_DIR:", PROJECT_DIR)
    print("BACK_DIR:", BACK_DIR)
    print("SERVER_DIR:", SERVER_DIR)
    print("FINAL_MODEL_DIR:", FINAL_MODEL_DIR)
    print("DEFAULT_MODEL_PATH:", DEFAULT_MODEL_PATH)
    print("SHOCK_AWARE_MODEL_PATH:", SHOCK_AWARE_MODEL_PATH)
    print("LATEST_FEATURE_DEFAULTS_PATH:", LATEST_FEATURE_DEFAULTS_PATH)
    print("LIVE_FEATURE_DEFAULTS_PATH:", LIVE_FEATURE_DEFAULTS_PATH)
