# model/config.py

from pathlib import Path

# ==============================
# Project paths
# ==============================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTION_DIR = DATA_DIR / "prediction"

BACK_DIR = PROJECT_DIR / "back"
MODEL_OUTPUT_DIR = BACK_DIR / "models"
EXPERIMENT_DIR = MODEL_OUTPUT_DIR / "experiments"
FINAL_MODEL_DIR = MODEL_OUTPUT_DIR / "final"


# ==============================
# Raw data paths
# ==============================

OIL_PRICE_PATH = RAW_DIR / "oil_price_daily.csv"
CONFLICT_PATH = RAW_DIR / "conflict_events_daily.csv"
GPR_PATH = RAW_DIR / "gpr_daily.csv"

DXY_PATH = RAW_DIR / "dxy_daily.csv"
VIX_PATH = RAW_DIR / "vix_daily.csv"
US10Y_PATH = RAW_DIR / "us10y_daily.csv"
CRUDE_INVENTORY_PATH = RAW_DIR / "us_crude_inventory_daily.csv"
GDELT_PATH = RAW_DIR / "gdelt_daily.csv"


# ==============================
# Processed data paths
# ==============================

ALL_OIL_DATASET_PATH = PROCESSED_DIR / "all_oil_dataset.csv"

DATASET_PATHS = {
    "brent": PROCESSED_DIR / "brent_dataset.csv",
    "dubai": PROCESSED_DIR / "dubai_dataset.csv",
    "wti": PROCESSED_DIR / "wti_dataset.csv",
}


# ==============================
# Target settings
# ==============================

OIL_TYPES = ["brent", "dubai", "wti"]

OIL_COLUMN_MAP = {
    "brent": "Brent",
    "dubai": "Dubai",
    "wti": "WTI",
}

CURRENT_PRICE_COLUMNS = {
    "brent": "current_Brent",
    "dubai": "current_Dubai",
    "wti": "current_WTI",
}

FORECAST_HORIZON_TRADING_DAYS = 10

TARGET_COL = "target"
DATE_COL = "date"
TARGET_DATE_COL = "target_date"


# ==============================
# Split settings
# ==============================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ==============================
# Feature settings
# ==============================

PRICE_MOMENTUM_WINDOWS = [5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20]
LAG_DAYS = [1, 2, 3]

SELECTED_EXTRA_TOP_K_LIST = [10, 20, 30]


# ==============================
# Model settings
# ==============================

RANDOM_STATE = 42

BASELINE_NAME = "baseline_no_change"

PRIMARY_OIL_TYPE = "dubai"

PRIMARY_FEATURE_SET = "price_momentum_gpr"
PRIMARY_MODEL_NAME = "random_forest"


# ==============================
# Output files
# ==============================

FEATURE_SCORE_PATH = EXPERIMENT_DIR / "feature_scores.csv"
SELECTED_FEATURES_PATH = EXPERIMENT_DIR / "selected_features.json"
TRAIN_RESULTS_PATH = EXPERIMENT_DIR / "train_results.csv"
TRAIN_RESULTS_JSON_PATH = EXPERIMENT_DIR / "train_results.json"

FINAL_MODEL_PATH = FINAL_MODEL_DIR / "dubai_model.pkl"
FINAL_FEATURE_COLUMNS_PATH = FINAL_MODEL_DIR / "dubai_feature_columns.pkl"
FINAL_MODEL_SUMMARY_PATH = FINAL_MODEL_DIR / "model_summary.json"

EVALUATION_RESULT_PATH = FINAL_MODEL_DIR / "dubai_evaluation_result.json"
TEST_PREDICTIONS_PATH = FINAL_MODEL_DIR / "dubai_test_predictions.csv"


# ==============================
# Directory setup
# ==============================


def ensure_directories():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def print_config_summary():
    print("=" * 80)
    print("Project configuration")
    print("=" * 80)
    print("PROJECT_DIR:", PROJECT_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("EXPERIMENT_DIR:", EXPERIMENT_DIR)
    print("FINAL_MODEL_DIR:", FINAL_MODEL_DIR)
    print("PRIMARY_OIL_TYPE:", PRIMARY_OIL_TYPE)
    print("FORECAST_HORIZON_TRADING_DAYS:", FORECAST_HORIZON_TRADING_DAYS)
