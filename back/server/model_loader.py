# back/server/model_loader.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# =========================
# Path settings
# =========================
# file path:
# ai_project/back/server/model_loader.py
#
# parents[0] = server
# parents[1] = back
# parents[2] = ai_project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_DIR = PROJECT_ROOT / "back" / "models" / "final"


OIL_TYPES = ["dubai", "wti", "brent"]
MODEL_TYPES = ["default", "shock_aware"]

MODEL_FILE_MAP = {
    "dubai": {
        "default": "dubai_default_model.pkl",
        "shock_aware": "dubai_shock_aware_model.pkl",
    },
    "wti": {
        "default": "wti_default_model.pkl",
        "shock_aware": "wti_shock_aware_model.pkl",
    },
    "brent": {
        "default": "brent_default_model.pkl",
        "shock_aware": "brent_shock_aware_model.pkl",
    },
}


def normalize_oil_type(oil_type: str) -> str:
    oil = str(oil_type).strip().lower()

    aliases = {
        "dubai": "dubai",
        "두바이": "dubai",
        "wti": "wti",
        "brent": "brent",
        "브렌트": "brent",
    }

    if oil not in aliases:
        raise ValueError(f"지원하지 않는 oil_type입니다: {oil_type}")

    return aliases[oil]


def load_pickle_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    required_keys = [
        "oil_type",
        "model_mode",
        "feature_set",
        "model_name",
        "target_type",
        "target_horizon",
        "train_cutoff",
        "feature_cols",
        "model",
    ]

    missing = [key for key in required_keys if key not in bundle]

    if missing:
        raise ValueError(f"{path.name} 모델 bundle 필수 key 누락: {missing}")

    return bundle


def load_models() -> dict[str, dict[str, dict[str, Any]]]:
    models = {}

    for oil_type in OIL_TYPES:
        models[oil_type] = {}

        for model_type in MODEL_TYPES:
            filename = MODEL_FILE_MAP[oil_type][model_type]
            path = FINAL_MODEL_DIR / filename

            bundle = load_pickle_model(path)
            models[oil_type][model_type] = bundle

    return models


def get_model_bundle(
    models: dict[str, dict[str, dict[str, Any]]],
    oil_type: str,
    model_type: str,
) -> dict[str, Any]:
    oil = normalize_oil_type(oil_type)
    model_type = str(model_type).strip().lower()

    if model_type not in MODEL_TYPES:
        raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")

    return models[oil][model_type]


def summarize_models(models: dict[str, dict[str, dict[str, Any]]]) -> dict:
    summary = {}

    for oil_type, oil_models in models.items():
        summary[oil_type] = {}

        for model_type, bundle in oil_models.items():
            summary[oil_type][model_type] = {
                "oil_type": bundle.get("oil_type"),
                "model_mode": bundle.get("model_mode"),
                "feature_set": bundle.get("feature_set"),
                "model_name": bundle.get("model_name"),
                "target_type": bundle.get("target_type"),
                "target_horizon": bundle.get("target_horizon"),
                "train_cutoff": bundle.get("train_cutoff"),
                "feature_count": len(bundle.get("feature_cols", [])),
            }

    return summary
