# back/server/model_loader.py

from __future__ import annotations

import pickle
from pathlib import Path

try:
    from back.server.config import (
        DEFAULT_MODEL_PATH,
        SHOCK_AWARE_MODEL_PATH,
    )
except ModuleNotFoundError:
    from config import (
        DEFAULT_MODEL_PATH,
        SHOCK_AWARE_MODEL_PATH,
    )


# ==============================
# Model loading
# ==============================


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n"
            f"먼저 final 모델 파일을 생성하거나 복사하세요."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def validate_model_bundle(bundle: dict, name: str) -> None:
    required_keys = ["feature_cols"]

    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"{name} model bundle에 '{key}' 키가 없습니다.")

    # 단일 회귀 모델 bundle
    if "model" in bundle:
        return

    # hybrid bundle도 일단 허용
    if all(key in bundle for key in ["classifier", "normal_reg", "shock_reg"]):
        return

    raise ValueError(
        f"{name} model bundle에서 예측 모델을 찾지 못했습니다. "
        f"'model' 또는 hybrid 구성 키가 필요합니다."
    )


def load_models() -> dict:
    default_model = load_pickle(DEFAULT_MODEL_PATH)
    shock_aware_model = load_pickle(SHOCK_AWARE_MODEL_PATH)

    validate_model_bundle(default_model, "default")
    validate_model_bundle(shock_aware_model, "shock_aware")

    return {
        "default": default_model,
        "shock_aware": shock_aware_model,
    }


def summarize_model_bundle(bundle: dict) -> dict:
    return {
        "oil_type": bundle.get("oil_type", "unknown"),
        "feature_set": bundle.get("feature_set", "unknown"),
        "model_name": bundle.get("model_name", bundle.get("model", "unknown")),
        "target_type": bundle.get("target_type", "return_pct"),
        "feature_count": len(bundle.get("feature_cols", [])),
    }


def summarize_loaded_models(models: dict) -> dict:
    return {
        model_type: summarize_model_bundle(bundle)
        for model_type, bundle in models.items()
    }
