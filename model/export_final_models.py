# model/export_final_models.py

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

from config import EXPERIMENT_DIR, FINAL_MODEL_DIR, ensure_directories

# ==============================
# Final model choices
# ==============================

FINAL_MODEL_SOURCES = {
    "dubai_default_model.pkl": (
        "final_candidates/"
        "dubai_default_price_momentum_gpr_gdelt_event_tone_xgboost.pkl"
    ),
    "dubai_shock_aware_model.pkl": (
        "final_candidates/" "dubai_shock_aware_price_momentum_gpr_extra_trees.pkl"
    ),
    "wti_default_model.pkl": (
        "final_candidates/"
        "wti_default_price_momentum_gpr_selected_extra_20_xgboost.pkl"
    ),
    "wti_shock_aware_model.pkl": (
        "final_candidates/"
        "wti_shock_aware_price_momentum_gpr_region_conflict_extra_trees.pkl"
    ),
    "brent_default_model.pkl": (
        "final_candidates/"
        "brent_default_price_momentum_gpr_selected_extra_10_extra_trees.pkl"
    ),
    "brent_shock_aware_model.pkl": (
        "final_candidates/"
        "brent_shock_aware_price_momentum_gpr_selected_extra_20_xgboost.pkl"
    ),
}


def validate_model_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"candidate model이 없습니다: {path}")

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
        raise ValueError(f"{path.name} bundle 필수 key 누락: {missing}")

    if not bundle["feature_cols"]:
        raise ValueError(f"{path.name} feature_cols가 비어 있습니다.")

    return bundle


def main():
    ensure_directories()
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("최종 모델 export 시작")
    print("=" * 80)
    print("source dir:", EXPERIMENT_DIR / "final_candidates")
    print("target dir:", FINAL_MODEL_DIR)

    manifest = {}

    for final_name, relative_source in FINAL_MODEL_SOURCES.items():
        source_path = EXPERIMENT_DIR / relative_source
        target_path = FINAL_MODEL_DIR / final_name

        bundle = validate_model_bundle(source_path)

        shutil.copy2(source_path, target_path)

        manifest[final_name] = {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "oil_type": bundle["oil_type"],
            "model_mode": bundle["model_mode"],
            "feature_set": bundle["feature_set"],
            "model_name": bundle["model_name"],
            "target_type": bundle["target_type"],
            "target_horizon": bundle["target_horizon"],
            "train_cutoff": bundle["train_cutoff"],
            "feature_count": len(bundle["feature_cols"]),
        }

        print("\n[저장 완료]")
        print("final:", final_name)
        print("from :", source_path)
        print("to   :", target_path)
        print("oil_type:", bundle["oil_type"])
        print("model_mode:", bundle["model_mode"])
        print("feature_set:", bundle["feature_set"])
        print("model_name:", bundle["model_name"])
        print("train_cutoff:", bundle["train_cutoff"])
        print("feature_count:", len(bundle["feature_cols"]))

    manifest_path = FINAL_MODEL_DIR / "final_model_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("최종 모델 export 완료")
    print("=" * 80)
    print("manifest:", manifest_path)

    print("\n최종 생성 파일:")
    for final_name in FINAL_MODEL_SOURCES:
        print("-", FINAL_MODEL_DIR / final_name)


if __name__ == "__main__":
    main()
