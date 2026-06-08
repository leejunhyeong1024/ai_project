# back/server/feature_builder.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# =========================
# Path settings
# =========================
# file path:
# ai_project/back/server/feature_builder.py
#
# parents[0] = server
# parents[1] = back
# parents[2] = ai_project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction"
LATEST_FEATURE_DEFAULTS_PATH = PREDICTION_DIR / "latest_feature_defaults.json"


def load_default_features() -> dict[str, Any]:
    """
    서버 예측에 사용할 최신 feature default 값을 로드한다.

    이 파일은 build_live_features.py가 생성한다.
    """
    if not LATEST_FEATURE_DEFAULTS_PATH.exists():
        raise FileNotFoundError(
            f"latest_feature_defaults.json이 없습니다: {LATEST_FEATURE_DEFAULTS_PATH}\n"
            f"먼저 아래 명령어를 실행하세요:\n"
            f"python3 back/server/jobs/fetch_latest_raw_api_data.py\n"
            f"python3 back/server/jobs/build_live_features.py"
        )

    with open(LATEST_FEATURE_DEFAULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def merge_user_features(
    default_features: dict[str, Any],
    selected_features: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    사용자가 입력한 feature 값으로 default feature를 덮어쓴다.
    """
    features = default_features.copy()

    if not selected_features:
        return features

    for key, value in selected_features.items():
        features[key] = value

    return features


def make_feature_vector(
    feature_values: dict[str, Any],
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    모델이 요구하는 feature_cols 순서에 맞춰 입력 dict를 만든다.
    없는 값은 None으로 둔다.
    모델 Pipeline의 SimpleImputer가 처리한다.
    """
    return {col: feature_values.get(col, None) for col in feature_cols}
