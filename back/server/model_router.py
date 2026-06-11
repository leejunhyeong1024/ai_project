# back/server/model_router.py

from __future__ import annotations

from typing import Any

SHOCK_KEYWORDS = [
    "hormuz",
    "strait_of_hormuz",
    "gulf_supply",
    "supply_disruption",
    "oil_infrastructure_attack",
    "infrastructure_attack",
    "pipeline_attack",
    "refinery_attack",
    "gdelt_global_hormuz_risk_count",
    "gdelt_middleeast_hormuz_risk_count",
    "gdelt_global_gulf_supply_disruption_count",
    "gdelt_middleeast_gulf_supply_disruption_count",
    "gdelt_global_oil_infrastructure_attack_count",
    "gdelt_middleeast_oil_infrastructure_attack_count",
]


def normalize_feature_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def is_shock_related_feature(feature_name: str) -> bool:
    normalized = normalize_feature_name(feature_name)

    return any(keyword in normalized for keyword in SHOCK_KEYWORDS)


def should_use_shock_aware_model(selected_features: dict[str, Any] | None) -> bool:
    if not selected_features:
        return False

    # 💡 순환 참조(Circular Import) 에러를 방지하기 위해 함수 내부에서 안전하게 로드
    try:
        from back.server.feature_builder import load_default_features
    except ImportError:
        from feature_builder import load_default_features

    try:
        defaults = load_default_features()
    except Exception:
        defaults = {}

    # 대소문자 및 띄어쓰기 억까 방지를 위해 소문자 정규화 장부 생성
    norm_defaults = {normalize_feature_name(k): v for k, v in defaults.items()}

    for feature_name, value in selected_features.items():
        if value is None:
            continue

        if is_shock_related_feature(feature_name):
            norm_key = normalize_feature_name(feature_name)
            
            # 💡 오늘 자 인터넷 실시간 뉴스 베이스라인 기본 수치 추출
            default_val = norm_defaults.get(norm_key, 0.0)
            
            # 🎯 핵심: 사용자가 오늘 기준치보다 슬라이더를 '더 높게 드래그하여 위기를 고조시켰을 때만' 쇼크 모드 발동!
            if float(value) > float(default_val):
                return True

    return False


def route_model_type(selected_features: dict[str, Any] | None) -> str:
    if should_use_shock_aware_model(selected_features):
        return "shock_aware"

    return "default"