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

    for feature_name, value in selected_features.items():
        if value is None:
            continue
        if is_shock_related_feature(feature_name):
            return True

    return False


def route_model_type(selected_features: dict[str, Any] | None) -> str:
    if should_use_shock_aware_model(selected_features):
        return "shock_aware"
    return "default"


# 🚨 모델팀이 누락한 찐 핵심 함수 복구!
def convert_user_features_to_model_features(user_features: dict[str, Any] | None) -> dict[str, Any]:
    """
    프론트엔드 슬라이더에서 들어오는 소문자/대문자 변수명을 
    모델팀 백엔드 내부에서 사용하는 규격(예: vix -> VIX, dxy -> DXY)으로 매핑 및 변환해주는 함수.
    """
    if not user_features:
        return {}
        
    converted = {}
    # 프론트엔드가 보내는 소문자 키값들을 백엔드 대문자 수치 및 쇼크 수치로 매핑
    mapping = {
        "vix": "VIX",
        "dxy": "DXY",
        "us10y": "US10Y",
        "gpr": "GPRD",
        "hormuz_lock": "gdelt_hormuz_risk_count"
    }
    
    for key, value in user_features.items():
        norm_key = key.strip().lower()
        if norm_key in mapping:
            # 호르무즈 해협 스위치(Shock)가 켜지면 카운트를 강제로 높여서 쇼크 모델 트리거
            if norm_key == "hormuz_lock" and value == 1:
                converted[mapping[norm_key]] = 250.0  # 쇼크 임계값 강제 주입
            else:
                converted[mapping[norm_key]] = value
        else:
            converted[key] = value
            
    return converted