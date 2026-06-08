# back/server/model_router.py

from __future__ import annotations

try:
    from back.server.config import SHOCK_USER_FEATURE_KEYS
except ModuleNotFoundError:
    from config import SHOCK_USER_FEATURE_KEYS


# ==============================
# User feature name -> model feature name
# ==============================

USER_TO_MODEL_FEATURE_MAP = {
    # shock / GDELT event
    "hormuz_risk": "gdelt_hormuz_risk_count",
    "gulf_supply_disruption": "gdelt_gulf_supply_disruption_count",
    "oil_infrastructure_attack": "gdelt_oil_infrastructure_attack_count",
    "news_tone": "gdelt_avg_tone",
    "gdelt_avg_tone": "gdelt_avg_tone",
    # market
    "vix": "VIX",
    "dxy": "DXY",
    "us10y": "US10Y",
    "crude_inventory": "crude_inventory",
    # risk
    "gpr": "GPRD",
    # price
    "current_dubai": "current_Dubai",
    "current_brent": "current_Brent",
    "current_wti": "current_WTI",
}


# ==============================
# Routing
# ==============================


def normalize_user_key(key: str) -> str:
    return str(key).strip().lower()


def convert_user_feature_name(user_key: str) -> str:
    normalized_key = normalize_user_key(user_key)
    return USER_TO_MODEL_FEATURE_MAP.get(normalized_key, user_key)


def convert_user_features_to_model_features(user_features: dict | None) -> dict:
    if not user_features:
        return {}

    converted = {}

    for key, value in user_features.items():
        model_key = convert_user_feature_name(key)
        converted[model_key] = value

    return converted


def has_shock_related_feature(user_features: dict | None) -> bool:
    if not user_features:
        return False

    selected_keys = {normalize_user_key(key) for key in user_features.keys()}

    return bool(selected_keys & SHOCK_USER_FEATURE_KEYS)


def select_model_type(user_features: dict | None) -> str:
    """
    일반 feature만 선택:
        default model

    Hormuz / Gulf supply disruption / Oil infrastructure attack / News tone 선택:
        shock-aware model
    """

    if has_shock_related_feature(user_features):
        return "shock_aware"

    return "default"


def get_available_features() -> list[dict]:
    """
    프론트엔드가 선택 UI를 만들 때 사용할 feature 목록.
    """

    return [
        {
            "key": "hormuz_risk",
            "label": "Hormuz Risk",
            "model_feature": "gdelt_hormuz_risk_count",
            "type": "shock",
            "unit": "news count",
            "description": "호르무즈 해협 관련 리스크 뉴스량",
        },
        {
            "key": "gulf_supply_disruption",
            "label": "Gulf Supply Disruption",
            "model_feature": "gdelt_gulf_supply_disruption_count",
            "type": "shock",
            "unit": "news count",
            "description": "중동/걸프 지역 원유 공급 차질 관련 뉴스량",
        },
        {
            "key": "oil_infrastructure_attack",
            "label": "Oil Infrastructure Attack",
            "model_feature": "gdelt_oil_infrastructure_attack_count",
            "type": "shock",
            "unit": "news count",
            "description": "정유시설, 송유관, 석유 인프라 공격 관련 뉴스량",
        },
        {
            "key": "news_tone",
            "label": "News Tone",
            "model_feature": "gdelt_avg_tone",
            "type": "shock",
            "unit": "tone score",
            "description": "GDELT 기반 뉴스 평균 tone",
        },
        {
            "key": "vix",
            "label": "VIX",
            "model_feature": "VIX",
            "type": "market",
            "unit": "index",
            "description": "시장 변동성 지수",
        },
        {
            "key": "dxy",
            "label": "DXY",
            "model_feature": "DXY",
            "type": "market",
            "unit": "index",
            "description": "미국 달러 지수",
        },
        {
            "key": "us10y",
            "label": "US 10Y Yield",
            "model_feature": "US10Y",
            "type": "market",
            "unit": "%",
            "description": "미국 10년물 국채금리",
        },
        {
            "key": "crude_inventory",
            "label": "US Crude Inventory",
            "model_feature": "crude_inventory",
            "type": "market",
            "unit": "barrels",
            "description": "미국 원유 재고",
        },
        {
            "key": "gpr",
            "label": "GPR",
            "model_feature": "GPRD",
            "type": "risk",
            "unit": "index",
            "description": "Geopolitical Risk Index",
        },
    ]
