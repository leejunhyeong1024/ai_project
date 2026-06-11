import os
import json
import requests
import yfinance as yf
from datetime import datetime

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏳ 실시간 일일 데이터 수집 가동 (latest_feature_defaults 대상)...")

def get_latest_market_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1d")
        if not df.empty:
            return round(df['Close'].iloc[-1], 3)
    except Exception as e:
        print(f"⚠️ {ticker_symbol} 데이터 수집 실패: {e}")
    return 0.0

print("📈 야후 파이낸스에서 시장 지표를 추출합니다...")
latest_vix = get_latest_market_data("^VIX")       
latest_tnx = get_latest_market_data("^TNX")       
latest_dxy = get_latest_market_data("DX-Y.NYB")   
latest_brent = get_latest_market_data("BZ=F")     
latest_wti = get_latest_market_data("CL=F")       
latest_dubai = round(latest_brent - 0.5, 3) if latest_brent > 0 else 0.0

def get_today_gdelt_tone(query):
    try:
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=Tone&format=json&timespan=1d"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "tone" in data and len(data["tone"]) > 0:
                return round(data["tone"][0].get("tone", 0.0), 3)
    except:
        pass
    return 0.0

def get_today_gdelt_count(query):
    try:
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=TimelineVolRaw&format=json&timespan=1d"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "timeline" in data and len(data["timeline"]) > 0:
                return int(data["timeline"][0].get("value", 0))
    except:
        pass
    return 0

print("📰 GDELT API에서 글로벌 뉴스 지표를 분석합니다...")
hormuz_risk_cnt = get_today_gdelt_count('("Strait of Hormuz" OR "Hormuz closure")')
gulf_supply_cnt = get_today_gdelt_count('("oil supply disruption" OR "production outage")')
oil_attack_cnt = get_today_gdelt_count('("oil facility" OR "pipeline") AND "attack"')
today_news_tone = get_today_gdelt_tone("oil OR economy OR crisis")

latest_gpr = 115.5          
latest_crude = 459000.0     

live_features = {
    "vix": float(latest_vix),
    "dxy": float(latest_dxy),
    "us10y": float(latest_tnx),
    "crude_inventory": float(latest_crude),
    "gpr": float(latest_gpr),
    
    "hormuz_risk": int(hormuz_risk_cnt),
    "gulf_supply_disruption": int(gulf_supply_cnt),
    "oil_infrastructure_attack": int(oil_attack_cnt),
    "news_tone": float(today_news_tone),
    "gdelt_avg_tone": float(today_news_tone),
    
    "current_Dubai": float(latest_dubai),
    "current_Brent": float(latest_brent),
    "current_WTI": float(latest_wti)
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if "server" in BASE_DIR or "back" in BASE_DIR:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) if "server" in BASE_DIR else os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

SERVER_TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "prediction")

os.makedirs(SERVER_TARGET_DIR, exist_ok=True)

target_file_path = os.path.join(SERVER_TARGET_DIR, "latest_feature_defaults.json")

with open(target_file_path, "w", encoding="utf-8") as f:
    json.dump(live_features, f, indent=4, ensure_ascii=False)

print("=" * 80)
print(f"🎯 엉뚱한 폴더 생성 방지 완료! 기존 찐 폴더에 완벽하게 덮어씌웠습니다.")
print(f"📍 덮어쓴 파일 위치: {os.path.abspath(target_file_path)}")
print("=" * 80)