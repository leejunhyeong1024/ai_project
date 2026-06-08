import os
import json
import requests
import yfinance as yf
from datetime import datetime

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏳ 실시간 일일 데이터 수집 가동 (latest_feature_defaults 대상)...")

# ==========================================
# 1. 금융 & 원유 시장 데이터 실시간 수집 (Yahoo Finance API)
# ==========================================
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

# ==========================================
# 2. 지정학적 뉴스 데이터 실시간 수집 (GDELT 2.0 API)
# ==========================================
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

# 고정 지표 설정 (GPR 및 원유 재고 최신값 반영)
latest_gpr = 115.5          
latest_crude = 459000.0     

# ==========================================
# 3. 서버 config 규격에 맞춘 JSON 구조 생성
# ==========================================
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

# ==========================================
# 4. 서버 진짜 루트 폴더(data/prediction) 역추적 및 덮어씌우기
# ==========================================
# 현재 파이썬 파일이 있는 위치 추적
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 만약 스크립트가 'back/server' 또는 'back' 안에 있다면 최상단(ai_project)으로 올라감
if "server" in BASE_DIR or "back" in BASE_DIR:
    # 찐 프로젝트 최상단 폴더 찾기
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) if "server" in BASE_DIR else os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

# 찐 프로젝트 최상단 기준 data/prediction 폴더 정확하게 조준
SERVER_TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "prediction")

# (혹시 몰라서) 폴더가 진짜 없으면 만들고, 있으면 무시
os.makedirs(SERVER_TARGET_DIR, exist_ok=True)

target_file_path = os.path.join(SERVER_TARGET_DIR, "latest_feature_defaults.json")

# 기존 파일 자비 없이 덮어씌우기 ("w" 모드)
with open(target_file_path, "w", encoding="utf-8") as f:
    json.dump(live_features, f, indent=4, ensure_ascii=False)

print("=" * 80)
print(f"🎯 엉뚱한 폴더 생성 방지 완료! 기존 찐 폴더에 완벽하게 덮어씌웠습니다.")
print(f"📍 덮어쓴 파일 위치: {os.path.abspath(target_file_path)}")
print("=" * 80)