// 1. 모델팀 백엔드 API 주소 설정 (FastAPI 기본 포트 8000)
const API_BASE_URL = 'http://127.0.0.1:8000';

// 2. 글로벌 원유별 피처 세트 정의 (모델팀 스펙 기반)
const FEATURE_SETS_BY_OIL = {
  WTI: [
    { id: 'wti_macro_gpr', label: '종합 거시경제 + GPR 세트', vars: ['dxy', 'us10y', 'vix', 'gpr'] }
  ],
  Brent: [
    { id: 'brent_geo_heavy', label: '지정학 리스크 집중 세트', vars: ['gpr', 'vix', 'dxy'] }
  ],
  Dubai: [
    { id: 'dubai_asia_macro', label: '아시아 매크로 연동 세트', vars: ['dxy', 'gpr'] }
  ]
};

// 3. 모델팀의 진짜 변수 스펙 정의
const ALL_VAR_SPECS = {
  vix: { id: 'vix', label: 'VIX 변동성 지수', badge: '시장 위험', color: '#7C3AED', bg: '#F5F3FF', min: 10, max: 50, step: 0.1, unit: '' },
  dxy: { id: 'dxy', label: '달러 인덱스 (DXY)', badge: '환율 영향', color: '#1A56DB', bg: '#EBF2FF', min: 90, max: 115, step: 0.01, unit: '' },
  us10y: { id: 'us10y', label: '미국 10년물 국채 금리', badge: '거시 경제', color: '#059669', bg: '#ECFDF5', min: 2.0, max: 6.0, step: 0.001, unit: '%' },
  gpr: { id: 'gpr', label: 'GPR 지정학적 리스크 지수', badge: '전쟁 리스크', color: '#E53E3E', bg: '#FFF1F1', min: 50, max: 300, step: 0.1, unit: 'p' }
};

const SHOCK_VARS = [
  { id: 'hormuz_lock', label: '호르무즈 해협 봉쇄 위기 고조 (Shock Mode 강제 진입)', coef: 7.5 }
];

let currentDefaults = {}; 
let forecastChart = null;
let predictTimeout = null; // 🚨 디바운싱용 타이머 변수

// [로드 스텝 1] 백엔드로부터 최신 기본 수치들 가져오기
async function fetchDefaultFeatures() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/defaults`);
    if (response.ok) {
      currentDefaults = await response.json();
    } else {
      throw new Error();
    }
  } catch (e) {
    console.warn("백엔드 기본값 API 연결 실패. 로컬 더미 데이터로 구동합니다.");
    currentDefaults = { dxy: 100.06, us10y: 4.536, vix: 19.76, gpr: 115.5, WTI_base: 94.58, Brent_base: 97.59, Dubai_base: 97.09 };
  }
}

// [로드 스텝 2] 유가 선택 변경 시 시뮬레이터 UI 재지정
function onTargetOilChange() {
  const oil = document.getElementById('targetOilSelect').value;
  const setSelect = document.getElementById('featureSetSelect');
  if (!setSelect) return;
  
  setSelect.innerHTML = '';
  FEATURE_SETS_BY_OIL[oil].forEach(set => {
    const opt = document.createElement('option');
    opt.value = set.id; 
    opt.textContent = set.label;
    setSelect.appendChild(opt);
  });
  
  renderSimulator();
  triggerDebouncedPrediction(); // 변경 시 즉시 혹은 디바운스 호출
}

function getActiveVars() {
  const oil = document.getElementById('targetOilSelect').value;
  const setId = document.getElementById('featureSetSelect').value;
  const currentSet = FEATURE_SETS_BY_OIL[oil].find(s => s.id === setId);
  return currentSet ? currentSet.vars.map(vId => ALL_VAR_SPECS[vId]) : [];
}

// 🚨 [2번 최적화] 디바운스 함수 추가: 연속적인 슬라이더 움직임 압축
function triggerDebouncedPrediction() {
  // 사용자가 마우스를 계속 움직이는 중이라면 이전 타이머를 취소
  if (predictTimeout) {
    clearTimeout(predictTimeout);
  }
  
  // 🚨 [1번 UX] 사용자가 입력을 멈추는 순간 화면 숫자를 흐리게(로딩 상태) 변경
  const heroBlock = document.querySelector('.predict-main');
  if (heroBlock) heroBlock.classList.add('loading-state');

  // 0.15초(150ms) 동안 멈추면 딱 한 번만 진짜 백엔드 서버에 API 요청을 보냄
  predictTimeout = setTimeout(() => {
    updatePrediction();
  }, 150000 / 1000000 ? 150 : 150); // 150ms 변환값
}

// [실시간 연동] FastAPI 서버로 데이터 전송 및 예측
async function updatePrediction() {
  const oil = document.getElementById('targetOilSelect').value;
  const basePrice = currentDefaults[`${oil}_base`] || 94.58;
  
  const payload = {
    target_oil: oil,
    feature_set_id: document.getElementById('featureSetSelect').value,
    features: {},
    shocks: {}
  };
  
  getActiveVars().forEach(v => {
    const el = document.getElementById('slider_' + v.id);
    payload.features[v.id] = el ? parseFloat(el.value) : (currentDefaults[v.id] || v.min);
  });
  
  SHOCK_VARS.forEach(s => {
    const el = document.getElementById('chk_' + s.id);
    payload.shocks[s.id] = el && el.checked ? 1 : 0;
  });

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) throw new Error();
    const data = await response.json(); 
    
    renderUI(data.predicted, data.delta, data.activated_mode, data.factors, basePrice);
  } catch (e) {
    // 폴백 가상 로직
    let delta = 0;
    const factors = [];
    getActiveVars().forEach(v => {
      let diff = payload.features[v.id] - (currentDefaults[v.id] || v.min);
      let coef = v.id === 'gpr' ? 0.08 : v.id === 'vix' ? 0.22 : -0.45;
      let contrib = diff * coef;
      delta += contrib;
      if (Math.abs(contrib) > 0.01) factors.push({ name: v.label, value: contrib });
    });
    if (payload.shocks.hormuz_lock === 1) {
      delta += 7.5;
      factors.push({ name: '호르무즈 해협 위기', value: 7.5 });
    }
    const mockMode = (delta > 5 || payload.shocks.hormuz_lock === 1) ? "SHOCK (Ridge)" : "NORMAL (RF)";
    renderUI(basePrice + delta, delta, mockMode, factors, basePrice);
  } finally {
    // 🚨 [1번 UX] 서버 통신 완료(혹은 연산 완료) 시 로딩 상태 해제해 흐림 복구
    const heroBlock = document.querySelector('.predict-main');
    if (heroBlock) heroBlock.classList.remove('loading-state');
  }
}

function renderUI(predicted, delta, mode, factors, basePrice) {
  const isUp = delta >= 0;
  const pct = ((delta / basePrice) * 100).toFixed(2);

  document.getElementById('predictNumber').textContent = '$' + predicted.toFixed(2);
  document.getElementById('predictNumber').className = 'predict-number ' + (isUp ? 'up' : 'dn');
  document.getElementById('predictDelta').className = 'predict-delta ' + (isUp ? 'up' : 'dn');
  document.getElementById('deltaText').textContent = `${isUp ? '+' : ''}$${delta.toFixed(2)} (${isUp ? '+' : ''}${pct}%)`;
  document.getElementById('predictRange').textContent = `$${(predicted - 2.5).toFixed(2)} – $${(predicted + 2.5).toFixed(2)}`;
  
  const tag = document.querySelector('.model-tag');
  if (tag) {
    tag.textContent = `🟢 MODE: ${mode}`;
    tag.style.background = mode.includes('SHOCK') ? '#FFF1F1' : '#ECFDF5';
    tag.style.color = mode.includes('SHOCK') ? '#D42B2B' : '#059669';
  }
  renderFactorRows(factors);
  if (forecastChart) {
    const trend = (predicted - basePrice) / 4;
    forecastChart.data.datasets[0].data = [basePrice, basePrice + trend, basePrice + trend*2, basePrice + trend*3, predicted];
    forecastChart.update();
  }
}

function renderSimulator() {
  const container = document.getElementById('simulatorGrid');
  if (!container) return;
  container.innerHTML = '';
  
  getActiveVars().forEach(v => {
    const defaultVal = currentDefaults[v.id] || v.min;
    const pct = ((defaultVal - v.min) / (v.max - v.min) * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'sim-item';
    item.innerHTML = `
      <div class="sim-header">
        <div class="sim-label">${v.label} <span class="sim-label-badge" style="color:${v.color}; background:${v.bg}">${v.badge}</span></div>
        <div class="sim-value" id="simVal_${v.id}">${defaultVal}${v.unit}</div>
      </div>
      <input type="range" class="sim-slider" id="slider_${v.id}" min="${v.min}" max="${v.max}" step="${v.step}" value="${defaultVal}" style="background: linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)" oninput="onSliderInput(this, '${v.id}')">
      <div class="sim-range-labels"><span>${v.min}${v.unit}</span><span>${v.max}${v.unit}</span></div>
    `;
    container.appendChild(item);
  });

  const div = document.createElement('div');
  div.innerHTML = `<div class="divider" style="margin:1.5rem 0; border-top:1px dashed var(--border); "></div><div class="card-title" style="font-size:0.75rem; margin-bottom:0.5rem;"><span style="color:#E53E3E">⚠️</span> 지정학적 게이트키퍼 활성화 스위치</div>`;
  container.appendChild(div);
  
  SHOCK_VARS.forEach(s => {
    const row = document.createElement('div');
    row.style = 'display:flex; align-items:center; gap:10px; margin-bottom:6px;';
    row.innerHTML = `<input type="checkbox" id="chk_${s.id}" onchange="triggerDebouncedPrediction()" style="width:15px; height:15px; cursor:pointer;"><label for="chk_${s.id}" style="font-size:0.78rem; color:var(--text-secondary); cursor:pointer;">${s.label}</label>`;
    container.appendChild(row);
  });
}

window.onSliderInput = function(slider, id) {
  const val = parseFloat(slider.value);
  const v = ALL_VAR_SPECS[id];
  slider.style.background = `linear-gradient(to right, #1A56DB ${((val - v.min) / (v.max - v.min) * 100).toFixed(1)}%, #E8ECF2 ${((val - v.min) / (v.max - v.min) * 100).toFixed(1)}%)`;
  document.getElementById('simVal_' + id).textContent = val + v.unit;
  
  // 🚨 [2번 최적화] 슬라이더 드래그할 때 매순간 쏘지 않고, 디바운스 타이머 가동
  triggerDebouncedPrediction();
}

function renderFactorRows(factors) {
  const container = document.getElementById('factorRows');
  if (!container) return;
  container.innerHTML = '';
  if (!factors || factors.length === 0) {
    container.innerHTML = '<div style="font-size:0.8rem; color:var(--text-muted); text-align:center; padding:1rem;">모든 지표가 최신 균형 상태입니다.</div>';
    return;
  }
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.value)), 1);
  factors.forEach(f => {
    const isP = f.value >= 0;
    const row = document.createElement('div');
    row.className = 'factor-row';
    row.innerHTML = `<div class="factor-name">${f.name}</div><div class="factor-bar-wrap"><div class="factor-bar ${isP?'up':'dn'}" style="width:${(Math.abs(f.value)/maxAbs*100).toFixed(1)}%"></div></div><div class="factor-score ${isP?'up':'dn'}">${isP?'+':''}$${f.value.toFixed(2)}</div>`;
    container.appendChild(row);
  });
}

function initChart() {
  const ctx = document.getElementById('forecastChart');
  if (ctx) forecastChart = new Chart(ctx.getContext('2d'), { type: 'line', data: { labels: ['현재','2일뒤','5일뒤','8일뒤','10일뒤(예측)'], datasets: [{ label: '시뮬레이션 경로', data: [94.58, 94.58, 94.58, 94.58, 94.58], borderColor: '#1A56DB', backgroundColor: 'rgba(26,86,219,0.05)', borderWidth: 2.5, tension: 0.3, fill: true }] }, options: { responsive: true, maintainAspectRatio: false } });
}

async function init() {
  await fetchDefaultFeatures();
  onTargetOilChange();
  initChart();
}
window.addEventListener('DOMContentLoaded', init);