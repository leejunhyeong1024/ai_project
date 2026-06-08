/* ────────────────────────────────────────────────────────────
   상수 & 초기 상태 데이터
──────────────────────────────────────────────────────────── */
const BASE_PRICE = 82.40;
const UNCERTAINTY = 3.6;

// 시뮬레이터 입력 변수 정의
const SIM_VARS = [
  {
    id: 'syriaBlasts',
    label: '시리아 폭발 사건 (건/주)',
    badge: '공급 리스크',
    badgeColor: '#E53E3E',
    badgeBg: '#FFF1F1',
    min: 0, max: 50, step: 1, value: 12,
    unit: '건',
    coef: 0.08,   // 1건당 유가 영향 (USD)
    impact: '1건 증가 시 약 +$0.08',
  },
  {
    id: 'yemenFights',
    label: '예멘 교전 건수 (건/주)',
    badge: '해상 리스크',
    badgeColor: '#D97706',
    badgeBg: '#FFFBEB',
    min: 0, max: 80, step: 1, value: 28,
    unit: '건',
    coef: 0.06,
    impact: '1건 증가 시 약 +$0.06',
  },
  {
    id: 'iranSanctions',
    label: '이란 제재 강도 (0–100)',
    badge: 'OFAC 지수',
    badgeColor: '#7C3AED',
    badgeBg: '#F5F3FF',
    min: 0, max: 100, step: 1, value: 72,
    unit: 'p',
    coef: 0.04,
    impact: '1p 상승 시 약 +$0.04',
  },
  {
    id: 'dxy',
    label: '달러 인덱스 (DXY)',
    badge: '환율 영향',
    badgeColor: '#1A56DB',
    badgeBg: '#EBF2FF',
    min: 90, max: 115, step: 0.1, value: 104.2,
    unit: '',
    coef: -0.55,   // DXY 100 초과분에 적용
    impact: '1p 상승 시 약 −$0.55',
    special: 'dxy', // 특수 처리
  },
  {
    id: 'opecCut',
    label: 'OPEC 감산량 (mb/d)',
    badge: '공급량',
    badgeColor: '#059669',
    badgeBg: '#ECFDF5',
    min: -2, max: 3, step: 0.1, value: -0.5,
    unit: 'mb/d',
    coef: -1.8,    // 감산(-) → 가격 ↑
    impact: '1mb/d 감산 시 약 +$1.8',
  },
  {
    id: 'eiaInventory',
    label: 'EIA 원유 재고 변화 (mb)',
    badge: '수급 지표',
    badgeColor: '#0694A2',
    badgeBg: '#ECFEFF',
    min: -10, max: 10, step: 0.5, value: -2,
    unit: 'mb',
    coef: -0.28,   // 재고 감소(-) → 가격 ↑
    impact: '1mb 감소 시 약 +$0.28',
  },
];

// Feature Importance (읽기 전용, 모델 학습 결과)
const FEAT_IMPORTANCE = [
  { name: 'DXY 달러 인덱스',       pct: 22.4, color: '#1A56DB' },
  { name: 'OPEC 생산 결정',          pct: 19.1, color: '#7C3AED' },
  { name: '분쟁 종합 지수',          pct: 17.6, color: '#E53E3E' },
  { name: 'EIA 원유 재고',           pct: 13.8, color: '#059669' },
  { name: '이란 제재 강도',          pct: 10.5, color: '#D97706' },
  { name: '예멘 교전 건수',          pct: 8.2,  color: '#0694A2' },
  { name: '시리아 폭발 건수',       pct: 5.3,  color: '#E53E3E' },
  { name: 'S&P 500 변동성(VIX)',    pct: 3.1,  color: '#6B7280' },
];

// 중동 지역별 분쟁 현황
const CONFLICT_DATA = [
  { region: 'Gaza / 이스라엘', score: 88, status: '전면전',  sColor:'#E53E3E', sBg:'#FFF1F1' },
  { region: '이란 제재·핵협상', score: 72, status: '고위험', sColor:'#D97706', sBg:'#FFFBEB' },
  { region: 'Yemen 후티 공격', score: 65, status: '고위험',  sColor:'#D97706', sBg:'#FFFBEB' },
  { region: 'Libya 내전',       score: 55, status: '중위험', sColor:'#D97706', sBg:'#FFFBEB' },
  { region: 'Syria 불안정',     score: 45, status: '주시',   sColor:'#6B7280', sBg:'#F3F4F6' },
  { region: 'Iraq 시설 위협',   score: 38, status: '주시',   sColor:'#6B7280', sBg:'#F3F4F6' },
];

// 빠른 시나리오 프리셋
const SCENARIOS = {
  base:  { syriaBlasts:12, yemenFights:28, iranSanctions:72, dxy:104.2, opecCut:-0.5, eiaInventory:-2.0 },
  crisis:{ syriaBlasts:45, yemenFights:70, iranSanctions:95, dxy:102.0, opecCut:-1.5, eiaInventory:-6.0 },
  peace: { syriaBlasts:3,  yemenFights:5,  iranSanctions:30, dxy:105.0, opecCut:0.0,  eiaInventory:2.0  },
  usd:   { syriaBlasts:12, yemenFights:28, iranSanctions:72, dxy:112.0, opecCut:-0.5, eiaInventory:-2.0 },
  opec:  { syriaBlasts:12, yemenFights:28, iranSanctions:72, dxy:104.2, opecCut:-2.0, eiaInventory:-3.0 },
};

/* ────────────────────────────────────────────────────────────
   유가 예측 모델 (진짜 API 연동 대응 비동기 함수)
──────────────────────────────────────────────────────────── */
async function computePrediction(vals) {
  try {
    // [팁] 백엔드 팀원이 서버 주소(IP/Port) 주면 여기 주소를 바꾸면 됨!
    const serverUrl = 'http://127.0.0.1:8000/api/predict'; 
    
    const response = await fetch(serverUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(vals)
    });

    if (!response.ok) throw new Error('서버 응답 오류');
    return await response.json();

  } catch (error) {
    // 서버가 꺼져있을 때는 기존 프론트 하드코딩 로직이 돌아서 화면이 유지되게 방어선 구축 (MOCK BUFFER)
    let delta = 0;
    const factors = [];

    SIM_VARS.forEach(v => {
      let contribution = v.special === 'dxy' ? v.coef * (vals[v.id] - 100) : v.coef * vals[v.id];
      delta += contribution;
      factors.push({ name: v.label, value: contribution });
    });

    const predicted = BASE_PRICE + delta;
    return { predicted, delta, factors };
  }
}

/* ────────────────────────────────────────────────────────────
   현재 슬라이더 값 읽기
──────────────────────────────────────────────────────────── */
function readVals() {
  const vals = {};
  SIM_VARS.forEach(v => {
    const el = document.getElementById('slider_' + v.id);
    vals[v.id] = el ? parseFloat(el.value) : v.value;
  });
  return vals;
}

/* ────────────────────────────────────────────────────────────
   예측 UI 업데이트
──────────────────────────────────────────────────────────── */
async function updatePrediction() {
  const vals = readVals();
  
  // API 응답 대기
  const { predicted, delta, factors } = await computePrediction(vals);
  
  const pct = ((delta / BASE_PRICE) * 100).toFixed(2);
  const isUp = delta >= 0;

  // 숫자 카드
  const numEl = document.getElementById('predictNumber');
  numEl.textContent = '$' + predicted.toFixed(2);
  numEl.className = 'predict-number ' + (isUp ? 'up' : 'dn');

  const deltaEl = document.getElementById('predictDelta');
  deltaEl.className = 'predict-delta ' + (isUp ? 'up' : 'dn');
  document.getElementById('deltaText').textContent =
    (isUp ? '+' : '') + '$' + delta.toFixed(2) + ' (' + (isUp ? '+' : '') + pct + '%)';

  // 신뢰구간
  document.getElementById('predictRange').textContent =
    '$' + (predicted - UNCERTAINTY).toFixed(2) + ' – $' + (predicted + UNCERTAINTY).toFixed(2);

  // DXY KPI 반영
  const dxyEl = document.getElementById('kpiDxy');
  if (dxyEl) dxyEl.textContent = parseFloat(vals.dxy).toFixed(1);

  // 요인 기여도 바
  renderFactorRows(factors);

  // AI 분석 텍스트
  renderReasoning(vals, delta, pct, isUp);

  // 차트 업데이트
  updateForecastChart(predicted);
}

function renderFactorRows(factors) {
  const container = document.getElementById('factorRows');
  if (!container) return;
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.value)), 1);
  container.innerHTML = '';
  factors.forEach(f => {
    const isP = f.value >= 0;
    const pct = (Math.abs(f.value) / maxAbs * 100).toFixed(1);
    const cls = isP ? 'up' : 'dn';
    const row = document.createElement('div');
    row.className = 'factor-row';
    row.innerHTML = `
      <div class="factor-name">${f.name.replace(' (0–100)', '').replace(' (건/주)', '').replace(' (mb/d)', '').replace(' (mb)', '').replace(' (DXY)', '')}</div>
      <div class="factor-bar-wrap"><div class="factor-bar ${cls}" style="width:${pct}%"></div></div>
      <div class="factor-score ${cls}">${isP ? '+' : ''}$${f.value.toFixed(2)}</div>
    `;
    container.appendChild(row);
  });
}

function renderReasoning(vals, delta, pct, isUp) {
  const dxy = parseFloat(vals.dxy);
  const conflict = parseInt(vals.iranSanctions);
  const syria = parseInt(vals.syriaBlasts);
  const yemen = parseInt(vals.yemenFights);

  let dxyMsg = dxy > 108 ? '강달러 기조(<strong>' + dxy.toFixed(1) + '</strong>)가 원자재 전반에 하방 압력을 가하고 있습니다.'
             : dxy < 100  ? '약달러(<strong>' + dxy.toFixed(1) + '</strong>)로 달러 표시 원자재 가격이 지지받고 있습니다.'
                          : '달러 인덱스(<strong>' + dxy.toFixed(1) + '</strong>)가 중립 수준입니다.';
  let cMsg = conflict >= 80 ? '이란 제재 지수가 <strong>' + conflict + 'p</strong>로 극도로 높아 호르무즈 해협 통과 차질 위험이 고조됩니다.'
           : conflict >= 50 ? '이란 제재 지수 <strong>' + conflict + 'p</strong>로 중동 공급 차질 프리미엄이 유가에 반영되고 있습니다.'
                            : '이란 제재 강도가 <strong>' + conflict + 'p</strong>로 낮아 공급 리스크는 제한적입니다.';
  let hotMsg = (syria > 20 || yemen > 40) ? `시리아(${syria}건)·예멘(${yemen}건) 교전이 지속되어 지역 불안정성이 높습니다.`
                                           : `시리아(${syria}건)·예멘(${yemen}건) 교전 건수가 비교적 낮아 단기 충격은 제한적입니다.`;

  document.getElementById('reasoningText').innerHTML =
    `${dxyMsg}<br><br>${cMsg}<br><br>${hotMsg}<br><br>
    종합 판단: 1주 후 유가 <strong style="color:${isUp ? 'var(--up)' : 'var(--dn)'}">${isUp ? '상승' : '하락'} ${Math.abs(pct)}%</strong> 예측.`;
}

/* ────────────────────────────────────────────────────────────
   Chart.js 시각화 로직
──────────────────────────────────────────────────────────── */
let forecastChart = null;
let historicalChart = null;

function buildForecastData(predicted) {
  const labels = ['현재', '1일', '2일', '3일', '4일', '5일', '6일', '7일(예측)'];
  const trend = (predicted - BASE_PRICE) / 7;
  const data  = labels.map((_, i) => parseFloat((BASE_PRICE + trend * i + Math.sin(i * 0.8) * 0.35).toFixed(2)));
  return {
    labels,
    data,
    upper: data.map(v => parseFloat((v + UNCERTAINTY).toFixed(2))),
    lower: data.map(v => parseFloat((v - UNCERTAINTY).toFixed(2))),
  };
}

function initForecastChart(predicted) {
  const { labels, data, upper, lower } = buildForecastData(predicted);
  const canvas = document.getElementById('forecastChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: '기본 예측',
          data,
          borderColor: '#1A56DB',
          backgroundColor: 'rgba(26,86,219,0.06)',
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: '#1A56DB',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0.4,
          fill: false,
        },
        {
          label: '상단 시나리오',
          data: upper,
          borderColor: 'rgba(229,62,62,0.45)',
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.4,
          fill: false,
        },
        {
          label: '하단 시나리오',
          data: lower,
          borderColor: 'rgba(5,150,105,0.45)',
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.4,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => `$${c.parsed.y.toFixed(2)}/bbl` } },
      },
      scales: {
        y: {
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { callback: v => `$${v}`, font: { size: 11 }, color: '#8898AA' },
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 11 }, color: '#8898AA' },
        },
      },
    },
  });
}

function updateForecastChart(predicted) {
  if (!forecastChart) return;
  const { data, upper, lower } = buildForecastData(predicted);
  forecastChart.data.datasets[0].data = data;
  forecastChart.data.datasets[1].data = upper;
  forecastChart.data.datasets[2].data = lower;
  forecastChart.update('active');
}

function initHistoricalChart() {
  const weeks = ['W-12','W-11','W-10','W-9','W-8','W-7','W-6','W-5','W-4','W-3','W-2','W-1'];
  const prices    = [74.2, 76.8, 75.1, 78.4, 80.2, 79.6, 81.1, 79.8, 82.3, 81.9, 83.2, 82.4];
  const conflicts = [52,   55,   54,   60,   62,   61,   65,   63,   67,   66,   68,   67.4];

  const canvas = document.getElementById('historicalChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  historicalChart = new Chart(ctx, {
    data: {
      labels: weeks,
      datasets: [
        {
          type: 'line',
          label: 'WTI 주간 종가',
          data: prices,
          borderColor: '#1A56DB',
          backgroundColor: 'rgba(26,86,219,0.06)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.4,
          yAxisID: 'y1',
          fill: true,
        },
        {
          type: 'bar',
          label: '분쟁 지수',
          data: conflicts,
          backgroundColor: 'rgba(229,62,62,0.15)',
          borderColor: 'rgba(229,62,62,0.4)',
          borderWidth: 1,
          yAxisID: 'y2',
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        y1: {
          position: 'left',
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { callback: v => `$${v}`, font: { size: 11 }, color: '#1A56DB' },
        },
        y2: {
          position: 'right',
          grid: { display: false },
          ticks: { font: { size: 11 }, color: '#E53E3E' },
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 11 }, color: '#8898AA' },
        },
      },
    },
  });
}

/* ────────────────────────────────────────────────────────────
   Feature Importance 렌더링
──────────────────────────────────────────────────────────── */
function renderFeatureImportance() {
  const container = document.getElementById('featTable');
  if (!container) return;
  container.innerHTML = '';
  FEAT_IMPORTANCE.forEach((f, i) => {
    const row = document.createElement('div');
    row.className = 'feat-row';
    row.innerHTML = `
      <div class="feat-name">
        <span class="feat-rank">${i+1}</span>
        ${f.name}
      </div>
      <div class="feat-bar-wrap">
        <div class="feat-bar" style="width:0%; background:${f.color}" data-target="${f.pct}"></div>
      </div>
      <div class="feat-pct">${f.pct}%</div>
    `;
    container.appendChild(row);
  });
  setTimeout(() => {
    container.querySelectorAll('.feat-bar').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 200);
}

/* ────────────────────────────────────────────────────────────
   시뮬레이터 렌더링 및 슬라이더 제어
──────────────────────────────────────────────────────────── */
function renderSimulator() {
  const container = document.getElementById('simulatorGrid');
  if (!container) return;
  container.innerHTML = '';
  SIM_VARS.forEach(v => {
    const displayVal = v.special === 'dxy'
      ? parseFloat(v.value).toFixed(1)
      : (v.id === 'opecCut' || v.id === 'eiaInventory')
        ? (v.value >= 0 ? '+' : '') + parseFloat(v.value).toFixed(1) + v.unit
        : v.value + v.unit;

    const pct = ((v.value - v.min) / (v.max - v.min) * 100).toFixed(1);

    const item = document.createElement('div');
    item.className = 'sim-item';
    item.innerHTML = `
      <div class="sim-header">
        <div class="sim-label">
          ${v.label}
          <span class="sim-label-badge" style="color:${v.badgeColor}; background:${v.badgeBg}">${v.badge}</span>
        </div>
        <div class="sim-value" id="simVal_${v.id}">${displayVal}</div>
      </div>
      <input
        type="range" class="sim-slider"
        id="slider_${v.id}"
        min="${v.min}" max="${v.max}" step="${v.step}" value="${v.value}"
        style="background: linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)"
        oninput="onSliderInput(this, '${v.id}')"
      />
      <div class="sim-range-labels">
        <span>${v.min}${v.unit}</span>
        <span>${v.max}${v.unit}</span>
      </div>
      <div class="sim-impact">💡 ${v.impact}</div>
    `;
    container.appendChild(item);
  });
}

window.onSliderInput = function(slider, id) {
  const v = SIM_VARS.find(x => x.id === id);
  const val = parseFloat(slider.value);
  const pct = ((val - v.min) / (v.max - v.min) * 100).toFixed(1);
  slider.style.background = `linear-gradient(to right, #1A56DB ${pct}%, #E8ECF2 ${pct}%)`;

  const displayEl = document.getElementById('simVal_' + id);
  if (v.special === 'dxy') {
    displayEl.textContent = val.toFixed(1);
  } else if (v.id === 'opecCut' || v.id === 'eiaInventory') {
    displayEl.textContent = (val >= 0 ? '+' : '') + val.toFixed(1) + v.unit;
  } else {
    displayEl.textContent = val + v.unit;
  }

  updatePrediction();
}

/* ────────────────────────────────────────────────────────────
   시나리오 프리셋 제어
──────────────────────────────────────────────────────────── */
window.applyScenario = function(key) {
  document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');

  const preset = SCENARIOS[key];
  SIM_VARS.forEach(v => {
    const slider = document.getElementById('slider_' + v.id);
    if (slider && preset[v.id] !== undefined) {
      slider.value = preset[v.id];
      window.onSliderInput(slider, v.id);
    }
  });
}

/* ────────────────────────────────────────────────────────────
   분쟁 지역 리스트 렌더링
──────────────────────────────────────────────────────────── */
function renderConflictList() {
  const container = document.getElementById('conflictList');
  if (!container) return;
  container.innerHTML = '';
  CONFLICT_DATA.forEach(c => {
    const barColor = c.score >= 70 ? '#E53E3E' : c.score >= 50 ? '#D97706' : '#6B7280';
    const row = document.createElement('div');
    row.className = 'conflict-row-item';
    row.innerHTML = `
      <div class="conflict-region">${c.region}</div>
      <div class="conflict-track">
        <div class="conflict-fill" style="width:${c.score}%; background:${barColor}"></div>
      </div>
      <div class="conflict-score-num" style="color:${barColor}">${c.score}</div>
      <span class="conflict-pill" style="color:${c.sColor}; background:${c.sBg}">${c.status}</span>
    `;
    container.appendChild(row);
  });
}

window.runUpdate = function() {
  updatePrediction();
}

/* ────────────────────────────────────────────────────────────
   초기화 구동 엔진
──────────────────────────────────────────────────────────── */
function init() {
  const now = new Date();
  const updateTimeEl = document.getElementById('updateTime');
  if (updateTimeEl) {
    updateTimeEl.textContent = now.toLocaleString('ko-KR', {
      month: 'long', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  }

  renderSimulator();
  renderFeatureImportance();
  renderConflictList();

  const initVals = {};
  SIM_VARS.forEach(v => { initVals[v.id] = v.value; });
  
  computePrediction(initVals).then(({ predicted }) => {
    initForecastChart(predicted);
    initHistoricalChart();
    updatePrediction();
  });
}

window.addEventListener('DOMContentLoaded', init);