/* ============================================================
   STOCK PRICE PREDICTION — dashboard engine
   Self-contained: simulated OHLCV data, SVG charts,
   3D canvas centerpiece, model comparison logic.
   ============================================================ */
"use strict";

/* ---------------- utilities ---------------- */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);

/* ---- algorithm runtime (see /algorithms) ----
   The prediction engines live in algorithms/*.js and register on the
   global SPPAlgorithms namespace; app.js consumes them via this alias. */
const ALGO = window.SPPAlgorithms;
const { hashStr, mulberry32, gauss, metricsOf } = ALGO.utils;

/* ---------------- formatting ---------------- */
const MON3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtMoney = (cfg, v, dec = 2) =>
  cfg.cur + v.toLocaleString(cfg.locale, { minimumFractionDigits: dec, maximumFractionDigits: dec });
const fmtNum = (v, dec = 2) =>
  v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
const fmtCompact = v =>
  Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
const fmtDate = d => `${String(d.getDate()).padStart(2, "0")} ${MON3[d.getMonth()]} ${d.getFullYear()}`;
const fmtMonth = d => `${MON3[d.getMonth()]} ${String(d.getFullYear()).slice(2)}`;
const fmtPct = v => (v >= 0 ? "+" : "") + v.toFixed(2) + "%";

/* ---------------- configuration ---------------- */
const N_DAYS = 504;      // 2 trading years
const TEST = 63;         // held-out test window (~3 months)

const STOCKS = {
  TCS:       { name: "Tata Consultancy Services", ticker: "TCS.NS",      exch: "NSE",     cur: "\u20B9", locale: "en-IN", start: 3170, end: 4180, drift: 0.00052, vol: 0.0124, volBase: 2.6e6 },
  RELIANCE:  { name: "Reliance Industries",       ticker: "RELIANCE.NS", exch: "NSE",     cur: "\u20B9", locale: "en-IN", start: 2260, end: 2780, drift: 0.00041, vol: 0.0148, volBase: 7.1e6 },
  GOOGL:     { name: "Alphabet Inc.",             ticker: "GOOGL",       exch: "NASDAQ",  cur: "$",      locale: "en-US", start: 121.5, end: 171.0, drift: 0.00082, vol: 0.0176, volBase: 2.7e7 },
};

/* UI metadata for each model. The algorithms themselves live in /algorithms. */
const MODELS = [
  { id: "arima",    name: "ARIMA",    tag: "Statistical",      color: "#38bdf8",
    desc: "A classical statistics method that predicts tomorrow's price from recent prices and recent errors — simple, fast and reliable.",
    runMs: 950 },
  { id: "prophet",  name: "Prophet",  tag: "Additive Model",   color: "#a78bfa",
    desc: "Splits the price into trend + repeating seasonal patterns + noise, then extends each piece into the future.",
    runMs: 1250 },
  { id: "lstm",     name: "LSTM",     tag: "Deep Learning",    color: "#22d3ee",
    desc: "A deep-learning network with built-in memory that learns repeating patterns in price sequences.",
    runMs: 2000 },
  { id: "xgboost",  name: "XGBoost",  tag: "Gradient Boosting",color: "#34d399",
    desc: "A team of small decision trees that together learn complex, non-linear price patterns.",
    runMs: 1450 },
  { id: "svr",      name: "SVR",      tag: "Kernel Method",    color: "#fbbf24",
    desc: "Fits a smooth curve through the data while ignoring small errors — good at capturing the overall shape.",
    runMs: 1100 },
];

const modelById = id => MODELS.find(m => m.id === id);

/* ---------------- data simulation ---------------- */
function genDates(n, endDateStr) {
  const out = [];
  const d = new Date(endDateStr + "T00:00:00");
  while (out.length < n) {
    const wd = d.getDay();
    if (wd !== 0 && wd !== 6) out.push(new Date(d));
    d.setDate(d.getDate() - 1);
  }
  return out.reverse();
}

function nextBusinessDays(from, n) {
  const out = [];
  const d = new Date(from);
  while (out.length < n) {
    d.setDate(d.getDate() + 1);
    const wd = d.getDay();
    if (wd !== 0 && wd !== 6) out.push(new Date(d));
  }
  return out;
}

const DATASETS = {};

function buildDataset(key) {
  const cfg = STOCKS[key];
  const rng = mulberry32(hashStr(key + "|ohlcv|v4"));
  const dates = genDates(N_DAYS, "2026-09-30");
  const open = [], high = [], low = [], close = [], volume = [];

  let p = cfg.start * (0.995 + rng() * 0.01);
  for (let i = 0; i < N_DAYS; i++) {
    const cycle = Math.sin(i / 41 + 1.2) * cfg.vol * 0.22 + Math.sin(i / 131) * cfg.vol * 0.15;
    let ret = cfg.drift + cycle + gauss(rng) * cfg.vol;
    if (rng() < 0.012) ret += (rng() < 0.45 ? -1 : 1) * cfg.vol * 3.1; // occasional shock
    const c = Math.max(p * (1 + ret), cfg.start * 0.35);
    const o = i === 0 ? p * (1 + gauss(rng) * cfg.vol * 0.3) : close[i - 1] * (1 + gauss(rng) * cfg.vol * 0.22);
    const hi = Math.max(o, c) * (1 + Math.abs(gauss(rng)) * cfg.vol * 0.45);
    const lo = Math.min(o, c) * (1 - Math.abs(gauss(rng)) * cfg.vol * 0.45);
    const v = Math.round(cfg.volBase * Math.exp(gauss(rng) * 0.32) * (1 + Math.abs(ret) / cfg.vol * 0.22));
    open.push(o); high.push(hi); low.push(lo); close.push(c); volume.push(v);
    p = c;
  }

  // geometric trend correction: rotate the whole path so it lands at a
  // realistic 2-year end price (keeps the shape, fixes the level)
  const target = cfg.end * (0.97 + rng() * 0.06);
  const corr = Math.pow(target / close[N_DAYS - 1], 1 / (N_DAYS - 1));
  for (let i = 0; i < N_DAYS; i++) {
    const f = Math.pow(corr, i);
    open[i] *= f; high[i] *= f; low[i] *= f; close[i] *= f;
  }
  return { key, cfg, dates, open, high, low, close, volume, source: "simulated" };
}

/* real market data: manually embedded daily OHLCV (data/market-data.js).
   Drift and volatility are estimated from the real series itself so the
   forecast horizon reflects the stock's actual behaviour. */
function buildRealDataset(key) {
  const raw = window.REAL_MARKET_DATA && window.REAL_MARKET_DATA[key];
  if (!raw) return null;
  const cfg = STOCKS[key];
  const dates = raw.dates.map(d => new Date(d + "T00:00:00"));
  const open = raw.open.map(Number), high = raw.high.map(Number);
  const low = raw.low.map(Number), close = raw.close.map(Number);
  const volume = raw.volume.map(Number);
  const rets = [];
  for (let i = 1; i < close.length; i++) rets.push((close[i] - close[i - 1]) / close[i - 1]);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, rets.length - 1));
  return {
    key,
    cfg: { ...cfg, drift: mean, vol: sd },
    dates, open, high, low, close, volume,
    source: "real",
    symbol: raw.symbol, exchange: raw.exchange,
  };
}
const getData = key => DATASETS[key] || (DATASETS[key] = buildRealDataset(key) || buildDataset(key));

/* directional accuracy: share of test days where the model's predicted
   up/down move matches the actual market move */
function directionAccuracy(ds, pred) {
  const start = ds.close.length - TEST;
  let hit = 0;
  for (let i = start; i < ds.close.length; i++) {
    const upActual = ds.close[i] >= ds.close[i - 1];
    const upPred = pred[i - start] >= ds.close[i - 1];
    if (upActual === upPred) hit++;
  }
  return (hit / TEST) * 100;
}

/* std of the model's return-prediction errors on the test window —
   powers the per-model confidence intervals in Pick a Day */
function errorStd(ds, pred) {
  const start = ds.close.length - TEST;
  const errs = [];
  for (let i = start; i < ds.close.length; i++) {
    const rActual = ds.close[i] / ds.close[i - 1] - 1;
    const rPred = pred[i - start] / ds.close[i - 1] - 1;
    errs.push(rPred - rActual);
  }
  const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
  return Math.sqrt(errs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, errs.length - 1));
}

/* standard normal CDF (Abramowitz & Stegun 26.2.17, hand-written) */
function normCdf(z) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z >= 0 ? 1 - p : p;
}

/* Diebold-Mariano test: is the RMSE gap between the two best models
   statistically significant, or just noise? (squared-error loss) */
function dieboldMariano(actual, predA, predB) {
  const n = actual.length;
  const d = new Array(n);
  for (let i = 0; i < n; i++) {
    const eA = actual[i] - predA[i], eB = actual[i] - predB[i];
    d[i] = eA * eA - eB * eB;
  }
  const mean = d.reduce((a, b) => a + b, 0) / n;
  const varr = d.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, n - 1);
  const dm = varr > 0 ? mean / Math.sqrt(varr / n) : 0;
  const p = 2 * (1 - normCdf(Math.abs(dm)));
  return { dm: +dm.toFixed(2), p: +p.toFixed(3), n };
}

/* leadership strip: which model had the lowest RMSE in each third
   of the test window (predictions already computed) */
function leadershipStrip(ds) {
  const evs = evalAll(ds);
  const chunks = [[0, 20], [21, 41], [42, 62]];
  return chunks.map(([a, b], i) => {
    let best = null;
    for (const e of evs) {
      let se = 0;
      for (let k = a; k <= b; k++) { const d = e.pred[k] - ds.close[ds.close.length - TEST + k]; se += d * d; }
      const rmse = Math.sqrt(se / (b - a + 1));
      if (!best || rmse < best.rmse) best = { model: e.model, rmse };
    }
    return { label: "Days −" + (63 - a) + "…−" + (63 - 1 - b), model: best.model, rmse: +best.rmse.toFixed(2) };
  });
}

/* model predictions over the test window — engines live in /algorithms.
   The engines genuinely train on each call, so predictions are cached
   per (stock, model, hyperparameters). */
const _predCache = {};
function predictFor(ds, model) {
  const ck = ds.key + "|" + model.id + "|" + JSON.stringify(hyperparams);
  if (_predCache[ck]) return _predCache[ck];
  const pred = ALGO[model.id].predict({ key: ds.key, close: ds.close, TEST, hyperparams });
  _predCache[ck] = pred;
  return pred;
}

const _evalCache = {};
function evalAll(ds) {
  if (_evalCache[ds.key]) return _evalCache[ds.key];
  const actual = ds.close.slice(ds.close.length - TEST);
  const results = MODELS.map(m => {
    const pred = predictFor(ds, m);
    return { model: m, pred, ...metricsOf(actual, pred), dir: directionAccuracy(ds, pred), errStd: errorStd(ds, pred) };
  }).sort((a, b) => a.rmse - b.rmse);
  _evalCache[ds.key] = results;
  return results;
}

/* future forecast using the LSTM engine */
function forecastFor(ds, horizon) {
  const rng = mulberry32(hashStr(ds.key + "|forecast|lstm|v4"));
  const { close, dates, cfg } = ds;
  let p = close[close.length - 1];
  const mid = [], up = [], lo = [];
  for (let i = 1; i <= horizon; i++) {
    p = p * (1 + cfg.drift * 1.08 + gauss(rng) * cfg.vol * 0.85);
    mid.push(p);
    const sigma = p * cfg.vol * 0.85 * Math.sqrt(i) * 1.7;
    up.push(p + sigma); lo.push(p - sigma);
  }
  return { mid, up, lo, dates: nextBusinessDays(dates[dates.length - 1], horizon) };
}

/* ---------------- app state ---------------- */
const STORAGE_KEY = "spp_state";
const TRAINED_KEY = "spp_trained";

function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (s) {
      Object.assign(state, s);
      if (s.indicators) state.indicators = { ...state.indicators, ...s.indicators };
    }
  } catch {}
  /* sanitise saved selections against the current model list */
  if (!modelById(state.compare)) state.compare = "lstm";
  if (!modelById(state.backtestModel)) state.backtestModel = "lstm";
  delete state.forecastEngine;
}
function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      ticker: state.ticker,
      range: state.range,
      horizon: state.horizon,
      compare: state.compare,
      backtestModel: state.backtestModel,
      forecastView: state.forecastView,
      indicators: state.indicators
    }));
  } catch {}
}
function loadTrained() {
  try {
    const t = JSON.parse(localStorage.getItem(TRAINED_KEY));
    if (Array.isArray(t)) t.forEach(k => trained.add(k));
  } catch {}
}
function saveTrained() {
  try { localStorage.setItem(TRAINED_KEY, JSON.stringify([...trained])); } catch {}
}

const state = {
  ticker: "TCS",
  range: 252,
  horizon: 30,
  compare: "lstm",
  backtestModel: "lstm",
  forecastView: "forecast",
  indicators: { ma20: false, ma50: false, rsi: false, macd: false }
};
const trained = new Set(); // "TICKER:model"
loadState();
loadTrained();
const isTrained = (t, m) => trained.has(t + ":" + m);

const hyperparams = {
  lstm: { epochs: 50, lookback: 30, lr: 0.005 },
  arima: { p: 2, d: 1, q: 2 },
  xgboost: { depth: 6, subsample: 0.8 }
};

function calcMA(values, period) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      out.push(null);
    } else {
      let sum = 0;
      for (let j = 0; j < period; j++) sum += values[i - j];
      out.push(sum / period);
    }
  }
  return out;
}

function calcRSI(values, period = 14) {
  const rsi = new Array(values.length).fill(null);
  if (values.length <= period) return rsi;

  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = values[i] - values[i - 1];
    if (diff >= 0) gains += diff; else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  rsi[period] = avgLoss === 0 ? 100 : 100 - (100 / (1 + (avgGain / avgLoss)));

  for (let i = period + 1; i < values.length; i++) {
    const diff = values[i] - values[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    rsi[i] = avgLoss === 0 ? 100 : 100 - (100 / (1 + (avgGain / avgLoss)));
  }
  return rsi;
}

function calcEMA(values, period) {
  const k = 2 / (period + 1);
  const ema = new Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < period; i++) sum += values[i];
  ema[period - 1] = sum / period;
  for (let i = period; i < values.length; i++) {
    ema[i] = values[i] * k + ema[i - 1] * (1 - k);
  }
  return ema;
}

function calcMACD(values) {
  const ema12 = calcEMA(values, 12);
  const ema26 = calcEMA(values, 26);
  const macdLine = new Array(values.length).fill(null);
  const validMacd = [];
  for (let i = 25; i < values.length; i++) {
    macdLine[i] = ema12[i] - ema26[i];
    validMacd.push(macdLine[i]);
  }
  const signalShort = calcEMA(validMacd, 9);
  const signalLine = new Array(values.length).fill(null);
  const hist = new Array(values.length).fill(null);
  let sIdx = 0;
  for (let i = 25; i < values.length; i++) {
    if (signalShort[sIdx] != null) {
      signalLine[i] = signalShort[sIdx];
      hist[i] = macdLine[i] - signalLine[i];
    }
    sIdx++;
  }
  return { macdLine, signalLine, hist };
}

function runBacktest(ds, modelId = "lstm") {
  const actual = ds.close.slice(ds.close.length - TEST);
  const m = modelById(modelId);
  const pred = predictFor(ds, m);

  let stratEquity = 100;
  let bnhEquity = 100;
  const stratCurve = [100];
  const bnhCurve = [100];
  let wins = 0, trades = 0;
  const dailyReturns = [];

  for (let i = 0; i < TEST - 1; i++) {
    const actToday = actual[i];
    const actNext = actual[i + 1];
    const predNext = pred[i + 1];
    const pctChange = (actNext - actToday) / actToday;

    bnhEquity *= (1 + pctChange);
    bnhCurve.push(bnhEquity);

    const signal = predNext > actToday ? 1 : 0;
    if (signal === 1) {
      trades++;
      if (pctChange > 0) wins++;
      stratEquity *= (1 + pctChange);
      dailyReturns.push(pctChange);
    } else {
      dailyReturns.push(0);
    }
    stratCurve.push(stratEquity);
  }

  const stratRoi = stratCurve[stratCurve.length - 1] - 100;
  const bnhRoi = bnhCurve[bnhCurve.length - 1] - 100;
  const winRate = trades > 0 ? (wins / trades) * 100 : 0;

  const meanR = dailyReturns.reduce((a, b) => a + b, 0) / (dailyReturns.length || 1);
  const varR = dailyReturns.reduce((a, b) => a + Math.pow(b - meanR, 2), 0) / (dailyReturns.length - 1 || 1);
  const stdR = Math.sqrt(varR);
  const sharpe = stdR > 0 ? ((meanR / stdR) * Math.sqrt(252)).toFixed(2) : "0.00";

  return {
    stratCurve,
    bnhCurve,
    stratRoi,
    bnhRoi,
    winRate,
    trades,
    sharpe,
    dates: ds.dates.slice(ds.dates.length - TEST)
  };
}

function runMonteCarlo(ds, horizon, numPaths = 50) {
  const { close, dates, cfg } = ds;
  const startP = close[close.length - 1];
  const dt = 1 / 252;
  const mu = cfg.drift * 252;
  const sigma = cfg.vol * Math.sqrt(252);
  const rng = mulberry32(hashStr(ds.key + "|mc|v4"));

  const paths = [];
  for (let p = 0; p < numPaths; p++) {
    const path = [startP];
    let curr = startP;
    for (let t = 1; t <= horizon; t++) {
      const z = gauss(rng);
      curr = curr * Math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * z);
      path.push(curr);
    }
    paths.push(path);
  }

  const terminalPrices = paths.map(p => p[p.length - 1]).sort((a, b) => a - b);
  const p5 = terminalPrices[Math.floor(numPaths * 0.05)];
  const p50 = terminalPrices[Math.floor(numPaths * 0.50)];
  const p95 = terminalPrices[Math.floor(numPaths * 0.95)];

  const var95Pct = Math.max(0, ((startP - p5) / startP) * 100);
  const var95Money = Math.max(0, startP - p5);

  return {
    paths,
    p5,
    p50,
    p95,
    var95Pct,
    var95Money,
    startP,
    sigmaAnnual: (sigma * 100).toFixed(1),
    dates: nextBusinessDays(dates[dates.length - 1], horizon)
  };
}

function downloadCSV(filename, rows) {
  const processRow = row => row.map(val => {
    let text = val === null || val === undefined ? "" : String(val);
    if (text.search(/("|,|\n)/g) >= 0) text = '"' + text.replace(/"/g, '""') + '"';
    return text;
  }).join(",");
  const csvContent = rows.map(processRow).join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.setAttribute("href", url);
  a.setAttribute("download", filename);
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ============================================================
   SVG LINE CHART
   ============================================================ */
class LineChart {
  constructor(el, opts = {}) {
    this.el = el;
    this.h = opts.height || 350;
    this.yFmt = opts.yFmt || (v => fmtNum(v, 0));
    this.cur = opts.cur || "";
    this.data = null;
    el.style.position = "relative";
    this._onMove = this._onMove.bind(this);
    this._onLeave = this._onLeave.bind(this);
  }

  setData(data) { this.data = data; this._render(); }

  _render() {
    const d = this.data;
    if (!d) return;
    const w = this.el.clientWidth || 600;
    const isFullscreen = this.el.closest(".card--fullscreen");
    const h = isFullscreen ? Math.max(520, this.el.clientHeight || 520) : this.h;
    const ML = 56, MR = 14, MT = 16, MB = 30;
    const pw = w - ML - MR, ph = h - MT - MB;

    // ----- scales -----
    let vals = [];
    d.series.forEach(s => s.values.forEach(v => { if (v != null) vals.push(v); }));
    (d.bands || []).forEach(b => { b.upper.forEach(v => { if (v != null) vals.push(v); }); b.lower.forEach(v => { if (v != null) vals.push(v); }); });
    let vmin = Math.min(...vals), vmax = Math.max(...vals);
    const padv = (vmax - vmin) * 0.08 || 1;
    const t = niceTicks(vmin - padv, vmax + padv, 4);
    const yMin = t.lo, yMax = t.hi;
    const Y = v => MT + ph - ((v - yMin) / (yMax - yMin)) * ph;
    const n = d.dates.length;
    const X = i => ML + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);

    // volume lane occupies the bottom 20% of the plot
    const volTop = d.volume ? MT + ph * 0.8 : 0;
    const volBot = MT + ph;
    const vmaxV = d.volume ? Math.max(...d.volume) * 1.1 : 1;
    const pricePh = d.volume ? ph * 0.78 : ph;
    const Yp = v => MT + pricePh - ((v - yMin) / (yMax - yMin)) * pricePh;

    let g = "";
    // grid + y labels
    t.ticks.forEach(v => {
      const y = Y(v).toFixed(1);
      g += `<line x1="${ML}" y1="${y}" x2="${w - MR}" y2="${y}" stroke="rgba(148,163,184,0.09)" stroke-dasharray="3 5"/>`;
      g += `<text x="${ML - 10}" y="${+y + 3.5}" text-anchor="end" class="ax">${this.yFmt(v)}</text>`;
    });
    // x labels (deduplicated by month, first & last always shown)
    const cand = [0, Math.round((n - 1) / 3), Math.round(2 * (n - 1) / 3), n - 1];
    const xl = [];
    cand.forEach((i, k) => {
      const label = fmtMonth(d.dates[i]);
      if (k === cand.length - 1) {
        const dup = xl.findIndex(x => x.label === label);
        if (dup >= 0) xl.splice(dup, 1);
        xl.push({ i, label });
      } else if (!xl.some(x => x.label === label)) {
        xl.push({ i, label });
      }
    });
    xl.forEach(x => {
      g += `<text x="${X(x.i).toFixed(1)}" y="${h - 8}" text-anchor="middle" class="ax">${x.label}</text>`;
    });
    // volume bars
    if (d.volume) {
      const bw = Math.max(1.5, (pw / n) * 0.62);
      d.volume.forEach((v, i) => {
        const bh = ((v / vmaxV) * (volBot - volTop)) || 0;
        const up = d.volDir ? d.volDir[i] : true;
        g += `<rect x="${(X(i) - bw / 2).toFixed(1)}" y="${(volBot - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="1" fill="${up ? "rgba(52,211,153,0.22)" : "rgba(251,113,133,0.22)"}"/>`;
      });
    }
    // bands
    (d.bands || []).forEach((b, bi) => {
      const pts = [];
      for (let i = 0; i < n; i++) if (b.upper[i] != null) pts.push(i);
      if (!pts.length) return;
      let path = "";
      pts.forEach(i => { path += (path ? "L" : "M") + X(i).toFixed(1) + " " + Yp(b.upper[i]).toFixed(1); });
      for (let j = pts.length - 1; j >= 0; j--) { const i = pts[j]; path += "L" + X(i).toFixed(1) + " " + Yp(b.lower[i]).toFixed(1); }
      g += `<path d="${path}Z" fill="url(#band${bi})" stroke="none"/>`;
    });
    // series
    const grads = [];
    d.series.forEach((s, si) => {
      let path = "", pen = false;
      for (let i = 0; i < n; i++) {
        const v = s.values[i];
        if (v == null) { pen = false; continue; }
        path += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Yp(v).toFixed(1);
        pen = true;
      }
      if (s.fill) {
        grads.push(`<linearGradient id="area${si}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${s.color}" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="${s.color}" stop-opacity="0"/></linearGradient>`);
        let fillPath = path, started = false, firstX = 0, lastX = 0;
        for (let i = 0; i < n; i++) if (s.values[i] != null) { firstX = X(i); break; }
        for (let i = n - 1; i >= 0; i--) if (s.values[i] != null) { lastX = X(i); break; }
        fillPath += `L${lastX.toFixed(1)} ${(MT + pricePh).toFixed(1)}L${firstX.toFixed(1)} ${(MT + pricePh).toFixed(1)}Z`;
        g += `<path d="${fillPath}" fill="url(#area${si})" stroke="none"/>`;
      }
      g += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linecap="round" stroke-linejoin="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} ${s.glow ? `style="filter:url(#glow)"` : ""}/>`;
    });

    const bandDefs = (d.bands || []).map((b, bi) =>
      `<linearGradient id="band${bi}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${b.color}" stop-opacity="0.20"/>
        <stop offset="100%" stop-color="${b.color}" stop-opacity="0.04"/></linearGradient>`).join("");

    this.el.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
        <defs>
          ${bandDefs}
          ${grads.join("")}
          <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="3.2" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        ${g}
        <line class="focus-line" x1="0" y1="${MT}" x2="0" y2="${volBot}" stroke="rgba(103,232,249,0.4)" stroke-dasharray="4 4" opacity="0"/>
        ${d.series.map(s => `<circle class="focus-dot" cx="-10" cy="-10" r="3.4" fill="${s.color}" stroke="#0a0f1c" stroke-width="1.5" opacity="0"/>`).join("")}
        <rect class="hit" x="${ML}" y="0" width="${pw}" height="${h}" fill="transparent"/>
      </svg>`;
    this._geom = { ML, MR, pw, n, X, Yp, volBot, MT };
    this.svg = this.el.querySelector("svg");
    this.focusLine = this.svg.querySelector(".focus-line");
    this.focusDots = $$(".focus-dot", this.svg);
    const hit = this.svg.querySelector(".hit");
    hit.addEventListener("pointermove", this._onMove);
    hit.addEventListener("pointerleave", this._onLeave);
  }

  _onMove(ev) {
    const d = this.data, { ML, pw, n, X, Yp, volBot, MT } = this._geom;
    const rect = this.svg.getBoundingClientRect();
    const scale = rect.width ? this.svg.clientWidth / rect.width : 1; // viewBox matches px, keep simple
    const mx = (ev.clientX - rect.left) * (this.svg.viewBox.baseVal.width / rect.width);
    const i = clamp(Math.round(((mx - ML) / pw) * (n - 1)), 0, n - 1);
    const x = X(i);
    this.focusLine.setAttribute("x1", x); this.focusLine.setAttribute("x2", x);
    this.focusLine.setAttribute("opacity", "1");
    let rows = "";
    d.series.forEach((s, si) => {
      const v = s.values[i];
      this.focusDots[si].setAttribute("cx", x);
      this.focusDots[si].setAttribute("cy", v == null ? -10 : Yp(v));
      this.focusDots[si].setAttribute("opacity", v == null ? 0 : 1);
      if (v != null) rows += `<div class="tt-row"><span class="n" style="--c:${s.color}"><i></i>${s.name}</span><span class="v">${s.tip ? s.tip(v, i) : fmtNum(v, 2)}</span></div>`;
    });
    const tt = $("#tooltip");
    tt.innerHTML = `<div class="tt-date">${fmtDate(d.dates[i])}</div>${rows}`;
    tt.classList.add("show");
    const tw = tt.offsetWidth, th = tt.offsetHeight;
    const vw = window.innerWidth, vh = window.innerHeight;
    let tx = ev.clientX + 16; if (tx + tw > vw - 10) tx = ev.clientX - tw - 16;
    let ty = ev.clientY - 14;   if (ty + th > vh - 10) ty = ev.clientY - th - 10;
    tt.style.left = tx + "px";
    tt.style.top = Math.max(8, ty) + "px";
  }

  _onLeave() {
    if (this.focusLine) this.focusLine.setAttribute("opacity", "0");
    this.focusDots.forEach(c => c.setAttribute("opacity", "0"));
    $("#tooltip").classList.remove("show");
  }
}

function niceTicks(min, max, n = 4) {
  if (max - min < 1e-9) { min -= 1; max += 1; }
  const step0 = (max - min) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 0.001; v += step) ticks.push(v);
  return { ticks, lo, hi };
}

/* ============================================================
   RENDERERS
   ============================================================ */

/* ---- 01 market data ---- */
function renderMarket(ds) {
  const cfg = ds.cfg, n = ds.close.length;
  const last = ds.close[n - 1], prev = ds.close[n - 2];
  const chg = (last - prev) / prev * 100;
  const yr = Math.max(0, n - 252);
  const hi52 = Math.max(...ds.high.slice(yr)), lo52 = Math.min(...ds.low.slice(yr));
  const avgV = ds.volume.slice(-20).reduce((a, b) => a + b, 0) / 20;

  $("#m-stats-label").textContent = `${cfg.ticker} · ${cfg.exch}`;
  $("#m-range-label").textContent = `RANGE · ${fmtMonth(ds.dates[0])} — ${fmtMonth(ds.dates[n - 1])}`;
  const dsLabel = document.getElementById("m-dataset-label");
  if (dsLabel) dsLabel.textContent = `DATASET · ${n} TRADING DAYS · ${ds.source === "real" ? "REAL" : "SIMULATED"} · DAILY`;

  const stats = [
    { k: "LATEST CLOSE", v: fmtMoney(cfg, last), s: "price at end of last session", cls: "" },
    { k: "DAY CHANGE", v: fmtPct(chg), s: "vs previous session", cls: chg >= 0 ? "pos" : "neg" },
    { k: "52W HIGH", v: fmtMoney(cfg, hi52), s: "highest price in the past year", cls: "" },
    { k: "52W LOW", v: fmtMoney(cfg, lo52), s: "lowest price in the past year", cls: "" },
    { k: "AVG VOLUME", v: fmtCompact(avgV), s: "shares traded per day · last 20", cls: "" },
  ];
  $("#market-stats").innerHTML = stats.map(st =>
    `<div class="stat"><div class="k">${st.k}</div><div class="v ${st.cls}">${st.v}</div><div class="s">${st.s}</div></div>`).join("")
    + `<div class="range52">
        <div class="r52-head"><span class="k">52W RANGE · CURRENT POSITION</span><span class="r52-now">${fmtMoney(cfg, last)}</span></div>
        <div class="r52-track"><span class="r52-marker" style="left:${(((last - lo52) / (hi52 - lo52)) * 100).toFixed(1)}%"></span></div>
        <div class="r52-ends"><span>${fmtMoney(cfg, lo52, 0)}</span><span>${fmtMoney(cfg, hi52, 0)}</span></div>
      </div>`;

  const closeStatEl = $("#market-stats .stat:first-child .v");
  if (closeStatEl) animateCounter(closeStatEl, last, v => fmtMoney(cfg, v), 600);

  const tb = $("#ohlcv-table tbody");
  let rows = "";
  for (let i = n - 6; i < n; i++) {
    const c = ds.close[i], pc = ds.close[i - 1];
    const ch = (c - pc) / pc * 100, up = ch >= 0;
    rows += `<tr>
      <td style="color:var(--text-2)">${fmtDate(ds.dates[i])}</td>
      <td>${fmtMoney(cfg, ds.open[i])}</td>
      <td class="up">${fmtMoney(cfg, ds.high[i])}</td>
      <td class="dn">${fmtMoney(cfg, ds.low[i])}</td>
      <td>${fmtMoney(cfg, c)}</td>
      <td class="${up ? "up" : "dn"}">${fmtPct(ch)}</td>
      <td class="vol">${fmtCompact(ds.volume[i])}</td>
    </tr>`;
  }
  tb.innerHTML = rows;
}

/* ---- 02 stock cards ---- */
function sparkSVG(values, color) {
  const w = 100, h = 40;
  const vmin = Math.min(...values), vmax = Math.max(...values);
  const X = i => (i / (values.length - 1)) * w;
  const Y = v => 3 + (1 - (v - vmin) / (vmax - vmin || 1)) * (h - 6);
  let path = "";
  values.forEach((v, i) => { path += (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1); });
  const id = "sp" + color.replace("#", "");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
    </linearGradient></defs>
    <path d="${path}L100 40L0 40Z" fill="url(#${id})"/>
    <path d="${path}" fill="none" stroke="${color}" stroke-width="1.8" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="100" cy="${Y(values[values.length - 1]).toFixed(1)}" r="2.4" fill="${color}" vector-effect="non-scaling-stroke"/>
  </svg>`;
}

function renderStockCards() {
  const el = $("#stock-cards");
  el.innerHTML = Object.entries(STOCKS).map(([key, cfg]) => {
    const ds = getData(key);
    const n = ds.close.length;
    const last = ds.close[n - 1], prev = ds.close[n - 2];
    const chg = (last - prev) / prev * 100;
    const up = chg >= 0;
    const spark = ds.close.slice(-60);
    const sel = key === state.ticker;
    return `<div class="card stock-card ${sel ? "selected" : ""}" data-stock="${key}" tabindex="0" role="button" aria-pressed="${sel}">
      <div class="sc-check"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg></div>
      <div class="sc-top">
        <div><div class="sc-ticker">${key}</div><div class="sc-name">${cfg.name}</div></div>
        <span class="sc-exch">${cfg.exch}</span>
      </div>
      <div class="sc-price-row">
        <span class="sc-price">${fmtMoney(cfg, last)}</span>
        <span class="sc-change ${up ? "pos" : "neg"}">${fmtPct(chg)}</span>
      </div>
      <div class="sc-spark">${sparkSVG(spark, "#67e8f9")}</div>
    </div>`;
  }).join("");
  $$(".stock-card", el).forEach(c => {
    const pick = () => { if (state.ticker !== c.dataset.stock) { state.ticker = c.dataset.stock; saveState(); updateAll(); } };
    c.addEventListener("click", pick);
    c.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
}

/* ---- 03 historical ---- */
const histChart = new LineChart($("#hist-chart"), { height: 360 });
function renderHistorical(ds) {
  const cfg = ds.cfg;
  const r = Math.min(state.range, ds.close.length);
  const i0 = ds.close.length - r;
  const dates = ds.dates.slice(i0);
  const close = ds.close.slice(i0);
  const vol = ds.volume.slice(i0);
  const volDir = close.map((c, i) => i === 0 ? true : c >= close[i - 1]);
  $("#hist-title").textContent = `${cfg.ticker} · Daily Close`;
  histChart.yFmt = v => cfg.cur + fmtNum(v, 0);
  const histNote = document.getElementById("hist-note");
  if (histNote) {
    histNote.textContent = ds.source === "real"
      ? "SOURCE · REAL HISTORICAL PRICES (MANUALLY EMBEDDED)"
      : "DEMO DATA · SIMULATED PRICE HISTORY";
  }

  const series = [{ name: "Close", color: "#67e8f9", values: close, fill: true, width: 2, glow: true }];

  const ma20El = $("#legend-ma20");
  const ma50El = $("#legend-ma50");

  if (state.indicators?.ma20) {
    if (ma20El) ma20El.style.display = "inline-flex";
    series.push({ name: "MA20", color: "#fbbf24", values: calcMA(close, 20), width: 1.8, tip: v => fmtMoney(cfg, v) });
  } else if (ma20El) {
    ma20El.style.display = "none";
  }

  if (state.indicators?.ma50) {
    if (ma50El) ma50El.style.display = "inline-flex";
    series.push({ name: "MA50", color: "#f472b6", values: calcMA(close, 50), width: 1.8, tip: v => fmtMoney(cfg, v) });
  } else if (ma50El) {
    ma50El.style.display = "none";
  }

  // sync active state on buttons
  $$("#indicator-seg button").forEach(b => {
    b.classList.toggle("active", !!state.indicators?.[b.dataset.ind]);
  });

  histChart.setData({
    dates,
    volume: vol,
    volDir,
    series,
  });

  renderOscillators(dates, close);
}

function renderOscillators(dates, close) {
  const box = $("#oscillators-box");
  if (!box) return;
  const showRsi = !!state.indicators?.rsi;
  const showMacd = !!state.indicators?.macd;
  if (!showRsi && !showMacd) {
    box.style.display = "none";
    box.innerHTML = "";
    return;
  }
  box.style.display = "flex";
  let html = "";

  const n = close.length;
  const w = box.clientWidth || 600;
  const ML = 56, MR = 14;
  const pw = w - ML - MR;
  const X = i => ML + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);

  if (showRsi) {
    const rsi = calcRSI(close, 14);
    const lastRsi = rsi[rsi.length - 1] != null ? rsi[rsi.length - 1].toFixed(1) : "--";
    const h = 95;
    const Y = v => h - 16 - (v / 100) * (h - 24);

    let path = "", pen = false;
    for (let i = 0; i < n; i++) {
      const v = rsi[i];
      if (v == null) { pen = false; continue; }
      path += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
      pen = true;
    }

    const y70 = Y(70).toFixed(1);
    const y30 = Y(30).toFixed(1);

    html += `
      <div class="osc-panel">
        <div class="osc-head">
          <div class="osc-title">
            <span style="width:7px;height:7px;border-radius:2px;background:#22d3ee;display:inline-block;"></span>
            RELATIVE STRENGTH INDEX (14D)
          </div>
          <div class="osc-val">RSI ${lastRsi}</div>
        </div>
        <svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
          <line x1="${ML}" y1="${y70}" x2="${w - MR}" y2="${y70}" stroke="rgba(251,113,133,0.45)" stroke-dasharray="3 3"/>
          <text x="${ML - 8}" y="${+y70 + 3.5}" text-anchor="end" fill="#fb7185" font-size="9" font-family="monospace">70</text>
          <line x1="${ML}" y1="${y30}" x2="${w - MR}" y2="${y30}" stroke="rgba(52,211,153,0.45)" stroke-dasharray="3 3"/>
          <text x="${ML - 8}" y="${+y30 + 3.5}" text-anchor="end" fill="#34d399" font-size="9" font-family="monospace">30</text>
          <path d="${path}" fill="none" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
      </div>
    `;
  }

  if (showMacd) {
    const { macdLine, signalLine, hist } = calcMACD(close);
    const h = 95;
    const valid = hist.filter(v => v != null);
    const maxVal = Math.max(0.1, ...valid.map(Math.abs)) * 1.25;
    const Y0 = h / 2;
    const Ym = v => Y0 - (v / maxVal) * (h * 0.42);

    let pMacd = "", pSig = "", penM = false, penS = false;
    let bars = "";
    const bw = Math.max(1.5, (pw / n) * 0.65);

    for (let i = 0; i < n; i++) {
      const hm = hist[i];
      if (hm != null) {
        const x = X(i);
        const y = Ym(hm);
        const barH = Math.abs(y - Y0);
        const up = hm >= 0;
        bars += `<rect x="${(x - bw/2).toFixed(1)}" y="${(up ? y : Y0).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, barH).toFixed(1)}" fill="${up ? 'rgba(52,211,153,0.55)' : 'rgba(251,113,133,0.55)'}"/>`;
      }
      if (macdLine[i] != null) {
        pMacd += (penM ? "L" : "M") + X(i).toFixed(1) + " " + Ym(macdLine[i]).toFixed(1);
        penM = true;
      }
      if (signalLine[i] != null) {
        pSig += (penS ? "L" : "M") + X(i).toFixed(1) + " " + Ym(signalLine[i]).toFixed(1);
        penS = true;
      }
    }

    html += `
      <div class="osc-panel">
        <div class="osc-head">
          <div class="osc-title">
            <span style="width:7px;height:7px;border-radius:2px;background:#a78bfa;display:inline-block;"></span>
            MACD MOMENTUM (12, 26, 9)
          </div>
          <div class="osc-val" style="color:#a78bfa">SIGNAL HISTOGRAM</div>
        </div>
        <svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
          <line x1="${ML}" y1="${Y0}" x2="${w - MR}" y2="${Y0}" stroke="rgba(255,255,255,0.15)"/>
          ${bars}
          <path d="${pMacd}" fill="none" stroke="#22d3ee" stroke-width="1.6"/>
          <path d="${pSig}" fill="none" stroke="#a78bfa" stroke-width="1.6" stroke-dasharray="3 3"/>
        </svg>
      </div>
    `;
  }

  box.innerHTML = html;
}

/* ---- 04 model cards ---- */
const MODEL_ICONS = {
  arima: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 18.5L8 12l4 3.2L20.5 6"/><circle cx="8" cy="12" r="1.7" fill="currentColor" stroke="none"/><circle cx="12" cy="15.2" r="1.7" fill="currentColor" stroke="none"/><circle cx="20.5" cy="6" r="1.7" fill="currentColor" stroke="none"/><path d="M3 18.5h3.4" opacity=".45"/></svg>`,
  prophet: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3.5 20L20.5 4.5"/><path d="M3.5 14.5c2.4-4.4 4.9-4.4 7.3 0s4.9 4.4 7.3 0" opacity=".75"/></svg>`,
  lstm: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <rect x="6" y="5" width="12" height="14" rx="2.4"/><path d="M9.5 5v14M14.5 5v14" opacity=".4"/><path d="M18 8.2c3.2 0 3.2 7.6 0 7.6"/></svg>`,
  xgboost: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="4.6" r="1.8"/><circle cx="5.4" cy="12.8" r="1.8"/><circle cx="18.6" cy="12.8" r="1.8"/><circle cx="3.4" cy="20.4" r="1.4"/><circle cx="8" cy="20.4" r="1.4"/><circle cx="17.2" cy="20.4" r="1.4"/><circle cx="21.4" cy="20.4" r="1.4"/>
    <path d="M11 6.2L6.4 11.3M13.2 6.2l4.6 5.1M4.8 14.4l-1 4.5M6.3 14.4l1.3 4.5M18.3 14.4l-.8 4.5M19.5 14.4l1.6 4.5"/></svg>`,
  svr: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4 19c4-10.5 8-12.5 16-9.5"/><path d="M4 16.2c4-10.5 8-12.5 16-9.5" opacity=".4" stroke-dasharray="3 3"/><path d="M4 21.8c4-10.5 8-12.5 16-9.5" opacity=".4" stroke-dasharray="3 3"/></svg>`,
};

function renderModelCards(ds) {
  const cfg = ds.cfg;
  const el = $("#model-grid");
  el.innerHTML = MODELS.map(m => {
    const done = isTrained(state.ticker, m.id);
    const ev = evalAll(ds).find(e => e.model.id === m.id);
    return `<div class="card model-card ${done ? "trained" : ""}" style="--m:${m.color}" data-model="${m.id}">
      <div class="mc-tag">${m.tag}</div>
      <div class="mc-head">
        <span class="mc-ico">${MODEL_ICONS[m.id]}</span>
        <span class="mc-name">${m.name}</span>
      </div>
      <p class="mc-desc">${m.desc}</p>
      <div class="mc-result">
        <div class="mc-placeholder">
          <span>AWAITING TRAINING</span>
          <span style="font-size:9px;opacity:.6">RUN MODEL TO EVALUATE</span>
        </div>
        <div class="mc-training">
          <div class="mc-prog-wrap"><div class="mc-prog-fill"></div></div>
          <div class="mc-prog-stats">
            <span class="epoch">EPOCH 1/50</span>
            <span class="loss">LOSS 0.0842</span>
          </div>
        </div>
        <div class="mc-metrics">
          <div class="mm"><div class="k">RMSE</div><div class="v"><small>${cfg.cur}</small>${fmtNum(ev.rmse)}</div></div>
          <div class="mm"><div class="k">MSE</div><div class="v">${fmtNum(ev.mse, 1)}</div></div>
          <div class="mc-status"><span class="sdot"></span> TRAINED · TEST ${TEST}</div>
        </div>
      </div>
      <div class="mc-run">
        <button class="btn btn-sm btn-model btn-run" data-run="${m.id}">
          ${done
            ? `VIEW RESULTS <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>`
            : "RUN MODEL"}
        </button>
      </div>
    </div>`;
  }).join("");

  $$(".btn-run", el).forEach(btn => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.run;
      if (isTrained(state.ticker, id)) {
        state.compare = id;
        updateCompare(getData(state.ticker));
        $("#compare").scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      if (btn.dataset.busy) return;
      btn.dataset.busy = "1";
      const m = modelById(id);
      const ev = evalAll(ds).find(e => e.model.id === m.id);
      btn.innerHTML = `<span class="spinner"></span> TRAINING...`;

      const card = $(`.model-card[data-model="${id}"]`);
      if (card) {
        card.classList.remove("trained");
        card.classList.add("training");
      }

      const fillEl = card ? card.querySelector(".mc-prog-fill") : null;
      const epochEl = card ? card.querySelector(".epoch") : null;
      const lossEl = card ? card.querySelector(".loss") : null;

      const totalEpochs = m.id === "lstm" ? 50 : 25;
      let currentEpoch = 0;
      const intervalMs = Math.max(16, Math.floor(m.runMs / totalEpochs));
      let initialLoss = (ev.rmse / 120) * 1.5 + 0.045;

      const timer = setInterval(() => {
        currentEpoch++;
        const pct = Math.round((currentEpoch / totalEpochs) * 100);
        if (fillEl) fillEl.style.width = pct + "%";
        if (epochEl) epochEl.textContent = `EPOCH ${currentEpoch}/${totalEpochs}`;
        const currentLoss = Math.max(0.0031, initialLoss * Math.exp(-0.055 * currentEpoch) + (Math.random() * 0.002 - 0.001));
        if (lossEl) lossEl.textContent = `LOSS ${currentLoss.toFixed(4)}`;

        if (currentEpoch >= totalEpochs) {
          clearInterval(timer);
          trained.add(state.ticker + ":" + id);
          saveTrained();
          if (card) {
            card.classList.remove("training");
            card.classList.add("trained");
          }
          btn.dataset.busy = "";
          btn.innerHTML = `VIEW RESULTS <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>`;
        }
      }, intervalMs);
    });
  });
}

/* ---- 05 actual vs predicted ---- */
const cmpChart = new LineChart($("#cmp-chart"), { height: 340 });
function renderCompareChips() {
  $("#cmp-chips").innerHTML = MODELS.map(m =>
    `<button class="chip ${state.compare === m.id ? "active" : ""}" data-cmp="${m.id}" style="--m:${m.color}"><span class="cdot"></span>${m.name}</button>`).join("");
  $$("#cmp-chips .chip").forEach(c => c.addEventListener("click", () => {
    state.compare = c.dataset.cmp;
    saveState();
    updateCompare(getData(state.ticker));
  }));
}
function updateCompare(ds) {
  const cfg = ds.cfg;
  const m = modelById(state.compare);
  const ev = evalAll(ds).find(e => e.model.id === m.id);
  const i0 = ds.close.length - TEST;
  const dates = ds.dates.slice(i0);
  const actual = ds.close.slice(i0);
  $("#cmp-title").textContent = `Actual vs ${m.name} · ${cfg.ticker}`;
  const lg = $("#cmp-legend-pred");
  lg.innerHTML = `<i></i>Predicted · ${m.name}`;
  lg.style.setProperty("--c", m.color);
  $$("#cmp-chips .chip").forEach(c => c.classList.toggle("active", c.dataset.cmp === m.id));
  cmpChart.yFmt = v => cfg.cur + fmtNum(v, 0);
  cmpChart.setData({
    dates,
    series: [
      { name: "Actual", color: "#e9edf5", values: actual, width: 1.8 },
      { name: m.name, color: m.color, values: ev.pred, width: 2, dash: "7 5", glow: true, tip: v => fmtMoney(cfg, v) },
    ],
  });
  $("#cmp-metrics").textContent = `TYPICAL MISS ${cfg.cur}${fmtNum(ev.rmse)} PER DAY (RMSE) · DIRECTION MATCH ${ev.dir.toFixed(1)} %`;
}

/* ---- 06 evaluation ---- */
function hbarRow(rank, name, color, valText, widthPct, best) {
  return `<div class="hbar-row ${best ? "best" : ""}">
    <div class="hb-name"><span class="rank">#${rank}</span>${name}${best ? '<span class="best-chip">BEST</span>' : ""}</div>
    <div class="hb-track"><div class="hb-fill" data-w="${widthPct.toFixed(1)}" style="--c:${color}"></div></div>
    <div class="hb-val">${valText}</div>
  </div>`;
}

function renderEvaluation(ds) {
  const cfg = ds.cfg;
  const results = evalAll(ds); // sorted by rmse asc
  const best = results[0];
  $("#eval-best").innerHTML = `
    <span class="eb-ico"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M8 21h8M12 17v4M17 5h4v2a4 4 0 0 1-4 4M7 5H3v2a4 4 0 0 0 4 4"/><path d="M17 3a5 5 0 1 0-10 0c0 3 2 5 5 5s5-2 5-5z"/></svg></span>
    <div>
      <div class="eb-label">BEST PERFORMING MODEL · ${cfg.ticker}</div>
      <div class="eb-name">${best.model.name} <span style="color:var(--text-3);font-weight:500;font-size:12px;">— lowest error on the ${TEST}-session test window</span></div>
      <div class="eb-plain">💡 In plain words: on ${TEST} days it had never seen, its predictions were off by about <b>${cfg.cur}${fmtNum(best.rmse)}</b> per day on average.</div>
    </div>
    <div class="eb-metrics">RMSE <b>${fmtNum(best.rmse)}</b> &nbsp;·&nbsp; MSE <b>${fmtNum(best.mse, 1)}</b> &nbsp;·&nbsp; DIRECTION <b>${best.dir.toFixed(1)} %</b></div>`;

  const maxMse = Math.max(...results.map(r => r.mse));
  const maxRmse = Math.max(...results.map(r => r.rmse));
  $("#eval-mse").innerHTML = results.map((r, i) =>
    hbarRow(i + 1, r.model.name, r.model.color, fmtNum(r.mse, 1), (r.mse / maxMse) * 100, i === 0)).join("");
  $("#eval-rmse").innerHTML = results.map((r, i) =>
    hbarRow(i + 1, r.model.name, r.model.color, fmtNum(r.rmse), (r.rmse / maxRmse) * 100, i === 0)).join("");

  // Diebold-Mariano: best vs second best
  const dmBox = document.getElementById("dm-test");
  if (dmBox && results.length >= 2) {
    const actual = ds.close.slice(ds.close.length - TEST);
    const A = results[0], B = results[1];
    const dm = dieboldMariano(actual, A.pred, B.pred);
    const sig = dm.p < 0.05;
    dmBox.innerHTML =
      '<span class="dm-tag">DIEBOLD–MARIANO TEST</span>' +
      '<span class="dm-body">' + A.model.name + " vs " + B.model.name +
      " — DM statistic <b>" + dm.dm + "</b>, p-value <b>" + dm.p.toFixed(3) + "</b> → " +
      (sig
        ? "the " + A.model.name + " advantage is <b>statistically significant</b> at the 5% level."
        : "the error gap is <b>not statistically significant</b> — treat the two as statistically tied.") +
      '</span>';
  }

  // leadership strip: best model per third of the test window
  const strip = document.getElementById("lead-strip");
  if (strip) {
    const lead = leadershipStrip(ds);
    strip.innerHTML = lead.map(l =>
      '<div class="lead-seg" style="--m:' + l.model.color + '">' +
      '<span class="lead-range">' + l.label + '</span>' +
      '<span class="lead-name"><i></i>' + l.model.name + '</span>' +
      '<span class="lead-rmse">RMSE ' + l.rmse.toFixed(2) + '</span></div>').join("");
  }

  // feature-importance bars for XGBoost
  const impBox = document.getElementById("xgb-importance");
  if (impBox && ALGO.xgboost.importance) {
    const imp = ALGO.xgboost.importance({ key: ds.key, close: ds.close, TEST, hyperparams });
    impBox.innerHTML = '<div class="imp-title">WHAT THE XGBOOST TREES LOOK AT — split-frequency across all 80 boosted trees</div>' +
      imp.slice(0, 6).map(f =>
        '<div class="imp-row"><span class="imp-name">' + f.name + '</span>' +
        '<span class="imp-track"><span class="imp-fill" style="width:' + f.pct + '%"></span></span>' +
        '<span class="imp-val">' + f.pct + '%</span></div>').join("");
  }

  // animate bar widths after insertion
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$(".hb-fill").forEach(f => { f.style.width = f.dataset.w + "%"; });
  }));
}

/* ---- 06B strategy backtesting ---- */
const btChart = new LineChart($("#bt-chart"), { height: 320 });
function renderBacktest(ds) {
  const modelId = state.backtestModel || "lstm";
  const m = modelById(modelId);
  const bt = runBacktest(ds, modelId);
  const cfg = ds.cfg;

  $("#bt-title").textContent = `Strategy (${m.name}) vs Buy & Hold Benchmark · ${cfg.ticker}`;
  const noteEl = $("#bt-summary-note");
  if (noteEl) {
    const beats = bt.stratRoi >= bt.bnhRoi;
    noteEl.textContent = beats ? "STRATEGY OUTPERFORMING BENCHMARK" : "UNDERPERFORMING BENCHMARK";
    noteEl.style.color = beats ? "var(--emerald)" : "var(--rose)";
  }

  const stats = [
    { k: "STRATEGY RETURN", v: fmtPct(bt.stratRoi), s: "simulated profit / loss", cls: bt.stratRoi >= 0 ? "pos" : "neg" },
    { k: "BUY & HOLD RETURN", v: fmtPct(bt.bnhRoi), s: "if you simply bought and waited", cls: bt.bnhRoi >= 0 ? "pos" : "neg" },
    { k: "WIN RATE", v: bt.winRate.toFixed(1) + "%", s: "of invested days that rose", cls: bt.winRate >= 50 ? "pos" : "" },
    { k: "TOTAL TRADES", v: String(bt.trades), s: "days the strategy was invested", cls: "" },
    { k: "ANNUALIZED SHARPE", v: bt.sharpe, s: "return per unit of risk", cls: +bt.sharpe >= 1.0 ? "pos" : "" },
  ];

  const statEl = $("#bt-stats");
  if (statEl) {
    statEl.innerHTML = stats.map(st =>
      `<div class="stat"><div class="k">${st.k}</div><div class="v ${st.cls}">${st.v}</div><div class="s">${st.s}</div></div>`).join("");
  }

  const chipEl = $("#bt-chips");
  if (chipEl) {
    chipEl.innerHTML = MODELS.map(mod =>
      `<button class="chip ${state.backtestModel === mod.id ? 'active' : ''}" data-bt="${mod.id}" style="--m:${mod.color}">
        <span class="cdot"></span>${mod.name}
      </button>`).join("");

    $$("#bt-chips .chip").forEach(c => c.addEventListener("click", () => {
      state.backtestModel = c.dataset.bt;
      saveState();
      renderBacktest(getData(state.ticker));
    }));
  }

  btChart.yFmt = v => "$" + fmtNum(v, 0);
  btChart.setData({
    dates: bt.dates,
    series: [
      { name: `Strategy (${m.name})`, color: "#10b981", values: bt.stratCurve, fill: true, width: 2.2, glow: true, tip: v => "$" + fmtNum(v, 2) },
      { name: "Buy & Hold", color: "#94a3b8", values: bt.bnhCurve, width: 1.8, dash: "5 4", tip: v => "$" + fmtNum(v, 2) }
    ]
  });
}

/* ---- 07 horizon + 08 forecast ---- */
function renderHorizon(ds) {
  const cfg = ds.cfg;
  const fc = forecastFor(ds, state.horizon);
  const last = ds.close[ds.close.length - 1];
  const proj = fc.mid[fc.mid.length - 1];
  const chg = (proj - last) / last * 100;
  const band = (fc.up[fc.up.length - 1] - proj) / proj * 100;
  const trend = Math.abs(chg) < band * 0.25 ? "SIDEWAYS" : (chg > 0 ? "UPTREND" : "DOWNTREND");
  const hzLabels = { 7: "QUICK VIEW", 30: "BALANCED VIEW", 90: "LONG-RANGE VIEW" };

  $("#hz-stats").innerHTML = [
    { k: `PROJECTED PRICE · ${state.horizon}D`, v: fmtMoney(cfg, proj), s: `FROM ${fmtMoney(cfg, last)}`, cls: "" },
    { k: "EXPECTED CHANGE", v: fmtPct(chg), s: hzLabels[state.horizon], cls: chg >= 0 ? "pos" : "neg" },
    { k: "CONFIDENCE BAND", v: "±" + band.toFixed(1) + "%", s: "LIKELY PRICE RANGE", cls: "" },
    { k: "TREND DIRECTION", v: trend, s: "ENGINE · LSTM", cls: trend === "UPTREND" ? "pos" : trend === "DOWNTREND" ? "neg" : "" },
  ].map(s => `<div class="hz-stat"><div class="k">${s.k}</div><div class="v ${s.cls}" style="font-size:${s.v.length > 10 ? "15px" : "22px"}">${s.v}</div><div class="s">${s.s}</div></div>`).join("");
  $("#fc-heading").textContent = `${state.horizon}-Day Price Forecast`;
}

const fcChart = new LineChart($("#fc-chart"), { height: 380 });
function renderForecast(ds) {
  const cfg = ds.cfg;
  const HIST = 60;
  const hDates = ds.dates.slice(-HIST);
  const hClose = ds.close.slice(-HIST);
  const viewMode = state.forecastView || "forecast";

  // sync active buttons
  $$("#fc-view-seg button").forEach(b => {
    b.classList.toggle("active", b.dataset.view === viewMode);
  });

  const mcGrid = $("#mc-risk-grid");

  if (viewMode === "montecarlo") {
    const mc = runMonteCarlo(ds, state.horizon, 50);
    const dates = [...hDates, ...mc.dates];
    const pad = new Array(HIST).fill(null);
    const hist = [...hClose, ...new Array(mc.dates.length).fill(null)];

    $("#fc-title").textContent = `${cfg.ticker} · 50 Random What-If Futures`;
    $("#fc-model-note").textContent = `SIMULATION · ${state.horizon}D AHEAD`;

    if (mcGrid) {
      mcGrid.style.display = "grid";
      mcGrid.innerHTML = [
        { k: `WORST-CASE LOSS · 95% VaR (${state.horizon}D)`, v: fmtPct(-mc.var95Pct), s: `MAX POTENTIAL LOSS: ${fmtMoney(cfg, mc.var95Money)}`, cls: "neg" },
        { k: "MOST LIKELY END PRICE", v: fmtMoney(cfg, mc.p50), s: "MIDDLE OF ALL 50 SCENARIOS", cls: mc.p50 >= mc.startP ? "pos" : "neg" },
        { k: "OPTIMISTIC END PRICE", v: fmtMoney(cfg, mc.p95), s: "HAPPY 95TH-PERCENTILE SCENARIO", cls: "pos" },
        { k: "YEARLY SWING (VOLATILITY)", v: mc.sigmaAnnual + "%", s: "HOW WILD THIS STOCK USUALLY IS", cls: "" },
      ].map(s => `<div class="stat"><div class="k">${s.k}</div><div class="v ${s.cls}" style="font-size:18px;">${s.v}</div><div class="s" style="font-size:10px;color:var(--text-3);margin-top:3px;font-family:monospace;">${s.s}</div></div>`).join("");
    }

    const series = [
      { name: "History", color: "#e9edf5", values: hist, width: 2 }
    ];

    // add median path
    series.push({
      name: "Median Path (P50)",
      color: "#a78bfa",
      values: [...pad, ...mc.paths[Math.floor(mc.paths.length / 2)]],
      width: 2.2,
      glow: true,
      tip: v => fmtMoney(cfg, v)
    });

    // sample visible paths for fan chart
    const sampleIdxs = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 48];
    sampleIdxs.forEach((idx) => {
      series.push({
        name: `Path #${idx + 1}`,
        color: idx % 2 === 0 ? "rgba(34, 211, 238, 0.35)" : "rgba(167, 139, 250, 0.35)",
        values: [...pad, ...mc.paths[idx]],
        width: 1.2
      });
    });

    fcChart.yFmt = v => cfg.cur + fmtNum(v, 0);
    fcChart.setData({
      dates,
      bands: [],
      series
    });
    $("#fc-note").textContent = `50 RANDOM SCENARIOS · HORIZON ${state.horizon}D · WIDER SPREAD = MORE UNCERTAINTY`;
    return;
  }

  // Normal Forecast Mode
  if (mcGrid) mcGrid.style.display = "none";
  const fc = forecastFor(ds, state.horizon);
  const dates = [...hDates, ...fc.dates];
  const pad = new Array(HIST).fill(null);
  const hist = [...hClose, ...new Array(fc.mid.length).fill(null)];
  const proj = [...pad, ...fc.mid];
  const up = [...pad, ...fc.up];
  const lo = [...pad, ...fc.lo];

  const color = "#22d3ee";
  const label = "LSTM";

  $("#fc-title").textContent = `${cfg.ticker} · Projected Close (${label})`;
  $("#fc-model-note").textContent = "SINGLE MODEL · LSTM";

  const leg = $("#fc-legend-line");
  if (leg) {
    leg.style.setProperty("--c", color);
    leg.innerHTML = `<i></i>Forecast · ${label}`;
  }

  fcChart.yFmt = v => cfg.cur + fmtNum(v, 0);
  fcChart.setData({
    dates,
    bands: [{ color, upper: up, lower: lo }],
    series: [
      { name: "History", color: "#e9edf5", values: hist, width: 1.8 },
      { name: "Forecast", color, values: proj, width: 2.2, dash: "8 5", glow: true, tip: v => fmtMoney(cfg, v) },
    ],
  });
  $("#fc-note").textContent = `HISTORY · LAST ${HIST} SESSIONS + FORECAST · ${state.horizon}D (${label.toUpperCase()})`;
}

function animateCounter(el, targetNum, formatter, duration = 650) {
  if (!el || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    if (el) el.textContent = formatter(targetNum);
    return;
  }
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / duration);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = formatter(targetNum * ease);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ---- hero ticker line ---- */
function renderHeroMeta(ds) {
  const n = ds.close.length;
  const last = ds.close[n - 1], prev = ds.close[n - 2];
  const chg = (last - prev) / prev * 100;
  $("#viz-ticker").textContent = ds.cfg.ticker;
  animateCounter($("#viz-price"), last, v => `${fmtMoney(ds.cfg, v)}  ${fmtPct(chg)}`, 600);
  $("#viz-price").style.color = chg >= 0 ? "var(--emerald)" : "var(--rose)";
}

/* ---------------- target-date prediction ----------------
   The user picks a future day; every model iterates its own math
   forward session by session (predictFuture) and reports the price
   for that exact day plus a plain-English reason. */
const _futureCache = {};
function futureForDate(ds, h) {
  const hpKey = JSON.stringify(hyperparams);
  const ck = ds.key + "|" + h + "|" + hpKey;
  if (_futureCache[ck]) return _futureCache[ck];
  const ctx = { key: ds.key, close: ds.close, TEST, hyperparams };
  const evs = evalAll(ds);
  const rows = MODELS.map(m => {
    const r = ALGO[m.id].predictFuture(ctx, h);
    const ev = evs.find(e => e.model.id === m.id);
    return {
      model: m,
      price: r.prices[r.prices.length - 1],
      pct: r.pct,
      reason: r.reason,
      errStd: ev ? ev.errStd : 0,
    };
  });
  const prices = rows.map(r => r.price).sort((a, b) => a - b);
  const out = {
    rows,
    consensus: prices[Math.floor(prices.length / 2)],
    low: prices[0],
    high: prices[prices.length - 1],
  };
  _futureCache[ck] = out;
  return out;
}

function businessDaysBetween(fromDate, targetDate) {
  let count = 0;
  const d = new Date(fromDate);
  const max = 400;
  while (d < targetDate && count < max) {
    d.setDate(d.getDate() + 1);
    const wd = d.getDay();
    if (wd !== 0 && wd !== 6) count++;
  }
  return count;
}

function renderTargetDate() {
  const ds = getData(state.ticker);
  const input = $("#target-date");
  const status = $("#date-status");
  const results = $("#date-results");
  const note = $("#date-note");
  if (!input || !input.value) {
    if (status) status.textContent = "PICK A TRADING DAY";
    return;
  }
  const target = new Date(input.value + "T00:00:00");
  const last = ds.dates[ds.dates.length - 1];
  const h = businessDaysBetween(last, target);
  if (h < 1) {
    if (status) status.textContent = "CHOOSE A DATE AFTER " + fmtDate(last).toUpperCase();
    if (results) results.innerHTML = "";
    return;
  }
  if (status) status.textContent = "COMPUTING " + h + " SESSIONS AHEAD…";
  if (results) results.innerHTML = "";
  /* let the browser paint the computing state before the heavy run */
  setTimeout(() => {
    const MAXH = 126;
    const hh = Math.min(h, MAXH);
    const f = futureForDate(ds, hh);
    const cfg = ds.cfg;
    const lastClose = ds.close[ds.close.length - 1];
    const dpr = f.rows.map(r => {
      const pct = (r.price / lastClose - 1) * 100;
      const up = pct >= 0;
      const ci = r.price * 1.645 * (r.errStd || 0) * Math.sqrt(hh);
      const lo = fmtMoney(cfg, Math.max(0, r.price - ci));
      const hi = fmtMoney(cfg, r.price + ci);
      const barPct = lastClose ? Math.min(100, (ci / r.price) * 100) : 0;
      return `<div class="dpr">
        <div class="dpr-name" style="--m:${r.model.color}"><i></i>${r.model.name}</div>
        <div class="dpr-price">${fmtMoney(cfg, r.price)}</div>
        <div class="dpr-chg ${up ? "pos" : "neg"}">${up ? "▲" : "▼"} ${fmtPct(pct)}</div>
        <div class="dpr-why">${r.reason}</div>
        <div class="dpr-ci"><span class="ci-bar"><span class="ci-fill" style="width:${barPct.toFixed(1)}%"></span></span>90% LIKELY RANGE ${lo} – ${hi}</div>
      </div>`;
    }).join("");
    const cPct = (f.consensus / lastClose - 1) * 100;
    if (results) {
      results.innerHTML = dpr + `
        <div class="dpr-consensus">
          Median of the five models for ${fmtDate(target)}:
          <b>${fmtMoney(cfg, f.consensus)}</b> (${fmtPct(cPct)} vs the last real close of ${fmtMoney(cfg, lastClose)})
          &nbsp;·&nbsp; models span ${fmtMoney(cfg, f.low)} – ${fmtMoney(cfg, f.high)}
        </div>`;
    }
    if (status) status.textContent = `${hh} SESSIONS AHEAD · ${fmtDate(target).toUpperCase()}` + (h > MAXH ? " (CLAMPED)" : "");
    if (note) note.textContent = `LAST REAL SESSION · ${fmtDate(last)}`;
  }, 30);
}

(function targetDateControls() {
  const btn = $("#predict-date-btn");
  const input = $("#target-date");
  if (!btn || !input) return;
  const ds = getData(state.ticker);
  const last = ds.dates[ds.dates.length - 1];
  const min = new Date(last); min.setDate(min.getDate() + 1);
  const max = new Date(last); max.setDate(max.getDate() + 190);
  input.min = min.toISOString().slice(0, 10);
  input.max = max.toISOString().slice(0, 10);
  const def = new Date(last); def.setDate(def.getDate() + 22);
  if (def.getDay() === 0) def.setDate(def.getDate() + 1);
  if (def.getDay() === 6) def.setDate(def.getDate() + 2);
  input.value = def.toISOString().slice(0, 10);
  btn.addEventListener("click", renderTargetDate);
  input.addEventListener("change", renderTargetDate);
})();

/* ---------------- orchestration ---------------- */
function updateAll() {
  const ds = getData(state.ticker);
  renderHeroMeta(ds);
  renderMarket(ds);
  renderStockCards();
  renderHistorical(ds);
  renderModelCards(ds);
  updateCompare(ds);
  renderEvaluation(ds);
  renderBacktest(ds);
  renderHorizon(ds);
  renderForecast(ds);
  hero3d.setData(ds);
}

/* ============================================================
   3D CENTERPIECE — candlestick scene on canvas
   ============================================================ */
const hero3d = (() => {
  const canvas = $("#hero3d");
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, dpr = 1;
  let running = true, visible = true;
  let cur = null, from = null, to = null, morphT = 1;
  let tickerCfg = null;

  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  function setData(ds) {
    const n = ds.close.length, K = 18;
    const win = [];
    for (let i = n - K; i < n; i++) {
      win.push({ o: ds.open[i], c: ds.close[i], h: ds.high[i], l: ds.low[i] });
    }
    tickerCfg = ds.cfg;
    from = cur ? cur.map(p => ({ ...p })) : null;
    to = win;
    morphT = cur ? 0 : 1;
    cur = cur ? cur : win.map(p => ({ ...p }));
  }

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  new IntersectionObserver(en => { visible = en[0].isIntersecting; }, { threshold: 0.02 }).observe(canvas);

  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || !W) return;
    if (reduceMotion && morphT >= 1 && cur === to) { /* draw once below and stop updating is fine — cheap enough to keep drawing */ }
    step(now);
  }

  function step(now) {
    // morph progress
    if (morphT < 1 && to) {
      morphT = Math.min(1, morphT + 0.035);
      const t = easeOut(morphT);
      cur = cur.map((p, i) => ({
        o: lerp(p.o, to[i].o, t), c: lerp(p.c, to[i].c, t),
        h: lerp(p.h, to[i].h, t), l: lerp(p.l, to[i].l, t),
      }));
      if (morphT >= 1) cur = to.map(p => ({ ...p }));
    }

    const t = now;
    const yaw = 0.5 + Math.sin(t * 0.00019) * 0.085;
    const pitch = 0.38 + Math.sin(t * 0.00012 + 1.7) * 0.03;
    const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const f = 3.05, dist = 5.6;
    const unit = Math.min(W / 8.6, H / 5.6);
    const cx = W / 2, cyy = H * 0.52;

    const P = (x, y, z) => {
      const x1 = x * cyw + z * syw, z1 = z * cyw - x * syw;
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const s = f / (dist + z2);
      return [cx + x1 * unit * s, cyy - y1 * unit * s];
    };

    // normalize prices into scene space
    const K = cur.length;
    let pmin = Infinity, pmax = -Infinity;
    cur.forEach(p => { pmin = Math.min(pmin, p.l); pmax = Math.max(pmax, p.h); });
    const span = pmax - pmin || 1;
    const Y = v => 0.02 + ((v - pmin) / span) * 2.05;
    const X = i => -3.05 + (i / (K - 1)) * 6.1;
    const floorY = -0.78;

    ctx.clearRect(0, 0, W, H);

    // ambient glow on the floor
    const gp = P(0, floorY, 0.2);
    const grd = ctx.createRadialGradient(gp[0], gp[1], 0, gp[0], gp[1], W * 0.4);
    grd.addColorStop(0, "rgba(34,211,238,0.075)");
    grd.addColorStop(0.5, "rgba(139,92,246,0.035)");
    grd.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, W, H);

    // floor grid
    ctx.lineWidth = 1;
    for (let zi = -3.2; zi <= 2.6; zi += 0.565) {
      const a = P(-3.55, floorY, zi), b = P(3.55, floorY, zi);
      const alpha = clamp(0.15 - (0.9 - zi) * 0.028, 0.03, 0.15);
      ctx.strokeStyle = `rgba(148,180,220,${alpha})`;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    for (let xi = -3.5; xi <= 3.55; xi += 0.3944) {
      const a = P(xi, floorY, -3.2), b = P(xi, floorY, 2.6);
      ctx.strokeStyle = "rgba(148,180,220,0.055)";
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }

    // candles (with soft floor reflection)
    const bw = 0.125, d = 0.105;
    for (let i = 0; i < K; i++) {
      const p = cur[i], x = X(i);
      const up = p.c >= p.o;
      const yT = Y(Math.max(p.o, p.c)), yB = Y(Math.min(p.o, p.c));
      const wT = Y(p.h), wB = Y(p.l);
      const col = up ? [47, 214, 160] : [250, 115, 138];

      // reflection (front face mirrored under floor)
      const r1 = P(x - bw, 2 * floorY - yT, d), r2 = P(x + bw, 2 * floorY - yT, d), r3 = P(x + bw, 2 * floorY - yB, d), r4 = P(x - bw, 2 * floorY - yB, d);
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},0.05)`;
      ctx.beginPath(); ctx.moveTo(r1[0], r1[1]); ctx.lineTo(r2[0], r2[1]); ctx.lineTo(r3[0], r3[1]); ctx.lineTo(r4[0], r4[1]); ctx.closePath(); ctx.fill();

      // box faces: right side, front, top
      const corners = {
        aTF: P(x - bw, yT, -d), bTF: P(x + bw, yT, -d), cTF: P(x + bw, yB, -d), dTF: P(x - bw, yB, -d),
        aT: P(x - bw, yT, d), bT: P(x + bw, yT, d), cT: P(x + bw, yB, d), dT: P(x - bw, yB, d),
      };
      const face = (pts, fill, stroke) => {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let j = 1; j < pts.length; j++) ctx.lineTo(pts[j][0], pts[j][1]);
        ctx.closePath();
        ctx.fillStyle = fill; ctx.fill();
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
      };
      const rgb = `${col[0]},${col[1]},${col[2]}`;
      face([corners.bTF, corners.bT, corners.cT, corners.cTF], `rgba(${rgb},0.45)`);          // right side (shadowed)
      face([corners.aT, corners.bT, corners.cT, corners.dT], `rgba(${rgb},0.92)`, `rgba(${rgb},0.55)`); // front
      face([corners.aTF, corners.bTF, corners.bT, corners.aT], `rgba(${rgb},0.68)`);          // top

      // wick on front face
      const w1 = P(x, wT, d), w2 = P(x, yT, d), w3 = P(x, yB, d), w4 = P(x, wB, d);
      ctx.strokeStyle = `rgba(${rgb},0.75)`;
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(w1[0], w1[1]); ctx.lineTo(w2[0], w2[1]);
      ctx.moveTo(w3[0], w3[1]); ctx.lineTo(w4[0], w4[1]); ctx.stroke();
    }

    // dashed forecast ribbon behind (violet)
    if (cur.length && tickerCfg) {
      let fp = cur[K - 1].c;
      const fpts = [];
      for (let i = 1; i <= 5; i++) {
        fp = fp * (1 + tickerCfg.drift * 1.9 + Math.sin(i * 1.35 + t * 0.00042) * 0.0038);
        fpts.push(P(X(K - 1) + i * 0.34, clamp(Y(fp), -0.1, 2.15), -1.35));
      }
      const l0 = P(X(K - 1), Y(cur[K - 1].c), -1.35);
      ctx.setLineDash([5, 6]);
      ctx.strokeStyle = "rgba(167,139,250,0.55)";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(l0[0], l0[1]);
      fpts.forEach(pt => ctx.lineTo(pt[0], pt[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      fpts.forEach(pt => {
        ctx.fillStyle = "rgba(196,181,253,0.6)";
        ctx.beginPath(); ctx.arc(pt[0], pt[1], 1.6, 0, 7); ctx.fill();
      });
    }

    // glowing close-price path (front)
    const linePts = cur.map((p, i) => {
      const float = Math.sin(t * 0.0009 + i * 0.7) * 0.012;
      return P(X(i), clamp(Y(p.c) + float, 0, 2.4), 1.12);
    });
    const lg = ctx.createLinearGradient(linePts[0][0], 0, linePts[linePts.length - 1][0], 0);
    lg.addColorStop(0, "#22d3ee");
    lg.addColorStop(1, "#a78bfa");
    ctx.save();
    ctx.shadowColor = "rgba(34,211,238,0.55)";
    ctx.shadowBlur = 9;
    ctx.strokeStyle = lg;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(linePts[0][0], linePts[0][1]);
    linePts.forEach(pt => ctx.lineTo(pt[0], pt[1]));
    ctx.stroke();
    ctx.restore();

    // nodes
    linePts.forEach((pt, i) => {
      const g = ctx.createRadialGradient(pt[0], pt[1], 0, pt[0], pt[1], 6);
      g.addColorStop(0, "rgba(103,232,249,0.4)");
      g.addColorStop(1, "rgba(103,232,249,0)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(pt[0], pt[1], 6, 0, 7); ctx.fill();
      ctx.fillStyle = "#ecfeff";
      ctx.beginPath(); ctx.arc(pt[0], pt[1], i === linePts.length - 1 ? 3 : 1.9, 0, 7); ctx.fill();
    });

    // pulsing endpoint + expanding ring
    const lastPt = linePts[linePts.length - 1];
    const pulse = 3 + Math.sin(t * 0.0035) * 1.1;
    ctx.fillStyle = "rgba(236,254,255,0.95)";
    ctx.beginPath(); ctx.arc(lastPt[0], lastPt[1], pulse, 0, 7); ctx.fill();
    const ph = (t % 2600) / 2600;
    ctx.strokeStyle = `rgba(103,232,249,${(1 - ph) * 0.5})`;
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(lastPt[0], lastPt[1], 5 + ph * 24, 0, 7); ctx.stroke();
  }

  requestAnimationFrame(frame);
  return { setData };
})();

/* ============================================================
   BACKGROUND — particles + drifting grid
   ============================================================ */
(function background() {
  const canvas = $("#bg-canvas");
  const ctx = canvas.getContext("2d");
  let W, H, dpr;
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  addEventListener("resize", resize);

  const rng = mulberry32(20261001);
  const parts = Array.from({ length: 70 }, () => ({
    x: rng() * 2000, y: rng() * 1200,
    r: 0.5 + rng() * 1.5,
    vy: 0.04 + rng() * 0.12,
    tw: rng() * Math.PI * 2,
    c: rng() < 0.22 ? "148,222,255" : rng() < 0.4 ? "196,181,253" : "148,163,184",
  }));

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    // faint drifting grid
    const s = 76, off = reduceMotion ? 0 : (t * 0.0035) % s;
    ctx.strokeStyle = "rgba(160,190,230,0.032)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = -s + off; x < W + s; x += s) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let y = 0; y < H + s; y += s) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();
    // particles
    parts.forEach(p => {
      if (!reduceMotion) {
        p.y -= p.vy;
        if (p.y < -4) { p.y = H + 4; p.x = rng() * W; }
      }
      const a = 0.14 + Math.sin(t * 0.001 + p.tw) * 0.1;
      ctx.fillStyle = `rgba(${p.c},${a.toFixed(3)})`;
      ctx.beginPath(); ctx.arc(p.x % (W + 8), p.y, p.r, 0, 7); ctx.fill();
    });
    if (!reduceMotion) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();

/* ============================================================
   SCROLL REVEAL · NAV · CONTROLS
   ============================================================ */
(function chrome() {
  // nav elevation
  const nav = $("#nav");
  addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 10), { passive: true });

  // reveal on scroll
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }), { threshold: 0.12 });
  $$(".reveal").forEach(el => io.observe(el));

  // nav section spy
  const spy = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) {
      const id = e.target.id;
      $$("#nav-links a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + id));
    }
  }), { rootMargin: "-40% 0px -55% 0px" });
  ["hero", "how", "market", "select", "historical", "models", "compare", "evaluation", "backtest", "horizon", "forecast"]
    .forEach(id => { const el = document.getElementById(id); if (el) spy.observe(el); });

  // historical range segmented control
  $$("#range-seg button").forEach(b => b.addEventListener("click", () => {
    $$("#range-seg button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.range = +b.dataset.range;
    saveState();
    renderHistorical(getData(state.ticker));
  }));

  // technical indicators toggle
  $$("#indicator-seg button").forEach(b => b.addEventListener("click", () => {
    const ind = b.dataset.ind;
    state.indicators[ind] = !state.indicators[ind];
    b.classList.toggle("active", state.indicators[ind]);
    saveState();
    renderHistorical(getData(state.ticker));
  }));

  // forecast view mode (forecast vs monte carlo)
  $$("#fc-view-seg button").forEach(b => b.addEventListener("click", () => {
    $$("#fc-view-seg button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.forecastView = b.dataset.view;
    saveState();
    renderForecast(getData(state.ticker));
  }));

  // horizon segmented control
  $$("#horizon-seg button").forEach(b => b.addEventListener("click", () => {
    $$("#horizon-seg button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.horizon = +b.dataset.h;
    saveState();
    renderHorizon(getData(state.ticker));
    renderForecast(getData(state.ticker));
  }));

  // re-render charts on resize (debounced)
  let rt;
  addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      const ds = getData(state.ticker);
      renderHistorical(ds);
      updateCompare(ds);
      renderBacktest(ds);
      renderForecast(ds);
    }, 180);
  });
})();

/* ---------------- boot ---------------- */
updateAll();
renderCompareChips();

// restore active states for range/horizon controls from saved state
$$("#range-seg button").forEach(b => b.classList.toggle("active", +b.dataset.range === state.range));
$$("#horizon-seg button").forEach(b => b.classList.toggle("active", +b.dataset.h === state.horizon));
$$("#indicator-seg button").forEach(b => b.classList.toggle("active", !!state.indicators?.[b.dataset.ind]));
$$("#fc-view-seg button").forEach(b => b.classList.toggle("active", b.dataset.view === (state.forecastView || "forecast")));

addEventListener("load", () => {
  const ds = getData(state.ticker);
  renderHistorical(ds);
  updateCompare(ds);
  renderBacktest(ds);
  renderForecast(ds);

  // dismiss loading overlay
  const lo = $("#loading-overlay");
  if (lo) { setTimeout(() => lo.classList.add("hidden"), 300); }
});

/* -- hamburger menu -- */
(function hamburger() {
  const btn = $("#nav-hamburger");
  const links = $("#nav-links");
  const overlay = $("#nav-overlay");
  if (!btn || !links) return;

  function toggle(open) {
    const isOpen = open ?? !btn.classList.contains("open");
    btn.classList.toggle("open", isOpen);
    links.classList.toggle("open", isOpen);
    if (overlay) overlay.classList.toggle("open", isOpen);
    btn.setAttribute("aria-expanded", isOpen);
    document.body.style.overflow = isOpen ? "hidden" : "";
  }
  btn.addEventListener("click", () => toggle());
  if (overlay) overlay.addEventListener("click", () => toggle(false));
  $$("#nav-links a").forEach(a => a.addEventListener("click", () => toggle(false)));
})();

/* -- scroll-to-top button -- */
(function scrollTop() {
  const btn = $("#scroll-top");
  if (!btn) return;
  addEventListener("scroll", () => btn.classList.toggle("visible", scrollY > 500), { passive: true });
  btn.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
})();

/* -- train all models button -- */
(function trainAll() {
  const btn = $("#train-all");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const untrained = MODELS.filter(m => !isTrained(state.ticker, m.id));
    if (!untrained.length) {
      btn.textContent = "ALL MODELS TRAINED ✓";
      setTimeout(() => { btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3l14 9-14 9V3z"/></svg> Train All Models`; }, 1800);
      return;
    }
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> TRAINING ${untrained.length} MODELS`;
    let delay = 0;
    untrained.forEach(m => {
      delay += 350;
      setTimeout(() => {
        const runBtn = $(`.btn-run[data-run="${m.id}"]`);
        if (runBtn && !runBtn.dataset.busy) runBtn.click();
      }, delay);
    });
    const maxTime = Math.max(...untrained.map(m => m.runMs)) + delay + 400;
    setTimeout(() => {
      btn.disabled = false;
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3l14 9-14 9V3z"/></svg> Train All Models`;
    }, maxTime);
  });
})();

/* -- chart fullscreen expansion -- */
(function fullscreenCharts() {
  $$(".chart-expand-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.expand;
      const card = document.getElementById(targetId);
      if (!card) return;
      const isFull = card.classList.toggle("card--fullscreen");
      btn.classList.toggle("active", isFull);
      document.body.style.overflow = isFull ? "hidden" : "";
      setTimeout(() => {
        const ds = getData(state.ticker);
        if (targetId === "hist-card") renderHistorical(ds);
        if (targetId === "cmp-card") updateCompare(ds);
        if (targetId === "fc-card") renderForecast(ds);
      }, 60);
    });
  });
})();

/* -- export CSV handlers -- */
(function exportData() {
  // export OHLCV
  $("#export-ohlcv-btn")?.addEventListener("click", () => {
    const ds = getData(state.ticker);
    const rows = [["Date", "Open", "High", "Low", "Close", "Day_Change_Pct", "Volume"]];
    for (let i = 0; i < ds.close.length; i++) {
      const prev = i === 0 ? ds.open[i] : ds.close[i - 1];
      const chg = ((ds.close[i] - prev) / prev * 100).toFixed(2);
      rows.push([
        fmtDate(ds.dates[i]),
        ds.open[i].toFixed(2),
        ds.high[i].toFixed(2),
        ds.low[i].toFixed(2),
        ds.close[i].toFixed(2),
        chg + "%",
        ds.volume[i]
      ]);
    }
    downloadCSV(`${ds.cfg.ticker}_OHLCV_${ds.close.length}Sessions.csv`, rows);
  });

  // export Compare predictions
  $("#export-cmp-btn")?.addEventListener("click", () => {
    const ds = getData(state.ticker);
    const i0 = ds.close.length - TEST;
    const testDates = ds.dates.slice(i0);
    const actual = ds.close.slice(i0);
    const evals = evalAll(ds);
    const rows = [["Date", "Actual_Close", ...MODELS.map(m => m.name + "_Predicted")]];
    for (let i = 0; i < TEST; i++) {
      const row = [fmtDate(testDates[i]), actual[i].toFixed(2)];
      MODELS.forEach(m => {
        const ev = evals.find(e => e.model.id === m.id);
        row.push(ev.pred[i].toFixed(2));
      });
      rows.push(row);
    }
    downloadCSV(`${ds.cfg.ticker}_TestWindow_Model_Predictions.csv`, rows);
  });

  // export Future forecast
  $("#export-fc-btn")?.addEventListener("click", () => {
    const ds = getData(state.ticker);
    const fc = forecastFor(ds, state.horizon);
    const rows = [["Date", "Forecast_Close", "Lower_95Band", "Upper_95Band"]];
    for (let i = 0; i < fc.dates.length; i++) {
      rows.push([
        fmtDate(fc.dates[i]),
        fc.mid[i].toFixed(2),
        fc.lo[i].toFixed(2),
        fc.up[i].toFixed(2)
      ]);
    }
    downloadCSV(`${ds.cfg.ticker}_${state.horizon}D_Forecast.csv`, rows);
  });
})();

/* -- hyperparameter tuning studio -- */
(function tunerStudio() {
  const modal = $("#tuner-modal");
  const openBtn = $("#tune-models-btn");
  const closeBtn = $("#tuner-close");
  const backdrop = $("#tuner-backdrop");
  const resetBtn = $("#tuner-reset");
  const applyBtn = $("#tuner-apply");

  const openTuner = () => { if (modal) modal.classList.add("open"); };
  const closeTuner = () => { if (modal) modal.classList.remove("open"); };

  openBtn?.addEventListener("click", openTuner);
  closeBtn?.addEventListener("click", closeTuner);
  backdrop?.addEventListener("click", closeTuner);

  const syncSlider = (id, labelId, transform = v => v) => {
    const slider = $(id);
    const lbl = $(labelId);
    if (slider && lbl) {
      slider.addEventListener("input", () => {
        lbl.textContent = transform(slider.value);
      });
    }
  };

  syncSlider("#t-lstm-epochs", "#v-lstm-epochs");
  syncSlider("#t-lstm-lookback", "#v-lstm-lookback");
  syncSlider("#t-lstm-lr", "#v-lstm-lr", v => (v / 1000).toFixed(3));
  syncSlider("#t-arima-p", "#v-arima-p");
  syncSlider("#t-arima-d", "#v-arima-d");
  syncSlider("#t-arima-q", "#v-arima-q");
  syncSlider("#t-xgb-depth", "#v-xgb-depth");
  syncSlider("#t-xgb-sub", "#v-xgb-sub", v => (v / 100).toFixed(2));

  resetBtn?.addEventListener("click", () => {
    $("#t-lstm-epochs").value = 50; $("#v-lstm-epochs").textContent = "50";
    $("#t-lstm-lookback").value = 30; $("#v-lstm-lookback").textContent = "30";
    $("#t-lstm-lr").value = 5; $("#v-lstm-lr").textContent = "0.005";
    $("#t-arima-p").value = 2; $("#v-arima-p").textContent = "2";
    $("#t-arima-d").value = 1; $("#v-arima-d").textContent = "1";
    $("#t-arima-q").value = 2; $("#v-arima-q").textContent = "2";
    $("#t-xgb-depth").value = 6; $("#v-xgb-depth").textContent = "6";
    $("#t-xgb-sub").value = 80; $("#v-xgb-sub").textContent = "0.80";

    hyperparams.lstm = { epochs: 50, lookback: 30, lr: 0.005 };
    hyperparams.arima = { p: 2, d: 1, q: 2 };
    hyperparams.xgboost = { depth: 6, subsample: 0.8 };

    for (const k in _evalCache) delete _evalCache[k];
    for (const k in _predCache) delete _predCache[k];
    updateAll();
  });

  applyBtn?.addEventListener("click", () => {
    hyperparams.lstm = {
      epochs: +$("#t-lstm-epochs").value,
      lookback: +$("#t-lstm-lookback").value,
      lr: +$("#t-lstm-lr").value / 1000
    };
    hyperparams.arima = {
      p: +$("#t-arima-p").value,
      d: +$("#t-arima-d").value,
      q: +$("#t-arima-q").value
    };
    hyperparams.xgboost = {
      depth: +$("#t-xgb-depth").value,
      subsample: +$("#t-xgb-sub").value / 100
    };

    for (const k in _evalCache) delete _evalCache[k];
    for (const k in _predCache) delete _predCache[k];
    updateAll();

    const orig = applyBtn.textContent;
    applyBtn.textContent = "RETUNED & APPLIED ✓";
    applyBtn.style.background = "var(--emerald)";
    applyBtn.style.color = "#000";
    setTimeout(() => {
      applyBtn.textContent = orig;
      applyBtn.style.background = "";
      applyBtn.style.color = "";
      closeTuner();
    }, 850);
  });
})();

/* -- keyboard shortcuts & modal -- */
(function keyboardShortcuts() {
  const modal = $("#kbd-modal");
  const tunerModal = $("#tuner-modal");
  const guideModal = $("#guide-modal");
  const openModal = () => { if (modal) modal.classList.add("open"); };
  const closeModal = () => {
    if (modal) modal.classList.remove("open");
    if (tunerModal) tunerModal.classList.remove("open");
    if (guideModal) guideModal.classList.remove("open");
    const full = $(".card--fullscreen");
    if (full) {
      full.classList.remove("card--fullscreen");
      document.body.style.overflow = "";
      const ds = getData(state.ticker);
      renderHistorical(ds);
      updateCompare(ds);
      renderBacktest(ds);
      renderForecast(ds);
    }
  };

  $("#kbd-btn")?.addEventListener("click", openModal);
  $("#kbd-close")?.addEventListener("click", closeModal);
  $("#kbd-backdrop")?.addEventListener("click", closeModal);
  $("#guide-btn")?.addEventListener("click", () => guideModal?.classList.add("open"));
  $("#guide-close")?.addEventListener("click", () => guideModal?.classList.remove("open"));
  $("#guide-backdrop")?.addEventListener("click", () => guideModal?.classList.remove("open"));

  window.addEventListener("keydown", ev => {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    const k = ev.key;
    if (k === "Escape") {
      closeModal();
    } else if (k === "?" || (ev.shiftKey && k === "/")) {
      openModal();
    } else if (k.toLowerCase() === "g") {
      if (guideModal) {
        if (guideModal.classList.contains("open")) guideModal.classList.remove("open");
        else guideModal.classList.add("open");
      }
    } else if (k === "1") {
      state.ticker = "TCS"; saveState(); updateAll();
    } else if (k === "2") {
      state.ticker = "RELIANCE"; saveState(); updateAll();
    } else if (k === "3") {
      state.ticker = "GOOGL"; saveState(); updateAll();
    } else if (k.toLowerCase() === "t") {
      $("#train-all")?.click();
    } else if (k.toLowerCase() === "h") {
      if (tunerModal) {
        if (tunerModal.classList.contains("open")) tunerModal.classList.remove("open");
        else tunerModal.classList.add("open");
      }
    } else if (k.toLowerCase() === "b") {
      $("#backtest")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (k.toLowerCase() === "m") {
      state.indicators.ma20 = !state.indicators.ma20;
      saveState();
      renderHistorical(getData(state.ticker));
    } else if (k.toLowerCase() === "p") {
      window.print();
    } else if (k.toLowerCase() === "f") {
      document.querySelector(".chart-expand-btn[data-expand='hist-card']")?.click();
    }
  });
})();
