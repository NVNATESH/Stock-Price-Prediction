/* ============================================================
   SVR — Support Vector Regression (kernel method)
   Hand-written implementation (Drucker et al., 1997; Vapnik).

   The ε-insensitive regression

     min ½‖w‖² + C Σ (ξᵢ + ξᵢ*)   s.t. |yᵢ − w·φ(xᵢ)| ≤ ε + ξᵢ

   is solved through its dual with the dual coordinate-descent
   algorithm (Hsieh et al., LIBLINEAR-style): the coefficients
   αᵢ, αᵢ* ∈ [0, C] are optimised one coordinate at a time against
   a precomputed RBF kernel matrix K(xᵢ, xⱼ) = exp(−γ‖xᵢ−xⱼ‖²).
   The prediction function is

     ŷ(x) = Σⱼ (αⱼ − αⱼ*) K(xⱼ, x) + b

   with the intercept b set from the average of the free support
   vectors' residual errors. Features are standardised lagged
   returns; the model predicts the next return. Deterministic —
   cyclic coordinate order, no randomness.
   ============================================================ */
(function (ns) {
  "use strict";
  const U = ns.utils;

  const NF = 5;          // features: r1, r2, r3, mean5, std5
  const C = 1.0;         // box constraint (target is standardised)
  const EPS = 0.1;       // ε-tube width in standardised units
  const GAMMA = 0.35;    // RBF kernel width
  const MAX_SWEEPS = 300;

  function predict(ctx) {
    const full = ctx.close, N = full.length, T = ctx.TEST;
    const split = N - T;

    /* ---- returns and feature builder (one-step-ahead safe) ---- */
    const ret = new Array(N);
    ret[0] = 0;
    for (let i = 1; i < N; i++) ret[i] = (full[i] - full[i - 1]) / full[i - 1];
    function features(i) {
      const f = new Array(NF);
      f[0] = ret[i - 1]; f[1] = ret[i - 2]; f[2] = ret[i - 3];
      let m5 = 0;
      for (let k = 1; k <= 5; k++) m5 += ret[i - k];
      m5 /= 5;
      let v5 = 0;
      for (let k = 1; k <= 5; k++) v5 += (ret[i - k] - m5) * (ret[i - k] - m5);
      f[3] = m5;
      f[4] = Math.sqrt(v5 / 5);
      return f;
    }

    /* ---- standardise with train statistics ---- */
    const i0 = 6;
    const rows = [], y = [];
    for (let i = i0; i < split; i++) {
      rows.push(features(i));
      y.push(ret[i]);
    }
    const n = rows.length;
    const mu = new Array(NF).fill(0), sd = new Array(NF).fill(0);
    for (let f = 0; f < NF; f++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += rows[r][f];
      mu[f] = s / n;
      s = 0;
      for (let r = 0; r < n; r++) s += (rows[r][f] - mu[f]) * (rows[r][f] - mu[f]);
      sd[f] = Math.sqrt(s / n) || 1e-9;
      for (let r = 0; r < n; r++) rows[r][f] = (rows[r][f] - mu[f]) / sd[f];
    }
    let ymu = 0;
    for (let r = 0; r < n; r++) ymu += y[r];
    ymu /= n;
    let ysd = 0;
    for (let r = 0; r < n; r++) ysd += (y[r] - ymu) * (y[r] - ymu);
    ysd = Math.sqrt(ysd / n) || 1e-9;
    const yn = y.map(v => (v - ymu) / ysd);

    /* ---- precompute the RBF kernel matrix on the training rows ---- */
    const K = new Float64Array(n * n);
    for (let a = 0; a < n; a++) {
      for (let b = a; b < n; b++) {
        let d2 = 0;
        for (let f = 0; f < NF; f++) {
          const d = rows[a][f] - rows[b][f];
          d2 += d * d;
        }
        const k = Math.exp(-GAMMA * d2);
        K[a * n + b] = k; K[b * n + a] = k;
      }
    }

    /* ---- dual coordinate descent on (α, α*) ---- */
    const w = new Float64Array(n);          // wᵢ = αᵢ − αᵢ*
    const G = new Float64Array(n);          // Gᵢ = Σⱼ wⱼKᵢⱼ − yᵢ
    for (let i = 0; i < n; i++) G[i] = -yn[i];

    for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
      let maxDelta = 0;
      for (let i = 0; i < n; i++) {
        const Kii = K[i * n + i];
        /* objective along wᵢ: ½Kii·w² + B·w + ε|w|,
           B = Gᵢ − wᵢKii (gradient without the own term) */
        const B = G[i] - w[i] * Kii;
        let wNew;
        if (B > EPS) wNew = -(B - EPS) / Kii;
        else if (B < -EPS) wNew = -(B + EPS) / Kii;
        else wNew = 0;
        if (wNew > C) wNew = C; else if (wNew < -C) wNew = -C;
        const delta = wNew - w[i];
        if (delta === 0) continue;
        w[i] = wNew;
        maxDelta = Math.max(maxDelta, Math.abs(delta));
        for (let j = 0; j < n; j++) G[j] += delta * K[i * n + j];
      }
      if (maxDelta < 1e-6) break;
    }

    /* ---- one-step-ahead inference on the test window ---- */
    const pred = new Array(T);
    for (let i = split; i < N; i++) {
      const f = features(i);
      for (let k = 0; k < NF; k++) f[k] = (f[k] - mu[k]) / sd[k];
      let s = 0;
      for (let j = 0; j < n; j++) {
        if (w[j] === 0) continue;
        let d2 = 0;
        for (let q = 0; q < NF; q++) {
          const d = rows[j][q] - f[q];
          d2 += d * d;
        }
        s += w[j] * Math.exp(-GAMMA * d2);
      }
      const rHat = ymu + Math.max(-4, Math.min(4, s)) * ysd;
      pred[i - split] = full[i - 1] * (1 + rHat);
    }
    return pred;
  }

  /* ---- target-date forecast: train once (cached), then roll the
     feature window forward feeding predicted returns back in ---- */
  let _fit = null;
  function getFit(ctx) {
    const tag = ctx.key + "|" + JSON.stringify(ctx.hyperparams || {});
    if (_fit && _fit.tag === tag) return _fit.m;

    const full = ctx.close, N = full.length, split = N - ctx.TEST;
    const ret = new Array(N);
    ret[0] = 0;
    for (let i = 1; i < N; i++) ret[i] = (full[i] - full[i - 1]) / full[i - 1];
    function feats(r) {
      const f = new Array(NF);
      f[0] = ret[r - 1]; f[1] = ret[r - 2]; f[2] = ret[r - 3];
      let m5 = 0;
      for (let k = 1; k <= 5; k++) m5 += ret[r - k];
      m5 /= 5;
      let v5 = 0;
      for (let k = 1; k <= 5; k++) v5 += (ret[r - k] - m5) * (ret[r - k] - m5);
      f[3] = m5;
      f[4] = Math.sqrt(v5 / 5);
      return f;
    }
    const i0 = 6;
    const rows = [], y = [];
    for (let i = i0; i < split; i++) {
      rows.push(feats(i));
      y.push(ret[i]);
    }
    const n = rows.length;
    const mu = new Array(NF).fill(0), sd = new Array(NF).fill(0);
    for (let f = 0; f < NF; f++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += rows[r][f];
      mu[f] = s / n;
      s = 0;
      for (let r = 0; r < n; r++) s += (rows[r][f] - mu[f]) * (rows[r][f] - mu[f]);
      sd[f] = Math.sqrt(s / n) || 1e-9;
      for (let r = 0; r < n; r++) rows[r][f] = (rows[r][f] - mu[f]) / sd[f];
    }
    let ymu = 0;
    for (let r = 0; r < n; r++) ymu += y[r];
    ymu /= n;
    let ysd = 0;
    for (let r = 0; r < n; r++) ysd += (y[r] - ymu) * (y[r] - ymu);
    ysd = Math.sqrt(ysd / n) || 1e-9;
    const yn = y.map(v => (v - ymu) / ysd);

    const K = new Float64Array(n * n);
    for (let a = 0; a < n; a++) {
      for (let b = a; b < n; b++) {
        let d2 = 0;
        for (let f = 0; f < NF; f++) {
          const d = rows[a][f] - rows[b][f];
          d2 += d * d;
        }
        const k = Math.exp(-GAMMA * d2);
        K[a * n + b] = k; K[b * n + a] = k;
      }
    }

    const w = new Float64Array(n);
    const G = new Float64Array(n);
    for (let i = 0; i < n; i++) G[i] = -yn[i];
    for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
      let maxDelta = 0;
      for (let i = 0; i < n; i++) {
        const Kii = K[i * n + i];
        const B = G[i] - w[i] * Kii;
        let wNew;
        if (B > EPS) wNew = -(B - EPS) / Kii;
        else if (B < -EPS) wNew = -(B + EPS) / Kii;
        else wNew = 0;
        if (wNew > C) wNew = C; else if (wNew < -C) wNew = -C;
        const delta = wNew - w[i];
        if (delta === 0) continue;
        w[i] = wNew;
        maxDelta = Math.max(maxDelta, Math.abs(delta));
        for (let j = 0; j < n; j++) G[j] += delta * K[i * n + j];
      }
      if (maxDelta < 1e-6) break;
    }

    let sv = 0;
    for (let i = 0; i < n; i++) if (w[i] !== 0) sv++;
    _fit = { tag, m: { w, rows, mu, sd, ymu, ysd, sv } };
    return _fit.m;
  }

  function predictFuture(ctx, h) {
    const m = getFit(ctx);
    const n = m.rows.length;
    /* rolling return tail: last 5 real returns feed the feature builder */
    const tail = [];
    const full = ctx.close;
    for (let i = full.length - 5; i < full.length; i++) tail.push((full[i] - full[i - 1]) / full[i - 1]);
    const prices = [];
    let price = full[full.length - 1];
    let stepR = 0;
    for (let s = 0; s < h; s++) {
      const raw = new Array(NF);
      raw[0] = tail[4]; raw[1] = tail[3]; raw[2] = tail[2];
      let m5 = 0;
      for (let k = 1; k <= 5; k++) m5 += tail[5 - k];
      m5 /= 5;
      let v5 = 0;
      for (let k = 1; k <= 5; k++) v5 += (tail[5 - k] - m5) * (tail[5 - k] - m5);
      raw[3] = m5;
      raw[4] = Math.sqrt(v5 / 5);
      const f = raw.map((v, k) => (v - m.mu[k]) / m.sd[k]);
      let pred = 0;
      for (let j = 0; j < n; j++) {
        if (m.w[j] === 0) continue;
        let d2 = 0;
        for (let q = 0; q < NF; q++) {
          const d = m.rows[j][q] - f[q];
          d2 += d * d;
        }
        pred += m.w[j] * Math.exp(-GAMMA * d2);
      }
      const rHat = m.ymu + Math.max(-4, Math.min(4, pred)) * m.ysd;
      stepR = rHat;
      price *= (1 + rHat);
      prices.push(price);
      tail.push(rHat);
      tail.shift();
    }
    const last = full[full.length - 1];
    const pct = (prices[prices.length - 1] / last - 1) * 100;
    const reason =
      "The kernel model compares the current 5-day pattern with the " + m.sv +
      " historical patterns it kept as support vectors — the weighted consensus of the most similar ones is a " +
      (stepR >= 0 ? "rise" : "fall") + " of about " + (stepR * 100).toFixed(2) +
      "% per session — " + (pct >= 0 ? "+" : "") + pct.toFixed(2) + "% by the chosen day.";
    return { prices, reason, pct };
  }

  ns.svr = { id: "svr", predict, predictFuture };
})(window.SPPAlgorithms = window.SPPAlgorithms || {});
