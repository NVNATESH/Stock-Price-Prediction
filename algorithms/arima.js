/* ============================================================
   ARIMA(p, d, q) — Autoregressive Integrated Moving Average
   Hand-written implementation (Box & Jenkins, 1970).

   The d-th differenced series z is modelled as

     z_t = φ₁z_{t−1} + … + φ_p z_{t−p}
           + θ₁e_{t−1} + … + θ_q e_{t−q} + noise

   Both coefficient vectors are fitted by conditional least squares
   (ridge-stabilised normal equations, solved with the hand-written
   linear solver in common.js). Prediction is one-step-ahead: the
   recursion runs over the actual history, so each test prediction
   uses only information available up to the previous session.
   ============================================================ */
(function (ns) {
  "use strict";
  const U = ns.utils;

  function predict(ctx) {
    const hp = (ctx.hyperparams && ctx.hyperparams.arima) || { p: 2, d: 1, q: 2 };
    const P = Math.max(1, Math.min(5, hp.p | 0));   // AR order
    const D = Math.max(0, Math.min(2, hp.d | 0));   // differencing degree
    const Q = Math.max(0, Math.min(4, hp.q | 0));   // MA order

    const full = ctx.close, N = full.length, T = ctx.TEST;
    const split = N - T;

    /* ---- d-th differencing of the actual series ----
       cur[j] is the D-th difference; it corresponds to the original
       index j + D. For D = 0 the model works on the levels directly. */
    let cur = full.slice();
    for (let d = 0; d < D; d++) {
      const dd = new Array(cur.length - 1);
      for (let i = 1; i < cur.length; i++) dd[i - 1] = cur[i] - cur[i - 1];
      cur = dd;
    }
    const off = D;                 // cur[j] ↔ original index j + off
    const M = cur.length;          // = N − D
    const jTest = split - off;     // first test position in cur

    /* ---- fit AR(P) on the training part of cur by least squares ---- */
    const rows = [], targets = [];
    for (let t = P; t < jTest; t++) {
      const row = new Array(P);
      for (let k = 1; k <= P; k++) row[k - 1] = cur[t - k];
      rows.push(row);
      targets.push(cur[t]);
    }
    const phi = U.linsolve(rows, targets, P, 1e-6);

    /* ---- AR residuals, then fit MA(Q) on them by least squares ---- */
    const e = new Array(jTest).fill(0);
    for (let t = P; t < jTest; t++) {
      let z = 0;
      for (let k = 1; k <= P; k++) z += phi[k - 1] * cur[t - k];
      e[t] = cur[t] - z;
    }
    let theta = null;
    if (Q > 0) {
      const mrows = [], mtargets = [];
      for (let t = P + Q; t < jTest; t++) {
        const row = new Array(Q);
        for (let k = 1; k <= Q; k++) row[k - 1] = e[t - k];
        mrows.push(row);
        mtargets.push(e[t]);
      }
      theta = U.linsolve(mrows, mtargets, Q, 1e-6);
    }

    /* ---- one-step-ahead recursion over the full history ----
       For every position the AR terms read the ACTUAL past (available
       in one-step-ahead forecasting) and the MA terms read the actual
       residuals e_t = z_t − ẑ_t, so no future information is used.   */
    const pred = new Array(T);
    for (let j = jTest; j < M; j++) {
      let z = 0;
      for (let k = 1; k <= P; k++) z += phi[k - 1] * cur[j - k];
      if (theta) {
        for (let k = 1; k <= Q; k++) z += theta[k - 1] * e[j - k];
      }
      /* reconstruct the level from the predicted difference:
           D = 0 → ŷ = ẑ                D = 1 → ŷ = y_{t−1} + ẑ
           D = 2 → ŷ = 2y_{t−1} − y_{t−2} + ẑ                        */
      let level;
      if (D === 0) level = z;
      else if (D === 1) level = full[j + off - 1] + z;
      else level = 2 * full[j + off - 1] - full[j + off - 2] + z;

      pred[j - jTest] = level;
      /* actual residual for the next step's MA feedback */
      e[j] = cur[j] - z;
    }
    return pred;
  }

  /* ---- target-date forecast: refit once (cached), iterate h sessions ---- */
  let _fit = null;
  function getFit(ctx) {
    const tag = ctx.key + "|" + JSON.stringify(ctx.hyperparams || {});
    if (_fit && _fit.tag === tag) return _fit.m;
    const hp = (ctx.hyperparams && ctx.hyperparams.arima) || { p: 2, d: 1, q: 2 };
    const P = Math.max(1, Math.min(5, hp.p | 0));
    const D = Math.max(0, Math.min(2, hp.d | 0));
    const Q = Math.max(0, Math.min(4, hp.q | 0));
    const full = ctx.close;
    let cur = full.slice();
    for (let d = 0; d < D; d++) {
      const dd = new Array(cur.length - 1);
      for (let i = 1; i < cur.length; i++) dd[i - 1] = cur[i] - cur[i - 1];
      cur = dd;
    }
    const split = full.length - ctx.TEST;
    const jTest = split - D;
    const rows = [], targets = [];
    for (let t = P; t < jTest; t++) {
      const row = new Array(P);
      for (let k = 1; k <= P; k++) row[k - 1] = cur[t - k];
      rows.push(row);
      targets.push(cur[t]);
    }
    const phi = U.linsolve(rows, targets, P, 1e-6);
    const e = new Array(jTest).fill(0);
    for (let t = P; t < jTest; t++) {
      let z = 0;
      for (let k = 1; k <= P; k++) z += phi[k - 1] * cur[t - k];
      e[t] = cur[t] - z;
    }
    let theta = null;
    if (Q > 0) {
      const mrows = [], mtargets = [];
      for (let t = P + Q; t < jTest; t++) {
        const row = new Array(Q);
        for (let k = 1; k <= Q; k++) row[k - 1] = e[t - k];
        mrows.push(row);
        mtargets.push(e[t]);
      }
      theta = U.linsolve(mrows, mtargets, Q, 1e-6);
    }
    _fit = { tag, m: { P, D, Q, phi, theta, cur, full } };
    return _fit.m;
  }

  /* Iterate the fitted recursion into the future: AR terms read the
     (increasingly predicted) differences, MA terms use the average miss
     of 0 — i.e. the expected residual. Returns the price path + reason. */
  function predictFuture(ctx, h) {
    const m = getFit(ctx);
    const cur = m.cur, D = m.D, P = m.P, Q = m.Q, phi = m.phi, theta = m.theta;
    const full = ctx.close;
    const M = cur.length;
    const e = new Array(M + h).fill(0);
    for (let t = P; t < M; t++) {
      let z = 0;
      for (let k = 1; k <= P; k++) z += phi[k - 1] * cur[t - k];
      if (theta) for (let k = 1; k <= Q; k++) z += theta[k - 1] * e[t - k];
      e[t] = cur[t] - z;
    }
    const fut = new Array(h).fill(0);
    const prices = [];
    let l1 = full[full.length - 1];
    let l2 = full[full.length - 2];
    for (let s = 0; s < h; s++) {
      const j = M + s;
      let z = 0;
      for (let k = 1; k <= P; k++) z += phi[k - 1] * (j - k >= M ? fut[j - k - M] : cur[j - k]);
      if (theta) for (let k = 1; k <= Q; k++) z += theta[k - 1] * e[j - k];
      fut[s] = z;
      let level;
      if (D === 0) level = z;
      else if (D === 1) level = l1 + z;
      else level = 2 * l1 - l2 + z;
      prices.push(level);
      l2 = l1; l1 = level;
    }
    const last = full[full.length - 1];
    const pct = (prices[prices.length - 1] / last - 1) * 100;
    const recent = cur.slice(-P).map(v => (v >= 0 ? "+" : "") + v.toFixed(2)).join(", ");
    const reason =
      "ARIMA(" + P + "," + D + "," + Q + ") reads the last " + P + " real daily change" + (P > 1 ? "s" : "") +
      " (" + recent + ") plus its average recent miss, and projects that pattern forward — a cumulative " +
      (pct >= 0 ? "+" : "") + pct.toFixed(2) + "% move over " + h + " sessions.";
    return { prices, reason, pct };
  }

  ns.arima = { id: "arima", predict, predictFuture };
})(window.SPPAlgorithms = window.SPPAlgorithms || {});
