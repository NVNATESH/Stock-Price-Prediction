/* ============================================================
   PROPHET — Additive decomposition model
   Hand-written implementation of the Prophet methodology
   (Taylor & Letham, 2018): the series is decomposed as

     y(t) = g(t) + s(t) + ε

   with a piecewise-linear trend g(t) built from hinge functions at
   evenly spaced changepoints, plus Fourier seasonal components for
   the 5-session trading week and the ~21-session month. All
   components are linear in their parameters, so the whole model is
   fitted in one ridge least-squares solve (common.js linsolve).

   As with the other engines, the decomposition is fitted on the
   stationary series (daily returns) and the predicted return is
   anchored to the previous close — the standard practitioner
   preprocessing for financial series, where a level-trend fit
   would extrapolate wildly.
   ============================================================ */
(function (ns) {
  "use strict";
  const U = ns.utils;

  function predict(ctx) {
    const full = ctx.close, N = full.length, T = ctx.TEST;
    const split = N - T;

    const CP = 6;   // trend changepoints
    const K5 = 3;   // Fourier order for the 5-session (weekly) cycle
    const K21 = 2;  // Fourier order for the 21-session (monthly) cycle
    const COLS = 2 + CP + 2 * K5 + 2 * K21;
    /* Ridge penalty: shrinkage prior on trend/seasonal coefficients so
       the fitted curve extrapolates smoothly (the role Prophet's
       Bayesian priors play in the reference implementation).          */
    const LAMBDA = 0.75;

    /* ---- design matrix row for session index t ----
       Time is normalised by the training length so the ridge penalty
       acts on a well-scaled basis (t̂ ∈ [0, ~1.14] over the horizon). */
    function row(t) {
      const tn = t / split;
      const r = new Array(COLS);
      r[0] = 1;                       // intercept
      r[1] = tn;                      // global linear trend
      for (let k = 0; k < CP; k++) {  // hinge (piecewise) trend terms
        const cp = (k + 1) / (CP + 1);
        r[2 + k] = Math.max(0, tn - cp);
      }
      let c = 2 + CP;
      for (let k = 1; k <= K5; k++) {
        r[c++] = Math.sin((2 * Math.PI * k * t) / 5);
        r[c++] = Math.cos((2 * Math.PI * k * t) / 5);
      }
      for (let k = 1; k <= K21; k++) {
        r[c++] = Math.sin((2 * Math.PI * k * t) / 21);
        r[c++] = Math.cos((2 * Math.PI * k * t) / 21);
      }
      return r;
    }

    /* ---- fit on training-window RETURNS (the stationary series) ---- */
    const rows = [], targets = [];
    for (let i = 1; i < split; i++) {
      rows.push(row(i));
      targets.push((full[i] - full[i - 1]) / full[i - 1]);
    }
    const beta = U.linsolve(rows, targets, COLS, LAMBDA);

    /* ---- predict each test return, anchored to the previous close ---- */
    const pred = new Array(T);
    for (let i = split; i < N; i++) {
      const r = row(i);
      let rHat = 0;
      for (let c = 0; c < COLS; c++) rHat += beta[c] * r[c];
      pred[i - split] = full[i - 1] * (1 + rHat);
    }
    return pred;
  }

  /* ---- target-date forecast: fit once (cached), evaluate the curve ahead ---- */
  let _fit = null;
  function getFit(ctx) {
    const tag = ctx.key + "|" + JSON.stringify(ctx.hyperparams || {});
    if (_fit && _fit.tag === tag) return _fit.m;
    const full = ctx.close;
    const split = full.length - ctx.TEST;
    const CP = 6, K5 = 3, K21 = 2;
    const COLS = 2 + CP + 2 * K5 + 2 * K21;
    const LAMBDA = 0.75;
    function row(t) {
      const tn = t / split;
      const r = new Array(COLS);
      r[0] = 1;
      r[1] = tn;
      for (let k = 0; k < CP; k++) {
        const cp = (k + 1) / (CP + 1);
        r[2 + k] = Math.max(0, tn - cp);
      }
      let c = 2 + CP;
      for (let k = 1; k <= K5; k++) {
        r[c++] = Math.sin((2 * Math.PI * k * t) / 5);
        r[c++] = Math.cos((2 * Math.PI * k * t) / 5);
      }
      for (let k = 1; k <= K21; k++) {
        r[c++] = Math.sin((2 * Math.PI * k * t) / 21);
        r[c++] = Math.cos((2 * Math.PI * k * t) / 21);
      }
      return r;
    }
    const rows = [], targets = [];
    for (let i = 1; i < split; i++) {
      rows.push(row(i));
      targets.push((full[i] - full[i - 1]) / full[i - 1]);
    }
    const beta = U.linsolve(rows, targets, COLS, LAMBDA);
    _fit = { tag, m: { row, beta, split, CP, COLS } };
    return _fit.m;
  }

  function predictFuture(ctx, h) {
    const m = getFit(ctx);
    const full = ctx.close, split = m.split;
    const prices = [];
    let price = full[full.length - 1];
    let trendPart = 0, seasonPart = 0;
    for (let s = 0; s < h; s++) {
      const r = m.row(split + s);
      let rHat = 0;
      for (let c = 0; c < m.COLS; c++) rHat += m.beta[c] * r[c];
      price *= (1 + rHat);
      prices.push(price);
      if (s === h - 1) {
        /* decompose the final day's predicted return into trend vs season */
        const trend = m.beta[1] * (r[1] - 1) +
          m.beta.slice(2, 2 + m.CP).reduce((a, v) => a + v, 0);
        trendPart = trend * 100;
        seasonPart = rHat * 100 - trendPart - m.beta[0] * 100;
      }
    }
    const last = full[full.length - 1];
    const pct = (prices[prices.length - 1] / last - 1) * 100;
    const reason =
      "The fitted additive curve extends its trend (" + (trendPart >= 0 ? "+" : "") + trendPart.toFixed(2) +
      "% by the chosen day) and its weekly/monthly seasonal cycle (" + (seasonPart >= 0 ? "+" : "") +
      seasonPart.toFixed(2) + "% that session) — compounding to " + (pct >= 0 ? "+" : "") +
      pct.toFixed(2) + "% over " + h + " sessions.";
    return { prices, reason, pct };
  }

  ns.prophet = { id: "prophet", predict, predictFuture };
})(window.SPPAlgorithms = window.SPPAlgorithms || {});
