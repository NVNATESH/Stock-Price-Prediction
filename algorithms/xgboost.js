/* ============================================================
   XGBOOST — Regularized gradient-boosted regression trees
   Hand-written implementation of the gradient boosting
   framework (Chen & Guestrin, 2016) with squared-error loss:

     F_0 = 0              F_m(x) = F_{m−1}(x) + η · h_m(x)

   where each h_m is a regression tree fitted to the current
   residuals y − F_{m−1}. Trees are grown manually (greedy split
   search over quantile bins, max_depth from the tuner, minimum
   leaf size) with row subsampling per tree (stochastic boosting,
   seeded → deterministic). Features are lagged returns and rolling
   statistics; the model predicts the next return.
   ============================================================ */
(function (ns) {
  "use strict";
  const U = ns.utils;

  const LAGS = 5;        // r_{t−1} … r_{t−5}
  const NF = LAGS + 4;   // feature count: lags + mean5/mean10/std5/momentum
  const ROUNDS = 80;     // boosting rounds
  const ETA = 0.1;       // learning rate (shrinkage)
  const MIN_LEAF = 8;
  const BINS = 12;       // split candidates per feature

  /* walk a tree down with a feature vector (module scope: shared by
     the one-step protocol and the target-date forecaster) */
  function treeEval(node, f) {
    while (node.f !== undefined) node = f[node.f] < node.thr ? node.l : node.r;
    return node.v;
  }

  function predict(ctx) {
    const hp = (ctx.hyperparams && ctx.hyperparams.xgboost) || { depth: 6, subsample: 0.8 };
    const maxDepth = Math.max(2, Math.min(10, hp.depth | 0));
    const subsample = Math.max(0.5, Math.min(1, hp.subsample || 0.8));

    const full = ctx.close, N = full.length, T = ctx.TEST;
    const split = N - T;

    /* ---- returns and feature builder ----
       features(i) uses only sessions up to i−1 (one-step-ahead). */
    const ret = new Array(N);
    ret[0] = 0;
    for (let i = 1; i < N; i++) ret[i] = (full[i] - full[i - 1]) / full[i - 1];
    function features(i) {
      const f = new Array(NF);
      for (let k = 1; k <= LAGS; k++) f[k - 1] = ret[i - k];
      let m5 = 0, m10 = 0;
      for (let k = 1; k <= 10; k++) { const v = ret[i - k]; if (k <= 5) m5 += v; m10 += v; }
      m5 /= 5; m10 /= 10;
      let v5 = 0;
      for (let k = 1; k <= 5; k++) v5 += (ret[i - k] - m5) * (ret[i - k] - m5);
      f[LAGS] = m5;
      f[LAGS + 1] = m10;
      f[LAGS + 2] = Math.sqrt(v5 / 5);
      f[LAGS + 3] = ret[i - 1] + ret[i - 2];   // 2-day momentum
      return f;
    }

    /* ---- standardised training rows and target ---- */
    const i0 = 11;                       // first index with full history
    const rows = [], y = [];
    for (let i = i0; i < split; i++) {
      rows.push(features(i));
      y.push(ret[i]);
    }
    const nTrain = rows.length;
    const mu = new Array(NF).fill(0), sd = new Array(NF).fill(0);
    for (let f = 0; f < NF; f++) {
      let s = 0;
      for (let r = 0; r < nTrain; r++) s += rows[r][f];
      mu[f] = s / nTrain;
      s = 0;
      for (let r = 0; r < nTrain; r++) s += (rows[r][f] - mu[f]) * (rows[r][f] - mu[f]);
      sd[f] = Math.sqrt(s / nTrain) || 1e-9;
      for (let r = 0; r < nTrain; r++) rows[r][f] = (rows[r][f] - mu[f]) / sd[f];
    }
    let ymu = 0;
    for (let r = 0; r < nTrain; r++) ymu += y[r];
    ymu /= nTrain;
    let ysd = 0;
    for (let r = 0; r < nTrain; r++) ysd += (y[r] - ymu) * (y[r] - ymu);
    ysd = Math.sqrt(ysd / nTrain) || 1e-9;
    const yn = y.map(v => (v - ymu) / ysd);   // normalised target

    /* ---- greedy regression-tree builder ----
       targets[] is the fitting array (the current residuals);
       returns the leaf-mean at the root when no split helps.    */
    function buildTree(idx, targets, depth) {
      let sum = 0;
      for (const r of idx) sum += targets[r];
      const node = { v: sum / idx.length };
      if (depth >= maxDepth || idx.length < 2 * MIN_LEAF) return node;
      let best = null;
      for (let f = 0; f < NF; f++) {
        const vals = idx.map(r => rows[r][f]).sort((a, b) => a - b);
        for (let b = 1; b < BINS; b++) {
          const thr = vals[Math.floor((b / BINS) * vals.length)];
          let ls = 0, rs = 0, ln = 0, rn = 0;
          for (const r of idx) {
            if (rows[r][f] < thr) { ls += targets[r]; ln++; }
            else { rs += targets[r]; rn++; }
          }
          if (ln < MIN_LEAF || rn < MIN_LEAF) continue;
          /* split quality = Σ leaf²/n  (equivalent to SSE reduction) */
          const score = (ls * ls) / ln + (rs * rs) / rn;
          if (!best || score > best.score) best = { f, thr, score };
        }
      }
      if (!best) return node;
      const left = [], right = [];
      for (const r of idx) (rows[r][best.f] < best.thr ? left : right).push(r);
      node.f = best.f; node.thr = best.thr;
      node.l = buildTree(left, targets, depth + 1);
      node.r = buildTree(right, targets, depth + 1);
      return node;
    }
    function treeEval(node, f) {
      while (node.f !== undefined) node = f[node.f] < node.thr ? node.l : node.r;
      return node.v;
    }

    /* ---- gradient boosting with row subsampling ---- */
    const trees = [];
    const F = new Float64Array(nTrain);        // ensemble estimate of yn
    const resid = new Float64Array(nTrain);
    for (let r = 0; r < nTrain; r++) resid[r] = yn[r];
    const rng = U.mulberry32(U.hashStr(ctx.key + "|xgboost|boost|v1"));
    for (let m = 0; m < ROUNDS; m++) {
      const idx = [];
      for (let r = 0; r < nTrain; r++) {
        if (subsample >= 1 || rng() < subsample) idx.push(r);
      }
      if (idx.length < 2 * MIN_LEAF) continue;
      const tree = buildTree(idx, resid, 0);
      for (let r = 0; r < nTrain; r++) {
        F[r] += ETA * treeEval(tree, rows[r]);
        resid[r] = yn[r] - F[r];
      }
      trees.push(tree);
    }

    /* ---- one-step-ahead inference on the test window ---- */
    const pred = new Array(T);
    for (let i = split; i < N; i++) {
      const f = features(i);
      for (let k = 0; k < NF; k++) f[k] = (f[k] - mu[k]) / sd[k];
      let rn = 0;
      for (const t of trees) rn += ETA * treeEval(t, f);
      const rHat = ymu + Math.max(-4, Math.min(4, rn)) * ysd;  // de-normalised
      pred[i - split] = full[i - 1] * (1 + rHat);
    }
    return pred;
  }

  /* ---- target-date forecast: train once (cached), then walk the
     feature window forward feeding predicted returns back in ---- */
  let _fit = null;
  function getFit(ctx) {
    const tag = ctx.key + "|" + JSON.stringify(ctx.hyperparams || {});
    if (_fit && _fit.tag === tag) return _fit.m;
    const hp = (ctx.hyperparams && ctx.hyperparams.xgboost) || { depth: 6, subsample: 0.8 };
    const maxDepth = Math.max(2, Math.min(10, hp.depth | 0));
    const subsample = Math.max(0.5, Math.min(1, hp.subsample || 0.8));

    const full = ctx.close, N = full.length, split = N - ctx.TEST;
    const ret = new Array(N);
    ret[0] = 0;
    for (let i = 1; i < N; i++) ret[i] = (full[i] - full[i - 1]) / full[i - 1];
    /* build the lag/rolling feature vector from any return array at index i */
    function featuresAt(arr, i) {
      const f = new Array(NF);
      for (let k = 1; k <= LAGS; k++) f[k - 1] = arr[i - k];
      let m5 = 0, m10 = 0;
      for (let k = 1; k <= 10; k++) { const v = arr[i - k]; if (k <= 5) m5 += v; m10 += v; }
      m5 /= 5; m10 /= 10;
      let v5 = 0;
      for (let k = 1; k <= 5; k++) v5 += (arr[i - k] - m5) * (arr[i - k] - m5);
      f[LAGS] = m5;
      f[LAGS + 1] = m10;
      f[LAGS + 2] = Math.sqrt(v5 / 5);
      f[LAGS + 3] = arr[i - 1] + arr[i - 2];
      return f;
    }
    const i0 = 11;
    const rows = [], y = [];
    for (let i = i0; i < split; i++) {
      rows.push(featuresAt(ret, i));
      y.push(ret[i]);
    }
    const nTrain = rows.length;
    const mu = new Array(NF).fill(0), sd = new Array(NF).fill(0);
    for (let f = 0; f < NF; f++) {
      let s = 0;
      for (let r = 0; r < nTrain; r++) s += rows[r][f];
      mu[f] = s / nTrain;
      s = 0;
      for (let r = 0; r < nTrain; r++) s += (rows[r][f] - mu[f]) * (rows[r][f] - mu[f]);
      sd[f] = Math.sqrt(s / nTrain) || 1e-9;
      for (let r = 0; r < nTrain; r++) rows[r][f] = (rows[r][f] - mu[f]) / sd[f];
    }
    let ymu = 0;
    for (let r = 0; r < nTrain; r++) ymu += y[r];
    ymu /= nTrain;
    let ysd = 0;
    for (let r = 0; r < nTrain; r++) ysd += (y[r] - ymu) * (y[r] - ymu);
    ysd = Math.sqrt(ysd / nTrain) || 1e-9;
    const yn = y.map(v => (v - ymu) / ysd);

    function buildTree(idx, targets, depth) {
      let sum = 0;
      for (const r of idx) sum += targets[r];
      const node = { v: sum / idx.length };
      if (depth >= maxDepth || idx.length < 2 * MIN_LEAF) return node;
      let best = null;
      for (let f = 0; f < NF; f++) {
        const vals = idx.map(r => rows[r][f]).sort((a, b) => a - b);
        for (let b = 1; b < BINS; b++) {
          const thr = vals[Math.floor((b / BINS) * vals.length)];
          let ls = 0, rs = 0, ln = 0, rn = 0;
          for (const r of idx) {
            if (rows[r][f] < thr) { ls += targets[r]; ln++; }
            else { rs += targets[r]; rn++; }
          }
          if (ln < MIN_LEAF || rn < MIN_LEAF) continue;
          const score = (ls * ls) / ln + (rs * rs) / rn;
          if (!best || score > best.score) best = { f, thr, score };
        }
      }
      if (!best) return node;
      const left = [], right = [];
      for (const r of idx) (rows[r][best.f] < best.thr ? left : right).push(r);
      node.f = best.f; node.thr = best.thr;
      node.l = buildTree(left, targets, depth + 1);
      node.r = buildTree(right, targets, depth + 1);
      return node;
    }
    function treeEval(node, f) {
      while (node.f !== undefined) node = f[node.f] < node.thr ? node.l : node.r;
      return node.v;
    }

    const trees = [];
    const F = new Float64Array(nTrain);
    const resid = new Float64Array(nTrain);
    for (let r = 0; r < nTrain; r++) resid[r] = yn[r];
    const rng = U.mulberry32(U.hashStr(ctx.key + "|xgboost|boost|v1"));
    for (let m = 0; m < ROUNDS; m++) {
      const idx = [];
      for (let r = 0; r < nTrain; r++) {
        if (subsample >= 1 || rng() < subsample) idx.push(r);
      }
      if (idx.length < 2 * MIN_LEAF) continue;
      const tree = buildTree(idx, resid, 0);
      for (let r = 0; r < nTrain; r++) {
        F[r] += ETA * treeEval(tree, rows[r]);
        resid[r] = yn[r] - F[r];
      }
      trees.push(tree);
    }

    _fit = { tag, m: { trees, mu, sd, ymu, ysd, ret } };
    return _fit.m;
  }

  /* split-frequency feature importance: how often each feature is used
     as a split across all boosted trees (hand-traversed) */
  function importance(ctx) {
    const m = getFit(ctx);
    const names = ["Return t−1", "Return t−2", "Return t−3", "Return t−4", "Return t−5",
                   "Mean 5-day", "Mean 10-day", "Vol 5-day", "2-day momentum"];
    const counts = new Array(NF).fill(0);
    function walk(node) {
      if (node.f === undefined) return;
      counts[node.f]++;
      walk(node.l);
      walk(node.r);
    }
    for (const t of m.trees) { walk(t); }
    let splits = 0;
    for (let f = 0; f < NF; f++) splits += counts[f];
    return names.map((name, f) => ({
      name,
      count: counts[f],
      pct: splits ? +((counts[f] / splits) * 100).toFixed(1) : 0,
    })).sort((a, b) => b.count - a.count);
  }

  function featuresFromTail(tail) {
    /* tail: rolling array of recent returns, tail[tail.length-1] = r_{t-1} */
    const f = new Array(NF);
    for (let k = 1; k <= LAGS; k++) f[k - 1] = tail[tail.length - k];
    let m5 = 0, m10 = 0;
    for (let k = 1; k <= 10; k++) { const v = tail[tail.length - k]; if (k <= 5) m5 += v; m10 += v; }
    m5 /= 5; m10 /= 10;
    let v5 = 0;
    for (let k = 1; k <= 5; k++) v5 += (tail[tail.length - k] - m5) * (tail[tail.length - k] - m5);
    f[LAGS] = m5;
    f[LAGS + 1] = m10;
    f[LAGS + 2] = Math.sqrt(v5 / 5);
    f[LAGS + 3] = tail[tail.length - 1] + tail[tail.length - 2];
    return f;
  }

  function predictFuture(ctx, h) {
    const m = getFit(ctx);
    const tail = m.ret.slice(-10);
    const prices = [];
    let price = ctx.close[ctx.close.length - 1];
    let stepR = 0;
    for (let s = 0; s < h; s++) {
      const f = featuresFromTail(tail);
      for (let k = 0; k < NF; k++) f[k] = (f[k] - m.mu[k]) / m.sd[k];
      let rn = 0;
      for (const t of m.trees) rn += ETA * treeEval(t, f);
      const rHat = m.ymu + Math.max(-4, Math.min(4, rn)) * m.ysd;
      stepR = rHat;
      price *= (1 + rHat);
      prices.push(price);
      tail.push(rHat);
      tail.shift();
    }
    const last = ctx.close[ctx.close.length - 1];
    const pct = (prices[prices.length - 1] / last - 1) * 100;
    const f5 = tail.slice(-5).map(v => (v * 100).toFixed(2) + "%").join(", ");
    const reason =
      "The boosted trees read the last five real daily moves (" + f5 +
      ") and the patterns most similar to it in training history resolved to a " +
      (stepR >= 0 ? "rise" : "fall") + " of about " + (stepR * 100).toFixed(2) +
      "% per session — " + (pct >= 0 ? "+" : "") + pct.toFixed(2) + "% by the chosen day.";
    return { prices, reason, pct };
  }

  ns.xgboost = { id: "xgboost", predict, predictFuture, importance };
})(window.SPPAlgorithms = window.SPPAlgorithms || {});
