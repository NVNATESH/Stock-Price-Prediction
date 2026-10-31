/* ============================================================
   STOCK PRICE PREDICTION — algorithms/common.js
   Shared numeric infrastructure for the hand-written engines:
     · deterministic seeded RNG (FNV-1a hash → mulberry32)
     · Gaussian noise (Box–Muller)
     · error metrics (MSE / RMSE)
     · ridge least-squares solver (hand-written Gaussian elimination)
   Every prediction algorithm implements its own mathematics from
   scratch — nothing here is imported from any library.
   Exposed on the global namespace: window.SPPAlgorithms
   ============================================================ */
(function (ns) {
  "use strict";

  /* ---- deterministic RNG: FNV-1a string hash → mulberry32 PRNG ---- */
  function hashStr(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---- standard normal noise (Box–Muller transform) ---- */
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---- error metrics on the test window ---- */
  function metricsOf(actual, pred) {
    let se = 0;
    for (let i = 0; i < actual.length; i++) { const d = pred[i] - actual[i]; se += d * d; }
    const mse = se / actual.length;
    return { mse, rmse: Math.sqrt(mse) };
  }

  /* ---- ridge least squares: minimise ||A·w − b||² + λ||w||² ----
     Hand-written normal-equations solver (Gaussian elimination with
     partial pivoting). A is an array of rows; replaces numpy.linalg.lstsq
     for the small dense systems used by ARIMA and Prophet.            */
  function linsolve(A, b, cols, lambda) {
    const n = A.length;
    const M = [];
    for (let i = 0; i < cols; i++) {
      const row = new Array(cols + 1).fill(0);
      for (let j = 0; j < cols; j++) {
        let s = 0;
        for (let r = 0; r < n; r++) s += A[r][i] * A[r][j];
        row[j] = s + (i === j ? lambda : 0);
      }
      let s2 = 0;
      for (let r = 0; r < n; r++) s2 += A[r][i] * b[r];
      row[cols] = s2;
      M.push(row);
    }
    for (let col = 0; col < cols; col++) {
      let piv = col;
      for (let r = col + 1; r < cols; r++) {
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      }
      if (Math.abs(M[piv][col]) < 1e-12) continue;
      const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
      for (let r = 0; r < cols; r++) {
        if (r === col) continue;
        const f = M[r][col] / M[col][col];
        if (!f) continue;
        for (let c = col; c <= cols; c++) M[r][c] -= f * M[col][c];
      }
    }
    const w = new Array(cols).fill(0);
    for (let i = 0; i < cols; i++) {
      w[i] = Math.abs(M[i][i]) > 1e-12 ? M[i][cols] / M[i][i] : 0;
    }
    return w;
  }

  ns.utils = { hashStr, mulberry32, gauss, metricsOf, linsolve };
})(window.SPPAlgorithms = window.SPPAlgorithms || {});
