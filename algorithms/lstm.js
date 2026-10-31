/* ============================================================
   LSTM — Long Short-Term Memory network
   Hand-written implementation (Hochreiter & Schmidhuber, 1997).

   A single-layer LSTM cell with hidden size 6 is trained from
   scratch on the training window by batch gradient descent with
   manual backpropagation through time (BPTT). The network reads
   the last `lookback` daily returns and predicts the next return;
   the forecast level is prev_close × (1 + r̂).

     f_t = σ(W_f x_t + U_f h_{t−1} + b_f)      forget gate
     i_t = σ(W_i x_t + U_i h_{t−1} + b_i)      input gate
     g_t = tanh(W_g x_t + U_g h_{t−1} + b_g)   candidate
     c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t           cell state
     o_t = σ(W_o x_t + U_o h_{t−1} + b_o)      output gate
     h_t = o_t ⊙ tanh(c_t)
     ŷ_t = w_y·h_t + b_y

   All weights are initialised from the seeded PRNG (deterministic
   training) and every forward/backward buffer is preallocated per
   call, so the training loop is allocation-free. U matrices are
   stored flat (row-major) for cache-friendly inner loops.
   ============================================================ */
(function (ns) {
  "use strict";
  const U = ns.utils;

  const H = 6;          // hidden units
  const GRAD_CLIP = 5;

  function predict(ctx) {
    const hp = (ctx.hyperparams && ctx.hyperparams.lstm) ||
      { epochs: 50, lookback: 30, lr: 0.005 };
    const L = Math.max(5, Math.min(60, hp.lookback | 0));
    const EPOCHS = Math.max(10, Math.min(100, hp.epochs | 0));
    const LR = Math.max(0.0005, Math.min(0.02, hp.lr));

    const full = ctx.close, N = full.length, T = ctx.TEST;
    const split = N - T;

    /* ---- daily returns, normalised with train statistics only ---- */
    const ret = new Array(N - 1);
    for (let i = 1; i < N; i++) ret[i - 1] = (full[i] - full[i - 1]) / full[i - 1];
    let mu = 0;
    for (let i = 0; i < split - 1; i++) mu += ret[i];
    mu /= (split - 1);
    let sd = 0;
    for (let i = 0; i < split - 1; i++) sd += (ret[i] - mu) * (ret[i] - mu);
    sd = Math.sqrt(sd / Math.max(1, split - 2)) || 1e-6;
    const x = ret.map(v => (v - mu) / sd);   // x[j] ↔ return of session j+1

    /* ---- parameters (seeded init → reproducible) ---- */
    const rng = U.mulberry32(U.hashStr(ctx.key + "|lstm|init|v3"));
    const rand = n => { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = (rng() * 2 - 1) * 0.25; return a; };
    const Wg = rand(H), Wi = rand(H), Wf = rand(H), Wo = rand(H), wy = rand(H);
    const bg = rand(H), bi = rand(H), bf = rand(H), bo = rand(H);
    bf.fill(1);                              // forget bias = 1 (remember by default)
    let by = 0;
    const UG = rand(H * H), UI = rand(H * H), UF = rand(H * H), UO = rand(H * H);

    /* ---- preallocated forward/backward buffers (per call) ---- */
    const XT = new Float64Array(L);
    const CP = new Float64Array(L * H), HP = new Float64Array(L * H);
    const Gg = new Float64Array(L * H), Ii = new Float64Array(L * H);
    const Ff = new Float64Array(L * H), Oo = new Float64Array(L * H);
    const CN = new Float64Array(L * H), HN = new Float64Array(L * H);
    const THC = new Float64Array(L * H);
    const cSt = new Float64Array(H), hSt = new Float64Array(H);
    const dag = new Float64Array(H), dai = new Float64Array(H);
    const daf = new Float64Array(H), dao = new Float64Array(H);
    const dh = new Float64Array(H), dc = new Float64Array(H);
    const dhn = new Float64Array(H), dcn = new Float64Array(H);

    const sig = v => 1 / (1 + Math.exp(-v));
    const clip = (v, lim) => v > lim ? lim : (v < -lim ? -lim : v);

    /* run the cell over XT[0..len−1]; state carried in cSt/hSt */
    function forward(len) {
      for (let t = 0; t < len; t++) {
        const xt = XT[t], o = t * H;
        for (let j = 0; j < H; j++) {
          CP[o + j] = cSt[j]; HP[o + j] = hSt[j];
          const ug = j * H;
          let sg = bg[j], si = bi[j], sf = bf[j], so = bo[j];
          for (let k = 0; k < H; k++) {
            const hk = hSt[k];
            sg += UG[ug + k] * hk; si += UI[ug + k] * hk;
            sf += UF[ug + k] * hk; so += UO[ug + k] * hk;
          }
          sg += Wg[j] * xt; si += Wi[j] * xt;
          sf += Wf[j] * xt; so += Wo[j] * xt;
          const gv = Math.tanh(sg);
          const iv = sig(si), fv = sig(sf), ov = sig(so);
          const cv = fv * cSt[j] + iv * gv;
          const hv = ov * Math.tanh(cv);
          Gg[o + j] = gv; Ii[o + j] = iv; Ff[o + j] = fv; Oo[o + j] = ov;
          CN[o + j] = cv; HN[o + j] = hv; THC[o + j] = Math.tanh(cv);
        }
        for (let j = 0; j < H; j++) { cSt[j] = CN[o + j]; hSt[j] = HN[o + j]; }
      }
      let y = by;
      const last = (len - 1) * H;
      for (let k = 0; k < H; k++) y += wy[k] * HN[last + k];
      return y;
    }

    /* ---- manual BPTT for the sequence just forwarded (target = tgt) ---- */
    function backward(len, tgt) {
      dh.fill(0); dc.fill(0);
      const dy = clip(2 * (forward_y - tgt), GRAD_CLIP);
      G.by += dy;
      const last = (len - 1) * H;
      for (let k = 0; k < H; k++) {
        G.wy[k] += dy * HN[last + k];
        dh[k] = dy * wy[k];
      }
      for (let t = len - 1; t >= 0; t--) {
        const o = t * H;
        dhn.fill(0);
        for (let j = 0; j < H; j++) {
          const thc = THC[o + j];
          const dCell = dc[j] + dh[j] * Oo[o + j] * (1 - thc * thc);
          dao[j] = clip(dh[j] * thc * Oo[o + j] * (1 - Oo[o + j]), GRAD_CLIP);
          daf[j] = clip(dCell * CP[o + j] * Ff[o + j] * (1 - Ff[o + j]), GRAD_CLIP);
          dai[j] = clip(dCell * Gg[o + j] * Ii[o + j] * (1 - Ii[o + j]), GRAD_CLIP);
          dag[j] = clip(dCell * Ii[o + j] * (1 - Gg[o + j] * Gg[o + j]), GRAD_CLIP);
          dcn[j] = dCell * Ff[o + j];
        }
        for (let j = 0; j < H; j++) {
          G.bg[j] += dag[j]; G.bi[j] += dai[j];
          G.bf[j] += daf[j]; G.bo[j] += dao[j];
          const xt = XT[t], ug = j * H;
          G.Wg[j] += dag[j] * xt; G.Wi[j] += dai[j] * xt;
          G.Wf[j] += daf[j] * xt; G.Wo[j] += dao[j] * xt;
          for (let k = 0; k < H; k++) {
            const hk = HP[o + k];                 // h_{t−1}
            G.UG[ug + k] += dag[j] * hk; G.UI[ug + k] += dai[j] * hk;
            G.UF[ug + k] += daf[j] * hk; G.UO[ug + k] += dao[j] * hk;
            dhn[k] += dag[j] * UG[ug + k] + dai[j] * UI[ug + k]
                    + daf[j] * UF[ug + k] + dao[j] * UO[ug + k];
          }
        }
        for (let j = 0; j < H; j++) { dc[j] = dcn[j]; dh[j] = dhn[j]; }
      }
    }

    /* ---- gradient accumulator + epoch apply (batch GD) ---- */
    function allocG() {
      return { Wg: new Float64Array(H), Wi: new Float64Array(H),
               Wf: new Float64Array(H), Wo: new Float64Array(H),
               wy: new Float64Array(H),
               bg: new Float64Array(H), bi: new Float64Array(H),
               bf: new Float64Array(H), bo: new Float64Array(H), by: 0,
               UG: new Float64Array(H * H), UI: new Float64Array(H * H),
               UF: new Float64Array(H * H), UO: new Float64Array(H * H) };
    }
    let G = allocG(), forward_y = 0;
    function applyG(n) {
      const upd = (W, dW) => { for (let i = 0; i < W.length; i++) W[i] -= LR * dW[i] / n; };
      upd(Wg, G.Wg); upd(Wi, G.Wi); upd(Wf, G.Wf); upd(Wo, G.Wo); upd(wy, G.wy);
      upd(bg, G.bg); upd(bi, G.bi); upd(bf, G.bf); upd(bo, G.bo);
      by -= LR * G.by / n;
      upd(UG, G.UG); upd(UI, G.UI); upd(UF, G.UF); upd(UO, G.UO);
    }

    /* ---- training: EPOCHS batch passes; each epoch fits a seeded
       random subset of the training sequences (mini-batch SGD) ---- */
    const seqRng = U.mulberry32(U.hashStr(ctx.key + "|lstm|epochs|v3"));
    const SEQ_PER_EPOCH = 150;
    for (let ep = 0; ep < EPOCHS; ep++) {
      G = allocG();
      let count = 0;
      for (let s = 0; s < SEQ_PER_EPOCH; s++) {
        const t = L + Math.floor(seqRng() * (split - 1 - L));
        for (let q = 0; q < L; q++) XT[q] = x[t - L + q];
        cSt.fill(0); hSt.fill(0);
        forward_y = forward(L);
        backward(L, x[t]);
        count++;
      }
      applyG(count);
    }

    /* ---- one-step-ahead inference on the test window ---- */
    const pred = new Array(T);
    for (let i = split; i < N; i++) {
      for (let s = 0; s < L; s++) XT[s] = x[i - 1 - L + s];
      cSt.fill(0); hSt.fill(0);
      const y = forward(L);
      const rHat = mu + clip(y, 4) * sd;
      pred[i - split] = full[i - 1] * (1 + rHat);
    }
    return pred;
  }

  /* ---- target-date forecast: train once (cached), then roll the cell
     forward feeding its own predictions back as inputs ----
     (deliberately a self-contained copy of the training so the one-step
     evaluation protocol in predict() stays untouched)                  */
  let _fit = null;
  function getFit(ctx) {
    const tag = ctx.key + "|" + JSON.stringify(ctx.hyperparams || {});
    if (_fit && _fit.tag === tag) return _fit.m;
    const hp = (ctx.hyperparams && ctx.hyperparams.lstm) ||
      { epochs: 50, lookback: 30, lr: 0.005 };
    const L = Math.max(5, Math.min(60, hp.lookback | 0));
    const EPOCHS = Math.max(10, Math.min(100, hp.epochs | 0));
    const LR = Math.max(0.0005, Math.min(0.02, hp.lr));

    const full = ctx.close, N = full.length, split = N - ctx.TEST;
    const ret = new Array(N - 1);
    for (let i = 1; i < N; i++) ret[i - 1] = (full[i] - full[i - 1]) / full[i - 1];
    let mu = 0;
    for (let i = 0; i < split - 1; i++) mu += ret[i];
    mu /= (split - 1);
    let sd = 0;
    for (let i = 0; i < split - 1; i++) sd += (ret[i] - mu) * (ret[i] - mu);
    sd = Math.sqrt(sd / Math.max(1, split - 2)) || 1e-6;
    const x = ret.map(v => (v - mu) / sd);

    const rng = U.mulberry32(U.hashStr(ctx.key + "|lstm|init|v3"));
    const zeros = n => new Float64Array(n);
    const rand = n => { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = (rng() * 2 - 1) * 0.25; return a; };
    const Wg = rand(H), Wi = rand(H), Wf = rand(H), Wo = rand(H), wy = rand(H);
    const bg = rand(H), bi = rand(H), bf = rand(H), bo = rand(H);
    bf.fill(1);
    let by = 0;
    const UG = rand(H * H), UI = rand(H * H), UF = rand(H * H), UO = rand(H * H);

    const XT = new Float64Array(L);
    const CP = new Float64Array(L * H), HP = new Float64Array(L * H);
    const Gg = new Float64Array(L * H), Ii = new Float64Array(L * H);
    const Ff = new Float64Array(L * H), Oo = new Float64Array(L * H);
    const CN = new Float64Array(L * H), HN = new Float64Array(L * H);
    const THC = new Float64Array(L * H);
    const cSt = new Float64Array(H), hSt = new Float64Array(H);
    const dag = new Float64Array(H), dai = new Float64Array(H);
    const daf = new Float64Array(H), dao = new Float64Array(H);
    const dh = new Float64Array(H), dc = new Float64Array(H);
    const dhn = new Float64Array(H), dcn = new Float64Array(H);

    const sig = v => 1 / (1 + Math.exp(-v));
    const clip = (v, lim) => v > lim ? lim : (v < -lim ? -lim : v);

    function forward(len) {
      for (let t = 0; t < len; t++) {
        const xt = XT[t], o = t * H;
        for (let j = 0; j < H; j++) {
          CP[o + j] = cSt[j]; HP[o + j] = hSt[j];
          const ug = j * H;
          let sg = bg[j], si = bi[j], sf = bf[j], so = bo[j];
          for (let k = 0; k < H; k++) {
            const hk = hSt[k];
            sg += UG[ug + k] * hk; si += UI[ug + k] * hk;
            sf += UF[ug + k] * hk; so += UO[ug + k] * hk;
          }
          sg += Wg[j] * xt; si += Wi[j] * xt;
          sf += Wf[j] * xt; so += Wo[j] * xt;
          const gv = Math.tanh(sg);
          const iv = sig(si), fv = sig(sf), ov = sig(so);
          const cv = fv * cSt[j] + iv * gv;
          const hv = ov * Math.tanh(cv);
          Gg[o + j] = gv; Ii[o + j] = iv; Ff[o + j] = fv; Oo[o + j] = ov;
          CN[o + j] = cv; HN[o + j] = hv; THC[o + j] = Math.tanh(cv);
        }
        for (let j = 0; j < H; j++) { cSt[j] = CN[o + j]; hSt[j] = HN[o + j]; }
      }
      let y = by;
      const last = (len - 1) * H;
      for (let k = 0; k < H; k++) y += wy[k] * HN[last + k];
      return y;
    }

    function backward(len, tgt) {
      dh.fill(0); dc.fill(0);
      const dy = clip(2 * (forward_y - tgt), GRAD_CLIP);
      G.by += dy;
      const last = (len - 1) * H;
      for (let k = 0; k < H; k++) {
        G.wy[k] += dy * HN[last + k];
        dh[k] = dy * wy[k];
      }
      for (let t = len - 1; t >= 0; t--) {
        const o = t * H;
        dhn.fill(0);
        for (let j = 0; j < H; j++) {
          const thc = THC[o + j];
          const dCell = dc[j] + dh[j] * Oo[o + j] * (1 - thc * thc);
          dao[j] = clip(dh[j] * thc * Oo[o + j] * (1 - Oo[o + j]), GRAD_CLIP);
          daf[j] = clip(dCell * CP[o + j] * Ff[o + j] * (1 - Ff[o + j]), GRAD_CLIP);
          dai[j] = clip(dCell * Gg[o + j] * Ii[o + j] * (1 - Ii[o + j]), GRAD_CLIP);
          dag[j] = clip(dCell * Ii[o + j] * (1 - Gg[o + j] * Gg[o + j]), GRAD_CLIP);
          dcn[j] = dCell * Ff[o + j];
        }
        for (let j = 0; j < H; j++) {
          G.bg[j] += dag[j]; G.bi[j] += dai[j];
          G.bf[j] += daf[j]; G.bo[j] += dao[j];
          const xt = XT[t], ug = j * H;
          G.Wg[j] += dag[j] * xt; G.Wi[j] += dai[j] * xt;
          G.Wf[j] += daf[j] * xt; G.Wo[j] += dao[j] * xt;
          for (let k = 0; k < H; k++) {
            const hk = HP[o + k];
            G.UG[ug + k] += dag[j] * hk; G.UI[ug + k] += dai[j] * hk;
            G.UF[ug + k] += daf[j] * hk; G.UO[ug + k] += dao[j] * hk;
            dhn[k] += dag[j] * UG[ug + k] + dai[j] * UI[ug + k]
                    + daf[j] * UF[ug + k] + dao[j] * UO[ug + k];
          }
        }
        for (let j = 0; j < H; j++) { dc[j] = dcn[j]; dh[j] = dhn[j]; }
      }
    }

    function allocG() {
      return { Wg: new Float64Array(H), Wi: new Float64Array(H),
               Wf: new Float64Array(H), Wo: new Float64Array(H),
               wy: new Float64Array(H),
               bg: new Float64Array(H), bi: new Float64Array(H),
               bf: new Float64Array(H), bo: new Float64Array(H), by: 0,
               UG: new Float64Array(H * H), UI: new Float64Array(H * H),
               UF: new Float64Array(H * H), UO: new Float64Array(H * H) };
    }
    let G = allocG(), forward_y = 0;
    function applyG(n) {
      const upd = (W, dW) => { for (let i = 0; i < W.length; i++) W[i] -= LR * dW[i] / n; };
      upd(Wg, G.Wg); upd(Wi, G.Wi); upd(Wf, G.Wf); upd(Wo, G.Wo); upd(wy, G.wy);
      upd(bg, G.bg); upd(bi, G.bi); upd(bf, G.bf); upd(bo, G.bo);
      by -= LR * G.by / n;
      upd(UG, G.UG); upd(UI, G.UI); upd(UF, G.UF); upd(UO, G.UO);
    }

    const seqRng = U.mulberry32(U.hashStr(ctx.key + "|lstm|epochs|v3"));
    const SEQ_PER_EPOCH = 150;
    for (let ep = 0; ep < EPOCHS; ep++) {
      G = allocG();
      let count = 0;
      for (let s = 0; s < SEQ_PER_EPOCH; s++) {
        const t = L + Math.floor(seqRng() * (split - 1 - L));
        for (let q = 0; q < L; q++) XT[q] = x[t - L + q];
        cSt.fill(0); hSt.fill(0);
        forward_y = forward(L);
        backward(L, x[t]);
        count++;
      }
      applyG(count);
    }

    _fit = { tag, m: { L, Wg, Wi, Wf, Wo, wy, bg, bi, bf, bo, by, UG, UI, UF, UO, mu, sd, x, sig, clip } };
    return _fit.m;
  }

  function predictFuture(ctx, h) {
    const m = getFit(ctx);
    const sig = m.sig, clip = m.clip;
    const L = m.L, H6 = H;
    const XT = new Float64Array(L);
    const cSt = new Float64Array(H6), hSt = new Float64Array(H6);
    const Gg = new Float64Array(L * H6), Ii = new Float64Array(L * H6);
    const Ff = new Float64Array(L * H6), Oo = new Float64Array(L * H6);
    const CN = new Float64Array(L * H6), HN = new Float64Array(L * H6);
    const THC = new Float64Array(L * H6);

    function roll(seq) {
      for (let t = 0; t < L; t++) {
        const xt = seq[t], o = t * H6;
        for (let j = 0; j < H6; j++) {
          const ug = j * H6;
          let sg = m.bg[j], si = m.bi[j], sf = m.bf[j], so = m.bo[j];
          for (let k = 0; k < H6; k++) {
            const hk = hSt[k];
            sg += m.UG[ug + k] * hk; si += m.UI[ug + k] * hk;
            sf += m.UF[ug + k] * hk; so += m.UO[ug + k] * hk;
          }
          sg += m.Wg[j] * xt; si += m.Wi[j] * xt;
          sf += m.Wf[j] * xt; so += m.Wo[j] * xt;
          const fv = sig(sf), iv = sig(si), ov = sig(so);
          const gv = Math.tanh(sg);
          const cv = fv * cSt[j] + iv * gv;
          const hv = ov * Math.tanh(cv);
          Gg[o + j] = Math.tanh(sg); Ii[o + j] = sig(si);
          Ff[o + j] = sig(sf); Oo[o + j] = sig(so);
          CN[o + j] = cv; HN[o + j] = hv; THC[o + j] = Math.tanh(cv);
        }
        for (let j = 0; j < H6; j++) { cSt[j] = CN[o + j]; hSt[j] = HN[o + j]; }
      }
      let y = m.by;
      const last = (L - 1) * H6;
      for (let k = 0; k < H6; k++) y += m.wy[k] * HN[last + k];
      return y;
    }

    const prices = [];
    let price = ctx.close[ctx.close.length - 1];
    const win = m.x.slice(m.x.length - L);
    let firstR = 0;
    for (let s = 0; s < h; s++) {
      for (let i = 0; i < L; i++) XT[i] = win[i];
      cSt.fill(0); hSt.fill(0);
      const y = roll(XT);
      const rHat = m.mu + m.clip(y, 4) * m.sd;
      if (s === 0) firstR = rHat;
      price *= (1 + rHat);
      prices.push(price);
      win.push((rHat - m.mu) / m.sd);
      win.shift();
    }
    const last = ctx.close[ctx.close.length - 1];
    const pct = (prices[prices.length - 1] / last - 1) * 100;
    let mom = 0;
    for (let i = m.x.length - L; i < m.x.length; i++) mom += m.x[i];
    mom = (mom / L) * m.sd * 100;
    const reason =
      "The trained network's memory of the last " + L + " real sessions (net momentum " +
      (mom >= 0 ? "+" : "") + mom.toFixed(2) + "%) leads it to expect a " +
      (firstR >= 0 ? "rise" : "fall") + " of " + (firstR * 100).toFixed(2) + "% next session," +
      " compounding to " + (pct >= 0 ? "+" : "") + pct.toFixed(2) + "% by the chosen day.";
    return { prices, reason, pct };
  }

  ns.lstm = { id: "lstm", predict, predictFuture };
})(window.SPPAlgorithms = window.SPPAlgorithms || {});
