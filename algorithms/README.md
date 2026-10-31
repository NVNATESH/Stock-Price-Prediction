# Algorithms

Hand-written prediction engines for the dashboard. Every algorithm here is
**implemented from scratch in vanilla JavaScript** — no ML libraries, no imports,
only `Math.*` and the small shared numeric utilities in `common.js` (seeded RNG,
Gaussian noise, MSE/RMSE, and a hand-written ridge least-squares solver).

Each file registers one engine on the global `window.SPPAlgorithms` namespace and is
loaded by `index.html` **before** `assets/js/app.js` (plain scripts, no build step,
works from `file://` and any static server).

| File | Algorithm | What is genuinely implemented |
|---|---|---|
| `common.js` | Shared numeric infrastructure | FNV-1a hash → mulberry32 PRNG, Box–Muller noise, MSE/RMSE, ridge normal-equations solver (Gaussian elimination) |
| `arima.js` | ARIMA(p, d, q) | Differencing (d = 0/1/2), AR coefficients and MA coefficients fitted by conditional least squares, one-step-ahead recursion with residual feedback |
| `prophet.js` | Prophet-style additive decomposition | Piecewise-linear trend (6 hinge changepoints) + Fourier seasonality (5-session week, 21-session month), fitted on returns by ridge least squares |
| `lstm.js` | LSTM recurrent network | Real LSTM cell (6 hidden units, 4 gates), seeded weight init, trained by batch gradient descent with **manual backpropagation through time**, allocation-free forward/backward buffers |
| `xgboost.js` | Gradient-boosted regression trees | 80 boosting rounds (η = 0.1) of hand-grown regression trees: greedy quantile-bin split search, max depth from the tuner, seeded row subsampling |
| `svr.js` | Support Vector Regression | ε-SVR dual solved by coordinate descent over a precomputed RBF kernel matrix (441 × 441), ε-tube and box constraint C |

## Common interface

Every engine exposes a single method:

```js
predict(ctx)   // ctx = { key, close, TEST, hyperparams } → 63 price predictions
```

* `key` — ticker key (seeds the RNG, so training is deterministic and reproducible)
* `close` — the full 504-session close series
* `TEST` — held-out test-window length (63)
* `hyperparams` — the live tuner state (LSTM epochs/lookback/learning rate, ARIMA
  (p, d, q), XGBoost depth/subsample — all of which genuinely change the training)

All engines follow the same protocol: fit on the first 441 sessions, predict the
final 63 sessions one step ahead (each prediction may use the actual history up to
the previous session — never the future), and anchor the predicted return to the
previous close.

## Results behaviour (what to expect)

On daily equity series the predictable component is tiny, so every honest engine
lands near the *random-walk floor* (the RMSE of simply predicting "tomorrow =
today"). The measured differences between engines are small — see Chapter 6 of
`docs/PROJECT_REPORT.md` for the recorded tables.

## Swapping in real data / other models

Replace the body of any `predict(ctx)` with a lookup of offline-computed
predictions (e.g. JSON produced by Python `statsmodels` / `Prophet` / `Keras` /
`XGBoost` / `scikit-learn`). The dashboard's scoring, backtesting and visualization
layers need no other change.
