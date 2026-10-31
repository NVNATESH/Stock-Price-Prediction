# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/) · versioning: [SemVer](https://semver.org/).

## [1.0.0] — 2026-10-31

### Added
- Five forecasting engines implemented from scratch in vanilla JavaScript
  (no ML libraries): ARIMA(p,d,q) conditional least squares, Prophet-style
  additive decomposition, LSTM with hand-written backpropagation through
  time, XGBoost-style boosted regression trees, ε-SVR dual coordinate descent.
- Real market data layer: daily OHLCV for TCS.NS, RELIANCE.NS and GOOGL
  (2024-10-28 → 2026-10-28) embedded from Yahoo Finance — no runtime
  network calls; seeded simulator as fallback.
- Interactive dashboard: custom SVG charting engine, 3-D candlestick hero,
  technical indicators (SMA, Wilder RSI, EMA, MACD), CSV exports,
  print stylesheet, keyboard shortcuts, plain-English glossary.
- Model Scoreboard: MSE / RMSE / directional accuracy on a 63-session
  holdout, Diebold–Mariano significance test, leadership timeline,
  XGBoost feature-importance extraction.
- Pick-a-Day: any chosen future date priced by all five models with
  per-model plain-English reasons and 90 % confidence intervals.
- Trading simulation (long/flat vs buy-and-hold with Sharpe ratio) and
  50-path Monte Carlo GBM simulation with 95 % Value-at-Risk.
- Hyperparameter Studio with live re-training and re-scoring.
- Python reference notebooks for all five models; academic report
  (Markdown + PDF) and themed review presentation.

### Notes
- Evaluation protocol: chronological split — last 63 sessions held out;
  no shuffling, no leakage. Naive baseline reported as the reference floor.
- Recorded results and the honesty statement (what is real vs simulated)
  are documented in `docs/PROJECT_REPORT.md` §6–§7.
