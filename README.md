# Stock Price Prediction — Time-Series Forecasting Dashboard

A beginner-friendly, single-page web dashboard for an academic machine-learning project:
**"Stock Price Prediction Using Time-Series Forecasting Models."**

It compares five prediction models — **ARIMA, Prophet, LSTM, XGBoost and SVR** —
all implemented from scratch in vanilla JavaScript — against **real historical
market data** (manually embedded daily OHLCV for TCS, Reliance Industries and
Google, collected once from Yahoo Finance).

- 📘 **Full project report:** [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)
- 📄 **Submission-ready Word report:** [docs/PROJECT_REPORT.docx](docs/PROJECT_REPORT.docx)
- 🖼️ **Dashboard screenshots:** [docs/screenshots/](docs/screenshots/)

## How to Run

No installation needed — it is pure HTML/CSS/JavaScript.

**Option 1:** double-click `index.html` (opens in your browser).

**Option 2 (recommended):** serve it locally so charts measure correctly:

```bash
# from this folder
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

## How to Use (4 steps)

1. **Pick a stock** — click TCS / RELIANCE / GOOGL. Everything updates instantly.
2. **Train the models** — click **Run Model** on a card, or **Train All Models**.
3. **Compare accuracy** — the *Model Scoreboard* section ranks all models by error
   (RMSE = typical per-day miss in ₹/$).
4. **See the forecast** — choose 7 / 30 / 90 days and view the projected price,
   the likely range (shaded band), or 50 random what-if futures (Monte Carlo).
5. **Pick a day** — choose any future date in the *Pick a Day* section and every
   model gives its predicted price for that exact day, each with a plain-English
   reason and a 90 % confidence interval from its own tested errors.
6. **Go deeper** — the scoreboard adds a Diebold–Mariano significance test, a
   monthly leadership timeline, and XGBoost feature-importance bars.

New to the jargon? Click the **book icon** in the top bar (or press **G**) for a
plain-English glossary of every term used.

## Project Structure

```
ML PROJECT/
├── index.html              # page structure and content
├── README.md               # this guide
├── data/
│   └── market-data.js      # ⭐ real embedded daily OHLCV (Yahoo Finance)
├── notebooks/              # Python reference implementations (Jupyter)
│   ├── ARIMA.ipynb         # statsmodels-style ARIMA walkthrough
│   ├── LSTM.ipynb          # Keras-style LSTM walkthrough
│   ├── Prophet.ipynb       # Prophet walkthrough
│   ├── XG_BOOST.ipynb      # XGBoost walkthrough
│   └── svr.ipynb           # scikit-learn SVR walkthrough
├── algorithms/             # ⭐ all prediction engines (separate folder)
│   ├── README.md           # module interface documentation
│   ├── common.js           # shared runtime: seeded RNG, noise, MSE/RMSE
│   ├── arima.js            # ARIMA(p, d, q) engine
│   ├── prophet.js          # Prophet decomposition engine
│   ├── lstm.js             # LSTM sequential-memory engine
│   ├── xgboost.js          # XGBoost boosted-trees engine
│   └── svr.js              # SVR kernel-regression engine
├── assets/
│   ├── css/styles.css      # design system, responsive + print styles
│   └── js/app.js           # data, charts, UI — consumes SPPAlgorithms
└── docs/
    ├── PROJECT_REPORT.md   # full academic project report
    ├── PROJECT_REPORT.docx # Word report for submission
    └── screenshots/        # captured dashboard views
```

The UI loads every engine from `algorithms/` before `app.js`. Each engine exposes
`predict({ key, close, TEST, hyperparams })`, so the folder is independently testable
and swappable — see [algorithms/README.md](algorithms/README.md).

## What Each Section Shows

| Section | Plain-English purpose |
|---|---|
| How This Dashboard Works | The 4-step tour for first-time visitors |
| The Stock Data Models Learn From | Raw daily prices (Open/High/Low/Close + Volume) |
| Choose a Company | Switch the whole dashboard to another stock |
| Price History & Trend Guides | Closing price, volume, moving averages, RSI, MACD |
| Meet the 5 Prediction Models | What each model is, in one sentence each |
| Actual vs Predicted | Overlay of real prices vs each model's predictions |
| Model Scoreboard | MSE / RMSE ranking — who predicted best |
| Trading Simulation | Would following the model have made money vs buy-and-hold? |
| Forecast | Future price path, confidence band, Monte Carlo futures |

## Honest Note (important for reports/viva)

The price data is **real**: genuine daily OHLCV history (2024-10-28 → 2026-10-28)
collected once from Yahoo Finance and embedded manually in `data/market-data.js` —
the dashboard makes **no network calls at runtime**. The five prediction engines in
`algorithms/` are **hand-written implementations with no ML libraries** — the LSTM
really trains by backpropagation, the trees really boost, the SVR really solves its
dual — and everything downstream (MSE/RMSE + directional-accuracy scoring,
indicators SMA/RSI/EMA/MACD, the backtest, the Monte Carlo simulation) is genuinely
computed on the real prices. Chapter 7 of the report documents the data provenance
and how to refresh it to the latest sessions.

## Keyboard Shortcuts

`1/2/3` select stock · `T` train all · `G` plain-English guide ·
`M` toggle moving averages · `F` fullscreen chart · `P` print report · `?` all shortcuts
