# Python Reference Notebooks

Jupyter notebooks that walk through the **same five algorithms as library-grade
Python implementations** — the reference point the hand-written JavaScript
engines in [`algorithms/`](../algorithms/) were validated against.

| Notebook | Algorithm | Library used |
|---|---|---|
| `ARIMA.ipynb` | ARIMA(p, d, q) | `statsmodels` |
| `Prophet.ipynb` | Prophet additive decomposition | `prophet` |
| `LSTM.ipynb` | LSTM recurrent network | `tensorflow` / `keras` |
| `XG_BOOST.ipynb` | Gradient-boosted trees | `xgboost` |
| `svr.ipynb` | Support Vector Regression | `scikit-learn` |

## Why these exist

The dashboard's JavaScript engines are hand-written with no ML libraries.
These notebooks serve two purposes:

1. **Cross-check** — run a library implementation on the same data and compare
   its behaviour against the JS engine's (they operate at the same scale, since
   daily equity moves are dominated by noise in both).
2. **Learning aid** — each notebook walks through the algorithm's standard
   workflow step by step, mirroring the report's methodology chapter.

## Run them

```bash
pip install pandas numpy matplotlib statsmodels prophet tensorflow xgboost scikit-learn jupyter
jupyter notebook
```

Notebooks are self-contained and do not affect the dashboard — the site runs
entirely from the JavaScript engines.
