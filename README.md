# Constrained Interpretable Regime-Aware Modeling of Commodity Dynamics

A scientific machine learning system for modeling weekly log-returns of **Gold** and **WTI Oil** under macro-driven regimes and exogenous shocks.

**Constraints enforced:**
- No neural networks or embeddings — classical ML only (HMM + pyGAM)
- Monotonicity constraints on macro sensitivities
- Bounded gradient sensitivity
- Exogenous shock amplitudes are L1-regularized
- Training data is strictly local CSV (gold/oil from Kaggle); Yahoo Finance is test-only

---

## Pipeline Overview

End-to-end flow: **Data → Preprocess → Regime (HMM) → GAM + Shocks → Evaluate → Visualize**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAIN PIPELINE (main.py --mode train)                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. Load data        → Local CSVs from Kaggle (gold/oil), FRED + yfinance macro   │
│  2. Align & preprocess → Weekly alignment, rolling z-score, chronological split │
│  3. Fit HMM          → K=3 regimes on macro state X_t                           │
│  4. Fit GAMs         → One constrained LinearGAM per regime                      │
│  5. Fit shock model  → Gaussian kernel shocks on residuals                      │
│  6. Alternating opt  → GAM ↔ shock updates (n_alternating_iter)                 │
│  7. Evaluate         → Validation metrics + baselines (ARIMA, LR, GAM)          │
│  8. Save             → models/*.pkl                                             │
│  9. Visualize        → regime timeline, PDPs, shock magnitude, stability         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TEST PIPELINE (main.py --mode test)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. Load frozen model → models/hmm_*.pkl, gam_*_regime*.pkl, shock_*.pkl         │
│  2. Fetch Yahoo       → 2020–present weekly (GC=F, CL=F + macro tickers)         │
│  3. Preprocess        → Same rolling z-score (using saved norm_stats)            │
│  4. Predict           → y_pred = f_r(X) + u_t                                    │
│  5. Evaluate          → RMSE, MAE, R²; append to yahoo_test_results.csv         │
│  6. Visualize         → actual vs predicted, shock magnitude, regime timeline    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Pipeline stages (code → step):**

| Stage | Module | What it does |
|-------|--------|--------------|
| Load prices | `data_loader.load_gold` / `load_oil` | Read local CSV from Kaggle (gold/oil), compute weekly `log_return` |
| Load macro | `data_loader.load_macro` | FRED (real_rate, inflation_exp) + yfinance (DXY, VIX, liquidity, demand_proxy) |
| Align | `preprocessing.align_weekly` | Merge to Friday weekly index, forward-fill macro |
| Normalize | `preprocessing.rolling_zscore` | 104-week rolling z-score (no lookahead) |
| Split | `preprocessing.chronological_split` | Train to 2016, Val 2017–2019 |
| Regimes | `regime_model.RegimeModel` | Gaussian HMM (K=3), persistence ≥ 0.85 |
| GAM | `gam_model.MultiRegimeGAM` | One LinearGAM per regime; monotonicity + gradient bound |
| Shocks | `shock_model.ShockModel` | Gaussian kernels at config events; L1 on amplitudes |
| Orchestrate | `composite_model.CompositeModel` | fit: HMM → GAMs → shocks → alternating; predict: f_r(X)+u |
| Loss | `loss_functions.composite_loss` | Huber + monotonicity + stability + shock + regime terms |
| Evaluate | `evaluation.evaluate_all` | RMSE, MAE, R², constraint checks |
| Baselines | `baselines.run_all_baselines` | ARIMA, Linear Regression, unconstrained GAM |
| Yahoo test | `yahoo_testing.run_yahoo_test` | Load model, fetch Yahoo, predict, save metrics |
| Plots | `visualization` | Regime timeline, PDPs, shock magnitude, actual vs predicted |

---

## Quickstart

```bash
# 1. Activate virtual environment (Windows)
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train (uses Kaggle gold/oil CSVs + FRED/yfinance macro)
python main.py --mode train --asset both

# 4. Out-of-sample test (frozen Yahoo Finance data, 2020–present)
python main.py --mode test --asset both
```

**Options:**
- `--asset gold | oil | both` — which asset(s) to run (default: both)
- `--config PATH` — config YAML (default: `configs/config.yaml`)

---

## Model Architecture

```
y_t = f_{r_t}(X_t)  +  u_t  +  ε_t
      └── GAM        └── Shock   └── noise
           └─ regime-specific (K=3 HMM)
```

| Component | Method | Constraints |
|-----------|--------|-------------|
| Regime r_t | Gaussian HMM (K=3) | Persistence ≥ 0.85 |
| f_r(X_t) | pyGAM LinearGAM (10 splines/var) | Monotonicity, ‖∂f/∂X‖₂ ≤ M |
| u_t | Gaussian kernel shocks | τ ∈ [1,52] wks, L1(A) |

---

## Project Structure

```
SciML Project/
├── gold/             ← Kaggle: XAU weekly data (XAU_1w_data.csv)
├── Oil/              ← Kaggle: Brent oil prices (BrentOilPrices.csv)
├── data/
│   ├── raw/          ← macro_cache.pkl + Yahoo cache
│   └── processed/    ← yahoo_test_results.csv
├── src/
│   ├── utils.py
│   ├── data_loader.py      ← FRED + yfinance macro fetch
│   ├── preprocessing.py    ← z-score, alignment, splits
│   ├── regime_model.py     ← HMM
│   ├── gam_model.py        ← constrained GAM per regime
│   ├── shock_model.py      ← Gaussian kernel shocks
│   ├── loss_functions.py   ← composite loss
│   ├── composite_model.py  ← orchestrator
│   ├── baselines.py        ← ARIMA, LR, unconstrained GAM
│   ├── evaluation.py       ← metrics
│   ├── yahoo_testing.py    ← frozen out-of-sample
│   └── visualization.py   ← all plots
├── configs/config.yaml     ← all hyperparameters
├── models/                 ← saved model artifacts (.pkl)
├── outputs/                ← all plots (.png)
├── main.py                 ← entry point (train / test)
└── requirements.txt
```

---

## Data Sources

### Training data (price series) — Kaggle

Gold and Brent Oil price data used for training are **downloaded from Kaggle**. Place the CSV files in the paths expected by the config:

| Asset | Config path | Description |
|-------|-------------|-------------|
| Gold | `gold/XAU_1w_data.csv` | Weekly gold (XAU) price series |
| Oil | `Oil/BrentOilPrices.csv` | Brent oil prices (daily; code resamples to weekly) |

Download the datasets from Kaggle and extract them into the `gold/` and `Oil/` folders in the project root. Out-of-sample evaluation uses Yahoo Finance (see Test pipeline); training uses only these local Kaggle CSVs.

### Macro data — FRED & yfinance

| Variable | Source | Series |
|----------|--------|--------|
| Real interest rate | FRED | `DFII10` (10-yr TIPS) |
| Inflation expectations | FRED | `T10YIE` (breakeven) |
| DXY | yfinance | `DX-Y.NYB` |
| VIX | yfinance | `^VIX` |
| Liquidity proxy | yfinance | `^TNX` (10-yr Treasury) |
| Oil demand proxy | yfinance | `USO` (oil only) |

> **Note:** FRED access does not require an API key for `pandas-datareader`. For higher rate limits, set `FRED_API_KEY` in your environment.

---

## Outputs

**After `--mode train`:**
- `models/hmm_{asset}.pkl`, `models/gam_{asset}_regime{r}.pkl`, `models/shock_{asset}.pkl`
- `outputs/{asset}_regime_timeline.png`
- `outputs/{asset}_pdp_{feature}.png` (one per macro variable)
- `outputs/{asset}_shock_magnitude.png`
- `outputs/{asset}_stability_{feature}.png`

**After `--mode test`:**
- `data/processed/yahoo_test_results.csv`
- `data/raw/yahoo_test_cache_{asset}.pkl`
- `outputs/yahoo_{asset}_actual_vs_predicted.png`

---

## Key plots

The pipeline produces the following visualizations (saved under `outputs/`). To display them in the README on GitHub, copy the desired PNGs into `docs/plots/` and commit.

| Plot | Description | Output file |
|------|-------------|-------------|
| **Regime timeline** | HMM regime assignment over time; optional price overlay. | `{asset}_regime_timeline.png` |
| **Actual vs predicted** | Out-of-sample Yahoo: actual vs predicted log-returns, GAM and shock components, prediction error. | `yahoo_{asset}_actual_vs_predicted.png` |
| **Shock magnitude** | Fitted exogenous shock \(u_t\) over time with event centers (e.g. 2008 crisis, COVID, 2022). | `{asset}_shock_magnitude.png` |
| **Partial dependence (PDP)** | Effect of each macro variable on the predicted return (per regime); reflects monotonicity constraints. | `{asset}_pdp_{feature}.png` |
| **Stability** | Sensitivity of predictions to small perturbations in a feature (bounded gradient). | `{asset}_stability_{feature}.png` |

---

## Reproducibility

- `seed = 42` set globally (numpy + Python random)
- Yahoo download timestamp and ticker logged and cached
- No data shuffling at any stage
- Chronological splits: Train 2000–2016, Val 2017–2019
