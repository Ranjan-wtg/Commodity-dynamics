"""
baselines.py
============
Classical baseline models for benchmark comparison.

Baselines:
  1. ARIMA(1,1,1) — via statsmodels
  2. Linear Regression — via sklearn
  3. Unconstrained GAM — via pyGAM (no monotonicity/gradient constraints)

NO neural networks.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional
from src.utils import get_logger

logger = get_logger("baselines")


# ── 1. ARIMA ─────────────────────────────────────────────────────────────────

def fit_arima(
    y_train: np.ndarray,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> object:
    """
    Fit ARIMA(p,d,q) on training log returns.

    Returns fitted statsmodels ARIMA results object.
    """
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y_train, order=order)
        result = model.fit()

    logger.info(f"ARIMA{order} fitted. AIC={result.aic:.4f}")
    return result


def predict_arima(arima_result, n_steps: int) -> np.ndarray:
    """Produce n_steps out-of-sample forecasts from a fitted ARIMA."""
    forecast = arima_result.forecast(steps=n_steps)
    return np.array(forecast)


# ── 2. Linear Regression ──────────────────────────────────────────────────────

def fit_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """
    Fit OLS linear regression on macro features.
    Returns sklearn LinearRegression fitted model.
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info(
        f"LinearRegression fitted. "
        f"R²_train={model.score(X_train, y_train):.4f}"
    )
    return model


def predict_linear(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# ── 3. Unconstrained GAM ──────────────────────────────────────────────────────

def fit_unconstrained_gam(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splines: int = 10,
    lam: float = 0.6,
) -> object:
    """
    Fit a pyGAM LinearGAM with no monotonicity or gradient constraints.
    """
    from pygam import LinearGAM, s

    n_features = X_train.shape[1]
    terms = s(0, n_splines=n_splines, lam=lam)
    for j in range(1, n_features):
        terms = terms + s(j, n_splines=n_splines, lam=lam)

    gam = LinearGAM(terms, max_iter=200)
    gam.fit(X_train, y_train)
    logger.info(f"Unconstrained GAM fitted. n_obs={len(y_train)}")
    return gam


def predict_unconstrained_gam(gam, X: np.ndarray) -> np.ndarray:
    return gam.predict(X)


# ── Evaluate all baselines ────────────────────────────────────────────────────

def run_all_baselines(
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Fit and evaluate all baselines on validation set.

    Returns dict: {baseline_name: {'model': ..., 'val_pred': ..., 'val_rmse': ...}}
    """
    from src.evaluation import rmse

    results = {}

    # ARIMA
    try:
        arima_res = fit_arima(y_train)
        arima_pred = predict_arima(arima_res, n_steps=len(y_val))
        results["arima"] = {
            "model": arima_res,
            "val_pred": arima_pred,
            "val_rmse": rmse(y_val, arima_pred),
        }
        logger.info(f"ARIMA val RMSE: {results['arima']['val_rmse']:.6f}")
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")

    # Linear Regression
    try:
        lr_model = fit_linear(X_train, y_train)
        lr_pred  = predict_linear(lr_model, X_val)
        results["linear"] = {
            "model": lr_model,
            "val_pred": lr_pred,
            "val_rmse": rmse(y_val, lr_pred),
        }
        logger.info(f"Linear val RMSE: {results['linear']['val_rmse']:.6f}")
    except Exception as e:
        logger.warning(f"Linear Regression failed: {e}")

    # Unconstrained GAM
    try:
        ugam = fit_unconstrained_gam(X_train, y_train)
        ugam_pred = predict_unconstrained_gam(ugam, X_val)
        results["unconstrained_gam"] = {
            "model": ugam,
            "val_pred": ugam_pred,
            "val_rmse": rmse(y_val, ugam_pred),
        }
        logger.info(f"Unconstrained GAM val RMSE: {results['unconstrained_gam']['val_rmse']:.6f}")
    except Exception as e:
        logger.warning(f"Unconstrained GAM failed: {e}")

    return results
