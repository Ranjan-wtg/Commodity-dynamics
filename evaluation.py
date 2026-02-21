"""
evaluation.py
=============
Evaluation metrics for model outputs.

Metrics computed:
  - RMSE
  - MAE
  - Directional accuracy
  - Stability index (mean output variance under perturbation)
  - Constraint violation score (fraction of monotonicity violations)
  - Shock contribution magnitude (mean |u_t|)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.utils import get_logger

logger = get_logger("evaluation")


# ── Point metrics ─────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of time steps where sign(y_true) == sign(y_pred).
    Measures whether the model gets the direction of movement right.
    """
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(correct.mean())


# ── Stability index ───────────────────────────────────────────────────────────

def stability_index(
    multi_regime_gam,
    X: np.ndarray,
    regime_labels: np.ndarray,
    n_trials: int = 50,
    epsilon_std: float = 0.01,
    seed: int = 42,
) -> float:
    """
    Mean variance of model output under random input perturbations.
    Lower = more stable.

    Stability Index = mean_t [ Var_trials(f(X_t + ε)) ]
    """
    rng = np.random.default_rng(seed)
    T = len(X)
    outputs = np.zeros((n_trials, T))

    for k in range(n_trials):
        X_perturbed = X + rng.normal(0.0, epsilon_std, size=X.shape)
        outputs[k] = multi_regime_gam.predict(X_perturbed, regime_labels)

    return float(outputs.var(axis=0).mean())


# ── Constraint violation score ────────────────────────────────────────────────

def constraint_violation_score(
    multi_regime_gam,
    X: np.ndarray,
    regime_labels: np.ndarray,
    constraints_per_regime: Dict[int, Dict[str, int]],
) -> float:
    """
    Fraction of (timestep, feature) pairs that violate monotonicity constraints.
    0.0 = fully constraint-consistent; 1.0 = all violated.
    """
    total_checks = 0
    total_violations = 0

    for r, gam_r in multi_regime_gam.gams.items():
        mask = regime_labels == r
        if mask.sum() == 0:
            continue
        cons = constraints_per_regime.get(r, {})
        X_r = X[mask]
        for feat_name, direction in cons.items():
            if feat_name not in gam_r.feature_names:
                continue
            j = gam_r.feature_names.index(feat_name)
            deriv = gam_r.numerical_derivative(X_r, j)
            violation = (-direction * deriv) > 0
            total_violations += int(violation.sum())
            total_checks += len(deriv)

    if total_checks == 0:
        return 0.0
    return float(total_violations / total_checks)


# ── Shock contribution ────────────────────────────────────────────────────────

def shock_contribution_magnitude(
    shock_model,
    dates: pd.DatetimeIndex,
) -> float:
    """Mean absolute shock contribution over a date range."""
    return shock_model.shock_magnitude(dates)


# ── Combined evaluator ────────────────────────────────────────────────────────

def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multi_regime_gam,
    shock_model,
    regime_labels: np.ndarray,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    constraints_per_regime: Dict[int, Dict[str, int]],
    asset: str = "unknown",
) -> Dict[str, float]:
    """
    Compute and log all evaluation metrics.
    Returns dict suitable for saving to CSV.
    """
    metrics = {
        "asset":                      asset,
        "rmse":                       rmse(y_true, y_pred),
        "mae":                        mae(y_true, y_pred),
        "directional_accuracy":       directional_accuracy(y_true, y_pred),
        "stability_index":            stability_index(multi_regime_gam, X, regime_labels),
        "constraint_violation_score": constraint_violation_score(
                                          multi_regime_gam, X, regime_labels,
                                          constraints_per_regime
                                      ),
        "shock_contribution_magnitude": shock_contribution_magnitude(shock_model, dates),
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"  Evaluation Results — {asset.upper()}")
    logger.info(f"{'='*55}")
    for k, v in metrics.items():
        if k != "asset":
            logger.info(f"  {k:<35} {v:.6f}")
    logger.info(f"{'='*55}")

    return metrics
