"""
loss_functions.py
=================
All loss components for the composite objective:

L_total = Huber(y_true, y_pred)
        + λ1 * L_monotonic
        + λ2 * L_stability
        + λ3 * L_shock
        + λ4 * L_regime
"""

import numpy as np
from scipy.special import rel_entr
from typing import Dict, Optional
from src.utils import get_logger

logger = get_logger("loss_functions")


# ── 1. Huber loss ─────────────────────────────────────────────────────────────

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """
    Huber loss (less sensitive to outliers than MSE).
        L_δ(r) = 0.5 * r²              if |r| ≤ δ
               = δ * (|r| - 0.5 * δ)  otherwise
    """
    r = y_true - y_pred
    mask = np.abs(r) <= delta
    loss = np.where(mask, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))
    return float(loss.mean())


# ── 2. Monotonicity loss ──────────────────────────────────────────────────────

def monotonicity_loss(
    multi_regime_gam,
    X: np.ndarray,
    regime_labels: np.ndarray,
    constraints_per_regime: Dict[int, Dict[str, int]],
) -> float:
    """
    L_mono = Σ_r Σ_j Σ_t max(0, -direction * ∂f_r/∂x_j)
    Aggregated across regimes.
    """
    return multi_regime_gam.total_monotonicity_penalty(
        X, regime_labels, constraints_per_regime
    )


# ── 3. Stability / perturbation loss ─────────────────────────────────────────

def stability_loss(
    multi_regime_gam,
    X: np.ndarray,
    regime_labels: np.ndarray,
    epsilon_std: float = 0.01,
    n_samples: int = 10,
    seed: int = 42,
) -> float:
    """
    L_stab = E[|| f(X + ε) - f(X) ||]
    where ε ~ N(0, epsilon_std²).

    Measures sensitivity to small input perturbations.
    """
    rng = np.random.default_rng(seed)
    f0 = multi_regime_gam.predict(X, regime_labels)
    diffs = []
    for _ in range(n_samples):
        X_perturbed = X + rng.normal(0.0, epsilon_std, size=X.shape)
        f_perturbed = multi_regime_gam.predict(X_perturbed, regime_labels)
        diffs.append(np.abs(f_perturbed - f0))
    return float(np.mean(diffs))


# ── 4. Shock loss ─────────────────────────────────────────────────────────────

def shock_loss(shock_model, l1_weight: float = 1.0) -> float:
    """
    L_shock = l1_weight * Σ_s |A_s|
    Encourages sparse shock amplitudes.
    """
    return float(l1_weight * np.sum(np.abs(shock_model.A)))


# ── 5. Regime consistency loss ────────────────────────────────────────────────

def regime_consistency_loss(regime_probs: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    L_regime = (1/T) Σ_t KL(p(r_t) || p(r_{t-1}))

    Penalizes abrupt regime jumps. Encourages smooth regime transitions.
    """
    p_t   = regime_probs[1:]   + epsilon
    p_tm1 = regime_probs[:-1]  + epsilon
    # Normalize
    p_t   = p_t   / p_t.sum(axis=1, keepdims=True)
    p_tm1 = p_tm1 / p_tm1.sum(axis=1, keepdims=True)
    kl = np.sum(rel_entr(p_t, p_tm1), axis=1)
    return float(kl.mean())


# ── 6. Composite loss ─────────────────────────────────────────────────────────

def composite_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multi_regime_gam,
    shock_model,
    regime_probs: np.ndarray,
    regime_labels: np.ndarray,
    X: np.ndarray,
    constraints_per_regime: Dict[int, Dict[str, int]],
    lambdas: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute all components of the composite loss.

    Returns dict with individual losses and 'total'.

    lambdas keys: huber_delta, lambda_monotonic, lambda_stability,
                  lambda_shock, lambda_regime, stability_epsilon_std,
                  stability_n_samples
    """
    delta   = lambdas.get("huber_delta", 1.0)
    lam1    = lambdas.get("lambda_monotonic", 0.5)
    lam2    = lambdas.get("lambda_stability", 0.3)
    lam3    = lambdas.get("lambda_shock",     0.2)
    lam4    = lambdas.get("lambda_regime",    0.1)
    eps_std = lambdas.get("stability_epsilon_std", 0.01)
    n_samp  = int(lambdas.get("stability_n_samples", 10))

    L_huber  = huber_loss(y_true, y_pred, delta)
    L_mono   = monotonicity_loss(multi_regime_gam, X, regime_labels, constraints_per_regime)
    L_stab   = stability_loss(multi_regime_gam, X, regime_labels, eps_std, n_samp)
    L_shock  = shock_loss(shock_model, lam3)
    L_regime = regime_consistency_loss(regime_probs)

    total = L_huber + lam1 * L_mono + lam2 * L_stab + lam3 * L_shock + lam4 * L_regime

    return {
        "huber":       L_huber,
        "monotonic":   L_mono,
        "stability":   L_stab,
        "shock":       L_shock,
        "regime_kl":   L_regime,
        "total":       total,
    }
