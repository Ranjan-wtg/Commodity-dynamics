"""
gam_model.py
============
Regime-specific constrained Generalized Additive Models.

For each regime r:
    f_r(X_t) = Σ_j g_{r,j}(X_{t,j})

Constraints implemented:
  1. Monotonicity (Gold: real_rate ≤ 0 in tightening, DXY ≤ 0 always;
                   Oil:  demand_proxy ≥ 0 in growth regime)
     → Numerical derivative check + L_mono = Σ max(0, violation)
  2. Bounded sensitivity: ||∂f/∂X||_2 ≤ M
     → Gradient penalty added to loss

Uses pyGAM LinearGAM with 10 splines per variable.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pygam import LinearGAM, s
from src.utils import get_logger, save_artifact, load_artifact

logger = get_logger("gam_model")


def _build_gam(n_features: int, n_splines: int = 10, lam: float = 0.6) -> LinearGAM:
    """Build a pyGAM LinearGAM with n_splines per feature."""
    terms = s(0, n_splines=n_splines, lam=lam)
    for j in range(1, n_features):
        terms = terms + s(j, n_splines=n_splines, lam=lam)
    return LinearGAM(terms, max_iter=200)


class RegimeGAM:
    """
    Wrapper around pyGAM LinearGAM for a single regime.

    Parameters
    ----------
    regime_id : int
    feature_names : list of str
    n_splines : int
    lam : float  (smoothing penalty)
    """

    def __init__(
        self,
        regime_id: int,
        feature_names: List[str],
        n_splines: int = 10,
        lam: float = 0.6,
    ):
        self.regime_id = regime_id
        self.feature_names = feature_names
        self.n_splines = n_splines
        self.lam = lam
        self.gam: Optional[LinearGAM] = None
        self.n_features = len(feature_names)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimeGAM":
        """Fit GAM on (X, y) for this regime."""
        if len(y) < 5:
            logger.warning(
                f"Regime {self.regime_id}: too few samples ({len(y)}). "
                "Skipping GAM fit."
            )
            return self
        logger.info(
            f"Fitting GAM [regime {self.regime_id}]: "
            f"n_obs={len(y)}, n_features={self.n_features}"
        )
        self.gam = _build_gam(self.n_features, self.n_splines, self.lam)
        self.gam.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return f_r(X) predictions."""
        if self.gam is None:
            return np.zeros(len(X))
        return self.gam.predict(X)

    # ── Partial dependence ────────────────────────────────────────────────────

    def partial_dependence(
        self, feature_idx: int, grid_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial dependence of f_r on feature j.
        Returns (grid, pdp_values).
        """
        if self.gam is None:
            raise RuntimeError("GAM not fitted.")
        XX = self.gam.generate_X_grid(term=feature_idx, n=grid_points)
        pdp = self.gam.partial_dependence(term=feature_idx, X=XX)
        grid = XX[:, feature_idx]
        return grid, pdp

    # ── Monotonicity check & penalty ─────────────────────────────────────────

    def numerical_derivative(
        self, X: np.ndarray, feature_idx: int, eps: float = 1e-4
    ) -> np.ndarray:
        """
        Compute numerical derivative ∂f/∂x_j at each row of X.
        Uses central differences.
        """
        X_plus  = X.copy(); X_plus[:, feature_idx]  += eps
        X_minus = X.copy(); X_minus[:, feature_idx] -= eps
        return (self.predict(X_plus) - self.predict(X_minus)) / (2 * eps)

    def monotonicity_penalty(
        self,
        X: np.ndarray,
        constraints: Dict[str, int],
    ) -> float:
        """
        Compute L_mono = Σ_j Σ_t max(0, violation_jt).

        constraints : dict {feature_name: direction}
            direction = -1  (non-increasing: ∂f/∂x ≤ 0)
            direction = +1  (non-decreasing: ∂f/∂x ≥ 0)
        """
        if self.gam is None or len(X) == 0:
            return 0.0

        total_penalty = 0.0
        for feat_name, direction in constraints.items():
            if feat_name not in self.feature_names:
                continue
            j = self.feature_names.index(feat_name)
            deriv = self.numerical_derivative(X, j)
            # violation: direction=-1 → deriv > 0 is a violation
            #            direction=+1 → deriv < 0 is a violation
            violation = -direction * deriv   # positive when violated
            total_penalty += np.maximum(0.0, violation).sum()

        return float(total_penalty)

    def violation_count(
        self,
        X: np.ndarray,
        constraints: Dict[str, int],
    ) -> Dict[str, int]:
        """Count number of time steps where monotonicity is violated per feature."""
        counts = {}
        if self.gam is None or len(X) == 0:
            return counts
        for feat_name, direction in constraints.items():
            if feat_name not in self.feature_names:
                continue
            j = self.feature_names.index(feat_name)
            deriv = self.numerical_derivative(X, j)
            violation = -direction * deriv
            counts[feat_name] = int((violation > 0).sum())
        return counts

    # ── Gradient / sensitivity penalty ───────────────────────────────────────

    def gradient_norm(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Compute ||∂f/∂X||_2 (Euclidean norm of gradient) at each row.
        Returns array of shape (T,).
        """
        if self.gam is None:
            return np.zeros(len(X))
        grads = np.stack(
            [self.numerical_derivative(X, j, eps) for j in range(self.n_features)],
            axis=1,
        )
        return np.linalg.norm(grads, axis=1)

    def gradient_penalty(
        self, X: np.ndarray, M: float = 2.0, eps: float = 1e-4
    ) -> float:
        """
        L_grad = Σ_t max(0, ||∂f/∂X||_2 - M)
        Penalizes gradient norms exceeding bound M.
        """
        if self.gam is None or len(X) == 0:
            return 0.0
        norms = self.gradient_norm(X, eps)
        return float(np.maximum(0.0, norms - M).sum())

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str) -> "RegimeGAM":
        return load_artifact(path)


# ── Multi-regime GAM container ────────────────────────────────────────────────

class MultiRegimeGAM:
    """
    Fits and stores one RegimeGAM per regime (K regimes total).

    Usage
    -----
    mrg = MultiRegimeGAM(n_regimes=3, feature_names=[...])
    mrg.fit(X_train, y_train, regime_labels_train)
    y_hat = mrg.predict(X, regime_labels)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        feature_names: List[str] = None,
        n_splines: int = 10,
        lam: float = 0.6,
    ):
        self.n_regimes = n_regimes
        self.feature_names = feature_names or []
        self.n_splines = n_splines
        self.lam = lam
        self.gams: Dict[int, RegimeGAM] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
    ) -> "MultiRegimeGAM":
        """Fit one GAM per regime using data split by regime label."""
        for r in range(self.n_regimes):
            mask = regime_labels == r
            X_r, y_r = X[mask], y[mask]
            gam_r = RegimeGAM(
                regime_id=r,
                feature_names=self.feature_names,
                n_splines=self.n_splines,
                lam=self.lam,
            )
            gam_r.fit(X_r, y_r)
            self.gams[r] = gam_r
        return self

    def predict(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Predict f_r(X_t) for each observation using its assigned regime.
        """
        y_hat = np.zeros(len(X))
        for r, gam_r in self.gams.items():
            mask = regime_labels == r
            if mask.sum() > 0:
                y_hat[mask] = gam_r.predict(X[mask])
        return y_hat

    def predict_all_regimes(self, X: np.ndarray) -> np.ndarray:
        """
        Return (T, K) matrix of predictions from all K GAMs.
        Useful for soft blending with regime_probs.
        """
        preds = np.zeros((len(X), self.n_regimes))
        for r, gam_r in self.gams.items():
            preds[:, r] = gam_r.predict(X)
        return preds

    def predict_soft(
        self,
        X: np.ndarray,
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Soft regime-blended prediction: Σ_r p(r_t) * f_r(X_t).
        """
        all_preds = self.predict_all_regimes(X)  # (T, K)
        return (all_preds * regime_probs).sum(axis=1)

    def total_monotonicity_penalty(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        constraints_per_regime: Dict[int, Dict[str, int]],
    ) -> float:
        """Sum monotonicity penalties across all regimes."""
        total = 0.0
        for r, gam_r in self.gams.items():
            mask = regime_labels == r
            if mask.sum() == 0:
                continue
            cons = constraints_per_regime.get(r, {})
            total += gam_r.monotonicity_penalty(X[mask], cons)
        return total

    def log_violations(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        constraints_per_regime: Dict[int, Dict[str, int]],
    ) -> None:
        """Log constraint violation counts for each regime."""
        for r, gam_r in self.gams.items():
            mask = regime_labels == r
            if mask.sum() == 0:
                continue
            cons = constraints_per_regime.get(r, {})
            counts = gam_r.violation_count(X[mask], cons)
            if counts:
                logger.info(f"  Regime {r} violations: {counts}")

    def save(self, dir_path: str, asset: str) -> None:
        import os
        os.makedirs(dir_path, exist_ok=True)
        for r, gam_r in self.gams.items():
            gam_r.save(os.path.join(dir_path, f"gam_{asset}_regime{r}.pkl"))

    @classmethod
    def load(cls, dir_path: str, asset: str, n_regimes: int = 3) -> "MultiRegimeGAM":
        import os
        obj = cls(n_regimes=n_regimes)
        for r in range(n_regimes):
            path = os.path.join(dir_path, f"gam_{asset}_regime{r}.pkl")
            if os.path.exists(path):
                obj.gams[r] = RegimeGAM.load(path)
        logger.info(f"MultiRegimeGAM loaded from {dir_path} for asset={asset}")
        return obj
