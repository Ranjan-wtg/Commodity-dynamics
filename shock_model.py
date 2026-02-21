"""
shock_model.py
==============
Exogenous shock model:

    u_t = Σ_s A_s * exp(-((t - t_s)^2) / (2 τ_s^2))

Predefined shock events:
  - 2008-10 Financial Crisis
  - 2020-03 COVID
  - 2022-03 Geopolitical Event

Constraints:
  - τ_s ∈ [1, 52] weeks
  - L1 penalty on A_s
  - Crisis dominance: |u_t| >= γ * |f_r(X_t)| when regime == crisis
  - Model is FROZEN during Yahoo testing (no refit)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple
from src.utils import get_logger, save_artifact, load_artifact

logger = get_logger("shock_model")


class ShockModel:
    """
    Gaussian kernel shock model.

    Parameters
    ----------
    events : list of dict, each with keys:
        'name'   : str
        'center' : str (date "YYYY-MM-DD")
        'A_init' : float (initial amplitude)
        'tau_init': float (initial decay width in weeks)
    tau_min : float
    tau_max : float
    l1_weight : float  (L1 penalty weight on amplitudes)
    crisis_gamma : float  (crisis dominance coefficient)
    """

    def __init__(
        self,
        events: List[Dict],
        tau_min: float = 1.0,
        tau_max: float = 52.0,
        l1_weight: float = 0.5,
        crisis_gamma: float = 0.5,
    ):
        self.events = events          # list of event dicts
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.l1_weight = l1_weight
        self.crisis_gamma = crisis_gamma

        # Learnable parameters — initialized from event dicts
        self.A   = np.array([e["A_init"]   for e in events], dtype=float)
        self.tau = np.array([e["tau_init"] for e in events], dtype=float)
        self._centers_weeks: np.ndarray = None  # set in fit/evaluate

        self._frozen = False

    # ── Internal helper ───────────────────────────────────────────────────────

    def _dates_to_week_index(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Convert dates to float week indices anchored at first date."""
        origin = pd.Timestamp("2000-01-01")
        return np.array([(d - origin).days / 7.0 for d in dates])

    def _event_center_weeks(self) -> np.ndarray:
        origin = pd.Timestamp("2000-01-01")
        return np.array(
            [(pd.Timestamp(e["center"]) - origin).days / 7.0 for e in self.events]
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Compute u_t = Σ_s A_s * exp(-((t - t_s)^2) / (2 τ_s^2))
        for each date in `dates`.

        Returns array of shape (T,).
        """
        t = self._dates_to_week_index(dates)
        t_s = self._event_center_weeks()
        u = np.zeros(len(t))
        for s_idx in range(len(self.events)):
            u += self.A[s_idx] * np.exp(
                -((t - t_s[s_idx]) ** 2) / (2.0 * self.tau[s_idx] ** 2)
            )
        return u

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        residuals: np.ndarray,
        dates: pd.DatetimeIndex,
        regime_labels: Optional[np.ndarray] = None,
        gam_preds: Optional[np.ndarray] = None,
        crisis_regime_id: Optional[int] = None,
    ) -> "ShockModel":
        """
        Fit shock parameters (A_s, τ_s) by minimizing:
            MSE(residual, u_t) + l1_weight * Σ|A_s|
            + crisis_dominance_penalty

        Parameters
        ----------
        residuals : array (T,) — y_t - f_r(X_t)
        dates     : DatetimeIndex of length T
        regime_labels : optional (T,) for crisis dominance penalty
        gam_preds : optional (T,) for crisis dominance penalty
        crisis_regime_id : regime index that is the "crisis" regime
        """
        if self._frozen:
            raise RuntimeError(
                "ShockModel is frozen. Do NOT refit during Yahoo testing."
            )

        t       = self._dates_to_week_index(dates)
        t_s     = self._event_center_weeks()
        n_events = len(self.events)

        # Pack parameters: [A_0, A_1, ..., tau_0, tau_1, ...]
        x0 = np.concatenate([self.A, self.tau])

        # Bounds: A unbounded (but L1 penalized), tau ∈ [tau_min, tau_max]
        A_bounds   = [(None, None)] * n_events
        tau_bounds = [(self.tau_min, self.tau_max)] * n_events
        bounds = A_bounds + tau_bounds

        def objective(x):
            A_   = x[:n_events]
            tau_ = x[n_events:]
            u_   = np.zeros(len(t))
            for i in range(n_events):
                u_ += A_[i] * np.exp(-((t - t_s[i]) ** 2) / (2.0 * tau_[i] ** 2))

            # Mean-squared error on residuals
            mse = np.mean((residuals - u_) ** 2)

            # L1 on amplitudes
            l1 = self.l1_weight * np.sum(np.abs(A_))

            # Crisis dominance penalty
            crisis_pen = 0.0
            if (
                regime_labels is not None
                and gam_preds is not None
                and crisis_regime_id is not None
            ):
                crisis_mask = regime_labels == crisis_regime_id
                if crisis_mask.sum() > 0:
                    u_crisis = u_[crisis_mask]
                    f_crisis = gam_preds[crisis_mask]
                    # Penalty when |u_t| < γ * |f_r(X_t)|
                    deficit = np.maximum(
                        0.0,
                        self.crisis_gamma * np.abs(f_crisis) - np.abs(u_crisis),
                    )
                    crisis_pen = deficit.mean()

            return mse + l1 + crisis_pen

        logger.info(f"Fitting shock model: {n_events} events, {len(residuals)} obs")
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        self.A   = result.x[:n_events]
        self.tau = result.x[n_events:]

        logger.info(
            f"Shock model fitted. A={np.round(self.A, 4)}, "
            f"tau={np.round(self.tau, 2)}, "
            f"obj={result.fun:.6f}"
        )
        return self

    # ── Crisis dominance check ────────────────────────────────────────────────

    def crisis_dominance_violation(
        self,
        dates: pd.DatetimeIndex,
        regime_labels: np.ndarray,
        gam_preds: np.ndarray,
        crisis_regime_id: int,
    ) -> Dict[str, float]:
        """
        Return fraction of crisis-regime timesteps where
        |u_t| < γ * |f_r(X_t)|.
        """
        u = self.evaluate(dates)
        crisis_mask = regime_labels == crisis_regime_id
        if crisis_mask.sum() == 0:
            return {"crisis_violation_frac": 0.0, "n_crisis_steps": 0}
        u_c = u[crisis_mask]
        f_c = gam_preds[crisis_mask]
        viol = np.abs(u_c) < self.crisis_gamma * np.abs(f_c)
        return {
            "crisis_violation_frac": float(viol.mean()),
            "n_crisis_steps": int(crisis_mask.sum()),
        }

    # ── Contribution magnitude ────────────────────────────────────────────────

    def shock_magnitude(self, dates: pd.DatetimeIndex) -> float:
        """Return mean absolute shock contribution over given dates."""
        u = self.evaluate(dates)
        return float(np.abs(u).mean())

    # ── Freeze / Serialize ────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze parameters — call before Yahoo testing."""
        self._frozen = True
        logger.info("ShockModel FROZEN — no further fitting allowed.")

    def save(self, path: str) -> None:
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str) -> "ShockModel":
        obj = load_artifact(path)
        logger.info(f"ShockModel loaded from {path}")
        return obj

    def summary(self) -> Dict:
        """Return summary dict of fitted parameters."""
        return {
            e["name"]: {"A": float(self.A[i]), "tau_weeks": float(self.tau[i])}
            for i, e in enumerate(self.events)
        }
