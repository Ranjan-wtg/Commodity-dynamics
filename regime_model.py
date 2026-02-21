"""
regime_model.py
===============
Gaussian Hidden Markov Model for regime discovery.

Spec:
  - Input: macro state X_t only (NOT price returns)
  - n_components = 3, covariance_type = 'full'
  - Persistence constraint: diagonal of transition matrix >= 0.85
  - Outputs: regime_probs, regime_labels, transition_matrix
  - FROZEN during Yahoo testing (no refit)
"""

import numpy as np
import os
from hmmlearn.hmm import GaussianHMM
from src.utils import get_logger, save_artifact, load_artifact

logger = get_logger("regime_model")

PERSISTENCE_MIN = 0.85


class RegimeModel:
    """
    Gaussian HMM wrapper with persistence enforcement.

    Attributes
    ----------
    model : GaussianHMM
        Fitted hmmlearn model.
    n_components : int
        Number of hidden regimes (default 3).
    _frozen : bool
        If True, calling fit() raises an error.
    """

    def __init__(self, n_components: int = 3, n_iter: int = 200, seed: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.seed = seed
        self.model: GaussianHMM = None
        self._frozen = False
        self.transition_matrix: np.ndarray = None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "RegimeModel":
        """
        Fit Gaussian HMM on macro state matrix X (T × d).
        After fitting, enforce persistence constraint.
        """
        if self._frozen:
            raise RuntimeError(
                "RegimeModel is frozen. Do NOT refit during Yahoo testing."
            )

        # Remove any NaN rows before fitting
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        logger.info(
            f"Fitting HMM: n_components={self.n_components}, "
            f"n_iter={self.n_iter}, n_obs={len(X_clean)}"
        )

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.seed,
            verbose=False,
        )
        self.model.fit(X_clean)

        self._enforce_persistence()

        logger.info(
            f"HMM converged (monitor: {self.model.monitor_.converged}). "
            f"Transition matrix after persistence enforcement:\n"
            f"{np.round(self.transition_matrix, 4)}"
        )
        return self

    # ── Persistence enforcement ───────────────────────────────────────────────

    def _enforce_persistence(self) -> None:
        """
        Clamp diagonal of transition matrix to >= PERSISTENCE_MIN.
        Renormalize rows so they sum to 1.
        """
        A = self.model.transmat_.copy()
        for i in range(self.n_components):
            if A[i, i] < PERSISTENCE_MIN:
                deficit = PERSISTENCE_MIN - A[i, i]
                off_diag = [j for j in range(self.n_components) if j != i]
                # Reduce off-diag proportionally
                off_sum = A[i, off_diag].sum()
                if off_sum > 1e-8:
                    A[i, off_diag] -= deficit * (A[i, off_diag] / off_sum)
                A[i, i] = PERSISTENCE_MIN
            # Renormalize row
            A[i] = np.clip(A[i], 0, 1)
            A[i] /= A[i].sum()

        self.model.transmat_ = A
        self.transition_matrix = A

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_regimes(self, X: np.ndarray):
        """
        Forward-filter HMM on X. Returns:
            regime_labels : (T,) int array of most-likely regime
            regime_probs  : (T, K) float array of posterior probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        # Posterior probabilities (forward-backward)
        regime_probs_clean = self.model.predict_proba(X_clean)
        regime_labels_clean = np.argmax(regime_probs_clean, axis=1)

        # Reconstruct full arrays (NaN rows get regime 0 by default)
        T = len(X)
        regime_probs = np.zeros((T, self.n_components))
        regime_labels = np.zeros(T, dtype=int)
        regime_probs[mask] = regime_probs_clean
        regime_labels[mask] = regime_labels_clean

        return regime_labels, regime_probs

    def get_transition_matrix(self) -> np.ndarray:
        """Return the (post-enforcement) transition matrix."""
        if self.transition_matrix is None:
            raise RuntimeError("Model not fitted.")
        return self.transition_matrix

    # ── Freeze / Serialize ───────────────────────────────────────────────────

    def freeze(self) -> None:
        """
        Freeze model parameters. After this, fit() will raise an error.
        Call this before Yahoo testing.
        """
        self._frozen = True
        logger.info("RegimeModel FROZEN — no further fitting allowed.")

    def save(self, path: str) -> None:
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str) -> "RegimeModel":
        obj = load_artifact(path)
        logger.info(f"RegimeModel loaded from {path}")
        return obj

    # ── Summary ──────────────────────────────────────────────────────────────

    def regime_summary(self, X: np.ndarray, labels=None) -> dict:
        """Return dict with regime counts and means."""
        if labels is None:
            labels, _ = self.predict_regimes(X)
        summary = {}
        for r in range(self.n_components):
            mask = labels == r
            summary[f"regime_{r}"] = {
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "mean_X": X[mask].mean(axis=0).tolist() if mask.sum() > 0 else [],
            }
        return summary
