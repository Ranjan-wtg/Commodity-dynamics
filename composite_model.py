"""
composite_model.py
==================
Full pipeline: y_t = f_{r_t}(X_t) + u_t + ε_t

Training procedure (Step 1–4):
  1. Fit HMM on macro state X_t
  2. Split by regime, fit regime-specific GAMs
  3. Compute residuals, fit shock model on residuals
  4. Alternating: update GAM → update shock → repeat N iterations

This model is the central orchestrator. It serializes all sub-models
to models/ and loads them for frozen Yahoo inference.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

import yaml

from src.utils import get_logger, save_artifact, load_artifact, ensure_dir
from src.regime_model import RegimeModel
from src.gam_model import MultiRegimeGAM
from src.shock_model import ShockModel
from src.loss_functions import composite_loss

logger = get_logger("composite_model")


def _build_constraints_per_regime(
    asset: str,
    n_regimes: int,
    tightening_regime: int = 0,
    growth_regime: int = 1,
) -> Dict[int, Dict[str, int]]:
    """
    Build the constraints_per_regime dict from the spec.

    Gold:
      - tightening regime: real_rate → -1
      - all regimes: DXY → -1
    Oil:
      - growth regime: demand_proxy → +1
    """
    cons = {r: {} for r in range(n_regimes)}
    if asset.lower() == "gold":
        # DXY constraint applies in all regimes
        for r in range(n_regimes):
            cons[r]["DXY"] = -1
        # real_rate constraint in tightening regime only
        cons[tightening_regime]["real_rate"] = -1
    elif asset.lower() == "oil":
        cons[growth_regime]["demand_proxy"] = 1
    return cons


def _build_shock_events(cfg: dict) -> List[Dict]:
    """Parse shock events from config dict."""
    shock_cfg = cfg.get("shock", {}).get("events", {})
    events = []
    for name, e in shock_cfg.items():
        events.append({
            "name":     name,
            "center":   e["center"],
            "A_init":   e["A_init"],
            "tau_init": e["tau_init"],
        })
    return events


class CompositeModel:
    """
    Regime-aware constrained commodity dynamics model.

    y_t = f_{r_t}(X_t) + u_t + ε_t

    Attributes
    ----------
    asset : str  ('gold' or 'oil')
    regime_model : RegimeModel
    gam_model    : MultiRegimeGAM
    shock_model  : ShockModel
    constraints_per_regime : dict
    _frozen : bool  — True after freeze(), prevents any refit
    """

    def __init__(self, asset: str, cfg: dict, seed: int = 42):
        self.asset  = asset
        self.cfg    = cfg
        self.seed   = seed
        self._frozen = False

        hmm_cfg = cfg.get("hmm", {})
        gam_cfg = cfg.get("gam", {})
        shock_cfg = cfg.get("shock", {})
        loss_cfg = cfg.get("loss", {})

        self.n_regimes = hmm_cfg.get("n_components", 3)

        # Sub-models
        self.regime_model = RegimeModel(
            n_components=self.n_regimes,
            n_iter=hmm_cfg.get("n_iter", 200),
            seed=seed,
        )
        self.gam_model = None     # built after HMM (need feature_names)
        self.shock_model = ShockModel(
            events=_build_shock_events(cfg),
            tau_min=shock_cfg.get("tau_min", 1.0),
            tau_max=shock_cfg.get("tau_max", 52.0),
            l1_weight=loss_cfg.get("lambda_shock", 0.2),
            crisis_gamma=shock_cfg.get("crisis_dominance_gamma", 0.5),
        )

        self.gam_n_splines = gam_cfg.get("n_splines", 10)
        self.gam_lam       = gam_cfg.get("lam", 0.6)
        self.n_alternating = cfg.get("training", {}).get("n_alternating_iter", 5)
        self.lambdas       = loss_cfg
        self.gradient_bound = cfg.get("gradient_bound", 2.0)

        # Regime assignment helpers (set after HMM fit)
        self.tightening_regime = 0
        self.growth_regime = 1
        self.crisis_regime = 2

        self.constraints_per_regime = None
        self.norm_stats = None
        self.feature_names: List[str] = []
        self.training_end: str = cfg.get("splits", {}).get("train_end", "2016-12-31")

        logger.info(f"CompositeModel initialized for asset='{asset}'")

    # ── Step 1: Fit HMM ──────────────────────────────────────────────────────

    def _fit_hmm(self, X_train: np.ndarray) -> np.ndarray:
        """Fit HMM and return regime labels for training data."""
        logger.info("[Step 1] Fitting HMM on macro state…")
        self.regime_model.fit(X_train)
        labels, probs = self.regime_model.predict_regimes(X_train)

        # Identify which regime is "crisis" (highest VIX mean)
        vix_idx = self.feature_names.index("VIX") if "VIX" in self.feature_names else 3
        vix_means = [X_train[labels == r, vix_idx].mean() for r in range(self.n_regimes)]
        self.crisis_regime    = int(np.argmax(vix_means))
        # Tightening = highest real_rate mean
        rr_idx = self.feature_names.index("real_rate") if "real_rate" in self.feature_names else 0
        rr_means = [X_train[labels == r, rr_idx].mean() for r in range(self.n_regimes)]
        self.tightening_regime = int(np.argmax(rr_means))
        # Growth = remaining
        remaining = [r for r in range(self.n_regimes)
                     if r != self.crisis_regime and r != self.tightening_regime]
        self.growth_regime = remaining[0] if remaining else 0

        logger.info(
            f"Regime assignment → crisis={self.crisis_regime}, "
            f"tightening={self.tightening_regime}, growth={self.growth_regime}"
        )

        self.constraints_per_regime = _build_constraints_per_regime(
            self.asset, self.n_regimes,
            tightening_regime=self.tightening_regime,
            growth_regime=self.growth_regime,
        )
        return labels

    # ── Step 2: Fit GAMs per regime ───────────────────────────────────────────

    def _fit_gams(self, X_train: np.ndarray, y_train: np.ndarray, labels: np.ndarray):
        logger.info("[Step 2] Fitting regime-specific constrained GAMs…")
        self.gam_model = MultiRegimeGAM(
            n_regimes=self.n_regimes,
            feature_names=self.feature_names,
            n_splines=self.gam_n_splines,
            lam=self.gam_lam,
        )
        self.gam_model.fit(X_train, y_train, labels)
        self.gam_model.log_violations(X_train, labels, self.constraints_per_regime)

    # ── Step 3: Fit shock model on residuals ──────────────────────────────────

    def _fit_shock(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        labels: np.ndarray,
        dates: pd.DatetimeIndex,
    ):
        logger.info("[Step 3] Fitting shock model on GAM residuals…")
        f_train = self.gam_model.predict(X_train, labels)
        residuals = y_train - f_train
        self.shock_model.fit(
            residuals, dates,
            regime_labels=labels,
            gam_preds=f_train,
            crisis_regime_id=self.crisis_regime,
        )
        logger.info(f"Shock summary: {self.shock_model.summary()}")

    # ── Step 4: Alternating optimization ─────────────────────────────────────

    def _alternating_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        labels: np.ndarray,
        dates: pd.DatetimeIndex,
    ):
        logger.info(f"[Step 4] Alternating optimization: {self.n_alternating} iterations…")
        for iteration in range(self.n_alternating):
            # Update GAM: predict using CURRENT shock, refit GAM on de-shocked residual
            u_train = self.shock_model.evaluate(dates)
            y_gam = y_train - u_train
            self.gam_model.fit(X_train, y_gam, labels)

            # Update shock: compute new GAM residuals, refit shock
            f_train = self.gam_model.predict(X_train, labels)
            residuals = y_train - f_train
            self.shock_model.fit(
                residuals, dates,
                regime_labels=labels,
                gam_preds=f_train,
                crisis_regime_id=self.crisis_regime,
            )

            # Compute composite loss for monitoring
            y_pred = f_train + u_train
            _, probs = self.regime_model.predict_regimes(X_train)
            losses = composite_loss(
                y_train, y_pred,
                self.gam_model, self.shock_model,
                probs, labels, X_train,
                self.constraints_per_regime,
                self.lambdas,
            )
            logger.info(
                f"  Iter {iteration+1}/{self.n_alternating} — "
                f"total={losses['total']:.6f}  "
                f"huber={losses['huber']:.6f}  "
                f"mono={losses['monotonic']:.4f}  "
                f"stab={losses['stability']:.6f}"
            )

    # ── Main fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dates_train: pd.DatetimeIndex,
        feature_names: List[str],
        norm_stats: dict = None,
    ) -> "CompositeModel":
        """
        Full four-step training procedure.

        Parameters
        ----------
        X_train      : (T, d) macro feature matrix (normalized)
        y_train      : (T,) log-return target
        dates_train  : DatetimeIndex of training weeks
        feature_names: list of macro column names
        norm_stats   : rolling normalization statistics (for Yahoo reuse)
        """
        if self._frozen:
            raise RuntimeError("Model is frozen. Cannot refit.")

        self.feature_names = feature_names
        self.norm_stats    = norm_stats

        # Remove NaN rows from training data
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[valid]
        y_train = y_train[valid]
        dates_train = dates_train[valid]

        labels = self._fit_hmm(X_train)
        self._fit_gams(X_train, y_train, labels)
        self._fit_shock(X_train, y_train, labels, dates_train)
        self._alternating_optimization(X_train, y_train, labels, dates_train)

        logger.info("Training complete.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        dates: pd.DatetimeIndex,
        use_soft_regime: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict y_hat = f_{r_t}(X_t) + u_t.

        Returns
        -------
        y_hat        : full prediction
        f_component  : GAM component  f_r(X)
        u_component  : shock component u_t
        """
        labels, probs = self.regime_model.predict_regimes(X)
        if use_soft_regime:
            f_component = self.gam_model.predict_soft(X, probs)
        else:
            f_component = self.gam_model.predict(X, labels)
        u_component = self.shock_model.evaluate(dates)
        y_hat = f_component + u_component
        return y_hat, f_component, u_component

    # ── Freeze ────────────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """
        Freeze all sub-models. Called before Yahoo testing.
        Prevents any accidental refitting.
        """
        self._frozen = True
        self.regime_model.freeze()
        self.shock_model.freeze()
        logger.info(
            "CompositeModel FROZEN — all sub-models locked for Yahoo testing."
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def save_all(self, models_dir: str) -> None:
        ensure_dir(models_dir)
        a = self.asset
        self.regime_model.save(os.path.join(models_dir, f"hmm_{a}.pkl"))
        self.gam_model.save(models_dir, a)
        self.shock_model.save(os.path.join(models_dir, f"shock_{a}.pkl"))
        save_artifact(
            {
                "feature_names":           self.feature_names,
                "norm_stats":              self.norm_stats,
                "constraints_per_regime":  self.constraints_per_regime,
                "tightening_regime":       self.tightening_regime,
                "growth_regime":           self.growth_regime,
                "crisis_regime":           self.crisis_regime,
                "training_end":            self.training_end,
            },
            os.path.join(models_dir, f"meta_{a}.pkl"),
        )
        logger.info(f"All sub-models saved to {models_dir}/")

    @classmethod
    def load_all(cls, asset: str, models_dir: str, cfg: dict) -> "CompositeModel":
        """Load all sub-models and metadata for frozen Yahoo inference."""
        obj = cls(asset=asset, cfg=cfg)

        a = asset
        obj.regime_model = RegimeModel.load(os.path.join(models_dir, f"hmm_{a}.pkl"))
        obj.gam_model    = MultiRegimeGAM.load(models_dir, a, n_regimes=obj.n_regimes)
        obj.shock_model  = ShockModel.load(os.path.join(models_dir, f"shock_{a}.pkl"))

        meta = load_artifact(os.path.join(models_dir, f"meta_{a}.pkl"))
        obj.feature_names          = meta["feature_names"]
        obj.norm_stats             = meta["norm_stats"]
        obj.constraints_per_regime = meta["constraints_per_regime"]
        obj.tightening_regime      = meta["tightening_regime"]
        obj.growth_regime          = meta["growth_regime"]
        obj.crisis_regime          = meta["crisis_regime"]
        obj.training_end           = meta["training_end"]

        obj.freeze()  # Always freeze after loading for test mode
        logger.info(f"CompositeModel loaded and frozen for asset='{asset}'")
        return obj
