"""
visualization.py
================
All plotting functions for the SciML project.

Plots generated:
  1. Regime timeline
  2. Actual vs predicted (Yahoo out-of-sample)
  3. Shock magnitude over time
  4. Partial dependence plots (per regime, per feature)
  5. Stability perturbation plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, List

from src.utils import get_logger, ensure_dir

logger = get_logger("visualization")

# ── Style defaults ─────────────────────────────────────────────────────────────
REGIME_COLORS = ["#2196F3", "#4CAF50", "#F44336"]   # blue=0, green=1, red=2
REGIME_NAMES  = {0: "Regime 0", 1: "Regime 1", 2: "Regime 2"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved → {path}")


# ── 1. Regime timeline ────────────────────────────────────────────────────────

def plot_regime_timeline(
    dates: pd.DatetimeIndex,
    regime_labels: np.ndarray,
    asset_name: str,
    price_series: Optional[np.ndarray] = None,
    out_dir: str = "outputs",
    regime_names: dict = None,
) -> str:
    """
    Colour-coded regime timeline. Optionally overlays the price series.
    """
    rnames = regime_names or REGIME_NAMES
    n_regimes = len(np.unique(regime_labels))

    fig, axes = plt.subplots(
        2 if price_series is not None else 1, 1,
        figsize=(14, 6 if price_series is not None else 3),
        sharex=True,
    )
    if price_series is None:
        axes = [axes]

    ax_reg = axes[0]
    for r in range(n_regimes):
        mask = regime_labels == r
        ax_reg.bar(
            dates[mask], np.ones(mask.sum()),
            color=REGIME_COLORS[r % len(REGIME_COLORS)],
            width=8, alpha=0.8, label=rnames.get(r, f"Regime {r}")
        )
    ax_reg.set_ylabel("Regime")
    ax_reg.set_yticks([])
    ax_reg.legend(loc="upper left", ncol=n_regimes)
    ax_reg.set_title(f"{asset_name.upper()} — Regime Timeline", fontweight="bold")
    ax_reg.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if price_series is not None:
        ax_p = axes[1]
        ax_p.plot(dates, price_series, color="#333333", linewidth=1.0)
        ax_p.set_ylabel("Price")
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

    path = os.path.join(out_dir, f"{asset_name}_regime_timeline.png")
    _save(fig, path)
    return path


# ── 2. Actual vs predicted ────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    asset_name: str,
    f_component: Optional[np.ndarray] = None,
    u_component: Optional[np.ndarray] = None,
    out_dir: str = "outputs",
) -> str:
    """
    Two-panel plot:
      Top: Actual vs Full Prediction (+ optional GAM / Shock components)
      Bottom: Prediction error
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(dates, y_true,  label="Actual",     color="#333333", linewidth=1.2, alpha=0.9)
    ax1.plot(dates, y_pred,  label="Predicted",  color="#E91E63", linewidth=1.2, alpha=0.9, linestyle="--")
    if f_component is not None:
        ax1.plot(dates, f_component, label="GAM f_r(X)", color="#2196F3", linewidth=0.8, alpha=0.6)
    if u_component is not None:
        ax1.plot(dates, u_component, label="Shock u_t",  color="#FF9800", linewidth=0.8, alpha=0.6)

    ax1.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Log Return")
    ax1.legend(loc="upper left", ncol=4)
    ax1.set_title(f"{asset_name.upper()} — Actual vs Predicted (Yahoo Out-of-Sample)", fontweight="bold")

    error = y_true - y_pred
    ax2.fill_between(dates, error, 0, where=error >= 0, color="#4CAF50", alpha=0.4, label="Over-pred")
    ax2.fill_between(dates, error, 0, where=error <  0, color="#F44336", alpha=0.4, label="Under-pred")
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    path = os.path.join(out_dir, f"yahoo_{asset_name}_actual_vs_predicted.png")
    _save(fig, path)
    return path


# ── 3. Shock magnitude over time ──────────────────────────────────────────────

def plot_shock_magnitude(
    dates: pd.DatetimeIndex,
    shock_values: np.ndarray,
    asset_name: str,
    shock_events: Optional[list] = None,
    out_dir: str = "outputs",
) -> str:
    """
    Bar chart / area chart of u_t shock function over time.
    Annotates named shock events with vertical lines.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(dates, shock_values, 0,
                    where=shock_values >= 0, color="#4CAF50", alpha=0.6, label="Positive shock")
    ax.fill_between(dates, shock_values, 0,
                    where=shock_values < 0,  color="#F44336", alpha=0.6, label="Negative shock")
    ax.axhline(0, color="grey", linewidth=0.7)

    if shock_events:
        for ev in shock_events:
            center = pd.Timestamp(ev.get("center", ""))
            name   = ev.get("name", "")
            if center in dates or True:  # always annotate
                ax.axvline(center, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
                ax.text(center, ax.get_ylim()[1] * 0.9, name,
                        rotation=90, fontsize=8, va="top", ha="right", color="black")

    ax.set_ylabel("Shock u_t")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.set_title(f"{asset_name.upper()} — Exogenous Shock Magnitude Over Time", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    path = os.path.join(out_dir, f"{asset_name}_shock_magnitude.png")
    _save(fig, path)
    return path


# ── 4. Partial dependence plots ───────────────────────────────────────────────

def plot_partial_dependence(
    multi_regime_gam,
    feature_names: List[str],
    asset_name: str,
    out_dir: str = "outputs",
) -> List[str]:
    """
    One figure per feature, showing PDP for each regime.
    """
    paths = []
    n_regimes = multi_regime_gam.n_regimes

    for feat_name in feature_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False

        for r in range(n_regimes):
            gam_r = multi_regime_gam.gams.get(r)
            if gam_r is None or gam_r.gam is None:
                continue
            if feat_name not in gam_r.feature_names:
                continue
            j = gam_r.feature_names.index(feat_name)
            try:
                grid, pdp = gam_r.partial_dependence(j, grid_points=100)
                ax.plot(
                    grid, pdp,
                    color=REGIME_COLORS[r % len(REGIME_COLORS)],
                    label=f"Regime {r}",
                    linewidth=1.8,
                )
                plotted = True
            except Exception as e:
                logger.warning(f"PDP failed for regime {r}, feature {feat_name}: {e}")

        if not plotted:
            plt.close(fig)
            continue

        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Partial Effect on log return")
        ax.set_title(
            f"{asset_name.upper()} — Partial Dependence: {feat_name}",
            fontweight="bold",
        )
        ax.legend()

        path = os.path.join(out_dir, f"{asset_name}_pdp_{feat_name}.png")
        _save(fig, path)
        paths.append(path)

    return paths


# ── 5. Stability perturbation plot ────────────────────────────────────────────

def plot_stability_perturbation(
    X: np.ndarray,
    regime_labels: np.ndarray,
    multi_regime_gam,
    asset_name: str,
    feature_idx: int = 0,
    feature_name: str = "feature_0",
    n_trials: int = 30,
    epsilon_std: float = 0.01,
    out_dir: str = "outputs",
    seed: int = 42,
) -> str:
    """
    Show how model predictions spread under input perturbations.
    Plots mean prediction ± std band across perturbation trials.
    """
    rng = np.random.default_rng(seed)
    T = len(X)
    pred_trials = np.zeros((n_trials, T))

    f0 = multi_regime_gam.predict(X, regime_labels)
    for k in range(n_trials):
        X_p = X + rng.normal(0.0, epsilon_std, size=X.shape)
        pred_trials[k] = multi_regime_gam.predict(X_p, regime_labels)

    mean_pred = pred_trials.mean(axis=0)
    std_pred  = pred_trials.std(axis=0)

    # Sort by the target feature for cleaner display
    sort_idx = np.argsort(X[:, feature_idx])
    x_sorted  = X[sort_idx, feature_idx]
    f0_sorted = f0[sort_idx]
    mu_sorted = mean_pred[sort_idx]
    sd_sorted = std_pred[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_sorted, f0_sorted, color="#333333", linewidth=1.5, label="Nominal f(X)")
    ax.plot(x_sorted, mu_sorted, color="#E91E63", linewidth=1.2,
            linestyle="--", label="Mean (perturbed)")
    ax.fill_between(x_sorted,
                    mu_sorted - sd_sorted,
                    mu_sorted + sd_sorted,
                    color="#E91E63", alpha=0.2, label="±1 SD (perturbed)")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("f(X) prediction")
    ax.set_title(
        f"{asset_name.upper()} — Stability Perturbation ({feature_name})",
        fontweight="bold",
    )
    ax.legend()

    path = os.path.join(out_dir, f"{asset_name}_stability_{feature_name}.png")
    _save(fig, path)
    return path


# ── Batch generate all plots ──────────────────────────────────────────────────

def generate_all_plots(
    asset_name: str,
    composite_model,
    dates_train: pd.DatetimeIndex,
    regime_labels_train: np.ndarray,
    X_train: np.ndarray,
    price_series: Optional[np.ndarray] = None,
    # Yahoo data (optional — only if test mode called first)
    dates_yahoo: Optional[pd.DatetimeIndex] = None,
    y_true_yahoo: Optional[np.ndarray] = None,
    y_pred_yahoo: Optional[np.ndarray] = None,
    f_yahoo: Optional[np.ndarray] = None,
    u_yahoo: Optional[np.ndarray] = None,
    out_dir: str = "outputs",
) -> List[str]:
    """Convenience wrapper — generate all plots for one asset."""
    paths = []
    ensure_dir(out_dir)

    # Regime timeline (training period)
    paths.append(
        plot_regime_timeline(
            dates_train, regime_labels_train, asset_name,
            price_series=price_series, out_dir=out_dir,
        )
    )

    # Shock magnitude (over full date range)
    shock_vals = composite_model.shock_model.evaluate(dates_train)
    shock_events_list = composite_model.shock_model.events
    paths.append(
        plot_shock_magnitude(dates_train, shock_vals, asset_name,
                             shock_events=shock_events_list, out_dir=out_dir)
    )

    # Partial dependence plots
    pdp_paths = plot_partial_dependence(
        composite_model.gam_model, composite_model.feature_names,
        asset_name, out_dir=out_dir,
    )
    paths.extend(pdp_paths)

    # Stability plot (first feature)
    if composite_model.feature_names:
        paths.append(
            plot_stability_perturbation(
                X_train, regime_labels_train,
                composite_model.gam_model,
                asset_name,
                feature_idx=0,
                feature_name=composite_model.feature_names[0],
                out_dir=out_dir,
            )
        )

    # Actual vs predicted (Yahoo — only if provided)
    if dates_yahoo is not None and y_true_yahoo is not None:
        paths.append(
            plot_actual_vs_predicted(
                dates_yahoo, y_true_yahoo, y_pred_yahoo,
                asset_name,
                f_component=f_yahoo,
                u_component=u_yahoo,
                out_dir=out_dir,
            )
        )

    logger.info(f"Generated {len(paths)} plots for {asset_name.upper()}")
    return paths
