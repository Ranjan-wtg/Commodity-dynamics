"""
main.py
=======
Entry point for the SciML Commodity Dynamics Model.

Usage:
    python main.py --mode train [--asset gold|oil|both]
    python main.py --mode test  [--asset gold|oil|both]

Modes:
    train — Full four-step training pipeline. Saves all model artifacts.
    test  — Frozen out-of-sample evaluation on Yahoo Finance data.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, get_logger, ensure_dir
from src.data_loader import load_gold, load_oil, load_macro
from src.preprocessing import (
    align_weekly, rolling_zscore, chronological_split,
    get_macro_columns, build_feature_matrix, get_target,
)
from src.composite_model import CompositeModel
from src.baselines import run_all_baselines
from src.yahoo_testing import run_yahoo_test
from src.visualization import generate_all_plots

logger = get_logger("main")


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── TRAIN ────────────────────────────────────────────────────────────────────

def train_asset(asset: str, cfg: dict) -> None:
    """Full training pipeline for one asset (gold or oil)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  TRAINING: {asset.upper()}")
    logger.info(f"{'='*60}")

    set_seed(cfg.get("seed", 42))
    data_cfg   = cfg.get("data", {})
    splits_cfg = cfg.get("splits", {})
    prep_cfg   = cfg.get("preprocessing", {})
    models_dir = cfg.get("outputs", {}).get("models_dir", "models")
    plots_dir  = cfg.get("outputs", {}).get("plots_dir",  "outputs")

    # ── 1. Load local price data ──────────────────────────────────────────────
    logger.info(f"Loading local {asset} price data…")
    if asset == "gold":
        price_df = load_gold(data_cfg.get("gold_path", "gold/XAU_1w_data.csv"))
    else:
        price_df = load_oil(data_cfg.get("oil_path", "Oil/BrentOilPrices.csv"))

    # ── 2. Fetch macro data (FRED + yfinance) ─────────────────────────────────
    train_start = splits_cfg.get("train_start", "2000-01-01")
    val_end     = splits_cfg.get("val_end",     "2019-12-31")
    macro_cache = data_cfg.get("macro_cache", "data/raw/macro_cache.pkl")

    logger.info(f"Fetching macro data [{train_start} → {val_end}]…")
    macro_df = load_macro(
        start=train_start,
        end=val_end,
        asset=asset,
        cache_path=macro_cache,
    )

    # ── 3. Align and preprocess ───────────────────────────────────────────────
    logger.info("Aligning and preprocessing…")
    merged = align_weekly(price_df, macro_df)

    feature_names = get_macro_columns(asset)
    feature_names = [c for c in feature_names if c in merged.columns]

    rolling_window = prep_cfg.get("rolling_window", 104)
    merged_norm, norm_stats = rolling_zscore(merged, feature_names, window=rolling_window)

    train_end = splits_cfg.get("train_end", "2016-12-31")
    train_df, val_df, _test_df = chronological_split(
        merged_norm,
        train_end=train_end,
        val_end=val_end,
    )

    # ── 4. Build arrays ───────────────────────────────────────────────────────
    X_train = build_feature_matrix(train_df, feature_names)
    y_train = get_target(train_df)
    dates_train = pd.DatetimeIndex(train_df.index)

    X_val = build_feature_matrix(val_df, feature_names)
    y_val = get_target(val_df)

    # ── 5. Fit composite model ────────────────────────────────────────────────
    logger.info("Fitting CompositeModel…")
    model = CompositeModel(asset=asset, cfg=cfg, seed=cfg.get("seed", 42))
    model.fit(
        X_train, y_train, dates_train,
        feature_names=feature_names,
        norm_stats=norm_stats,
    )

    # ── 6. Validation evaluation ──────────────────────────────────────────────
    logger.info("Evaluating on validation set…")
    dates_val = pd.DatetimeIndex(val_df.index)
    labels_val, _ = model.regime_model.predict_regimes(X_val)
    y_pred_val, _, _ = model.predict(X_val, dates_val)

    from src.evaluation import evaluate_all
    val_metrics = evaluate_all(
        y_true=y_val,
        y_pred=y_pred_val,
        multi_regime_gam=model.gam_model,
        shock_model=model.shock_model,
        regime_labels=labels_val,
        X=X_val,
        dates=dates_val,
        constraints_per_regime=model.constraints_per_regime,
        asset=f"{asset}_val",
    )

    # ── 7. Baselines ──────────────────────────────────────────────────────────
    logger.info("Running baselines…")
    baseline_results = run_all_baselines(y_train, X_train, X_val, y_val)

    logger.info("\n── Baseline vs Model (Validation RMSE) ──")
    for name, bres in baseline_results.items():
        logger.info(f"  {name:<25} RMSE={bres['val_rmse']:.6f}")
    logger.info(f"  {'composite_model':<25} RMSE={val_metrics['rmse']:.6f}")

    # ── 8. Save model artifacts ───────────────────────────────────────────────
    logger.info(f"Saving model artifacts → {models_dir}/")
    model.save_all(models_dir)

    # ── 9. Training-period visualizations ─────────────────────────────────────
    logger.info("Generating training visualizations…")
    ensure_dir(plots_dir)
    labels_train, _ = model.regime_model.predict_regimes(X_train)
    generate_all_plots(
        asset_name=asset,
        composite_model=model,
        dates_train=dates_train,
        regime_labels_train=labels_train,
        X_train=X_train,
        price_series=train_df["close"].values if "close" in train_df.columns else None,
        out_dir=plots_dir,
    )

    logger.info(f"DONE: {asset.upper()} training complete.\n")


# ── TEST ─────────────────────────────────────────────────────────────────────

def test_asset(asset: str, cfg: dict) -> None:
    """Frozen out-of-sample Yahoo Finance evaluation for one asset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  TESTING (Yahoo Out-of-Sample): {asset.upper()}")
    logger.info(f"{'='*60}")

    models_dir  = cfg.get("outputs", {}).get("models_dir", "models")
    plots_dir   = cfg.get("outputs", {}).get("plots_dir",  "outputs")
    results_path = cfg.get("data", {}).get("results_path", "data/processed/yahoo_test_results.csv")

    # Load frozen model
    logger.info(f"Loading frozen model from {models_dir}/…")
    model = CompositeModel.load_all(asset=asset, models_dir=models_dir, cfg=cfg)
    # freeze() is called inside load_all automatically

    # Run Yahoo test
    (
        metrics, y_pred, f_comp, u_comp, dates, y_true, labels
    ) = run_yahoo_test(
        composite_model=model,
        asset=asset,
        cfg=cfg,
        results_path=results_path,
        plots_dir=plots_dir,
    )

    # Yahoo visualizations
    logger.info("Generating Yahoo visualizations…")
    ensure_dir(plots_dir)

    from src.visualization import (
        plot_actual_vs_predicted, plot_shock_magnitude, plot_regime_timeline
    )
    plot_actual_vs_predicted(dates, y_true, y_pred, asset,
                             f_component=f_comp, u_component=u_comp,
                             out_dir=plots_dir)
    plot_shock_magnitude(dates, u_comp, asset,
                         shock_events=model.shock_model.events,
                         out_dir=plots_dir)
    plot_regime_timeline(dates, labels, asset, out_dir=plots_dir)

    logger.info(f"DONE: {asset.upper()} Yahoo testing complete.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SciML Commodity Dynamics Model"
    )
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True,
        help="'train' to fit the model; 'test' for Yahoo out-of-sample evaluation."
    )
    parser.add_argument(
        "--asset", choices=["gold", "oil", "both"], default="both",
        help="Which asset to run (default: both)."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    assets = ["gold", "oil"] if args.asset == "both" else [args.asset]

    for asset in assets:
        if args.mode == "train":
            train_asset(asset, cfg)
        else:
            test_asset(asset, cfg)


if __name__ == "__main__":
    main()
