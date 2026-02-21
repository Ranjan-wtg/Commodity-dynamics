"""
yahoo_testing.py
================
Out-of-sample testing on Yahoo Finance data.

STRICT RULES:
  - Do NOT refit HMM, GAM, or shock model (all frozen)
  - Use training-period normalization statistics only
  - Forward-filter only for regime inference
  - Cache Yahoo data with timestamp + ticker log
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, Tuple

from src.utils import get_logger, ensure_dir
from src.preprocessing import apply_normalization, get_macro_columns, build_feature_matrix, get_target
from src.data_loader import load_macro
from src.evaluation import evaluate_all

logger = get_logger("yahoo_testing")


# ── Yahoo price download ──────────────────────────────────────────────────────

def download_yahoo(
    ticker: str,
    fallback: str,
    start: str,
    end: Optional[str],
    cache_path: str,
    interval: str = "1wk",
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Download weekly OHLCV data from Yahoo Finance.
    Caches to disk as a pickle with metadata.

    Returns DataFrame with columns [close, log_return].
    """
    ensure_dir(os.path.dirname(os.path.abspath(cache_path)))

    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logger.info(
            f"Yahoo cache hit: ticker={cache.get('ticker_used')}, "
            f"downloaded_at={cache.get('downloaded_at')}"
        )
        return cache["data"]

    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    used_ticker = None
    df_raw = None

    for t in [ticker, fallback]:
        try:
            logger.info(f"Downloading Yahoo Finance: ticker={t}, start={start}, end={end}")
            raw = yf.download(
                t,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if raw is not None and len(raw) > 5:
                used_ticker = t
                df_raw = raw
                break
        except Exception as e:
            logger.warning(f"yfinance failed for {t}: {e}")

    if df_raw is None or len(df_raw) == 0:
        raise RuntimeError(
            f"Could not download Yahoo data for ticker={ticker} or fallback={fallback}"
        )

    close = df_raw["Close"].squeeze()
    close.index = pd.to_datetime(close.index)
    close = close.resample("W-FRI").last().dropna()

    df = pd.DataFrame({"close": close})
    df["log_return"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df = df.dropna()

    downloaded_at = datetime.datetime.utcnow().isoformat() + "Z"
    logger.info(
        f"Yahoo download complete: ticker={used_ticker}, "
        f"obs={len(df)}, downloaded_at={downloaded_at}"
    )

    cache_obj = {
        "data": df,
        "ticker_used": used_ticker,
        "downloaded_at": downloaded_at,
        "start": start,
        "end": end,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_obj, f)
    logger.info(f"Yahoo cache saved → {cache_path}")

    return df


# ── Normalization using TRAINING stats only ───────────────────────────────────

def prepare_yahoo_features(
    yahoo_df: pd.DataFrame,
    asset: str,
    cfg: dict,
    norm_stats: dict,
    training_end: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list]:
    """
    Fetch macro data for the Yahoo test period, align to price dates,
    and normalize using TRAINING-PERIOD statistics only.

    Returns (X, y, dates, feature_names).
    """
    macro_cfg = cfg.get("macro", {})
    yahoo_cfg = cfg.get("yahoo", {})
    macro_cache = cfg.get("data", {}).get("macro_cache", "data/raw/macro_cache.pkl")

    start = yahoo_cfg.get("start_date", "2020-01-01")
    end   = datetime.date.today().strftime("%Y-%m-%d")

    # Fetch macro for the Yahoo test period
    macro_df = load_macro(
        start=start,
        end=end,
        asset=asset,
        cache_path=macro_cache,
    )

    # Align macro to Yahoo price dates (forward-fill)
    macro_aligned = macro_df.reindex(yahoo_df.index, method="ffill")

    feature_names = get_macro_columns(asset)
    # Keep only available columns
    feature_names = [c for c in feature_names if c in macro_aligned.columns]

    # Apply TRAINING-PERIOD normalization (no recalculation on Yahoo data)
    logger.info(
        "Applying training-period normalization to Yahoo features "
        "(no lookahead / no recalculation on Yahoo data)"
    )
    macro_norm = apply_normalization(
        macro_aligned[feature_names].copy(),
        feature_names,
        norm_stats,
        training_end,
    )

    # Join and drop NaN
    yahoo_df = yahoo_df.copy()
    for col in feature_names:
        yahoo_df[col] = macro_norm[col] if col in macro_norm.columns else np.nan

    yahoo_df = yahoo_df.dropna()

    X      = yahoo_df[feature_names].values.astype(np.float64)
    y      = yahoo_df["log_return"].values.astype(np.float64)
    dates  = pd.DatetimeIndex(yahoo_df.index)

    logger.info(
        f"Yahoo test set ready: {len(y)} obs  "
        f"[{dates[0].date()} – {dates[-1].date()}]"
    )
    return X, y, dates, feature_names


# ── Main Yahoo test runner ────────────────────────────────────────────────────

def run_yahoo_test(
    composite_model,
    asset: str,
    cfg: dict,
    results_path: str,
    plots_dir: str = "outputs",
) -> Dict[str, float]:
    """
    Run frozen out-of-sample evaluation on Yahoo Finance data.

    Steps:
      1. Download Yahoo price data (or load cache)
      2. Fetch Yahoo-period macro data (cached separately)
      3. Normalize with TRAINING stats (frozen)
      4. Regime inference: forward-filter only (no refit)
      5. Predict: y_hat = f_r(X) + u_t   (frozen params)
      6. Evaluate all metrics
      7. Save results CSV

    STRICT: composite_model must already be frozen before calling this.
    """
    if not composite_model._frozen:
        raise RuntimeError(
            "CompositeModel must be frozen before Yahoo testing. "
            "Call composite_model.freeze() first."
        )

    yahoo_cfg  = cfg.get("yahoo", {})
    data_cfg   = cfg.get("data", {})

    # ── 1. Download Yahoo price ───────────────────────────────────────────────
    ticker_map = {
        "gold": (yahoo_cfg.get("gold_ticker", "GC=F"),   yahoo_cfg.get("gold_fallback", "GLD")),
        "oil":  (yahoo_cfg.get("oil_ticker",  "CL=F"),   yahoo_cfg.get("oil_fallback",  "USO")),
    }
    ticker, fallback = ticker_map[asset]
    cache_path = data_cfg.get("yahoo_cache", "data/raw/yahoo_test_cache.pkl")
    # Separate cache per asset
    base, ext   = os.path.splitext(cache_path)
    cache_path  = f"{base}_{asset}{ext}"

    yahoo_df = download_yahoo(
        ticker=ticker,
        fallback=fallback,
        start=yahoo_cfg.get("start_date", "2020-01-01"),
        end=None,
        cache_path=cache_path,
        interval=yahoo_cfg.get("interval", "1wk"),
    )

    # ── 2–3. Prepare features (frozen normalization) ──────────────────────────
    X, y_true, dates, feature_names = prepare_yahoo_features(
        yahoo_df=yahoo_df,
        asset=asset,
        cfg=cfg,
        norm_stats=composite_model.norm_stats,
        training_end=composite_model.training_end,
    )

    # ── 4–5. Frozen inference ─────────────────────────────────────────────────
    logger.info("Running frozen inference (no refit of any model)…")
    y_pred, f_component, u_component = composite_model.predict(X, dates)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    labels, _ = composite_model.regime_model.predict_regimes(X)
    metrics = evaluate_all(
        y_true=y_true,
        y_pred=y_pred,
        multi_regime_gam=composite_model.gam_model,
        shock_model=composite_model.shock_model,
        regime_labels=labels,
        X=X,
        dates=dates,
        constraints_per_regime=composite_model.constraints_per_regime,
        asset=asset,
    )
    metrics["n_obs"] = len(y_true)
    metrics["date_start"] = str(dates[0].date())
    metrics["date_end"]   = str(dates[-1].date())

    # ── 7. Save results CSV ───────────────────────────────────────────────────
    ensure_dir(os.path.dirname(os.path.abspath(results_path)))
    results_df = pd.DataFrame([metrics])

    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        # Remove old entry for this asset
        existing = existing[existing["asset"] != asset]
        results_df = pd.concat([existing, results_df], ignore_index=True)

    results_df.to_csv(results_path, index=False)
    logger.info(f"Yahoo test results saved → {results_path}")

    return metrics, y_pred, f_component, u_component, dates, y_true, labels
