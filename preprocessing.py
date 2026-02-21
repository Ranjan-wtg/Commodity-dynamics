"""
preprocessing.py
================
Weekly alignment, forward-fill, rolling z-score normalization,
and chronological train/val/test splitting.

Rules enforced:
- No lookahead bias: rolling stats computed only on past window.
- No shuffling of time-series data.
- Macro data forward-filled to fill weekend/holiday gaps.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from src.utils import get_logger

logger = get_logger("preprocessing")


def align_weekly(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price and macro data on Friday-aligned weekly index.

    - Reindexes macro to match price dates (forward-fill).
    - Returns merged DataFrame with all price + macro columns.
    """
    # Ensure both have DatetimeIndex
    price_df.index = pd.to_datetime(price_df.index)
    macro_df.index = pd.to_datetime(macro_df.index)

    # Reindex macro to price dates, forward-fill any gaps
    macro_aligned = macro_df.reindex(price_df.index, method="ffill")

    merged = price_df.join(macro_aligned, how="left")

    # Final forward-fill pass on any remaining NaN in macro columns
    macro_cols = macro_df.columns.tolist()
    merged[macro_cols] = merged[macro_cols].ffill()

    # Drop rows where BOTH price AND macro are NaN
    merged = merged.dropna(subset=["log_return"])

    logger.info(
        f"Aligned dataset: {len(merged)} weekly obs, "
        f"{merged.shape[1]} columns, "
        f"[{merged.index[0].date()} – {merged.index[-1].date()}]"
    )
    return merged


def rolling_zscore(
    df: pd.DataFrame,
    feature_cols: list,
    window: int = 104,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.Series, pd.Series]]]:
    """
    Apply rolling 104-week z-score normalization to feature_cols.

    Uses only past data (no lookahead):
        z_t = (x_t - μ_{t-window:t-1}) / σ_{t-window:t-1}

    Rows within the first `window` steps receive NaN and are dropped.

    Returns:
        df_norm   — DataFrame with normalized feature columns
        norm_stats — dict of {col: (rolling_mean, rolling_std)} Series
                     (training-period statistics for reuse on Yahoo data)
    """
    df = df.copy()
    norm_stats: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    for col in feature_cols:
        roll_mean = df[col].shift(1).rolling(window=window, min_periods=window).mean()
        roll_std  = df[col].shift(1).rolling(window=window, min_periods=window).std()
        norm_stats[col] = (roll_mean, roll_std)
        df[col] = (df[col] - roll_mean) / roll_std.replace(0, 1e-8)

    before = len(df)
    df = df.dropna(subset=feature_cols)
    logger.info(
        f"Rolling z-score (window={window}): dropped {before - len(df)} warm-up rows. "
        f"Remaining: {len(df)} obs."
    )
    return df, norm_stats


def apply_normalization(
    df: pd.DataFrame,
    feature_cols: list,
    norm_stats: Dict[str, Tuple[pd.Series, pd.Series]],
    training_end: str,
) -> pd.DataFrame:
    """
    Apply TRAINING-PERIOD normalization statistics to new data (Yahoo/test).
    Uses the last available rolling mean/std from the training period.

    This prevents any lookahead contamination from Yahoo data.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in norm_stats:
            logger.warning(f"No normalization stats for '{col}' — skipping.")
            continue
        roll_mean, roll_std = norm_stats[col]
        # Use the last training-period value
        last_mean = roll_mean.loc[:training_end].iloc[-1]
        last_std  = roll_std.loc[:training_end].iloc[-1]
        df[col] = (df[col] - last_mean) / (last_std if last_std != 0 else 1e-8)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2016-12-31",
    val_end: str = "2019-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train / val / test.
    NO shuffling.

    Returns:
        train — data up to train_end (inclusive)
        val   — data between train_end+1 and val_end (inclusive)
        test  — data after val_end (may be empty if no local test data)
    """
    df.index = pd.to_datetime(df.index)
    train = df.loc[:train_end]
    val   = df.loc[train_end:val_end].iloc[1:]   # exclude train_end overlap
    test  = df.loc[val_end:].iloc[1:]            # exclude val_end overlap

    logger.info(
        f"Chronological split → "
        f"train: {len(train)} obs [{train.index[0].date()}–{train.index[-1].date()}]  "
        f"val: {len(val)} obs [{val.index[0].date() if len(val) else 'N/A'}–"
        f"{val.index[-1].date() if len(val) else 'N/A'}]  "
        f"test: {len(test)} obs"
    )
    return train, val, test


def get_macro_columns(asset: str) -> list:
    """Return ordered list of macro feature column names for a given asset."""
    base = ["real_rate", "DXY", "inflation_exp", "VIX", "liquidity_proxy"]
    if asset.lower() == "oil":
        base.append("demand_proxy")
    return base


def build_feature_matrix(df: pd.DataFrame, macro_cols: list) -> np.ndarray:
    """Extract macro feature matrix X from aligned DataFrame, drop any NaN rows."""
    X = df[macro_cols].values.astype(np.float64)
    return X


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract target log-return vector y from aligned DataFrame."""
    return df["log_return"].values.astype(np.float64)
