"""
data_loader.py
==============
Load local Gold (weekly) and Brent Oil (daily→weekly) price CSVs.
Fetch real macro variables from FRED and yfinance.

Macro sources:
  - Real interest rate  → FRED: DFII10 (10-yr TIPS real yield)
  - Inflation exp       → FRED: T10YIE (10-yr breakeven)
  - DXY                 → yfinance: DX-Y.NYB
  - VIX                 → yfinance: ^VIX
  - Liquidity proxy     → yfinance: ^TNX (10-yr Treasury yield)
  - Demand proxy (oil)  → yfinance: USO  (fallback CL=F)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import get_logger

logger = get_logger("data_loader")

# ── FRED fetching ────────────────────────────────────────────────────────────

def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Download a single FRED series, return daily Series indexed by date."""
    try:
        import pandas_datareader.data as web
        s = web.DataReader(series_id, "fred", start, end)[series_id]
        return s
    except Exception as e:
        logger.warning(f"FRED pandas-datareader failed for {series_id}: {e}. Trying fredapi…")
    try:
        from fredapi import Fred
        api_key = os.environ.get("FRED_API_KEY", "")
        fred = Fred(api_key=api_key)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        return s
    except Exception as e2:
        logger.error(f"Could not fetch {series_id} from FRED: {e2}")
        return pd.Series(dtype=float)


def _fetch_yfinance(ticker: str, fallback: str, start: str, end: str,
                    interval: str = "1wk") -> pd.Series:
    """Download a yfinance ticker weekly Close, return Series indexed by date."""
    for t in [ticker, fallback]:
        try:
            df = yf.download(t, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) > 10:
                close = df["Close"].squeeze()
                close.name = t
                return close
        except Exception as e:
            logger.warning(f"yfinance failed for {t}: {e}")
    return pd.Series(dtype=float)


# ── Gold loader ──────────────────────────────────────────────────────────────

def load_gold(path: str) -> pd.DataFrame:
    """
    Parse XAU_1w_data.csv (semicolon-delimited, YYYY.MM.DD HH:MM format).
    Returns weekly DataFrame with columns: [close, log_return].
    Index: Friday-aligned DatetimeIndex.
    """
    raw = pd.read_csv(path, sep=";", header=0)
    raw.columns = [c.strip() for c in raw.columns]
    # Date format: "2004.06.06 00:00"
    raw["Date"] = pd.to_datetime(raw["Date"], format="%Y.%m.%d %H:%M")
    raw = raw.set_index("Date").sort_index()
    df = raw[["Close"]].rename(columns={"Close": "close"})
    # align to Friday
    df = df.resample("W-FRI").last().dropna()
    df["log_return"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df = df.dropna()
    logger.info(f"Gold loaded: {len(df)} weekly obs  [{df.index[0].date()} – {df.index[-1].date()}]")
    return df


# ── Oil loader ───────────────────────────────────────────────────────────────

def load_oil(path: str) -> pd.DataFrame:
    """
    Parse BrentOilPrices.csv (comma-delimited, DD-Mon-YY date format, daily).
    Resample to Friday-weekly close, compute log return.
    Returns DataFrame with columns: [close, log_return].
    """
    raw = pd.read_csv(path, header=0)
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], format="mixed", dayfirst=True)
    raw = raw.set_index("Date").sort_index()
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    # Resample daily → weekly Friday close
    df = raw[["Price"]].rename(columns={"Price": "close"})
    df = df.resample("W-FRI").last().dropna()
    df["log_return"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df = df.dropna()
    logger.info(f"Oil loaded:  {len(df)} weekly obs  [{df.index[0].date()} – {df.index[-1].date()}]")
    return df


# ── Macro data loader ────────────────────────────────────────────────────────

def load_macro(
    start: str,
    end: str,
    asset: str = "gold",
    cache_path: str = "data/raw/macro_cache.pkl",
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Fetch macro variables from FRED / yfinance and align to weekly Friday.

    Variables fetched:
      real_rate       — FRED DFII10 (10-yr TIPS real yield)
      inflation_exp   — FRED T10YIE (10-yr breakeven inflation)
      DXY             — yfinance DX-Y.NYB
      VIX             — yfinance ^VIX
      liquidity_proxy — yfinance ^TNX (10-yr Treasury yield, inverted ≈ liquidity)
      demand_proxy    — yfinance USO (oil demand proxy; only included when asset='oil')

    Returns DataFrame indexed by Friday-aligned DatetimeIndex.
    Forward-fill applied to handle FRED weekend/holiday gaps.
    """
    cache_key = f"{asset}_{start}_{end}"

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache_key in cache:
            logger.info(f"Macro data loaded from cache for key='{cache_key}'")
            return cache[cache_key]
    else:
        cache = {}

    logger.info(f"Fetching macro data [{start} → {end}] for asset='{asset}' …")

    # ── FRED series ──────────────────────────────────────────────────────────
    real_rate     = _fetch_fred("DFII10", start, end).rename("real_rate")
    inflation_exp = _fetch_fred("T10YIE", start, end).rename("inflation_exp")

    # ── yfinance series (weekly) ─────────────────────────────────────────────
    dxy     = _fetch_yfinance("DX-Y.NYB", "UUP",  start, end, "1wk").rename("DXY")
    vix     = _fetch_yfinance("^VIX",     "VIXY", start, end, "1wk").rename("VIX")
    tnx     = _fetch_yfinance("^TNX",     "SHY",  start, end, "1wk").rename("liquidity_proxy")

    # ── Build weekly Friday-aligned frame ────────────────────────────────────
    # Resample FRED daily → weekly Friday
    fred_weekly = pd.concat([real_rate, inflation_exp], axis=1)
    fred_weekly = fred_weekly.resample("W-FRI").last()

    # Align yfinance weekly (already weekly, but may not be Friday)
    yf_weekly = pd.concat([dxy, vix, tnx], axis=1)
    yf_weekly.index = pd.to_datetime(yf_weekly.index)
    yf_weekly = yf_weekly.resample("W-FRI").last()

    macro = pd.concat([fred_weekly, yf_weekly], axis=1)

    if asset.lower() == "oil":
        demand = _fetch_yfinance("USO", "CL=F", start, end, "1wk").rename("demand_proxy")
        demand.index = pd.to_datetime(demand.index)
        demand = demand.resample("W-FRI").last()
        macro = macro.join(demand, how="left")

    # ── Forward-fill (macro data published with lags) ────────────────────────
    macro = macro.ffill()

    # ── Trim to requested window ─────────────────────────────────────────────
    macro = macro.loc[start:end]

    logger.info(
        f"Macro data ready: {macro.shape[1]} variables, "
        f"{len(macro)} weeks  [{macro.index[0].date()} – {macro.index[-1].date()}]"
    )

    # ── Cache to disk ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    cache[cache_key] = macro
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"Macro data cached → {cache_path}")

    return macro
