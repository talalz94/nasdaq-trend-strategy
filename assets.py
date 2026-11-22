"""
Multi-asset loaders -> bt-compatible cache dicts. All source dirs are READ-ONLY.
An asset 'cache' is {name: {col: np.ndarray}} with the columns bt.simulate needs.
"""
from __future__ import annotations
import glob
import os
import numpy as np, pandas as pd
from indicators import compute_all

HERE = os.path.dirname(os.path.abspath(__file__))
# local data from two separate projects (not included in this repository); override with environment variables
GOLD_DIR = os.environ.get("GOLD_DATA_DIR", os.path.join(HERE, "..", "..", "Binance", "data"))
STOCK_DIR = os.environ.get("STOCK_PRICES_DIR", os.path.join(HERE, "..", "..", "Stock Screener", "data", "prices"))

CACHE_COLS = ["open", "high", "low", "close", "wt1", "wt2", "wt_cross_up",
              "wt_cross_dn", "mom", "mom_color", "mom_rising", "sqz_on",
              "sqz_off", "sqz_release", "datetime",
              "ma8", "ma8_up", "ma8_turn_up", "ma8_turn_dn",
              "mashift_osc", "mashift_regime_up", "mashift_sig_up",
              "mashift_sig_dn", "mashift_up"]


def cache_from_df(df: pd.DataFrame, name: str, extra=None) -> dict:
    """df must have datetime, date, open, high, low, close, volume (sorted)."""
    ind = compute_all(df)
    cols = CACHE_COLS + (extra or [])
    return {name: {c: ind[c].to_numpy() for c in cols if c in ind}}


def load_gold(tf: str = "1h", source: str = "dukascopy_cfd") -> pd.DataFrame:
    p = f"{GOLD_DIR}/{source}_XAUUSD_{tf}.parquet"
    df = pd.read_parquet(p).rename(columns={"open_time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date.astype(str)
    return df.sort_values("datetime").reset_index(drop=True)


def gold_cache(tf: str = "1h") -> dict:
    return cache_from_df(load_gold(tf), "XAUUSD")


if __name__ == "__main__":
    for tf in ["1h", "4h"]:
        c = gold_cache(tf)["XAUUSD"]
        print(f"gold {tf}: {len(c['close'])} bars, "
              f"wt cross_up={int(c['wt_cross_up'].sum())}, releases={int(c['sqz_release'].sum())}")
