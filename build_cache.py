"""
Read the S&P 500 price store (read-only) and cache OHLC + indicators per
timeframe into the scratchpad, so the backtests never recompute or touch source.
"""
import glob, os, sys, time
import pandas as pd
from indicators import compute_all

HERE = os.path.dirname(os.path.abspath(__file__))
# the S&P 500 price store is a separate local project (not included); override with STOCK_PRICES_DIR
SRC = os.environ.get("STOCK_PRICES_DIR", os.path.join(HERE, "..", "..", "Stock Screener", "data", "prices"))
CACHE = os.environ.get("CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
os.makedirs(CACHE, exist_ok=True)

MIN_BARS = 40  # need enough warmup for indicators


def load_interval(iv: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{SRC}/{iv}/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def build(iv: str) -> pd.DataFrame:
    raw = load_interval(iv)
    out = []
    for tkr, g in raw.groupby("ticker", sort=False):
        if len(g) < MIN_BARS:
            continue
        ind = compute_all(g[["datetime", "date", "open", "high", "low", "close", "volume"]])
        ind["ticker"] = tkr
        out.append(ind)
    res = pd.concat(out, ignore_index=True)
    path = f"{CACHE}/ind_{iv}.parquet"
    res.to_parquet(path)
    return res


if __name__ == "__main__":
    for iv in ["15m", "1h", "4h", "1d"]:
        t = time.time()
        res = build(iv)
        print(f"{iv}: {res['ticker'].nunique()} tickers, {len(res)} rows, "
              f"{time.time()-t:.1f}s -> ind_{iv}.parquet")
