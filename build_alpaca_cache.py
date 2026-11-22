"""Compute indicators on Alpaca data, save cache + coverage. Args: input outcache minbars."""
import sys, time, numpy as np, pandas as pd
import bt
from indicators import compute_all

CACHE = bt.CACHE

def main(inp="data_alpaca/stocks_1h.parquet", outname="ind_alpaca_1h.parquet", min_bars=600):
    t0 = time.time()
    raw = pd.read_parquet(inp)
    print(f"loaded {len(raw):,} rows, {raw.ticker.nunique()} tickers, {time.time()-t0:.0f}s", flush=True)
    out, cov = [], []
    for tkr, g in raw.groupby("ticker", sort=False):
        if len(g) < min_bars:
            cov.append((tkr, g.datetime.min(), len(g), "too_short")); continue
        ind = compute_all(g[["datetime","date","open","high","low","close","volume"]])
        ind["ticker"] = tkr
        out.append(ind)
        cov.append((tkr, g.datetime.min(), len(g), "ok"))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(f"{CACHE}/{outname}")
    covdf = pd.DataFrame(cov, columns=["ticker","first","nbars","status"])
    covdf.to_parquet(f"{CACHE}/alpaca_coverage.parquet")
    covdf["first"] = pd.to_datetime(covdf["first"])
    full = covdf[(covdf.status=="ok") & (covdf["first"] <= pd.Timestamp("2016-02-01"))]
    print(f"cache: {res.ticker.nunique()} usable tickers, {len(res):,} rows, {time.time()-t0:.0f}s", flush=True)
    print(f"full-2016 survivors: {len(full)} | later/short: {covdf.ticker.nunique()-len(full)}")
    print(f"median bars/ticker: {covdf[covdf.status=='ok'].nbars.median():.0f}")

if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a)>1 else "data_alpaca/stocks_1h.parquet",
         a[2] if len(a)>2 else "ind_alpaca_1h.parquet",
         int(a[3]) if len(a)>3 else 600)
