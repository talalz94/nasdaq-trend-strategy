"""
Master pipeline — runs end to end on the clean 10yr data:
  1. verify data completeness
  2. rebuild indicator cache
  3. comprehensive sweep (train/test, regime, hour, weekday, sector, per-stock, per-regime)
  4. champion walk-forward
  5. timeframe comparison (1h vs 4h vs 1d) for the strongest signals
  6. emit consolidated report data
"""
import sys, time, json, numpy as np, pandas as pd
import bt
CACHE = bt.CACHE
DATA = "data_alpaca/stocks_1h.parquet"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------- 1. verify ----------
def verify():
    df = pd.read_parquet(DATA, columns=["ticker","datetime"])
    span = df.groupby("ticker")["datetime"].agg(["min","max","size"])
    reach = (pd.to_datetime(span["max"]) >= pd.Timestamp("2026-07-01")).sum()
    log(f"data: {df.ticker.nunique()} tickers, {len(df):,} rows, {reach} reach 2026-07, "
        f"median bars={span['size'].median():.0f}")
    if reach < df.ticker.nunique() * 0.8:
        log("WARNING: many tickers still truncated — check fetch!")
    return reach

# ---------- 5. timeframe comparison ----------
import sweep
from indicators import compute_all
TOP_SPECS = [
    dict(name="confl_os-53_stop0.02", type="confluence", os=-53, stop=0.02),
    dict(name="sqz_color_ma40", type="sqz_color", filters=["above_ma40"]),
    dict(name="sqz_zero_long", type="sqz_zero"),
    dict(name="mashift_reg", type="mashift", regime=True),
]
def resample_tf(raw, rule):
    out=[]
    for t,g in raw.groupby("ticker", sort=False):
        s=g.set_index("datetime")[["open","high","low","close","volume"]]
        a=s.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
        a=a.reset_index(); a["ticker"]=t; a["date"]=a["datetime"].dt.date.astype(str)
        out.append(a)
    return pd.concat(out, ignore_index=True)
def cache_from_raw(raw, minbars):
    cache={}
    for t,g in raw.groupby("ticker", sort=False):
        if len(g)<minbars: continue
        ind=compute_all(g[["datetime","date","open","high","low","close","volume"]])
        cache[t]={c:ind[c].to_numpy() for c in sweep.CACHE_COLS if c in ind}
    return cache
def timeframe_compare():
    raw=pd.read_parquet(DATA)
    raw["datetime"]=pd.to_datetime(raw["datetime"])
    rows=[]
    for tf,rule,minb in [("1h",None,600),("4h","4h",300),("1d","1D",300)]:
        r = raw if rule is None else resample_tf(raw, rule)
        cache = cache_from_raw(r, minb)
        for spec in TOP_SPECS:
            df=sweep.run_spec(cache, spec)
            if len(df)==0: continue
            _,te,full=sweep.split_eval(df)
            rows.append(dict(tf=tf, signal=spec["name"], n=full["n"], full_avg=full["avg"],
                             full_pf=full["pf"], te_n=te["n"], te_avg=te["avg"], te_pf=te["pf"]))
            log(f"  TF {tf:3s} {spec['name']:22s} full n={full['n']:6d} avg={full['avg']:6.1f} pf={full['pf']:.2f} | test avg={te['avg']:6.1f} pf={te['pf']:.2f}")
    pd.DataFrame(rows).to_csv(f"{CACHE}/timeframe_sweep.csv", index=False)
    return rows

def main():
    t0=time.time()
    log("=== STEP 1: verify data ==="); verify()
    log("=== STEP 2: rebuild indicator cache ===")
    import build_alpaca_cache; build_alpaca_cache.main()
    log("=== STEP 3: comprehensive sweep ===")
    sweep.main()
    log("=== STEP 4: champion walk-forward ===")
    import alpaca_wf; alpaca_wf.main(DATA)
    log("=== STEP 5: timeframe comparison ===")
    tfrows = timeframe_compare()
    log(f"=== ALL DONE in {(time.time()-t0)/60:.1f} min ===")

if __name__ == "__main__":
    main()
