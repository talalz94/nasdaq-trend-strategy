"""Isolation test of the two new indicators on stocks (1h) and gold (1h,4h)."""
import numpy as np, pandas as pd
import bt, assets

def bench(cache, bars, cost):
    c = 2*cost/1e4; h=max(int(round(bars)),1); rr=[]
    for d in cache.values():
        C=d["close"]
        if len(C)>h:
            f=C[h:]/C[:-h]-1-c; rr.append(f[np.isfinite(f)])
    r=np.concatenate(rr); return float(r.mean())

def boot_p(r,n=8000,seed=3):
    if len(r)<50: return np.nan
    rng=np.random.default_rng(seed)
    return float((rng.choice(r,size=(n,len(r)),replace=True).mean(axis=1)<=0).mean())

def run(cache, strats, cost, label):
    print(f"\n===== {label} (net {cost}bps/side) =====")
    print(f"{'strategy':22s}{'n':>6}{'win':>6}{'avg_bps':>9}{'edge':>8}{'pf':>6}{'shrp':>6}{'bars':>6}{'p0':>6}")
    for s in strats:
        for stop in [None]:
            tr=bt.run_strategy(cache,s,stop=stop,cost_bps=cost)
            if len(tr)==0: print(f"{s:22s}   (no trades)"); continue
            m=bt.metrics(tr); e=(m['avg']-bench(cache,m['avg_bars'],cost))*1e4
            p0=boot_p(tr['ret'].to_numpy())
            print(f"{s:22s}{m['n']:6d}{m['win']:6.2f}{m['avg']*1e4:9.2f}{e:8.2f}"
                  f"{m['pf']:6.2f}{m['sharpe']:6.2f}{m['avg_bars']:6.1f}"
                  f"{(p0 if not np.isnan(p0) else -1):6.2f}")

# STOCKS 1h + 15m (MA-shift needs depth -> 15m)
stk1h = bt.load_cache("1h")
run(stk1h, bt.STRATS_P2, 2.0, "STOCKS 1h — new indicators")
stk15 = bt.load_cache("15m")
run(stk15, ["EMA8_slope_long","EMA8_slope_both","MAShift","MAShift_long"], 2.0, "STOCKS 15m — new indicators")

# GOLD 1h + 4h
for tf in ["1h","4h"]:
    gc = assets.gold_cache(tf)
    run(gc, bt.STRATS_P2, 1.5, f"GOLD {tf} — new indicators")
