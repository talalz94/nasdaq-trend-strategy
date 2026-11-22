"""
INTEGRITY REVIEW of oos_live -- step 5: anatomy of the 1988-1999 lag-1 advantage (claim C8). Prints only.

For every Friday EMA-gate flip, the NDX return of the next session (the day lag-1 holds the new weights
and lag-2 still holds the old ones). If the 1990s lag-1 edge is real momentum it should look the same in
every era; if it is a stale-index artifact it should be concentrated where NDX lags the S&P 500.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

P = C.panel()
idx = P.index
mk = idx[P["mark"].values]
ab = P["above"].astype(int)
flip = ab.loc[mk].diff().fillna(0)
pos = {t: i for i, t in enumerate(idx)}
rows = []
for t, f in flip.items():
    if f == 0 or pos[t] + 1 >= len(idx):
        continue
    nx = idx[pos[t] + 1]
    rows.append(dict(t=t, dir="up" if f > 0 else "down", nxt=P.at[nx, "ndx_ret"], ndx_t=P.at[t, "ndx_ret"]))
F = pd.DataFrame(rows).set_index("t")
w = C.strategy_weights(P, lev="tqqq", voltgt=0.20, off="ief")
r1, r2 = C.backtest(P, w, lag=1), C.backtest(P, w, lag=2)
for nm, (a, b) in {"1988-1994": ("1988-01-01", "1994-12-31"), "1995-1999": ("1995-01-01", "1999-12-31"),
                   "2000-2009": ("2000-01-01", "2009-12-31"), "2010-freeze": ("2010-01-01", "2026-07-24")}.items():
    f = F.loc[a:b]
    up, dn = f[f.dir == "up"], f[f.dir == "down"]
    s1, s2 = C.stats(r1, a, b, start=None), C.stats(r2, a, b, start=None)
    x = P.loc[a:b]
    print(f"{nm}: F3 CAGR lag1 {s1['cagr']*100:5.1f}% lag2 {s2['cagr']*100:5.1f}% (gap {100*(s1['cagr']-s2['cagr']):+.1f}pp) | "
          f"flips up {len(up)} next-day NDX mean {up.nxt.mean()*100:+.2f}% | down {len(dn)} next-day mean {dn.nxt.mean()*100:+.2f}% | "
          f"NDX daily autocorr {x.ndx_ret.autocorr(1):+.3f} | corr(NDX_t, SPX_t-1) {x.ndx_ret.corr(x.spx.shift(1)):+.3f}")
print("all-days mean NDX return 1988-1999:", round(P.loc['1988':'1999', 'ndx_ret'].mean() * 100, 3), "%")
