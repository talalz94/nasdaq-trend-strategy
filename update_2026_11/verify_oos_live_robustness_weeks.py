"""verify_oos_live_robustness_weeks.py -- frequency of large F3 / F2 weekly losses (Friday-to-Friday, lag 1), by era.
Read-only; prints only."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
P = C.panel(); IDX = P.index; HIST_A = IDX[IDX >= C.START][0]; END = IDX[-1]
m = IDX[P["mark"].values]
for nm, kw in [("F3", dict(lev="tqqq", voltgt=0.20, off="ief")), ("F2", dict(lev="qld", voltgt=0.30, off="ief"))]:
    r = C.backtest(P, C.strategy_weights(P, **kw))
    eq = (1 + r.fillna(0)).cumprod().loc[m]
    wk = (eq / eq.shift(1) - 1).dropna(); wk = wk[wk.index >= HIST_A]
    for thr in [-0.10, -0.12, -0.1388]:
        x = wk[wk <= thr]
        yrs = (END - HIST_A).days / 365.25
        re_ = x[x.index >= "2010-02-12"]; pre = x[x.index < "2008-01-01"]
        print(f"{nm} weeks <= {thr*100:6.2f}% (1988-01..2026-11-17 incl. OOS): {len(x):2d} = 1 per {yrs/max(len(x),1):4.1f}y | "
              f"1988-2007: {len(pre)} | 2008+: {len(x)-len(pre)} | real-TQQQ era 2010-02+: {len(re_)} "
              f"({', '.join(str(d.date()) for d in re_.index)})")
