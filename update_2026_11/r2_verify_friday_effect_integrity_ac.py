"""
Integrity reviewer (C6/C7 attribution): NDX next-session autocorrelation by weekday (Pearson / Spearman) and the
F2 Friday-minus-tranche first-holding-day share + lag-2 removal, for 1988-94, 1995-99, 2000-06, 2007-26.
Also: same split excluding the 5 largest |Monday| moves after Friday decisions. Writes nothing to the project.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

P = C.panel(); IDX = P.index; dow = IDX.dayofweek.values
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
r = P["ndx_ret"].fillna(0).values
nx = np.r_[r[1:], np.nan]
w = C.strategy_weights(P, lev="qld", voltgt=0.30)
rs1 = {d: C.backtest(P, w, mark=pd.Series(dow == i, index=IDX)) for i, d in enumerate(DAYS)}
rs2 = {d: C.backtest(P, w, mark=pd.Series(dow == i, index=IDX), lag=2) for i, d in enumerate(DAYS)}
t1 = sum(rs1[d] for d in DAYS) / 5; t2 = sum(rs2[d] for d in DAYS) / 5
g = (rs1["Fri"] - t1).values
k1 = np.r_[False, (dow == 4)[:-1]]
for nm, (a, b) in {"1988-94": ("1988-01-01", "1994-12-31"), "1995-99": ("1995-01-01", "1999-12-31"),
                   "2000-06": ("2000-01-01", "2006-12-31"), "2007-26": ("2007-01-01", "2026-07-24")}.items():
    m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    mm = m & np.r_[m[1:], False]
    ac = {}
    for i, d in enumerate(DAYS):
        s = mm & (dow == i)
        ac[d] = (round(float(np.corrcoef(r[s], nx[s])[0, 1]), 3),
                 round(float(pd.Series(r[s]).corr(pd.Series(nx[s]), method="spearman")), 3))
    allac = round(float(np.corrcoef(r[mm], nx[mm])[0, 1]), 3)
    g1 = (C.stats(rs1["Fri"], a=a, b=b)["cagr"] - C.stats(t1, a=a, b=b)["cagr"]) * 100
    g2 = (C.stats(rs2["Fri"], a=a, b=b)["cagr"] - C.stats(t2, a=a, b=b)["cagr"]) * 100
    fd = g[m & k1].sum() / g[m].sum()
    # robustness: drop the 5 largest |gap| first-holding days
    idxk = np.where(m & k1)[0]
    top = idxk[np.argsort(-np.abs(g[idxk]))[:5]]
    keep = np.ones(len(g), bool); keep[top] = False
    fd_ex = g[m & k1 & keep].sum() / g[m & keep].sum()
    print(f"{nm}: next-session corr (pearson, spearman) {ac} | all-days AC1 {allac} | gap lag1 {g1:+.2f} lag2 {g2:+.2f} "
          f"removed {1 - g2 / g1:.2f} | first-day share {fd:.2f} (excl. top-5 first days {fd_ex:.2f}; "
          f"top-5 dates {[str(IDX[i].date()) for i in top]})", flush=True)
