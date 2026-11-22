"""
INTEGRITY REVIEW of update_2026_11/ucits.py -- step 7: NaN / zero-return runs inside the backtest windows,
and component B (UCITS lag2 3bp - US lag2 3bp) with calendar 2020 removed. Read-only; prints only.
"""
import os, io, contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index('US2 = dict(lev="qld"')
g = {"__name__": "ucits_partial", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code[:cut], SRC, "exec"), g)
C, P2, END, run = g["C"], g["P2"], g["END"], g["run"]

print("NaN and exact-zero daily returns inside the windows (panel US calendar)")
for c, a in [("u_lqq", "2011-04-28"), ("u_cbu0", "2011-04-28"), ("u_igln", "2011-04-28"), ("u_qqq3", "2013-01-03"),
             ("u_ibtm", "2009-01-21"), ("u_phau", "2009-01-21"), ("qld", "2011-04-28"), ("ief", "2011-04-28"),
             ("gold_phys", "2011-04-28")]:
    s = P2[c][(P2.index >= a) & (P2.index <= END)]
    z = (s == 0)
    worst = z.groupby(s.index.year).mean().sort_values(ascending=False).head(3)
    print(f"  {c:9s} NaN {int(s.isna().sum()):3d}  zero-return share {z.mean()*100:5.1f}%  worst years "
          + " ".join(f"{y}:{v*100:.0f}%" for y, v in worst.items()))

with contextlib.redirect_stdout(io.StringIO()):
    rU, _ = run("x", "u_lqq", 0.30, "u_cbu0", "u_igln", lag=2)
    rS, _ = run("x", "qld", 0.30, "ief", "gold_phys", lag=2)
    rU3, _ = run("x", "u_qqq3", 0.20, "u_cbu0", "u_igln", lag=2)
    rS3, _ = run("x", "tqqq", 0.20, "ief", "gold_phys", lag=2)
for nm, x, y, a in [("F2", rU, rS, "2011-04-28"), ("F3", rU3, rS3, "2013-01-03")]:
    m = (x.index >= a) & (x.index <= END)
    full = C.stats(x[m], start=None)["cagr"] - C.stats(y[m], start=None)["cagr"]
    ex = m & (x.index.year != 2020)
    xx, yy = x[ex], y[ex]
    yrs = (END - pd.Timestamp(a)).days / 365.25 - 1.0
    gx, gy = np.prod(1 + xx) ** (1 / yrs) - 1, np.prod(1 + yy) ** (1 / yrs) - 1
    neg = sum(1 for yr in range(int(a[:4]), 2027) if np.prod(1 + x[m & (x.index.year == yr)]) < np.prod(1 + y[m & (y.index.year == yr)]))
    print(f"{nm}: component B full window {full*100:+.2f}%/yr; excluding calendar 2020 {(gx-gy)*100:+.2f}%/yr; "
          f"years with B<0: {neg} of {2027-int(a[:4])}")

print("\nComponent A (one-session delay: US lag2 - US lag1) by period, F2 (QLD vt .30) and F3 (TQQQ vt .20)")
P = C.panel()
for lev, vt in [("qld", 0.30), ("tqqq", 0.20)]:
    w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", off="ief")
    r1, r2 = C.backtest(P, w, lag=1), C.backtest(P, w, lag=2)
    out = []
    for a, b in [("1988-01-01", "2006-12-31"), ("2007-01-01", "2026-07-24"), ("1988-01-01", "2026-07-24"),
                 ("2009-01-21", str(END.date())), ("2011-04-28", str(END.date()))]:
        s1, s2 = C.stats(r1, a, b, start=None), C.stats(r2, a, b, start=None)
        out.append(f"{a[:4]}-{b[:4]} {(s2['cagr']-s1['cagr'])*100:+.2f}")
    bb = C.block_bootstrap_cagr_diff(r2[:"2026-07-24"], r1[:"2026-07-24"])
    print(f"  {lev}: " + " | ".join(out) + f" | 1988-freeze bootstrap [{bb['lo']*100:+.2f}, {bb['hi']*100:+.2f}]")
