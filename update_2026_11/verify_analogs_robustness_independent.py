"""
verify_analogs_robustness_independent.py -- recompute a few load-bearing analogs numbers directly from
common.py (no code from analogs.py): 12m F2/F3/QQQ returns from selected 2022 start dates, the OOS
since-record / since-mark returns, today's state and two feature percentiles.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

P = C.panel(); idx = P.index
r2 = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief"))
r3 = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_phys", gold_w=0.15, off="ief"))
q = P["qqq"].fillna(0); ndx = P["ndx_ret"].fillna(0)


def fwd(r, d, h=252):
    i = idx.get_loc(pd.Timestamp(d))
    return float((1 + r.iloc[i + 1:i + h + 1]).prod() - 1)


print("12m forward from 2022 start dates (F3 / F2 / QQQ):")
for d in ["2022-01-07", "2022-01-28", "2022-02-09", "2022-02-14", "2022-04-28"]:
    print(f"  {d}: F3 {fwd(r3, d)*100:+.1f}%  F2 {fwd(r2, d)*100:+.1f}%  QQQ {fwd(q, d)*100:+.1f}%")


def per(r, a, b):
    ia, ib = idx.get_loc(pd.Timestamp(a)), idx.get_loc(pd.Timestamp(b))
    return float((1 + r.iloc[ia + 1:ib + 1]).prod() - 1)


last = idx[-1]
for nm, r in [("NDX", ndx), ("QQQ", q), ("F2", r2), ("F3", r3)]:
    print(f"  {nm}: record 10-06 -> exit mark 10-16 {per(r, '2026-10-06', '2026-10-16')*100:+.2f}%   "
          f"10-06 -> {last.date()} {per(r, '2026-10-06', last)*100:+.2f}%   10-16 -> {last.date()} "
          f"{per(r, '2026-10-16', last)*100:+.2f}%   freeze -> {last.date()} {per(r, C.FREEZE, last)*100:+.2f}%")
lvl = P["ndx"]
print(f"  NDX {lvl.iloc[-1]:.0f}, record {lvl.max():.0f} on {lvl.idxmax().date()}, dd {lvl.iloc[-1]/lvl.max()-1:+.4f}")
below = ~P["above"].astype(bool)
tail = below[idx >= "2026-10-01"]
print("  below-EMA flags since 2026-10-01: " + "".join("B" if v else "a" for v in tail.values))
ins = (idx >= C.START) & (idx <= C.FREEZE)
for col in ["vol20", "vix"]:
    x = P[col].ffill()
    a = x[ins].dropna()
    print(f"  {col} today {x.iloc[-1]:.4f} pct-rank in-sample {(a < x.iloc[-1]).mean()*100:.1f}")
