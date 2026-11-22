"""
INTEGRITY REVIEW of leverage_execution  --  step 5: where does the ~12.6x/yr turnover come from? (claim C9)
Constant-mix engine, Friday marks, lag 1, IS. Each weight change at a mark is attributed to ONE cause in priority:
  on/off switch  >  chop flip (er20 crosses 0.25)  >  Aug/Sep season factor change  >  vol-target rescale only
Read-only (prints only).  usage: python update_2026_11/verify_leverage_execution_integrity_turnover.py
"""
import sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

P = C.panel(); IDX = P.index
for nm, lev, vt in (("F2", "qld", 0.30), ("F3", "tqqq", 0.20)):
    w = C.strategy_weights(P, lev=lev, voltgt=vt)
    m = P["mark"].astype(bool)
    ron = P["above"].astype(bool); cln = (P["er20"] >= 0.25).fillna(False); sea = P["month"].isin((8, 9))
    tw = w[m]; d = tw.diff().abs().sum(axis=1)
    st = pd.DataFrame({"ron": ron[m], "cln": cln[m], "sea": sea[m]})
    ch = st.ne(st.shift())
    cause = np.where(ch["ron"], "on/off switch",
             np.where(~st["ron"], "off (no change)",
             np.where(ch["cln"], "chop flip",
             np.where(ch["sea"], "season Aug/Sep", "vol-target rescale"))))
    s = pd.DataFrame({"turn": d, "cause": cause}, index=tw.index)
    s = s[(s.index >= C.START) & (s.index <= C.FREEZE)]
    yrs = (C.FREEZE - C.START).days / 365.25
    tot = s["turn"].sum() / yrs
    g = s.groupby("cause")["turn"].agg(["sum", "count"])
    g["per_yr"] = g["sum"] / yrs; g["share"] = g["sum"] / s["turn"].sum()
    print(f"\n{nm}: constant-mix turnover {tot:.2f}/yr (IS); by cause:")
    for c, row in g.sort_values("per_yr", ascending=False).iterrows():
        big = s[(s.cause == c) & (s.turn > 0.04)]["turn"].sum() / yrs
        print(f"   {c:20s} {row.per_yr:5.2f}/yr  share {row.share*100:4.1f}%  events {int(row['count']):5d}  "
              f"of which in steps >2pp/leg (|dw|sum>0.04): {big:5.2f}/yr")
