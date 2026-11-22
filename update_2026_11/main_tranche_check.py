"""
Independent (main-session) check of the rebalance-timing-luck finding.
For each weekday d (Mon..Fri) rebalance on every trading day that is weekday d; the
'tranche' book holds 1/5 of capital in each weekday book (= average of the 5 return streams,
exact under the constant-mix convention used everywhere). Same rules, costs, lag=1.
"""
import sys, json
sys.path.insert(0, "update_2026_11")
import numpy as np, pandas as pd
import common as C
P = C.panel()
idx = P.index
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
marks = {d: pd.Series(idx.dayofweek == i, index=idx) for i, d in enumerate(DAYS)}

CFG = {"F1-IEF": dict(lev="qqq", voltgt=0.20), "F2-IEF": dict(lev="qld", voltgt=0.30),
       "F3-IEF": dict(lev="tqqq", voltgt=0.20), "F3-TLT": dict(lev="tqqq", voltgt=0.20, off="tlt")}
out = {}
for name, kw in CFG.items():
    w = C.strategy_weights(P, **kw)
    rs = {d: C.backtest(P, w, mark=marks[d]) for d in DAYS}
    rs["tranche"] = sum(rs[d] for d in DAYS) / 5
    rs["engine-Fri (P.mark)"] = C.backtest(P, w)          # holiday-Friday weeks skipped, as July
    out[name] = {}
    print(f"\n{name}")
    for k, r in rs.items():
        s = C.stats(r); s2 = C.stats(r, a="2000-01-01", b=C.FREEZE); s3 = C.stats(r, b=C.FREEZE)
        oos = C.period_return(r, "2026-07-25", "2026-12-31")
        out[name][k] = dict(full=s, to_freeze=s3, since2000=s2, oos=oos)
        print(f"  {k:20s} to-freeze CAGR {s3['cagr']*100:5.2f}% DD {s3['maxdd']*100:6.1f}% Cal {s3['calmar']:.2f} | "
              f"2000-freeze CAGR {s2['cagr']*100:5.2f}% DD {s2['maxdd']*100:6.1f}% Cal {s2['calmar']:.2f} | OOS {oos*100:+6.2f}%")
qqq = P["qqq"].fillna(0)
s3 = C.stats(qqq, b=C.FREEZE); s2 = C.stats(qqq, a="2000-01-01", b=C.FREEZE)
print(f"\nQQQ buy&hold        to-freeze CAGR {s3['cagr']*100:5.2f}% DD {s3['maxdd']*100:6.1f}% Cal {s3['calmar']:.2f} | "
      f"2000-freeze CAGR {s2['cagr']*100:5.2f}% DD {s2['maxdd']*100:6.1f}% Cal {s2['calmar']:.2f}")
json.dump(out, open("update_2026_11/main_tranche_check.json", "w"), indent=1, default=float)
