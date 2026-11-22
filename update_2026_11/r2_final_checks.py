"""
r2 FINAL CRITIC - quick reproduction checks used in the final decision (read-only on project data).
(1) US basis (5-weekday tranche, lag 1, 5 bps) vs QQQ on 2010-01-01..2026-07-24 and 2013-01-01..freeze:
    checks the integrity reviewer's 'material misstatement' on recommendation 4 of candidates_tranche.
(2) The same tranche basis IS88/IS00/IS07 to confirm the PLAN rows from the frozen code path.
Nothing is tuned; all parameters are the frozen July values. In-sample windows end at the freeze.
Output JSON only to argv[1] (scratchpad).
"""
import sys, os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

OUT = sys.argv[1] if len(sys.argv) > 1 else None
P = C.panel()
idx = P.index
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
marks = {d: pd.Series(idx.dayofweek == i, index=idx) for i, d in enumerate(DAYS)}
TIERS = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
WIN = {"IS88": ("1988-01-01", C.FREEZE), "IS00": ("2000-01-01", C.FREEZE), "IS07": ("2007-01-01", C.FREEZE),
       "S2010": ("2010-01-01", C.FREEZE), "S2013": ("2013-01-01", C.FREEZE)}


def ulcer(r):
    e = (1 + r).cumprod()
    dd = e / e.cummax() - 1
    return float(np.sqrt((dd ** 2).mean()) * 100)


res = {}
for t, kw in TIERS.items():
    w = C.strategy_weights(P, **kw)
    tr = sum(C.backtest(P, w, cost=5e-4, mark=marks[d]) for d in DAYS) / 5
    res[t] = {}
    for k, (a, b) in WIN.items():
        s = C.stats(tr, a=a, b=b, start=None)
        rr = tr[(tr.index >= pd.Timestamp(a)) & (tr.index <= pd.Timestamp(b))]
        res[t][k] = dict(cagr=round(s["cagr"] * 100, 2), maxdd=round(s["maxdd"] * 100, 1),
                         calmar=round(s["calmar"], 3), ulcer=round(ulcer(rr), 2))
q = P["qqq"].fillna(0)
res["QQQ"] = {}
for k, (a, b) in WIN.items():
    s = C.stats(q, a=a, b=b, start=None)
    rr = q[(q.index >= pd.Timestamp(a)) & (q.index <= pd.Timestamp(b))]
    res["QQQ"][k] = dict(cagr=round(s["cagr"] * 100, 2), maxdd=round(s["maxdd"] * 100, 1),
                         calmar=round(s["calmar"], 3), ulcer=round(ulcer(rr), 2))
txt = json.dumps(res, indent=1)
print(txt)
if OUT:
    open(OUT, "w").write(txt)
