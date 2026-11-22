"""Report series on the timing-luck-free basis (5-weekday tranche) + weekday comparison. Main session."""
import sys, json
sys.path.insert(0, "update_2026_11")
import numpy as np, pandas as pd
import common as C
P = C.panel(); idx = P.index
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
marks = {d: pd.Series(idx.dayofweek == i, index=idx) for i, d in enumerate(DAYS)}
CFG = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}

def curve(r, start=C.START, end=None):
    s = r[r.index >= start]
    if end is not None: s = s[s.index <= end]
    e = (1 + s.fillna(0)).cumprod().resample("ME").last()
    return [[t.strftime("%Y-%m"), round(float(v), 5)] for t, v in e.items()]

out = {"weekday": {}, "tranche_long": {}, "tranche_stats": {}, "friday_stats": {}}
R = {}
for k, kw in CFG.items():
    w = C.strategy_weights(P, **kw)
    rs = {d: C.backtest(P, w, mark=marks[d]) for d in DAYS}
    tr = sum(rs[d] for d in DAYS) / 5
    R[k] = tr
    out["weekday"][k] = {d: C.stats(rs[d], b=C.FREEZE) for d in DAYS}
    out["weekday"][k]["5-day average"] = C.stats(tr, b=C.FREEZE)
    out["tranche_long"][k] = curve(tr)
    out["tranche_stats"][k] = {"to_freeze": C.stats(tr, b=C.FREEZE), "since2000": C.stats(tr, a="2000-01-01", b=C.FREEZE),
                               "full": C.stats(tr), "oos": C.period_return(tr, "2026-07-25", "2026-12-31"),
                               "decades": C.by_decade(tr), "crises": C.crises(tr)}
    fr = C.backtest(P, w)
    out["friday_stats"][k] = {"to_freeze": C.stats(fr, b=C.FREEZE), "since2000": C.stats(fr, a="2000-01-01", b=C.FREEZE),
                              "oos": C.period_return(fr, "2026-07-25", "2026-12-31")}
q = P["qqq"].fillna(0)
out["tranche_long"]["QQQ buy & hold"] = curve(q)
out["qqq_stats"] = {"to_freeze": C.stats(q, b=C.FREEZE), "since2000": C.stats(q, a="2000-01-01", b=C.FREEZE),
                    "oos": C.period_return(q, "2026-07-25", "2026-12-31"), "decades": C.by_decade(q), "crises": C.crises(q)}
json.dump(out, open("update_2026_11/report_series2.json", "w"), indent=0, default=float)
for k in CFG:
    t = out["tranche_stats"][k]
    print(f"{k} tranche: to-freeze {C.fmt(t['to_freeze'])} | since2000 CAGR {t['since2000']['cagr']*100:.1f}% Cal {t['since2000']['calmar']:.2f} | OOS {t['oos']*100:+.1f}%")
    print("   decades:", {d: round(v['cagr']*100, 1) for d, v in t['decades'].items()})
    print("   crises :", {c: round(v*100, 1) for c, v in t['crises'].items()})
print("QQQ decades:", {d: round(v['cagr']*100, 1) for d, v in out['qqq_stats']['decades'].items()})
