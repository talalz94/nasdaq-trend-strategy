"""Chart series for the Nov-2026 report (main session). Extend CONFIGS once the workflows finish."""
import sys, json
sys.path.insert(0, "update_2026_11")
import numpy as np, pandas as pd
import common as C
P = C.panel()

CONFIGS = {
    "F2-IEF": dict(lev="qld", voltgt=0.30, off="ief", gold="gold_phys"),
    "F3-IEF": dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys"),
    "F3-TLT (as scanner)": dict(lev="tqqq", voltgt=0.20, off="tlt", gold="gold_phys"),
}
R = {k: C.backtest(P, C.strategy_weights(P, **v)) for k, v in CONFIGS.items()}
R["QQQ buy & hold"] = P["qqq"].fillna(0)
R["TQQQ buy & hold"] = P["tqqq"].fillna(0)

def monthly_curve(r, start=C.START):
    e = (1 + r[r.index >= start].fillna(0)).cumprod().resample("ME").last()
    return [[t.strftime("%Y-%m"), round(float(v), 5)] for t, v in e.items()]

def dd_curve(r, start=C.START):
    e = (1 + r[r.index >= start].fillna(0)).cumprod()
    d = (e / e.cummax() - 1).resample("ME").min()
    return [[t.strftime("%Y-%m"), round(float(v), 4)] for t, v in d.items()]

out = {"asof": str(P.index[-1].date()), "long": {}, "dd": {}, "oos": {}, "stats": {}, "oos_stats": {}}
for k, r in R.items():
    out["long"][k] = monthly_curve(r)
    out["dd"][k] = dd_curve(r)
    out["stats"][k] = C.stats(r)
# daily OOS path (real ETFs in this window) indexed to 100 at the 2026-07-24 close
oos = P.index[P.index > C.FREEZE]
for k, r in R.items():
    e = (1 + r.loc[oos].fillna(0)).cumprod() * 100
    out["oos"][k] = [[t.strftime("%Y-%m-%d"), round(float(v), 3)] for t, v in e.items()]
    s = r.loc[oos]; eq = (1 + s).cumprod()
    out["oos_stats"][k] = dict(ret=float(eq.iloc[-1] - 1), maxdd=float((eq / eq.cummax() - 1).min()))
# held weights through the OOS window (what the rules actually held)
for k in ["F2-IEF", "F3-IEF"]:
    hw = C.held_weights(P, C.strategy_weights(P, **CONFIGS[k])).loc[oos]
    wk = hw[P.loc[oos, "mark"]]
    out.setdefault("oos_weights", {})[k] = [
        {"date": t.strftime("%Y-%m-%d"), **{c: round(float(v), 3) for c, v in row.items()}} for t, row in wk.iterrows()]
# market dashboard
last = P.index[-1]
ndx = P["ndx"]
ath = ndx.cummax()
below = (~P["above"]).astype(int)
streak = int(below[::-1].cumprod().sum())
out["market"] = dict(
    date=str(last.date()), ndx=float(ndx.iloc[-1]), ath=float(ath.iloc[-1]), ath_date=str(ndx.idxmax().date()),
    dd_from_ath=float(ndx.iloc[-1] / ath.iloc[-1] - 1), ema100=float(P["ema100"].iloc[-1]),
    days_below_ema=streak, vol20=float(P["vol20"].iloc[-1]), er20=float(P["er20"].iloc[-1]),
    tnx=float(P["tnx"].iloc[-1]), tnx_6m_change=float(P["tnx"].iloc[-1] - P["tnx"].iloc[-127]),
    curve=float(P["curve"].iloc[-1]), dxy=float(P["dxy"].iloc[-1]),
    dxy_6m_change=float(P["dxy"].iloc[-1] / P["dxy"].iloc[-127] - 1), vix=float(P["vix"].iloc[-1]),
    gold_3m=float((1 + P["gold_phys"].iloc[-63:]).prod() - 1))
json.dump(out, open("update_2026_11/report_series.json", "w"), indent=0, default=str)
print(json.dumps(out["market"], indent=1))
print({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out["oos_stats"].items()})
print("weeks in OOS log:", len(out["oos_weights"]["F2-IEF"]))
for row in out["oos_weights"]["F2-IEF"]: print(row)
