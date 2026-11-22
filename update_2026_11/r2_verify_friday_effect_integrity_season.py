"""
Integrity reviewer: (S1) OOS late-Jul/early-Aug 2026 - target/held QLD weights per weekday book around the
Aug x0.5 seasonal boundary (decision-day month); (S2) IS share of the F2 Friday-minus-tranche gap earned in the
seasonal-boundary windows (Jul 25..Aug 10 and Sep 25..Oct 10) and with the seasonal rule switched off;
(S3) IXIC 1975-87 next-session autocorrelation by weekday (stale-close vs weekend attribution, C7).
Writes nothing to the project.
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

J = json.load(open(os.path.join(HERE, "r2_friday_effect.json")))
P = C.panel(); IDX = P.index
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
dow = IDX.dayofweek.values

# S1
w = C.strategy_weights(P, lev="qld", voltgt=0.30)
m = (IDX >= "2026-07-17") & (IDX <= "2026-08-11")
tab = pd.DataFrame({"dow": IDX[m].day_name().str[:3], "month": P["month"][m], "above": P["above"][m],
                    "vol20": P["vol20"][m].round(3), "er20": P["er20"][m].round(2), "target_qld": w["qld"][m].round(3),
                    "ndx_ret_%": (P["ndx_ret"][m] * 100).round(2)})
for i, d in enumerate(DAYS):
    tab[f"held_{d}"] = C.held_weights(P, w, mark=pd.Series(dow == i, index=IDX))["qld"][m].round(3)
print(tab.to_string(), flush=True)

# S2
rs = {d: C.backtest(P, w, mark=pd.Series(dow == i, index=IDX)) for i, d in enumerate(DAYS)}
tr = sum(rs[d] for d in DAYS) / 5
g = rs["Fri"] - tr
isw = (IDX >= C.START) & (IDX <= C.FREEZE)
md = IDX.month * 100 + IDX.day
bnd = ((md >= 725) & (md <= 810)) | ((md >= 925) & (md <= 1010))
yrs = (C.FREEZE - C.START).days / 365.25
print("S2 IS arith gap pp/yr %.3f ; in seasonal-boundary windows %.3f (share %.2f, %.1f%% of days)"
      % (g[isw].sum() / yrs * 100, g[isw & bnd].sum() / yrs * 100, g[isw & bnd].sum() / g[isw].sum(),
         100 * (isw & bnd).sum() / isw.sum()), flush=True)
w0 = C.strategy_weights(P, lev="qld", voltgt=0.30, seasonal=False)
rs0 = {d: C.backtest(P, w0, mark=pd.Series(dow == i, index=IDX)) for i, d in enumerate(DAYS)}
tr0 = sum(rs0[d] for d in DAYS) / 5
for wn, (a, b) in {"IS": (C.START, C.FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", C.FREEZE)}.items():
    f, t = C.stats(rs0["Fri"], a=a, b=b), C.stats(tr0, a=a, b=b)
    print(f"S2 seasonal OFF {wn}: Fri-tranche CAGR {(f['cagr'] - t['cagr']) * 100:+.2f}pp Calmar {f['calmar'] - t['calmar']:+.3f}", flush=True)
oos = IDX >= "2026-07-25"
print("S2 OOS total Fri-tranche (seasonal ON) %.2fpp ; (seasonal OFF) %.2fpp" % (
    (C.period_return(rs["Fri"], "2026-07-25", "2026-12-31") - C.period_return(tr, "2026-07-25", "2026-12-31")) * 100,
    (C.period_return(rs0["Fri"], "2026-07-25", "2026-12-31") - C.period_return(tr0, "2026-07-25", "2026-12-31")) * 100), flush=True)

# S3
ac = J["c_cross_market"]["IXIC"]["windows"]["pre-NDX 1975-1987"]["next_session_autocorr_by_weekday"]
print("S3 IXIC 1975-87 next-session corr by weekday:", {d: round(v["pearson"], 3) for d, v in ac.items()}, flush=True)
for mk in ("NDX-2xsim (reference)", "SPX", "N225", "DAX", "HSI-own-calendar (robustness)"):
    for wn in ("H1", "H2"):
        ac = J["c_cross_market"][mk]["windows"][wn]["next_session_autocorr_by_weekday"]
        print(f"S3 {mk[:12]:12s} {wn}:", {d: round(v["pearson"], 3) for d, v in ac.items()}, flush=True)
