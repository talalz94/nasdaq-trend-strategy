"""
r2_verify_scenario_stress_robustness_followup.py -- ROBUSTNESS follow-up: bear-replay phase sensitivity.

The specialist's bear numbers (2022-01-21 / 2000-04-14 replays, F2 Friday 12m -12.7% / -9.6%, maxDD -19.3% /
-18.6%) are single paths. Here every replay start offset -10..+10 trading days around X is run from today's state
(literal convention otherwise; offset 0 = specialist 'literal', +1 = 'after'), reporting F1/F2/F3 Friday and tranche
12m return and max drawdown, and how often F2 breaches the in-sample worst rolling-12m (-21.3% Friday, -25.3% tranche).
usage: python -u update_2026_11/r2_verify_scenario_stress_robustness_followup.py <out.json>
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_verify_scenario_stress_robustness_lib as L  # noqa: E402

OUT = os.path.abspath(sys.argv[1])
assert not OUT.startswith(os.path.abspath(HERE))
g = L.load_specialist()
P = g["P"]
idx = P.index
H = g["H"]
F = L.Fast()
J = json.load(open(os.path.join(HERE, "r2_scenario_stress.json")))
worst_fri = J["realism"]["insample_rolling12m_F2_Fri"]["min"]
worst_tr = J["realism"]["insample_rolling12m_F2_tr"]["min"]
res = {}
for dte in J["replays"]:
    i = idx.get_loc(pd.Timestamp(dte))
    rows = []
    for o in range(-10, 11):
        sg = P.iloc[i + o:i + o + H]
        assert sg.index[-1] <= g["C"].FREEZE
        rr = F.run(sg["ndx_ret"].values, sg["ief"].values, sg["gold_phys"].values)
        row = dict(off=o, QQQ=L.cum(rr["qqq"]), QQQ_mdd=L.mdd(rr["qqq"]))
        for t in ("F1", "F2", "F3"):
            for b in ("Fri", "tr"):
                row[f"{t}_{b}"] = L.cum(rr["rets"][t][b])
                row[f"{t}_{b}_mdd"] = L.mdd(rr["rets"][t][b])
        rows.append(row)
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    res[dte] = dict(rows=rows,
                    F2_Fri=dict(min=float(a["F2_Fri"].min()), median=float(np.median(a["F2_Fri"])), max=float(a["F2_Fri"].max())),
                    F2_tr=dict(min=float(a["F2_tr"].min()), median=float(np.median(a["F2_tr"])), max=float(a["F2_tr"].max())),
                    F2_Fri_mdd=dict(min=float(a["F2_Fri_mdd"].min()), median=float(np.median(a["F2_Fri_mdd"]))),
                    F2_tr_mdd=dict(min=float(a["F2_tr_mdd"].min()), median=float(np.median(a["F2_tr_mdd"]))),
                    n_F2_Fri_below_insample_worst=int((a["F2_Fri"] < worst_fri).sum()),
                    n_F2_tr_below_insample_worst=int((a["F2_tr"] < worst_tr).sum()),
                    n_F2_Fri_beats_QQQ=int((a["F2_Fri"] > a["QQQ"]).sum()),
                    n_F2_tr_beats_QQQ=int((a["F2_tr"] > a["QQQ"]).sum()),
                    n_F1_best_tier_Fri=int(((a["F1_Fri"] >= a["F2_Fri"]) & (a["F1_Fri"] >= a["F3_Fri"])).sum()),
                    n=len(rows))
    v = res[dte]
    print(f"{dte}: offsets -10..+10 (21 paths) | F2 Fri 12m {v['F2_Fri']['min']*100:+.1f} / {v['F2_Fri']['median']*100:+.1f} / "
          f"{v['F2_Fri']['max']*100:+.1f} (min/med/max), maxDD worst {v['F2_Fri_mdd']['min']*100:.1f} med "
          f"{v['F2_Fri_mdd']['median']*100:.1f} | tranche {v['F2_tr']['min']*100:+.1f} / {v['F2_tr']['median']*100:+.1f} / "
          f"{v['F2_tr']['max']*100:+.1f}, maxDD worst {v['F2_tr_mdd']['min']*100:.1f} | F2 Fri < in-sample worst "
          f"{worst_fri*100:.1f}%: {v['n_F2_Fri_below_insample_worst']}/21; tranche < {worst_tr*100:.1f}%: "
          f"{v['n_F2_tr_below_insample_worst']}/21 | F2 Fri>QQQ {v['n_F2_Fri_beats_QQQ']}/21, tranche>QQQ "
          f"{v['n_F2_tr_beats_QQQ']}/21 | F1 best tier {v['n_F1_best_tier_Fri']}/21", flush=True)
# ---- C1 balance check: off-leg risk shares over longer windows (real GLD era) as well as the trailing year
import math  # noqa: E402
C = g["C"]
rs = {}
for lab, a_, b_ in (("2004-11-18..freeze", "2004-11-18", C.FREEZE), ("2015..freeze", "2015-01-01", C.FREEZE),
                    ("trailing 252d to freeze", None, C.FREEZE), ("trailing 252d to 2026-11-17", None, idx[-1]),
                    ("trailing 63d to 2026-11-17", None, idx[-1])):
    seg = P.loc[:b_]
    seg = seg.loc[a_:] if a_ else seg.iloc[-(63 if "63d" in lab else 252):]
    x = np.vstack([0.85 * seg["ief"].values, 0.15 * seg["gold_phys"].values])
    cv = np.cov(x)
    tot = cv.sum()
    rs[lab] = dict(vol_085IEF=float(math.sqrt(cv[0, 0] * 252)), vol_015GLD=float(math.sqrt(cv[1, 1] * 252)),
                   corr=float(cv[0, 1] / math.sqrt(cv[0, 0] * cv[1, 1])),
                   risk_share_IEF=float((cv[0, 0] + cv[0, 1]) / tot), risk_share_GLD=float((cv[1, 1] + cv[0, 1]) / tot))
    v = rs[lab]
    print(f"C1 {lab:28s}: 0.85 IEF {v['vol_085IEF']*100:.2f}% | 0.15 GLD {v['vol_015GLD']*100:.2f}% | corr {v['corr']:+.2f} | "
          f"risk share IEF {v['risk_share_IEF']*100:.0f}% / GLD {v['risk_share_GLD']*100:.0f}%")
res["C1_risk_shares"] = rs
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("saved", OUT)
