"""
r2_verify_chop_vol_matched_robustness_era.py -- ROBUSTNESS review of r2_chop_vol_matched, claim C2
("the chop rule's matched edge comes from one era and ONE market: strongly positive only on NDX in 1988-99").

For each of the 7 registered cross-market indices (2x, vt .30, tranche, lag 1, 3 bps, IEF + 15% gold_phys,
exactly the specialist's construction) and for NDX F2/F3, the chop-rule Calmar difference B - A is computed in
the sub-periods 1990-99, 2000-06, 2000-12 and 2013-2026-07-24 with A's vol target matched to B's exposure
(i) over the market's full registered window (the specialist's convention) and (ii) re-matched INSIDE each
sub-period (removes any within-period exposure mismatch). Information only; no decision is affected.
usage: python update_2026_11/r2_verify_chop_vol_matched_robustness_era.py <scratch_out.json>
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_verify_chop_vol_matched_robustness_lib as L  # noqa: E402

OUTP = os.path.abspath(sys.argv[1])
FREEZE = L.FREEZE
SUBS = {"1990-99": ("1990-01-01", "1999-12-31"), "2000-06": ("2000-01-01", "2006-12-31"),
        "2000-12": ("2000-01-01", "2012-12-31"), "2013-frz": ("2013-01-01", FREEZE)}
MK = dict(L.cross_markets())
MK["NDX-F2"] = L.NDXM["F2"]
MK["NDX-F3"] = L.NDXM["F3"]
VT = {"NDX-F3": 0.20}
OUT = {}
for nm, M in MK.items():
    vt = VT.get(nm, 0.30)
    a0 = "1988-01-01" if nm.startswith("NDX") else "1990-01-01"
    vtA, cV = L.match_expo(M, vt, "tranche", a0, FREEZE)
    S = L.series(M, vt, vtA, cV)
    row = {}
    for k, (a, b) in SUBS.items():
        d_full = L.calmar(S["B"], a, b) - L.calmar(S["A"], a, b)
        vtA_k, _ = L.match_expo(M, vt, "tranche", a, b)
        rA_k = L.ret(M, L.wlev(M, vtA_k, chop=False))
        d_re = L.calmar(S["B"], a, b) - L.calmar(rA_k, a, b)
        row[k] = dict(full_match=d_full, rematched=d_re, vtA_rematched=vtA_k)
    OUT[nm] = row
    print(f"{nm:7s} " + " | ".join(f"{k} {v['full_match']:+.3f} (re-match {v['rematched']:+.3f})" for k, v in row.items()), flush=True)
others = [m for m in MK if not m.startswith("NDX")]
summ = {}
for k in SUBS:
    for conv in ["full_match", "rematched"]:
        vals = [OUT[m][k][conv] for m in others]
        summ[f"{k}|{conv}"] = dict(median=float(np.median(vals)), n_pos=int(sum(v > 0 for v in vals)),
                                   n_ge_0p15=int(sum(v >= 0.15 for v in vals)))
        print(f"  7-market {k:8s} {conv:10s}: median {np.median(vals):+.3f}, positive {sum(v > 0 for v in vals)}/7, >= +0.15: {sum(v >= 0.15 for v in vals)}/7")
OUT["summary_7_markets"] = summ
json.dump(OUT, open(OUTP, "w"), indent=1, default=float)
print("saved", OUTP)
