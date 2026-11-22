"""
ROBUSTNESS REVIEW of update_2026_11/ucits.py  --  step 4: FX-timing sensitivity and a primary-line data spot check.

  X1  ucits.py re-run with the FX interpolation weight changed from 16.5/24 (0.6875) to 0.78 -- the weight that the
      same-product USD vs EUR/GBp lines imply since 2018 (verify_ucits_robustness_checks.py R7: 0.74 / 0.79 / 0.80) --
      and to 0.0 / 1.0 as bounds; reports the EUR loadings and the strategy gaps.  Output JSONs go to the scratch dir.
  X2  QQQ3.L (primary 3x line) vs its EUR / GBp lines and TQQQ on the dates where the same-product lines disagree by
      > 5% in a day (2020-03-20, 2022-09-13/16/21): which line is off?
usage: python update_2026_11/verify_ucits_robustness_fxw.py <scratch_dir>
"""
import os
import sys
import io
import json
import contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
scratch = sys.argv[1]
base_code = open(SRC, encoding="utf-8").read()
pat = 'open(os.path.join(HERE, "ucits.json"), "w")'
assert base_code.count(pat) == 1 and base_code.count("FX_W = 16.5 / 24") == 1

res = {}
G = None
for wv in ["16.5 / 24", "0.78", "0.0", "1.0"]:
    out = os.path.join(scratch, f"ucits_fxw_{wv.replace(' ', '').replace('/', 'over')}.json")
    code = base_code.replace(pat, f"open({out!r}, 'w')").replace("FX_W = 16.5 / 24", f"FX_W = {wv}")
    g = {"__name__": "__main__", "__file__": SRC}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, SRC, "exec"), g)
    d = json.load(open(out))
    f2 = d["strategy"]["F2 (2x, vt .30): LQQ.PA + CBU0.L + IGLN.L vs QLD + IEF + GLD"]
    f3 = d["strategy"]["F3 (3x, vt .20): QQQ3.L + CBU0.L + IGLN.L vs TQQQ + IEF + GLD"]
    fr = d["fx_residual_regressions"]
    res[wv] = dict(LQQ_PA_fx_coef=fr["LQQ.PA"]["interpolated_fx (used)"]["coef_fx"],
                   LQQ_PA_fx_t=fr["LQQ.PA"]["interpolated_fx (used)"]["t_fx"],
                   SGLN_fx_coef=fr["SGLN.L"]["interpolated_fx (used)"]["coef_fx"],
                   QQQ3MI_fx_coef=fr["QQQ3.MI"]["interpolated_fx (used)"]["coef_fx"],
                   placebo_QQQ3L=fr["QQQ3.L (USD-line placebo vs EURUSD)"]["interpolated_fx (used)"]["coef_fx"],
                   LQQ_PA_cagr_gap_full=d["overlap_full_history"]["LQQ.PA"]["cagr_gap"],
                   LQQ_PA_te=d["overlap_full_history"]["LQQ.PA"]["te_weekly_ann"],
                   F2_total_gap=f2["gap_decomposition_cagr"]["TOTAL measured (UCITS lag2 realistic - US lag1 3bp) = A+B+C"],
                   F2_B=f2["gap_decomposition_cagr"]["B. instruments + European close, same session (UCITS lag2 - US lag2)"],
                   F3_total_gap=f3["gap_decomposition_cagr"]["TOTAL measured (UCITS lag2 realistic - US lag1 3bp) = A+B+C"])
    print(f"FX_W={wv:9s} " + "  ".join(f"{k} {v:+.4f}" for k, v in res[wv].items()))
    G = g if wv == "16.5 / 24" else G
json.dump(res, open(os.path.join(scratch, "verify_ucits_robustness_fxw.json"), "w"), indent=1)

# ---------------------------------------------------------------- X2 primary 3x line spot check
CLEAN = G["CLEAN"]
A = G["us_level"]("tqqq_real")
for d0, d1 in [("2020-03-17", "2020-03-25"), ("2022-09-09", "2022-09-23")]:
    df = pd.DataFrame({"QQQ3.L": CLEAN["QQQ3.L"], "QQQ3.MI": CLEAN["QQQ3.MI"], "LQQ3.L": CLEAN.get("LQQ3.L"),
                       "TQQQ_lvl": A}).loc[d0:d1]
    r = df.pct_change(fill_method=None) * 100
    print(f"\n{d0}..{d1} daily % returns (USD; European lines at the European close, TQQQ at the US close)")
    print(r.round(2).to_string())
