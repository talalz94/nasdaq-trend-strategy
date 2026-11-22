"""
INTEGRITY REVIEW of update_2026_11/ucits.py  --  step 5: remaining claim checks.
  a. same-close-time test of the 'close timing, not tracking error' claim: European lines vs each other
  b. 1x control (CNDX.L vs QQQ) with the stale-price 2010 quarter excluded
  c. chop-flag flips per year inside the UCITS windows (vs the 15.8/yr quoted from 1988-freeze)
  d. current state (NDX vs EMA-100 on 2026-11-16), cost grid, control bootstrap as saved in ucits.json
Read-only: writes nothing except stdout.
"""
import os, io, json, contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index('US2 = dict(lev="qld"')
g = {"__name__": "ucits_partial", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code[:cut], SRC, "exec"), g)
C, P, P2, PX, CLEAN, END, FREEZE, WEND = g["C"], g["P"], g["P2"], g["PX"], g["CLEAN"], g["END"], g["FREEZE"], g["WEND"]
run = g["run"]
J = json.load(open(os.path.join(HERE, "ucits.json")))


def wk(s):
    return s.resample("W-FRI").last()


print("a. European lines against each other (same ~16:30-17:30 CET close) vs against the US analogue")
for t1, t2, lev_ratio in [("LQQ.PA", "QQQ3.L", 1.5), ("LQQ.PA", "CNDX.L", 2.0), ("QQQ3.L", "CNDX.L", 3.0), ("CBU0.L", "IGLN.L", None)]:
    W = pd.concat([wk(CLEAN[t1]), wk(CLEAN[t2])], axis=1, keys=["x", "y"]).dropna()
    W = W[(W.index <= WEND) & (W.index >= "2015-01-01")]
    r = W.pct_change().dropna()
    r4 = W.iloc[::4].pct_change().dropna()
    print(f"  {t1} vs {t2} (2015+): weekly corr {r.x.corr(r.y):.4f}  4-week corr {r4.x.corr(r4.y):.4f}  n={len(r)}")
for tk, col in [("LQQ.PA", "qld_real"), ("QQQ3.L", "tqqq_real"), ("CNDX.L", "qqq_real")]:
    lv = g["us_level"](col)
    W = pd.concat([wk(CLEAN[tk]), wk(lv)], axis=1, keys=["x", "y"]).dropna()
    W = W[(W.index <= WEND) & (W.index >= "2015-01-01")]
    r = W.pct_change().dropna()
    print(f"  {tk} vs its US analogue (2015+): weekly corr {r.x.corr(r.y):.4f}")

print("\nb. 1x control CNDX.L vs QQQ (F2 rules, lag2, 3bp), excluding the stale 2010 quarter")
x = PX[PX.ticker == "CNDX.L"].set_index("date").sort_index()
st = (x.close.diff() == 0)
print("  CNDX.L unchanged-close share: " + " ".join(f"{y}:{st[st.index.year == y].mean()*100:.0f}%" for y in range(2010, 2016)))
with contextlib.redirect_stdout(io.StringIO()):
    r_q, _ = run("x", "qqq", 0.30, "ief", "gold_phys", lag=2)
    r_c, _ = run("x", "u_cndx", 0.30, "ief", "gold_phys", lag=2)
for a in ["2010-10-01", "2011-01-03", "2014-01-02"]:
    sq, sc = C.stats(r_q, a, END, start=None), C.stats(r_c, a, END, start=None)
    bb = C.block_bootstrap_cagr_diff(r_c[:END], r_q[:END], start=pd.Timestamp(a))
    print(f"  from {a}: CNDX {sc['cagr']*100:.2f}% vs QQQ {sq['cagr']*100:.2f}%  gap {(sc['cagr']-sq['cagr'])*100:+.2f}%/yr  "
          f"[{bb['lo']*100:+.2f}, {bb['hi']*100:+.2f}]")
m = (r_c.index >= "2010-10-01") & (r_c.index <= "2010-12-31")
print(f"  Q4-2010 alone: CNDX-version {np.prod(1+r_c[m])-1:+.2%}  QQQ-version {np.prod(1+r_q[m])-1:+.2%}")
print(f"  saved control: {J['control_1x_CNDX_vs_QQQ']['cagr_gap']*100:+.2f}%  bootstrap {J['control_1x_CNDX_vs_QQQ']['bootstrap']}")

print("\nc. chop-flag (ER20 >= 0.25) changes per year at weekly marks")
mk = P2["mark"].fillna(False)
fl = (P2["er20"] >= 0.25).fillna(False)
for a, b in [("1988-01-01", str(FREEZE.date())), ("2011-04-28", str(END.date())), ("2013-01-03", str(END.date()))]:
    s = fl[mk & (P2.index >= a) & (P2.index <= b)]
    yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    on = P2["above"][mk & (P2.index >= a) & (P2.index <= b)]
    ch = (s != s.shift()).iloc[1:]
    print(f"  {a}..{b}: {ch.sum()/yrs:.1f} flips/yr (all weeks); {(ch & on.iloc[1:]).sum()/yrs:.1f}/yr while risk-on")

print("\nd. saved outputs")
last = P.loc[:END].iloc[-1]
print(f"  panel {P.loc[:END].index[-1].date()}: ndx {last['ndx']:.0f} ema100 {last['ema100']:.0f} above {bool(last['above'])}")
print(f"  current_state saved: {J['current_state']}")
f2 = J["strategy"]["F2 (2x, vt .30): LQQ.PA + CBU0.L + IGLN.L vs QLD + IEF + GLD"]
for k in ["lev 10bp / bond+gold 5bp one-way", "lev 15bp / bond+gold 8bp one-way", "lev 20bp / bond+gold 8bp one-way",
          "lev 30bp / bond+gold 12bp one-way"]:
    print(f"  F2 cost grid {k}: {f2['analytic_extra_cost_drag_grid_per_year'][k]*100:.2f}%/yr")
print(f"  F2 annual turnover UCITS lag2: {f2['annual_turnover_UCITS_lag2']}")
print(f"  hybrid: {json.dumps(J['hybrid_US_leveraged_plus_UCITS_bonds_gold']['F2']['bootstrap'])}")
print(f"  meta variants_tested: {J['meta']['variants_tested']}")
