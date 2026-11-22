"""
INTEGRITY REVIEW of update_2026_11/ucits.py  --  step 6.
  1. F2 'longer history' (2009-01-21..2026-11-16) with the known-bad IBTM.L Dist/GBP line replaced by CBU0.L from
     CBU0's first clean return on (IBTM.L only before) -> how much of the -2.09%/yr is the IBTM data lag?
  2. stale CBU0 bond leg isolated: bond leg executed only on a fresh CBU0 close, leveraged/gold legs unchanged
  3. robustness of the 'one-session delay' (component A) to the window start
Read-only: writes nothing except stdout.
"""
import os, io, contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index('US2 = dict(lev="qld"')
g = {"__name__": "ucits_partial", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code[:cut], SRC, "exec"), g)
C, P, P2, PX, CLEAN, END = g["C"], g["P"], g["P2"], g["PX"], g["CLEAN"], g["END"]
run, COST_REAL, held_weights_eu, eu_open_mask = g["run"], g["COST_REAL"], g["held_weights_eu"], g["eu_open_mask"]

print("1. F2 longer history with IBTM.L -> CBU0.L splice for the bond sleeve")
cb_first = P2["u_cbu0"].first_valid_index()
P2["u_bspl"] = np.where(P2.index >= cb_first, P2["u_cbu0"], P2["u_ibtm"])
P2.loc[P2.index < P2["u_ibtm"].first_valid_index(), "u_bspl"] = np.nan
g["COL2TK"]["u_bspl"] = "CBU0.L"          # London calendar (same exchange as IBTM.L)
COST = dict(COST_REAL, u_bspl=8e-4)
a = pd.Timestamp("2009-01-21")
with contextlib.redirect_stdout(io.StringIO()):
    r_us, _ = run("x", "qld", 0.30, "ief", "gold_phys", lag=1)
    r_old, _ = run("x", "u_lqq", 0.30, "u_ibtm", "u_phau", lag=2, costs=COST_REAL)
    r_new, _ = run("x", "u_lqq", 0.30, "u_bspl", "u_phau", lag=2, costs=COST)
su, so, sn = (C.stats(x, a, END, start=None) for x in (r_us, r_old, r_new))
print(f"  US lag1 3bp               {C.fmt(su)}")
print(f"  UCITS IBTM (as ucits.py)  {C.fmt(so)}  gap {(so['cagr']-su['cagr'])*100:+.2f}")
print(f"  UCITS IBTM->CBU0 splice   {C.fmt(sn)}  gap {(sn['cagr']-su['cagr'])*100:+.2f}")
bb = C.block_bootstrap_cagr_diff(r_new[:END], r_us[:END], start=a)
print(f"  splice gap bootstrap [{bb['lo']*100:+.2f}, {bb['hi']*100:+.2f}]")

print("\n2. Stale CBU0 bond leg isolated (bond leg waits for a fresh CBU0 close; lev + gold legs as in ucits.py)")
x = PX[PX.ticker == "CBU0.L"].set_index("date").sort_index()
x = x[x.index <= END].dropna(subset=["close"])
fresh = pd.Series(P2.index.isin(x.index[(x.close.diff().fillna(1) != 0).values]), index=P2.index)
for nm, lev, vt, a in [("F2", "u_lqq", 0.30, "2011-04-28"), ("F3", "u_qqq3", 0.20, "2013-01-03")]:
    a = pd.Timestamp(a)
    w = C.strategy_weights(P2, lev=lev, voltgt=vt, off="u_cbu0", gold="u_igln")
    m0 = eu_open_mask([g["COL2TK"][lev], "CBU0.L", "IGLN.L"])
    ww0 = held_weights_eu(P2, w, 2, m0)
    ww1 = ww0.copy()
    ww1["u_cbu0"] = held_weights_eu(P2, w, 2, m0 & fresh)["u_cbu0"]
    out = []
    for ww in (ww0, ww1):
        gross = sum(ww[c] * P2[c].fillna(0) for c in ww.columns)
        tc = sum(ww[c].diff().abs().fillna(0) * COST_REAL.get(c, C.COST) for c in ww.columns)
        out.append(C.stats(gross - tc, a, END, start=None))
    gap_days = int(((ww1["u_cbu0"] - ww0["u_cbu0"]).abs() > 1e-12)[(P2.index >= a) & (P2.index <= END)].sum())
    print(f"  {nm}: as ucits.py {C.fmt(out[0])} | bond leg on fresh closes only {C.fmt(out[1])}  "
          f"dCAGR {(out[1]['cagr']-out[0]['cagr'])*100:+.2f}  (days with differing bond weight: {gap_days})")

print("\n3. Component A (US lag2 - US lag1, F2 rules) by window start")
with contextlib.redirect_stdout(io.StringIO()):
    r1, _ = run("x", "qld", 0.30, "ief", "gold_phys", lag=1)
    r2, _ = run("x", "qld", 0.30, "ief", "gold_phys", lag=2)
for a in ["1988-01-01", "2009-01-21", "2010-01-04", "2011-04-28", "2013-01-03", "2015-01-02"]:
    s1, s2 = C.stats(r1, a, END, start=None), C.stats(r2, a, END, start=None)
    print(f"  from {a}: A = {(s2['cagr']-s1['cagr'])*100:+.2f}%/yr")
