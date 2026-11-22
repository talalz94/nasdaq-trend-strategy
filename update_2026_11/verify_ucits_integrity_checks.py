"""
INTEGRITY REVIEW of update_2026_11/ucits.py  --  step 2: targeted checks.

Loads ucits.py's own definitions (exec of the source up to the strategy section, stdout muted, nothing written)
and then probes:
  1. raw-data defects in the lines used by the backtests (stale closes, zero volume, 2025-10-24 USD-NAV prints,
     2015-01-02 LQQ.PA split snap)
  2. execution-calendar integrity: do 'execution' sessions of the UCITS lag-2 backtests fall on rows whose price is
     stale (unchanged close / zero volume)?  A stale close on the execution day = trading at a pre-signal price
     (the same mechanism as the lag-1 look-ahead artifact).
  3. FX-interpolation sensitivity of the strategy result (FX_W = 0, 16.5/24, 1)
  4. where component B (instrument + European close) comes from (2020 concentration)
Read-only: writes nothing except stdout.
usage: python update_2026_11/verify_ucits_integrity_checks.py
"""
import os, sys, io, contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index('US2 = dict(lev="qld"')
g = {"__name__": "ucits_partial", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code[:cut], SRC, "exec"), g)
C, P, P2, PX, CLEAN, END, FREEZE = g["C"], g["P"], g["P2"], g["PX"], g["CLEAN"], g["END"], g["FREEZE"]
run, on_dates, us_level, to_usd, clean = g["run"], g["on_dates"], g["us_level"], g["to_usd"], g["clean"]
COST_REAL, held_weights_eu, eu_open_mask = g["COST_REAL"], g["held_weights_eu"], g["eu_open_mask"]
EXCH_CAL = g["EXCH_CAL"]


def raw(tk):
    x = PX[PX.ticker == tk].set_index("date").sort_index()
    return x[x.index <= END]


print("=" * 100)
print("1a. STALE / ZERO-VOLUME ROWS BY YEAR (raw Yahoo rows, <= END)")
for tk in ["LQQ.PA", "QQQ3.L", "CBU0.L", "IGLN.L", "IBTM.L", "PHAU.L", "CNDX.L"]:
    x = raw(tk).dropna(subset=["close"])
    st = (x.close.diff() == 0)
    zv = (x.volume == 0)
    yrs = sorted(set(x.index.year))
    line = " ".join(f"{y % 100:02d}:{st[x.index.year == y].mean()*100:3.0f}/{zv[x.index.year == y].mean()*100:3.0f}" for y in yrs)
    print(f"  {tk:7s} stale%/zeroVol% {line}")

print("\n1b. 2025-10-24 (claimed USD-NAV print into EUR/GBp lines): raw closes and USD value / US analogue")
for tk in ["LQQ.PA", "LQQ.MI", "QQQ3.L", "LQQ3.L", "QQQ3.MI", "IGLN.L", "SGLN.L", "CNDX.L", "EQQQ.L", "CBU0.L", "IBTM.L"]:
    x = raw(tk)
    sl = x.loc["2025-10-20":"2025-10-31", "close"]
    cu = CLEAN[tk].loc["2025-10-20":"2025-10-31"]
    a = us_level(g["CAT"][tk]["us"])
    rel = cu / on_dates(a, cu.index)
    rel = rel / rel.iloc[0]
    print(f"  {tk:8s} raw " + " ".join(f"{d.strftime('%m-%d')}:{v:.4g}" for d, v in sl.items()))
    print(f"  {'':8s} clean-USD/analogue (norm) " + " ".join(f"{d.strftime('%m-%d')}:{v:.3f}" for d, v in rel.items()))

print("\n1c. LQQ.PA around the 2015-01-02 snap (raw EUR close, cleaned USD, QLD level)")
x = raw("LQQ.PA").loc["2014-12-22":"2015-01-09", "close"]
q = us_level("qld_real").loc["2014-12-22":"2015-01-09"]
cu = CLEAN["LQQ.PA"].loc["2014-12-22":"2015-01-09"]
for d in x.index:
    print(f"  {d.date()} raw {x[d]:10.3f}  clean-USD {cu.get(d, np.nan):8.4f}  QLD {q.get(d, np.nan):8.4f}")
c0, c1 = cu.loc[:"2014-12-30"].iloc[-1], cu.loc["2015-01-02"]
q0, q1 = q.loc[:"2014-12-30"].iloc[-1], q.loc["2015-01-02"]
print(f"  2014-12-30 -> 2015-01-02: LQQ clean {c1/c0-1:+.2%}  vs QLD {q1/q0-1:+.2%}  (US closes 12-30 -> 01-02)")

print("\n" + "=" * 100)
print("2. EXECUTION SESSIONS ON STALE ROWS (UCITS lag2 backtests, as run by ucits.py)")


def stale_flags(tk):
    """per panel date: True if the Yahoo row of tk on that date is missing, has zero volume or an unchanged close"""
    x = raw(tk).dropna(subset=["close"])
    st = (x.close.diff() == 0) | (x.volume == 0)
    present = P2.index.isin(x.index)
    s = pd.Series(True, index=P2.index)                      # missing row -> value is carried from an earlier session
    s[present] = st.reindex(P2.index[present]).values
    return s


for nm, lev, off, gold, vt in [("F2", "u_lqq", "u_cbu0", "u_igln", 0.30), ("F3", "u_qqq3", "u_cbu0", "u_igln", 0.20),
                               ("F2 longer", "u_lqq", "u_ibtm", "u_phau", 0.30)]:
    with contextlib.redirect_stdout(io.StringIO()):
        r, ww = run("chk", lev, vt, off, gold, lag=2, costs=COST_REAL)
    a0 = max(P2[c].first_valid_index() for c in (lev, off, gold))
    chg = ww.diff().abs()
    chg = chg[(chg.index > a0) & (chg.index <= END)]
    tkmap = g["COL2TK"]
    print(f"  {nm}:")
    for c in (lev, off, gold):
        tk = tkmap[c]
        sf = stale_flags(tk)
        ex_next = chg.index[chg[c] > 1e-12]                       # first day of new holdings
        ex_day = P2.index[P2.index.get_indexer(ex_next) - 1]      # execution close = previous panel session
        bad = sf.reindex(ex_day).values
        dw = chg[c].reindex(ex_next).values
        print(f"    {tk:7s} executions {len(ex_day):4d}  on stale/zero-vol/missing row {bad.sum():4d} "
              f"({bad.mean()*100:4.1f}%)  |dw| on stale rows {dw[bad].sum():6.2f} of {dw.sum():7.2f}")
        if bad.sum():
            yy = pd.Series(bad, index=ex_day).groupby(ex_day.year).sum()
            print(f"            by year: " + " ".join(f"{y}:{int(v)}" for y, v in yy.items() if v))

print("\n  Holiday rows in the union exchange calendars (date in EXCH_CAL but zero volume on EVERY line of that exchange):")
for sfx in (".L", ".PA"):
    tks = sorted(set(PX.ticker[PX.ticker.str.endswith(sfx)]))
    vol = PX[PX.ticker.isin(tks)].pivot_table(index="date", columns="ticker", values="volume", aggfunc="last")
    cal = EXCH_CAL[sfx]
    v = vol.reindex(cal)
    allzero = (v.fillna(0) <= 0).all(axis=1)
    allzero = allzero[(allzero.index <= END)]
    days = allzero[allzero].index
    md = [d for d in days if (d.month, d.day) in ((12, 25), (12, 26), (1, 1), (5, 1))]
    print(f"    {sfx}: {len(days)} calendar days with zero volume on all lines; of which Dec25/Dec26/Jan1/May1: "
          f"{[str(d.date()) for d in md][:20]}")
    if len(days):
        print(f"       sample: {[str(d.date()) for d in days[:25]]}")

print("\n" + "=" * 100)
print("3. FX-INTERPOLATION SENSITIVITY (F2 primary, UCITS lag2 realistic cost, LQQ.PA only EUR line)")
a = pd.Timestamp("2011-04-28")
base_r, _ = None, None
for wfx in [0.0, 16.5 / 24, 1.0]:
    g["FX_W"] = wfx
    s = PX[PX.ticker == "LQQ.PA"].set_index("date").sort_index()["adjclose"]
    s = s[(s > 0) & (s.index <= END)].dropna()
    with contextlib.redirect_stdout(io.StringIO()):
        cl, lg = clean(to_usd("LQQ.PA", s), us_level("qld_real"))
    P2["u_lqq"] = on_dates(cl, P2.index).pct_change(fill_method=None)
    with contextlib.redirect_stdout(io.StringIO()):
        r, _ = run("chk", "u_lqq", 0.30, "u_cbu0", "u_igln", lag=2, costs=COST_REAL)
    st = C.stats(r, a, END, start=None)
    print(f"  FX_W={wfx:.4f}  {C.fmt(st)}   (cleaning log entries {len(lg)})")
g["FX_W"] = 16.5 / 24
s = PX[PX.ticker == "LQQ.PA"].set_index("date").sort_index()["adjclose"]
s = s[(s > 0) & (s.index <= END)].dropna()
with contextlib.redirect_stdout(io.StringIO()):
    cl, lg = clean(to_usd("LQQ.PA", s), us_level("qld_real"))
P2["u_lqq"] = on_dates(cl, P2.index).pct_change(fill_method=None)

print("\n" + "=" * 100)
print("4. COMPONENT B BY YEAR (UCITS lag2 3bp - US lag2 3bp), F2 primary; and top daily contributions in 2020")
with contextlib.redirect_stdout(io.StringIO()):
    rU, wU = run("chk", "u_lqq", 0.30, "u_cbu0", "u_igln", lag=2)
    rS, wS = run("chk", "qld", 0.30, "ief", "gold_phys", lag=2)
    rL, wL = run("chk", "u_lqq", 0.30, "ief", "gold_phys", lag=2)
yrs = range(2011, 2027)
for y in yrs:
    m = (rU.index.year == y) & (rU.index >= a) & (rU.index <= END)
    print(f"  {y}: UCITS {np.prod(1+rU[m])-1:+7.2%}  US {np.prod(1+rS[m])-1:+7.2%}  diff {np.prod(1+rU[m])-np.prod(1+rS[m]):+6.2%}"
          f"   lev-swap-only diff {np.prod(1+rL[m])-np.prod(1+rS[m]):+6.2%}")
m = (rU.index.year == 2020)
d = (rU - rS)[m]
print("  2020 daily (UCITS - US) largest |diff|; with held lev weights UCITS/US and LQQ/QLD returns:")
for dt in d.abs().sort_values(ascending=False).index[:12]:
    print(f"    {dt.date()} diff {d[dt]:+.2%}  w_lev U {wU.loc[dt,'u_lqq']:.2f} / US {wS.loc[dt,'qld']:.2f}  "
          f"r LQQ {P2.loc[dt,'u_lqq']:+.2%}  r QLD {P2.loc[dt,'qld']:+.2%}")
cum = (1 + d).cumprod()
print(f"  2020 cumulative (1+diff) product: {cum.iloc[-1]-1:+.2%}")
