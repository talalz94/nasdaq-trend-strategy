"""
INTEGRITY REVIEW of update_2026_11/ucits.py  --  step 4: Yahoo FX bar timing is NOT constant, and the stale CBU0 era.
  1. year-by-year corr of the Yahoo EURUSD / GBPUSD bar change with DXY change of D-1 and D, and a joint
     regression on DXY D-1, D, D+1 (effective bar time)
  2. era-correct USD conversion (pre-switch: same-date bar, which sits close to the European close; post-switch:
     the specialist's 16.5/24 interpolation) -> F2 / F3 / F2-longer UCITS results and the LQQ.PA EURUSD loading
  3. bond-sleeve swap effect (B2) by year, F2 2011-2026 gap from 2015 (clean CBU0 data) onwards
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
C, P, P2, PX, CLEAN, END, FREEZE, WEND = g["C"], g["P"], g["P2"], g["PX"], g["CLEAN"], g["END"], g["FREEZE"], g["WEND"]
on_dates, us_level, clean, run, COST_REAL = g["on_dates"], g["us_level"], g["clean"], g["run"], g["COST_REAL"]
FX = g["FX"]

dx = np.log(P["dxy"]).diff()
print("1. Yahoo FX bar timing by year: corr(bar change D, DXY change D-1) | corr(.., DXY D) ; joint betas on DXY D-1/D/D+1")
for nm in ["EURUSD=X", "GBPUSD=X"]:
    lf = np.log(FX[nm]).diff()
    rows = []
    for y in range(2004, 2027):
        z = pd.concat([lf, dx.shift(1), dx, dx.shift(-1)], axis=1, keys=["fx", "dm1", "d0", "dp1"]).dropna()
        z = z[z.index.year == y]
        if len(z) < 100:
            continue
        X = np.column_stack([np.ones(len(z)), z.dm1, z.d0, z.dp1])
        b = np.linalg.lstsq(X, z.fx.values, rcond=None)[0]
        rows.append(f"{y}:{z.fx.corr(z.dm1):+.2f}|{z.fx.corr(z.d0):+.2f} (b {b[1]:+.2f}/{b[2]:+.2f}/{b[3]:+.2f})")
    print(f"  {nm}:")
    for i in range(0, len(rows), 4):
        print("    " + "   ".join(rows[i:i + 4]))

# find the switch date: rolling 120-session corr with DXY D vs D-1
lf = np.log(FX["EURUSD=X"]).diff()
z = pd.concat([lf, dx.shift(1), dx], axis=1, keys=["fx", "dm1", "d0"]).dropna()
rc = (z.fx.rolling(60).corr(z.d0) - z.fx.rolling(60).corr(z.dm1))
flip = rc[(rc.index >= "2011-01-01") & (rc.index <= "2014-12-31")]
print("  rolling-60 [corr(D) - corr(D-1)] for EURUSD, monthly samples 2011-2014 (negative = bar aligned with DXY D):")
print("    " + " ".join(f"{d.strftime('%y-%m')}:{v:+.2f}" for d, v in flip.resample("ME").last().items()))

print("\n2. ERA-CORRECT FX CONVERSION")
SWITCH = None
s = rc.dropna()
# first date after which the rolling difference stays positive (bar aligned with DXY D-1) for 120 sessions
pos = (s > 0).astype(int)
for i in range(len(s) - 120):
    if s.index[i] >= pd.Timestamp("2010-01-01") and pos.iloc[i:i + 120].mean() > 0.95:
        SWITCH = s.index[i]
        break
print(f"  detected switch (EURUSD bar moves from end-of-day-D to start-of-day-D timing): {SWITCH.date()}")
FX_W = g["FX_W"]


def fx_era(fx, dates):
    lfx = np.log(fx)
    cur = on_dates(lfx, dates)
    pos_ = np.searchsorted(lfx.index.values, dates.values, side="right")
    nxt = pd.Series(np.where(pos_ < len(lfx), lfx.values[np.minimum(pos_, len(lfx) - 1)], np.nan), index=dates)
    post = (1 - FX_W) * cur + FX_W * nxt.fillna(cur)
    return np.exp(np.where(dates < SWITCH, cur, post))


def usd_era(tk, s):
    fxn, div = g["fx_for"](g["INFO"][tk]["currency"])
    if fxn is None:
        return s / div
    return s / div * pd.Series(fx_era(FX[fxn], s.index), index=s.index)


ALT = {}
for tk in ["LQQ.PA", "IBTM.L"]:
    s = PX[PX.ticker == tk].set_index("date").sort_index()["adjclose"]
    s = s[(s > 0) & (s.index <= END)].dropna()
    with contextlib.redirect_stdout(io.StringIO()):
        ALT[tk], lg = clean(usd_era(tk, s), us_level(g["CAT"][tk]["us"]))
    print(f"  {tk}: era-correct cleaned series, log entries {len(lg)}")
    for lbl, ser in [("specialist", CLEAN[tk]), ("era-correct", ALT[tk])]:
        o = g["overlap"](ser, g["CAT"][tk]["us"])
        o5 = g["overlap"](ser, g["CAT"][tk]["us"], start="2009-01-01", end=pd.Timestamp("2012-12-31"))
        print(f"    {lbl:12s} full: corr_w {o['corr_weekly']:.3f} corr_4w {o['corr_4week']:.3f} TE {o['te_weekly_ann']*100:5.2f}% "
              f"CAGR gap {o['cagr_gap']*100:+.2f}%   | 2009-2012: corr_w {o5['corr_weekly']:.3f} TE {o5['te_weekly_ann']*100:5.2f}%")

# LQQ.PA EURUSD loading with era-correct FX (full / pre-switch / post-switch)
fxr = g["fx_reg"]
for lbl, ser in [("specialist", CLEAN["LQQ.PA"]), ("era-correct", ALT["LQQ.PA"])]:
    o = fxr(ser, "qld_real", "EURUSD=X")
    o_pre = fxr(ser[ser.index < SWITCH], "qld_real", "EURUSD=X")
    o_post = fxr(ser[ser.index >= SWITCH], "qld_real", "EURUSD=X")
    print(f"  LQQ.PA EURUSD loading ({lbl}): full {o['coef_fx']:+.3f} (t {o['t_fx']:+.1f}) | pre-switch {o_pre['coef_fx']:+.3f} "
          f"(t {o_pre['t_fx']:+.1f}, n={o_pre['weeks']}) | post-switch {o_post['coef_fx']:+.3f} (t {o_post['t_fx']:+.1f})")
o_pl = fxr(CLEAN["QQQ3.L"][CLEAN["QQQ3.L"].index >= SWITCH], "tqqq_real", "EURUSD=X")
print(f"  QQQ3.L USD-line placebo post-switch: {o_pl['coef_fx']:+.3f} (t {o_pl['t_fx']:+.1f})")

for nm, lev, off, gold, vt, a in [("F2 primary", "u_lqq", "u_cbu0", "u_igln", 0.30, "2011-04-28"),
                                  ("F2 longer", "u_lqq", "u_ibtm", "u_phau", 0.30, "2009-01-21")]:
    a = pd.Timestamp(a)
    with contextlib.redirect_stdout(io.StringIO()):
        r0, _ = run("x", lev, vt, off, gold, lag=2, costs=COST_REAL)
        rus, _ = run("x", "qld", vt, "ief", "gold_phys", lag=1)
    keep = {c: P2[c].copy() for c in ("u_lqq", "u_ibtm")}
    P2["u_lqq"] = on_dates(ALT["LQQ.PA"], P2.index).pct_change(fill_method=None)
    P2["u_ibtm"] = on_dates(ALT["IBTM.L"], P2.index).pct_change(fill_method=None)
    with contextlib.redirect_stdout(io.StringIO()):
        r1, _ = run("x", lev, vt, off, gold, lag=2, costs=COST_REAL)
    for c, v in keep.items():
        P2[c] = v
    s0, s1, su = (C.stats(x, a, END, start=None) for x in (r0, r1, rus))
    print(f"  {nm:11s} UCITS lag2 realistic: specialist FX {C.fmt(s0)} | era-correct FX {C.fmt(s1)}  "
          f"(gap vs US lag1 {(s0['cagr']-su['cagr'])*100:+.2f} -> {(s1['cagr']-su['cagr'])*100:+.2f})")

print("\n3. BOND-SLEEVE SWAP (B2) BY YEAR, F2 primary (UCITS CBU0 bond only, lag2) - (US lag2); CBU0 stale share")
with contextlib.redirect_stdout(io.StringIO()):
    rb, _ = run("x", "qld", 0.30, "u_cbu0", "gold_phys", lag=2)
    rs, _ = run("x", "qld", 0.30, "ief", "gold_phys", lag=2)
    ruc, _ = run("x", "u_lqq", 0.30, "u_cbu0", "u_igln", lag=2, costs=COST_REAL)
    rus, _ = run("x", "qld", 0.30, "ief", "gold_phys", lag=1)
x = PX[PX.ticker == "CBU0.L"].set_index("date").sort_index()
stale = (x.close.diff() == 0)
a = pd.Timestamp("2011-04-28")
line = []
for y in range(2011, 2027):
    m = (rb.index.year == y) & (rb.index >= a) & (rb.index <= END)
    line.append(f"{y}:{(np.prod(1+rb[m])-np.prod(1+rs[m]))*100:+.2f}pp(stale {stale[stale.index.year == y].mean()*100:.0f}%)")
for i in range(0, len(line), 4):
    print("  " + "  ".join(line[i:i + 4]))
for a_, b_ in [("2011-04-28", "2014-12-31"), ("2015-01-01", str(END.date())), ("2011-04-28", str(END.date()))]:
    sb, ss = C.stats(rb, a_, b_, start=None), C.stats(rs, a_, b_, start=None)
    print(f"  B2 {a_}..{b_}: {(sb['cagr']-ss['cagr'])*100:+.2f}%/yr")
for a_, b_ in [("2015-01-02", str(END.date())), ("2011-04-28", "2014-12-31")]:
    su_, sc_ = C.stats(rus, a_, b_, start=None), C.stats(ruc, a_, b_, start=None)
    bb = C.block_bootstrap_cagr_diff(ruc[:b_], rus[:b_], start=pd.Timestamp(a_))
    print(f"  total F2 gap (UCITS lag2 realistic - US lag1) {a_}..{b_}: {(sc_['cagr']-su_['cagr'])*100:+.2f}%/yr "
          f"[{bb['lo']*100:+.2f}, {bb['hi']*100:+.2f}]")
