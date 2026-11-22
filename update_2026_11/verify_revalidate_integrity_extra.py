"""
INTEGRITY REVIEW of update_2026_11/revalidate.py  --  step 3: follow-ups.

  E1  excess-return Sharpe for every candidate/baseline (the specialist's Sharpe ignores rf)
  E2  is the synthetic-treasury duration era-dependent?  Re-fit duration on sub-periods of the
      real-ETF era; then rebuild pre-2002-07-30 IEF/TLT with a yield-dependent (par-bond) duration
      and re-score F3-IEF, F2-IEF, F3-TLT (H1 and IS)
  E3  C5/C4 consistency: gold_w = 0.25 vs 0.15 holding the walk-forward picks' other knobs fixed
      (already in checks K7) -> here: same for the DEFAULT configs in each half
usage: python update_2026_11/verify_revalidate_integrity_extra.py      (read-only)
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel()
IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
rf = P["rf_d"].fillna(0)
H1 = ("1988-01-01", "2006-12-31"); H2 = ("2007-01-01", FREEZE); IS = (None, FREEZE)


def st(r, a=None, b=None):
    s = r[r.index >= START]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    e = np.cumprod(1 + s.values)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    x = (s - rf.reindex(s.index))
    return dict(cagr=cagr, maxdd=dd, calmar=cagr / abs(dd), sharpe=s.mean() / s.std() * np.sqrt(252),
                xsharpe=x.mean() / x.std() * np.sqrt(252), vol=s.std() * np.sqrt(252))


print("=" * 90 + "\nE1  raw (rf ignored) vs excess Sharpe, IS 1988-01-01..2026-07-24\n" + "=" * 90)
CFG = {"F3-IEF": dict(lev="tqqq", voltgt=0.20), "F3-TLT": dict(lev="tqqq", voltgt=0.20, off="tlt"),
       "F2-IEF": dict(lev="qld", voltgt=0.30), "F2-IEF vt.25": dict(lev="qld", voltgt=0.25),
       "F1-IEF": dict(lev="qqq", voltgt=0.20),
       "A": dict(lev="tqqq", voltgt=0.25, chop_pos=1.0, seasonal=False, gold_w=0.0, off="bills")}
R = {k: C.backtest(P, C.strategy_weights(P, **v)) for k, v in CFG.items()}
R["QQQ B&H"] = P["qqq"].fillna(0)
R["60/40"] = 0.6 * P["spx"].fillna(0) + 0.4 * P["ief"].fillna(0)
for k, r in R.items():
    s = st(r, *IS)
    print(f"  {k:14s} vol {s['vol']*100:5.1f}%  raw Sharpe {s['sharpe']:.3f}  excess Sharpe {s['xsharpe']:.3f}"
          f"  (rf inflation {s['sharpe']-s['xsharpe']:+.3f})")

print("\n" + "=" * 90 + "\nE2  synthetic treasury duration: era dependence\n" + "=" * 90)


def fit(real, y, a=None, b=None):
    yd = y / 100.0
    dy = yd.diff(); carry = yd.shift(1) / 252
    df = pd.concat([real, carry, dy, yd.shift(1)], axis=1).dropna(); df.columns = ["r", "carry", "dy", "y"]
    if a is not None:
        df = df[df.index >= pd.Timestamp(a)]
    if b is not None:
        df = df[df.index <= pd.Timestamp(b)]
    Y = (df.r - df.carry).values
    X = np.column_stack([np.ones(len(df)), -df.dy.values, 0.5 * df.dy.values ** 2])
    al, D, Cv = np.linalg.lstsq(X, Y, rcond=None)[0]
    return D, Cv, al * 252, df.y.mean() * 100


for nm, real, y in [("IEF", P["ief_real"], P["tnx"]), ("TLT", P["tlt_real"], P["tyx"])]:
    for a, b in [("2002-07-30", "2026-11-17"), ("2002-07-30", "2008-12-31"), ("2009-01-01", "2016-12-31"),
                 ("2017-01-01", "2021-12-31"), ("2022-01-01", "2026-11-17")]:
        D, Cv, al, ym = fit(real, y, a, b)
        print(f"  {nm} fit {a}..{b}: duration {D:5.2f} (mean yield {ym:4.2f}%)")
print("  mean 10y yield 1988-2002-07: %.2f%% | mean 30y yield 1988-2002-07: %.2f%%" % (
    P.loc["1988":"2002-07-29", "tnx"].mean(), P.loc["1988":"2002-07-29", "tyx"].mean()))


def dpar(y, n):
    """modified duration of an n-year semi-annual par bond at yield y (decimal)."""
    y = np.maximum(y, 1e-4)
    return (1 / y) * (1 - (1 + y / 2) ** (-2 * n))


def solve_n(D_target, y_mean):
    lo, hi = 0.5, 60.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if dpar(y_mean, mid) < D_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


VAL = json.load(open(os.path.join(HERE, "panel_validation.json")))
Pa = P.copy()
for nm, ycol, key in [("ief", "tnx", "ief_calib"), ("tlt", "tyx", "tlt_calib")]:
    D0, C0, a0 = VAL[key]["duration"], VAL[key]["convexity"], VAL[key]["alpha_ann"] / 252
    yd = Pa[ycol] / 100
    ym = yd[Pa[f"{nm}_real"].notna()].mean()
    n = solve_n(D0, ym)
    Dt = dpar(yd.shift(1), n)
    Ct = C0 * (Dt / D0) ** 2
    dy = yd.diff()
    syn_era = a0 + yd.shift(1) / 252 - Dt * dy + 0.5 * Ct * dy ** 2
    Pa[f"{nm}_era"] = Pa[f"{nm}_real"].fillna(syn_era).fillna(0)
    pre = Pa.index < "2002-07-30"
    print(f"  {nm}: equivalent maturity {n:4.1f}y (D={D0} at mean real-era yield {ym*100:.2f}%) -> mean D 1988-2002 "
          f"{Dt[pre & (Pa.index >= '1988-01-01')].mean():.2f} | pre-2002 vol const-D {Pa.loc[pre, nm].std()*np.sqrt(252)*100:.2f}% "
          f"vs era-D {Pa.loc[pre, nm + '_era'].std()*np.sqrt(252)*100:.2f}%")
    yr = lambda s: (1 + s[(s.index >= '1988-01-01') & pre]).groupby(s[(s.index >= '1988-01-01') & pre].index.year).prod() - 1  # noqa
    t = pd.concat([yr(Pa[nm]), yr(Pa[nm + "_era"])], axis=1); t.columns = ["constD", "eraD"]
    print("     calendar-year returns const-D / era-D: " + " ".join(f"{i}:{r.constD*100:+.1f}/{r.eraD*100:+.1f}"
                                                            for i, r in t.iterrows() if i in (1989, 1994, 1995, 1999, 2000)))

for lab, kw, offs in [("F3", dict(lev="tqqq", voltgt=0.20), ["ief", "tlt"]), ("F2", dict(lev="qld", voltgt=0.30), ["ief", "tlt"])]:
    for off in offs:
        r0 = C.backtest(Pa, C.strategy_weights(Pa, off=off, **kw))
        r1 = C.backtest(Pa, C.strategy_weights(Pa, off=off + "_era", **kw))
        a0_, a1_ = st(r0, *H1), st(r1, *H1)
        i0, i1 = st(r0, *IS), st(r1, *IS)
        print(f"  {lab}-{off.upper()}: H1 Cal {a0_['calmar']:.3f} -> era-D {a1_['calmar']:.3f} (CAGR {a0_['cagr']*100:.2f} -> "
              f"{a1_['cagr']*100:.2f}%, DD {a0_['maxdd']*100:.1f} -> {a1_['maxdd']*100:.1f}%) | IS Cal {i0['calmar']:.3f} -> {i1['calmar']:.3f}"
              f" (CAGR {i0['cagr']*100:.2f} -> {i1['cagr']*100:.2f}%, DD {i0['maxdd']*100:.1f} -> {i1['maxdd']*100:.1f}%)")

print("\n" + "=" * 90 + "\nE3  gold 0.25 vs 0.15 for the DEFAULT knobs, per window\n" + "=" * 90)
for lab, kw in [("F3", dict(lev="tqqq", voltgt=0.20)), ("F2", dict(lev="qld", voltgt=0.30))]:
    out = []
    for gw in [0.15, 0.25]:
        r = C.backtest(P, C.strategy_weights(P, gold_w=gw, **kw))
        out.append({w: st(r, *ab)["calmar"] for w, ab in {"H1": H1, "H2": H2, "IS": IS}.items()})
    print(f"  {lab} default knobs: " + " | ".join(f"{w} g.15 {out[0][w]:.3f} -> g.25 {out[1][w]:.3f}" for w in out[0]))

print("\n" + "=" * 90 + "\nE4  'monthly rebalance' ablation: month-END marks shift the Aug-Sep halving to Sep-Oct holdings\n" + "=" * 90)
_ml = pd.Series(IDX, index=IDX).groupby(IDX.to_period("M")).max()
MONTHLY = pd.Series(False, index=IDX); MONTHLY[pd.DatetimeIndex(_ml.values)] = True
for lab, kw in [("F3", dict(lev="tqqq", voltgt=0.20)), ("F2", dict(lev="qld", voltgt=0.30))]:
    rw = C.backtest(P, C.strategy_weights(P, **kw))
    rm = C.backtest(P, C.strategy_weights(P, **kw), mark=MONTHLY)                       # specialist's ablation
    rm2 = C.backtest(P, C.strategy_weights(P, season_months=(7, 8), **kw), mark=MONTHLY)   # halving held in Aug-Sep
    rmn = C.backtest(P, C.strategy_weights(P, seasonal=False, **kw), mark=MONTHLY)
    rwn = C.backtest(P, C.strategy_weights(P, seasonal=False, **kw))
    for nm_, ab in {"H1": H1, "H2": H2, "IS": IS}.items():
        w_, m_, m2_ = st(rw, *ab)["calmar"], st(rm, *ab)["calmar"], st(rm2, *ab)["calmar"]
        print(f"  {lab} {nm_}: weekly {w_:.3f} | monthly as ablated {m_:.3f} (d {w_-m_:+.3f}) | monthly w/ halving on"
              f" Aug-Sep holdings {m2_:.3f} (d {w_-m2_:+.3f}) | both without Aug-Sep: weekly {st(rwn,*ab)['calmar']:.3f}"
              f" monthly {st(rmn,*ab)['calmar']:.3f} (d {st(rwn,*ab)['calmar']-st(rmn,*ab)['calmar']:+.3f})")
print("done")
