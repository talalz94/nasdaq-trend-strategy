"""
INTEGRITY REVIEW of update_2026_11/ucits.py  --  step 3: follow-ups.
  A. executions on TRULY stale closes (unchanged close or missing row) and a 'fresh-price execution' variant
     (execute only when every European line whose weight changes has a fresh close) -> size of stale-price bias
  B. CBU0.L vs IEF weekly / 4-week correlation by sub-period (is the low full-history correlation timing or stale data?)
  C. same-product TE with the un-cleaned 2025-10-24 prints removed (LQQ3.L, QQQ3.MI)
  D. Yahoo FX bar timing: corr(EURUSD daily change, DXY change at D-1 / D / D+1); 2015-01-22 ECB bars
  E. LQQ.PA / LQQ.MI around 2026-07-09 and 2015-01-02 (what the '200:1 split' really is) + volume units
  F. US F2 / F3 1988..freeze directly with common.py (C16)
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
on_dates, us_level, bt_pc, eu_open_mask = g["on_dates"], g["us_level"], g["bt_pc"], g["eu_open_mask"]
COST_REAL, COL2TK, held_weights_eu = g["COST_REAL"], g["COL2TK"], g["held_weights_eu"]


def raw(tk):
    x = PX[PX.ticker == tk].set_index("date").sort_index()
    return x[x.index <= END].dropna(subset=["close"])


def fresh_mask(tk):
    """panel dates on which tk has a row whose close differs from the previous row's close"""
    x = raw(tk)
    fr = x.close.diff().fillna(1) != 0
    return pd.Series(P2.index.isin(x.index[fr.values]), index=P2.index)


print("A. STALE-CLOSE EXECUTIONS AND A FRESH-PRICE EXECUTION VARIANT")
for nm, lev, off, gold, vt, a in [("F2", "u_lqq", "u_cbu0", "u_igln", 0.30, "2011-04-28"),
                                  ("F3", "u_qqq3", "u_cbu0", "u_igln", 0.20, "2013-01-03"),
                                  ("F2 longer", "u_lqq", "u_ibtm", "u_phau", 0.30, "2009-01-21")]:
    a = pd.Timestamp(a)
    w = C.strategy_weights(P2, lev=lev, voltgt=vt, off=off, gold=gold)
    eu = [COL2TK[c] for c in (lev, off, gold)]
    m0 = eu_open_mask(eu)
    r0, ww0 = bt_pc(w, COST_REAL, 2, P2, m0)
    # stale executions in the original run
    chg = ww0.diff().abs()
    chg = chg[(chg.index > a) & (chg.index <= END)]
    for c in (lev, off):
        tk = COL2TK[c]
        fm = fresh_mask(tk)
        ex_next = chg.index[chg[c] > 1e-12]
        ex_day = P2.index[P2.index.get_indexer(ex_next) - 1]
        st = ~fm.reindex(ex_day).values
        print(f"  {nm:9s} {tk:7s}: {len(ex_day)} executions, {st.sum()} on an unchanged/missing close "
              f"({', '.join(sorted(set(str(d.year) for d in ex_day[st])))})")
    # variant: also require fresh closes of the lev and off lines (gold weight is constant)
    m1 = m0 & fresh_mask(COL2TK[lev]) & fresh_mask(COL2TK[off])
    r1, ww1 = bt_pc(w, COST_REAL, 2, P2, m1)
    m2 = m0 & fresh_mask(COL2TK[lev])
    r2, _ = bt_pc(w, COST_REAL, 2, P2, m2)
    s0, s1, s2 = (C.stats(x, a, END, start=None) for x in (r0, r1, r2))
    print(f"  {nm:9s} original (as ucits.py)        {C.fmt(s0)}")
    print(f"  {nm:9s} fresh lev+bond closes required {C.fmt(s1)}   dCAGR {(s1['cagr']-s0['cagr'])*100:+.2f}")
    print(f"  {nm:9s} fresh lev close required      {C.fmt(s2)}   dCAGR {(s2['cagr']-s0['cagr'])*100:+.2f}")

print("\nB. CBU0.L vs IEF and QQQ3.L vs TQQQ by sub-period (weekly corr / 4-week corr / weekly TE)")


def sub_overlap(su, a_col, a, b):
    lv = us_level(a_col)
    W = pd.concat([su.resample("W-FRI").last(), lv.resample("W-FRI").last()], axis=1, keys=["u", "a"]).dropna()
    W = W[(W.index >= pd.Timestamp(a)) & (W.index <= pd.Timestamp(b))]
    rw = W.pct_change().dropna()
    r4 = W.iloc[::4].pct_change().dropna()
    return rw.u.corr(rw.a), r4.u.corr(r4.a), (rw.u - rw.a).std() * np.sqrt(52), len(rw)


for tk, col in [("CBU0.L", "ief_real"), ("QQQ3.L", "tqqq_real"), ("LQQ.PA", "qld_real")]:
    for a, b in [("2011-01-01", "2014-12-31"), ("2015-01-01", "2018-12-31"), ("2019-01-01", str(WEND.date()))]:
        if tk == "QQQ3.L" and a == "2011-01-01":
            a = "2013-01-01"
        cw, c4, te, n = sub_overlap(CLEAN[tk], col, a, b)
        print(f"  {tk:7s} {a[:4]}-{b[:4]}: weekly corr {cw:.3f}  4-week corr {c4:.3f}  weekly TE {te*100:5.2f}%  (n={n})")

print("\nC. SAME-PRODUCT TE WITH THE 2025-10-24 PRINTS REMOVED")
for t1, t2 in [("LQQ3.L", "QQQ3.L"), ("QQQ3.MI", "QQQ3.L")]:
    for drop in (False, True):
        s1 = CLEAN[t1].copy()
        if drop:
            s1 = s1.drop(pd.Timestamp("2025-10-24"), errors="ignore")
        W = pd.concat([s1.resample("W-FRI").last(), CLEAN[t2].resample("W-FRI").last()], axis=1).dropna()
        W = W[W.index <= WEND]
        r = W.pct_change().dropna()
        te = (r.iloc[:, 0] - r.iloc[:, 1]).std() * np.sqrt(52)
        print(f"  {t1} vs {t2}  drop 2025-10-24={drop!s:5s}  weekly TE {te*100:5.2f}%  corr {r.iloc[:,0].corr(r.iloc[:,1]):.4f}")
    a_col = "tqqq_real"
    for drop in (False, True):
        s1 = CLEAN[t1].copy()
        if drop:
            s1 = s1.drop(pd.Timestamp("2025-10-24"), errors="ignore")
        o = g["overlap"](s1, a_col)
        print(f"  {t1} vs TQQQ drop={drop!s:5s} TE {o['te_weekly_ann']*100:5.2f}%  corr_w {o['corr_weekly']:.3f}")

print("\nD. YAHOO FX BAR TIMING")
fx = g["FX"]["EURUSD=X"]
lf = np.log(fx).diff().dropna()
dx = np.log(P["dxy"]).diff()
for lag in (-1, 0, 1):
    z = pd.concat([lf, dx.shift(lag)], axis=1, keys=["fx", "dxy"]).dropna()
    z = z[z.index >= "2004-01-01"]
    print(f"  corr(EURUSD bar D change, DXY change of D{-lag:+d}) = {z.fx.corr(z.dxy):+.3f}  (n={len(z)})")
for y0, y1 in [("2004", "2012"), ("2013", "2019"), ("2020", "2026")]:
    out = []
    for lag in (1, 0):
        z = pd.concat([lf, dx.shift(lag)], axis=1, keys=["fx", "dxy"]).dropna().loc[y0:y1]
        out.append(z.fx.corr(z.dxy))
    print(f"  {y0}-{y1}: corr with DXY D-1 {out[0]:+.3f}  with DXY D {out[1]:+.3f}")
print("  EURUSD bars around ECB QE (announced 2015-01-22 ~13:30 London):",
      " ".join(f"{d.date()}:{v:.4f}" for d, v in fx.loc["2015-01-20":"2015-01-26"].items()))
print("  EURUSD bars around the SNB floor removal (2015-01-15 ~09:30 London):",
      " ".join(f"{d.date()}:{v:.4f}" for d, v in fx.loc["2015-01-13":"2015-01-19"].items()))

print("\nE. LQQ.PA / LQQ.MI raw close & volume around 2026-07-09 and the 2015-01-02 boundary")
for tk in ["LQQ.PA", "LQQ.MI"]:
    x = raw(tk)
    sl = x.loc["2026-07-03":"2026-07-15", ["close", "volume", "splits"]]
    print(f"  {tk}: " + " | ".join(f"{d.strftime('%m-%d')} {r.close:.4g} v{int(r.volume)}" for d, r in sl.iterrows()))
x = raw("LQQ.PA")
for a, b in [("2014-10-01", "2014-12-31"), ("2015-01-02", "2015-03-31"), ("2026-04-01", "2026-07-08"), ("2026-07-10", "2026-11-16")]:
    s = x.loc[a:b]
    print(f"  LQQ.PA {a}..{b}: median close {s.close.median():9.3f} EUR  median volume {s.volume.median():10.0f}  "
          f"median EUR value traded {(s.close*s.volume).median()/1e6:7.2f}m")
y = raw("LQQ.MI")
for a, b in [("2026-04-01", "2026-07-08"), ("2026-07-10", "2026-11-16")]:
    s = y.loc[a:b]
    print(f"  LQQ.MI {a}..{b}: median close {s.close.median():9.3f} EUR  median volume {s.volume.median():10.0f}  "
          f"median EUR value traded {(s.close*s.volume).median()/1e6:7.2f}m")
pa, mi = raw("LQQ.PA").close, raw("LQQ.MI").close
z = pd.concat([pa, mi], axis=1, keys=["pa", "mi"]).dropna()
rat = (z.mi / z.pa)
print("  ratio LQQ.MI/LQQ.PA close: " + " ".join(f"{d.strftime('%Y-%m-%d')}:{v:.1f}" for d, v in rat.iloc[[0, 100, 500, 1000, -95, -92, -90, -88, -1]].items()))

print("\nF. US F2 / F3 1988..freeze directly with common.py")
for lev, vt in [("qld", 0.30), ("tqqq", 0.20)]:
    w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", off="ief")
    r = C.backtest(P, w)
    print(f"  {lev}: {C.fmt(C.stats(r, b=FREEZE))}")
