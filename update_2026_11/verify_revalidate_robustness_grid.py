"""
ROBUSTNESS REVIEW of update_2026_11/revalidate.py  --  part 2 (grid-based)
  * rebuilds the 2 x 540 parameter grid with common.py (same values as revalidate.py) and cross-checks
    IS Calmar against revalidate.json
  * paired F2 vt.30 vs F3 vt.20 on EXCESS-return Sharpe (rf subtracted)
  * plateau 'box with Aug-Sep on' numbers quoted in the claims
  * walk-forward under alternative split years and rolling-origin (pre-freeze data only)
  * component ablation under alternative splits (thirds, 2000 / 2010 splits)
Writes nothing except stdout.   run: python -u update_2026_11/verify_revalidate_robustness_grid.py
"""
import sys, os, json, time, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
P = C.panel(); IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
J = json.load(open(os.path.join(HERE, "revalidate.json")))
RFD = P["rf_d"].values


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- vectorised stats identical to C.stats
def mstats(R, a=None, b=FREEZE):
    """R: T x N array aligned to IDX.  returns dict of arrays (cagr, maxdd, calmar, sharpe, exsharpe)."""
    m = IDX >= START
    if a is not None:
        m &= IDX >= pd.Timestamp(a)
    if b is not None:
        m &= IDX <= pd.Timestamp(b)
    x = R[m]; d = IDX[m]
    e = np.cumprod(1 + x, axis=0)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
    sh = x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(252)
    xe = x - RFD[m][:, None]
    exs = xe.mean(axis=0) / xe.std(axis=0, ddof=1) * np.sqrt(252)
    return dict(cagr=cagr, maxdd=dd, calmar=cagr / np.abs(dd), sharpe=sh, exsharpe=exs)


# ---------------------------------------------------------------- grid
GV = dict(voltgt=[0.15, 0.20, 0.25, 0.30, 0.35], chop_thr=[0.20, 0.25, 0.30], chop_pos=[0.35, 0.50, 0.65],
          gold_w=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25], seasonal=[False, True])
rows, cols = [], []
log("building grid returns")
for lev in ["tqqq", "qld"]:
    for vt, th, po, gw, se in itertools.product(*GV.values()):
        w = C.strategy_weights(P, lev=lev, voltgt=vt, chop_thr=th, chop_pos=po, seasonal=se,
                               gold="gold_phys", gold_w=gw, off="ief")
        cols.append(C.backtest(P, w).values)
        rows.append(dict(lev=lev, voltgt=vt, chop_thr=th, chop_pos=po, gold_w=gw, seasonal=se))
R = np.column_stack(cols); G = pd.DataFrame(rows)
log(f"grid built: {R.shape}")
isx = mstats(R)
jg = pd.DataFrame(J["grid"])
assert len(jg) == len(G)
mx = np.nanmax(np.abs(jg["IS_calmar"].values - isx["calmar"]))
print(f"  cross-check vs revalidate.json grid: max |IS Calmar diff| = {mx:.2e}  (rows aligned by construction order: "
      f"{bool((jg['lev'].values == G['lev'].values).all() and np.allclose(jg['voltgt'], G['voltgt']))})")

WINS = {"IS": (None, FREEZE), "H1": (None, "2006-12-31"), "H2": ("2007-01-01", FREEZE),
        "H2_FULL": ("2007-01-01", LATEST), "FULL": (None, LATEST)}
for wn, (a, b) in WINS.items():
    s = mstats(R, a, b)
    for k, v in s.items():
        G[f"{wn}_{k}"] = v
DEF = {"tqqq": dict(voltgt=0.20, chop_thr=0.25, chop_pos=0.50, gold_w=0.15, seasonal=True),
       "qld": dict(voltgt=0.30, chop_thr=0.25, chop_pos=0.50, gold_w=0.15, seasonal=True)}


def locate_i(lev, p):
    m = (G.lev == lev) & (G.seasonal == p["seasonal"])
    for k in ["voltgt", "chop_thr", "chop_pos", "gold_w"]:
        m &= np.isclose(G[k], p[k])
    return int(np.flatnonzero(m.values)[0])


DI = {lev: locate_i(lev, DEF[lev]) for lev in DEF}

# ---------------------------------------------------------------- 1. paired 2x vs 3x on excess Sharpe
log("1. paired F2 vt.30 vs F3 vt.20 (108 nuisance combos): raw vs excess Sharpe")
key = ["chop_thr", "chop_pos", "gold_w", "seasonal"]
a2 = G[(G.lev == "qld") & np.isclose(G.voltgt, 0.30)].set_index(key)
a3 = G[(G.lev == "tqqq") & np.isclose(G.voltgt, 0.20)].set_index(key)
for wn in ["H1", "H2", "IS", "FULL"]:
    ds = a2[f"{wn}_sharpe"] - a3[f"{wn}_sharpe"]; de = a2[f"{wn}_exsharpe"] - a3[f"{wn}_exsharpe"]
    dc = a2[f"{wn}_calmar"] - a3[f"{wn}_calmar"]
    print(f"  {wn:5s} n={len(ds)} F2 higher RAW Sharpe {(ds>0).mean()*100:5.1f}% (median d {ds.median():+.3f}) | "
          f"F2 higher EXCESS Sharpe {(de>0).mean()*100:5.1f}% (median d {de.median():+.3f}, min {de.min():+.3f}, max {de.max():+.3f})"
          f" | F2 higher Calmar {(dc>0).mean()*100:5.1f}%")
# all 2x configs vs all 3x configs (no pairing): grid medians of excess Sharpe
for wn in ["H1", "H2", "IS"]:
    print(f"  grid median excess Sharpe {wn}: qld {G[G.lev=='qld'][f'{wn}_exsharpe'].median():.3f} "
          f"tqqq {G[G.lev=='tqqq'][f'{wn}_exsharpe'].median():.3f} | raw qld {G[G.lev=='qld'][f'{wn}_sharpe'].median():.3f} "
          f"tqqq {G[G.lev=='tqqq'][f'{wn}_sharpe'].median():.3f}")

# ---------------------------------------------------------------- 2. plateau numbers quoted in the claims
log("2. plateau checks")
IDXK = {k: (lambda v, k=k: int(np.argmin(np.abs(np.array(GV[k]) - v)))) for k in ["voltgt", "chop_thr", "chop_pos", "gold_w"]}
for k in IDXK:
    G[k + "_i"] = G[k].apply(IDXK[k])
G["seasonal_i"] = G["seasonal"].astype(int)
PI = ["voltgt_i", "chop_thr_i", "chop_pos_i", "gold_w_i", "seasonal_i"]
for lev in ["tqqq", "qld"]:
    df = G[G.lev == lev]; d = G.iloc[DI[lev]]
    di = np.array([d[c] for c in PI])
    diff = np.abs(df[PI].values - di[None, :])
    box = df[diff.max(axis=1) <= 1]; box_on = box[box.seasonal]
    print(f"  {lev}: default IS Cal {d['IS_calmar']:.3f} pctile {(df['IS_calmar'] < d['IS_calmar']).mean():.3f} | box n {len(box)} "
          f"median {box['IS_calmar'].median():.3f} | box AS-on n {len(box_on)} median {box_on['IS_calmar'].median():.3f} "
          f"share>=0.70 {(box_on['IS_calmar']>=0.70).mean():.3f} (all-box share>=0.70 {(box['IS_calmar']>=0.70).mean():.3f})")
    for wn in ["H1", "H2"]:
        print(f"       {wn}: default {d[f'{wn}_calmar']:.3f} pctile {(df[f'{wn}_calmar'] < d[f'{wn}_calmar']).mean():.3f}"
              f" grid median {df[f'{wn}_calmar'].median():.3f}")

# ---------------------------------------------------------------- 3. walk-forward picks: did gold 0.25 help in the test half?
log("3. walk-forward picks: gold 0.25 vs 0.15 (same other knobs) in the test window")
for lev, pk, test in [("tqqq", dict(voltgt=0.20, chop_thr=0.30, chop_pos=0.35, seasonal=True), "H2_FULL"),
                      ("tqqq", dict(voltgt=0.20, chop_thr=0.30, chop_pos=0.35, seasonal=True), "H2"),
                      ("tqqq", dict(voltgt=0.25, chop_thr=0.25, chop_pos=0.50, seasonal=True), "H1"),
                      ("qld", dict(voltgt=0.35, chop_thr=0.25, chop_pos=0.35, seasonal=True), "H2_FULL"),
                      ("qld", dict(voltgt=0.35, chop_thr=0.25, chop_pos=0.35, seasonal=True), "H2"),
                      ("qld", dict(voltgt=0.25, chop_thr=0.20, chop_pos=0.50, seasonal=True), "H1")]:
    i25 = locate_i(lev, {**pk, "gold_w": 0.25}); i15 = locate_i(lev, {**pk, "gold_w": 0.15})
    print(f"  {lev} {pk} test {test}: g.25 {G.iloc[i25][f'{test}_calmar']:.3f} vs g.15 {G.iloc[i15][f'{test}_calmar']:.3f}"
          f" -> gold 0.25 {'HELPED' if G.iloc[i25][f'{test}_calmar'] > G.iloc[i15][f'{test}_calmar'] else 'did not help'}")

# ---------------------------------------------------------------- 4. walk-forward under alternative split years (pre-freeze only)
log("4. walk-forward with alternative split years (choose by train Calmar, score on test; all pre-freeze)")
WF = []
for scope_lev in ["tqqq", "qld"]:
    sidx = np.flatnonzero((G.lev == scope_lev).values)
    for Y in [1996, 1999, 2002, 2006, 2009, 2012, 2015]:
        trA, teA = (None, f"{Y}-12-31"), (f"{Y+1}-01-01", FREEZE)
        for direction, tr, te in [("early->late", trA, teA), ("late->early", teA, trA)]:
            str_ = mstats(R[:, sidx], *tr); ste = mstats(R[:, sidx], *te)
            pk = int(np.argmax(str_["calmar"])); di = int(np.flatnonzero(sidx == DI[scope_lev])[0])
            tc = ste["calmar"]
            WF.append(dict(lev=scope_lev, split=Y, dir=direction, pick_train=float(str_["calmar"][pk]),
                           pick_test=float(tc[pk]), default_test=float(tc[di]), median_test=float(np.median(tc)),
                           oracle_test=float(tc.max()), pick_test_pct=float((tc < tc[pk]).mean()),
                           default_test_pct=float((tc < tc[di]).mean()),
                           pick=G.iloc[sidx[pk]][["voltgt", "chop_thr", "chop_pos", "gold_w", "seasonal"]].to_dict()))
W = pd.DataFrame(WF)
W["pick_minus_default"] = W.pick_test - W.default_test
W["degradation"] = W.pick_train - W.pick_test
W["pick_minus_median"] = W.pick_test - W.median_test
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(W[["lev", "split", "dir", "pick_train", "pick_test", "default_test", "median_test", "oracle_test",
             "pick_test_pct", "default_test_pct", "pick_minus_default", "degradation"]].round(3).to_string(index=False))
for lev in ["tqqq", "qld"]:
    w = W[W.lev == lev]
    print(f"  {lev}: pick beats default in {int((w.pick_minus_default>0).sum())}/{len(w)} tests; mean pick-default "
          f"{w.pick_minus_default.mean():+.3f}; mean pick test pctile {w.pick_test_pct.mean():.2f} vs default "
          f"{w.default_test_pct.mean():.2f}; mean train->test degradation of pick {w.degradation.mean():+.3f}; "
          f"mean pick-minus-grid-median {w.pick_minus_median.mean():+.3f}")
    print(f"        picks' gold_w: {w.pick.apply(lambda p: p['gold_w']).value_counts().to_dict()}  seasonal: "
          f"{w.pick.apply(lambda p: p['seasonal']).value_counts().to_dict()}")

# rolling-origin: expanding train 1988..Y, test next 5 calendar years (pre-freeze only)
log("4b. rolling-origin walk-forward (train 1988..Y, test Y+1..Y+5)")
RO = []
for lev in ["tqqq", "qld"]:
    sidx = np.flatnonzero((G.lev == lev).values); di = int(np.flatnonzero(sidx == DI[lev])[0])
    for Y in [1995, 2000, 2005, 2010, 2015, 2020]:
        str_ = mstats(R[:, sidx], None, f"{Y}-12-31")
        ste = mstats(R[:, sidx], f"{Y+1}-01-01", min(pd.Timestamp(f"{Y+5}-12-31"), FREEZE))
        pk = int(np.argmax(str_["calmar"])); tc = ste["calmar"]
        RO.append(dict(lev=lev, Y=Y, pick_test=tc[pk], default_test=tc[di], median=np.median(tc),
                       pick_pct=(tc < tc[pk]).mean(), def_pct=(tc < tc[di]).mean()))
RO = pd.DataFrame(RO)
print(RO.round(3).to_string(index=False))
for lev in ["tqqq", "qld"]:
    r = RO[RO.lev == lev]
    print(f"  {lev}: rolling-origin pick beats default {int((r.pick_test > r.default_test).sum())}/{len(r)}; pick beats grid median "
          f"{int((r.pick_test > r['median']).sum())}/{len(r)}; mean pick pctile {r.pick_pct.mean():.2f}, default {r.def_pct.mean():.2f}")

# ---------------------------------------------------------------- 5. component ablation under alternative splits
log("5. component ablation under alternative splits (faithful instruments)")
MON = pd.Series(IDX, index=IDX).groupby(IDX.to_period("M")).max()
MONTHLY = pd.Series(False, index=IDX); MONTHLY[pd.DatetimeIndex(MON.values)] = True
ABL = {"no EMA gate": dict(risk_on=pd.Series(True, index=IDX)), "no vol target": dict(voltgt=99.0),
       "chop full size": dict(chop_pos=1.0), "chop zero": dict(chop_pos=0.0), "BIL off-leg": dict(off="bills"),
       "no gold": dict(gold_w=0.0), "no Aug-Sep": dict(seasonal=False), "daily rebal": "daily", "monthly rebal": "monthly"}
SPL = {"H1": (None, "2006-12-31"), "H2": ("2007-01-01", FREEZE), "T1 88-99": (None, "1999-12-31"),
       "T2 00-12": ("2000-01-01", "2012-12-31"), "T3 13-frz": ("2013-01-01", FREEZE),
       "A 88-99": (None, "1999-12-31"), "B 00-frz": ("2000-01-01", FREEZE),
       "C 88-09": (None, "2009-12-31"), "D 10-frz": ("2010-01-01", FREEZE)}
for fam, base in [("F3", dict(lev="tqqq", voltgt=0.20)), ("F2", dict(lev="qld", voltgt=0.30))]:
    rb = C.backtest(P, C.strategy_weights(P, **base)).values[:, None]
    colsA, names = [], []
    for ak, av in ABL.items():
        if isinstance(av, str):
            mk = pd.Series(True, index=IDX) if av == "daily" else MONTHLY
            colsA.append(C.backtest(P, C.strategy_weights(P, **base), mark=mk).values)
        else:
            colsA.append(C.backtest(P, C.strategy_weights(P, **{**base, **av})).values)
        names.append(ak)
    RA = np.column_stack(colsA)
    res = {}
    for sn, (a, b) in SPL.items():
        sb = mstats(rb, a, b)["calmar"][0]; sa = mstats(RA, a, b)["calmar"]
        res[sn] = sb - sa
    T = pd.DataFrame(res, index=names)
    T["min_over_thirds"] = T[["T1 88-99", "T2 00-12", "T3 13-frz"]].min(axis=1)
    T["helps_all_thirds"] = T[["T1 88-99", "T2 00-12", "T3 13-frz"]].gt(0).all(axis=1)
    T["helps_both_halves_alt2000"] = T[["A 88-99", "B 00-frz"]].gt(0).all(axis=1)
    T["helps_both_halves_alt2010"] = T[["C 88-09", "D 10-frz"]].gt(0).all(axis=1)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"  {fam}: dCalmar (default - ablated; + = component helps)")
        print(T.round(3).to_string())
    print(f"  {fam}: helps in all thirds {int(T.helps_all_thirds.sum())}/9 | both halves split@2000 "
          f"{int(T.helps_both_halves_alt2000.sum())}/9 | split@2010 {int(T.helps_both_halves_alt2010.sum())}/9 | "
          f"components with a half-delta < 0.02: {list(T.index[(T[['H1','H2']].abs().min(axis=1) < 0.02)])}")

# ---------------------------------------------------------------- 6. OOS facts quoted in C10/C12
log("6. OOS / current-drawdown facts")
ab = J["ablation"]
print(f"  no-Aug-Sep OOS: F3 {ab['F3|faithful']['no Aug-Sep half']['OOS_total']*100:.2f}%  F2 "
      f"{ab['F2|faithful']['no Aug-Sep half']['OOS_total']*100:.2f}%")
F3 = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20))
e = (1 + F3[F3.index >= START]).cumprod()
pk = e[:FREEZE].idxmax()
print(f"  F3-IEF: peak {pk.date()} -> freeze DD {(e[FREEZE]/e[pk]-1)*100:.1f}% ; -> latest DD {(e.iloc[-1]/e[pk]-1)*100:.1f}%; "
      f"pre-2026-06-02 peak? {pk}")
log("done")
