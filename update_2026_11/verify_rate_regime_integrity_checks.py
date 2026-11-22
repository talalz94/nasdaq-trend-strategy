"""
INTEGRITY REVIEW of update_2026_11/rate_regime.py  --  step 2: independent checks.

  A. independent re-implementation of the strategy + backtest (no common.py) for the key
     policies -> do the headline numbers reproduce (CAGR within 0.3pp, maxDD within 1pp)?
  B. risk-off share of in-sample days / average off-leg weight (claim C1)
  C. timing / lookahead sensitivity: off-leg signal delayed 1 extra day; whole strategy lag=2
  D. pre-2000 gold interpolation leak: replace monthly-interpolated gold (uses the NEXT
     month-end price) with causal month-end booking; does F3's worst DD / the ranking change?
  E. freeze misuse in the synthetic IEF: panel calibration (alpha, D, C) was fitted on
     2002-07..2026-11-17 (includes post-freeze data). Re-fit on pre-freeze only, and alpha=0.
  F. cost sensitivity (0 / 3 / 10 bp) for bond trend vs static IEF
  G. Reality Check on the RISK-ADJUSTED objective (Sharpe and Calmar differences), since the
     specialist's RC tests mean log-return, which a de-risking rule is expected to lose
  H. misc: count of F2 policies above static IEF; H1 CAGR cost of bond trend; same-day T-bill yield

Writes nothing inside the project; results JSON goes to the scratch dir given as argv[1].
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

T0 = time.time()
OUTDIR = sys.argv[1]
P0 = C.panel()
idx = P0.index
FREEZE, START = C.FREEZE, C.START
H1 = ("1988-01-01", "2006-12-31"); H2 = ("2007-01-01", str(FREEZE.date()))
SAVED = json.load(open(os.path.join(HERE, "rate_regime.json")))
OUT = {}


def ma(s, n):
    return s.rolling(n, min_periods=n).mean()


# ------------------------------------------------------------------ signals on an arbitrary panel
def sigs(P):
    ief_px = (1 + P["ief"].fillna(0)).cumprod()
    return {
        "durA_tnx200_bills": P["tnx"] > ma(P["tnx"], 200),
        "bondtrend200_bills": ief_px < ma(ief_px, 200),
        "corr126_bills": P["ndx_ret"].rolling(126, min_periods=126).corr(P["ief"]) > 0,
        "corr63_bills": P["ndx_ret"].rolling(63, min_periods=63).corr(P["ief"]) > 0,
        "bondtrend100_bills": ief_px < ma(ief_px, 100),
    }


def alloc_static(**kw):
    return pd.DataFrame({k: pd.Series(float(v), index=idx) for k, v in kw.items()})


def alloc_switch(cond, short="bills", long="ief", delay=0):
    c = cond.reindex(idx).fillna(False).astype(float)
    if delay:
        c = c.shift(delay).fillna(0.0)
    return pd.DataFrame({long: 1 - c, short: c})


# ------------------------------------------------------------------ A. independent engine
def indep_backtest(P, lev, vt, alloc, cost=3e-4, gold_w=0.15):
    """loop implementation, written without common.py. Decision at each Friday close,
    held from the next session until (and including) the next Friday."""
    ndx = P["ndx"].values
    n = len(P)
    ema = pd.Series(ndx).ewm(span=100, adjust=False).mean().values
    above = ndx > ema
    dr = pd.Series(ndx).pct_change()
    vol = (dr.rolling(20).std() * np.sqrt(252)).values
    er = (pd.Series(ndx).diff(20).abs() / pd.Series(ndx).diff().abs().rolling(20).sum()).values
    fri = (P.index.dayofweek == 4)
    month = P.index.month.values
    cols = [lev, "gold_phys"] + [c for c in alloc.columns]
    cols = list(dict.fromkeys(cols))
    R = np.column_stack([P[c].fillna(0).values for c in cols])
    A = alloc.reindex(P.index).ffill().fillna(0)
    held = np.zeros(len(cols)); prev = np.zeros(len(cols))
    target = None
    out = np.zeros(n); W = np.zeros((n, len(cols)))
    for t in range(n):
        if target is not None:                 # weights decided at the previous mark start today
            held = target; target = None
        W[t] = held
        out[t] = float(held @ R[t]) - cost * float(np.abs(held - prev).sum())
        prev = held.copy()
        if fri[t]:
            s = 1.0 if not np.isfinite(vol[t]) or vol[t] <= 0 else min(1.0, vt / vol[t])
            if month[t] in (8, 9):
                s *= 0.5
            if above[t]:
                eq = s if (np.isfinite(er[t]) and er[t] >= 0.25) else 0.5 * s
            else:
                eq = 0.0
            w = np.zeros(len(cols))
            w[cols.index(lev)] += (1 - gold_w) * eq
            rest = max(0.0, (1 - gold_w) * (1 - eq))
            for c in alloc.columns:
                w[cols.index(c)] += rest * A[c].iloc[t]
            w[cols.index("gold_phys")] += gold_w
            target = w
    return pd.Series(out, index=P.index), pd.DataFrame(W, index=P.index, columns=cols)


def st(r, a=START, b=FREEZE):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    e = np.concatenate([[1.0], (1 + s).cumprod().values])
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return dict(cagr=cagr, maxdd=dd, calmar=cagr / abs(dd),
                sharpe=s.mean() / s.std() * np.sqrt(252))


def pr(r, a, b):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    return float((1 + s).prod() - 1)


assert ((P0.index.dayofweek == 4) == P0["mark"].values).all(), "mark != Fridays"
S0 = sigs(P0)
POL = {"static_ief": alloc_static(ief=1), "static_tlt": alloc_static(tlt=1),
       "static_bills": alloc_static(bills=1),
       "durA_tnx200_bills": alloc_switch(S0["durA_tnx200_bills"]),
       "bondtrend200_bills": alloc_switch(S0["bondtrend200_bills"]),
       "corr126_bills": alloc_switch(S0["corr126_bills"])}
STR = {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}
OUT["A_independent_engine"] = {}
print("A. independent engine vs specialist (IS 1988-01-01..2026-07-24)")
INDEP_W = {}
for s, (lev, vt) in STR.items():
    for p, al in POL.items():
        r, W = indep_backtest(P0, lev, vt, al)
        INDEP_W[(s, p)] = W
        m = st(r); sp = SAVED["results"][s][p]["IS_full"]
        y22 = pr(r, "2022-01-01", "2022-12-31"); oos = pr(r, "2026-07-25", "2026-11-17")
        h1 = st(r, *H1); h2 = st(r, *H2)
        OUT["A_independent_engine"][f"{s}/{p}"] = dict(
            cagr=m["cagr"], maxdd=m["maxdd"], calmar=m["calmar"], sharpe=m["sharpe"], y2022=y22, oos=oos,
            calmar_H1=h1["calmar"], calmar_H2=h2["calmar"],
            d_cagr_pp=(m["cagr"] - sp["cagr"]) * 100, d_maxdd_pp=(m["maxdd"] - sp["maxdd"]) * 100)
        print(f"  {s} {p:20s} CAGR {m['cagr']*100:6.2f} (spec {sp['cagr']*100:6.2f})  maxDD {m['maxdd']*100:6.2f} "
              f"(spec {sp['maxdd']*100:6.2f})  Cal {m['calmar']:.3f} (spec {sp['calmar']:.3f})  H1/H2 {h1['calmar']:.3f}/{h2['calmar']:.3f} "
              f" 2022 {y22*100:6.1f}  OOS {oos*100:6.2f}")

# ------------------------------------------------------------------ B. risk-off share, off-leg weight
print("\nB. risk-off share of IS days and off-leg weight")
OUT["B"] = {}
for s, (lev, vt) in STR.items():
    W = INDEP_W[(s, "static_ief")]
    Wi = W[(W.index >= START) & (W.index <= FREEZE)]
    riskoff = (Wi[lev] == 0)
    off = Wi["ief"]
    OUT["B"][s] = dict(share_days_riskoff=float(riskoff.mean()), avg_offleg=float(off.mean()),
                       offleg_when_riskoff=float(off[riskoff].mean()),
                       share_days_offleg_ge_50pct=float((off >= 0.5).mean()))
    print(f"  {s}: risk-off share of IS days {riskoff.mean()*100:.1f}%  avg off-leg {off.mean()*100:.1f}%  "
          f"off-leg when risk-off {off[riskoff].mean()*100:.1f}%   days with off-leg>=50%: {(off>=0.5).mean()*100:.1f}%")


# ------------------------------------------------------------------ helper using common.py mechanics
def cm_run(P, s, alloc, lag=1, cost=C.COST):
    lev, vt = STR[s]
    w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off_weights=alloc)
    return C.backtest(P, w, cost=cost, lag=lag, mark="mark")


def summ(r):
    rr = r[r.index <= FREEZE]
    f = C.stats(rr); h1 = C.stats(rr, *H1); h2 = C.stats(rr, *H2)
    return dict(cagr=f["cagr"], maxdd=f["maxdd"], calmar=f["calmar"], sharpe=f["sharpe"],
                calH1=h1["calmar"], calH2=h2["calmar"], cagrH1=h1["cagr"], cagrH2=h2["cagr"],
                y2022=C.period_return(r, "2022-01-01", "2022-12-31"),
                dotcom=C.period_return(r, "2000-03-01", "2002-10-31"),
                gfc=C.period_return(r, "2007-10-01", "2009-03-31"))


def line(lab, m):
    return (f"  {lab:44s} CAGR {m['cagr']*100:6.2f} maxDD {m['maxdd']*100:6.2f} Cal {m['calmar']:.3f} "
            f"H1 {m['calH1']:.3f} H2 {m['calH2']:.3f} 2022 {m['y2022']*100:6.1f} dotcom {m['dotcom']*100:6.1f} GFC {m['gfc']*100:5.1f}")


# ------------------------------------------------------------------ C. timing sensitivity
print("\nC. timing: off-leg signal delayed 1 extra trading day; whole strategy lag=2")
OUT["C"] = {}
for s in STR:
    for p in ["static_ief", "durA_tnx200_bills", "bondtrend200_bills", "corr126_bills"]:
        base = summ(cm_run(P0, s, POL[p]))
        d1 = summ(cm_run(P0, s, alloc_switch(S0[p], delay=1))) if p != "static_ief" else base
        l2 = summ(cm_run(P0, s, POL[p], lag=2))
        OUT["C"][f"{s}/{p}"] = dict(base=base, sig_delay1=d1, lag2=l2)
        print(line(f"{s} {p} base", base)); print(line(f"{s} {p} signal +1d", d1)); print(line(f"{s} {p} lag=2", l2))

# ------------------------------------------------------------------ D. gold interpolation leak
print("\nD. pre-2000-08 gold: panel (monthly log-interp, uses next month-end) vs causal month-end booking")
P_g = P0.copy()
interp = P0["gold_src"] == "monthly-interp"
g = P0["gold_phys"].copy()
gi = g[interp]
mon = gi.groupby(gi.index.to_period("M"))
booked = pd.Series(0.0, index=gi.index)
for per, grp in mon:
    booked.loc[grp.index[-1]] = float((1 + grp).prod() - 1)
g.loc[interp] = booked
P_g["gold_phys"] = g
chk = ((1 + P0["gold_phys"][interp]).prod(), (1 + P_g["gold_phys"][interp]).prod())
print(f"  total growth over interp period: panel {chk[0]:.4f}  booked {chk[1]:.4f} (must match)")
OUT["D"] = {}
for s in STR:
    for p in ["static_ief", "bondtrend200_bills", "durA_tnx200_bills", "static_bills", "static_tlt"]:
        a = cm_run(P0, s, POL[p]); b = cm_run(P_g, s, POL[p])
        ma_, mb = summ(a), summ(b)
        rr = b[(b.index >= START) & (b.index <= FREEZE)]
        e = (1 + rr).cumprod(); dd = e / e.cummax() - 1; tr = dd.idxmin(); pk = e.loc[:tr].idxmax()
        OUT["D"][f"{s}/{p}"] = dict(panel=ma_, causal_gold=mb, causal_dd_peak=str(pk.date()), causal_dd_trough=str(tr.date()))
        print(line(f"{s} {p} panel gold", ma_)); print(line(f"{s} {p} causal gold", mb) + f"  DD {pk.date()}->{tr.date()}")

# ------------------------------------------------------------------ E. synthetic IEF calibration refit
print("\nE. synthetic IEF before 2002-07-30: panel calibration (fit to 2026-11-17) vs pre-freeze fit vs alpha=0")
yd = P0["tnx"] / 100.0
dy = yd.diff(); carry = yd.shift(1) / 252


def fit(real, a_end):
    df = pd.concat([real, carry, dy], axis=1).dropna(); df.columns = ["r", "carry", "dy"]
    df = df[df.index <= pd.Timestamp(a_end)]
    Y = (df.r - df.carry).values
    X = np.column_stack([np.ones(len(df)), -df.dy.values, 0.5 * df.dy.values ** 2])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


coef_full = fit(P0["ief_real"], "2026-11-17")
coef_pre = fit(P0["ief_real"], str(FREEZE.date()))
print(f"  full-sample fit: alpha {coef_full[0]*252*100:+.3f}%/yr D {coef_full[1]:.3f} C {coef_full[2]:.1f}")
print(f"  pre-freeze fit : alpha {coef_pre[0]*252*100:+.3f}%/yr D {coef_pre[1]:.3f} C {coef_pre[2]:.1f}")
OUT["E"] = dict(coef_full=list(map(float, coef_full)), coef_pre=list(map(float, coef_pre)))
for lab, (a_, D_, C_) in [("prefreeze_fit", coef_pre), ("alpha0", (0.0, coef_full[1], coef_full[2]))]:
    syn = a_ + carry - D_ * dy + 0.5 * C_ * dy ** 2
    P_e = P0.copy()
    P_e["ief"] = P0["ief_real"].fillna(syn).fillna(0)
    Se = sigs(P_e)
    pols = {"static_ief": alloc_static(ief=1), "static_bills": alloc_static(bills=1),
            "bondtrend200_bills": alloc_switch(Se["bondtrend200_bills"]),
            "durA_tnx200_bills": alloc_switch(Se["durA_tnx200_bills"]),
            "corr126_bills": alloc_switch(Se["corr126_bills"])}
    OUT["E"][lab] = {}
    for s in STR:
        for p, al in pols.items():
            m = summ(cm_run(P_e, s, al))
            OUT["E"][lab][f"{s}/{p}"] = m
            print(line(f"{lab} {s} {p}", m))

# ------------------------------------------------------------------ F. cost sensitivity
print("\nF. cost sensitivity (bp per unit one-way turnover)")
OUT["F"] = {}
for s in STR:
    for cst in (0.0, 3e-4, 10e-4):
        a = summ(cm_run(P0, s, POL["static_ief"], cost=cst)); b = summ(cm_run(P0, s, POL["bondtrend200_bills"], cost=cst))
        OUT["F"][f"{s}/{cst*1e4:.0f}bp"] = dict(ief=a, bondtrend=b)
        print(f"  {s} {cst*1e4:4.0f}bp: IEF CAGR {a['cagr']*100:.2f} Cal {a['calmar']:.3f} | bondtrend CAGR {b['cagr']*100:.2f} "
              f"Cal {b['calmar']:.3f} | diff CAGR {(b['cagr']-a['cagr'])*100:+.2f} Cal {b['calmar']-a['calmar']:+.3f}")
sw = alloc_switch(S0["bondtrend200_bills"])["bills"]
smk = sw[P0["mark"]]; smk = smk[(smk.index >= START) & (smk.index <= FREEZE)]
flips = float((smk.diff().abs() > 0).sum() / ((smk.index[-1] - smk.index[0]).days / 365.25))
print(f"  bond-trend weekly flips per year (IS): {flips:.2f}")
OUT["F"]["bondtrend_flips_per_year"] = flips

# ------------------------------------------------------------------ G. Reality Check on Sharpe & Calmar
print("\nG. White Reality Check on RISK-ADJUSTED metrics (main 12 and switch-only), IS, 63d blocks, 2000 draws")
import importlib.util
MAIN_NAMES = ["static_ief", "static_tlt", "static_shy", "static_bills", "ief50_bills50", "thirds_bills_shy_ief",
              "durA_tnx200_bills", "durB_tnx200_shy", "corr63_bills", "corr126_bills", "bondtrend200_bills", "infl_gsci_bills"]
gsci = P0["gsci"].ffill()
ief_px0 = (1 + P0["ief"].fillna(0)).cumprod()
SIGM = {"durA_tnx200_bills": P0["tnx"] > ma(P0["tnx"], 200), "durB_tnx200_shy": P0["tnx"] > ma(P0["tnx"], 200),
        "corr63_bills": S0["corr63_bills"], "corr126_bills": S0["corr126_bills"],
        "bondtrend200_bills": ief_px0 < ma(ief_px0, 200),
        "infl_gsci_bills": (gsci > ma(gsci, 200)) & (gsci / gsci.shift(252) - 1 > 0)}
ALL = {"static_ief": alloc_static(ief=1), "static_tlt": alloc_static(tlt=1), "static_shy": alloc_static(shy=1),
       "static_bills": alloc_static(bills=1), "ief50_bills50": alloc_static(ief=0.5, bills=0.5),
       "thirds_bills_shy_ief": alloc_static(bills=1/3, shy=1/3, ief=1/3)}
for k, sg in SIGM.items():
    ALL[k] = alloc_switch(sg, short="shy" if k.startswith("durB") else "bills")
OUT["G"] = {}


def rc_metric(R, names, metric, n=2000, block=63, seed=7):
    A = R[names].values
    T, K = A.shape
    b = names.index("static_ief"); keep = [k for k in range(K) if k != b]

    def met(X):
        if metric == "sharpe":
            return X.mean(0) / X.std(0) * np.sqrt(252)
        e = np.cumprod(1 + X, axis=0)
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(0)
        g = e[-1] ** (252 / len(X)) - 1
        return g / np.abs(dd)
    m0 = met(A); d0 = m0[keep] - m0[b]
    stat = d0.max()
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); mx = np.empty(n)
    for i in range(n):
        st_ = rng.integers(0, T, nb)
        ix = (st_[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        mi = met(A[ix]); mx[i] = ((mi[keep] - mi[b]) - d0).max()
    best = [names[k] for k in keep][int(np.argmax(d0))]
    return dict(best=best, best_diff=float(stat), p_value=float((mx >= stat).mean()))


for s in STR:
    R = pd.concat({p: cm_run(P0, s, al) for p, al in ALL.items()}, axis=1)
    R = R[(R.index >= START) & (R.index <= FREEZE)]
    # sanity: these must equal the specialist's series
    sp_cal = {p: SAVED["results"][s][p]["IS_full"]["calmar"] for p in MAIN_NAMES}
    my_cal = {p: C.stats(R[p])["calmar"] for p in MAIN_NAMES}
    assert max(abs(sp_cal[p] - my_cal[p]) for p in MAIN_NAMES) < 1e-5, "policy series do not match specialist"
    OUT["G"][s] = {}
    for metric in ["sharpe", "calmar"]:
        for lab, names in [("main12", MAIN_NAMES), ("switch_only", ["static_ief"] + list(SIGM))]:
            res = rc_metric(R, names, metric)
            OUT["G"][s][f"{metric}/{lab}"] = res
            print(f"  {s} {metric:6s} {lab:12s} best {res['best']:20s} diff {res['best_diff']:+.4f}  RC p = {res['p_value']:.3f}")

# ------------------------------------------------------------------ H. misc claim checks
print("\nH. misc")
OUT["H"] = {}
for s in STR:
    b = SAVED["results"][s]["static_ief"]
    above = [p for p in SAVED["results"][s] if p in MAIN_NAMES + ["carry_inverted_bills", "vote2of4_bills"]
             and p != "static_ief" and SAVED["results"][s][p]["IS_full"]["calmar"] > b["IS_full"]["calmar"]]
    bt = SAVED["results"][s]["bondtrend200_bills"]
    OUT["H"][s] = dict(policies_above_static_ief_fullIS=above,
                       bondtrend_cagr_cost_full_pp=(bt["IS_full"]["cagr"] - b["IS_full"]["cagr"]) * 100,
                       bondtrend_cagr_cost_H1_pp=(bt["IS_H1"]["cagr"] - b["IS_H1"]["cagr"]) * 100,
                       bondtrend_cagr_cost_H2_pp=(bt["IS_H2"]["cagr"] - b["IS_H2"]["cagr"]) * 100)
    neigh_both = [p for p in SAVED["results"][s] if p not in MAIN_NAMES and p != "static_ief"
                  and SAVED["results"][s][p]["IS_H1"]["calmar"] > b["IS_H1"]["calmar"]
                  and SAVED["results"][s][p]["IS_H2"]["calmar"] > b["IS_H2"]["calmar"]]
    OUT["H"][s]["nonmain_variants_beating_ief_in_both_halves"] = {
        p: (round(SAVED["results"][s][p]["IS_full"]["calmar"] - b["IS_full"]["calmar"], 4)) for p in neigh_both}
    print(f"  {s}: full-IS Calmar above static IEF: {above}")
    print(f"  {s}: bond-trend CAGR cost full {OUT['H'][s]['bondtrend_cagr_cost_full_pp']:+.2f}pp  H1 "
          f"{OUT['H'][s]['bondtrend_cagr_cost_H1_pp']:+.2f}pp  H2 {OUT['H'][s]['bondtrend_cagr_cost_H2_pp']:+.2f}pp")
    print(f"  {s}: neighbourhood/lag/extra variants beating static IEF in BOTH halves (full-IS Calmar gain): "
          f"{OUT['H'][s]['nonmain_variants_beating_ief_in_both_halves']}")
# same-day T-bill yield used for the pre-BIL bills return (bills_t = irx_t/100/252 - fee)
pre = P0.index < pd.Timestamp("2007-05-31")
b_same = (P0["rf"] / 252 - 0.0013 / 252)[pre]
b_lag = (P0["rf"].shift(1) / 252 - 0.0013 / 252)[pre]
cg = lambda x: (1 + x[x.index >= START]).prod() ** (252 / len(x[x.index >= START])) - 1
OUT["H"]["bills_same_day_vs_lagged_yield_cagr_pp"] = float((cg(b_same) - cg(b_lag)) * 100)
print(f"  pre-2007 bills: same-day vs prior-day yield CAGR difference {OUT['H']['bills_same_day_vs_lagged_yield_cagr_pp']:+.4f}pp/yr")
# OOS window Oct 7-Nov 17 for all main policies (claim 'every policy -17.2%')
oo = {s: {p: SAVED["results"][s][p]["windows"]["OOS since NDX record 2026-10-06"]["ret"] for p in MAIN_NAMES} for s in STR}
OUT["H"]["oos_since_record"] = oo
print("  Oct7-Nov17 F3 range:", round(min(oo["F3"].values()) * 100, 2), "to", round(max(oo["F3"].values()) * 100, 2))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(OUTDIR, "verify_rate_regime_integrity_checks.json"), "w"), indent=1)
print(f"\ndone [{time.time()-T0:.0f}s]")
