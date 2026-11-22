"""
update_2026_11/rate_regime.py  --  OFF-LEG POLICY FOR RISING-RATE / STOCKS-AND-BONDS-DOWN REGIMES

Question: when the engine is (partly or fully) out of the Nasdaq sleeve, where should the
"rest of the 85%" sit?  The July research used a synthetic 10y (IEF-like).  1994, 2022 and
the Oct-Nov 2026 episode are regimes where yields rose while stocks fell, so duration HURT.
We compare static and regime-switching off-legs on F3 (TQQQ vt .20) and F2 (QLD vt .30),
both with a permanent 15% physical-gold sleeve, using the shared mechanics in common.py.

All switch signals use data up to the day-t close only; common.py samples weights on the
Friday mark and applies them from t+1 (lag=1), exactly like the EMA-100 gate itself.
No signal uses gold.  Nothing is selected with post-freeze (after 2026-07-24) data.

Outputs: update_2026_11/rate_regime.json (full numbers) + console summary.
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)
import common as C

T0 = time.time()
P = C.panel()
idx = P.index
FREEZE = C.FREEZE
LAST = idx[-1]
H1 = (pd.Timestamp("1988-01-01"), pd.Timestamp("2006-12-31"))
H2 = (pd.Timestamp("2007-01-01"), FREEZE)
OOS_A = pd.Timestamp("2026-07-25")
SEL_COST = (0.07, 0.15)                       # measured selection cost range (Calmar), July research
OUT = os.path.join(HERE, "rate_regime.json")

VARIANT_LOG = []                              # every backtested configuration is appended here


# ============================================================ signals (causal)
def ma(s, n):
    return s.rolling(n, min_periods=n).mean()


tnx = P["tnx"]
ief_px = (1 + P["ief"].fillna(0)).cumprod()  # total-return index of the spliced IEF sleeve
gsci = P["gsci"].ffill()


def sig_rate_up(n=200):            # 10y yield above its n-day MA  -> rates in an up-trend
    return tnx > ma(tnx, n)


def sig_bond_down(n=200):          # IEF total-return index below its n-day MA -> bond down-trend
    return ief_px < ma(ief_px, n)


def corr_series(n=63):
    return P["ndx_ret"].rolling(n, min_periods=n).corr(P["ief"])


def sig_corr_pos(n=63, thr=0.0):   # stocks and bonds moving together -> bonds are not a hedge
    return corr_series(n) > thr


def sig_infl(n=200, mom=252):      # commodity up-trend AND positive 12m momentum -> inflation regime
    return (gsci > ma(gsci, n)) & (gsci / gsci.shift(mom) - 1 > 0)


def sig_inverted():                # EXTRA: carry - bills yield more than the 10y
    return P["irx"] > P["tnx"]


# ============================================================ off-leg allocation frames
def static(**kw):
    return pd.DataFrame({k: pd.Series(float(v), index=idx) for k, v in kw.items()})


def switch(cond, short="bills", long="ief", extra_lag=0):
    c = cond.reindex(idx).fillna(False).astype(float)
    if extra_lag:
        c = c.shift(extra_lag).fillna(0.0)
    df = pd.DataFrame(0.0, index=idx, columns=[long, short])
    df[short] = c
    df[long] = 1.0 - c
    return df


SIGNALS = {   # the switch signals that drive the six pre-specified switch policies
    "durA_tnx200_bills": sig_rate_up(200),
    "durB_tnx200_shy": sig_rate_up(200),
    "corr63_bills": sig_corr_pos(63),
    "corr126_bills": sig_corr_pos(126),
    "bondtrend200_bills": sig_bond_down(200),
    "infl_gsci_bills": sig_infl(200, 252),
}
SHORT_LEG = {"durB_tnx200_shy": "shy"}

MAIN = {   # the 12 pre-specified policies (the task list), static IEF = baseline
    "static_ief": static(ief=1),
    "static_tlt": static(tlt=1),
    "static_shy": static(shy=1),
    "static_bills": static(bills=1),
    "ief50_bills50": static(ief=0.5, bills=0.5),
    "thirds_bills_shy_ief": static(bills=1 / 3, shy=1 / 3, ief=1 / 3),
}
for k, s in SIGNALS.items():
    MAIN[k] = switch(s, SHORT_LEG.get(k, "bills"))
MAIN_ORDER = list(MAIN.keys())

# parameter neighbourhood of every switch rule (robustness only - NOT used to pick a winner)
NEIGH = {}
for n in (100, 150, 250, 300):
    NEIGH[f"durA_tnx{n}_bills"] = switch(sig_rate_up(n))
    NEIGH[f"bondtrend{n}_bills"] = switch(sig_bond_down(n))
for n in (42, 252):
    NEIGH[f"corr{n}_bills"] = switch(sig_corr_pos(n))
for n in (100, 300):
    NEIGH[f"infl_gsci{n}_bills"] = switch(sig_infl(n, 252))

# one extra week of signal delay (does the edge depend on precise timing?)
LAGGED = {f"{k}__lag5": switch(s, SHORT_LEG.get(k, "bills"), extra_lag=5) for k, s in SIGNALS.items()}

# two extra, economically-motivated ideas (clearly labelled extras, counted)
vote = (sig_rate_up(200).astype(int) + sig_corr_pos(63).astype(int)
        + sig_bond_down(200).astype(int) + sig_infl(200).astype(int)) >= 2
EXTRAS = {"carry_inverted_bills": switch(sig_inverted()),
          "vote2of4_bills": switch(vote)}

STRATS = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}


def weights(strat, ow):
    return C.strategy_weights(P, **STRATS[strat], gold="gold_phys", gold_w=0.15, off_weights=ow)


def run(strat, ow, name, group):
    w = weights(strat, ow)
    r = C.backtest(P, w, cost=C.COST, lag=1, mark="mark")
    VARIANT_LOG.append(dict(strategy=strat, policy=name, group=group))
    return r, w


# ============================================================ metrics
riskoff_mark = None
ab = P["above"]
mk = P["mark"]
for d in idx[::-1]:                                   # last Friday mark that turned the gate off
    if mk.loc[d] and not ab.loc[d]:
        riskoff_mark = d
    elif mk.loc[d] and ab.loc[d]:
        break
EXTRA_WIN = {
    "2013 taper tantrum": ("2013-05-01", "2013-09-05"),
    "2023 Aug-Oct 10y->5%": ("2023-07-31", "2023-10-27"),
    "OOS pre-peak 2026-07-25..10-06": ("2026-07-25", "2026-10-06"),
    "OOS since NDX record 2026-10-06": ("2026-10-07", str(LAST.date())),
    "OOS Nov 5-17 yields back up": ("2026-11-06", str(LAST.date())),
    "OOS since risk-off (applied)": (str(idx[idx > riskoff_mark][0].date()), str(LAST.date())),
}
WINDOWS = {**C.CRISES, **EXTRA_WIN}
DEFL = ["1998 LTCM", "dot-com 2000-02", "GFC 2007-09", "COVID 2020"]
INFL = ["1994 bond massacre", "2022 inflation"]


def window_dd(r, a, b):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    if not len(s):
        return np.nan
    e = np.concatenate([[1.0], (1 + s).cumprod().values])
    return float((e / np.maximum.accumulate(e) - 1).min())


def dd_location(r):
    s = r[(r.index >= C.START) & (r.index <= FREEZE)].dropna()
    e = (1 + s).cumprod()
    dd = e / e.cummax() - 1
    trough = dd.idxmin()
    peak = e.loc[:trough].idxmax()
    return dict(peak=str(peak.date()), trough=str(trough.date()), maxdd=float(dd.min()))


def evaluate(r):
    rr = r[r.index <= FREEZE]
    return dict(
        IS_maxdd_location=dd_location(r),
        IS_H1_maxdd_location=dd_location(r[r.index <= H1[1]]),
        IS_H2_maxdd_location=dd_location(r[r.index >= H2[0]]),
        IS_full=C.stats(rr),
        IS_H1=C.stats(rr, H1[0], H1[1]),
        IS_H2=C.stats(rr, H2[0], FREEZE),
        IS_plus_OOS_to_latest=C.stats(r),
        OOS=dict(ret=C.period_return(r, OOS_A, LAST), maxdd=window_dd(r, OOS_A, LAST)),
        windows={k: dict(ret=C.period_return(r, a, b), maxdd=window_dd(r, a, b)) for k, (a, b) in WINDOWS.items()},
        decades_IS=C.by_decade(rr),
    )


def off_diag(w, ow_name, sig=None):
    h = C.held_weights(P, w)
    h = h[(h.index >= C.START) & (h.index <= FREEZE)]
    off_cols = [c for c in h.columns if c not in ("tqqq", "qld", "gold_phys")]
    out = dict(avg_offleg_weight=float(h[off_cols].sum(axis=1).mean()))
    for c in off_cols:
        out[f"avg_{c}"] = float(h[c].mean())
    if sig is not None:
        s = sig.reindex(idx).fillna(False).astype(int)[P["mark"]]
        s = s[(s.index >= C.START) & (s.index <= FREEZE)]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        out["share_of_weeks_short"] = float(s.mean())
        out["flips_per_year"] = float((s.diff().abs() > 0).sum() / yrs)
    return out


# ============================================================ run all variants
RES = {s: {} for s in STRATS}     # strat -> policy -> metrics
RET = {s: {} for s in STRATS}     # strat -> policy -> daily return series
GROUPS = [("main", MAIN), ("neighbourhood", NEIGH), ("lag5", LAGGED), ("extra", EXTRAS)]
for strat in STRATS:
    for gname, pols in GROUPS:
        for name, ow in pols.items():
            r, w = run(strat, ow, name, gname)
            RET[strat][name] = r
            m = evaluate(r)
            base_sig = SIGNALS.get(name.replace("__lag5", ""))
            if gname == "neighbourhood" or name in EXTRAS:
                base_sig = (ow.iloc[:, 1] > 0.5)
            m["offleg"] = off_diag(w, name, base_sig)
            m["group"] = gname
            RES[strat][name] = m
print(f"[{time.time()-T0:5.1f}s] ran {len(VARIANT_LOG)} variants", flush=True)

# ------------------------------------------------ equal-drawdown risk dial (is a DD cut just less risk?)
# static IEF with a lower vol target; compare CAGR at the same in-sample maxDD as each switch rule.
DIAL = {s: {} for s in STRATS}
for strat, grid in [("F3", (0.16, 0.17, 0.18, 0.19)), ("F2", (0.24, 0.26, 0.27, 0.28))]:
    for vt in grid:
        w = C.strategy_weights(P, lev=STRATS[strat]["lev"], voltgt=vt, gold="gold_phys", gold_w=0.15,
                               off_weights=MAIN["static_ief"])
        r = C.backtest(P, w, cost=C.COST, lag=1, mark="mark")
        VARIANT_LOG.append(dict(strategy=strat, policy=f"static_ief_vt{vt}", group="risk_dial"))
        rr = r[r.index <= FREEZE]
        DIAL[strat][f"{vt:.2f}"] = dict(IS_full=C.stats(rr), IS_H1=C.stats(rr, H1[0], H1[1]),
                                        IS_H2=C.stats(rr, H2[0], FREEZE),
                                        w2022=C.period_return(r, *C.CRISES["2022 inflation"]))
    base_pts = [(RES[strat]["static_ief"]["IS_full"]["maxdd"], RES[strat]["static_ief"]["IS_full"]["cagr"])]
    pts = sorted([(v["IS_full"]["maxdd"], v["IS_full"]["cagr"]) for v in DIAL[strat].values()] + base_pts)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    DIAL[strat]["_equal_dd_cagr"] = {}
    for p in MAIN_ORDER + list(EXTRAS):
        dd = RES[strat][p]["IS_full"]["maxdd"]
        cg = float(np.interp(dd, xs, ys)) if xs.min() <= dd <= xs.max() else None
        DIAL[strat]["_equal_dd_cagr"][p] = dict(policy_maxdd=dd, policy_cagr=RES[strat][p]["IS_full"]["cagr"],
                                                dial_cagr_at_same_dd=cg,
                                                excess_vs_dial=(RES[strat][p]["IS_full"]["cagr"] - cg) if cg is not None else None)

# cash-mix dial: a x (static-IEF strategy) + (1-a) x bills, rebalanced on the same weekly marks.
# This is the 'plain risk dial at equal drawdown' benchmark from the July meta-lessons.
CASHMIX = {s: {} for s in STRATS}
for strat in STRATS:
    w0 = weights(strat, MAIN["static_ief"])
    pts = []
    for a in np.round(np.arange(0.80, 1.0001, 0.02), 2):
        w = w0 * a
        w["bills"] = w.get("bills", 0.0) + (1 - a)
        r = C.backtest(P, w, cost=C.COST, lag=1, mark="mark")
        VARIANT_LOG.append(dict(strategy=strat, policy=f"cashmix_{a:.2f}", group="cash_mix_dial"))
        s_ = C.stats(r[r.index <= FREEZE])
        CASHMIX[strat][f"{a:.2f}"] = s_
        pts.append((s_["maxdd"], s_["cagr"], s_["sharpe"]))
    pts.sort()
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts]); zs = np.array([p[2] for p in pts])
    CASHMIX[strat]["_equal_dd"] = {}
    for p in MAIN_ORDER + list(EXTRAS):
        dd = RES[strat][p]["IS_full"]["maxdd"]
        ok = xs.min() <= dd <= xs.max()
        cg = float(np.interp(dd, xs, ys)) if ok else None
        CASHMIX[strat]["_equal_dd"][p] = dict(
            policy_maxdd=dd, policy_cagr=RES[strat][p]["IS_full"]["cagr"],
            cashmix_cagr_at_same_dd=cg,
            excess_cagr_vs_cashmix=(RES[strat][p]["IS_full"]["cagr"] - cg) if ok else None,
            note=None if ok else "policy maxDD not inside the dial range (dial only reduces risk)")


# ============================================================ walk-forward selection
def rank_on(strat, pols, a, b, key="calmar"):
    sc = {p: C.stats(RET[strat][p][RET[strat][p].index <= FREEZE], a, b)[key] for p in pols}
    return sorted(sc, key=lambda p: -sc[p]), sc


WF = {s: {} for s in STRATS}
for strat in STRATS:
    for tr, te, lab in [(H1, H2, "pick_H1_test_H2"), (H2, H1, "pick_H2_test_H1")]:
        order_tr, sc_tr = rank_on(strat, MAIN_ORDER, *tr)
        order_te, sc_te = rank_on(strat, MAIN_ORDER, *te)
        pick = order_tr[0]
        WF[strat][lab] = dict(
            picked=pick, train_calmar=sc_tr[pick], test_calmar=sc_te[pick],
            test_rank_of_12=order_te.index(pick) + 1,
            test_static_ief_calmar=sc_te["static_ief"],
            test_rank_static_ief=order_te.index("static_ief") + 1,
            test_best=order_te[0], test_best_calmar=sc_te[order_te[0]],
            train_order=order_tr, test_order=order_te,
            train_scores=sc_tr, test_scores=sc_te)


def expanding_wf(strat, pool, first_sel_year=1995, key="calmar"):
    """at each Dec-31 pick the best-Calmar policy on 1988..Y (expanding) and run it in Y+1."""
    picks = {}
    for Y in range(first_sel_year, 2026):
        b = pd.Timestamp(f"{Y}-12-31")
        sc = {p: C.stats(RET[strat][p], C.START, b)[key] for p in pool}
        picks[Y + 1] = max(sc, key=lambda p: sc[p])
    pols = {**MAIN, **NEIGH, **EXTRAS}
    cols = sorted({c for p in pool for c in pols[p].columns} | {"ief"})
    ow = pd.DataFrame(0.0, index=idx, columns=cols)
    ow["ief"] = 1.0                                  # before the first selection year
    for yr, p in picks.items():
        m = idx.year == yr
        ow.loc[m, :] = pols[p].reindex(columns=cols, fill_value=0.0).loc[m].values
    r, _ = run(strat, ow, f"WF_expanding_{len(pool)}pool", "walkforward")
    a = pd.Timestamp(f"{first_sel_year+1}-01-01")
    rr = r[r.index <= FREEZE]
    base = RET[strat]["static_ief"]
    base_rr = base[base.index <= FREEZE]
    # what an in-sample (hindsight) pick would have shown on the same window
    is_order, is_sc = rank_on(strat, pool, C.START, FREEZE)
    hind = RET[strat][is_order[0]]
    hind_rr = hind[hind.index <= FREEZE]
    return dict(
        picks={str(k): v for k, v in picks.items()},
        n_distinct_picks=len(set(picks.values())),
        window=f"{a.date()}..{FREEZE.date()}",
        wf_realised=C.stats(rr, a, FREEZE), static_ief=C.stats(base_rr, a, FREEZE),
        hindsight_best=is_order[0], hindsight_best_stats=C.stats(hind_rr, a, FREEZE),
        wf_OOS_ret=C.period_return(r, OOS_A, LAST), static_ief_OOS_ret=C.period_return(base, OOS_A, LAST),
        wf_windows={k: C.period_return(r, *WINDOWS[k]) for k in DEFL + INFL},
    ), r


for strat in STRATS:
    WF[strat]["expanding_main12"], _ = expanding_wf(strat, MAIN_ORDER)
    pool_all = MAIN_ORDER + list(NEIGH) + list(EXTRAS)
    WF[strat]["expanding_all"], _ = expanding_wf(strat, pool_all)
print(f"[{time.time()-T0:5.1f}s] walk-forward done", flush=True)


# ============================================================ bootstrap vs static IEF (in-sample only)
def boot_all(strat, pols, n=2000, block=63, seed=0):
    R = pd.concat({p: RET[strat][p] for p in pols}, axis=1)
    R = R[(R.index >= C.START) & (R.index <= FREEZE)].dropna()
    A = R.values
    T, K = A.shape
    b = pols.index("static_ief")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    cg = np.empty((n, K)); cl = np.empty((n, K)); dd_ = np.empty((n, K))
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        e = np.cumprod(1 + A[ix], axis=0)
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        g = e[-1] ** (252 / T) - 1
        cal = g / np.abs(dd)
        cg[i] = g - g[b]; cl[i] = cal - cal[b]; dd_[i] = dd - dd[b]
    out = {}
    for k, p in enumerate(pols):
        if p == "static_ief":
            continue
        out[p] = dict(
            cagr_diff=dict(mean=float(cg[:, k].mean()), lo=float(np.percentile(cg[:, k], 2.5)),
                           hi=float(np.percentile(cg[:, k], 97.5)), p_le0=float((cg[:, k] <= 0).mean())),
            calmar_diff=dict(mean=float(cl[:, k].mean()), lo=float(np.percentile(cl[:, k], 2.5)),
                             hi=float(np.percentile(cl[:, k], 97.5)), p_le0=float((cl[:, k] <= 0).mean()),
                             p_gt_selcost_lo=float((cl[:, k] > SEL_COST[0]).mean())),
            maxdd_diff=dict(mean=float(dd_[:, k].mean()), lo=float(np.percentile(dd_[:, k], 2.5)),
                            hi=float(np.percentile(dd_[:, k], 97.5))),
        )
        # cross-check with the shared helper (same seed/block -> identical CAGR draws)
        rp = RET[strat][p][RET[strat][p].index <= FREEZE]
        rb = RET[strat]["static_ief"][RET[strat]["static_ief"].index <= FREEZE]
        out[p]["common_block_bootstrap_cagr_diff"] = C.block_bootstrap_cagr_diff(rp, rb)
    return out


def reality_check(strat, pols, n=2000, block=63, seed=1):
    """White (2000) Reality Check on mean daily log-return differential vs static IEF:
    is the BEST policy's edge larger than what the best of K no-edge policies would show?"""
    R = pd.concat({p: RET[strat][p] for p in pols}, axis=1)
    R = R[(R.index >= C.START) & (R.index <= FREEZE)].dropna()
    L = np.log1p(R.values)
    b = pols.index("static_ief")
    keep = [k for k in range(len(pols)) if k != b]
    D = L[:, keep] - L[:, [b]]
    T = len(D)
    dbar = D.mean(0)
    stat = np.sqrt(T) * dbar.max()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    mx = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        mx[i] = np.sqrt(T) * (D[ix].mean(0) - dbar).max()
    best = [pols[k] for k in keep][int(np.argmax(dbar))]
    return dict(best_by_mean_logret=best, best_ann_logret_diff=float(dbar.max() * 252),
                p_value=float((mx >= stat).mean()), n_candidates=len(keep))


BOOT, RC = {}, {}
for strat in STRATS:
    BOOT[strat] = boot_all(strat, MAIN_ORDER + list(EXTRAS))
    RC[strat] = dict(main12=reality_check(strat, MAIN_ORDER),
                     switch_rules_only=reality_check(strat, ["static_ief"] + list(SIGNALS)),
                     whole_search_excl_lag=reality_check(strat, MAIN_ORDER + list(NEIGH) + list(EXTRAS)))
print(f"[{time.time()-T0:5.1f}s] bootstrap done", flush=True)


# ============================================================ regime attribution (monthly, IS)
def monthly(r):
    return (1 + r).resample("ME").prod() - 1


ndx_m = monthly(P["ndx_ret"]); ief_m = monthly(P["ief"])
tnx_m = P["tnx"].resample("ME").last()
dy6 = tnx_m.diff(6)
mi = ndx_m.index[(ndx_m.index >= "1988-01-31") & (ndx_m.index <= "2026-06-30")]
BUCKETS = {
    "stocks DOWN & bonds DOWN": (ndx_m < 0) & (ief_m < 0),
    "stocks DOWN & bonds UP": (ndx_m < 0) & (ief_m >= 0),
    "stocks UP & bonds DOWN": (ndx_m >= 0) & (ief_m < 0),
    "stocks UP & bonds UP": (ndx_m >= 0) & (ief_m >= 0),
    "10y +50bp or more over 6m": dy6 > 0.5,
    "10y within +-50bp over 6m": dy6.abs() <= 0.5,
    "10y -50bp or more over 6m": dy6 < -0.5,
}
ATTR = {s: {} for s in STRATS}
for strat in STRATS:
    lb = np.log1p(monthly(RET[strat]["static_ief"]))
    for p in MAIN_ORDER:
        if p == "static_ief":
            continue
        dl = (np.log1p(monthly(RET[strat][p])) - lb).reindex(mi)
        ATTR[strat][p] = {k: dict(months=int(m.reindex(mi).fillna(False).sum()),
                                  sum_log_excess_pct=float(dl[m.reindex(mi).fillna(False)].sum() * 100),
                                  mean_bp_per_month=float(dl[m.reindex(mi).fillna(False)].mean() * 1e4))
                          for k, m in BUCKETS.items()}


# ============================================================ instrument context
INSTR = {}
for k, (a, b) in WINDOWS.items():
    INSTR[k] = {c: C.period_return(P[c], a, b) for c in ["ndx_ret", "qqq", "qld", "tqqq", "ief", "tlt", "shy", "bills", "gold_phys"]}
    _pre = P["tnx"].loc[:pd.Timestamp(a) - pd.Timedelta(days=1)]          # close before the window starts
    INSTR[k]["tnx_change_pp"] = float(P["tnx"].loc[a:b].iloc[-1] - _pre.iloc[-1]) if len(P["tnx"].loc[a:b]) else np.nan


# ============================================================ what each rule says TODAY
last_mark = idx[P["mark"]][-1]
corr63 = corr_series(63); corr126 = corr_series(126)
def reading(d):
    d = pd.Timestamp(d)
    return dict(
        date=str(d.date()),
        tnx=float(tnx.loc[d]), tnx_ma200=float(ma(tnx, 200).loc[d]),
        tnx_minus_ma200_bp=float((tnx.loc[d] - ma(tnx, 200).loc[d]) * 100),
        corr63=float(corr63.loc[d]), corr126=float(corr126.loc[d]),
        ief_px_vs_ma200_pct=float((ief_px.loc[d] / ma(ief_px, 200).loc[d] - 1) * 100),
        gsci=float(gsci.loc[d]), gsci_vs_ma200_pct=float((gsci.loc[d] / ma(gsci, 200).loc[d] - 1) * 100),
        gsci_12m_mom_pct=float((gsci.loc[d] / gsci.shift(252).loc[d] - 1) * 100),
        irx=float(P["irx"].loc[d]), curve_pp=float(P["curve"].loc[d]),
        ndx_above_ema100=bool(P["above"].loc[d]),
        signals={k: ("SHORT (" + SHORT_LEG.get(k, "bills") + ")") if bool(s.loc[d]) else "LONG (ief)"
                 for k, s in {**SIGNALS, "carry_inverted_bills": sig_inverted(), "vote2of4_bills": vote}.items()},
    )


def last_flip(s):
    s = s.fillna(False).astype(bool)
    ch = s.ne(s.shift())
    ch.iloc[0] = False
    d = ch[ch].index
    return str(d[-1].date()) if len(d) else None


NEIGH_SIG = {}
for n in (100, 150, 250, 300):
    NEIGH_SIG[f"durA_tnx{n}_bills"] = sig_rate_up(n)
    NEIGH_SIG[f"bondtrend{n}_bills"] = sig_bond_down(n)
for n in (42, 252):
    NEIGH_SIG[f"corr{n}_bills"] = sig_corr_pos(n)
for n in (100, 300):
    NEIGH_SIG[f"infl_gsci{n}_bills"] = sig_infl(n, 252)
TODAY_NEIGH = {k: dict(last_mark="SHORT" if bool(s.loc[last_mark]) else "IEF",
                       latest="SHORT" if bool(s.loc[LAST]) else "IEF", last_flip=last_flip(s))
               for k, s in NEIGH_SIG.items()}
TODAY_NEIGH["_ief_px_vs_ma_pct_latest"] = {n: float((ief_px.loc[LAST] / ma(ief_px, n).loc[LAST] - 1) * 100)
                                           for n in (100, 150, 200, 250, 300)}
TODAY_NEIGH["_tnx_minus_ma_bp_latest"] = {n: float((tnx.loc[LAST] - ma(tnx, n).loc[LAST]) * 100)
                                          for n in (100, 150, 200, 250, 300)}
TODAY = dict(as_of_last_mark=reading(last_mark), as_of_latest=reading(LAST), neighbourhood_rules=TODAY_NEIGH,
             walkforward_current_pick={s: {k: WF[s][k]["picks"]["2026"] for k in ["expanding_main12", "expanding_all"]}
                                       for s in STRATS},
             last_flip_date={k: last_flip(s) for k, s in {**SIGNALS, "carry_inverted_bills": sig_inverted(),
                                                            "vote2of4_bills": vote}.items()},
             target_weights_latest={})
for strat in STRATS:
    TODAY["target_weights_latest"][strat] = {}
    for p in MAIN_ORDER + list(EXTRAS):
        w = weights(strat, {**MAIN, **EXTRAS}[p])
        TODAY["target_weights_latest"][strat][p] = {c: round(float(v), 4) for c, v in w.iloc[-1].items() if abs(v) > 1e-9}
# recent monthly path of the switch inputs (context for the current episode)
path = pd.DataFrame({"tnx": tnx, "tnx_ma200": ma(tnx, 200), "corr63": corr63, "corr126": corr126,
                     "ief_vs_ma200_pct": (ief_px / ma(ief_px, 200) - 1) * 100,
                     "gsci_vs_ma200_pct": (gsci / ma(gsci, 200) - 1) * 100,
                     "gsci_12m_pct": (gsci / gsci.shift(252) - 1) * 100})
TODAY["recent_path_weekly"] = {str(d.date()): {k: round(float(v), 3) for k, v in row.items()}
                               for d, row in path[P["mark"]].loc["2026-05-01":].iterrows()}
TODAY["recent_path_weekly"][str(LAST.date())] = {k: round(float(v), 3) for k, v in path.loc[LAST].items()}


# ============================================================ verdicts
VERDICT = {s: {} for s in STRATS}
for strat in STRATS:
    b = RES[strat]["static_ief"]
    for p in MAIN_ORDER + list(EXTRAS):
        m = RES[strat][p]
        d_full = m["IS_full"]["calmar"] - b["IS_full"]["calmar"]
        d_h1 = m["IS_H1"]["calmar"] - b["IS_H1"]["calmar"]
        d_h2 = m["IS_H2"]["calmar"] - b["IS_H2"]["calmar"]
        bt = BOOT[strat].get(p)
        VERDICT[strat][p] = dict(
            calmar_diff_full=d_full, calmar_diff_H1=d_h1, calmar_diff_H2=d_h2,
            cagr_diff_full=m["IS_full"]["cagr"] - b["IS_full"]["cagr"],
            maxdd_diff_full=m["IS_full"]["maxdd"] - b["IS_full"]["maxdd"],
            both_halves=bool(d_h1 > 0 and d_h2 > 0),
            beats_selcost_lo=bool(d_full > SEL_COST[0]),
            beats_selcost_hi=bool(d_full > SEL_COST[1]),
            boot_calmar_p_le0=(bt["calmar_diff"]["p_le0"] if bt else None),
            robust=bool(d_h1 > 0 and d_h2 > 0 and d_full > SEL_COST[0] and bt is not None
                        and bt["calmar_diff"]["p_le0"] < 0.10),
        ) if p != "static_ief" else dict(baseline=True)


# ============================================================ save
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
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


n_var = len(VARIANT_LOG)
summary = dict(
    meta=dict(script="update_2026_11/rate_regime.py", data_end=str(LAST.date()), freeze=str(FREEZE.date()),
              in_sample=f"{C.START.date()}..{FREEZE.date()}", H1=f"{H1[0].date()}..{H1[1].date()}",
              H2=f"{H2[0].date()}..{H2[1].date()}", OOS=f"{OOS_A.date()}..{LAST.date()}",
              riskoff_mark=str(riskoff_mark.date()), last_mark=str(last_mark.date()),
              selection_cost_calmar=SEL_COST, cost_bps=C.COST * 1e4,
              variants_tested=n_var,
              variant_breakdown={g: sum(1 for v in VARIANT_LOG if v["group"] == g)
                                 for g in sorted({v["group"] for v in VARIANT_LOG})},
              robustness_criteria="robust = Calmar beats static IEF in BOTH halves AND by > 0.07 full-IS AND "
                                  "block-bootstrap P(Calmar diff <= 0) < 0.10"),
    results=RES, walkforward=WF, bootstrap_vs_static_ief=BOOT, reality_check=RC, risk_dial_equal_dd=DIAL,
    cash_mix_dial_equal_dd=CASHMIX,
    regime_attribution_monthly_IS=ATTR, instrument_returns_in_windows=INSTR, today=TODAY,
    verdict=VERDICT, variant_log=VARIANT_LOG,
)
json.dump(clean(summary), open(OUT, "w"), indent=1)


# ============================================================ console summary
def pc(x):
    return "   n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:6.1f}"


for strat in STRATS:
    print(f"\n==================== {strat}  ({STRATS[strat]})  + 15% gold_phys ====================")
    print(f"{'policy':24s} | IS 1988-2026/07: CAGR  maxDD  Cal  Shrp | H1 Cal  H2 Cal | 1994   2022  dotcom  GFC  COVID 1998 | OOS   OOS-dd | off%  short%")
    for p in MAIN_ORDER + list(EXTRAS):
        m = RES[strat][p]; f = m["IS_full"]; W = m["windows"]
        o = m["offleg"]
        print(f"{p:24s} | {pc(f['cagr'])} {pc(f['maxdd'])} {f['calmar']:5.2f} {f['sharpe']:5.2f} | "
              f"{m['IS_H1']['calmar']:5.2f}  {m['IS_H2']['calmar']:5.2f} | "
              f"{pc(W['1994 bond massacre']['ret'])}{pc(W['2022 inflation']['ret'])}{pc(W['dot-com 2000-02']['ret'])}"
              f"{pc(W['GFC 2007-09']['ret'])}{pc(W['COVID 2020']['ret'])}{pc(W['1998 LTCM']['ret'])} | "
              f"{pc(m['OOS']['ret'])} {pc(m['OOS']['maxdd'])} | {o['avg_offleg_weight']*100:4.0f} "
              f"{(o.get('share_of_weeks_short', np.nan) or 0)*100:5.0f}")
    print("-- neighbourhood / lag5 (Calmar full, H1, H2, 2022, dot-com, GFC):")
    for p in list(NEIGH) + list(LAGGED):
        m = RES[strat][p]
        print(f"   {p:28s} {m['IS_full']['calmar']:5.2f} {m['IS_H1']['calmar']:5.2f} {m['IS_H2']['calmar']:5.2f} "
              f"{pc(m['windows']['2022 inflation']['ret'])} {pc(m['windows']['dot-com 2000-02']['ret'])} "
              f"{pc(m['windows']['GFC 2007-09']['ret'])}")
    print("-- bootstrap vs static IEF (IS): CAGR diff [95%], Calmar diff [95%], P(Cal<=0)")
    for p, bt in BOOT[strat].items():
        c, k = bt["cagr_diff"], bt["calmar_diff"]
        print(f"   {p:24s} {c['mean']*100:+5.2f} [{c['lo']*100:+5.2f},{c['hi']*100:+5.2f}]  "
              f"{k['mean']:+5.2f} [{k['lo']:+5.2f},{k['hi']:+5.2f}]  p={k['p_le0']:.2f}")
    print("-- walk-forward:")
    for lab in ["pick_H1_test_H2", "pick_H2_test_H1"]:
        x = WF[strat][lab]
        print(f"   {lab}: picked {x['picked']} (train Cal {x['train_calmar']:.2f}) -> test Cal {x['test_calmar']:.2f}"
              f" rank {x['test_rank_of_12']}/12 ; static IEF test Cal {x['test_static_ief_calmar']:.2f} rank {x['test_rank_static_ief']}")
    for lab in ["expanding_main12", "expanding_all"]:
        x = WF[strat][lab]
        print(f"   {lab}: {x['window']} WF {C.fmt(x['wf_realised'])} | static IEF {C.fmt(x['static_ief'])} | "
              f"hindsight {x['hindsight_best']} Cal {x['hindsight_best_stats']['calmar']:.2f} | distinct picks {x['n_distinct_picks']}")
    print("-- reality check:", RC[strat])
    print("-- maxDD location (IS full / H1 / H2):")
    for p in ["static_ief", "static_tlt", "static_bills", "ief50_bills50", "durA_tnx200_bills", "bondtrend200_bills"]:
        m = RES[strat][p]
        print(f"   {p:22s} " + "  ".join(f"{m[k]['maxdd']*100:6.1f}% {m[k]['peak']}->{m[k]['trough']}"
                                         for k in ["IS_maxdd_location", "IS_H1_maxdd_location", "IS_H2_maxdd_location"]))
    print("-- OOS sub-windows (%), rate-shock analogs:")
    for p in MAIN_ORDER:
        W = RES[strat][p]["windows"]
        print(f"   {p:22s} " + " ".join(f"{k[:18]}:{pc(W[k]['ret'])}" for k in
              ["OOS pre-peak 2026-07-25..10-06", "OOS since NDX record 2026-10-06", "OOS Nov 5-17 yields back up",
               "OOS since risk-off (applied)", "2013 taper tantrum", "2023 Aug-Oct 10y->5%"]))
    print("-- risk dial (static IEF, lower vol target):",
          {k: (round(v['IS_full']['cagr']*100, 2), round(v['IS_full']['maxdd']*100, 1), round(v['IS_full']['calmar'], 2))
           for k, v in DIAL[strat].items() if not k.startswith('_')})
    for p in ["static_bills", "ief50_bills50", "durA_tnx200_bills", "corr126_bills", "bondtrend200_bills", "infl_gsci_bills"]:
        x = DIAL[strat]["_equal_dd_cagr"][p]
        print(f"   equal-DD {p:22s} policy CAGR {x['policy_cagr']*100:5.2f}% @DD {x['policy_maxdd']*100:5.1f}%  "
              f"dial CAGR {('%5.2f%%' % (x['dial_cagr_at_same_dd']*100)) if x['dial_cagr_at_same_dd'] is not None else '  n/a'}")
    for p in ["ief50_bills50", "thirds_bills_shy_ief", "durA_tnx200_bills", "durB_tnx200_shy", "corr126_bills",
              "bondtrend200_bills", "infl_gsci_bills", "static_shy"]:
        x = CASHMIX[strat]["_equal_dd"][p]
        if x["cashmix_cagr_at_same_dd"] is not None:
            print(f"   cash-mix equal-DD {p:22s} policy {x['policy_cagr']*100:5.2f}% vs dial {x['cashmix_cagr_at_same_dd']*100:5.2f}% "
                  f"@DD {x['policy_maxdd']*100:5.1f}%  -> excess {x['excess_cagr_vs_cashmix']*100:+5.2f}%/yr")
        else:
            print(f"   cash-mix equal-DD {p:22s} policy DD {x['policy_maxdd']*100:5.1f}% (not lower than static IEF)")
    print("-- expanding WF picks:", WF[strat]["expanding_main12"]["picks"])
print("\ninstrument returns in OOS sub-windows (%):")
for k in ["OOS Jul-Nov 2026", "OOS pre-peak 2026-07-25..10-06", "OOS since NDX record 2026-10-06",
          "OOS Nov 5-17 yields back up", "OOS since risk-off (applied)"]:
    v = INSTR[k]
    print(f"   {k:34s} " + " ".join(f"{c}:{v[c]*100:6.2f}" for c in ["ndx_ret", "tqqq", "qld", "ief", "tlt", "shy", "bills", "gold_phys"]
                                    if c in v) + f"  dTNX {v['tnx_change_pp']:+.3f}pp")
print("\nTODAY (last mark", last_mark.date(), "):", json.dumps(clean(TODAY["as_of_last_mark"]), indent=0)[:1500])
print("TODAY (latest numbers):", {k: v for k, v in clean(TODAY["as_of_latest"]).items() if k != "signals"})
print("TODAY (latest):", json.dumps(clean(TODAY["as_of_latest"]["signals"])))
print("last flips:", TODAY["last_flip_date"])
print("neighbourhood rules today:", json.dumps(clean(TODAY["neighbourhood_rules"])))
print("walk-forward current (2026) picks:", TODAY["walkforward_current_pick"])
print(f"\nvariants_tested = {n_var}   {summary['meta']['variant_breakdown']}")
print(f"saved {OUT}   [{time.time()-T0:.1f}s]")
