"""
ROUND 2 - FINAL CANDIDATE TABLE ON THE TIMING-LUCK-FREE BASIS, WITH IMPLEMENTATION COSTS
KEY = candidates_tranche.  Writes update_2026_11/r2_candidates_tranche.json.

PRE-REGISTERED DESIGN (fixed here before any number was computed; nothing below is selected or tuned):
  Configs (6)  tiers F1 = QQQ vt .20, F2 = QLD vt .30, F3 = TQQQ vt .20 (July rules, common.strategy_weights)
               x off-leg {IEF ; 50/50 IEF + bills via off_weights}; permanent 15% gold_phys.
  Bases (4)    fri     = Friday book (P.mark), lag 1, 3 bps                       (July convention)
               tranche = 5-weekday tranche (mean of the Mon..Fri books), lag 1, 3 bps
               us      = tranche, lag 1, 5 bps                                     (US realistic)
               ucits   = tranche, lag 2, 15 bps on the Nasdaq sleeve, 8 bps on IEF/bills/gold (UCITS proxy)
               Per-sleeve costs use backtest_ps(), which is bit-identical to common.backtest when costs are equal.
  Windows      IS88 1988-01-01..2026-07-24 (primary), IS00 2000-01-01..2026-07-24, IS07 2007-01-01..2026-07-24.
               OOS 2026-07-25..end is reported separately and is never used for any decision.
  Metrics      CAGR, maxDD, Calmar (as common.stats); Sharpe of EXCESS returns (r - rf_d, ddof 1, x sqrt 252);
               Ulcer = RMS of the daily drawdown series; mean depth of the 5 deepest distinct drawdown episodes
               (peak -> new high); CDaR95 = mean of the worst 5% of daily drawdown observations; worst rolling
               252-trading-day return; % negative COMPLETE calendar years inside the window.
  Benchmarks   QQQ buy-and-hold (P.qqq); 60/40 = 0.6 spx + 0.4 ief and 85% ief + 15% gold_phys, both daily
               constant mixes with no costs (the same constant-mix convention as every backtest here).
  Bootstrap    stationary bootstrap (Politis & Romano 1994: geometric block lengths, circular wrap), mean block
               length 126 and 252 trading days, 5,000 paired replications (one index draw is applied to every
               series), primary window IS88; IS00 and IS07 reported as supplementary.
               Pairs: each of the 24 candidates minus QQQ; F2 minus F3 and F2 minus F1 (per off-leg x basis);
               50/50 minus IEF (per tier x basis).  Metrics: CAGR, Calmar, Ulcer (+ excess Sharpe as extra).
               Reported: sample difference, bootstrap mean and median, percentile 95% CI, P(diff <= 0).
               Sign convention: diff = first - second.  For Ulcer, a NEGATIVE diff means the first is better.
  Planning     per tier x route (US = 'us' basis, UCITS = 'ucits' basis) x off-leg (IEF primary, 50/50 alt):
               CAGR range and Calmar range = min..max of the IS88 / IS00 / IS07 point estimates;
               maxDD = realized IS88 value (the worst of the nested windows), bootstrap median and 5th percentile
               (L = 252, IS88).  No Friday-basis number enters the planning table.
  Critic check re-implement every spec in critic_out_critic_candidates.json with this script's own pandas
               engine, compare every stored number, and re-run its circular 63-day bootstrap (seed 0, 2,000 reps).
               The critic's F1 'UCITS' spec (8 bps on QQQ) is also run through the pre-registered stationary
               bootstrap as a reproduction-only series (it is NOT one of the 24 candidates).
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT_JSON = os.path.join(HERE, "r2_candidates_tranche.json")
CRITIC_JSON = os.path.join(HERE, "critic_out_critic_candidates.json")
MAIN_TRANCHE_JSON = os.path.join(HERE, "main_tranche_check.json")

P = C.panel()
IDX = P.index
FREEZE = C.FREEZE
LATEST = IDX[-1]
RF = P["rf_d"].fillna(0)

# ------------------------------------------------------------------ pre-registered design
WINDOWS = {"IS88": ("1988-01-01", FREEZE), "IS00": ("2000-01-01", FREEZE), "IS07": ("2007-01-01", FREEZE)}
OOS_WIN = (pd.Timestamp("2026-07-25"), LATEST)
TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
OFF_5050 = pd.DataFrame({"ief": 0.5, "bills": 0.5}, index=IDX)
OFFS = {"IEF": None, "5050": OFF_5050}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAY_MARKS = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(DAYS)}
BASES = {
    "fri": dict(kind="fri", lag=1, desc="Friday book (P.mark), lag 1, 3 bps (July convention)"),
    "tranche": dict(kind="tranche", lag=1, desc="5-weekday tranche, lag 1, 3 bps"),
    "us": dict(kind="tranche", lag=1, desc="US realistic: 5-weekday tranche, lag 1, 5 bps"),
    "ucits": dict(kind="tranche", lag=2,
                  desc="UCITS proxy: 5-weekday tranche, lag 2, 15 bps Nasdaq sleeve, 8 bps IEF/bills/gold"),
}
BOOT_L = [126, 252]
BOOT_REPS = 5000
BOOT_WINDOWS = ["IS88", "IS00", "IS07"]          # IS88 = primary
BOOT_SEED = {"IS88": 11, "IS00": 22, "IS07": 33}  # fixed seeds (seed + L used per run)
PLAN_L = 252


def basis_cost(basis, lev):
    if basis in ("fri", "tranche"):
        return 3e-4
    if basis == "us":
        return 5e-4
    if basis == "ucits":
        return {lev: 15e-4, "ief": 8e-4, "bills": 8e-4, "gold_phys": 8e-4}
    raise KeyError(basis)


VARIANTS = []          # every return series evaluated, with its category


def reg(name, category, note=""):
    VARIANTS.append(dict(name=name, category=category, note=note))


# ------------------------------------------------------------------ engine
def backtest_ps(w, cost, lag=1, mark="mark"):
    """common.backtest with a per-sleeve cost (float, or dict sleeve -> cost per unit one-way turnover).
    Sleeves are grouped by cost level in column order, so with equal costs the arithmetic is exactly
    common.backtest's (verified bit-for-bit in the integrity section)."""
    m = P[mark] if isinstance(mark, str) else mark
    ww = w.where(m).ffill().fillna(0).shift(lag).fillna(0)
    gross = sum(ww[c] * P[c].fillna(0) for c in ww.columns)
    cmap = {c: (cost[c] if isinstance(cost, dict) else cost) for c in ww.columns}
    tc = None
    for k in dict.fromkeys(cmap[c] for c in ww.columns):
        cols = [c for c in ww.columns if cmap[c] == k]
        t = ww[cols].diff().abs().sum(axis=1).fillna(0) * k
        tc = t if tc is None else tc + t
    return gross - tc


def turnover(w, lag, mark):
    hw = C.held_weights(P, w, lag=lag, mark=mark)
    return hw.diff().abs().sum(axis=1).fillna(0)


def run_spec(w, cost, lag, kind):
    """kind 'fri' -> P.mark book; 'tranche' -> mean of the 5 weekday books. Returns (r, books, turnover)."""
    if kind == "fri":
        return backtest_ps(w, cost, lag=lag, mark="mark"), None, turnover(w, lag, "mark")
    books = {d: backtest_ps(w, cost, lag=lag, mark=DAY_MARKS[d]) for d in DAYS}
    tr = sum(books[d] for d in DAYS) / 5
    tov = sum(turnover(w, lag, DAY_MARKS[d]) for d in DAYS) / 5
    return tr, books, tov


# ------------------------------------------------------------------ metrics
def _slice(r, a, b):
    a = max(pd.Timestamp(a), C.START)
    return r[(r.index >= a) & (r.index <= pd.Timestamp(b))]


def metrics(r, a, b):
    s = _slice(r, a, b)
    if s.isna().any():
        raise ValueError("NaN inside evaluation window")
    x = s.values.astype(float)
    n = len(x)
    e = np.cumprod(1.0 + x)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e[-1] ** (1.0 / yrs) - 1.0
    dd = e / np.maximum.accumulate(e) - 1.0
    mdd = float(dd.min())
    ex = x - RF.reindex(s.index).values
    exsh = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
    ulcer = float(np.sqrt(np.mean(dd ** 2)))
    under = dd < 0
    starts = under & ~np.r_[False, under[:-1]]
    eid = np.cumsum(starts)
    depths = (np.sort(pd.Series(dd[under]).groupby(eid[under]).min().values) if under.any() else np.array([]))
    k5 = min(5, len(depths))
    top5 = float(depths[:k5].mean()) if k5 else 0.0
    kc = int(np.ceil(0.05 * n))
    cdar = float(np.sort(dd)[:kc].mean())
    if n > 252:
        cs = np.r_[0.0, np.cumsum(np.log1p(x))]
        w12 = float(np.expm1((cs[252:] - cs[:-252]).min()))
    else:
        w12 = np.nan
    a_ts = max(pd.Timestamp(a), C.START)
    b_ts = pd.Timestamp(b)
    full = [y for y in range(s.index[0].year, s.index[-1].year + 1)
            if pd.Timestamp(year=y, month=1, day=1) >= a_ts and pd.Timestamp(year=y, month=12, day=31) <= b_ts]
    yr = (1.0 + s).groupby(s.index.year).prod() - 1.0
    yr = yr[yr.index.isin(full)]
    return dict(start=str(s.index[0].date()), end=str(s.index[-1].date()), days=int(n), years=float(yrs),
                total=float(e[-1] - 1), cagr=float(cagr), maxdd=mdd, calmar=float(cagr / abs(mdd)) if mdd < 0 else np.nan,
                ex_sharpe=float(exsh), ulcer=ulcer, top5dd_mean=top5, n_dd_episodes=int(len(depths)),
                cdar95=cdar, worst_12m=w12, n_full_years=int(len(yr)), n_neg_years=int((yr < 0).sum()),
                pct_neg_years=float((yr < 0).mean()) if len(yr) else np.nan,
                worst_year=float(yr.min()) if len(yr) else np.nan,
                worst_year_label=int(yr.idxmin()) if len(yr) else None)


def all_windows(r):
    out = {w: metrics(r, a, b) for w, (a, b) in WINDOWS.items()}
    out["OOS"] = metrics(r, *OOS_WIN)
    return out


# ------------------------------------------------------------------ 1. candidates
print("building candidates ...", flush=True)
CFGS = {}
for tier, (lev, vt) in TIERS.items():
    for off, ow in OFFS.items():
        CFGS[f"{tier}-{off}"] = dict(tier=tier, lev=lev, voltgt=vt, off=off,
                                     w=C.strategy_weights(P, lev=lev, voltgt=vt, off_weights=ow))

SER, BOOKS, TOV = {}, {}, {}
for cfg, c in CFGS.items():
    for basis, b in BASES.items():
        r, books, tov = run_spec(c["w"], basis_cost(basis, c["lev"]), b["lag"], b["kind"])
        SER[(cfg, basis)] = r
        BOOKS[(cfg, basis)] = books
        TOV[(cfg, basis)] = tov
        reg(f"{cfg}|{basis}", "candidate", b["desc"])
        if books is not None:
            for d in DAYS:
                reg(f"{cfg}|{basis}|book-{d}", "component", "single-weekday book inside the tranche")

BENCH = {"QQQ B&H": P["qqq"].fillna(0),
         "60/40 spx/ief": 0.6 * P["spx"].fillna(0) + 0.4 * P["ief"].fillna(0),
         "85% IEF + 15% gold": 0.85 * P["ief"].fillna(0) + 0.15 * P["gold_phys"].fillna(0)}
for k in BENCH:
    reg(k, "benchmark")

# ------------------------------------------------------------------ 2. integrity checks
print("integrity checks ...", flush=True)
INTEG = {}
eq = []
for cfg, c in CFGS.items():
    for mk_name, mk in [("P.mark", "mark"), ("Wed", DAY_MARKS["Wed"])]:
        for lag in (1, 2):
            for cost in (3e-4, 5e-4, 15e-4):
                a1 = backtest_ps(c["w"], cost, lag=lag, mark=mk).values
                a2 = C.backtest(P, c["w"], cost=cost, lag=lag, mark=mk).values
                eq.append(dict(cfg=cfg, mark=mk_name, lag=lag, cost=cost, bit_identical=bool(np.array_equal(a1, a2)),
                               max_abs_diff=float(np.max(np.abs(a1 - a2)))))
INTEG["equal_cost_backtest_ps_vs_common_backtest"] = dict(
    n_cases=len(eq), all_bit_identical=all(e_["bit_identical"] for e_ in eq),
    max_abs_diff=max(e_["max_abs_diff"] for e_ in eq))

# independent numpy engine (held weights via last-mark index, per-sleeve |dh| x cost, tranche by averaging
# held weights) for the unequal-cost UCITS series
N = len(IDX)


def _last_idx(mask):
    pos = np.where(mask, np.arange(N), -1)
    return np.maximum.accumulate(pos)


LAST = {d: _last_idx(DAY_MARKS[d].values) for d in DAYS}
LAST["P.mark"] = _last_idx(P["mark"].values.astype(bool))


def np_engine(w, cost, lag, kind):
    keys = ["P.mark"] if kind == "fri" else DAYS
    Hs = []
    for k in keys:
        li = LAST[k]
        H = {}
        for col in w.columns:
            v = w[col].values
            h = np.where(li >= 0, v[np.maximum(li, 0)], 0.0)
            H[col] = np.concatenate([np.zeros(lag), h[:-lag]])
        Hs.append(H)
    H = {col: sum(h[col] for h in Hs) / len(Hs) for col in w.columns}
    gross = np.zeros(N)
    tc = np.zeros(N)
    for col, h in H.items():
        gross += h * P[col].fillna(0).values
        dh = np.abs(np.diff(h, prepend=np.nan))
        dh[0] = 0.0
        tc += dh * (cost[col] if isinstance(cost, dict) else cost)
    return gross - tc


ind = {}
for cfg, c in CFGS.items():
    for basis in ("ucits", "us", "fri"):
        b = BASES[basis]
        a1 = SER[(cfg, basis)].values
        a2 = np_engine(c["w"], basis_cost(basis, c["lev"]), b["lag"], b["kind"])
        ind[f"{cfg}|{basis}"] = float(np.max(np.abs(a1 - a2)))
INTEG["independent_numpy_engine_max_abs_diff"] = dict(per_series=ind, max=max(ind.values()))

# Friday weekday book (dayofweek == 4) == P.mark book; tranche Friday component == 'fri' basis
INTEG["fri_weekday_book_equals_P.mark_book"] = all(
    bool(np.array_equal(BOOKS[(cfg, "tranche")]["Fri"].values, SER[(cfg, "fri")].values)) for cfg in CFGS)
INTEG["weights_row_sum_max_abs_dev_from_1"] = float(max(np.max(np.abs(c["w"].sum(axis=1).values - 1)) for c in CFGS.values()))

# reproduce main_tranche_check.json (the round-1 main-session reference)
mt = json.load(open(MAIN_TRANCHE_JSON))
rep = {}
for cfg, key in [("F1-IEF", "F1-IEF"), ("F2-IEF", "F2-IEF"), ("F3-IEF", "F3-IEF")]:
    for basis, mk in [("fri", "engine-Fri (P.mark)"), ("tranche", "tranche")]:
        r = SER[(cfg, basis)]
        mine_tf = C.stats(r, b=FREEZE)
        mine_00 = C.stats(r, a="2000-01-01", b=FREEZE)
        ref = mt[key][mk]
        rep[f"{cfg}|{basis}"] = dict(
            cagr_to_freeze=[mine_tf["cagr"], ref["to_freeze"]["cagr"]],
            maxdd_to_freeze=[mine_tf["maxdd"], ref["to_freeze"]["maxdd"]],
            calmar_to_freeze=[mine_tf["calmar"], ref["to_freeze"]["calmar"]],
            calmar_since2000=[mine_00["calmar"], ref["since2000"]["calmar"]],
            oos=[C.period_return(r, "2026-07-25", "2026-12-31"), ref["oos"]],
            max_abs_diff=float(max(abs(mine_tf["cagr"] - ref["to_freeze"]["cagr"]),
                                   abs(mine_tf["maxdd"] - ref["to_freeze"]["maxdd"]),
                                   abs(mine_tf["calmar"] - ref["to_freeze"]["calmar"]),
                                   abs(mine_00["calmar"] - ref["since2000"]["calmar"]),
                                   abs(C.period_return(r, "2026-07-25", "2026-12-31") - ref["oos"]))))
INTEG["reproduce_main_tranche_check"] = dict(per_series=rep, max_abs_diff=max(v["max_abs_diff"] for v in rep.values()))

# ------------------------------------------------------------------ 3. metrics tables
print("metrics ...", flush=True)
CAND = {}
for cfg in CFGS:
    CAND[cfg] = {}
    for basis in BASES:
        r = SER[(cfg, basis)]
        mw = all_windows(r)
        tv = TOV[(cfg, basis)]
        m_is = (tv.index >= C.START) & (tv.index <= FREEZE)
        for w in mw:
            mw[w]["turnover_per_year_IS88"] = float(tv[m_is].sum() / mw["IS88"]["years"]) if w == "IS88" else None
        CAND[cfg][basis] = mw
BENCHM = {k: all_windows(r) for k, r in BENCH.items()}

WEEKDAY = {}
for cfg in CFGS:
    WEEKDAY[cfg] = {}
    for basis in ("tranche", "us", "ucits"):
        WEEKDAY[cfg][basis] = {}
        for d in DAYS:
            m88 = metrics(BOOKS[(cfg, basis)][d], *WINDOWS["IS88"])
            m00 = metrics(BOOKS[(cfg, basis)][d], *WINDOWS["IS00"])
            WEEKDAY[cfg][basis][d] = dict(IS88=dict(cagr=m88["cagr"], maxdd=m88["maxdd"], calmar=m88["calmar"],
                                                    ex_sharpe=m88["ex_sharpe"], ulcer=m88["ulcer"]),
                                          IS00=dict(cagr=m00["cagr"], maxdd=m00["maxdd"], calmar=m00["calmar"]),
                                          OOS_total=C.period_return(BOOKS[(cfg, basis)][d], *OOS_WIN))

# ------------------------------------------------------------------ 4. critic reproduction
print("critic reproduction ...", flush=True)
crit = json.load(open(CRITIC_JSON))
UC = crit["meta"]["ucits_cost_model"]
CRIT_SPECS = {
    "F2-IEF (QLD vt.30)": dict(lev="qld", vt=0.30, off=None, lag=1, cost=C.COST),
    "F2-IEF lag2 (Monday execution)": dict(lev="qld", vt=0.30, off=None, lag=2, cost=C.COST),
    "F2-IEF lag2 + UCITS-level costs 15/8bp": dict(lev="qld", vt=0.30, off=None, lag=2,
                                                  cost={"qld": UC["qld"], "ief": UC["ief"], "gold_phys": UC["gold_phys"]}),
    "F3-IEF (TQQQ vt.20)": dict(lev="tqqq", vt=0.20, off=None, lag=1, cost=C.COST),
    "F2 off-leg 50/50 IEF/BIL": dict(lev="qld", vt=0.30, off="5050", lag=1, cost=C.COST),
    "F3 off-leg 50/50 IEF/BIL": dict(lev="tqqq", vt=0.20, off="5050", lag=1, cost=C.COST),
    "F1-IEF (QQQ vt.20)": dict(lev="qqq", vt=0.20, off=None, lag=1, cost=C.COST),
    "F1-IEF lag2 + UCITS-level costs 8/8bp": dict(lev="qqq", vt=0.20, off=None, lag=2,
                                                 cost={"qqq": 8e-4, "ief": 8e-4, "gold_phys": 8e-4}),
    "F2-TLT (scanner-style off-leg)": dict(lev="qld", vt=0.30, off="tlt", lag=1, cost=C.COST),
}
CWINS = {"IS": ("1988-01-01", FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
         "2000-frz": ("2000-01-01", FREEZE), "2010-frz": ("2010-01-01", FREEZE)}
CCRI = {"2022": ("2022-01-01", "2022-12-31"), "dotcom": ("2000-03-01", "2002-10-31"),
        "GFC": ("2007-10-01", "2009-03-31"), "OOS": ("2026-07-25", str(LATEST.date()))}
CSER = {}
crit_cmp = {}
dup = {}
for name, s in CRIT_SPECS.items():
    if s["off"] == "5050":
        w = C.strategy_weights(P, lev=s["lev"], voltgt=s["vt"], off_weights=OFF_5050)
    elif s["off"] == "tlt":
        w = C.strategy_weights(P, lev=s["lev"], voltgt=s["vt"], off="tlt")
    else:
        w = C.strategy_weights(P, lev=s["lev"], voltgt=s["vt"])
    for mode in ("Fri", "tranche"):
        r, _, _ = run_spec(w, s["cost"], s["lag"], "fri" if mode == "Fri" else "tranche")
        CSER[(name, mode)] = r
        same = [f"{cfg}|{basis}" for (cfg, basis), rr in SER.items() if np.array_equal(rr.values, r.values)]
        dup[f"{name} [{mode}]"] = same
        reg(f"critic:{name} [{mode}]", "reproduction" if not same else "reproduction (identical to candidate)",
            ",".join(same))
        ref = crit[name][mode]
        diffs = {}
        for wn, (a, b) in CWINS.items():
            m = metrics(r, a, b)
            for k in ("cagr", "maxdd", "calmar", "ex_sharpe"):
                diffs[f"{wn}.{k}"] = [m[k], ref[wn][k], m[k] - ref[wn][k]]
        for cn, (a, b) in CCRI.items():
            v = C.period_return(r, a, b)
            diffs[f"crisis.{cn}"] = [v, ref["crises"][cn], v - ref["crises"][cn]]
        crit_cmp[f"{name} [{mode}]"] = dict(max_abs_diff=float(max(abs(d[2]) for d in diffs.values())), values=diffs)
qqq_np = BENCH["QQQ B&H"]
crit_cmp["QQQ buy-and-hold"] = {}
for wn in ("IS", "2000-frz"):
    m = metrics(qqq_np, *CWINS[wn])
    ref = crit["QQQ buy-and-hold"][wn]
    crit_cmp["QQQ buy-and-hold"][wn] = {k: [m[k], ref[k], m[k] - ref[k]] for k in ("cagr", "maxdd", "calmar", "ex_sharpe")}


def critic_boot(r1, r2, n=2000, block=63, seed=0):
    """exact re-implementation of critic_candidates.boot (circular fixed 63-day blocks, 252/T annualisation)."""
    m = (IDX >= C.START) & (IDX <= FREEZE)
    a = np.column_stack([r1.values[m], r2.values[m]])
    T = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    dc, dk, dd1, dd2 = [], [], [], []
    for _ in range(n):
        stt = rng.integers(0, T, nb)
        ix = (stt[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = a[ix]
        e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        mdd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = g / np.abs(mdd)
        dc.append(g[0] - g[1]); dk.append(cal[0] - cal[1]); dd1.append(mdd[0]); dd2.append(mdd[1])
    dc, dk = np.array(dc), np.array(dk)
    return dict(dcagr=[float(np.mean(dc)), float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
                p_dcagr_le0=float((dc <= 0).mean()),
                dcalmar=[float(np.mean(dk)), float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5))],
                p_dcalmar_le0=float((dk <= 0).mean()),
                maxdd_median=[float(np.median(dd1)), float(np.median(dd2))],
                maxdd_p5=[float(np.percentile(dd1, 5)), float(np.percentile(dd2, 5))])


def _flat(d):
    out = []
    for k in ("dcagr", "dcalmar", "maxdd_median", "maxdd_p5"):
        out += list(d[k])
    out += [d["p_dcagr_le0"], d["p_dcalmar_le0"]]
    return np.array(out)


crit_boot = {}
for nm in ["F2-IEF (QLD vt.30)", "F3-IEF (TQQQ vt.20)", "F1-IEF (QQQ vt.20)", "F2-IEF lag2 + UCITS-level costs 15/8bp",
           "F2 off-leg 50/50 IEF/BIL", "F1-IEF lag2 + UCITS-level costs 8/8bp"]:
    for mode in ("Fri", "tranche"):
        key = f"{nm} [{mode}] minus QQQ"
        mine = critic_boot(CSER[(nm, mode)], qqq_np)
        crit_boot[key] = dict(mine=mine, critic=crit["bootstrap_IS"][key],
                              max_abs_diff=float(np.max(np.abs(_flat(mine) - _flat(crit["bootstrap_IS"][key])))))
for mode in ("Fri", "tranche"):
    for key, (x1, x2) in {f"F2 minus F3 [{mode}]": ("F2-IEF (QLD vt.30)", "F3-IEF (TQQQ vt.20)"),
                          f"F2 50/50 minus F2 IEF [{mode}]": ("F2 off-leg 50/50 IEF/BIL", "F2-IEF (QLD vt.30)")}.items():
        mine = critic_boot(CSER[(x1, mode)], CSER[(x2, mode)])
        crit_boot[key] = dict(mine=mine, critic=crit["bootstrap_IS"][key],
                              max_abs_diff=float(np.max(np.abs(_flat(mine) - _flat(crit["bootstrap_IS"][key])))))
CRITIC = dict(
    metrics_max_abs_diff=float(max(v["max_abs_diff"] for k, v in crit_cmp.items() if k != "QQQ buy-and-hold")),
    qqq_max_abs_diff=float(max(abs(v[2]) for wn in crit_cmp["QQQ buy-and-hold"].values() for v in wn.values())),
    bootstrap_max_abs_diff=float(max(v["max_abs_diff"] for v in crit_boot.values())),
    tolerance_note="critic JSON values are rounded to 5 decimals, so |diff| <= 5e-6 means exact reproduction",
    identical_series=dup, metrics=crit_cmp, bootstrap_63d=crit_boot)

# ------------------------------------------------------------------ 5. stationary bootstrap
print("stationary bootstrap ...", flush=True)
COLS = [f"{cfg}|{basis}" for cfg in CFGS for basis in BASES] + list(BENCH) + ["critic:F1-IEF lag2 8/8bp [tranche]"]
COLSER = {f"{cfg}|{basis}": SER[(cfg, basis)] for cfg in CFGS for basis in BASES}
COLSER.update(BENCH)
COLSER["critic:F1-IEF lag2 8/8bp [tranche]"] = CSER[("F1-IEF lag2 + UCITS-level costs 8/8bp", "tranche")]
CI = {c: i for i, c in enumerate(COLS)}

PAIRS = []
for cfg in CFGS:
    for basis in BASES:
        PAIRS.append(("cand_minus_QQQ", f"{cfg}|{basis}", "QQQ B&H"))
for off in OFFS:
    for basis in BASES:
        PAIRS.append(("F2_minus_F3", f"F2-{off}|{basis}", f"F3-{off}|{basis}"))
        PAIRS.append(("F2_minus_F1", f"F2-{off}|{basis}", f"F1-{off}|{basis}"))
for tier in TIERS:
    for basis in BASES:
        PAIRS.append(("5050_minus_IEF", f"{tier}-5050|{basis}", f"{tier}-IEF|{basis}"))
PAIRS.append(("critic_repro_minus_QQQ", "critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"))


def sb_indices(T, L, rng):
    """Politis-Romano stationary bootstrap: new block with prob 1/L each day, uniform start, circular wrap."""
    new = rng.random(T) < 1.0 / L
    new[0] = True
    pos = np.flatnonzero(new)
    src = rng.integers(0, T, size=pos.size)
    bid = np.cumsum(new) - 1
    return (src[bid] + np.arange(T) - pos[bid]) % T, pos.size


def bstats(X, rfv, yrs, ix):
    R = X[:, ix]
    e = np.cumprod(1.0 + R, axis=1)
    cagr = e[:, -1] ** (1.0 / yrs) - 1.0
    dd = e / np.maximum.accumulate(e, axis=1) - 1.0
    mdd = dd.min(axis=1)
    ulcer = np.sqrt((dd * dd).mean(axis=1))
    ex = R - rfv[ix][None, :]
    exsh = ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(252)
    return cagr, mdd, ulcer, exsh


def q(a, p):
    return float(np.percentile(a, p))


BOOT = {}
block_check = {}
for wname in BOOT_WINDOWS:
    a, b = WINDOWS[wname]
    sl = {c: _slice(COLSER[c], a, b) for c in COLS}
    dates = sl[COLS[0]].index
    for c in COLS:
        assert sl[c].index.equals(dates) and not sl[c].isna().any()
    X = np.vstack([sl[c].values for c in COLS]).astype(float)
    rfv = RF.reindex(dates).values.astype(float)
    T = X.shape[1]
    yrs = (dates[-1] - dates[0]).days / 365.25
    pc, pm, pu, pe = bstats(X, rfv, yrs, np.arange(T))
    point = dict(cagr=pc, maxdd=pm, ulcer=pu, exsh=pe, calmar=pc / np.abs(pm))
    # point estimates must equal metrics()
    chk = 0.0
    for c in COLS:
        mm = metrics(COLSER[c], a, b)
        chk = max(chk, abs(pc[CI[c]] - mm["cagr"]), abs(pm[CI[c]] - mm["maxdd"]),
                  abs(pu[CI[c]] - mm["ulcer"]), abs(pe[CI[c]] - mm["ex_sharpe"]))
    BOOT[wname] = dict(T=int(T), years=float(yrs), point_vs_metrics_max_abs_diff=float(chk))
    for L in BOOT_L:
        t1 = time.time()
        rng = np.random.default_rng(BOOT_SEED[wname] + L)
        D = {m: np.empty((BOOT_REPS, len(COLS))) for m in ("cagr", "maxdd", "ulcer", "exsh")}
        nblocks = []
        for i in range(BOOT_REPS):
            ix, nb_ = sb_indices(T, L, rng)
            nblocks.append(nb_)
            D["cagr"][i], D["maxdd"][i], D["ulcer"][i], D["exsh"][i] = bstats(X, rfv, yrs, ix)
        D["calmar"] = D["cagr"] / np.abs(D["maxdd"])
        block_check[f"{wname}|L{L}"] = float(T / np.mean(nblocks))
        own = {}
        for c in COLS:
            j = CI[c]
            own[c] = {m: dict(sample=float(point[m][j]), median=q(D[m][:, j], 50), p5=q(D[m][:, j], 5),
                              p95=q(D[m][:, j], 95), p2_5=q(D[m][:, j], 2.5), p97_5=q(D[m][:, j], 97.5))
                      for m in ("cagr", "maxdd", "calmar", "ulcer", "exsh")}
            own[c]["maxdd"]["p_worse_than_-50pct"] = float((D["maxdd"][:, j] < -0.50).mean())
        pairs = {}
        for fam, x1, x2 in PAIRS:
            i1, i2 = CI[x1], CI[x2]
            ent = dict(family=fam, first=x1, second=x2)
            for m in ("cagr", "calmar", "ulcer", "exsh"):
                d = D[m][:, i1] - D[m][:, i2]
                ent[m] = dict(sample=float(point[m][i1] - point[m][i2]), boot_mean=float(d.mean()),
                              boot_median=q(d, 50), ci95=[q(d, 2.5), q(d, 97.5)], p_le0=float((d <= 0).mean()))
            pairs[f"{x1} minus {x2}"] = ent
        BOOT[wname][f"L{L}"] = dict(reps=BOOT_REPS, seed=BOOT_SEED[wname] + L, own=own, pairs=pairs,
                                    runtime_s=round(time.time() - t1, 1))
        print(f"  {wname} L={L}: {time.time() - t1:5.1f}s", flush=True)
INTEG["stationary_bootstrap_mean_block_length_check"] = block_check
INTEG["bootstrap_point_equals_metrics"] = {w: BOOT[w]["point_vs_metrics_max_abs_diff"] for w in BOOT_WINDOWS}

# ------------------------------------------------------------------ 6. planning table (pre-registered rule)
PLAN = []
for tier in TIERS:
    for route, basis in (("US", "us"), ("UCITS", "ucits")):
        for off in OFFS:
            cfg = f"{tier}-{off}"
            ms = {w: CAND[cfg][basis][w] for w in WINDOWS}
            own = BOOT["IS88"][f"L{PLAN_L}"]["own"][f"{cfg}|{basis}"]
            note = ""
            if tier == "F3" and route == "UCITS":
                note = "hypothetical: round-1 ucits found no viable UCITS 3x fund (QQQ3.L is a non-UCITS ETP)"
            if tier == "F1" and route == "UCITS":
                note = "15 bps on the 1x sleeve per the pre-registered proxy (critic used 8 bps)"
            PLAN.append(dict(tier=tier, route=route, off_leg=off, basis=basis,
                             cagr_range=[min(m["cagr"] for m in ms.values()), max(m["cagr"] for m in ms.values())],
                             cagr_by_window={w: m["cagr"] for w, m in ms.items()},
                             calmar_range=[min(m["calmar"] for m in ms.values()), max(m["calmar"] for m in ms.values())],
                             calmar_by_window={w: m["calmar"] for w, m in ms.items()},
                             maxdd_realized_IS88=ms["IS88"]["maxdd"],
                             maxdd_boot_median=own["maxdd"]["median"], maxdd_boot_p5=own["maxdd"]["p5"],
                             p_maxdd_worse_than_50=own["maxdd"]["p_worse_than_-50pct"],
                             cagr_boot_p5_p95=[own["cagr"]["p5"], own["cagr"]["p95"]],
                             ex_sharpe_IS88=ms["IS88"]["ex_sharpe"], ulcer_IS88=ms["IS88"]["ulcer"],
                             oos_total_SEPARATE=CAND[cfg][basis]["OOS"]["total"], note=note))

# ------------------------------------------------------------------ 7. variant count + save
vc = {}
for v in VARIANTS:
    vc[v["category"]] = vc.get(v["category"], 0) + 1
META = dict(script="update_2026_11/r2_candidates_tranche.py", run_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            panel_end=str(LATEST.date()), freeze=str(FREEZE.date()), windows={k: [str(pd.Timestamp(a).date()), str(pd.Timestamp(b).date())]
                                                                                for k, (a, b) in WINDOWS.items()},
            oos=[str(OOS_WIN[0].date()), str(OOS_WIN[1].date())], bases={k: v["desc"] for k, v in BASES.items()},
            tiers={k: dict(sleeve=v[0], voltgt=v[1]) for k, v in TIERS.items()},
            off_legs={"IEF": "100% ief", "5050": "50% ief + 50% bills (off_weights)"},
            bootstrap=dict(method="stationary (Politis-Romano), circular", mean_block=BOOT_L, reps=BOOT_REPS,
                           windows=BOOT_WINDOWS, primary="IS88", seeds={w: [BOOT_SEED[w] + L for L in BOOT_L] for w in BOOT_WINDOWS},
                           sign="diff = first - second; for ulcer a negative diff means the first is better"),
            planning_rule=("CAGR/Calmar range = min..max over IS88/IS00/IS07 point estimates on the route basis "
                           "(US='us', UCITS='ucits'); maxDD = realized IS88 + bootstrap median and 5th pct (L=252, IS88). "
                           "No Friday-basis number used."),
            metric_defs=dict(ulcer="sqrt(mean(dd_t^2)), dd_t = equity/running max - 1 (fraction)",
                             top5dd_mean="mean depth of the 5 deepest distinct peak-to-new-high drawdown episodes",
                             cdar95="mean of the worst ceil(5% x n) daily drawdown observations",
                             worst_12m="minimum rolling 252-trading-day compounded return",
                             pct_neg_years="share of COMPLETE calendar years in the window with a negative return",
                             ex_sharpe="mean(r - rf_d)/std(r - rf_d, ddof 1) x sqrt(252)"),
            data_caveats=["spx in the panel is the S&P 500 PRICE index (1988-2025 CAGR 9.1%); the 60/40 benchmark omits "
                          "equity dividends. A total-return fetch (^SP500TR) failed: no network in this sandbox.",
                          "qqq before 1999-03-10 is NDX price return - 0.20%/yr (no dividends); qld/tqqq before inception "
                          "are simulations (2x/3x NDX - 0.95% - financing at T-bill).",
                          "gold_phys before 2000-08-30 is a monthly-average log-interpolation (return sleeve only, never a signal).",
                          "ief/tlt/shy before 2002-07-30 are calibrated synthetics; bills before 2007-05-30 = T-bill - 0.13%/yr."],
            variant_counts=vc, n_series_total=len(VARIANTS))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 7)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


OUT = dict(meta=META, integrity=INTEG, candidates=CAND, benchmarks=BENCHM, weekday_books=WEEKDAY,
           bootstrap=BOOT, critic_reproduction=CRITIC, planning_table=PLAN, variants=VARIANTS)
OUT["meta"]["runtime_s"] = round(time.time() - T0, 1)
json.dump(clean(OUT), open(OUT_JSON, "w"), indent=1)

# ------------------------------------------------------------------ 8. console summary
pc_ = lambda v: f"{v * 100:6.2f}" if v is not None and np.isfinite(v) else "   n/a"
print("\nINTEGRITY", json.dumps(clean({k: v for k, v in INTEG.items() if k not in ("reproduce_main_tranche_check", "independent_numpy_engine_max_abs_diff")}), indent=0)[:1500])
print("  independent numpy engine max diff", INTEG["independent_numpy_engine_max_abs_diff"]["max"],
      "| main_tranche_check max diff", INTEG["reproduce_main_tranche_check"]["max_abs_diff"])
print(f"\nCRITIC: metrics max|diff| {CRITIC['metrics_max_abs_diff']:.2e}  QQQ {CRITIC['qqq_max_abs_diff']:.2e}  "
      f"bootstrap max|diff| {CRITIC['bootstrap_max_abs_diff']:.2e}")
print("\nCANDIDATES (IS88 | IS00 | IS07; CAGR/maxDD/Calmar; exSh, Ulcer, top5, CDaR95, worst12m, %negY) + OOS total")
for cfg in CFGS:
    for basis in BASES:
        m = CAND[cfg][basis]
        a = m["IS88"]
        print(f"{cfg:8s} {basis:7s} {pc_(a['cagr'])}/{pc_(a['maxdd'])}/{a['calmar']:.2f} exSh {a['ex_sharpe']:.2f} "
              f"U {pc_(a['ulcer'])} t5 {pc_(a['top5dd_mean'])} CD {pc_(a['cdar95'])} w12 {pc_(a['worst_12m'])} "
              f"neg {a['n_neg_years']}/{a['n_full_years']} TO {a['turnover_per_year_IS88']:.1f} | "
              f"00 {pc_(m['IS00']['cagr'])}/{pc_(m['IS00']['maxdd'])}/{m['IS00']['calmar']:.2f} | "
              f"07 {pc_(m['IS07']['cagr'])}/{pc_(m['IS07']['maxdd'])}/{m['IS07']['calmar']:.2f} | OOS {pc_(m['OOS']['total'])}")
for k, m in BENCHM.items():
    a = m["IS88"]
    print(f"{k:16s} {pc_(a['cagr'])}/{pc_(a['maxdd'])}/{a['calmar']:.2f} exSh {a['ex_sharpe']:.2f} U {pc_(a['ulcer'])} "
          f"t5 {pc_(a['top5dd_mean'])} CD {pc_(a['cdar95'])} w12 {pc_(a['worst_12m'])} neg {a['n_neg_years']}/{a['n_full_years']} | "
          f"00 {pc_(m['IS00']['cagr'])}/{pc_(m['IS00']['maxdd'])}/{m['IS00']['calmar']:.2f} | "
          f"07 {pc_(m['IS07']['cagr'])}/{pc_(m['IS07']['maxdd'])}/{m['IS07']['calmar']:.2f} | OOS {pc_(m['OOS']['total'])}")
for L in BOOT_L:
    print(f"\nBOOTSTRAP IS88 L={L}: diff sample [CI95] p<=0  (CAGR | Calmar | Ulcer)")
    for k, e_ in BOOT["IS88"][f"L{L}"]["pairs"].items():
        c_, k_, u_ = e_["cagr"], e_["calmar"], e_["ulcer"]
        print(f"  {k:52s} {c_['sample']*100:+6.2f} [{c_['ci95'][0]*100:+6.2f},{c_['ci95'][1]*100:+6.2f}] {c_['p_le0']:.3f} | "
              f"{k_['sample']:+.3f} [{k_['ci95'][0]:+.3f},{k_['ci95'][1]:+.3f}] {k_['p_le0']:.3f} | "
              f"{u_['sample']*100:+6.2f} [{u_['ci95'][0]*100:+6.2f},{u_['ci95'][1]*100:+6.2f}] {u_['p_le0']:.3f}")
print("\nPLANNING TABLE (tranche bases only)")
for p in PLAN:
    print(f"  {p['tier']} {p['route']:5s} {p['off_leg']:4s} CAGR {pc_(p['cagr_range'][0])}..{pc_(p['cagr_range'][1])} "
          f"Calmar {p['calmar_range'][0]:.2f}..{p['calmar_range'][1]:.2f} maxDD real {pc_(p['maxdd_realized_IS88'])} "
          f"boot med {pc_(p['maxdd_boot_median'])} p5 {pc_(p['maxdd_boot_p5'])} | OOS {pc_(p['oos_total_SEPARATE'])} {p['note']}")
print(f"\nvariants: {vc}  total series {len(VARIANTS)}  runtime {time.time() - T0:.1f}s")
print(f"saved {OUT_JSON}")
