"""
INTEGRITY REVIEW of r2_candidates_tranche (independent re-implementation; does NOT import the specialist's code
and does NOT use common.backtest / common.strategy_weights).  Read-only w.r.t. project files: the only output is a
JSON in the scratch dir given on the command line.

Checks
  A  signals: recompute ema100/above/vol20/er20 from ndx; causality (truncate ndx at FREEZE -> identical to FREEZE)
  B  weights: rules re-implemented from the brief; gold weight is a constant 0.15 (never a signal)
  C  data: NaN / exact-zero counts of every sleeve in the windows, gold_src splice dates, rf
  D  engine: searchsorted-based held weights (last mark <= t-lag), per-sleeve |dh| x cost, tranche = mean of 5 books
     -> compare every candidate x basis x window metric with the saved JSON
  E  truncation test: rebuild everything on the panel cut at FREEZE -> in-sample metrics must be unchanged
  F  IEF synthetic calibrated only on data <= FREEZE (the panel used real IEF to 2026-11-17) -> effect on IS numbers
  G  benchmarks, weekday books, Friday premium (lag 1 and lag 2), turnover / cost arithmetic
  H  planning rule re-applied from scratch; claim numbers in the review text vs saved JSON
  I  critic JSON reproduced with this engine (+ critic 63-day bootstrap rerun on these series)
usage: python -u update_2026_11/r2_verify_candidates_tranche_integrity_checks.py <scratch_dir>
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = sys.argv[1]
os.makedirs(SCR, exist_ok=True)
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
SAVED = json.load(open(os.path.join(HERE, "r2_candidates_tranche.json")))
CRIT = json.load(open(os.path.join(HERE, "critic_out_critic_candidates.json")))
START = pd.Timestamp("1988-01-01")
FREEZE = pd.Timestamp("2026-07-24")
OUT = {}


def pr(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ A. signals
def signals(ndx):
    ema = ndx.ewm(span=100, adjust=False).mean()
    above = ndx > ema
    vol20 = ndx.pct_change().rolling(20).std() * np.sqrt(252)
    er20 = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum().replace(0, np.nan)
    return pd.DataFrame(dict(ema100=ema, above=above, vol20=vol20, er20=er20))


S_full = signals(P["ndx"])
S_tr = signals(P["ndx"][P.index <= FREEZE])
A = {}
for c in ["ema100", "vol20", "er20"]:
    A[f"{c}_panel_vs_recomputed_maxabs"] = float(np.nanmax(np.abs(S_full[c].values - P[c].values)))
    A[f"{c}_nan_pattern_equal"] = bool((S_full[c].isna() == P[c].isna()).all())
    A[f"{c}_trunc_vs_full_to_FREEZE_maxabs"] = float(np.nanmax(np.abs(S_tr[c].values - S_full[c][P.index <= FREEZE].values)))
A["above_panel_vs_recomputed_equal"] = bool((S_full["above"] == P["above"]).all())
A["above_trunc_vs_full_equal"] = bool((S_tr["above"] == S_full["above"][P.index <= FREEZE]).all())
win = (P.index >= START)
A["nan_in_signals_from_1988"] = {c: int(P.loc[win, c].isna().sum()) for c in ["ndx", "vol20", "er20", "ema100"]}
OUT["A_signals"] = A
pr("A", json.dumps(A))

# ------------------------------------------------------------------ B. weights (from the brief, not common.py)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def weights(Pn, Sg, lev, vt, off):
    vol = Sg["vol20"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.minimum(1.0, vt / vol)
    scale = np.where(np.isnan(scale), 1.0, np.clip(scale, 0, 1))
    mon = Pn.index.month.values
    scale = np.where(np.isin(mon, [8, 9]), 0.5 * scale, scale)
    er = Sg["er20"].values
    clean = np.where(np.isnan(er), False, er >= 0.25)
    on = Sg["above"].values.astype(bool)
    eng = np.where(on, np.where(clean, scale, 0.5 * scale), 0.0)
    W = {lev: 0.85 * eng}
    rest = 0.85 - 0.85 * eng
    if off == "IEF":
        W["ief"] = rest
    else:
        W["ief"] = 0.5 * rest
        W["bills"] = 0.5 * rest
    W["gold_phys"] = np.full(len(Pn), 0.15)
    return W


TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
CFG = {f"{t}-{o}": (lev, vt, o) for t, (lev, vt) in TIERS.items() for o in ("IEF", "5050")}

# ------------------------------------------------------------------ C. data checks
C_ = {}
for c in ["qqq", "qld", "tqqq", "ief", "bills", "gold_phys", "spx", "rf_d", "rf", "ndx_ret"]:
    s = P.loc[win, c]
    s_is = P.loc[(P.index >= START) & (P.index <= FREEZE), c]
    C_[c] = dict(nan=int(s.isna().sum()), exact_zero=int((s == 0).sum()), exact_zero_IS=int((s_is == 0).sum()))
# zero runs of the NDX itself (missing data proxies)
gs = P["gold_src"]
C_["gold_src_first_dates"] = {k: str(P.index[gs == k][0].date()) for k in gs.unique()}
C_["gold_src_last_dates"] = {k: str(P.index[gs == k][-1].date()) for k in gs.unique()}
gp = P["gold_phys"]
C_["gold_daily_vol_ann"] = dict(
    interp_1988_2000=float(gp[(P.index >= START) & (P.index < "2000-08-30")].std() * np.sqrt(252)),
    gcf_2000_2004=float(gp[(P.index >= "2000-09-01") & (P.index < "2004-11-18")].std() * np.sqrt(252)),
    gld_2004_freeze=float(gp[(P.index >= "2004-11-19") & (P.index <= FREEZE)].std() * np.sqrt(252)))
C_["dayofweek4_equals_P.mark"] = bool(((P.index.dayofweek == 4) == P["mark"].values.astype(bool)).all())
C_["n_weekend_rows"] = int((P.index.dayofweek >= 5).sum())
C_["panel_index_monotonic_unique"] = bool(P.index.is_monotonic_increasing and P.index.is_unique)
OUT["C_data"] = C_
pr("C", json.dumps(C_))


# ------------------------------------------------------------------ D. independent engine
def book(Pn, W, mark_mask, lag, cost):
    """held weight on day i = target at the last mark day j <= i - lag (searchsorted); cost per sleeve x |dh|."""
    n = len(Pn)
    mpos = np.flatnonzero(mark_mask)
    k = np.searchsorted(mpos, np.arange(n) - lag, side="right") - 1
    valid = k >= 0
    src = np.where(valid, mpos[np.maximum(k, 0)], 0)
    gross = np.zeros(n)
    tc = np.zeros(n)
    for c, w in W.items():
        h = np.where(valid, w[src], 0.0)
        gross += h * Pn[c].fillna(0).values
        dh = np.zeros(n)
        dh[1:] = np.abs(h[1:] - h[:-1])
        tc += dh * (cost[c] if isinstance(cost, dict) else cost)
    return gross - tc, tc


def run(Pn, W, lev, kind, lag, cost):
    dow = Pn.index.dayofweek.values
    if kind == "fri":
        r, tc = book(Pn, W, Pn["mark"].values.astype(bool), lag, cost)
        return pd.Series(r, Pn.index), None
    bk = {d: pd.Series(book(Pn, W, dow == i, lag, cost)[0], Pn.index) for i, d in enumerate(DAYS)}
    return sum(bk[d] for d in DAYS) / 5.0, bk


def cost_of(basis, lev):
    return {"fri": 3e-4, "tranche": 3e-4, "us": 5e-4,
            "ucits": {lev: 15e-4, "ief": 8e-4, "bills": 8e-4, "gold_phys": 8e-4}}[basis]


BASES = {"fri": ("fri", 1), "tranche": ("tranche", 1), "us": ("tranche", 1), "ucits": ("tranche", 2)}
WINDOWS = {"IS88": ("1988-01-01", FREEZE), "IS00": ("2000-01-01", FREEZE), "IS07": ("2007-01-01", FREEZE)}


def mets(r, rf, a, b):
    a = max(pd.Timestamp(a), START)
    s = r[(r.index >= a) & (r.index <= pd.Timestamp(b))]
    x = s.values
    e = np.cumprod(1 + x)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = e / np.maximum.accumulate(e) - 1
    mdd = dd.min()
    ex = x - rf.reindex(s.index).values
    # episodes: contiguous dd<0 runs
    depths = []
    cur = 0.0
    inside = False
    for v in dd:
        if v < 0:
            cur = min(cur, v) if inside else v
            inside = True
        elif inside:
            depths.append(cur); inside = False; cur = 0.0
    if inside:
        depths.append(cur)
    depths = sorted(depths)
    kc = int(math.ceil(0.05 * len(x)))
    roll = [np.prod(1 + x[i:i + 252]) - 1 for i in range(0, len(x) - 251)] if len(x) > 252 else [np.nan]
    full = [y for y in range(s.index[0].year, s.index[-1].year + 1)
            if pd.Timestamp(y, 1, 1) >= a and pd.Timestamp(y, 12, 31) <= pd.Timestamp(b)]
    yr = {y: np.prod(1 + x[s.index.year == y]) - 1 for y in full}
    return dict(cagr=cagr, maxdd=mdd, calmar=cagr / abs(mdd), ex_sharpe=ex.mean() / ex.std(ddof=1) * np.sqrt(252),
                ulcer=np.sqrt(np.mean(dd ** 2)), top5dd_mean=np.mean(depths[:5]), cdar95=np.sort(dd)[:kc].mean(),
                worst_12m=float(np.min(roll)), n_neg_years=int(sum(v < 0 for v in yr.values())),
                n_full_years=len(full), total=e[-1] - 1, days=len(x))


RF = P["rf_d"].fillna(0)
SER, BOOKS, W_ALL = {}, {}, {}
for cfg, (lev, vt, off) in CFG.items():
    W = weights(P, S_full, lev, vt, off)
    W_ALL[cfg] = W
    for basis, (kind, lag) in BASES.items():
        SER[(cfg, basis)], BOOKS[(cfg, basis)] = run(P, W, lev, kind, lag, cost_of(basis, lev))

D = {"gold_weight_constant_0.15": all(bool(np.all(W["gold_phys"] == 0.15)) for W in W_ALL.values()),
     "weights_sum_to_1_maxdev": float(max(np.max(np.abs(sum(W.values()) - 1)) for W in W_ALL.values()))}
cmp_ = {}
worst = (0.0, None)
OOS_A = pd.Timestamp("2026-07-25")
for cfg in CFG:
    for basis in BASES:
        r = SER[(cfg, basis)]
        for wn, (a, b) in WINDOWS.items():
            m = mets(r, RF, a, b)
            sv = SAVED["candidates"][cfg][basis][wn]
            for k in ["cagr", "maxdd", "calmar", "ex_sharpe", "ulcer", "top5dd_mean", "cdar95", "worst_12m", "n_neg_years",
                      "n_full_years", "days"]:
                dlt = abs(float(m[k]) - float(sv[k]))
                cmp_[f"{cfg}|{basis}|{wn}|{k}"] = dlt
                if dlt > worst[0]:
                    worst = (dlt, f"{cfg}|{basis}|{wn}|{k}")
        oos = float(np.prod(1 + r[r.index >= OOS_A].values) - 1)
        dlt = abs(oos - SAVED["candidates"][cfg][basis]["OOS"]["total"])
        cmp_[f"{cfg}|{basis}|OOS|total"] = dlt
        if dlt > worst[0]:
            worst = (dlt, f"{cfg}|{basis}|OOS|total")
D["candidate_metrics_vs_saved_max_abs_diff"] = worst[0]
D["candidate_metrics_vs_saved_argmax"] = worst[1]
D["n_compared"] = len(cmp_)
OUT["D_engine"] = D
pr("D", json.dumps(D))

# ------------------------------------------------------------------ E. truncation test (no post-freeze information)
Pt = P[P.index <= FREEZE]
St = signals(Pt["ndx"])
E = {}
worstE = 0.0
for cfg, (lev, vt, off) in CFG.items():
    W = weights(Pt, St, lev, vt, off)
    for basis, (kind, lag) in BASES.items():
        r, _ = run(Pt, W, lev, kind, lag, cost_of(basis, lev))
        for wn, (a, b) in WINDOWS.items():
            m = mets(r, RF, a, b)
            sv = SAVED["candidates"][cfg][basis][wn]
            for k in ["cagr", "maxdd", "calmar", "ex_sharpe", "ulcer"]:
                worstE = max(worstE, abs(float(m[k]) - float(sv[k])))
E["truncated_panel_IS_metrics_vs_saved_max_abs_diff"] = worstE
OUT["E_truncation"] = E
pr("E", json.dumps(E))

# ------------------------------------------------------------------ F. IEF synthetic calibrated only to FREEZE


def calib(real, ylvl, end=None):
    yd = ylvl / 100.0
    dy = yd.diff()
    carry = yd.shift(1) / 252
    df = pd.concat([real, carry, dy], axis=1).dropna()
    df.columns = ["r", "carry", "dy"]
    if end is not None:
        df = df[df.index <= end]
    Y = (df.r - df.carry).values
    X = np.column_stack([np.ones(len(df)), -df.dy.values, 0.5 * df.dy.values ** 2])
    a, Dd, Cc = np.linalg.lstsq(X, Y, rcond=None)[0]
    return a + carry - Dd * dy + 0.5 * Cc * dy ** 2, (a * 252, Dd, Cc)


syn_full, par_full = calib(P["ief_real"], P["tnx"])
syn_frz, par_frz = calib(P["ief_real"], P["tnx"], end=FREEZE)
F = dict(params_full=[float(v) for v in par_full], params_to_freeze=[float(v) for v in par_frz],
         panel_ief_syn_reproduced_maxabs=float(np.nanmax(np.abs(syn_full.values - P["ief_syn"].values))))
P2 = P.copy()
P2["ief"] = P["ief_real"].fillna(syn_frz).fillna(0)
F["ief_changed_days_IS"] = int((np.abs(P2["ief"] - P["ief"])[(P.index >= START) & (P.index <= FREEZE)] > 0).sum())
for cfg in ["F1-IEF", "F2-IEF", "F3-IEF"]:
    lev, vt, off = CFG[cfg]
    W = weights(P2, S_full, lev, vt, off)
    for basis in ["us", "ucits"]:
        kind, lag = BASES[basis]
        r, _ = run(P2, W, lev, kind, lag, cost_of(basis, lev))
        m = mets(r, RF, *WINDOWS["IS88"])
        sv = SAVED["candidates"][cfg][basis]["IS88"]
        F[f"{cfg}|{basis}|IS88"] = dict(cagr=[float(m["cagr"]), sv["cagr"]], maxdd=[float(m["maxdd"]), sv["maxdd"]],
                                        calmar=[float(m["calmar"]), sv["calmar"]])
OUT["F_ief_calibration_to_freeze"] = F
pr("F", json.dumps(F))

# ------------------------------------------------------------------ G. benchmarks, weekday books, Friday premium, costs
G = {}
BEN = {"QQQ B&H": P["qqq"].fillna(0), "60/40 spx/ief": 0.6 * P["spx"].fillna(0) + 0.4 * P["ief"].fillna(0),
       "85% IEF + 15% gold": 0.85 * P["ief"].fillna(0) + 0.15 * P["gold_phys"].fillna(0)}
wb = 0.0
for k, r in BEN.items():
    for wn, (a, b) in WINDOWS.items():
        m = mets(r, RF, a, b)
        sv = SAVED["benchmarks"][k][wn]
        for kk in ["cagr", "maxdd", "calmar", "ex_sharpe", "ulcer", "top5dd_mean", "cdar95", "worst_12m", "n_neg_years"]:
            wb = max(wb, abs(float(m[kk]) - float(sv[kk])))
    G[f"{k}|OOS_total"] = [float(np.prod(1 + r[r.index >= OOS_A].values) - 1), SAVED["benchmarks"][k]["OOS"]["total"]]
G["benchmarks_vs_saved_max_abs_diff"] = wb
# weekday books
wk = 0.0
best_oos = {}
for cfg in CFG:
    for basis in ["tranche", "us", "ucits"]:
        oo = {}
        for d in DAYS:
            r = BOOKS[(cfg, basis)][d]
            m = mets(r, RF, *WINDOWS["IS88"])
            sv = SAVED["weekday_books"][cfg][basis][d]
            wk = max(wk, abs(m["cagr"] - sv["IS88"]["cagr"]), abs(m["calmar"] - sv["IS88"]["calmar"]))
            oo[d] = float(np.prod(1 + r[r.index >= OOS_A].values) - 1)
            wk = max(wk, abs(oo[d] - sv["OOS_total"]))
        best_oos[f"{cfg}|{basis}"] = dict(best=max(oo, key=oo.get), vals={k: round(v, 4) for k, v in oo.items()})
G["weekday_books_vs_saved_max_abs_diff"] = wk
G["weekday_OOS_best"] = best_oos
G["friday_best_OOS_all_us_ucits"] = all(v["best"] == "Fri" for k, v in best_oos.items() if not k.endswith("tranche"))
# Friday premium at lag 1 (3bp) and at lag 2 + UCITS costs (Friday component of the ucits tranche)
fp = {}
for cfg in ["F1-IEF", "F2-IEF", "F3-IEF"]:
    f1 = mets(SER[(cfg, "fri")], RF, *WINDOWS["IS88"])["cagr"]
    t1 = mets(SER[(cfg, "tranche")], RF, *WINDOWS["IS88"])["cagr"]
    f2 = mets(BOOKS[(cfg, "ucits")]["Fri"], RF, *WINDOWS["IS88"])["cagr"]
    t2 = mets(SER[(cfg, "ucits")], RF, *WINDOWS["IS88"])["cagr"]
    fp[cfg] = dict(lag1_3bp_pp=round((f1 - t1) * 100, 3), lag2_ucits_pp=round((f2 - t2) * 100, 3))
G["friday_premium"] = fp
# weekday F2 US range
f2us = {d: mets(BOOKS[("F2-IEF", "us")][d], RF, *WINDOWS["IS88"]) for d in DAYS}
G["F2_us_weekday_cagr"] = {d: round(float(v["cagr"]) * 100, 2) for d, v in f2us.items()}
G["F2_us_weekday_calmar"] = {d: round(float(v["calmar"]), 3) for d, v in f2us.items()}
# turnover per year (mean of book turnovers) and cost arithmetic
tov = {}
for cfg in ["F1-IEF", "F2-IEF", "F3-IEF"]:
    lev, vt, off = CFG[cfg]
    W = W_ALL[cfg]
    dow = P.index.dayofweek.values
    tt = []
    for i, d in enumerate(DAYS):
        _, tc = book(P, W, dow == i, 1, 1.0)   # cost 1.0 -> tc = turnover
        tt.append(tc)
    tv = pd.Series(np.mean(tt, axis=0), P.index)
    m_is = (tv.index >= START) & (tv.index <= FREEZE)
    yrs = (P.index[m_is][-1] - P.index[m_is][0]).days / 365.25
    tov[cfg] = dict(turnover_per_yr=float(tv[m_is].sum() / yrs),
                    saved=SAVED["candidates"][cfg]["us"]["IS88"]["turnover_per_year_IS88"],
                    us_minus_tranche_cagr_pp=round((mets(SER[(cfg, "us")], RF, *WINDOWS["IS88"])["cagr"]
                                                    - mets(SER[(cfg, "tranche")], RF, *WINDOWS["IS88"])["cagr"]) * 100, 3))
G["turnover"] = tov
OUT["G"] = G
pr("G", json.dumps(G)[:3000])

# ------------------------------------------------------------------ H. planning rule re-applied from saved candidates
H = {}
mism = []
for p in SAVED["planning_table"]:
    cfg = f"{p['tier']}-{p['off_leg']}"
    basis = {"US": "us", "UCITS": "ucits"}[p["route"]]
    ms = [SAVED["candidates"][cfg][basis][w] for w in ["IS88", "IS00", "IS07"]]
    own = SAVED["bootstrap"]["IS88"]["L252"]["own"][f"{cfg}|{basis}"]
    exp = dict(cagr_range=[min(m["cagr"] for m in ms), max(m["cagr"] for m in ms)],
               calmar_range=[min(m["calmar"] for m in ms), max(m["calmar"] for m in ms)],
               maxdd_realized_IS88=ms[0]["maxdd"], maxdd_boot_median=own["maxdd"]["median"], maxdd_boot_p5=own["maxdd"]["p5"])
    for k, v in exp.items():
        got = p[k]
        if isinstance(v, list):
            if any(abs(x - y) > 1e-9 for x, y in zip(v, got)):
                mism.append((cfg, basis, k, v, got))
        elif abs(v - got) > 1e-9:
            mism.append((cfg, basis, k, v, got))
    # is the IS88 maxDD the worst of the nested windows (as the docstring asserts)?
    if not (ms[0]["maxdd"] <= ms[1]["maxdd"] + 1e-12 and ms[0]["maxdd"] <= ms[2]["maxdd"] + 1e-12):
        mism.append((cfg, basis, "IS88 maxdd not worst", [m["maxdd"] for m in ms]))
H["planning_rule_mismatches"] = mism
H["planning_rows"] = len(SAVED["planning_table"])
H["any_fri_basis_in_plan"] = any(p["basis"] not in ("us", "ucits") for p in SAVED["planning_table"])
OUT["H_planning"] = H
pr("H", json.dumps(H))

# ------------------------------------------------------------------ I. critic reproduction with this engine
UC = CRIT["meta"]["ucits_cost_model"]
CS = {
    "F2-IEF (QLD vt.30)": ("qld", 0.30, "IEF", 1, 3e-4),
    "F2-IEF lag2 (Monday execution)": ("qld", 0.30, "IEF", 2, 3e-4),
    "F2-IEF lag2 + UCITS-level costs 15/8bp": ("qld", 0.30, "IEF", 2, {"qld": UC["qld"], "ief": UC["ief"], "gold_phys": UC["gold_phys"]}),
    "F3-IEF (TQQQ vt.20)": ("tqqq", 0.20, "IEF", 1, 3e-4),
    "F2 off-leg 50/50 IEF/BIL": ("qld", 0.30, "5050", 1, 3e-4),
    "F3 off-leg 50/50 IEF/BIL": ("tqqq", 0.20, "5050", 1, 3e-4),
    "F1-IEF (QQQ vt.20)": ("qqq", 0.20, "IEF", 1, 3e-4),
    "F1-IEF lag2 + UCITS-level costs 8/8bp": ("qqq", 0.20, "IEF", 2, {"qqq": 8e-4, "ief": 8e-4, "gold_phys": 8e-4}),
}
CW = {"IS": ("1988-01-01", FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
      "2000-frz": ("2000-01-01", FREEZE), "2010-frz": ("2010-01-01", FREEZE)}
I = {}
wc = (0.0, None)
CSER = {}
for nm, (lev, vt, off, lag, cost) in CS.items():
    W = weights(P, S_full, lev, vt, off)
    for mode in ["Fri", "tranche"]:
        r, _ = run(P, W, lev, "fri" if mode == "Fri" else "tranche", lag, cost)
        CSER[(nm, mode)] = r
        for wn, (a, b) in CW.items():
            m = mets(r, RF, a, b)
            for k in ["cagr", "maxdd", "calmar", "ex_sharpe"]:
                dlt = abs(float(m[k]) - CRIT[nm][mode][wn][k])
                if dlt > wc[0]:
                    wc = (dlt, f"{nm}|{mode}|{wn}|{k}")
I["critic_metrics_max_abs_diff"] = wc
I["F2_tranche_IS"] = {k: CRIT["F2-IEF (QLD vt.30)"]["tranche"]["IS"][k] for k in ["cagr", "maxdd", "calmar"]}
I["F2_tranche_2000"] = {k: CRIT["F2-IEF (QLD vt.30)"]["tranche"]["2000-frz"][k] for k in ["cagr", "maxdd", "calmar"]}
I["F1_8bp_tranche_IS"] = {k: CRIT["F1-IEF lag2 + UCITS-level costs 8/8bp"]["tranche"]["IS"][k] for k in ["cagr", "maxdd", "calmar"]}
I["F1_15bp_ucits_IS_saved"] = {k: SAVED["candidates"]["F1-IEF"]["ucits"]["IS88"][k] for k in ["cagr", "maxdd", "calmar"]}
I["critic_ucits_cost_model"] = UC


def critic_boot(r1, r2, n=2000, block=63, seed=0):
    m = (P.index >= START) & (P.index <= FREEZE)
    a = np.column_stack([r1.values[m], r2.values[m]]); T = len(a); rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block)); dc, dk, dd1, dd2 = [], [], [], []
    for _ in range(n):
        stt = rng.integers(0, T, nb)
        ix = (stt[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = a[ix]; e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        mdd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = g / np.abs(mdd)
        dc.append(g[0] - g[1]); dk.append(cal[0] - cal[1]); dd1.append(mdd[0]); dd2.append(mdd[1])
    dk = np.array(dk); dc = np.array(dc)
    return dict(dcalmar=[float(dk.mean()), float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5))],
                p_dcalmar_le0=float((dk <= 0).mean()), dcagr_mean=float(dc.mean()),
                maxdd_median=[float(np.median(dd1)), float(np.median(dd2))])


q = BEN["QQQ B&H"]
for nm in ["F1-IEF lag2 + UCITS-level costs 8/8bp", "F2-IEF lag2 + UCITS-level costs 15/8bp"]:
    mine = critic_boot(CSER[(nm, "tranche")], q)
    ref = CRIT["bootstrap_IS"][f"{nm} [tranche] minus QQQ"]
    I[f"boot63|{nm}"] = dict(mine=mine, critic=dict(dcalmar=ref["dcalmar"], p=ref["p_dcalmar_le0"], mdd_med=ref["maxdd_median"]))
OUT["I_critic"] = I
pr("I", json.dumps(I))
OUT["runtime_s"] = round(time.time() - T0, 1)
json.dump(OUT, open(os.path.join(SCR, "integrity_checks.json"), "w"), indent=1, default=str)
pr("done", OUT["runtime_s"])
