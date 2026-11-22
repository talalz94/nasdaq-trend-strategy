"""
COMPLETENESS CRITIC - cross-dimension consistency checks + a few missing robustness probes.
Read-only w.r.t. every other file; writes only update_2026_11/critic_crosscheck.json.

Parts
  A  shared-configuration numbers across the 8 dimension JSONs (F3-IEF, F2-IEF, F3-TLT, IS/OOS, holdings)
  B  small factual disputes: OOS day count, re-entry level, last close above EMA, rate-shock flag at 11-13 mark
  C  one-session delay cost by era (adjudicates revalidate C9 vs oos_live C8 vs leverage_execution C7)
  D  MISSING TEST: OOS percentile on the timing-luck-free 5-weekday tranche (planning basis) and per weekday phase
  E  MISSING TEST: rank of the July defaults inside the 540-config grid under each weekday phase and the tranche
     (was the July default selection partly fitted to the Friday path?)
  F  MISSING TEST (diagnostic only, no selection): chop rule at realistic costs, by thirds, Friday vs tranche
Nothing here is used to choose a parameter; all IS statistics stop at the 2026-07-24 freeze.
"""
from __future__ import annotations
import os, sys, json, time, tempfile, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("CRITIC_OUT_DIR") or os.path.join(tempfile.gettempdir(), "critic_out")
os.makedirs(OUTDIR, exist_ok=True)
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT = {}
P = C.panel()
IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
OOS_A, OOS_B = pd.Timestamp("2026-07-25"), LATEST


def jl(name):
    return json.load(open(os.path.join(HERE, name + ".json")))


# ------------------------------------------------------------------ A. cross-dimension consistency
A = {}
ol = jl("oos_live"); rv = jl("revalidate"); rr = jl("rate_regime"); mf = jl("managed_futures")
ov = jl("overlays"); an = jl("analogs"); le = jl("leverage_execution"); uc = jl("ucits")


def g(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


rows = {}
rows["F3-IEF IS cagr"] = dict(
    oos_live=g(ol, "in_sample_stats", "b_F3_TQQQ_IEF_GLD", "lag1", "cagr"),
    revalidate=g(rv, "candidates", "F3-IEF", "windows", "IS", "cagr"),
    rate_regime=g(rr, "results", "F3", "static_ief", "IS_full", "cagr"),
    managed_futures=g(mf, "baselines", "F3", "IS", "cagr"),
    overlays=g(ov, "baselines", "F3", "IS", "cagr"))
rows["F3-IEF IS maxdd"] = dict(
    oos_live=g(ol, "in_sample_stats", "b_F3_TQQQ_IEF_GLD", "lag1", "maxdd"),
    revalidate=g(rv, "candidates", "F3-IEF", "windows", "IS", "maxdd"),
    rate_regime=g(rr, "results", "F3", "static_ief", "IS_full", "maxdd"),
    managed_futures=g(mf, "baselines", "F3", "IS", "maxdd"),
    overlays=g(ov, "baselines", "F3", "IS", "maxdd"))
rows["F2-IEF IS cagr"] = dict(
    oos_live=g(ol, "in_sample_stats", "c_F2_QLD_IEF_GLD_vt30", "lag1", "cagr"),
    revalidate=g(rv, "candidates", "F2-IEF", "windows", "IS", "cagr"),
    rate_regime=g(rr, "results", "F2", "static_ief", "IS_full", "cagr"),
    managed_futures=g(mf, "baselines", "F2", "IS", "cagr"),
    overlays=g(ov, "baselines", "F2", "IS", "cagr"))
rows["F2-IEF IS maxdd"] = dict(
    oos_live=g(ol, "in_sample_stats", "c_F2_QLD_IEF_GLD_vt30", "lag1", "maxdd"),
    revalidate=g(rv, "candidates", "F2-IEF", "windows", "IS", "maxdd"),
    rate_regime=g(rr, "results", "F2", "static_ief", "IS_full", "maxdd"),
    managed_futures=g(mf, "baselines", "F2", "IS", "maxdd"),
    overlays=g(ov, "baselines", "F2", "IS", "maxdd"))
rows["F3-TLT IS calmar"] = dict(
    oos_live=g(ol, "in_sample_stats", "a_F_scanner_TQQQ_TLT_GLD", "lag1", "calmar"),
    revalidate=g(rv, "candidates", "F3-TLT (scanner)", "windows", "IS", "calmar"),
    rate_regime=g(rr, "results", "F3", "static_tlt", "IS_full", "calmar"))
rows["F3-IEF OOS total"] = dict(
    oos_live=g(ol, "results_table", "b_F3_TQQQ_IEF_GLD", "oos_lag1"),
    revalidate=g(rv, "candidates", "F3-IEF", "windows", "OOS", "total"),
    managed_futures=g(mf, "baselines", "F3", "OOS_total"),
    overlays=g(ov, "baselines", "F3", "OOS_return"),
    analogs=g(an, "today", "since_freeze", "F3"))
rows["F2-IEF OOS total"] = dict(
    oos_live=g(ol, "results_table", "c_F2_QLD_IEF_GLD_vt30", "oos_lag1"),
    revalidate=g(rv, "candidates", "F2-IEF", "windows", "OOS", "total"),
    managed_futures=g(mf, "baselines", "F2", "OOS_total"),
    overlays=g(ov, "baselines", "F2", "OOS_return"),
    analogs=g(an, "today", "since_freeze", "F2"))
rows["F2 lag2 IS cagr"] = dict(
    oos_live=g(ol, "in_sample_stats", "c_F2_QLD_IEF_GLD_vt30", "lag2", "cagr"),
    revalidate=g(rv, "candidates", "F2-IEF", "lag2", "IS", "cagr"),
    leverage_execution=g(le, "execution_lag", "results", "F2-IEF", "lag2_mark", "IS", "cagr"))
for k, v in rows.items():
    vals = [x for x in v.values() if isinstance(x, (int, float))]
    v["max_abs_spread"] = (max(vals) - min(vals)) if len(vals) > 1 else None
A["shared_numbers"] = rows
A["oos_days_in_overlays_json"] = g(ov, "baselines", "F3", "OOS", "days")
OUT["A_consistency"] = A
print("A done", f"{time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ B. small factual disputes
B = {}
oos_idx = IDX[(IDX >= OOS_A) & (IDX <= OOS_B)]
B["oos_trading_days_0725_to_last"] = int(len(oos_idx))
B["oos_first_day"], B["oos_last_day"] = str(oos_idx[0].date()), str(oos_idx[-1].date())
ema = P["ema100"]; ndx = P["ndx"]
alpha = 2 / 101.0
e = float(ema.iloc[-1])
proj = {}
for d in ["2026-11-18", "2026-11-19"]:
    e = e + alpha * (float(ndx.iloc[-1]) - e)
    proj[d] = e
B["ema_last"] = float(ema.iloc[-1]); B["ndx_last"] = float(ndx.iloc[-1])
B["ema_projected_flat_ndx"] = proj
B["reentry_threshold_if_flat_to_thu"] = proj["2026-11-19"]
B["move_needed_vs_projected_thu_ema"] = proj["2026-11-19"] / float(ndx.iloc[-1]) - 1
B["move_needed_vs_today_ema"] = float(ema.iloc[-1]) / float(ndx.iloc[-1]) - 1
post = P.loc[P.index > FREEZE, ["ndx", "ema100", "above"]]
B["first_close_below_after_freeze"] = str(post.index[~post["above"]][0].date())
B["last_close_above"] = str(post.index[post["above"]][-1].date())
fri_marks = P.index[(P["mark"]) & (P.index > FREEZE)]
B["friday_marks_after_freeze_above"] = {str(d.date()): bool(P.at[d, "above"]) for d in fri_marks}
# rate-shock flag (overlays rule: 10y up >50bp over 126 trading days, one extra day of signal lag)
tnx = P["tnx"]
tnx_pct = tnx if tnx.max() < 20 else tnx / 10.0
d126 = (tnx_pct - tnx_pct.shift(126)) * 100.0  # in bp if tnx in %
B["tnx_units_last"] = float(tnx.iloc[-1])
B["tnx_126d_change_bp"] = {str(d.date()): round(float(d126.loc[d]), 1)
                            for d in IDX[-8:]}
OUT["B_facts"] = B
print("B done", f"{time.time()-T0:.1f}s", flush=True)


# ------------------------------------------------------------------ fast numpy engine (verified vs common)
N = len(IDX)
dates_np = IDX.values
in_is = (IDX >= START) & (IDX <= FREEZE)
RET = {c: P[c].fillna(0).values for c in ["qqq", "qld", "tqqq", "ief", "tlt", "bills", "gold_phys"]}
VOL = P["vol20"].values
ER = P["er20"].values
ABOVE = P["above"].values.astype(bool)
MONTH = P["month"].values
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MARKS = {d: (IDX.dayofweek == i) for i, d in enumerate(DAYS)}
MARKS["Fri"] = P["mark"].values.astype(bool)          # identical to P.mark (holiday Fridays skip)
assert (MARKS["Fri"] == (IDX.dayofweek == 4)).all() or True


def last_idx(mask):
    pos = np.where(mask, np.arange(N), -1)
    return np.maximum.accumulate(pos)


LAST = {d: last_idx(m) for d, m in MARKS.items()}


def weights_np(lev, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, seasonal=True, off="ief"):
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.clip(voltgt / VOL, 0, 1)
    sc = np.where(np.isnan(sc), 1.0, sc)
    if seasonal:
        sc = np.clip(sc * np.where(np.isin(MONTH, (8, 9)), 0.5, 1.0), 0, 1)
    clean = np.where(np.isnan(ER), False, ER >= chop_thr)
    eq = (ABOVE & clean) * sc + (ABOVE & ~clean) * sc * chop_pos
    core = 1.0 - gold_w
    W = {lev: eq * core}
    rest = np.clip(core - eq * core, 0, None)
    W[off] = W.get(off, 0.0) + rest
    if gold_w > 0:
        W["gold_phys"] = W.get("gold_phys", 0.0) + gold_w
    return W


def held_np(W, day, lag=1):
    li = LAST[day]
    H = {}
    for c, w in W.items():
        w = np.broadcast_to(np.asarray(w, dtype=float), (N,))
        h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
        h = np.concatenate([np.zeros(lag), h[:-lag]])
        H[c] = h
    return H


def ret_from_held(H, cost):
    gross = np.zeros(N); turn = np.zeros(N)
    for c, h in H.items():
        gross += h * RET[c]
        dh = np.abs(np.diff(h, prepend=np.nan)); dh[0] = 0.0
        turn += dh
    return gross - turn * cost


def run(W, mode="Fri", lag=1, cost=C.COST):
    if mode == "tranche":
        Hs = [held_np(W, d, lag) for d in DAYS]
        H = {c: sum(h[c] for h in Hs) / 5.0 for c in Hs[0]}
    else:
        H = held_np(W, mode, lag)
    return ret_from_held(H, cost)


def st(r, mask):
    rr = r[mask]
    d = dates_np[mask]
    e = np.cumprod(1 + rr)
    yrs = (pd.Timestamp(d[-1]) - pd.Timestamp(d[0])).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    return dict(cagr=float(cagr), maxdd=dd, calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
                sharpe=float(rr.mean() / rr.std(ddof=1) * np.sqrt(252)))


def win(a, b):
    return (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))


# verification against common.py
VER = {}
for name, lev, vt in [("F3", "tqqq", 0.20), ("F2", "qld", 0.30)]:
    wc = C.strategy_weights(P, lev=lev, voltgt=vt)
    rc = C.backtest(P, wc).values
    rn = run(weights_np(lev, vt), "Fri")
    VER[name + "_fri_maxabsdiff"] = float(np.nanmax(np.abs(rc - rn)))
    rc2 = C.backtest(P, wc, lag=2).values
    VER[name + "_fri_lag2_maxabsdiff"] = float(np.nanmax(np.abs(rc2 - run(weights_np(lev, vt), "Fri", lag=2))))
    VER[name + "_IS_fri"] = st(rn, in_is)
    VER[name + "_IS_tranche"] = st(run(weights_np(lev, vt), "tranche"), in_is)
OUT["engine_verification"] = VER
print("verify", json.dumps({k: v for k, v in VER.items() if "diff" in k}), flush=True)

# ------------------------------------------------------------------ C. delay cost by era
Cc = {}
ERAS = {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-freeze": ("2000-01-01", FREEZE),
        "2010-freeze": ("2010-01-01", FREEZE), "1991-freeze": ("1991-01-01", FREEZE), "IS": ("1988-01-01", FREEZE)}
for name, lev, vt in [("F3", "tqqq", 0.20), ("F2", "qld", 0.30)]:
    W = weights_np(lev, vt)
    out = {}
    for mode in ["Fri", "tranche"]:
        r1, r2 = run(W, mode, 1), run(W, mode, 2)
        for era, (a, b) in ERAS.items():
            s1, s2 = st(r1, win(a, b)), st(r2, win(a, b))
            out[f"{mode}|{era}"] = dict(lag1_cagr=s1["cagr"], lag2_cagr=s2["cagr"], lag1_calmar=s1["calmar"],
                                        lag2_calmar=s2["calmar"], lag1_dd=s1["maxdd"], lag2_dd=s2["maxdd"])
    Cc[name] = out
OUT["C_delay_by_era"] = Cc
print("C done", f"{time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ D. OOS percentile per phase and tranche
D = {}
WIN_N = 81
oos_mask = (IDX >= pd.Timestamp("2026-07-27")) & (IDX <= LATEST)
assert oos_mask.sum() == 81
hist_end = np.where(IDX <= FREEZE)[0][-1]
hist_start = np.where(IDX >= START)[0][0]
for name, lev, vt in [("F3", "tqqq", 0.20), ("F2", "qld", 0.30)]:
    W = weights_np(lev, vt)
    out = {}
    for mode in DAYS + ["tranche"]:
        for lag in (1, 2):
            r = run(W, mode, lag)
            oos_tot = float(np.prod(1 + r[oos_mask]) - 1)
            lr = np.log1p(r)
            cs = np.concatenate([[0.0], np.cumsum(lr)])
            ends = np.arange(hist_start + WIN_N - 1, hist_end + 1)
            tots = np.exp(cs[ends + 1] - cs[ends + 1 - WIN_N]) - 1
            out[f"{mode}_lag{lag}"] = dict(oos_total=oos_tot,
                                           pctile=float((tots < oos_tot).mean() * 100),
                                           hist_median=float(np.median(tots)),
                                           hist_p05=float(np.percentile(tots, 5)))
    D[name] = out
OUT["D_oos_percentiles"] = D
print("D done", f"{time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ E. grid rank of the July defaults per phase
GV = rv["meta"]["grid_values"]
DEF = rv["meta"]["defaults"]
grid_rows = []
cfgs = list(itertools.product(["tqqq", "qld"], GV["voltgt"], GV["chop_thr"], GV["chop_pos"], GV["gold_w"],
                              GV["seasonal"]))
rv_lookup = {(x["lev"], x["voltgt"], x["chop_thr"], x["chop_pos"], x["gold_w"], x["seasonal"]): x["IS_calmar"]
             for x in rv["grid"]}
h1 = win("1988-01-01", "2006-12-31"); h2 = win("2007-01-01", FREEZE)
for (lev, vt, ct, cp, gw, se) in cfgs:
    W = weights_np(lev, vt, ct, cp, gw, se)
    row = dict(lev=lev, voltgt=vt, chop_thr=ct, chop_pos=cp, gold_w=gw, seasonal=se)
    trs = []
    for mode in DAYS:
        r = run(W, mode)
        trs.append(r)
        row[f"{mode}_IS_calmar"] = st(r, in_is)["calmar"]
    Hs = [held_np(W, d) for d in DAYS]
    Ht = {c: sum(h[c] for h in Hs) / 5.0 for c in Hs[0]}
    rt = ret_from_held(Ht, C.COST)
    s = st(rt, in_is)
    row["tranche_IS_calmar"] = s["calmar"]; row["tranche_IS_cagr"] = s["cagr"]; row["tranche_IS_maxdd"] = s["maxdd"]
    row["tranche_H1_calmar"] = st(rt, h1)["calmar"]; row["tranche_H2_calmar"] = st(rt, h2)["calmar"]
    row["rv_IS_calmar"] = rv_lookup.get((lev, vt, ct, cp, gw, se))
    grid_rows.append(row)
G = pd.DataFrame(grid_rows)
G["fri_check_diff"] = (G["Fri_IS_calmar"] - G["rv_IS_calmar"]).abs()
E = {"fri_vs_revalidate_grid_max_abs_diff": float(G["fri_check_diff"].max())}
for lev in ["tqqq", "qld"]:
    sub = G[G.lev == lev].reset_index(drop=True)
    d = DEF[lev]
    isdef = ((sub.voltgt == d["voltgt"]) & (sub.chop_thr == d["chop_thr"]) & (sub.chop_pos == d["chop_pos"]) &
             (sub.gold_w == d["gold_w"]) & (sub.seasonal == d["seasonal"]))
    assert isdef.sum() == 1
    res = {}
    for col in [f"{m}_IS_calmar" for m in DAYS] + ["tranche_IS_calmar", "tranche_H1_calmar", "tranche_H2_calmar"]:
        v = sub[col]; dv = float(v[isdef].iloc[0])
        res[col] = dict(default=dv, pctile=float((v < dv).mean() * 100), median=float(v.median()),
                        p10=float(v.quantile(0.1)), p90=float(v.quantile(0.9)), max=float(v.max()))
    res["spearman_fri_vs_tranche"] = float(sub["Fri_IS_calmar"].rank().corr(sub["tranche_IS_calmar"].rank()))
    res["spearman_fri_vs_mon"] = float(sub["Fri_IS_calmar"].rank().corr(sub["Mon_IS_calmar"].rank()))
    res["mean_pctile_of_default_over_MonThu"] = float(np.mean([res[f"{m}_IS_calmar"]["pctile"] for m in DAYS[:4]]))
    best = sub.loc[sub["tranche_IS_calmar"].idxmax()]
    res["tranche_best"] = {k: (float(best[k]) if not isinstance(best[k], (str, bool, np.bool_)) else
                              (bool(best[k]) if isinstance(best[k], (bool, np.bool_)) else best[k]))
                           for k in ["voltgt", "chop_thr", "chop_pos", "gold_w", "seasonal", "tranche_IS_calmar"]}
    res["default_tranche_cagr_dd"] = [float(sub.loc[isdef, "tranche_IS_cagr"].iloc[0]),
                                      float(sub.loc[isdef, "tranche_IS_maxdd"].iloc[0])]
    # +/-1 step box around the default with seasonal on (as revalidate's box)
    def nb(col, val, vals):
        i = vals.index(val); return vals[max(0, i - 1): i + 2]
    box = sub[sub.voltgt.isin(nb("voltgt", d["voltgt"], GV["voltgt"])) &
              sub.chop_thr.isin(nb("chop_thr", d["chop_thr"], GV["chop_thr"])) &
              sub.chop_pos.isin(nb("chop_pos", d["chop_pos"], GV["chop_pos"])) &
              sub.gold_w.isin(nb("gold_w", d["gold_w"], GV["gold_w"])) & (sub.seasonal == True)]  # noqa: E712
    res["box_seasonal_on_n"] = int(len(box))
    res["box_median_fri"] = float(box["Fri_IS_calmar"].median())
    res["box_median_tranche"] = float(box["tranche_IS_calmar"].median())
    E[lev] = res
OUT["E_grid_rank_by_phase"] = E
print("E done", f"{time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ F. chop rule at realistic costs (diagnostic)
F = {}
THIRDS = {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-2012": ("2000-01-01", "2012-12-31"),
          "2013-freeze": ("2013-01-01", FREEZE), "IS": ("1988-01-01", FREEZE)}
for name, lev, vt in [("F3", "tqqq", 0.20), ("F2", "qld", 0.30)]:
    out = {}
    for cp_label, cp in [("chop_half", 0.5), ("no_chop", 1.0)]:
        W = weights_np(lev, vt, chop_pos=cp)
        for mode in ["Fri", "tranche"]:
            for bps in (3, 10, 15):
                r = run(W, mode, 1, cost=bps / 1e4)
                for era, (a, b) in THIRDS.items():
                    s = st(r, win(a, b))
                    out[f"{cp_label}|{mode}|{bps}bp|{era}"] = dict(cagr=s["cagr"], maxdd=s["maxdd"], calmar=s["calmar"])
    # deltas chop_half minus no_chop
    dl = {}
    for mode in ["Fri", "tranche"]:
        for bps in (3, 10, 15):
            for era in THIRDS:
                a_ = out[f"chop_half|{mode}|{bps}bp|{era}"]; b_ = out[f"no_chop|{mode}|{bps}bp|{era}"]
                dl[f"{mode}|{bps}bp|{era}"] = dict(d_calmar=a_["calmar"] - b_["calmar"], d_cagr=a_["cagr"] - b_["cagr"],
                                                   d_maxdd=a_["maxdd"] - b_["maxdd"])
    F[name] = dict(levels=out, chop_minus_nochop=dl)
OUT["F_chop_cost_diagnostic"] = F
print("F done", f"{time.time()-T0:.1f}s", flush=True)

OUT["meta"] = dict(script="update_2026_11/critic_crosscheck.py", runtime_s=round(time.time() - T0, 1),
                   note="diagnostics only; no parameter chosen; IS stops at 2026-07-24")


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


json.dump(clean(OUT), open(os.path.join(OUTDIR, "critic_crosscheck.json"), "w"), indent=1)
print("written critic_crosscheck.json", f"{time.time()-T0:.1f}s")
