"""
r2_verify_friday_effect_robustness_ndx.py -- adversarial ROBUSTNESS checks of r2_friday_effect (NDX part).

Read-only w.r.t. every project file. Writes its numeric output to the path given as argv[1] (scratch).
Engine helpers are copied verbatim from r2_friday_effect.py (that engine == common.backtest, max diff 0, which the
re-run verified); part R1 re-derives the headline books independently with common.backtest / common.stats.

R1  independent reproduction (common.py only): F1/F2/F3 weekday books + tranche, IS/H1/H2, decades (CAGR+Calmar)
R2  criterion-A fragility: (a) half split year, (b) null design (per-week seeds, block 4/13/26/52 + seeds),
    (c) EMA span (per-week placebo at every span), (d) single-episode removal, (e) maxDD episode dates
R3  switch-count placebo (Friday 189 vs 214-217)
R4  selection-adjusted null: max over the July EMA grid {100,125,150,200,250} per random schedule
R5  paired circular block bootstrap of Friday - tranche CAGR (block 21/63/252)
R6  F3 single-day planning numbers under the block-13 null (claim C12 quotes the per-week null for F3)
R7  OOS gap events (claim C13 '2 whipsaw events')
R8  Fri->Mon continuation robustness (sub-eras, trimmed) and the 1985-99 Wednesday mean (NDX vs IXIC vs SPX)
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT_PATH = os.path.abspath(sys.argv[1])
assert not OUT_PATH.startswith(os.path.abspath(HERE)), "write scratch output outside the project"
P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
OOS_A = pd.Timestamp("2026-07-25")
LAST = IDX[-1]
H1 = (START, pd.Timestamp("2006-12-31"))
H2 = (pd.Timestamp("2007-01-01"), FREEZE)
WIN = {"IS": (START, FREEZE), "H1": H1, "H2": H2}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SEED = 20261118
NPL = 1000
TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
OUT = {}
NVAR = {}


def count(k, n=1):
    NVAR[k] = NVAR.get(k, 0) + n


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ engine (verbatim copy of the specialist's)
def last_idx(mask):
    pos = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(pos)


def held(W, mask, lag=1):
    li = last_idx(mask)
    H = np.where((li >= 0)[:, None], W[np.maximum(li, 0)], 0.0)
    if lag:
        H = np.vstack([np.zeros((lag, W.shape[1])), H[:-lag]])
    return H


def run_book(W, R, mask, lag=1, cost=C.COST):
    H = held(W, mask, lag)
    gross = (H * R).sum(1)
    turn = np.abs(np.diff(H, axis=0, prepend=np.zeros((1, H.shape[1])))).sum(1)
    return gross - turn * cost


def wm(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def st1(dates, r, a, b):
    m = wm(dates, a, b)
    rr = r[m]; d = dates[m]
    e = np.cumprod(1 + rr)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    sd = rr.std(ddof=1)
    return dict(cagr=float(cagr), maxdd=dd, calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
                sharpe=float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan)


def stm(dates, Rm, a, b):
    m = wm(dates, a, b)
    X = Rm[:, m]; d = dates[m]
    E = np.cumprod(1 + X, axis=1)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = E[:, -1] ** (1 / yrs) - 1
    dd = (E / np.maximum.accumulate(E, axis=1) - 1).min(axis=1)
    return cagr, dd, cagr / np.abs(dd)


def week_codes(dates):
    return pd.factorize(pd.DatetimeIndex(dates).to_period("W-FRI"))[0]


def placebo_masks(dates, S, rng, block_weeks=1):
    wk = week_codes(dates)
    dow = pd.DatetimeIndex(dates).dayofweek.values
    grp = wk // block_weeks
    draws = rng.integers(0, 5, size=(S, grp.max() + 1))
    return dow[None, :] == draws[:, grp]


def placebo_returns(W, R, masks, lag=1, cost=C.COST, chunk=100):
    """full matrix of daily net returns, one row per schedule (same arithmetic as the specialist's placebo_stats)."""
    T = R.shape[0]; ar = np.arange(T); k = W.shape[1]
    out = np.empty((masks.shape[0], T))
    for i in range(0, masks.shape[0], chunk):
        mk = masks[i:i + chunk]; s = mk.shape[0]
        li = np.maximum.accumulate(np.where(mk, ar[None, :], -1), axis=1)
        H = np.where((li >= 0)[..., None], W[np.maximum(li, 0)], 0.0)
        H = np.concatenate([np.zeros((s, lag, k)), H[:, :-lag]], axis=1)
        gross = (H * R[None]).sum(2)
        turn = np.abs(np.diff(H, axis=1, prepend=np.zeros((s, 1, k)))).sum(2)
        out[i:i + s] = gross - turn * cost
    return out


def pctile(dist, x):
    dist = np.asarray(dist)
    return float(100 * ((dist < x).mean() + 0.5 * (dist == x).mean()))


def f_weights(lvl, month, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, span=100):
    lvl = pd.Series(lvl)
    ema = lvl.ewm(span=span, adjust=False).mean()
    above = (lvl > ema)
    vol20 = lvl.pct_change().rolling(20).std() * np.sqrt(252)
    er20 = lvl.diff(20).abs() / lvl.diff().abs().rolling(20).sum().replace(0, np.nan)
    scale = (voltgt / vol20).clip(0, 1).fillna(1.0)
    scale = (scale * np.where(np.isin(month, (8, 9)), 0.5, 1.0)).clip(0, 1)
    clean = (er20 >= chop_thr).fillna(False)
    eq = (above & clean) * scale + (above & ~clean) * scale * chop_pos
    core = 1.0 - gold_w
    lev = (eq * core).values
    off = np.clip(core - lev, 0, None)
    gold = np.full(len(lvl), gold_w)
    return np.column_stack([lev, off, gold]), above.values.astype(bool)


DOW = IDX.dayofweek.values
DM = {d: (DOW == i) for i, d in enumerate(DAYS)}


def books_from(W, R, lag=1):
    rs = {d: run_book(W, R, DM[d], lag=lag) for d in DAYS}
    rs["tranche"] = sum(rs[d] for d in DAYS) / 5
    return rs


# ================================================================== R1 independent reproduction via common.py
log("R1 independent reproduction via common.backtest/common.stats")
marks = {d: pd.Series(DM[d], index=IDX) for d in DAYS}
R1 = {}
COMMON_BOOKS = {}
for t, (lev, vt) in TIERS.items():
    w = C.strategy_weights(P, lev=lev, voltgt=vt)
    rs = {d: C.backtest(P, w, mark=marks[d]) for d in DAYS}
    rs["tranche"] = sum(rs[d] for d in DAYS) / 5
    COMMON_BOOKS[t] = rs
    row = {}
    for wn, (a, b) in WIN.items():
        row[wn] = {k: {m: round(v, 5) for m, v in C.stats(rs[k], a, b).items() if m in ("cagr", "maxdd", "calmar")}
                   for k in ("Fri", "tranche")}
        row[wn]["d_cagr_pp"] = round((row[wn]["Fri"]["cagr"] - row[wn]["tranche"]["cagr"]) * 100, 3)
        row[wn]["d_calmar"] = round(row[wn]["Fri"]["calmar"] - row[wn]["tranche"]["calmar"], 4)
    dec = {}
    for k, (a, b) in C.DECADES.items():
        a = max(pd.Timestamp(a), START); b = min(pd.Timestamp(b), FREEZE)
        sf, stt = C.stats(rs["Fri"], a, b), C.stats(rs["tranche"], a, b)
        dec[k] = dict(d_cagr_pp=round((sf["cagr"] - stt["cagr"]) * 100, 2), d_calmar=round(sf["calmar"] - stt["calmar"], 3))
    row["decades"] = dec
    row["oos_total"] = {k: round(C.period_return(rs[k], OOS_A, LAST) * 100, 2) for k in DAYS + ["tranche"]}
    R1[t] = row
    log(t, "IS", row["IS"], "H1 d", row["H1"]["d_cagr_pp"], row["H1"]["d_calmar"], "H2 d", row["H2"]["d_cagr_pp"],
        row["H2"]["d_calmar"])
    log(t, "decades", dec)
    log(t, "OOS", row["oos_total"])
OUT["R1_independent_reproduction"] = R1

# ================================================================== F2 setup
wdf = C.strategy_weights(P, lev="qld", voltgt=0.30)
W2 = wdf.values; R2 = P[list(wdf.columns)].fillna(0).values
BK2 = books_from(W2, R2)
assert np.allclose(BK2["Fri"], COMMON_BOOKS["F2"]["Fri"].values, atol=1e-14)
ABOVE = P["above"].values.astype(bool)

# ================================================================== R2a split-year sensitivity of criterion A
log("R2a criterion A vs half split year (per-week placebo, spec seed)")
rng = np.random.default_rng(SEED)
MASK_W = placebo_masks(IDX, NPL, rng)
RM_W = placebo_returns(W2, R2, MASK_W); count("placebo_perweek_F2_specseed_rebuilt", NPL)
# check: identical to the specialist's placebo percentile in IS/H1/H2
chk = {}
for wn, ab in WIN.items():
    cg, dd, cal = stm(IDX, RM_W, *ab)
    chk[wn] = dict(pct_calmar=pctile(cal, st1(IDX, BK2["Fri"], *ab)["calmar"]),
                   pct_cagr=pctile(cg, st1(IDX, BK2["Fri"], *ab)["cagr"]))
OUT["R2_placebo_rebuild_check"] = chk
log("rebuild check (spec: IS 100.0/99.9, H1 98.0/99.3, H2 99.1/99.6):", chk)


def a_verdict(rf, rt, RM, h1, h2):
    out = {}
    ok = True
    for nm, ab in (("H1", h1), ("H2", h2)):
        sf, stt = st1(IDX, rf, *ab), st1(IDX, rt, *ab)
        cg, dd, cal = stm(IDX, RM, *ab)
        pc = pctile(cal, sf["calmar"])
        dcg = (sf["cagr"] - stt["cagr"]) * 100; dcal = sf["calmar"] - stt["calmar"]
        out[nm] = dict(d_cagr_pp=round(dcg, 3), d_calmar=round(dcal, 4), pct_calmar=round(pc, 1),
                       pct_cagr=round(pctile(cg, sf["cagr"]), 1))
        ok = ok and (dcg > 0) and (dcal > 0) and (pc >= 95)
    out["A_holds"] = bool(ok)
    return out


split = {}
for Y in [1994, 1996, 1998, 2000, 2002, 2004, 2007, 2009, 2010, 2012, 2014, 2016]:
    h1 = (START, pd.Timestamp(f"{Y - 1}-12-31")); h2 = (pd.Timestamp(f"{Y}-01-01"), FREEZE)
    split[Y] = a_verdict(BK2["Fri"], BK2["tranche"], RM_W, h1, h2)
    log(f"  split {Y}: {split[Y]}")
OUT["R2a_split_year"] = split
OUT["R2a_split_year_summary"] = dict(n=len(split), n_A_holds=int(sum(v["A_holds"] for v in split.values())),
                                     failing=[y for y, v in split.items() if not v["A_holds"]])

# ================================================================== R2b null-design sensitivity
log("R2b null design: per-week seeds, block nulls 4/13/26/52 weeks")
nulls = {}
fri_s = {wn: st1(IDX, BK2["Fri"], *ab) for wn, ab in WIN.items()}
tr_s = {wn: st1(IDX, BK2["tranche"], *ab) for wn, ab in WIN.items()}


def null_row(RM):
    row = {}
    for wn, ab in WIN.items():
        cg, dd, cal = stm(IDX, RM, *ab)
        row[wn] = dict(pct_calmar=round(pctile(cal, fri_s[wn]["calmar"]), 1), pct_cagr=round(pctile(cg, fri_s[wn]["cagr"]), 1),
                       null_cagr_mean=round(float(cg.mean()) * 100, 2), null_cagr_p50=round(float(np.median(cg)) * 100, 2),
                       null_maxdd_p50=round(float(np.median(dd)) * 100, 1), null_maxdd_p5=round(float(np.percentile(dd, 5)) * 100, 1),
                       null_calmar_p50=round(float(np.median(cal)), 3),
                       tranche_pct_cagr=round(pctile(cg, tr_s[wn]["cagr"]), 1))
    return row


nulls["perweek_seed_spec"] = null_row(RM_W)
for k, sd in enumerate([SEED + 7001, SEED + 7002]):
    m_ = placebo_masks(IDX, NPL, np.random.default_rng(sd))
    nulls[f"perweek_seed_alt{k + 1}"] = null_row(placebo_returns(W2, R2, m_)); count("placebo_perweek_altseed", NPL)
for bw, seeds in [(4, [SEED + 7101]), (13, [SEED + 100, SEED + 7201, SEED + 7202, SEED + 7203]), (26, [SEED + 7301]),
                  (52, [SEED + 7401])]:
    for sd in seeds:
        m_ = placebo_masks(IDX, NPL, np.random.default_rng(sd), block_weeks=bw)
        RMb = placebo_returns(W2, R2, m_); count(f"placebo_block{bw}", NPL)
        nulls[f"block{bw}_seed{'spec' if sd == SEED + 100 else sd - SEED}"] = null_row(RMb)
        if bw == 13 and sd == SEED + 100:
            RM_B13 = RMb; MASK_B13 = m_
for k, v in nulls.items():
    log(f"  {k:24s} Fri Calmar pct IS/H1/H2 {v['IS']['pct_calmar']}/{v['H1']['pct_calmar']}/{v['H2']['pct_calmar']} "
        f"| CAGR pct {v['IS']['pct_cagr']}/{v['H1']['pct_cagr']}/{v['H2']['pct_cagr']} | null CAGR mean IS {v['IS']['null_cagr_mean']}"
        f" maxDD p50 {v['IS']['null_maxdd_p50']} | tranche CAGR pct {v['IS']['tranche_pct_cagr']}")
OUT["R2b_null_design"] = nulls
# A placebo leg under each null
OUT["R2b_A_placebo_leg_by_null"] = {k: bool(v["H1"]["pct_calmar"] >= 95 and v["H2"]["pct_calmar"] >= 95) for k, v in nulls.items()}

# ================================================================== R2c EMA span: full criterion A at each span
log("R2c criterion A at each EMA span (per-week placebo, spec seed masks)")
MON = IDX.month.values
R_F2 = P[["qld", "ief", "gold_phys"]].fillna(0).values
span_res = {}
SPAN_IS_CAL_W = {}; SPAN_IS_CAL_B = {}
for span in [50, 75, 100, 125, 150, 200, 250]:
    Ws, ab_s = f_weights(P["ndx"], MON, 0.30, span=span)
    bs = books_from(Ws, R_F2); count("span_weekday_books", 5)
    RMs = placebo_returns(Ws, R_F2, MASK_W); count("span_placebo_perweek", NPL)
    v = a_verdict(bs["Fri"], bs["tranche"], RMs, H1, H2)
    cg, dd, cal = stm(IDX, RMs, START, FREEZE)
    SPAN_IS_CAL_W[span] = cal
    sF = st1(IDX, bs["Fri"], START, FREEZE)
    v["IS_fri_calmar"] = round(sF["calmar"], 3)
    v["IS_pct_calmar"] = round(pctile(cal, sF["calmar"]), 1)
    v["IS_pct_cagr"] = round(pctile(cg, sF["cagr"]), 1)
    RMsb = placebo_returns(Ws, R_F2, MASK_B13); count("span_placebo_block13", NPL)
    cgb, ddb, calb = stm(IDX, RMsb, START, FREEZE)
    SPAN_IS_CAL_B[span] = calb
    v["IS_pct_calmar_block13"] = round(pctile(calb, sF["calmar"]), 1)
    for wn in ("H1", "H2"):
        cgb2, ddb2, calb2 = stm(IDX, RMsb, *WIN[wn])
        v[f"{wn}_pct_calmar_block13"] = round(pctile(calb2, st1(IDX, bs["Fri"], *WIN[wn])["calmar"]), 1)
    mk = np.where(DM["Fri"] & wm(IDX, START, FREEZE))[0]
    v["fri_switches_IS"] = int((ab_s[mk][1:] != ab_s[mk][:-1]).sum())
    v["other_switches_IS"] = {d: int((ab_s[np.where(DM[d] & wm(IDX, START, FREEZE))[0]][1:] !=
                                      ab_s[np.where(DM[d] & wm(IDX, START, FREEZE))[0]][:-1]).sum()) for d in DAYS[:4]}
    span_res[span] = v
    log(f"  span {span:3d}: A={v['A_holds']} H1 {v['H1']} H2 {v['H2']} | IS pct Cal {v['IS_pct_calmar']} (block13 "
        f"{v['IS_pct_calmar_block13']}, H1 {v['H1_pct_calmar_block13']}, H2 {v['H2_pct_calmar_block13']}) | switches Fri "
        f"{v['fri_switches_IS']} others {v['other_switches_IS']}")
OUT["R2c_ema_span_A"] = span_res
OUT["R2c_summary"] = dict(spans_A_holds=[s for s, v in span_res.items() if v["A_holds"]],
                          july_grid_100_250_A_holds=[s for s in (100, 125, 150, 200, 250) if span_res[s]["A_holds"]])

# ================================================================== R4 selection-adjusted null (July grid 100..250)
log("R4 selection-adjusted null: best-of-July-grid Calmar per random schedule")
grid = [100, 125, 150, 200, 250]
fri_best = max(span_res[s]["IS_fri_calmar"] for s in grid)
maxW = np.max(np.vstack([SPAN_IS_CAL_W[s] for s in grid]), axis=0)
maxB = np.max(np.vstack([SPAN_IS_CAL_B[s] for s in grid]), axis=0)
OUT["R4_selection_adjusted"] = dict(
    fri_best_over_grid_IS_calmar=fri_best,
    fri_best_span=[s for s in grid if span_res[s]["IS_fri_calmar"] == fri_best],
    pct_vs_max_over_grid_perweek=round(pctile(maxW, fri_best), 1),
    pct_vs_max_over_grid_block13=round(pctile(maxB, fri_best), 1),
    max_over_grid_block13_p50=round(float(np.median(maxB)), 3), max_over_grid_block13_p95=round(float(np.percentile(maxB, 95)), 3),
    unadjusted_pct_block13_span100=span_res[100]["IS_pct_calmar_block13"])
log("  ", OUT["R4_selection_adjusted"])

# ================================================================== R2d single-episode removal
log("R2d single-episode removal (both books' returns set to 0 inside the episode)")
EPIS = {"1988-89 (first 2y)": ("1988-01-01", "1989-12-31"), "1989-90": ("1989-01-01", "1990-12-31"),
        "1998 (top year)": ("1998-01-01", "1998-12-31"), "2000-02 dot-com": ("2000-03-01", "2002-10-31"),
        "GFC 2007-10..2009-03": ("2007-10-01", "2009-03-31"), "2018": ("2018-01-01", "2018-12-31"),
        "2012": ("2012-01-01", "2012-12-31"), "COVID 2020-02..04": ("2020-02-15", "2020-04-30"),
        "2022": ("2022-01-01", "2022-12-31"), "2025 tariff": ("2025-02-15", "2025-05-31")}
ep_res = {}
for nm, (a, b) in EPIS.items():
    z = wm(IDX, a, b)
    rf = np.where(z, 0.0, BK2["Fri"]); rt = np.where(z, 0.0, BK2["tranche"])
    RMz = np.where(z[None, :], 0.0, RM_W)
    ep_res[nm] = a_verdict(rf, rt, RMz, H1, H2)
    sF, sT = st1(IDX, rf, START, FREEZE), st1(IDX, rt, START, FREEZE)
    ep_res[nm]["IS_d_cagr_pp"] = round((sF["cagr"] - sT["cagr"]) * 100, 3)
    ep_res[nm]["IS_d_calmar"] = round(sF["calmar"] - sT["calmar"], 4)
    log(f"  drop {nm:22s} A={ep_res[nm]['A_holds']} IS d {ep_res[nm]['IS_d_cagr_pp']:+.2f}pp/{ep_res[nm]['IS_d_calmar']:+.3f} "
        f"H1 {ep_res[nm]['H1']} H2 {ep_res[nm]['H2']}")
OUT["R2d_episode_removal"] = ep_res


# ================================================================== R2e maxDD episodes
def dd_info(r, a, b):
    m = wm(IDX, a, b)
    e = np.cumprod(1 + r[m]); d = IDX[m]
    pk = np.maximum.accumulate(e); ddv = e / pk - 1
    j = int(np.argmin(ddv)); i = int(np.argmax(e[:j + 1]))
    return dict(maxdd=round(float(ddv[j]) * 100, 2), peak=str(d[i].date()), trough=str(d[j].date()))


dde = {}
for wn, ab in WIN.items():
    dde[wn] = {k: dd_info(BK2[k], *ab) for k in DAYS + ["tranche"]}
OUT["R2e_maxdd_episodes"] = dde
log("R2e maxDD episodes:", json.dumps(dde))
# Calmar gap if Friday's maxDD were replaced by the tranche's (CAGR advantage only)
OUT["R2e_calmar_gap_with_equal_maxdd"] = {wn: round(fri_s[wn]["cagr"] / abs(tr_s[wn]["maxdd"]) - tr_s[wn]["calmar"], 4)
                                          for wn in WIN}
OUT["R2e_fri_pct_calmar_if_tranche_maxdd"] = {}
for wn, ab in WIN.items():
    cg, dd, cal = stm(IDX, RM_W, *ab)
    OUT["R2e_fri_pct_calmar_if_tranche_maxdd"][wn] = round(pctile(cal, fri_s[wn]["cagr"] / abs(tr_s[wn]["maxdd"])), 1)
log("R2e Calmar gap with equal maxDD:", OUT["R2e_calmar_gap_with_equal_maxdd"], "pct:", OUT["R2e_fri_pct_calmar_if_tranche_maxdd"])

# ================================================================== R3 switch counts under the nulls
log("R3 switch counts")
mis = wm(IDX, START, FREEZE)


def switches(mask):
    mk = np.where(mask & mis)[0]
    return int((ABOVE[mk][1:] != ABOVE[mk][:-1]).sum())


sw_book = {d: switches(DM[d]) for d in DAYS}
sw_w = np.array([switches(MASK_W[i]) for i in range(NPL)])
sw_b = np.array([switches(MASK_B13[i]) for i in range(NPL)])
sw_off = {o: switches((np.arange(N) % 5) == o) for o in range(5)}
OUT["R3_switch_counts"] = dict(books=sw_book, offset_cycles=sw_off,
                               perweek=dict(p5=float(np.percentile(sw_w, 5)), p50=float(np.median(sw_w)), p95=float(np.percentile(sw_w, 95)),
                                            min=int(sw_w.min()), pct_of_fri=pctile(sw_w, sw_book["Fri"])),
                               block13=dict(p5=float(np.percentile(sw_b, 5)), p50=float(np.median(sw_b)), p95=float(np.percentile(sw_b, 95)),
                                            min=int(sw_b.min()), pct_of_fri=pctile(sw_b, sw_book["Fri"])))
log("  ", OUT["R3_switch_counts"])
# do the switch counts explain the placebo Calmar? correlation across block13 schedules
cgb, ddb, calb = stm(IDX, RM_B13, START, FREEZE)
OUT["R3_corr_switches_vs_calmar_block13"] = float(np.corrcoef(sw_b, calb)[0, 1])
OUT["R3_corr_switches_vs_cagr_block13"] = float(np.corrcoef(sw_b, cgb)[0, 1])
log("  corr(switches, Calmar) block13:", OUT["R3_corr_switches_vs_calmar_block13"], "cagr:", OUT["R3_corr_switches_vs_cagr_block13"])

# ================================================================== R5 block bootstrap
log("R5 paired circular block bootstrap Fri - tranche CAGR")
bb = {}
rfS = pd.Series(BK2["Fri"], index=IDX); rtS = pd.Series(BK2["tranche"], index=IDX)
for wn, (a, b) in WIN.items():
    x = rfS[(IDX >= a) & (IDX <= b)]; y = rtS[(IDX >= a) & (IDX <= b)]
    for L in (21, 63, 252):
        bb[f"{wn}_block{L}"] = {k: round(v, 4) for k, v in C.block_bootstrap_cagr_diff(x, y, n=2000, block=L, seed=1, start=a).items()}
        count("bootstrap_runs", 1)
OUT["R5_block_bootstrap"] = bb
log("  ", json.dumps(bb))

# ================================================================== R6 F3 / F1 planning numbers under block-13 null
log("R6 single-day planning numbers under the block-13 null for F1/F3")
r6 = {}
for t in ("F1", "F3"):
    lev, vt = TIERS[t]
    w_ = C.strategy_weights(P, lev=lev, voltgt=vt)
    Wt = w_.values; Rt = P[list(w_.columns)].fillna(0).values
    RMt = placebo_returns(Wt, Rt, MASK_B13); count("placebo_block13_F1F3", NPL)
    cg, dd, cal = stm(IDX, RMt, START, FREEZE)
    RMtw = placebo_returns(Wt, Rt, placebo_masks(IDX, NPL, np.random.default_rng(SEED + list(TIERS).index(t) + 1)))
    count("placebo_perweek_F1F3_rebuilt", NPL)
    cgw, ddw, calw = stm(IDX, RMtw, START, FREEZE)
    r6[t] = dict(block13=dict(cagr_p50=round(float(np.median(cg)) * 100, 2), maxdd_p50=round(float(np.median(dd)) * 100, 1),
                              maxdd_p5=round(float(np.percentile(dd, 5)) * 100, 1), calmar_p50=round(float(np.median(cal)), 3)),
                 perweek_rebuilt=dict(cagr_p50=round(float(np.median(cgw)) * 100, 2), maxdd_p50=round(float(np.median(ddw)) * 100, 1),
                                      calmar_p50=round(float(np.median(calw)), 3)))
cg, dd, cal = stm(IDX, RM_B13, START, FREEZE)
r6["F2"] = dict(block13=dict(cagr_p50=round(float(np.median(cg)) * 100, 2), cagr_p5=round(float(np.percentile(cg, 5)) * 100, 2),
                             cagr_p95=round(float(np.percentile(cg, 95)) * 100, 2), maxdd_p50=round(float(np.median(dd)) * 100, 1),
                             maxdd_p5=round(float(np.percentile(dd, 5)) * 100, 1), calmar_p50=round(float(np.median(cal)), 3)))
OUT["R6_planning_null"] = r6
log("  ", json.dumps(r6))

# ================================================================== R7 OOS gap events
log("R7 OOS gap events")
g = BK2["Fri"] - BK2["tranche"]
mo = wm(IDX, OOS_A, LAST)
gi = np.where(mo)[0]
ev = [dict(date=str(IDX[i].date()), gap_bp=round(float(g[i]) * 1e4, 1), ndx_bp=round(float(P["ndx_ret"].values[i]) * 1e4, 1),
           fri_lev=round(float(held(W2, DM["Fri"])[i, 0]), 3),
           others_lev={d: round(float(held(W2, DM[d])[i, 0]), 3) for d in DAYS[:4]})
      for i in gi if abs(g[i]) >= 0.003]
tot = float(np.prod(1 + BK2["Fri"][mo]) - np.prod(1 + BK2["tranche"][mo]))
OUT["R7_oos_gap_days_ge_30bp"] = ev
OUT["R7_oos"] = dict(days=int(mo.sum()), total_gap_pp=round(tot * 100, 2), sum_daily_gap_pp=round(float(g[mo].sum()) * 100, 2),
                     sum_gap_on_listed_days_pp=round(sum(e["gap_bp"] for e in ev) / 100, 2), n_listed=len(ev))
log("  ", OUT["R7_oos"])
for e_ in ev:
    log("   ", e_)

# ================================================================== R8 Fri->Mon continuation and the Wednesday mean
log("R8 Fri->next-session continuation and 1985-99 Wednesday mean")
Y = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))


def yser(t, col="close"):
    s = Y[Y.ticker == t].set_index("date")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index <= LAST].astype(float)


ndx = P["ndx_ret"].values.astype(float)
spx = P["spx"].values.astype(float); spx[0] = np.nan
ixs = yser("^IXIC")
ixr = ixs.reindex(ixs.index.union(IDX)).ffill().reindex(IDX).pct_change().values
nx = lambda x: np.r_[x[1:], np.nan]  # noqa: E731
r8 = {}
for era, (a, b) in {"1988-1993": ("1988-01-01", "1993-12-31"), "1994-1999": ("1994-01-01", "1999-12-31"),
                    "1988-1999": ("1988-01-01", "1999-12-31"), "2000-2006": ("2000-01-01", "2006-12-31"),
                    "2007-2016": ("2007-01-01", "2016-12-31"), "2017-2026IS": ("2017-01-01", str(FREEZE.date()))}.items():
    m = wm(IDX, a, b) & np.r_[wm(IDX, a, b)[1:], False]
    row = {}
    for nm, x in (("NDX", ndx), ("SPX", spx), ("IXIC", ixr)):
        xn = nx(x); rr = {}
        for i, d in enumerate(DAYS):
            mm = m & DM[d] & np.isfinite(x) & np.isfinite(xn)
            xs, ys = x[mm], xn[mm]
            # trimmed: drop the 10 largest |x| days
            keep = np.argsort(-np.abs(xs))[10:]
            rr[d] = dict(n=int(mm.sum()), pearson=round(float(np.corrcoef(xs, ys)[0, 1]), 3),
                         spearman=round(float(pd.Series(xs).corr(pd.Series(ys), method="spearman")), 3),
                         pearson_trim10=round(float(np.corrcoef(xs[keep], ys[keep])[0, 1]), 3))
        row[nm] = rr
    r8[era] = row
    log(f"  {era}: NDX Fri {row['NDX']['Fri']} | others Pearson {[row['NDX'][d]['pearson'] for d in DAYS[:4]]} "
        f"Spearman {[row['NDX'][d]['spearman'] for d in DAYS[:4]]} | SPX Fri {row['SPX']['Fri']['pearson']} IXIC Fri {row['IXIC']['Fri']['pearson']}")
OUT["R8_next_session_corr"] = r8
wed = {}
for nm, x in (("NDX", ndx), ("SPX", spx), ("IXIC", ixr)):
    m = wm(IDX, "1985-10-03", "1999-12-31") & np.isfinite(x)
    row = {}
    for i, d in enumerate(DAYS):
        xx = x[m & DM[d]]
        row[d] = dict(mean_bp=round(float(xx.mean()) * 1e4, 1), t=round(float(xx.mean() / (xx.std(ddof=1) / np.sqrt(len(xx)))), 2),
                      median_bp=round(float(np.median(xx)) * 1e4, 1), pos=round(float((xx > 0).mean()), 3))
    wed[nm] = row
mm = wm(IDX, "1985-10-03", "1999-12-31") & np.isfinite(ixr)
wed["corr_NDX_IXIC_same_day_by_weekday"] = {d: round(float(np.corrcoef(ndx[mm & DM[d]], ixr[mm & DM[d]])[0, 1]), 3) for d in DAYS}
wed["NDX_minus_IXIC_mean_bp_by_weekday"] = {d: round(float((ndx[mm & DM[d]] - ixr[mm & DM[d]]).mean()) * 1e4, 1) for d in DAYS}
OUT["R8_weekday_means_1985_1999"] = wed
log("  weekday means 1985-99:", json.dumps(wed))

OUT["variants_evaluated_by_this_check"] = dict(NVAR, total=int(sum(NVAR.values())))
OUT["runtime_s"] = time.time() - T0


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(OUT_PATH, "w"), indent=1)
log("variants", OUT["variants_evaluated_by_this_check"])
log("saved", OUT_PATH)
