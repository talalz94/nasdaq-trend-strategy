"""
ROBUSTNESS REVIEW of update_2026_11/leverage_execution.py  --  step 2: independent checks.

Uses only update_2026_11/common.py + panel.parquet (read-only). Writes results to
<scratch>/verify_leverage_execution_robustness_checks.json ; modifies no project file.

  A  FRIDAY EFFECT (C5/C6/C7/C8)
     A1 timing-luck null: 400 random schedules (one random trading day per week) and 400 schedules with a
        random weekday per 13-week block -> where do Friday / Monday sit in that distribution?
     A2 non-calendar phases: rebalance every 5th trading day, offsets 0..4 (rotate through weekdays at
        every holiday) -> dispersion and the bootstrap "significance" of the best one
     A3 episode dependence of Fri - mean(Mon..Thu): yearly gaps, t-stat, drop top years
     A4 cross-market: the same F2 machinery (EMA-100 gate, vt .30, chop, Aug/Sep) on S&P 500 and Russell
        2000 with a synthetic 2x sleeve, and plain 1x EMA-100 timing on NDX/SPX/RUT/TSX -> is Friday best?
     A5 holiday substitution (trade the next day of the same week when the weekday is a holiday)
     A6 mechanism statistic robustness (median / trimmed / ex-2008-2020)
     A7 the OOS weekday ordering: exit dates of each phase
  B  LEVERAGE (C1-C4)
     B1 vol-matched and bootstrap-median-DD-matched 3x vs F2 with paired-bootstrap CIs (Friday + tranche)
     B2 bootstrap tail sensitivity to block length and seed (F2 vs F3)
     B3 instrument effect decomposition: freed 1/3 in bills instead of IEF (fee/financing vs term premium)
     B4 leveraged-sleeve sim gap vs the T-bill rate (2010-2026), bond synthetic effective duration by regime
  C  EXECUTION (C7/C8)  stale-signal cost averaged over weekdays
usage: python update_2026_11/verify_leverage_execution_robustness_checks.py <scratch_dir>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
scratch = sys.argv[1]
os.makedirs(scratch, exist_ok=True)
OUTF = os.path.join(scratch, "verify_leverage_execution_robustness_checks.json")


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


P = C.panel()
IDX = P.index
T = len(IDX)
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
OOS_A = pd.Timestamp("2026-07-25")
WIN = {"IS": (None, FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE)}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DOW = IDX.dayofweek.values
MK = {d: pd.Series(DOW == i, index=IDX) for i, d in enumerate(DAYS)}
OUT = {}
NVAR = [0]


def W(lev, vt, PP=None, **kw):
    return C.strategy_weights(P if PP is None else PP, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15,
                              off="ief", **kw)


def bt(w, mark, lag=1, PP=None, cost=C.COST):
    NVAR[0] += 1
    return C.backtest(P if PP is None else PP, w, cost=cost, lag=lag, mark=mark)


def s3(r):
    return {wn: C.stats(r, *WIN[wn]) for wn in WIN}


def cg(r, wn="IS"):
    return C.stats(r, *WIN[wn])["cagr"]


def tranche(w, marks, lag=1, PP=None):
    PP = P if PP is None else PP
    H = sum(C.held_weights(PP, w, lag=lag, mark=m) for m in marks) / len(marks)
    gross = sum(H[c] * PP[c].fillna(0) for c in H.columns)
    NVAR[0] += 1
    return gross - H.diff().abs().sum(axis=1).fillna(0) * C.COST


def switches(w, lev, mark, lag=1):
    H = C.held_weights(P, w, lag=lag, mark=mark)
    sw = (H[lev] > 1e-12).astype(int).diff().abs()
    return int(sw[(sw.index >= START) & (sw.index <= FREEZE)].sum())


def r2(x, n=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


wF2, wF3 = W("qld", 0.30), W("tqqq", 0.20)
STR = {"F2": (wF2, "qld"), "F3": (wF3, "tqqq")}

# ---------------------------------------------------------------- reproduce the weekday table
log("A0 reproduce weekday phases")
PH = {s: {d: bt(w, MK[d]) for d in DAYS} for s, (w, lev) in STR.items()}
TR = {s: tranche(w, [MK[d] for d in DAYS]) for s, (w, lev) in STR.items()}
A0 = {}
for s in STR:
    oth = sum(PH[s][d] for d in DAYS[:4]) / 4.0
    A0[s] = dict(phase_IS_cagr={d: r2(cg(PH[s][d]) * 100) for d in DAYS},
                 phase_H1_cagr={d: r2(cg(PH[s][d], "H1") * 100) for d in DAYS},
                 phase_H2_cagr={d: r2(cg(PH[s][d], "H2") * 100) for d in DAYS},
                 tranche_IS=r2(cg(TR[s]) * 100), tranche_maxdd=r2(C.stats(TR[s], *WIN["IS"])["maxdd"] * 100, 1),
                 fri_minus_other4_boot=C.block_bootstrap_cagr_diff(PH[s]["Fri"].loc[:FREEZE], oth.loc[:FREEZE]),
                 switches={d: switches(STR[s][0], STR[s][1], MK[d]) for d in DAYS})
OUT["A0_reproduce"] = A0
log("A0", json.dumps({s: (A0[s]["phase_IS_cagr"], A0[s]["tranche_IS"], A0[s]["switches"]) for s in A0}))

# ---------------------------------------------------------------- A1 random schedules
log("A1 random-schedule timing-luck null")
wk_code, _ = pd.factorize(IDX.to_period("W-FRI"))
starts = np.r_[0, np.where(np.diff(wk_code) != 0)[0] + 1]
lens = np.diff(np.r_[starts, T])
nwk = len(starts)
rng = np.random.default_rng(20261118)
NSIM = 400


def rand_week_mark():
    pick = starts + np.floor(rng.random(nwk) * lens).astype(int)
    m = np.zeros(T, bool); m[pick] = True
    return pd.Series(m, index=IDX)


def rand_block_mark(block=13):
    blk = wk_code // block
    d = rng.integers(0, 5, blk.max() + 1)
    return pd.Series(DOW == d[blk], index=IDX)


A1 = {}
for kind, fn in (("random_day_each_week", rand_week_mark), ("random_weekday_per_13wk_block", rand_block_mark)):
    res = {s: dict(IS=[], H1=[], H2=[], dd=[], sw=[]) for s in STR}
    for i in range(NSIM):
        m = fn()
        for s, (w, lev) in STR.items():
            r = bt(w, m)
            for wn in WIN:
                res[s][wn].append(cg(r, wn))
            res[s]["dd"].append(C.stats(r, *WIN["IS"])["maxdd"])
            if s == "F2":
                res[s]["sw"].append(switches(w, lev, m))
    summ = {}
    for s in STR:
        a = np.array(res[s]["IS"]) * 100
        fri, mon = A0[s]["phase_IS_cagr"]["Fri"], A0[s]["phase_IS_cagr"]["Mon"]
        h1 = np.array(res[s]["H1"]) * 100; h2 = np.array(res[s]["H2"]) * 100
        dd = np.array(res[s]["dd"]) * 100
        summ[s] = dict(n=NSIM, IS_mean=r2(a.mean()), IS_sd=r2(a.std()), IS_p1=r2(np.percentile(a, 1)),
                       IS_p5=r2(np.percentile(a, 5)), IS_p50=r2(np.percentile(a, 50)), IS_p95=r2(np.percentile(a, 95)),
                       IS_p99=r2(np.percentile(a, 99)), IS_max=r2(a.max()), IS_min=r2(a.min()),
                       share_ge_friday=r2((a >= fri).mean(), 4), share_le_monday=r2((a <= mon).mean(), 4),
                       friday_z=r2((fri - a.mean()) / a.std()), monday_z=r2((mon - a.mean()) / a.std()),
                       H1_mean=r2(h1.mean()), H1_sd=r2(h1.std()), H2_mean=r2(h2.mean()), H2_sd=r2(h2.std()),
                       share_ge_fri_H1=r2((h1 >= A0[s]["phase_H1_cagr"]["Fri"]).mean(), 4),
                       share_ge_fri_H2=r2((h2 >= A0[s]["phase_H2_cagr"]["Fri"]).mean(), 4),
                       share_ge_fri_both_halves=r2(((h1 >= A0[s]["phase_H1_cagr"]["Fri"]) &
                                                    (h2 >= A0[s]["phase_H2_cagr"]["Fri"])).mean(), 4),
                       maxdd_p50=r2(np.percentile(dd, 50), 1), maxdd_p5=r2(np.percentile(dd, 5), 1),
                       maxdd_p95=r2(np.percentile(dd, 95), 1),
                       share_dd_better_than_friday=r2((dd >= C.stats(PH[s]["Fri"], *WIN["IS"])["maxdd"] * 100).mean(), 4))
        if s == "F2":
            sw = np.array(res[s]["sw"])
            summ[s]["switches_mean"] = r2(sw.mean(), 1); summ[s]["switches_p1"] = r2(np.percentile(sw, 1), 1)
            summ[s]["switches_min"] = int(sw.min()); summ[s]["share_switches_le_189"] = r2((sw <= 189).mean(), 4)
    A1[kind] = summ
    log("A1", kind, json.dumps(summ))
OUT["A1_random_schedules"] = A1

# ---------------------------------------------------------------- A2 trading-day offset phases
log("A2 every-5th-trading-day offset phases")
A2 = {}
for s, (w, lev) in STR.items():
    rr = {}
    for off in range(5):
        m = pd.Series((np.arange(T) % 5) == off, index=IDX)
        rr[off] = bt(w, m)
    cag = {o: cg(rr[o]) * 100 for o in rr}
    best = max(cag, key=cag.get)
    oth = sum(rr[o] for o in rr if o != best) / 4.0
    boot = C.block_bootstrap_cagr_diff(rr[best].loc[:FREEZE], oth.loc[:FREEZE])
    sw = {o: switches(w, lev, pd.Series((np.arange(T) % 5) == o, index=IDX)) for o in range(5)} if s == "F2" else None
    vals = np.array(list(cag.values()))
    A2[s] = dict(offset_IS_cagr={o: r2(v) for o, v in cag.items()},
                 H1={o: r2(cg(rr[o], "H1") * 100) for o in rr}, H2={o: r2(cg(rr[o], "H2") * 100) for o in rr},
                 spread_max_minus_min=r2(vals.max() - vals.min()), sd=r2(vals.std(ddof=1)),
                 best_minus_mean_other4=r2(vals.max() - np.mean([v for o, v in cag.items() if o != best])),
                 best_offset=int(best), best_vs_other4_boot=boot, switches=sw,
                 weekday_phase_sd=r2(np.std(list(A0[s]["phase_IS_cagr"].values()), ddof=1)),
                 weekday_spread=r2(max(A0[s]["phase_IS_cagr"].values()) - min(A0[s]["phase_IS_cagr"].values())))
    log("A2", s, json.dumps(A2[s]))
OUT["A2_offset_phases"] = A2

# ---------------------------------------------------------------- A3 episode dependence
log("A3 episode dependence")
A3 = {}
for s in STR:
    lr = pd.DataFrame({d: np.log1p(PH[s][d]) for d in DAYS})
    lr = lr[(lr.index >= START) & (lr.index <= FREEZE)]
    yr = lr.groupby(lr.index.year).sum()
    gap = (yr["Fri"] - yr[DAYS[:4]].mean(axis=1)) * 100
    srt = gap.sort_values(ascending=False)
    res = dict(mean_pp=r2(gap.mean()), median_pp=r2(gap.median()), t_yearly=r2(gap.mean() / (gap.std() / np.sqrt(len(gap)))),
               n_pos=int((gap > 0).sum()), n=int(len(gap)), top5=[(int(k), r2(v, 1)) for k, v in srt.head(5).items()])
    for k in (1, 3, 5):
        g2 = gap.drop(srt.index[:k])
        res[f"drop_top{k}_mean_pp"] = r2(g2.mean()); res[f"drop_top{k}_t"] = r2(g2.mean() / (g2.std() / np.sqrt(len(g2))))
    # Friday vs each single other day (yearly log gap)
    res["fri_minus_each_mean_pp"] = {d: r2(((yr["Fri"] - yr[d]) * 100).mean()) for d in DAYS[:4]}
    res["fri_minus_each_years_pos"] = {d: int(((yr["Fri"] - yr[d]) > 0).sum()) for d in DAYS[:4]}
    A3[s] = res
log("A3", json.dumps(A3))
OUT["A3_episode_dependence"] = A3

# ---------------------------------------------------------------- A4 cross-market
log("A4 cross-market weekday test")


def market_panel(ret_col):
    """copy of the panel whose signal columns are rebuilt from another index (no file is touched)."""
    PP = P.copy()
    lvl = (1 + PP[ret_col].fillna(0)).cumprod()
    PP["above"] = lvl > lvl.ewm(span=100, adjust=False).mean()
    PP["vol20"] = lvl.pct_change().rolling(20).std() * np.sqrt(252)
    PP["er20"] = lvl.diff(20).abs() / lvl.diff().abs().rolling(20).sum().replace(0, np.nan)
    PP[f"{ret_col}_2x"] = 2 * PP[ret_col].fillna(0) - 0.0095 / 252 - PP["rf_d"]
    PP[f"{ret_col}_1x"] = PP[ret_col].fillna(0)
    return PP


A4 = {}
for mkt in ("ndx_ret", "spx", "rut", "tsx"):
    PP = market_panel(mkt)
    out = {}
    # (i) F2 machinery with a synthetic 2x sleeve on that index
    w2 = W(f"{mkt}_2x", 0.30, PP=PP)
    ph2 = {d: bt(w2, MK[d], PP=PP) for d in DAYS}
    c2 = {d: cg(ph2[d]) * 100 for d in DAYS}
    oth2 = sum(ph2[d] for d in DAYS[:4]) / 4.0
    out["F2_machinery_2x"] = dict(IS={d: r2(v) for d, v in c2.items()},
                                  H1={d: r2(cg(ph2[d], "H1") * 100) for d in DAYS},
                                  H2={d: r2(cg(ph2[d], "H2") * 100) for d in DAYS},
                                  fri_rank=int(sorted(c2.values(), reverse=True).index(c2["Fri"]) + 1),
                                  fri_minus_mean_other4=r2(c2["Fri"] - np.mean([c2[d] for d in DAYS[:4]])),
                                  boot=C.block_bootstrap_cagr_diff(ph2["Fri"].loc[:FREEZE], oth2.loc[:FREEZE], n=1000))
    # (ii) plain 1x EMA-100 timing: index when above, bills when below; no vol target, no gold
    above = PP["above"].astype(float)
    w1 = pd.DataFrame({f"{mkt}_1x": above, "bills": 1 - above}, index=IDX)
    ph1 = {d: bt(w1, MK[d], PP=PP) for d in DAYS}
    c1 = {d: cg(ph1[d]) * 100 for d in DAYS}
    out["plain_1x_ema100"] = dict(IS={d: r2(v) for d, v in c1.items()},
                                  fri_rank=int(sorted(c1.values(), reverse=True).index(c1["Fri"]) + 1),
                                  fri_minus_mean_other4=r2(c1["Fri"] - np.mean([c1[d] for d in DAYS[:4]])))
    A4[mkt] = out
    log("A4", mkt, json.dumps({k: (v["IS"], v["fri_rank"], v["fri_minus_mean_other4"]) for k, v in out.items()}))
OUT["A4_cross_market"] = A4

# ---------------------------------------------------------------- A5 holiday substitution
log("A5 holiday substitution")
wk_s = pd.Series(wk_code, index=IDX)


def subst_mark(d):
    """weekday d; if it is a holiday, the next trading day of the same week (Fri: the previous day)."""
    df = pd.DataFrame({"wk": wk_code, "dow": DOW, "pos": np.arange(T)})
    m = np.zeros(T, bool)
    for _, g in df.groupby("wk"):
        if d < 4:
            c = g[g.dow >= d]
            p = c.pos.iloc[0] if len(c) else g.pos.iloc[-1]
        else:
            p = g.pos.iloc[-1]
        m[p] = True
    return pd.Series(m, index=IDX)


SM = {d: subst_mark(i) for i, d in enumerate(DAYS)}
A5 = {"skipped_weeks_without_substitution": {d: int(nwk - MK[d].sum()) for d in DAYS}}
for s, (w, lev) in STR.items():
    rr = {d: bt(w, SM[d]) for d in DAYS}
    c = {d: cg(rr[d]) * 100 for d in DAYS}
    trs = tranche(w, [SM[d] for d in DAYS])
    oth = sum(rr[d] for d in DAYS[:4]) / 4.0
    A5[s] = dict(IS={d: r2(v) for d, v in c.items()}, tranche_IS=r2(cg(trs) * 100),
                 tranche_maxdd=r2(C.stats(trs, *WIN["IS"])["maxdd"] * 100, 1),
                 fri_minus_mean_other4=r2(c["Fri"] - np.mean([c[d] for d in DAYS[:4]])),
                 boot=C.block_bootstrap_cagr_diff(rr["Fri"].loc[:FREEZE], oth.loc[:FREEZE], n=1000))
log("A5", json.dumps(A5))
OUT["A5_holiday_substitution"] = A5

# ---------------------------------------------------------------- A6 mechanism robustness
log("A6 mechanism statistic robustness")
nxt = P["ndx_ret"].shift(-1)
abv = P["above"].astype(bool).values
A6 = {}
for wn in ("H1", "H2"):
    a, b = WIN[wn]
    sel = (IDX >= pd.Timestamp(a or START)) & (IDX <= pd.Timestamp(b))
    row = {}
    for i, d in enumerate(DAYS):
        x = nxt[sel & (DOW == i) & ~abv].dropna()
        xs = np.sort(x.values)
        k = int(len(xs) * 0.05)
        ex = x[~x.index.year.isin([2008, 2020])]
        top5 = x.sort_values(ascending=False).head(5)
        row[d] = dict(n=int(len(x)), mean_bp=r2(x.mean() * 1e4, 1), median_bp=r2(x.median() * 1e4, 1),
                      trim5_mean_bp=r2(xs[k:len(xs) - k].mean() * 1e4, 1),
                      ex_2008_2020_mean_bp=r2(ex.mean() * 1e4, 1),
                      ex_2008_2020_t=r2(ex.mean() / (ex.std() / np.sqrt(len(ex)))),
                      share_pos=r2((x > 0).mean(), 3),
                      top5_days=[(str(k2.date()), r2(v * 100, 1)) for k2, v in top5.items()],
                      mean_without_top5_bp=r2(x.drop(top5.index).mean() * 1e4, 1))
    A6[wn] = row
log("A6", json.dumps({wn: {d: (v["mean_bp"], v["median_bp"], v["trim5_mean_bp"], v["ex_2008_2020_mean_bp"],
                               v["mean_without_top5_bp"]) for d, v in A6[wn].items()} for wn in A6}))
OUT["A6_mechanism"] = A6

# ---------------------------------------------------------------- A7 OOS exits by phase
log("A7 OOS exit dates by phase")
A7 = {}
for s, (w, lev) in STR.items():
    rows = {}
    for d in DAYS:
        H = C.held_weights(P, w, mark=MK[d])[lev]
        h = H[H.index >= pd.Timestamp("2026-07-20")]
        on = (h > 1e-12).astype(int)
        ch = on.diff().fillna(0)
        ev = [(str(t.date()), "on" if v > 0 else "off", r2(P["ndx"].shift(1).loc[t], 0)) for t, v in ch.items() if v != 0]
        rows[d] = dict(events_first_held_day_and_ndx_at_trade_close=ev,
                       oos_total=r2(C.period_return(PH[s][d], OOS_A, LATEST) * 100, 1))
    A7[s] = rows
log("A7", json.dumps(A7["F2"]))
OUT["A7_oos_events"] = A7

# ---------------------------------------------------------------- B1 matched-risk with CIs
log("B1 vol-matched and bootstrap-DD-matched 3x vs 2x with CIs")


def paired_boot_mat(series, n=1000, block=63, seed=11, a=START, b=FREEZE):
    M = pd.concat(series, axis=1)
    M = M[(M.index >= a) & (M.index <= b)].fillna(0).values
    TT, k = M.shape
    g = np.random.default_rng(seed)
    nb = int(np.ceil(TT / block))
    cgm = np.empty((n, k)); ddm = np.empty((n, k))
    for i in range(n):
        st_ = g.integers(0, TT, nb)
        ix = ((st_[:, None] + np.arange(block)[None, :]).ravel()[:TT]) % TT
        e = np.vstack([np.ones((1, k)), np.cumprod(1 + M[ix], axis=0)])
        cgm[i] = e[-1] ** (252.0 / TT) - 1
        ddm[i] = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
    return cgm, ddm


B1 = {}
for path in ("friday", "tranche"):
    run = (lambda w: bt(w, MK["Fri"])) if path == "friday" else (lambda w: tranche(w, [MK[d] for d in DAYS]))
    r2x = run(wF2)
    vol2 = float(r2x[(r2x.index >= START) & (r2x.index <= FREEZE)].std() * np.sqrt(252))
    grid = np.round(np.arange(0.05, 0.20, 0.0025), 4)
    cand = {}
    for vt in grid:
        rr = run(W("tqqq", float(vt)))
        cand[float(vt)] = rr
    vols = {vt: float(r[(r.index >= START) & (r.index <= FREEZE)].std() * np.sqrt(252)) for vt, r in cand.items()}
    vt_vm = min(vols, key=lambda v: abs(vols[v] - vol2))
    r3 = cand[vt_vm]
    res = dict(F2_vol=r2(vol2 * 100), vt3_volmatched=vt_vm, vol3=r2(vols[vt_vm] * 100),
               cagr={wn: dict(F2=r2(cg(r2x, wn) * 100), x3=r2(cg(r3, wn) * 100),
                              diff_3x_minus_2x=r2((cg(r3, wn) - cg(r2x, wn)) * 100)) for wn in WIN},
               maxdd={wn: dict(F2=r2(C.stats(r2x, *WIN[wn])["maxdd"] * 100, 1),
                               x3=r2(C.stats(r3, *WIN[wn])["maxdd"] * 100, 1)) for wn in WIN},
               boot_3x_minus_2x=C.block_bootstrap_cagr_diff(r3.loc[:FREEZE], r2x.loc[:FREEZE]))
    # bootstrap-median-maxDD matching on the SAME draws
    sub = [vt for vt in grid if 0.07 <= vt <= 0.16]
    cgm, ddm = paired_boot_mat([r2x] + [cand[vt] for vt in sub])
    med = np.median(ddm, axis=0)
    j = 1 + int(np.argmin(np.abs(med[1:] - med[0])))
    d = cgm[:, j] - cgm[:, 0]
    res["boot_median_dd_matched"] = dict(F2_median_dd=r2(med[0] * 100, 1), vt3=sub[j - 1], x3_median_dd=r2(med[j] * 100, 1),
                                         F2_med_cagr=r2(np.median(cgm[:, 0]) * 100), x3_med_cagr=r2(np.median(cgm[:, j]) * 100),
                                         diff_med_pp=r2(np.median(d) * 100), diff_p2_5=r2(np.percentile(d, 2.5) * 100),
                                         diff_p97_5=r2(np.percentile(d, 97.5) * 100), p_3x_better=r2((d > 0).mean(), 3),
                                         F2_p5_dd=r2(np.percentile(ddm[:, 0], 5) * 100, 1),
                                         x3_p5_dd=r2(np.percentile(ddm[:, j], 5) * 100, 1))
    B1[path] = res
    log("B1", path, json.dumps(res))
OUT["B1_matched_risk_CI"] = B1

# ---------------------------------------------------------------- B2 bootstrap tail sensitivity
log("B2 bootstrap tail sensitivity (block length, seed)")
rF2, rF3 = PH["F2"]["Fri"], PH["F3"]["Fri"]
B2 = {}
for block in (21, 63, 126, 252):
    for seed in (11, 12, 13):
        cgm, ddm = paired_boot_mat([rF2, rF3], n=1000, block=block, seed=seed)
        cal = cgm / -ddm
        B2[f"block{block}_seed{seed}"] = dict(
            F2_med_dd=r2(np.median(ddm[:, 0]) * 100, 1), F3_med_dd=r2(np.median(ddm[:, 1]) * 100, 1),
            F2_p5_dd=r2(np.percentile(ddm[:, 0], 5) * 100, 1), F3_p5_dd=r2(np.percentile(ddm[:, 1], 5) * 100, 1),
            F2_p_dd50=r2((ddm[:, 0] <= -0.5).mean(), 3), F3_p_dd50=r2((ddm[:, 1] <= -0.5).mean(), 3),
            p_calmar_F2_gt_F3=r2((cal[:, 0] > cal[:, 1]).mean(), 3))
log("B2", json.dumps(B2))
OUT["B2_boot_sensitivity"] = B2

# ---------------------------------------------------------------- B3 instrument decomposition
log("B3 instrument effect decomposition")
w_inst = W("tqqq", 0.30, extra_scale=pd.Series(2.0 / 3.0, index=IDX))
freed = wF2["qld"] - w_inst["tqqq"]                      # = qld weight / 3
w_bill = w_inst.copy()
w_bill["ief"] = w_bill["ief"] - freed
w_bill["bills"] = freed
assert float((w_bill["ief"]).min()) > -1e-12
B3 = {}
for vn, ren3, ren2 in [("spliced", {}, {}),
                       ("sim_all", {"tqqq": "tqqq_sim", "ief": "ief_syn"}, {"qld": "qld_sim", "ief": "ief_syn"})]:
    rq = bt(wF2.rename(columns=ren2), MK["Fri"])
    ri = bt(w_inst.rename(columns=ren3), MK["Fri"])
    rb = bt(w_bill.rename(columns=ren3), MK["Fri"])
    row = {}
    for lab, a, b in (("IS", None, FREEZE), ("H1", "1988-01-01", "2006-12-31"), ("H2", "2007-01-01", FREEZE),
                      ("2010-03..freeze", "2010-03-01", FREEZE), ("2023-01..freeze", "2023-01-01", FREEZE),
                      ("2023-01..latest", "2023-01-01", LATEST)):
        cq, ci, cb = (C.stats(x, a, b)["cagr"] for x in (rq, ri, rb))
        row[lab] = dict(tqqq23_ief_minus_qld=r2((ci - cq) * 100), tqqq23_bills_minus_qld=r2((cb - cq) * 100),
                        term_premium_part=r2((ci - cb) * 100))
    if vn == "spliced":
        row["boot_IS_bills_version"] = C.block_bootstrap_cagr_diff(rb.loc[:FREEZE], rq.loc[:FREEZE])
    B3[vn] = row
log("B3", json.dumps(B3))
OUT["B3_instrument_decomposition"] = B3

# ---------------------------------------------------------------- B4 sim gap vs rates; bond duration by regime
log("B4 sim financing gap vs T-bill rate; bond synthetic duration by regime")
B4 = {"years": {}}
rows = []
for y in range(2010, 2027):
    a = max(pd.Timestamp("2010-03-01"), pd.Timestamp(f"{y}-01-01")); b = min(LATEST, pd.Timestamp(f"{y}-12-31"))
    sl = (IDX >= a) & (IDX <= b)
    g = {k: float((P[f"{k}_real"] - P[f"{k}_sim"])[sl].mean() * 252 * 100) for k in ("qqq", "qld", "tqqq")}
    rfm = float(P["rf"][sl].mean() * 100)
    # implied financing spread over rf: TQQQ ~ 3*div - 2*spread (+fee diff); QQQ gap ~ dividend yield
    g["rf"] = rfm
    g["implied_spread_tqqq"] = (3 * g["qqq"] - g["tqqq"]) / 2
    g["implied_spread_qld"] = (2 * g["qqq"] - g["qld"]) / 1
    B4["years"][str(y)] = {k: r2(v) for k, v in g.items()}
    rows.append(g)
df = pd.DataFrame(rows)
for k in ("tqqq", "qld", "implied_spread_tqqq", "implied_spread_qld"):
    sl_, ic = np.polyfit(df["rf"], df[k], 1)
    B4[f"ols_{k}_on_rf"] = dict(slope=r2(sl_, 3), intercept=r2(ic, 3), corr=r2(np.corrcoef(df["rf"], df[k])[0, 1], 3))
B4["avg_rf_pct"] = {lab: r2(P["rf"][(IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))].mean() * 100)
                    for lab, a, b in (("1988-2006", "1988-01-01", "2006-12-31"), ("1988-2010-02", "1988-01-01", "2010-02-28"),
                                      ("2010-03..2021", "2010-03-01", "2021-12-31"), ("2023..latest", "2023-01-01", LATEST))}
# effective duration of the real bond ETFs by yield regime vs the fixed calibrated duration
dur = {}
for etf, ycol in (("tlt", "tyx"), ("ief", "tnx")):
    dy = (P[ycol] / 100).diff()
    carry = (P[ycol] / 100).shift(1) / 252
    dd = {}
    for lab, a, b in (("2002-08..2007", "2002-08-01", "2007-12-31"), ("2008..2015", "2008-01-01", "2015-12-31"),
                      ("2016..2021", "2016-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
                      ("2023..latest", "2023-01-01", LATEST)):
        sl = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
        m = pd.concat([P[f"{etf}_real"] - carry, dy], axis=1)[sl].dropna()
        m.columns = ["y", "dy"]
        D = -np.polyfit(m["dy"], m["y"], 1)[0]
        dd[lab] = dict(eff_duration=r2(D, 2), avg_yield=r2(P[ycol][sl].mean(), 2))
    dur[etf] = dd
B4["bond_effective_duration_by_regime"] = dur
B4["bond_avg_yield_pre_2002"] = {c: r2(P[c][(IDX >= START) & (IDX < pd.Timestamp("2002-07-30"))].mean(), 2) for c in ("tyx", "tnx")}
log("B4", json.dumps({k: v for k, v in B4.items() if k != "years"}))
OUT["B4_sim_gap_and_duration"] = B4

# ---------------------------------------------------------------- C stale-signal cost averaged over weekdays
log("C stale-signal cost averaged over weekday phases")
Cres = {}
for s, (w, lev) in STR.items():
    st_ = {d: bt(w.shift(1), MK[d]) for d in DAYS}   # signal at the previous session's close, trade at day d close
    c_st = {d: cg(st_[d]) * 100 for d in DAYS}
    c_l1 = A0[s]["phase_IS_cagr"]
    Cres[s] = dict(stale_IS={d: r2(v) for d, v in c_st.items()},
                   stale_cost_by_day={d: r2(c_l1[d] - c_st[d]) for d in DAYS},
                   stale_cost_avg_over_days=r2(np.mean([c_l1[d] - c_st[d] for d in DAYS])),
                   thu_lag2_IS=r2(cg(bt(w, MK["Thu"], lag=2)) * 100))
log("C", json.dumps(Cres))
OUT["C_stale_signal"] = Cres

# ---------------------------------------------------------------- current state detail
ab = P["above"].astype(bool)
last_true = ab[ab].index[-1]
OUT["current_state"] = dict(last_close_above=str(last_true.date()),
                            first_close_below_after_freeze=str(ab[(~ab) & (IDX > FREEZE)].index[0].date()),
                            rally_needed_pct=r2((P["ema100"].iloc[-1] / P["ndx"].iloc[-1] - 1) * 100),
                            ndx=r2(P["ndx"].iloc[-1], 1), ema100=r2(P["ema100"].iloc[-1], 1))
OUT["n_backtests_run_by_this_check"] = NVAR[0]
OUT["runtime_sec"] = time.time() - T0
with open(OUTF, "w") as f:
    json.dump(OUT, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
log("saved", OUTF, "backtests:", NVAR[0])
