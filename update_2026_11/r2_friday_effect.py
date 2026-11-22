"""
ROUND 2 - DIAGNOSIS OF THE FRIDAY REBALANCE PREMIUM                       KEY = friday_effect
Writes update_2026_11/r2_friday_effect.json. Reads panel.parquet / common.py (shared, read-only) and the
helper cache update_2026_11/r2_friday_effect_yf.parquet (built by r2_friday_effect_fetch.py).

PRE-REGISTERED DECISION RULE (fixed before any round-2 number was computed; applied exactly as written)
  'Friday rebalancing is STRUCTURALLY better'  iff  A and B.
  A (both halves, NDX, F2-IEF = the recommended tier): Friday book minus the 5-weekday tranche is > 0 in
    BOTH CAGR and Calmar in BOTH halves, H1 = 1988-01-01..2006-12-31 and H2 = 2007-01-01..2026-07-24,
    AND the Friday book's Calmar is at or above the 95th percentile of the 1,000-schedule random-weekday
    placebo (one weekday drawn independently each week; seed 20261118) in BOTH halves.
  B (other markets): with the F2 rules (2x sim sleeve 2r - 0.95%/252 - rf_d, own-price EMA-100 / vol20 /
    er20 signals, vt 0.30, chop 0.25 -> x0.5, Aug-Sep x0.5, IEF off-leg, permanent 15% gold_phys), Friday
    book minus tranche is > 0 in BOTH CAGR and Calmar over the market's in-sample window (its start ..
    2026-07-24) in at least 5 of the 7 other markets: SPX, RUT, TSX, HSI (panel returns, US calendar) and
    DAX, N225, IXIC (Yahoo, own trading calendars; IXIC and N225 from 1975-01-01, every other market from
    max(1988-01-01, first data + 250 sessions)).
  If not (A and B): the planning basis is the 5-weekday tranche and the Friday premium is luck, treated as
    upside. F1/F3, OOS (2026-07-25..2026-11-17) and every other number are reported, never used to decide.
  Diagnostic (b), also fixed in advance: 'the first holding day explains the premium' iff BOTH
    (i) >= 2/3 of the IS Friday-minus-tranche CAGR gap disappears when every book trades one session later
        (lag 2, i.e. the first holding day is neutralised for every book), AND
    (ii) >= 2/3 of the IS arithmetic Friday-minus-tranche gap is earned on the Friday book's first holding
        day (the session after each Friday decision).
No parameter is tuned or introduced; every strategy parameter is the July one. Gold before 2000-08-30 is a
monthly interpolation and is only ever used as a (common, constant-weight) return sleeve, never as a signal.

ADDED AFTER THE FIRST RUN (diagnostics only, NOT part of the pre-registered rule, never used to decide):
  (f) selection-artifact check: the July EMA length / vol target / chop settings were chosen on Friday-
      rebalanced backtests. Friday-minus-tranche is re-measured across EMA spans {50,75,100,125,150,200,250}
      and the round-1 plateau sub-grid vt {.15..35} x chop_thr {.20,.25,.30} x chop_pos {.35,.50,.65}
      (F2 sleeve, gold 15%, Aug-Sep on). Nothing is selected from this grid.
  (c+) per-market lag-2 Friday-minus-tranche and Friday->next-session return autocorrelation by half.
  (d+) maxDD distribution of the 13-week-block null; single-fixed-day planning distribution.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT_JSON = os.path.join(HERE, "r2_friday_effect.json")
YF_PATH = os.path.join(HERE, "r2_friday_effect_yf.parquet")
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
VAR = {}
OUT = {"meta": dict(script="update_2026_11/r2_friday_effect.py", key="friday_effect",
                    windows={k: [str(a.date()), str(b.date())] for k, (a, b) in WIN.items()},
                    oos=[str(OOS_A.date()), str(LAST.date())], placebo_n=NPL, seed=SEED,
                    note="IS = in-sample to the 2026-07-24 freeze (decision data). OOS reported separately.")}


def count(k, n=1):
    VAR[k] = VAR.get(k, 0) + n


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# =============================================================== engine (numpy, == common.backtest)
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
    return gross - turn * cost, H, turn


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
                sharpe=float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan, total=float(e[-1] - 1),
                days=int(m.sum()))


def stm(dates, Rm, a, b):
    m = wm(dates, a, b)
    X = Rm[:, m]; d = dates[m]
    E = np.cumprod(1 + X, axis=1)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = E[:, -1] ** (1 / yrs) - 1
    dd = (E / np.maximum.accumulate(E, axis=1) - 1).min(axis=1)
    return cagr, dd, cagr / np.abs(dd), E[:, -1] - 1


def period_ret(dates, r, a, b):
    m = wm(dates, a, b)
    return float(np.prod(1 + r[m]) - 1)


def yrs_of(dates, a, b):
    d = dates[wm(dates, a, b)]
    return (d[-1] - d[0]).days / 365.25


def week_codes(dates):
    return pd.factorize(pd.DatetimeIndex(dates).to_period("W-FRI"))[0]


def placebo_masks(dates, S, rng, block_weeks=1):
    wk = week_codes(dates)
    dow = pd.DatetimeIndex(dates).dayofweek.values
    grp = wk // block_weeks
    draws = rng.integers(0, 5, size=(S, grp.max() + 1))
    return dow[None, :] == draws[:, grp]


def placebo_stats(W, R, dates, masks, windows, lag=1, cost=C.COST, chunk=100):
    """CAGR / maxDD / Calmar / total of every schedule in `masks` over each window (chunked)."""
    res = {wn: {"cagr": [], "maxdd": [], "calmar": [], "total": []} for wn in windows}
    T = len(dates); ar = np.arange(T); k = W.shape[1]
    for i in range(0, masks.shape[0], chunk):
        mk = masks[i:i + chunk]; s = mk.shape[0]
        li = np.maximum.accumulate(np.where(mk, ar[None, :], -1), axis=1)
        H = np.where((li >= 0)[..., None], W[np.maximum(li, 0)], 0.0)
        H = np.concatenate([np.zeros((s, lag, k)), H[:, :-lag]], axis=1)
        gross = (H * R[None]).sum(2)
        turn = np.abs(np.diff(H, axis=1, prepend=np.zeros((s, 1, k)))).sum(2)
        Rm = gross - turn * cost
        for wn, (a, b) in windows.items():
            cg, dd, cal, tot = stm(dates, Rm, a, b)
            res[wn]["cagr"].append(cg); res[wn]["maxdd"].append(dd)
            res[wn]["calmar"].append(cal); res[wn]["total"].append(tot)
    return {wn: {m: np.concatenate(v) for m, v in d.items()} for wn, d in res.items()}


def pctile(dist, x):
    dist = np.asarray(dist)
    return float(100 * ((dist < x).mean() + 0.5 * (dist == x).mean()))


def dsum(x):
    x = np.asarray(x, float)
    return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), p5=float(np.percentile(x, 5)),
                p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)), min=float(x.min()),
                max=float(x.max()))


def f_weights(lvl, month, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, span=100):
    """F-family target weights from an arbitrary price level (mirrors common.strategy_weights)."""
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
    return np.column_stack([lev, off, gold]), above.values.astype(bool), (lvl / ema - 1).values


# =============================================================== NDX books (section a)
log("building NDX weekday books")
DOW = IDX.dayofweek.values
DM = {d: (DOW == i) for i, d in enumerate(DAYS)}
ABOVE = P["above"].values.astype(bool)
DIST = (P["ndx"] / P["ema100"] - 1).values
NDXR = P["ndx_ret"].fillna(0).values
BK = {}
check = {}
ref = json.load(open(os.path.join(HERE, "main_tranche_check.json")))
for t, (lev, vt) in TIERS.items():
    wdf = C.strategy_weights(P, lev=lev, voltgt=vt)
    cols = list(wdf.columns)
    W = wdf.values; R = P[cols].fillna(0).values
    b = dict(W=W, R=R, cols=cols, r={}, H={}, turn={}, r2={})
    maxdiff = 0.0
    for d in DAYS:
        r, H, tu = run_book(W, R, DM[d]); count("ndx_weekday_books_lag1")
        b["r"][d], b["H"][d], b["turn"][d] = r, H, tu
        rc = C.backtest(P, wdf, mark=pd.Series(DM[d], index=IDX)).values
        maxdiff = max(maxdiff, float(np.abs(rc - r).max()))
        b["r2"][d] = run_book(W, R, DM[d], lag=2)[0]; count("ndx_weekday_books_lag2")
    b["r"]["tranche"] = sum(b["r"][d] for d in DAYS) / 5
    b["r2"]["tranche"] = sum(b["r2"][d] for d in DAYS) / 5
    # reproduction vs main_tranche_check.json (to_freeze CAGR/maxDD)
    rep = {}
    for k in DAYS + ["tranche"]:
        s = st1(IDX, b["r"][k], START, FREEZE)
        rr = ref[f"{t}-IEF"][k]["to_freeze"]
        rep[k] = dict(cagr=s["cagr"], ref_cagr=rr["cagr"], maxdd=s["maxdd"], ref_maxdd=rr["maxdd"],
                      calmar=s["calmar"], abs_diff_cagr=abs(s["cagr"] - rr["cagr"]))
    check[t] = dict(max_abs_daily_diff_vs_common_backtest=maxdiff,
                    max_abs_cagr_diff_vs_main_tranche_check=max(v["abs_diff_cagr"] for v in rep.values()))
    BK[t] = b
    log(t, "repro", check[t])
OUT["reproduction_check"] = check
count("ndx_tranches", 3 * 2)


def book_table(t):
    b = BK[t]; out = {}
    for k in DAYS + ["tranche"]:
        out[k] = {wn: st1(IDX, b["r"][k], a, bb) for wn, (a, bb) in WIN.items()}
        out[k]["OOS_total"] = period_ret(IDX, b["r"][k], OOS_A, LAST)
    return out


SEC_A = {}
for t in TIERS:
    b = BK[t]
    rf, rt = b["r"]["Fri"], b["r"]["tranche"]
    g = rf - rt
    tab = book_table(t)
    gaps = {}
    for wn, (a, bb) in WIN.items():
        sf, stt = tab["Fri"][wn], tab["tranche"][wn]
        m = wm(IDX, a, bb)
        gaps[wn] = dict(d_cagr_pp=(sf["cagr"] - stt["cagr"]) * 100, d_calmar=sf["calmar"] - stt["calmar"],
                        d_maxdd_pp=(sf["maxdd"] - stt["maxdd"]) * 100,
                        arith_pp_per_yr=float(g[m].sum() / yrs_of(IDX, a, bb) * 100))
    gaps["OOS"] = dict(fri_total=tab["Fri"]["OOS_total"], tranche_total=tab["tranche"]["OOS_total"],
                       d_total_pp=(tab["Fri"]["OOS_total"] - tab["tranche"]["OOS_total"]) * 100)
    # ---- by decade
    dec = {}
    isy = yrs_of(IDX, START, FREEZE)
    tot_arith = g[wm(IDX, START, FREEZE)].sum()
    for k, (a, bb) in C.DECADES.items():
        a = max(pd.Timestamp(a), START); bb = min(pd.Timestamp(bb), FREEZE)
        sf, stt = st1(IDX, rf, a, bb), st1(IDX, rt, a, bb)
        m = wm(IDX, a, bb)
        dec[k] = dict(fri_cagr=sf["cagr"], tranche_cagr=stt["cagr"], d_cagr_pp=(sf["cagr"] - stt["cagr"]) * 100,
                      d_calmar=sf["calmar"] - stt["calmar"], fri_maxdd=sf["maxdd"], tranche_maxdd=stt["maxdd"],
                      arith_pp_per_yr=float(g[m].sum() / yrs_of(IDX, a, bb) * 100),
                      share_of_IS_arith_gap=float(g[m].sum() / tot_arith))
    # ---- by calendar year
    yr = {}
    years = sorted(set(IDX[wm(IDX, START, FREEZE)].year))
    for y in years:
        a = pd.Timestamp(f"{y}-01-01"); bb = min(pd.Timestamp(f"{y}-12-31"), FREEZE)
        m = wm(IDX, a, bb)
        fy = float(np.prod(1 + rf[m]) - 1); ty = float(np.prod(1 + rt[m]) - 1)
        yr[int(y)] = dict(fri=fy, tranche=ty, gap_pp=(fy - ty) * 100,
                          others={d: float(np.prod(1 + b["r"][d][m]) - 1) for d in DAYS[:4]})
    gp = pd.Series({y: v["gap_pp"] for y, v in yr.items()})
    srt = gp.sort_values(ascending=False)
    lg = pd.Series({y: (np.log1p(v["fri"]) - np.log1p(v["tranche"])) for y, v in yr.items()})
    yr_sum = dict(n_years=int(len(gp)), n_fri_beats_tranche=int((gp > 0).sum()),
                  n_fri_best_of_5=int(sum(1 for y, v in yr.items()
                                          if v["fri"] >= max(v["others"].values()))),
                  mean_gap_pp=float(gp.mean()), median_gap_pp=float(gp.median()),
                  t_stat=float(gp.mean() / (gp.std(ddof=1) / np.sqrt(len(gp)))),
                  top5=[(int(k), float(v)) for k, v in srt.head(5).items()],
                  bottom5=[(int(k), float(v)) for k, v in srt.tail(5).items()],
                  share_of_log_gap_top5=float(lg.loc[srt.index[:5]].sum() / lg.sum()),
                  mean_gap_pp_drop_top3=float(gp.drop(srt.index[:3]).mean()),
                  mean_gap_pp_drop_top5=float(gp.drop(srt.index[:5]).mean()),
                  H1_years_pos=int((gp[gp.index <= 2006] > 0).sum()), H1_years=int((gp.index <= 2006).sum()),
                  H2_years_pos=int((gp[gp.index >= 2007] > 0).sum()), H2_years=int((gp.index >= 2007).sum()))
    # ---- by event type of the Friday book's decision
    fm = DM["Fri"]; mk = np.where(fm)[0]
    wlev = b["W"][:, 0]
    cls_at = {}
    for i, p in enumerate(mk):
        if i == 0:
            cls_at[p] = "first"; continue
        q = mk[i - 1]
        if ABOVE[p] != ABOVE[q]:
            c = ("switch_on" if ABOVE[p] else "switch_off") + ("_nearEMA" if abs(DIST[p]) < 0.01 else "_farEMA")
        elif abs(wlev[p] - wlev[q]) > 1e-12:
            c = "size_only"
        else:
            c = "stay_off" if not ABOVE[p] else "stay_on_same_size"
        cls_at[p] = c
    li_f = last_idx(fm)
    decn = np.r_[-1, li_f[:-1]]
    day_cls = np.array([cls_at.get(p, "none") if p >= 0 else "none" for p in decn])
    ev = {}
    for wn, (a, bb) in WIN.items():
        m = wm(IDX, a, bb); yy = yrs_of(IDX, a, bb)
        rows = {}
        wk_g = pd.Series(g[m]).groupby(decn[m]).sum()
        wk_c = pd.Series(day_cls[m]).groupby(decn[m]).first()
        for c in sorted(set(day_cls[m])):
            sel = wk_g[wk_c == c]
            rows[c] = dict(n_weeks=int(len(sel)), pp_per_yr=float(sel.sum() / yy * 100),
                           mean_bp_per_week=float(sel.mean() * 1e4),
                           t_weekly=float(sel.mean() / (sel.std(ddof=1) / np.sqrt(len(sel)))) if len(sel) > 2 else None,
                           share_of_gap=float(sel.sum() / g[m].sum()))
        # coarse classes
        coarse = {"switch": [c for c in rows if c.startswith("switch")], "size_only": ["size_only"],
                  "no_change": ["stay_off", "stay_on_same_size"], "near_EMA_switch": [c for c in rows if "nearEMA" in c]}
        rows["_coarse"] = {k: dict(pp_per_yr=float(sum(rows[c]["pp_per_yr"] for c in v if c in rows)),
                                   n_weeks=int(sum(rows[c]["n_weeks"] for c in v if c in rows)),
                                   share_of_gap=float(sum(rows[c]["share_of_gap"] for c in v if c in rows)))
                           for k, v in coarse.items()}
        ev[wn] = rows
    # ---- state agreement / disagreement among the 5 books (held state, lag 1)
    HS = np.vstack([np.r_[False, (ABOVE[np.maximum(last_idx(DM[d]), 0)] & (last_idx(DM[d]) >= 0))[:-1]]
                    for d in DAYS])
    non = HS.sum(0)
    agree_on, agree_off = non == 5, non == 0
    dis = ~(agree_on | agree_off)
    fri_on = HS[4]
    st_dec = {}
    for wn, (a, bb) in WIN.items():
        m = wm(IDX, a, bb); yy = yrs_of(IDX, a, bb); tg = g[m].sum()
        st_dec[wn] = {k: dict(days=int((m & s).sum()), pp_per_yr=float(g[m & s].sum() / yy * 100),
                              share_of_gap=float(g[m & s].sum() / tg))
                      for k, s in [("all_5_on (size differences only)", agree_on),
                                   ("all_5_off", agree_off), ("books_disagree_on_state", dis),
                                   ("disagree_and_Fri_on", dis & fri_on), ("disagree_and_Fri_off", dis & ~fri_on)]}
    # ---- disagreement episodes (maximal runs of disagreement days), IS
    m_is = wm(IDX, START, FREEZE)
    runs = []
    i = 0
    while i < N:
        if dis[i] and m_is[i]:
            j = i
            while j + 1 < N and dis[j + 1] and m_is[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    eps = []
    for (i, j) in runs:
        eps.append(dict(start=str(IDX[i].date()), end=str(IDX[j].date()), days=int(j - i + 1),
                        gap_pp=float(g[i:j + 1].sum() * 100),
                        ndx_ret=float(np.prod(1 + NDXR[i:j + 1]) - 1),
                        fri_on_share=float(fri_on[i:j + 1].mean())))
    epg = np.array([e["gap_pp"] for e in eps])
    order = np.argsort(-epg)
    tot_pp = g[m_is].sum() * 100
    ep_sum = dict(n_episodes=len(eps), n_fri_positive=int((epg > 0).sum()), n_fri_negative=int((epg < 0).sum()),
                  total_gap_pp_IS=float(tot_pp), episodes_gap_pp=float(epg.sum()),
                  top10=[eps[k] for k in order[:10]], bottom5=[eps[k] for k in order[::-1][:5]],
                  top10_share_of_total=float(epg[order[:10]].sum() / tot_pp),
                  top20_share_of_total=float(epg[order[:20]].sum() / tot_pp),
                  gap_pp_per_yr_excl_top10=float((tot_pp - epg[order[:10]].sum()) / isy),
                  gap_pp_per_yr_excl_top10_and_bottom10=float((tot_pp - epg[order[:10]].sum()
                                                               - epg[order[::-1][:10]].sum()) / isy))
    ep_sum["episodes_in_top5_years"] = {
        int(y): sorted([e for e in eps if e["start"][:4] == str(y) and abs(e["gap_pp"]) >= 0.5],
                       key=lambda e: -abs(e["gap_pp"]))[:6] for y, _ in yr_sum["top5"]}
    # ---- sleeve decomposition of the daily gap (IS): lev sleeve / IEF / gold / costs
    Hbar = sum(b["H"][d] for d in DAYS) / 5
    tbar = sum(b["turn"][d] for d in DAYS) / 5
    sl = {}
    for wn, (a, bb) in WIN.items():
        m = wm(IDX, a, bb); yy = yrs_of(IDX, a, bb)
        comp = {c: float(((b["H"]["Fri"][:, j] - Hbar[:, j]) * b["R"][:, j])[m].sum() / yy * 100)
                for j, c in enumerate(b["cols"])}
        comp["costs"] = float((-(b["turn"]["Fri"] - tbar) * C.COST)[m].sum() / yy * 100)
        comp["turnover_per_yr_fri"] = float(b["turn"]["Fri"][m].sum() / yy)
        comp["turnover_per_yr_tranche_avg_book"] = float(tbar[m].sum() / yy)
        sl[wn] = comp
    SEC_A[t] = dict(books=tab, gap=gaps, by_decade=dec, by_year=yr, year_summary=yr_sum,
                    by_friday_event=ev, by_state_agreement=st_dec, episodes=ep_sum, sleeve_decomposition=sl)
    BK[t]["g"] = g; BK[t]["decn"] = decn; BK[t]["day_cls"] = day_cls; BK[t]["dis"] = dis
    log(t, "gap IS", gaps["IS"], "years", yr_sum["n_fri_beats_tranche"], "/", yr_sum["n_years"])
OUT["a_decomposition"] = SEC_A

# switch counts per book (F2)
sw = {}
for d in DAYS:
    mk = np.where(DM[d] & wm(IDX, START, FREEZE))[0]
    sw[d] = int((ABOVE[mk][1:] != ABOVE[mk][:-1]).sum())
OUT["a_switch_counts_IS"] = sw

# =============================================================== section (b): day-of-week & mechanism
log("section b: day-of-week statistics and first-holding-day tests")
ERAS = {"1985-1999": (pd.Timestamp("1985-10-02"), pd.Timestamp("1999-12-31")),
        "2000-2009": (pd.Timestamp("2000-01-01"), pd.Timestamp("2009-12-31")),
        "2010-2026IS": (pd.Timestamp("2010-01-01"), FREEZE),
        "OOS": (OOS_A, LAST)}


def dow_table(ret, dates):
    dow = pd.DatetimeIndex(dates).dayofweek.values
    out = {}
    for en, (a, b) in ERAS.items():
        m = wm(dates, a, b) & np.isfinite(ret)
        e = {}
        for i, d in enumerate(DAYS):
            x = ret[m & (dow == i)]
            if len(x) < 3:
                continue
            e[d] = dict(n=int(len(x)), mean_bp=float(x.mean() * 1e4), sd_bp=float(x.std(ddof=1) * 1e4),
                        ann_vol=float(x.std(ddof=1) * np.sqrt(252)),
                        t=float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), pos_share=float((x > 0).mean()))
        mo = ret[m & (dow == 0)]; ot = ret[m & (dow != 0)]
        if len(mo) > 3:
            e["Mon_minus_rest"] = dict(diff_bp=float((mo.mean() - ot.mean()) * 1e4),
                                       welch_t=float((mo.mean() - ot.mean()) /
                                                     np.sqrt(mo.var(ddof=1) / len(mo) + ot.var(ddof=1) / len(ot))),
                                       var_ratio_Mon_vs_rest=float(mo.var(ddof=1) / ot.var(ddof=1)))
        # next-session autocorrelation by weekday of the first day
        nxt = np.r_[ret[1:], np.nan]
        ac = {}
        for i, d in enumerate(DAYS):
            mm = m & (dow == i) & np.isfinite(nxt) & np.r_[wm(dates, a, b)[1:], False]
            if mm.sum() > 10:
                ac[d] = float(np.corrcoef(ret[mm], nxt[mm])[0, 1])
        e["corr_with_next_session"] = ac
        mm = m & np.isfinite(nxt) & np.r_[wm(dates, a, b)[1:], False]
        e["corr_with_next_session_all"] = float(np.corrcoef(ret[mm], nxt[mm])[0, 1])
        out[en] = e
    return out


SPXR = P["spx"].values.astype(float)
SPXR = np.where(np.arange(N) == 0, np.nan, SPXR)
SEC_B = dict(dow_ndx=dow_table(NDXR.astype(float), IDX), dow_spx=dow_table(SPXR, IDX))

mech = {}
for t in TIERS:
    b = BK[t]; rt = b["r"]["tranche"]; isy = {wn: yrs_of(IDX, *ab) for wn, ab in WIN.items()}
    res = {}
    # M1: each book's excess over the tranche by holding-day number k (sessions since its own decision)
    hd = {}
    for d in DAYS:
        li = last_idx(DM[d])
        k = np.full(N, -1); k[1:] = np.arange(1, N) - li[:-1]; k[1:][li[:-1] < 0] = -1
        kb = np.where(k >= 6, 6, k)
        e = b["r"][d] - rt
        hd[d] = {wn: {("k6+" if kk == 6 else f"k{kk}"): float(e[wm(IDX, *ab) & (kb == kk)].sum() / isy[wn] * 100)
                      for kk in range(1, 7)} for wn, ab in WIN.items()}
        for wn, ab in WIN.items():
            tot = float(e[wm(IDX, *ab)].sum() / isy[wn] * 100)
            hd[d][wn]["total"] = tot
            hd[d][wn]["first_day_share"] = hd[d][wn]["k1"] / tot if abs(tot) > 1e-9 else None
    res["M1_excess_vs_tranche_by_holding_day_pp_per_yr"] = hd
    # Friday gap by weekday of the return day (Mon = first holding day for the Friday book)
    g = b["g"]
    res["M1b_fri_gap_by_return_weekday_pp_per_yr"] = {
        wn: {d: float(g[wm(IDX, *ab) & DM[d]].sum() / isy[wn] * 100) for d in DAYS} for wn, ab in WIN.items()}
    # M2: lag-2 difference-in-differences
    m2 = {}
    for wn, ab in WIN.items():
        f1, t1 = st1(IDX, b["r"]["Fri"], *ab), st1(IDX, b["r"]["tranche"], *ab)
        f2, t2 = st1(IDX, b["r2"]["Fri"], *ab), st1(IDX, b["r2"]["tranche"], *ab)
        gap1 = (f1["cagr"] - t1["cagr"]) * 100; gap2 = (f2["cagr"] - t2["cagr"]) * 100
        m2[wn] = dict(fri_lag1_cagr=f1["cagr"], tranche_lag1_cagr=t1["cagr"], fri_lag2_cagr=f2["cagr"],
                      tranche_lag2_cagr=t2["cagr"], gap_lag1_pp=gap1, gap_lag2_pp=gap2,
                      share_of_gap_removed_by_lag2=(1 - gap2 / gap1) if abs(gap1) > 1e-9 else None,
                      calmar_gap_lag1=f1["calmar"] - t1["calmar"], calmar_gap_lag2=f2["calmar"] - t2["calmar"],
                      lag_cost_fri_pp=(f1["cagr"] - f2["cagr"]) * 100,
                      lag_cost_tranche_pp=(t1["cagr"] - t2["cagr"]) * 100,
                      lag2_books_cagr={d: st1(IDX, b["r2"][d], *ab)["cagr"] for d in DAYS})
    res["M2_lag2_diff_in_diff"] = m2
    # M3: value of the last day's information for an identical holding window: d lag1 vs previous-day lag2
    prev = {"Mon": "Fri", "Tue": "Mon", "Wed": "Tue", "Thu": "Wed", "Fri": "Thu"}
    m3 = {}
    for wn, ab in WIN.items():
        v = {d: (st1(IDX, b["r"][d], *ab)["cagr"] - st1(IDX, b["r2"][prev[d]], *ab)["cagr"]) * 100 for d in DAYS}
        m3[wn] = dict(info_value_pp=v, mean_over_weekdays=float(np.mean(list(v.values()))),
                      fri_minus_mean=float(v["Fri"] - np.mean(list(v.values()))))
    res["M3_last_day_information_value"] = m3
    # M4: mean channel vs timing channel of the lev-sleeve gap by return weekday (exact identity)
    x = b["R"][:, 0] - b["R"][:, 1]                    # lev sleeve minus IEF (gold weight is constant)
    dh = b["H"]["Fri"][:, 0] - sum(b["H"][d][:, 0] for d in DAYS) / 5
    m4 = {}
    for wn, ab in WIN.items():
        m = wm(IDX, *ab); rows = {}
        for d in DAYS:
            mm = m & DM[d]; n = mm.sum()
            mean_ch = n * dh[mm].mean() * x[mm].mean()
            tim_ch = n * (np.mean(dh[mm] * x[mm]) - dh[mm].mean() * x[mm].mean())
            rows[d] = dict(mean_channel_pp_per_yr=float(mean_ch / isy[wn] * 100),
                           timing_channel_pp_per_yr=float(tim_ch / isy[wn] * 100),
                           mean_dh=float(dh[mm].mean()), mean_x_bp=float(x[mm].mean() * 1e4))
        rows["total_mean_channel"] = float(sum(v["mean_channel_pp_per_yr"] for k, v in rows.items() if k in DAYS))
        rows["total_timing_channel"] = float(sum(v["timing_channel_pp_per_yr"] for k, v in rows.items() if k in DAYS))
        m4[wn] = rows
    res["M4_mean_vs_timing_channel"] = m4
    mech[t] = res

# M5 (signal-level, tier-independent): next-session / holding-window NDX return after each book's decisions
m5 = {}
nx = np.r_[NDXR[1:], np.nan]
for wn, ab in WIN.items():
    rows = {}
    for d in DAYS:
        allm = np.where(DM[d])[0]
        mk = np.where(DM[d] & wm(IDX, *ab))[0]
        mk = mk[mk < N - 1]
        st_ = ABOVE[mk]
        j0 = int(np.searchsorted(allm, mk[0]))
        first_prev = ABOVE[allm[j0 - 1]] if j0 > 0 else st_[0]
        prev_ = np.r_[first_prev, st_[:-1]]
        nxt = nx[mk]
        win_ret = []
        for i, p in enumerate(mk):
            q = mk[i + 1] if i + 1 < len(mk) else min(p + 5, N - 1)
            win_ret.append(np.prod(1 + NDXR[p + 1:q + 1]) - 1)
        win_ret = np.array(win_ret)
        def agg(sel):
            return dict(n=int(sel.sum()), next_bp=float(np.nanmean(nxt[sel]) * 1e4) if sel.sum() else None,
                        window_pct=float(np.nanmean(win_ret[sel]) * 100) if sel.sum() else None)
        rows[d] = dict(close_above=agg(st_), close_below=agg(~st_),
                       switch_on=agg(st_ & ~prev_), switch_off=agg(~st_ & prev_))
    m5[wn] = rows
mech["M5_ndx_after_decisions_by_weekday"] = m5
SEC_B["mechanism"] = mech

# diagnostic criterion (b) evaluated for F2 (IS) and reported for all tiers
diag = {}
for t in TIERS:
    m2 = mech[t]["M2_lag2_diff_in_diff"]["IS"]
    m1 = mech[t]["M1_excess_vs_tranche_by_holding_day_pp_per_yr"]["Fri"]["IS"]
    c1 = (m2["share_of_gap_removed_by_lag2"] or 0) >= 2 / 3
    c2 = (m1["first_day_share"] or 0) >= 2 / 3
    diag[t] = dict(lag2_share_removed=m2["share_of_gap_removed_by_lag2"], first_day_share=m1["first_day_share"],
                   crit_i=bool(c1), crit_ii=bool(c2), first_holding_day_explains=bool(c1 and c2))
SEC_B["diagnostic_first_holding_day"] = diag
OUT["b_weekday_and_mechanism"] = SEC_B
log("b done", json.dumps(diag))

# =============================================================== section (d) placebo on NDX tiers
log("section d: placebo schedules")
WIN_PL = dict(WIN); WIN_PL["OOS"] = (OOS_A, LAST)
SEC_D = {}
for ti, t in enumerate(TIERS):
    b = BK[t]
    rng = np.random.default_rng(SEED if t == "F2" else SEED + ti + 1)
    masks = placebo_masks(IDX, NPL, rng)
    ps = placebo_stats(b["W"], b["R"], IDX, masks, WIN_PL); count("placebo_ndx_random_weekday_per_week", NPL)
    res = {}
    for wn, ab in WIN_PL.items():
        e = {}
        for k in DAYS + ["tranche"]:
            if wn == "OOS":
                v = period_ret(IDX, b["r"][k], *ab)
                e[k] = dict(total=v, pct_total=pctile(ps[wn]["total"], v))
            else:
                s = st1(IDX, b["r"][k], *ab)
                e[k] = dict(cagr=s["cagr"], calmar=s["calmar"], maxdd=s["maxdd"],
                            pct_cagr=pctile(ps[wn]["cagr"], s["cagr"]), pct_calmar=pctile(ps[wn]["calmar"], s["calmar"]),
                            pct_maxdd=pctile(ps[wn]["maxdd"], s["maxdd"]))
        if wn == "OOS":
            e["placebo_total"] = dsum(ps[wn]["total"])
        else:
            e["placebo_cagr"] = dsum(ps[wn]["cagr"]); e["placebo_calmar"] = dsum(ps[wn]["calmar"])
            e["placebo_maxdd"] = dsum(ps[wn]["maxdd"])
            e["fri_z_cagr"] = (e["Fri"]["cagr"] - e["placebo_cagr"]["mean"]) / e["placebo_cagr"]["sd"]
            e["fri_z_calmar"] = (e["Fri"]["calmar"] - e["placebo_calmar"]["mean"]) / e["placebo_calmar"]["sd"]
        res[wn] = e
    if t == "F2":
        both = ((ps["H1"]["calmar"] >= res["H1"]["Fri"]["calmar"]) & (ps["H2"]["calmar"] >= res["H2"]["Fri"]["calmar"]))
        res["share_placebo_ge_fri_calmar_both_halves"] = float(both.mean())
        # secondary null: one random weekday per 13-week block (regular 5-day spacing within blocks)
        rngb = np.random.default_rng(SEED + 100)
        mb = placebo_masks(IDX, NPL, rngb, block_weeks=13)
        pb = placebo_stats(b["W"], b["R"], IDX, mb, WIN); count("placebo_ndx_random_weekday_per_13wk_block", NPL)
        res["block13_null"] = {wn: dict(pct_cagr=pctile(pb[wn]["cagr"], res[wn]["Fri"]["cagr"]),
                                        pct_calmar=pctile(pb[wn]["calmar"], res[wn]["Fri"]["calmar"]),
                                        pct_tranche_cagr=pctile(pb[wn]["cagr"], res[wn]["tranche"]["cagr"]),
                                        placebo_cagr=dsum(pb[wn]["cagr"]), placebo_calmar=dsum(pb[wn]["calmar"]),
                                        placebo_maxdd=dsum(pb[wn]["maxdd"]))
                               for wn in WIN}
        # every-5th-session offset cycles (no calendar effect possible)
        off = {}
        for o in range(5):
            r_o = run_book(b["W"], b["R"], (np.arange(N) % 5) == o)[0]; count("ndx_5session_offset_cycles")
            off[o] = {wn: st1(IDX, r_o, *ab) for wn, ab in WIN.items()}
        res["offset_cycles_5session"] = dict(
            IS_cagr={o: v["IS"]["cagr"] for o, v in off.items()},
            IS_calmar={o: v["IS"]["calmar"] for o, v in off.items()},
            IS_cagr_spread_pp=float((max(v["IS"]["cagr"] for v in off.values()) -
                                     min(v["IS"]["cagr"] for v in off.values())) * 100),
            weekday_IS_cagr_spread_pp=float((max(SEC_A["F2"]["books"][d]["IS"]["cagr"] for d in DAYS) -
                                             min(SEC_A["F2"]["books"][d]["IS"]["cagr"] for d in DAYS)) * 100))
    SEC_D[t] = res
    log(t, "placebo pct Fri calmar IS/H1/H2:", [round(res[w]["Fri"]["pct_calmar"], 1) for w in WIN],
        "cagr:", [round(res[w]["Fri"]["pct_cagr"], 1) for w in WIN])
OUT["d_placebo_ndx"] = SEC_D

# =============================================================== section (c) cross-market replication
log("section c: cross-market replication")
Y = pd.read_parquet(YF_PATH)


def yser(t, col="close"):
    s = Y[Y.ticker == t].set_index("date")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index <= LAST].astype(float)


def asof(s, dates):
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


# US-calendar total-return indices that reach back before the panel (1985-10-02) for own-calendar markets
irx, tnx = yser("^IRX"), yser("^TNX")
rf_us = pd.concat([irx[irx.index < IDX[0]] / 100.0, P["rf"]]).sort_index()
yv = P["tnx"] / 100.0; dy = yv.diff(); carry = yv.shift(1) / 252
cal = pd.concat([P["ief_syn"] - carry, dy], axis=1).dropna(); cal.columns = ["y", "dy"]
Xc = np.column_stack([np.ones(len(cal)), -cal["dy"].values, 0.5 * cal["dy"].values ** 2])
a_, D_, Cx_ = np.linalg.lstsq(Xc, cal["y"].values, rcond=None)[0]
yt = tnx / 100.0
ief_pre = (a_ + yt.shift(1) / 252 - D_ * yt.diff() + 0.5 * Cx_ * yt.diff() ** 2)
ief_pre = ief_pre[ief_pre.index < IDX[0]].fillna(0)
ief_us = pd.concat([ief_pre, P["ief"].fillna(0)]).sort_index()
gold_pre = (irx[irx.index < IDX[0]] / 100.0 / 252)          # cash stands in for gold before 1985-10-02
gold_us = pd.concat([gold_pre, P["gold_phys"].fillna(0)]).sort_index()
IEF_IDX = (1 + ief_us).cumprod(); GOLD_IDX = (1 + gold_us).cumprod()
syn_chk = pd.concat([(a_ + yt.shift(1) / 252 - D_ * yt.diff() + 0.5 * Cx_ * yt.diff() ** 2).reindex(IDX),
                     P["ief_syn"]], axis=1).dropna()
syn_chk = syn_chk[syn_chk.index < "2002-07-30"]
OUT["c_meta"] = dict(ief_calibration_recovered=dict(alpha_ann=float(a_ * 252), duration=float(D_), convexity=float(Cx_)),
                     ief_synthetic_rebuild_corr_vs_panel=float(syn_chk.iloc[:, 0].corr(syn_chk.iloc[:, 1])),
                     note=("Panel markets (SPX, RUT, TSX, HSI) use panel returns on the US calendar; DAX, N225, IXIC "
                           "(and an HSI own-calendar robustness row) use Yahoo closes on their own calendars. Off-leg "
                           "IEF and gold are the panel sleeves mapped as-of onto each calendar; before 1985-10-02 IEF "
                           "is rebuilt from ^TNX with the panel's own calibration and gold is replaced by T-bills "
                           "(gold is a constant 15% in every book, so it cannot favour a weekday). rf = panel rf "
                           "(^IRX before 1985-10-02). The 2x sleeve is 2r - 0.95%/252 - rf_d on every market."))


def market_panel(col):
    r = P[col].fillna(0).values.astype(float)
    nz = np.where(r != 0)[0]
    start = max(START, IDX[min(nz[0] + 250, N - 1)])
    lvl = np.cumprod(1 + r)
    return dict(dates=IDX, lvl=lvl, ret=r, rf_d=P["rf_d"].values, ief=P["ief"].fillna(0).values,
                gold=P["gold_phys"].fillna(0).values, start=start, calendar="US (panel)")


def market_yf(ticker, start_fixed=None):
    c = yser(ticker)
    c = c[c > 0]
    dates = c.index
    r = c.pct_change().fillna(0).values
    start = pd.Timestamp(start_fixed) if start_fixed else max(START, dates[250])
    rf_d = (asof(rf_us, dates).bfill() / 252).values
    ief = asof(IEF_IDX, dates).pct_change().fillna(0).values
    gold = asof(GOLD_IDX, dates).pct_change().fillna(0).values
    return dict(dates=dates, lvl=c.values, ret=r, rf_d=rf_d, ief=ief, gold=gold, start=start, calendar="own")


MKTS = {"NDX-2xsim (reference)": market_panel("ndx_ret"), "SPX": market_panel("spx"), "RUT": market_panel("rut"),
        "TSX": market_panel("tsx"), "HSI": market_panel("hsi"), "DAX": market_yf("^GDAXI"),
        "N225": market_yf("^N225", "1975-01-01"), "IXIC": market_yf("^IXIC", "1975-01-01"),
        "HSI-own-calendar (robustness)": market_yf("^HSI")}
DECIDING = ["SPX", "RUT", "TSX", "HSI", "DAX", "N225", "IXIC"]
SEC_C = {}
for mi, (name, M) in enumerate(MKTS.items()):
    dates = pd.DatetimeIndex(M["dates"])
    month = dates.month.values
    W, above_m, dist_m = f_weights(pd.Series(M["lvl"], index=dates), month, 0.30)
    lev_r = 2 * M["ret"] - 0.0095 / 252 - M["rf_d"]
    R = np.column_stack([lev_r, M["ief"], M["gold"]])
    dow = dates.dayofweek.values
    st, en = M["start"], FREEZE
    mid = st + (en - st) / 2
    mid = pd.Timestamp(mid.date())
    wins = {"IS": (st, en), "H1": (st, mid), "H2": (mid + pd.Timedelta(days=1), en)}
    if name == "IXIC":
        wins["pre-NDX 1975-1987"] = (pd.Timestamp("1975-01-01"), pd.Timestamp("1987-12-31"))
        wins["1988-freeze"] = (START, FREEZE)
    books, books2 = {}, {}
    for i, d in enumerate(DAYS):
        books[d] = run_book(W, R, dow == i)[0]; count("cross_market_weekday_books")
        books2[d] = run_book(W, R, dow == i, lag=2)[0]; count("cross_market_weekday_books_lag2")
    books["tranche"] = sum(books[d] for d in DAYS) / 5
    books2["tranche"] = sum(books2[d] for d in DAYS) / 5
    rnx = np.r_[M["ret"][1:], np.nan]
    res = dict(calendar=M["calendar"], start=str(st.date()), mid=str(mid.date()), windows={})
    rng = np.random.default_rng(SEED + 10 + mi)
    masks = placebo_masks(dates, NPL, rng)
    ps = placebo_stats(W, R, dates, masks, wins); count("placebo_cross_market", NPL)
    for wn, ab in wins.items():
        s = {k: st1(dates, books[k], *ab) for k in DAYS + ["tranche"]}
        cg = {d: s[d]["cagr"] for d in DAYS}; cm = {d: s[d]["calmar"] for d in DAYS}
        res["windows"][wn] = dict(
            stats={k: dict(cagr=v["cagr"], maxdd=v["maxdd"], calmar=v["calmar"]) for k, v in s.items()},
            best_by_cagr=max(cg, key=cg.get), best_by_calmar=max(cm, key=cm.get),
            fri_rank_cagr=int(sorted(cg.values(), reverse=True).index(cg["Fri"]) + 1),
            fri_rank_calmar=int(sorted(cm.values(), reverse=True).index(cm["Fri"]) + 1),
            worst_by_calmar=min(cm, key=cm.get),
            fri_minus_tranche_cagr_pp=(s["Fri"]["cagr"] - s["tranche"]["cagr"]) * 100,
            fri_minus_tranche_calmar=s["Fri"]["calmar"] - s["tranche"]["calmar"],
            fri_beats_tranche_both=bool(s["Fri"]["cagr"] > s["tranche"]["cagr"] and
                                        s["Fri"]["calmar"] > s["tranche"]["calmar"]),
            placebo_pct_fri_cagr=pctile(ps[wn]["cagr"], s["Fri"]["cagr"]),
            placebo_pct_fri_calmar=pctile(ps[wn]["calmar"], s["Fri"]["calmar"]),
            placebo_pct_by_day_calmar={d: pctile(ps[wn]["calmar"], s[d]["calmar"]) for d in DAYS})
        s2f, s2t = st1(dates, books2["Fri"], *ab), st1(dates, books2["tranche"], *ab)
        res["windows"][wn]["lag2_fri_minus_tranche_cagr_pp"] = (s2f["cagr"] - s2t["cagr"]) * 100
        res["windows"][wn]["lag2_fri_minus_tranche_calmar"] = s2f["calmar"] - s2t["calmar"]
        mwin = wm(dates, *ab) & np.isfinite(rnx) & np.r_[wm(dates, *ab)[1:], False]
        acd = {}
        for i, d in enumerate(DAYS):
            mm = mwin & (dow == i)
            acd[d] = dict(pearson=float(np.corrcoef(M["ret"][mm], rnx[mm])[0, 1]),
                          spearman=float(pd.Series(M["ret"][mm]).corr(pd.Series(rnx[mm]), method="spearman")))
        res["windows"][wn]["next_session_autocorr_by_weekday"] = acd
    SEC_C[name] = res
    w = res["windows"]
    log(f"{name:30s} start {res['start']} IS best(Calmar) {w['IS']['best_by_calmar']} Fri rank {w['IS']['fri_rank_calmar']}"
        f" | H1 best {w['H1']['best_by_calmar']} Fri {w['H1']['fri_rank_calmar']} | H2 best {w['H2']['best_by_calmar']}"
        f" Fri {w['H2']['fri_rank_calmar']} | Fri-tr CAGR {w['IS']['fri_minus_tranche_cagr_pp']:+.2f}pp "
        f"Cal {w['IS']['fri_minus_tranche_calmar']:+.3f} | placebo pct Cal {w['IS']['placebo_pct_fri_calmar']:.0f}")
OUT["c_cross_market"] = SEC_C

# =============================================================== section (e) data / mechanical artifacts
log("section e: artifact checks")
SEC_E = {}
g_sp = yser("^GSPC"); g_sp = g_sp[(g_sp.index >= IDX[0])]
SEC_E["E1_calendar_vs_GSPC"] = dict(gspc_days_missing_from_ndx=[str(d.date()) for d in g_sp.index.difference(IDX)],
                                    ndx_days_not_gspc_days=[str(d.date()) for d in IDX.difference(g_sp.index)],
                                    weekend_rows=int((DOW >= 5).sum()))
ndx_lvl = P["ndx"].values
zr = {}
for en, (a, b) in {"1985-1990": ("1985-10-02", "1990-12-31"), "1991-1999": ("1991-01-01", "1999-12-31"),
                   "2000-2009": ("2000-01-01", "2009-12-31"), "2010-2026": ("2010-01-01", str(LAST.date()))}.items():
    m = wm(IDX, a, b) & (np.arange(N) > 0)
    rr = NDXR[m]
    zr[en] = dict(days=int(m.sum()), zero_returns=int((rr == 0).sum()),
                  abs_ret_lt_1bp=int((np.abs(rr) < 1e-4).sum()),
                  zero_by_weekday={d: int(((NDXR == 0) & m & DM[d]).sum()) for d in DAYS})
SEC_E["E2_stale_closes_ndx"] = zr
ixr = asof(yser("^IXIC"), IDX).pct_change().values
spr = asof(g_sp, IDX).pct_change().values
al = {}
for en, (a, b) in {"1985-1989": ("1985-10-03", "1989-12-31"), "1990-1999": ("1990-01-01", "1999-12-31"),
                   "2000-2009": ("2000-01-01", "2009-12-31"), "2010-2026": ("2010-01-01", str(FREEZE.date()))}.items():
    m = wm(IDX, a, b)
    def cc(x, y, k):
        ys = np.roll(y, -k)
        mm = m & np.isfinite(x) & np.isfinite(ys)
        if k > 0:
            mm[-k:] = False
        if k < 0:
            mm[:-k] = False
        return float(np.corrcoef(x[mm], ys[mm])[0, 1])
    al[en] = dict(ndx_vs_ixic={f"lag{k:+d}": cc(NDXR, ixr, k) for k in (-1, 0, 1)},
                  ndx_vs_gspc={f"lag{k:+d}": cc(NDXR, spr, k) for k in (-1, 0, 1)},
                  ndx_ac1=cc(NDXR, NDXR, 1), ixic_ac1=cc(ixr, ixr, 1), gspc_ac1=cc(spr, spr, 1))
SEC_E["E3_date_alignment_and_autocorr"] = dict(
    note="corr(ndx_t, other_{t+k}); a 1-day timestamp shift would put the peak at k=+-1 instead of 0", by_era=al)
e3b = {}
nxn = np.r_[NDXR[1:], np.nan]; nxs = np.r_[spr[1:], np.nan]
for en, (a, b) in {"1985-10..1999 (incl. Oct-87)": ("1985-10-03", "1999-12-31"), "1988-1999": ("1988-01-01", "1999-12-31"),
                   "2000-2006": ("2000-01-01", "2006-12-31"), "2007-2026IS": ("2007-01-01", str(FREEZE.date()))}.items():
    m = wm(IDX, a, b) & np.r_[wm(IDX, a, b)[1:], False]
    row = {}
    for nm, x, nx_ in (("NDX", NDXR, nxn), ("SPX", spr, nxs)):
        rr = {}
        for i, d in enumerate(DAYS):
            mm = m & DM[d] & np.isfinite(x) & np.isfinite(nx_)
            rr[d] = dict(pearson=float(np.corrcoef(x[mm], nx_[mm])[0, 1]),
                         spearman=float(pd.Series(x[mm]).corr(pd.Series(nx_[mm]), method="spearman")))
        row[nm] = rr
    e3b[en] = row
SEC_E["E3b_next_session_autocorr_by_weekday"] = e3b
o_ = asof(yser("^NDX", "open"), IDX).values
pc_ = np.r_[np.nan, ndx_lvl[:-1]]
SEC_E["E4_ndx_open_quality"] = {str(y): float(np.mean(o_[(IDX.year == y)] == pc_[(IDX.year == y)]))
                                for y in range(1985, 1999)}
# E5 holidays
wk = week_codes(IDX)
m_is = wm(IDX, START, FREEZE)
nwk_is = len(set(wk[m_is]))
hol = {d: int(nwk_is - len(set(wk[m_is & DM[d]]))) for d in DAYS}
wkday_present = pd.DataFrame({"wk": wk, "dow": DOW})


def subst_mark(dn):
    """weekday dn; if it is a holiday, the next trading day of the same week (Fri: last trading day)."""
    m = np.zeros(N, bool)
    df = pd.DataFrame({"wk": wk, "dow": DOW, "pos": np.arange(N)})
    for _, gg in df.groupby("wk"):
        c = gg[gg.dow >= dn]
        m[(c.pos.iloc[0] if (len(c) and dn < 4) else gg.pos.iloc[-1])] = True
    return m


SUB = {d: subst_mark(i) for i, d in enumerate(DAYS)}
e5 = dict(weeks_IS=nwk_is, weeks_without_that_weekday_IS=hol,
          mark_equals_trading_fridays=bool((P["mark"].values == DM["Fri"]).all()),
          mark_equals_engine_WFRI_rule=bool((P["mark"].values ==
                                             pd.Series(False, index=IDX).mask(
                                                 IDX.isin(P["ndx"].resample("W-FRI").last().index), True).values).all()),
          mark_wk_equals_last_trading_day_of_week=bool((P["mark_wk"].values == SUB["Fri"]).all()),
          fridays_skipped_IS=int((m_is & P["mark_wk"].values & ~P["mark"].values).sum()))
for t in TIERS:
    b = BK[t]
    rs = {d: run_book(b["W"], b["R"], SUB[d])[0] for d in DAYS}; count("holiday_substituted_books", 5)
    rs["tranche"] = sum(rs[d] for d in DAYS) / 5
    rwk = run_book(b["W"], b["R"], P["mark_wk"].values)[0]; count("mark_wk_book")
    e5[t] = dict(subst={k: {wn: dict(cagr=st1(IDX, v, *ab)["cagr"], calmar=st1(IDX, v, *ab)["calmar"],
                                     maxdd=st1(IDX, v, *ab)["maxdd"]) for wn, ab in WIN.items()} for k, v in rs.items()},
                 mark_wk={wn: st1(IDX, rwk, *ab) for wn, ab in WIN.items()},
                 fri_skip={wn: st1(IDX, b["r"]["Fri"], *ab) for wn, ab in WIN.items()})
    e5[t]["subst_fri_minus_tranche"] = {wn: dict(
        d_cagr_pp=(e5[t]["subst"]["Fri"][wn]["cagr"] - e5[t]["subst"]["tranche"][wn]["cagr"]) * 100,
        d_calmar=e5[t]["subst"]["Fri"][wn]["calmar"] - e5[t]["subst"]["tranche"][wn]["calmar"]) for wn in WIN}
SEC_E["E5_holidays_and_mark_definition"] = e5
# E6 options-expiry Fridays (3rd Friday) vs other Fridays: gap per decision (F2)
fpos = np.where(DM["Fri"])[0]
third = np.array([(15 <= IDX[p].day <= 21) for p in fpos])
opx = set(fpos[third])
e6 = {}
for t in TIERS:
    g = BK[t]["g"]; decn = BK[t]["decn"]
    rows = {}
    for wn, ab in WIN.items():
        m = wm(IDX, *ab)
        s = pd.Series(g[m]).groupby(decn[m]).sum()
        isop = np.array([p in opx for p in s.index])
        so, sn = s[isop].values, s[~isop].values
        rows[wn] = dict(opex_weeks=int(isop.sum()), other_weeks=int((~isop).sum()),
                        opex_mean_bp_per_week=float(so.mean() * 1e4),
                        other_mean_bp_per_week=float(sn.mean() * 1e4),
                        opex_share_of_gap=float(so.sum() / s.sum()),
                        welch_t_opex_minus_other=float((so.mean() - sn.mean()) /
                                                       np.sqrt(so.var(ddof=1) / len(so) + sn.var(ddof=1) / len(sn))))
    e6[t] = rows
SEC_E["E6_opex_fridays"] = e6
# E7 data quality of the other markets used in (c)
e7 = {}
for name in ["DAX", "N225", "IXIC", "HSI-own-calendar (robustness)"]:
    M = MKTS[name]; dts = pd.DatetimeIndex(M["dates"]); r = M["ret"]
    rows = {}
    for dec_ in range(1970, 2030, 10):
        m = (dts.year >= dec_) & (dts.year < dec_ + 10) & (dts >= M["start"] - pd.Timedelta(days=400))
        if m.sum() < 50:
            continue
        rows[f"{dec_}s"] = dict(days=int(m.sum()), zero_ret_share=float((r[m] == 0).mean()),
                                days_per_year=float(m.sum() / max(1, len(set(dts.year[m])))),
                                weekday_counts={d: int((m & (dts.dayofweek == i)).sum()) for i, d in enumerate(DAYS)})
    e7[name] = rows
SEC_E["E7_other_market_data_quality"] = e7
OUT["e_artifacts"] = SEC_E

# =============================================================== section (f) selection-artifact grid (diagnostic)
log("section f: selection-artifact grid (diagnostic only; nothing is selected)")
R_F2 = P[["qld", "ief", "gold_phys"]].fillna(0).values
MON_NDX = IDX.month.values
W_chk = f_weights(P["ndx"], MON_NDX, 0.30)[0]
SEC_F = dict(check_span100_equals_F2_weights=float(np.abs(W_chk - BK["F2"]["W"]).max()))


def fri_vs_tranche(W):
    rr = {d: run_book(W, R_F2, DM[d])[0] for d in DAYS}
    count("selection_grid_weekday_books", 5)
    rr["tranche"] = sum(rr[d] for d in DAYS) / 5
    out = {}
    for wn, ab in WIN.items():
        s = {k: st1(IDX, v, *ab) for k, v in rr.items()}
        cm = {d: s[d]["calmar"] for d in DAYS}; cg = {d: s[d]["cagr"] for d in DAYS}
        out[wn] = dict(fri_cagr=s["Fri"]["cagr"], tranche_cagr=s["tranche"]["cagr"],
                       fri_calmar=s["Fri"]["calmar"], tranche_calmar=s["tranche"]["calmar"],
                       d_cagr_pp=(s["Fri"]["cagr"] - s["tranche"]["cagr"]) * 100,
                       d_calmar=s["Fri"]["calmar"] - s["tranche"]["calmar"],
                       fri_rank_calmar=int(sorted(cm.values(), reverse=True).index(cm["Fri"]) + 1),
                       fri_rank_cagr=int(sorted(cg.values(), reverse=True).index(cg["Fri"]) + 1),
                       mon_rank_calmar=int(sorted(cm.values(), reverse=True).index(cm["Mon"]) + 1),
                       best_calmar=max(cm, key=cm.get))
    return out


SEC_F["ema_span"] = {}
for span in [50, 75, 100, 125, 150, 200, 250]:
    SEC_F["ema_span"][span] = fri_vs_tranche(f_weights(P["ndx"], MON_NDX, 0.30, span=span)[0])
    v = SEC_F["ema_span"][span]
    log(f"  EMA {span:3d}: Fri-tr CAGR IS/H1/H2 {v['IS']['d_cagr_pp']:+.2f}/{v['H1']['d_cagr_pp']:+.2f}/"
        f"{v['H2']['d_cagr_pp']:+.2f}  Calmar {v['IS']['d_calmar']:+.3f}/{v['H1']['d_calmar']:+.3f}/"
        f"{v['H2']['d_calmar']:+.3f}  Fri rank IS {v['IS']['fri_rank_calmar']}  best {v['IS']['best_calmar']}")
grid = []
for vt in [0.15, 0.20, 0.25, 0.30, 0.35]:
    for th in [0.20, 0.25, 0.30]:
        for po in [0.35, 0.50, 0.65]:
            o = fri_vs_tranche(f_weights(P["ndx"], MON_NDX, vt, chop_thr=th, chop_pos=po)[0])
            grid.append(dict(voltgt=vt, chop_thr=th, chop_pos=po, **{f"{wn}_{k}": v for wn in WIN for k, v in o[wn].items()}))
G = pd.DataFrame(grid)
summ = {}
for wn in WIN:
    summ[wn] = dict(n=int(len(G)), share_fri_beats_tranche_both=float(((G[f"{wn}_d_cagr_pp"] > 0) &
                                                                         (G[f"{wn}_d_calmar"] > 0)).mean()),
                    median_d_cagr_pp=float(G[f"{wn}_d_cagr_pp"].median()), min_d_cagr_pp=float(G[f"{wn}_d_cagr_pp"].min()),
                    max_d_cagr_pp=float(G[f"{wn}_d_cagr_pp"].max()),
                    median_d_calmar=float(G[f"{wn}_d_calmar"].median()),
                    share_fri_rank1_calmar=float((G[f"{wn}_fri_rank_calmar"] == 1).mean()),
                    share_mon_rank5_calmar=float((G[f"{wn}_mon_rank_calmar"] == 5).mean()))
SEC_F["plateau_subgrid_summary"] = summ
SEC_F["plateau_subgrid"] = grid
ema_rows = SEC_F["ema_span"]
SEC_F["ema_span_summary"] = {wn: dict(
    share_fri_beats_tranche_both=float(np.mean([(v[wn]["d_cagr_pp"] > 0) and (v[wn]["d_calmar"] > 0) for v in ema_rows.values()])),
    median_d_cagr_pp=float(np.median([v[wn]["d_cagr_pp"] for v in ema_rows.values()])),
    share_fri_rank1=float(np.mean([v[wn]["fri_rank_calmar"] == 1 for v in ema_rows.values()])))
    for wn in WIN}
log("  plateau sub-grid", json.dumps({wn: {k: round(v, 3) for k, v in s.items()} for wn, s in summ.items()}))
OUT["f_selection_artifact_grid"] = SEC_F

# =============================================================== conclusion: apply the pre-registered rule
log("applying the pre-registered decision rule")
f2 = SEC_A["F2"]["gap"]; pl = SEC_D["F2"]
A_parts = dict(
    H1_cagr_pos=f2["H1"]["d_cagr_pp"] > 0, H1_calmar_pos=f2["H1"]["d_calmar"] > 0,
    H2_cagr_pos=f2["H2"]["d_cagr_pp"] > 0, H2_calmar_pos=f2["H2"]["d_calmar"] > 0,
    H1_placebo_calmar_pct=pl["H1"]["Fri"]["pct_calmar"], H2_placebo_calmar_pct=pl["H2"]["Fri"]["pct_calmar"])
A_parts["H1_placebo_ge95"] = A_parts["H1_placebo_calmar_pct"] >= 95
A_parts["H2_placebo_ge95"] = A_parts["H2_placebo_calmar_pct"] >= 95
A = all(bool(v) for k, v in A_parts.items() if k.endswith(("_pos", "_ge95")))
B_rows = {m: SEC_C[m]["windows"]["IS"]["fri_beats_tranche_both"] for m in DECIDING}
B_n = int(sum(B_rows.values()))
B = B_n >= 5
STRUCT = bool(A and B)
B_sens = dict(  # reporting only - the rule above is applied exactly as written
    with_HSI_own_calendar_instead_of_panel=int(B_n - B_rows["HSI"] + SEC_C["HSI-own-calendar (robustness)"]["windows"]["IS"]["fri_beats_tranche_both"]),
    H1_count={m: SEC_C[m]["windows"]["H1"]["fri_beats_tranche_both"] for m in DECIDING},
    H2_count={m: SEC_C[m]["windows"]["H2"]["fri_beats_tranche_both"] for m in DECIDING},
    fri_rank1_calmar_IS=[m for m in DECIDING if SEC_C[m]["windows"]["IS"]["fri_rank_calmar"] == 1],
    mon_worst_calmar_IS=[m for m in DECIDING if SEC_C[m]["windows"]["IS"]["worst_by_calmar"] == "Mon"],
    placebo_pct_fri_calmar_IS={m: SEC_C[m]["windows"]["IS"]["placebo_pct_fri_calmar"] for m in DECIDING},
    placebo_pct_mon_calmar_IS={m: SEC_C[m]["windows"]["IS"]["placebo_pct_by_day_calmar"]["Mon"] for m in DECIDING})
B_sens["H1_n"] = int(sum(B_sens["H1_count"].values())); B_sens["H2_n"] = int(sum(B_sens["H2_count"].values()))
OUT["conclusion"] = dict(
    A_both_halves_NDX_F2=dict(parts={k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in A_parts.items()},
                              holds=bool(A)),
    B_other_markets=dict(per_market_fri_beats_tranche_both=B_rows, n_markets=B_n, needed=5, holds=bool(B),
                         sensitivity_reporting_only=B_sens),
    friday_structurally_better=STRUCT,
    planning_basis=("Friday book" if STRUCT else "5-weekday tranche (Friday premium = luck, treated as upside)"),
    first_holding_day_explains_premium_F2=SEC_B["diagnostic_first_holding_day"]["F2"]["first_holding_day_explains"])
OUT["variants_evaluated"] = dict(VAR, total=int(sum(VAR.values())))
OUT["runtime_s"] = time.time() - T0


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


json.dump(clean(OUT), open(OUT_JSON, "w"), indent=1)
log("conclusion", json.dumps(clean(OUT["conclusion"])))
log("variants", json.dumps(OUT["variants_evaluated"]))
log("saved", OUT_JSON)
