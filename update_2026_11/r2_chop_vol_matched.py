"""
ROUND 2 (PRE-REGISTERED) - matched-exposure test of the CHOP RULE and the VOLATILITY TARGET.
KEY = chop_vol_matched.   Writes update_2026_11/r2_chop_vol_matched.json and prints a summary.

Question: do the chop rule (half size when er20 < 0.25) and the vol target (scale = min(1, vt/vol20)) add
value beyond simply holding less NDX exposure?

Constructions (tiers F2 = QLD vt .30, F3 = TQQQ vt .20; IEF off-leg; permanent 15% gold_phys; 3 bps; lag 1):
  B   baseline = July-2026 rules as specified (nothing re-tuned)
  A0  no chop rule (engine = scale whenever risk-on) at the ORIGINAL vt                 [reference only]
  A   no chop rule with vt lowered by bisection so that average NDX exposure (= leverage x HELD sleeve
      weight, averaged over all days 1988-01-01..2026-07-24) equals B's; solved per tier AND per basis
  V0  no vol scaling (engine = 1 when risk-on; chop half-size and Aug-Sep half kept)    [reference only]
  V   V0 x constant c with c = E[B]/E[V0] (exposure is exactly linear in c); per tier and basis
Bases: PRIMARY = 5-weekday tranche, identical to main_tranche_check.py (mean of the five weekday books' daily
net returns; each book rebalances on its weekday, lag 1, 3 bps x its own turnover). Secondary = Friday book.
Cross-market (2x, vt .30, tranche): spx, rut, tsx, hsi from the panel (price = cumprod(1+r)), plus ^GDAXI,
^N225, ^IXIC from yfinance (cache r2_chop_vol_matched_yf.parquet; closes reindexed to the panel calendar with
ffill). Own-price EMA-100 / vol20 / er20; sleeve = 2r - 0.95%/252 - rf_d; same IEF + 15% gold_phys off-legs;
1990-01-01 (or first full year after a 1-year warm-up) .. 2026-07-24; exposure matched over that window.
PRE-REGISTERED DECISION RULE (applied exactly; separately for chop / vol target and for F2 / F3):
  RETAINED     tranche dCalmar(B - matched alt) >= +0.03 in BOTH halves (1988-01-01..2006-12-31 and
               2007-01-01..2026-07-24)  AND  cross-market median dCalmar >= +0.02
  REPLACEABLE  dCalmar < +0.03 in at least one half  AND  cross-market median dCalmar <= +0.02
  INCONCLUSIVE any other combination (retain by default)
All decisions use data to 2026-07-24 only. 2026-07-25..end is out-of-sample, information only.
Information-only extras (never used in a decision): paired block bootstrap of dCalmar, exposure re-matched
within each half, per-weekday books, max-drawdown episodes / binding rates, rule outcome at 10/15 bps and on the
Friday book, leave-one-out cross-market medians, and a NOT-pre-registered 3x / vt .20 cross-market run.
Conventions: "one-way turnover" = sum over sleeves |change in held weight| (common.py convention: a 10% switch
IEF->QLD is 0.20); "N bps" = N bps of NAV per unit of that turnover. CDaR95 = mean of the worst 5% of daily
drawdown values (negative numbers, like maxDD). Worst-5 = mean depth of the 5 deepest distinct drawdown
episodes (an episode ends at a new equity high).
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
OOS_A = pd.Timestamp("2026-07-25")
END = IDX[-1]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DOW = IDX.dayofweek.values
MARKS = {d: DOW == i for i, d in enumerate(DAYS)}
assert np.array_equal(P["mark"].values.astype(bool), MARKS["Fri"]), "P.mark must equal the Friday weekday mark"
LAST = {d: np.maximum.accumulate(np.where(m, np.arange(N), -1)) for d, m in MARKS.items()}

CHOP_THR, CHOP_POS, GOLD_W, CORE = 0.25, 0.5, 0.15, 0.85
SEASON = np.where(np.isin(P["month"].values, (8, 9)), 0.5, 1.0)
R_IEF = P["ief"].fillna(0).values
R_GOLD = P["gold_phys"].fillna(0).values
RF_D = P["rf_d"].fillna(0).values
GOLD_T = np.full(N, GOLD_W)
COSTS = {"3bps": 3e-4, "10bps": 10e-4, "15bps": 15e-4}
TRADE_EPS = 1e-6                         # a sleeve "trades" when its held weight changes by more than this
H1_END, H2_START = pd.Timestamp("2006-12-31"), pd.Timestamp("2007-01-01")
WIN_NDX = {"IS": (START, FREEZE), "H1": (START, H1_END), "H2": (H2_START, FREEZE),
           "T1_1988-99": (START, pd.Timestamp("1999-12-31")), "T2_2000-12": (pd.Timestamp("2000-01-01"), pd.Timestamp("2012-12-31")),
           "T3_2013-frz": (pd.Timestamp("2013-01-01"), FREEZE), "OOS": (OOS_A, END)}
RET_THR, MED_THR = 0.03, 0.02            # pre-registered thresholds

VARIANTS = []                            # every strategy variant whose RETURNS were evaluated
BISECT = {"exposure_only_evaluations": 0}  # bisection steps: exposure only, no return data touched


def log(*a):
    print(*a, flush=True)


def count(**kw):
    VARIANTS.append(kw)


def mask(a, b):
    return (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))


# ------------------------------------------------------------------ markets
def signals_from_returns(r):
    """own-price signals exactly as panel.py builds them for NDX (price = cumprod(1+r))."""
    price = pd.Series(np.cumprod(1.0 + np.nan_to_num(r)), index=IDX)
    ema = price.ewm(span=100, adjust=False).mean()
    vol20 = price.pct_change().rolling(20).std() * np.sqrt(252)
    er20 = price.diff(20).abs() / price.diff().abs().rolling(20).sum().replace(0, np.nan)
    return (price > ema).values, vol20.values, er20.values


def make_market(name, mult, r_lev, above, vol20, er20):
    er20 = np.asarray(er20, float)
    return dict(name=name, mult=float(mult), r_lev=np.nan_to_num(np.asarray(r_lev, float)),
                above=np.asarray(above, bool), vol20=np.asarray(vol20, float),
                clean=(~np.isnan(er20)) & (er20 >= CHOP_THR))


def targets(M, vt=None, chop=True, vol=True, c=1.0):
    """target weights (leveraged sleeve, IEF off-leg); gold is the constant GOLD_T. Mirrors common.strategy_weights."""
    if vol:
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.clip(vt / M["vol20"], 0, 1)
        sc = np.where(np.isnan(sc), 1.0, sc)
    else:
        sc = np.ones(N)
    sc = np.clip(sc * SEASON, 0, 1)
    if c != 1.0:
        sc = np.clip(sc * c, 0, 1)
    if chop:
        eq = np.where(M["above"], np.where(M["clean"], sc, CHOP_POS * sc), 0.0)
    else:
        eq = np.where(M["above"], sc, 0.0)
    wl = CORE * eq
    return wl, np.clip(CORE - wl, 0, None)


def held(w, d):
    """weight held on each day by the book that rebalances on weekday d (signal at the mark close, lag 1)."""
    li = LAST[d]
    h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
    out = np.empty(N)
    out[0] = 0.0
    out[1:] = h[:-1]
    return out


def books_of(basis):
    return DAYS if basis == "tranche" else ["Fri"]


def run(M, wl, wo, basis):
    """gross return, turnover and trade counts; tranche = mean of the 5 weekday books (main_tranche_check)."""
    bk = books_of(basis)
    k = float(len(bk))
    out = dict(gross=np.zeros(N), turn=np.zeros(N), turn_lev=np.zeros(N), hl=np.zeros(N),
               reb=np.zeros(N), orders=np.zeros(N), lev_trades=np.zeros(N))
    for d in bk:
        hl, ho, hg = held(wl, d), held(wo, d), held(GOLD_T, d)
        dl = np.abs(np.diff(hl, prepend=0.0))
        do = np.abs(np.diff(ho, prepend=0.0))
        dg = np.abs(np.diff(hg, prepend=0.0))
        out["gross"] += (hl * M["r_lev"] + ho * R_IEF + hg * R_GOLD) / k
        out["turn"] += (dl + do + dg) / k
        out["turn_lev"] += dl / k
        out["hl"] += hl / k
        tl, to, tg = dl > TRADE_EPS, do > TRADE_EPS, dg > TRADE_EPS
        out["reb"] += (tl | to | tg)          # executions summed over books (not divided by k)
        out["orders"] += tl.astype(float) + to + tg
        out["lev_trades"] += tl
    out["nbooks"] = int(k)
    return out


def net(R, cost):
    return R["gross"] - R["turn"] * cost


def avg_expo(M, wl, basis, a, b):
    hl = sum(held(wl, d) for d in books_of(basis)) / len(books_of(basis))
    return float(M["mult"] * hl[mask(a, b)].mean())


# ------------------------------------------------------------------ statistics
def worst_episodes(dd, k=5):
    under = (dd < 0).astype(int)
    edges = np.diff(np.concatenate([[0], under, [0]]))
    s, e = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if len(s) == 0:
        return 0.0, []
    depths = np.sort(np.array([dd[i:j].min() for i, j in zip(s, e)]))[:k]
    return float(depths.mean()), [float(x) for x in depths]


def st(r, a, b):
    """C.stats conventions (CAGR on calendar years, Sharpe without rf) + Ulcer, CDaR95, worst-5 episodes."""
    m = mask(a, b)
    rr = r[m]
    d = IDX[m]
    if len(rr) < 20:
        return None
    e = np.cumprod(1 + rr)
    yrs = max((d[-1] - d[0]).days / 365.25, 1 / 365.25)
    cagr = e[-1] ** (1 / yrs) - 1
    dd = e / np.maximum.accumulate(e) - 1
    mdd = float(dd.min())
    sd = rr.std(ddof=1)
    q = np.quantile(dd, 0.05)
    w5, w5l = worst_episodes(dd)
    it = int(np.argmin(dd))
    ip = int(np.argmax(e[:it + 1]))
    rec = np.where(e[it:] >= e[ip])[0]
    return dict(cagr=float(cagr), maxdd=mdd, calmar=float(cagr / abs(mdd)) if mdd < 0 else None,
                sharpe=float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else None, vol=float(sd * np.sqrt(252)),
                ulcer=float(np.sqrt(np.mean((100 * dd) ** 2))), cdar95=float(dd[dd <= q].mean()),
                worst5_mean=w5, worst5=w5l, total=float(e[-1] - 1), days=int(len(rr)),
                maxdd_peak=str(d[ip].date()), maxdd_trough=str(d[it].date()),
                maxdd_recovered=(str(d[it + rec[0]].date()) if len(rec) else None),
                start=str(d[0].date()), end=str(d[-1].date()))


def activity(R, a, b):
    m = mask(a, b)
    d = IDX[m]
    yrs = (d[-1] - d[0]).days / 365.25
    k = R["nbooks"]
    return dict(turnover_per_yr=float(R["turn"][m].sum() / yrs),
                turnover_lev_sleeve_per_yr=float(R["turn_lev"][m].sum() / yrs),
                rebalances_per_yr_all_books=float(R["reb"][m].sum() / yrs),
                orders_per_yr_all_books=float(R["orders"][m].sum() / yrs),
                lev_sleeve_trades_per_yr_all_books=float(R["lev_trades"][m].sum() / yrs),
                rebalances_per_yr_per_book=float(R["reb"][m].sum() / yrs / k),
                orders_per_yr_per_book=float(R["orders"][m].sum() / yrs / k),
                n_trades_window_all_books=int(R["reb"][m].sum()),
                n_orders_window_all_books=int(R["orders"][m].sum()),
                cost_drag_pct_per_yr={c: float(R["turn"][m].sum() / yrs * v) for c, v in COSTS.items()},
                books=k)


def binding_profile(M, vt, a, b):
    """how often each component actually changes the position (descriptive; risk-on days in the window)."""
    m = mask(a, b) & M["above"]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = vt / M["vol20"][m]
    vb = np.where(np.isnan(raw), False, raw < 1)
    sc = np.where(np.isnan(raw), 1.0, np.clip(raw, 0, 1))
    return dict(riskon_share_of_days=float(m.sum() / mask(a, b).sum()),
                vol_target_binding_share_of_riskon_days=float(vb.mean()),
                mean_vol_scale_on_riskon_days=float(sc.mean()),
                chop_half_share_of_riskon_days=float((~M["clean"][m]).mean()))


def expo_profile(M, R, a, b):
    m = mask(a, b)
    e = M["mult"] * R["hl"][m]
    on = e > 1e-12
    cap = CORE * M["mult"]
    return dict(avg_all_days=float(e.mean()), avg_riskon_days=float(e[on].mean()) if on.any() else 0.0,
                share_days_invested=float(on.mean()), share_invested_days_at_cap=float((e[on] >= cap * 0.999).mean()) if on.any() else 0.0,
                max=float(e.max()))


# ------------------------------------------------------------------ solvers (exposure only - no returns used)
def solve_A(M, vt_b, basis, a, b):
    tgt = avg_expo(M, targets(M, vt_b)[0], basis, a, b)

    def f(vt):
        BISECT["exposure_only_evaluations"] += 1
        return avg_expo(M, targets(M, vt, chop=False)[0], basis, a, b) - tgt

    lo, hi = 1e-4, float(vt_b)
    flo, fhi = f(lo), f(hi)
    if not (flo < 0 <= fhi):
        raise RuntimeError(f"bisection bracket fails for {M['name']} {basis}: f(lo)={flo}, f(hi)={fhi}")
    it = 0
    while hi - lo > 1e-12 and it < 200:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        it += 1
    vt = 0.5 * (lo + hi)
    got = avg_expo(M, targets(M, vt, chop=False)[0], basis, a, b)
    return dict(vt=float(vt), target_expo=tgt, achieved_expo=got, abs_err=abs(got - tgt), iterations=it,
                expo_A0=fhi + tgt)


def solve_V(M, vt_b, basis, a, b):
    tgt = avg_expo(M, targets(M, vt_b)[0], basis, a, b)
    e1 = avg_expo(M, targets(M, None, vol=False)[0], basis, a, b)
    BISECT["exposure_only_evaluations"] += 2
    c = tgt / e1
    got = avg_expo(M, targets(M, None, vol=False, c=c)[0], basis, a, b)
    return dict(c=float(c), target_expo=tgt, achieved_expo=got, abs_err=abs(got - tgt), expo_V0=e1)


def solve_Ac(M, vt_b, basis, a, b):
    """INFO ONLY: no chop at the ORIGINAL vt, times a constant c (a pure, uniform de-levering dial)."""
    tgt = avg_expo(M, targets(M, vt_b)[0], basis, a, b)
    e0 = avg_expo(M, targets(M, vt_b, chop=False)[0], basis, a, b)
    BISECT["exposure_only_evaluations"] += 2
    c = tgt / e0
    got = avg_expo(M, targets(M, vt_b, chop=False, c=c)[0], basis, a, b)
    return dict(c=float(c), target_expo=tgt, achieved_expo=got, abs_err=abs(got - tgt), expo_A0=e0)


# ------------------------------------------------------------------ one complete comparison set
def variant_set(M, vt_b, basis, a, b, tag, windows, costs=("3bps",), with_activity=True, with_Ac=False):
    """B, A0, A(matched), V0, V(matched) [+ info-only Ac] for one market/tier/basis; specs, stats, raw series."""
    sA = solve_A(M, vt_b, basis, a, b)
    sV = solve_V(M, vt_b, basis, a, b)
    specs = {"B": dict(vt=vt_b, chop=True, vol=True, c=1.0),
             "A0": dict(vt=vt_b, chop=False, vol=True, c=1.0),
             "A": dict(vt=sA["vt"], chop=False, vol=True, c=1.0),
             "V0": dict(vt=None, chop=True, vol=False, c=1.0),
             "V": dict(vt=None, chop=True, vol=False, c=sV["c"])}
    sAc = None
    if with_Ac:
        sAc = solve_Ac(M, vt_b, basis, a, b)
        specs["Ac"] = dict(vt=vt_b, chop=False, vol=True, c=sAc["c"])
    res, ser = {}, {}
    for k, sp in specs.items():
        wl, wo = targets(M, **sp)
        R = run(M, wl, wo, basis)
        count(tag=tag, market=M["name"], basis=basis, variant=k, **{kk: (None if v is None else float(v)) for kk, v in sp.items()})
        out = dict(spec=sp, exposure=expo_profile(M, R, a, b))
        for c in costs:
            r = net(R, COSTS[c])
            out[c] = {w: st(r, *ab) for w, ab in windows.items()}
            if c == "3bps":
                ser[k] = r
        if with_activity:
            out["activity"] = activity(R, a, b)
        out["expo_by_window"] = {w: float(M["mult"] * R["hl"][mask(*ab)].mean()) for w, ab in windows.items() if w != "OOS"}
        res[k] = out
    return dict(solve_A=sA, solve_V=sV, solve_Ac_info_only=sAc, binding=binding_profile(M, vt_b, a, b),
                variants=res), ser


def dcal(res, x, w, cost="3bps", ref="B"):
    sb, sx = res["variants"][ref][cost][w], res["variants"][x][cost][w]
    return float(sb["calmar"] - sx["calmar"])


def decide(dh1, dh2, med):
    both = (dh1 >= RET_THR) and (dh2 >= RET_THR)
    if both and med >= MED_THR:
        return "RETAINED"
    if (not both) and med <= MED_THR:
        return "REPLACEABLE"
    return "INCONCLUSIVE (retain by default)"


def boot_dcal(r1, r2, a, b, n=1000, block=63, seed=7):
    """paired circular block bootstrap (63d blocks) of Calmar(r1) - Calmar(r2); information only."""
    m = mask(a, b)
    X = np.column_stack([r1[m], r2[m]])
    T = len(X)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        s0 = rng.integers(0, T, nb)
        ix = ((s0[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        e = np.cumprod(1 + X[ix], axis=0)
        cagr = e[-1] ** (252.0 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = cagr / np.abs(dd)
        out[i] = cal[0] - cal[1]
    return dict(mean=float(out.mean()), p05=float(np.percentile(out, 5)), p50=float(np.median(out)),
                p95=float(np.percentile(out, 95)), p_le_0=float((out <= 0).mean()),
                p_ge_0p03=float((out >= RET_THR).mean()), n=n, block=block)


OUT = dict(meta=dict(script="update_2026_11/r2_chop_vol_matched.py", run_date=str(pd.Timestamp.now().date()),
                     panel_end=str(END.date()), freeze=str(FREEZE.date()), in_sample=f"{START.date()}..{FREEZE.date()}",
                     oos=f"{OOS_A.date()}..{END.date()} (information only)",
                     costs="3 bps per unit one-way turnover (sum over sleeves |dw|) unless stated; 10/15 bps = UCITS-route sensitivity",
                     tranche="mean of the 5 weekday books' daily net returns (main_tranche_check.py); each weekday has at most one "
                             "executing book, so this equals netted-turnover accounting",
                     trade_threshold=f"a sleeve trades when |change in held weight| > {TRADE_EPS}",
                     cross_market_calendar="all markets on the panel (US/NDX) trading calendar; foreign closes forward-filled"))

# =================================================================== 0. validation of mechanics
log("== 0. validation")
val = {}
NDX = {"F2": make_market("NDX-F2", 2, P["qld"].values, P["above"].values, P["vol20"].values, P["er20"].values),
       "F3": make_market("NDX-F3", 3, P["tqqq"].values, P["above"].values, P["vol20"].values, P["er20"].values)}
VT = {"F2": 0.30, "F3": 0.20}
LEVCOL = {"F2": "qld", "F3": "tqqq"}
for t in ["F2", "F3"]:
    wl, wo = targets(NDX[t], VT[t])
    r_np = net(run(NDX[t], wl, wo, "Fri"), COSTS["3bps"])
    w_c = C.strategy_weights(P, lev=LEVCOL[t], voltgt=VT[t], gold="gold_phys", gold_w=0.15, off="ief")
    r_c = C.backtest(P, w_c).values
    r_tr_c = sum(C.backtest(P, w_c, mark=pd.Series(MARKS[d], index=IDX)).values for d in DAYS) / 5
    r_tr_np = net(run(NDX[t], wl, wo, "tranche"), COSTS["3bps"])
    val[f"{t}_friday_maxabs_diff_vs_common_backtest"] = float(np.abs(r_np - r_c).max())
    val[f"{t}_tranche_maxabs_diff_vs_common_backtest"] = float(np.abs(r_tr_np - r_tr_c).max())
    s_f, s_t = st(r_np, START, FREEZE), st(r_tr_np, START, FREEZE)
    val[f"{t}_friday_IS"] = dict(cagr=s_f["cagr"], maxdd=s_f["maxdd"], calmar=s_f["calmar"])
    val[f"{t}_tranche_IS"] = dict(cagr=s_t["cagr"], maxdd=s_t["maxdd"], calmar=s_t["calmar"])
ref = json.load(open(os.path.join(HERE, "main_tranche_check.json")))
for t in ["F2", "F3"]:
    for k, kk in [("Fri", "friday"), ("tranche", "tranche")]:
        rr = ref[f"{t}-IEF"][k]["to_freeze"]
        mine = val[f"{t}_{kk}_IS"]
        val[f"{t}_{kk}_vs_main_tranche_check_dcalmar"] = float(mine["calmar"] - rr["calmar"])
        val[f"{t}_{kk}_vs_main_tranche_check_dcagr"] = float(mine["cagr"] - rr["cagr"])
sig = signals_from_returns(P["ndx_ret"].values)
val["ndx_rebuilt_signals_match_panel"] = dict(
    above_mismatch_days=int((sig[0] != P["above"].values.astype(bool)).sum()),
    vol20_maxabs=float(np.nanmax(np.abs(sig[1] - P["vol20"].values))),
    er20_maxabs=float(np.nanmax(np.abs(sig[2] - P["er20"].values))))
OUT["validation"] = val
for k, v in val.items():
    log(f"   {k}: {v}")
assert max(val[f"{t}_friday_maxabs_diff_vs_common_backtest"] for t in ["F2", "F3"]) < 1e-12
assert max(val[f"{t}_tranche_maxabs_diff_vs_common_backtest"] for t in ["F2", "F3"]) < 1e-12

# =================================================================== 1. NDX tiers (decision basis = tranche)
log("\n== 1. NDX tiers")
SER = {}
OUT["ndx"] = {}
for t in ["F2", "F3"]:
    OUT["ndx"][t] = {}
    for basis in ["tranche", "Fri"]:
        res, ser = variant_set(NDX[t], VT[t], basis, START, FREEZE, tag=f"NDX-{t}", windows=WIN_NDX,
                               costs=("3bps", "10bps", "15bps"), with_Ac=True)
        SER[(t, basis)] = ser
        dd = {}
        for comp, x in [("chop", "A"), ("chop_unmatched_ref", "A0"), ("vol", "V"), ("vol_unmatched_ref", "V0"),
                        ("chop_const_info", "Ac")]:
            dd[comp] = {}
            for c in COSTS:
                dd[comp][c] = {w: dcal(res, x, w, cost=c) for w in WIN_NDX if w != "OOS"}
            v = res["variants"]
            dd[comp]["oos_total_return_diff"] = float(v["B"]["3bps"]["OOS"]["total"] - v[x]["3bps"]["OOS"]["total"])
        res["dcalmar_B_minus_alt"] = dd
        OUT["ndx"][t][basis] = res
        v = res["variants"]
        log(f"  {t} {basis:7s} vt_A={res['solve_A']['vt']:.4f} (expo {res['solve_A']['target_expo']:.4f}, err {res['solve_A']['abs_err']:.1e}) "
            f"c_V={res['solve_V']['c']:.4f} c_Ac(info)={res['solve_Ac_info_only']['c']:.4f}")
        for k in ["B", "A0", "A", "V0", "V", "Ac"]:
            s = v[k]["3bps"]["IS"]
            h1, h2 = v[k]["3bps"]["H1"], v[k]["3bps"]["H2"]
            ac = v[k]["activity"]
            log(f"     {k:3s} IS CAGR {s['cagr']*100:6.2f}% DD {s['maxdd']*100:6.1f}% Cal {s['calmar']:.3f} Sh {s['sharpe']:.2f} "
                f"Ulc {s['ulcer']:5.2f} CDaR {s['cdar95']*100:6.1f}% W5 {s['worst5_mean']*100:6.1f}% | H1 Cal {h1['calmar']:.3f} "
                f"H2 Cal {h2['calmar']:.3f} | expo {v[k]['exposure']['avg_all_days']:.3f} | turn {ac['turnover_per_yr']:5.2f} "
                f"reb/bk {ac['rebalances_per_yr_per_book']:4.1f} | OOS {v[k]['3bps']['OOS']['total']*100:+6.2f}%")
        log(f"     dCal B-A  H1 {dd['chop']['3bps']['H1']:+.3f} H2 {dd['chop']['3bps']['H2']:+.3f} IS {dd['chop']['3bps']['IS']:+.3f}"
            f" | B-V H1 {dd['vol']['3bps']['H1']:+.3f} H2 {dd['vol']['3bps']['H2']:+.3f} IS {dd['vol']['3bps']['IS']:+.3f}"
            f" | [info] B-Ac H1 {dd['chop_const_info']['3bps']['H1']:+.3f} H2 {dd['chop_const_info']['3bps']['H2']:+.3f}")

# cross-check against the round-1 critic ablation (A0 = 'chop half vs full', V0 = 'vol target'), info only
try:
    cr = json.load(open(os.path.join(HERE, "critic_out_critic_ablation_tranche.json")))
    xc = {}
    for t in ["F2", "F3"]:
        for comp, lab, x in [("chop", "chop half vs full", "A0"), ("vol", "vol target", "V0")]:
            for basis, cb in [("tranche", "tranche"), ("Fri", "Fri")]:
                for w in ["IS", "H1", "H2"]:
                    mine = dcal(OUT["ndx"][t][basis], x, w)
                    theirs = cr[t][lab][f"{cb}|{w}"]["d_calmar"]
                    xc[f"{t}|{comp}|{basis}|{w}"] = dict(mine=mine, critic=theirs, diff=mine - theirs)
    OUT["validation"]["critic_ablation_crosscheck_unmatched"] = xc
    log("   critic cross-check max |diff| =", max(abs(v["diff"]) for v in xc.values()))
except Exception as e:  # noqa: BLE001
    OUT["validation"]["critic_ablation_crosscheck_unmatched"] = f"not run: {e}"

# =================================================================== 2. cross-market (2x, vt .30, tranche)
log("\n== 2. cross-market")
YF = os.path.join(HERE, "r2_chop_vol_matched_yf.parquet")
yfd = pd.read_parquet(YF) if os.path.exists(YF) else pd.DataFrame(columns=["date", "ticker", "close"])
XM = {"spx": ("panel", "spx"), "rut": ("panel", "rut"), "tsx": ("panel", "tsx"), "hsi": ("panel", "hsi"),
      "dax": ("yfinance", "^GDAXI"), "n225": ("yfinance", "^N225"), "ixic": ("yfinance", "^IXIC")}
XREF = {"ndx_sim2x": ("panel", "ndx_ret")}      # reference only, excluded from the median
OUT["cross_market"] = {"markets": {}, "missing": [], "reference_excluded_from_median": {}}
XSER = {}


def market_returns(src, col):
    if src == "panel":
        r = P[col].values.astype(float)
        first = IDX[np.argmax((r != 0) & ~np.isnan(r))]
        return np.nan_to_num(r), first, dict(source="panel", column=col)
    s = yfd[yfd["ticker"] == col]
    if len(s) == 0:
        return None, None, None
    c = pd.Series(s["close"].values, index=pd.DatetimeIndex(s["date"])).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    first = c.index[0]
    lv = c.reindex(c.index.union(IDX)).ffill().reindex(IDX)
    r = lv.pct_change().values
    r[IDX <= first] = np.nan
    info = dict(source="yfinance", ticker=col, raw_first=str(first.date()), raw_last=str(c.index[-1].date()),
                raw_rows=int(len(c)), max_abs_daily_ret_on_panel_calendar=float(np.nanmax(np.abs(r))),
                zero_ret_days_after_1990=int(((np.nan_to_num(r) == 0) & (IDX >= "1990-01-01")).sum()))
    return np.nan_to_num(r), first, info


def eval_start(first):
    warm = pd.Timestamp(first) + pd.Timedelta(days=365)
    jan = pd.Timestamp(year=warm.year + (0 if (warm.month == 1 and warm.day == 1) else 1), month=1, day=1)
    return max(pd.Timestamp("1990-01-01"), jan)


def cross_market(name, src, col, mult=2, vt=0.30, tag="cross-market", with_Ac=True):
    r, first, info = market_returns(src, col)
    if r is None:
        return None
    above, vol20, er20 = signals_from_returns(r)
    r_lev = mult * r - 0.0095 / 252 - (mult - 1) * RF_D          # 2x: 2r - 0.95%/252 - rf_d (as specified)
    M = make_market(name, mult, r_lev, above, vol20, er20)
    a = eval_start(first)
    wins = {"FULL": (a, FREEZE), "H1": (a, H1_END), "H2": (H2_START, FREEZE), "OOS": (OOS_A, END)}
    res, ser = variant_set(M, vt, "tranche", a, FREEZE, tag=tag, windows=wins, costs=tuple(COSTS), with_Ac=with_Ac)
    res["info"] = dict(info, data_first=str(pd.Timestamp(first).date()), eval_start=str(a.date()), eval_end=str(FREEZE.date()))
    pairs = [("chop", "A"), ("chop_unmatched_ref", "A0"), ("vol", "V"), ("vol_unmatched_ref", "V0")]
    if with_Ac:
        pairs.append(("chop_const_info", "Ac"))
    res["dcalmar_B_minus_alt"] = {comp: {w: dcal(res, x, w) for w in ["FULL", "H1", "H2"]} for comp, x in pairs}
    res["dcalmar_B_minus_alt_FULL_by_cost"] = {c: {comp: dcal(res, x, "FULL", cost=c) for comp, x in pairs} for c in COSTS}
    res["oos_total_return"] = {k: res["variants"][k]["3bps"]["OOS"]["total"] for k in res["variants"]}
    v = res["variants"]
    bd = res["binding"]
    log(f"  {name:9s} start {a.date()} vt_A={res['solve_A']['vt']:.4f} c_V={res['solve_V']['c']:.4f} "
        f"| Cal B {v['B']['3bps']['FULL']['calmar']:.3f} A {v['A']['3bps']['FULL']['calmar']:.3f} "
        f"V {v['V']['3bps']['FULL']['calmar']:.3f} A0 {v['A0']['3bps']['FULL']['calmar']:.3f} V0 {v['V0']['3bps']['FULL']['calmar']:.3f} "
        f"| dChop {res['dcalmar_B_minus_alt']['chop']['FULL']:+.3f} dVol {res['dcalmar_B_minus_alt']['vol']['FULL']:+.3f} "
        f"| vt binds {bd['vol_target_binding_share_of_riskon_days']*100:4.1f}% chop {bd['chop_half_share_of_riskon_days']*100:4.1f}% of risk-on days")
    return res, ser, M


XMKT = {}
for name, (src, col) in XM.items():
    out = cross_market(name, src, col)
    if out is None:
        OUT["cross_market"]["missing"].append(dict(market=name, ticker=col, reason="no cached yfinance data: run "
                                                   "update_2026_11/r2_chop_vol_matched_fetch.py first"))
        log(f"  {name}: MISSING ({col}) - WARNING: the cross-market median below leaves this market out; run "
            f"update_2026_11/r2_chop_vol_matched_fetch.py first to reproduce the published result")
        continue
    OUT["cross_market"]["markets"][name], XSER[name], XMKT[name] = out
for name, (src, col) in XREF.items():
    out = cross_market(name, src, col)
    OUT["cross_market"]["reference_excluded_from_median"][name] = out[0]

# independent re-computation of two cross-market rows through common.py (substituted signal/sleeve columns)
xchk = {}
for name in [m for m in ["spx", "dax"] if m in XSER]:
    src, col = XM[name]
    r, first, _ = market_returns(src, col)
    ab, v20, e20 = signals_from_returns(r)
    P2 = P.copy()
    P2["above"], P2["vol20"], P2["er20"] = ab, v20, e20
    P2["lev_x"] = 2.0 * r - 0.0095 / 252 - RF_D
    rc = OUT["cross_market"]["markets"][name]
    kws = {"B": dict(voltgt=0.30), "A": dict(voltgt=rc["solve_A"]["vt"], chop_pos=1.0),
           "V": dict(voltgt=1e9, extra_scale=pd.Series(rc["solve_V"]["c"], index=IDX))}
    for k, kw in kws.items():
        w = C.strategy_weights(P2, lev="lev_x", gold="gold_phys", gold_w=0.15, off="ief", **kw)
        rr = sum(C.backtest(P2, w, mark=pd.Series(MARKS[d], index=IDX)).values for d in DAYS) / 5
        xchk[f"{name}|{k}"] = float(np.abs(rr - XSER[name][k]).max())
OUT["validation"]["cross_market_vs_common_py_maxabs_daily_diff"] = xchk
log("   cross-market common.py re-computation max |diff|:", {k: f"{v:.1e}" for k, v in xchk.items()})
assert max(xchk.values()) < 1e-12

mk = OUT["cross_market"]["markets"]
med = {}
for comp in ["chop", "vol", "chop_unmatched_ref", "vol_unmatched_ref", "chop_const_info"]:
    vals = {m: mk[m]["dcalmar_B_minus_alt"][comp]["FULL"] for m in mk}
    med[comp] = dict(median=float(np.median(list(vals.values()))) if vals else None, per_market=vals,
                     n_markets=len(vals), n_positive=int(sum(v > 0 for v in vals.values())),
                     n_ge_0p02=int(sum(v >= MED_THR for v in vals.values())),
                     median_H1=float(np.median([mk[m]["dcalmar_B_minus_alt"][comp]["H1"] for m in mk])) if vals else None,
                     median_H2=float(np.median([mk[m]["dcalmar_B_minus_alt"][comp]["H2"] for m in mk])) if vals else None)
OUT["cross_market"]["median_dcalmar"] = med
OUT["cross_market"]["median_dcalmar_FULL_by_cost"] = {
    c: {comp: float(np.median([mk[m]["dcalmar_B_minus_alt_FULL_by_cost"][c][comp] for m in mk])) for comp in ["chop", "vol"]}
    for c in COSTS}
loo = {}
for comp in ["chop", "vol"]:
    vals = med[comp]["per_market"]
    l1 = {m: float(np.median([v for k, v in vals.items() if k != m])) for m in vals}
    ref8 = list(vals.values()) + [OUT["cross_market"]["reference_excluded_from_median"]["ndx_sim2x"]["dcalmar_B_minus_alt"][comp]["FULL"]]
    loo[comp] = dict(leave_one_out_medians=l1, min=min(l1.values()), max=max(l1.values()),
                     median_incl_ndx_sim2x_8_markets=float(np.median(ref8)))
OUT["cross_market"]["median_sensitivity_info_only"] = loo
log(f"  median dCal chop {med['chop']['median']:+.4f} ({med['chop']['n_positive']}/{med['chop']['n_markets']} >0) | "
    f"vol {med['vol']['median']:+.4f} ({med['vol']['n_positive']}/{med['vol']['n_markets']} >0)")
log(f"  leave-one-out median range: chop [{loo['chop']['min']:+.4f}, {loo['chop']['max']:+.4f}] "
    f"vol [{loo['vol']['min']:+.4f}, {loo['vol']['max']:+.4f}] | incl. NDX-sim: chop {loo['chop']['median_incl_ndx_sim2x_8_markets']:+.4f} "
    f"vol {loo['vol']['median_incl_ndx_sim2x_8_markets']:+.4f}")
log(f"  medians by cost: {OUT['cross_market']['median_dcalmar_FULL_by_cost']}")

# =================================================================== 3. pre-registered decisions
log("\n== 3. decisions (tranche basis, 3 bps)")
DEC = {}
for t in ["F2", "F3"]:
    for comp in ["chop", "vol"]:
        d = OUT["ndx"][t]["tranche"]["dcalmar_B_minus_alt"][comp]["3bps"]
        m = med[comp]["median"]
        dec = decide(d["H1"], d["H2"], m)
        fri = OUT["ndx"][t]["Fri"]["dcalmar_B_minus_alt"][comp]["3bps"]
        by_cost = {}
        for c in ["10bps", "15bps"]:
            dc = OUT["ndx"][t]["tranche"]["dcalmar_B_minus_alt"][comp][c]
            mc = OUT["cross_market"]["median_dcalmar_FULL_by_cost"][c][comp]
            by_cost[c] = dict(dcalmar_H1=dc["H1"], dcalmar_H2=dc["H2"], cross_market_median=mc,
                              rule_outcome=decide(dc["H1"], dc["H2"], mc))
        DEC[f"{comp}|{t}"] = dict(component=("chop rule" if comp == "chop" else "vol target"), tier=t,
                                  dcalmar_H1=d["H1"], dcalmar_H2=d["H2"], dcalmar_IS=d["IS"],
                                  both_halves_ge_0p03=bool(d["H1"] >= RET_THR and d["H2"] >= RET_THR),
                                  cross_market_median=m, median_ge_0p02=bool(m >= MED_THR), median_le_0p02=bool(m <= MED_THR),
                                  margin_weaker_half_minus_0p03=float(min(d["H1"], d["H2"]) - RET_THR),
                                  margin_median_minus_0p02=float(m - MED_THR),
                                  decision=dec,
                                  info_friday_book=dict(dcalmar_H1=fri["H1"], dcalmar_H2=fri["H2"], dcalmar_IS=fri["IS"],
                                                        rule_if_applied_to_friday=decide(fri["H1"], fri["H2"], m)),
                                  info_rule_at_higher_costs=by_cost)
        if comp == "chop":
            dc = OUT["ndx"][t]["tranche"]["dcalmar_B_minus_alt"]["chop_const_info"]["3bps"]
            mc = med["chop_const_info"]["median"]
            DEC[f"{comp}|{t}"]["info_if_alternative_were_Ac_constant_dial"] = dict(
                dcalmar_H1=dc["H1"], dcalmar_H2=dc["H2"], cross_market_median=mc, rule_outcome=decide(dc["H1"], dc["H2"], mc))
        log(f"  {comp:5s} {t}: dCal H1 {d['H1']:+.4f} H2 {d['H2']:+.4f} | x-mkt median {m:+.4f} -> {dec}"
            f"   [info: Friday book -> {DEC[f'{comp}|{t}']['info_friday_book']['rule_if_applied_to_friday']}; "
            f"10bps -> {by_cost['10bps']['rule_outcome']}; 15bps -> {by_cost['15bps']['rule_outcome']}"
            + (f"; vs Ac -> {DEC[f'{comp}|{t}']['info_if_alternative_were_Ac_constant_dial']['rule_outcome']}" if comp == "chop" else "")
            + "]")
OUT["decisions"] = DEC

# =================================================================== 4. information-only robustness
log("\n== 4. information-only checks")
info = {}
# 4a. paired block bootstrap of the IS tranche Calmar difference
info["bootstrap_dcalmar_IS_tranche"] = {}
for t in ["F2", "F3"]:
    s = SER[(t, "tranche")]
    for comp, x in [("chop", "A"), ("vol", "V")]:
        b = boot_dcal(s["B"], s[x], START, FREEZE)
        info["bootstrap_dcalmar_IS_tranche"][f"{comp}|{t}"] = b
        log(f"  boot {comp} {t}: mean {b['mean']:+.3f} p05 {b['p05']:+.3f} p95 {b['p95']:+.3f} P(<=0) {b['p_le_0']:.2f}")
# 4b. re-match exposure WITHIN each half (removes any within-half exposure mismatch)
info["within_half_rematch_tranche"] = {}
for t in ["F2", "F3"]:
    for h, (a, b) in [("H1", (START, H1_END)), ("H2", (H2_START, FREEZE))]:
        res, _ = variant_set(NDX[t], VT[t], "tranche", a, b, tag=f"NDX-{t}-rematch-{h}", windows={h: (a, b)},
                             with_activity=False)
        dch, dvo = dcal(res, "A", h), dcal(res, "V", h)
        info["within_half_rematch_tranche"][f"{t}|{h}"] = dict(vt_A=res["solve_A"]["vt"], c_V=res["solve_V"]["c"],
                                                                 dcalmar_chop=dch, dcalmar_vol=dvo,
                                                                 calmar=dict((k, res["variants"][k]["3bps"][h]["calmar"]) for k in res["variants"]))
        log(f"  rematch {t} {h}: vt_A {res['solve_A']['vt']:.4f} dChop {dch:+.3f} | c_V {res['solve_V']['c']:.4f} dVol {dvo:+.3f}")
# 4c. per-weekday spread of the chop / vol differences (shows whether the tranche result is a 5-book average of
#     consistent or of offsetting effects)
info["per_weekday_dcalmar_IS"] = {}
for t in ["F2", "F3"]:
    vtA = OUT["ndx"][t]["tranche"]["solve_A"]["vt"]
    cV = OUT["ndx"][t]["tranche"]["solve_V"]["c"]
    row = {}
    WS = {"B": targets(NDX[t], VT[t]), "A": targets(NDX[t], vtA, chop=False), "V": targets(NDX[t], None, vol=False, c=cV)}
    for d in DAYS:
        out = {}
        for nm, (wl, wo) in WS.items():
            hl, ho, hg = held(wl, d), held(wo, d), held(GOLD_T, d)
            g = hl * NDX[t]["r_lev"] + ho * R_IEF + hg * R_GOLD
            tn = np.abs(np.diff(hl, prepend=0.0)) + np.abs(np.diff(ho, prepend=0.0)) + np.abs(np.diff(hg, prepend=0.0))
            out[nm] = st(g - tn * COSTS["3bps"], START, FREEZE)
        row[d] = dict(calmar_B=out["B"]["calmar"], dchop=out["B"]["calmar"] - out["A"]["calmar"],
                      dvol=out["B"]["calmar"] - out["V"]["calmar"])
    info["per_weekday_dcalmar_IS"][t] = row
    log(f"  per-weekday {t}: " + " ".join(f"{d} dChop {row[d]['dchop']:+.3f}/dVol {row[d]['dvol']:+.3f}" for d in DAYS))
# 4d. where the differences come from: max-drawdown episodes and component binding rates (NDX, tranche)
info["ndx_episodes_and_binding"] = {}
for t in ["F2", "F3"]:
    rt = OUT["ndx"][t]["tranche"]
    ep = {k: {w: dict(maxdd=rt["variants"][k]["3bps"][w]["maxdd"], peak=rt["variants"][k]["3bps"][w]["maxdd_peak"],
                      trough=rt["variants"][k]["3bps"][w]["maxdd_trough"], cagr=rt["variants"][k]["3bps"][w]["cagr"])
              for w in ["IS", "H1", "H2"]} for k in ["B", "A", "V"]}
    info["ndx_episodes_and_binding"][t] = dict(maxdd_episodes=ep, binding_IS=rt["binding"],
                                               binding_H1=binding_profile(NDX[t], VT[t], START, H1_END),
                                               binding_H2=binding_profile(NDX[t], VT[t], H2_START, FREEZE))
    for k in ["B", "A", "V"]:
        log(f"  {t} {k}: " + " | ".join(f"{w} CAGR {ep[k][w]['cagr']*100:5.2f}% DD {ep[k][w]['maxdd']*100:5.1f}% "
                                         f"{ep[k][w]['peak']}->{ep[k][w]['trough']}" for w in ["H1", "H2"]))
    b = info["ndx_episodes_and_binding"][t]
    log(f"  {t} binding (share of risk-on days): vol target H1 {b['binding_H1']['vol_target_binding_share_of_riskon_days']*100:.1f}% "
        f"H2 {b['binding_H2']['vol_target_binding_share_of_riskon_days']*100:.1f}% | chop-half H1 "
        f"{b['binding_H1']['chop_half_share_of_riskon_days']*100:.1f}% H2 {b['binding_H2']['chop_half_share_of_riskon_days']*100:.1f}%")
# 4e. NOT PRE-REGISTERED, information only: cross-market at the F3 construction (3x, vt .20), where the vol
#     target binds far more often than at 2x/vt .30 - addresses the low power of the registered 2x leg for vol.
info["cross_market_3x_vt020_info_only"] = {"markets": {}}
x3 = {}
for name, (src, col) in XM.items():
    if name not in XSER:
        continue
    res3, _, _ = cross_market(name + "_3x", src, col, mult=3, vt=0.20, tag="cross-market-3x-info", with_Ac=False)
    info["cross_market_3x_vt020_info_only"]["markets"][name] = dict(
        solve_A=res3["solve_A"], solve_V=res3["solve_V"], binding=res3["binding"],
        dcalmar_B_minus_alt=res3["dcalmar_B_minus_alt"],
        calmar_FULL={k: res3["variants"][k]["3bps"]["FULL"]["calmar"] for k in res3["variants"]})
    x3[name] = res3["dcalmar_B_minus_alt"]
info["cross_market_3x_vt020_info_only"]["median_dcalmar_FULL"] = {
    comp: float(np.median([x3[m][comp]["FULL"] for m in x3])) for comp in ["chop", "vol"]}
info["cross_market_3x_vt020_info_only"]["n_positive"] = {
    comp: int(sum(x3[m][comp]["FULL"] > 0 for m in x3)) for comp in ["chop", "vol"]}
log(f"  [info only] 3x vt .20 cross-market median dCal: {info['cross_market_3x_vt020_info_only']['median_dcalmar_FULL']} "
    f"positive {info['cross_market_3x_vt020_info_only']['n_positive']} of {len(x3)}")
OUT["information_only"] = info

# =================================================================== 5. counts + save
OUT["variant_count"] = dict(
    total_return_evaluated_variants=len(VARIANTS),
    by_tag={t: sum(1 for v in VARIANTS if v["tag"] == t) for t in sorted(set(v["tag"] for v in VARIANTS))},
    note="each variant = one weight rule on one basis for one market (cost sensitivities and windows of the same "
         "variant are not counted again); bisection/linear solves touch exposure only, never returns. Tags "
         "'NDX-*-rematch-*' and 'cross-market-3x-info' are information-only and never enter a decision; "
         "'cross-market' includes the ndx_sim2x reference row (excluded from the median)",
    exposure_only_solver_evaluations=BISECT["exposure_only_evaluations"],
    per_weekday_info_runs=2 * 5 * 3)
OUT["variants_list"] = VARIANTS


def _brief(t, basis):
    r = OUT["ndx"][t][basis]
    v = r["variants"]
    return dict(vt_A=r["solve_A"]["vt"], c_V=r["solve_V"]["c"], c_Ac_info=r["solve_Ac_info_only"]["c"],
                avg_ndx_exposure=r["solve_A"]["target_expo"],
                IS_3bps={k: dict(cagr=v[k]["3bps"]["IS"]["cagr"], maxdd=v[k]["3bps"]["IS"]["maxdd"],
                                 calmar=v[k]["3bps"]["IS"]["calmar"], ulcer=v[k]["3bps"]["IS"]["ulcer"],
                                 cdar95=v[k]["3bps"]["IS"]["cdar95"], worst5_mean=v[k]["3bps"]["IS"]["worst5_mean"],
                                 calmar_H1=v[k]["3bps"]["H1"]["calmar"], calmar_H2=v[k]["3bps"]["H2"]["calmar"],
                                 calmar_10bps=v[k]["10bps"]["IS"]["calmar"], calmar_15bps=v[k]["15bps"]["IS"]["calmar"],
                                 turnover_per_yr=v[k]["activity"]["turnover_per_yr"],
                                 rebalances_per_yr_per_book=v[k]["activity"]["rebalances_per_yr_per_book"],
                                 oos_total=v[k]["3bps"]["OOS"]["total"])
                         for k in ["B", "A", "A0", "V", "V0", "Ac"]},
                dcalmar_B_minus_alt_3bps={c: r["dcalmar_B_minus_alt"][c]["3bps"] for c in ["chop", "vol", "chop_const_info"]})


OUT["summary"] = dict(
    decisions={k: v["decision"] for k, v in DEC.items()},
    rule="RETAINED if tranche dCalmar >= +0.03 in both halves AND cross-market median >= +0.02; REPLACEABLE if "
         "< +0.03 in at least one half AND median <= +0.02; else INCONCLUSIVE (retain by default)",
    cross_market_median_dcalmar={c: med[c]["median"] for c in ["chop", "vol", "chop_const_info"]},
    ndx={f"{t}|{b}": _brief(t, b) for t in ["F2", "F3"] for b in ["tranche", "Fri"]})


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(HERE, "r2_chop_vol_matched.json"), "w"), indent=1)
log(f"\nsaved update_2026_11/r2_chop_vol_matched.json | variants {len(VARIANTS)} | runtime {time.time()-T0:.1f}s")
