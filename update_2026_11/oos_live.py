"""
oos_live.py -- TRUE OUT-OF-SAMPLE CHECK WITH REAL ETF RETURNS  (Nov-2026 refresh, KEY = oos_live)

Window    : first trading day after the July freeze (2026-07-24) .. last panel date.
Decisions : weights are decided ONLY from data available at each Friday close. The indicators
            (EMA-100, 20d realised vol, 20d Kaufman efficiency ratio) are causal and warmed up on the
            1985+ NDX history; re-verified here by recomputing them on data truncated at every decision
            date (full history and a 5-year window, as the live scanner does).
Sleeves   : REAL ETF daily total returns: TQQQ, QLD, QQQ, TLT, IEF, SHY, BIL, GLD (panel *_real columns,
            independently re-checked against a fresh yfinance pull made by oos_live_fetch.py) + SPY.
Mechanics : (1) update_2026_11/common.py  -- constant weights between Friday marks, 3 bps x turnover,
                lag 1 (trade at the decision close) / lag 2 (trade at the next session's close);
            (2) a share-level DRIFT simulation -- holdings drift between weekly rebalances, every Friday
                decision resets to target; executed at the Friday close, Monday close or Monday OPEN.
Nothing here chooses a parameter: every configuration is the pre-registered July specification.
Output    : update_2026_11/oos_live.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OUTJ = os.path.join(HERE, "oos_live.json")
YFP = os.path.join(HERE, "oos_live_yf.parquet")
SCANNER_STATE = os.path.join(ROOT, "scanner_state.json")       # read-only cross-check

P = C.panel().copy()                 # never mutate the cached shared panel
IDX = P.index
FREEZE = C.FREEZE
OOS = IDX[IDX > FREEZE]
A0, END = OOS[0], OOS[-1]
N = len(OOS)
HIST_A = IDX[IDX >= C.START][0]
VARIANTS: list[str] = []


def reg(name):
    if name not in VARIANTS:
        VARIANTS.append(name)


def nxt(t):
    j = IDX.get_loc(t)
    return IDX[j + 1] if j + 1 < len(IDX) else None


def prv(t):
    return IDX[IDX.get_loc(t) - 1]


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


print(f"OOS window {A0.date()} .. {END.date()}  ({N} trading days)   history {HIST_A.date()} .. {FREEZE.date()}")

# ============================================================== 0. data: real ETFs, sims, fetched
YC = YO = None
if os.path.exists(YFP):
    _y = pd.read_parquet(YFP)
    YC = _y.pivot(index="date", columns="ticker", values="close")
    YO = _y.pivot(index="date", columns="ticker", values="open")

REAL = {"qqq": "qqq_real", "qld": "qld_real", "tqqq": "tqqq_real", "tlt": "tlt_real",
        "ief": "ief_real", "shy": "shy_real", "bills": "bil_real", "gold_phys": "gld_real"}
TICK = {"qqq": "QQQ", "qld": "QLD", "tqqq": "TQQQ", "tlt": "TLT", "ief": "IEF", "shy": "SHY",
        "bills": "BIL", "gold_phys": "GLD", "spy": "SPY"}


def yret(t):
    """close-to-close return on the NDX calendar from the fetched (dividend-adjusted) closes."""
    c = YC[t].dropna()
    r = c.reindex(IDX).ffill().pct_change(fill_method=None)
    r[IDX <= c.index.min()] = np.nan
    return r


ON_FILLS = {}


def yon(t):
    """overnight return  open_t / close_{t-1} - 1  (dividend-adjusted OHLC). Missing opens after
    inception (none expected) are set to 0 (= open at previous close) and counted in ON_FILLS."""
    c = YC[t].dropna()
    o = YO[t].reindex(c.index)
    on = (o / c.shift(1) - 1).reindex(IDX)
    live = IDX > c.index.min()
    ON_FILLS[t] = int((on.isna() & live).sum())
    on[on.isna() & live] = 0.0
    return on


PR = P.copy()                                   # REAL-ETF panel
for k, c in REAL.items():
    PR[k] = P[c]
if YC is not None:
    PR["spy"] = yret("SPY")
    P["spy"] = PR["spy"]                        # only ever used inside the OOS window

chk = IDX[(IDX >= pd.Timestamp("2026-01-02")) & (IDX <= END)]
nan_real = {k: int(PR.loc[chk, k].isna().sum()) for k in list(REAL) + (["spy"] if "spy" in PR else [])}
assert all(v == 0 for v in nan_real.values()), nan_real
spliced_vs_real = {k: float((P.loc[OOS, k] - P.loc[OOS, c]).abs().max()) for k, c in REAL.items()}

indep = {}
if YC is not None:
    for k, t in TICK.items():
        if k == "spy":
            continue
        d = (yret(t).loc[OOS] - PR.loc[OOS, k]).abs()
        indep[t] = dict(max_abs_daily_diff=float(d.max()),
                        total_yf=float((1 + yret(t).loc[OOS]).prod() - 1),
                        total_panel=float((1 + PR.loc[OOS, k]).prod() - 1))
    indep["NDX_level_max_abs_diff"] = float((YC["^NDX"].reindex(OOS) - P.loc[OOS, "ndx"]).abs().max())
    indep["SPY_total_OOS"] = float((1 + PR.loc[OOS, "spy"]).prod() - 1)
    indep["SPX_price_index_total_OOS"] = float((1 + P.loc[OOS, "spx"]).prod() - 1)

PS = P.copy()                                   # SIMULATED-sleeve panel (what the backtest uses pre-inception)
for k in ["qqq", "qld", "tqqq"]:
    PS[k] = P[f"{k}_sim"]
for k in ["tlt", "ief", "shy"]:
    PS[k] = P[f"{k}_syn"]
PS["bills"] = P["rf_d"] - 0.0013 / 252          # panel's pre-BIL formula
GOLD_SIM = "GLD itself (no synthetic gold available)"
if YC is not None and "GC=F" in YC.columns:
    PS["gold_phys"] = yret("GC=F") - 0.0040 / 252   # panel's pre-GLD splice source: GC=F - 0.40%/yr
    GOLD_SIM = "GC=F front-month futures - 0.40%/yr (the panel's 2000-2004 splice source)"
    assert PS.loc[OOS, "gold_phys"].notna().all()

# ============================================================== 1. no-look-ahead verification
def recompute(nd):
    ema = nd.ewm(span=100, adjust=False).mean().iloc[-1]
    vol = nd.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    er = abs(nd.diff(20).iloc[-1]) / nd.diff().abs().rolling(20).sum().iloc[-1]
    return ema, vol, er


DEC = IDX[P["mark"].values & (IDX >= FREEZE) & (IDX <= END)]       # Friday decisions in/at the window
lookahead = dict(max_abs_diff_ema_full=0.0, max_abs_diff_vol_full=0.0, max_abs_diff_er_full=0.0,
                 max_abs_diff_ema_5y=0.0, n_decisions=len(DEC))
for t in list(DEC) + [END]:
    nd = P.loc[:t, "ndx"]
    for tag, s in [("full", nd), ("5y", nd[nd.index > t - pd.DateOffset(years=5)])]:
        e, v, r_ = recompute(s)
        lookahead[f"max_abs_diff_ema_{tag}"] = max(lookahead[f"max_abs_diff_ema_{tag}"], abs(e - P.at[t, "ema100"]))
        if tag == "full":
            lookahead["max_abs_diff_vol_full"] = max(lookahead["max_abs_diff_vol_full"], abs(v - P.at[t, "vol20"]))
            lookahead["max_abs_diff_er_full"] = max(lookahead["max_abs_diff_er_full"], abs(r_ - P.at[t, "er20"]))
print("no-look-ahead check:", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in lookahead.items()})

# ============================================================== 2. strategies and benchmarks
inv = (P["curve"] < 0).fillna(False)
inv_recent = inv.rolling(252, min_periods=1).max().astype(bool)
scanner_ron = P["above"] & ~(inv_recent & ~P["above"])        # scanner.py logic, literally
curve_rule_noop = bool((scanner_ron == P["above"]).all())

STRATS = {
    "a_F_scanner_TQQQ_TLT_GLD": dict(lev="tqqq", voltgt=0.20, off="tlt", gold="gold_phys", gold_w=0.15,
                                     risk_on=scanner_ron),
    "b_F3_TQQQ_IEF_GLD": dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15),
    "c_F2_QLD_IEF_GLD_vt30": dict(lev="qld", voltgt=0.30, off="ief", gold="gold_phys", gold_w=0.15),
    "d_A_TQQQ_BIL_vt25": dict(lev="tqqq", voltgt=0.25, chop_pos=1.0, seasonal=False, gold_w=0.0, off="bills"),
    "ref_Fjuly_TQQQ_bond10syn_XAUminers": dict(lev="tqqq", voltgt=0.20, off="bond10_old", gold="gold_miners",
                                               gold_w=0.15),
}
LEV = {k: v["lev"] for k, v in STRATS.items()}
BENCH = {
    "QQQ_buyhold": {"qqq": 1.0}, "QLD_buyhold": {"qld": 1.0}, "TQQQ_buyhold": {"tqqq": 1.0},
    "6040_SPXprice_IEF": {"spx": 0.6, "ief": 0.4},
    "6040_SPY_IEF": {"spy": 0.6, "ief": 0.4},
    "static_IEF85_GLD15": {"ief": 0.85, "gold_phys": 0.15},
}
if "spy" not in PR:
    BENCH.pop("6040_SPY_IEF")

W = {k: C.strategy_weights(P, **kw) for k, kw in STRATS.items()}
for k, d in BENCH.items():
    W[k] = pd.DataFrame({c: float(v) for c, v in d.items()}, index=IDX)
assert np.allclose(W["a_F_scanner_TQQQ_TLT_GLD"]["tqqq"], C.strategy_weights(P, lev="tqqq", off="tlt")["tqqq"])

# ============================================================== 3. drift (share-level) simulator
def drift_sim(R, Wt, marks, mode, cost=C.COST, t0=pd.Timestamp("2026-01-02"), ON=None):
    """Holdings drift between trades; each Friday decision fully resets holdings to target.
    mode: 'close1' = trade at the decision (Friday) close   (== common lag 1)
          'close2' = trade at the next session's close       (== common lag 2)
          'open1'  = trade at the next session's OPEN        (needs overnight returns ON)
    cost = cost x sum|trade| / V, charged in the return of the first day the new holdings are held."""
    cols = list(Wt.columns)
    d = R.index[R.index >= t0]
    m = marks.reindex(d).fillna(False).values.astype(bool)
    Rv = R.loc[d, cols].values.astype(float)
    Wv = Wt.loc[d, cols].values.astype(float)
    ONv = ON.loc[d, cols].values.astype(float) if ON is not None else None
    assert not np.isnan(Rv).any()
    first = int(np.argmax(m))
    h = Wv[first].copy()                       # start at target at the first decision close (V = 1)
    out = pd.Series(np.nan, index=d)
    tot_cost = 0.0
    for i in range(first + 1, len(d)):
        V0 = h.sum()
        if mode in ("close1", "close2"):
            tgt = Wv[i - 1] if (mode == "close1" and m[i - 1]) else (
                Wv[i - 2] if (mode == "close2" and i >= 2 and m[i - 2]) else None)
            if tgt is not None:
                cf = cost * np.abs(tgt - h / V0).sum()
                tot_cost += cf * V0
                h = V0 * (1 - cf) * tgt
            h = h * (1 + Rv[i])
        else:
            on = ONv[i]
            assert not np.isnan(on).any()
            h = h * (1 + on)
            if m[i - 1]:
                Vo = h.sum()
                cf = cost * np.abs(Wv[i - 1] - h / Vo).sum()
                tot_cost += cf * Vo
                h = Vo * (1 - cf) * Wv[i - 1]
            h = h * (1 + Rv[i]) / (1 + on)
        out.iloc[i] = h.sum() / V0 - 1
    return out


ON = {}
if YO is not None:
    for k, t in TICK.items():
        ON[k] = yon(t)
ONF = pd.DataFrame(ON) if ON else None

# ============================================================== 4. run everything
RUN = {}
for k, w in W.items():
    RUN[(k, "model_spliced_lag1")] = C.backtest(P, w, lag=1)
    RUN[(k, "model_spliced_lag2")] = C.backtest(P, w, lag=2)
    RUN[(k, "real_lag1")] = C.backtest(PR, w, lag=1)
    RUN[(k, "real_lag2")] = C.backtest(PR, w, lag=2)
    RUN[(k, "sim_sleeves_lag1")] = C.backtest(PS, w, lag=1)
    cols = list(w.columns)
    R = PR[cols]
    for mode in ["close1", "close2", "open1"]:
        if mode == "open1" and (ONF is None or any(c not in ONF.columns for c in cols)):
            continue
        RUN[(k, f"real_drift_{mode}")] = drift_sim(R, w, P["mark"], mode,
                                                   ON=ONF[cols] if mode == "open1" else None)
    for (kk, mm) in list(RUN):
        if kk == k:
            reg(f"{kk} | {mm}")

# real == spliced inside the window (sanity)
real_eq_spliced = {k: float((RUN[(k, "real_lag1")].loc[OOS] - RUN[(k, "model_spliced_lag1")].loc[OOS]).abs().max())
                   for k in W}
assert max(real_eq_spliced.values()) < 1e-12, real_eq_spliced


def oos_stats(r):
    s = r.loc[A0:END]
    assert s.notna().all()
    e = (1 + s).cumprod()
    e0 = pd.concat([pd.Series([1.0], index=[FREEZE]), e])
    dd = e0 / e0.cummax() - 1
    return dict(total=float(e.iloc[-1] - 1), maxdd=float(dd.min()), maxdd_date=str(dd.idxmin().date()),
                peak_gain=float(e0.max() - 1), peak_date=str(e0.idxmax().date()),
                vol_ann=float(s.std() * np.sqrt(252)), worst_day=float(s.min()),
                worst_day_date=str(s.idxmin().date()), days=int(len(s)))


def is_stats(r):
    s = C.stats(r, b=FREEZE)
    return {k: (float(v) if k != "days" else int(v)) for k, v in s.items()}


OOSR = {f"{k}|{m}": oos_stats(r) for (k, m), r in RUN.items()}
ISR = {k: {"lag1": is_stats(RUN[(k, "model_spliced_lag1")]), "lag2": is_stats(RUN[(k, "model_spliced_lag2")])}
       for k in W if k != "6040_SPY_IEF"}

# ============================================================== 4b. execution timing IN-SAMPLE (context for the OOS lag result)
CORE4 = ["a_F_scanner_TQQQ_TLT_GLD", "b_F3_TQQQ_IEF_GLD", "c_F2_QLD_IEF_GLD_vt30", "d_A_TQQQ_BIL_vt25"]
SUB = {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-2009": ("2000-01-01", "2009-12-31"),
       "2010-freeze": ("2010-01-01", str(FREEZE.date()))}
LAGCOST = {}
for k in CORE4 + ["ref_Fjuly_TQQQ_bond10syn_XAUminers"]:
    r1, r2 = RUN[(k, "model_spliced_lag1")], RUN[(k, "model_spliced_lag2")]
    LAGCOST[k] = {}
    for nm, (a_, b_) in SUB.items():
        s1, s2 = C.stats(r1, a_, b_, start=None), C.stats(r2, a_, b_, start=None)
        LAGCOST[k][nm] = dict(cagr_lag1=s1["cagr"], cagr_lag2=s2["cagr"], cagr_cost=s1["cagr"] - s2["cagr"],
                              calmar_lag1=s1["calmar"], calmar_lag2=s2["calmar"])
NDX_AUTOCORR = {nm: float(P["ndx_ret"].loc[a_:b_].autocorr(1)) for nm, (a_, b_) in SUB.items()}

EXEC_IS = {}
if ONF is not None:
    need = ["tqqq_real", "qld_real", "ief_real", "tlt_real", "gld_real", "bil_real"]
    first_ok = max([P[c].first_valid_index() for c in need] +
                   [ONF[c].first_valid_index() for c in ["tqqq", "qld", "ief", "tlt", "gold_phys", "bills"]])
    T0_REAL = IDX[(IDX > first_ok) & P["mark"].values][0]
    for k in CORE4:
        cols = list(W[k].columns)
        EXEC_IS[k] = {}
        for mode in ["close1", "close2", "open1"]:
            r = drift_sim(PR[cols], W[k], P["mark"], mode, t0=T0_REAL, ON=ONF[cols] if mode == "open1" else None)
            reg(f"{k} | IS real-ETF era drift_{mode}")
            s = C.stats(r, b=FREEZE, start=None)
            s["oos_total_same_run"] = float((1 + r.loc[OOS]).prod() - 1)
            EXEC_IS[k][mode] = s
        for lag in [1, 2]:
            s = C.stats(RUN[(k, f"real_lag{lag}")], a=nxt(T0_REAL), b=FREEZE, start=None)
            EXEC_IS[k][f"model_constmix_lag{lag}"] = s
    EXEC_IS["_span"] = [str(nxt(T0_REAL).date()), str(FREEZE.date())]

# market context (not variants): each instrument over the window
mkt = {}
for nm, col, frame in [("NDX index", "ndx_ret", P), ("QQQ", "qqq", PR), ("QLD", "qld", PR), ("TQQQ", "tqqq", PR),
                       ("TLT", "tlt", PR), ("IEF", "ief", PR), ("SHY", "shy", PR), ("BIL", "bills", PR),
                       ("GLD", "gold_phys", PR), ("SPY", "spy", PR), ("S&P500 price idx", "spx", P),
                       ("XAU gold miners idx", "gold_miners", P), ("July synthetic 10y", "bond10_old", P)]:
    if col in frame:
        mkt[nm] = oos_stats(frame[col].fillna(0))
ndx_o = P.loc[OOS, "ndx"]
mkt["NDX_levels"] = dict(freeze_close=float(P.at[FREEZE, "ndx"]), record_close=float(ndx_o.max()),
                         record_date=str(ndx_o.idxmax().date()), last=float(ndx_o.iloc[-1]),
                         last_vs_record=float(ndx_o.iloc[-1] / ndx_o.max() - 1),
                         min_close=float(ndx_o.min()), min_date=str(ndx_o.idxmin().date()),
                         ema100_last=float(P.at[END, "ema100"]), vol20_last=float(P.at[END, "vol20"]),
                         er20_last=float(P.at[END, "er20"]), tnx_last=float(P.at[END, "tnx"]),
                         vix_last=float(P.at[END, "vix"]), dxy_last=float(P.at[END, "dxy"]))

# ============================================================== 5. tracking gap: real vs simulated sleeves
def te_row(real, sim, prior_te=None):
    a = pd.concat([real.loc[OOS], sim.loc[OOS]], axis=1).dropna()
    a.columns = ["r", "s"]
    out = dict(total_real=float((1 + a.r).prod() - 1), total_sim=float((1 + a.s).prod() - 1),
               corr=float(a.r.corr(a.s)), te_ann=float((a.r - a.s).std() * np.sqrt(252)))
    out["gap_pp"] = out["total_real"] - out["total_sim"]
    if prior_te is not None:
        out["expected_1sigma_gap_from_full_history_TE"] = float(prior_te * np.sqrt(len(a) / 252))
    return out


VAL = json.load(open(os.path.join(HERE, "panel_validation.json")))
sleeve_track = {
    "QQQ vs qqq_sim (NDX - 0.20%)": te_row(P.qqq_real, P.qqq_sim, VAL["qqq sim vs real"]["te_ann"]),
    "QLD vs qld_sim (2x NDX - fee - rf)": te_row(P.qld_real, P.qld_sim, VAL["qld sim vs real"]["te_ann"]),
    "TQQQ vs tqqq_sim (3x NDX - fee - 2rf)": te_row(P.tqqq_real, P.tqqq_sim, VAL["tqqq sim vs real"]["te_ann"]),
    "TLT vs tlt_syn (dur 15.8)": te_row(P.tlt_real, P.tlt_syn, VAL["tlt syn vs real"]["te_ann"]),
    "IEF vs ief_syn (dur 7.1)": te_row(P.ief_real, P.ief_syn, VAL["ief syn vs real"]["te_ann"]),
    "SHY vs shy_syn (dur 1.8)": te_row(P.shy_real, P.shy_syn, VAL["shy syn vs real"]["te_ann"]),
    "BIL vs T-bill - 0.13%": te_row(P.bil_real, P.rf_d - 0.0013 / 252),
    "IEF vs July bond10_old": te_row(P.ief_real, P.bond10_old, VAL["July bond10 vs IEF"]["te_ann"]),
    "TLT vs July bond10_old": te_row(P.tlt_real, P.bond10_old, VAL["July bond10 vs TLT"]["te_ann"]),
    "GLD vs XAU gold miners (July proxy)": te_row(P.gld_real, P.gold_miners),
}
if "GC=F" in GOLD_SIM:
    sleeve_track["GLD vs GC=F - 0.40%"] = te_row(P.gld_real, PS.gold_phys, VAL["GC=F vs GLD"]["te_ann"])

strat_track = {}
for k in W:
    a, s = RUN[(k, "real_lag1")], RUN[(k, "sim_sleeves_lag1")]
    strat_track[k] = te_row(a, s)

# ============================================================== 6. phase attribution (risk-on until exit vs risk-off)
NDX_PEAK = ndx_o.idxmax()


def exit_day(Pp, w, lev, lag):
    held = C.held_weights(Pp, w, lag=lag).loc[OOS, lev]
    on = held > 1e-12
    if not on.iloc[0]:
        return None
    off = held.index[~on.values]
    return off[0] if len(off) else None


EXIT_B1 = exit_day(PR, W["b_F3_TQQQ_IEF_GLD"], "tqqq", 1)


def attribution(Pp, w, r, lag, ex):
    held = C.held_weights(Pp, w, lag=lag)
    turn = held.diff().abs().sum(axis=1).fillna(0)
    s = r.loc[A0:END]
    V = (1 + s).cumprod()
    Vp = V.shift(1).fillna(1.0)
    pnl = pd.DataFrame({c: held[c].loc[A0:END] * Pp[c].fillna(0).loc[A0:END] * Vp for c in held.columns})
    pnl["cost"] = -(turn * C.COST).loc[A0:END] * Vp
    resid = float((pnl.sum(axis=1) - (V - Vp)).abs().max())
    segs = [("1a risk-on: freeze -> NDX record close", A0, NDX_PEAK)]
    if ex is not None:
        segs += [("1b risk-on: record -> exit", nxt(NDX_PEAK), prv(ex)), ("2 risk-off: exit -> last date", ex, END)]
        segs_main = [("risk-on phase (until exit)", A0, prv(ex)), ("risk-off phase", ex, END)]
    else:
        segs += [("1b after record", nxt(NDX_PEAK), END)]
        segs_main = [("whole window (no exit)", A0, END)]
    out = {}
    for nm, a, b in segs_main + segs:
        Va = 1.0 if a == A0 else float(V.loc[prv(a)])
        seg = pnl.loc[a:b]
        out[nm] = dict(start=str(a.date()), end=str(b.date()), days=int(len(seg)),
                       compounded_return=float(V.loc[b] / Va - 1),
                       pnl_pct_of_start_capital={c: float(seg[c].sum()) for c in seg.columns},
                       avg_weights={c: float(held[c].loc[a:b].mean()) for c in held.columns})
    out["_check_max_abs_residual"] = resid
    out["_total"] = float(V.iloc[-1] - 1)
    return out


ATTR = {}
for k in W:
    lev = LEV.get(k)
    for lag in [1, 2]:
        r = RUN[(k, f"real_lag{lag}")]
        ex = exit_day(PR, W[k], lev, lag) if lev else EXIT_B1
        ATTR[f"{k}|lag{lag}"] = attribution(PR, W[k], r, lag, ex)
        ATTR[f"{k}|lag{lag}"]["_exit_day"] = str(ex.date()) if ex is not None else None

# counterfactual: same engine, EMA gate removed (always risk-on) -> value of the exit (diagnostic only)
always = pd.Series(True, index=IDX)
CF = {}
for k in ["b_F3_TQQQ_IEF_GLD", "c_F2_QLD_IEF_GLD_vt30"]:
    kw = {**STRATS[k], "risk_on": always}
    wcf = C.strategy_weights(P, **kw)
    rcf = C.backtest(PR, wcf, lag=1)
    reg(f"{k} NO-GATE counterfactual | real_lag1")
    act = RUN[(k, "real_lag1")]
    CF[k] = dict(oos_total_no_gate=float((1 + rcf.loc[OOS]).prod() - 1),
                 oos_total_actual=float((1 + act.loc[OOS]).prod() - 1),
                 riskoff_phase_no_gate=float((1 + rcf.loc[EXIT_B1:END]).prod() - 1),
                 riskoff_phase_actual=float((1 + act.loc[EXIT_B1:END]).prod() - 1),
                 maxdd_no_gate=oos_stats(rcf)["maxdd"])

# rule-removal decomposition (DIAGNOSTIC ONLY: explains where the OOS result came from; one 81-day window
# is never a reason to change a rule -- the in-sample columns are printed next to it for that reason)
RULES = {}
for k in ["b_F3_TQQQ_IEF_GLD", "c_F2_QLD_IEF_GLD_vt30"]:
    base = {kk: vv for kk, vv in STRATS[k].items()}
    var = {"as specified": {},
           "no Aug-Sep halving": dict(seasonal=False),
           "no chop halving": dict(chop_pos=1.0),
           "no Aug-Sep + no chop": dict(seasonal=False, chop_pos=1.0),
           "BIL instead of IEF off-leg": dict(off="bills"),
           "no gold (100% engine)": dict(gold_w=0.0)}
    RULES[k] = {}
    for nm, ch in var.items():
        w_ = C.strategy_weights(P, **{**base, **ch})
        r_oos = C.backtest(PR, w_, lag=1)
        r_is = C.backtest(P, w_, lag=1)
        if nm != "as specified":
            reg(f"{k} rule-removal '{nm}' | real_lag1 (diagnostic)")
        s_is = C.stats(r_is, b=FREEZE)
        held_ = C.held_weights(PR, w_, lag=1)
        RULES[k][nm] = dict(oos_total=float((1 + r_oos.loc[OOS]).prod() - 1),
                            oos_to_ndx_record=float((1 + r_oos.loc[A0:NDX_PEAK]).prod() - 1),
                            oos_record_to_exit=float((1 + r_oos.loc[nxt(NDX_PEAK):prv(EXIT_B1)]).prod() - 1),
                            oos_maxdd=oos_stats(r_oos)["maxdd"],
                            avg_lev_weight_freeze_to_record=float(held_[LEV[k]].loc[A0:NDX_PEAK].mean()),
                            is_cagr=s_is["cagr"], is_maxdd=s_is["maxdd"], is_calmar=s_is["calmar"])

# ============================================================== 7. rolling-window percentiles
def windows(r, n=N, a=HIST_A, b=FREEZE):
    s = r.loc[a:b].fillna(0)
    L = np.concatenate([[0.0], np.cumsum(np.log1p(s.values))])
    rets = np.expm1(L[n:] - L[:-n])
    Wl = sliding_window_view(L, n + 1)
    dd = np.expm1((Wl - np.maximum.accumulate(Wl, axis=1)).min(axis=1))
    return rets, dd, s.index[:len(rets)]


def oos_ret_dd(r):
    s = r.loc[OOS]
    L = np.concatenate([[0.0], np.cumsum(np.log1p(s.values))])
    return float(np.expm1(L[-1])), float(np.expm1((L - np.maximum.accumulate(L)).min()))


ndx_ret_series = P["ndx_ret"]
ndx_w, ndx_dd, _ = windows(ndx_ret_series)
ndx_oos, ndx_oos_dd = oos_ret_dd(ndx_ret_series)
REAL_ERA = P["tqqq_real"].first_valid_index()
ndx_w_re, _, _ = windows(ndx_ret_series, a=REAL_ERA)


def episodes(mask_starts):
    """approximate number of non-overlapping episodes among qualifying window start positions."""
    pos = np.flatnonzero(mask_starts)
    if len(pos) == 0:
        return 0
    return int(1 + (np.diff(pos) > N).sum())


PCT = {}
for k in W:
    if k == "6040_SPY_IEF":
        continue
    r = RUN[(k, "model_spliced_lag1")]
    hw, hdd, st = windows(r)
    o_ret, o_dd = oos_ret_dd(RUN[(k, "real_lag1")])
    b, a = np.polyfit(ndx_w, hw, 1)
    res_h = hw - (a + b * ndx_w)
    res_o = o_ret - (a + b * ndx_oos)
    c1 = np.abs(ndx_w - ndx_oos) <= 0.04
    c2 = c1 & (ndx_dd <= -0.10)
    hw_re, hdd_re, _ = windows(r, a=REAL_ERA)
    worst = np.argsort(hw)[:1]
    PCT[k] = dict(
        oos_return=o_ret, oos_maxdd=o_dd, n_windows=int(len(hw)),
        pct_rank_return=float(100 * (hw <= o_ret).mean()),
        pct_rank_maxdd=float(100 * (hdd <= o_dd).mean()),
        hist_quantiles_return={q: float(np.percentile(hw, q)) for q in [1, 5, 10, 25, 50, 75, 90, 95]},
        hist_mean=float(hw.mean()), hist_std=float(hw.std()), z_score=float((o_ret - hw.mean()) / hw.std()),
        share_windows_negative=float((hw < 0).mean()),
        hist_quantiles_maxdd={q: float(np.percentile(hdd, q)) for q in [1, 5, 10, 25, 50]},
        real_tqqq_era=dict(start=str(REAL_ERA.date()), n_windows=int(len(hw_re)),
                           pct_rank_return=float(100 * (hw_re <= o_ret).mean()),
                           pct_rank_maxdd=float(100 * (hdd_re <= o_dd).mean()),
                           median=float(np.median(hw_re))),
        cond_ndx_within_4pp=dict(n_windows=int(c1.sum()), approx_episodes=episodes(c1),
                                 pct_rank_return=float(100 * (hw[c1] <= o_ret).mean()) if c1.any() else None,
                                 median=float(np.median(hw[c1])) if c1.any() else None),
        cond_ndx_within_4pp_and_ndx_dd_le_10=dict(
            n_windows=int(c2.sum()), approx_episodes=episodes(c2),
            pct_rank_return=float(100 * (hw[c2] <= o_ret).mean()) if c2.any() else None,
            median=float(np.median(hw[c2])) if c2.any() else None),
        regression_on_ndx=dict(beta=float(b), alpha=float(a), expected_given_ndx=float(a + b * ndx_oos),
                               residual=float(res_o), pct_rank_residual=float(100 * (res_h <= res_o).mean())),
        worst_hist_window=dict(start=str(st[worst[0]].date()), ret=float(hw[worst[0]])),
    )
PCT["_NDX"] = dict(oos_return=ndx_oos, oos_maxdd=ndx_oos_dd, pct_rank_return=float(100 * (ndx_w <= ndx_oos).mean()),
                   pct_rank_maxdd=float(100 * (ndx_dd <= ndx_oos_dd).mean()))

# tail check of the whipsaw itself: the record->exit stretch and the worst held week, vs history
WHIP = {}
if EXIT_B1 is not None:
    seg = IDX[(IDX > NDX_PEAK) & (IDX < EXIT_B1)]
    nseg = len(seg)
    for k in CORE4:
        r = RUN[(k, "model_spliced_lag1")]
        hw, _, st = windows(r, n=nseg)
        o = float((1 + RUN[(k, "real_lag1")].loc[seg]).prod() - 1)
        # Friday-to-Friday held-week returns (decision t -> next decision), history vs the worst OOS week
        m = P.index[P["mark"].values]
        eq = (1 + r.fillna(0)).cumprod()
        wk = (eq.loc[m] / eq.loc[m].shift(1) - 1).dropna()
        wk_h = wk[(wk.index >= HIST_A) & (wk.index <= FREEZE)]
        wk_o = wk[wk.index > FREEZE]
        worse = np.flatnonzero(hw <= o)
        WHIP[k] = dict(stretch=[str(seg[0].date()), str(seg[-1].date())], days=nseg, stretch_return=o,
                       pct_rank_among_same_length_windows=float(100 * (hw <= o).mean()),
                       approx_episodes_as_bad=int(0 if len(worse) == 0 else 1 + (np.diff(worse) > nseg).sum()),
                       worst_oos_week=float(wk_o.min()), worst_oos_week_ending=str(wk_o.idxmin().date()),
                       pct_rank_worst_week_among_hist_weeks=float(100 * (wk_h <= wk_o.min()).mean()),
                       hist_weeks_worse=int((wk_h < wk_o.min()).sum()), hist_weeks=int(len(wk_h)),
                       hist_weeks_worse_dates=[str(d.date()) for d in wk_h[wk_h < wk_o.min()].index])

# consistency: compounding the held-week returns of the decision log == OOS total
WEEK_CHECK = {}

# same calendar window every year (first trading day after Jul-24 .. last trading day <= Nov-17)
def cal_ret(r, y):
    a = IDX[IDX > pd.Timestamp(f"{y}-07-24")][0]
    b = IDX[IDX <= pd.Timestamp(f"{y}-11-17")][-1]
    return float((1 + r.loc[a:b].fillna(0)).prod() - 1)


CAL = {}
for k in ["a_F_scanner_TQQQ_TLT_GLD", "b_F3_TQQQ_IEF_GLD", "c_F2_QLD_IEF_GLD_vt30", "d_A_TQQQ_BIL_vt25",
          "QQQ_buyhold", "TQQQ_buyhold"]:
    r = RUN[(k, "model_spliced_lag1")]
    yrs = {y: cal_ret(r, y) for y in range(1988, 2026)}
    v26 = float((1 + RUN[(k, "real_lag1")].loc[OOS]).prod() - 1)
    arr = np.array(list(yrs.values()))
    CAL[k] = dict(y2026=v26, rank_from_worst=int((arr < v26).sum() + 1), n_years=int(len(arr) + 1),
                  median_1988_2025=float(np.median(arr)), worse_years={y: v for y, v in yrs.items() if v < v26},
                  by_year=yrs)
CAL["_NDX"] = {y: cal_ret(P["ndx_ret"], y) for y in range(1988, 2026)}

# ============================================================== 8. weekly decision log
def engine_row(t, voltgt, chop_pos=0.5, seasonal=True):
    vol, er, ab, mo = P.at[t, "vol20"], P.at[t, "er20"], bool(P.at[t, "above"]), int(P.at[t, "month"])
    raw = min(1.0, voltgt / vol)
    sm = 0.5 if (seasonal and mo in (8, 9)) else 1.0
    chop = bool(er < 0.25)
    eng = (raw * sm * (chop_pos if chop else 1.0)) if ab else 0.0
    return raw, sm, chop, eng


log_dates = list(DEC) + ([END] if not P.at[END, "mark"] else [])
DECLOG = []
for i, t in enumerate(log_dates):
    raw_b, sm, chop, eng_b = engine_row(t, 0.20)
    raw_c, _, _, eng_c = engine_row(t, 0.30)
    raw_d, _, _, eng_d = engine_row(t, 0.25, chop_pos=1.0, seasonal=False)
    wb, wc = W["b_F3_TQQQ_IEF_GLD"].loc[t], W["c_F2_QLD_IEF_GLD_vt30"].loc[t]
    wa, wd = W["a_F_scanner_TQQQ_TLT_GLD"].loc[t], W["d_A_TQQQ_BIL_vt25"].loc[t]
    assert abs(wb["tqqq"] - 0.85 * eng_b) < 1e-12 and abs(wc["qld"] - 0.85 * eng_c) < 1e-12
    assert abs(wd["tqqq"] - eng_d) < 1e-12
    indicative = not bool(P.at[t, "mark"])
    eff = nxt(t)
    row = dict(date=str(t.date()), decision=("INDICATIVE (not a Friday; scanner run at last close)" if indicative
                                             else "Friday decision"),
               effective_from=(str(eff.date()) if eff is not None else "next session"),
               ndx=float(P.at[t, "ndx"]), ema100=float(P.at[t, "ema100"]),
               ndx_vs_ema=float(P.at[t, "ndx"] / P.at[t, "ema100"] - 1), above=bool(P.at[t, "above"]),
               vol20=float(P.at[t, "vol20"]), er20=float(P.at[t, "er20"]), month=int(P.at[t, "month"]),
               chop=chop, seasonal_mult=sm, curve_inverted_last_12m=bool(inv_recent.at[t]),
               b_scale_raw=raw_b, b_engine=eng_b,
               b_weights=dict(TQQQ=float(wb["tqqq"]), IEF=float(wb["ief"]), GLD=float(wb["gold_phys"])),
               c_scale_raw=raw_c, c_engine=eng_c,
               c_weights=dict(QLD=float(wc["qld"]), IEF=float(wc["ief"]), GLD=float(wc["gold_phys"])),
               a_weights=dict(TQQQ=float(wa["tqqq"]), TLT=float(wa["tlt"]), GLD=float(wa["gold_phys"])),
               d_weights=dict(TQQQ=float(wd["tqqq"]), BIL=float(wd["bills"])))
    if not indicative and eff is not None:
        later = [x for x in log_dates[i + 1:] if P.at[x, "mark"]]
        hold_end = later[0] if later else END
        for k, tag in [("b_F3_TQQQ_IEF_GLD", "b"), ("c_F2_QLD_IEF_GLD_vt30", "c"),
                       ("a_F_scanner_TQQQ_TLT_GLD", "a"), ("d_A_TQQQ_BIL_vt25", "d")]:
            row[f"{tag}_return_while_held"] = float((1 + RUN[(k, "real_lag1")].loc[eff:hold_end]).prod() - 1)
        row["ndx_return_while_held"] = float(P.at[hold_end, "ndx"] / P.at[t, "ndx"] - 1)
        row["held_until_close_of"] = str(hold_end.date())
    DECLOG.append(row)

for tag, k in [("a", "a_F_scanner_TQQQ_TLT_GLD"), ("b", "b_F3_TQQQ_IEF_GLD"), ("c", "c_F2_QLD_IEF_GLD_vt30"),
               ("d", "d_A_TQQQ_BIL_vt25")]:
    comp = float(np.prod([1 + r_[f"{tag}_return_while_held"] for r_ in DECLOG if f"{tag}_return_while_held" in r_]) - 1)
    WEEK_CHECK[tag] = dict(compounded_log=comp, oos_total=OOSR[f"{k}|real_lag1"]["total"])
    assert abs(comp - OOSR[f"{k}|real_lag1"]["total"]) < 1e-10, WEEK_CHECK

# scanner cross-check (scanner_state.json written by the live scanner on 2026-08-01)
scan_chk = None
if os.path.exists(SCANNER_STATE):
    ss = json.load(open(SCANNER_STATE))
    t = pd.Timestamp(ss["asof"])
    if "strategy_F" not in ss:                      # scanner v2 (Nov 2026) writes tiers, not the July v1 keys
        scan_chk = dict(asof=ss["asof"], skipped="scanner_state.json is from scanner v2; this cross-check reads the "
                                                   "July v1 fields (strategy_F / strategy_A), so it is not applicable")
        print("scanner cross-check skipped: scanner_state.json is from scanner v2")
    elif t in IDX:
        wa = W["a_F_scanner_TQQQ_TLT_GLD"].loc[t]
        scan_chk = dict(asof=ss["asof"], scanner=dict(ndx=ss["ndx"], ema100=ss["ema100"], vol=ss["realized_vol"],
                                                      er=ss["efficiency_ratio"], F=ss["strategy_F"], A=ss["strategy_A"]),
                        replicated=dict(ndx=round(float(P.at[t, "ndx"]), 1), ema100=round(float(P.at[t, "ema100"]), 1),
                                        vol=round(float(P.at[t, "vol20"]), 4), er=round(float(P.at[t, "er20"]), 3),
                                        F=dict(TQQQ=round(float(wa["tqqq"]), 3), TLT=round(float(wa["tlt"]), 3),
                                               GLD=round(float(wa["gold_phys"]), 3)),
                                        A=dict(TQQQ=round(float(W["d_A_TQQQ_BIL_vt25"].at[t, "tqqq"]), 3))))
        scan_chk["match"] = (scan_chk["scanner"]["F"] == scan_chk["replicated"]["F"]
                             and abs(ss["ndx"] - scan_chk["replicated"]["ndx"]) < 0.05)

# ============================================================== 9. current holdings and re-entry level
alpha = 2 / 101
ema_proj = float(P.at[END, "ema100"])
for _ in range(2):                                  # Wed 11-18, Thu 11-19 with NDX unchanged at the last close
    ema_proj = ema_proj + alpha * (float(P.at[END, "ndx"]) - ema_proj)
CUR = {}
for k, tag in [("b_F3_TQQQ_IEF_GLD", "b"), ("c_F2_QLD_IEF_GLD_vt30", "c"),
               ("a_F_scanner_TQQQ_TLT_GLD", "a"), ("d_A_TQQQ_BIL_vt25", "d")]:
    held = C.held_weights(PR, W[k], lag=1).loc[END]
    CUR[tag] = dict(held_on_last_date_from_decision=str(DEC[-1].date()),
                    held={c: float(v) for c, v in held.items() if abs(v) > 1e-12},
                    indicative_target_at_last_close={c: float(v) for c, v in W[k].loc[END].items() if abs(v) > 1e-12})
eng_if_on_b = min(1, 0.20 / P.at[END, "vol20"]) * (0.5 if P.at[END, "er20"] < 0.25 else 1.0)
eng_if_on_c = min(1, 0.30 / P.at[END, "vol20"]) * (0.5 if P.at[END, "er20"] < 0.25 else 1.0)
REENTRY = dict(next_decision="2026-11-20 (Friday close)",
               rule="risk-on iff Friday close NDX > EMA-100, equivalently NDX_fri > EMA_100(Thursday)",
               ema100_last=float(P.at[END, "ema100"]),
               approx_threshold_if_ndx_flat_until_thursday=ema_proj,
               required_ndx_move_from_last=float(ema_proj / P.at[END, "ndx"] - 1),
               weights_if_reentered_at_current_vol_er=dict(
                   b_TQQQ=float(0.85 * eng_if_on_b), c_QLD=float(0.85 * eng_if_on_c),
                   note=f"vol20 {P.at[END, 'vol20']:.3f}, er20 {P.at[END, 'er20']:.3f}; Nov/Dec -> no seasonal cut"))

# ============================================================== 10. summary print + JSON
order = ["a_F_scanner_TQQQ_TLT_GLD", "b_F3_TQQQ_IEF_GLD", "c_F2_QLD_IEF_GLD_vt30", "d_A_TQQQ_BIL_vt25",
         "ref_Fjuly_TQQQ_bond10syn_XAUminers"] + list(BENCH)
print(f"\n{'strategy':38s}{'IS CAGR':>8}{'IS DD':>7}{'Cal':>5} | {'OOS l1':>7}{'OOS l2':>7}{'drift1':>7}"
      f"{'drift2':>7}{'open':>7}{'sim':>7}{'OOS DD':>7}{'pct':>6}{'pctDD':>6}")
TABLE = {}
for k in order:
    g = lambda m: OOSR.get(f"{k}|{m}", {}).get("total", np.nan)
    isr = ISR.get(k, {}).get("lag1", {})
    pc = PCT.get(k, {})
    TABLE[k] = dict(is_cagr=isr.get("cagr"), is_maxdd=isr.get("maxdd"), is_calmar=isr.get("calmar"),
                    is_cagr_lag2=ISR.get(k, {}).get("lag2", {}).get("cagr"),
                    oos_lag1=g("real_lag1"), oos_lag2=g("real_lag2"), oos_drift_fri_close=g("real_drift_close1"),
                    oos_drift_mon_close=g("real_drift_close2"), oos_drift_mon_open=g("real_drift_open1"),
                    oos_sim_sleeves=g("sim_sleeves_lag1"), oos_maxdd=OOSR[f"{k}|real_lag1"]["maxdd"],
                    pct_rank=pc.get("pct_rank_return"), pct_rank_dd=pc.get("pct_rank_maxdd"))
    t_ = TABLE[k]
    fm = lambda x, w=7: f"{x*100:{w}.1f}" if x is not None and np.isfinite(x) else f"{'-':>{w}}"
    print(f"{k:38s}{fm(t_['is_cagr'],8)}{fm(t_['is_maxdd'])}{(t_['is_calmar'] or np.nan):5.2f} |"
          f"{fm(t_['oos_lag1'])}{fm(t_['oos_lag2'])}{fm(t_['oos_drift_fri_close'])}{fm(t_['oos_drift_mon_close'])}"
          f"{fm(t_['oos_drift_mon_open'])}{fm(t_['oos_sim_sleeves'])}{fm(t_['oos_maxdd'])}"
          f"{(t_['pct_rank'] if t_['pct_rank'] is not None else np.nan):6.1f}"
          f"{(t_['pct_rank_dd'] if t_['pct_rank_dd'] is not None else np.nan):6.1f}")

print(f"\nNDX OOS {ndx_oos*100:.2f}%  (pct {PCT['_NDX']['pct_rank_return']:.1f})  record {mkt['NDX_levels']['record_close']:.0f}"
      f" on {mkt['NDX_levels']['record_date']}  last {mkt['NDX_levels']['last']:.0f}  exit (lag1) effective {EXIT_B1.date()}")
print("\nPHASE ATTRIBUTION (lag 1, compounded return; P&L % of start capital by sleeve)")
for k in ["a_F_scanner_TQQQ_TLT_GLD", "b_F3_TQQQ_IEF_GLD", "c_F2_QLD_IEF_GLD_vt30", "d_A_TQQQ_BIL_vt25",
          "QQQ_buyhold", "TQQQ_buyhold", "static_IEF85_GLD15"]:
    at = ATTR[f"{k}|lag1"]
    segs = [s for s in at if not s.startswith("_")]
    print(f"  {k:36s} " + " | ".join(f"{s.split(':')[0][:26]} {at[s]['compounded_return']*100:6.2f}%" for s in segs))
    if k.startswith(("a_", "b_", "c_", "d_")):
        for s in segs[:2]:
            print(f"      {s:30s} " + "  ".join(f"{c} {v*100:+.2f}" for c, v in at[s]["pnl_pct_of_start_capital"].items()))
print("\nPERCENTILES (81-day windows 1988..freeze): ret pct | cond NDX+-4pp | cond +NDX dd<=-10% | resid pct | real era")
for k in order:
    if k in PCT:
        p = PCT[k]
        print(f"  {k:38s} {p['pct_rank_return']:5.1f} | {p['cond_ndx_within_4pp']['pct_rank_return']:5.1f}"
              f" (n={p['cond_ndx_within_4pp']['approx_episodes']}) | {p['cond_ndx_within_4pp_and_ndx_dd_le_10']['pct_rank_return'] or float('nan'):5.1f}"
              f" (n={p['cond_ndx_within_4pp_and_ndx_dd_le_10']['approx_episodes']}) | {p['regression_on_ndx']['pct_rank_residual']:5.1f}"
              f" | {p['real_tqqq_era']['pct_rank_return']:5.1f}   median {p['hist_quantiles_return'][50]*100:5.1f}%"
              f" 5th {p['hist_quantiles_return'][5]*100:6.1f}%")
print("\nSAME CALENDAR WINDOW (Jul-27..Nov-17) rank of 2026 from worst:",
      {k: f"{v['rank_from_worst']}/{v['n_years']}" for k, v in CAL.items() if not k.startswith("_")})
print("\nCURRENT:", json.dumps(jsafe(CUR), indent=None))
print("RE-ENTRY:", json.dumps(jsafe(REENTRY)))
print("\ntracking (strategy real vs sim sleeves):",
      {k: round(v["gap_pp"] * 100, 2) for k, v in strat_track.items()})
print("\nLAG COST IN-SAMPLE (model, CAGR lag1 -> lag2 by sub-period);  NDX daily autocorr:",
      {k: round(v, 3) for k, v in NDX_AUTOCORR.items()})
for k, v in LAGCOST.items():
    print(f"  {k:38s} " + " | ".join(f"{nm} {x['cagr_lag1']*100:5.1f}->{x['cagr_lag2']*100:5.1f}" for nm, x in v.items()))
if EXEC_IS:
    print(f"\nEXECUTION TIMING, REAL ETFs, drift sim {EXEC_IS['_span'][0]}..{EXEC_IS['_span'][1]}  (CAGR / maxDD / Calmar)")
    for k in CORE4:
        e = EXEC_IS[k]
        print(f"  {k:30s} " + " | ".join(f"{m} {e[m]['cagr']*100:5.1f}% {e[m]['maxdd']*100:5.1f}% {e[m]['calmar']:.2f}"
                                         for m in ["close1", "open1", "close2", "model_constmix_lag1", "model_constmix_lag2"]))
print("overnight-return fills:", ON_FILLS)
print("\nWHIPSAW TAIL CHECK (record -> exit stretch, and worst held week):")
for k, v in WHIP.items():
    print(f"  {k:30s} {v['stretch'][0]}..{v['stretch'][1]} ({v['days']}d) {v['stretch_return']*100:6.2f}%  pct "
          f"{v['pct_rank_among_same_length_windows']:5.2f} (~{v['approx_episodes_as_bad']} episodes as bad) | worst week "
          f"{v['worst_oos_week']*100:6.2f}% ({v['worst_oos_week_ending']}) pct {v['pct_rank_worst_week_among_hist_weeks']:5.2f}"
          f" ({v['hist_weeks_worse']}/{v['hist_weeks']} hist weeks worse)")
print("weekly-log compounding check:", {k: round(v['compounded_log'] * 100, 3) for k, v in WEEK_CHECK.items()})
print("\nRULE-REMOVAL DECOMPOSITION (diagnostic only): OOS total | to record | record->exit | avg lev wt to record || IS CAGR / DD / Calmar")
for k, v in RULES.items():
    for nm, x in v.items():
        print(f"  {k[:22]:22s} {nm:28s} {x['oos_total']*100:6.1f}% | {x['oos_to_ndx_record']*100:6.1f}% | "
              f"{x['oos_record_to_exit']*100:6.1f}% | {x['avg_lev_weight_freeze_to_record']:.3f} || "
              f"{x['is_cagr']*100:5.1f}% {x['is_maxdd']*100:6.1f}% {x['is_calmar']:.2f}")
print("variants evaluated:", len(VARIANTS))

out = dict(
    meta=dict(key="oos_live", window=dict(start=str(A0.date()), end=str(END.date()), trading_days=N),
              freeze=str(FREEZE.date()), history_for_percentiles=[str(HIST_A.date()), str(FREEZE.date())],
              cost_bps_per_turnover=C.COST * 1e4, rebalance="Friday close marks (P.mark)",
              gold_sim_in_tracking_test=GOLD_SIM,
              no_parameter_chosen="all configurations are the pre-registered July specs; nothing tuned on OOS"),
    checks=dict(no_lookahead=lookahead, curve_rule_is_noop_everywhere=curve_rule_noop,
                spliced_equals_real_in_window=spliced_vs_real, real_run_equals_spliced_run=real_eq_spliced,
                nan_real_columns_2026=nan_real, independent_yfinance_check=indep, scanner_state_crosscheck=scan_chk),
    market_context=mkt,
    results_table=TABLE,
    oos_stats_all_runs=OOSR,
    in_sample_stats=ISR,
    tracking_gap=dict(sleeves=sleeve_track, strategies_real_minus_sim=strat_track),
    phase_attribution=ATTR,
    exit_counterfactual_no_gate=CF,
    rule_removal_decomposition_diagnostic_only=RULES,
    execution_timing_in_sample=dict(model_lag_cost_by_subperiod=LAGCOST, ndx_daily_autocorr=NDX_AUTOCORR,
                                    real_etf_drift_sims=EXEC_IS, overnight_open_fills=ON_FILLS,
                                    modes=dict(close1="trade at Friday close (lag 1)",
                                               open1="trade at Monday open", close2="trade at Monday close (lag 2)")),
    percentiles=PCT,
    whipsaw_tail_check=WHIP,
    same_calendar_window=CAL,
    decision_log_compounding_check=WEEK_CHECK,
    decision_log=DECLOG,
    current_holdings=CUR,
    reentry=REENTRY,
    variants=dict(count=len(VARIANTS), list=VARIANTS),
)
json.dump(jsafe(out), open(OUTJ, "w"), indent=1)
print(f"\nsaved {OUTJ}")
