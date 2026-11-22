"""
managed_futures.py  -  Nov-2026 refresh, dimension: MANAGED-FUTURES (TSMOM) DIVERSIFIER

Question: would a trend-following / managed-futures sleeve (the DBMF / KMLM kind of product) have
made Strategy F more robust, especially in stocks-AND-bonds-down regimes like 2022?

1) A transparent time-series-momentum (TSMOM) proxy is built from long-history panel series.
   Everything below was fixed BEFORE looking at results (textbook Moskowitz-Ooi-Pedersen style):
     markets   spx, ndx, intl  (price index return - rf_d; dividends NOT in the panel -> sensitivity)
               ief, tlt        (total return - rf_d)
               gold            (gold_phys - rf_d)
               dxy             (spot DXY change; FX carry not available -> no rf subtraction)
               gsci            (S&P GSCI SPOT change; roll yield not available -> no rf subtraction;
                                sensitivity uses GSG = futures-based GSCI total return from 2006-07)
     signal    sign of trailing 12m (252d) cumulative excess return   [alt: mean sign of 1/3/12m]
     sizing    each market targets 40%/sqrt(N) vol using its 63d realised vol at the signal date
     scaling   whole book -> 12% ex-ante vol: 126d historical simulation of the NEW positions
               (only data up to the signal date); scale capped at 1.5
     trading   month-end close signal, applied from the next session (lag 1);
               3 bp x notional turnover + 0.75%/yr fee/slippage drag
     return    rf_d (collateral) + sum(position x excess return) - costs
   Gold before 2000-09 (gold_phys = monthly-average series, log-interpolated -> future info):
     signal = gold_phys momentum LAGGED 21 trading days (>= 1 month), [sensitivity: gold_miners]
     vol    = 36m std of monthly returns lagged one month / 0.774 smoothing correction
              (vol_interp / vol_gcf from panel_validation.json), used until 126 real daily obs exist;
              during that time gold is excluded from the historical-simulation window and added
              back as an independent variance term.
2) Validation vs real DBMF (2019-05+), KMLM (2020-12+), CTA (2022-03+).
3) Tests inside F3 (TQQQ, vt .20) and F2 (QLD, vt .30), both with 15% gold_phys + IEF off-leg
   (the faithful baselines from the Nov audit):
     (a) permanent TSMOM sleeve 10/20/30% (all other sleeves scaled pro-rata)
     (b) TSMOM replaces the 15% gold   (c) TSMOM replaces IEF as the off-leg
     (d) off-leg 50% IEF + 50% TSMOM
4) Robustness grid (no selection): every structure re-run with ~20 proxy variants
   (lookback, fee, market list, risk budget, dividends, commodity data, gold signal, scaling).
Every evaluated variant is registered and counted. In-sample = 1988-01-01..2026-07-24;
out-of-sample = 2026-07-25..latest (never used to choose anything).

Run:  python -u update_2026_11/managed_futures.py      -> update_2026_11/managed_futures.json
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C          # noqa: E402

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

P = C.panel().copy()        # in-memory copy only; panel.parquet is never written
idx = P.index
START, FREEZE = C.START, C.FREEZE
OOS_A = pd.Timestamp("2026-07-25")
LAST = idx[-1]
H1 = ("1988-01-01", "2006-12-31")
H2 = ("2007-01-01", str(FREEZE.date()))
SEL_COST_LO, SEL_COST_HI = 0.07, 0.15

VALJ = json.load(open(os.path.join(HERE, "panel_validation.json")))
_g = VALJ["gold_monthly_interp_vs_gcf_monthly"]
SMOOTH = _g["vol_interp"] / _g["vol_gcf"]          # ~0.774

REG = []                    # registry of EVERY evaluated variant
def reg(kind, name, **kw):
    REG.append(dict(kind=kind, name=name, **kw))

MKTS = ["spx", "ndx", "intl", "ief", "tlt", "gold", "dxy", "gsci"]
CLASS = dict(spx="equity", ndx="equity", intl="equity", ief="bond", tlt="bond",
             gold="gold", dxy="dollar", gsci="commodity")
DIV = dict(spx=0.020, ndx=0.008, intl=0.018)        # rough long-run dividend yields (sensitivity only)

# ------------------------------------------------------------------ optional data: GSG
GSG_CACHE = os.path.join(HERE, "managed_futures_gsg.parquet")
def load_gsg():
    s = None
    try:
        import yfinance as yf
        h = yf.Ticker("GSG").history(period="max", auto_adjust=True)["Close"].dropna()
        h.index = h.index.tz_localize(None).normalize()
        s = h[~h.index.duplicated()]
        if len(s) > 1000:
            s.to_frame("gsg").to_parquet(GSG_CACHE)
    except Exception as e:                       # offline -> cached copy
        log("GSG fetch failed:", repr(e))
    if (s is None or len(s) < 1000) and os.path.exists(GSG_CACHE):
        s = pd.read_parquet(GSG_CACHE)["gsg"]
    if s is None or len(s) < 1000:
        return None
    r = s.reindex(idx).ffill().pct_change()
    r[idx <= s.index.min()] = np.nan
    return r

GSG = load_gsg()
log("GSG loaded:", None if GSG is None else str(GSG.first_valid_index().date()))

# ------------------------------------------------------------------ building blocks
def excess_returns(div=False, gsg=False):
    X = pd.DataFrame(index=idx)
    X["spx"] = P["spx"] - P["rf_d"]
    X["ndx"] = P["ndx_ret"] - P["rf_d"]
    X["intl"] = P["intl"] - P["rf_d"]
    X["ief"] = P["ief"] - P["rf_d"]
    X["tlt"] = P["tlt"] - P["rf_d"]
    X["gold"] = P["gold_phys"] - P["rf_d"]
    X["dxy"] = P["dxy_ret"]                       # spot FX change (zero-cost forward, carry ignored)
    X["gsci"] = P["gsci_ret"]                     # spot commodity change (roll yield ignored)
    if div:
        for k, v in DIV.items():
            X[k] = X[k] + v / 252
    if gsg and GSG is not None:
        g = GSG + 0.0075 / 252 - P["rf_d"]        # GSG + its fee - T-bill  ~= GSCI excess return
        X["gsci"] = g.where(g.notna(), X["gsci"])
    return X.fillna(0.0)

_me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
if LAST < LAST + pd.offsets.BMonthEnd(0):         # panel ends mid-month -> drop incomplete month
    _me = _me.iloc[:-1]
ME = pd.DatetimeIndex(_me.values)                  # month-end rebalance dates (complete months)

GOLD_SIG_OK = pd.Timestamp("2000-09-01")
_first_real = idx[(P["gold_src"] != "monthly-interp").values][0]
GOLD_DAILY_OK = idx[idx.get_loc(_first_real) + 126]
log("first real daily gold:", _first_real.date(), " gold daily-vol/HS usable from:", GOLD_DAILY_OK.date())


def make_signals(X, lbs, gold_alt=None):
    cum = (1 + X).cumprod()
    tot = pd.DataFrame(0.0, index=idx, columns=X.columns)
    early = idx < GOLD_SIG_OK
    for L in lbs:
        s = np.sign(cum / cum.shift(L) - 1)
        if "gold" in X.columns:
            if gold_alt is None:
                g = s["gold"].shift(21)            # lag >= one month (monthly-interp gold)
            else:
                ca = (1 + gold_alt).cumprod()
                g = np.sign(ca / ca.shift(L) - 1)  # gold-miners momentum (real daily data)
            s.loc[early, "gold"] = g[early]
        tot = tot + s.fillna(0.0)
    return tot / len(lbs)


def make_vols(X):
    V = X.rolling(63).std() * np.sqrt(252)
    if "gold" in X.columns:
        per = idx.to_period("M")
        gm = (1 + X["gold"]).groupby(per).prod() - 1
        gv = (gm.rolling(36, min_periods=12).std() * np.sqrt(12) / SMOOTH).shift(1)
        gv = pd.Series(gv.reindex(per).values, index=idx)
        early = idx < GOLD_DAILY_OK
        V.loc[early, "gold"] = gv[early]
    return V.clip(lower=0.01)


def build_proxy(name, mkts=tuple(MKTS), lbs=(252,), fee=0.0075, div=False, gsg=False, gold_alt=None,
                budget="market", scaling="hs", tgt=0.12, win=126, kcap=1.5, note="", force_sign=None):
    mkts = list(mkts)
    X = excess_returns(div=div, gsg=gsg)[mkts]
    S = make_signals(X, lbs, gold_alt)
    if force_sign is not None:                     # control: same sizing/scaling, no timing
        S = pd.DataFrame(float(force_sign), index=idx, columns=mkts)
    V = make_vols(X)
    N = len(mkts)
    if budget == "market":
        tv = pd.Series(0.40 / np.sqrt(N), index=mkts)
    else:                                          # equal risk per asset class, split within class
        cl = pd.Series({m: CLASS[m] for m in mkts})
        nin = cl.map(cl.value_counts())
        tv = 0.40 / np.sqrt(N) * N / (cl.nunique() * nin)
    first_ok = idx[252 + 21 + 5]
    RB = ME[ME >= first_ok]
    # live "as-if rebalanced today" point (reporting only, never traded in the backtest)
    RBL = RB.append(pd.DatetimeIndex([LAST])) if LAST not in RB else RB
    raw = (S.loc[RBL] * tv / V.loc[RBL]).fillna(0.0)
    has_gold = "gold" in mkts
    ex = [m for m in mkts if m != "gold"]
    if scaling == "realized":
        rawheld = raw.loc[RB].reindex(idx).ffill().shift(1).fillna(0.0)
        rr_all = (rawheld * X).sum(axis=1)
        rr_ex = (rawheld[ex] * X[ex]).sum(axis=1) if has_gold else rr_all
    K = []
    for t in RBL:
        i = idx.get_loc(t)
        early = has_gold and t < GOLD_DAILY_OK
        cols = ex if early else mkts
        if scaling == "hs":
            W = X.iloc[i - win + 1:i + 1][cols]
            var = float((W @ raw.loc[t, cols]).var() * 252)
        else:
            rr = rr_ex if early else rr_all
            var = float(rr.iloc[i - win + 1:i + 1].var() * 252)
        if early:
            var += float((raw.loc[t, "gold"] * V.loc[t, "gold"]) ** 2)
        sh = np.sqrt(var) if var > 0 else np.nan
        K.append(min(tgt / sh, kcap) if (sh is not None and np.isfinite(sh) and sh > 0) else 0.0)
    K = pd.Series(K, index=RBL)
    POSL = raw.mul(K, axis=0)
    POS = POSL.loc[RB]
    held = POS.reindex(idx).ffill().shift(1).fillna(0.0)
    gross = (held * X).sum(axis=1)
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    ret = P["rf_d"] + gross - turn * C.COST - fee / 252
    out = dict(name=name, ret=ret, POS=POS, POS_live=POSL.loc[LAST], K=K.loc[RB], K_live=float(K.loc[LAST]),
               S=S, V=V, X=X, tv=tv, held=held, turn=turn, mkts=mkts, lbs=lbs,
               spec=dict(mkts=mkts, lbs=list(lbs), fee=fee, div=div, gsg=gsg,
                         gold_signal_pre2000=("gold_miners" if gold_alt is not None else "gold_phys lag 21d"),
                         budget=budget, scaling=scaling, tgt=tgt, win=win, kcap=kcap, note=note,
                         force_sign=force_sign))
    return out


def dd_episode(r, a=START, b=FREEZE):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    e = (1 + s).cumprod()
    dd = e / e.cummax() - 1
    tr = dd.idxmin()
    pk = e[:tr].idxmax()
    after = e[tr:]
    rec = after[after >= e[pk]]
    return dict(depth=float(dd.min()), peak=str(pk.date()), trough=str(tr.date()),
                recovered=str(rec.index[0].date()) if len(rec) else None)


def top_drawdowns(r, n=3, a=START, b=FREEZE):
    """n deepest non-overlapping drawdown episodes (peak->trough)."""
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    e = (1 + s).cumprod()
    dd = e / e.cummax() - 1
    out, used = [], pd.Series(False, index=dd.index)
    for _ in range(n):
        d2 = dd[~used]
        if len(d2) == 0 or d2.min() >= 0:
            break
        tr = d2.idxmin()
        pk = e[:tr].idxmax()
        after = e[tr:]
        rec = after[after >= e[pk]]
        end = rec.index[0] if len(rec) else dd.index[-1]
        out.append(dict(depth=float(dd[tr]), peak=str(pk.date()), trough=str(tr.date()),
                        recovered=str(end.date()) if len(rec) else None))
        used[(dd.index >= pk) & (dd.index <= end)] = True
    return out


# ------------------------------------------------------------------ statistics helpers
def wk(r):
    return (1 + r).groupby(r.index.to_period("W-FRI")).prod() - 1

def mo(r):
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1

def yearly(r, a="1988-01-01"):
    s = r[r.index >= pd.Timestamp(a)]
    return {int(k): round(float(v), 4) for k, v in ((1 + s).groupby(s.index.year).prod() - 1).items()}

def maxdd_window(r, a, b):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    if len(s) == 0:
        return np.nan
    e = (1 + s).cumprod()
    e = pd.concat([pd.Series([1.0]), e.reset_index(drop=True)])
    return float((e / e.cummax() - 1).min())

def rnd(d, n=4):
    if isinstance(d, dict):
        return {k: rnd(v, n) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [rnd(v, n) for v in d]
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), n)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d

def proxy_report(pr):
    r = pr["ret"]
    m = (idx >= START) & (idx <= FREEZE)
    ex = (r - P["rf_d"])[m]
    d = dict(spec=pr["spec"])
    d["IS"] = C.stats(r, b=FREEZE)
    d["IS"]["vol"] = float(r[m].std() * np.sqrt(252))
    d["IS"]["sharpe_excess"] = float(ex.mean() / ex.std() * np.sqrt(252))
    d["IS"]["excess_cagr"] = float((1 + ex).prod() ** (252 / len(ex)) - 1)
    d["H1"] = C.stats(r, *H1)
    d["H2"] = C.stats(r, *H2)
    for h, (a, b) in (("H1", H1), ("H2", H2)):
        mm = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        e2 = (r - P["rf_d"])[mm]
        d[h]["sharpe_excess"] = float(e2.mean() / e2.std() * np.sqrt(252))
    d["OOS_total"] = C.period_return(r, OOS_A, LAST)
    d["OOS_maxdd"] = maxdd_window(r, OOS_A, LAST)
    d["y2022"] = C.period_return(r, "2022-01-01", "2022-12-31")
    d["crises"] = C.crises(r)
    held = pr["held"][m]
    d["avg_gross_notional"] = float(held.abs().sum(axis=1).mean())
    d["avg_scale_k"] = float(pr["K"][pr["K"].index <= FREEZE].mean())
    d["kcap_binding_share"] = float((pr["K"] >= pr["spec"]["kcap"] - 1e-9).mean())
    d["annual_turnover_notional"] = float(pr["turn"][m].sum() / (m.sum() / 252))
    return d


def paired_boot(r1, r2, n=2000, block=63, seed=0):
    """paired circular block bootstrap (same draws as C.block_bootstrap_cagr_diff) on the
    IN-SAMPLE window; returns CAGR / Sharpe / maxDD / Calmar differences r1 - r2."""
    a = pd.concat([r1, r2], axis=1).dropna()
    a = a[(a.index >= START) & (a.index <= FREEZE)].values
    T = len(a); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    ar = np.arange(block)
    res = np.empty((n, 4))
    for k in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + ar[None, :]).ravel()[:T] % T
        s = a[ix]
        e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        sh = s.mean(0) / s.std(0, ddof=1) * np.sqrt(252)
        cal = g / np.abs(dd)
        res[k] = (g[0] - g[1], sh[0] - sh[1], dd[0] - dd[1], cal[0] - cal[1])
    out = {}
    for j, kk in enumerate(["d_cagr", "d_sharpe", "d_maxdd", "d_calmar"]):
        x = res[:, j]
        out[kk] = dict(mean=float(x.mean()), lo=float(np.percentile(x, 2.5)),
                       hi=float(np.percentile(x, 97.5)), p_le0=float((x <= 0).mean()))
    return out


# ------------------------------------------------------------------ Strategy F plumbing
def F(lev, vt, gold="gold_phys", off="ief", off_weights=None):
    return C.strategy_weights(P, lev=lev, voltgt=vt, gold=gold, gold_w=0.15, off=off, off_weights=off_weights)

def sleeve(w, col, s):
    w2 = w * (1 - s)
    w2[col] = w2.get(col, 0.0) + s
    return w2

ENGINES = {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}
STRUCTS = ["sleeve10", "sleeve20", "sleeve30", "replace_gold", "replace_ief_offleg", "offleg_50ief_50tsmom"]

def structure_weights(lev, vt, col, st):
    if st.startswith("sleeve"):
        return sleeve(F(lev, vt), col, int(st[6:]) / 100)
    if st == "replace_gold":
        return F(lev, vt, gold=col)
    if st == "replace_ief_offleg":
        return F(lev, vt, off=col)
    if st == "offleg_50ief_50tsmom":
        return F(lev, vt, off=None, off_weights=pd.DataFrame({"ief": 0.5, col: 0.5}, index=idx))
    raise ValueError(st)

def evaluate(name, w, base=None, boot=False, tag=None, col=None):
    r = C.backtest(P, w)
    d = dict(name=name)
    d["IS"] = C.stats(r, b=FREEZE)
    _m = (idx >= START) & (idx <= FREEZE)
    _x = (r - P["rf_d"])[_m]
    d["IS"]["sharpe_excess"] = float(_x.mean() / _x.std() * np.sqrt(252))     # rf-adjusted (C.stats ignores rf)
    d["H1"] = C.stats(r, *H1)
    d["H2"] = C.stats(r, *H2)
    d["latest_incl_OOS"] = C.stats(r)
    d["OOS_total"] = C.period_return(r, OOS_A, LAST)
    d["OOS_maxdd"] = maxdd_window(r, OOS_A, LAST)
    d["crises"] = C.crises(r)
    if col is not None:
        hw = C.held_weights(P, w)
        mm = (idx >= START) & (idx <= FREEZE)
        d["avg_weight_tsmom"] = float(hw[col][mm].mean()) if col in hw else 0.0
    if base is not None:
        b = base
        d["delta_vs_base"] = {k: {m: d[k][m] - b["d"][k][m] for m in ("cagr", "maxdd", "sharpe", "calmar")}
                              for k in ("IS", "H1", "H2")}
        d["delta_vs_base"]["OOS_total"] = d["OOS_total"] - b["d"]["OOS_total"]
        d["delta_vs_base"]["y2022"] = d["crises"]["2022 inflation"] - b["d"]["crises"]["2022 inflation"]
        if boot:
            d["boot"] = paired_boot(r, b["r"])
            dc = d["delta_vs_base"]
            ok_halves = dc["H1"]["calmar"] > 0 and dc["H2"]["calmar"] > 0
            sig = d["boot"]["d_calmar"]["lo"] > 0
            if dc["IS"]["calmar"] > SEL_COST_HI and ok_halves and sig:
                v = "PASS"
            elif dc["IS"]["calmar"] > SEL_COST_LO and ok_halves:
                v = "MARGINAL (beats low selection cost & both halves, not bootstrap-significant)" if not sig else \
                    "MARGINAL (significant & both halves, but below the 0.15 selection-cost ceiling)"
            else:
                v = "FAIL"
            d["verdict"] = v
    reg("portfolio", name, tag=tag, IS_calmar=d["IS"]["calmar"], IS_cagr=d["IS"]["cagr"])
    return dict(r=r, d=d)


# ================================================================== 1. PROXIES
log("building proxies ...")
PROX = {}
def add_proxy(key, **kw):
    pr = build_proxy(key, **kw)
    PROX[key] = pr
    reg("proxy", key, spec=pr["spec"])
    return pr

add_proxy("main_12m")
add_proxy("blend_1_3_12m", lbs=(21, 63, 252))
# --- sensitivities (robustness only; nothing is selected from these) ---
add_proxy("lb_1m", lbs=(21,))
add_proxy("lb_3m", lbs=(63,))
add_proxy("lb_6m", lbs=(126,))
add_proxy("fee_0", fee=0.0)
add_proxy("fee_2pct", fee=0.02)
for m in MKTS:
    add_proxy(f"drop_{m}", mkts=tuple(x for x in MKTS if x != m))
add_proxy("ex_equities", mkts=("ief", "tlt", "gold", "dxy", "gsci"), note="KMLM-like: no equity futures")
add_proxy("class_risk_parity", budget="class")
add_proxy("div_addback", div=True)
add_proxy("gold_sig_miners_pre2000", gold_alt=(P["gold_miners"] - P["rf_d"]).fillna(0.0))
add_proxy("realized_vol_scaling", scaling="realized")
if GSG is not None:
    add_proxy("gsg_commodity_2006plus", gsg=True, note="GSCI futures excess return (GSG) from 2006-07")
    add_proxy("faithful_div_gsg", div=True, gsg=True, note="dividends + GSG commodity leg")
# CONTROL (not a candidate): identical markets / sizing / 12% scaling / costs but ALWAYS LONG.
# Separates the value of trend TIMING from plain multi-asset diversification + secular bond/equity beta.
add_proxy("control_always_long", force_sign=1.0, note="control: no timing")
log(f"{len(PROX)} proxies built")

PREP = {k: proxy_report(v) for k, v in PROX.items()}
main, blend = PROX["main_12m"], PROX["blend_1_3_12m"]

# correlations of every proxy with main (IS daily) + summary row
for k, v in PROX.items():
    m = (idx >= START) & (idx <= FREEZE)
    PREP[k]["corr_with_main_daily_IS"] = float(v["ret"][m].corr(main["ret"][m]))
    PREP[k]["corr_with_main_monthly_IS"] = float(mo(v["ret"][m]).corr(mo(main["ret"][m])))

# yearly table for main & blend
YEARLY = {"main_12m": yearly(main["ret"]), "blend_1_3_12m": yearly(blend["ret"]),
          "ief": yearly(P["ief"]), "spx": yearly(P["spx"]), "ndx": yearly(P["ndx_ret"]),
          "gold_phys": yearly(P["gold_phys"])}

# ================================================================== 2. VALIDATION vs real ETFs
log("validating vs DBMF / KMLM / CTA ...")
def validate(pr_ret, col):
    d = pd.concat([pr_ret.rename("proxy"), P[col].rename("etf"), P["rf_d"].rename("rf")], axis=1).dropna()
    out = dict(etf=col, start=str(d.index[0].date()), end=str(d.index[-1].date()), days=int(len(d)))
    exd = d[["proxy", "etf"]].sub(d["rf"], axis=0)
    out["corr_daily"] = float(exd.proxy.corr(exd.etf))
    W = pd.DataFrame({c: wk(d[c]) for c in ["proxy", "etf", "rf"]})
    M = pd.DataFrame({c: mo(d[c]) for c in ["proxy", "etf", "rf"]})
    W = W.iloc[1:-1]; M = M.iloc[1:-1]                      # drop partial first/last periods
    we = W[["proxy", "etf"]].sub(W["rf"], axis=0)
    me_ = M[["proxy", "etf"]].sub(M["rf"], axis=0)
    out["corr_weekly"] = float(we.proxy.corr(we.etf))
    out["corr_monthly"] = float(me_.proxy.corr(me_.etf))
    cv = np.cov(we.etf, we.proxy)
    out["beta_etf_on_proxy_weekly"] = float(cv[0, 1] / cv[1, 1])
    out["beta_proxy_on_etf_weekly"] = float(cv[0, 1] / cv[0, 0])
    out["r2_weekly"] = float(out["corr_weekly"] ** 2)
    out["tracking_error_weekly_ann"] = float((we.etf - we.proxy).std() * np.sqrt(52))
    for lbl, s in (("proxy", d.proxy), ("etf", d.etf)):
        e = (1 + s).cumprod()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        out[f"cagr_{lbl}"] = float(e.iloc[-1] ** (1 / yrs) - 1)
        out[f"vol_{lbl}"] = float(s.std() * np.sqrt(252))
        out[f"maxdd_{lbl}"] = float((e / e.cummax() - 1).min())
        out[f"y2022_{lbl}"] = C.period_return(s, "2022-01-01", "2022-12-31") if s.index[0] <= pd.Timestamp("2022-01-03") else None
        out[f"to_freeze_total_{lbl}"] = C.period_return(s, s.index[0], FREEZE)
        out[f"OOS_total_{lbl}"] = C.period_return(s, OOS_A, LAST)
        out[f"OOS_maxdd_{lbl}"] = maxdd_window(s, OOS_A, LAST)
        out[f"yearly_{lbl}"] = yearly(s, a=str(s.index[0].date()))
    return out

VALID = {}
for pk in ["main_12m", "blend_1_3_12m", "ex_equities", "faithful_div_gsg", "gsg_commodity_2006plus"]:
    if pk in PROX:
        VALID[pk] = {c: validate(PROX[pk]["ret"], c) for c in ["dbmf", "kmlm", "cta"]}

# ================================================================== 3. F + TSMOM portfolios
log("baselines ...")
BASE = {}
for e, (lev, vt) in ENGINES.items():
    BASE[e] = evaluate(f"{e}+GLD+IEF (baseline)", F(lev, vt), tag="baseline")
    log(f"  {e} baseline IS:", C.fmt(BASE[e]["d"]["IS"]), " 2022", round(BASE[e]["d"]["crises"]["2022 inflation"]*100, 1))

# make every proxy available as a sleeve column
for k, v in PROX.items():
    P[f"ts_{k}"] = v["ret"]

log("main structures (main + blend proxies, with bootstrap) ...")
PORT = {}
for pk in ["main_12m", "blend_1_3_12m"]:
    col = f"ts_{pk}"
    for e, (lev, vt) in ENGINES.items():
        for st in STRUCTS:
            nm = f"{e} | {st} | {pk}"
            PORT[nm] = evaluate(nm, structure_weights(lev, vt, col, st), base=BASE[e], boot=True,
                                tag="main_grid", col=col)
    log(f"  done {pk}")

# sanity: my paired bootstrap reproduces common.block_bootstrap_cagr_diff
_k0 = next(iter(PORT))
_off = C.block_bootstrap_cagr_diff(PORT[_k0]["r"][PORT[_k0]["r"].index <= FREEZE],
                                    BASE[_k0.split(" | ")[0]]["r"][BASE[_k0.split(" | ")[0]]["r"].index <= FREEZE])
BOOT_CHECK = dict(common_mean=_off["mean"], mine_mean=PORT[_k0]["d"]["boot"]["d_cagr"]["mean"],
                  common_lo=_off["lo"], mine_lo=PORT[_k0]["d"]["boot"]["d_cagr"]["lo"])
log("bootstrap check:", BOOT_CHECK)

log("de-lever control: same-size T-bill sleeve, and TSMOM sleeve vs T-bill sleeve ...")
CASH = {}
VS_CASH = {}
for e, (lev, vt) in ENGINES.items():
    for s in (10, 20, 30):
        nm = f"{e} | sleeve{s} | CONTROL bills (plain de-lever)"
        CASH[nm] = evaluate(nm, sleeve(F(lev, vt), "bills", s / 100), base=BASE[e], boot=True, tag="control_cash")
        for pk in ["main_12m", "blend_1_3_12m"]:
            tk = f"{e} | sleeve{s} | {pk}"
            bt = paired_boot(PORT[tk]["r"], CASH[nm]["r"])
            dt, dc = PORT[tk]["d"], CASH[nm]["d"]
            VS_CASH[tk] = dict(
                d_calmar_IS=dt["IS"]["calmar"] - dc["IS"]["calmar"], d_calmar_H1=dt["H1"]["calmar"] - dc["H1"]["calmar"],
                d_calmar_H2=dt["H2"]["calmar"] - dc["H2"]["calmar"], d_cagr_IS=dt["IS"]["cagr"] - dc["IS"]["cagr"],
                d_sharpe_excess_IS=dt["IS"]["sharpe_excess"] - dc["IS"]["sharpe_excess"],
                d_maxdd_IS=dt["IS"]["maxdd"] - dc["IS"]["maxdd"], d_2022=dt["crises"]["2022 inflation"] - dc["crises"]["2022 inflation"],
                d_OOS=dt["OOS_total"] - dc["OOS_total"], boot=bt)

log("robustness grid (all structures x sensitivity proxies, no bootstrap) ...")
ROB = {}
for pk in PROX:
    if pk in ("main_12m", "blend_1_3_12m"):
        continue
    col = f"ts_{pk}"
    for e, (lev, vt) in ENGINES.items():
        for st in STRUCTS:
            nm = f"{e} | {st} | {pk}"
            ROB[nm] = evaluate(nm, structure_weights(lev, vt, col, st), base=BASE[e], boot=False,
                               tag="robustness_grid", col=col)
log(f"  {len(ROB)} robustness portfolios")
# bootstrap for the two economically-motivated alternatives (pre-specified rows, reported whatever they show)
for pk in ["ex_equities", "faithful_div_gsg"]:
    if pk not in PROX:
        continue
    for e in ENGINES:
        for st in ["sleeve20", "replace_ief_offleg"]:
            k = f"{e} | {st} | {pk}"
            ROB[k]["d"]["boot"] = paired_boot(ROB[k]["r"], BASE[e]["r"])
log("  extra bootstraps done")

# robustness summary per (engine, structure)
ROBSUM = {}
for e in ENGINES:
    for st in STRUCTS:
        rows = [v["d"] for k, v in {**PORT, **ROB}.items()
                if k.startswith(f"{e} | {st} | ") and not k.endswith("control_always_long")]
        dIS = np.array([r["delta_vs_base"]["IS"]["calmar"] for r in rows])
        dH1 = np.array([r["delta_vs_base"]["H1"]["calmar"] for r in rows])
        dH2 = np.array([r["delta_vs_base"]["H2"]["calmar"] for r in rows])
        dC = np.array([r["delta_vs_base"]["IS"]["cagr"] for r in rows])
        d22 = np.array([r["delta_vs_base"]["y2022"] for r in rows])
        ROBSUM[f"{e} | {st}"] = dict(n_proxies=len(rows),
            d_calmar_IS_median=float(np.median(dIS)), d_calmar_IS_min=float(dIS.min()), d_calmar_IS_max=float(dIS.max()),
            share_d_calmar_IS_gt_0=float((dIS > 0).mean()), share_gt_sel_cost_hi=float((dIS > SEL_COST_HI).mean()),
            share_both_halves_gt_0=float(((dH1 > 0) & (dH2 > 0)).mean()),
            d_cagr_IS_median=float(np.median(dC)), d_2022_median=float(np.median(d22)))

# control (always-long basket) vs TSMOM main, same structures
CONTROL = {}
for e in ENGINES:
    for st in STRUCTS:
        c = ROB[f"{e} | {st} | control_always_long"]["d"]
        t = PORT[f"{e} | {st} | main_12m"]["d"]
        CONTROL[f"{e} | {st}"] = dict(
            control_d_calmar_IS=c["delta_vs_base"]["IS"]["calmar"], tsmom_d_calmar_IS=t["delta_vs_base"]["IS"]["calmar"],
            control_d_calmar_H1=c["delta_vs_base"]["H1"]["calmar"], control_d_calmar_H2=c["delta_vs_base"]["H2"]["calmar"],
            control_d_cagr_IS=c["delta_vs_base"]["IS"]["cagr"], tsmom_d_cagr_IS=t["delta_vs_base"]["IS"]["cagr"],
            control_y2022=c["crises"]["2022 inflation"], tsmom_y2022=t["crises"]["2022 inflation"],
            control_OOS=c["OOS_total"], tsmom_OOS=t["OOS_total"])

# ------------------------------------------------------------------ real-ETF reality check (2019-05+)
log("real DBMF/KMLM sleeve check on their live windows ...")
REAL = {}
for etf, a in (("dbmf", "2019-05-13"), ("kmlm", "2020-12-07")):
    for e, (lev, vt) in ENGINES.items():
        rb = BASE[e]["r"]
        rowb = dict(IS=C.stats(rb, a, FREEZE, start=None), OOS_total=C.period_return(rb, OOS_A, LAST),
                    y2022=C.period_return(rb, "2022-01-01", "2022-12-31"))
        REAL[f"{e} baseline [{etf} window]"] = rowb
        for st in ["sleeve20", "replace_ief_offleg", "offleg_50ief_50tsmom"]:
            for src, col in ((etf.upper() + " real", etf), ("proxy main_12m", "ts_main_12m")):
                w = structure_weights(lev, vt, col, st)
                r = C.backtest(P, w)
                nm = f"{e} | {st} | {src} [{etf} window]"
                REAL[nm] = dict(IS=C.stats(r, a, FREEZE, start=None), OOS_total=C.period_return(r, OOS_A, LAST),
                                y2022=C.period_return(r, "2022-01-01", "2022-12-31"))
                REAL[nm]["d_calmar_vs_base"] = REAL[nm]["IS"]["calmar"] - rowb["IS"]["calmar"]
                REAL[nm]["d_cagr_vs_base"] = REAL[nm]["IS"]["cagr"] - rowb["IS"]["cagr"]
                if src.endswith("real"):
                    reg("portfolio", nm, tag="real_etf_window_check")

# ================================================================== 4. diagnostics
log("diagnostics ...")
m = (idx >= START) & (idx <= FREEZE)
DIAG = {}
for pk in ["main_12m", "blend_1_3_12m"]:
    r = PROX[pk]["ret"]
    dd = {}
    for e in ENGINES:
        dd[f"corr_daily_vs_{e}_base"] = float(r[m].corr(BASE[e]["r"][m]))
        dd[f"corr_monthly_vs_{e}_base"] = float(mo(r[m]).corr(mo(BASE[e]["r"][m])))
    dd["corr_daily_vs_ndx"] = float(r[m].corr(P["ndx_ret"][m]))
    dd["corr_daily_vs_ief"] = float(r[m].corr(P["ief"][m]))
    dd["corr_monthly_vs_ief"] = float(mo(r[m]).corr(mo(P["ief"][m])))
    # regime: F risk-off days (held state), annualised excess return of proxy vs IEF
    ron = P["above"].astype(float).where(P["mark"]).ffill().shift(1).fillna(0.0) > 0.5
    for lbl, mask in (("F_risk_on", ron & m), ("F_risk_off", (~ron) & m)):
        ex_p = (r - P["rf_d"])[mask]; ex_i = (P["ief"] - P["rf_d"])[mask]
        dd[f"{lbl}_days"] = int(mask.sum())
        dd[f"{lbl}_proxy_excess_ann"] = float(ex_p.mean() * 252)
        dd[f"{lbl}_ief_excess_ann"] = float(ex_i.mean() * 252)
    # stocks-and-bonds-down months (SPX<0 and IEF<0 in the same month), IS
    Mx = pd.DataFrame({"spx": mo(P["spx"][m]), "ief": mo(P["ief"][m]), "proxy": mo(r[m]), "rf": mo(P["rf_d"][m])})
    sb = Mx[(Mx.spx < 0) & (Mx.ief < 0)]
    dd["stocks_bonds_down_months"] = int(len(sb))
    dd["sbd_avg_month_proxy"] = float(sb.proxy.mean())
    dd["sbd_avg_month_ief"] = float(sb.ief.mean())
    dd["sbd_hit_rate_proxy_gt_rf"] = float((sb.proxy > sb.rf).mean())
    eqd = Mx[Mx.spx < -0.05]
    dd["spx_down5pct_months"] = int(len(eqd))
    dd["spx_down5_avg_month_proxy"] = float(eqd.proxy.mean())
    dd["spx_down5_avg_month_ief"] = float(eqd.ief.mean())
    # per-market P&L attribution (gross of costs), IS / 2022 / OOS
    pr = PROX[pk]
    contrib = pr["held"] * pr["X"]
    att = {}
    for mk in pr["mkts"]:
        c = contrib[mk]
        cm = c[m]
        att[mk] = dict(ann_contrib_IS=float(cm.mean() * 252),
                       leg_sharpe_IS=float(cm.mean() / cm.std() * np.sqrt(252)) if cm.std() > 0 else None,
                       H1_ann=float(c[(idx >= pd.Timestamp(H1[0])) & (idx <= pd.Timestamp(H1[1]))].mean() * 252),
                       H2_ann=float(c[(idx >= pd.Timestamp(H2[0])) & (idx <= pd.Timestamp(H2[1]))].mean() * 252),
                       y2022_sum=float(c[(idx >= "2022-01-01") & (idx <= "2022-12-31")].sum()),
                       OOS_sum=float(c[idx >= OOS_A].sum()),
                       share_long_IS=float((pr["held"][mk][m] > 0).mean()))
    dd["attribution_gross"] = att
    dd["costs_ann_IS"] = float((pr["turn"][m] * C.COST).mean() * 252 + pr["spec"]["fee"])
    DIAG[pk] = dd

# drawdown anatomy: deepest IS episodes of baselines and main-proxy structures
DDX = {}
for e in ENGINES:
    DDX[f"{e} baseline"] = top_drawdowns(BASE[e]["r"])
    for st in STRUCTS:
        k = f"{e} | {st} | main_12m"
        DDX[k] = top_drawdowns(PORT[k]["r"])
DDX["proxy main_12m"] = top_drawdowns(PROX["main_12m"]["ret"])
DDX["proxy control_always_long"] = top_drawdowns(PROX["control_always_long"]["ret"])

# real ETF vs proxy: correlation with the F baselines on the ETF's live window (weekly)
LIVECORR = {}
for etf, a in (("dbmf", "2019-05-13"), ("kmlm", "2020-12-07")):
    mm = (idx >= pd.Timestamp(a)) & (idx <= LAST)
    for e in ENGINES:
        b = wk(BASE[e]["r"][mm])
        LIVECORR[f"{etf}_vs_{e}_weekly"] = float(wk(P[etf][mm].fillna(0)).corr(b))
        LIVECORR[f"proxy_main_vs_{e}_weekly_same_window"] = float(wk(PROX["main_12m"]["ret"][mm]).corr(b))
    LIVECORR[f"{etf}_vs_ief_weekly"] = float(wk(P[etf][mm].fillna(0)).corr(wk(P["ief"][mm])))
    LIVECORR[f"proxy_main_vs_ief_weekly_{etf}_window"] = float(wk(PROX["main_12m"]["ret"][mm]).corr(wk(P["ief"][mm])))

# ================================================================== 5. positions today
def positions(pr):
    X = pr["X"]; cum = (1 + X).cumprod()
    last_rb = pr["POS"].index[-1]
    out = dict(last_rebalance=str(last_rb.date()), as_of=str(LAST.date()),
               scale_k_at_last_rebalance=float(pr["K"].iloc[-1]), scale_k_live=pr["K_live"], markets={})
    for mk in pr["mkts"]:
        L = max(pr["lbs"])
        r12_rb = float(cum[mk].loc[last_rb] / cum[mk].shift(252).loc[last_rb] - 1)
        r12_now = float(cum[mk].iloc[-1] / cum[mk].shift(252).iloc[-1] - 1)
        mom = {f"{l}d_excess_ret_now": float(cum[mk].iloc[-1] / cum[mk].shift(l).iloc[-1] - 1) for l in pr["lbs"]}
        out["markets"][mk] = dict(
            held_signal=float(pr["S"].loc[last_rb, mk]), held_notional=float(pr["POS"].loc[last_rb, mk]),
            held_risk_vol=float(abs(pr["POS"].loc[last_rb, mk]) * pr["V"].loc[last_rb, mk]),
            live_signal=float(pr["S"].iloc[-1][mk]), live_notional=float(pr["POS_live"][mk]),
            r12m_excess_at_last_rebalance=r12_rb, r12m_excess_now=r12_now, vol63_now=float(pr["V"].iloc[-1][mk]), **mom)
    out["held_gross_notional"] = float(pr["POS"].iloc[-1].abs().sum())
    out["live_gross_notional"] = float(pr["POS_live"].abs().sum())
    return out

POSN = {pk: positions(PROX[pk]) for pk in ["main_12m", "blend_1_3_12m", "ex_equities"]}

# ================================================================== 6. output
def pick(d):
    return {k: d[k] for k in d if k not in ("name",)}

summary_rows = []
for nm, v in {**{k: BASE[k] for k in BASE}, **PORT}.items():
    d = v["d"]
    summary_rows.append(dict(name=d["name"], IS=d["IS"], H1=d["H1"], H2=d["H2"], OOS_total=d["OOS_total"],
                             y2022=d["crises"]["2022 inflation"], verdict=d.get("verdict"),
                             d_calmar_IS=d.get("delta_vs_base", {}).get("IS", {}).get("calmar"),
                             boot_d_calmar=d.get("boot", {}).get("d_calmar"), boot_d_cagr=d.get("boot", {}).get("d_cagr"),
                             avg_weight_tsmom=d.get("avg_weight_tsmom")))

n_proxy = sum(1 for x in REG if x["kind"] == "proxy")
n_port = sum(1 for x in REG if x["kind"] == "portfolio")
OUT = dict(
    meta=dict(script="update_2026_11/managed_futures.py", run_date=str(pd.Timestamp.today().date()),
              panel_last=str(LAST.date()), in_sample=f"{START.date()}..{FREEZE.date()}",
              oos=f"{OOS_A.date()}..{LAST.date()}", halves=[H1, H2], gold_smoothing_correction=SMOOTH,
              gold_signal_switch=str(GOLD_SIG_OK.date()), gold_daily_vol_from=str(GOLD_DAILY_OK.date()),
              gsg_available=GSG is not None, selection_cost_calmar=[SEL_COST_LO, SEL_COST_HI],
              verdict_rule=("PASS = dCalmar(IS) > 0.15 AND dCalmar > 0 in both halves AND paired block-bootstrap "
                            "95% CI of dCalmar excludes 0; MARGINAL = dCalmar > 0.07 and both halves > 0"),
              variants_tested=dict(total=len(REG), proxies=n_proxy, portfolios=n_port,
                                   main_grid=sum(1 for x in REG if x.get("tag") == "main_grid"),
                                   robustness_grid=sum(1 for x in REG if x.get("tag") == "robustness_grid"),
                                   baselines=sum(1 for x in REG if x.get("tag") == "baseline"),
                                   real_etf_window_checks=sum(1 for x in REG if x.get("tag") == "real_etf_window_check"),
                                   control_cash=sum(1 for x in REG if x.get("tag") == "control_cash"),
                                   of_which_control_always_long=sum(1 for x in REG if x["name"].endswith("control_always_long"))),
              bootstrap_check=BOOT_CHECK),
    proxies=PREP, proxy_yearly=YEARLY, validation=VALID,
    baselines={k: pick(v["d"]) for k, v in BASE.items()},
    main_grid={k: pick(v["d"]) for k, v in PORT.items()},
    robustness_summary=ROBSUM, control_always_long_vs_tsmom=CONTROL,
    drawdown_anatomy=DDX, live_window_correlations=LIVECORR,
    robustness_grid={k: dict(IS=v["d"]["IS"], H1=v["d"]["H1"], H2=v["d"]["H2"], OOS_total=v["d"]["OOS_total"],
                             y2022=v["d"]["crises"]["2022 inflation"], delta_vs_base=v["d"]["delta_vs_base"],
                             **({"boot": v["d"]["boot"]} if "boot" in v["d"] else {}))
                     for k, v in ROB.items()},
    control_cash_sleeves={k: pick(v["d"]) for k, v in CASH.items()},
    tsmom_sleeve_vs_cash_sleeve=VS_CASH,
    real_etf_window_check=REAL, diagnostics=DIAG, positions_today=POSN, summary=summary_rows,
    registry=[dict(kind=x["kind"], name=x["name"], tag=x.get("tag")) for x in REG])

with open(os.path.join(HERE, "managed_futures.json"), "w") as f:
    json.dump(rnd(OUT, 5), f, indent=1, default=str)
log(f"saved managed_futures.json   variants: {len(REG)} (proxies {n_proxy}, portfolios {n_port})")

# ------------------------------------------------------------------ console summary
def pct(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{x*100:6.1f}%"
print("\n=== PROXIES (in-sample 1988..2026-07-24) ===")
print(f"{'proxy':26s}{'CAGR':>8}{'exCAGR':>8}{'vol':>7}{'exShrp':>7}{'maxDD':>8}{'H1 exS':>7}{'H2 exS':>7}{'2022':>8}{'OOS':>8}{'corrM':>6}")
for k, d in PREP.items():
    print(f"{k:26s}{pct(d['IS']['cagr'])}{pct(d['IS']['excess_cagr'])}{pct(d['IS']['vol'])[1:]}"
          f"{d['IS']['sharpe_excess']:7.2f}{pct(d['IS']['maxdd'])}{d['H1']['sharpe_excess']:7.2f}{d['H2']['sharpe_excess']:7.2f}"
          f"{pct(d['y2022'])}{pct(d['OOS_total'])}{d['corr_with_main_daily_IS']:6.2f}")
print("\n=== VALIDATION vs real ETFs (main_12m / blend) ===")
for pk, vv in VALID.items():
    for c, v in vv.items():
        print(f"{pk:24s}{c:5s} {v['start']} corr d/w/m {v['corr_daily']:.2f}/{v['corr_weekly']:.2f}/{v['corr_monthly']:.2f} "
              f"beta(etf~proxy) {v['beta_etf_on_proxy_weekly']:.2f}  CAGR proxy {pct(v['cagr_proxy'])} etf {pct(v['cagr_etf'])}  "
              f"2022 proxy {pct(v['y2022_proxy']) if v['y2022_proxy'] is not None else '   n/a'} etf {pct(v['y2022_etf']) if v['y2022_etf'] is not None else '   n/a'}  "
              f"OOS proxy {pct(v['OOS_total_proxy'])} etf {pct(v['OOS_total_etf'])}")
print("\n=== F + TSMOM (in-sample; deltas vs same-engine baseline) ===")
print(f"{'portfolio':52s}{'CAGR':>7}{'maxDD':>7}{'Calm':>6}{'Shrp':>6}{'dCal':>6}{'dCalH1':>7}{'dCalH2':>7}{'bootCal lo..hi':>16}{'2022':>7}{'OOS':>7}  verdict")
for k in BASE:
    d = BASE[k]["d"]
    print(f"{d['name']:52s}{pct(d['IS']['cagr'])}{pct(d['IS']['maxdd'])}{d['IS']['calmar']:6.2f}{d['IS']['sharpe']:6.2f}"
          f"{'':6s}{'':7s}{'':7s}{'':16s}{pct(d['crises']['2022 inflation'])}{pct(d['OOS_total'])}")
for k, v in PORT.items():
    d = v["d"]; dc = d["delta_vs_base"]; b = d["boot"]["d_calmar"]
    print(f"{k:52s}{pct(d['IS']['cagr'])}{pct(d['IS']['maxdd'])}{d['IS']['calmar']:6.2f}{d['IS']['sharpe']:6.2f}"
          f"{dc['IS']['calmar']:6.2f}{dc['H1']['calmar']:7.2f}{dc['H2']['calmar']:7.2f}   {b['lo']:6.2f}..{b['hi']:5.2f}"
          f"{pct(d['crises']['2022 inflation'])}{pct(d['OOS_total'])}  {d['verdict'][:8]}")
print("\n=== ROBUSTNESS SUMMARY (all proxy variants) ===")
for k, v in ROBSUM.items():
    print(f"{k:34s} n={v['n_proxies']:2d}  dCalmar IS median {v['d_calmar_IS_median']:+.2f} [{v['d_calmar_IS_min']:+.2f},{v['d_calmar_IS_max']:+.2f}]"
          f"  >0: {v['share_d_calmar_IS_gt_0']*100:4.0f}%  >0.15: {v['share_gt_sel_cost_hi']*100:4.0f}%  both halves>0: {v['share_both_halves_gt_0']*100:4.0f}%"
          f"  dCAGR med {v['d_cagr_IS_median']*100:+.2f}%  d2022 med {v['d_2022_median']*100:+.1f}%")
print("\n=== DE-LEVER CONTROL: T-bill sleeve vs baseline, and TSMOM sleeve minus same-size T-bill sleeve ===")
for k, v in CASH.items():
    d = v["d"]; dc = d["delta_vs_base"]
    print(f"{k:52s} CAGR {pct(d['IS']['cagr'])} maxDD {pct(d['IS']['maxdd'])} Calmar {d['IS']['calmar']:.2f} "
          f"Shrp {d['IS']['sharpe']:.2f} exShrp {d['IS']['sharpe_excess']:.2f}  dCal {dc['IS']['calmar']:+.2f}  2022 {pct(d['crises']['2022 inflation'])}")
for k, v in VS_CASH.items():
    b = v["boot"]
    print(f"{k:40s} vs cash: dCal {v['d_calmar_IS']:+.2f} (H1 {v['d_calmar_H1']:+.2f} H2 {v['d_calmar_H2']:+.2f}) boot [{b['d_calmar']['lo']:+.2f},{b['d_calmar']['hi']:+.2f}]"
          f"  dCAGR {v['d_cagr_IS']*100:+.2f}% [{b['d_cagr']['lo']*100:+.2f},{b['d_cagr']['hi']*100:+.2f}]  dExShrp {v['d_sharpe_excess_IS']:+.3f}"
          f"  dShrp boot [{b['d_sharpe']['lo']:+.3f},{b['d_sharpe']['hi']:+.3f}]  d2022 {v['d_2022']*100:+.1f}  dOOS {v['d_OOS']*100:+.1f}")
print("\n=== extra bootstraps (ex_equities / faithful_div_gsg) ===")
for k, v in ROB.items():
    if "boot" in v["d"]:
        d = v["d"]; dc = d["delta_vs_base"]; b = d["boot"]
        print(f"{k:48s} dCal {dc['IS']['calmar']:+.2f} (H1 {dc['H1']['calmar']:+.2f} H2 {dc['H2']['calmar']:+.2f}) boot [{b['d_calmar']['lo']:+.2f},{b['d_calmar']['hi']:+.2f}] "
              f"dCAGR {dc['IS']['cagr']*100:+.2f}%  dShrp boot [{b['d_sharpe']['lo']:+.3f},{b['d_sharpe']['hi']:+.3f}]  2022 {pct(d['crises']['2022 inflation'])}  OOS {pct(d['OOS_total'])}")
print("\n=== CONTROL: always-long basket vs TSMOM (dCalmar IS vs baseline) ===")
for k, v in CONTROL.items():
    print(f"{k:34s} control {v['control_d_calmar_IS']:+.2f} (H1 {v['control_d_calmar_H1']:+.2f} H2 {v['control_d_calmar_H2']:+.2f})"
          f"  tsmom {v['tsmom_d_calmar_IS']:+.2f}   2022 control {pct(v['control_y2022'])} tsmom {pct(v['tsmom_y2022'])}")
print("\n=== DRAWDOWN ANATOMY (deepest IS episodes) ===")
for k, v in DDX.items():
    print(f"{k:44s} " + " | ".join(f"{x['depth']*100:6.1f}% {x['peak']}->{x['trough']}" for x in v))
print("\n=== ATTRIBUTION main_12m (gross, ann. IS) ===")
for mk, v in DIAG["main_12m"]["attribution_gross"].items():
    print(f"  {mk:5s} {v['ann_contrib_IS']*100:+5.2f}%/yr  legSharpe {v['leg_sharpe_IS']:+.2f}  H1 {v['H1_ann']*100:+5.2f}  H2 {v['H2_ann']*100:+5.2f}"
          f"  2022 {v['y2022_sum']*100:+5.1f}  OOS {v['OOS_sum']*100:+5.1f}  long {v['share_long_IS']*100:3.0f}%")
print("costs ann IS", round(DIAG["main_12m"]["costs_ann_IS"] * 100, 2), "%")
print("\n=== LIVE-WINDOW CORRELATIONS ===", {k: round(v, 2) for k, v in LIVECORR.items()})
print("\n=== POSITIONS (main_12m) ===")
pp = POSN["main_12m"]
print(f"held since {pp['last_rebalance']} (k={pp['scale_k_at_last_rebalance']:.2f}); live as of {pp['as_of']} (k={pp['scale_k_live']:.2f})")
for mk, v in pp["markets"].items():
    print(f"  {mk:5s} held {v['held_signal']:+.0f} notional {v['held_notional']*100:+6.1f}%   live {v['live_signal']:+.0f} "
          f"notional {v['live_notional']*100:+6.1f}%   12m xs now {v['r12m_excess_now']*100:+6.1f}%  (at rebal {v['r12m_excess_at_last_rebalance']*100:+6.1f}%)")
log("done")
