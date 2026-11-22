"""
update_2026_11/leverage_execution.py  --  LEVERAGE CHOICE AND EXECUTION REALISM  (Nov-2026 refresh)

Questions answered (all mechanics from common.py unless stated; physical gold 15%, IEF off-leg):
  1  LEVERAGE x VOL-TARGET GRID   lev {qqq 1x, qld 2x, tqqq 3x} x voltgt {.15,.20,.25,.30,.35,.45}
       - CAGR / maxDD / Calmar / Sharpe / vol for IS, H1, H2 (+ OOS and FULL for reference)
       - Pareto frontier of the 18 points (CAGR vs |maxDD| and CAGR vs vol)
       - MATCHED-RISK comparison on a fine voltgt grid (frontier mapping, 0.05..0.80 step 0.01):
         CAGR of each leverage interpolated at equal max drawdown and at equal volatility,
         separately in IS, H1 and H2
       - exposure-matched instrument test: TQQQ at 2/3 weight (vt .30) has EXACTLY F2's NDX exposure
         -> splits "F3 vs F2" into an exposure-cap effect and an instrument (fees / bond carry) effect
       - paired circular block bootstrap (63d) of CAGR and maxDD; bootstrap-matched drawdown curves
       - gap / tail risk at the exposure cap (instantaneous NDX shocks), worst day / week, 1987 crash
  2  EXECUTION LAG   lag 1 vs 2, Friday marks vs last-trading-day-of-week marks, and a
                     "signal one day stale" bound, for F2-IEF and F3-IEF
  3  NO-TRADE REBALANCE BAND  own drift-faithful holdings loop (holdings drift between rebalances,
                     unlike common.backtest which holds target weights constant every day):
                     band 0 / 2 / 5 / 10 percentage points (absolute), pure and with a regime-change
                     override; turnover, orders, cost at 0 / 3 / 10 / 20 bps per unit traded notional
  4  REAL-ETF REALITY CHECK  2010-03-01..latest: spliced (real ETFs) vs pure-simulated sleeves
                     (qld_sim / tqqq_sim, ief_syn / tlt_syn) for F2 / F3 x IEF / TLT, with
                     sleeve attribution and calendar-year tracking (2020, 2022, 2025, 2026)
  5  RECOMMENDATION INPUTS (current state, chosen configuration evidence)

Windows (stats always start 1988-01-01):
    IS   = 1988-01-01 .. 2026-07-24   (July freeze; the ONLY window used to choose anything)
    H1   = 1988-01-01 .. 2006-12-31
    H2   = 2007-01-01 .. 2026-07-24
    OOS  = 2026-07-25 .. 2026-11-17   (genuinely out-of-sample, reported only)
    FULL = 1988-01-01 .. 2026-11-17
Turnover convention (same as common.backtest): sum over sleeves of |traded notional| / NAV, i.e. a
switch of 10% from IEF to TQQQ is 0.20 turnover; "N bps" = N bps of NAV per unit of that turnover.

Output: update_2026_11/leverage_execution.json    (run: python -u update_2026_11/leverage_execution.py)
"""
import sys, os, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                    # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


P = C.panel()
IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
OOS_A = pd.Timestamp("2026-07-25")
WIN = {"IS": (None, FREEZE), "H1": ("1988-01-01", "2006-12-31"),
       "H2": ("2007-01-01", FREEZE), "FULL": (None, LATEST)}
MULT = {"qqq": 1, "qld": 2, "tqqq": 3, "qqq_sim": 1, "qld_sim": 2, "tqqq_sim": 3}
LEVS = ["qqq", "qld", "tqqq"]
LEVNAME = {"qqq": "1x", "qld": "2x", "tqqq": "3x"}
VTS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
FINE = [round(float(x), 2) for x in np.arange(0.05, 0.8001, 0.01)]
BOOT_VTS = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60, 0.80]
FINE_T = [round(float(x), 3) for x in np.arange(0.05, 0.8001, 0.025)]     # tranche frontier (coarser)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MK_DOW = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(DAYS)}   # Fri == P.mark
DD_TARGETS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
VOL_TARGETS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]
OUT = os.path.join(HERE, "leverage_execution.json")
RES = {"meta": dict(run_date="2026-11-18", panel_end=str(LATEST.date()), freeze=str(FREEZE.date()),
                    windows={k: [str(a) if a else "1988-01-01", str(pd.Timestamp(b).date())] for k, (a, b) in WIN.items()},
                    oos=[str(OOS_A.date()), str(LATEST.date())],
                    turnover_convention="sum over sleeves |traded notional|/NAV (both legs); cost = bps x turnover")}

# ------------------------------------------------------------------ variant registry
VARIANTS = {}


def count(section, **cfg):
    c = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cfg.items()}
    key = json.dumps(c, sort_keys=True, default=str)
    VARIANTS.setdefault(key, []).append(section)


# ------------------------------------------------------------------ helpers
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def seg(r, a=None, b=None, start=START):
    s = r
    if start is not None:
        s = s[s.index >= start]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    return s.fillna(0)


def ann_vol(r, a=None, b=None, start=START):
    return float(seg(r, a, b, start).std() * np.sqrt(252))


def oos(r):
    s = r[r.index >= OOS_A].fillna(0)
    e = np.r_[1.0, (1 + s).cumprod().values]
    return dict(total=float(e[-1] - 1), maxdd=float((e / np.maximum.accumulate(e) - 1).min()), days=int(len(s)))


def wstats(r):
    d = {}
    for k, (a, b) in WIN.items():
        s = C.stats(r, a, b)
        s["vol"] = ann_vol(r, a, b)
        d[k] = s
    d["OOS"] = oos(r)
    return d


def weights(lev, vt, off="ief", **kw):
    return C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off=off, **kw)


def ndx_exposure(w, lag=1, mark="mark"):
    H = C.held_weights(P, w, lag=lag, mark=mark)
    return sum(H[c] * MULT[c] for c in H.columns if c in MULT)


def expo_stats(w, lev):
    e = ndx_exposure(w)
    e = e[(e.index >= START) & (e.index <= FREEZE)]
    on = e > 1e-9
    cap = 0.85 * MULT[lev]
    return dict(cap=cap, max=float(e.max()), avg_all_days=float(e.mean()), avg_riskon_days=float(e[on].mean()),
                share_days_riskon=float(on.mean()), share_riskon_days_at_cap=float((e[on] >= cap * 0.999).mean()))


def gap_losses(expo, lev):
    """instantaneous one-day NDX shock with no chance to trade; bonds/gold assumed flat."""
    m = MULT[lev]
    out = {}
    for s in (-0.10, -0.15, -0.20, -0.25):
        sleeve = max(-1.0, m * s)
        out[f"ndx{int(s*100)}%"] = dict(at_cap=0.85 * sleeve, at_avg_riskon=(expo["avg_riskon_days"] / m) * sleeve)
    return out


def tail(r):
    s = seg(r, None, FREEZE)
    lg = np.log1p(s)
    w5 = np.expm1(lg.rolling(5).sum())
    w21 = np.expm1(lg.rolling(21).sum())
    return dict(worst_day=float(s.min()), worst_day_date=str(s.idxmin().date()),
                worst_5d=float(w5.min()), worst_5d_end=str(w5.idxmin().date()),
                worst_21d=float(w21.min()), worst_21d_end=str(w21.idxmin().date()))


def turnover_cm(w, lag=1, mark="mark", a=START, b=FREEZE):
    H = C.held_weights(P, w, lag=lag, mark=mark)
    t = H.diff().abs().sum(axis=1).fillna(0)
    t = t[(t.index >= a) & (t.index <= b)]
    yrs = (t.index[-1] - t.index[0]).days / 365.25
    return float(t.sum() / yrs)


def pareto(points):
    """points: list of (name, cagr, risk>0). Non-dominated = no other point with >= cagr and <= risk."""
    out = []
    for n, c, rk in points:
        dom = any((c2 >= c and rk2 <= rk) and (c2 > c or rk2 < rk) for n2, c2, rk2 in points if n2 != n)
        if not dom:
            out.append(n)
    return out


def interp_at(risk, val, vts, target):
    """first crossing of `target` by `risk` (risk rises with vt); linear interpolation of val and vt."""
    risk = np.asarray(risk, float); val = np.asarray(val, float); vts = np.asarray(vts, float)
    hit = np.where(risk >= target)[0]
    if len(hit) == 0:
        return dict(cagr=None, voltgt=None, status=f"unreachable (max {risk.max():.3f})")
    i = hit[0]
    if i == 0:
        return dict(cagr=None, voltgt=None, status=f"below grid min ({risk[0]:.3f})")
    x0, x1 = risk[i - 1], risk[i]
    f = (target - x0) / (x1 - x0) if x1 > x0 else 0.0
    return dict(cagr=float(val[i - 1] + f * (val[i] - val[i - 1])),
                voltgt=float(vts[i - 1] + f * (vts[i] - vts[i - 1])), status="ok")


def paired_boot(series, a=START, b=FREEZE, n=1000, block=63, seed=11):
    """paired circular block bootstrap: same block draws for every series -> CAGR and maxDD distributions."""
    names = list(series)
    M = pd.concat([series[k] for k in names], axis=1)
    M = M[(M.index >= a) & (M.index <= b)].fillna(0).values
    T, k = M.shape
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    cg = np.empty((n, k)); dd = np.empty((n, k))
    ones = np.ones((1, k))
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = ((st[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        e = np.vstack([ones, np.cumprod(1 + M[ix], axis=0)])
        cg[i] = e[-1] ** (252.0 / T) - 1
        dd[i] = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
    return names, cg, dd


def boot_summary(cg, dd):
    return dict(cagr_med=float(np.median(cg)), cagr_p5=float(np.percentile(cg, 5)),
                cagr_p95=float(np.percentile(cg, 95)),
                maxdd_med=float(np.median(dd)), maxdd_p5=float(np.percentile(dd, 5)),
                maxdd_p1=float(np.percentile(dd, 1)),
                p_dd_worse_40=float((dd <= -0.40).mean()), p_dd_worse_50=float((dd <= -0.50).mean()))


def dd_episode(r, a=None, b=FREEZE):
    s = seg(r, a, b)
    e = (1 + s).cumprod()
    dd = e / e.cummax() - 1
    t = dd.idxmin()
    pk = e.loc[:t].idxmax()
    return dict(maxdd=float(dd.min()), peak=str(pk.date()), trough=str(t.date()))


def tranche_run(w, lag=1, cost=C.COST):
    """5 equal tranches, each rebalanced weekly on a different weekday (Mon..Fri) -> removes rebalance-timing
    luck. Constant-mix convention as common.backtest; trades between tranches are netted."""
    H = sum(C.held_weights(P, w, lag=lag, mark=MK_DOW[d]) for d in DAYS) / 5.0
    gross = sum(H[c] * P[c].fillna(0) for c in H.columns)
    return gross - H.diff().abs().sum(axis=1).fillna(0) * cost


SER, WTS = {}, {}


def base_run(lev, vt, section):
    """the canonical grid backtest (constant-mix engine, lag 1, Friday marks, 3 bps)."""
    k = (lev, round(float(vt), 2))
    count(section, kind="strategyF", lev=lev, voltgt=k[1], off="ief", gold="gold_phys", lag=1, mark="mark",
          cost_bps=3, engine="constant_mix", data="spliced")
    if k not in SER:
        w = weights(lev, k[1])
        WTS[k] = w
        SER[k] = C.backtest(P, w)
    return SER[k]


# ================================================================== SECTION 1: GRID
log("SECTION 1  leverage x vol-target grid")
grid = {}
for lev in LEVS:
    for vt in VTS:
        r = base_run(lev, vt, "1a_grid")
        w = WTS[(lev, vt)]
        ex = expo_stats(w, lev)
        name = f"{lev}_vt{vt:.2f}"
        grid[name] = dict(lev=lev, leverage=LEVNAME[lev], voltgt=vt, stats=wstats(r), crises=C.crises(r),
                          exposure=ex, gap_shock_loss=gap_losses(ex, lev), tail_IS=tail(r),
                          turnover_per_yr_IS=turnover_cm(w),
                          dd_episodes={k: dd_episode(r, *WIN[k]) for k in ("IS", "H1", "H2")})

# Strategy A reference point (aggressive, as researched in July): TQQQ x min(1,.25/vol) above EMA-100 else cash
wA = C.strategy_weights(P, lev="tqqq", voltgt=0.25, chop_pos=1.0, seasonal=False, gold_w=0.0, off="bills")
rA = C.backtest(P, wA)
count("1a_reference", kind="strategyA", lev="tqqq", voltgt=0.25, chop=False, seasonal=False, gold_w=0, off="bills")
exA = expo_stats(wA, "tqqq")
RES["strategyA_reference"] = dict(stats=wstats(rA), crises=C.crises(rA), exposure=exA, tail_IS=tail(rA))

print("\n  GRID (IS = 1988-01-01..2026-07-24; H1 1988-2006; H2 2007-2026-07-24; OOS after freeze)")
print(f"  {'config':14s} {'CAGR':>7s} {'maxDD':>7s} {'Calm':>5s} {'Shrp':>5s} {'vol':>5s} | "
      f"{'H1 CAGR':>7s} {'H1 DD':>6s} {'H1 Cal':>6s} | {'H2 CAGR':>7s} {'H2 DD':>6s} {'H2 Cal':>6s} | "
      f"{'OOS':>6s} {'2022':>6s} {'avgExp':>6s} {'@cap':>5s}")
for name, g in grid.items():
    s = g["stats"]
    print(f"  {name:14s} {s['IS']['cagr']*100:6.2f}% {s['IS']['maxdd']*100:6.1f}% {s['IS']['calmar']:5.2f} "
          f"{s['IS']['sharpe']:5.2f} {s['IS']['vol']*100:4.1f}% | {s['H1']['cagr']*100:6.2f}% {s['H1']['maxdd']*100:5.1f}% "
          f"{s['H1']['calmar']:6.2f} | {s['H2']['cagr']*100:6.2f}% {s['H2']['maxdd']*100:5.1f}% {s['H2']['calmar']:6.2f} | "
          f"{s['OOS']['total']*100:5.1f}% {g['crises']['2022 inflation']*100:5.1f}% {g['exposure']['avg_riskon_days']:6.2f} "
          f"{g['exposure']['share_riskon_days_at_cap']*100:4.0f}%")
sA = RES["strategyA_reference"]["stats"]
print(f"  {'StrategyA ref':14s} {sA['IS']['cagr']*100:6.2f}% {sA['IS']['maxdd']*100:6.1f}% {sA['IS']['calmar']:5.2f} "
      f"{sA['IS']['sharpe']:5.2f} {sA['IS']['vol']*100:4.1f}% | OOS {sA['OOS']['total']*100:5.1f}%")

# Pareto frontier (IS) of the 18 grid points
pts_dd = [(n, g["stats"]["IS"]["cagr"], -g["stats"]["IS"]["maxdd"]) for n, g in grid.items()]
pts_vol = [(n, g["stats"]["IS"]["cagr"], g["stats"]["IS"]["vol"]) for n, g in grid.items()]
par = {"IS_cagr_vs_maxdd": pareto(pts_dd), "IS_cagr_vs_vol": pareto(pts_vol)}
for wn in ("H1", "H2"):
    par[f"{wn}_cagr_vs_maxdd"] = pareto([(n, g["stats"][wn]["cagr"], -g["stats"][wn]["maxdd"]) for n, g in grid.items()])
RES["grid"] = grid
RES["grid_pareto"] = par
print("\n  Pareto (CAGR vs |maxDD|) IS:", par["IS_cagr_vs_maxdd"])
print("  Pareto (CAGR vs |maxDD|) H1:", par["H1_cagr_vs_maxdd"])
print("  Pareto (CAGR vs |maxDD|) H2:", par["H2_cagr_vs_maxdd"])
print("  Pareto (CAGR vs vol)     IS:", par["IS_cagr_vs_vol"])

# ------------------------------------------------------------------ 1b: fine grid -> matched-risk frontier
log("SECTION 1b  fine voltgt grid (frontier mapping) for matched-risk comparison")
FG = {lev: {"vt": FINE, **{wn: {"cagr": [], "maxdd": [], "vol": [], "sharpe": []} for wn in ("IS", "H1", "H2")}}
      for lev in LEVS}
for lev in LEVS:
    for vt in FINE:
        r = base_run(lev, vt, "1b_fine_grid")
        for wn in ("IS", "H1", "H2"):
            a, b = WIN[wn]
            s = C.stats(r, a, b)
            FG[lev][wn]["cagr"].append(s["cagr"]); FG[lev][wn]["maxdd"].append(s["maxdd"])
            FG[lev][wn]["sharpe"].append(s["sharpe"]); FG[lev][wn]["vol"].append(ann_vol(r, a, b))

F2_IS = grid["qld_vt0.30"]["stats"]["IS"]; F3_IS = grid["tqqq_vt0.20"]["stats"]["IS"]
matched = {}
for wn in ("IS", "H1", "H2"):
    tab = {}
    targets = DD_TARGETS + ([round(-F2_IS["maxdd"], 4), round(-F3_IS["maxdd"], 4)] if wn == "IS" else [])
    for tgt in targets:
        row = {}
        for lev in LEVS:
            risk = [-x for x in FG[lev][wn]["maxdd"]]; val = FG[lev][wn]["cagr"]
            d = interp_at(risk, val, FINE, tgt)
            d["envelope_cagr"] = max([c for c, rk in zip(val, risk) if rk <= tgt], default=None)
            row[LEVNAME[lev]] = d
        c1, c2, c3 = (row[x]["cagr"] for x in ("1x", "2x", "3x"))
        row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
        row["2x_minus_1x_pp"] = None if (c1 is None or c2 is None) else (c2 - c1) * 100
        tab[f"maxDD_{tgt:.3f}"] = row
    vtab = {}
    for tgt in VOL_TARGETS:
        row = {}
        for lev in LEVS:
            d = interp_at(FG[lev][wn]["vol"], FG[lev][wn]["cagr"], FINE, tgt)
            row[LEVNAME[lev]] = d
        c1, c2, c3 = (row[x]["cagr"] for x in ("1x", "2x", "3x"))
        row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
        row["2x_minus_1x_pp"] = None if (c1 is None or c2 is None) else (c2 - c1) * 100
        vtab[f"vol_{tgt:.3f}"] = row
    matched[wn] = dict(by_maxdd=tab, by_vol=vtab)

# which family is on top of the combined frontier at each drawdown budget (IS, H1, H2)
frontier_owner = {}
for wn in ("IS", "H1", "H2"):
    lst = []
    for tgt in np.round(np.arange(0.10, 0.4501, 0.01), 2):
        best, bc = None, -9
        cg = {}
        for lev in LEVS:
            d = interp_at([-x for x in FG[lev][wn]["maxdd"]], FG[lev][wn]["cagr"], FINE, tgt)
            cg[LEVNAME[lev]] = d["cagr"]
            if d["cagr"] is not None and d["cagr"] > bc:
                best, bc = LEVNAME[lev], d["cagr"]
        lst.append(dict(maxdd=float(tgt), best=best, **{f"cagr_{k}": v for k, v in cg.items()}))
    frontier_owner[wn] = lst
RES["fine_grid"] = {"voltgt": FINE, **{LEVNAME[l]: {wn: FG[l][wn] for wn in ("IS", "H1", "H2")} for l in LEVS}}
RES["matched_risk"] = matched
RES["frontier_owner_by_dd_budget"] = frontier_owner

print("\n  MATCHED MAX-DRAWDOWN: CAGR interpolated on fine voltgt grid (each window matched on its own maxDD)")
for wn in ("IS", "H1", "H2"):
    for key, row in matched[wn]["by_maxdd"].items():
        f = lambda x: "   n/a" if row[x]["cagr"] is None else f"{row[x]['cagr']*100:5.2f}%"
        g = lambda x: "  -  " if row[x]["voltgt"] is None else f"{row[x]['voltgt']:.3f}"
        d32 = "  n/a" if row["3x_minus_2x_pp"] is None else f"{row['3x_minus_2x_pp']:+5.2f}"
        print(f"   {wn:3s} {key:13s}  1x {f('1x')} (vt {g('1x')})  2x {f('2x')} (vt {g('2x')})  "
              f"3x {f('3x')} (vt {g('3x')})   3x-2x {d32}pp")
print("\n  MATCHED VOLATILITY (IS):")
for key, row in matched["IS"]["by_vol"].items():
    f = lambda x: "   n/a" if row[x]["cagr"] is None else f"{row[x]['cagr']*100:5.2f}%"
    d32 = "  n/a" if row["3x_minus_2x_pp"] is None else f"{row['3x_minus_2x_pp']:+5.2f}"
    print(f"   {key:10s}  1x {f('1x')}  2x {f('2x')}  3x {f('3x')}   3x-2x {d32}pp")

# ------------------------------------------------------------------ 1c: matched pairs + instrument test
log("SECTION 1c  matched pairs, exposure-matched instrument test")
def nearest_vt(lev, target_dd, wn="IS"):
    dds = FG[lev][wn]["maxdd"]
    i = int(np.argmin([abs(d - target_dd) for d in dds]))
    return FINE[i], dds[i]

vt3m, dd3m = nearest_vt("tqqq", F2_IS["maxdd"])
vt2m, dd2m = nearest_vt("qld", F3_IS["maxdd"])
qld_max_dd = min(FG["qld"]["IS"]["maxdd"])
r_F2 = SER[("qld", 0.30)]; r_F3 = SER[("tqqq", 0.20)]
r_3m = SER[("tqqq", vt3m)]; r_2m = SER[("qld", vt2m)]

# exposure-matched instrument test: TQQQ at 2/3 weight with vt .30 -> identical NDX exposure to F2
w_inst = weights("tqqq", 0.30, extra_scale=pd.Series(2.0 / 3.0, index=IDX))
r_inst = C.backtest(P, w_inst)
count("1c_instrument", kind="strategyF", lev="tqqq", voltgt=0.30, extra_scale="2/3", off="ief", gold="gold_phys",
      lag=1, mark="mark", cost_bps=3, engine="constant_mix", data="spliced")
e_F2 = ndx_exposure(WTS[("qld", 0.30)]); e_inst = ndx_exposure(w_inst)
inst_check = float((e_F2 - e_inst).abs().max())

def pair_block(r1, r2, label1, label2):
    out = {}
    for wn in ("IS", "H1", "H2"):
        a, b = WIN[wn]
        s1, s2 = C.stats(r1, a, b), C.stats(r2, a, b)
        out[wn] = {label1: s1, label2: s2, "cagr_diff_pp": (s1["cagr"] - s2["cagr"]) * 100,
                   "maxdd_diff_pp": (s1["maxdd"] - s2["maxdd"]) * 100}
    out["OOS"] = {label1: oos(r1), label2: oos(r2)}
    c1, c2 = C.crises(r1), C.crises(r2)
    out["crises"] = {k: {label1: c1[k], label2: c2[k]} for k in c1}
    out["bootstrap_cagr_diff_IS"] = C.block_bootstrap_cagr_diff(r1.loc[:FREEZE], r2.loc[:FREEZE])
    return out

pairs = {
    "F3_vt.20_vs_F2_vt.30 (incumbents)": pair_block(r_F3, r_F2, "F3", "F2"),
    f"F3_vt{vt3m:.2f}_matchedDD_vs_F2_vt.30": pair_block(r_3m, r_F2, "F3m", "F2"),
    f"F2_vt{vt2m:.2f}_matchedDD_vs_F3_vt.20": pair_block(r_2m, r_F3, "F2m", "F3"),
    "TQQQx2/3_vt.30 (same NDX exposure as F2)_vs_F2": pair_block(r_inst, r_F2, "TQQQ_2/3", "F2"),
    "F3_vt.20_vs_TQQQx2/3_vt.30 (exposure-cap effect)": pair_block(r_F3, r_inst, "F3", "TQQQ_2/3"),
}
RES["matched_pairs"] = dict(
    note=("F3m = TQQQ with the fine-grid voltgt whose IS maxDD is closest to F2's; F2m = QLD with voltgt whose "
          "IS maxDD is closest to F3's (QLD max reachable IS maxDD on the grid: %.3f). The instrument test holds "
          "TQQQ at exactly 2/3 of F2's QLD weight (max |exposure difference| = %.2e), the freed 1/3 goes to IEF."
          % (qld_max_dd, inst_check)),
    vt_F3_matched_to_F2_dd=vt3m, dd_F3_matched=dd3m, vt_F2_matched_to_F3_dd=vt2m, dd_F2_matched=dd2m,
    qld_max_reachable_IS_maxdd=qld_max_dd, instrument_exposure_max_abs_diff=inst_check, pairs=pairs)
print(f"\n  F3 voltgt matching F2's IS maxDD ({F2_IS['maxdd']*100:.1f}%): vt={vt3m:.2f} (DD {dd3m*100:.1f}%);  "
      f"F2 voltgt matching F3's IS maxDD ({F3_IS['maxdd']*100:.1f}%): vt={vt2m:.2f} (DD {dd2m*100:.1f}%), "
      f"QLD max reachable {qld_max_dd*100:.1f}%")
for k, v in pairs.items():
    b = v["bootstrap_cagr_diff_IS"]
    print(f"   {k:52s} IS dCAGR {v['IS']['cagr_diff_pp']:+5.2f}pp dDD {v['IS']['maxdd_diff_pp']:+5.1f}pp | "
          f"H1 dCAGR {v['H1']['cagr_diff_pp']:+5.2f} dDD {v['H1']['maxdd_diff_pp']:+5.1f} | "
          f"H2 dCAGR {v['H2']['cagr_diff_pp']:+5.2f} dDD {v['H2']['maxdd_diff_pp']:+5.1f} | "
          f"boot 95% [{b['lo']*100:+.2f},{b['hi']*100:+.2f}] p<=0 {b['p_le0']:.2f}")

# ------------------------------------------------------------------ 1d: paired bootstrap of CAGR and maxDD
log("SECTION 1d  paired block bootstrap (CAGR and maxDD distributions)")
bser = {}
for lev in LEVS:
    for vt in BOOT_VTS:
        bser[f"{LEVNAME[lev]}_vt{vt:.2f}"] = base_run(lev, vt, "1d_boot_curve")
bser["F3m_matched"] = r_3m; bser["F2m_matched"] = r_2m; bser["TQQQx2/3_vt.30"] = r_inst
names, CG, DD = paired_boot(bser, n=1000)
bs = {n: boot_summary(CG[:, i], DD[:, i]) for i, n in enumerate(names)}
ix = {n: i for i, n in enumerate(names)}
def pairp(a, b):
    i, j = ix[a], ix[b]
    return dict(p_cagr_a_gt_b=float((CG[:, i] > CG[:, j]).mean()),
                median_cagr_diff_pp=float(np.median(CG[:, i] - CG[:, j]) * 100),
                p_maxdd_a_worse=float((DD[:, i] < DD[:, j]).mean()),
                median_maxdd_diff_pp=float(np.median(DD[:, i] - DD[:, j]) * 100),
                p_calmar_a_gt_b=float(((CG[:, i] / -DD[:, i]) > (CG[:, j] / -DD[:, j])).mean()))
bpairs = {"3x_vt0.20 vs 2x_vt0.30": pairp("3x_vt0.20", "2x_vt0.30"),
          "F3m_matched vs 2x_vt0.30": pairp("F3m_matched", "2x_vt0.30"),
          "TQQQx2/3_vt.30 vs 2x_vt0.30": pairp("TQQQx2/3_vt.30", "2x_vt0.30"),
          "2x_vt0.30 vs 1x_vt0.45": pairp("2x_vt0.30", "1x_vt0.45")}
# bootstrap-matched frontier: median CAGR at matched median maxDD, and at matched 1-in-20 (p5) maxDD
bmatch = {"median_dd": {}, "p5_dd": {}}
for tgt in DD_TARGETS:
    row = {}
    for lev in LEVS:
        ks = [f"{LEVNAME[lev]}_vt{vt:.2f}" for vt in BOOT_VTS]
        row[LEVNAME[lev]] = interp_at([-bs[k]["maxdd_med"] for k in ks], [bs[k]["cagr_med"] for k in ks], BOOT_VTS, tgt)
    c2, c3 = row["2x"]["cagr"], row["3x"]["cagr"]
    row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
    bmatch["median_dd"][f"{tgt:.2f}"] = row
for tgt in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    row = {}
    for lev in LEVS:
        ks = [f"{LEVNAME[lev]}_vt{vt:.2f}" for vt in BOOT_VTS]
        row[LEVNAME[lev]] = interp_at([-bs[k]["maxdd_p5"] for k in ks], [bs[k]["cagr_med"] for k in ks], BOOT_VTS, tgt)
    c2, c3 = row["2x"]["cagr"], row["3x"]["cagr"]
    row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
    bmatch["p5_dd"][f"{tgt:.2f}"] = row
RES["bootstrap"] = dict(settings="paired circular block bootstrap, 63-day blocks, 1000 draws, IS 1988-01-01..2026-07-24",
                        per_series=bs, pairs=bpairs, matched_frontier=bmatch)
print("\n  BOOTSTRAP per series (median CAGR | median maxDD | 1-in-20 maxDD | P(DD<-40%) | P(DD<-50%))")
for n in names:
    s = bs[n]
    print(f"   {n:16s} {s['cagr_med']*100:6.2f}% | {s['maxdd_med']*100:6.1f}% | {s['maxdd_p5']*100:6.1f}% | "
          f"{s['p_dd_worse_40']*100:5.1f}% | {s['p_dd_worse_50']*100:5.1f}%")
print("  bootstrap pairs:")
for k, v in bpairs.items():
    print(f"   {k:30s} P(CAGR a>b) {v['p_cagr_a_gt_b']:.2f}  med dCAGR {v['median_cagr_diff_pp']:+.2f}pp  "
          f"P(DD a worse) {v['p_maxdd_a_worse']:.2f}  med dDD {v['median_maxdd_diff_pp']:+.1f}pp  "
          f"P(Calmar a>b) {v['p_calmar_a_gt_b']:.2f}")
print("  bootstrap-matched frontier (median CAGR at equal MEDIAN maxDD):")
for k, row in bmatch["median_dd"].items():
    f = lambda x: "  n/a " if row[x]["cagr"] is None else f"{row[x]['cagr']*100:5.2f}%"
    d32 = " n/a" if row["3x_minus_2x_pp"] is None else f"{row['3x_minus_2x_pp']:+.2f}"
    print(f"   DD {k}: 1x {f('1x')} 2x {f('2x')} 3x {f('3x')}  3x-2x {d32}pp")
print("  bootstrap-matched frontier (median CAGR at equal 1-in-20 maxDD):")
for k, row in bmatch["p5_dd"].items():
    f = lambda x: "  n/a " if row[x]["cagr"] is None else f"{row[x]['cagr']*100:5.2f}%"
    d32 = " n/a" if row["3x_minus_2x_pp"] is None else f"{row['3x_minus_2x_pp']:+.2f}"
    print(f"   p5DD {k}: 1x {f('1x')} 2x {f('2x')} 3x {f('3x')}  3x-2x {d32}pp")

# ------------------------------------------------------------------ 1e: same grid without rebalance-timing luck
log("SECTION 1e  grid + matched frontier on the 5-weekday TRANCHE (timing-luck-free)")
tgrid = {}
for lev in LEVS:
    for vt in VTS:
        count("1e_tranche_grid", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=1,
              mark="tranche5", cost_bps=3, engine="constant_mix", data="spliced")
        rt = tranche_run(WTS[(lev, vt)])
        tgrid[f"{lev}_vt{vt:.2f}"] = dict(stats=wstats(rt), crises=C.crises(rt),
                                          dd_episodes={k: dd_episode(rt, *WIN[k]) for k in ("IS", "H1", "H2")})
TFG = {lev: {wn: {"cagr": [], "maxdd": [], "vol": []} for wn in ("IS", "H1", "H2")} for lev in LEVS}
TSER = {}
for lev in LEVS:
    for vt in FINE_T:
        count("1e_tranche_fine", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=1,
              mark="tranche5", cost_bps=3, engine="constant_mix", data="spliced")
        rt = tranche_run(weights(lev, vt))
        TSER[(lev, vt)] = rt
        for wn in ("IS", "H1", "H2"):
            a, b = WIN[wn]
            s = C.stats(rt, a, b)
            TFG[lev][wn]["cagr"].append(s["cagr"]); TFG[lev][wn]["maxdd"].append(s["maxdd"])
            TFG[lev][wn]["vol"].append(ann_vol(rt, a, b))
tmatched = {}
for wn in ("IS", "H1", "H2"):
    tab, vtab = {}, {}
    for tgt in DD_TARGETS:
        row = {LEVNAME[l]: interp_at([-x for x in TFG[l][wn]["maxdd"]], TFG[l][wn]["cagr"], FINE_T, tgt) for l in LEVS}
        c2, c3 = row["2x"]["cagr"], row["3x"]["cagr"]
        row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
        tab[f"maxDD_{tgt:.3f}"] = row
    for tgt in VOL_TARGETS:
        row = {LEVNAME[l]: interp_at(TFG[l][wn]["vol"], TFG[l][wn]["cagr"], FINE_T, tgt) for l in LEVS}
        c2, c3 = row["2x"]["cagr"], row["3x"]["cagr"]
        row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
        vtab[f"vol_{tgt:.3f}"] = row
    tmatched[wn] = dict(by_maxdd=tab, by_vol=vtab)
# paired bootstrap of the tranche frontier (distributional drawdown, no timing luck)
tb = {f"{LEVNAME[l]}_vt{vt:.3f}": TSER[(l, vt)] for l in LEVS for vt in FINE_T}
tnames, TCG, TDD = paired_boot(tb, n=600, seed=13)
tbs = {n: boot_summary(TCG[:, i], TDD[:, i]) for i, n in enumerate(tnames)}
tbmatch = {"median_dd": {}, "p5_dd": {}}
for key, stat, tgts in (("median_dd", "maxdd_med", (0.20, 0.25, 0.30, 0.35, 0.40)),
                        ("p5_dd", "maxdd_p5", (0.30, 0.35, 0.40, 0.45, 0.50, 0.55))):
    for tgt in tgts:
        row = {}
        for l in LEVS:
            ks = [f"{LEVNAME[l]}_vt{vt:.3f}" for vt in FINE_T]
            row[LEVNAME[l]] = interp_at([-tbs[k][stat] for k in ks], [tbs[k]["cagr_med"] for k in ks], FINE_T, tgt)
        c2, c3 = row["2x"]["cagr"], row["3x"]["cagr"]
        row["3x_minus_2x_pp"] = None if (c2 is None or c3 is None) else (c3 - c2) * 100
        tbmatch[key][f"{tgt:.2f}"] = row
RES["tranche_grid"] = dict(note=("same 18 configs, held as 5 weekday tranches (Mon..Fri rebalance phases averaged): "
                                 "removes rebalance-timing luck; matched-risk tables on a 0.025-step voltgt grid; "
                                 "bootstrap 600 paired draws"),
                           grid=tgrid, matched_risk=tmatched, bootstrap_matched=tbmatch,
                           bootstrap_per_series={k: tbs[k] for k in tbs})
print("\n  TRANCHE grid (5 weekday phases; IS / H1 / H2):")
for name, g in tgrid.items():
    s = g["stats"]
    print(f"   {name:14s} IS {s['IS']['cagr']*100:6.2f}% {s['IS']['maxdd']*100:6.1f}% Cal {s['IS']['calmar']:4.2f} Sh {s['IS']['sharpe']:4.2f} | "
          f"H1 {s['H1']['cagr']*100:6.2f}% {s['H1']['maxdd']*100:6.1f}% | H2 {s['H2']['cagr']*100:6.2f}% {s['H2']['maxdd']*100:6.1f}% | "
          f"OOS {s['OOS']['total']*100:5.1f}%")
def _pc(d):
    return "n/a" if d["cagr"] is None else f"{d['cagr'] * 100:.1f}"


for wn in ("IS", "H1", "H2"):
    print(f"  TRANCHE matched maxDD ({wn}): " + " | ".join(
        f"{k[6:]}: 1x {_pc(r['1x'])} 2x {_pc(r['2x'])} 3x {_pc(r['3x'])}" for k, r in tmatched[wn]["by_maxdd"].items()))
print("  TRANCHE matched vol (IS): " + " | ".join(
    f"{k[4:]}: 1x {_pc(r['1x'])} 2x {_pc(r['2x'])} 3x {_pc(r['3x'])}" for k, r in tmatched["IS"]["by_vol"].items()))
for key in ("median_dd", "p5_dd"):
    print(f"  TRANCHE bootstrap-matched ({key}): " + " | ".join(
        f"{k}: 1x {_pc(r['1x'])} 2x {_pc(r['2x'])} 3x {_pc(r['3x'])}" for k, r in tbmatch[key].items()))

# ================================================================== SECTION 2: EXECUTION LAG / MARKS
log("SECTION 2  execution lag and rebalance-day convention")
STRATS = {"F2-IEF": ("qld", 0.30), "F3-IEF": ("tqqq", 0.20)}
lagres = {}
for sn, (lev, vt) in STRATS.items():
    w = WTS[(lev, vt)]
    base = SER[(lev, vt)]
    rows = {}
    for lag in (1, 2):
        for mk in ("mark", "mark_wk"):
            count("2_lag", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=lag, mark=mk,
                  cost_bps=3, engine="constant_mix", data="spliced")
            r = base if (lag == 1 and mk == "mark") else C.backtest(P, w, lag=lag, mark=mk)
            rows[f"lag{lag}_{mk}"] = r
    count("2_lag", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=1, mark="mark",
          signal="t-1 (stale by one day)", cost_bps=3, engine="constant_mix", data="spliced")
    rows["stale_signal_trade_fri_close"] = C.backtest(P, w.shift(1), lag=1, mark="mark")
    out = {}
    for k, r in rows.items():
        cr = C.crises(r)
        out[k] = dict(stats=wstats(r), crises=cr, ret_1987_10_19=float(r.loc["1987-10-19"]),
                      ret_oct1987=C.period_return(r, "1987-10-01", "1987-10-31"),
                      vs_lag1_mark_boot=(None if k == "lag1_mark" else
                                         C.block_bootstrap_cagr_diff(r.loc[:FREEZE], base.loc[:FREEZE])))
    lagres[sn] = out
    print(f"\n  {sn}:  config                         IS CAGR  IS DD  Calmar | H1 CAGR  H2 CAGR | OOS    2022   COVID  Oct-87 19-Oct-87")
    for k, v in out.items():
        s = v["stats"]
        print(f"   {k:34s} {s['IS']['cagr']*100:6.2f}% {s['IS']['maxdd']*100:5.1f}% {s['IS']['calmar']:5.2f} | "
              f"{s['H1']['cagr']*100:6.2f}% {s['H2']['cagr']*100:6.2f}% | {s['OOS']['total']*100:5.1f}% "
              f"{v['crises']['2022 inflation']*100:6.1f}% {v['crises']['COVID 2020']*100:6.1f}% "
              f"{v['ret_oct1987']*100:6.1f}% {v['ret_1987_10_19']*100:6.1f}%")
RES["execution_lag"] = dict(
    note=("lag1 = weights from Friday-close signal earn returns from the next session, i.e. traded AT the Friday "
          "close (achievable only with a ~15:45 ET pre-close computation + market-on-close orders). lag2 = trade at "
          "the next session's close. 'stale_signal' = Thursday-close signal traded at Friday close (upper bound on "
          "the error of computing before the close). mark = Fridays only (holiday Fridays skip that week); "
          "mark_wk = last trading day of each week. Oct-1987 is before the 1988 stats start (shown as a stress test)."),
    mark_vs_markwk_differing_days=int((P["mark"] != P["mark_wk"]).sum()),
    results=lagres)

# ------------------------------------------------------------------ 2b: rebalance weekday (timing luck) + tranche
log("SECTION 2b  rebalance weekday phases, daily rebalancing, 5-weekday tranche")
ALLDAYS = pd.Series(True, index=IDX)
DEC4 = {"1988-1996": ("1988-01-01", "1996-12-31"), "1997-2006": ("1997-01-01", "2006-12-31"),
        "2007-2016": ("2007-01-01", "2016-12-31"), "2017-freeze": ("2017-01-01", str(FREEZE.date()))}


def cfg_count(lev, vt, lag, mark):
    count("2b_weekday", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=lag, mark=mark,
          cost_bps=3, engine="constant_mix", data="spliced")


dow_res = {}
for sn, (lev, vt) in STRATS.items():
    w = WTS[(lev, vt)]
    runs, extra = {}, {}
    for d in DAYS:
        for lag in (1, 2):
            cfg_count(lev, vt, lag, "mark" if d == "Fri" else f"dow_{d}")
            runs[f"{d}_lag{lag}"] = C.backtest(P, w, lag=lag, mark=MK_DOW[d])
            Hd = C.held_weights(P, w, lag=lag, mark=MK_DOW[d])
            sw = (Hd[lev] > 1e-12).astype(int).diff().abs()
            extra[f"{d}_lag{lag}"] = dict(onoff_switches_IS=int(sw[(sw.index >= START) & (sw.index <= FREEZE)].sum()),
                                          turnover_per_yr_IS=turnover_cm(w, lag=lag, mark=MK_DOW[d]))
    for lag in (1, 2):
        cfg_count(lev, vt, lag, "daily")
        runs[f"daily_lag{lag}"] = C.backtest(P, w, lag=lag, mark=ALLDAYS)
        extra[f"daily_lag{lag}"] = dict(turnover_per_yr_IS=turnover_cm(w, lag=lag, mark=ALLDAYS))
        cfg_count(lev, vt, lag, "tranche5")
        runs[f"tranche_lag{lag}"] = tranche_run(w, lag=lag)
        Ht = sum(C.held_weights(P, w, lag=lag, mark=MK_DOW[d]) for d in DAYS) / 5.0
        tt = Ht.diff().abs().sum(axis=1).fillna(0)
        tt = tt[(tt.index >= START) & (tt.index <= FREEZE)]
        extra[f"tranche_lag{lag}"] = dict(turnover_per_yr_IS=float(tt.sum() / ((tt.index[-1] - tt.index[0]).days / 365.25)))
    out = {}
    for k, r in runs.items():
        cr = C.crises(r)
        out[k] = dict(stats=wstats(r), decades={dk: C.stats(r, a, b)["cagr"] for dk, (a, b) in DEC4.items()},
                      crises={c: cr[c] for c in ("2022 inflation", "COVID 2020", "GFC 2007-09", "dot-com 2000-02",
                                                 "2011 EU", "OOS Jul-Nov 2026")},
                      dd_episode_IS=dd_episode(r), **extra[k])
    # Friday vs the other phases
    oth = sum(runs[f"{d}_lag1"] for d in ("Mon", "Tue", "Wed", "Thu")) / 4.0
    lr = pd.DataFrame({d: np.log1p(runs[f"{d}_lag1"]) for d in DAYS})
    lr = lr[(lr.index >= START) & (lr.index <= FREEZE)]
    yr = lr.groupby(lr.index.year).sum()
    gap = (yr["Fri"] - yr[["Mon", "Tue", "Wed", "Thu"]].mean(axis=1)) * 100
    lagcost = {wn: float(np.mean([out[f"{d}_lag1"]["stats"][wn]["cagr"] - out[f"{d}_lag2"]["stats"][wn]["cagr"]
                                  for d in DAYS]) * 100) for wn in ("IS", "H1", "H2")}
    fri_lagcost = {wn: (out["Fri_lag1"]["stats"][wn]["cagr"] - out["Fri_lag2"]["stats"][wn]["cagr"]) * 100
                   for wn in ("IS", "H1", "H2")}
    dow_res[sn] = dict(
        runs=out,
        fri_vs_mean_other4_boot=C.block_bootstrap_cagr_diff(runs["Fri_lag1"].loc[:FREEZE], oth.loc[:FREEZE]),
        fri_vs_tranche_boot=C.block_bootstrap_cagr_diff(runs["Fri_lag1"].loc[:FREEZE], runs["tranche_lag1"].loc[:FREEZE]),
        yearly_logret_gap_fri_minus_mean_other4_pp={str(k): float(v) for k, v in gap.items()},
        years_fri_beats_mean_other4=f"{int((gap > 0).sum())}/{len(gap)}",
        top5_years_share_of_gap=float(gap.sort_values(ascending=False).head(5).sum() / gap.sum()),
        mean_phase_cagr_lag1={wn: float(np.mean([out[f"{d}_lag1"]["stats"][wn]["cagr"] for d in DAYS])) for wn in ("IS", "H1", "H2")},
        lag1_minus_lag2_cagr_pp_avg_over_phases=lagcost, lag1_minus_lag2_cagr_pp_friday=fri_lagcost)
    print(f"\n  {sn}: rebalance weekday x lag  (IS CAGR / maxDD | H1 | H2 | OOS | 2022)")
    for k in [f"{d}_lag{l}" for l in (1, 2) for d in DAYS] + ["daily_lag1", "daily_lag2", "tranche_lag1", "tranche_lag2"]:
        s = out[k]["stats"]
        sw = out[k].get("onoff_switches_IS", "")
        print(f"   {k:14s} {s['IS']['cagr']*100:6.2f}% {s['IS']['maxdd']*100:6.1f}% Cal {s['IS']['calmar']:4.2f} | "
              f"H1 {s['H1']['cagr']*100:6.2f}% | H2 {s['H2']['cagr']*100:6.2f}% | OOS {s['OOS']['total']*100:5.1f}% | "
              f"2022 {out[k]['crises']['2022 inflation']*100:5.1f}% | turn {out[k]['turnover_per_yr_IS']:5.2f} | sw {sw}")
    b1, b2 = dow_res[sn]["fri_vs_mean_other4_boot"], dow_res[sn]["fri_vs_tranche_boot"]
    print(f"   Fri - mean(other 4): boot {b1['mean']*100:+.2f}pp [{b1['lo']*100:+.2f},{b1['hi']*100:+.2f}] p<=0 {b1['p_le0']:.3f}; "
          f"Fri - tranche {b2['mean']*100:+.2f}pp [{b2['lo']*100:+.2f},{b2['hi']*100:+.2f}]; years Fri>avg "
          f"{dow_res[sn]['years_fri_beats_mean_other4']}; top-5 yrs share {dow_res[sn]['top5_years_share_of_gap']*100:.0f}%")
    print(f"   lag1-lag2 CAGR (pp): Friday {fri_lagcost}  | averaged over 5 weekday phases {lagcost}")

# mechanism: next-session NDX return after closes above / below EMA-100, by weekday of the signal close
mech = {}
nxt = P["ndx_ret"].shift(-1)
f5 = np.expm1(np.log1p(P["ndx_ret"]).rolling(5).sum().shift(-5))
dow = IDX.dayofweek
abv = P["above"].astype(bool).values
for wn in ("H1", "H2"):
    a, b = WIN[wn]
    sel = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    rows = {}
    for i, d in enumerate(DAYS):
        m = sel & (dow == i)
        on, off = nxt[m & abv], nxt[m & ~abv]
        rows[d] = dict(next_ret_bp_above=float(on.mean() * 1e4), next_ret_bp_below=float(off.mean() * 1e4),
                       t_below=float(off.mean() / (off.std() / np.sqrt(off.count()))), n_below=int(off.count()),
                       next5_ret_bp_below=float(f5[m & ~abv].mean() * 1e4), next5_ret_bp_above=float(f5[m & abv].mean() * 1e4))
    mech[wn] = rows
dow_res["mechanism_next_session_ndx_return_by_signal_weekday"] = mech
print("\n  MECHANISM: mean next-session NDX return (bp) after a close BELOW EMA-100, by weekday of that close")
for wn in ("H1", "H2"):
    print(f"   {wn}: " + "  ".join(f"{d} {mech[wn][d]['next_ret_bp_below']:+6.1f} (t {mech[wn][d]['t_below']:+.1f})" for d in DAYS))
    print(f"   {wn} next-5 after below: " + "  ".join(f"{d} {mech[wn][d]['next5_ret_bp_below']:+6.1f}" for d in DAYS))
RES["rebalance_weekday"] = dict(
    note=("Weekly rebalance on each weekday (holiday weeks skip, as P.mark does for Fridays), daily rebalance, and a "
          "5-tranche book (each fifth rebalanced on a different weekday). Fri_lag1 is the research baseline. "
          "The tranche removes rebalance-timing luck and is the conservative planning estimate."),
    results=dow_res)

# ================================================================== SECTION 3: NO-TRADE BAND, DRIFT-FAITHFUL
log("SECTION 3  drift-faithful rebalance loop with no-trade bands")


def drift_loop(w, lev, mark="mark", lag=1, band=0.0, cost=C.COST, override=False):
    """
    Holdings drift with market moves between rebalances. A target computed at mark day t (close) is
    executed at the close of day t+lag-1. Rebalance is SKIPPED when every sleeve's drifted weight is within
    `band` (absolute weight) of the new target. override=True always trades when the risk-on/off state of
    the leveraged sleeve changes (held weight >0 vs target 0 or vice versa). Trades go fully to target.
    """
    cols = list(w.columns)
    R = np.nan_to_num(P[cols].to_numpy(dtype=float))
    W = np.nan_to_num(w.to_numpy(dtype=float))
    m = P[mark].to_numpy(dtype=bool)
    T, n = R.shape
    exe = np.full(T, -1)
    for t in np.where(m)[0]:
        e = t + lag - 1
        if e < T:
            exe[e] = t
    li = cols.index(lev)
    h = np.zeros(n); V = 1.0; inv = False; cur = np.zeros(n)
    ret = np.zeros(T); turn = np.zeros(T); orders = np.zeros(T); did = np.zeros(T, bool); skip = np.zeros(T, bool)
    skip_regime = np.zeros(T, bool); held_lev = np.zeros(T); tgt_lev = np.zeros(T)
    for t in range(T):
        V0 = V
        if inv:
            h = h * (1.0 + R[t]); V = h.sum()
        if exe[t] >= 0:
            tw = W[exe[t]]; cur = tw
            cw = h / V if inv else np.zeros(n)
            regime_change = (tw[li] > 1e-12) != (cw[li] > 1e-12)
            force = (not inv) or (override and regime_change)
            if band > 0 and np.abs(cw - tw).max() <= band and not force:
                skip[t] = True
                skip_regime[t] = bool(regime_change)
            else:
                tr = np.abs(tw * V - h); tn = tr.sum() / V
                c = tn * cost
                V = V * (1.0 - c); h = tw * V
                turn[t] = tn; orders[t] = float((tr / max(V, 1e-12) > 1e-6).sum()); did[t] = True; inv = True
        ret[t] = V / V0 - 1.0
        if inv:
            held_lev[t] = h[li] / V
            tgt_lev[t] = cur[li]
    S = lambda x: pd.Series(x, index=IDX)
    return dict(r=S(ret), turn=S(turn), orders=S(orders), did=S(did), skip=S(skip), skip_regime=S(skip_regime),
                held_lev=S(held_lev), tgt_lev=S(tgt_lev))


def loop_metrics(o, lev, a=START, b=FREEZE):
    sel = (IDX >= a) & (IDX <= b)
    yrs = (IDX[sel][-1] - IDX[sel][0]).days / 365.25
    did, skip = o["did"][sel], o["skip"][sel]
    held, tgt = o["held_lev"][sel], o["tgt_lev"][sel]
    dev = (held - tgt).abs()
    act = (held > 1e-9) | (tgt > 1e-9)                    # days with a leveraged position held or targeted
    return dict(turnover_per_yr=float(o["turn"][sel].sum() / yrs),
                rebalances_per_yr=float(did.sum() / yrs), skipped_per_yr=float(skip.sum() / yrs),
                skip_share=float(skip.sum() / max(did.sum() + skip.sum(), 1)),
                skipped_regime_switches=int(o["skip_regime"][sel].sum()),
                orders_per_yr=float(o["orders"][sel].sum() / yrs),
                avg_abs_dev_lev_weight_active_days=float(dev[act].mean()),
                max_abs_dev_lev_weight=float(dev.max()),
                avg_abs_dev_ndx_exposure_active_days=float(dev[act].mean() * MULT[lev]),
                days_lev_held_while_target_zero=int(((held > 1e-9) & (tgt <= 1e-12)).sum()))


band_res = {}
COSTS = [0.0, 3.0, 10.0, 20.0]
BANDS = [(0.0, False), (0.02, False), (0.05, False), (0.10, False), (0.02, True), (0.05, True), (0.10, True)]
for sn, (lev, vt) in STRATS.items():
    w = WTS[(lev, vt)]
    out = {"constant_mix_reference": {}}
    for cb in COSTS:
        count("3_band", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=1, mark="mark",
              cost_bps=cb, engine="constant_mix", data="spliced")
        r = SER[(lev, vt)] if cb == 3.0 else C.backtest(P, w, cost=cb / 1e4)
        out["constant_mix_reference"][f"{cb:g}bps"] = dict(stats=wstats(r), turnover_per_yr=turnover_cm(w))
    for band, ovr in BANDS:
        bk = f"band{int(band*100)}pp" + ("_override" if ovr else "")
        out[bk] = {}
        for cb in COSTS:
            count("3_band", kind="strategyF", lev=lev, voltgt=vt, off="ief", gold="gold_phys", lag=1, mark="mark",
                  cost_bps=cb, engine="drift_loop", band=band, override=ovr, data="spliced")
            o = drift_loop(w, lev, band=band, cost=cb / 1e4, override=ovr)
            d = dict(stats=wstats(o["r"]), crises=C.crises(o["r"]), **loop_metrics(o, lev))
            d["annual_cost_pct_of_nav"] = d["turnover_per_yr"] * cb / 1e4
            # fixed-ticket illustration: IBKR Pro Fixed minimum ~$1 per order (assumption)
            d["fixed_$1_per_order_drag_bps"] = {f"acct_{acct//1000}k": d["orders_per_yr"] * 1.0 / acct * 1e4
                                               for acct in (25000, 60000, 150000)}
            out[bk][f"{cb:g}bps"] = d
            if cb == 10.0 and bk != "band0pp":
                o0 = drift_loop(w, lev, band=0.0, cost=cb / 1e4)   # same as band0pp @10bps (deterministic)
                d["boot_vs_band0_10bps"] = C.block_bootstrap_cagr_diff(o["r"].loc[:FREEZE], o0["r"].loc[:FREEZE], n=1000)
    band_res[sn] = out
    print(f"\n  {sn}: no-trade band, drift-faithful loop (IS 1988-01..2026-07-24)")
    print(f"   {'config':20s} {'turn/yr':>7s} {'rebal/yr':>8s} {'orders/yr':>9s} {'skip%':>5s} {'avgDev':>6s} | "
          f"{'CAGR 0bp':>8s} {'3bp':>7s} {'10bp':>7s} {'20bp':>7s} | {'DD 10bp':>7s} {'Cal10':>5s} | H1/H2 CAGR@10bp")
    for bk in ["band0pp", "band2pp", "band5pp", "band10pp", "band2pp_override", "band5pp_override", "band10pp_override"]:
        d3 = out[bk]["3bps"]
        cs = [out[bk][f"{c:g}bps"]["stats"]["IS"]["cagr"] * 100 for c in COSTS]
        s10 = out[bk]["10bps"]["stats"]
        print(f"   {bk:20s} {d3['turnover_per_yr']:7.2f} {d3['rebalances_per_yr']:8.1f} {d3['orders_per_yr']:9.1f} "
              f"{d3['skip_share']*100:5.0f} {d3['avg_abs_dev_lev_weight_active_days']*100:5.2f}% "
              f"rs{d3['skipped_regime_switches']:<3d}| "
              f"{cs[0]:7.2f}% {cs[1]:6.2f}% {cs[2]:6.2f}% {cs[3]:6.2f}% | {s10['IS']['maxdd']*100:6.1f}% "
              f"{s10['IS']['calmar']:5.2f} | {s10['H1']['cagr']*100:5.2f}% / {s10['H2']['cagr']*100:5.2f}%")
    cm = out["constant_mix_reference"]
    print(f"   {'constant-mix (common)':20s} {cm['3bps']['turnover_per_yr']:7.2f} {'':8s} {'':9s} {'':5s} {'':6s} | "
          + " ".join(f"{cm[f'{c:g}bps']['stats']['IS']['cagr']*100:7.2f}%" for c in COSTS))
RES["rebalance_band"] = dict(
    note=("Drift-faithful loop: holdings drift between rebalances (common.backtest instead re-sets target weights "
          "every day for free). band = absolute weight tolerance per sleeve; rebalance skipped when ALL sleeves are "
          "within the band; '_override' = always trade when the leveraged sleeve switches on/off. Costs are bps of "
          "NAV per unit of turnover (both legs). Fixed $1/order drag is an illustrative assumption (IBKR Pro Fixed "
          "minimum), shown for 25k/60k/150k accounts."),
    results=band_res)

# ================================================================== SECTION 4: REAL vs SIMULATED SLEEVES
log("SECTION 4  real-ETF reality check 2010-03..latest")
REAL_A = pd.Timestamp("2010-03-01")
SPECIAL = {"COVID 2020-02-15..04-30": ("2020-02-15", "2020-04-30"), "2020 FY": ("2020-01-01", "2020-12-31"),
           "2022 FY": ("2022-01-01", "2022-12-31"), "2025 tariff 02-15..05-31": ("2025-02-15", "2025-05-31"),
           "2025 FY": ("2025-01-01", "2025-12-31"), "2026 Jan1..freeze": ("2026-01-01", str(FREEZE.date())),
           "OOS 2026-07-25..11-17": (str(OOS_A.date()), str(LATEST.date())),
           "2026 YTD (to 11-17)": ("2026-01-01", str(LATEST.date()))}


def track(real, sim, a, b):
    m = pd.concat([real, sim], axis=1)
    m.columns = ["real", "sim"]
    m = m[(m.index >= pd.Timestamp(a)) & (m.index <= pd.Timestamp(b))].dropna()
    if len(m) < 5:
        return None
    d = m["real"] - m["sim"]
    rr, rs = float((1 + m["real"]).prod() - 1), float((1 + m["sim"]).prod() - 1)
    return dict(ret_real=rr, ret_sim=rs, diff_pp=(rr - rs) * 100, te_ann=float(d.std() * np.sqrt(252)),
                corr=float(m["real"].corr(m["sim"])), days=int(len(m)))


def year_table(real, sim):
    out = {}
    for y in range(2010, LATEST.year + 1):
        a = max(REAL_A, pd.Timestamp(f"{y}-01-01")); b = min(LATEST, pd.Timestamp(f"{y}-12-31"))
        out[str(y)] = track(real, sim, a, b)
    return out


# sleeve-level tracking (the raw instruments)
sleeves = {}
for real, sim in [("tqqq_real", "tqqq_sim"), ("qld_real", "qld_sim"), ("qqq_real", "qqq_sim"),
                  ("ief_real", "ief_syn"), ("tlt_real", "tlt_syn")]:
    sleeves[f"{real} vs {sim}"] = dict(
        whole_2010_03_latest=track(P[real], P[sim], REAL_A, LATEST),
        windows={k: track(P[real], P[sim], a, b) for k, (a, b) in SPECIAL.items()},
        years=year_table(P[real], P[sim]))

# strategy-level: same weights, spliced (real) vs simulated return columns
rs_res = {}
for sn, (lev, vt) in STRATS.items():
    for off in ("ief", "tlt"):
        w = weights(lev, vt, off=off)
        vers = {"spliced_real": {}, "sim_all": {lev: f"{lev}_sim", off: f"{off}_syn"},
                "sim_lev_only": {lev: f"{lev}_sim"}, "sim_bond_only": {off: f"{off}_syn"}}
        rr = {}
        for vn, ren in vers.items():
            count("4_real_vs_sim", kind="strategyF", lev=lev, voltgt=vt, off=off, gold="gold_phys", lag=1, mark="mark",
                  cost_bps=3, engine="constant_mix", data=("spliced" if vn == "spliced_real" else vn))
            rr[vn] = C.backtest(P, w.rename(columns=ren))
        key = f"{sn.split('-')[0]}-{off.upper()}"
        H = C.held_weights(P, w)
        contrib = {}
        for y in range(2010, LATEST.year + 1):
            a = max(REAL_A, pd.Timestamp(f"{y}-01-01")); b = min(LATEST, pd.Timestamp(f"{y}-12-31"))
            sl = (IDX >= a) & (IDX <= b)
            contrib[str(y)] = dict(
                lev_sleeve_pp=float((H[lev] * (P[lev] - P[f"{lev}_sim"]).fillna(0))[sl].sum() * 100),
                bond_sleeve_pp=float((H[off] * (P[off] - P[f"{off}_syn"]).fillna(0))[sl].sum() * 100))
        stats_w = {}
        for vn, r in rr.items():
            stats_w[vn] = {"2010-03..freeze": C.stats(r, REAL_A, FREEZE, start=None),
                           "2010-03..latest": C.stats(r, REAL_A, LATEST, start=None),
                           "OOS": oos(r)}
        rs_res[key] = dict(
            stats=stats_w,
            tracking_real_vs_sim=dict(whole_2010_03_latest=track(rr["spliced_real"], rr["sim_all"], REAL_A, LATEST),
                                      windows={k: track(rr["spliced_real"], rr["sim_all"], a, b) for k, (a, b) in SPECIAL.items()},
                                      years=year_table(rr["spliced_real"], rr["sim_all"])),
            attribution_windows={vn: {k: C.period_return(rr[vn], a, b) for k, (a, b) in SPECIAL.items()} for vn in rr},
            exposure_weighted_sleeve_gap_by_year=contrib)

# instrument effect with REAL ETFs: TQQQ at 2/3 weight vs QLD at full weight, identical NDX exposure (2010-03+)
inst_real = {}
for vn, ren3, ren2 in [("spliced_real", {}, {}), ("sim_all", {"tqqq": "tqqq_sim", "ief": "ief_syn"},
                                                   {"qld": "qld_sim", "ief": "ief_syn"})]:
    dv = "spliced" if vn == "spliced_real" else vn
    count("4_real_vs_sim", kind="strategyF", lev="tqqq", voltgt=0.30, extra_scale="2/3", off="ief",
          gold="gold_phys", lag=1, mark="mark", cost_bps=3, engine="constant_mix", data=dv)
    count("4_real_vs_sim", kind="strategyF", lev="qld", voltgt=0.30, off="ief", gold="gold_phys", lag=1,
          mark="mark", cost_bps=3, engine="constant_mix", data=dv)
    a3 = C.backtest(P, w_inst.rename(columns=ren3)); a2 = C.backtest(P, WTS[("qld", 0.30)].rename(columns=ren2))
    s3 = C.stats(a3, REAL_A, FREEZE, start=None); s2 = C.stats(a2, REAL_A, FREEZE, start=None)
    yrs_diff = {}
    for y in range(2010, LATEST.year + 1):
        a = max(REAL_A, pd.Timestamp(f"{y}-01-01")); b = min(LATEST, pd.Timestamp(f"{y}-12-31"))
        yrs_diff[str(y)] = (C.period_return(a3, a, b) - C.period_return(a2, a, b)) * 100
    s3r = C.stats(a3, "2023-01-01", LATEST, start=None); s2r = C.stats(a2, "2023-01-01", LATEST, start=None)
    inst_real[vn] = dict(tqqq_2_3=s3, qld=s2, cagr_diff_pp=(s3["cagr"] - s2["cagr"]) * 100,
                         year_diff_pp=yrs_diff,
                         cagr_diff_pp_2023_latest=(s3r["cagr"] - s2r["cagr"]) * 100,
                         windows={k: dict(tqqq_2_3=C.period_return(a3, a, b), qld=C.period_return(a2, a, b))
                                  for k, (a, b) in SPECIAL.items()})
RES["real_vs_sim"] = dict(
    note=("2010-03-01..2026-11-17 (all four instruments real). Same weights; only the return columns differ. "
          "Gold = gold_phys (= GLD real) in every version. sim sleeves: qld_sim = 2*NDX price return - 0.95%/yr - rf, "
          "tqqq_sim = 3*NDX - 0.95% - 2rf (price index, no dividends, no swap spread); ief_syn/tlt_syn = yield-based "
          "duration/convexity models calibrated on the real ETFs (in-sample calibration over 2002-2026)."),
    sleeves=sleeves, strategies=rs_res, instrument_effect_2010_03_freeze=inst_real)

print("\n  SLEEVE tracking (real - sim, pp; TE annualised)")
for k, v in sleeves.items():
    wl = v["whole_2010_03_latest"]
    yy = v["years"]
    print(f"   {k:22s} whole: real {wl['ret_real']*100:8.1f}% sim {wl['ret_sim']*100:8.1f}%  TE {wl['te_ann']*100:4.1f}%  corr {wl['corr']:.4f}"
          f" | 2020 {yy['2020']['diff_pp']:+5.1f}  2022 {yy['2022']['diff_pp']:+5.1f}  2025 {yy['2025']['diff_pp']:+5.1f}  "
          f"2026 {yy['2026']['diff_pp']:+5.1f}")
print("\n  STRATEGY real vs sim (2010-03..)")
for k, v in rs_res.items():
    st = v["stats"]
    tr = v["tracking_real_vs_sim"]
    yy = tr["years"]
    print(f"   {k:8s} CAGR to freeze: real {st['spliced_real']['2010-03..freeze']['cagr']*100:5.2f}% sim "
          f"{st['sim_all']['2010-03..freeze']['cagr']*100:5.2f}% (lev-only sim {st['sim_lev_only']['2010-03..freeze']['cagr']*100:5.2f}%, "
          f"bond-only sim {st['sim_bond_only']['2010-03..freeze']['cagr']*100:5.2f}%)  maxDD real "
          f"{st['spliced_real']['2010-03..freeze']['maxdd']*100:5.1f}% sim {st['sim_all']['2010-03..freeze']['maxdd']*100:5.1f}%  "
          f"TE {tr['whole_2010_03_latest']['te_ann']*100:4.2f}%")
    print(f"            year diffs (pp): " + "  ".join(f"{y}:{yy[y]['diff_pp']:+.1f}" for y in yy if yy[y]))
    wnd = tr["windows"]
    print(f"            windows real/sim: " + " | ".join(f"{k2} {wnd[k2]['ret_real']*100:+.1f}/{wnd[k2]['ret_sim']*100:+.1f}"
                                                     for k2 in SPECIAL if wnd[k2]))
print("\n  INSTRUMENT effect (TQQQ x2/3 vs QLD, identical NDX exposure) 2010-03..freeze:")
for vn, d in inst_real.items():
    print(f"   {vn:13s} TQQQx2/3 {d['tqqq_2_3']['cagr']*100:5.2f}%  QLD {d['qld']['cagr']*100:5.2f}%  diff {d['cagr_diff_pp']:+.2f}pp  "
          f"(2022: {d['windows']['2022 FY']['tqqq_2_3']*100:+.1f}% vs {d['windows']['2022 FY']['qld']*100:+.1f}%)  "
          f"2023..latest CAGR diff {d['cagr_diff_pp_2023_latest']:+.2f}pp")
    print("      by year (pp): " + " ".join(f"{y}:{v:+.2f}" for y, v in d["year_diff_pp"].items()))

# ================================================================== SECTION 5: CURRENT STATE + SUMMARY
log("SECTION 5  current state and summary")
cur = {}
for sn, (lev, vt) in STRATS.items():
    w = WTS[(lev, vt)]
    cur[sn] = dict(date=str(IDX[-1].date()), target_weights_last_row={c: float(w[c].iloc[-1]) for c in w.columns},
                   held_weights_last_row={c: float(v) for c, v in C.held_weights(P, w).iloc[-1].items()},
                   above_ema100=bool(P["above"].iloc[-1]), vol20=float(P["vol20"].iloc[-1]), er20=float(P["er20"].iloc[-1]),
                   ndx=float(P["ndx"].iloc[-1]), ema100=float(P["ema100"].iloc[-1]))
RES["current_state"] = cur

by_sec = {}
for k, secs in VARIANTS.items():
    by_sec[secs[0]] = by_sec.get(secs[0], 0) + 1
RES["variants"] = dict(total_distinct=len(VARIANTS), by_first_section=by_sec,
                       note=("every distinct backtest configuration (strategy x engine x lag x marks x band x cost x data "
                             "version) counted once; fine grid = frontier mapping (not a selection search)"))
RES["runtime_sec"] = time.time() - T0
with open(OUT, "w") as f:
    json.dump(clean(RES), f, indent=1)
log(f"variants evaluated (distinct): {len(VARIANTS)}  by section: {by_sec}")
log(f"saved {OUT}")
