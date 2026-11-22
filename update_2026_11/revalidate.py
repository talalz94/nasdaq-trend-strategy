"""
update_2026_11/revalidate.py  --  FULL RE-VALIDATION ON EXTENDED DATA  (Nov-2026 refresh)

Re-tests the July-2026 leveraged Nasdaq-100 trend strategy ("Strategy F" family) on the extended
panel 1988-01-01 .. 2026-11-17 with FAITHFUL instruments:
    gold  = physical gold (GLD real <- GC=F <- monthly-interpolated; NOT the ^XAU miners July used)
    bonds = IEF / TLT (real ETF <- calibrated synthetic before inception)
All mechanics come from common.py (Friday marks, 1-day lag, 3 bps x one-way turnover).

Windows (stats always start 1988-01-01):
    IS      = 1988-01-01 .. 2026-07-24   (the July freeze: everything the July research could see)
    OOS     = 2026-07-25 .. 2026-11-17   (genuinely out-of-sample; NEVER used to choose anything)
    H1      = 1988-01-01 .. 2006-12-31   (first IS half)
    H2      = 2007-01-01 .. 2026-07-24   (second IS half)
    H2_FULL = 2007-01-01 .. 2026-11-17   (second half incl. OOS)
    FULL    = 1988-01-01 .. 2026-11-17

Sections
  1  candidates + baselines : stats per window, decades, crises, worst rolling 3y/5y CAGR,
                              % negative calendar years, longest time under water, max-DD recovery,
                              current drawdown, lag-2 robustness, current held weights
  2  instrument decomposition (July proxies -> faithful) : "4 more months" vs "instrument fix"
  3  component ablation : every July conclusion re-tested, faithful AND July instruments, by IS half
  4  parameter plateau grid : F2 (QLD) and F3 (TQQQ) with IEF + physical gold, 540 configs each
  5  walk-forward : choose by Calmar on one half, score on the other (post-freeze never used to choose)
  6  paired block bootstrap : F2 vs F3, each vs QQQ buy-and-hold, plus component tests
  7  verdicts
Output: update_2026_11/revalidate.json      (run: python -u update_2026_11/revalidate.py)
"""
import sys, os, json, time, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                   # noqa: E402

T0 = time.time()
P = C.panel()
IDX = P.index
START, FREEZE = C.START, C.FREEZE
LATEST = IDX[-1]
OOS_A = pd.Timestamp("2026-07-25")
OUT_JSON = os.path.join(HERE, "revalidate.json")

WIN = {"IS": (None, FREEZE), "FULL": (None, LATEST),
       "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
       "H2_FULL": ("2007-01-01", LATEST)}
METRICS = ("cagr", "maxdd", "calmar", "sharpe")


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ rebalance marks
_mlast = pd.Series(IDX, index=IDX).groupby(IDX.to_period("M")).max()
MONTHLY = pd.Series(False, index=IDX)
MONTHLY[pd.DatetimeIndex(_mlast.values)] = True
MARKS = {"weekly": "mark", "daily": pd.Series(True, index=IDX), "monthly": MONTHLY}

# ------------------------------------------------------------------ variant registry
# every distinct configuration evaluated is stored once here -> len(REG) = variants tested
REG, TAGS = {}, {}


def S(lev="tqqq", voltgt=0.20, chop_thr=0.25, chop_pos=0.5, seasonal=True, gold="gold_phys",
      gold_w=0.15, off="ief", gate="ema100", lag=1, rebal="weekly"):
    """canonical (fully specified) Strategy-F-family config."""
    return dict(kind="strat", lev=lev, voltgt=round(float(voltgt), 4), chop_thr=round(float(chop_thr), 4),
                chop_pos=round(float(chop_pos), 4), seasonal=bool(seasonal), gold=gold,
                gold_w=round(float(gold_w), 4), off=off, gate=gate, lag=int(lag), rebal=rebal)


def STATIC(weights, lag=1, rebal="weekly"):
    return dict(kind="static", weights={k: float(v) for k, v in sorted(weights.items())},
                lag=int(lag), rebal=rebal)


def _key(cfg):
    return json.dumps(cfg, sort_keys=True)


def weights_of(cfg):
    c = dict(cfg)
    kind = c.pop("kind"); c.pop("lag"); c.pop("rebal")
    if kind == "static":
        return pd.DataFrame({a: v for a, v in c["weights"].items()}, index=IDX)
    gate = c.pop("gate")
    ron = None if gate == "ema100" else pd.Series(True, index=IDX)   # 'none' = always risk-on
    return C.strategy_weights(P, risk_on=ron, **c)


def run(cfg, tag):
    k = _key(cfg)
    if k not in REG:
        REG[k] = C.backtest(P, weights_of(cfg), lag=cfg["lag"], mark=MARKS[cfg["rebal"]])
    TAGS.setdefault(k, set()).add(tag)
    return REG[k]


# ------------------------------------------------------------------ metrics
def seg(r, a=None, b=None):
    s = r[r.index >= START]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    return s.fillna(0)


def oos_stats(r):
    s = r[r.index >= OOS_A].fillna(0)
    e = np.r_[1.0, (1 + s).cumprod().values]          # include starting capital
    return dict(total=float(e[-1] - 1), maxdd=float((e / np.maximum.accumulate(e) - 1).min()),
                days=int(len(s)), a=str(s.index[0].date()), b=str(s.index[-1].date()))


def win_stats(r):
    d = {k: C.stats(r, a, b) for k, (a, b) in WIN.items()}
    d["OOS"] = oos_stats(r)
    return d


def rolling_worst(r):
    """worst / median rolling 3y and 5y CAGR (trading-day windows: 756 / 1260 days), full sample."""
    s = seg(r)
    e = pd.Series(np.r_[1.0, (1 + s).cumprod().values])
    out = {}
    for yrs in (3, 5):
        N = 252 * yrs
        ann = ((e / e.shift(N)) ** (252.0 / N) - 1).iloc[1:]
        ann.index = s.index
        ann = ann.dropna()
        out[f"worst_{yrs}y"] = float(ann.min())
        out[f"worst_{yrs}y_end"] = str(ann.idxmin().date())
        out[f"median_{yrs}y"] = float(ann.median())
        out[f"pct_{yrs}y_windows_neg"] = float((ann < 0).mean())
        out[f"worst_{yrs}y_IS_only"] = float(ann[ann.index <= FREEZE].min())
    return out


def cal_years(r):
    s = seg(r)
    y = (1 + s).groupby(s.index.year).prod() - 1
    full = y[y.index <= 2025]                          # complete calendar years 1988-2025 (all IS)
    p26 = s[s.index.year == 2026]
    return dict(n_full_years=int(len(full)), n_negative=int((full < 0).sum()),
                pct_negative=float((full < 0).mean()),
                worst_year=int(full.idxmin()), worst_year_ret=float(full.min()),
                best_year=int(full.idxmax()), best_year_ret=float(full.max()),
                ytd2026_to_freeze_IS=float((1 + p26[p26.index <= FREEZE]).prod() - 1),
                ytd2026_after_freeze_OOS=float((1 + p26[p26.index > FREEZE]).prod() - 1),
                ytd2026_total=float((1 + p26).prod() - 1),
                years={int(k): float(v) for k, v in y.items()})


def underwater(r):
    """longest time under water + max-DD peak/trough/recovery dates (full sample 1988..latest)."""
    s = seg(r)
    e = np.r_[1.0, (1 + s).cumprod().values]
    dates = pd.DatetimeIndex([s.index[0] - pd.Timedelta(days=1)]).append(s.index)
    dd = e / np.maximum.accumulate(e) - 1
    hi = np.flatnonzero(dd >= 0)                       # days at an equity high (exactly 0 drawdown)
    spells = [(int(i0), int(i1), False) for i0, i1 in zip(hi[:-1], hi[1:]) if i1 - i0 > 1]
    if hi[-1] < len(e) - 1:
        spells.append((int(hi[-1]), len(e) - 1, True))  # still under water at the end
    cd = lambda i0, i1: int((dates[i1] - dates[i0]).days)   # noqa: E731
    L = max(spells, key=lambda t: cd(t[0], t[1]))
    it = int(np.argmin(dd)); ip = int(hi[hi <= it][-1]); later = hi[hi > it]
    ir = int(later[0]) if len(later) else None
    return dict(longest_uw_years=round(cd(L[0], L[1]) / 365.25, 2),
                longest_uw_from=str(dates[L[0]].date()), longest_uw_to=str(dates[L[1]].date()),
                longest_uw_ongoing=bool(L[2]),
                maxdd=float(dd[it]), maxdd_peak=str(dates[ip].date()), maxdd_trough=str(dates[it].date()),
                maxdd_recovered_on=(str(dates[ir].date()) if ir is not None else None),
                peak_to_trough_years=round(cd(ip, it) / 365.25, 2),
                trough_to_recovery_years=(round(cd(it, ir) / 365.25, 2) if ir is not None else None),
                peak_to_recovery_years=(round(cd(ip, ir) / 365.25, 2) if ir is not None else None),
                current_dd=float(dd[-1]), current_dd_since_peak=str(dates[int(hi[-1])].date()))


def decades(r):
    d = C.by_decade(r)                                   # NB: '2020s' runs to latest (incl. OOS)
    d["2020s_to_freeze_IS"] = C.stats(r, "2020-01-01", FREEZE, start=None)
    return d


def profile(r):
    return dict(windows=win_stats(r), decades=decades(r), crises=C.crises(r),
                rolling=rolling_worst(r), calendar=cal_years(r), underwater=underwater(r))


def brief(r):
    """compact window stats used by ablation tables."""
    d = {k: C.stats(r, a, b) for k, (a, b) in WIN.items()}
    o = oos_stats(r); cr = C.crises(r)
    return dict(**{f"{k}_{m}": d[k][m] for k in WIN for m in METRICS},
                OOS_total=o["total"], OOS_maxdd=o["maxdd"],
                y2022=cr["2022 inflation"], dotcom=cr["dot-com 2000-02"], gfc=cr["GFC 2007-09"],
                eu2011=cr["2011 EU"], covid=cr["COVID 2020"])


# ------------------------------------------------------------------ bootstrap
def boot_pair(r1, r2, n=2000, block=63, seed=0, b=None):
    """paired circular block bootstrap (same resampling scheme/seed as C.block_bootstrap_cagr_diff)
    of CAGR, maxDD, Calmar and Sharpe differences r1 - r2.  maxDD diff > 0 => r1 shallower."""
    x = pd.concat([r1, r2], axis=1)
    if b is not None:
        x = x[x.index <= pd.Timestamp(b)]
    x = x.dropna()
    x = x[x.index >= START].values
    T = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    res = {k: np.empty(n) for k in METRICS}
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = x[ix]
        e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = g / np.abs(dd)
        sh = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
        res["cagr"][i] = g[0] - g[1]; res["maxdd"][i] = dd[0] - dd[1]
        res["calmar"][i] = cal[0] - cal[1]; res["sharpe"][i] = sh[0] - sh[1]
    out = {k: dict(mean=float(v.mean()), lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)),
                   p_gt0=float((v > 0).mean())) for k, v in res.items()}
    out["days"] = int(T); out["block"] = block; out["n"] = n
    return out


# ================================================================== 1. candidates + baselines
log("section 1: candidates and baselines")
CANDS = {
    "F3-IEF": S("tqqq", 0.20, off="ief"),
    "F3-TLT (scanner)": S("tqqq", 0.20, off="tlt"),
    "F2-IEF": S("qld", 0.30, off="ief"),
    "F2-IEF vt.25": S("qld", 0.25, off="ief"),
    "F1-IEF": S("qqq", 0.20, off="ief"),
    "A (TQQQ vt.25, cash)": S("tqqq", 0.25, chop_pos=1.0, seasonal=False, gold_w=0.0, off="bills"),
}
BASES = {
    "QQQ B&H": STATIC({"qqq": 1.0}),
    "QLD B&H": STATIC({"qld": 1.0}),
    "TQQQ B&H": STATIC({"tqqq": 1.0}),
    "60/40 SPX/IEF": STATIC({"spx": 0.6, "ief": 0.4}),
    "85/15 IEF/gold": STATIC({"ief": 0.85, "gold_phys": 0.15}),
}
ALL = {**CANDS, **BASES}
R = {k: run(cfg, "candidates" if k in CANDS else "baselines") for k, cfg in ALL.items()}
PROF = {k: profile(r) for k, r in R.items()}

LAG2 = {}
for k, cfg in CANDS.items():
    r2 = run({**cfg, "lag": 2}, "lag2")                 # lag 2 = trade at the NEXT session's close
    LAG2[k] = dict(IS=C.stats(r2, None, FREEZE), FULL=C.stats(r2), OOS=oos_stats(r2),
                   H1=C.stats(r2, *WIN["H1"]), H2=C.stats(r2, *WIN["H2"]))

CURRENT = {}
for k, cfg in CANDS.items():
    hw = C.held_weights(P, weights_of(cfg)).iloc[-1]
    CURRENT[k] = {c: round(float(v), 4) for c, v in hw.items() if abs(v) > 1e-9}
lastrow = P.iloc[-1]
STATE = dict(asof=str(LATEST.date()), last_mark=str(P.index[P["mark"].values.astype(bool)][-1].date()),
             ndx=float(lastrow["ndx"]), ema100=float(lastrow["ema100"]),
             pct_vs_ema100=float(lastrow["ndx"] / lastrow["ema100"] - 1), above=bool(lastrow["above"]),
             vol20=float(lastrow["vol20"]), er20=float(lastrow["er20"]),
             held_weights_on_latest_day=CURRENT)

print(f"\n{'strategy':24s}{'IS CAGR':>8}{'IS DD':>7}{'IS Cal':>7}{'IS Shp':>7}{'FULL Cal':>9}"
      f"{'OOS':>7}{'OOS DD':>7}{'2022':>7}{'w3y':>7}{'w5y':>7}{'%negY':>6}{'UWyrs':>6}{'curDD':>7}")
for k in ALL:
    p = PROF[k]; w = p["windows"]
    print(f"{k:24s}{w['IS']['cagr']*100:7.1f}%{w['IS']['maxdd']*100:6.1f}%{w['IS']['calmar']:7.2f}"
          f"{w['IS']['sharpe']:7.2f}{w['FULL']['calmar']:9.2f}{w['OOS']['total']*100:6.1f}%"
          f"{w['OOS']['maxdd']*100:6.1f}%{p['crises']['2022 inflation']*100:6.1f}%"
          f"{p['rolling']['worst_3y']*100:6.1f}%{p['rolling']['worst_5y']*100:6.1f}%"
          f"{p['calendar']['pct_negative']*100:5.0f}%{p['underwater']['longest_uw_years']:6.1f}"
          f"{p['underwater']['current_dd']*100:6.1f}%")
print("\nhalves (Calmar H1 1988-2006 | H2 2007-freeze) and lag-2 IS Calmar")
for k in ALL:
    w = PROF[k]["windows"]
    l2 = LAG2.get(k, {}).get("IS", {}).get("calmar", np.nan)
    print(f"  {k:24s} H1 {w['H1']['calmar']:5.2f} (CAGR {w['H1']['cagr']*100:5.1f}%, DD {w['H1']['maxdd']*100:5.1f}%)"
          f" | H2 {w['H2']['calmar']:5.2f} (CAGR {w['H2']['cagr']*100:5.1f}%, DD {w['H2']['maxdd']*100:5.1f}%)"
          f" | lag2 IS Cal {l2:5.2f}")
print("\nmax-DD recovery / longest under water (full sample)")
for k in ALL:
    u = PROF[k]["underwater"]
    print(f"  {k:24s} maxDD {u['maxdd']*100:6.1f}% peak {u['maxdd_peak']} trough {u['maxdd_trough']}"
          f" recovered {u['maxdd_recovered_on']} ({u['peak_to_recovery_years']} y) | longest UW "
          f"{u['longest_uw_years']} y {u['longest_uw_from']}..{u['longest_uw_to']}"
          f"{' ONGOING' if u['longest_uw_ongoing'] else ''}")
print(f"\ncurrent state {STATE['asof']}: NDX {STATE['ndx']:.0f} vs EMA100 {STATE['ema100']:.0f} "
      f"({STATE['pct_vs_ema100']*100:+.1f}%), vol20 {STATE['vol20']*100:.1f}%, er20 {STATE['er20']:.2f}")
for k, v in CURRENT.items():
    print(f"  {k:24s} {v}")

# ================================================================== 2. instrument decomposition
log("section 2: instrument decomposition")
FAM = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
INSTR = {"July proxies (miners + syn10y)": dict(gold="gold_miners", off="bond10_old"),
         "phys gold + syn10y": dict(gold="gold_phys", off="bond10_old"),
         "miners + IEF": dict(gold="gold_miners", off="ief"),
         "faithful (phys gold + IEF)": dict(gold="gold_phys", off="ief"),
         "phys gold + TLT (scanner)": dict(gold="gold_phys", off="tlt")}
DECOMP = {}
for f, fk in FAM.items():
    DECOMP[f] = {}
    for ik, iv in INSTR.items():
        DECOMP[f][ik] = brief(run(S(**fk, **iv), "instrument"))
    print(f"\n{f}: {'instrument set':34s}{'IS CAGR':>8}{'IS DD':>7}{'IS Cal':>7}{'FULL Cal':>9}{'H1 Cal':>7}"
          f"{'H2 Cal':>7}{'2022':>7}{'OOS':>7}")
    for ik, b in DECOMP[f].items():
        print(f"    {ik:34s}{b['IS_cagr']*100:7.1f}%{b['IS_maxdd']*100:6.1f}%{b['IS_calmar']:7.2f}"
              f"{b['FULL_calmar']:9.2f}{b['H1_calmar']:7.2f}{b['H2_calmar']:7.2f}{b['y2022']*100:6.1f}%"
              f"{b['OOS_total']*100:6.1f}%")

# ================================================================== 3. component ablation
log("section 3: component ablation (faithful and July instruments)")
ABL = {"default": {}, "no EMA gate (always on)": dict(gate="none"), "no vol target": dict(voltgt=99.0),
       "chop full size (no chop rule)": dict(chop_pos=1.0), "chop zero (full de-lever)": dict(chop_pos=0.0),
       "cash off-leg (BIL)": dict(off="bills"), "no gold": dict(gold_w=0.0),
       "no Aug-Sep half": dict(seasonal=False), "daily rebalance": dict(rebal="daily"),
       "monthly rebalance": dict(rebal="monthly")}
ISET = {"faithful": dict(gold="gold_phys", off="ief"), "july": dict(gold="gold_miners", off="bond10_old")}
ABLR = {}
for f, fk in FAM.items():
    for ins, iv in ISET.items():
        base = brief(run(S(**fk, **iv), "ablation"))
        tab = {}
        for ak, av in ABL.items():
            cfg = S(**{**fk, **iv, **av})
            b = brief(run(cfg, "ablation"))
            # delta = default minus ablated (positive => the July component HELPS)
            b["dCal_H1"] = base["H1_calmar"] - b["H1_calmar"]; b["dCal_H2"] = base["H2_calmar"] - b["H2_calmar"]
            b["dCal_IS"] = base["IS_calmar"] - b["IS_calmar"]; b["dCal_FULL"] = base["FULL_calmar"] - b["FULL_calmar"]
            b["dCAGR_IS"] = base["IS_cagr"] - b["IS_cagr"]; b["dDD_IS"] = base["IS_maxdd"] - b["IS_maxdd"]
            b["dDD_H1"] = base["H1_maxdd"] - b["H1_maxdd"]; b["dDD_H2"] = base["H2_maxdd"] - b["H2_maxdd"]
            b["helps_both_halves"] = bool(b["dCal_H1"] > 0 and b["dCal_H2"] > 0)
            tab[ak] = b
        ABLR[f"{f}|{ins}"] = tab
for kk, tab in ABLR.items():
    print(f"\n{kk}: component removed -> dCalmar (default - ablated; + = component helps)")
    print(f"    {'ablation':32s}{'IS Cal':>7}{'dH1':>7}{'dH2':>7}{'dIS':>7}{'dFULL':>7}{'dDD IS':>8}{'2022':>7}{'OOS':>7}")
    for ak, b in tab.items():
        print(f"    {ak:32s}{b['IS_calmar']:7.2f}{b['dCal_H1']:+7.2f}{b['dCal_H2']:+7.2f}{b['dCal_IS']:+7.2f}"
              f"{b['dCal_FULL']:+7.2f}{b['dDD_IS']*100:+7.1f}%{b['y2022']*100:6.1f}%{b['OOS_total']*100:6.1f}%")

# ================================================================== 4. plateau grid
log("section 4: plateau grid (2 x 540 configs)")
GV = dict(voltgt=[0.15, 0.20, 0.25, 0.30, 0.35], chop_thr=[0.20, 0.25, 0.30], chop_pos=[0.35, 0.50, 0.65],
          gold_w=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25], seasonal=[False, True])
PNUM = ["voltgt", "chop_thr", "chop_pos", "gold_w"]
DEF = {"tqqq": dict(voltgt=0.20, chop_thr=0.25, chop_pos=0.50, gold_w=0.15, seasonal=True),
       "qld": dict(voltgt=0.30, chop_thr=0.25, chop_pos=0.50, gold_w=0.15, seasonal=True)}
rows = []
for lev in ["tqqq", "qld"]:
    for vt, th, po, gw, se in itertools.product(*GV.values()):
        r = run(S(lev, vt, th, po, se, gold="gold_phys", gold_w=gw, off="ief"), "grid")
        row = dict(lev=lev, voltgt=vt, chop_thr=th, chop_pos=po, gold_w=gw, seasonal=se)
        for wk, (a, b) in WIN.items():
            st = C.stats(r, a, b)
            for m in METRICS:
                row[f"{wk}_{m}"] = st[m]
        o = oos_stats(r); row["OOS_total"] = o["total"]; row["OOS_maxdd"] = o["maxdd"]
        rows.append(row)
    log(f"  grid {lev} done")
G = pd.DataFrame(rows)
for k in PNUM:
    G[k + "_i"] = G[k].apply(lambda v, k=k: int(np.argmin(np.abs(np.array(GV[k]) - v))))
G["seasonal_i"] = G["seasonal"].astype(int)
PIDX = [k + "_i" for k in PNUM] + ["seasonal_i"]


def locate(lev, p):
    m = (G.lev == lev) & (G.seasonal == p["seasonal"])
    for k in PNUM:
        m &= np.isclose(G[k], p[k])
    return G[m].iloc[0]


def plateau(lev, win):
    df = G[G.lev == lev]; col = f"{win}_calmar"; cal = df[col]
    d = locate(lev, DEF[lev]); dc = float(d[col])
    di = np.array([d[c] for c in PIDX])
    diff = np.abs(df[PIDX].values - di[None, :])
    one = df[diff.sum(axis=1) == 1]                     # 9 one-step neighbours
    box = df[diff.max(axis=1) <= 1]                     # +-1 step in every knob (incl. seasonal both ways)
    best = df.loc[cal.idxmax()]
    marg = {k: {str(v): float(df[np.isclose(df[k], v) if k != "seasonal" else (df[k] == v)][col].median())
                for v in GV[k]} for k in GV}
    one_tab = []
    for _, rr in one.iterrows():
        ch = [k for k in PNUM if not np.isclose(rr[k], d[k])] + (["seasonal"] if rr["seasonal"] != d["seasonal"] else [])
        one_tab.append(dict(change=f"{ch[0]}={rr[ch[0]]}", calmar=float(rr[col]), d=float(rr[col] - dc)))
    res = dict(window=win, n=int(len(df)),
               dist=dict(min=float(cal.min()), p10=float(cal.quantile(.1)), p25=float(cal.quantile(.25)),
                         median=float(cal.median()), p75=float(cal.quantile(.75)), p90=float(cal.quantile(.9)),
                         max=float(cal.max())),
               default_calmar=dc, default_pctile=float((cal < dc).mean()),
               best=dict(**{k: (bool(best[k]) if k == "seasonal" else float(best[k])) for k in GV},
                         calmar=float(best[col])),
               one_step=dict(min=float(one[col].min()), median=float(one[col].median()), max=float(one[col].max()),
                             max_abs_dev=float(np.abs(one[col] - dc).max()), detail=one_tab),
               box=dict(n=int(len(box)), min=float(box[col].min()), median=float(box[col].median()),
                        max=float(box[col].max())),
               marginal_median=marg)
    res["on_plateau"] = bool(res["one_step"]["max_abs_dev"] <= 0.10 and res["box"]["median"] >= dc - 0.10)
    return res


PLAT = {f"{lev}|{win}": plateau(lev, win) for lev in ["tqqq", "qld"] for win in ["IS", "H1", "H2", "FULL"]}
for kk, p in PLAT.items():
    print(f"  {kk:10s} grid Calmar min {p['dist']['min']:.2f} med {p['dist']['median']:.2f} max {p['dist']['max']:.2f}"
          f" | default {p['default_calmar']:.2f} (pctile {p['default_pctile']*100:.0f}%) | 1-step "
          f"{p['one_step']['min']:.2f}..{p['one_step']['max']:.2f} (max|d| {p['one_step']['max_abs_dev']:.2f})"
          f" | box({p['box']['n']}) med {p['box']['median']:.2f} | plateau={p['on_plateau']}")
for lev in ["tqqq", "qld"]:
    print(f"  marginal median IS Calmar ({lev}):")
    for k, v in PLAT[f"{lev}|IS"]["marginal_median"].items():
        print(f"      {k:9s} " + "  ".join(f"{a}:{b:.2f}" for a, b in v.items()))

# July-paired 2x vs 3x across all nuisance knobs: F2 vt .30 vs F3 vt .20.  NOTE: equal NDX notional only
# while vol20 > 30%; in calmer markets F3 carries up to 2.55x notional vs F2's 1.7x (see 'exposure').
MATCH = {}
for win in ["H1", "H2", "IS", "FULL"]:
    a = G[(G.lev == "qld") & np.isclose(G.voltgt, 0.30)].set_index(["chop_thr", "chop_pos", "gold_w", "seasonal"])
    b = G[(G.lev == "tqqq") & np.isclose(G.voltgt, 0.20)].set_index(["chop_thr", "chop_pos", "gold_w", "seasonal"])
    d = (a[f"{win}_calmar"] - b[f"{win}_calmar"]).dropna()
    ds = (a[f"{win}_sharpe"] - b[f"{win}_sharpe"]).dropna()
    dc = (a[f"{win}_cagr"] - b[f"{win}_cagr"]).dropna()
    MATCH[win] = dict(n=int(len(d)), pct_F2_higher_calmar=float((d > 0).mean()), median_dCalmar=float(d.median()),
                      pct_F2_higher_sharpe=float((ds > 0).mean()), median_dSharpe=float(ds.median()),
                      median_dCAGR=float(dc.median()))
grid_F2_vs_F3 = {win: dict(F2_median=float(G[G.lev == "qld"][f"{win}_calmar"].median()),
                           F3_median=float(G[G.lev == "tqqq"][f"{win}_calmar"].median()),
                           F2_max=float(G[G.lev == "qld"][f"{win}_calmar"].max()),
                           F3_max=float(G[G.lev == "tqqq"][f"{win}_calmar"].max()))
                 for win in ["H1", "H2", "IS", "FULL"]}
print("  July-paired F2(vt.30) vs F3(vt.20) over 108 nuisance combos (share F2 higher Calmar, median dCal):",
      {k: (round(v['pct_F2_higher_calmar'], 2), round(v['median_dCalmar'], 3)) for k, v in MATCH.items()})

# ================================================================== 5. walk-forward
log("section 5: walk-forward")
PCOLS = ["lev", "voltgt", "chop_thr", "chop_pos", "gold_w", "seasonal"]


def pdesc(row):
    return {k: (row[k] if k == "lev" else (bool(row[k]) if k == "seasonal" else float(row[k]))) for k in PCOLS}


def wstat(row, win):
    return {m: float(row[f"{win}_{m}"]) for m in METRICS}


def walk(scope_df, train, tests, label, def_levs):
    pick = scope_df.loc[scope_df[f"{train}_calmar"].idxmax()]
    out = dict(scope=label, n_configs=int(len(scope_df)), train=train, pick=pdesc(pick),
               pick_train=wstat(pick, train),
               defaults_train={lv: wstat(locate(lv, DEF[lv]), train) for lv in def_levs}, tests={})
    for t in tests:
        col = f"{t}_calmar"; orc = float(scope_df[col].max())
        dd = {}
        for lv in def_levs:
            drow = locate(lv, DEF[lv])
            dd[lv] = dict(**wstat(drow, t), pctile=float((scope_df[col] < drow[col]).mean()))
        out["tests"][t] = dict(pick=dict(**wstat(pick, t), pctile=float((scope_df[col] < pick[col]).mean())),
                               defaults=dd, oracle_calmar=orc, grid_median_calmar=float(scope_df[col].median()),
                               selection_cost=orc - float(pick[col]),
                               pick_minus_default={lv: float(pick[col]) - dd[lv]["calmar"] for lv in def_levs})
    out["pick_OOS_total"] = float(pick["OOS_total"])
    out["defaults_OOS_total"] = {lv: float(locate(lv, DEF[lv])["OOS_total"]) for lv in def_levs}
    return out


SCOPES = {"F3 4-param (Aug-Sep on)": (G[(G.lev == "tqqq") & G.seasonal], ["tqqq"]),
          "F3 5-param": (G[G.lev == "tqqq"], ["tqqq"]),
          "F2 4-param (Aug-Sep on)": (G[(G.lev == "qld") & G.seasonal], ["qld"]),
          "F2 5-param": (G[G.lev == "qld"], ["qld"]),
          "joint 2x/3x 5-param": (G, ["tqqq", "qld"])}
WF = {}
for sl, (df, dl) in SCOPES.items():
    WF[f"{sl} | early->late"] = walk(df, "H1", ["H2_FULL", "H2"], sl, dl)
    WF[f"{sl} | late->early"] = walk(df, "H2", ["H1"], sl, dl)     # trained on 2007..FREEZE only
    WF[f"{sl} | IS->OOS"] = walk(df, "IS", [], sl, dl)               # 4 months: anecdotal only
for k, w in WF.items():
    p = w["pick"]
    s = (f"  {k:42s} pick {p['lev']} vt{p['voltgt']:.2f} thr{p['chop_thr']:.2f} pos{p['chop_pos']:.2f} "
         f"g{p['gold_w']:.2f} AS{int(p['seasonal'])} trainCal {w['pick_train']['calmar']:.2f}")
    for t, tt in w["tests"].items():
        dtxt = " ".join(f"def[{lv}] {v['calmar']:.2f}" for lv, v in tt["defaults"].items())
        s += (f" | TEST {t}: pick {tt['pick']['calmar']:.2f} (CAGR {tt['pick']['cagr']*100:.1f}% DD "
              f"{tt['pick']['maxdd']*100:.1f}%) {dtxt} oracle {tt['oracle_calmar']:.2f} selcost {tt['selection_cost']:.2f}")
    s += f" | OOS pick {w['pick_OOS_total']*100:.1f}% vs def " + " ".join(
        f"{lv}:{v*100:.1f}%" for lv, v in w["defaults_OOS_total"].items())
    print(s)
OOS_GRID = {lev: dict(min=float(G[G.lev == lev].OOS_total.min()), median=float(G[G.lev == lev].OOS_total.median()),
                      max=float(G[G.lev == lev].OOS_total.max()),
                      default=float(locate(lev, DEF[lev])["OOS_total"]),
                      default_pctile=float((G[G.lev == lev].OOS_total < locate(lev, DEF[lev])["OOS_total"]).mean()))
            for lev in ["tqqq", "qld"]}
print("  OOS (Jul 25-Nov 17) grid distribution:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in OOS_GRID.items()})

# ================================================================== 6. bootstrap
log("section 6: block bootstrap")
rF3, rF2, rQ = R["F3-IEF"], R["F2-IEF"], R["QQQ B&H"]
BOOT = {}
for name, (a, b) in {"F2-IEF minus F3-IEF": (rF2, rF3), "F2-IEF minus QQQ B&H": (rF2, rQ),
                     "F3-IEF minus QQQ B&H": (rF3, rQ)}.items():
    BOOT[name] = dict(FULL_cagr_common=C.block_bootstrap_cagr_diff(a, b),
                      FULL=boot_pair(a, b), IS=boot_pair(a, b, b=FREEZE),
                      FULL_block252=boot_pair(a, b, block=252))
# component tests (faithful instruments, full sample): default minus ablated
comp = {"F3 Aug-Sep on minus off": (rF3, run(S("tqqq", 0.20, seasonal=False), "ablation")),
        "F2 Aug-Sep on minus off": (rF2, run(S("qld", 0.30, seasonal=False), "ablation")),
        "F3 gold15 minus no-gold": (rF3, run(S("tqqq", 0.20, gold_w=0.0), "ablation")),
        "F2 gold15 minus no-gold": (rF2, run(S("qld", 0.30, gold_w=0.0), "ablation")),
        "F3 IEF minus TLT off-leg": (rF3, R["F3-TLT (scanner)"]),
        "F3 IEF minus BIL off-leg": (rF3, run(S("tqqq", 0.20, off="bills"), "ablation")),
        "F2 IEF minus BIL off-leg": (rF2, run(S("qld", 0.30, off="bills"), "ablation"))}
for name, (a, b) in comp.items():
    BOOT[name] = dict(FULL=boot_pair(a, b), IS=boot_pair(a, b, b=FREEZE))
for k, v in BOOT.items():
    f = v["FULL"]
    print(f"  {k:28s} FULL dCAGR {f['cagr']['mean']*100:+.2f}% [{f['cagr']['lo95']*100:+.2f},{f['cagr']['hi95']*100:+.2f}]"
          f" P>0 {f['cagr']['p_gt0']:.2f} | dCalmar {f['calmar']['mean']:+.2f} [{f['calmar']['lo95']:+.2f},"
          f"{f['calmar']['hi95']:+.2f}] P>0 {f['calmar']['p_gt0']:.2f} | dSharpe P>0 {f['sharpe']['p_gt0']:.2f}"
          f" | dMaxDD P>0 {f['maxdd']['p_gt0']:.2f}")
    if "FULL_block252" in v:
        g = v["FULL_block252"]["cagr"]
        print(f"  {'':28s} block252 dCAGR {g['mean']*100:+.2f}% [{g['lo95']*100:+.2f},{g['hi95']*100:+.2f}]"
              f" | IS-only dCAGR {v['IS']['cagr']['mean']*100:+.2f}% P>0 {v['IS']['cagr']['p_gt0']:.2f}"
              f" | common fn {v['FULL_cagr_common']['mean']*100:+.2f}%")

# ================================================================== 7. verdicts
log("section 7: verdicts")
w3, w2 = PROF["F3-IEF"]["windows"], PROF["F2-IEF"]["windows"]
fa3, fa2 = ABLR["F3|faithful"], ABLR["F2|faithful"]
ju3 = ABLR["F3|july"]


def comp_status(tab_f, tab_j, ak):
    bf, bj = tab_f[ak], tab_j[ak]
    return dict(faithful_dCal_H1=bf["dCal_H1"], faithful_dCal_H2=bf["dCal_H2"], faithful_dCal_IS=bf["dCal_IS"],
                faithful_dCal_FULL=bf["dCal_FULL"], july_instr_dCal_IS=bj["dCal_IS"],
                faithful_dDD_IS=bf["dDD_IS"], faithful_dCAGR_IS=bf["dCAGR_IS"],
                helps_both_halves_faithful=bf["helps_both_halves"],
                sign_changed_by_instruments=bool(np.sign(bf["dCal_IS"]) != np.sign(bj["dCal_IS"])),
                sign_changed_by_4_months=bool(np.sign(bf["dCal_IS"]) != np.sign(bf["dCal_FULL"])))


VERD = {f"F3 | {ak}": comp_status(fa3, ju3, ak) for ak in ABL if ak != "default"}
VERD.update({f"F2 | {ak}": comp_status(fa2, ABLR["F2|july"], ak) for ak in ABL if ak != "default"})
# NDX notional exposure actually carried (lev multiple x held sleeve weight), IS window
LEVX = {"tqqq": 3.0, "qld": 2.0, "qqq": 1.0}
EXPO = {}
for k in ["F3-IEF", "F2-IEF", "F2-IEF vt.25", "F1-IEF", "A (TQQQ vt.25, cash)"]:
    cfg = CANDS[k]
    hw = C.held_weights(P, weights_of(cfg))
    nx = (hw[cfg["lev"]] * LEVX[cfg["lev"]])
    nx = nx[(nx.index >= START) & (nx.index <= FREEZE)]
    on = nx[nx > 1e-9]
    EXPO[k] = dict(mean_notional_IS=float(nx.mean()), pct_days_in_market=float((nx > 1e-9).mean()),
                   mean_notional_when_in=float(on.mean()), max_notional=float(nx.max()),
                   pct_in_market_days_notional_ge_2=float((on >= 2.0 - 1e-9).mean()))
print("\nNDX notional exposure (IS):")
for k, v in EXPO.items():
    print(f"  {k:24s} mean {v['mean_notional_IS']:.2f}x | in market {v['pct_days_in_market']*100:.0f}% of days |"
          f" mean when in {v['mean_notional_when_in']:.2f}x | max {v['max_notional']:.2f}x |"
          f" >=2x on {v['pct_in_market_days_notional_ge_2']*100:.0f}% of in-market days")

# OOS weekly path: where the post-freeze loss came from (Friday targets + following-week return)
OOSW = []
_mk = [d for d in IDX[P["mark"].values.astype(bool)] if d >= FREEZE]
_W = {k: weights_of(CANDS[k]) for k in ["F3-IEF", "F2-IEF", "A (TQQQ vt.25, cash)"]}
for i, d in enumerate(_mk):
    nd = _mk[i + 1] if i + 1 < len(_mk) else LATEST
    if nd <= d:
        continue
    wk = lambda r: float((1 + r[(r.index > d) & (r.index <= nd)].fillna(0)).prod() - 1)   # noqa: E731
    OOSW.append(dict(friday=str(d.date()), ndx_vs_ema100=float(P.at[d, "ndx"] / P.at[d, "ema100"] - 1),
                     vol20=float(P.at[d, "vol20"]), er20=float(P.at[d, "er20"]),
                     F3_tqqq=float(_W["F3-IEF"].at[d, "tqqq"]), F2_qld=float(_W["F2-IEF"].at[d, "qld"]),
                     A_tqqq=float(_W["A (TQQQ vt.25, cash)"].at[d, "tqqq"]),
                     next_week=dict(F3=wk(R["F3-IEF"]), F2=wk(R["F2-IEF"]), A=wk(R["A (TQQQ vt.25, cash)"]),
                                    NDX=wk(P["ndx_ret"]))))
print("\nOOS weekly path (Friday target -> following week):")
for o in OOSW:
    nw = o["next_week"]
    print(f"  {o['friday']} NDX/EMA {o['ndx_vs_ema100']*100:+5.1f}% vol {o['vol20']:.2f} er {o['er20']:.2f} | "
          f"F3 {o['F3_tqqq']:.2f} F2 {o['F2_qld']:.2f} A {o['A_tqqq']:.2f} | F3 {nw['F3']*100:+5.1f}% "
          f"F2 {nw['F2']*100:+5.1f}% A {nw['A']*100:+5.1f}% NDX {nw['NDX']*100:+5.1f}%")

TWO_VS_THREE = dict(
    IS=dict(F2=w2["IS"], F3=w3["IS"]), FULL=dict(F2=w2["FULL"], F3=w3["FULL"]),
    H1=dict(F2=w2["H1"], F3=w3["H1"]), H2=dict(F2=w2["H2"], F3=w3["H2"]),
    OOS=dict(F2=w2["OOS"], F3=w3["OOS"]),
    lag2=dict(F2=LAG2["F2-IEF"], F3=LAG2["F3-IEF"]), exposure=dict(F2=EXPO["F2-IEF"], F3=EXPO["F3-IEF"]),
    july_paired_grid=MATCH, grid_medians=grid_F2_vs_F3,
    bootstrap_full=BOOT["F2-IEF minus F3-IEF"]["FULL"],
    joint_walkforward_picks={k: WF[k]["pick"]["lev"] for k in WF if k.startswith("joint")})

# ------------------------------------------------------------------ save
counts = {}
for k, tg in TAGS.items():
    for t in tg:
        counts[t] = counts.get(t, 0) + 1


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


grid_cols = PCOLS + [f"{w}_{m}" for w in WIN for m in METRICS] + ["OOS_total", "OOS_maxdd"]
OUT = dict(
    meta=dict(script="update_2026_11/revalidate.py", data_start=str(IDX[0].date()), data_end=str(LATEST.date()),
              stats_start=str(START.date()), freeze=str(FREEZE.date()), windows={k: [str(a), str(b)] for k, (a, b) in WIN.items()},
              oos_window=[str(OOS_A.date()), str(LATEST.date())],
              variants_tested_unique=len(REG), variants_by_section=counts,
              grid_values=GV, defaults=DEF, runtime_s=round(time.time() - T0, 1),
              notes=["IS = to 2026-07-24 (July freeze); OOS = 2026-07-25..2026-11-17, never used for any choice",
                     "late->early walk-forward trains on 2007-01-01..2026-07-24 (post-freeze excluded)",
                     "gold_phys before 2000-08-30 is monthly-interpolated (used only as a return series, no signal)",
                     "spx (60/40 baseline) is ^GSPC price-only; QQQ/QLD/TQQQ before inception are simulated from "
                     "the NDX price index (no dividends) -> baselines and sims understate total return slightly",
                     "Strategy A cash = BIL (T-bill minus 0.135% fee before BIL inception)",
                     "rolling 3y/5y = 756/1260 trading-day windows; calendar-year stats use complete years 1988-2025"]),
    candidates={k: dict(config=ALL[k], **PROF[k], lag2=LAG2.get(k)) for k in ALL},
    current_state=STATE,
    instrument_decomposition=DECOMP,
    ablation=ABLR,
    plateau=PLAT, july_paired_F2vt30_vs_F3vt20=MATCH, grid_F2_vs_F3=grid_F2_vs_F3,
    exposure=EXPO, oos_weekly_path=OOSW,
    walk_forward=WF, oos_grid_distribution=OOS_GRID,
    bootstrap=BOOT,
    verdicts=dict(components=VERD, two_vs_three=TWO_VS_THREE),
    grid=G[grid_cols].to_dict("records"))
with open(OUT_JSON, "w") as fh:
    json.dump(clean(OUT), fh, indent=1)
log(f"saved {OUT_JSON}  | unique variants evaluated: {len(REG)}  by section: {counts}")
