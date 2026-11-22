"""
r2_scenario_stress.py -- ROUND 2: SCENARIO STRESS OF TODAY'S BOOK AND RE-ENTRY BEHAVIOUR
=========================================================================================
DESCRIPTIVE ONLY. No strategy parameter is chosen, tuned or ranked. The July rules are applied
exactly through common.strategy_weights / common.backtest (lag 1, 3 bp, 15% gold_phys, IEF off-leg).
Scenario paths are hypothetical; historical replays reuse in-sample history (all replay windows end
before the 2026-07-24 freeze); the 2026-11-17 starting state itself is out-of-sample context.

TODAY (last panel day 2026-11-17 close): NDX 27,001.78, EMA-100 28,968.42 (-6.79%), vol20 35.5%,
er20 0.260, 10y 5.103%. All five weekday books of F1/F2/F3-IEF hold 85% IEF + 15% GLD.

FORWARD SIMULATOR (pre-registered design)
  * 252 forward trading days on the NYSE calendar from 2026-11-18. Weekday books mark every trading
    day with that weekday (main_tranche_check.py); holiday Fridays (2026-12-25, 2027-01-01,
    2027-03-26, 2027-06-18) therefore have no Friday mark, exactly like P.mark.
  * Signals are recomputed on history + path exactly as panel.py: ema100 by the adjust=False
    recursion from the panel's last value (== ewm(span=100, adjust=False) on the full series),
    above = ndx > ema100, vol20 = 20d std of pct_change * sqrt(252), er20 = |diff20| / sum|diff|
    (20d), month. Then common.strategy_weights + common.backtest per weekday book; the tranche is
    the mean of the 5 daily return streams (constant-mix, as main_tranche_check.py).
  * Sleeves from the path: qqq = ndx - 0.20%/252; qld = 2 ndx - 0.95%/252 - rf_d;
    tqqq = 3 ndx - 0.95%/252 - 2 rf_d; rf held at the 2026-11-17 value (4.138%) for every
    scenario AND replay (the cost of leverage paid from today).
  * IEF (synthetic scenarios): price index f(dy) = 1 - 7.09 dy + 0.5*76.9 dy^2 on the cumulative
    10y change dy from 5.103% (convexity of the whole move captured) plus carry
    (y_{t-1} + 0.305% panel-calibration alpha)/252. TLT (reference only): 15.8 / 134.4 / 0.183%
    on a parallel 30y shift from today's 30y. BIL = rf_d - 0.13%/252 (panel convention).
  * Noise: Brownian bridges between pins, so every path hits its scenario skeleton exactly at
    each pin. PRIMARY design A: pins only at the scenario's own waypoints + day 252 (behaves like
    a random walk: noise VR(21) 0.90 / VR(63) 0.68 = identical to an unpinned random walk under
    the same estimator). Disclosed sensitivity design B (= the FIRST run of this script): extra
    pins every 21 days. B forces month-end mean reversion (VR(21) 0.35, VR(63) 0.11 vs NDX
    0.81 / 0.85) and gave 7.6 whipsaws/yr in flat markets vs a historical max of 6, so it is a
    'range-bound market' stress, not the primary. NDX daily vol 30%/yr (sensitivity 25% and 35%);
    10y 4.0 bp/day and gold 28.1%/yr = trailing-252-day realised to the freeze. Independent
    Gaussian noise; seeds shared by every scenario (common random numbers): A@30% 40 seeds,
    A@25%, A@35% and B@30% 20 seeds each; plus a disclosed DIAGNOSTIC A@5% (10 seeds, added
    after the main runs) that separates skeleton effects from noise-driven whipsaws.
  * After a scenario's stated window every skeleton is FLAT at its end level (noise continues).
  * S0 control (everything flat, noise only) shares the NDX path of S1/S2 seed by seed, so
    S1/S2 minus S0 isolates the pure rate / gold shock effect.

SCENARIOS (skeletons)
  S0       control: NDX, 10y and gold flat (noise only).
  S1+/S1-  10y +100 / -100 bp linearly over 21 days, then held; NDX flat; gold flat.
  S2-/S2+  gold -15% / +15% over 42 days, then held; NDX flat; 10y flat.
  S3       V-rebound: NDX to the 2026-10-06 record 31,224 over 30 days (6 weeks), then +1%/month
           for 126 days (to day 156), then flat; 10y -25 bp over the first 30 days; gold flat.
  S3c      [added AFTER seeing S3, disclosed] same skeleton as S3, but NDX noise vol drops after
           the 6-week climb to the in-sample (1988..freeze) median vol20, because real V-recoveries
           calm down (the 2018-12 / 2020-03 / 2025-04 replays are the independent check).
  S4       2022-like: NDX -25% over 126 days; 10y +75 bp and gold -10% over the same 126 days.
  S5       deflationary bust: NDX -35% over 189 days; 10y -150 bp and gold +10% over 189 days.
  S6a/S6b  whipsaw: NDX = c*(1 + 0.08 sin(2 pi t/T + phi)) with c = today's EMA-100 (28,968),
           phi set so the path starts at today's level (rising); T = 42 days (3 cycles) and
           T = 21 days (6 cycles) for 126 days, ending back at today's level; then flat.
HISTORICAL REPLAYS: actual daily NDX, IEF (spliced) and gold_phys returns from the start date
  applied to today's state. PRIMARY convention 'literal': day 1 = the return dated X.
  Sensitivities: 'after' (day 1 = the day after X, the round-1 analogs convention) and
  'literal + IEF carry re-based' (IEF_t + (5.103% - 10y_{t-1})/252).

METRICS per path: 1/3/6/12-month returns (21/63/126/252 trading days) of F1/F2/F3 (Friday book and
  tranche), QQQ and QLD buy-and-hold, NDX, the legs and three static risk-off books (85% IEF /
  50:50 IEF:BIL / 85% TLT, each + 15% GLD); 12-month max drawdown; Friday re-entry decision date,
  day number and NDX level; tranche first-book and all-5-books re-entry; whipsaws = on->off switches
  (re-entries reversed) within 6 and 12 months (analogs.py definition); time in market;
  rebound capture over [NDX trough, highest close after it] when that rebound is >= 10%:
  participation = share of the rebound's NDX log-move earned while invested, capture-vs-QQQ =
  strategy log-return / QQQ log-return over the rebound, missed = share of the rebound's log-move
  completed before the book was invested.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T_START = time.time()
OUT = os.path.join(HERE, "r2_scenario_stress.json")
P = C.panel()
idx = P.index
POS_LAST = len(idx) - 1
T0 = idx[-1]
assert str(T0.date()) == "2026-11-17", T0
L0 = float(P["ndx"].iloc[-1])
EMA0 = float(P["ema100"].iloc[-1])
Y0 = float(P["tnx"].iloc[-1])
Y30_0 = float(P["tyx"].iloc[-1])
RF = float(P["rf"].iloc[-1])
RFD = RF / 252
EXP = 0.0095
ALPHA = 2.0 / 101.0
H = 252
HZ = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
IEF_D, IEF_C, IEF_A = 7.09, 76.9, 0.00305           # panel_validation.json ief_calib
TLT_D, TLT_C, TLT_A = 15.8, 134.4, 0.00183          # panel_validation.json tlt_calib
BIL_FEE = 0.0013
TIERS = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30),
         "F3": dict(lev="tqqq", voltgt=0.20)}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SEEDS = list(range(40))
RUNS = [("A", 0.30, 40), ("A", 0.25, 20), ("A", 0.35, 20), ("B", 0.30, 20),    # (noise design, vol, seeds)
        ("A", 0.05, 10)]   # diagnostic only (added after the main runs): near-noise-free skeletons, to
                           # separate skeleton effects from noise-driven whipsaw effects
VOLS = sorted({v for _, v, _ in RUNS})
BASE_VOL = 0.30
RECORD = 31224.0                                     # 2026-10-06 NDX record close (brief)
REB_MIN = 0.10                                       # rebound-capture metrics only if rebound >= 10%
VARIANTS = []                                        # every scenario configuration evaluated

# ------------------------------------------------------------------ noise calibration (to freeze)
_Q = P[P.index <= C.FREEZE]
SD_Y = float(_Q["tnx"].diff().iloc[-252:].std())     # percentage points per day
SD_G = float(_Q["gold_phys"].iloc[-252:].std())      # daily
CALM_VOL = float(P.loc[(idx >= C.START) & (idx <= C.FREEZE), "vol20"].median())   # S3c calm-phase vol
_post = dict(sd_10y_bp_last252_to_2026_11_17=float(P["tnx"].diff().iloc[-252:].std() * 100),
             gold_vol_last252_to_2026_11_17=float(P["gold_phys"].iloc[-252:].std() * np.sqrt(252)))

# ------------------------------------------------------------------ forward NYSE calendar
NYSE_HOL = pd.to_datetime([
    "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04"])
_bd = pd.bdate_range("2026-11-18", "2028-06-30")
FWD = _bd[~_bd.isin(NYSE_HOL)][:H]
assert len(FWD) == H
HOLIDAY_FRIDAYS = [str(d.date()) for d in NYSE_HOL if d.dayofweek == 4 and FWD[0] <= d <= FWD[-1]]


def nz(x):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        xf = float(x)
    except Exception:
        return x
    return None if (math.isnan(xf) or math.isinf(xf)) else xf


# ------------------------------------------------------------------ extended frame
_MARK_CACHE = {}


def marks_for(index):
    key = (index[0], index[-1], len(index))
    if key not in _MARK_CACHE:
        _MARK_CACHE[key] = {d: pd.Series(index.dayofweek == i, index=index) for i, d in enumerate(DAYS)}
    return _MARK_CACHE[key]


def build(hist_end, fwd_idx, ndx_r, ief_r, gold_r, sleeves=None, rfd=None, tail=300):
    """panel rows [hist_end-tail .. hist_end] + forward rows; forward signals recomputed exactly."""
    ht = P.iloc[hist_end - tail: hist_end + 1]
    n = len(fwd_idx)
    ndx_r = np.asarray(ndx_r, dtype=float)
    rfd_ = RFD if rfd is None else np.asarray(rfd, dtype=float)
    f = pd.DataFrame(index=fwd_idx)
    f["ndx_ret"] = ndx_r
    f["ndx"] = float(ht["ndx"].iloc[-1]) * np.cumprod(1 + ndx_r)
    if sleeves is None:
        f["qqq"] = ndx_r - 0.0020 / 252
        f["qld"] = 2 * ndx_r - EXP / 252 - rfd_
        f["tqqq"] = 3 * ndx_r - EXP / 252 - 2 * rfd_
    else:
        for k in ["qqq", "qld", "tqqq"]:
            f[k] = np.asarray(sleeves[k], dtype=float)
    f["ief"] = np.asarray(ief_r, dtype=float)
    f["gold_phys"] = np.asarray(gold_r, dtype=float)
    ema = np.empty(n)
    prev = float(ht["ema100"].iloc[-1])
    x = f["ndx"].values
    for t in range(n):
        prev = ALPHA * x[t] + (1 - ALPHA) * prev
        ema[t] = prev
    f["ema100"] = ema
    E = pd.concat([ht[["ndx", "ndx_ret", "qqq", "qld", "tqqq", "ief", "gold_phys", "ema100"]], f])
    nd = E["ndx"]
    vol = (nd.pct_change().rolling(20).std() * np.sqrt(252)).values
    er = (nd.diff(20).abs() / nd.diff().abs().rolling(20).sum().replace(0, np.nan)).values
    k = len(ht)
    E["vol20"] = np.concatenate([ht["vol20"].values, vol[k:]])
    E["er20"] = np.concatenate([ht["er20"].values, er[k:]])
    E["above"] = (E["ndx"] > E["ema100"]).astype(bool)
    E["month"] = E.index.month
    return E, k - 1


def run_books(E, pos0):
    """per tier x weekday: forward daily returns, decision-state sleeve weight (index 0 = start close)."""
    M = marks_for(E.index)
    rets, tgts = {}, {}
    for tier, kw in TIERS.items():
        w = C.strategy_weights(E, lev=kw["lev"], voltgt=kw["voltgt"], gold="gold_phys", gold_w=0.15, off="ief")
        for d in DAYS:
            r = C.backtest(E, w, mark=M[d])
            rets[(tier, d)] = r.values[pos0 + 1:]
            tgts[(tier, d)] = w[kw["lev"]].where(M[d]).ffill().fillna(0).values[pos0:]
    return rets, tgts


# ------------------------------------------------------------------ metric helpers
def cum(r, h):
    return float(np.prod(1 + r[:h]) - 1)


def mdd(r):
    e = np.concatenate([[1.0], np.cumprod(1 + r)])
    return float((e / np.maximum.accumulate(e) - 1).min())


def first_on(st, start=1):
    w = np.flatnonzero(st[start:])
    return int(w[0] + start) if len(w) else None


def spells(st, h):
    """reversed spells (entry k_in -> exit k_out) inside days 1..h of a state array (index 0 = start)."""
    out, k_in = [], None
    for t in range(1, h + 1):
        if st[t] and not st[t - 1]:
            k_in = t
        elif (not st[t]) and st[t - 1]:
            out.append((k_in if k_in is not None else 0, t))
            k_in = None
    return out


def evaluate(ndx_r, ief_r, gold_r, tlt_r, bil_r):
    """run the 3 tiers x 5 weekday books on a 252-day forward path from today's state."""
    E, pos0 = build(POS_LAST, FWD, ndx_r, ief_r, gold_r)
    rets, tgts = run_books(E, pos0)
    lv = np.concatenate([[L0], E["ndx"].values[pos0 + 1:]])
    above = E["above"].values[pos0:]                                  # index 0 = today
    ndx_r = np.asarray(ndx_r, float)
    ief_r, gold_r = np.asarray(ief_r, float), np.asarray(gold_r, float)
    tlt_r, bil_r = np.asarray(tlt_r, float), np.asarray(bil_r, float)
    qqq_r = E["qqq"].values[pos0 + 1:]
    qld_r = E["qld"].values[pos0 + 1:]
    st = {d: tgts[("F2", d)] > 0 for d in DAYS}
    for d in DAYS:                                                    # gate is shared by the tiers
        for tier in ("F1", "F3"):
            assert np.array_equal(tgts[(tier, d)] > 0, st[d]), "tier states differ"
        assert not st[d][0], "a book is risk-on at the start"
    m = {}
    # ---- benchmarks, legs and static risk-off books
    static = {"NDX": ndx_r, "QQQ": qqq_r, "QLD": qld_r, "IEF": ief_r, "GLD": gold_r, "TLT": tlt_r,
              "OFF_IEF": 0.85 * ief_r + 0.15 * gold_r,
              "OFF_5050": 0.425 * ief_r + 0.425 * bil_r + 0.15 * gold_r,
              "OFF_TLT": 0.85 * tlt_r + 0.15 * gold_r}
    for k, r in static.items():
        for hn, h in HZ.items():
            m[f"{k}_r{hn}"] = cum(r, h)
        m[f"{k}_mdd"] = mdd(r)
    m["NDX_min_vs_today"] = float(lv.min() / L0 - 1)
    m["NDX_end_level"] = float(lv[-1])
    m["daily_cross_day"] = first_on(above, 1)
    # ---- rebound window [trough, highest close after it]
    kt = int(np.argmin(lv))
    kp = kt + int(np.argmax(lv[kt:]))
    reb = float(lv[kp] / lv[kt] - 1)
    reb_ok = (kp > kt) and reb >= REB_MIN
    m["rebound_trough_day"] = kt
    m["rebound_peak_day"] = kp
    m["rebound_ndx"] = reb
    logmove = math.log(lv[kp] / lv[kt]) if kp > kt else np.nan
    ndx_log = np.log1p(ndx_r)
    # ---- strategy returns
    tr = {}
    for tier in TIERS:
        rf_ = rets[(tier, "Fri")]
        rt = sum(rets[(tier, d)] for d in DAYS) / 5
        tr[tier] = rt
        for book, r in (("Fri", rf_), ("tr", rt)):
            for hn, h in HZ.items():
                m[f"{tier}_{book}_r{hn}"] = cum(r, h)
            m[f"{tier}_{book}_mdd"] = mdd(r)
            if reb_ok:
                q = math.log(np.prod(1 + qqq_r[kt:kp]))
                s = math.log(np.prod(1 + r[kt:kp]))
                m[f"{tier}_{book}_capture_vs_qqq"] = s / q if q > 0 else None
            else:
                m[f"{tier}_{book}_capture_vs_qqq"] = None
    # ---- timing: Friday book
    s = st["Fri"]
    k = first_on(s, 1)
    m["Fri_reentry_day"] = k
    m["Fri_reentry_date"] = str(FWD[k - 1].date()) if k else None
    m["Fri_reentry_ndx"] = float(lv[k]) if k else None
    m["Fri_reentry_ndx_vs_today"] = float(lv[k] / L0 - 1) if k else None
    m["Fri_reentry_ndx_vs_prior_low"] = float(lv[k] / lv[:k + 1].min() - 1) if k else None
    m["Fri_lag_after_daily_cross"] = (k - m["daily_cross_day"]) if (k and m["daily_cross_day"]) else None
    for tier in TIERS:
        m[f"{tier}_Fri_size_at_reentry"] = float(tgts[(tier, "Fri")][k]) if k else None
    m["Fri_vol20_at_reentry"] = float(E["vol20"].values[pos0 + k]) if k else None
    m["Fri_er20_at_reentry"] = float(E["er20"].values[pos0 + k]) if k else None
    for hn, h in (("6m", 126), ("12m", 252)):
        m[f"Fri_whip_{hn}"] = int(np.sum(s[:h] & ~s[1:h + 1]))
        m[f"Fri_entries_{hn}"] = int(np.sum(~s[:h] & s[1:h + 1]))
    sp = spells(s, H)
    m["Fri_whip_ndx_entry_to_exit_mean"] = float(np.mean([lv[b] / lv[a] - 1 for a, b in sp])) if sp else None
    m["Fri_time_in_mkt_12m"] = float(s[:H].mean())
    m["Fri_on_at_12m"] = bool(s[H])
    # ---- timing: tranche
    ks = [first_on(st[d], 1) for d in DAYS]
    kk = [x for x in ks if x is not None]
    k1 = min(kk) if kk else None
    all5 = np.logical_and.reduce([st[d] for d in DAYS])
    k5 = first_on(all5, 1)
    m["tr_first_reentry_day"] = k1
    m["tr_first_reentry_date"] = str(FWD[k1 - 1].date()) if k1 else None
    m["tr_first_reentry_ndx"] = float(lv[k1]) if k1 else None
    m["tr_all5_day"] = k5
    m["tr_all5_date"] = str(FWD[k5 - 1].date()) if k5 else None
    m["tr_all5_ndx"] = float(lv[k5]) if k5 else None
    m["tr_books_reentered_12m"] = int(len(kk))
    for hn, h in (("6m", 126), ("12m", 252)):
        m[f"tr_whip_{hn}"] = float(np.mean([np.sum(st[d][:h] & ~st[d][1:h + 1]) for d in DAYS]))
    m["tr_time_in_mkt_12m"] = float(np.mean([st[d][:H].mean() for d in DAYS]))
    # ---- rebound participation / missed share (tier-independent)
    held_fri = s[:H].astype(float)                      # exposure flag on forward day t = state at t-1
    held_tr = np.mean([st[d][:H].astype(float) for d in DAYS], axis=0)
    for book, held, sts in (("Fri", held_fri, [s]), ("tr", held_tr, [st[d] for d in DAYS])):
        if reb_ok:
            m[f"{book}_participation"] = float(np.sum(held[kt:kp] * ndx_log[kt:kp]) / logmove)
            miss = []
            for z in sts:
                if z[kt]:
                    miss.append(0.0)
                else:
                    kin = first_on(z, kt + 1)
                    miss.append(1.0 if (kin is None or kin >= kp) else math.log(lv[kin] / lv[kt]) / logmove)
            m[f"{book}_missed_before_invested"] = float(np.mean(miss))
        else:
            m[f"{book}_participation"] = None
            m[f"{book}_missed_before_invested"] = None
    return m


# ------------------------------------------------------------------ scenario construction
tt = np.arange(1, H + 1)
GRID21 = list(range(21, H + 1, 21))


def ramp(total, a, b):
    """0 up to day a, linear to `total` at day b, flat after (cumulative skeleton at days 1..H)."""
    return total * np.clip((tt - a) / (b - a), 0.0, 1.0)


def osc(T, A=0.08, n=126):
    c = EMA0
    phi = math.asin((L0 / c - 1) / A)                # starts at today's level, rising
    s = math.log(c / L0) + np.log(1 + A * np.sin(2 * np.pi * tt / T + phi))
    s[tt > n] = s[n - 1]
    return s, phi


def bridge(z, pins):
    out = np.zeros_like(z)
    s0 = 0
    for e in pins:
        k = e - s0
        if k >= 2:
            seg = z[s0:e]
            out[s0:e] = (seg - seg.mean()) * math.sqrt(k / (k - 1))
        s0 = e
    return out


def pins_for(wp, design):
    """design A: the series' own waypoints + day H;  design B: + every 21 trading days."""
    s = {int(x) for x in wp if 0 < int(x) <= H} | {H}
    if design == "B":
        s |= set(GRID21)
    return sorted(s)


def quarter_pins(T, n=126):
    return sorted({int(x) for x in np.round(np.arange(T / 4, n + 1e-9, T / 4))} | {n})


ZERO = np.zeros(H)
_s6a, _phi_a = osc(42)
_s6b, _phi_b = osc(21)
# wn / wy / wg = waypoints of the NDX, 10y and gold skeletons (pins are per series, so S0 and S1/S2
# share the identical NDX path for a given seed)
SCEN = {
    "S0 control: all flat (noise only)": dict(ndx=ZERO, y=ZERO, g=ZERO, wn=[], wy=[], wg=[]),
    "S1+ 10y +100bp/1m, equities flat": dict(ndx=ZERO, y=ramp(1.00, 0, 21), g=ZERO, wn=[], wy=[21], wg=[]),
    "S1- 10y -100bp/1m, equities flat": dict(ndx=ZERO, y=ramp(-1.00, 0, 21), g=ZERO, wn=[], wy=[21], wg=[]),
    "S2- gold -15%/2m": dict(ndx=ZERO, y=ZERO, g=ramp(math.log(0.85), 0, 42), wn=[], wy=[], wg=[42]),
    "S2+ gold +15%/2m": dict(ndx=ZERO, y=ZERO, g=ramp(math.log(1.15), 0, 42), wn=[], wy=[], wg=[42]),
    "S3 V-rebound to 31,224/6w then +1%/m": dict(
        ndx=ramp(math.log(RECORD / L0), 0, 30) + ramp(6 * math.log(1.01), 30, 156),
        y=ramp(-0.25, 0, 30), g=ZERO, wn=[30, 156], wy=[30], wg=[]),
    "S3c calm V: S3 with noise vol -> in-sample median vol20 after day 30": dict(
        ndx=ramp(math.log(RECORD / L0), 0, 30) + ramp(6 * math.log(1.01), 30, 156),
        y=ramp(-0.25, 0, 30), g=ZERO, wn=[30, 156], wy=[30], wg=[], calm_after=30),
    "S4 2022-like NDX -25%/6m, 10y +75bp, gold -10%": dict(
        ndx=ramp(math.log(0.75), 0, 126), y=ramp(0.75, 0, 126), g=ramp(math.log(0.90), 0, 126),
        wn=[126], wy=[126], wg=[126]),
    "S5 deflationary bust NDX -35%/9m, 10y -150bp, gold +10%": dict(
        ndx=ramp(math.log(0.65), 0, 189), y=ramp(-1.50, 0, 189), g=ramp(math.log(1.10), 0, 189),
        wn=[189], wy=[189], wg=[189]),
    "S6a whipsaw +/-8% around EMA, 42d cycle x3": dict(ndx=_s6a, y=ZERO, g=ZERO, wn=quarter_pins(42), wy=[], wg=[]),
    "S6b whipsaw +/-8% around EMA, 21d cycle x6": dict(ndx=_s6b, y=ZERO, g=ZERO, wn=quarter_pins(21), wy=[], wg=[]),
}
SCEN_SHORT = {k: k.split(" ")[0] for k in SCEN}


def synth_path(spec, vol, seed, design="A"):
    rng = np.random.default_rng([20261118, seed])
    zn, zy, zg = rng.standard_normal(H), rng.standard_normal(H), rng.standard_normal(H)
    sn = spec["ndx"]
    sig = np.full(H, vol)
    if "calm_after" in spec:                         # S3c: calm phase after the climb (a pin day)
        sig[spec["calm_after"]:] = CALM_VOL
    dn = np.diff(np.concatenate([[0.0], sn])) + bridge(zn * sig / math.sqrt(252), pins_for(spec["wn"], design))
    ndx_r = np.expm1(dn)
    ycum = spec["y"] + np.cumsum(bridge(zy * SD_Y, pins_for(spec["wy"], design)))   # pp change from Y0
    y = Y0 + ycum
    yprev = np.concatenate([[Y0], y[:-1]])
    dy = ycum / 100.0
    f = 1 - IEF_D * dy + 0.5 * IEF_C * dy ** 2
    fprev = np.concatenate([[1.0], f[:-1]])
    ief_r = f / fprev - 1 + (yprev / 100 + IEF_A) / 252
    y30prev = Y30_0 + np.concatenate([[0.0], ycum[:-1]])
    ft = 1 - TLT_D * dy + 0.5 * TLT_C * dy ** 2
    ftprev = np.concatenate([[1.0], ft[:-1]])
    tlt_r = ft / ftprev - 1 + (y30prev / 100 + TLT_A) / 252
    gcum = spec["g"] + np.cumsum(bridge(zg * SD_G, pins_for(spec["wg"], design)))
    gold_r = np.expm1(np.diff(np.concatenate([[0.0], gcum])))
    bil_r = np.full(H, RFD - BIL_FEE / 252)
    return ndx_r, ief_r, gold_r, tlt_r, bil_r, y


# ------------------------------------------------------------------ VALIDATION
print("validation ...", flush=True)
val = {}
# V1: start the machinery at the freeze and feed the ACTUAL post-freeze returns (real ETF sleeves):
#     every weekday book of every tier must reproduce the full-panel backtest exactly.
pos_fr = int(idx.searchsorted(C.FREEZE, side="right") - 1)
oos_idx = idx[pos_fr + 1:]
seg = P.iloc[pos_fr + 1:]
E_v, p0_v = build(pos_fr, oos_idx, seg["ndx_ret"].values, seg["ief"].values, seg["gold_phys"].values,
                  sleeves={k: seg[k].values for k in ["qqq", "qld", "tqqq"]})
rets_v, tgts_v = run_books(E_v, p0_v)
Mfull = {d: pd.Series(idx.dayofweek == i, index=idx) for i, d in enumerate(DAYS)}
maxdiff, maxdiff_sig, oos_rep = 0.0, 0.0, {}
for c_ in ["ema100", "vol20", "er20"]:
    a_ = E_v[c_].values[p0_v + 1:]
    b_ = P[c_].values[pos_fr + 1:]
    maxdiff_sig = max(maxdiff_sig, float(np.nanmax(np.abs(a_ - b_) / np.maximum(1e-12, np.abs(b_)))))
assert np.array_equal(E_v["above"].values[p0_v + 1:], P["above"].values[pos_fr + 1:].astype(bool))
for tier, kw in TIERS.items():
    w_full = C.strategy_weights(P, lev=kw["lev"], voltgt=kw["voltgt"], gold="gold_phys", gold_w=0.15, off="ief")
    rs = {}
    for d in DAYS:
        r_full = C.backtest(P, w_full, mark=Mfull[d]).values[pos_fr + 1:]
        maxdiff = max(maxdiff, float(np.abs(r_full - rets_v[(tier, d)]).max()))
        rs[d] = r_full
    oos_rep[tier] = dict(Fri=float(np.prod(1 + rs["Fri"]) - 1),
                         tranche=float(np.prod(1 + sum(rs[d] for d in DAYS) / 5) - 1))
val["V1_oos_replay_max_abs_return_diff"] = maxdiff
val["V1_oos_replay_max_rel_signal_diff"] = maxdiff_sig
val["V1_oos_returns_reproduced"] = oos_rep
assert maxdiff < 1e-12 and maxdiff_sig < 1e-9, (maxdiff, maxdiff_sig)
print(f"  V1 OOS replay through the forward machinery: max |dr| {maxdiff:.2e}, max rel signal diff "
      f"{maxdiff_sig:.2e}; F2 Fri {oos_rep['F2']['Fri']*100:+.2f}% tranche {oos_rep['F2']['tranche']*100:+.2f}%, "
      f"F3 Fri {oos_rep['F3']['Fri']*100:+.2f}%", flush=True)
# V2: today's holdings in every tier x weekday book (decision state at the 2026-11-17 close)
hold = {}
for tier, kw in TIERS.items():
    w_full = C.strategy_weights(P, lev=kw["lev"], voltgt=kw["voltgt"], gold="gold_phys", gold_w=0.15, off="ief")
    for d in DAYS:
        tg = w_full.where(Mfull[d]).ffill().fillna(0).iloc[-1]
        last_mark = str(idx[Mfull[d].values][-1].date())
        hold[f"{tier}_{d}"] = dict(last_mark=last_mark, **{c_: round(float(v), 6) for c_, v in tg.items() if abs(v) > 1e-12})
        assert abs(tg.get("ief", 0) - 0.85) < 1e-12 and abs(tg.get("gold_phys", 0) - 0.15) < 1e-12 \
            and abs(tg.get(kw["lev"], 0)) < 1e-12, (tier, d, tg)
val["V2_holdings_2026_11_17_close"] = hold
# V3: truncated-history frame == full-history frame on a synthetic path
_n, _i, _g, _t, _b, _ = synth_path(SCEN["S3 V-rebound to 31,224/6w then +1%/m"], BASE_VOL, 0)
E_a, p_a = build(POS_LAST, FWD, _n, _i, _g)
E_b, p_b = build(POS_LAST, FWD, _n, _i, _g, tail=POS_LAST)
ra, ta = run_books(E_a, p_a)
rb, tb = run_books(E_b, p_b)
v3 = max(float(np.abs(ra[k] - rb[k]).max()) for k in ra)
full_ema = pd.concat([P["ndx"], E_a["ndx"].iloc[p_a + 1:]]).ewm(span=100, adjust=False).mean().values[-H:]
v3e = float(np.abs(full_ema - E_a["ema100"].values[p_a + 1:]).max())
val["V3_truncated_vs_full_history_max_abs_diff"] = v3
val["V3_ema_recursion_vs_full_ewm_max_abs_diff"] = v3e
assert v3 < 1e-12 and v3e < 1e-6, (v3, v3e)
print(f"  V3 truncated vs full history: {v3:.2e}; EMA recursion vs ewm: {v3e:.2e}", flush=True)
# V4: skeleton pins hit exactly (noise bridges vanish at pins)
lv_chk = L0 * np.cumprod(1 + _n)
val["V4_S3_level_day30"] = float(lv_chk[29])
val["V4_S3_level_day156"] = float(lv_chk[155])
assert abs(lv_chk[29] / RECORD - 1) < 1e-9 and abs(lv_chk[155] / (RECORD * 1.01 ** 6) - 1) < 1e-9
print(f"  V4 S3 pins: day30 {lv_chk[29]:.2f} (target {RECORD:.0f}), day156 {lv_chk[155]:.2f}", flush=True)

# ------------------------------------------------------------------ TODAY: thresholds, sizes, shocks
today = dict(date=str(T0.date()), ndx=L0, ema100=EMA0, gap_to_ema=L0 / EMA0 - 1,
             vol20=float(P["vol20"].iloc[-1]), er20=float(P["er20"].iloc[-1]), tnx=Y0, tyx=Y30_0, rf=RF,
             ndx_record_2026_10_06=float(P["ndx"].loc[:T0].max()),
             holiday_fridays_in_horizon=HOLIDAY_FRIDAYS,
             forward_window=[str(FWD[0].date()), str(FWD[-1].date())])
# re-entry size if the gate opened at today's vol20/er20 (outside Aug/Sep)
sz = {}
for tier, kw in TIERS.items():
    sc = min(1.0, kw["voltgt"] / today["vol20"])
    sz[tier] = dict(clean=0.85 * sc, chop=0.85 * sc * 0.5,
                    ndx_beta_clean=0.85 * sc * {"qqq": 1, "qld": 2, "tqqq": 3}[kw["lev"]])
today["reentry_size_at_today_vol"] = sz
# flat-path threshold ladder: if NDX stayed at 27,001.78 until the decision day, the Friday rule
# NDX_t > EMA_t  <=>  NDX_t > EMA_{t-1}; threshold = previous trading day's EMA on the flat path
lad = []
for j, d in enumerate(FWD[:70]):
    if d.dayofweek != 4:
        continue
    ema_prev = L0 + (EMA0 - L0) * (1 - ALPHA) ** j            # EMA at the close before d
    lad.append(dict(friday=str(d.date()), threshold=round(ema_prev, 1), pct_above_today=round(ema_prev / L0 - 1, 4)))
    if len(lad) >= 12:
        break
today["friday_threshold_ladder_flat_path"] = lad
print(f"  next Friday threshold {lad[0]['friday']}: NDX > {lad[0]['threshold']:.0f} "
      f"({lad[0]['pct_above_today']*100:+.2f}%)", flush=True)


def bond_shock(dy_bp, D, Cx):
    dy = dy_bp / 1e4
    return -D * dy + 0.5 * Cx * dy * dy


shocks = {}
for bp in (100, -100, 75, -75, -150, 50, -50):
    ief_ = bond_shock(bp, IEF_D, IEF_C)
    tlt_ = bond_shock(bp, TLT_D, TLT_C)
    shocks[f"10y {bp:+d}bp instantaneous"] = dict(
        IEF=ief_, book_85IEF_15GLD=0.85 * ief_, book_5050_IEF_BIL=0.425 * ief_, ref_85TLT_parallel=0.85 * tlt_)
for gp in (-0.15, -0.10, 0.10, 0.15):
    shocks[f"gold {gp*100:+.0f}% instantaneous"] = dict(book_any_offleg=0.15 * gp)
shocks["NDX -10% instantaneous (risk-off book)"] = dict(book=0.0)
for tier in TIERS:
    shocks[f"NDX -10% if {tier} had re-entered at today's vol (clean)"] = dict(book=-0.10 * sz[tier]["ndx_beta_clean"])
shocks["1-month carry, 85% IEF (yield unchanged)"] = dict(book=0.85 * (Y0 / 100 + IEF_A) / 12)
shocks["combo S4-type 10y +75bp & gold -10%"] = dict(book_85IEF_15GLD=0.85 * bond_shock(75, IEF_D, IEF_C) - 0.015,
                                                     book_5050_IEF_BIL=0.425 * bond_shock(75, IEF_D, IEF_C) - 0.015)
shocks["combo 10y +100bp & gold -15%"] = dict(book_85IEF_15GLD=0.85 * bond_shock(100, IEF_D, IEF_C) - 0.0225)
VARIANTS.append("instantaneous shock table (deterministic)")
VARIANTS.append("flat-path Friday threshold ladder (deterministic)")

# ------------------------------------------------------------------ SYNTHETIC SCENARIOS
KEYS_TABLE = ["NDX_r1m", "NDX_r3m", "NDX_r6m", "NDX_r12m", "QQQ_r12m", "QLD_r12m", "QQQ_mdd", "QLD_mdd"]
DATE_KEYS = {"Fri_reentry_date": "Fri_reentry_day", "tr_first_reentry_date": "tr_first_reentry_day",
             "tr_all5_date": "tr_all5_day"}


def aggregate(rows):
    keys = rows[0].keys()
    agg = {}
    for k in keys:
        if k in DATE_KEYS:
            continue
        vals = [r[k] for r in rows]
        num = [float(v) for v in vals if v is not None and not isinstance(v, str)]
        n_none = sum(v is None for v in vals)
        if not num:
            agg[k] = dict(n=0, n_none=n_none)
            continue
        a = np.array(num)
        agg[k] = dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                      mean=float(a.mean()), min=float(a.min()), max=float(a.max()), n=int(len(a)), n_none=n_none)
    for dk, nk in DATE_KEYS.items():
        days = [r[nk] for r in rows if r[nk] is not None]
        # lower median: always an actual decision day (a plain median of an even count can fall mid-week)
        agg[dk] = dict(median=str(FWD[int(np.percentile(days, 50, method="lower")) - 1].date()) if days else None,
                       share_reentered_12m=len(days) / len(rows))
    return agg


SEED_KEYS = ["NDX_r12m", "QQQ_r12m", "QLD_r12m", "F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F2_tr_r12m",
             "F2_Fri_r3m", "F2_Fri_mdd", "F3_Fri_mdd", "Fri_reentry_date", "Fri_reentry_ndx", "Fri_whip_12m",
             "tr_first_reentry_date", "Fri_participation", "OFF_IEF_r12m"]
PAIR_KEYS = [f"{t}_{b}_r{h}" for t in TIERS for b in ("Fri", "tr") for h in HZ] + \
            [f"OFF_IEF_r{h}" for h in HZ] + [f"OFF_5050_r{h}" for h in HZ] + [f"OFF_TLT_r{h}" for h in HZ] + \
            ["F2_Fri_mdd", "F2_tr_mdd"]
synthetic = {}
raw_rows = {}
print("\nsynthetic scenarios ...", flush=True)
for name, spec in SCEN.items():
    synthetic[name] = {"short": SCEN_SHORT[name], "waypoints_ndx": spec["wn"], "waypoints_10y": spec["wy"],
                       "waypoints_gold": spec["wg"]}
    if name.startswith("S6a"):
        synthetic[name]["phi"] = _phi_a
    if name.startswith("S6b"):
        synthetic[name]["phi"] = _phi_b
    for design, vol, nseed in RUNS:
        rows = []
        for sd in SEEDS[:nseed]:
            ndx_r, ief_r, gold_r, tlt_r, bil_r, ypath = synth_path(spec, vol, sd, design)
            mm = evaluate(ndx_r, ief_r, gold_r, tlt_r, bil_r)
            mm["seed"] = sd
            mm["y_end"] = float(ypath[-1])
            rows.append(mm)
        tag = f"{design}_vol{vol:.2f}"
        raw_rows[(name, tag)] = rows
        VARIANTS.append(f"{SCEN_SHORT[name]} noise-{design} vol{int(vol*100)} x{nseed} seeds")
        agg = aggregate(rows)
        blk = dict(design=design, vol=vol, n_seeds=nseed, agg=agg)
        if design == "A" and vol == BASE_VOL:
            blk["per_seed"] = [{k: nz(r[k]) for k in ["seed"] + SEED_KEYS} for r in rows]
        synthetic[name][tag] = blk
        g = lambda k: agg[k].get("median")  # noqa: E731
        print(f"  {SCEN_SHORT[name]:4s} {tag}  NDX12 {g('NDX_r12m')*100:+6.1f}  QQQ {g('QQQ_r12m')*100:+6.1f}  "
              f"QLD {g('QLD_r12m')*100:+6.1f} | F1 {g('F1_Fri_r12m')*100:+6.1f} F2 {g('F2_Fri_r12m')*100:+6.1f} "
              f"F3 {g('F3_Fri_r12m')*100:+6.1f} F2tr {g('F2_tr_r12m')*100:+6.1f} | F2mdd {g('F2_Fri_mdd')*100:6.1f} | "
              f"reentry {agg['Fri_reentry_date']['median']} ({agg['Fri_reentry_date']['share_reentered_12m']*100:.0f}%) "
              f"NDX {g('Fri_reentry_ndx') or float('nan'):,.0f} | whip12 {agg['Fri_whip_12m']['mean']:.1f} | "
              f"part {g('Fri_participation') if g('Fri_participation') is not None else float('nan'):.2f} | "
              f"{time.time()-T_START:.0f}s", flush=True)

# paired shock effects: S1/S2 minus the S0 control, seed by seed (identical NDX path and gate)
S0_NAME = "S0 control: all flat (noise only)"
paired = {}
for name in SCEN:
    if not name.startswith(("S1", "S2")):
        continue
    paired[SCEN_SHORT[name]] = {}
    for design, vol, nseed in RUNS:
        tag = f"{design}_vol{vol:.2f}"
        a_rows, b_rows = raw_rows[(name, tag)], raw_rows[(S0_NAME, tag)]
        for ra_, rb_ in zip(a_rows, b_rows):
            assert ra_["seed"] == rb_["seed"] and ra_["Fri_reentry_day"] == rb_["Fri_reentry_day"]
        dd = {}
        for k in PAIR_KEYS:
            v = np.array([ra_[k] - rb_[k] for ra_, rb_ in zip(a_rows, b_rows)])
            dd[k] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                         p90=float(np.percentile(v, 90)), mean=float(v.mean()))
        paired[SCEN_SHORT[name]][tag] = dd
    d_ = paired[SCEN_SHORT[name]][f"A_vol{BASE_VOL:.2f}"]
    print(f"  paired {SCEN_SHORT[name]:4s} - S0 (A, 30%): risk-off book 1m {d_['OFF_IEF_r1m']['median']*100:+5.2f}pp | "
          f"F1/F2/F3 Fri 1m {d_['F1_Fri_r1m']['median']*100:+5.2f}/{d_['F2_Fri_r1m']['median']*100:+5.2f}/"
          f"{d_['F3_Fri_r1m']['median']*100:+5.2f}pp | 12m {d_['F1_Fri_r12m']['median']*100:+5.2f}/"
          f"{d_['F2_Fri_r12m']['median']*100:+5.2f}/{d_['F3_Fri_r12m']['median']*100:+5.2f}pp", flush=True)

# ------------------------------------------------------------------ realism diagnostics
def _vr(x, q):
    return float(pd.Series(x).rolling(q).sum().dropna().var() / (q * np.var(x, ddof=1)))


_r_ins = np.log1p(P["ndx_ret"][(idx >= C.START) & (idx <= C.FREEZE)].values)
_rng = np.random.default_rng(7)
_zz = [_rng.standard_normal(H) for _ in range(400)]
realism = dict(
    ndx_variance_ratio_1988_freeze=dict(vr5=_vr(_r_ins, 5), vr21=_vr(_r_ins, 21), vr63=_vr(_r_ins, 63)),
    noise_variance_ratio_252d_windows={
        nm: dict(vr21=float(np.mean([pd.Series(bridge(z, pins)).rolling(21).sum().dropna().var() for z in _zz])
                       / (21 * np.mean([bridge(z, pins).var(ddof=1) for z in _zz]))),
                 vr63=float(np.mean([pd.Series(bridge(z, pins)).rolling(63).sum().dropna().var() for z in _zz])
                       / (63 * np.mean([bridge(z, pins).var(ddof=1) for z in _zz]))))
        for nm, pins in (("design_A_endpoint_pin", [H]), ("design_B_monthly_pins", GRID21))})
realism["noise_variance_ratio_252d_windows"]["unpinned_random_walk"] = dict(
    vr21=float(np.mean([pd.Series(z).rolling(21).sum().dropna().var() for z in _zz]) / (21 * np.mean([z.var(ddof=1) for z in _zz]))),
    vr63=float(np.mean([pd.Series(z).rolling(63).sum().dropna().var() for z in _zz]) / (63 * np.mean([z.var(ddof=1) for z in _zz]))))
_wF2 = C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
_st = (_wF2["qld"].where(P["mark"]).ffill().fillna(0) > 0).values
_ins = np.where((idx >= C.START) & (idx <= C.FREEZE))[0]
_a, _b = int(_ins[0]), int(_ins[-1])
_cnt = np.array([int(np.sum(_st[i:i + H] & ~_st[i + 1:i + H + 1])) for i in range(_a, _b - H) if not _st[i]])
realism["historical_whipsaws_12m_from_riskoff_days_1988_freeze"] = dict(
    median=float(np.median(_cnt)), mean=float(_cnt.mean()), p90=float(np.percentile(_cnt, 90)),
    max=int(_cnt.max()), share_ge5=float((_cnt >= 5).mean()), n_days=int(len(_cnt)),
    switches_off_per_year=float(np.sum(_st[_a:_b] & ~_st[_a + 1:_b + 1]) / ((idx[_b] - idx[_a]).days / 365.25)))
# in-sample rolling 12-month outcome distribution of the Friday book (context for the scenario losses)
for _tier, _kw in TIERS.items():
    _w = C.strategy_weights(P, lev=_kw["lev"], voltgt=_kw["voltgt"], gold="gold_phys", gold_w=0.15, off="ief")
    for _book, _mk in (("Fri", P["mark"]), ("tr", None)):
        if _mk is None:
            _r = sum(C.backtest(P, _w, mark=pd.Series(idx.dayofweek == i_, index=idx)) for i_ in range(5)) / 5
        else:
            _r = C.backtest(P, _w, mark=_mk)
        _lw = np.cumsum(np.log1p(_r.values))
        _r12 = np.array([np.exp(_lw[i + H] - _lw[i]) - 1 for i in range(_a, _b - H + 1)])
        _r12_off = np.array([np.exp(_lw[i + H] - _lw[i]) - 1 for i in range(_a, _b - H + 1) if not _st[i]])
        realism[f"insample_rolling12m_{_tier}_{_book}"] = dict(
            min=float(_r12.min()), p1=float(np.percentile(_r12, 1)), p5=float(np.percentile(_r12, 5)),
            p50=float(np.median(_r12)), from_riskoff_min=float(_r12_off.min()),
            from_riskoff_p5=float(np.percentile(_r12_off, 5)), from_riskoff_p50=float(np.median(_r12_off)),
            worst_start=str(idx[_a + int(np.argmin(_r12))].date()))
realism["calm_vol_S3c_insample_median_vol20"] = CALM_VOL
for design, vol, nseed in RUNS:
    tag = f"{design}_vol{vol:.2f}"
    w_ = np.array([r["Fri_whip_12m"] for r in raw_rows[(S0_NAME, tag)]])
    realism[f"S0_flat_whipsaws_12m_{tag}"] = dict(median=float(np.median(w_)), mean=float(w_.mean()),
                                                  max=int(w_.max()), share_ge5=float((w_ >= 5).mean()))
print(f"  realism: NDX VR21 {realism['ndx_variance_ratio_1988_freeze']['vr21']:.2f}; noise VR21 A "
      f"{realism['noise_variance_ratio_252d_windows']['design_A_endpoint_pin']['vr21']:.2f} / B "
      f"{realism['noise_variance_ratio_252d_windows']['design_B_monthly_pins']['vr21']:.2f} / RW "
      f"{realism['noise_variance_ratio_252d_windows']['unpinned_random_walk']['vr21']:.2f}; hist whipsaws 12m "
      f"from risk-off med {realism['historical_whipsaws_12m_from_riskoff_days_1988_freeze']['median']:.0f} max "
      f"{realism['historical_whipsaws_12m_from_riskoff_days_1988_freeze']['max']}; S0 A30 mean "
      f"{realism['S0_flat_whipsaws_12m_A_vol0.30']['mean']:.2f} / B30 mean {realism['S0_flat_whipsaws_12m_B_vol0.30']['mean']:.2f}",
      flush=True)

# ------------------------------------------------------------------ HISTORICAL REPLAYS
REPLAYS = ["2018-12-07", "2022-01-21", "2000-04-14", "2011-08-05", "2025-04-04", "2020-03-06",
           "2015-08-24", "1990-08-10", "1998-08-31"]
w_hist = {tier: C.strategy_weights(P, lev=kw["lev"], voltgt=kw["voltgt"], gold="gold_phys", gold_w=0.15, off="ief")
          for tier, kw in TIERS.items()}
r_hist = {tier: {d: C.backtest(P, w_hist[tier], mark=Mfull[d]) for d in DAYS} for tier in TIERS}
st_hist = {d: (w_hist["F2"]["qld"].where(Mfull[d]).ffill().fillna(0) > 0).values for d in DAYS}


def hist_context(i0):
    """the ORIGINAL episode, from the close of position i0 (state then, what the strategy then did)."""
    ndx = P["ndx"]
    h52 = ndx.rolling(252, min_periods=1).max()
    s = st_hist["Fri"]
    k = first_on(s[i0:i0 + H + 1], 1) if not s[i0] else 0
    d = dict(date_close=str(idx[i0].date()), gap_to_ema=float(ndx.iloc[i0] / P["ema100"].iloc[i0] - 1),
             dd_52w=float(ndx.iloc[i0] / h52.iloc[i0] - 1), vol20=float(P["vol20"].iloc[i0]),
             er20=float(P["er20"].iloc[i0]), tnx=float(P["tnx"].iloc[i0]),
             fri_book_risk_on=bool(s[i0]), fri_reentry_day_original=k,
             gold_src=str(P["gold_src"].iloc[i0]), ief_real=bool(not np.isnan(P["ief_real"].iloc[i0])))
    for tier in TIERS:
        rr = r_hist[tier]["Fri"].values[i0 + 1:i0 + 1 + H]
        rt = (sum(r_hist[tier][dd] for dd in DAYS) / 5).values[i0 + 1:i0 + 1 + H]
        d[f"{tier}_Fri_r12m_original"] = cum(rr, H)
        d[f"{tier}_tr_r12m_original"] = cum(rt, H)
    d["NDX_r12m_original"] = cum(P["ndx_ret"].values[i0 + 1:i0 + 1 + H], H)
    return d


replays = {}
print("\nhistorical replays ...", flush=True)
for dte in REPLAYS:
    i = idx.get_loc(pd.Timestamp(dte))
    out = {}
    for conv, s0, cadj in (("literal", i, False), ("after", i + 1, False), ("literal_ief_carry_rebased", i, True)):
        sg = P.iloc[s0:s0 + H]
        assert len(sg) == H and sg.index[-1] <= C.FREEZE, (dte, conv)
        ndx_r = sg["ndx_ret"].values
        ief_r = sg["ief"].values.copy()
        tlt_r = sg["tlt"].values.copy()
        gold_r = sg["gold_phys"].values
        assert not (np.isnan(ndx_r).any() or np.isnan(ief_r).any() or np.isnan(gold_r).any())
        if cadj:
            yprev = P["tnx"].ffill().values[s0 - 1:s0 - 1 + H]
            y30prev = P["tyx"].ffill().values[s0 - 1:s0 - 1 + H]
            ief_r = ief_r + (Y0 - yprev) / 100 / 252
            tlt_r = tlt_r + (Y30_0 - y30prev) / 100 / 252
        bil_r = np.full(H, RFD - BIL_FEE / 252)
        mm = evaluate(ndx_r, ief_r, gold_r, tlt_r, bil_r)
        mm["path_dates"] = [str(sg.index[0].date()), str(sg.index[-1].date())]
        mm["tnx_change_12m_pp"] = float(P["tnx"].ffill().iloc[s0 + H - 1] - P["tnx"].ffill().iloc[s0 - 1])
        mm["gold_src_share_interp"] = float((sg["gold_src"] == "monthly-interp").mean())
        mm["ief_share_synthetic"] = float(sg["ief_real"].isna().mean())
        out[conv] = {k: nz(v) if not isinstance(v, list) else v for k, v in mm.items()}
        VARIANTS.append(f"replay {dte} {conv}")
    out["history_context_close_before_path"] = hist_context(i - 1)
    out["history_context_close_of_date"] = hist_context(i)
    replays[dte] = out
    m_ = out["literal"]
    print(f"  {dte} lit: NDX12 {m_['NDX_r12m']*100:+6.1f} QQQ {m_['QQQ_r12m']*100:+6.1f} QLD {m_['QLD_r12m']*100:+6.1f} | "
          f"F1 {m_['F1_Fri_r12m']*100:+6.1f} F2 {m_['F2_Fri_r12m']*100:+6.1f} F3 {m_['F3_Fri_r12m']*100:+6.1f} "
          f"F2tr {m_['F2_tr_r12m']*100:+6.1f} | F2mdd {m_['F2_Fri_mdd']*100:6.1f} | reentry {m_['Fri_reentry_date']} "
          f"NDX {m_['Fri_reentry_ndx'] or float('nan'):,.0f} | whip12 {m_['Fri_whip_12m']} | part "
          f"{m_['Fri_participation'] if m_['Fri_participation'] is not None else float('nan'):.2f} | after: F2 "
          f"{out['after']['F2_Fri_r12m']*100:+6.1f} | carry-adj F2 {out['literal_ief_carry_rebased']['F2_Fri_r12m']*100:+6.1f}",
          flush=True)


# ------------------------------------------------------------------ cross-scenario summary
PRIMARY = f"A_vol{BASE_VOL:.2f}"


def med(name, key, tag=PRIMARY):
    return synthetic[name][tag]["agg"][key].get("median")


summary = {}
for name in SCEN:
    a = synthetic[name][PRIMARY]["agg"]
    summary[SCEN_SHORT[name]] = dict(
        F1_Fri_r12m=med(name, "F1_Fri_r12m"), F2_Fri_r12m=med(name, "F2_Fri_r12m"), F3_Fri_r12m=med(name, "F3_Fri_r12m"),
        F2_tr_r12m=med(name, "F2_tr_r12m"), QQQ_r12m=med(name, "QQQ_r12m"), QLD_r12m=med(name, "QLD_r12m"),
        OFF_IEF_r1m=med(name, "OFF_IEF_r1m"), OFF_IEF_r3m=med(name, "OFF_IEF_r3m"),
        reentry_share=a["Fri_reentry_date"]["share_reentered_12m"], reentry_median=a["Fri_reentry_date"]["median"],
        whip12_mean=a["Fri_whip_12m"]["mean"], F2_Fri_mdd=med(name, "F2_Fri_mdd"),
        F2_Fri_minus_QQQ_r12m=(med(name, "F2_Fri_r12m") - med(name, "QQQ_r12m")),
        noiseB_F2_Fri_r12m=med(name, "F2_Fri_r12m", "B_vol0.30"),
        noiseB_whip12_mean=synthetic[name]["B_vol0.30"]["agg"]["Fri_whip_12m"]["mean"])
summary_replays = {}
for d, r in replays.items():
    L_, hc = r["literal"], r["history_context_close_before_path"]
    summary_replays[d] = dict(
        F1_Fri_r12m=L_["F1_Fri_r12m"], F2_Fri_r12m=L_["F2_Fri_r12m"], F3_Fri_r12m=L_["F3_Fri_r12m"],
        F2_tr_r12m=L_["F2_tr_r12m"], QQQ_r12m=L_["QQQ_r12m"], QLD_r12m=L_["QLD_r12m"],
        F2_Fri_mdd=L_["F2_Fri_mdd"], QQQ_mdd=L_["QQQ_mdd"], reentry=L_["Fri_reentry_date"],
        whip12=L_["Fri_whip_12m"], F2_Fri_r12m_after_convention=r["after"]["F2_Fri_r12m"],
        F2_tr_r12m_after_convention=r["after"]["F2_tr_r12m"],
        F2_Fri_r12m_carry_rebased=r["literal_ief_carry_rebased"]["F2_Fri_r12m"],
        original_episode_F2_Fri_r12m_from_own_state=hc["F2_Fri_r12m_original"],
        original_episode_gap_to_ema=hc["gap_to_ema"], original_episode_risk_on=hc["fri_book_risk_on"])
n_rep_beat = sum(1 for v in summary_replays.values() if v["F2_Fri_r12m"] > v["QQQ_r12m"])
n_rep_dd = sum(1 for v in summary_replays.values() if v["F2_Fri_mdd"] > v["QQQ_mdd"])
# replay diagnostics: (a) Friday-minus-tranche spread (weekday phase luck: replayed days land on
# arbitrary forward weekdays); (b) today's state vs the episode's own state, same path start (X+1)
_ph = np.array([replays[d][c]["F2_Fri_r12m"] - replays[d][c]["F2_tr_r12m"] for d in REPLAYS for c in ("literal", "after")])
_sf = np.array([replays[d]["after"]["F2_Fri_r12m"] - replays[d]["history_context_close_of_date"]["F2_Fri_r12m_original"]
                for d in REPLAYS])
_stc = np.array([replays[d]["after"]["F2_tr_r12m"] - replays[d]["history_context_close_of_date"]["F2_tr_r12m_original"]
                 for d in REPLAYS])
replay_diag = dict(
    F2_Fri_minus_tranche_r12m=dict(mean=float(_ph.mean()), median=float(np.median(_ph)),
                                   mean_abs=float(np.abs(_ph).mean()), min=float(_ph.min()), max=float(_ph.max())),
    F2_today_state_minus_own_state_r12m_Fri=dict(median=float(np.median(_sf)), mean=float(_sf.mean())),
    F2_today_state_minus_own_state_r12m_tranche=dict(median=float(np.median(_stc)), mean=float(_stc.mean())),
    literal_medians={k: float(np.median([replays[d]["literal"][k] for d in REPLAYS]))
                     for k in ["F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F2_tr_r12m", "QQQ_r12m", "QLD_r12m",
                               "F2_Fri_mdd", "QQQ_mdd", "QLD_mdd", "OFF_IEF_r12m"]},
    literal_worst={k: float(np.min([replays[d]["literal"][k] for d in REPLAYS]))
                   for k in ["F2_Fri_r12m", "F2_tr_r12m", "QQQ_r12m", "QLD_r12m", "F2_Fri_mdd", "QQQ_mdd", "QLD_mdd"]})
print(f"  replay diag: Fri-minus-tranche mean {_ph.mean()*100:+.1f}pp, mean |.| {np.abs(_ph).mean()*100:.1f}pp; "
      f"today-state minus own-state F2 tranche median {np.median(_stc)*100:+.1f}pp", flush=True)

res = dict(
    meta=dict(script="update_2026_11/r2_scenario_stress.py", descriptive_only=True,
              horizon_days=H, horizons=HZ, runs=[dict(design=a_, vol=b_, seeds=c_) for a_, b_, c_ in RUNS],
              primary_run=PRIMARY, ndx_noise_vols=VOLS, base_vol=BASE_VOL,
              design_note=("design A (primary): pins only at scenario waypoints + day 252; design B (first run, "
                           "disclosed): extra pins every 21 days = month-end mean reversion (range-bound stress)"),
              sd_10y_pp_per_day_to_freeze=SD_Y, gold_daily_sd_to_freeze=SD_G,
              gold_vol_ann_to_freeze=SD_G * math.sqrt(252), post_freeze_calibration_for_reference=_post,
              ief_model=dict(duration=IEF_D, convexity=IEF_C, alpha_ann=IEF_A),
              tlt_model=dict(duration=TLT_D, convexity=TLT_C, alpha_ann=TLT_A),
              rf_financing=RF, record_level=RECORD, rebound_min=REB_MIN,
              n_variants=len(VARIANTS), variants=VARIANTS,
              n_synthetic_paths=len(SCEN) * sum(c_ for _, _, c_ in RUNS),
              runtime_sec=round(time.time() - T_START, 1)),
    validation=val, today=today, instant_shocks=shocks, realism=realism, paired_shock_effects_vs_S0=paired,
    synthetic=synthetic, replays=replays,
    summary=dict(synthetic_vol30_medians=summary, replays_literal=summary_replays, replay_diagnostics=replay_diag,
                 replays_F2_beats_QQQ_12m=f"{n_rep_beat}/{len(REPLAYS)}",
                 replays_F2_mdd_shallower_than_QQQ=f"{n_rep_dd}/{len(REPLAYS)}"))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return nz(o)


json.dump(clean(res), open(OUT, "w"), indent=1)
print(f"\nvariants evaluated: {len(VARIANTS)}  paths: {len(SCEN)*sum(c_ for _, _, c_ in RUNS)} synthetic + "
      f"{len(REPLAYS)*3} replay  | saved {OUT}  ({time.time()-T_START:.0f}s)", flush=True)
