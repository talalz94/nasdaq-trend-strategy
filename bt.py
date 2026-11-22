"""
Backtest engine + strategy library for the WaveTrend / Squeeze study.

Conventions (no lookahead):
  - Signals are evaluated at bar CLOSE i, position is opened at OPEN of bar i+1.
  - Stops/targets are checked intrabar with high/low; if both hit in one bar we
    assume the STOP fills first (conservative).
  - One position per ticker at a time (no pyramiding).
  - Returns are net of `cost_bps` per side (entry+exit).
Metrics are pooled across all tickers within a timeframe.
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

CACHE = os.environ.get("CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
os.makedirs(CACHE, exist_ok=True)                  # every research script writes its results here

OB1, OB2, OS1, OS2 = 60, 53, -60, -53


# ----------------------------------------------------------------------------- data
def load_cache(iv: str) -> dict[str, dict]:
    df = pd.read_parquet(f"{CACHE}/ind_{iv}.parquet")
    cols = ["open", "high", "low", "close", "wt1", "wt2", "wt_cross_up",
            "wt_cross_dn", "mom", "mom_color", "mom_rising", "sqz_on",
            "sqz_off", "sqz_release", "datetime",
            "ma8", "ma8_up", "ma8_turn_up", "ma8_turn_dn",
            "mashift_osc", "mashift_regime_up", "mashift_sig_up",
            "mashift_sig_dn", "mashift_up"]
    out = {}
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        d = {c: g[c].to_numpy() for c in cols if c in g}
        out[tkr] = d
    return out


# ----------------------------------------------------------------------------- engine
def simulate(d, long_entry, long_exit, short_entry=None, short_exit=None,
             target=None, stop=None, time_bars=None, reverse=False,
             trail=None, cost_bps=2.0):
    """Return list of trade dicts for one ticker. cost_bps is per side."""
    O, H, L, C = d["open"], d["high"], d["low"], d["close"]
    n = len(C)
    c = cost_bps / 1e4
    trades = []
    pos = 0            # 0 flat, +1 long, -1 short
    entry_px = 0.0
    entry_i = 0
    tp = sp = np.nan

    def close_trade(exit_i, exit_px, reason, direction):
        if direction == 1:
            gross = exit_px / entry_px - 1.0
        else:
            gross = entry_px / exit_px - 1.0
        net = gross - 2 * c
        trades.append(dict(entry_i=entry_i, exit_i=exit_i, dir=direction,
                           entry_px=entry_px, exit_px=exit_px, ret=net,
                           bars=exit_i - entry_i, reason=reason))

    i = 0
    while i < n - 1:
        go_long = bool(long_entry[i])
        go_short = bool(short_entry[i]) if short_entry is not None else False
        if not (go_long or go_short):
            i += 1
            continue
        pos = 1 if go_long else -1
        entry_i = i + 1
        entry_px = O[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue
        tp = entry_px * (1 + target) if (target and pos == 1) else (
             entry_px * (1 - target) if (target and pos == -1) else np.nan)
        sp = entry_px * (1 - stop) if (stop and pos == 1) else (
             entry_px * (1 + stop) if (stop and pos == -1) else np.nan)

        j = entry_i
        exit_bar = None      # bar to resume outer scan from (always > i => no reentry loop)
        peak = entry_px      # for trailing stop (long: running max high; short: running min low)
        trough = entry_px
        while j < n:
            hi, lo = H[j], L[j]
            if pos == 1:
                if stop and lo <= sp:
                    close_trade(j, sp, "stop", pos); exit_bar = j; break
                if target and hi >= tp:
                    close_trade(j, tp, "target", pos); exit_bar = j; break
                if trail:
                    peak = max(peak, hi)
                    tsp = peak * (1 - trail)
                    if lo <= tsp and tsp > entry_px * (1 - (stop or 1)):
                        close_trade(j, min(tsp, hi), "trail", pos); exit_bar = j; break
            else:
                if stop and hi >= sp:
                    close_trade(j, sp, "stop", pos); exit_bar = j; break
                if target and lo <= tp:
                    close_trade(j, tp, "target", pos); exit_bar = j; break
                if trail:
                    trough = min(trough, lo)
                    tsp = trough * (1 + trail)
                    if hi >= tsp:
                        close_trade(j, max(tsp, lo), "trail", pos); exit_bar = j; break
            if time_bars and (j - entry_i) >= time_bars:
                close_trade(j, C[j], "time", pos); exit_bar = j; break
            exit_sig = (bool(long_exit[j]) if pos == 1 else
                        (bool(short_exit[j]) if short_exit is not None else False))
            rev_sig = False
            if reverse:
                rev_sig = (short_entry is not None and pos == 1 and bool(short_entry[j])) or \
                          (pos == -1 and bool(long_entry[j]))
            if (exit_sig or rev_sig) and j + 1 < n:
                close_trade(j + 1, O[j + 1], "signal", pos); exit_bar = j + 1; break
            j += 1
        if exit_bar is None:                       # ran off the end
            close_trade(n - 1, C[n - 1], "eod", pos)
            break
        i = exit_bar                               # resume from the exit bar (strictly > entry-1)
    return trades


# ----------------------------------------------------------------------------- metrics
def metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, med=np.nan, pf=np.nan,
                    exp=np.nan, sharpe=np.nan, maxdd=np.nan, avg_bars=np.nan,
                    sum_ret=0.0, avg_win=np.nan, avg_loss=np.nan)
    r = trades["ret"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    gp, gl = wins.sum(), -losses.sum()
    pf = gp / gl if gl > 0 else np.inf
    # equity curve by exit order
    order = trades.sort_values("exit_i") if "exit_i" in trades else trades
    eq = order["ret"].cumsum().to_numpy()
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq - peak).min()) if len(eq) else 0.0
    return dict(
        n=len(r),
        win=float((r > 0).mean()),
        avg=float(r.mean()),
        med=float(np.median(r)),
        pf=float(pf),
        exp=float(r.mean()),
        sharpe=float(r.mean() / r.std()) if r.std() > 0 else np.nan,
        maxdd=maxdd,
        avg_bars=float(trades["bars"].mean()),
        sum_ret=float(r.sum()),
        avg_win=float(wins.mean()) if len(wins) else np.nan,
        avg_loss=float(losses.mean()) if len(losses) else np.nan,
    )


# ----------------------------------------------------------------------------- strategies
def falsearr(d):
    return np.zeros(len(d["close"]), dtype=bool)


def build_signals(d, name):
    """Return dict(long_entry, long_exit, short_entry, short_exit, opts)."""
    cu, cd = d["wt_cross_up"], d["wt_cross_dn"]
    wt2 = d["wt2"]
    mom, color, rising = d["mom"], d["mom_color"], d["mom_rising"]
    mom_prev = np.roll(mom, 1); mom_prev[0] = np.nan
    sqz_rel = d["sqz_release"]
    F = falsearr(d)
    opts = {}

    if name == "WT_cross":                         # baseline reversal
        return dict(le=cu, lx=cd, se=cd, sx=cu, opts=dict(reverse=True))
    if name == "WT_cross_long":                    # long-only crossover
        return dict(le=cu, lx=cd, se=F, sx=F, opts={})
    if name == "WT_oversold_long":                 # bullish cross while oversold
        le = cu & (wt2 < OS2)
        return dict(le=le, lx=cd, se=F, sx=F, opts={})
    if name == "WT_ob_short":                       # bearish cross while overbought
        se = cd & (wt2 > OB2)
        return dict(le=F, lx=F, se=se, sx=cu, opts={})
    if name == "WT_meanrev_both":                   # oversold long + overbought short
        le = cu & (wt2 < OS2); se = cd & (wt2 > OB2)
        return dict(le=le, lx=cd, se=se, sx=cu, opts=dict(reverse=False))
    if name == "SQZ_color_long":                    # neg & turning up (maroon) -> long
        le = (mom < 0) & (mom > mom_prev) & (np.roll(mom,1) <= np.roll(mom,2))
        lx = (mom > 0) & (mom < mom_prev)           # pos & falling -> exit
        return dict(le=le, lx=lx, se=F, sx=F, opts={})
    if name == "SQZ_zero_cross":                    # momentum zero-line cross
        le = (mom > 0) & (mom_prev <= 0)
        lx = (mom < 0) & (mom_prev >= 0)
        se = lx.copy(); sx = le.copy()
        return dict(le=le, lx=lx, se=se, sx=sx, opts=dict(reverse=True))
    if name == "SQZ_fire":                           # squeeze release in mom direction
        le = sqz_rel & (mom > 0)
        se = sqz_rel & (mom < 0)
        lx = (mom < mom_prev) & (mom < 0)            # momentum rolls over
        sx = (mom > mom_prev) & (mom > 0)
        return dict(le=le, lx=lx, se=se, sx=sx, opts={})
    if name == "Confluence_long":                    # WT bull cross + momentum rising
        le = cu & rising
        lx = cd
        return dict(le=le, lx=lx, se=F, sx=F, opts={})
    if name == "Confluence_oversold":                # WT oversold cross + mom turning up
        le = cu & (wt2 < OS2) & rising
        lx = cd
        return dict(le=le, lx=lx, se=F, sx=F, opts={})
    if name == "WT_oversold_ob_exit":                # oversold cross, exit at overbought or bear cross
        le = cu & (wt2 < OS2)
        lx = cd | (wt2 > OB2)
        return dict(le=le, lx=lx, se=F, sx=F, opts={})
    if name == "Confluence_oversold_ob_exit":
        le = cu & (wt2 < OS2) & rising
        lx = cd | (wt2 > OB2)
        return dict(le=le, lx=lx, se=F, sx=F, opts={})
    if name == "Confluence_strict":                  # WT bull cross + mom>0 + not squeezed
        le = cu & (mom > 0) & (~d["sqz_on"])
        lx = cd | ((mom < 0) & (mom < mom_prev))
        se = cd & (mom < 0)
        sx = cu
        return dict(le=le, lx=lx, se=se, sx=sx, opts={})
    # ---------------- phase-2 indicators ----------------
    ma_up = d.get("ma8_up"); ma_tu = d.get("ma8_turn_up"); ma_td = d.get("ma8_turn_dn")
    ms_up = d.get("mashift_sig_up"); ms_dn = d.get("mashift_sig_dn"); ms_reg = d.get("mashift_regime_up")
    if name == "EMA8_slope_long":                    # trend-follow: long while EMA8 rising
        return dict(le=ma_tu, lx=ma_td, se=F, sx=F, opts={})
    if name == "EMA8_slope_both":                    # long up-slope / short down-slope
        return dict(le=ma_tu, lx=ma_td, se=ma_td, sx=ma_tu, opts=dict(reverse=True))
    if name == "MAShift":                            # oscillator diamonds, both-sided
        return dict(le=ms_up, lx=ms_dn, se=ms_dn, sx=ms_up, opts=dict(reverse=True))
    if name == "MAShift_long":                       # long-only oversold-turn
        return dict(le=ms_up, lx=ms_dn, se=F, sx=F, opts={})
    if name == "MAShift_regime_long":                # long-turn only when price above MA
        le = ms_up & ms_reg
        return dict(le=le, lx=ms_dn, se=F, sx=F, opts={})
    raise ValueError(name)


STRATS = ["WT_cross", "WT_cross_long", "WT_oversold_long", "WT_ob_short",
          "WT_meanrev_both", "SQZ_color_long", "SQZ_zero_cross", "SQZ_fire",
          "Confluence_long", "Confluence_oversold", "Confluence_strict"]
STRATS_P2 = ["EMA8_slope_long", "EMA8_slope_both", "MAShift", "MAShift_long",
             "MAShift_regime_long"]


def run_strategy(cache, name, target=None, stop=None, time_bars=None,
                 trail=None, cost_bps=2.0):
    all_trades = []
    for tkr, d in cache.items():
        s = build_signals(d, name)
        tr = simulate(d, s["le"], s["lx"], s.get("se"), s.get("sx"),
                      target=target, stop=stop, time_bars=time_bars, trail=trail,
                      cost_bps=cost_bps, **s.get("opts", {}))
        dt = d.get("datetime")
        for t in tr:
            t["ticker"] = tkr
            if dt is not None:
                t["entry_time"] = dt[t["entry_i"]]
                t["exit_time"] = dt[min(t["exit_i"], len(dt) - 1)]
        all_trades.extend(tr)
    return pd.DataFrame(all_trades)
