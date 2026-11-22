"""
Shared, side-effect-free backtest mechanics for the Nov-2026 refresh.
Every update_2026_11 script imports this so all results use IDENTICAL mechanics.

    import sys; sys.path.insert(0, "update_2026_11"); import common as C
    P = C.panel()
    w = C.strategy_weights(P, lev="tqqq", gold="gold_phys", off="ief")
    r = C.backtest(P, w)
    C.stats(r)

Mechanics (unchanged from the July research unless a kwarg says otherwise):
  * signal computed on day-t close; weights only change on rebalance marks
    (P.mark = Fridays, identical to engine.py); applied from day t+1  (lag=1).
    lag=2 = conservative "trade next session's close" robustness check.
  * cost = 3 bps x one-way turnover (sum |dw|), charged on the day the new weights start.
  * stats from START (1988-01-01) by default: CAGR, maxDD, Sharpe (rf ignored, as July), Calmar.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH = os.path.join(HERE, "panel.parquet")
START = pd.Timestamp("1988-01-01")
FREEZE = pd.Timestamp("2026-07-24")        # last data day of the July research
COST = 3.0 / 1e4

_P = None
def panel() -> pd.DataFrame:
    global _P
    if _P is None:
        _P = pd.read_parquet(PANEL_PATH)
    return _P


# ------------------------------------------------------------------ strategy
def strategy_weights(P, lev="tqqq", voltgt=0.20, chop_thr=0.25, chop_pos=0.5,
                     seasonal=True, season_months=(8, 9), season_mult=0.5,
                     gold="gold_phys", gold_w=0.15, off="ief",
                     risk_on=None, extra_scale=None, off_weights=None):
    """
    The researched strategy ("Strategy F" family) as a daily target-weight frame.
      risk-on        = NDX > EMA-100     (the July 'confirmed curve' term is a logical no-op:
                       above & ~(inv_recent & ~above) == above, so it is omitted here)
      scale          = min(1, voltgt / 20d realised vol)       (x season_mult in Aug/Sep)
      engine weight  = scale if efficiency-ratio >= chop_thr else scale*chop_pos
      sleeve         = (1-gold_w) * engine weight in `lev`
      remainder      = `off` sleeve  (or a dict/DataFrame of off-leg weights via off_weights)
      gold           = permanent gold_w in `gold` sleeve
    risk_on      : optional boolean Series overriding the EMA-100 gate
    extra_scale  : optional Series in [0,1] multiplied into the engine weight (overlays)
    off_weights  : optional DataFrame of off-leg allocations whose rows sum to 1
                   (e.g. a regime switch between 'ief' and 'bills')
    """
    idx = P.index
    scale = (voltgt / P["vol20"]).clip(0, 1).fillna(1.0)
    if seasonal:
        scale = (scale * np.where(P["month"].isin(season_months), season_mult, 1.0)).clip(0, 1)
    if extra_scale is not None:
        scale = (scale * extra_scale.reindex(idx).fillna(1.0)).clip(0, 1)
    ron = P["above"] if risk_on is None else risk_on.reindex(idx).fillna(False).astype(bool)
    clean = (P["er20"] >= chop_thr).fillna(False)
    eq = (ron & clean) * scale + (ron & ~clean) * scale * chop_pos
    core = 1.0 - gold_w
    w = pd.DataFrame(0.0, index=idx, columns=[lev])
    w[lev] = eq * core
    rest = (core - eq * core).clip(lower=0)
    if off_weights is not None:
        ow = off_weights.reindex(idx).ffill().fillna(0)
        for c in ow.columns:
            w[c] = w.get(c, 0.0) + rest * ow[c]
    elif off is not None:
        w[off] = w.get(off, 0.0) + rest
    if gold_w > 0:
        w[gold] = w.get(gold, 0.0) + gold_w
    return w


def backtest(P, w, cost=COST, lag=1, mark="mark"):
    """daily net returns of target weights w (columns = panel return columns)."""
    m = P[mark] if isinstance(mark, str) else mark
    ww = w.where(m).ffill().fillna(0).shift(lag).fillna(0)
    gross = sum(ww[c] * P[c].fillna(0) for c in ww.columns)
    turn = ww.diff().abs().sum(axis=1).fillna(0)
    return gross - turn * cost


def held_weights(P, w, lag=1, mark="mark"):
    m = P[mark] if isinstance(mark, str) else mark
    return w.where(m).ffill().fillna(0).shift(lag).fillna(0)


# ------------------------------------------------------------------ statistics
def stats(r, a=None, b=None, start=START):
    r = r.copy()
    if a is not None:
        r = r[r.index >= pd.Timestamp(a)]
    if b is not None:
        r = r[r.index <= pd.Timestamp(b)]
    if start is not None:
        r = r[r.index >= start]
    r = r.dropna()
    if len(r) < 20:
        return dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan, calmar=np.nan, total=np.nan, days=len(r))
    e = (1 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1/365.25)
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    return dict(cagr=float(cagr), maxdd=float(dd),
                sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                calmar=float(cagr / abs(dd)) if dd < 0 else np.nan,
                total=float(e.iloc[-1] - 1), days=int(len(r)))


def period_return(r, a, b):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    return float((1 + s).prod() - 1) if len(s) else np.nan


DECADES = {"1988-89": ("1988-01-01", "1989-12-31"), "1990s": ("1990-01-01", "1999-12-31"),
           "2000s": ("2000-01-01", "2009-12-31"), "2010s": ("2010-01-01", "2019-12-31"),
           "2020s": ("2020-01-01", "2030-12-31")}
CRISES = {"1987 crash": ("1987-10-01", "1987-12-31"), "1994 bond massacre": ("1994-01-01", "1994-12-31"),
          "1998 LTCM": ("1998-07-01", "1998-10-31"), "dot-com 2000-02": ("2000-03-01", "2002-10-31"),
          "GFC 2007-09": ("2007-10-01", "2009-03-31"), "2011 EU": ("2011-04-01", "2011-12-31"),
          "2018 Q4": ("2018-10-01", "2018-12-31"), "COVID 2020": ("2020-02-15", "2020-04-30"),
          "2022 inflation": ("2022-01-01", "2022-12-31"),
          "2025 tariff shock": ("2025-02-15", "2025-05-31"),
          "OOS Jul-Nov 2026": ("2026-07-25", "2026-12-31")}


def by_decade(r):
    return {k: stats(r, a, b, start=None) for k, (a, b) in DECADES.items()}


def crises(r):
    return {k: period_return(r, a, b) for k, (a, b) in CRISES.items()}


def block_bootstrap_cagr_diff(r1, r2, n=2000, block=63, seed=0, start=START):
    """paired circular block bootstrap of the CAGR difference r1 - r2 (annualised)."""
    a = pd.concat([r1, r2], axis=1).dropna()
    a = a[a.index >= start].values
    T = len(a); rng = np.random.default_rng(seed); out = []
    nb = int(np.ceil(T / block))
    for _ in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = a[ix]
        g = np.prod(1 + s, axis=0) ** (252 / T) - 1
        out.append(g[0] - g[1])
    out = np.array(out)
    return dict(mean=float(out.mean()), lo=float(np.percentile(out, 2.5)),
                hi=float(np.percentile(out, 97.5)), p_le0=float((out <= 0).mean()))


def fmt(s):
    return (f"CAGR {s['cagr']*100:6.2f}%  maxDD {s['maxdd']*100:6.1f}%  "
            f"Calmar {s['calmar']:.2f}  Sharpe {s['sharpe']:.2f}")
