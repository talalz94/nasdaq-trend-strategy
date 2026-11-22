"""
Clean, side-effect-free strategy engine (imports do nothing but define).
Everything downstream (geo tests, indicator tests, final validation, scanner)
builds on this so results stay consistent and imports stay fast.

Data it expects (built by letf_data.py / macro_data.py / multi_panel.py / geo_panel.py):
  data_letf/letf_daily.parquet    ndx OHLC, qqq/tqqq/sqqq returns, rf
  data_letf/macro_daily.parquet   curve, vix, dxy, credit, tlt/gold returns
  data_letf/multi_panel.parquet   multi-asset daily returns (incl. synthetic bond, gold miners)
  data_letf/geo_panel.parquet     oil/commodities/dollar/EM regime inputs   (optional)
"""
from __future__ import annotations
import numpy as np, pandas as pd, os
import bt
CACHE = bt.CACHE
COST = 3.0/1e4
START = pd.Timestamp("1988-01-01")     # all sleeves + 12m lookback available


# ------------------------------------------------------------------ data
class Data:
    def __init__(self):
        d = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
        self.d = d
        self.idx = d.index
        self.ndx = d["ndx"]
        self.rf_d = (d["rf"]/252).fillna(0)
        m = pd.read_parquet("data_letf/macro_daily.parquet")
        self.m = m.reindex(self.idx)
        P = pd.read_parquet("data_letf/multi_panel.parquet").reindex(self.idx)
        self.P = P
        # tradeable sleeves
        self.R = {
            "tqqq": d["tqqq_ret"].fillna(0),
            "qqq":  d["qqq_ret"].fillna(0),
            "sqqq": d["sqqq_ret"].fillna(0),
            "cash": self.rf_d,
            "bond": P["bond"].fillna(0),
            "gold": P["gold"].fillna(0),
            "spx":  P["spx"].fillna(0),
            "rut":  P["rut"].fillna(0),
            "intl": P["intl"].fillna(0),
            "hsi":  P["hsi"].fillna(0),
            "tsx":  P["tsx"].fillna(0),
        }
        self.geo = None
        if os.path.exists("data_letf/geo_panel.parquet"):
            self.geo = pd.read_parquet("data_letf/geo_panel.parquet").reindex(self.idx)
        # ---- core signals ----
        self.ema100 = self.ndx.ewm(span=100, adjust=False).mean()
        self.above = self.ndx > self.ema100
        self.vol = self.ndx.pct_change().rolling(20).std()*np.sqrt(252)
        self.er = (self.ndx.diff(20).abs() /
                   self.ndx.diff().abs().rolling(20).sum().replace(0, np.nan))
        self.curve = self.m["curve"]
        inv = (self.curve < 0).fillna(False)
        self.inv_recent = inv.rolling(252, min_periods=1).max().astype(bool)
        self.confirmed = (self.inv_recent & ~self.above)      # leading recession signal
        self.vix = self.m["vix"].ffill()
        # weekly rebalance marks
        self.mark = pd.Series(False, index=self.idx)
        self.mark[self.ndx.resample("W-FRI").last().index.intersection(self.idx)] = True

    # -------------------------------------------------------------- helpers
    def px(self, sleeve):     # synthetic price series of a sleeve (for trend/momentum)
        return (1+self.R[sleeve].fillna(0)).cumprod()

    def mom(self, sleeve, n=252):
        p = self.px(sleeve); return p/p.shift(n)-1

    def above_ma(self, sleeve, n=200):
        p = self.px(sleeve); return p > p.rolling(n).mean()


_D = None
def data() -> Data:
    global _D
    if _D is None: _D = Data()
    return _D


# ------------------------------------------------------------------ strategy
def weights(voltgt=0.20, chop_thr=0.25, chop_pos=0.5, use_curve=True,
            off="bond", gold_w=0.15, extra_veto=None, gold_series=None,
            lev="tqqq", D=None):
    """
    The canonical strategy weight matrix (Phase C final by default).
      voltgt/chop_thr/chop_pos : Phase-B tuned engine
      use_curve                : confirmed-curve risk-off
      off                      : where non-equity capital sits ('bond'|'cash'|'gold')
      gold_w                   : permanent gold sleeve (0 to disable)
      extra_veto               : optional boolean Series; True = force risk-off
      gold_series              : optional Series of time-varying gold weight (overrides gold_w)
    """
    D = D or data()
    idx = D.idx
    scale = (voltgt/D.vol).clip(0, 1).fillna(1.0)
    ron = D.above.copy()
    if use_curve: ron = ron & ~D.confirmed
    if extra_veto is not None: ron = ron & ~extra_veto.reindex(idx).fillna(False)
    clean = (D.er >= chop_thr).fillna(False)
    eq = (ron & clean)*scale + (ron & ~clean)*scale*chop_pos      # equity-sleeve weight
    gw = (gold_series.reindex(idx).fillna(0) if gold_series is not None
          else pd.Series(gold_w, index=idx))
    gw = gw.clip(0, 0.6)
    core = 1.0 - gw
    cols = sorted(set([lev, off, "gold", "cash"]))
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w[lev] = eq*core
    w[off] = w[off] + (core - eq*core).clip(lower=0)
    w["gold"] = w["gold"] + gw
    return w


def backtest(w, D=None):
    D = D or data()
    w = w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    gross = sum(w[c]*D.R[c].fillna(0) for c in w.columns)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    return gross - turn*COST


def stats(r, a=None, b=None, start=START):
    r = r.copy()
    if a is not None:
        r = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    if start is not None:
        r = r[r.index >= start]
    r = r.dropna()
    if len(r) < 60:
        return dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan, calmar=np.nan, total=np.nan)
    e = (1+r).cumprod(); yrs = (r.index[-1]-r.index[0]).days/365.25
    cagr = e.iloc[-1]**(1/yrs)-1; dd = (e/e.cummax()-1).min()
    return dict(cagr=float(cagr), maxdd=float(dd),
                sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                calmar=float(cagr/abs(dd)) if dd < 0 else np.nan,
                total=float(e.iloc[-1]-1))


def run(**kw):
    return backtest(weights(**kw))


DECADES = {"1988-89": ("1988-01-01", "1989-12-31"), "1990s": ("1990-01-01", "1999-12-31"),
           "2000s": ("2000-01-01", "2009-12-31"), "2010s": ("2010-01-01", "2019-12-31"),
           "2020s": ("2020-01-01", "2026-07-24")}
WINDOWS = {"1992-95": ("1992-01-01", "1995-12-31"), "2000-2010": ("2000-01-01", "2010-12-31")}


def by_decade(r):
    return {k: stats(r, a, b) for k, (a, b) in DECADES.items()}


def eqpts(r, start=START):
    e = (1+r[r.index >= start].fillna(0)).cumprod().resample("ME").last().dropna()
    return [{"t": t.strftime("%Y-%m"), "e": round(float(v), 4)} for t, v in e.items()]


def proj1000(r, start=START):
    e = (1+r[r.index >= start].fillna(0)).cumprod().values
    out = {}
    for hn, dd in [("1 yr", 252), ("3 yr", 756), ("5 yr", 1260), ("10 yr", 2520)]:
        if len(e) <= dd: continue
        w = e[dd:]/e[:-dd]
        out[hn] = {"typ": round(1000*float(np.median(w))),
                   "lo": round(1000*float(np.percentile(w, 10))),
                   "hi": round(1000*float(np.percentile(w, 90)))}
    return out
