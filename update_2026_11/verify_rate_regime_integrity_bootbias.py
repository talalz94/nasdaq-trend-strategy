"""
INTEGRITY REVIEW of update_2026_11/rate_regime.py  --  step 3: is the Calmar bootstrap a valid test?

The specialist's paired circular block bootstrap (63-day blocks) gives a MEAN Calmar difference
that is far from the in-sample point estimate (F2 bond trend: -0.006 vs +0.063), i.e. the
statistic is strongly biased under resampling (maxDD is path-dependent; 63-day blocks cannot
rebuild a 13-month drawdown like 2022). This script:
  1. reproduces the specialist's draws exactly (seed 0, 63d, 2000 draws) for bond trend vs IEF,
  2. reports percentile CI (what the specialist used), basic (bias-reversed) CI, and the
     bias-corrected P(gain > 0.07),
  3. repeats with longer blocks (126 / 252 / 504 days) to show how the answer moves.
Writes nothing inside the project.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

P = C.panel(); idx = P.index


def ma(s, n):
    return s.rolling(n, min_periods=n).mean()


ief_px = (1 + P["ief"].fillna(0)).cumprod()
bt = (ief_px < ma(ief_px, 200)).astype(float)
ALLOC = {"static_ief": pd.DataFrame({"ief": 1.0}, index=idx),
         "bondtrend200_bills": pd.DataFrame({"ief": 1 - bt, "bills": bt}),
         "static_tlt": pd.DataFrame({"tlt": 1.0}, index=idx)}
STR = {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}


def run(s, al):
    lev, vt = STR[s]
    w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off_weights=al)
    return C.backtest(P, w)


def calmar_cols(X):
    e = np.cumprod(1 + X, axis=0)
    dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
    g = e[-1] ** (252 / len(X)) - 1
    return g / np.abs(dd)


for s in STR:
    R = pd.concat({p: run(s, a) for p, a in ALLOC.items()}, axis=1)
    R = R[(R.index >= C.START) & (R.index <= C.FREEZE)].dropna()
    A = R.values; T = len(A)
    pt = calmar_cols(A)
    for j, name in [(1, "bondtrend200_bills"), (2, "static_tlt")]:
        d0 = pt[j] - pt[0]
        for block in (63, 126, 252, 504):
            rng = np.random.default_rng(0); nb = int(np.ceil(T / block)); D = np.empty(2000)
            for i in range(2000):
                st = rng.integers(0, T, nb)
                ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
                c = calmar_cols(A[ix]); D[i] = c[j] - c[0]
            lo, hi = np.percentile(D, [2.5, 97.5]); bias = D.mean() - d0
            print(f"{s} {name:20s} block {block:3d}: point {d0:+.3f} | boot mean {D.mean():+.3f} (bias {bias:+.3f}) | "
                  f"percentile CI [{lo:+.3f},{hi:+.3f}] P(<=0) {np.mean(D<=0):.3f} P(>0.07) {np.mean(D>0.07):.3f} | "
                  f"basic CI [{2*d0-hi:+.3f},{2*d0-lo:+.3f}] bias-corr P(>0.07) {np.mean(D-bias>0.07):.3f} P(<=0) {np.mean(D-bias<=0):.3f}")
