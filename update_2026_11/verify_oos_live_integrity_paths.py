"""
INTEGRITY REVIEW of oos_live -- step 4: robustness of the C2 percentiles (prints only, writes nothing).

 1. execution convention: OOS lag-2 return ranked against lag-2 history (both Monday-close)
 2. 'similar NDX path': OOS NDX rose +11.0% to a peak inside the window, then fell; condition on
    NDX return within 4pp of the OOS AND within-window NDX peak gain >= +6% AND NDX drawdown <= -10%
 3. same, restricted to 2000+ (post-interpolated-gold, post-stale-index era)
No parameter is chosen here; these are diagnostics of the published claims.
"""
import os, sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

P = C.panel()
idx = P.index
FREEZE = C.FREEZE
OOS = idx[idx > FREEZE]
N = len(OOS)
A = idx[idx >= C.START][0]


def win(r, a=A, b=FREEZE):
    s = r.loc[a:b].fillna(0)
    L = np.concatenate([[0.0], np.cumsum(np.log1p(s.values))])
    W = sliding_window_view(L, N + 1) - L[:-N, None] if False else sliding_window_view(L, N + 1)
    base = W[:, :1]
    rel = W - base
    ret = np.expm1(rel[:, -1])
    peak = np.expm1(rel.max(axis=1))
    dd = np.expm1((W - np.maximum.accumulate(W, axis=1)).min(axis=1))
    return ret, peak, dd, s.index[:len(ret)]


def episodes(mask):
    pos = np.flatnonzero(mask)
    return 0 if len(pos) == 0 else int(1 + (np.diff(pos) > N).sum())


nr, npk, ndd, st = win(P["ndx_ret"])
o = P.loc[OOS, "ndx_ret"]
Lo = np.concatenate([[0.0], np.cumsum(np.log1p(o.values))])
o_ret, o_peak, o_dd = np.expm1(Lo[-1]), np.expm1(Lo.max()), np.expm1((Lo - np.maximum.accumulate(Lo)).min())
print(f"OOS NDX: ret {o_ret*100:.2f}%  peak gain {o_peak*100:.2f}%  maxDD {o_dd*100:.2f}%")

S = {"b F3": dict(lev="tqqq", voltgt=0.20, off="ief"), "c F2": dict(lev="qld", voltgt=0.30, off="ief"),
     "a scanner": dict(lev="tqqq", voltgt=0.20, off="tlt")}
PR = P.copy()
for k, c in {"tqqq": "tqqq_real", "qld": "qld_real", "ief": "ief_real", "tlt": "tlt_real", "gold_phys": "gld_real"}.items():
    PR[k] = P[c]
c1 = np.abs(nr - o_ret) <= 0.04
cp = c1 & (npk >= 0.06) & (ndd <= -0.10)
post = st >= pd.Timestamp("2000-01-01")
print(f"windows: cond NDX+-4pp {c1.sum()} ({episodes(c1)} ep) | +peak>=6% & DD<=-10% {cp.sum()} ({episodes(cp)} ep) "
      f"| same, 2000+ {(cp & post).sum()} ({episodes(cp & post)} ep)")
for nm, kw in S.items():
    w = C.strategy_weights(P, **kw)
    for lag in [1, 2]:
        hist = C.backtest(P, w, lag=lag)
        oo = float((1 + C.backtest(PR, w, lag=lag).loc[OOS]).prod() - 1)
        hr, _, _, _ = win(hist)
        line = (f"  {nm:10s} lag{lag}: OOS {oo*100:6.2f}% | uncond pct {100*(hr<=oo).mean():5.1f} | 2000+ {100*(hr[post]<=oo).mean():5.1f}"
                f" | NDX+-4pp {100*(hr[c1]<=oo).mean():5.1f} | path-similar {100*(hr[cp]<=oo).mean():5.1f}"
                f" (median {np.median(hr[cp])*100:5.1f}%) | path-similar 2000+ {100*(hr[cp & post]<=oo).mean():5.1f}")
        print(line)
