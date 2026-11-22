"""
r2_verify_model_hygiene_robustness_driftval.py -- validates the weekly-drift backtest used in
r2_verify_model_hygiene_robustness_duration.py: with targets forward-filled from the Friday marks and a reset on EVERY
day (cost 0), the drift engine must reproduce common.backtest (constant mix, cost 0) exactly.  Read-only.
usage: python -u update_2026_11/r2_verify_model_hygiene_robustness_driftval.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel()


def drift_bt(PP, W, mark, cost=0.0):
    cols = list(W.columns)
    R = np.column_stack([PP[c].fillna(0).values for c in cols])
    Wv = W[cols].values
    m = np.asarray(mark, bool)
    cur = np.zeros(len(cols))
    rp = np.zeros(len(R))
    pend = 0.0
    for t in range(len(R)):
        g = float(cur @ R[t])
        rp[t] = g - pend
        pend = 0.0
        tot = cur.sum() + g
        endw = cur * (1 + R[t]) / tot if tot > 0 else cur
        if m[t]:
            pend = np.abs(Wv[t] - endw).sum() * cost
            cur = Wv[t].copy()
        else:
            cur = endw
    return pd.Series(rp, index=PP.index)


for tn, kw in (("F1", dict(lev="qqq", voltgt=0.20)), ("F3", dict(lev="tqqq", voltgt=0.20))):
    w = C.strategy_weights(P, **kw)
    ref = C.backtest(P, w, cost=0.0)
    tgt = w.where(P["mark"]).ffill().fillna(0)          # Friday targets held through the week
    dr = drift_bt(P, tgt, np.ones(len(P), bool))         # reset every day -> constant mix
    print(f"{tn}: max |drift-engine(daily reset) - common.backtest| = {np.max(np.abs(dr.values - ref.values)):.2e}")
