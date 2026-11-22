"""
Fidelity check (main session): common.backtest() holds target weights constant every day
between weekly marks, i.e. it implicitly rebalances DAILY for free. A real weekly portfolio
lets holdings drift between Fridays. Simulate the true drift portfolio and compare.
Trades happen at the close of each mark day (same timing convention as lag=1).
"""
import sys, json
sys.path.insert(0, "update_2026_11")
import numpy as np, pandas as pd
import common as C

P = C.panel()

def drift_sim(w, cost=C.COST, mark="mark"):
    m = P[mark].values
    cols = list(w.columns)
    R = np.nan_to_num(P[cols].values)            # daily returns per sleeve
    W = w.values
    T, n = R.shape
    h = np.zeros(n); V = 1.0; out = np.zeros(T); started = False
    for t in range(T):
        # 1) today's market move on current holdings
        if started:
            h = h * (1 + R[t])
            Vn = h.sum()
            out[t] = Vn / V - 1
            V = Vn
        # 2) at the close of a mark day: rebalance to today's target
        if m[t]:
            tgt = W[t] * V
            tc = np.abs(tgt - h).sum() * cost
            V = V - tc
            h = W[t] * V
            if started: out[t] = (V / (V + tc)) * (1 + out[t]) - 1 if out[t] != 0 else -tc / (V + tc)
            started = True
    return pd.Series(out, index=P.index)

rows = []
for name, kw in [("F3-IEF", dict(lev="tqqq", voltgt=0.20)),
                 ("F2-IEF", dict(lev="qld", voltgt=0.30)),
                 ("F3-TLT (scanner)", dict(lev="tqqq", voltgt=0.20, off="tlt"))]:
    w = C.strategy_weights(P, **kw)
    r_cm = C.backtest(P, w)
    r_dr = drift_sim(w)
    s1, s2 = C.stats(r_cm), C.stats(r_dr)
    c1, c2 = C.crises(r_cm), C.crises(r_dr)
    rows.append(dict(name=name, constant_mix=s1, drift=s2,
                     crises_cm={k: c1[k] for k in ["2022 inflation", "COVID 2020", "dot-com 2000-02", "OOS Jul-Nov 2026"]},
                     crises_drift={k: c2[k] for k in ["2022 inflation", "COVID 2020", "dot-com 2000-02", "OOS Jul-Nov 2026"]}))
    print(f"{name:18s} constant-mix: {C.fmt(s1)}")
    print(f"{'':18s} weekly drift: {C.fmt(s2)}   (CAGR diff {(s2['cagr']-s1['cagr'])*100:+.2f}pp, DD diff {(s2['maxdd']-s1['maxdd'])*100:+.1f}pp)")
    print(f"{'':18s} crises cm {[round(v*100,1) for v in rows[-1]['crises_cm'].values()]}  drift {[round(v*100,1) for v in rows[-1]['crises_drift'].values()]}")
json.dump(rows, open("update_2026_11/main_drift_check.json", "w"), indent=1)
print("saved update_2026_11/main_drift_check.json")
