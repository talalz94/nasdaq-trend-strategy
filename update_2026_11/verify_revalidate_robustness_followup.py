"""
ROBUSTNESS REVIEW of update_2026_11/revalidate.py  --  part 3 (follow-ups)
  * C4 evidence: grid marginal medians of gold_w by half (from revalidate.json)
  * gold 0.20/0.25 at the DEFAULT configs by half / third (does more gold 'help in both halves'?)
  * C10 narrative facts: weights held in the week after 2026-07-31 and the -5.5% NDX week
  * C1: which episode sets the July-proxy F3 max drawdown
  * F2 minus F3 excess-Sharpe bootstrap by half
Writes nothing except stdout.  run: python -u update_2026_11/verify_revalidate_robustness_followup.py
"""
import sys, os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel(); IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
J = json.load(open(os.path.join(HERE, "revalidate.json")))

print("== C4: grid marginal median Calmar by gold_w (revalidate.json plateau section)")
for lev in ["tqqq", "qld"]:
    for wn in ["H1", "H2", "IS"]:
        mm = J["plateau"][f"{lev}|{wn}"]["marginal_median"]["gold_w"]
        print(f"  {lev} {wn}: " + "  ".join(f"{k}:{v:.3f}" for k, v in mm.items()))

print("\n== gold 0.15 vs 0.20 vs 0.25 at the default configs (other knobs fixed)")
SPL = {"IS": (None, FREEZE), "H1": (None, "2006-12-31"), "H2": ("2007-01-01", FREEZE),
       "T1 88-99": (None, "1999-12-31"), "T2 00-12": ("2000-01-01", "2012-12-31"), "T3 13-frz": ("2013-01-01", FREEZE)}
for lev, vt in [("tqqq", 0.20), ("qld", 0.30)]:
    out = {}
    for gw in [0.15, 0.20, 0.25]:
        r = C.backtest(P, C.strategy_weights(P, lev=lev, voltgt=vt, gold_w=gw))
        out[gw] = {k: C.stats(r, a, b)["calmar"] for k, (a, b) in SPL.items()}
        out[gw]["IS_cagr"] = C.stats(r, None, FREEZE)["cagr"]; out[gw]["IS_dd"] = C.stats(r, None, FREEZE)["maxdd"]
    for gw in [0.20, 0.25]:
        d = {k: out[gw][k] - out[0.15][k] for k in SPL}
        print(f"  {lev} gold {gw:.2f} minus 0.15: " + "  ".join(f"{k} {v:+.3f}" for k, v in d.items()) +
              f" | IS CAGR {out[gw]['IS_cagr']*100:.2f}% vs {out[0.15]['IS_cagr']*100:.2f}%, DD {out[gw]['IS_dd']*100:.1f}% vs {out[0.15]['IS_dd']*100:.1f}%")

print("\n== C10 narrative checks")
w3 = C.strategy_weights(P, lev="tqqq", voltgt=0.20)
hw = C.held_weights(P, w3)["tqqq"]
for a, b in [("2026-08-03", "2026-08-07"), ("2026-08-10", "2026-09-30"), ("2026-10-05", "2026-10-09"), ("2026-10-12", "2026-10-16")]:
    s = hw[(hw.index >= a) & (hw.index <= b)]
    nd = P["ndx_ret"][(IDX >= a) & (IDX <= b)]
    print(f"  F3 TQQQ weight held {a}..{b}: {s.min():.3f}-{s.max():.3f} | NDX over window {((1+nd).prod()-1)*100:+.2f}% "
          f"| worst NDX day {nd.min()*100:+.2f}% on {nd.idxmin().date()}")
print("  NDX daily returns 2026-10-09..2026-10-16:", (P["ndx_ret"]["2026-10-09":"2026-10-16"] * 100).round(2).to_dict())

print("\n== C1: max-DD episode of the July-proxy F3 (miners + bond10_old)")
rj = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_miners", off="bond10_old"))
s = rj[(rj.index >= START) & (rj.index <= FREEZE)]
e = (1 + s).cumprod(); dd = e / e.cummax() - 1; t = dd.idxmin(); pk = e[:t].idxmax()
print(f"  July-proxy F3 IS maxDD {dd.min()*100:.1f}% peak {pk.date()} trough {t.date()}; Calmar from 1991: "
      f"{C.stats(rj, '1991-01-01', FREEZE)['calmar']:.3f} vs faithful {C.stats(C.backtest(P, w3), '1991-01-01', FREEZE)['calmar']:.3f}")

print("\n== F2 minus F3 excess-Sharpe bootstrap by window (63-day blocks, n=2000, seed 0)")
F3 = C.backtest(P, w3); F2 = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30))
for wn, (a, b) in {"H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE), "IS": ("1988-01-01", FREEZE)}.items():
    X = pd.concat([F2, F3, P["rf_d"]], axis=1)
    X = X[(X.index >= a) & (X.index <= b)].dropna().values
    xe = X[:, :2] - X[:, 2:3]
    T = len(xe); rng = np.random.default_rng(0); blk = 63; nb = int(np.ceil(T / blk)); out = []; outr = []
    for _ in range(2000):
        st = rng.integers(0, T, nb); ix = (st[:, None] + np.arange(blk)[None, :]).ravel()[:T] % T
        s_ = xe[ix]; sh = s_.mean(0) / s_.std(0, ddof=1) * np.sqrt(252); out.append(sh[0] - sh[1])
        r_ = X[ix, :2]; shr = r_.mean(0) / r_.std(0, ddof=1) * np.sqrt(252); outr.append(shr[0] - shr[1])
    out, outr = np.array(out), np.array(outr)
    print(f"  {wn}: excess dSharpe {out.mean():+.3f} [{np.percentile(out,2.5):+.3f},{np.percentile(out,97.5):+.3f}] P>0 {(out>0).mean():.2f}"
          f" | raw dSharpe {outr.mean():+.3f} [{np.percentile(outr,2.5):+.3f},{np.percentile(outr,97.5):+.3f}] P>0 {(outr>0).mean():.2f}")
