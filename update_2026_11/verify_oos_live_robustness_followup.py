"""
verify_oos_live_robustness_followup.py -- follow-up checks for the oos_live adversarial review.
  A. independent no-gate counterfactual (C5)
  B. sub-phase decomposition of the risk-on phase: rally leg (freeze->record) vs crash leg
     (record->exit) -- does 'stocks and bonds fell together' (C6) describe the risk-on phase?
  C. overlay benefit at lag 2 and by halves at lag 2; paired block bootstrap of the Calmar gap (C4)
  D. path-analog episodes (rally then >=10% fall, peak in 2nd half): F3 per episode (C2)
  E. Poisson interval for the 'once every 8 years' tail-week rate; the June-2026 near miss (C3)
Read-only; prints to stdout only.
"""
import os, sys, json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel().copy()
IDX = P.index
FREEZE = C.FREEZE
OOS = IDX[IDX > FREEZE]
A0, END = OOS[0], OOS[-1]
N = len(OOS)
HIST_A = IDX[IDX >= C.START][0]
REC = pd.Timestamp("2026-10-06")
EXIT = pd.Timestamp("2026-10-19")
nxt = lambda t: IDX[IDX.get_loc(t) + 1]
prv = lambda t: IDX[IDX.get_loc(t) - 1]
tot = lambda r, a, b: float((1 + r.loc[a:b]).prod() - 1)
COUNT = []

B = dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15)
F2 = dict(lev="qld", voltgt=0.30, off="ief", gold="gold_phys", gold_w=0.15)

print("================ A. no-gate counterfactual (own code)")
always = pd.Series(True, index=IDX)
for nm, kw in [("F3", B), ("F2", F2)]:
    w0 = C.strategy_weights(P, **kw)
    w1 = C.strategy_weights(P, **kw, risk_on=always)
    r0, r1 = C.backtest(P, w0), C.backtest(P, w1)
    COUNT += [f"{nm} no-gate lag1"]
    e = pd.concat([pd.Series([1.0], index=[FREEZE]), (1 + r1.loc[OOS]).cumprod()])
    print(f"  {nm}: actual {tot(r0, A0, END)*100:.2f}%  no-gate {tot(r1, A0, END)*100:.2f}%  | risk-off phase actual "
          f"{tot(r0, EXIT, END)*100:.2f}% no-gate {tot(r1, EXIT, END)*100:.2f}% | no-gate OOS maxDD {(e/e.cummax()-1).min()*100:.1f}%"
          f" | no-gate avg lev weight after exit {C.held_weights(P, w1).loc[EXIT:END, kw['lev']].mean():.3f}")

print("\n================ B. sub-phases of the risk-on phase")
legs = [("rally leg  07-27..10-06 (to record)", A0, REC), ("crash leg  10-07..10-16 (record->exit)", nxt(REC), prv(EXIT)),
        ("risk-on    07-27..10-16", A0, prv(EXIT)), ("risk-off   10-19..11-17", EXIT, END)]
for nm, a, b in legs:
    print(f"  {nm}: QQQ {tot(P['qqq_real'], a, b)*100:+6.2f}%  TQQQ {tot(P['tqqq_real'], a, b)*100:+7.2f}%  "
          f"IEF {tot(P['ief_real'], a, b)*100:+5.2f}%  TLT {tot(P['tlt_real'], a, b)*100:+5.2f}%  GLD {tot(P['gld_real'], a, b)*100:+5.2f}%  "
          f"10y {P.at[prv(a), 'tnx']:.2f}->{P.at[b, 'tnx']:.2f}")
J = json.load(open(os.path.join(HERE, "oos_live.json")))
for k in ["b_F3_TQQQ_IEF_GLD", "a_F_scanner_TQQQ_TLT_GLD"]:
    at = J["phase_attribution"][f"{k}|lag1"]
    for seg in ["1a risk-on: freeze -> NDX record close", "1b risk-on: record -> exit"]:
        s = at[seg]
        print(f"  {k[:24]:24s} {seg[:30]:30s} ret {s['compounded_return']*100:+6.2f}%  pnl(pp of start cap): "
              + "  ".join(f"{c} {v*100:+.2f}" for c, v in s["pnl_pct_of_start_capital"].items())
              + "  | avg w: " + "  ".join(f"{c} {v:.2f}" for c, v in s["avg_weights"].items()))
# daily correlation of NDX and IEF in the window
x = P.loc[A0:prv(EXIT), ["ndx_ret", "ief_real"]].dropna()
print(f"  daily corr(NDX, IEF) risk-on phase {x.corr().iloc[0,1]:+.2f}; crash leg "
      f"{P.loc[nxt(REC):prv(EXIT), ['ndx_ret','ief_real']].corr().iloc[0,1]:+.2f}; "
      f"rally leg {P.loc[A0:REC, ['ndx_ret','ief_real']].corr().iloc[0,1]:+.2f}")

print("\n================ C. overlay benefit: lag 2, halves, paired block bootstrap of Calmar")
var = {"spec": B, "neither": {**B, "chop_pos": 1.0, "seasonal": False}}
RR = {}
for lag in [1, 2]:
    for nm, kw in var.items():
        RR[(nm, lag)] = C.backtest(P, C.strategy_weights(P, **kw), lag=lag)
        COUNT.append(f"F3 {nm} lag{lag}")
    for half, (a, b) in {"full": ("1988-01-01", str(FREEZE.date())), "1988-2006": ("1988-01-01", "2006-12-31"),
                         "2007-freeze": ("2007-01-01", str(FREEZE.date()))}.items():
        s1 = C.stats(RR[("spec", lag)], a, b, start=None); s0 = C.stats(RR[("neither", lag)], a, b, start=None)
        print(f"  lag{lag} {half:11s} spec {s1['cagr']*100:5.1f}%/{s1['maxdd']*100:6.1f}%/{s1['calmar']:.2f}   "
              f"neither {s0['cagr']*100:5.1f}%/{s0['maxdd']*100:6.1f}%/{s0['calmar']:.2f}")


def calmar(x):
    e = np.cumprod(1 + x); dd = (e / np.maximum.accumulate(e) - 1).min()
    g = e[-1] ** (252 / len(x)) - 1
    return g / abs(dd), dd


a = pd.concat([RR[("spec", 1)], RR[("neither", 1)]], axis=1).loc[HIST_A:FREEZE].values
T = len(a); rng = np.random.default_rng(7); blk = 126; nb = int(np.ceil(T / blk)); dc, ddd = [], []
for _ in range(1000):
    st = rng.integers(0, T, nb)
    ix = (st[:, None] + np.arange(blk)[None, :]).ravel()[:T] % T
    c1, d1 = calmar(a[ix, 0]); c0, d0 = calmar(a[ix, 1])
    dc.append(c1 - c0); ddd.append(d1 - d0)
dc, ddd = np.array(dc), np.array(ddd)
print(f"  block bootstrap (126d blocks, 1000x) Calmar(spec) - Calmar(neither): mean {dc.mean():+.3f}  95% [{np.percentile(dc,2.5):+.3f}, "
      f"{np.percentile(dc,97.5):+.3f}]  P(<=0) {(dc<=0).mean():.3f};  maxDD gap (spec - neither) mean {ddd.mean()*100:+.1f}pp  P(<=0) {(ddd<=0).mean():.3f}")

print("\n================ D. path-analog episodes (NDX peak >= +6% above start, end >= 10% below the peak, peak in 2nd half)")
s = P["ndx_ret"].loc[HIST_A:FREEZE].fillna(0).values
e = np.concatenate([[1.0], np.cumprod(1 + s)])
Wl = swv(e, N + 1); rel = Wl / Wl[:, :1]
peak = rel.max(1) - 1; fpk = rel[:, -1] / rel.max(1) - 1; apk = rel.argmax(1); endr = rel[:, -1] - 1
r3 = C.backtest(P, C.strategy_weights(P, **B)).loc[HIST_A:FREEZE].fillna(0).values
e3 = np.concatenate([[1.0], np.cumprod(1 + r3)]); f3 = e3[N:] / e3[:-N] - 1
m = (peak >= 0.06) & (fpk <= -0.10) & (apk >= N // 2)
starts = P.loc[HIST_A:FREEZE].index[:len(f3)]
pos = np.flatnonzero(m)
groups = np.split(pos, np.flatnonzero(np.diff(pos) > N) + 1)
rows = []
for g in groups:
    j = g[np.argmin(np.abs(endr[g] - (-0.040)))]           # closest end-return window within the episode
    rows.append((starts[g[0]].date(), len(g), endr[j], peak[j], fpk[j], f3[j], f3[g].min(), np.median(f3[g])))
print(f"  {len(groups)} episodes; per episode: first start | n windows | (closest-end window) NDX end / peak / from-peak | F3 | F3 min / median over episode")
for r_ in rows:
    print(f"    {r_[0]}  n={r_[1]:3d}  NDX {r_[2]*100:+6.1f}% / {r_[3]*100:+5.1f}% / {r_[4]*100:6.1f}%   F3 {r_[5]*100:+6.1f}%   "
          f"F3 min {r_[6]*100:+6.1f}%  median {r_[7]*100:+6.1f}%")
med_ep = np.array([r_[7] for r_ in rows]); close_ep = np.array([r_[5] for r_ in rows])
print(f"  episode-level: share of episodes whose closest-end window F3 <= OOS F3 (-8.99%): {(close_ep <= -0.0899).mean()*100:.1f}% "
      f"({(close_ep <= -0.0899).sum()}/{len(rows)});  episode-median F3 <= OOS: {(med_ep <= -0.0899).sum()}/{len(rows)}")

print("\n================ E. tail-week rate and the June-2026 near miss")
k_ev, yrs = 5, (END - HIST_A).days / 365.25
# exact Poisson 95% CI for 5 events (chi-square quantiles, tabulated): [1.623, 11.668]
lo, hi = 1.623 / yrs, 11.668 / yrs
print(f"  {k_ev} weeks <= -13.88% in {yrs:.1f}y: rate 1 per {yrs/k_ev:.1f}y, exact 95% CI 1 per {1/hi:.1f}y .. 1 per {1/lo:.1f}y")
w = C.strategy_weights(P, **B)
for d in [pd.Timestamp("2026-05-29"), pd.Timestamp("2026-06-05")]:
    print(f"  {d.date()} decision/close: NDX {P.at[d,'ndx']:.0f} vol {P.at[d,'vol20']:.3f} er {P.at[d,'er20']:.3f} "
          f"above {bool(P.at[d,'above'])} TQQQ target {w.at[d,'tqqq']:.3f}")
print(f"  NDX week 05-29 -> 06-05: {P.at[pd.Timestamp('2026-06-05'),'ndx']/P.at[pd.Timestamp('2026-05-29'),'ndx']-1:+.2%}")
print(f"\nreviewer variants evaluated in this follow-up: {len(COUNT)}")
