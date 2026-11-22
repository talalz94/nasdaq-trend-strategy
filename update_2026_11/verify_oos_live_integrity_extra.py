"""
INTEGRITY REVIEW of oos_live -- step 3: follow-up probes (prints only, writes nothing).

 1. post-2000 / post-2010 maxDD and Calmar of F3/F2 and the rule-removal variants at lag 1 and lag 2
    (the headline F3 IS maxDD is set on 1990-05-02, inside the era the specialist calls lag-1-flattered)
 2. bonds vs stocks inside the risk-on phase: did they fall TOGETHER (claim C6)?
 3. true size of the 'independent' yfinance diff (claim C1: 'within 2e-7')
 4. Monday-open minus Friday-close, per day (claim C8: '+3.47pp on 10-19')
 5. frequency of F3 weeks <= -12% (claim C3 / rec. 7 'once every 8 years')
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # read-only use of the shared mechanics (same as the specialist)

P = C.panel()
idx = P.index
FREEZE = C.FREEZE
OOS = idx[idx > FREEZE]


def st(r, a, b):
    s = C.stats(r, a, b, start=None)
    e = (1 + r.loc[a:b].dropna()).cumprod()
    dd = e / e.cummax() - 1
    return f"{s['cagr']*100:5.1f}% / {s['maxdd']*100:6.1f}% [{dd.idxmin().date()}] / {s['calmar']:.2f}"


print("1. maxDD / Calmar by era (model, spliced) -- lag 1 | lag 2")
base_b = dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15)
base_c = dict(lev="qld", voltgt=0.30, off="ief", gold="gold_phys", gold_w=0.15)
VAR = {"as specified": {}, "no Aug-Sep": dict(seasonal=False), "no chop": dict(chop_pos=1.0),
       "no Aug-Sep + no chop": dict(seasonal=False, chop_pos=1.0)}
for nm_b, base in [("F3", base_b), ("F2", base_c)]:
    for nm, ch in VAR.items():
        w = C.strategy_weights(P, **{**base, **ch})
        for lag in [1, 2]:
            r = C.backtest(P, w, lag=lag)
            print(f"  {nm_b} {nm:22s} lag{lag}: 1988-1999 {st(r,'1988-01-01','1999-12-31')} | 2000-freeze "
                  f"{st(r,'2000-01-01',str(FREEZE.date()))} | 2010-freeze {st(r,'2010-01-01',str(FREEZE.date()))} | full {st(r,'1988-01-01',str(FREEZE.date()))}")

print("\n2. bonds vs stocks within the OOS risk-on phase")
seg = {"freeze->record 07-27..10-06": ("2026-07-27", "2026-10-06"),
       "record->exit 10-07..10-16": ("2026-10-07", "2026-10-16"),
       "risk-off 10-19..11-17": ("2026-10-19", "2026-11-17")}
for nm, (a, b) in seg.items():
    x = P.loc[a:b]
    print(f"  {nm}: NDX {((1+x.ndx_ret).prod()-1)*100:+6.2f}%  IEF {((1+x.ief_real).prod()-1)*100:+6.2f}%  "
          f"TLT {((1+x.tlt_real).prod()-1)*100:+6.2f}%  GLD {((1+x.gld_real).prod()-1)*100:+6.2f}%  "
          f"10y {x.tnx.iloc[0]:.3f}->{x.tnx.iloc[-1]:.3f}  daily corr(IEF,NDX) {x.ief_real.corr(x.ndx_ret):+.2f}")
w = C.strategy_weights(P, **base_b)
H = C.held_weights(P, w, lag=1).loc[OOS]
x = P.loc[OOS]
both_down = (x.ndx_ret < 0) & (x.ief_real < 0)
print(f"  OOS days with NDX<0 and IEF<0: {int(both_down.sum())}/{len(x)}; IEF P&L on those days (F3 weights) "
      f"{(H.ief*x.ief_real)[both_down].sum()*100:+.2f}pp vs on NDX-up days {(H.ief*x.ief_real)[x.ndx_ret>0].sum()*100:+.2f}pp")
print("  F3 average IEF weight 07-27..10-06:", round(float(H.ief.loc[:'2026-10-06'].mean()), 3),
      " 10-07..10-16:", round(float(H.ief.loc['2026-10-07':'2026-10-16'].mean()), 3))

print("\n3. 'independent' yfinance check: actual max |daily diff| in the OOS window")
Y = pd.read_parquet(os.path.join(HERE, "oos_live_yf.parquet"))
YC = Y.pivot(index="date", columns="ticker", values="close")
for t, c in [("QQQ", "qqq_real"), ("QLD", "qld_real"), ("TQQQ", "tqqq_real"), ("TLT", "tlt_real"),
             ("IEF", "ief_real"), ("SHY", "shy_real"), ("BIL", "bil_real"), ("GLD", "gld_real")]:
    r = YC[t].dropna().reindex(idx).ffill().pct_change(fill_method=None)
    d = (r.loc[OOS] - P.loc[OOS, c]).abs().max()
    dfull = (r - P[c]).abs().loc["2011-01-01":].max()
    print(f"  {t:5s} OOS max {d:.2e}   2011+ max {dfull:.2e}")

print("\n4. Monday-open minus Friday-close (drift), F3, per day")
YO = Y.pivot(index="date", columns="ticker", values="open")
TK = {"tqqq": "TQQQ", "ief": "IEF", "gold_phys": "GLD"}
cols = list(w.columns)
R = P.loc[:, ["tqqq_real", "ief_real", "gld_real"]].copy()
R.columns = ["tqqq", "ief", "gold_phys"]
ON = pd.DataFrame({k: (YO[t] / YC[t].shift(1) - 1).reindex(idx) for k, t in TK.items()})
d0 = idx[idx >= pd.Timestamp("2026-07-24")]
m = P["mark"].reindex(d0).values
Wv = w.loc[d0, cols].values
Rv = R.loc[d0, cols].values
Ov = ON.loc[d0, cols].values


def sim(mode):
    h = Wv[0].copy()
    out = []
    for i in range(1, len(d0)):
        V0 = h.sum()
        if mode == "close":
            if m[i - 1]:
                cf = 3e-4 * np.abs(Wv[i - 1] - h / V0).sum()
                h = V0 * (1 - cf) * Wv[i - 1]
            h = h * (1 + Rv[i])
        else:
            h = h * (1 + Ov[i])
            if m[i - 1]:
                Vo = h.sum()
                cf = 3e-4 * np.abs(Wv[i - 1] - h / Vo).sum()
                h = Vo * (1 - cf) * Wv[i - 1]
            h = h * (1 + Rv[i]) / (1 + Ov[i])
        out.append(h.sum() / V0 - 1)
    return pd.Series(out, index=d0[1:])


rc, ro = sim("close"), sim("open")
print(f"  totals: close {((1+rc).prod()-1)*100:.3f}%  open {((1+ro).prod()-1)*100:.3f}%")
dl = (ro - rc)
print("  days with |open-close| > 0.2pp:", {str(k.date()): round(v * 100, 2) for k, v in dl[dl.abs() > 0.002].items()},
      " sum of rest:", round(float(dl[dl.abs() <= 0.002].sum() * 100), 2))

print("\n5. F3 weekly (Fri->Fri held week) returns <= -12%, model lag 1, 1988..last")
r = C.backtest(P, w, lag=1)
eq = (1 + r.fillna(0)).cumprod()
mk = idx[P["mark"].values]
wk = (eq.loc[mk] / eq.loc[mk].shift(1) - 1).dropna()
wk = wk[wk.index >= "1988-01-04"]
bad = wk[wk <= -0.12].sort_values()
print("  ", {str(k.date()): round(v * 100, 2) for k, v in bad.items()})
yrs = (wk.index[-1] - wk.index[0]).days / 365.25
print(f"   count {len(bad)} in {yrs:.1f} yrs -> one every {yrs/len(bad):.1f} yrs;  <= -13.88%: {(wk <= -0.1388).sum()} -> one every {yrs/(wk <= -0.1388).sum():.1f} yrs")
