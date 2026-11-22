"""
ROBUSTNESS REVIEW of update_2026_11/overlays.py  --  step 3: single-episode dependence of the Calmar gains.

Which drawdown episode carries each 'best' Calmar number in overlays.json?
  * dollar shock x0.75 (F3 +0.005, F2 +0.062)    * rate shock 100bp -> bills (F2 +0.077)
For each: IS max-drawdown episode of base and overlay, the overlay's depth inside the base's max-DD episode,
the return difference in the sub-window where they diverge, and the IS Calmar with that one episode's
returns set equal to the base (i.e. 'what is left without this episode').
usage: python update_2026_11/verify_overlays_robustness_followup.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel()
idx = P.index
IS = (C.START, C.FREEZE)
BASES = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
COMMON = dict(gold="gold_phys", gold_w=0.15, off="ief", chop_thr=0.25, chop_pos=0.5, seasonal=True)


def run(base, **kw):
    return C.backtest(P, C.strategy_weights(P, **{**COMMON, **BASES[base], **kw}))


def lag1(flag):
    return flag.reindex(idx).astype(float).shift(1).fillna(0.0).astype(bool)


def dd_window(r, a, b):
    x = r[(r.index >= a) & (r.index <= b)]
    e = (1 + x).cumprod()
    return float((e / e.cummax() - 1).min())


def dd_ep(r):
    x = r[(r.index >= IS[0]) & (r.index <= IS[1])]
    e = (1 + x).cumprod()
    d = e / e.cummax() - 1
    t = d.idxmin()
    return float(d.min()), e.loc[:t].idxmax(), t


usd = lag1(P["dxy"] / P["dxy"].shift(126) - 1 > 0.05)
rs100 = lag1(P["tnx"] - P["tnx"].shift(126) > 1.0)
cases = {
    "dollar x0.75": lambda b: run(b, extra_scale=pd.Series(np.where(usd.values, 0.75, 1.0), index=idx)),
    "rate100 -> bills": lambda b: run(b, off_weights=pd.DataFrame({"ief": 1.0 - rs100.astype(float).values,
                                                                     "bills": rs100.astype(float).values}, index=idx)),
}
for name, f in cases.items():
    for b in BASES:
        rb, r = run(b), f(b)
        sb, so = C.stats(rb, *IS, start=None), C.stats(r, *IS, start=None)
        ddb, pb, tb = dd_ep(rb)
        ddo, po, to = dd_ep(r)
        inside = dd_window(r, pb, tb)
        print(f"\n{name:16s} {b}: IS Calmar {so['calmar']:.4f} vs {sb['calmar']:.4f} ({so['calmar']-sb['calmar']:+.4f}); "
              f"CAGR {so['cagr']*100:.2f} vs {sb['cagr']*100:.2f}")
        print(f"   base max-DD {ddb*100:.2f}% {pb.date()}->{tb.date()} ; overlay max-DD {ddo*100:.2f}% {po.date()}->{to.date()} ; "
              f"overlay depth inside base episode {inside*100:.2f}%")
        # neutralise the base's max-DD episode (overlay returns := base returns inside it) and recompute
        r2 = r.copy()
        m = (r2.index >= pb) & (r2.index <= tb)
        r2[m] = rb[m]
        s2 = C.stats(r2, *IS, start=None)
        print(f"   with the base max-DD episode neutralised: Calmar {s2['calmar']:.4f} ({s2['calmar']-sb['calmar']:+.4f}) "
              f"CAGR {s2['cagr']*100:.2f}  maxDD {s2['maxdd']*100:.2f}%")
        if name.startswith("dollar"):
            for a0, b0 in (("2021-11-19", "2021-12-31"), ("2022-01-01", "2022-12-28"), ("1995-07-17", "1996-03-08"),
                           ("1989-10-09", "1990-05-02")):
                print(f"   {a0}..{b0}: overlay {C.period_return(r, a0, b0)*100:+.2f}%  base {C.period_return(rb, a0, b0)*100:+.2f}%  "
                      f"| depth overlay {dd_window(r, a0, b0)*100:.2f}% base {dd_window(rb, a0, b0)*100:.2f}%")
        flag = usd if name.startswith("dollar") else rs100
        on = flag[(flag.index >= pb) & (flag.index <= tb)]
        print(f"   flag on {int(on.sum())} of {len(on)} days inside the base max-DD episode")
