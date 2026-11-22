"""
INTEGRITY REVIEW of update_2026_11/overlays.py  --  step 3: small follow-ups on specific wording.
  * exact half-sample deltas of the rate-shock x0.5 variants (claim C5 'not positive in either half',
    claim C6 'negative in-sample evidence in both halves') and the bootstrap p-range quoted in C6
  * how continuous the post-freeze rate-shock 'spans' quoted in C6 are
  * F2/F3 dollar-shock full-sample maxDD location (why IS dCal << both half dCal)
Read-only: uses overlays.json + panel.parquet; prints only.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "overlays.json")))
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
idx = P.index
V = {(v["base"], v["name"]): v for v in J["variants"]}

print("rate-shock variants: exact halves (dCal, dSh) and bootstrap p(Sharpe diff<=0)")
for b in ("F3", "F2"):
    for bp in (50, 75, 100):
        for kind in ("engine x0.5", "off-leg IEF->bills"):
            v = V[(b, f"rate shock 10y +{bp}bp/126d: {kind}")]
            h = v["halves"]
            print(f"  {b} {bp:3d}bp {kind:20s} dCal {v['d_calmar']:+.5f} | H1 dCal {h[0]['d_calmar']:+.5f} dSh {h[0]['d_sharpe']:+.5f} | "
                  f"H2 dCal {h[1]['d_calmar']:+.5f} dSh {h[1]['d_sharpe']:+.5f} | pSh {v['boot_vs_base']['sharpe']['p_le0']:.4f} | OOS ov {v['oos']['ov']*100:+.2f}% base {v['oos']['base']*100:+.2f}%")

d126 = P["tnx"] - P["tnx"].shift(126)
for thr, a, b in ((0.50, "2026-07-31", "2026-11-13"), (0.75, "2026-08-28", "2026-10-21")):
    seg = d126[(idx >= a) & (idx <= b)]
    on = seg > thr
    runs = (on != on.shift()).cumsum()
    nruns = int(on.groupby(runs).first().sum())
    print(f"10y d126 > {thr*100:.0f}bp between {a} and {b}: on {int(on.sum())} of {len(seg)} trading days in {nruns} separate runs; "
          f"max {seg.max()*100:.1f}bp on {seg.idxmax().date()}")

print("\ndollar shock: where is the full in-sample maxDD?")
import sys
sys.path.insert(0, HERE)
import common as C
for b, kw in (("F3", dict(lev="tqqq", voltgt=0.20)), ("F2", dict(lev="qld", voltgt=0.30))):
    rb = C.backtest(P, C.strategy_weights(P, **kw))
    raw = (P["dxy"] / P["dxy"].shift(126) - 1) > 0.05
    f = raw.astype(float).shift(1).fillna(0).astype(bool)
    ro = C.backtest(P, C.strategy_weights(P, **kw, extra_scale=pd.Series(np.where(f, 0.75, 1.0), index=idx)))
    for nm, r in (("base", rb), ("dollar", ro)):
        x = r[(r.index >= C.START) & (r.index <= C.FREEZE)]
        e = (1 + x).cumprod(); dd = e / e.cummax() - 1
        trough = dd.idxmin(); peak = e[:trough].idxmax()
        print(f"  {b} {nm:6s} maxDD {dd.min()*100:6.2f}%  peak {peak.date()} trough {trough.date()}")
    for a, z in (("2021-11-19", "2022-12-28"), ("2021-11-19", "2022-01-31"), ("2022-02-01", "2022-12-28")):
        print(f"  {b} {a}..{z}: base {C.period_return(rb, a, z)*100:+.2f}%  dollar {C.period_return(ro, a, z)*100:+.2f}%")
    on = f[(idx >= "2021-09-01") & (idx <= "2022-12-31")]
    s_b, s_o = C.stats(rb, None, C.FREEZE), C.stats(ro, None, C.FREEZE)
    print(f"  lagged dollar flag first ON {on[on].index[0].date()};  IS Calmar base {s_b['calmar']:.4f}, overlay {s_o['calmar']:.4f}, "
          f"overlay CAGR / base maxDD = {s_o['cagr']/abs(s_b['maxdd']):.4f}  (=> CAGR part {s_o['cagr']/abs(s_b['maxdd'])-s_b['calmar']:+.4f}, "
          f"drawdown part {s_o['calmar']-s_o['cagr']/abs(s_b['maxdd']):+.4f})")
