"""
verify_managed_futures_robustness_realfunds.py -- does the proxy's verdict on the OFF-LEG SWAP survive when the
TSMOM leg is a REAL managed-futures fund with a longer live record than DBMF/KMLM?

Real funds (Yahoo adjusted NAV, net of their 1-2% fees; fetched in memory, cached ONLY in the scratch dir):
  RYMFX (Guggenheim Managed Futures, 2007-03), AQMIX (AQR Managed Futures I, 2010-01),
  ASFYX (AlphaSimplex Managed Futures Y, 2010-07), PQTIX (PIMCO TRENDS MF Instl, 2011-01), WTMF (WisdomTree MF ETF, 2011-01)
For each fund window (first full month after inception .. FREEZE 2026-07-24) compare, for F3 and F2:
  baseline (IEF off-leg) | off-leg = real fund | off-leg = proxy main_12m | +20% real-fund sleeve | +20% proxy sleeve
plus the episodes the specialist blames (2011 EU, 2018 Q4) and 2022. Nothing is selected; reported as found.

usage: python update_2026_11/verify_managed_futures_robustness_realfunds.py <scratch_dir>
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = os.path.abspath(sys.argv[1])
CACHE = os.path.join(SCR, "verify_mf_realfunds_prices.parquet")
P = C.panel().copy()
idx = P.index
FREEZE = C.FREEZE
D = pd.read_parquet(os.path.join(SCR, "managed_futures_rerun_series.parquet"))
P["ts_main_12m"] = D["prox::main_12m"]
P["ts_blend"] = D["prox::blend_1_3_12m"]
P["ts_exeq"] = D["prox::ex_equities"]
TICK = ["RYMFX", "AQMIX", "ASFYX", "PQTIX", "WTMF"]

if os.path.exists(CACHE):
    PX = pd.read_parquet(CACHE)
else:
    import yfinance as yf
    cols = {}
    for t in TICK:
        try:
            h = yf.Ticker(t).history(period="max", auto_adjust=True)["Close"].dropna()
            h.index = h.index.tz_localize(None).normalize()
            cols[t] = h[~h.index.duplicated()]
            print(f"fetched {t}: {h.index[0].date()}..{h.index[-1].date()} n={len(h)}")
        except Exception as e:
            print("fetch failed", t, repr(e))
    PX = pd.DataFrame(cols)
    PX.to_parquet(CACHE)

NV = 0
res = {}
for t in PX.columns:
    s = PX[t].dropna()
    if len(s) < 500:
        continue
    r = s.reindex(idx).ffill().pct_change()
    r[idx <= s.index[0]] = np.nan
    big = r[r.abs() > 0.06]
    col = t.lower()
    P[col] = r
    a = (s.index[0] + pd.offsets.MonthBegin(1)).normalize()
    a = max(a, pd.Timestamp("2007-04-01"))
    # standalone sanity: yearly returns + suspicious jumps (unadjusted distributions would show as big drops)
    yrs = {y: C.period_return(r, f"{y}-01-01", f"{y}-12-31") for y in range(a.year, 2026)}
    print(f"\n=== {t}: window {a.date()}..{FREEZE.date()}  |daily ret|>6%: {len(big)} "
          + ("" if len(big) == 0 else str([(str(k.date()), round(v, 3)) for k, v in big.items()][:6])))
    print("  calendar years: " + ", ".join(f"{y}:{v*100:+.1f}" for y, v in yrs.items()))
    pr = D["prox::main_12m"]
    m = (idx >= a) & (idx <= FREEZE)
    wk = lambda x: (1 + x).groupby(x.index.to_period("W-FRI")).prod() - 1
    cw = wk((r - P["rf_d"])[m]).corr(wk((pr - P["rf_d"])[m]))
    cwb = wk((r - P["rf_d"])[m]).corr(wk((D["prox::blend_1_3_12m"] - P["rf_d"])[m]))
    print(f"  weekly excess corr with proxy main_12m {cw:.2f}, blend {cwb:.2f};  fund CAGR {C.stats(r, a, FREEZE, start=None)['cagr']*100:.1f}% "
          f"vs proxy {C.stats(pr, a, FREEZE, start=None)['cagr']*100:.1f}%")
    for ep, (ea, eb) in (("2011 EU", ("2011-04-01", "2011-12-31")), ("2018 Q4", ("2018-10-01", "2018-12-31")),
                         ("2015-08..2016-01", ("2015-08-01", "2016-01-31")), ("2022", ("2022-01-01", "2022-12-31"))):
        if pd.Timestamp(ea) >= a:
            print(f"  episode {ep:17s} fund {C.period_return(r, ea, eb)*100:+6.1f}%  proxy main {C.period_return(pr, ea, eb)*100:+6.1f}%  "
                  f"blend {C.period_return(D['prox::blend_1_3_12m'], ea, eb)*100:+6.1f}%  IEF {C.period_return(P['ief'], ea, eb)*100:+6.1f}%")
    for e, (lev, vt) in {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}.items():
        rb = C.backtest(P, C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off="ief"))
        sb = C.stats(rb, a, FREEZE, start=None)
        line = [f"base Cal {sb['calmar']:.2f} (CAGR {sb['cagr']*100:.1f} DD {sb['maxdd']*100:.1f})"]
        for lbl, offc in ((f"off={t}", col), ("off=proxy", "ts_main_12m"), ("off=blend", "ts_blend")):
            rr = C.backtest(P, C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off=offc))
            if offc == col:
                NV += 1
            s2 = C.stats(rr, a, FREEZE, start=None)
            e11 = C.period_return(rr, "2011-04-01", "2011-12-31") if a <= pd.Timestamp("2011-04-01") else np.nan
            e18 = C.period_return(rr, "2018-10-01", "2018-12-31")
            line.append(f"{lbl} dCal {s2['calmar']-sb['calmar']:+.2f} (DD {s2['maxdd']*100:.1f}, 2011 {e11*100:+.1f}, 18Q4 {e18*100:+.1f})")
            res[(t, e, lbl)] = s2["calmar"] - sb["calmar"]
        for lbl, offc in ((f"+20% {t}", col), ("+20% proxy", "ts_main_12m")):
            w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off="ief") * 0.8
            w[offc] = 0.2
            rr = C.backtest(P, w)
            if offc == col:
                NV += 1
            s2 = C.stats(rr, a, FREEZE, start=None)
            line.append(f"{lbl} dCal {s2['calmar']-sb['calmar']:+.2f}")
            res[(t, e, lbl)] = s2["calmar"] - sb["calmar"]
        b11 = C.period_return(rb, "2011-04-01", "2011-12-31") if a <= pd.Timestamp("2011-04-01") else np.nan
        print(f"  {e}: base 2011 {b11*100:+.1f} 18Q4 {C.period_return(rb, '2018-10-01', '2018-12-31')*100:+.1f} | " + " | ".join(line))
    # pre-2019 sub-window (before DBMF existed): is the swap still positive with the real fund?
    b2 = pd.Timestamp("2018-12-31")
    if a < pd.Timestamp("2016-01-01"):
        for e, (lev, vt) in {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}.items():
            rb = C.backtest(P, C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off="ief"))
            sb = C.stats(rb, a, b2, start=None)
            out = []
            for lbl, offc in ((f"off={t}", col), ("off=proxy", "ts_main_12m")):
                rr = C.backtest(P, C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off=offc))
                s2 = C.stats(rr, a, b2, start=None)
                out.append(f"{lbl} Cal {s2['calmar']:.2f} (DD {s2['maxdd']*100:.1f})")
            print(f"  {e} {a.date()}..2018-12-31 (pre-DBMF, ex-2022): base Cal {sb['calmar']:.2f} (DD {sb['maxdd']*100:.1f}) | " + " | ".join(out))
print(f"\nreviewer variants (real-fund portfolios) evaluated: {NV}")
