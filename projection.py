"""
$1000 -> ? over 1..10 yr for the best strategies.
Honest method: take each strategy's ACTUAL daily returns over 1985-2026, then look at
EVERY rolling N-year window (not just the average) -> distribution of outcomes.
Reports typical (median) + range (10th-90th pct) + worst window, plus the naive
"compound the average CAGR" number for reference, and the worst drawdown dip.
"""
import numpy as np, pandas as pd
d = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
ndx = d["ndx"]; rf_d = (d["rf"]/252).fillna(0)
RET = {"tqqq": d["tqqq_ret"].fillna(0), "qqq": d["qqq_ret"].fillna(0), "cash": rf_d}
ema100 = ndx.ewm(span=100, adjust=False).mean()
on = (ndx > ema100)
vol20 = ndx.pct_change().rolling(20).std()*np.sqrt(252)
mark = pd.Series(False, index=ndx.index); mark[ndx.resample("W-FRI").last().index.intersection(ndx.index)] = True
COST = 3.0/1e4

def strat_ret(kind):
    w = pd.DataFrame(0.0, index=ndx.index, columns=["tqqq","qqq","cash"])
    if kind == "champ":                     # vol-targeted TQQQ / cash
        sc = (0.25/vol20).clip(0,1).fillna(1.0)
        w["tqqq"] = on.astype(float)*sc; w["cash"] = 1 - w["tqqq"]
    elif kind == "unlev":                   # QQQ / cash
        w["qqq"] = on.astype(float); w["cash"] = 1 - w["qqq"]
    elif kind == "bh_qqq":
        w["qqq"] = 1.0
    elif kind == "bh_tqqq":
        w["tqqq"] = 1.0
    w = w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    gross = sum(w[a]*RET[a] for a in RET)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    return gross - turn*COST

STRAT = {"Vol-targeted TQQQ (best growth)":"champ",
         "Unleveraged QQQ timer (best risk-adj)":"unlev",
         "Buy & hold QQQ":"bh_qqq", "Buy & hold TQQQ":"bh_tqqq"}
HOR = [("1 yr",252),("2 yr",504),("3 yr",756),("5 yr",1260),("10 yr",2520)]

for name, k in STRAT.items():
    r = strat_ret(k).dropna()
    eq = (1+r).cumprod()
    yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1
    dd=(eq/eq.cummax()-1).min()
    print(f"\n===== {name} =====")
    print(f"  full-history CAGR {cagr*100:.1f}%  |  worst drawdown {dd*100:.0f}%  "
          f"(a $1000 stake dipped to ${1000*(1+dd):,.0f} at the low)")
    print(f"  {'horizon':8s}{'$1000 -> TYPICAL':>18s}{'  (range 10th–90th pct)':>28s}{'   worst window':>16s}")
    growth = eq/eq.shift(1) - 1
    for hn, days in HOR:
        wins = eq.values[days:]/eq.values[:-days]      # rolling N-day gross multiple
        if len(wins) < 20:
            print(f"  {hn:8s}   (not enough history)"); continue
        med=np.median(wins); p10,p90=np.percentile(wins,[10,90]); lo=wins.min()
        print(f"  {hn:8s}{'$'+format(1000*med,',.0f'):>18s}"
              f"{'  $'+format(1000*p10,',.0f')+' – $'+format(1000*p90,',.0f'):>28s}"
              f"{'$'+format(1000*lo,',.0f'):>16s}")
