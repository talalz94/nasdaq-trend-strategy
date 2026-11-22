"""
Does parking the risk-OFF leg in gold/bonds (instead of 0%-ish cash) fix the
2000-2012 dead decade? Champion = vol-targeted TQQQ when Nasdaq>EMA100 else OFF-leg.
Off-leg in {cash(T-bills), gold, bonds, best-recent-trend}. Weekly rebalance.
"""
import numpy as np, pandas as pd, yfinance as yf
d = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
def close(t):
    h=yf.Ticker(t).history(period="max",auto_adjust=True)["Close"]; h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()]
gold=close("GC=F"); bonds=close("TLT")            # gold futures 2000+, long treasuries 2002+
d["gold_ret"]=gold.reindex(d.index).pct_change()
d["bond_ret"]=bonds.reindex(d.index).pct_change()
d=d.loc[d[["gold_ret","bond_ret"]].dropna().index.min():]   # start where both exist (~2002)
ndx=d["ndx"]; rf_d=(d["rf"]/252).fillna(0)
RET={"tqqq":d["tqqq_ret"].fillna(0),"qqq":d["qqq_ret"].fillna(0),"cash":rf_d,
     "gold":d["gold_ret"].fillna(0),"bond":d["bond_ret"].fillna(0)}
ema100=ndx.ewm(span=100,adjust=False).mean(); on=(ndx>ema100)
vol20=ndx.pct_change().rolling(20).std()*np.sqrt(252)
mark=pd.Series(False,index=ndx.index); mark[ndx.resample("W-FRI").last().index.intersection(ndx.index)]=True
mom63={a:RET[a].add(1).rolling(63).apply(np.prod,raw=True)-1 for a in ["gold","bond","cash"]}
COST=3.0/1e4

def run(on_asset, off_mode):
    w=pd.DataFrame(0.0,index=ndx.index,columns=list(RET))
    on_w = on.astype(float)*( (0.25/vol20).clip(0,1).fillna(1) if on_asset=="tqqq" else 1.0)
    w[on_asset]=on_w; off=1-on_w
    if off_mode=="best":     # dual-momentum: risk-off -> best 3mo trend of gold/bond/cash
        pick=pd.concat([mom63["gold"],mom63["bond"],mom63["cash"]],axis=1)
        pick.columns=["gold","bond","cash"]; choice=pick.idxmax(axis=1).fillna("cash")
        for a in ["gold","bond","cash"]: w[a]+=off*(choice==a).astype(float)
    else:
        w[off_mode]+=off
    w=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    ret=sum(w[a]*RET[a] for a in RET)-w.diff().abs().sum(axis=1).fillna(0)*COST
    return ret

def stats(ret,sub=None):
    r=ret if sub is None else ret.loc[sub]; e=(1+r).cumprod()
    yrs=(r.index[-1]-r.index[0]).days/365.25; cagr=e.iloc[-1]**(1/yrs)-1
    dd=(e/e.cummax()-1).min(); sh=r.mean()/r.std()*np.sqrt(252)
    return cagr,float(dd),float(sh),e.iloc[-1]-1

full=d.index; dead=(d.index<=pd.Timestamp("2013-01-01"))   # ~2002-2012 dead decade
print(f"period {full.min().date()}..{full.max().date()}\n")
for on_asset in ["tqqq","qqq"]:
    lbl="Vol-tgt TQQQ" if on_asset=="tqqq" else "Unlev QQQ"
    print(f"===== {lbl} timer — off-leg variants =====")
    print(f"{'off-leg':10s}{'FULL cagr':>11s}{'maxDD':>8s}{'Sharpe':>8s}{'| 2002-2012 cagr':>18s}{'maxDD':>8s}")
    for off in ["cash","gold","bond","best"]:
        r=run(on_asset,off); c,dd,sh,tot=stats(r); c2,dd2,sh2,_=stats(r,sub=dead)
        print(f"{off:10s}{c*100:10.1f}%{dd*100:7.0f}%{sh:8.2f}{c2*100:16.1f}%{dd2*100:7.0f}%")
    print()
# buy&hold refs over same window
for a in ["qqq","tqqq"]:
    c,dd,sh,tot=stats(RET[a]); c2,*_=stats(RET[a],sub=dead)
    print(f"BH {a.upper():4s}: FULL {c*100:.1f}% maxDD {dd*100:.0f}% | 2002-2012 {c2*100:.1f}%")
