"""
NEW EXTERNAL INDICATORS (#5) — seasonality, breadth, volatility term structure.
Unlike the geo flags (mostly trailing momentum), SEASONALITY is a calendar rule so it
cannot lag, and BREADTH/VIX-TERM are structural rather than price-momentum.
Ablation on the Phase-C baseline.
"""
import json, numpy as np, pandas as pd, yfinance as yf
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE

def px(t):
    h=yf.Ticker(t).history(period="max")["Close"]; h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).ffill()
rsp,spy=px("RSP"),px("SPY"); vix3m,vxn=px("^VIX3M"),px("^VXN")
rut_px=(1+D.R["rut"].fillna(0)).cumprod(); spx_px=(1+D.R["spx"].fillna(0)).cumprod()

def build(gold_w=0.15, veto=None, chop_pos=0.5, voltgt=0.20, scale_mult=None):
    scale=(voltgt/D.vol).clip(0,1).fillna(1.0)
    if scale_mult is not None: scale=(scale*scale_mult.reindex(idx).fillna(1.0)).clip(0,1)
    ron=D.above & ~D.confirmed
    if veto is not None: ron=ron & ~veto.reindex(idx).fillna(False)
    clean=(D.er>=0.25).fillna(False)
    eq=(ron&clean)*scale+(ron&~clean)*scale*chop_pos
    gw=pd.Series(gold_w,index=idx); core=1.0-gw
    w=pd.DataFrame(0.0,index=idx,columns=["tqqq","bond","gold","cash"])
    w["tqqq"]=eq*core; w["bond"]+=(core-eq*core).clip(lower=0); w["gold"]+=gw
    return E.backtest(w)

res=[]
def add(name,grp,r):
    s=E.stats(r); res.append(dict(name=name,grp=grp,**s,
        dec={k:round(v["cagr"],4) for k,v in E.by_decade(r).items()},
        w0010=E.stats(r,*E.WINDOWS["2000-2010"])["cagr"]))
add("BASELINE Phase C","base",build())

# ---------- SEASONALITY ----------
mo=pd.Series(idx.month,index=idx)
# sell in May: de-lever (not exit) May-Oct
for mult in [0.5,0.75,0.0]:
    sm=pd.Series(np.where(mo.isin([5,6,7,8,9,10]),mult,1.0),index=idx)
    add(f"S1 May-Oct scale x{mult}","season",build(scale_mult=sm))
# worst months only (Aug/Sep historically weakest)
for mult in [0.5,0.0]:
    sm=pd.Series(np.where(mo.isin([8,9]),mult,1.0),index=idx)
    add(f"S2 Aug-Sep scale x{mult}","season",build(scale_mult=sm))
# boost the strong seasonal window (Nov-Apr)
for b in [1.15,1.3]:
    sm=pd.Series(np.where(mo.isin([11,12,1,2,3,4]),b,1.0),index=idx)
    add(f"S3 Nov-Apr boost x{b}","season",build(scale_mult=sm))
# turn of month: boost last+first 3 trading days
dom=pd.Series(idx.day,index=idx)
tom=((dom>=28)|(dom<=3))
add("S4 turn-of-month boost x1.3","season",build(scale_mult=pd.Series(np.where(tom,1.3,1.0),index=idx)))
# election cycle: year 3 strongest, year 1-2 weakest
yr=pd.Series(idx.year,index=idx); cyc=(yr-1988)%4
add("S5 de-lever x0.7 in cycle yrs 1-2","season",build(scale_mult=pd.Series(np.where(cyc.isin([1,2]),0.7,1.0),index=idx)))

# ---------- BREADTH ----------
rel_rut=(rut_px/spx_px); breadth_bad=(rel_rut<rel_rut.rolling(200).mean()).fillna(False)
add("B1 veto when small-caps lagging","breadth",build(veto=breadth_bad))
add("B1b de-lever x0.6 when small-caps lagging","breadth",
    build(scale_mult=pd.Series(np.where(breadth_bad,0.6,1.0),index=idx)))
rel_eq=(rsp/spy); narrow=(rel_eq<rel_eq.rolling(200).mean()).fillna(False)   # 2003+
add("B2 de-lever x0.6 when breadth narrow (RSP/SPY)","breadth",
    build(scale_mult=pd.Series(np.where(narrow,0.6,1.0),index=idx)))

# ---------- VOL TERM STRUCTURE ----------
backwardation=((vix3m/D.vix)<1.0).fillna(False)          # 2006+
add("V1 veto on VIX backwardation","vol",build(veto=backwardation))
add("V1b de-lever x0.5 on VIX backwardation","vol",
    build(scale_mult=pd.Series(np.where(backwardation,0.5,1.0),index=idx)))
vxn_hi=(vxn>vxn.rolling(252).quantile(0.8)).fillna(False)
add("V2 de-lever x0.6 when VXN high","vol",build(scale_mult=pd.Series(np.where(vxn_hi,0.6,1.0),index=idx)))

df=pd.DataFrame(res); base=df.iloc[0]
df["d_cal"]=df["calmar"]-base["calmar"]; df["d_cagr"]=df["cagr"]-base["cagr"]; df["d_0010"]=df["w0010"]-base["w0010"]
print(f"{'variant':44s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'2000-10':>9}{'dCal':>7}")
for _,r in df.sort_values("calmar",ascending=False).iterrows():
    print(f"{r['name']:44s}{r['cagr']*100:6.1f}%{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}{r['w0010']*100:8.1f}%{r['d_cal']:+7.2f}")
print("\n=== DECADE VIEW: baseline vs top 4 ===")
top=df.sort_values("calmar",ascending=False).head(5)
print(f"{'variant':44s}"+"".join(f"{k:>9s}" for k in E.DECADES))
for _,r in top.iterrows():
    print(f"{r['name']:44s}"+"".join(f"{r['dec'][k]*100:8.1f}%" for k in E.DECADES))
json.dump(df.to_dict("records"),open(f"{CACHE}/indicator_test.json","w"),indent=1,default=str)
print("\nsaved indicator_test.json")
