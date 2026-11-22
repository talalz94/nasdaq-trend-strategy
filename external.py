"""
HEAD-TO-HEAD vs PUBLISHED / COMMUNITY STRATEGIES.
Every strategy re-implemented on the SAME data, SAME period (1988-2026), SAME cost model
(3 bps per unit turnover) and SAME metrics, so the comparison is apples-to-apples.
Published CAGR figures are NOT comparable across sources (different eras/costs/assets) —
that's exactly why this re-implementation matters.

  1. HFEA (Hedgefundie, Bogleheads)      55% UPRO / 45% TMF, quarterly
  2. Gayed "Leverage for the Long Run"   3x SPX when SPX>200d SMA else T-bills (2016 Dow Award)
  3. Gayed applied to Nasdaq             3x NDX when NDX>200d SMA else T-bills
  4. Antonacci GEM (dual momentum)       monthly: best 12m of SPX/intl if > T-bills, else bonds
  5. Faber GTAA (4-asset)                monthly: each asset held if > its 10m SMA, else cash
  6. Vol-target 12% (SeekingAlpha)       3x equity/3x bond scaled to 12% vol, monthly
  7. VAA-G4 (Keller)                     13612W momentum, canary crash protection, monthly
  8. Risk-parity-ish 60/40 x1.5          leveraged balanced, quarterly
  9. OUR STRATEGY (3x and 2x)
"""
import json, numpy as np, pandas as pd
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE; rf_d=D.rf_d; COST=E.COST
P=D.P
# ---- build the sleeves the external strategies need ----
tnx=D.m["tnx"].reindex(idx).ffill()/1.0 if "tnx" in D.m else None
import yfinance as yf
_t=yf.Ticker("^TNX").history(period="max")["Close"]/100
_t.index=_t.index.tz_localize(None).normalize(); tnx=_t[~_t.index.duplicated()].reindex(idx).ffill()
dy=tnx.diff()
bond20=(tnx.shift(1)/252 - 17.0*dy).fillna(0)          # ~TLT (20y+, duration ~17)
bond10=D.R["bond"]                                      # ~IEF/10y, duration 7.5
def lev(r,k,exp): return (k*r - exp/252 - (k-1)*rf_d).fillna(0)
R=dict(D.R)
R["upro"]=lev(P["spx"],3,0.0091)        # 3x S&P
R["tmf"] =lev(bond20,3,0.0106)          # 3x 20y+ treasuries
R["bond20"]=bond20
R["sso"] =lev(P["spx"],2,0.0089)        # 2x S&P
qld=lev(D.d["ndx_ret"],2,0.0095); R["qld"]=qld
mo_end=pd.Series(False,index=idx); mo_end[idx.to_series().resample("ME").last().dropna()]=True
q_end=pd.Series(False,index=idx);  q_end[idx.to_series().resample("QE").last().dropna()]=True

def run(w, mark):
    ww=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(ww[c]*R[c].fillna(0) for c in ww.columns)-ww.diff().abs().sum(axis=1).fillna(0)*COST
def px(c): return (1+R[c].fillna(0)).cumprod()
def mom(c,n): p=px(c); return p/p.shift(n)-1
def sma_ratio(c,n):  p=px(c); return p/p.rolling(n).mean()-1

res=[]
def add(name,src,r):
    s=E.stats(r); res.append(dict(name=name,src=src,**s,
      dec={k:round(v["cagr"],4) for k,v in E.by_decade(r).items()},
      ev2022=float((1+r[(r.index>="2022-01-01")&(r.index<="2022-12-31")].dropna()).prod()-1),
      ev0002=float((1+r[(r.index>="2000-03-01")&(r.index<="2002-10-31")].dropna()).prod()-1),
      evgfc=float((1+r[(r.index>="2007-10-01")&(r.index<="2009-03-31")].dropna()).prod()-1)))

# 1 HFEA
w=pd.DataFrame(0.0,index=idx,columns=["upro","tmf"]); w["upro"]=0.55; w["tmf"]=0.45
add("HFEA 55/45 UPRO/TMF (quarterly)","Bogleheads",run(w,q_end))
# 2 Gayed LRS on S&P
sig=(sma_ratio("spx",200)>0).fillna(False)
w=pd.DataFrame(0.0,index=idx,columns=["upro","cash"]); w["upro"]=sig.astype(float); w["cash"]=1-w["upro"]
add("Gayed LRS: 3x SPX / T-bills","Dow Award 2016",run(w,mo_end))
w2=w.copy(); add("Gayed LRS (weekly rebalance)","Dow Award 2016",run(w2,D.mark))
# 3 Gayed on Nasdaq
sig=(D.ndx/D.ndx.rolling(200).mean()>1).fillna(False)
w=pd.DataFrame(0.0,index=idx,columns=["tqqq","cash"]); w["tqqq"]=sig.astype(float); w["cash"]=1-w["tqqq"]
add("Gayed LRS applied to Nasdaq (3x)","adapted",run(w,mo_end))
# 4 Antonacci GEM
m_spx,m_int,m_cash=mom("spx",252),mom("intl",252),mom("cash",252)
best_int=(m_int>m_spx)
absok=(np.maximum(m_spx,m_int)>m_cash)
w=pd.DataFrame(0.0,index=idx,columns=["spx","intl","bond"])
w["spx"]=(absok&~best_int).astype(float); w["intl"]=(absok&best_int).astype(float)
w["bond"]=(~absok).astype(float)
add("Antonacci GEM dual momentum","Antonacci 2014",run(w,mo_end))
# 5 Faber GTAA (4-asset: SPX, intl, bond, gold)
assets=["spx","intl","bond","gold"]
w=pd.DataFrame(0.0,index=idx,columns=assets+["cash"])
for a in assets:
    on=(sma_ratio(a,210)>0).fillna(False)     # 10-month SMA ~ 210 trading days
    w[a]=on.astype(float)*0.25; w["cash"]+= (~on).astype(float)*0.25
add("Faber GTAA (4-asset, 10m SMA)","Faber 2007",run(w,mo_end))
# 6 Vol-target 12% with 3x sleeves
port=0.6*R["upro"]+0.4*R["tmf"]
rv=port.rolling(21).std()*np.sqrt(252)
k=(0.12/rv).clip(0,1).fillna(0)
w=pd.DataFrame(0.0,index=idx,columns=["upro","tmf","cash"])
w["upro"]=k*0.6; w["tmf"]=k*0.4; w["cash"]=1-k
add("Vol-target 12% (3x eq/bond)","SeekingAlpha",run(w,mo_end))
# 7 VAA-G4 (Keller) 13612W
def w13612(c):
    p=px(c); return (12*(p/p.shift(21)-1)+4*(p/p.shift(63)-1)+2*(p/p.shift(126)-1)+(p/p.shift(252)-1))/19
off=["spx","intl","hsi","bond"]; sc=pd.concat([w13612(a).rename(a) for a in off],axis=1)
canary_bad=(sc<0).any(axis=1)                      # any negative -> defensive
pick=sc.idxmax(axis=1)
w=pd.DataFrame(0.0,index=idx,columns=off+["cash"])
for a in off: w[a]=((~canary_bad)&(pick==a)).astype(float)
w["cash"]=canary_bad.astype(float)
add("VAA-G4 (13612W + canary)","Keller 2017",run(w,mo_end))
# 8 leveraged 60/40
w=pd.DataFrame(0.0,index=idx,columns=["sso","bond20"]); w["sso"]=0.6*1.5/2*2; w["sso"]=0.45; w["bond20"]=0.55
add("Leveraged balanced (2x SPX 45 / TLT 55)","risk-parity-ish",run(w,q_end))
# 9 OURS
add("★ OURS — 3x (Strategy F)","this project",E.run())
def ours(lev_sleeve="tqqq", voltgt=0.20, gold_w=0.15):
    mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)
    scale=((voltgt/D.vol).clip(0,1).fillna(1.0)*AS).clip(0,1)
    ron=D.above & ~D.confirmed; clean=(D.er>=0.25).fillna(False)
    eq=(ron&clean)*scale+(ron&~clean)*scale*0.5
    gw=pd.Series(gold_w,index=idx); core=1.0-gw
    w=pd.DataFrame(0.0,index=idx,columns=[lev_sleeve,"bond","gold"])
    w[lev_sleeve]=eq*core; w["bond"]=(core-eq*core).clip(lower=0); w["gold"]=gw
    return run(w, D.mark)
add("★ OURS — 2x variant","this project", ours("qld",0.30))

df=pd.DataFrame(res).sort_values("calmar",ascending=False)
print(f"{'strategy':40s}{'source':16s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}{'2022':>8}{'dotcom':>8}{'GFC':>8}")
for _,r in df.iterrows():
    star='★' if r['src']=='this project' else ' '
    print(f"{star}{r['name'][:39]:39s}{r['src'][:15]:16s}{r['cagr']*100:6.1f}%{r['maxdd']*100:6.0f}%"
          f"{r['calmar']:7.2f}{r['sharpe']:7.2f}{r['ev2022']*100:7.0f}%{r['ev0002']*100:7.0f}%{r['evgfc']*100:7.0f}%")
print("\n=== DECADES (CAGR) ===")
print(f"{'strategy':40s}"+"".join(f"{k:>9s}" for k in E.DECADES))
for _,r in df.iterrows():
    print(f"{r['name'][:39]:40s}"+"".join(f"{r['dec'][k]*100:8.1f}%" for k in E.DECADES))
json.dump(df.to_dict("records"),open(f"{CACHE}/external.json","w"),indent=1,default=str)
print("\nsaved external.json")
