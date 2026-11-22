"""
TIER 1 IMPROVEMENTS — the three untested ideas with real potential.
  T1  2x leverage (QLD) instead of 3x (TQQQ)      -> less decay, hold more, longer
  T2  SIGNAL ENSEMBLE (avg of 4 EMA lookbacks)    -> continuous exposure, less whipsaw
                                                     [target: the 2011 -21.6% failure]
  T3  DURATION-MATCHED off-leg (short bills when rates rising, long bonds when falling)
                                                     [target: the 2022 -21.9% failure]
Each tested alone and in combination against the Phase-D FINAL strategy.
"""
import json, numpy as np, pandas as pd, yfinance as yf
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE
d=D.d; rf_d=D.rf_d; m=D.m

# ---------- build 2x sleeve (QLD) + validate vs real ----------
EXP=0.0095
qld_sim = 2*d["ndx_ret"] - EXP/252 - 1*rf_d          # 2x borrows 1x notional
def real(t):
    h=yf.Ticker(t).history(period="max",auto_adjust=True)["Close"]
    h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).pct_change()
qld_real=real("QLD")
ov=pd.concat([qld_sim,qld_real],axis=1).dropna(); ov.columns=["sim","real"]
print(f"VALIDATE 2x: overlap {len(ov)}d  sim x{(1+ov['sim']).prod():.2f} vs real x{(1+ov['real']).prod():.2f}"
      f"  corr {ov['sim'].corr(ov['real']):.3f}  trackErr {(ov['sim']-ov['real']).std()*np.sqrt(252)*100:.1f}%/yr")
R=dict(D.R); R["qld"]=qld_sim.fillna(0)

# ---------- duration sleeves ----------
tnx=(m["tnx"] if "tnx" in m else None)
tnx=(yf.Ticker("^TNX").history(period="max")["Close"]/100)
tnx.index=tnx.index.tz_localize(None).normalize(); tnx=tnx[~tnx.index.duplicated()].reindex(idx).ffill()
dy=tnx.diff()
R["bond2y"]=(tnx.shift(1)/252 - 1.9*dy).fillna(0)     # ~2y duration
R["bond10y"]=R["bond"]                                 # ~7.5y duration (existing)
rate_rising=(tnx>tnx.rolling(200).mean()).fillna(False)

# ---------- ensemble signal ----------
def ens_exposure(spans=(50,100,150,200)):
    votes=[(D.ndx>D.ndx.ewm(span=s,adjust=False).mean()).astype(float) for s in spans]
    return sum(votes)/len(votes)                       # 0..1 continuous

mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)

def build(lev="tqqq", voltgt=0.20, chop_thr=0.25, chop_pos=0.5, gold_w=0.15,
          off="bond10y", ensemble=False, spans=(50,100,150,200), duration=False,
          seasonal=True, use_curve=True):
    scale=(voltgt/D.vol).clip(0,1).fillna(1.0)
    if seasonal: scale=(scale*AS).clip(0,1)
    if ensemble:
        expo=ens_exposure(spans)
        if use_curve: expo=expo*(~D.confirmed).astype(float)
        clean=(D.er>=chop_thr).fillna(False)
        eq=expo*scale*np.where(clean,1.0,chop_pos)
        eq=pd.Series(eq,index=idx)
    else:
        ron=D.above & ((~D.confirmed) if use_curve else True)
        clean=(D.er>=chop_thr).fillna(False)
        eq=(ron&clean)*scale+(ron&~clean)*scale*chop_pos
    gw=pd.Series(gold_w,index=idx); core=1.0-gw
    cols=[lev,"bond2y","bond10y","gold","cash"]
    w=pd.DataFrame(0.0,index=idx,columns=cols)
    w[lev]=eq*core
    free=(core-eq*core).clip(lower=0)
    if duration:
        w["bond2y"]+=free*rate_rising.astype(float); w["bond10y"]+=free*(~rate_rising).astype(float)
    else:
        w[off]+=free
    w["gold"]+=gw
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(ww[c]*R[c].fillna(0) for c in cols)-ww.diff().abs().sum(axis=1).fillna(0)*E.COST

EV={"2011 EU crisis":("2011-05-01","2011-10-31"),"2022 bear":("2022-01-01","2022-12-31"),
    "dot-com":("2000-03-01","2002-10-31"),"GFC":("2007-10-01","2009-03-31")}
def evt(r,k):
    a,b=EV[k]; x=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))].dropna(); return (1+x).prod()-1

res=[]
def add(name,grp,**kw):
    r=build(**kw); s=E.stats(r)
    res.append(dict(name=name,grp=grp,**s,
        dec={k:round(v["cagr"],4) for k,v in E.by_decade(r).items()},
        **{f"ev_{k}":evt(r,k) for k in EV}))
    return res[-1]

add("FINAL (3x, single EMA, TLT)","base")
# T1: 2x
for vt in [0.20,0.30,0.40]:
    add(f"T1 2x QLD voltgt{vt}","T1",lev="qld",voltgt=vt)
# T2: ensemble
add("T2 ensemble 50/100/150/200","T2",ensemble=True)
add("T2 ensemble 20/50/100/200","T2",ensemble=True,spans=(20,50,100,200))
add("T2 ensemble + 2x","T2",ensemble=True,lev="qld",voltgt=0.30)
# T3: duration-matched
add("T3 duration-matched off-leg","T3",duration=True)
add("T3 short-bills only off-leg","T3",off="bond2y")
add("T3 cash off-leg (for contrast)","T3",off="cash")
# combos
add("C1 2x + ensemble + duration","combo",lev="qld",voltgt=0.30,ensemble=True,duration=True)
add("C2 3x + ensemble + duration","combo",ensemble=True,duration=True)
add("C3 2x + duration","combo",lev="qld",voltgt=0.30,duration=True)
add("C4 2x voltgt0.40 + ensemble + duration","combo",lev="qld",voltgt=0.40,ensemble=True,duration=True)

df=pd.DataFrame(res); b=df.iloc[0]
df["dCal"]=df["calmar"]-b["calmar"]
print(f"\n{'variant':36s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}{'dCal':>7}{'2011':>8}{'2022':>8}")
for _,r in df.sort_values("calmar",ascending=False).iterrows():
    print(f"{r['name']:36s}{r['cagr']*100:6.1f}%{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}{r['sharpe']:7.2f}"
          f"{r['dCal']:+7.2f}{r['ev_2011 EU crisis']*100:7.1f}%{r['ev_2022 bear']*100:7.1f}%")
print(f"\n{'variant':36s}{'dot-com':>9}{'GFC':>9}   decades")
for _,r in df.sort_values("calmar",ascending=False).head(6).iterrows():
    print(f"{r['name']:36s}{r['ev_dot-com']*100:8.1f}%{r['ev_GFC']*100:8.1f}%   "
          +" ".join(f"{r['dec'][k]*100:.0f}%" for k in E.DECADES))
json.dump(df.to_dict("records"),open(f"{CACHE}/tier1.json","w"),indent=1,default=str)
print("\nsaved tier1.json")
