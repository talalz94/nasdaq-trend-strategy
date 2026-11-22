"""
PHASE C — can multi-asset rotation create growth in Nasdaq's flat decades?

Three designs, all no-lookahead (signal from t-1, weekly rebalance):
  CORE-SAT   : B-v2 rules on the Nasdaq sleeve; when Nasdaq is risk-OFF, rotate the
               capital into the best-trending ALTERNATIVE (gold/bond/rut/intl/hsi/tsx)
               that passes absolute momentum, else cash.  (minimal change to B-v2)
  ROTATE-N   : pure dual-momentum across ALL sleeves incl. a leveraged Nasdaq sleeve;
               hold top-N by 12m momentum that are above their own 200d MA, else cash.
  BLEND      : half core-sat + half rotate.
Benchmarks: B-v2, A champion, buy&hold NDX/SPX/gold/60-40/equal-weight.
"""
import json, numpy as np, pandas as pd
import bt, frontier as F
CACHE=bt.CACHE
R=pd.read_parquet("data_letf/multi_panel.parquet")
idx=R.index
START=pd.Timestamp("1988-01-01")          # all sleeves + 12m lookback available
ALTS=["gold","bond","rut","intl","hsi","tsx"]
COST=3.0/1e4
mark=pd.Series(False,index=idx); mark[F.ndx.resample("W-FRI").last().index.intersection(idx)]=True

# price proxies (cum returns) for each sleeve, for trend/momentum
PX={c:(1+R[c].fillna(0)).cumprod() for c in ALTS+["ndx","spx","cash","tqqq","qqq"]}
def mom(c,n=252): return PX[c]/PX[c].shift(n)-1
def above_ma(c,n=200): return PX[c]>PX[c].rolling(n).mean()

# --- B-v2 Nasdaq sleeve state (from Phase B) ---
scale=(0.20/F.vol).clip(0,1).fillna(1.0)
ron=F.above & ~F.confirmed                       # trend up AND not confirmed-recession
clean=(F.er>=0.25).fillna(False)
ndx_w=(ron&clean)*scale + (ron&~clean)*scale*0.5  # TQQQ weight (full or half)

def bt_run(w):
    w=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    gross=sum(w[c]*R[c].fillna(0) for c in w.columns)
    return gross - w.diff().abs().sum(axis=1).fillna(0)*COST

def stats(r,a=None,b=None):
    if a: r=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    r=r.dropna(); e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(cagr),maxdd=float(dd),sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                calmar=float(cagr/abs(dd)) if dd<0 else np.nan)

cols=["tqqq","qqq","cash"]+ALTS
def blank(): return pd.DataFrame(0.0,index=idx,columns=cols)

# ---------- 1. CORE-SATELLITE ----------
def core_sat(alt_on=True):
    w=blank(); w["tqqq"]=ndx_w
    free=(1-ndx_w).clip(lower=0)
    if not alt_on:
        w["cash"]+=free; return w
    score=pd.concat([mom(c).rename(c) for c in ALTS],axis=1)
    ok=pd.concat([(above_ma(c)&(mom(c)>0)).rename(c) for c in ALTS],axis=1)
    score=score.where(ok)                       # only trending alts eligible
    best=score.idxmax(axis=1)
    anyok=ok.any(axis=1)
    for c in ALTS:
        w[c]+=free*((best==c)&anyok).astype(float)
    w["cash"]+=free*(~anyok).astype(float)
    return w

# ---------- 2. ROTATE-N (dual momentum across sleeves incl. leveraged Nasdaq) ----------
def rotate(n=3, lev=True):
    sleeves=ALTS+["ndx","spx"]
    score=pd.concat([mom(c).rename(c) for c in sleeves],axis=1)
    ok=pd.concat([(above_ma(c)&(mom(c)>0)).rename(c) for c in sleeves],axis=1)
    score=score.where(ok)
    rank=score.rank(axis=1,ascending=False)
    w=blank()
    for c in sleeves:
        sel=(rank[c]<=n)&ok[c]
        tgt = "tqqq" if (c=="ndx" and lev) else c
        add=sel.astype(float)/n
        if c=="ndx" and lev: add=add*scale        # vol-target the leveraged sleeve
        if tgt in w.columns: w[tgt]+=add
        else: w[c]=w.get(c,0)+add
    w["cash"]+=(1-w.drop(columns=["cash"]).sum(axis=1)).clip(lower=0)
    return w

# ---------- benchmarks ----------
def bh(c):
    w=blank(); w[c if c in cols else "cash"]=1.0
    if c not in cols: w=blank(); w["cash"]=0.0; w[c]=1.0
    return w
def sixty40():
    w=blank(); w["spx"]=0.0
    w=pd.DataFrame(0.0,index=idx,columns=["spx","bond"]); w["spx"]=0.6; w["bond"]=0.4; return w
def eqw():
    w=pd.DataFrame(0.0,index=idx,columns=ALTS+["ndx"]);
    for c in w.columns: w[c]=1/len(w.columns)
    return w

WINS={"1992-1995":("1992-01-01","1995-12-31"),"2000-2010":("2000-01-01","2010-12-31"),
      "full 1988-2026":("1988-01-01","2026-07-24")}
runs={}
runs["CORE-SAT (B-v2 + alt rotation)"]=bt_run(core_sat(True))
runs["B-v2 (cash off-leg)"]=bt_run(core_sat(False))
for n in [2,3,4]:
    runs[f"ROTATE top-{n} (levered NDX)"]=bt_run(rotate(n,True))
runs["ROTATE top-3 (unlevered)"]=bt_run(rotate(3,False))
runs["BLEND 50/50 core+rot"]=(runs["CORE-SAT (B-v2 + alt rotation)"]+runs["ROTATE top-3 (levered NDX)"])/2
runs["BH Nasdaq-100"]=R["ndx"].fillna(0)
runs["BH S&P 500"]=R["spx"].fillna(0)
runs["BH gold miners"]=R["gold"].fillna(0)
runs["BH 60/40"]=0.6*R["spx"].fillna(0)+0.4*R["bond"].fillna(0)
runs["BH equal-weight all"]=R[ALTS+["ndx"]].fillna(0).mean(axis=1)
runs["A champion (TQQQ/cash)"]=F.build(voltgt=0.25)

print(f"{'strategy':34s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}"
      f"{'|1992-95':>9}{'2000-10':>9}")
out=[]
for nm,r in runs.items():
    r=r[r.index>=START]
    s=stats(r); w1=stats(r,*WINS["1992-1995"]); w2=stats(r,*WINS["2000-2010"])
    print(f"{nm:34s}{s['cagr']*100:6.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:7.2f}"
          f"{w1['cagr']*100:8.1f}%{w2['cagr']*100:8.1f}%")
    out.append(dict(name=nm,**s,w9295=w1["cagr"],w0010=w2["cagr"],
                    dd9295=w1["maxdd"],dd0010=w2["maxdd"]))

def eqpts(r):
    e=(1+r[r.index>=START]).cumprod().resample("ME").last().dropna()
    return [{"t":t.strftime("%Y-%m"),"e":round(float(v),4)} for t,v in e.items()]
keep=["CORE-SAT (B-v2 + alt rotation)","B-v2 (cash off-leg)","ROTATE top-3 (levered NDX)",
      "BH Nasdaq-100","BH 60/40","A champion (TQQQ/cash)"]
json.dump({"rows":out,"eq":{k:eqpts(runs[k]) for k in keep},
           "asset_windows":None},open(f"{CACHE}/rotation.json","w"),indent=1,default=str)
print("\nsaved rotation.json")
