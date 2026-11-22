"""
PHASE C (refined) — rotation v1 failed because it rotated into equity-like alts that
are correlated with Nasdaq and very volatile. Test the tighter hypotheses:

  (a) DEFENSIVE OFF-LEG only: when the Nasdaq sleeve is risk-off, park in
      cash / bond / gold / best-of-those-3 (small, uncorrelated choice set).
  (b) STATIC BLEND: hold X% of the B-v2 engine + (1-X)% of a diversifier, always.
  (c) RISK-SCALED gold sleeve: small permanent gold allocation (vol-scaled).
Question: which, if any, produces growth in 1992-95 and 2000-2010 without wrecking
the full-period result?
"""
import json, numpy as np, pandas as pd
import bt, frontier as F
CACHE=bt.CACHE
R=pd.read_parquet("data_letf/multi_panel.parquet"); idx=R.index
START=pd.Timestamp("1988-01-01"); COST=3.0/1e4
mark=pd.Series(False,index=idx); mark[F.ndx.resample("W-FRI").last().index.intersection(idx)]=True
PX={c:(1+R[c].fillna(0)).cumprod() for c in R.columns}
def mom(c,n=252): return PX[c]/PX[c].shift(n)-1
def above_ma(c,n=200): return PX[c]>PX[c].rolling(n).mean()

scale=(0.20/F.vol).clip(0,1).fillna(1.0)
ron=F.above & ~F.confirmed
clean=(F.er>=0.25).fillna(False)
ndx_w=(ron&clean)*scale + (ron&~clean)*scale*0.5

COLS=["tqqq","cash","bond","gold","spx"]
def run(w):
    w=w.reindex(columns=COLS).fillna(0)
    w=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(w[c]*R[c].fillna(0) for c in COLS) - w.diff().abs().sum(axis=1).fillna(0)*COST
def stats(r,a=None,b=None):
    if a: r=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    r=r[r.index>=START].dropna(); e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(cagr),maxdd=float(dd),sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                calmar=float(cagr/abs(dd)) if dd<0 else np.nan)

def offleg(kind):
    w=pd.DataFrame(0.0,index=idx,columns=COLS); w["tqqq"]=ndx_w
    free=(1-ndx_w).clip(lower=0)
    if kind in ("cash","bond","gold"): w[kind]+=free
    elif kind=="best3":                       # best trending of cash/bond/gold
        cands=["bond","gold"]
        sc=pd.concat([mom(c).where(above_ma(c)&(mom(c)>0)).rename(c) for c in cands],axis=1)
        best=sc.idxmax(axis=1); anyok=sc.notna().any(axis=1)
        for c in cands: w[c]+=free*((best==c)&anyok).astype(float)
        w["cash"]+=free*(~anyok).astype(float)
    elif kind=="bond+gold":                   # static split of the off-leg
        w["bond"]+=free*0.6; w["gold"]+=free*0.4
    return w

runs={}
for k in ["cash","bond","gold","best3","bond+gold"]:
    runs[f"B-v2 off-leg = {k}"]=run(offleg(k))
base_w=offleg("bond")
# static blends: X% engine + rest diversifier held always
eng=run(base_w)
for x,div,dn in [(0.85,"gold",0.15),(0.75,"gold",0.25),(0.85,"bond",0.15),(0.75,"bond",0.25)]:
    dr=R[div].fillna(0)
    runs[f"{int(x*100)}% engine + {int(dn*100)}% {div}"]=x*eng+dn*dr
# 60/40 and NDX for context
runs["BH 60/40"]=0.6*R["spx"].fillna(0)+0.4*R["bond"].fillna(0)
runs["BH Nasdaq-100"]=R["ndx"].fillna(0)
runs["A champion"]=F.build(voltgt=0.25)

W={"1992-95":("1992-01-01","1995-12-31"),"2000-2010":("2000-01-01","2010-12-31")}
print(f"{'variant':32s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}{'|1992-95':>9}{'2000-10':>9}{'DD 00-10':>9}")
out=[]
for nm,r in runs.items():
    s=stats(r); a=stats(r,*W["1992-95"]); b=stats(r,*W["2000-2010"])
    print(f"{nm:32s}{s['cagr']*100:6.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:7.2f}"
          f"{a['cagr']*100:8.1f}%{b['cagr']*100:8.1f}%{b['maxdd']*100:8.0f}%")
    out.append(dict(name=nm,**s,w9295=a["cagr"],w0010=b["cagr"],dd0010=b["maxdd"]))

# decade-by-decade for the two best candidates
print("\n=== DECADE VIEW ===")
DEC={"1988-89":("1988-01-01","1989-12-31"),"1990s":("1990-01-01","1999-12-31"),
     "2000s":("2000-01-01","2009-12-31"),"2010s":("2010-01-01","2019-12-31"),
     "2020s":("2020-01-01","2026-07-24")}
pick=["B-v2 off-leg = cash","B-v2 off-leg = bond","85% engine + 15% gold","BH 60/40","BH Nasdaq-100"]
print(f"{'strategy':32s}"+"".join(f"{k:>10s}" for k in DEC))
dec={}
for nm in pick:
    row="";dec[nm]={}
    for k,(a,b) in DEC.items():
        s=stats(runs[nm],a,b); row+=f"{s['cagr']*100:9.1f}%"; dec[nm][k]=round(s["cagr"],4)
    print(f"{nm:32s}{row}")

def eqpts(r):
    e=(1+r[r.index>=START]).cumprod().resample("ME").last().dropna()
    return [{"t":t.strftime("%Y-%m"),"e":round(float(v),4)} for t,v in e.items()]
json.dump({"rows":out,"dec":dec,"eq":{k:eqpts(runs[k]) for k in pick}},
          open(f"{CACHE}/rotation2.json","w"),indent=1,default=str)
print("\nsaved rotation2.json")
