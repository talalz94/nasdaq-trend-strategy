"""
(A) FULL COST ACCOUNTING — every real-world drag, itemised.
(B) SHOULD YOU RE-OPTIMISE ANNUALLY? — rolling annual re-fit vs frozen rules.
"""
import json, numpy as np, pandas as pd
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE
mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)

def build(voltgt=0.20,chop_thr=0.25,chop_pos=0.5,gold_w=0.15,seasonal=True,
          cost=3.0/1e4, extra_fees=None):
    scale=(voltgt/D.vol).clip(0,1).fillna(1.0)
    if seasonal: scale=(scale*AS).clip(0,1)
    ron=D.above & ~D.confirmed
    clean=(D.er>=chop_thr).fillna(False)
    eq=(ron&clean)*scale+(ron&~clean)*scale*chop_pos
    gw=pd.Series(gold_w,index=idx); core=1.0-gw
    w=pd.DataFrame(0.0,index=idx,columns=["tqqq","bond","gold","cash"])
    w["tqqq"]=eq*core; w["bond"]+=(core-eq*core).clip(lower=0); w["gold"]+=gw
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    r=sum(ww[c]*D.R[c].fillna(0) for c in ww.columns)-ww.diff().abs().sum(axis=1).fillna(0)*cost
    if extra_fees:                       # per-sleeve annual expense ratios
        for c,er in extra_fees.items():
            if c in ww: r=r-ww[c]*(er/252)
    return r, ww

print("="*80); print(" (A) FULL COST ACCOUNTING — what each real-world drag costs"); print("="*80)
base,ww=build()
turn_ann=ww.diff().abs().sum(axis=1).resample("YE").sum().mean()
print(f" annual turnover: {turn_ann:.1f}x  (~{turn_ann*52/52:.1f} units traded per year)")
steps=[
 ("idealised (3 bps, LETF fee only)", dict()),
 ("+ TLT 0.15% & GLD 0.40% expense ratios", dict(extra_fees={"bond":0.0015,"gold":0.0040})),
 ("+ realistic spreads (6 bps/turnover)", dict(cost=6/1e4, extra_fees={"bond":0.0015,"gold":0.0040})),
 ("+ wider spreads / slippage (10 bps)", dict(cost=10/1e4, extra_fees={"bond":0.0015,"gold":0.0040})),
 ("+ pessimistic (20 bps)", dict(cost=20/1e4, extra_fees={"bond":0.0015,"gold":0.0040})),
]
print(f"\n{'scenario':44s}{'CAGR':>8}{'maxDD':>7}{'Calmar':>7}{'drag':>7}")
prev=None; rows=[]
for lbl,kw in steps:
    r,_=build(**kw); s=E.stats(r)
    drag = (prev-s['cagr'])*100 if prev is not None else 0.0
    print(f"{lbl:44s}{s['cagr']*100:7.2f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{drag:+6.2f}%")
    rows.append(dict(label=lbl,**s,drag=drag)); prev=s['cagr']
tot=(rows[0]['cagr']-rows[-1]['cagr'])*100
print(f"\n TOTAL modelled cost drag, idealised -> pessimistic: {tot:.2f}%/yr")
print(" NOT modelled: TAXES. Weekly rebalancing realises SHORT-TERM gains.")
print(f"   At a 35% short-term rate, a ~24% gross return could net ~{24*0.65:.0f}% in a TAXABLE account.")
print("   => this strategy belongs in a TAX-ADVANTAGED account (IRA/401k/ISA-equivalent).")
print(" Commissions: ~$0 at major US brokers for ETFs (Alpaca/IBKR/Schwab).")

print("\n"+"="*80); print(" (B) SHOULD YOU RE-OPTIMISE EVERY YEAR?"); print("="*80)
GRID=[dict(voltgt=v,chop_thr=t,chop_pos=p,gold_w=g)
      for v in [0.15,0.20,0.25] for t in [0.20,0.25,0.30] for p in [0.35,0.5,0.65] for g in [0.10,0.15,0.20]]
CA={}
def rr(k):
    key=tuple(sorted(k.items()))
    if key not in CA: CA[key]=build(**k)[0]
    return CA[key]
FROZEN=dict(voltgt=0.20,chop_thr=0.25,chop_pos=0.5,gold_w=0.15)
years=list(range(1998,2027))
def seg(r,y): 
    m=(r.index>=pd.Timestamp(f"{y}-01-01"))&(r.index<=pd.Timestamp(f"{y}-12-31")); return r[m]
def fit(upto_y, lookback=None):
    a=f"{upto_y-lookback}-01-01" if lookback else "1988-01-01"; b=f"{upto_y-1}-12-31"
    best=None
    for k in GRID:
        s=E.stats(rr(k),a,b)
        if np.isnan(s["calmar"]): continue
        if best is None or s["calmar"]>best[1]: best=(k,s["calmar"])
    return best[0]
modes={"FROZEN rules (never re-optimise)":None,"re-optimise yearly on ALL history":0,
       "re-optimise yearly on last 10y":10,"re-optimise yearly on last 5y":5}
res={}
for lbl,lb in modes.items():
    parts=[]; changes=0; prev=None
    for y in years:
        k=FROZEN if lb is None else fit(y, None if lb==0 else lb)
        if prev is not None and k!=prev: changes+=1
        prev=k; parts.append(seg(rr(k),y))
    r=pd.concat(parts); s=E.stats(r,start=None)
    res[lbl]=dict(**s,param_changes=changes)
    print(f" {lbl:36s} CAGR {s['cagr']*100:5.1f}%  maxDD {s['maxdd']*100:4.0f}%  "
          f"Calmar {s['calmar']:.2f}  Sharpe {s['sharpe']:.2f}   param changes: {changes}/{len(years)}")
print("\n (all measured on 1998-2026 only, so each mode is scored on the same out-of-sample period)")
json.dump({"costs":rows,"reopt":res},open(f"{CACHE}/fees_reopt.json","w"),indent=1,default=str)
print("\nsaved fees_reopt.json")
