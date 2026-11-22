"""
PHASE B2b — combine the winners from frontier.py:
  partial de-lever (chop -> HALF TQQQ)  +  CONFIRMED curve  +  off-leg choice
and sweep the partial-position level & vol target to find the tuned frontier.
"""
import json, numpy as np, pandas as pd
import bt, frontier as F      # reuse build/stats/data
CACHE=bt.CACHE
res=[]
def add(name,grp,**kw):
    r=F.build(**kw); s=F.stats(r); s09=F.stats(r,F.m0009)
    res.append(dict(name=name,group=grp,**s,cagr0009=s09["cagr"],maxdd0009=s09["maxdd"],
                    _kw=kw)); return res[-1]

# baseline dial for the frontier test
for v in [0.12,0.15,0.20,0.25,0.30,0.40]:
    add(f"vol{int(v*100)} dial","dial",voltgt=v)
# tuned combos: partial de-lever x curve mode x off-leg
for thr in [0.20,0.25,0.30]:
    for pos in [0.35,0.5,0.65]:
        add(f"chop<{thr}->{int(pos*100)}%TQQQ","partial",chop_thr=thr,chop_pos=pos)
        add(f"chop<{thr}->{int(pos*100)}% +curve:confirm","combo",chop_thr=thr,chop_pos=pos,curve_mode="confirm")
        add(f"chop<{thr}->{int(pos*100)}% +confirm +bond","combo",chop_thr=thr,chop_pos=pos,curve_mode="confirm",off="bond")
# vol-target sweep on the best structure
for v in [0.20,0.25,0.30,0.35]:
    add(f"vol{int(v*100)} chop<0.25->50% +confirm","voltune",voltgt=v,chop_thr=0.25,chop_pos=0.5,curve_mode="confirm")
# incumbents
add("A: champion","incumbent",voltgt=0.25)
add("B: hybrid","incumbent",curve_mode="raw",chop_thr=0.30,off="bond")

df=pd.DataFrame(res)
dial=df[df.group=="dial"].sort_values("maxdd")
df["dial_equiv"]=df["maxdd"].apply(lambda dd: float(np.interp(dd,dial["maxdd"].values,dial["cagr"].values)))
df["edge"]=df["cagr"]-df["dial_equiv"]
print("=== TUNED VARIANTS (edge = CAGR above a pure vol-dial at the same drawdown) ===")
print(f"{'variant':40s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'2000-09':>9}{'edge':>8}")
for _,r in df.sort_values("edge",ascending=False).head(18).iterrows():
    print(f"{r['name']:40s}{r['cagr']*100:6.1f}%{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}{r['cagr0009']*100:8.1f}%{r['edge']*100:+7.1f}%")

best=df.sort_values("calmar",ascending=False).iloc[0]
print(f"\nBEST BY CALMAR: {best['name']}  CAGR {best['cagr']*100:.1f}%  maxDD {best['maxdd']*100:.0f}%  "
      f"Calmar {best['calmar']:.2f}  2000-09 {best['cagr0009']*100:.1f}%")
inc=df[df.name=="B: hybrid"].iloc[0]; a=df[df.name=="A: champion"].iloc[0]
print(f"vs OLD hybrid B: CAGR {inc['cagr']*100:.1f}% maxDD {inc['maxdd']*100:.0f}% Calmar {inc['calmar']:.2f}")
print(f"vs champion  A: CAGR {a['cagr']*100:.1f}% maxDD {a['maxdd']*100:.0f}% Calmar {a['calmar']:.2f}")

# equity curves for the report
def eqpts(r):
    e=(1+r).cumprod().resample("ME").last().dropna()
    return [{"t":t.strftime("%Y-%m"),"e":round(float(v),4)} for t,v in e.items()]
rb=F.build(**best["_kw"]); rA=F.build(voltgt=0.25); rB=F.build(curve_mode="raw",chop_thr=0.30,off="bond")
per={"1985-99":("1985","1999"),"2000-09":("2000","2009"),"2010-19":("2010","2019"),"2020-26":("2020","2026")}
def sub(r):
    o={}
    for k,(s,e) in per.items():
        msk=(F.idx>=pd.Timestamp(s))&(F.idx<=pd.Timestamp(e+"-12-31"))
        o[k]={kk:round(vv,4) for kk,vv in F.stats(r,msk).items()}
    return o
proj={}
e=(1+rb).cumprod().values
for hn,dd in [("1 yr",252),("3 yr",756),("5 yr",1260),("10 yr",2520)]:
    w=e[dd:]/e[:-dd]; proj[hn]={"typ":round(1000*np.median(w)),"lo":round(1000*np.percentile(w,10)),"hi":round(1000*np.percentile(w,90))}
out={"rows":df.drop(columns=["_kw"]).round(4).to_dict("records"),
     "best":{k:(round(v,4) if isinstance(v,(int,float,np.floating)) else str(v)) for k,v in best.drop("_kw").items()},
     "best_cfg":{k:str(v) for k,v in best["_kw"].items()},
     "sub_best":sub(rb),"sub_A":sub(rA),"sub_B":sub(rB),"proj":proj,
     "eq_best":eqpts(rb),"eq_A":eqpts(rA),"eq_B":eqpts(rB)}
json.dump(out,open(f"{CACHE}/frontier2.json","w"),indent=1,default=str)
print(f"\n$1000 -> {proj}")
print("saved frontier2.json")
