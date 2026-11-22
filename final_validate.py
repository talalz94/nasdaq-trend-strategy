"""
FINAL VALIDATION — everything that survived, stress-tested properly.
 1. walk-forward both directions (choose on train, score on unseen test)
 2. per-decade consistency of each component
 3. bootstrap significance of the Aug-Sep addition
 4. parameter perturbation robustness (all knobs +/-)
 5. cost sensitivity (3 -> 20 bps)
 6. crisis stress table
"""
import json, numpy as np, pandas as pd
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE
mo=pd.Series(idx.month,index=idx)
AUGSEP=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)

def build(voltgt=0.20,chop_thr=0.25,chop_pos=0.5,gold_w=0.15,off="bond",
          use_curve=True,seasonal=False,cost=E.COST):
    scale=(voltgt/D.vol).clip(0,1).fillna(1.0)
    if seasonal: scale=(scale*AUGSEP).clip(0,1)
    ron=D.above & (~D.confirmed if use_curve else True)
    clean=(D.er>=chop_thr).fillna(False)
    eq=(ron&clean)*scale+(ron&~clean)*scale*chop_pos
    gw=pd.Series(gold_w,index=idx); core=1.0-gw
    w=pd.DataFrame(0.0,index=idx,columns=["tqqq","bond","gold","cash"])
    w["tqqq"]=eq*core; w[off]+=(core-eq*core).clip(lower=0); w["gold"]+=gw
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(ww[c]*D.R[c].fillna(0) for c in ww.columns)-ww.diff().abs().sum(axis=1).fillna(0)*cost

FINAL=dict(seasonal=True); BASE=dict(seasonal=False)
rF,rB=build(**FINAL),build(**BASE)
print("=== 1. WALK-FORWARD (grid chosen on train by Calmar, scored on unseen test) ===")
GRID=[dict(voltgt=v,chop_thr=t,chop_pos=p,gold_w=g,seasonal=s)
      for v in [0.15,0.20,0.25] for t in [0.20,0.25,0.30] for p in [0.35,0.5,0.65]
      for g in [0.0,0.10,0.15,0.20] for s in [False,True]]
def nm(k): return f"vol{int(k['voltgt']*100)} thr{k['chop_thr']} pos{int(k['chop_pos']*100)} gold{int(k['gold_w']*100)}{' +AS' if k['seasonal'] else ''}"
CA={}
def rr(k):
    key=nm(k)
    if key not in CA: CA[key]=build(**k)
    return CA[key]
def msk(a,b): return (idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b))
for lbl,a1,b1,a2,b2 in [("early->late","1988-01-01","2006-12-31","2007-01-01","2026-07-24"),
                        ("late->early","2007-01-01","2026-07-24","1988-01-01","2006-12-31")]:
    tr,te=msk(a1,b1),msk(a2,b2)
    sc=[(nm(k),E.stats(rr(k),a1,b1),E.stats(rr(k),a2,b2)) for k in GRID]
    pick=max(sc,key=lambda x:x[1]["calmar"]); orc=max(sc,key=lambda x:x[2]["calmar"])
    print(f"  [{lbl}] picked '{pick[0]}'  train Cal {pick[1]['calmar']:.2f} -> "
          f"TEST Cal {pick[2]['calmar']:.2f} (CAGR {pick[2]['cagr']*100:.1f}%, DD {pick[2]['maxdd']*100:.0f}%)"
          f" | oracle {orc[2]['calmar']:.2f} -> selection cost {orc[2]['calmar']-pick[2]['calmar']:.2f}")
    fi=E.stats(rF,a2,b2); bi=E.stats(rB,a2,b2)
    print(f"           FINAL on this test: Cal {fi['calmar']:.2f} | no-seasonal: {bi['calmar']:.2f}")

print("\n=== 2. Aug-Sep component: consistency by decade ===")
print(f"{'decade':10s}{'base CAGR':>11}{'+AS CAGR':>10}{'base DD':>9}{'+AS DD':>8}{'better?':>9}")
for k,(a,b) in E.DECADES.items():
    sb,sf=E.stats(rB,a,b),E.stats(rF,a,b)
    better="yes" if (sf["calmar"]>=sb["calmar"]) else "no"
    print(f"{k:10s}{sb['cagr']*100:10.1f}%{sf['cagr']*100:9.1f}%{sb['maxdd']*100:8.0f}%{sf['maxdd']*100:7.0f}%{better:>9}")

print("\n=== 3. BOOTSTRAP: is +Aug-Sep real? (block bootstrap of the return difference) ===")
diff=(rF-rB)[rF.index>=E.START].dropna()
rng=np.random.default_rng(11); n=len(diff); bl=63; nb=n//bl
means=[]
for _ in range(4000):
    st=rng.integers(0,n-bl,size=nb)
    means.append(np.mean(np.concatenate([diff.values[s:s+bl] for s in st])))
means=np.array(means)*252
print(f"  mean annual difference {means.mean()*100:+.2f}%  95% CI [{np.percentile(means,2.5)*100:+.2f}%, {np.percentile(means,97.5)*100:+.2f}%]"
      f"  P(better)={100*(means>0).mean():.0f}%")

print("\n=== 4. PARAMETER PERTURBATION (each knob moved, FINAL config) ===")
print(f"{'perturbation':30s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}")
base_s=E.stats(rF); print(f"{'FINAL (as specified)':30s}{base_s['cagr']*100:6.1f}%{base_s['maxdd']*100:6.0f}%{base_s['calmar']:7.2f}")
for lbl,kw in [("voltgt 0.15",dict(voltgt=0.15)),("voltgt 0.25",dict(voltgt=0.25)),
               ("chop_thr 0.20",dict(chop_thr=0.20)),("chop_thr 0.30",dict(chop_thr=0.30)),
               ("chop_pos 0.35",dict(chop_pos=0.35)),("chop_pos 0.65",dict(chop_pos=0.65)),
               ("gold 10%",dict(gold_w=0.10)),("gold 20%",dict(gold_w=0.20)),
               ("no curve rule",dict(use_curve=False)),("cash off-leg",dict(off="cash"))]:
    s=E.stats(build(**{**FINAL,**kw})); print(f"{lbl:30s}{s['cagr']*100:6.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}")

print("\n=== 5. COST SENSITIVITY ===")
for c in [3,6,10,20]:
    s=E.stats(build(**FINAL,cost=c/1e4)); print(f"  {c:2d} bps/turnover: CAGR {s['cagr']*100:5.1f}%  Calmar {s['calmar']:.2f}")

print("\n=== 6. CRISIS STRESS ===")
CR={"1987 aftermath":("1988-01-01","1988-12-31"),"1990 recession":("1990-01-01","1990-12-31"),
    "1994 bond massacre":("1994-01-01","1994-12-31"),"1998 LTCM":("1998-07-01","1998-10-31"),
    "dot-com bust":("2000-03-01","2002-10-31"),"GFC":("2007-10-01","2009-03-31"),
    "2011 EU crisis":("2011-05-01","2011-10-31"),"COVID crash":("2020-02-01","2020-04-30"),
    "2022 bear":("2022-01-01","2022-12-31")}
print(f"{'event':22s}{'FINAL':>9}{'BH NDX':>9}{'BH SPX':>9}{'60/40':>9}")
ndx_r=D.R["qqq"]; spx_r=D.R["spx"]; s4=0.6*D.R["spx"]+0.4*D.R["bond"]
for k,(a,b) in CR.items():
    def tot(r): 
        x=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))].dropna(); return (1+x).prod()-1
    print(f"{k:22s}{tot(rF)*100:8.1f}%{tot(ndx_r)*100:8.1f}%{tot(spx_r)*100:8.1f}%{tot(s4)*100:8.1f}%")

out={"final":E.stats(rF),"base":E.stats(rB),"dec_final":{k:E.stats(rF,a,b) for k,(a,b) in E.DECADES.items()},
     "dec_base":{k:E.stats(rB,a,b) for k,(a,b) in E.DECADES.items()},
     "boot":{"mean":float(means.mean()),"lo":float(np.percentile(means,2.5)),"hi":float(np.percentile(means,97.5)),
             "p":float((means>0).mean())},
     "crisis":{k:{"final":float((1+rF[(rF.index>=pd.Timestamp(a))&(rF.index<=pd.Timestamp(b))].dropna()).prod()-1),
                  "ndx":float((1+ndx_r[(ndx_r.index>=pd.Timestamp(a))&(ndx_r.index<=pd.Timestamp(b))].dropna()).prod()-1),
                  "spx":float((1+spx_r[(spx_r.index>=pd.Timestamp(a))&(spx_r.index<=pd.Timestamp(b))].dropna()).prod()-1),
                  "s4":float((1+s4[(s4.index>=pd.Timestamp(a))&(s4.index<=pd.Timestamp(b))].dropna()).prod()-1)}
              for k,(a,b) in CR.items()},
     "proj":E.proj1000(rF),"eq_final":E.eqpts(rF),"eq_base":E.eqpts(rB)}
json.dump(out,open(f"{CACHE}/final_validate.json","w"),indent=1,default=str)
print(f"\n$1000 FINAL -> {out['proj']}")
print("saved final_validate.json")
