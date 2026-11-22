"""
PHASE B3 — Is the tuned strategy real, or did I curve-fit ~50 variants?

Three independent checks:
  1. WALK-FORWARD: pick the best variant by Calmar on TRAIN only, then score it on
     unseen TEST. Both directions (early->late and late->early). Compare to the
     "oracle" (best on test) to measure selection degradation.
  2. PARAMETER STABILITY: is the neighbourhood of the chosen params smooth, or a
     knife-edge spike? (Smooth = robust; spike = fit to noise.)
  3. CREDIT-SPREAD overlay (HYG/IEF, 2007+) as an extra leading macro signal.
"""
import json, numpy as np, pandas as pd
import bt, frontier as F
CACHE=bt.CACHE
idx=F.idx

GRID=[]
for thr in [0.20,0.25,0.30]:
    for pos in [0.35,0.5,0.65]:
        for cm in [None,"confirm"]:
            for off in ["cash","bond"]:
                for v in [0.20,0.25,0.30]:
                    GRID.append(dict(voltgt=v,chop_thr=thr,chop_pos=pos,curve_mode=cm,off=off))
GRID.append(dict(voltgt=0.25))                                    # A champion
GRID.append(dict(curve_mode="raw",chop_thr=0.30,off="bond"))      # B old hybrid
def nm(k):
    if k=={"voltgt":0.25}: return "A champion"
    if k.get("curve_mode")=="raw": return "B old hybrid"
    return (f"vol{int(k['voltgt']*100)} chop<{k['chop_thr']}->{int(k['chop_pos']*100)}%"
            f"{' +confirm' if k.get('curve_mode')=='confirm' else ''}{' +bond' if k.get('off')=='bond' else ''}")

CACHE_R={}
def rets(k):
    key=json.dumps(k,sort_keys=True)
    if key not in CACHE_R: CACHE_R[key]=F.build(**k)
    return CACHE_R[key]

def msk(a,b): return (idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b))
SPLITS=[("early->late","1985-10-02","2005-12-31","2006-01-01","2026-07-24"),
        ("late->early","2006-01-01","2026-07-24","1985-10-02","2005-12-31")]

print("=== 1. WALK-FORWARD (choose on TRAIN by Calmar, score on unseen TEST) ===")
wf=[]
for lbl,a1,b1,a2,b2 in SPLITS:
    tr,te=msk(a1,b1),msk(a2,b2)
    scored=[(nm(k),F.stats(rets(k),tr),F.stats(rets(k),te)) for k in GRID]
    pick=max(scored,key=lambda x:x[1]["calmar"])
    oracle=max(scored,key=lambda x:x[2]["calmar"])
    inc={n:(s1,s2) for n,s1,s2 in scored}
    print(f"\n  [{lbl}]  train {a1[:4]}-{b1[:4]} -> test {a2[:4]}-{b2[:4]}")
    print(f"    picked on train : {pick[0]}")
    print(f"      train  CAGR {pick[1]['cagr']*100:5.1f}%  DD {pick[1]['maxdd']*100:4.0f}%  Calmar {pick[1]['calmar']:.2f}")
    print(f"      TEST   CAGR {pick[2]['cagr']*100:5.1f}%  DD {pick[2]['maxdd']*100:4.0f}%  Calmar {pick[2]['calmar']:.2f}   <-- out of sample")
    print(f"      oracle (best on test) {oracle[0]}: Calmar {oracle[2]['calmar']:.2f}  -> selection cost {oracle[2]['calmar']-pick[2]['calmar']:.2f}")
    for base in ["A champion","B old hybrid"]:
        s=inc[base][1]; print(f"      vs {base:14s} TEST CAGR {s['cagr']*100:5.1f}%  DD {s['maxdd']*100:4.0f}%  Calmar {s['calmar']:.2f}")
    wf.append(dict(split=lbl,pick=pick[0],train=pick[1],test=pick[2],oracle=oracle[0],
                   oracle_calmar=oracle[2]["calmar"],
                   base={b:inc[b][1] for b in ["A champion","B old hybrid"]}))

print("\n=== 2. PARAMETER STABILITY around the chosen region (full history Calmar) ===")
print(f"{'chop_thr':>9}{'pos=0.35':>10}{'0.50':>8}{'0.65':>8}{'0.80':>8}{'1.00':>8}")
stab=[]
for thr in [0.15,0.20,0.25,0.30,0.35]:
    row=[]
    for pos in [0.35,0.5,0.65,0.8,1.0]:
        s=F.stats(F.build(voltgt=0.20,chop_thr=thr,chop_pos=pos,curve_mode="confirm"))
        row.append(s["calmar"]); stab.append(dict(thr=thr,pos=pos,calmar=s["calmar"],cagr=s["cagr"],maxdd=s["maxdd"]))
    print(f"{thr:>9}"+"".join(f"{v:>8.2f}" for v in row))
print("  (smooth surface = robust; a lone spike = fit to noise)")

print("\n=== 3. CREDIT-SPREAD overlay (HYG/IEF trend, 2007+ only) ===")
m=pd.read_parquet("data_letf/macro_daily.parquet")
credit_ok=m["credit_up"].reindex(idx).ffill().fillna(True).astype(bool)
m07=msk("2007-01-01","2026-07-24")
def with_credit(k):
    base=F.build(**k)
    kk=dict(k); r=F.build(**kk)
    # apply credit veto: when credit deteriorating, force off-leg (approximate overlay)
    scale=(kk.get("voltgt",0.25)/F.vol).clip(0,1).fillna(1.0)
    ron=F.above & credit_ok
    if kk.get("curve_mode")=="confirm": ron=ron&~F.confirmed
    clean=(F.er>=kk.get("chop_thr",0.25)).fillna(False)
    w=pd.DataFrame(0.0,index=idx,columns=list(F.R))
    w["tqqq"]=(ron&clean)*scale+(ron&~clean)*scale*kk.get("chop_pos",0.5)
    off="bond" if kk.get("off")=="bond" else "cash"
    w[off]=(1-w["tqqq"]).clip(lower=0)
    w=w.where(F.mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(w[a]*F.R[a] for a in F.R)-w.diff().abs().sum(axis=1).fillna(0)*F.COST
best_k=dict(voltgt=0.20,chop_thr=0.25,chop_pos=0.5,curve_mode="confirm")
for lbl,r in [("tuned (no credit)",rets(best_k)),("tuned + credit veto",with_credit(best_k))]:
    s=F.stats(r,m07); print(f"  {lbl:22s} 2007-2026: CAGR {s['cagr']*100:5.1f}%  DD {s['maxdd']*100:4.0f}%  Calmar {s['calmar']:.2f}")

json.dump({"wf":wf,"stability":stab},open(f"{CACHE}/validate.json","w"),indent=1,default=str)
print("\nsaved validate.json")
