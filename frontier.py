"""
PHASE B2 — Tune the last leg. Framed as an EFFICIENT FRONTIER, not a single winner.

B1 showed every hybrid variant sits at Calmar ~0.50, i.e. the chop filter is a
RISK DIAL, not alpha. So the real test: at the SAME realized drawdown, does any
rule beat simply holding less leverage (a lower vol target)?

Variants:
  volN        pure dial: trend + vol-target N, cash off-leg  (the benchmark curve)
  chopX/pP    de-lever to P (0=QQQ,0.5=half TQQQ) when ER<X
  curve*      raw inversion / confirmed (inverted AND trend down) / steepening-out
  boost*      MORE aggressive: raise vol target when trend is strong & clean
Anything ABOVE the pure-dial curve is genuine improvement; anything on it is noise.
"""
import json, itertools, numpy as np, pandas as pd
import bt
CACHE=bt.CACHE
d=pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
m=pd.read_parquet("data_letf/macro_daily.parquet")
idx=d.index; rf_d=(d["rf"]/252).fillna(0)
bond=pd.Series(np.where(idx>=pd.Timestamp("2002-08-01"),m["tlt_ret"].values,rf_d.values),index=idx).fillna(0)
R={"tqqq":d["tqqq_ret"].fillna(0),"qqq":d["qqq_ret"].fillna(0),"cash":rf_d,"bond":bond}
ndx=d["ndx"]; ema100=ndx.ewm(span=100,adjust=False).mean()
vol=ndx.pct_change().rolling(20).std()*np.sqrt(252)
er=(ndx.diff(20).abs()/ndx.diff().abs().rolling(20).sum().replace(0,np.nan))
curve=m["curve"].reindex(idx)
mark=pd.Series(False,index=idx); mark[ndx.resample("W-FRI").last().index.intersection(idx)]=True
COST=3.0/1e4
above=ndx>ema100
dist=(ndx/ema100-1)                      # how far above the trend line
curve_inv=(curve<0).fillna(False)
# "confirmed" recession signal: curve was inverted in the last 12m AND trend is now down
inv_recent=curve_inv.rolling(252,min_periods=1).max().astype(bool)
confirmed=(inv_recent & ~above)
# steepening out of inversion (classic recession trigger)
steepen=(inv_recent & (curve>curve.shift(60)) & (curve<0.5)).fillna(False)

def build(voltgt=0.25, chop_thr=None, chop_pos=0.0, curve_mode=None, off="cash",
          boost=None, boost_thr=0.40):
    scale=(voltgt/vol).clip(0,1).fillna(1.0)
    ron=above.copy()
    if curve_mode=="raw":       ron=ron&~curve_inv
    elif curve_mode=="confirm": ron=ron&~confirmed
    elif curve_mode=="steepen": ron=ron&~steepen
    w=pd.DataFrame(0.0,index=idx,columns=list(R))
    if boost:                                   # more aggressive in strong clean trends
        strong=(er>=boost_thr)&(dist>0.02)
        scale=np.minimum(scale*np.where(strong,boost,1.0),1.0)
        scale=pd.Series(scale,index=idx)
    if chop_thr is None:
        w["tqqq"]=ron*scale
    else:
        clean=(er>=chop_thr).fillna(False)
        w["tqqq"]=(ron&clean)*scale + (ron&~clean)*scale*chop_pos
        w["qqq"]=(ron&~clean).astype(float)*(1.0 if chop_pos==0 else 0.0)
    w["cash"]+= (1-w[["tqqq","qqq"]].sum(axis=1)).clip(lower=0)*(1 if off=="cash" else 0)
    if off=="bond":
        w["bond"]=(1-w[["tqqq","qqq"]].sum(axis=1)).clip(lower=0)
    w=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(w[a]*R[a] for a in R)-w.diff().abs().sum(axis=1).fillna(0)*COST

def stats(r,msk=None):
    r=r if msk is None else r[msk]
    e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(cagr),maxdd=float(dd),sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                calmar=float(cagr/abs(dd)) if dd<0 else np.nan)
m0009=(idx>=pd.Timestamp("2000-01-01"))&(idx<=pd.Timestamp("2009-12-31"))

rows=[]
def add(name,grp,**kw):
    r=build(**kw); s=stats(r); s09=stats(r,m0009)
    rows.append(dict(name=name,group=grp,**s,cagr0009=s09["cagr"],maxdd0009=s09["maxdd"]))
    return rows[-1]

# 1) pure vol-target dial (the benchmark frontier)
for v in [0.12,0.15,0.20,0.25,0.30,0.40,0.60]:
    add(f"vol{int(v*100)} (pure dial)","dial",voltgt=v)
# 2) chop filter variants
for thr in [0.15,0.20,0.25,0.30]:
    add(f"chop<{thr} -> QQQ","chop",chop_thr=thr,chop_pos=0.0)
    add(f"chop<{thr} -> half TQQQ","chop",chop_thr=thr,chop_pos=0.5)
# 3) curve variants (with bonds off-leg, as in hybrid B)
for cm in ["raw","confirm","steepen"]:
    add(f"curve:{cm}","curve",curve_mode=cm,off="bond")
    add(f"curve:{cm}+chop<0.30","curve",curve_mode=cm,chop_thr=0.30,off="bond")
# 4) AGGRESSIVE: boost leverage in strong clean trends
for b in [1.2,1.5,2.0]:
    add(f"boost x{b} in strong trend","boost",boost=b)
    add(f"boost x{b} + curve:confirm","boost",boost=b,curve_mode="confirm",off="bond")
# 5) the two incumbents
add("A: champion (vol25)","incumbent",voltgt=0.25)
add("B: hybrid (curve raw+chop+bond)","incumbent",curve_mode="raw",chop_thr=0.30,off="bond")

res=pd.DataFrame(rows)
res.to_csv(f"{CACHE}/frontier.csv",index=False)

# frontier test: interpolate the pure-dial CAGR at each variant's drawdown
dial=res[res.group=="dial"].sort_values("maxdd")
def dial_cagr_at(dd):
    return float(np.interp(dd, dial["maxdd"].values, dial["cagr"].values))
res["dial_equiv"]=res["maxdd"].apply(dial_cagr_at)
res["edge_vs_dial"]=res["cagr"]-res["dial_equiv"]      # >0 = genuinely better than just de-levering

print("=== ALL VARIANTS (sorted by edge over the pure vol-dial at equal drawdown) ===")
print(f"{'variant':34s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'2000-09':>9}{'vs dial':>9}")
for _,r in res.sort_values("edge_vs_dial",ascending=False).iterrows():
    print(f"{r['name']:34s}{r['cagr']*100:6.1f}%{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}"
          f"{r['cagr0009']*100:8.1f}%{r['edge_vs_dial']*100:+8.1f}%")

print("\n=== KEY COMPARISON: does the chop filter beat simply de-levering? ===")
hyb=res[res.name=="B: hybrid (curve raw+chop+bond)"].iloc[0]
print(f"  hybrid B      : CAGR {hyb['cagr']*100:.1f}% at maxDD {hyb['maxdd']*100:.0f}%")
print(f"  pure vol-dial : CAGR {hyb['dial_equiv']*100:.1f}% at the SAME drawdown")
print(f"  -> chop+curve rules are worth {hyb['edge_vs_dial']*100:+.1f}%/yr vs just holding less leverage")
json.dump({"rows":res.round(4).to_dict("records")},open(f"{CACHE}/frontier.json","w"),indent=1,default=str)
print("\nsaved frontier.csv/json")
