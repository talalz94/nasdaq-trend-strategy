"""
Item 5 completion + the crucial overfit test: param sweeps and TRUE walk-forward
(train 2005-2018 / test 2019-2026) on the gold champions, plus an in-window split
for the stock champion. If an edge is real it survives out-of-sample.
"""
import json, numpy as np, pandas as pd
import bt, assets
from indicators import colored_ma, ma_shift

CACHE = bt.CACHE

def st(r):
    r = np.asarray(r)
    if len(r)==0: return dict(n=0, avg=np.nan, pf=np.nan, win=np.nan)
    gp=r[r>0].sum(); gl=-r[r<=0].sum()
    return dict(n=len(r), avg=r.mean()*1e4, win=(r>0).mean(), pf=(gp/gl if gl>0 else np.inf))

def sig_feat(cache, tr, col):
    d0=cache; vals=[]
    for _,t in tr.iterrows():
        d=cache[t["ticker"]]; si=max(int(t["entry_i"])-1,0)
        v=d[col][si] if col in d else np.nan
        vals.append(bool(v) if np.isfinite(v) else False)
    return np.array(vals)

def walkforward(tr, split_year=2019, label=""):
    tr=tr.copy(); tr["year"]=pd.to_datetime(tr["entry_time"]).dt.year
    tr_tr=tr[tr.year<split_year]; te=tr[tr.year>=split_year]
    a=st(tr_tr["ret"].to_numpy()); b=st(te["ret"].to_numpy())
    print(f"  {label:38s} TRAIN(<{split_year}) n={a['n']:4d} avg={a['avg']:6.1f} pf={a['pf']:.2f} | "
          f"TEST(>={split_year}) n={b['n']:4d} avg={b['avg']:6.1f} pf={b['pf']:.2f} win={b['win']:.2f}")
    return dict(train=a, test=b)

print("=============== GOLD champions — WALK-FORWARD (train<2019 / test>=2019) ===============")
wf={}
# champion A: SQZ_color_long 4h + price_above_MA40
gc4=assets.gold_cache("4h")
tr=bt.run_strategy(gc4,"SQZ_color_long",cost_bps=1.5)
keep=sig_feat(gc4,tr,"mashift_regime_up"); tr_f=tr[keep]
wf["gold_A"]=walkforward(tr_f, label="SQZ_color_long 4h +price>MA40")
# champion B: SQZ_zero_cross 1h + price_above_MA40 + mashift_up
gc1=assets.gold_cache("1h")
tr=bt.run_strategy(gc1,"SQZ_zero_cross",cost_bps=1.5)
keep=sig_feat(gc1,tr,"mashift_regime_up") & sig_feat(gc1,tr,"mashift_up"); tr_f=tr[keep]
wf["gold_B"]=walkforward(tr_f, label="SQZ_zero_cross 1h +trend+osc")
# champion C: MAShift_regime_long 1h
tr=bt.run_strategy(gc1,"MAShift_regime_long",cost_bps=1.5)
wf["gold_C"]=walkforward(tr, label="MAShift_regime_long 1h")
# baseline (unfiltered) for contrast
tr=bt.run_strategy(gc4,"SQZ_color_long",cost_bps=1.5)
wf["gold_A_base"]=walkforward(tr, label="SQZ_color_long 4h (no filter)")

print("\n=============== PARAM SWEEPS (robustness = edge stable across params) ===============")
# EMA slope length sweep on gold 4h
g4=assets.load_gold("4h")
base4=assets.gold_cache("4h")  # for other cols
print("EMA-slope length sweep (gold 4h, long-only trend-follow):")
sweep_ema={}
for L in [5,8,13,21,34,55]:
    cm=colored_ma(g4,L,"EMA")
    d=dict(base4["XAUUSD"]); d["ma8_up"]=cm[f"ma{L}_up"].to_numpy()
    d["ma8_turn_up"]=cm[f"ma{L}_turn_up"].to_numpy(); d["ma8_turn_dn"]=cm[f"ma{L}_turn_dn"].to_numpy()
    tr=bt.run_strategy({"XAUUSD":d},"EMA8_slope_long",cost_bps=1.5)
    s=st(tr["ret"].to_numpy()); sweep_ema[L]=s
    print(f"   EMA{L:<3d} n={s['n']:5d} avg={s['avg']:6.2f}bps pf={s['pf']:.2f} win={s['win']:.2f}")

# MA-Shift threshold sweep on gold 1h (regime_long)
print("MA-Shift threshold sweep (gold 1h, regime_long):")
sweep_ms={}
for th in [0.3,0.5,0.7,1.0]:
    msh=ma_shift(g:=assets.load_gold("1h"), threshold=th)
    d=dict(assets.gold_cache("1h")["XAUUSD"])
    d["mashift_sig_up"]=msh["mashift_sig_up"].to_numpy(); d["mashift_sig_dn"]=msh["mashift_sig_dn"].to_numpy()
    d["mashift_regime_up"]=msh["mashift_regime_up"].to_numpy()
    tr=bt.run_strategy({"XAUUSD":d},"MAShift_regime_long",cost_bps=1.5)
    s=st(tr["ret"].to_numpy()); sweep_ms[th]=s
    print(f"   thr={th} n={s['n']:5d} avg={s['avg']:6.2f}bps pf={s['pf']:.2f} win={s['win']:.2f}")

print("\n=============== STOCK champion — in-window split (42d: 1st half / 2nd half) ===============")
stk=bt.load_cache("1h")
import filter_ladder as FL
df=FL.enrich(stk,"Confluence_oversold",0.02)
df=df[df["sector"].isin(FL.GOOD_SEC)].copy()
df["dt"]=pd.to_datetime(df["entry_time"]); mid=df["dt"].min()+(df["dt"].max()-df["dt"].min())/2
for lbl,part in [("1st half",df[df.dt<mid]),("2nd half",df[df.dt>=mid])]:
    s=st(part["ret"].to_numpy())
    print(f"  stock champ (sector-filtered) {lbl}: n={s['n']:3d} avg={s['avg']:.1f}bps pf={s['pf']:.2f} win={s['win']:.2f}")

json.dump({"wf":{k:{kk:{k3:float(v3) if isinstance(v3,(int,float,np.floating)) else v3
                       for k3,v3 in vv.items()} for kk,vv in v.items()} for k,v in wf.items()},
           "sweep_ema":{str(k):{kk:float(vv) for kk,vv in s.items()} for k,s in sweep_ema.items()},
           "sweep_ms":{str(k):{kk:float(vv) for kk,vv in s.items()} for k,s in sweep_ms.items()}},
          open(f"{CACHE}/walkforward.json","w"), indent=1)
print("\nsaved walkforward.json")
