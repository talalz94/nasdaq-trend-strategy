"""
Turn the daily sweep winners into deployable results:
  - realistic capital-constrained PORTFOLIO curve for the champion
  - a REGIME-SWITCHED strategy (bull=MA-Shift, bear=SQZ-color, choppy=WT-70) portfolio
  - buy&hold benchmark + edge-over-drift
  - daily -> weekly timeframe check
"""
import json, numpy as np, pandas as pd
import bt, sweep, regime as R
from indicators import compute_all
CACHE = bt.CACHE

cache = sweep.load_cache()            # daily ind cache (SWEEP_CACHE=ind_alpaca_1d.parquet not set here -> default 1h!)
# force daily source:
def load_daily():
    df = pd.read_parquet(f"{CACHE}/ind_alpaca_1d.parquet")
    return {t: {c: g[c].to_numpy() for c in sweep.CACHE_COLS if c in g} for t,g in df.groupby("ticker",sort=False)}
cache = load_daily()
reg = R.regime_series(cache)

def st(r):
    r=np.asarray(r,float)
    if len(r)==0: return dict(n=0,avg=np.nan,pf=np.nan,win=np.nan)
    gp,gl=r[r>0].sum(),-r[r<=0].sum()
    return dict(n=len(r),avg=r.mean()*1e4,win=(r>0).mean(),pf=(gp/gl if gl>0 else np.inf))

def portfolio(trades, capital=100_000, max_pos=15, risk=0.06):
    ev=[]
    for i,t in trades.reset_index(drop=True).iterrows():
        ev.append((pd.Timestamp(t.entry_time),"o",i)); ev.append((pd.Timestamp(t.exit_time),"c",i))
    ev.sort(key=lambda e:(e[0], 0 if e[1]=="c" else 1))
    eq=capital; slots=max_pos; alloc={}; curve=[]; rmap=trades.reset_index(drop=True)["ret"]
    for tm,k,i in ev:
        if k=="c":
            if i in alloc: eq+=alloc.pop(i)*rmap.iloc[i]; slots+=1
        else:
            if slots>0: alloc[i]=eq*risk; slots-=1
        curve.append((tm,eq))
    ec=pd.DataFrame(curve,columns=["t","e"]).groupby("t").last()
    yrs=(ec.index[-1]-ec.index[0]).days/365.25
    cagr=(eq/capital)**(1/yrs)-1
    dd=(ec["e"]/ec["e"].cummax()-1).min()
    dr=ec["e"].resample("D").last().dropna().pct_change().dropna()
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else np.nan
    return ec, dict(cagr=cagr, maxdd=float(dd), sharpe=float(sharpe), final=float(eq), yrs=yrs,
                    total=float(eq/capital-1))

def spec_trades(spec):
    df=sweep.run_spec(cache, spec)
    df["regime"]=R.tag_trades(df, reg)
    return df

print("=== CHAMPION portfolio: MA-Shift + trend filter ===")
champ=spec_trades(dict(name="mashift_reg", type="mashift", regime=True))
for lbl,sub in [("full 10yr",champ),("test 2022-26",champ[pd.to_datetime(champ.entry_time).dt.year>=2022])]:
    ec,s=portfolio(sub)
    print(f"  {lbl}: {s['total']*100:+.0f}% total, CAGR {s['cagr']*100:.1f}%, maxDD {s['maxdd']*100:.0f}%, Sharpe {s['sharpe']:.2f}, n={len(sub)}")

print("\n=== REGIME-SWITCHED portfolio (bull=MAShift, bear=SQZcolor, choppy=WT-70) ===")
bull=spec_trades(dict(name="mashift_reg",type="mashift",regime=True))
bear=spec_trades(dict(name="sqz_color",type="sqz_color"))
chop=spec_trades(dict(name="wt_os-70",type="wt_os",os=-70))
combo=pd.concat([bull[bull.regime=="BULL"], bear[bear.regime=="BEAR"], chop[chop.regime=="CHOPPY"]], ignore_index=True)
combo=combo.sort_values("entry_time")
for lbl,sub in [("full 10yr",combo),("test 2022-26",combo[pd.to_datetime(combo.entry_time).dt.year>=2022])]:
    ec,s=portfolio(sub)
    print(f"  {lbl}: {s['total']*100:+.0f}% total, CAGR {s['cagr']*100:.1f}%, maxDD {s['maxdd']*100:.0f}%, Sharpe {s['sharpe']:.2f}, n={len(sub)}")
eccombo,scombo=portfolio(combo)

# buy & hold equal-weight benchmark
mi=R.market_index(cache)
bh_total=mi["idx"].iloc[-1]/mi["idx"].iloc[0]-1
bh_cagr=(1+bh_total)**(1/((mi.index[-1]-mi.index[0]).days/365.25))-1
print(f"\nBuy&hold (equal-weight): {bh_total*100:+.0f}% total, CAGR {bh_cagr*100:.1f}%")

# champion edge over drift (matched ~ hold days)
hold=int(champ["bars"].mean())
c=2*sweep.COST/1e4; rr=[]
for d in cache.values():
    C=d["close"]
    if len(C)>hold: f=C[hold:]/C[:-hold]-1-c; rr.append(f[np.isfinite(f)])
drift=np.concatenate(rr).mean()*1e4
print(f"champion avg {st(champ['ret'])['avg']:.0f}bps vs {hold}-day drift {drift:.0f}bps -> edge {st(champ['ret'])['avg']-drift:.0f}bps")

# ---- daily -> weekly timeframe check ----
print("\n=== TIMEFRAME: daily vs weekly (top signals) ===")
raw=pd.read_parquet("data_alpaca/stocks_1d.parquet"); raw["datetime"]=pd.to_datetime(raw["datetime"])
wk=[]
for t,g in raw.groupby("ticker",sort=False):
    s=g.set_index("datetime")[["open","high","low","close","volume"]].resample("W-FRI").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["close"])
    if len(s)<200: continue
    s=s.reset_index(); s["ticker"]=t; s["date"]=s["datetime"].dt.date.astype(str); wk.append(s)
wkraw=pd.concat(wk,ignore_index=True)
wkcache={}
for t,g in wkraw.groupby("ticker",sort=False):
    ind=compute_all(g[["datetime","date","open","high","low","close","volume"]])
    wkcache[t]={c:ind[c].to_numpy() for c in sweep.CACHE_COLS if c in ind}
tf_rows=[]
for lbl,cch in [("daily",cache),("weekly",wkcache)]:
    for spec in [dict(name="mashift_reg",type="mashift",regime=True),
                 dict(name="wt_os-70",type="wt_os",os=-70),
                 dict(name="sqz_zero_long",type="sqz_zero")]:
        df=sweep.run_spec(cch,spec); s=st(df["ret"])
        tf_rows.append(dict(tf=lbl,signal=spec["name"],**s))
        print(f"  {lbl:6s} {spec['name']:16s} n={s['n']:6d} avg={s['avg']:6.0f}bps pf={s['pf']:.2f} win={s['win']:.2f}")

# save for report
out={"champ_port":{}, "combo_port":{}, "bh":{"total":bh_total*100,"cagr":bh_cagr*100},
     "combo_equity":[{"t":t.strftime("%Y-%m"),"e":round(float(v))} for t,v in eccombo["e"].resample("ME").last().dropna().items()],
     "tf":tf_rows, "regime_days":reg.value_counts().to_dict()}
for lbl,sub in [("full",champ),("test",champ[pd.to_datetime(champ.entry_time).dt.year>=2022])]:
    _,s=portfolio(sub); out["champ_port"][lbl]=s
for lbl,sub in [("full",combo),("test",combo[pd.to_datetime(combo.entry_time).dt.year>=2022])]:
    _,s=portfolio(sub); out["combo_port"][lbl]=s
json.dump(out, open(f"{CACHE}/finalize_daily.json","w"), indent=1, default=str)
print("\nsaved finalize_daily.json")
