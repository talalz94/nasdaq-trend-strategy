"""
Hybrid / macro-aware leveraged Nasdaq strategy — ablation ladder.
Base = vol-targeted TQQQ when NDX>EMA100 else off-leg. Layer on:
  bonds off-leg · chop-detector (de-lever OR mean-revert in chop) · macro overlays
  (yield-curve inversion, VIX stress, strong dollar) · selective SQQQ · adaptive EMA.
Each variant measured full-history + per sub-period (esp. 2000-2012). No lookahead.
"""
import json, numpy as np, pandas as pd
import bt
CACHE=bt.CACHE
d=pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
m=pd.read_parquet("data_letf/macro_daily.parquet")
idx=d.index; rf_d=(d["rf"]/252).fillna(0)
# bonds effective: real TLT from 2002 else T-bills
bond_eff=np.where(idx>=pd.Timestamp("2002-08-01"), m["tlt_ret"].values, rf_d.values)
RET={"tqqq":d["tqqq_ret"].fillna(0).values,"qqq":d["qqq_ret"].fillna(0).values,
     "sqqq":d["sqqq_ret"].fillna(0).values,"cash":rf_d.values,
     "bond":pd.Series(bond_eff,index=idx).fillna(0).values,"gold":m["gold_ret"].fillna(0).values}
ndx=d["ndx"]
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    dif=s.diff(); up=dif.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-dif.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
def eff_ratio(s,n=20):    # Kaufman efficiency ratio: 1=trend, 0=chop
    chg=s.diff(n).abs(); vol=s.diff().abs().rolling(n).sum()
    return (chg/vol.replace(0,np.nan)).fillna(0)
VIX=m["vix"].reindex(idx).ffill(); CURVE=m["curve"].reindex(idx); DXY=m["dxy"]; DXY_up=m["dxy_up"]
vol20=ndx.pct_change().rolling(20).std()*np.sqrt(252)
mark=pd.Series(False,index=idx); mark[ndx.resample("W-FRI").last().index.intersection(idx)]=True
COST=3.0/1e4
RS=rsi(ndx); ER=eff_ratio(ndx,20)

def run(cfg):
    ema_n=cfg.get("ema",100)
    trend_on=(ndx>ema(ndx,ema_n))
    if cfg.get("adaptive_ema"):    # shorter EMA when choppy (faster), longer when trending
        choppy0=ER<cfg.get("chop_thr",0.3)
        trend_on=np.where(choppy0, ndx>ema(ndx,50), ndx>ema(ndx,150))
        trend_on=pd.Series(trend_on,index=idx)
    choppy=ER<cfg.get("chop_thr",0.3)
    # macro risk-off overlay
    ro=pd.Series(False,index=idx)
    if cfg.get("curve"): ro=ro|(CURVE<0)
    if cfg.get("vix"): ro=ro|(VIX>cfg.get("vix_thr",28))
    if cfg.get("dollar"): ro=ro|(DXY_up & (DXY>DXY.shift(20)))
    ro=ro.fillna(False)
    riskon=trend_on & (~ro)
    sc=(0.25/vol20).clip(0,1).fillna(1.0)
    off=cfg.get("off","cash")
    w=pd.DataFrame(0.0,index=idx,columns=list(RET))
    # default risk-on instrument
    if cfg.get("chop")=="delever":
        onw=riskon & (~choppy)               # levered only in clean trends
        w["tqqq"]=(onw*sc); w["qqq"]=(riskon & choppy).astype(float)
    elif cfg.get("chop")=="meanrev":
        # trending: TQQQ; choppy: dip-buy QQQ via RSI hysteresis (in<35 out>60)
        trendpart=(riskon & ~choppy)
        w["tqqq"]=trendpart*sc
        trig=pd.Series(np.where(RS<35,1.0,np.where(RS>60,0.0,np.nan)),index=idx).ffill().fillna(0)
        w["qqq"]=(riskon & choppy).astype(float)*trig
    else:
        w["tqqq"]=riskon.astype(float)*sc
    onsum=w[["tqqq","qqq"]].sum(axis=1)
    # selective SQQQ: strong confirmed downtrend
    sqqq_on=pd.Series(False,index=idx)
    if cfg.get("sqqq"):
        sqqq_on=(ndx<ema(ndx,200))&(ema(ndx,50)<ema(ndx,50).shift(20))&(VIX>25)
        sqqq_on=sqqq_on.fillna(False)
    w["sqqq"]=np.where(sqqq_on & (onsum<0.01),0.5,0.0)   # half-size short only when flat
    resid=(1-w[["tqqq","qqq","sqqq"]].sum(axis=1)).clip(lower=0)
    w[off]=w[off]+resid
    w=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    ret=sum(w[a]*RET[a] for a in RET)-w.diff().abs().sum(axis=1).fillna(0)*COST
    return ret

def stats(ret,sub=None):
    r=ret if sub is None else ret[sub]
    if len(r)<50: return dict(cagr=np.nan,maxdd=np.nan,sharpe=np.nan,calmar=np.nan)
    e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min(); sh=r.mean()/r.std()*np.sqrt(252)
    return dict(cagr=float(cagr),maxdd=float(dd),sharpe=float(sh),calmar=float(cagr/abs(dd) if dd<0 else np.nan))

PERIODS={"full":None,"2000-09":("2000","2009"),"2010-19":("2010","2019"),"2020-26":("2020","2026")}
def submask(p):
    if p is None: return None
    return (idx>=pd.Timestamp(p[0]))&(idx<=pd.Timestamp(p[1]+"-12-31"))

LADDER=[
 ("BASE champion (TQQQ/cash)",dict(off="cash")),
 ("+ bonds off-leg",dict(off="bond")),
 ("+ chop de-lever",dict(off="bond",chop="delever")),
 ("+ chop mean-revert",dict(off="bond",chop="meanrev")),
 ("+ yield-curve overlay",dict(off="bond",chop="delever",curve=True)),
 ("+ VIX overlay",dict(off="bond",chop="delever",vix=True)),
 ("+ curve+VIX",dict(off="bond",chop="delever",curve=True,vix=True)),
 ("+ dollar overlay",dict(off="bond",chop="delever",curve=True,vix=True,dollar=True)),
 ("adaptive-EMA (50 chop/150 trend)",dict(off="bond",adaptive_ema=True)),
 ("selective SQQQ",dict(off="bond",chop="delever",curve=True,vix=True,sqqq=True)),
]
rows=[]
print(f"{'variant':34s}{'FULL cagr/DD/Cal':>22s}{'2000-09 cagr/DD':>18s}{'2010-19':>10s}{'2020-26':>10s}")
for name,cfg in LADDER:
    ret=run(cfg); rec={"name":name}
    line=f"{name:34s}"
    for p,pr in PERIODS.items():
        s=stats(ret,submask(pr)); rec[p]=s
        if p=="full": line+=f"{s['cagr']*100:6.1f}%/{s['maxdd']*100:3.0f}%/{s['calmar']:.2f}".rjust(22)
        elif p=="2000-09": line+=f"{s['cagr']*100:6.1f}%/{s['maxdd']*100:3.0f}%".rjust(18)
        else: line+=f"{s['cagr']*100:6.1f}%".rjust(10)
    print(line); rows.append(rec)

# benchmarks
bh={}
for a in ["qqq","tqqq"]:
    r=pd.Series(RET[a],index=idx); bh[a]={**stats(r),"s2000":stats(r,submask(('2000','2009')))}
    print(f"BH {a.upper():4s}: full CAGR {bh[a]['cagr']*100:.1f}% DD {bh[a]['maxdd']*100:.0f}% | 2000-09 {bh[a]['s2000']['cagr']*100:.1f}%")

# equity curves + $1000 rolling projection for the WINNER (bonds+chop-delever+curve)
WIN=dict(off="bond",chop="delever",curve=True)
def eqpts(ret):
    e=(1+ret).cumprod().resample("ME").last().dropna(); return [{"t":t.strftime("%Y-%m"),"e":round(float(v),4)} for t,v in e.items()]
win_ret=run(WIN); base_ret=run(dict(off="cash"))
proj={}
for hn,days in [("1 yr",252),("3 yr",756),("5 yr",1260),("10 yr",2520)]:
    e=(1+win_ret).cumprod().values; w=e[days:]/e[:-days]
    proj[hn]={"typ":round(1000*np.median(w)),"lo":round(1000*np.percentile(w,10)),"hi":round(1000*np.percentile(w,90))}
out={"ladder":rows,"periods":list(PERIODS),"bh":bh,
     "winner":{"cfg":"bonds off + de-lever in chop + yield-curve overlay","stats_full":stats(win_ret),
               "sub":{p:stats(win_ret,submask(pr)) for p,pr in PERIODS.items()},"proj1000":proj},
     "eq_base":eqpts(base_ret),"eq_win":eqpts(win_ret),
     "eq_bh_qqq":eqpts(pd.Series(RET["qqq"],index=idx)),"eq_bh_tqqq":eqpts(pd.Series(RET["tqqq"],index=idx))}
json.dump(out,open(f"{CACHE}/hybrid.json","w"),indent=1,default=str)
print(f"\nWINNER $1000 -> {proj}")
print("saved hybrid.json")
