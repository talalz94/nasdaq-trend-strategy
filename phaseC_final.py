"""
PHASE C final — combine the two winners (bond off-leg + small static gold sleeve),
sweep the sleeve size, and benchmark against every proven alternative.
"""
import json, numpy as np, pandas as pd, yfinance as yf
import bt, frontier as F
CACHE=bt.CACHE
R=pd.read_parquet("data_letf/multi_panel.parquet"); idx=R.index
START=pd.Timestamp("1988-01-01"); COST=3.0/1e4
mark=pd.Series(False,index=idx); mark[F.ndx.resample("W-FRI").last().index.intersection(idx)]=True
scale=(0.20/F.vol).clip(0,1).fillna(1.0)
ron=F.above & ~F.confirmed
clean=(F.er>=0.25).fillna(False)
ndx_w=(ron&clean)*scale + (ron&~clean)*scale*0.5
COLS=["tqqq","cash","bond","gold","spx"]

def engine(off="bond", gold_w=0.0):
    """B-v2 engine, chosen off-leg, plus an always-on gold sleeve of gold_w."""
    core=1.0-gold_w
    w=pd.DataFrame(0.0,index=idx,columns=COLS)
    w["tqqq"]=ndx_w*core
    w[off]+= (core-ndx_w*core).clip(lower=0)
    w["gold"]+=gold_w
    w=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(w[c]*R[c].fillna(0) for c in COLS)-w.diff().abs().sum(axis=1).fillna(0)*COST

def stats(r,a=None,b=None):
    if a: r=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    r=r[r.index>=START].dropna()
    if len(r)<100: return dict(cagr=np.nan,maxdd=np.nan,sharpe=np.nan,calmar=np.nan)
    e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(cagr),maxdd=float(dd),sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                calmar=float(cagr/abs(dd)) if dd<0 else np.nan)

print("=== GOLD-SLEEVE SWEEP (bond off-leg) ===")
print(f"{'gold sleeve':14s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}{'|1992-95':>9}{'2000-10':>9}")
sweep=[]
for g in [0.0,0.05,0.10,0.15,0.20,0.25,0.30]:
    r=engine("bond",g); s=stats(r)
    a=stats(r,"1992-01-01","1995-12-31"); b=stats(r,"2000-01-01","2010-12-31")
    print(f"{int(g*100)}%{'':11s}{s['cagr']*100:6.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:7.2f}"
          f"{a['cagr']*100:8.1f}%{b['cagr']*100:8.1f}%")
    sweep.append(dict(gold=g,**s,w9295=a["cagr"],w0010=b["cagr"]))

BEST_G=0.15
final=engine("bond",BEST_G)

# ---------- benchmarks incl. Berkshire & Bitcoin ----------
def ext(t):
    h=yf.Ticker(t).history(period="max",auto_adjust=True)["Close"]
    h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).pct_change()
brk=ext("BRK-B"); btc=ext("BTC-USD")

BM={
 "★ FINAL: B-v2 + bond off-leg + 15% gold":final,
 "B-v2 (cash off-leg)":engine("cash",0.0),
 "A champion (TQQQ/cash)":F.build(voltgt=0.25),
 "Buy&hold Nasdaq-100":R["ndx"].fillna(0),
 "Buy&hold S&P 500":R["spx"].fillna(0),
 "Buy&hold gold miners":R["gold"].fillna(0),
 "Buy&hold 10y Treasuries":R["bond"].fillna(0),
 "60/40 (SPX/bonds)":0.6*R["spx"].fillna(0)+0.4*R["bond"].fillna(0),
 "Berkshire (BRK-B, 1996+)":brk,
 "Bitcoin (2014+)":btc,
}
DEC={"1988-89":("1988-01-01","1989-12-31"),"1990s":("1990-01-01","1999-12-31"),
     "2000s":("2000-01-01","2009-12-31"),"2010s":("2010-01-01","2019-12-31"),
     "2020s":("2020-01-01","2026-07-24")}
print("\n=== BENCHMARK TABLE (full period from each asset's start, >=1988) ===")
print(f"{'strategy':40s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}")
rows=[]
for nm,r in BM.items():
    s=stats(r); print(f"{nm:40s}{s['cagr']*100:6.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:7.2f}")
    d={k:stats(r,a,b)["cagr"] for k,(a,b) in DEC.items()}
    rows.append(dict(name=nm,**s,dec={k:(None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(v,4)) for k,v in d.items()}))

print("\n=== DECADE VIEW (CAGR) ===")
print(f"{'strategy':40s}"+"".join(f"{k:>10s}" for k in DEC))
for row in rows:
    print(f"{row['name']:40s}"+"".join(
        (f"{row['dec'][k]*100:9.1f}%" if row['dec'][k] is not None else f"{'—':>10s}") for k in DEC))

# $1000 projection for the final
e=(1+final[final.index>=START]).cumprod().values
proj={}
for hn,dd in [("1 yr",252),("3 yr",756),("5 yr",1260),("10 yr",2520)]:
    w=e[dd:]/e[:-dd]; proj[hn]={"typ":round(1000*np.median(w)),"lo":round(1000*np.percentile(w,10)),"hi":round(1000*np.percentile(w,90))}
print(f"\n$1000 in the FINAL strategy -> {proj}")

def eqpts(r):
    x=(1+r[r.index>=START].fillna(0)).cumprod().resample("ME").last().dropna()
    return [{"t":t.strftime("%Y-%m"),"e":round(float(v),4)} for t,v in x.items()]
keep=["★ FINAL: B-v2 + bond off-leg + 15% gold","B-v2 (cash off-leg)","A champion (TQQQ/cash)",
      "Buy&hold Nasdaq-100","60/40 (SPX/bonds)"]
json.dump({"sweep":sweep,"rows":rows,"proj":proj,"decades":list(DEC),
           "eq":{k:eqpts(BM[k]) for k in keep}},open(f"{CACHE}/phaseC.json","w"),indent=1,default=str)
print("saved phaseC.json")
