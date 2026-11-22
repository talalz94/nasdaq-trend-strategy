"""
BEST-ASSET SEARCH — is the Nasdaq actually the right thing to run this on?
The earlier 8-market test asked "do the RULES generalize". This asks the different
question: "is there a BETTER ASSET for these rules?"

Identical rules everywhere (EMA-100 gate, vol-target 0.20, ER<0.25 -> half size,
Aug-Sep half, weekly rebalance, bond off-leg, no gold sleeve), each asset with its own
simulated leveraged sleeve. Ranked by Calmar over each asset's full history, THEN
re-checked on a common period and for sub-period stability (because searching 30 assets
for a winner is itself a selection process that can manufacture false champions).
"""
import json, numpy as np, pandas as pd, yfinance as yf
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE; rf_d=D.rf_d; COST=E.COST; EXP=0.0095
BOND=D.R["bond"]
mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)

U={"^NDX":"Nasdaq-100","^SOX":"Semiconductors","^GSPC":"S&P 500","^RUT":"Russell 2000",
   "^NSEI":"India Nifty","^KS11":"Korea","^TWII":"Taiwan","^N225":"Nikkei","^GDAXI":"DAX",
   "^HSI":"Hang Seng","^BVSP":"Brazil","^MXX":"Mexico","^STOXX50E":"EuroStoxx",
   "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SMH":"Semis ETF","XLK":"Tech ETF","XBI":"Biotech",
   "ARKK":"ARKK","AAPL":"Apple","MSFT":"Microsoft","NVDA":"Nvidia","AMZN":"Amazon",
   "GOOGL":"Google","TSLA":"Tesla","GC=F":"Gold","^XAU":"Gold miners","^IXIC":"Nasdaq Comp","^DJI":"Dow"}
print("fetching universe ...",flush=True)
PX={}
for t,n in U.items():
    try:
        h=yf.Ticker(t).history(period="max")["Close"]
        h.index=h.index.tz_localize(None).normalize()
        s=h[~h.index.duplicated()].reindex(idx).ffill()
        if s.dropna().shape[0]>800: PX[n]=s
    except Exception: pass
print(f"  {len(PX)} assets usable",flush=True)

def strat(p,lev=3,voltgt=0.20,chop=0.25,pos=0.5,seasonal=True,a=None,b=None):
    r1=p.pct_change()
    rl=(lev*r1-EXP/252-(lev-1)*rf_d).fillna(0) if lev>1 else (r1-0.0020/252).fillna(0)
    above=p>p.ewm(span=100,adjust=False).mean()
    vol=r1.rolling(20).std()*np.sqrt(252)
    er=(p.diff(20).abs()/p.diff().abs().rolling(20).sum().replace(0,np.nan))
    sc=(voltgt/vol).clip(0,1).fillna(1.0)
    if seasonal: sc=(sc*AS).clip(0,1)
    clean=(er>=chop).fillna(False)
    eq=(above&clean)*sc+(above&~clean)*sc*pos
    w=pd.DataFrame({"lev":eq,"bond":(1-eq).clip(lower=0)},index=idx)
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    r=ww["lev"]*rl+ww["bond"]*BOND.fillna(0)-ww.diff().abs().sum(axis=1).fillna(0)*COST
    valid=p.notna()
    return r[valid], r1.fillna(0)[valid]

def st(r,a=None,b=None):
    if a: r=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    r=r.dropna()
    if len(r)<400: return None
    e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    c=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(c),maxdd=float(dd),calmar=float(c/abs(dd)) if dd<0 else np.nan,
                sharpe=float(r.mean()/r.std()*np.sqrt(252)),yrs=float(yrs))

rows=[]
for n,p in PX.items():
    for lev in [1,2,3]:
        r,r1=strat(p,lev=lev); s=st(r)
        if s is None: continue
        bh=st(r1)
        rows.append(dict(asset=n,lev=lev,**s,bh_cagr=bh["cagr"],bh_dd=bh["maxdd"],
                         start=str(p.dropna().index.min().date())))
df=pd.DataFrame(rows)
best=df.sort_values("calmar",ascending=False)
print(f"\n=== TOP 20 by CALMAR (own full history — NOT comparable periods) ===")
print(f"{'asset':16s}{'lev':>4}{'from':>12}{'yrs':>5}{'CAGR':>8}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}{'BH cagr':>9}")
for _,r in best.head(20).iterrows():
    print(f"{r['asset']:16s}{r['lev']:>3}x{r['start']:>12}{r['yrs']:5.0f}{r['cagr']*100:7.1f}%"
          f"{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}{r['sharpe']:7.2f}{r['bh_cagr']*100:8.1f}%")

print(f"\n=== COMMON PERIOD 2015-2026 (apples-to-apples, all assets alive) ===")
rows2=[]
for n,p in PX.items():
    if p.dropna().index.min()>pd.Timestamp("2015-01-01"): continue
    for lev in [1,2,3]:
        r,_=strat(p,lev=lev); s=st(r,"2015-01-01","2026-07-24")
        if s: rows2.append(dict(asset=n,lev=lev,**s))
d2=pd.DataFrame(rows2).sort_values("calmar",ascending=False)
print(f"{'asset':16s}{'lev':>4}{'CAGR':>8}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}")
for _,r in d2.head(15).iterrows():
    print(f"{r['asset']:16s}{r['lev']:>3}x{r['cagr']*100:7.1f}%{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}{r['sharpe']:7.2f}")

print(f"\n=== SUB-PERIOD STABILITY of the top candidates (is the winner just lucky?) ===")
cand=["Nasdaq-100","Semiconductors","Bitcoin","Nvidia","India Nifty","Apple","Microsoft","Tech ETF","Semis ETF"]
print(f"{'asset':16s}{'lev':>4}{'1st half':>19}{'2nd half':>19}  consistent?")
stab=[]
for n in cand:
    if n not in PX: continue
    p=PX[n]; v=p.dropna(); mid=v.index[len(v)//2]
    for lev in [2,3]:
        r,_=strat(p,lev=lev)
        s1=st(r,str(v.index[0].date()),str(mid.date())); s2=st(r,str(mid.date()),str(v.index[-1].date()))
        if not s1 or not s2: continue
        ok="YES" if (s1["calmar"]>0.3 and s2["calmar"]>0.3) else ("no" if min(s1["calmar"],s2["calmar"])<0.15 else "weak")
        print(f"{n:16s}{lev:>3}x  Cal {s1['calmar']:5.2f} ({s1['cagr']*100:5.1f}%)  Cal {s2['calmar']:5.2f} ({s2['cagr']*100:5.1f}%)  {ok}")
        stab.append(dict(asset=n,lev=lev,h1=s1,h2=s2,ok=ok))
json.dump({"full":rows,"common":rows2,"stab":stab},open(f"{CACHE}/bestasset.json","w"),indent=1,default=str)
print("\nsaved bestasset.json")
