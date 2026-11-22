"""
"WHAT IF THE INDEX CRASHES AND NEVER RECOVERS?" + PRE-1985 VALIDATION.

Two natural experiments the modern data cannot provide:
  NIKKEI 1990-2026 : crashed from 38,900 and spent 23 years going DOWN (-5.6%/yr).
                     This is literally the scenario "what if Nasdaq tops and dies".
  S&P 500 1927-2026: 99 years incl. the Great Depression (1929-54: +1.4%/yr),
                     WWII, and the 1966-82 stagflation decade (+2.5%/yr).
Risk-free rate: ^IRX from 1960; before that an approximate piecewise historical
3-month T-bill series (flagged - financing is small vs the effect being measured).
Off-leg = cash/T-bills (bond data doesn't reach 1927).
"""
import numpy as np, pandas as pd, yfinance as yf, json
import bt
CACHE=bt.CACHE; EXP=0.0095; COST=3.0/1e4

def px(t):
    h=yf.Ticker(t).history(period="max")["Close"]
    h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].dropna()

irx=px("^IRX")/100
def rf_for(idx):
    """3-mo T-bill; ^IRX from 1960, approximate history before."""
    hist={1927:0.035,1930:0.015,1933:0.003,1940:0.003,1945:0.004,1950:0.015,1955:0.019}
    base=pd.Series(index=idx,dtype=float)
    for y,v in hist.items(): base[idx.year>=y]=v
    r=irx.reindex(idx).ffill()
    return r.fillna(base)/252

def strat(p, lev=3, voltgt=0.20, chop_thr=0.25, chop_pos=0.5, seasonal=True, off_ret=None):
    idx=p.index; r1=p.pct_change(); rf=rf_for(idx)
    rlev=(lev*r1 - EXP/252 - (lev-1)*rf).fillna(0) if lev>1 else (r1-0.0020/252).fillna(0)
    ema=p.ewm(span=100,adjust=False).mean(); above=p>ema
    vol=r1.rolling(20).std()*np.sqrt(252)
    er=(p.diff(20).abs()/p.diff().abs().rolling(20).sum().replace(0,np.nan))
    scale=(voltgt/vol).clip(0,1).fillna(1.0)
    if seasonal:
        mo=pd.Series(idx.month,index=idx); scale=(scale*np.where(mo.isin([8,9]),0.5,1.0)).clip(0,1)
    clean=(er>=chop_thr).fillna(False)
    eq=(above&clean)*scale+(above&~clean)*scale*chop_pos
    off=off_ret if off_ret is not None else rf
    mark=pd.Series(False,index=idx); mark[p.resample("W-FRI").last().index.intersection(idx)]=True
    w=pd.DataFrame({"lev":eq,"off":(1-eq).clip(lower=0)},index=idx).where(mark).ffill().fillna(0).shift(1).fillna(0)
    return w["lev"]*rlev + w["off"]*off - w.diff().abs().sum(axis=1).fillna(0)*COST, r1.fillna(0), rlev

def st(r,a=None,b=None):
    if a: r=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    r=r.dropna()
    if len(r)<200: return dict(cagr=np.nan,maxdd=np.nan,calmar=np.nan,sharpe=np.nan,total=np.nan)
    e=(1+r).cumprod(); yrs=(r.index[-1]-r.index[0]).days/365.25
    c=e.iloc[-1]**(1/yrs)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(c),maxdd=float(dd),calmar=float(c/abs(dd)) if dd<0 else np.nan,
                sharpe=float(r.mean()/r.std()*np.sqrt(252)),total=float(e.iloc[-1]-1))

out={}
print("="*84)
print(" TEST 1 — NIKKEI: the index that crashed in 1990 and fell for 23 years")
print("="*84)
nk=px("^N225")
ERAS={"1965-1989 (the bubble)":("1965-01-01","1989-12-31"),
      "1990-2012 (THE CRASH+LOST DECADES, -5.6%/yr)":("1990-01-01","2012-12-31"),
      "1990-2026 (full aftermath)":("1990-01-01","2026-07-24"),
      "2013-2026 (recovery)":("2013-01-01","2026-07-24")}
print(f"{'era':46s}{'lev':>4}{'STRAT cagr':>11}{'maxDD':>7}{'BH cagr':>9}{'BH3x cagr':>10}")
nkrows=[]
for lbl,(a,b) in ERAS.items():
    for lev in [1,2,3]:
        s,r1,r3=strat(nk,lev=lev)
        S=st(s,a,b); B=st(r1,a,b); L=st(r3,a,b)
        print(f"{lbl if lev==1 else '':46s}{lev:>3}x{S['cagr']*100:10.1f}%{S['maxdd']*100:6.0f}%"
              f"{B['cagr']*100:8.1f}%{L['cagr']*100:9.1f}%")
        nkrows.append(dict(era=lbl,lev=lev,strat=S,bh=B,bh_lev=L))
    print()
out["nikkei"]=nkrows

print("="*84)
print(" TEST 2 — S&P 500 back to 1927: 99 years, Depression, WWII, stagflation")
print("="*84)
sp=px("^GSPC")
SERAS={"1927-1954 (crash+Depression+WWII)":("1928-01-01","1954-12-31"),
       "1955-1965 (post-war boom)":("1955-01-01","1965-12-31"),
       "1966-1982 (STAGFLATION, +2.5%/yr)":("1966-01-01","1982-12-31"),
       "1983-1999 (great bull)":("1983-01-01","1999-12-31"),
       "2000-2026 (modern)":("2000-01-01","2026-07-24"),
       "FULL 1928-2026 (99 years)":("1928-01-01","2026-07-24")}
print(f"{'era':38s}{'lev':>4}{'STRAT cagr':>11}{'maxDD':>7}{'Calmar':>7}{'BH cagr':>9}{'BH maxDD':>9}")
sprows=[]
for lbl,(a,b) in SERAS.items():
    for lev in [1,2,3]:
        s,r1,r3=strat(sp,lev=lev)
        S=st(s,a,b); B=st(r1,a,b)
        print(f"{lbl if lev==1 else '':38s}{lev:>3}x{S['cagr']*100:10.1f}%{S['maxdd']*100:6.0f}%"
              f"{S['calmar']:7.2f}{B['cagr']*100:8.1f}%{B['maxdd']*100:8.0f}%")
        sprows.append(dict(era=lbl,lev=lev,strat=S,bh=B))
    print()
out["spx"]=sprows

print("="*84)
print(" TEST 3 — Nasdaq Composite 1971-2026 (includes the 1970s, unlike NDX)")
print("="*84)
ix=px("^IXIC")
print(f"{'era':30s}{'lev':>4}{'STRAT cagr':>11}{'maxDD':>7}{'Calmar':>7}{'BH cagr':>9}{'BH maxDD':>9}")
ixrows=[]
for lbl,(a,b) in {"1971-1989":("1971-01-01","1989-12-31"),"1990-2026":("1990-01-01","2026-07-24"),
                  "FULL 1971-2026":("1971-01-01","2026-07-24")}.items():
    for lev in [1,2,3]:
        s,r1,_=strat(ix,lev=lev); S=st(s,a,b); B=st(r1,a,b)
        print(f"{lbl if lev==1 else '':30s}{lev:>3}x{S['cagr']*100:10.1f}%{S['maxdd']*100:6.0f}%{S['calmar']:7.2f}"
              f"{B['cagr']*100:8.1f}%{B['maxdd']*100:8.0f}%")
        ixrows.append(dict(era=lbl,lev=lev,strat=S,bh=B))
    print()
out["ixic"]=ixrows
json.dump(out,open(f"{CACHE}/longrun.json","w"),indent=1,default=str)
print("saved longrun.json")
