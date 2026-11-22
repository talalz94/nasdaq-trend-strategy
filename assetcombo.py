"""
(A) HINDSIGHT CHECK — the single-stock "winners" (NVDA/AAPL) were chosen BECAUSE we know
    they won. Honest test: pick by a RULE knowable at the time (largest Nasdaq weight /
    prior 12m momentum leader), not by hindsight.
(B) DOES COMBINING ASSETS HELP? Nasdaq + Bitcoin sleeves are different return drivers.
"""
import json, numpy as np, pandas as pd, yfinance as yf
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE; rf_d=D.rf_d; COST=E.COST; EXP=0.0095
BOND=D.R["bond"]; mo=pd.Series(idx.month,index=idx)
AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)
def px(t):
    h=yf.Ticker(t).history(period="max")["Close"]; h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).ffill()

def sleeve(p,lev):
    r1=p.pct_change()
    rl=(lev*r1-EXP/252-(lev-1)*rf_d).fillna(0) if lev>1 else (r1-0.0020/252).fillna(0)
    above=p>p.ewm(span=100,adjust=False).mean()
    vol=r1.rolling(20).std()*np.sqrt(252)
    er=(p.diff(20).abs()/p.diff().abs().rolling(20).sum().replace(0,np.nan))
    sc=((0.20/vol).clip(0,1).fillna(1.0)*AS).clip(0,1)
    clean=(er>=0.25).fillna(False)
    eq=(above&clean)*sc+(above&~clean)*sc*0.5
    return eq.fillna(0), rl
def bt(parts):
    """parts = list of (weight_share, eq_series, lev_return); remainder -> bonds"""
    tot=pd.Series(0.0,index=idx); wsum=pd.Series(0.0,index=idx)
    cols={}
    for i,(share,eq,rl) in enumerate(parts):
        cols[f"a{i}"]=eq*share; wsum=wsum+eq*share
    w=pd.DataFrame(cols,index=idx); w["bond"]=(1-wsum).clip(lower=0)
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    r=ww["bond"]*BOND.fillna(0)
    for i,(share,eq,rl) in enumerate(parts): r=r+ww[f"a{i}"]*rl
    return r-ww.diff().abs().sum(axis=1).fillna(0)*COST
def st(r,a=None,b=None):
    if a: r=r[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    r=r.dropna()
    if len(r)<400: return None
    e=(1+r).cumprod(); y=(r.index[-1]-r.index[0]).days/365.25
    c=e.iloc[-1]**(1/y)-1; dd=(e/e.cummax()-1).min()
    return dict(cagr=float(c),maxdd=float(dd),calmar=float(c/abs(dd)) if dd<0 else np.nan,
                sharpe=float(r.mean()/r.std()*np.sqrt(252)))

print("="*76); print(" (A) HINDSIGHT CHECK — rule-picked stock vs hindsight-picked stock"); print("="*76)
names=["AAPL","MSFT","NVDA","AMZN","GOOGL","INTC","CSCO","ORCL","QCOM","TXN","AMD","ADBE"]
P={n:px(n) for n in names}
PXd=pd.DataFrame({n:P[n] for n in names})
mom12=PXd/PXd.shift(252)-1                       # knowable at the time
# monthly: hold the prior-12m momentum leader among the mega-caps
m_end=pd.Series(False,index=idx); m_end[idx.to_series().resample("ME").last().dropna()]=True
leader=mom12.idxmax(axis=1).ffill()
for lev in [2,3]:
    eqs={}; rls={}
    for n in names:
        e,rl=sleeve(P[n],lev); eqs[n]=e; rls[n]=rl
    sel=leader.where(m_end).ffill()
    eq=pd.Series(0.0,index=idx); rl=pd.Series(0.0,index=idx)
    for n in names:
        m=(sel==n)
        eq=eq+eqs[n].where(m,0.0); rl=rl+rls[n].where(m,0.0)
    s=st(bt([(1.0,eq,rl)]))
    sN=st(bt([(1.0,*sleeve(P["NVDA"],lev))]))
    sA=st(bt([(1.0,*sleeve(P["AAPL"],lev))]))
    print(f" {lev}x  RULE-PICKED momentum leader : CAGR {s['cagr']*100:5.1f}%  DD {s['maxdd']*100:4.0f}%  Calmar {s['calmar']:.2f}")
    print(f" {lev}x  HINDSIGHT: always NVDA      : CAGR {sN['cagr']*100:5.1f}%  DD {sN['maxdd']*100:4.0f}%  Calmar {sN['calmar']:.2f}")
    print(f" {lev}x  HINDSIGHT: always AAPL      : CAGR {sA['cagr']*100:5.1f}%  DD {sA['maxdd']*100:4.0f}%  Calmar {sA['calmar']:.2f}")
    ndx_s=st(bt([(1.0,*sleeve(D.ndx,lev))]))
    print(f" {lev}x  Nasdaq-100 index           : CAGR {ndx_s['cagr']*100:5.1f}%  DD {ndx_s['maxdd']*100:4.0f}%  Calmar {ndx_s['calmar']:.2f}\n")

print("="*76); print(" (B) DOES ADDING A BITCOIN SLEEVE HELP? (BTC era only: 2015-2026)"); print("="*76)
btc=px("BTC-USD")
A="2015-01-01"; B="2026-07-24"
print(f"{'allocation':40s}{'CAGR':>8}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}")
for lbl,parts in [
  ("100% Nasdaq sleeve (3x)",      [(1.0,*sleeve(D.ndx,3))]),
  ("100% Bitcoin sleeve (3x)",     [(1.0,*sleeve(btc,3))]),
  ("100% Bitcoin sleeve (2x)",     [(1.0,*sleeve(btc,2))]),
  ("90% Nasdaq 3x + 10% BTC 2x",   [(0.9,*sleeve(D.ndx,3)),(0.1,*sleeve(btc,2))]),
  ("80% Nasdaq 3x + 20% BTC 2x",   [(0.8,*sleeve(D.ndx,3)),(0.2,*sleeve(btc,2))]),
  ("70% Nasdaq 3x + 30% BTC 2x",   [(0.7,*sleeve(D.ndx,3)),(0.3,*sleeve(btc,2))]),
  ("50/50 Nasdaq 3x + BTC 2x",     [(0.5,*sleeve(D.ndx,3)),(0.5,*sleeve(btc,2))]),
  ("80% Nasdaq 2x + 20% BTC 2x",   [(0.8,*sleeve(D.ndx,2)),(0.2,*sleeve(btc,2))]),
]:
    s=st(bt(parts),A,B)
    print(f"{lbl:40s}{s['cagr']*100:7.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:7.2f}")
print("\n  BTC sub-period: 2015-2020 vs 2021-2026 (is the crypto edge decaying?)")
for lbl,parts in [("Bitcoin sleeve 2x",[(1.0,*sleeve(btc,2))]),("Nasdaq sleeve 3x",[(1.0,*sleeve(D.ndx,3))])]:
    a=st(bt(parts),"2015-01-01","2020-12-31"); b=st(bt(parts),"2021-01-01","2026-07-24")
    print(f"   {lbl:22s} 2015-20 Calmar {a['calmar']:.2f} ({a['cagr']*100:5.1f}%)  ->  2021-26 Calmar {b['calmar']:.2f} ({b['cagr']*100:5.1f}%)")
print("\nsaved")
