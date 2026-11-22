"""
IS OUR EDGE THE RULES, OR JUST THE ASSET?
Run OUR ruleset and GAYED's simpler ruleset on the SAME asset, SAME rebalance frequency,
SAME off-leg, SAME costs. Do it on Nasdaq AND S&P. If ours only wins on Nasdaq, our
"win" in the head-to-head is asset selection, not skill.
"""
import numpy as np, pandas as pd, json, yfinance as yf
import engine as E
D=E.data(); idx=D.idx; COST=E.COST; rf_d=D.rf_d; CACHE=E.CACHE
_t=yf.Ticker("^TNX").history(period="max")["Close"]/100
_t.index=_t.index.tz_localize(None).normalize(); tnx=_t[~_t.index.duplicated()].reindex(idx).ffill()
bond20=(tnx.shift(1)/252-17.0*tnx.diff()).fillna(0)
def lev(r,k,exp=0.0095): return (k*r-exp/252-(k-1)*rf_d).fillna(0)
mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)
mo_end=pd.Series(False,index=idx); mo_end[idx.to_series().resample("ME").last().dropna()]=True

MK={"Nasdaq-100":D.ndx,"S&P 500":(1+D.R["spx"].fillna(0)).cumprod()}
def bt(w,mark,R):
    ww=w.where(mark).ffill().fillna(0).shift(1).fillna(0)
    return sum(ww[c]*R[c].fillna(0) for c in ww.columns)-ww.diff().abs().sum(axis=1).fillna(0)*COST

def gayed(p,k,off,mark,R):
    sig=(p>p.rolling(200).mean()).fillna(False)
    w=pd.DataFrame(0.0,index=idx,columns=["lev",off]); w["lev"]=sig.astype(float); w[off]=1-w["lev"]
    return bt(w,mark,R)
def ours(p,k,off,mark,R,voltgt=0.20,gold=0.15):
    r1=p.pct_change(); vol=r1.rolling(20).std()*np.sqrt(252)
    er=(p.diff(20).abs()/p.diff().abs().rolling(20).sum().replace(0,np.nan))
    above=p>p.ewm(span=100,adjust=False).mean()
    scale=((voltgt/vol).clip(0,1).fillna(1.0)*AS).clip(0,1)
    ron=above & ~D.confirmed; clean=(er>=0.25).fillna(False)
    eq=(ron&clean)*scale+(ron&~clean)*scale*0.5
    core=1.0-gold
    cols=["lev",off]+(["gold"] if gold>0 else [])
    w=pd.DataFrame(0.0,index=idx,columns=cols)
    w["lev"]=eq*core; w[off]=(core-eq*core).clip(lower=0)
    if gold>0: w["gold"]=gold
    return bt(w,mark,R)

print(f"{'market':12s}{'ruleset':34s}{'lev':>4}{'CAGR':>8}{'maxDD':>7}{'Calmar':>7}{'Sharpe':>7}")
rows=[]
for mname,p in MK.items():
    for k in [1,3]:
        for offname,offser in [("cash",rf_d),("bond20",bond20)]:
            R={"lev":lev(p.pct_change(),k) if k>1 else (p.pct_change()-0.0020/252).fillna(0),
               "cash":rf_d,"bond20":bond20,"gold":D.R["gold"]}
            for rname,fn,mark,gold in [
                ("Gayed 200d SMA (monthly)",lambda:gayed(p,k,offname,mo_end,R),mo_end,0),
                ("OURS full (weekly)",lambda:ours(p,k,offname,D.mark,R,gold=0.15),D.mark,0.15),
                ("OURS engine only, no gold",lambda:ours(p,k,offname,D.mark,R,gold=0.0),D.mark,0)]:
                r=fn(); s=E.stats(r)
                lbl=f"{rname} [{offname} off-leg]"
                print(f"{mname if (k==1 and offname=='cash' and rname.startswith('Gayed')) else '':12s}"
                      f"{lbl:34s}{k:>3}x{s['cagr']*100:7.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:7.2f}")
                rows.append(dict(market=mname,rules=rname,off=offname,lev=k,**s))
    print()
df=pd.DataFrame(rows)
print("=== WHO WINS ON EACH MARKET (best Calmar per market/leverage) ===")
for mname in MK:
    for k in [1,3]:
        sub=df[(df.market==mname)&(df.lev==k)].sort_values("calmar",ascending=False)
        b=sub.iloc[0]; g=sub[sub.rules.str.startswith("Gayed")].iloc[0]
        o=sub[sub.rules.str.startswith("OURS full")].iloc[0]
        print(f"  {mname:12s} {k}x -> winner: {b['rules'][:26]:28s} Calmar {b['calmar']:.2f}   "
              f"(ours {o['calmar']:.2f} vs Gayed {g['calmar']:.2f})")
json.dump(rows,open(f"{CACHE}/fairtest.json","w"),indent=1,default=str)
print("\nsaved fairtest.json")
