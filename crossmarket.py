"""
CROSS-MARKET VALIDATION — the single most informative remaining test.
Apply the IDENTICAL timing rules (EMA-100 trend gate, vol-target 0.20, ER<0.25 -> half size,
Aug-Sep half, bond off-leg) to OTHER markets, each with its own 3x-simulated sleeve.

If the rules are a general property of trending markets, they should work (perhaps weaker)
on S&P, Russell, DAX, Nikkei, Hang Seng, Canada and gold. If they ONLY work on the Nasdaq,
the whole thing is fitted and should be distrusted.

Engine only (no gold sleeve) so we isolate the TIMING rules.
"""
import json, numpy as np, pandas as pd, yfinance as yf
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE; rf_d=D.rf_d
EXP=0.0095

def px(t):
    h=yf.Ticker(t).history(period="max")["Close"]
    h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).ffill()

MKT={"Nasdaq-100":"^NDX","S&P 500":"^GSPC","Russell 2000":"^RUT","DAX":"^GDAXI",
     "Nikkei":"^N225","Hang Seng":"^HSI","Canada TSX":"^GSPTSE","Gold miners":"^XAU"}
print("fetching indices ...", flush=True)
PX={k:px(t) for k,t in MKT.items()}
BOND=D.R["bond"]                       # synthetic 10y UST as the common off-leg
mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)

def run_market(p, lev=3, voltgt=0.20, chop_thr=0.25, chop_pos=0.5, seasonal=True):
    r1=p.pct_change()
    rlev=(lev*r1 - EXP/252 - (lev-1)*rf_d).fillna(0)
    ema=p.ewm(span=100,adjust=False).mean(); above=p>ema
    vol=r1.rolling(20).std()*np.sqrt(252)
    er=(p.diff(20).abs()/p.diff().abs().rolling(20).sum().replace(0,np.nan))
    scale=(voltgt/vol).clip(0,1).fillna(1.0)
    if seasonal: scale=(scale*AS).clip(0,1)
    clean=(er>=chop_thr).fillna(False)
    eq=(above&clean)*scale+(above&~clean)*scale*chop_pos
    w=pd.DataFrame({"lev":eq,"bond":(1-eq).clip(lower=0)},index=idx)
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    ret=ww["lev"]*rlev+ww["bond"]*BOND.fillna(0)-ww.diff().abs().sum(axis=1).fillna(0)*E.COST
    return ret, r1.fillna(0), rlev

rows=[]
print(f"\n{'market':14s}{'STRATEGY':^30s}   {'BUY&HOLD 1x':^20s}  {'BH 3x':^12s}")
print(f"{'':14s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'Shrp':>6}   {'CAGR':>7}{'maxDD':>7}{'Calmar':>6}  {'CAGR':>6}{'maxDD':>6}")
for k,p in PX.items():
    if p.dropna().shape[0]<3000: continue
    ret,r1,rlev=run_market(p)
    s=E.stats(ret); b=E.stats(r1); l=E.stats(rlev)
    print(f"{k:14s}{s['cagr']*100:6.1f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}{s['sharpe']:6.2f}   "
          f"{b['cagr']*100:6.1f}%{b['maxdd']*100:6.0f}%{b['calmar']:6.2f}  {l['cagr']*100:5.1f}%{l['maxdd']*100:6.0f}%")
    rows.append(dict(market=k,strat=s,bh1=b,bh3=l,
                     beats_bh1=bool(s["calmar"]>b["calmar"]),beats_bh3=bool(s["calmar"]>l["calmar"]),
                     dec={kk:round(v["cagr"],4) for kk,v in E.by_decade(ret).items()}))

n=len(rows); w1=sum(r["beats_bh1"] for r in rows); w3=sum(r["beats_bh3"] for r in rows)
print(f"\n=== VERDICT: strategy Calmar beat 1x buy&hold in {w1}/{n} markets, 3x buy&hold in {w3}/{n} ===")
print(f"median strategy Calmar {np.median([r['strat']['calmar'] for r in rows]):.2f} "
      f"vs median 1x buy&hold {np.median([r['bh1']['calmar'] for r in rows]):.2f} "
      f"vs median 3x {np.median([r['bh3']['calmar'] for r in rows]):.2f}")

print("\n=== PER-MARKET DECADES (strategy CAGR) ===")
print(f"{'market':14s}"+"".join(f"{k:>9s}" for k in E.DECADES))
for r in rows:
    print(f"{r['market']:14s}"+"".join(f"{r['dec'][k]*100:8.1f}%" for k in E.DECADES))

# parameter sensitivity across markets: does voltgt/chop generalize?
print("\n=== SAME RULES, PARAMETERS PERTURBED (median Calmar across all markets) ===")
for lbl,kw in [("as specified",{}),("voltgt 0.15",dict(voltgt=0.15)),("voltgt 0.30",dict(voltgt=0.30)),
               ("chop_thr 0.20",dict(chop_thr=0.20)),("chop_thr 0.30",dict(chop_thr=0.30)),
               ("chop_pos 1.0 (no chop rule)",dict(chop_pos=1.0)),("no seasonal",dict(seasonal=False)),
               ("2x leverage",dict(lev=2)),("1x (no leverage)",dict(lev=1))]:
    cals=[E.stats(run_market(p,**kw)[0])["calmar"] for k,p in PX.items() if p.dropna().shape[0]>=3000]
    print(f"  {lbl:28s} median Calmar {np.median(cals):.2f}   range {min(cals):.2f}-{max(cals):.2f}")

json.dump(rows,open(f"{CACHE}/crossmarket.json","w"),indent=1,default=str)
print("\nsaved crossmarket.json")
