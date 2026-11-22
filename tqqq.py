"""
Can a timing strategy on TQQQ/SQQQ beat buy & hold?
Leverage makes timing valuable: 3x ETFs suffer volatility decay + brutal drawdowns,
so sidestepping bad regimes can beat leveraged buy&hold. Daily, no lookahead
(position on day t decided from data through t-1). Compared to BH-QQQ and BH-TQQQ.
"""
import time, json, numpy as np, pandas as pd, requests
from indicators import compute_all
import bt
CACHE = bt.CACHE
KV = dict(l.strip().split("=",1) for l in open(".env") if "=" in l)
H = {"APCA-API-KEY-ID": KV["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": KV["ALPACA_SECRET"]}

def fetch(sym, start="2016-01-01T00:00:00Z"):
    out=[]; tok=None
    while True:
        p=dict(symbols=sym,timeframe="1Day",start=start,limit=10000,feed="sip",adjustment="split",sort="asc")
        if tok: p["page_token"]=tok
        j=requests.get("https://data.alpaca.markets/v2/stocks/bars",headers=H,params=p,timeout=30).json()
        out.extend(j.get("bars",{}).get(sym,[])); tok=j.get("next_page_token")
        if not tok: break
    df=pd.DataFrame(out); df["dt"]=pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize()
    return df.set_index("dt")[["o","h","l","c","v"]].rename(columns=dict(o="open",h="high",l="low",c="close",v="volume"))

print("fetching QQQ/TQQQ/SQQQ ...", flush=True)
qqq=fetch("QQQ"); tqqq=fetch("TQQQ"); sqqq=fetch("SQQQ")
idx=qqq.index.intersection(tqqq.index).intersection(sqqq.index)
qqq,tqqq,sqqq=qqq.loc[idx],tqqq.loc[idx],sqqq.loc[idx]
print(f"aligned {len(idx)} days: {idx.min().date()} -> {idx.max().date()}", flush=True)

r_qqq=qqq.close.pct_change().fillna(0)
r_tqqq=tqqq.close.pct_change().fillna(0)
r_sqqq=sqqq.close.pct_change().fillna(0)

# indicators on QQQ (for signal-based timers)
ind=compute_all(qqq.reset_index().rename(columns={"dt":"datetime"}).assign(date=idx.astype(str)))
ind.index=idx
sma200=qqq.close.rolling(200).mean()
sma50=qqq.close.rolling(50).mean()

# risk-on states (True = hold TQQQ). shifted by 1 day = decide from prior close.
states={
 "200SMA":            (qqq.close>sma200),
 "50>200":            (sma50>sma200),
 "WT(wt1>wt2)":       (ind.wt1>ind.wt2),
 "MAShift regime":    (ind.mashift_regime_up.astype(bool)),
 "SQZ mom>0":         (ind.mom>0),
 "200SMA & mom>0":    ((qqq.close>sma200)&(ind.mom>0)),
}

def metrics(ret):
    ret=ret.dropna(); eq=(1+ret).cumprod()
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1
    dd=(eq/eq.cummax()-1).min()
    sharpe=ret.mean()/ret.std()*np.sqrt(252) if ret.std()>0 else np.nan
    return dict(total=eq.iloc[-1]-1, cagr=cagr, maxdd=float(dd), sharpe=float(sharpe), eq=eq)

def timer(state, on_ret, off_ret, cost_bps=2.0):
    pos=state.shift(1).fillna(False).astype(float)         # 1=risk-on
    turn=pos.diff().abs().fillna(0)                        # trades on switches
    ret=pos*on_ret+(1-pos)*off_ret - turn*(cost_bps/1e4)
    return ret, float(pos.mean())

def run(off_name, off_ret):
    print(f"\n===== timers: risk-ON=TQQQ, risk-OFF={off_name} (net 2bps/switch) =====")
    print(f"{'strategy':20s}{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}{'%in':>6}")
    rows=[]
    for nm,s in states.items():
        ret,inm=timer(s,r_tqqq,off_ret); m=metrics(ret)
        print(f"{nm:20s}{m['total']*100:8.0f}%{m['cagr']*100:7.1f}%{m['maxdd']*100:7.0f}%{m['sharpe']:8.2f}{inm*100:5.0f}%")
        rows.append(dict(strategy=nm, off=off_name, **{k:v for k,v in m.items() if k!='eq'}, pct_in=inm, eq=m['eq']))
    return rows

# benchmarks
mq=metrics(r_qqq); mt=metrics(r_tqqq)
print(f"\nBUY&HOLD  QQQ : total {mq['total']*100:.0f}%  CAGR {mq['cagr']*100:.1f}%  maxDD {mq['maxdd']*100:.0f}%  Sharpe {mq['sharpe']:.2f}")
print(f"BUY&HOLD TQQQ : total {mt['total']*100:.0f}%  CAGR {mt['cagr']*100:.1f}%  maxDD {mt['maxdd']*100:.0f}%  Sharpe {mt['sharpe']:.2f}")

rows_cash=run("cash", pd.Series(0.0,index=idx))
rows_short=run("SQQQ", r_sqqq)

# best timer, train/test split (2022 = the TQQQ -80% stress test)
best=max(rows_cash+rows_short, key=lambda r:r['sharpe'])
print(f"\nBEST by Sharpe: {best['strategy']} (off={best['off']}) Sharpe {best['sharpe']:.2f}")
# walk-forward on the two headline timers
for nm in ["200SMA","200SMA & mom>0"]:
    s=states[nm]
    for off_name,off in [("cash",pd.Series(0.0,index=idx)),("SQQQ",r_sqqq)]:
        ret,_=timer(s,r_tqqq,off)
        for lbl,msk in [("2016-2021",idx.year<=2021),("2022-2026",idx.year>=2022)]:
            m=metrics(ret[msk])
            print(f"  {nm:16s} off={off_name:4s} {lbl}: total {m['total']*100:6.0f}% CAGR {m['cagr']*100:5.1f}% maxDD {m['maxdd']*100:5.0f}% Sharpe {m['sharpe']:.2f}")

# save equity curves (monthly) for report
def eqpts(eq):
    e=eq.resample("ME").last().dropna(); return [{"t":t.strftime("%Y-%m"),"e":round(float(v),3)} for t,v in e.items()]
out={"period":f"{idx.min().date()} to {idx.max().date()}","ndays":len(idx),
     "bh_qqq":{k:v for k,v in mq.items() if k!='eq'}, "bh_tqqq":{k:v for k,v in mt.items() if k!='eq'},
     "eq_qqq":eqpts(mq['eq']),"eq_tqqq":eqpts(mt['eq']),
     "timers":[{k:(v if k!='eq' else None) for k,v in r.items()} for r in rows_cash+rows_short]}
# add equity for the two headline + best
for nm in ["200SMA","200SMA & mom>0"]:
    ret,_=timer(states[nm],r_tqqq,pd.Series(0.0,index=idx)); out[f"eq_{nm.replace(' ','').replace('&','').replace('>','')}_cash"]=eqpts(metrics(ret)['eq'])
    ret,_=timer(states[nm],r_tqqq,r_sqqq); out[f"eq_{nm.replace(' ','').replace('&','').replace('>','')}_sqqq"]=eqpts(metrics(ret)['eq'])
json.dump(out, open(f"{CACHE}/tqqq_results.json","w"), indent=1, default=str)
print("\nsaved tqqq_results.json")
