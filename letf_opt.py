"""
Holistic leveraged-ETF regime optimizer on 40yr of Nasdaq-100 (1985-2026).
Grid over:  signal (SMA/EMA/crossover) x signal-timeframe (D/W/M) x rebalance freq
(D/W/M) x allocation (on/off asset in TQQQ/QQQ/SQQQ/cash) x vol-target x scale-in.
Vectorized daily allocation backtest, no lookahead (weights shifted 1 day).
Ranked by CALMAR (CAGR/|maxDD|) — the metric that matters for leverage — and Sharpe,
with sub-period robustness (esp. the 2000-2009 dot-com/2008 out-of-sample stress).
"""
import json, itertools, numpy as np, pandas as pd
import bt
CACHE = bt.CACHE
COST_BPS = 3.0            # per unit turnover (round-trip-ish); ETFs are liquid

d = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
ndx = d["ndx"]; rf_d = d["rf"]/252.0
RET = {"tqqq": d["tqqq_ret"].fillna(0), "qqq": d["qqq_ret"].fillna(0),
       "sqqq": d["sqqq_ret"].fillna(0), "cash": rf_d.fillna(0)}
vol20 = d["ndx_ret"].rolling(20).std()*np.sqrt(252)

def ma(series, n, kind):
    return series.ewm(span=n, adjust=False).mean() if kind=="EMA" else series.rolling(n).mean()

def signal_riskon(tf, spec):
    """boolean daily risk-on state from a trend signal computed on timeframe tf."""
    if tf=="D": s=ndx
    elif tf=="W": s=ndx.resample("W-FRI").last()
    else: s=ndx.resample("ME").last()
    if spec[0]=="ma":                       # price vs MA(n)
        on = s > ma(s, spec[1], spec[2])
    else:                                   # crossover fast>slow
        on = ma(s, spec[1], spec[3]) > ma(s, spec[2], spec[3])
    return on.reindex(ndx.index, method="ffill").fillna(False)

def build_weights(on, alloc, rebal, voltgt, scalein):
    on_a, off_a = alloc
    w = pd.DataFrame(0.0, index=ndx.index, columns=["tqqq","qqq","sqqq","cash"])
    w[on_a] = on.astype(float); w[off_a] += (~on).astype(float)
    if voltgt and on_a=="tqqq":             # scale TQQQ by target/realized vol, rest to cash
        scale = (voltgt/vol20).clip(0,1).fillna(1.0)
        moved = w["tqqq"]*(1-scale); w["tqqq"]*=scale; w["cash"]+=moved
    if rebal!="D":                          # only rebalance on W/M boundaries
        rule="W-FRI" if rebal=="W" else "ME"
        mark=pd.Series(False,index=ndx.index); mark[ndx.resample(rule).last().index.intersection(ndx.index)]=True
        w=w.where(mark).ffill().fillna(0)
    if scalein>1:                           # ramp weight changes over N days
        w=w.rolling(scalein,min_periods=1).mean()
    return w.shift(1).fillna(0)             # decide from prior close

def backtest(w):
    gross = sum(w[a]*RET[a] for a in RET)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    ret = gross - turn*(COST_BPS/1e4)
    eq = (1+ret).cumprod()
    return ret, eq

def metrics(ret, eq, sub=None):
    r = ret if sub is None else ret.loc[sub]; e=(1+r).cumprod()
    yrs=(r.index[-1]-r.index[0]).days/365.25
    cagr=e.iloc[-1]**(1/yrs)-1
    dd=(e/e.cummax()-1).min()
    sharpe=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan
    calmar=cagr/abs(dd) if dd<0 else np.nan
    return dict(cagr=cagr,maxdd=float(dd),sharpe=float(sharpe),calmar=float(calmar),total=e.iloc[-1]-1)

# ---------------- grid ----------------
SIGNALS = ([("ma",n,k) for n in [100,150,200,250] for k in ["SMA","EMA"]] +
           [("x",f,s,"SMA") for f,s in [(20,100),(50,200),(50,150),(100,200)]])
SIGNAMES = {("ma",n,k):f"{k}{n}" for n in [100,150,200,250] for k in ["SMA","EMA"]}
SIGNAMES.update({("x",f,s,"SMA"):f"cross{f}/{s}" for f,s in [(20,100),(50,200),(50,150),(100,200)]})
TFS=["D","W"]; REBAL=["D","W","M"]
ALLOCS=[("tqqq","cash"),("tqqq","qqq"),("tqqq","sqqq"),("qqq","cash")]
VOLT=[0,0.25]; SCALEIN=[1,5]

rows=[]
for sig in SIGNALS:
    for tf in TFS:
        on=signal_riskon(tf,sig)
        for rebal,alloc,vt,si in itertools.product(REBAL,ALLOCS,VOLT,SCALEIN):
            if vt and alloc[0]!="tqqq": continue
            w=build_weights(on,alloc,rebal,vt,si)
            ret,eq=backtest(w); m=metrics(ret,eq)
            rows.append(dict(signal=SIGNAMES[sig],tf=tf,rebal=rebal,
                on=alloc[0],off=alloc[1],voltgt=vt,scalein=si,
                pct_tqqq=float(w["tqqq"].mean()),**m))
res=pd.DataFrame(rows)
res.to_csv(f"{CACHE}/letf_opt.csv",index=False)

# benchmarks
full=ndx.index
bh={}
for a in ["qqq","tqqq"]:
    r=RET[a]; e=(1+r).cumprod(); bh[a]=metrics(r,e)
print("=== BENCHMARKS (1985-2026, full incl. simulated pre-2010) ===")
for a in ["qqq","tqqq"]:
    print(f"  BH {a.upper():4s}: total {bh[a]['total']*100:>12,.0f}%  CAGR {bh[a]['cagr']*100:5.1f}%  maxDD {bh[a]['maxdd']*100:4.0f}%  Sharpe {bh[a]['sharpe']:.2f}  Calmar {bh[a]['calmar']:.2f}")

print("\n=== TOP 15 by CALMAR (CAGR / |maxDD|) — full history ===")
cols=["signal","tf","rebal","on","off","voltgt","scalein","cagr","maxdd","sharpe","calmar","total","pct_tqqq"]
top=res.sort_values("calmar",ascending=False).head(15)
print(top[cols].to_string(index=False,float_format=lambda x:f"{x:.2f}"))

print("\n=== TOP 10 by SHARPE ===")
print(res.sort_values("sharpe",ascending=False).head(10)[cols].to_string(index=False,float_format=lambda x:f"{x:.2f}"))

# ---- sub-period robustness for the Calmar champion + a simple baseline ----
periods={"1985-99":slice("1985","1999"),"2000-09(dotcom+GFC)":slice("2000","2009"),
         "2010-19":slice("2010","2019"),"2020-26":slice("2020","2026")}
def rebuild(row):
    sig=[s for s in SIGNALS if SIGNAMES[s]==row["signal"]][0]
    on=signal_riskon(row["tf"],sig)
    w=build_weights(on,(row["on"],row["off"]),row["rebal"],row["voltgt"],row["scalein"])
    return backtest(w)
print("\n=== SUB-PERIOD ROBUSTNESS (champion vs simple 200SMA/cash daily) ===")
champ=top.iloc[0]
base=res[(res.signal=="SMA200")&(res.tf=="D")&(res.rebal=="D")&(res.on=="tqqq")&(res.off=="cash")&(res.voltgt==0)&(res.scalein==1)].iloc[0]
for label,row in [("CHAMPION: "+f"{champ['signal']} {champ['tf']}/{champ['rebal']} {champ['on']}/{champ['off']} vt{champ['voltgt']} si{champ['scalein']}",champ),
                  ("BASELINE: SMA200 D/D tqqq/cash",base)]:
    ret,eq=rebuild(row)
    print(f"\n {label}")
    for pn,sl in periods.items():
        try:
            m=metrics(ret,eq,sub=(ndx.index>=pd.Timestamp(sl.start))&(ndx.index<=pd.Timestamp(sl.stop+'-12-31')))
            print(f"   {pn:20s} CAGR {m['cagr']*100:6.1f}%  maxDD {m['maxdd']*100:5.0f}%  Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:.2f}")
        except Exception as e: print(f"   {pn}: {repr(e)[:50]}")

# best UNLEVERAGED (qqq/cash) by sharpe — the risk-adjusted winner
unlev=res[(res.on=="qqq")].sort_values("sharpe",ascending=False).iloc[0]
def subperiods(row):
    ret,eq=rebuild(row); o={}
    for pn,sl in periods.items():
        msk=(ndx.index>=pd.Timestamp(sl.start))&(ndx.index<=pd.Timestamp(sl.stop+'-12-31'))
        m=metrics(ret,eq,sub=msk); o[pn]={k:round(v,3) for k,v in m.items()}
    return o

# save equity curves + benchmarks for report
def eqpts(eq):
    e=eq.resample("ME").last().dropna(); return [{"t":t.strftime("%Y-%m"),"e":round(float(v),4)} for t,v in e.items()]
champ_ret,champ_eq=rebuild(champ); base_ret,base_eq=rebuild(base); unlev_ret,unlev_eq=rebuild(unlev)
def rowdict(r): return {k:(round(v,3) if isinstance(v,(int,float,np.floating)) else v) for k,v in r[cols].to_dict().items()}
out={"bh":bh,"top":top[cols].round(3).to_dict("records"),
     "champion":rowdict(champ),"baseline":rowdict(base),"unlev":rowdict(unlev),
     "sub_champ":subperiods(champ),"sub_base":subperiods(base),"sub_unlev":subperiods(unlev),
     "eq_bh_qqq":eqpts((1+RET["qqq"]).cumprod()),"eq_bh_tqqq":eqpts((1+RET["tqqq"]).cumprod()),
     "eq_champ":eqpts(champ_eq),"eq_base":eqpts(base_eq),"eq_unlev":eqpts(unlev_eq),
     "periods":list(periods.keys()),"period":"1985-2026"}
json.dump(out,open(f"{CACHE}/letf_opt.json","w"),indent=1,default=str)
print("\nsaved letf_opt.csv + letf_opt.json")
