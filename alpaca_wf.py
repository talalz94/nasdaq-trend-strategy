"""
Put the EQUITY champion through gold's gauntlet on ~10 years of Alpaca data:
matched-drift edge, bootstrap significance, and true walk-forward (train 2016-21 /
test 2022-26) + year-by-year. Strategy frozen: Confluence_oversold + 2% stop + sector whitelist.
"""
import sys, json, numpy as np, pandas as pd
import bt
from indicators import compute_all

CACHE = bt.CACHE
COST = 2.0
GOOD_SEC = {"Health Care","Utilities","Information Technology","Financials",
            "Consumer Staples","Real Estate","Industrials"}
sectors = pd.read_parquet("sectors.parquet").set_index("ticker")["sector"].to_dict()

def build_cache(path):
    raw = pd.read_parquet(path)
    cache = {}
    for tkr, g in raw.groupby("ticker", sort=False):
        if len(g) < 300: continue
        ind = compute_all(g[["datetime","date","open","high","low","close","volume"]])
        d = {c: ind[c].to_numpy() for c in ["open","high","low","close","wt1","wt2",
             "wt_cross_up","wt_cross_dn","mom","mom_color","mom_rising","sqz_on","sqz_off",
             "sqz_release","datetime","ma8","ma8_up","mashift_regime_up","mashift_up"] if c in ind}
        cache[tkr] = d
    return cache

def st(r):
    r=np.asarray(r)
    if len(r)==0: return dict(n=0,avg=np.nan,pf=np.nan,win=np.nan)
    gp=r[r>0].sum(); gl=-r[r<=0].sum()
    return dict(n=len(r),avg=r.mean()*1e4,win=(r>0).mean(),pf=(gp/gl if gl>0 else np.inf))

def boot_p(r,n=8000,seed=5):
    if len(r)<50: return (np.nan, (np.nan, np.nan))
    rng=np.random.default_rng(seed)
    b=rng.choice(r,size=(n,len(r)),replace=True).mean(axis=1)
    return float((b<=0).mean()), np.percentile(b,[2.5,97.5])*1e4

def bench(cache, bars, cost=COST):
    c=2*cost/1e4; h=max(int(round(bars)),1); rr=[]
    for d in cache.values():
        C=d["close"]
        if len(C)>h: f=C[h:]/C[:-h]-1-c; rr.append(f[np.isfinite(f)])
    r=np.concatenate(rr); return r.mean()

def enrich_filter(cache, tr):
    keep=[]
    for _,t in tr.iterrows():
        keep.append(sectors.get(t["ticker"],"?") in GOOD_SEC)
    return tr[np.array(keep)]

def main(path):
    print(f"loading {path} ...", flush=True)
    cache = build_cache(path)
    nb = sum(len(d["close"]) for d in cache.values())
    yrs = (max(pd.Timestamp(d["datetime"][-1]) for d in cache.values()) -
           min(pd.Timestamp(d["datetime"][0]) for d in cache.values())).days/365.25
    print(f"cache: {len(cache)} tickers, {nb:,} bars, ~{yrs:.1f}y span", flush=True)

    tr = bt.run_strategy(cache, "Confluence_oversold", stop=0.02, cost_bps=COST)
    trf = enrich_filter(cache, tr)
    for label, T in [("ALL sectors", tr), ("sector-filtered (champion)", trf)]:
        m = st(T["ret"].to_numpy()); e=(m["avg"]/1e4 - bench(cache, np.nanmean(T["bars"])))*1e4
        p0,ci = boot_p(T["ret"].to_numpy())
        print(f"\n{label}: n={m['n']} win={m['win']:.2f} avg={m['avg']:.1f}bps edge={e:.1f} "
              f"pf={m['pf']:.2f} p(<=0)={p0:.4f} CI=[{ci[0]:.0f},{ci[1]:.0f}]")

    # walk-forward on champion
    T = trf.copy(); T["year"]=pd.to_datetime(T["entry_time"]).dt.year
    print("\n=== WALK-FORWARD (champion, sector-filtered) ===")
    for lbl, sub in [("TRAIN 2016-2021", T[T.year<=2021]), ("TEST 2022-2026", T[T.year>=2022])]:
        s=st(sub["ret"].to_numpy()); print(f"  {lbl}: n={s['n']} avg={s['avg']:.1f}bps pf={s['pf']:.2f} win={s['win']:.2f}")
    print("  by year:")
    wf=T.groupby("year").agg(n=("ret","size"),win=("ret",lambda x:(x>0).mean()),
        avg=("ret",lambda x:x.mean()*1e4),pf=("ret",lambda x:x[x>0].sum()/(-x[x<=0].sum()) if (x<=0).any() else np.inf))
    print(wf.to_string(float_format=lambda x:f"{x:.2f}"))
    out={"years":yrs,"tickers":len(cache),
         "all":st(tr["ret"].to_numpy()),"champ":st(trf["ret"].to_numpy()),
         "wf_year":wf.reset_index().to_dict("records"),
         "train":st(T[T.year<=2021]["ret"].to_numpy()),"test":st(T[T.year>=2022]["ret"].to_numpy())}
    json.dump(out, open(f"{CACHE}/alpaca_wf.json","w"), indent=1, default=str)
    print("\nsaved alpaca_wf.json")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "data_alpaca/stocks_1h.parquet")
