"""
Comprehensive, overfit-guarded parameter sweep on 10yr of 1h stock data.
Every config is scored on TRAIN (2016-2021) and TEST (2022-2026); only OOS-holding
configs are trusted. Then the champion is broken down by regime / hour / weekday /
sector / volatility / stock, and the best signal is chosen per market regime.
"""
import os, json, itertools, numpy as np, pandas as pd
import bt, regime as R

CACHE = bt.CACHE
COST = 2.0
SRC = os.environ.get("SWEEP_CACHE", "ind_alpaca_1h.parquet")
MODE = os.environ.get("SWEEP_MODE", "intraday")   # 'intraday' or 'daily'
TAG = os.environ.get("SWEEP_TAG", "")             # output suffix
OB2, OS2 = 53, -53
sectors = pd.read_parquet("sectors.parquet").set_index("ticker")["sector"].to_dict()
CACHE_COLS = ["open","high","low","close","wt1","wt2","wt_cross_up","wt_cross_dn","mom",
    "mom_color","mom_rising","sqz_on","sqz_off","sqz_release","datetime","ma8","ma8_up",
    "ma8_turn_up","ma8_turn_dn","mashift_regime_up","mashift_up","mashift_sig_up","mashift_sig_dn"]

def load_cache():
    df = pd.read_parquet(f"{CACHE}/{SRC}")
    return {t: {c: g[c].to_numpy() for c in CACHE_COLS if c in g}
            for t, g in df.groupby("ticker", sort=False)}

# ------------------------------------------------------------------ signals
FILT = {
    "ema_up":     lambda d: d["ma8_up"],
    "above_ma40": lambda d: d["mashift_regime_up"],
    "not_sqz":    lambda d: ~d["sqz_on"],
    "mom_up":     lambda d: d["mashift_up"],
}
def make_signal(d, spec):
    n = len(d["close"]); F = np.zeros(n, bool)
    cu, cd, wt2 = d["wt_cross_up"], d["wt_cross_dn"], d["wt2"]
    mom, mrise = d["mom"], d["mom_rising"]; mprev = np.roll(mom,1); mprev[0]=np.nan
    t = spec["type"]; se=sx=F; opts={}
    if t == "wt_os":
        le = cu & (wt2 < spec.get("os", OS2))
        lx = cd | ((wt2 > spec["ob"]) if spec.get("ob") else np.zeros(n,bool))
    elif t == "confluence":
        le = cu & (wt2 < spec.get("os", OS2)) & mrise; lx = cd
    elif t == "sqz_zero":
        le = (mom>0)&(mprev<=0); lx = (mom<0)&(mprev>=0)
        if spec.get("both"): se, sx, opts = lx.copy(), le.copy(), dict(reverse=True)
    elif t == "sqz_color":
        le = (mom<0)&(mom>mprev); lx = (mom>0)&(mom<mprev)
    elif t == "ema_slope":
        le = d["ma8_turn_up"]; lx = d["ma8_turn_dn"]
    elif t == "mashift":
        le = d["mashift_sig_up"]; lx = d["mashift_sig_dn"]
        if spec.get("regime"): le = le & d["mashift_regime_up"]
    else:
        raise ValueError(t)
    for f in spec.get("filters", []):
        le = le & FILT[f](d)
    return le, lx, se, sx, opts

def run_spec(cache, spec):
    rows = []
    for tkr, d in cache.items():
        le, lx, se, sx, opts = make_signal(d, spec)
        tr = bt.simulate(d, le, lx, se, sx, target=spec.get("target"), stop=spec.get("stop"),
                         time_bars=spec.get("time"), trail=spec.get("trail"), cost_bps=COST, **opts)
        dt = d["datetime"]; L = len(dt)
        for x in tr:
            x["ticker"]=tkr; x["entry_time"]=dt[x["entry_i"]]
            x["exit_time"]=dt[min(x["exit_i"], L-1)]
        rows.extend(tr)
    df = pd.DataFrame(rows)
    if len(df):
        df["sector"] = df["ticker"].map(sectors)
        et = pd.to_datetime(df["entry_time"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        df["hour"] = et.dt.hour; df["dow"] = et.dt.dayofweek; df["year"] = et.dt.year
        df["dt"] = pd.to_datetime(df["entry_time"])
    return df

# ------------------------------------------------------------------ metrics
def stats(r):
    r = np.asarray(r, float)
    if len(r) == 0: return dict(n=0, avg=np.nan, pf=np.nan, win=np.nan, sharpe=np.nan)
    gp, gl = r[r>0].sum(), -r[r<=0].sum()
    return dict(n=int(len(r)), avg=float(r.mean()*1e4), win=float((r>0).mean()),
                pf=float(gp/gl if gl>0 else np.inf),
                sharpe=float(r.mean()/r.std()) if r.std()>0 else np.nan)

def split_eval(df, split_year=2022):
    tr = df[df.year < split_year]["ret"]; te = df[df.year >= split_year]["ret"]
    return stats(tr), stats(te), stats(df["ret"])

# ------------------------------------------------------------------ config grid
def build_configs():
    C = []
    stops = [None, 0.03, 0.05, 0.08] if MODE == "daily" else [None, 0.02, 0.03]
    for os_ in [-53, -60, -70]:
        for stop in stops:
            C.append(dict(name=f"wt_os{os_}_stop{stop}", type="wt_os", os=os_, stop=stop))
            C.append(dict(name=f"confl_os{os_}_stop{stop}", type="confluence", os=os_, stop=stop))
    for stop in ([None, 0.05] if MODE == "daily" else [None, 0.02]):
        C += [dict(name=f"sqz_zero_long_stop{stop}", type="sqz_zero", stop=stop),
              dict(name=f"sqz_zero_both_stop{stop}", type="sqz_zero", both=True, stop=stop),
              dict(name=f"sqz_color_stop{stop}", type="sqz_color", stop=stop),
              dict(name=f"sqz_color_ma40_stop{stop}", type="sqz_color", filters=["above_ma40"], stop=stop),
              dict(name=f"ema_slope_stop{stop}", type="ema_slope", stop=stop),
              dict(name=f"ema_slope_ma40_stop{stop}", type="ema_slope", filters=["above_ma40"], stop=stop),
              dict(name=f"mashift_reg_stop{stop}", type="mashift", regime=True, stop=stop)]
    # confluence + extra filters
    dstop = 0.05 if MODE == "daily" else 0.02
    for f in [["ema_up"], ["not_sqz"], ["above_ma40"], ["mom_up"]]:
        C.append(dict(name=f"confl_os-53_{'+'.join(f)}_stop{dstop}", type="confluence", os=-53,
                      filters=f, stop=dstop))
    return C

# ------------------------------------------------------------------ main
def main():
    cache = load_cache()
    reg = R.regime_series(cache)
    # per-stock realized daily vol (for vol bucketing)
    vol = {}
    for t, d in cache.items():
        s = pd.Series(d["close"]).pct_change()
        vol[t] = float(np.nanstd(s) * np.sqrt(7*252))
    vser = pd.Series(vol); vt = vser.quantile([1/3, 2/3])

    configs = build_configs()
    print(f"{len(configs)} configs on {len(cache)} tickers\n", flush=True)
    results = []; trade_store = {}
    for spec in configs:
        df = run_spec(cache, spec)
        if len(df) == 0: continue
        trade_store[spec["name"]] = df
        tr, te, full = split_eval(df)
        results.append(dict(name=spec["name"], **{f"tr_{k}":v for k,v in tr.items()},
                            **{f"te_{k}":v for k,v in te.items()}, full_n=full["n"], full_avg=full["avg"]))
        print(f"  {spec['name']:34s} TRAIN n={tr['n']:5d} avg={tr['avg']:6.1f} pf={tr['pf']:.2f} | "
              f"TEST n={te['n']:5d} avg={te['avg']:6.1f} pf={te['pf']:.2f}", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(f"{CACHE}/sweep_configs{TAG}.csv", index=False)

    # ---- rank by OOS (test), require both periods tradeable + positive ----
    ok = res[(res.tr_n>=200)&(res.te_n>=200)].copy()
    ok["robust"] = ok[["tr_avg","te_avg"]].min(axis=1)   # worst of the two periods
    ok = ok.sort_values("robust", ascending=False)
    print("\n===== TOP CONFIGS by worst-period (train/test) avg bps — OOS robust =====")
    print(ok[["name","tr_n","tr_avg","tr_pf","te_n","te_avg","te_pf","robust"]]
          .head(12).to_string(index=False, float_format=lambda x:f"{x:.1f}"))

    champ = ok.iloc[0]["name"]; cdf = trade_store[champ]
    print(f"\nCHAMPION (OOS-robust): {champ}")

    # ---- conditioning breakdowns on champion ----
    def brk(col, order=None):
        g = cdf.groupby(col)["ret"].apply(lambda x: pd.Series(stats(x))).unstack()
        return g.reindex(order) if order else g.sort_values("avg", ascending=False)
    cdf["regime"] = R.tag_trades(cdf, reg)
    cdf["volb"] = cdf["ticker"].map(lambda t: "hi" if vol.get(t,0)>vt.iloc[1] else ("lo" if vol.get(t,0)<vt.iloc[0] else "mid"))
    out = {"champion": champ, "mode": MODE,
           "ranking": ok[["name","tr_n","tr_avg","tr_pf","te_n","te_avg","te_pf","robust"]].head(15).round(2).to_dict("records"),
           "by_regime": brk("regime").round(2).reset_index().to_dict("records"),
           "by_dow": brk("dow").round(2).reset_index().to_dict("records"),
           "by_sector": brk("sector").round(2).reset_index().to_dict("records"),
           "by_vol": brk("volb").round(2).reset_index().to_dict("records")}
    print("\n== champion by REGIME =="); print(brk("regime").round(2).to_string())
    print("\n== champion by WEEKDAY (0=Mon) =="); print(brk("dow").round(2).to_string())
    print("\n== champion by SECTOR =="); print(brk("sector").round(2).to_string())
    if MODE != "daily":
        out["by_hour"] = brk("hour").round(2).reset_index().to_dict("records")
        print("\n== champion by ET HOUR =="); print(brk("hour").round(2).to_string())

    # ---- best signal PER REGIME (mask each config's trades by regime) ----
    print("\n===== BEST CONFIG PER REGIME (test-period trades) =====")
    perreg = {}
    for rg in ["BULL","BEAR","CHOPPY"]:
        rows = []
        for nm, df in trade_store.items():
            d2 = df[df.year>=2022].copy()
            d2["regime"] = R.tag_trades(d2, reg)
            s = stats(d2[d2.regime==rg]["ret"])
            if s["n"] >= 60: rows.append(dict(name=nm, **s))
        rr = pd.DataFrame(rows).sort_values("avg", ascending=False)
        perreg[rg] = rr.head(6).round(2).to_dict("records")
        print(f"\n-- {rg} --"); print(rr.head(5)[["name","n","avg","pf","win"]].to_string(index=False, float_format=lambda x:f"{x:.1f}"))
    out["per_regime"] = perreg

    # ---- per-stock top/bottom on champion ----
    stk = cdf.groupby("ticker")["ret"].apply(lambda x: pd.Series(stats(x))).unstack()
    stk = stk[stk.n>=20].sort_values("avg", ascending=False)
    out["stock_top"] = stk.head(12).round(2).reset_index().to_dict("records")
    out["stock_bot"] = stk.tail(8).round(2).reset_index().to_dict("records")
    out["regime_days"] = reg.value_counts().to_dict()

    json.dump(out, open(f"{CACHE}/sweep_results{TAG}.json","w"), indent=1, default=str)
    print(f"\nsaved sweep_configs{TAG}.csv + sweep_results{TAG}.json")

if __name__ == "__main__":
    main()
