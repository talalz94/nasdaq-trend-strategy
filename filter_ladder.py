"""
Idea #6: freeze the winner, then greedily add ONE veto filter at a time, keeping a
filter only if it lifts BOTH expectancy and profit factor while leaving enough trades.
Never changes the entry/exit — only vetoes bad trades.
"""
import json, numpy as np, pandas as pd
import bt, assets

CACHE = bt.CACHE
sectors = pd.read_parquet("sectors.parquet").set_index("ticker")["sector"].to_dict()
GOOD_SEC = {"Health Care","Utilities","Information Technology","Financials",
            "Consumer Staples","Real Estate","Industrials"}

def enrich(cache, strat, stop, cost=2.0):
    """Run strategy, attach signal-bar features to each trade (no lookahead)."""
    tr = bt.run_strategy(cache, strat, stop=stop, cost_bps=cost)
    feats = []
    for _, t in tr.iterrows():
        d = cache[t["ticker"]]; si = int(t["entry_i"]) - 1
        si = max(si, 0)
        dt = pd.Timestamp(d["datetime"][si])
        feats.append(dict(
            sector=sectors.get(t["ticker"], "?"),
            ema8_up=bool(d["ma8_up"][si]) if "ma8_up" in d and np.isfinite(d["ma8_up"][si]) else False,
            wt2=float(d["wt2"][si]),
            mom_color=int(d["mom_color"][si]),
            mom_rising=bool(d["mom_rising"][si]),
            sqz_on=bool(d["sqz_on"][si]),
            sqz_release=bool(d["sqz_release"][si]),
            hour=dt.hour,
            ms_regime=bool(d["mashift_regime_up"][si]) if "mashift_regime_up" in d and np.isfinite(d["mashift_regime_up"][si]) else False,
            ms_up=bool(d["mashift_up"][si]) if "mashift_up" in d and np.isfinite(d["mashift_up"][si]) else False,
        ))
    return pd.concat([tr.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

def stats(r):
    r = r.to_numpy() if hasattr(r, "to_numpy") else r
    if len(r) == 0: return dict(n=0, avg=np.nan, pf=np.nan, win=np.nan)
    gp = r[r > 0].sum(); gl = -r[r <= 0].sum()
    return dict(n=len(r), avg=r.mean()*1e4, win=(r > 0).mean(),
                pf=(gp/gl if gl > 0 else np.inf))

# candidate veto filters: name -> boolean mask over the enriched trade frame (True = keep)
def candidates(df, kind="stock"):
    c = {
        "ema8_rising":       df["ema8_up"],
        "deep_oversold<-60": df["wt2"] < -60,
        "mom_maroon(neg_up)":df["mom_color"] == -1,
        "not_in_squeeze":    ~df["sqz_on"],
        "squeeze_release":   df["sqz_release"],
    }
    if kind == "stock":
        c["sector_whitelist"] = df["sector"].isin(GOOD_SEC)
        c["session_am(<16h)"] = df["hour"] < 16
    else:  # gold: trend-alignment filters instead of sector
        c["ema8_trend_up"] = df["ema8_up"]
        c["price_above_MA40"] = df["ms_regime"]
        c["mashift_osc_up"] = df["ms_up"]
    return c

def greedy(df, min_n=150, label="", kind="stock"):
    base = stats(df["ret"]); keep_mask = pd.Series(True, index=df.index); chosen=[]
    ladder=[dict(step="baseline", **base)]
    print(f"\n===== FILTER LADDER — {label} =====")
    print(f"{'step':22s}{'n':>6}{'win':>6}{'avg_bps':>9}{'pf':>6}")
    print(f"{'baseline':22s}{base['n']:6d}{base['win']:6.2f}{base['avg']:9.2f}{base['pf']:6.2f}")
    cur = base
    while True:
        best=None
        for nm, m in candidates(df, kind).items():
            if nm in chosen: continue
            mask = keep_mask & m.fillna(False)
            s = stats(df.loc[mask, "ret"])
            if s["n"] < min_n: continue
            # keep only if BOTH expectancy and PF improve
            if s["avg"] > cur["avg"] + 0.5 and s["pf"] >= cur["pf"]:
                if best is None or s["avg"] > best[1]["avg"]:
                    best=(nm, s, mask)
        if best is None: break
        nm, s, mask = best; chosen.append(nm); keep_mask = mask; cur = s
        ladder.append(dict(step="+ "+nm, **s))
        print(f"{'+ '+nm:22s}{s['n']:6d}{s['win']:6.2f}{s['avg']:9.2f}{s['pf']:6.2f}")
    print(f"final filters: {chosen}")
    print(f"lift: avg {base['avg']:.1f} -> {cur['avg']:.1f} bps | pf {base['pf']:.2f} -> {cur['pf']:.2f} | "
          f"kept {cur['n']}/{base['n']} trades ({cur['n']/base['n']*100:.0f}%)")
    # diagnostic: every candidate's standalone effect on top of the chosen set (incl. rejected)
    print("  -- rejected/other filters on top of final set --")
    keep = pd.Series(True, index=df.index)
    for nm in chosen: keep &= candidates(df, kind)[nm].fillna(False)
    for nm, m in candidates(df, kind).items():
        if nm in chosen: continue
        s = stats(df.loc[keep & m.fillna(False), "ret"])
        print(f"     +{nm:20s} -> n={s['n']:4d} avg={s['avg']:6.1f} pf={s['pf']:.2f}")
    return ladder, chosen

if __name__ == "__main__":
    # ---- stocks: the frozen equity winner ----
    stk = bt.load_cache("1h")
    df = enrich(stk, "Confluence_oversold", 0.02)
    ladder_stk, chosen_stk = greedy(df, min_n=150, label="Confluence_oversold 1h (stocks)", kind="stock")

    # ---- gold: build a champion from the momentum base + trend filters ----
    out = {"stocks": ladder_stk, "chosen_stocks": chosen_stk}
    for base_strat, tf in [("SQZ_zero_cross","1h"), ("MAShift_regime_long","1h"), ("SQZ_color_long","4h")]:
        gc = assets.gold_cache(tf)
        gdf = enrich(gc, base_strat, None, cost=1.5)  # gold: no hard stop in base ladder
        lad, ch = greedy(gdf, min_n=120, label=f"{base_strat} {tf} (gold)", kind="gold")
        out[f"gold_{base_strat}_{tf}"] = lad; out[f"gold_{base_strat}_{tf}_chosen"] = ch

    json.dump(out, open(f"{CACHE}/filter_ladder.json","w"), indent=1, default=str)
    print("\nsaved filter_ladder.json")
