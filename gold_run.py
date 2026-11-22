"""Run the WT+Squeeze engine on 21yr XAUUSD: edge-over-drift, bootstrap, walk-forward."""
import json, numpy as np, pandas as pd
import bt, assets

CACHE = bt.CACHE
COST = 1.5  # bps/side, tight gold CFD

def bench_at(cache, bars):
    c = 2 * COST / 1e4
    C = cache["XAUUSD"]["close"]
    h = max(int(round(bars)), 1)
    fwd = C[h:] / C[:-h] - 1.0 - c
    fwd = fwd[np.isfinite(fwd)]
    return float(fwd.mean())

def boot_ci(r, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    b = rng.choice(r, size=(n, len(r)), replace=True).mean(axis=1)
    return np.percentile(b, [2.5, 97.5]) * 1e4, float((b <= 0).mean())

STRATS = ["WT_oversold_long", "Confluence_oversold", "WT_meanrev_both",
          "WT_ob_short", "SQZ_zero_cross", "SQZ_color_long"]

def run_tf(tf):
    cache = assets.gold_cache(tf)
    rows = []; trades_store = {}
    for s in STRATS:
        for stop in [None, 0.02]:
            tr = bt.run_strategy(cache, s, stop=stop, cost_bps=COST)
            if len(tr) == 0: continue
            m = bt.metrics(tr)
            edge = m["avg"] - bench_at(cache, m["avg_bars"])
            row = dict(tf=tf, strat=s, stop=("2%" if stop else "none"),
                       n=m["n"], win=m["win"], avg_bps=m["avg"]*1e4,
                       edge_bps=edge*1e4, pf=m["pf"], sharpe=m["sharpe"],
                       bars=m["avg_bars"])
            if m["n"] >= 100:
                (lo, hi), p0 = boot_ci(tr["ret"].to_numpy())
                row["ci_lo"], row["ci_hi"], row["p0"] = lo, hi, p0
            rows.append(row)
            trades_store[(s, "2%" if stop else "none")] = tr
    return pd.DataFrame(rows), trades_store, cache

allrows = []; wf_out = {}
for tf in ["1h", "4h"]:
    df, trades_store, cache = run_tf(tf)
    allrows.append(df)
    print(f"\n===== GOLD {tf} (net {COST}bps/side) =====")
    cols = ["strat","stop","n","win","avg_bps","edge_bps","pf","sharpe","bars","p0"]
    show = df[[c for c in cols if c in df]].copy()
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # walk-forward on the frozen winner (Confluence_oversold + 2% stop); fallback to WT_oversold_long
    for key in [("Confluence_oversold","2%"), ("WT_oversold_long","2%")]:
        if key in trades_store and len(trades_store[key]) >= 200:
            tr = trades_store[key].copy()
            tr["year"] = pd.to_datetime(tr["entry_time"]).dt.year
            wf = tr.groupby("year").agg(n=("ret","size"), win=("ret",lambda x:(x>0).mean()),
                                        avg_bps=("ret",lambda x:x.mean()*1e4),
                                        pf=("ret",lambda x: x[x>0].sum()/(-x[x<=0].sum()) if (x<=0).any() else np.inf))
            print(f"\n  walk-forward by year — {key[0]} +{key[1]} stop, {tf}:")
            print(wf.to_string(float_format=lambda x: f"{x:.2f}"))
            wf_out[f"{tf}_{key[0]}"] = wf.reset_index().to_dict("records")
            # train/test split (first 60% vs last 40% of trades by time)
            tr = tr.sort_values("entry_time").reset_index(drop=True)
            k = int(len(tr)*0.6)
            for lbl, part in [("train(1st60%)", tr.iloc[:k]), ("test(last40%)", tr.iloc[k:])]:
                r = part["ret"].to_numpy()
                print(f"    {lbl}: n={len(r)} avg={r.mean()*1e4:.1f}bps win={(r>0).mean():.2f} "
                      f"pf={r[r>0].sum()/(-r[r<=0].sum()):.2f}")
            break

res = pd.concat(allrows, ignore_index=True)
res.to_csv(f"{CACHE}/gold_results.csv", index=False)
json.dump({"wf": wf_out}, open(f"{CACHE}/gold_wf.json","w"), indent=1, default=str)
print("\nsaved gold_results.csv + gold_wf.json")
