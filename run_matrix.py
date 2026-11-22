"""Run the full strategy x exit-config x timeframe matrix; save results + benchmark."""
import os, time, json
import numpy as np, pandas as pd
import bt

CACHE = bt.CACHE
TFS = ["15m", "1h", "4h", "1d"]
COST = 2.0  # bps per side

# exit configs (idea #7 = quick small wins). None = pure signal exit.
CONFIGS = {
    "signal":   dict(),
    "scalp_a":  dict(target=0.004, stop=0.006, time_bars=6),
    "scalp_b":  dict(target=0.006, stop=0.008, time_bars=12),
    "runlet":   dict(target=0.012, stop=0.02),
    "time6":    dict(time_bars=6),
}


def benchmark(cache, horizons=(1, 3, 6, 12)):
    """Unconditional mean forward return per bar (net of 2*cost) = drift baseline."""
    rows = []
    c = 2 * COST / 1e4
    for h in horizons:
        rr = []
        for d in cache.values():
            C = d["close"]
            if len(C) > h:
                fwd = C[h:] / C[:-h] - 1.0 - c
                rr.append(fwd[np.isfinite(fwd)])
        r = np.concatenate(rr)
        rows.append(dict(horizon=h, mean=float(r.mean()), median=float(np.median(r)),
                         win=float((r > 0).mean()), n=len(r)))
    return pd.DataFrame(rows)


def main():
    results, benches = [], {}
    for tf in TFS:
        t = time.time()
        cache = bt.load_cache(tf)
        benches[tf] = benchmark(cache).assign(tf=tf)
        for strat in bt.STRATS:
            for cfg_name, cfg in CONFIGS.items():
                tr = bt.run_strategy(cache, strat, cost_bps=COST, **cfg)
                m = bt.metrics(tr)
                m.update(strat=strat, config=cfg_name, tf=tf)
                results.append(m)
        print(f"{tf}: done in {time.time()-t:.1f}s", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(f"{CACHE}/matrix_results.csv", index=False)
    bench = pd.concat(benches.values(), ignore_index=True)
    bench.to_csv(f"{CACHE}/benchmark.csv", index=False)
    print("\n=== BENCHMARK (unconditional drift, net) ===")
    print(bench.to_string(index=False))
    # top strategies by expectancy with >=200 trades
    good = res[res["n"] >= 200].copy()
    good["avg_bps"] = good["avg"] * 1e4
    print("\n=== TOP 20 by avg net return/trade (n>=200) ===")
    cols = ["tf", "strat", "config", "n", "win", "avg_bps", "pf", "sharpe", "sum_ret", "avg_bars"]
    print(good.sort_values("avg", ascending=False)[cols].head(20)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
