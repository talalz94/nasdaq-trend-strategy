"""Edge-over-drift ranking + bootstrap significance + market regime."""
import numpy as np, pandas as pd
import bt

CACHE = bt.CACHE
res = pd.read_csv(f"{CACHE}/matrix_results.csv")
bench = pd.read_csv(f"{CACHE}/benchmark.csv")

SHORT_STRATS = {"WT_ob_short"}  # pure short; meanrev/zero/fire are mixed


def bench_at(tf, bars):
    b = bench[bench.tf == tf].sort_values("horizon")
    return float(np.interp(bars, b.horizon, b["mean"]))


res["bench_matched"] = [bench_at(tf, hb) for tf, hb in zip(res.tf, res.avg_bars)]
# for pure-short strat, drift helps shorts negatively -> edge = avg - (-bench) = avg + bench
res["edge"] = np.where(res.strat.isin(SHORT_STRATS),
                       res["avg"] + res["bench_matched"],
                       res["avg"] - res["bench_matched"])
res["avg_bps"] = res["avg"] * 1e4
res["edge_bps"] = res["edge"] * 1e4
res["bench_bps"] = res["bench_matched"] * 1e4

good = res[res.n >= 250].copy()
cols = ["tf", "strat", "config", "n", "win", "avg_bps", "bench_bps", "edge_bps", "pf", "sharpe", "avg_bars"]
print("=== TOP 18 by EDGE over matched-horizon drift (n>=250) ===")
print(good.sort_values("edge", ascending=False)[cols].head(18)
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

res.to_csv(f"{CACHE}/matrix_edge.csv", index=False)

# ---- bootstrap 95% CI on mean net return for headline strategies (1h) ----
print("\n=== BOOTSTRAP 95% CI on mean net return/trade (10k resamples) ===")
rng = np.random.default_rng(42)
cache1h = bt.load_cache("1h")
cache4h = bt.load_cache("4h")
for tf, cache, strat, cfg in [("1h", cache1h, "WT_oversold_long", {}),
                              ("1h", cache1h, "Confluence_oversold", {}),
                              ("4h", cache4h, "WT_oversold_long", {}),
                              ("1h", cache1h, "SQZ_color_long", {})]:
    tr = bt.run_strategy(cache, strat, cost_bps=2.0, **cfg)
    r = tr["ret"].to_numpy()
    boot = rng.choice(r, size=(10000, len(r)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 1e4
    # p(mean<=0)
    p0 = float((boot <= 0).mean())
    print(f"{tf:3s} {strat:20s} n={len(r):5d} mean={r.mean()*1e4:7.2f}bps  "
          f"95%CI=[{lo:6.2f},{hi:6.2f}]  p(mean<=0)={p0:.4f}")

# ---- market regime: equal-weight cumulative over window (1h closes) ----
print("\n=== MARKET REGIME (equal-weight avg of per-bar returns, 1h) ===")
rets = []
for d in cache1h.values():
    C = d["close"]
    rets.append(pd.Series(C).pct_change().to_numpy())
m = np.nanmean(np.vstack([r[:min(len(x) for x in rets)] for r in
                          [np.resize(r, max(len(x) for x in rets)) for r in rets]]), axis=0)
# simpler: pool by concatenating equal-weight bar means using aligned min length
minlen = min(len(d["close"]) for d in cache1h.values())
mat = np.vstack([pd.Series(d["close"][:minlen]).pct_change().to_numpy() for d in cache1h.values()])
ew = np.nanmean(mat, axis=0)
cum = np.nancumprod(1 + ew) - 1
print(f"equal-weight cumulative over window: {cum[-1]*100:+.2f}%  "
      f"max={np.nanmax(cum)*100:+.2f}%  min={np.nanmin(cum)*100:+.2f}%")
print(f"peak-to-trough drawdown in window: {(np.nanmin(cum[np.nanargmax(cum):])-np.nanmax(cum))*100:.2f}% "
      f"(shows the window had both up and down legs)")
