"""Improved-exit experiments + stock/sector breakdown for the winning entry."""
import numpy as np, pandas as pd
import bt

CACHE = bt.CACHE
sectors = pd.read_parquet("sectors.parquet").set_index("ticker")["sector"].to_dict()


def summarize(tr):
    m = bt.metrics(tr)
    return dict(n=m["n"], win=m["win"], avg_bps=m["avg"] * 1e4, pf=m["pf"],
                sharpe=m["sharpe"], sum=m["sum_ret"] * 100, bars=m["avg_bars"],
                maxdd=m["maxdd"] * 100)


print("=== EXIT EXPERIMENTS on the winning oversold entry (1h & 4h) ===")
rows = []
for tf in ["1h", "4h"]:
    cache = bt.load_cache(tf)
    experiments = [
        ("WT_oversold_long", "bearish-cross exit (base)", {}),
        ("WT_oversold_ob_exit", "exit@overbought|cross", {}),
        ("WT_oversold_long", "trail 1.5%", dict(trail=0.015)),
        ("WT_oversold_long", "trail 2.5%", dict(trail=0.025)),
        ("WT_oversold_long", "stop 2% + cross exit", dict(stop=0.02)),
        ("WT_oversold_long", "stop 2% + trail 2%", dict(stop=0.02, trail=0.02)),
        ("WT_oversold_long", "target 3% + stop 2%", dict(target=0.03, stop=0.02)),
        ("Confluence_oversold", "confirm + bearish-cross (base)", {}),
        ("Confluence_oversold_ob_exit", "confirm + exit@overbought", {}),
        ("Confluence_oversold", "confirm + stop 2%", dict(stop=0.02)),
        ("Confluence_oversold", "confirm + trail 2%", dict(trail=0.02)),
    ]
    for strat, label, kw in experiments:
        tr = bt.run_strategy(cache, strat, cost_bps=2.0, **kw)
        s = summarize(tr); s.update(tf=tf, strat=strat, exit=label)
        rows.append(s)
exp = pd.DataFrame(rows)
exp.to_csv(f"{CACHE}/exit_experiments.csv", index=False)
cols = ["tf", "strat", "exit", "n", "win", "avg_bps", "pf", "sharpe", "sum", "bars", "maxdd"]
print(exp[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# ---- best config -> sector & stock breakdown (1h) ----
print("\n=== SECTOR BREAKDOWN: Confluence_oversold, 1h, bearish-cross exit ===")
cache = bt.load_cache("1h")
tr = bt.run_strategy(cache, "WT_oversold_long", cost_bps=2.0)
tr["sector"] = tr["ticker"].map(sectors)
sec = tr.groupby("sector").agg(n=("ret", "size"), win=("ret", lambda x: (x > 0).mean()),
                               avg_bps=("ret", lambda x: x.mean() * 1e4),
                               sum_pct=("ret", lambda x: x.sum() * 100))
sec["pf"] = tr.groupby("sector")["ret"].apply(lambda x: x[x > 0].sum() / -x[x <= 0].sum() if (x <= 0).any() else np.inf)
sec = sec.sort_values("avg_bps", ascending=False)
sec.to_csv(f"{CACHE}/sector_breakdown.csv")
print(sec.to_string(float_format=lambda x: f"{x:.2f}"))

# ---- stock breakdown ----
stk = tr.groupby("ticker").agg(n=("ret", "size"), win=("ret", lambda x: (x > 0).mean()),
                               avg_bps=("ret", lambda x: x.mean() * 1e4),
                               sum_pct=("ret", lambda x: x.sum() * 100))
stk = stk[stk.n >= 3]
stk.to_csv(f"{CACHE}/stock_breakdown.csv")
print("\n=== TOP 15 STOCKS by avg net/trade (>=3 trades), WT_oversold_long 1h ===")
print(stk.sort_values("avg_bps", ascending=False).head(15).to_string(float_format=lambda x: f"{x:.2f}"))
print("\n=== BOTTOM 10 STOCKS ===")
print(stk.sort_values("avg_bps").head(10).to_string(float_format=lambda x: f"{x:.2f}"))
print(f"\n% of stocks with positive expectancy: {(stk.avg_bps>0).mean()*100:.1f}%  (n stocks={len(stk)})")
