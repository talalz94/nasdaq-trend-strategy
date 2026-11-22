"""Consolidate all backtest outputs into report_data.json for the HTML report."""
import json, numpy as np, pandas as pd
import bt
CACHE = bt.CACHE

def rd(name): return pd.read_csv(f"{CACHE}/{name}")

edge = rd("matrix_edge.csv")
edge = edge[edge.n >= 250].sort_values("edge", ascending=False)
edge_top = edge.head(14)[["tf","strat","config","n","win","avg_bps","bench_bps","edge_bps","pf","sharpe","avg_bars"]]

exits = rd("exit_experiments.csv")
sector = rd("sector_breakdown.csv")
stock = rd("stock_breakdown.csv")
bench = rd("benchmark.csv")
opt = rd("options_sim.csv")
eq = rd("equity_curve.csv")

# downsample equity curve to ~200 pts
eq["time"] = pd.to_datetime(eq["time"])
eq = eq.set_index("time")["equity"]
eqd = eq.resample("6h").last().dropna()
equity_pts = [{"t": t.strftime("%m-%d %H:%M"), "e": round(float(v),1)} for t,v in eqd.items()]

# option return histogram buckets
bins = [-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,1.0,2.0,5.0]
labels = ["-100..-75","-75..-50","-50..-25","-25..0","0..25","25..50","50..100","100..200","200..500"]
hist = pd.cut(opt["opt_ret"], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels).fillna(0)

# full timeframe summary of the winning entry (from matrix_edge)
win_tf = edge[edge.strat.isin(["WT_oversold_long","Confluence_oversold"]) & (edge.config=="signal")]
win_tf = win_tf[["tf","strat","n","win","avg_bps","edge_bps","pf","sharpe"]]

out = {
  "meta": {"tickers": int(len(pd.read_parquet("sectors.parquet"))),
           "window": "2026-05-26 to 2026-07-24 (42 trading days)",
           "cost_bps_side": 2.0, "ew_buyhold_pct": 0.76},
  "edge_top": edge_top.round(2).to_dict("records"),
  "exits": exits.round(2).to_dict("records"),
  "sector": sector.round(2).to_dict("records"),
  "stock_top": stock.sort_values("avg_bps",ascending=False).head(12).round(1).to_dict("records"),
  "stock_bot": stock.sort_values("avg_bps").head(8).round(1).to_dict("records"),
  "stock_pos_frac": round(float((stock.avg_bps>0).mean())*100,1),
  "benchmark": bench.round(4).to_dict("records"),
  "win_tf": win_tf.round(2).to_dict("records"),
  "portfolio": {
     "variants": [
        {"max_pos":10,"risk":"10%","total":7.5,"maxdd":-3.6,"sharpe":3.50,"taken":191},
        {"max_pos":20,"risk":"5%","total":5.4,"maxdd":-1.9,"sharpe":3.73,"taken":337},
        {"max_pos":5,"risk":"20%","total":4.5,"maxdd":-5.0,"sharpe":2.15,"taken":93}],
     "equity": equity_pts},
  "options": {
     "n": int(len(opt)),
     "stock_avg_pct": round(float(opt.stock_ret.mean())*100,3),
     "stock_win": round(float((opt.stock_ret>0).mean()),3),
     "opt_avg_pct": round(float(opt.opt_ret.mean())*100,2),
     "opt_win": round(float((opt.opt_ret>0).mean()),3),
     "opt_median_pct": round(float(opt.opt_ret.median())*100,1),
     "opt_best_pct": round(float(opt.opt_ret.max())*100,0),
     "opt_worst_pct": round(float(opt.opt_ret.min())*100,0),
     "opt_pf": 1.56,
     "hist": [{"bucket":b,"n":int(n)} for b,n in hist.items()]},
}
with open("report_data.json","w") as f:
    json.dump(out, f, indent=1)
print("wrote report_data.json")
print("edge_top rows:", len(out["edge_top"]), "| sectors:", len(out["sector"]),
      "| equity pts:", len(equity_pts), "| option hist:", sum(h['n'] for h in out['options']['hist']))
