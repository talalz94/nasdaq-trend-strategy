"""Consolidate daily-sweep + finalize results into report_data3.json."""
import json, numpy as np, pandas as pd
import bt
CACHE = bt.CACHE
sw = json.load(open(f"{CACHE}/sweep_results_1d.json"))
fn = json.load(open(f"{CACHE}/finalize_daily.json"))
cfg = pd.read_csv(f"{CACHE}/sweep_configs_1d.csv")

# ranking already in sw['ranking']; add robust sort ensure
rank = sw["ranking"][:12]

# per-regime best (test)
perreg = {k: v[:5] for k, v in sw["per_regime"].items()}

def num(x):
    try: return float(x)
    except: return None

out = {
  "meta": {"tickers": 498, "survivors": 462, "years": 10.5, "cost_bps": 2.0},
  "ranking": rank,
  "by_regime": sw["by_regime"],
  "by_sector": sw["by_sector"],
  "by_dow": sw["by_dow"],
  "by_vol": sw["by_vol"],
  "per_regime": perreg,
  "regime_days": sw["regime_days"],
  "stock_top": sw["stock_top"][:12],
  "stock_bot": sw["stock_bot"][:8],
  "champ_port": fn["champ_port"],
  "combo_port": fn["combo_port"],
  "bh": fn["bh"],
  "combo_equity": fn["combo_equity"],
  "tf": fn["tf"],
  "edge_note": {"champ_avg": 304, "drift": 308, "edge": -4, "hold_days": 48},
}
json.dump(out, open("report_data3.json", "w"), indent=1, default=str)
print("wrote report_data3.json | ranking:", len(rank), "| sectors:", len(out["by_sector"]),
      "| equity pts:", len(out["combo_equity"]), "| tf:", len(out["tf"]))
