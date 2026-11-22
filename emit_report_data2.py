"""Consolidate phase-2 outputs into report_data2.json."""
import json, numpy as np, pandas as pd
import bt
CACHE = bt.CACHE

gold = pd.read_csv(f"{CACHE}/gold_results.csv")
lad = json.load(open(f"{CACHE}/filter_ladder.json"))
wf = json.load(open(f"{CACHE}/walkforward.json"))
lev = json.load(open(f"{CACHE}/leverage.json"))
goldwf = json.load(open(f"{CACHE}/gold_wf.json"))

# gold strategy table (signal-exit, no-stop rows for clarity), 1h & 4h
g = gold[gold["stop"] == "none"][["tf","strat","n","win","avg_bps","edge_bps","pf","p0"]].round(2)

# new-indicator isolation summary (from newind_run printout)
newind = [
 {"strat":"EMA8 slope","asset":"Stocks 1h","n":13768,"avg":1.8,"pf":1.03,"p0":0.18,"note":"whipsaw, ~flat"},
 {"strat":"EMA8 slope","asset":"Gold 4h","n":3335,"avg":4.7,"pf":1.17,"p0":0.00,"note":"weak but significant trend-follow"},
 {"strat":"MAShift long","asset":"Gold 1h","n":1125,"avg":8.1,"pf":1.14,"p0":0.05,"note":"mostly drift"},
 {"strat":"MAShift +regime","asset":"Gold 1h","n":429,"avg":19.4,"pf":1.33,"p0":0.02,"note":"trend-aligned pullback works"},
 {"strat":"MAShift +regime","asset":"Gold 4h","n":131,"avg":43.6,"pf":1.44,"p0":0.05,"note":"strong, small n"},
]

# multi-asset contrast (the headline): oversold-bounce on stocks vs gold
contrast = [
 {"strat":"WT oversold long","stocks_1h":41.4,"gold_1h":-5.8,"verdict":"stocks win / gold loses"},
 {"strat":"Confluence oversold","stocks_1h":48.1,"gold_1h":-2.7,"verdict":"stocks win / gold loses"},
 {"strat":"SQZ zero-cross (momentum)","stocks_1h":12.5,"gold_1h":7.3,"verdict":"works on both"},
]

# filter ladders (stocks + gold champions)
def ladder_rows(key):
    return [{"step":r["step"],"n":r["n"],"win":round(r.get("win",0),2),
             "avg":round(r["avg"],1),"pf":round(r["pf"],2)} for r in lad.get(key,[])]

# walk-forward champions
wf_rows = []
for k,label in [("gold_A","SQZ_color_long 4h + trend"),("gold_B","SQZ_zero_cross 1h + filters"),
                ("gold_C","MAShift_regime_long 1h"),("gold_A_base","SQZ_color_long 4h (no filter)")]:
    if k in wf["wf"]:
        w=wf["wf"][k]
        wf_rows.append({"strat":label,
            "tr_n":int(w["train"]["n"]),"tr_avg":round(w["train"]["avg"],1),"tr_pf":round(w["train"]["pf"],2),
            "te_n":int(w["test"]["n"]),"te_avg":round(w["test"]["avg"],1),"te_pf":round(w["test"]["pf"],2),
            "te_win":round(w["test"]["win"],2)})

sweep_ema=[{"p":k,**{kk:round(vv,2) for kk,vv in v.items()}} for k,v in wf["sweep_ema"].items()]
sweep_ms=[{"p":k,**{kk:round(vv,2) for kk,vv in v.items()}} for k,v in wf["sweep_ms"].items()]

# leverage table (gold_4h champion) + gold_1h
def lev_rows(key):
    return [{"L":r["L"],"final":round(r["final"]),"cagr":round(r["cagr"]*100),
             "maxdd":round(r["maxdd"]*100),"pruin":round(r["p_ruin"]*100,1),
             "med":round(r["med"]),"p05":round(r["p05"]),"p95":round(r["p95"])}
            for r in lev[key]["rows"]]

out = {
  "gold_table": g.to_dict("records"),
  "newind": newind,
  "contrast": contrast,
  "ladder_stocks": ladder_rows("stocks"),
  "ladder_gold_4h": ladder_rows("gold_SQZ_color_long_4h"),
  "ladder_gold_1h": ladder_rows("gold_SQZ_zero_cross_1h"),
  "wf": wf_rows,
  "sweep_ema": sweep_ema,
  "sweep_ms": sweep_ms,
  "lev_1h": lev_rows("gold_1h"),
  "lev_4h": lev_rows("gold_4h"),
  "lev_meta": {"kelly": round(lev["gold_4h"]["kelly"],1), "years": round(lev["gold_4h"]["years"],1),
               "start": 1000},
  "gold_wf_year": goldwf.get("wf", {}),
}
json.dump(out, open("report_data2.json","w"), indent=1)
print("wrote report_data2.json | gold rows:",len(out["gold_table"]),
      "| wf:",len(wf_rows),"| lev_4h:",len(out["lev_4h"]),"| ladders:",
      len(out["ladder_stocks"]),len(out["ladder_gold_4h"]))
