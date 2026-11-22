"""Survivor stats derived from the completed backfill parquet (no extra API calls)."""
import sys, pandas as pd
path = sys.argv[1] if len(sys.argv) > 1 else "data_alpaca/stocks_1h.parquet"
df = pd.read_parquet(path, columns=["ticker", "datetime"])
sec = pd.read_parquet("sectors.parquet").set_index("ticker")["sector"].to_dict()
universe = set(pd.read_parquet("sectors.parquet")["ticker"])

first = df.groupby("ticker")["datetime"].min().reset_index()
last = df.groupby("ticker")["datetime"].max()
nbar = df.groupby("ticker").size()
first["year"] = first["datetime"].dt.year
first["sector"] = first["ticker"].map(sec)
present = set(first["ticker"])
missing = sorted(universe - present)   # returned no data (incl. dotted BRK-B/BF-B)

FLOOR = pd.Timestamp("2016-02-01")
full = first[first["datetime"] <= FLOOR]

print(f"universe (current S&P 500): {len(universe)}")
print(f"tickers with ANY data     : {len(present)}   missing: {len(missing)} {missing[:12]}")
print(f"\n=== SURVIVOR PANEL: full history from 2016 ===")
print(f"  survivors (data <= 2016-01): {len(full)}  = {len(full)/len(universe)*100:.0f}% of {len(universe)}")
print(f"  later IPO/addition         : {len(present)-len(full)}")
print(f"\n=== first-bar YEAR distribution (listing/addition recency) ===")
yd = first["year"].value_counts().sort_index()
for y, n in yd.items():
    print(f"  {y}: {n:3d}  {'#'*n}")
print(f"\n=== survivors by sector ===")
tab = pd.concat([full.groupby("sector").size().rename("survivors"),
                 first.groupby("sector").size().rename("with_data")], axis=1).fillna(0).astype(int)
tab["survivor_%"] = (tab["survivors"]/tab["with_data"]*100).round(0)
print(tab.sort_values("survivors", ascending=False).to_string())
full[["ticker","sector"]].to_parquet("data_alpaca/survivors.parquet")
print(f"\nsaved data_alpaca/survivors.parquet ({len(full)} names) — the clean panel for bias-controlled tests")
