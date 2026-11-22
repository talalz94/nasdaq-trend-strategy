"""
How many current S&P 500 names have FULL history back to Alpaca's 2016 floor
(the 'survived the whole window' panel) vs later IPOs/additions.
Light probe: one limit=1 request per ticker for the earliest 1h bar.
"""
import time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, pandas as pd

KV = dict(l.strip().split("=",1) for l in open(".env") if "=" in l and not l.startswith("#"))
H = {"APCA-API-KEY-ID": KV["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": KV["ALPACA_SECRET"]}
BASE = "https://data.alpaca.markets/v2/stocks/bars"
_loc = threading.local()
def sess():
    if not hasattr(_loc,"s"): _loc.s=requests.Session()
    return _loc.s

def earliest(sym):
    asym = sym.replace("-", ".")   # Alpaca uses dots for class shares (BRK.B, BF.B)
    p = dict(symbols=asym, timeframe="1Hour", start="2010-01-01T00:00:00Z",
             limit=1, feed="sip", adjustment="split", sort="asc")
    try:
        r = None
        for a in range(5):
            try:
                r = sess().get(BASE, headers=H, params=p, timeout=30)
            except requests.RequestException:
                time.sleep(1+a); continue
            if r.status_code == 429:
                time.sleep(2+a*2); continue
            break
        if r is None or r.status_code != 200:
            return sym, None
        b = (r.json().get("bars") or {}).get(asym, [])
        return sym, (b[0]["t"][:10] if b else None)
    except Exception:
        return sym, None

uni = pd.read_parquet("sectors.parquet")
sec = uni.set_index("ticker")["sector"].to_dict()
tickers = uni["ticker"].tolist()
res = {}
t0=time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for fut in as_completed([ex.submit(earliest,s) for s in tickers]):
        try:
            s, d = fut.result(); res[s]=d
        except Exception:
            pass

df = pd.DataFrame([(s, d, sec.get(s,"?")) for s,d in res.items()], columns=["ticker","first","sector"])
df["first"] = pd.to_datetime(df["first"])
df["year"] = df["first"].dt.year
FLOOR = pd.Timestamp("2016-02-01")  # Alpaca data floor ~2016-01; allow Jan slack
full = df[df["first"] <= FLOOR]
none_ = df[df["first"].isna()]

df.to_parquet("data_alpaca/coverage.parquet")
print(f"probed {len(df)} tickers in {time.time()-t0:.0f}s\n")
print(f"=== SURVIVOR PANEL (full history from 2016) ===")
print(f"  full 2016 history : {len(full)}  ({len(full)/len(df)*100:.0f}% of {len(df)})")
print(f"  started later     : {len(df)-len(full)-len(none_)}")
print(f"  no data at all    : {len(none_)}  {sorted(none_['ticker'].tolist())[:12]}")
print("\n=== first-bar YEAR distribution (listing recency) ===")
print(df["year"].value_counts().sort_index().to_string())
print("\n=== survivor count by sector (full-2016 names) ===")
tab = pd.concat([full.groupby("sector").size().rename("survivors"),
                 df.groupby("sector").size().rename("current")], axis=1).fillna(0).astype(int)
tab["pct"] = (tab["survivors"]/tab["current"]*100).round(0)
print(tab.sort_values("survivors", ascending=False).to_string())
