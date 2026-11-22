"""
Helper for r2_chop_vol_matched.py: fetch the three cross-market indices that are NOT in the panel
(DAX ^GDAXI, Nikkei 225 ^N225, Nasdaq Composite ^IXIC) from yfinance and cache their daily closes to
update_2026_11/r2_chop_vol_matched_yf.parquet (long format: date, ticker, close).

Yahoo was returning HTTP 429 (rate limit) when this round started, so each ticker is retried with spaced
waits. Tickers already cached are not re-fetched. Nothing else is written.
"""
import os, sys, time
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "r2_chop_vol_matched_yf.parquet")
TICKERS = ["^GDAXI", "^N225", "^IXIC"]
WAITS = [0, 20, 40, 60, 90, 120]          # seconds before each attempt (python sleep; total < 6 min)


def fetch(t):
    for i, w in enumerate(WAITS):
        if w:
            time.sleep(w)
        try:
            h = yf.Ticker(t).history(period="max", auto_adjust=False)
            c = h["Close"].dropna() if len(h) else pd.Series(dtype=float)
            if len(c) > 1000:
                c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
                c = c[~c.index.duplicated(keep="last")]
                print(f"  {t}: {len(c)} rows {c.index[0].date()}..{c.index[-1].date()} (attempt {i+1})", flush=True)
                return c
            print(f"  {t}: empty (attempt {i+1})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: error {type(e).__name__}: {str(e)[:120]} (attempt {i+1})", flush=True)
    return None


have = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["date", "ticker", "close"])
done = set(have["ticker"].unique()) if len(have) else set()
parts = [have] if len(have) else []
for t in TICKERS:
    if t in done:
        print(f"  {t}: cached", flush=True)
        continue
    c = fetch(t)
    if c is None:
        continue
    parts.append(pd.DataFrame({"date": c.index, "ticker": t, "close": c.values.astype(float)}))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT, index=False)          # save after every ticker so partial progress survives
missing = [t for t in TICKERS if t not in (set(pd.read_parquet(OUT)["ticker"]) if os.path.exists(OUT) else set())]
print("missing:", missing, flush=True)
sys.exit(0 if not missing else 2)
