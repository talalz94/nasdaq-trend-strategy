"""
Round-2 helper for r2_friday_effect.py: fetch the extra index histories once (own trading calendars)
and cache them to update_2026_11/r2_friday_effect_yf.parquet (long format). Read-only w.r.t. every other file.

  ^GDAXI  DAX            (cross-market replication, own calendar)
  ^N225   Nikkei 225     (cross-market replication, own calendar)
  ^IXIC   Nasdaq Comp.   (cross-market replication 1975+, and NDX data cross-check)
  ^HSI    Hang Seng      (own-calendar robustness of the panel 'hsi' column)
  ^GSPC   S&P 500        (US trading-calendar cross-check of the NDX history)
  ^NDX    Nasdaq-100     (fresh OHLC copy for the 1985-1990 data-quality audit)
  ^IRX, ^TNX             (rf and synthetic-IEF inputs for 1975-1985, before the panel starts)
"""
import os
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "r2_friday_effect_yf.parquet")
TICKERS = ["^GDAXI", "^N225", "^IXIC", "^HSI", "^GSPC", "^NDX", "^IRX", "^TNX"]

frames = []
for t in TICKERS:
    h = yf.Ticker(t).history(period="max", auto_adjust=False)
    if len(h) == 0:
        print(f"{t}: EMPTY", flush=True)
        continue
    h.index = h.index.tz_localize(None).normalize()
    h = h[~h.index.duplicated(keep="last")]
    df = pd.DataFrame({"date": h.index, "ticker": t,
                       "open": h["Open"].values, "high": h["High"].values, "low": h["Low"].values,
                       "close": h["Close"].values, "volume": h["Volume"].values})
    frames.append(df)
    print(f"{t:8s} {h.index[0].date()} .. {h.index[-1].date()}  rows {len(h)}", flush=True)

out = pd.concat(frames, ignore_index=True)
out.to_parquet(OUT)
print("saved", OUT, out.shape)
