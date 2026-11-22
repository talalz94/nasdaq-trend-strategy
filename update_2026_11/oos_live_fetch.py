"""
oos_live helper: fetch daily OHLC (dividend-adjusted) for the instruments the live
strategies actually hold, for an INDEPENDENT check of the shared panel's real-ETF
returns over the out-of-sample window and for a Monday-open execution variant.

Writes only: update_2026_11/oos_live_yf.parquet   (long format: date, ticker, open, high, low, close)
Never touches panel.parquet or any other shared file.
"""
import os
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oos_live_yf.parquet")
TICKERS = ["^NDX", "QQQ", "QLD", "TQQQ", "TLT", "IEF", "SHY", "BIL", "GLD", "SPY", "GC=F"]
START = "2002-07-01"     # long enough for the in-sample (real-ETF era) execution-timing check

frames = []
for t in TICKERS:
    try:
        h = yf.Ticker(t).history(start=START, auto_adjust=True)
    except Exception as e:                      # network / ticker failure -> recorded, not fatal
        print(f"  {t:6s} FAILED: {e}", flush=True)
        continue
    if h is None or len(h) == 0:
        print(f"  {t:6s} no data", flush=True)
        continue
    h.index = h.index.tz_localize(None).normalize()
    h = h[~h.index.duplicated()]
    f = pd.DataFrame({"date": h.index, "ticker": t, "open": h["Open"].values, "high": h["High"].values,
                      "low": h["Low"].values, "close": h["Close"].values})
    frames.append(f)
    print(f"  {t:6s} {len(f):4d} rows  {f.date.min().date()}..{f.date.max().date()}", flush=True)

if frames:
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT)
    print(f"saved {OUT}  rows={len(df)}")
else:
    print("nothing fetched; main script will skip the independent check and the open-execution variant")
