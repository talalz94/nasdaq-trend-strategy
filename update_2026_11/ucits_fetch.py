"""
ucits helper: verify UCITS / non-US candidate tickers on Yahoo (Ticker.info) and fetch their full daily
history (unadjusted Close + Adj Close + Dividends + Splits + OHLCV), plus the US analogues and FX.

Writes ONLY:
  update_2026_11/ucits_prices.parquet   long format: date, ticker, open, high, low, close, adjclose, volume,
                                        dividends, splits   (dates = exchange-local trading dates)
  update_2026_11/ucits_info.json        per ticker: exists flag, selected Ticker.info fields, history span
Never touches panel.parquet or any other shared file.  Every probed ticker is recorded, including the ones
Yahoo does not know, so the report can say plainly which candidates do NOT exist.
"""
import os, json, time, warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PX = os.path.join(HERE, "ucits_prices.parquet")
OUT_INFO = os.path.join(HERE, "ucits_info.json")

# role -> tickers.  Only the tickers that verify (non-empty history + matching longName) are used downstream.
CANDIDATES = {
    "nasdaq_2x":   ["LQQ.PA", "LQQ.MI", "LQQ.DE", "LQQ.SW", "QQQ2.L", "LQQ2.L", "2QQQ.L"],
    "nasdaq_3x":   ["QQQ3.L", "LQQ3.L", "3QQQ.L", "QQQ3.MI", "TQQ3.L", "3NDX.L"],
    "nasdaq_1x":   ["CNDX.L", "EQQQ.L"],
    "ust_7_10":    ["CBU0.L", "IBTM.L"],
    "ust_20plus":  ["DTLA.L", "IDTL.L"],
    "ust_1_3":     ["IBTS.L"],
    "ust_0_1":     ["IB01.L"],
    "gold":        ["IGLN.L", "SGLN.L", "PHAU.L", "SGLD.L"],
    "us_analogue": ["QLD", "TQQQ", "QQQ", "IEF", "TLT", "SHY", "BIL", "GLD"],
    "fx":          ["EURUSD=X", "GBPUSD=X"],
}
INFO_KEYS = ["longName", "shortName", "currency", "financialCurrency", "exchange", "fullExchangeName",
             "exchangeTimezoneName", "quoteType", "legalType", "fundFamily", "category",
             "netExpenseRatio", "annualReportExpenseRatio", "totalAssets", "fundInceptionDate",
             "averageVolume", "averageVolume10days", "averageDailyVolume10Day", "averageDailyVolume3Month",
             "bid", "ask", "bidSize", "askSize", "regularMarketPrice", "regularMarketTime", "navPrice",
             "previousClose", "marketState"]


def jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


frames, info_out = [], {}
for role, tickers in CANDIDATES.items():
    for t in tickers:
        rec = {"role": role, "exists": False}
        try:
            tk = yf.Ticker(t)
            try:
                inf = tk.info or {}
            except Exception as e:                       # 404 -> Yahoo does not know the symbol
                inf = {}
                rec["info_error"] = repr(e)[:200]
            rec.update({k: jsonable(inf.get(k)) for k in INFO_KEYS})
            h = tk.history(period="max", auto_adjust=False, actions=True)
        except Exception as e:
            rec["history_error"] = repr(e)[:200]
            h = None
        if h is not None and len(h):
            h.index = h.index.tz_localize(None).normalize()
            h = h[~h.index.duplicated(keep="last")]
            f = pd.DataFrame({"date": h.index, "ticker": t,
                              "open": h["Open"].values, "high": h["High"].values, "low": h["Low"].values,
                              "close": h["Close"].values,
                              "adjclose": (h["Adj Close"] if "Adj Close" in h else h["Close"]).values,
                              "volume": h["Volume"].values,
                              "dividends": h["Dividends"].values if "Dividends" in h else 0.0,
                              "splits": h["Stock Splits"].values if "Stock Splits" in h else 0.0})
            frames.append(f)
            rec.update(exists=True, rows=int(len(f)), first=str(f.date.min().date()), last=str(f.date.max().date()),
                       n_dividends=int((f.dividends > 0).sum()), n_splits=int((f.splits > 0).sum()))
        info_out[t] = rec
        print(f"  {t:10s} {role:12s} exists={rec['exists']!s:5s} rows={rec.get('rows', 0):5d} "
              f"{rec.get('first', '')}..{rec.get('last', '')}  {rec.get('longName')}  ccy={rec.get('currency')}  "
              f"TER={rec.get('netExpenseRatio')}", flush=True)
        time.sleep(0.3)

if frames:
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT_PX)
    print(f"saved {OUT_PX} rows={len(df)}")
json.dump({"fetched_utc": pd.Timestamp.utcnow().isoformat(), "yfinance": yf.__version__, "tickers": info_out},
          open(OUT_INFO, "w"), indent=1, default=str)
print(f"saved {OUT_INFO}")
