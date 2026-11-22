"""
Market data. No credentials: everything comes from Yahoo Finance via yfinance (as scanner.py does).

FrameMarketData serves prices from in-memory frames (tests, historical replay); YahooMarketData
fills those frames lazily from Yahoo. Every accessor takes an explicit date, so a replay can never
read a price from the future of the day it is processing.

Price conventions (Yahoo, auto_adjust=False):
  Close        official close, adjusted for splits but NOT for dividends -> fills and NAV
  Adj Close    total-return close (dividends reinvested)                  -> the shadow tranche
  Dividends    cash per share on the ex-date                              -> credited to the account
  Stock Splits split ratio on the effective date (informational: the account detects splits from the
               change in Yahoo's back-adjusted closes, which also catches splits Yahoo never records)
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

CURRENCY = {"LQQ.PA": "EUR"}           # every other instrument used here is quoted in USD
FX_TICKER = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X"}

# Known breaks in Yahoo's history that Yahoo never back-adjusted: (first date on the new basis, factor);
# earlier prices are divided by the factor. LQQ.PA: Amundi's 200:1 unit change shows as a -99.5% day
# on 2015-01-02 (update_2026_11/ucits.json cleaning log, the same fix as the research).
KNOWN_BREAKS = {"LQQ.PA": [("2015-01-02", 200.0)]}
PRICE_COLS = ("Open", "High", "Low", "Close", "Adj Close", "Dividends")


class DataError(RuntimeError):
    pass


def _ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _clean(h: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    h = h.copy()
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h.index = idx.normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    for c in ("Dividends", "Stock Splits"):
        if c not in h.columns:
            h[c] = 0.0
    if "Adj Close" not in h.columns:
        h["Adj Close"] = h["Close"]
    for first_new, factor in KNOWN_BREAKS.get(symbol, []):
        old = h.index < pd.Timestamp(first_new)
        for c in PRICE_COLS:
            if c in h.columns:
                h.loc[old, c] = h.loc[old, c] / factor
    return h


class FrameMarketData:
    def __init__(self, frames: dict | None = None, ndx: pd.Series | None = None):
        self._frames = {k: _clean(v, k) for k, v in (frames or {}).items()}
        self._ndx = ndx

    # ------------------------------------------------------------- loading hooks
    def _frame(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._frames:
            raise DataError(f"no data for {symbol}")
        return self._frames[symbol]

    def _ndx_series(self) -> pd.Series:
        if self._ndx is None:
            raise DataError("no ^NDX data")
        return self._ndx

    # ------------------------------------------------------------- accessors
    def ndx_closes(self, upto=None) -> pd.Series:
        s = self._ndx_series()
        if upto is not None:
            s = s[s.index <= _ts(upto)]
        return s

    def has_bar(self, symbol, d) -> bool:
        h = self._frame(symbol)
        return _ts(d) in h.index and pd.notna(h.at[_ts(d), "Close"])

    def close_on(self, symbol, d):
        h = self._frame(symbol)
        ts = _ts(d)
        if ts not in h.index or pd.isna(h.at[ts, "Close"]):
            return None
        return float(h.at[ts, "Close"])

    def last_close_before(self, symbol, d, max_days=10):
        """(date, close) of the last bar strictly before d within max_days, else None."""
        h = self._frame(symbol)["Close"].dropna()
        h = h[(h.index < _ts(d)) & (h.index >= _ts(d) - pd.Timedelta(days=max_days))]
        return (h.index[-1].date(), float(h.iloc[-1])) if len(h) else None

    def bar_dates(self, symbol) -> set:
        h = self._frame(symbol)["Close"].dropna()
        return set(h.index)

    def dividends_between(self, symbol, after, upto) -> list:
        """Cash dividends with ex-date in (after, upto]."""
        h = self._frame(symbol)
        sel = h[(h.index > _ts(after)) & (h.index <= _ts(upto)) & (h["Dividends"] > 0)]
        return [(t.date(), float(v)) for t, v in sel["Dividends"].items()]

    def splits_between(self, symbol, after, upto) -> list:
        """Recorded splits (date, ratio) with effective date in (after, upto]."""
        h = self._frame(symbol)
        sp = h["Stock Splits"].fillna(0.0)
        sel = sp[(h.index > _ts(after)) & (h.index <= _ts(upto)) & (sp > 0) & (sp != 1.0)]
        return [(t.date(), float(v)) for t, v in sel.items()]

    def split_ratio_on(self, symbol, d) -> float:
        h = self._frame(symbol)
        ts = _ts(d)
        if ts in h.index:
            v = float(h.at[ts, "Stock Splits"] or 0.0)
            if v > 0 and v != 1.0:
                return v
        return 1.0

    def fx_to_usd(self, currency, d):
        if currency == "USD":
            return 1.0
        h = self._frame(FX_TICKER[currency])["Close"].dropna()
        h = h[h.index <= _ts(d)]
        return float(h.iloc[-1]) if len(h) else None

    def adj_closes(self, symbols, start, end) -> pd.DataFrame:
        cols = {}
        for s in symbols:
            h = self._frame(s)
            cols[s] = h.loc[(h.index >= _ts(start)) & (h.index <= _ts(end)), "Adj Close"]
        return pd.DataFrame(cols)

    def last_bar_date(self, symbol):
        h = self._frame(symbol)["Close"].dropna()
        return h.index[-1].date() if len(h) else None


class YahooMarketData(FrameMarketData):
    """Lazily downloads each symbol once per run (no credentials). `start` (a date) downloads from that
    date - the live engine uses the account start minus the shadow warm-up, so long-running accounts keep
    their whole history; otherwise `period`. ^NDX defaults to its full history ('max')."""

    def __init__(self, period: str = "2y", ndx_period: str = "max", start=None, retries: int = 3):
        super().__init__()
        self.period, self.ndx_period, self.start, self.retries = period, ndx_period, start, retries

    def _download(self, symbol):
        import yfinance as yf
        last = None
        for attempt in range(self.retries):
            try:
                t = yf.Ticker(symbol)
                if self.start is not None:
                    h = t.history(start=str(self.start), auto_adjust=False, actions=True)
                else:
                    h = t.history(period=self.period, auto_adjust=False, actions=True)
                if h is not None and len(h):
                    return h
            except Exception as e:      # network / Yahoo hiccup: retry
                last = e
            time.sleep(2 * (attempt + 1))
        raise DataError(f"Yahoo returned no data for {symbol}" + (f": {last}" if last else ""))

    def _frame(self, symbol):
        if symbol not in self._frames:
            self._frames[symbol] = _clean(self._download(symbol), symbol)
        return self._frames[symbol]

    def _ndx_series(self):
        if self._ndx is None:
            import scanner                  # the scanner's own fetch, for exact parity
            last = None
            for attempt in range(self.retries):
                try:
                    s = scanner.fetch("^NDX", period=self.ndx_period)
                    if len(s):
                        self._ndx = s
                        break
                except Exception as e:
                    last = e
                time.sleep(2 * (attempt + 1))
            if self._ndx is None:
                raise DataError(f"Yahoo returned no ^NDX data" + (f": {last}" if last else ""))
        return self._ndx


def today_et(now=None) -> dt.date:
    return (now or now_et()).date()


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York")
