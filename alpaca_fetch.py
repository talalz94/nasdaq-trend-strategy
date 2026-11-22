"""
Alpaca backfill -> this project's own data_alpaca/ (Stock Screener dir untouched).
SIP feed (full consolidated volume, history back to 2016-01-01), split-adjusted,
filtered to a clean 7-bar regular session (09:00-15:00 ET anchored).
Keys read from .env (never hardcoded/printed).
"""
import os, sys, time, argparse, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, pandas as pd

class RateLimiter:
    """Global cap of `rate` requests per `per` seconds, shared across threads."""
    def __init__(self, rate=170, per=60.0):
        self.rate, self.per = rate, per
        self.lock = threading.Lock(); self.times = collections.deque()
    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.times and now - self.times[0] > self.per:
                    self.times.popleft()
                if len(self.times) < self.rate:
                    self.times.append(now); return
                wait = self.per - (now - self.times[0])
            time.sleep(min(max(wait, 0.01), 1.0))

LIMITER = RateLimiter(rate=170)  # stay safely under Alpaca's 200/min free cap

KV = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))
H = {"APCA-API-KEY-ID": KV["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": KV["ALPACA_SECRET"]}
BASE = "https://data.alpaca.markets/v2/stocks/bars"
OUT = "data_alpaca"
os.makedirs(OUT, exist_ok=True)
RTH = set(range(9, 16))  # ET hours 9..15 -> 7 session bars/day

TF_MAP = {"1h": "1Hour", "15m": "15Min", "1d": "1Day"}


_local = threading.local()
def _sess():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s

def _get_page(params):
    """Fetch one page robustly. Retries 429/5xx/network with backoff; RAISES on hard
    failure so pagination is never silently truncated."""
    s = _sess()
    last = None
    for attempt in range(15):
        LIMITER.acquire()
        try:
            r = s.get(BASE, headers=H, params=params, timeout=45)
        except requests.RequestException as e:
            last = e; time.sleep(min(1.5 * (attempt + 1), 20)); continue
        if r.status_code == 429 or r.status_code >= 500:
            last = r; time.sleep(min(1.5 * (attempt + 1), 20)); continue
        r.raise_for_status()   # other 4xx (e.g. bad symbol) -> raise immediately
        return r.json()
    raise RuntimeError(f"page failed after retries: {last}")

def fetch_one(symbol, start, end, timeframe):
    """Paginate a single symbol's bars. Dash->dot for class shares (BRK.B/BF.B)."""
    asym = symbol.replace("-", ".")
    out, token = [], None
    while True:
        params = dict(symbols=asym, timeframe=timeframe, start=start,
                      limit=10000, feed="sip", adjustment="split", sort="asc")
        if end: params["end"] = end
        if token: params["page_token"] = token
        j = _get_page(params)
        out.extend((j.get("bars") or {}).get(asym, []))
        token = j.get("next_page_token")
        if not token:
            break
    return out, asym


def sym_frame(symbol, arr, timeframe):
    if not arr:
        return None
    df = pd.DataFrame(arr)
    df["datetime"] = pd.to_datetime(df["t"])
    if timeframe != "1Day":  # RTH filter on intraday
        et = df["datetime"].dt.tz_convert("America/New_York")
        df = df[et.dt.hour.isin(RTH)]
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["ticker"] = symbol
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date.astype(str)
    return df[["ticker", "datetime", "date", "open", "high", "low", "close", "volume"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2016-01-01T00:00:00Z")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #tickers (0=all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    universe = pd.read_parquet("sectors.parquet")["ticker"].tolist()
    if a.limit:
        universe = universe[:a.limit]
    tf = TF_MAP[a.interval]
    frames = []
    t0 = time.time()

    def run_pass(tickers):
        done, failed = 0, []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(fetch_one, s, a.start, a.end, tf): s for s in tickers}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    arr, _ = fut.result()
                    fr = sym_frame(s, arr, tf)
                    if fr is not None and not fr.empty:
                        frames.append(fr)
                except Exception as e:
                    failed.append(s); print(f"  !! {s}: {repr(e)[:80]}", flush=True)
                done += 1
                if done % 25 == 0 or done == len(tickers):
                    print(f"  {done}/{len(tickers)} tickers, {sum(len(f) for f in frames):,} rows, {time.time()-t0:.0f}s", flush=True)
        return failed

    failed = run_pass(universe)
    if failed:
        print(f"retrying {len(failed)} failed: {failed}", flush=True)
        failed = run_pass(failed)
        if failed: print(f"STILL FAILED (no data): {failed}", flush=True)

    full = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "datetime"]) \
             .sort_values(["ticker", "datetime"]).reset_index(drop=True)
    out = a.out or f"{OUT}/stocks_{a.interval}.parquet"
    full.to_parquet(out)
    # coverage sanity
    span = full.groupby("ticker")["datetime"].agg(["min", "max", "size"])
    complete = (pd.to_datetime(span["max"]) >= pd.Timestamp("2026-07-01")).sum()
    print(f"SAVED {out}: {full['ticker'].nunique()} tickers, {len(full)} rows, "
          f"{full['datetime'].min()} -> {full['datetime'].max()}, {time.time()-t0:.0f}s")
    print(f"  tickers reaching 2026-07: {complete}  (should be ~all)")


if __name__ == "__main__":
    main()
