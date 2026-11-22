"""
Accurate market-regime definition from an equal-weight market proxy built off the
universe itself (no extra data). Daily index -> trend (200/50 SMA) + realized vol.

  BULL   : index > SMA200 AND SMA200 rising AND vol not elevated
  BEAR   : index < SMA200 AND SMA200 falling
  CHOPPY : everything else (sideways / transitional / high-volatility stress)

Returns a daily Series date->regime; use `tag_trades` to label each trade by entry date.
"""
import numpy as np, pandas as pd


def market_index(cache) -> pd.DataFrame:
    daily = {}
    for t, d in cache.items():
        s = pd.Series(d["close"], index=pd.to_datetime(d["datetime"]))
        dd = s.resample("1D").last().dropna()
        if len(dd) > 50:
            daily[t] = dd.pct_change()
    ret = pd.DataFrame(daily).mean(axis=1).dropna()      # equal-weight daily return
    idx = (1 + ret).cumprod()
    return pd.DataFrame({"ret": ret, "idx": idx})


def classify(mi: pd.DataFrame) -> pd.Series:
    idx = mi["idx"]
    sma200 = idx.rolling(200, min_periods=100).mean()
    vol = mi["ret"].rolling(20).std() * np.sqrt(252)
    volmed = vol.median()
    rising = sma200 > sma200.shift(20)
    reg = pd.Series("CHOPPY", index=idx.index)
    reg[(idx > sma200) & rising] = "BULL"
    reg[(idx < sma200) & (~rising)] = "BEAR"
    reg[vol > volmed * 1.6] = "CHOPPY"                   # stress overrides
    reg[sma200.isna()] = "CHOPPY"
    return reg


def regime_series(cache) -> pd.Series:
    return classify(market_index(cache))


def tag_trades(trades: pd.DataFrame, reg: pd.Series) -> pd.Series:
    d = pd.to_datetime(trades["entry_time"]).dt.normalize()
    rmap = reg.copy(); rmap.index = pd.to_datetime(rmap.index).normalize()
    rmap = rmap[~rmap.index.duplicated()]
    return d.map(rmap).fillna("CHOPPY")


if __name__ == "__main__":
    import bt, pandas as pd
    df = pd.read_parquet(f"{bt.CACHE}/ind_alpaca_1h.parquet")
    cache = {t: {c: g[c].to_numpy() for c in ["close", "datetime"]}
             for t, g in df.groupby("ticker", sort=False)}
    reg = regime_series(cache)
    print("regime day counts:\n", reg.value_counts())
    print("\nby year:\n", reg.groupby(reg.index.year).value_counts().unstack().fillna(0).astype(int))
