"""
Two realistic overlays on the recommended signal:
  (A) Capital-constrained PORTFOLIO simulation -> real equity curve, DD, Sharpe.
  (B) OPTIONS simulation (ATM weekly calls via Black-Scholes) vs the stock trade.
Recommended config: Confluence_oversold entry + 2% hard stop, exit on bearish WT cross, 1h.
"""
import numpy as np, pandas as pd
from scipy.stats import norm
import bt

CACHE = bt.CACHE
sectors = pd.read_parquet("sectors.parquet").set_index("ticker")["sector"].to_dict()

TF = "1h"
BARS_PER_DAY = 7          # ~7 one-hour bars per US session
cache = bt.load_cache(TF)

# ---------- get recommended-strategy trades with timestamps ----------
tr = bt.run_strategy(cache, "Confluence_oversold", stop=0.02, cost_bps=2.0)
tr = tr.dropna(subset=["entry_time", "exit_time"]).copy()
tr["entry_time"] = pd.to_datetime(tr["entry_time"])
tr["exit_time"] = pd.to_datetime(tr["exit_time"])
tr = tr.sort_values("entry_time").reset_index(drop=True)
print(f"Recommended strategy trades: {len(tr)}  avg={tr.ret.mean()*1e4:.1f}bps  win={ (tr.ret>0).mean():.3f}")

# ================================================================= (A) PORTFOLIO
def portfolio_sim(trades, capital=100_000.0, max_pos=10, risk_frac=0.10):
    """Event-driven: each trade risks risk_frac of *current* equity; skip if no free slot."""
    events = []
    for idx, t in trades.iterrows():
        events.append((t.entry_time, "open", idx))
        events.append((t.exit_time, "close", idx))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))  # process closes first
    equity = capital
    open_slots = max_pos
    alloc = {}
    curve = []
    taken = 0
    for time, kind, idx in events:
        if kind == "close":
            if idx in alloc:
                equity += alloc.pop(idx) * trades.loc[idx, "ret"]
                open_slots += 1
        else:
            if open_slots > 0:
                stake = equity * risk_frac
                alloc[idx] = stake
                open_slots -= 1
                taken += 1
        curve.append((time, equity))
    eq = pd.DataFrame(curve, columns=["time", "equity"]).groupby("time").last()
    ret_total = equity / capital - 1
    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1).min()
    # daily sharpe
    daily = eq["equity"].resample("D").last().dropna()
    dret = daily.pct_change().dropna()
    sharpe = dret.mean() / dret.std() * np.sqrt(252) if dret.std() > 0 else np.nan
    return eq, dict(total_ret=ret_total, maxdd=dd, sharpe_ann=sharpe,
                    trades_taken=taken, final=equity)

for mp, rf in [(10, 0.10), (20, 0.05), (5, 0.20)]:
    eq, stats = portfolio_sim(tr, max_pos=mp, risk_frac=rf)
    print(f"portfolio max_pos={mp} risk={rf:.0%}: total={stats['total_ret']*100:+.1f}%  "
          f"maxDD={stats['maxdd']*100:.1f}%  Sharpe(ann)={stats['sharpe_ann']:.2f}  "
          f"taken={stats['trades_taken']}/{len(tr)}")

# save the base curve for the report
eq, stats = portfolio_sim(tr, max_pos=10, risk_frac=0.10)
eq.reset_index().to_csv(f"{CACHE}/equity_curve.csv", index=False)

# market benchmark: equal-weight buy&hold over same window
minlen = min(len(d["close"]) for d in cache.values())
mat = np.vstack([pd.Series(d["close"][:minlen]).pct_change().to_numpy() for d in cache.values()])
ew = np.nanmean(mat, axis=0)
bh = (np.nancumprod(1 + ew) - 1)[-1]
print(f"equal-weight buy&hold over window: {bh*100:+.2f}%")

# ================================================================= (B) OPTIONS
def bs_call(S, K, T, r, sig):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# realized annualized vol per ticker from 1h returns
ann = np.sqrt(BARS_PER_DAY * 252)
vol = {}
for tkr, d in cache.items():
    r = pd.Series(d["close"]).pct_change().to_numpy()
    s = np.nanstd(r) * ann
    vol[tkr] = min(max(s, 0.10), 2.5)  # clamp to sane 10%..250%

r_rate = 0.04
DTE_days = 5.0                    # weekly option
opt_rows = []
for _, t in tr.iterrows():
    if t["dir"] != 1:            # model long-call only on long trades
        continue
    S0 = t["entry_px"]; S1 = t["exit_px"]
    sig = vol.get(t["ticker"], 0.4)
    T0 = DTE_days / 252.0
    held_days = t["bars"] / BARS_PER_DAY
    T1 = max(T0 - held_days / 252.0, 0.0)
    K = S0                       # ATM
    v0 = bs_call(S0, K, T0, r_rate, sig)
    v1 = bs_call(S1, K, T1, r_rate, sig)
    if v0 <= 0.01:
        continue
    # 1% round-trip option friction (spreads are wide) applied on premium
    opt_ret = max((v1 * (1 - 0.01)) / (v0 * (1 + 0.01)) - 1.0, -1.0)
    opt_rows.append(dict(ticker=t["ticker"], stock_ret=t["ret"], opt_ret=opt_ret,
                         sig=sig, bars=t["bars"]))
opt = pd.DataFrame(opt_rows)
opt.to_csv(f"{CACHE}/options_sim.csv", index=False)
print("\n=== OPTIONS SIM (ATM weekly calls on long signals; BS, 1% premium friction) ===")
print(f"n={len(opt)}")
print(f"STOCK : avg={opt.stock_ret.mean()*100:+.3f}%  win={ (opt.stock_ret>0).mean():.3f}  "
      f"median={opt.stock_ret.median()*100:+.3f}%")
print(f"OPTION: avg={opt.opt_ret.mean()*100:+.2f}%  win={ (opt.opt_ret>0).mean():.3f}  "
      f"median={opt.opt_ret.median()*100:+.2f}%  "
      f"best={opt.opt_ret.max()*100:+.0f}%  worst={opt.opt_ret.min()*100:.0f}%")
# option expectancy is per-premium; account-level depends on sizing. Show profit factor.
w = opt.opt_ret[opt.opt_ret > 0].sum(); l = -opt.opt_ret[opt.opt_ret <= 0].sum()
print(f"OPTION profit factor={w/l:.2f}   % total return if equal $ premium each={opt.opt_ret.mean()*100:+.1f}% per trade")
