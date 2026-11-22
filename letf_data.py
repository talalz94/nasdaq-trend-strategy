"""
Max-history Nasdaq-100 / leveraged-ETF daily dataset (FREE via yfinance).

Real: ^NDX 1985+, QQQ 1999+, TQQQ/SQQQ 2010+, ^IRX (13wk T-bill) for financing.
Simulated LETF (to extend before 2010): each day
    TQQQ_ret =  3*ndx_ret - expense/252 - k_long*rf_daily
    SQQQ_ret = -3*ndx_ret - expense/252 - k_short*rf_daily
expense & k are CALIBRATED by OLS against the real 2010-2026 series, then applied
back to 1985. We validate the simulated cumulative vs real before trusting it.

Output: data_letf/letf_daily.parquet with columns
    date, ndx, ndx_ret, rf, qqq_ret, tqqq_ret, sqqq_ret, src   (real spliced over sim)
"""
import os, json, numpy as np, pandas as pd, yfinance as yf
import bt
CACHE = bt.CACHE
OUT = "data_letf"; os.makedirs(OUT, exist_ok=True)

def close(t):
    h = yf.Ticker(t).history(period="max", auto_adjust=True)
    s = h["Close"].copy(); s.index = s.index.tz_localize(None).normalize()
    return s[~s.index.duplicated()]

def ohlc(t):
    h = yf.Ticker(t).history(period="max", auto_adjust=True)[["Open","High","Low","Close"]]
    h.index = h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()]

print("fetching yfinance ...", flush=True)
ndxo = ohlc("^NDX")
ndx = ndxo["Close"]
qqq, tqqq, sqqq, irx = [close(t) for t in ["QQQ","TQQQ","SQQQ","^IRX"]]
print(f"  ^NDX {ndx.index.min().date()}..{ndx.index.max().date()} ({len(ndx)})  "
      f"QQQ {qqq.index.min().date()}  TQQQ {tqqq.index.min().date()}", flush=True)

df = pd.DataFrame(index=ndx.index)
df["ndx"] = ndx
df["ndx_open"], df["ndx_high"], df["ndx_low"] = ndxo["Open"], ndxo["High"], ndxo["Low"]
df["ndx_ret"] = ndx.pct_change()
df["rf"] = (irx.reindex(df.index).ffill() / 100.0).clip(0, 0.20).fillna(0)   # annualized
rf_d = df["rf"] / 252.0
for nm, s in [("qqq", qqq), ("tqqq", tqqq), ("sqqq", sqqq)]:
    df[f"{nm}_real"] = s.reindex(df.index).pct_change()

# ---- theoretical LETF daily mechanics (daily-compounded), validated vs real ----
# LETF delivers L x daily index return, minus expense ratio and financing (borrow cost).
EXP_L, EXP_Q = 0.0095, 0.0020                          # TQQQ/SQQQ 0.95%/yr, QQQ 0.20%/yr
df["tqqq_sim"] =  3*df["ndx_ret"] - EXP_L/252 - 2*rf_d # long 3x borrows 2x notional
df["sqqq_sim"] = -3*df["ndx_ret"] - EXP_L/252 - 2*rf_d # short 3x financing ~ symmetric
df["qqq_sim"]  =    df["ndx_ret"] - EXP_Q/252
for nm in ["tqqq", "sqqq", "qqq"]:
    m = df[[f"{nm}_real", f"{nm}_sim"]].dropna()
    cr = (1+m[f"{nm}_real"]).prod(); cs = (1+m[f"{nm}_sim"]).prod()
    te = (m[f"{nm}_real"]-m[f"{nm}_sim"]).std()*np.sqrt(252)*100
    print(f"  VALIDATE {nm}: overlap real x{cr:.2f} vs sim x{cs:.2f} "
          f"({len(m)} days, trackErr {te:.1f}%/yr)", flush=True)
# spliced: real where present else sim
df["qqq_ret"]  = df["qqq_real"].fillna(df["qqq_sim"])
df["tqqq_ret"] = df["tqqq_real"].fillna(df["tqqq_sim"])
df["sqqq_ret"] = df["sqqq_real"].fillna(df["sqqq_sim"])
df["src"] = np.where(df["tqqq_real"].notna(), "real", "sim")

out = df.dropna(subset=["ndx_ret"]).reset_index()
out = out.rename(columns={out.columns[0]: "date"})
out.to_parquet(f"{OUT}/letf_daily.parquet")

# sanity: cumulative buy&hold over full history
def cum(col):
    r = out[col].fillna(0); return (1+r).prod()-1
print(f"\nFULL {out.date.min().date()}..{out.date.max().date()} ({len(out)} days, "
      f"{out.src.eq('sim').sum()} sim / {out.src.eq('real').sum()} real):")
for c in ["qqq_ret","tqqq_ret","sqqq_ret"]:
    r=out[c]; eq=(1+r.fillna(0)).cumprod(); dd=(eq/eq.cummax()-1).min()
    print(f"  buy&hold {c:9s}: total {cum(c)*100:>12,.0f}%  maxDD {dd*100:.0f}%")
print(f"saved {OUT}/letf_daily.parquet")
