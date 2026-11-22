"""
Macro dataset aligned to the NDX daily index (from letf_daily). Free via yfinance.
Signals (economically motivated, kept simple to resist overfitting):
  curve   = 10y - 3mo yield  (inversion < 0 precedes recessions/bears)
  dxy_up  = dollar index > its 200d MA (strong-$ = risk headwind)
  vix     = level (fear/stress; >~28 = stress)
  credit  = HYG/IEF ratio trend (risk appetite; 2007+ only)
Tradeable defensive returns saved too: tlt (long bonds), ief (7-10y), gold, hyg.
"""
import numpy as np, pandas as pd, yfinance as yf
base = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
idx = base.index

def lvl(t):    # level series (yields, vix, dxy) - no adjust
    h=yf.Ticker(t).history(period="max")["Close"]; h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).ffill()
def ret(t):    # tradeable total-return
    h=yf.Ticker(t).history(period="max",auto_adjust=True)["Close"]; h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).pct_change()

print("fetching macro ...", flush=True)
tnx, irx, dxy, vix = lvl("^TNX"), lvl("^IRX"), lvl("DX-Y.NYB"), lvl("^VIX")
tlt, ief, gold, hyg = ret("TLT"), ret("IEF"), ret("GC=F"), ret("HYG")
hyg_lvl = lvl("HYG"); ief_lvl = lvl("IEF")

m = pd.DataFrame(index=idx)
m["curve"] = tnx - irx                                  # 10y - 3mo (%)
m["dxy"] = dxy; m["dxy_up"] = (dxy > dxy.rolling(200).mean())
m["vix"] = vix
m["tnx"] = tnx; m["tnx_up"] = (tnx > tnx.rolling(200).mean())   # rising rates
m["credit"] = (hyg_lvl/ief_lvl); m["credit_up"] = m["credit"] > m["credit"].rolling(50).mean()
m["tlt_ret"]=tlt.fillna(0); m["ief_ret"]=ief.fillna(0); m["gold_ret"]=gold.fillna(0); m["hyg_ret"]=hyg.fillna(0)
m.to_parquet("data_letf/macro_daily.parquet")

def cov(s):
    v=m[s].dropna(); return f"{v.index.min().date()}..{v.index.max().date()} ({len(v)})"
print("saved data_letf/macro_daily.parquet")
for s in ["curve","dxy","vix","credit","tlt_ret","gold_ret"]:
    print(f"  {s:10s} {cov(s)}")
print(f"\ncurve inverted (<0) days: {(m['curve']<0).sum()} ; vix>28 days: {(m['vix']>28).sum()}")
print(f"curve now={m['curve'].iloc[-1]:.2f}  vix now={m['vix'].iloc[-1]:.1f}  dxy_up now={bool(m['dxy_up'].iloc[-1])}")
