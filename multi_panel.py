"""
PHASE C — multi-asset panel with LONG history (1985-2026), free via yfinance.

Purpose: the strategy is Nasdaq-or-cash, so it cannot grow when Nasdaq goes nowhere
(2000-2010) . This builds the opportunity set needed to rotate into whatever IS
trending, using only assets with history back to the mid-80s.

Assets (daily total-ish returns):
  ndx    Nasdaq-100          ^NDX    1985
  spx    S&P 500             ^GSPC   1927
  rut    Russell 2000        ^RUT    1987
  gold   Gold miners index   ^XAU    1983   (long-history gold proxy)
  eafe   Nikkei/FTSE/DAX avg (intl)   1985
  hsi    Hang Seng           ^HSI    1986
  tsx    Canada (commodity)  ^GSPTSE 1979
  bond   synthetic 10y UST total return from ^TNX  (validated vs IEF)
  cash   3-month T-bill      ^IRX
Also carries tqqq/sqqq/qqq returns from the existing letf dataset.
"""
import numpy as np, pandas as pd, yfinance as yf
import bt
CACHE=bt.CACHE
base=pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
idx=base.index

def px(t):
    h=yf.Ticker(t).history(period="max")["Close"]
    h.index=h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).ffill()

print("fetching multi-asset panel ...", flush=True)
raw={k:px(t) for k,t in [("ndx","^NDX"),("spx","^GSPC"),("rut","^RUT"),("gold","^XAU"),
                         ("n225","^N225"),("ftse","^FTSE"),("dax","^GDAXI"),("hsi","^HSI"),
                         ("tsx","^GSPTSE")]}
tnx=px("^TNX")/100.0; irx=px("^IRX")/100.0

P=pd.DataFrame(index=idx)
for k,s in raw.items(): P[k]=s
# international composite (equal weight of what exists each day)
intl=pd.concat([raw["n225"].pct_change(),raw["ftse"].pct_change(),raw["dax"].pct_change()],axis=1).mean(axis=1)

R=pd.DataFrame(index=idx)
for k in ["ndx","spx","rut","gold","hsi","tsx"]: R[k]=P[k].pct_change()
R["intl"]=intl
# synthetic 10y Treasury total return: carry + duration*(-dy)
DUR=7.5
dy=tnx.diff()
R["bond"]=(tnx.shift(1)/252.0)-DUR*dy
R["cash"]=irx/252.0
# leveraged sleeves from the existing dataset
for k in ["tqqq_ret","qqq_ret","sqqq_ret"]: R[k.replace("_ret","")]=base[k]
R=R.replace([np.inf,-np.inf],np.nan)

# --- validate synthetic bond vs real IEF (2002+) ---
ief=yf.Ticker("IEF").history(period="max",auto_adjust=True)["Close"]
ief.index=ief.index.tz_localize(None).normalize()
ief_r=ief[~ief.index.duplicated()].reindex(idx).pct_change()
ov=pd.concat([R["bond"],ief_r],axis=1).dropna()
ov.columns=["syn","real"]
print(f"  synthetic bond vs IEF: corr {ov['syn'].corr(ov['real']):.2f} | "
      f"ann vol syn {ov['syn'].std()*np.sqrt(252)*100:.1f}% vs real {ov['real'].std()*np.sqrt(252)*100:.1f}% | "
      f"cum syn {((1+ov['syn']).prod()-1)*100:.0f}% vs real {((1+ov['real']).prod()-1)*100:.0f}%")

R.to_parquet("data_letf/multi_panel.parquet")
print(f"saved data_letf/multi_panel.parquet  {R.shape[0]} rows x {R.shape[1]} assets")

# --- what actually worked during the flat windows? (the research question) ---
print("\n=== ANNUALIZED RETURN BY ASSET, IN EACH WINDOW ===")
wins={"1992-1995":("1992-01-01","1995-12-31"),"2000-2010":("2000-01-01","2010-12-31"),
      "full 1987-2026":("1987-01-01","2026-07-24")}
cols=["ndx","spx","rut","gold","intl","hsi","tsx","bond","cash"]
print(f"{'window':16s}"+"".join(f"{c:>8s}" for c in cols))
for lbl,(a,b) in wins.items():
    m=(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b))
    row=""
    for c in cols:
        r=R[c][m].dropna()
        if len(r)<100: row+=f"{'—':>8s}"; continue
        ann=(1+r).prod()**(252/len(r))-1
        row+=f"{ann*100:7.1f}%"
    print(f"{lbl:16s}{row}")
print("\n-> assets with strong trends during Nasdaq's flat decade are the diversification candidates")
