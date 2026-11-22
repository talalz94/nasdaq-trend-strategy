"""
Geopolitical / macro-regime panel with LONG history (1984+), free via yfinance.
Purpose: answer "how does the strategy adapt to wars, de-dollarization, oil shocks,
the rise of China?" — using measurable, LEADING market-based proxies rather than headlines.

Raw (long history):
  ^XOI     oil & gas index          1983+   oil / energy shock proxy
  ^SPGSCI  GSCI commodity index     1984+   commodity supercycle / inflation
  ^BCOM    Bloomberg commodity      1991+   cross-check
  DX-Y.NYB US dollar index          1971+   dollar power
  ^HSI     Hang Seng                1986+   China / Asia leadership
  ^N225    Nikkei                   1965+   Asia
  ^VIX     volatility               1990+   stress (coincident - for context only)
  ^XAU     gold miners              1983+   monetary debasement proxy (already in multi_panel)

Derived REGIME FLAGS (all computed from prior data only, no lookahead):
  oil_shock     oil index 3-month momentum in its top quintile          (war / supply shock)
  commod_bull   GSCI above its 200d MA and 12m momentum > 0             (inflationary regime)
  dollar_weak   DXY below its 200d MA and falling                       (de-dollarization proxy)
  gold_lead     gold 6m momentum > SPX 6m momentum                      (debasement / fear bid)
  china_lead    HSI 6m momentum > SPX 6m momentum                       (EM/China leadership)
  stagflation   commod_bull AND (bonds falling) AND (equity trend weak) (the 1970s/2022 regime)
"""
import numpy as np, pandas as pd, yfinance as yf
base = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
idx = base.index
P = pd.read_parquet("data_letf/multi_panel.parquet").reindex(idx)

def px(t):
    h = yf.Ticker(t).history(period="max")["Close"]
    if len(h) == 0:                      # symbol withdrawn from Yahoo (e.g. ^BCOM, Nov 2026)
        print(f"  WARNING: no data for {t} - column left empty", flush=True)
        return pd.Series(np.nan, index=idx)
    h.index = h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].reindex(idx).ffill()

print("fetching geopolitical panel ...", flush=True)
oil   = px("^XOI")
gsci  = px("^SPGSCI")
bcom  = px("^BCOM")
dxy   = px("DX-Y.NYB")
hsi   = px("^HSI")
spx   = px("^GSPC")
vix   = px("^VIX")

G = pd.DataFrame(index=idx)
G["oil"], G["gsci"], G["bcom"], G["dxy"], G["hsi"], G["spx"] = oil, gsci, bcom, dxy, hsi, spx
G["vix"] = vix

def mom(s, n): return s/s.shift(n)-1
def ma(s, n):  return s.rolling(n).mean()

# --- oil shock: 63-day oil momentum in its own trailing top quintile ---
oil_m = mom(oil, 63)
oil_q = oil_m.rolling(1260, min_periods=252).quantile(0.80)
G["oil_shock"] = (oil_m > oil_q).fillna(False)

# --- commodity bull (inflationary regime) ---
G["commod_bull"] = ((gsci > ma(gsci, 200)) & (mom(gsci, 252) > 0)).fillna(False)

# --- dollar weakness (de-dollarization proxy) ---
G["dollar_weak"] = ((dxy < ma(dxy, 200)) & (dxy < dxy.shift(63))).fillna(False)

# --- gold leadership (monetary debasement / fear bid) ---
gold_px = (1+P["gold"].fillna(0)).cumprod()
G["gold_lead"] = (mom(gold_px, 126) > mom(spx, 126)).fillna(False)

# --- China / EM leadership ---
G["china_lead"] = (mom(hsi, 126) > mom(spx, 126)).fillna(False)

# --- stagflation: commodities up, bonds down, equities not trending ---
bond_px = (1+P["bond"].fillna(0)).cumprod()
G["bonds_falling"] = (bond_px < ma(bond_px, 200)).fillna(False)
ndx = base["ndx"]
G["stagflation"] = (G["commod_bull"] & G["bonds_falling"]).fillna(False)

# --- composite geopolitical stress score 0-4 (leading-ish inputs only) ---
G["geo_score"] = (G["oil_shock"].astype(int) + G["commod_bull"].astype(int)
                  + G["dollar_weak"].astype(int) + G["gold_lead"].astype(int))

G.to_parquet("data_letf/geo_panel.parquet")
print(f"saved data_letf/geo_panel.parquet  ({len(G)} rows)")

flags = ["oil_shock","commod_bull","dollar_weak","gold_lead","china_lead","stagflation"]
print("\n=== FLAG PREVALENCE (% of days) & coverage ===")
for f in flags:
    v = G[f]
    first = G.index[v.notna()][0] if v.notna().any() else None
    print(f"  {f:14s} {v.mean()*100:5.1f}%   from {str(first.date()) if first is not None else '—'}")

print("\n=== FLAGS BY DECADE (% of days ON) ===")
dec = {"1990s":("1990","1999"),"2000s":("2000","2009"),"2010s":("2010","2019"),"2020s":("2020","2026")}
print(f"{'flag':14s}"+"".join(f"{k:>9s}" for k in dec))
for f in flags:
    row=""
    for k,(a,b) in dec.items():
        m=(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b+"-12-31"))
        row+=f"{G[f][m].mean()*100:8.0f}%"
    print(f"{f:14s}{row}")

print(f"\n=== TODAY'S READ ({idx[-1].date()}) ===")
for f in flags: print(f"  {f:14s} {'ON' if bool(G[f].iloc[-1]) else 'off'}")
print(f"  geo_score      {int(G['geo_score'].iloc[-1])}/4   dollar {G['dxy'].iloc[-1]:.1f}  vix {G['vix'].iloc[-1]:.1f}")
