"""
NOV-2026 REFRESH — one shared, implementation-faithful daily panel (1985-10 .. latest).

Why this exists: an audit of the July-2026 research found the backtests and the
live scanner were NOT describing the same portfolio:
  * the backtest "gold" sleeve was ^XAU GOLD MINERS, but the scanner buys GLD (physical gold)
  * the backtest "bond" sleeve was a synthetic 10-year (IEF-like, duration 7.5), but the
    scanner buys TLT (20y+, duration ~17)
This panel carries BOTH the old proxies and faithful versions of every instrument the
scanner can actually hold, each spliced as  REAL ETF (when it exists) <- calibrated
synthetic (before inception), so every later test runs on one consistent dataset.

Output: update_2026_11/panel.parquet  +  update_2026_11/panel_validation.json
Columns (daily simple returns unless noted):
  ndx (level), ndx_ret, rf (annualised), rf_d
  qqq, qld, tqqq        spliced real<-sim Nasdaq sleeves (1x / 2x / 3x)
  qqq_sim, qld_sim, tqqq_sim, *_real
  bills, shy, ief, tlt  spliced real<-calibrated synthetic treasury sleeves
  bond10_old            the July synthetic 10y (duration 7.5) for continuity
  gold_phys             physical gold: GLD real <- GC=F <- monthly LBMA-type series (log-interp)
  gold_miners           ^XAU gold miners (the July proxy)
  spx, rut, intl, hsi, tsx   (from multi_panel)
  gsci, oil, dxy (levels) ; dxy_ret, gsci_ret
  dbmf, kmlm, cta       real managed-futures ETFs (NaN before inception)
  tnx, tyx, fvx, irx    yields in % ; curve = tnx - irx (pct points)
  vix
  ema100, above, vol20, er20, mark (weekly W-FRI rebalance flag), month
"""
import os, io, json, urllib.request
import numpy as np, pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
OUT = os.path.join(HERE, "panel.parquet")
VAL = os.path.join(HERE, "panel_validation.json")

base = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
idx = base.index
mp = pd.read_parquet("data_letf/multi_panel.parquet").reindex(idx)
geo = pd.read_parquet("data_letf/geo_panel.parquet").reindex(idx)
mac = pd.read_parquet("data_letf/macro_daily.parquet").reindex(idx)


def yclose(t, adjust=True):
    h = yf.Ticker(t).history(period="max", auto_adjust=adjust)["Close"].dropna()
    if len(h) == 0:
        return pd.Series(dtype=float)
    h.index = h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()]


def lvl(t):            # yields / indices: forward-filled level on NDX calendar
    return yclose(t, adjust=False).reindex(idx).ffill()


def real_ret(t):       # tradeable total return on NDX calendar, NaN before inception
    s = yclose(t)
    if len(s) == 0:
        return pd.Series(np.nan, index=idx)
    r = s.reindex(idx).ffill().pct_change()
    r[idx < s.index.min()] = np.nan
    r[idx == idx[idx >= s.index.min()][0]] = np.nan
    return r


print("fetching ...", flush=True)
P = pd.DataFrame(index=idx)
P["ndx"] = base["ndx"]
P["ndx_ret"] = base["ndx_ret"]
P["rf"] = base["rf"]
P["rf_d"] = (base["rf"] / 252).fillna(0)
rf_d = P["rf_d"]
EXP = 0.0095

# ---------------- Nasdaq sleeves ----------------
P["qqq_real"] = base["qqq_real"]; P["tqqq_real"] = base["tqqq_real"]
P["qld_real"] = real_ret("QLD")
P["qqq_sim"] = base["ndx_ret"] - 0.0020/252
P["qld_sim"] = 2*base["ndx_ret"] - EXP/252 - 1*rf_d
P["tqqq_sim"] = 3*base["ndx_ret"] - EXP/252 - 2*rf_d
for k in ["qqq", "qld", "tqqq"]:
    P[k] = P[f"{k}_real"].fillna(P[f"{k}_sim"]).fillna(0)

val = {}
def track(real, sim, name):
    m = pd.concat([real, sim], axis=1).dropna()
    m.columns = ["real", "sim"]
    if len(m) < 100: return
    val[name] = dict(days=int(len(m)), start=str(m.index[0].date()),
                     corr=round(float(m.real.corr(m.sim)), 4),
                     te_ann=round(float((m.real-m.sim).std()*np.sqrt(252)), 4),
                     cagr_real=round(float((1+m.real).prod()**(252/len(m))-1), 4),
                     cagr_sim=round(float((1+m.sim).prod()**(252/len(m))-1), 4))
    print(f"  VALIDATE {name:22s} corr {val[name]['corr']:.3f}  TE {val[name]['te_ann']*100:4.1f}%/yr  "
          f"CAGR real {val[name]['cagr_real']*100:5.2f}% vs sim {val[name]['cagr_sim']*100:5.2f}%  "
          f"({val[name]['start']}+)", flush=True)

for k in ["qqq", "qld", "tqqq"]:
    track(P[f"{k}_real"], P[f"{k}_sim"], f"{k} sim vs real")

# ---------------- yields ----------------
P["tnx"] = lvl("^TNX"); P["tyx"] = lvl("^TYX"); P["fvx"] = lvl("^FVX"); P["irx"] = lvl("^IRX")
P["tyx"] = P["tyx"].fillna(P["tnx"] + 0.25)          # pre-1977 safety (unused: panel starts 1985)
P["curve"] = P["tnx"] - P["irx"]

# ---------------- treasury sleeves: calibrated synthetic <- real ----------------
def calibrate(real, y, name):
    """fit  r = a + y_{t-1}/252 - D*dy + 0.5*C*dy^2  on the real ETF (dy in decimal)."""
    yd = y/100.0
    dy = yd.diff(); carry = yd.shift(1)/252
    df = pd.concat([real, carry, dy], axis=1).dropna(); df.columns = ["r", "carry", "dy"]
    Y = (df.r - df.carry).values
    X = np.column_stack([np.ones(len(df)), -df.dy.values, 0.5*df.dy.values**2])
    a, D, C = np.linalg.lstsq(X, Y, rcond=None)[0]
    syn = a + carry - D*dy + 0.5*C*dy**2
    val[f"{name}_calib"] = dict(alpha_ann=round(float(a*252), 5), duration=round(float(D), 2),
                                convexity=round(float(C), 1))
    print(f"  CALIB {name}: duration {D:5.2f}  convexity {C:7.1f}  alpha {a*252*100:+.2f}%/yr", flush=True)
    return syn

tlt_r, ief_r, shy_r, bil_r = real_ret("TLT"), real_ret("IEF"), real_ret("SHY"), real_ret("BIL")
P["tlt_real"], P["ief_real"], P["shy_real"], P["bil_real"] = tlt_r, ief_r, shy_r, bil_r
two_y = (P["irx"] + P["fvx"]) / 2                      # 2y-yield proxy (no long 2y series)
syn_tlt = calibrate(tlt_r, P["tyx"], "tlt")
syn_ief = calibrate(ief_r, P["tnx"], "ief")
syn_shy = calibrate(shy_r, two_y, "shy")
P["tlt_syn"], P["ief_syn"], P["shy_syn"] = syn_tlt, syn_ief, syn_shy
for k in ["tlt", "ief", "shy"]:
    track(P[f"{k}_real"], P[f"{k}_syn"], f"{k} syn vs real")
    P[k] = P[f"{k}_real"].fillna(P[f"{k}_syn"]).fillna(0)
P["bills"] = P["bil_real"].fillna(rf_d - 0.0013/252).fillna(0)   # BIL 0.135% fee
P["bond10_old"] = mp["bond"].fillna(0)                            # July proxy (dur 7.5)
track(P["ief_real"], P["bond10_old"], "July bond10 vs IEF")
track(P["tlt_real"], P["bond10_old"], "July bond10 vs TLT")

# ---------------- gold: physical (faithful) vs miners (July proxy) ----------------
url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}),
                             timeout=60).read().decode()
gm = pd.read_csv(io.StringIO(raw))
gm["Date"] = pd.to_datetime(gm["Date"]) + pd.offsets.MonthEnd(0)
gm = gm.set_index("Date")["Price"].astype(float)
# log-linear interpolation of month-end points onto the daily calendar (approximation:
# understates daily volatility; used ONLY before GC=F starts in Aug-2000)
grid = gm.reindex(gm.index.union(idx)).apply(np.log).interpolate(method="time").apply(np.exp)
gold_monthly_ret = grid.reindex(idx).pct_change()
gcf = yclose("GC=F", adjust=False)
gcf_ret = gcf.reindex(idx).ffill().pct_change(); gcf_ret[idx <= gcf.index.min()] = np.nan
gld_r = real_ret("GLD")
P["gld_real"] = gld_r
track(gld_r, gcf_ret - 0.0040/252, "GC=F vs GLD")
# monthly-interp vs GC=F on monthly resolution
mm = pd.concat([(1+gold_monthly_ret).resample("ME").prod()-1, (1+gcf_ret).resample("ME").prod()-1], axis=1).dropna()
mm.columns = ["interp", "gcf"]
mm = mm[mm.index >= "2000-10-31"]
val["gold_monthly_interp_vs_gcf_monthly"] = dict(corr=round(float(mm.interp.corr(mm.gcf)), 3),
    vol_interp=round(float(mm.interp.std()*np.sqrt(12)), 4), vol_gcf=round(float(mm.gcf.std()*np.sqrt(12)), 4))
print(f"  monthly-interp gold vs GC=F (monthly returns, 2000+): {val['gold_monthly_interp_vs_gcf_monthly']}", flush=True)
gold_phys = gld_r.copy()
gold_phys = gold_phys.fillna(gcf_ret - 0.0040/252)
gold_phys = gold_phys.fillna(gold_monthly_ret - 0.0040/252)
P["gold_phys"] = gold_phys.fillna(0)
P["gold_miners"] = mp["gold"].fillna(0)
P["gold_src"] = np.where(gld_r.notna(), "GLD", np.where(gcf_ret.notna(), "GC=F", "monthly-interp"))

# ---------------- other assets ----------------
for k in ["spx", "rut", "intl", "hsi", "tsx"]:
    P[k] = mp[k].fillna(0)
P["gsci"] = geo["gsci"]; P["oil"] = geo["oil"]; P["dxy"] = geo["dxy"]
P["gsci_ret"] = P["gsci"].pct_change(); P["dxy_ret"] = P["dxy"].pct_change()
P["vix"] = mac["vix"].ffill()
for t, k in [("DBMF", "dbmf"), ("KMLM", "kmlm"), ("CTA", "cta")]:
    P[k] = real_ret(t)

# ---------------- canonical signals (identical to engine.py / scanner.py) ----------------
ndx = P["ndx"]
P["ema100"] = ndx.ewm(span=100, adjust=False).mean()
P["above"] = ndx > P["ema100"]
P["vol20"] = ndx.pct_change().rolling(20).std()*np.sqrt(252)
P["er20"] = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum().replace(0, np.nan)
mark = pd.Series(False, index=idx)                 # identical to engine.py (Fridays only;
mark[ndx.resample("W-FRI").last().index.intersection(idx)] = True   # holiday-Friday weeks skip)
P["mark"] = mark
wk_last = pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max()
mark_wk = pd.Series(False, index=idx); mark_wk[pd.DatetimeIndex(wk_last.values)] = True
P["mark_wk"] = mark_wk                              # last trading day of every week (live-faithful)
P["month"] = idx.month

P = P.replace([np.inf, -np.inf], np.nan)
P.to_parquet(OUT)
val["rows"] = int(len(P)); val["start"] = str(idx[0].date()); val["end"] = str(idx[-1].date())
json.dump(val, open(VAL, "w"), indent=1)
print(f"\nsaved {OUT}  rows={len(P)}  {idx[0].date()}..{idx[-1].date()}  cols={P.shape[1]}")
print(f"saved {VAL}")
