"""
INTEGRITY REVIEW of r2_candidates_tranche - extra checks (read-only w.r.t. project files; prints only, plus one JSON in
the scratch dir given on the command line).
  1. Key-table rows of the review vs the saved JSON (every number in the 21 rows).
  2. Window 2010-01-01..FREEZE (the recommendation says 'if the next years look like 2010-2026, roughly match QQQ's
     return with about half its drawdown') - US basis vs QQQ.
  3. Depth of the 2022 drawdown for the 50/50 off-leg rows (claim C11 'both are the 2022 episode').
  4. Pre-2000 gold interpolation: sensitivity of IS88 metrics to restoring daily gold noise (within-month zero-sum
     noise, so every month-end value of the interpolated series is unchanged), 200 draws.
usage: python -u update_2026_11/r2_verify_candidates_tranche_integrity_extra.py <scratch_dir>
"""
import os, sys, json, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = sys.argv[1]
os.makedirs(SCR, exist_ok=True)
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
S = json.load(open(os.path.join(HERE, "r2_candidates_tranche.json")))
START, FREEZE = pd.Timestamp("1988-01-01"), pd.Timestamp("2026-07-24")
OUT = {}

# ------------------------------------------------------------------ 1. key table
TABLE = """F1 IEF|fri|11.88|-19.2|0.62|0.89|4.1|-13.0|-11.3|-17.4|3|10.3|11.9|0.54|0.62|-16.6|-21.7|+0.44|+0.09|+0.97|0.009|-4.4
F1 IEF|tranche|10.90|-20.8|0.52|0.81|4.4|-13.0|-12.0|-19.1|2|9.4|10.9|0.45|0.52|-17.6|-23.5|+0.35|-0.01|+0.91|0.027|-6.6
F1 IEF|us|10.62|-20.9|0.51|0.79|4.5|-13.0|-12.1|-19.1|2|9.1|10.6|0.44|0.51|-17.8|-23.7|+0.33|-0.02|+0.88|0.031|-6.7
F1 IEF|ucits|9.68|-21.5|0.45|0.69|4.8|-13.9|-12.9|-19.8|3|8.3|9.7|0.39|0.45|-18.7|-24.7|+0.27|-0.09|+0.73|0.055|-7.0
F1 5050|us|9.67|-14.6|0.66|0.74|4.1|-11.2|-9.6|-12.8|5|8.3|9.7|0.57|0.66|-14.1|-17.8|+0.48|+0.10|+0.86|0.006|-5.4
F1 5050|ucits|8.71|-15.1|0.58|0.64|4.5|-11.9|-10.3|-13.4|6|7.5|8.7|0.50|0.58|-15.1|-19.2|+0.40|+0.00|+0.71|0.023|-5.8
F2 IEF|fri|19.71|-24.0|0.82|0.87|8.8|-21.5|-19.2|-21.3|6|15.6|19.7|0.65|0.82|-25.6|-33.8|+0.64|+0.16|+0.88|0.003|-6.0
F2 IEF|tranche|17.29|-27.7|0.62|0.78|9.6|-22.6|-20.9|-25.3|6|13.4|17.3|0.48|0.62|-27.7|-36.4|+0.45|+0.04|+0.71|0.016|-10.7
F2 IEF|us|16.99|-27.8|0.61|0.76|9.7|-22.7|-21.0|-25.4|6|13.1|17.0|0.47|0.61|-27.8|-36.7|+0.43|+0.02|+0.69|0.019|-10.8
F2 IEF|ucits|15.86|-28.6|0.55|0.71|10.1|-23.3|-21.7|-26.3|8|12.4|15.9|0.43|0.55|-28.6|-38.2|+0.38|-0.03|+0.62|0.037|-11.2
F2 5050|us|15.94|-23.5|0.68|0.73|10.1|-21.2|-20.3|-20.4|8|12.2|15.9|0.52|0.70|-26.8|-35.2|+0.50|+0.03|+0.66|0.016|-9.6
F2 5050|ucits|14.80|-24.1|0.62|0.67|10.5|-22.0|-20.7|-21.2|11|11.5|14.8|0.48|0.63|-28.0|-36.8|+0.44|-0.03|+0.58|0.036|-10.0
F3 IEF|fri|24.13|-30.0|0.80|0.85|12.5|-27.8|-24.4|-24.2|7|19.7|24.3|0.72|0.88|-33.6|-43.0|+0.63|+0.13|+0.81|0.005|-9.0
F3 IEF|tranche|20.88|-32.5|0.64|0.76|13.6|-28.9|-26.5|-28.6|9|17.0|21.9|0.54|0.70|-35.3|-45.7|+0.46|+0.02|+0.66|0.020|-14.9
F3 IEF|us|20.58|-32.6|0.63|0.75|13.7|-29.0|-26.7|-28.7|9|16.7|21.6|0.53|0.68|-35.6|-46.0|+0.45|+0.01|+0.64|0.024|-15.0
F3 IEF|ucits|19.35|-32.9|0.59|0.70|14.3|-29.6|-27.4|-29.6|10|16.0|20.3|0.49|0.63|-36.9|-48.0|+0.41|-0.05|+0.58|0.041|-15.4
F3 5050|us|19.53|-30.7|0.64|0.72|14.2|-27.5|-25.3|-24.1|11|15.8|21.0|0.61|0.81|-35.4|-45.6|+0.46|-0.01|+0.62|0.028|-13.8
F3 5050|ucits|18.29|-31.4|0.58|0.67|14.7|-28.3|-25.9|-25.6|12|15.1|19.7|0.54|0.70|-36.8|-47.7|+0.40|-0.06|+0.55|0.047|-14.3"""
BEN = """QQQ B&H|14.78|-83.0|0.18|0.55|36.8|-40.5|-76.3|-69.7|7|8.6|16.1|0.10|0.30|-70.6|-87.6|-3.9
60/40 spx/ief|8.36|-32.8|0.26|0.53|6.6|-22.4|-20.5|-28.4|7|6.1|8.4|0.19|0.26|-29.7|-40.0|-1.1
85% IEF + 15% gold|5.87|-22.2|0.26|0.46|5.5|-12.3|-16.6|-16.7|7|4.3|5.9|0.20|0.26|-17.6|-25.9|-1.9"""


def dp(s):
    s = s.strip().replace("+", "")
    return len(s.split(".")[1]) if "." in s else 0


def close(txt, val, scale=1.0):
    """does the quoted text equal round(val*scale, dp(txt))? (half-unit tolerance + 1e-9)"""
    d = dp(txt)
    return abs(float(txt.replace("+", "")) - val * scale) <= 0.5 * 10 ** (-d) + 1e-9


bad = []
B252 = S["bootstrap"]["IS88"]["L252"]
for line in TABLE.splitlines():
    f = line.split("|")
    cfg = f[0].replace(" ", "-")
    basis = f[1]
    m = S["candidates"][cfg][basis]
    a = m["IS88"]
    ws = [m[w] for w in ("IS88", "IS00", "IS07")]
    own = B252["own"][f"{cfg}|{basis}"]
    pr_ = B252["pairs"][f"{cfg}|{basis} minus QQQ B&H"]["calmar"]
    checks = [(f[2], a["cagr"], 100), (f[3], a["maxdd"], 100), (f[4], a["calmar"], 1), (f[5], a["ex_sharpe"], 1),
              (f[6], a["ulcer"], 100), (f[7], a["top5dd_mean"], 100), (f[8], a["cdar95"], 100), (f[9], a["worst_12m"], 100),
              (f[10], a["n_neg_years"], 1), (f[11], min(w["cagr"] for w in ws), 100), (f[12], max(w["cagr"] for w in ws), 100),
              (f[13], min(w["calmar"] for w in ws), 1), (f[14], max(w["calmar"] for w in ws), 1),
              (f[15], own["maxdd"]["median"], 100), (f[16], own["maxdd"]["p5"], 100),
              (f[17], pr_["sample"], 1), (f[18], pr_["ci95"][0], 1), (f[19], pr_["ci95"][1], 1), (f[20], pr_["p_le0"], 1),
              (f[21], m["OOS"]["total"], 100)]
    for i, (t, v, sc) in enumerate(checks):
        if not close(t, v, sc):
            bad.append((cfg, basis, i, t, round(v * sc, 4)))
for line in BEN.splitlines():
    f = line.split("|")
    m = S["benchmarks"][f[0]]
    a = m["IS88"]
    ws = [m[w] for w in ("IS88", "IS00", "IS07")]
    own = B252["own"][f[0]]
    checks = [(f[1], a["cagr"], 100), (f[2], a["maxdd"], 100), (f[3], a["calmar"], 1), (f[4], a["ex_sharpe"], 1),
              (f[5], a["ulcer"], 100), (f[6], a["top5dd_mean"], 100), (f[7], a["cdar95"], 100), (f[8], a["worst_12m"], 100),
              (f[9], a["n_neg_years"], 1), (f[10], min(w["cagr"] for w in ws), 100), (f[11], max(w["cagr"] for w in ws), 100),
              (f[12], min(w["calmar"] for w in ws), 1), (f[13], max(w["calmar"] for w in ws), 1),
              (f[14], own["maxdd"]["median"], 100), (f[15], own["maxdd"]["p5"], 100), (f[16], m["OOS"]["total"], 100)]
    for i, (t, v, sc) in enumerate(checks):
        if not close(t, v, sc):
            bad.append((f[0], "bench", i, t, round(v * sc, 4)))
OUT["key_table_mismatches"] = bad
print("KEY TABLE mismatches (cfg, basis, field#, quoted, actual):", bad)

# ------------------------------------------------------------------ independent rebuild of US-basis series
ndx = P["ndx"]
ema = ndx.ewm(span=100, adjust=False).mean()
ABOVE = (ndx > ema).values
VOL = (ndx.pct_change().rolling(20).std() * np.sqrt(252)).values
ER = (ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum().replace(0, np.nan)).values
MON = P.index.month.values


def weights(lev, vt, off):
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.minimum(1.0, vt / VOL)
    sc = np.where(np.isnan(sc), 1.0, np.clip(sc, 0, 1))
    sc = np.where(np.isin(MON, [8, 9]), 0.5 * sc, sc)
    clean = np.where(np.isnan(ER), False, ER >= 0.25)
    eng = np.where(ABOVE, np.where(clean, sc, 0.5 * sc), 0.0)
    W = {lev: 0.85 * eng}
    rest = 0.85 - 0.85 * eng
    if off == "IEF":
        W["ief"] = rest
    else:
        W["ief"] = 0.5 * rest; W["bills"] = 0.5 * rest
    W["gold_phys"] = np.full(len(P), 0.15)
    return W


def tranche(W, lag, cost, gold=None):
    n = len(P)
    dow = P.index.dayofweek.values
    tot = np.zeros(n)
    for i in range(5):
        mpos = np.flatnonzero(dow == i)
        k = np.searchsorted(mpos, np.arange(n) - lag, side="right") - 1
        valid = k >= 0
        src = np.where(valid, mpos[np.maximum(k, 0)], 0)
        for c, w in W.items():
            h = np.where(valid, w[src], 0.0)
            rr = gold if (c == "gold_phys" and gold is not None) else P[c].fillna(0).values
            tot += h * rr
            dh = np.zeros(n); dh[1:] = np.abs(np.diff(h))
            tot -= dh * (cost[c] if isinstance(cost, dict) else cost)
    return pd.Series(tot / 5.0, P.index)


def st(r, a, b):
    s = r[(r.index >= max(pd.Timestamp(a), START)) & (r.index <= pd.Timestamp(b))]
    e = (1 + s).cumprod()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    dd = e / e.cummax() - 1
    ex = s - P["rf_d"].fillna(0).reindex(s.index)
    tr = dd.idxmin()
    return dict(cagr=round(float(cagr) * 100, 2), maxdd=round(float(dd.min()) * 100, 2), calmar=round(float(cagr / -dd.min()), 3),
                ulcer=round(float(np.sqrt((dd ** 2).mean())) * 100, 2),
                ex_sharpe=round(float(ex.mean() / ex.std() * np.sqrt(252)), 3),
                dd_peak=str(e[:tr].idxmax().date()), dd_trough=str(tr.date()))


TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
SER = {}
for t, (lev, vt) in TIERS.items():
    for off in ("IEF", "5050"):
        SER[f"{t}-{off}"] = tranche(weights(lev, vt, off), 1, 5e-4)
q = P["qqq"].fillna(0)

# ------------------------------------------------------------------ 2. 2010-2026 window
w10 = {k: st(v, "2010-01-01", FREEZE) for k, v in SER.items()}
w10["QQQ"] = st(q, "2010-01-01", FREEZE)
w07 = {k: st(v, "2007-01-01", FREEZE) for k, v in SER.items() if k.endswith("IEF")}
w07["QQQ"] = st(q, "2007-01-01", FREEZE)
OUT["window_2010_freeze_us"] = w10
OUT["window_2007_freeze_us"] = w07
print("2010-01-01..FREEZE (US basis):")
for k, v in w10.items():
    print(f"   {k:8s} {v}")
print("2007-01-01..FREEZE (US basis):")
for k, v in w07.items():
    print(f"   {k:8s} {v}")

# ------------------------------------------------------------------ 3. 2022 drawdown depth for 50/50 vs IEF
d22 = {}
for k, v in SER.items():
    s = v[(v.index >= "2021-01-01") & (v.index <= "2023-12-31")]
    e = (1 + s).cumprod()
    d22[k] = round(float((e / e.cummax() - 1).min()) * 100, 2)
OUT["dd_2021_2023_window"] = d22
print("maxDD inside 2021-2023 (US basis):", d22)
print("IS88 maxDD episode (US basis):", {k: (st(v, START, FREEZE)["maxdd"], st(v, START, FREEZE)["dd_peak"], st(v, START, FREEZE)["dd_trough"]) for k, v in SER.items()})

# ------------------------------------------------------------------ 4. gold interpolation sensitivity
g = P["gold_phys"].fillna(0).values
src = P["gold_src"].values
interp = (src == "monthly-interp")
real_sd = float(np.std(P["gold_phys"][(P.index >= "2004-11-19") & (P.index <= "2007-12-31")]))   # GLD daily sd, pre-GFC
months = pd.PeriodIndex(P.index, freq="M")
rng = np.random.default_rng(20261118)
base = {k: st(SER[k], START, FREEZE) for k in ["F1-IEF", "F2-IEF", "F3-IEF"]}
base["85/15"] = st(0.85 * P["ief"].fillna(0) + 0.15 * P["gold_phys"].fillna(0), START, FREEZE)
res = {k: [] for k in base}
lg = np.log1p(g)
mi = months[interp]
for it in range(200):
    z = rng.normal(0, real_sd, interp.sum())
    zs = pd.Series(z).groupby(np.asarray(mi.astype(str))).transform(lambda x: x - x.mean()).values  # month path unchanged
    lg2 = lg.copy()
    lg2[interp] = lg[interp] + zs
    g2 = np.expm1(lg2)
    for k in ["F1-IEF", "F2-IEF", "F3-IEF"]:
        t_, (lev, vt) = k.split("-")[0], TIERS[k.split("-")[0]]
        r = tranche(weights(lev, vt, "IEF"), 1, 5e-4, gold=g2)
        res[k].append(st(r, START, FREEZE))
    res["85/15"].append(st(pd.Series(0.85 * P["ief"].fillna(0).values + 0.15 * g2, P.index), START, FREEZE))
    if it == 49:
        print("   ...50 gold draws done", flush=True)
summ = {}
for k in base:
    arr = {m: np.array([x[m] for x in res[k]]) for m in ("cagr", "maxdd", "calmar", "ulcer", "ex_sharpe")}
    summ[k] = dict(base={m: base[k][m] for m in arr},
                   noisy_mean={m: round(float(arr[m].mean()), 3) for m in arr},
                   noisy_p5_p95={m: [round(float(np.percentile(arr[m], 5)), 3), round(float(np.percentile(arr[m], 95)), 3)] for m in arr})
OUT["gold_interp_sensitivity"] = dict(real_gld_daily_sd=real_sd, interp_daily_sd=float(np.std(g[interp & (P.index >= START)])),
                                      n_draws=200, summary=summ)
print("gold sensitivity:", json.dumps(OUT["gold_interp_sensitivity"], indent=0))
json.dump(OUT, open(os.path.join(SCR, "integrity_extra.json"), "w"), indent=1)
