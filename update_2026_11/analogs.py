"""
analogs.py -- CURRENT-REGIME ANALOGS AND FORWARD BASE RATES  (Nov-2026 refresh)

Question: the strategy is RISK-OFF since the 2026-10-16 Friday mark.  When NDX was in a
similar state in 1988 .. 2026-07-24 (the in-sample period), what happened next?
CONDITIONAL BASE RATES ONLY -- nothing here is a forecast.

State of the last panel day (out-of-sample, after the 2026-07-24 freeze) is the QUERY only;
no strategy parameter is chosen anywhere in this file.

Analog sets (every set is counted in N_VARIANTS):
  rule-based (chronological first hit, >= 60 trading days apart)
    R1   brief spec : DD from ATH in [-20%,-8%], below-EMA cluster age 10-40d, vol20 > 25%
    R1a  R1 but the date in each episode whose cluster age is closest to today's (aligned)
    R2   R1 with DD measured from the 52-week high (the 2000 ATH was not regained until
         2015-17, so ATH-based DD silently drops every 2001-2016 episode)
    R3   R2 + 10y yield up > 25bp over 126d (rates-rising sell-offs, like now)
    R4   tighter band around today: DD52 [-17%,-10%], age 15-35d, vol20 > 28%
  nearest neighbours (standardised features, greedy de-clustering: one date per +-60 days, k=10)
    K1   price state  : dd52, log(1+age), log vol20, er20
    K2   K1 + macro   : 10y level, 10y 126d change, curve, dxy 126d change, log VIX (1990+)
    K3   K2 + gold 63d return (only dates >= 2000-12 where gold_phys is daily-real)
  every analog date must also be: in-sample, NDX below EMA-100, and the weekly strategy risk-off.

Per episode: forward 1/3/6/12m returns of NDX, QQQ, F2-IEF, F3-IEF (+ IEF/TLT/GLD/BIL legs),
further NDX decline, deepest DD from the 52w peak within 12m (>25% = bear), days until the
daily close is back above EMA-100 and until the weekly strategy re-enters, whipsaws in 6m.

Null test: is an analog set's median different from N random (de-clustered) risk-off dates?
Block bootstrap (63-day blocks, 12m = 4 blocks) conditioned on starting risk-off.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OUT = os.path.join(HERE, "analogs.json")
P = C.panel()
idx = P.index
N = len(idx)
FREEZE, START = C.FREEZE, C.START
POS_FREEZE = int(idx.searchsorted(FREEZE, side="right") - 1)   # last in-sample position
POS_START = int(idx.searchsorted(START))
LAST = N - 1
HZ = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
RADIUS = 60            # de-clustering: at most one analog per 60 trading days
KNN_K = 10
GAP = 5                # below-EMA spells separated by <= 5 above-EMA days are one cluster
SEED = 20261118
N_VARIANTS = 0         # incremented for every analog set / bootstrap scheme evaluated


def nz(x):
    """json-safe float."""
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return x
    return None if (math.isnan(xf) or math.isinf(xf)) else xf


# ------------------------------------------------------------------ strategies (common.py)
wF2 = C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
wF3 = C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_phys", gold_w=0.15, off="ief")
RET = pd.DataFrame({
    "F2": C.backtest(P, wF2), "F3": C.backtest(P, wF3),
    "QQQ": P["qqq"].fillna(0), "NDX": P["ndx_ret"].fillna(0),
    "IEF": P["ief"].fillna(0), "TLT": P["tlt"].fillna(0),
    "GLD": P["gold_phys"].fillna(0), "BIL": P["bills"].fillna(0)}, index=idx)
LOGW = {c: np.cumsum(np.log1p(RET[c].values)) for c in RET.columns}   # log wealth at close t
STRATS = ["NDX", "QQQ", "F2", "F3"]
LEGS = ["IEF", "TLT", "GLD", "BIL"]

# weekly strategy state at the close of t (the weights held from t+1); identical for F2/F3
tgt = wF3.where(P["mark"]).ffill().fillna(0.0)
ON = (tgt["tqqq"] > 0).values
ABOVE = P["above"].values.astype(bool)
NDX = P["ndx"].values

# sanity: the state series reproduces the held weights of common.held_weights
_hw = C.held_weights(P, wF3)["tqqq"].values
assert np.array_equal(_hw[1:] > 0, ON[:-1]), "state series inconsistent with held weights"


# ------------------------------------------------------------------ state features (causal)
def cluster_age(above, gap=GAP):
    """trading days since the start of the current below-EMA cluster (0 if above today).
    spells separated by <= gap above-EMA days are merged; uses past data only."""
    age = np.zeros(len(above), dtype=int)
    start, last_below = -1, -10**9
    for t in range(len(above)):
        if not above[t]:
            if start < 0 or (t - last_below - 1) > gap:
                start = t
            last_below = t
            age[t] = t - start + 1
    return age


def run_length(flag):
    out = np.zeros(len(flag), dtype=int)
    c = 0
    for t in range(len(flag)):
        c = c + 1 if flag[t] else 0
        out[t] = c
    return out


ndx_s = P["ndx"]
F = pd.DataFrame(index=idx)
F["dd_ath"] = ndx_s / ndx_s.cummax() - 1
H52 = ndx_s.rolling(252, min_periods=1).max()
F["dd52"] = ndx_s / H52 - 1
F["age"] = cluster_age(ABOVE)                    # below-EMA cluster age (brief: "days below")
F["run"] = run_length(~ABOVE)                    # continuous below run (resets on any re-cross)
F["off_age"] = run_length(~ON)                   # days since the weekly strategy went risk-off
F["vol20"] = P["vol20"]
F["er20"] = P["er20"]
tnx = P["tnx"].ffill()
F["tnx"] = tnx
F["tnx_chg126"] = tnx - tnx.shift(126)           # percentage points
F["curve"] = (P["tnx"] - P["irx"]).ffill()
dxy = P["dxy"].ffill()
F["dxy_chg126"] = dxy / dxy.shift(126) - 1
F["vix"] = P["vix"].ffill()
gw = (1 + P["gold_phys"].fillna(0)).cumprod()
interp = (P["gold_src"] == "monthly-interp").astype(float)
gold63 = gw / gw.shift(63) - 1
gold63[interp.rolling(63, min_periods=1).max() > 0] = np.nan   # never a signal from interpolated gold
F["gold63"] = gold63
iw = (1 + P["ief"].fillna(0)).cumprod()
F["ief63"] = iw / iw.shift(63) - 1               # descriptive only
F["corr63_ndx_ief"] = P["ndx_ret"].rolling(63).corr(P["ief"])   # descriptive only
F["ndx63"] = ndx_s / ndx_s.shift(63) - 1          # descriptive only

T0 = LAST
today = {k: nz(F[k].iloc[T0]) for k in F.columns}
today["date"] = str(idx[T0].date())
today["ndx"] = nz(NDX[T0])
today["ath"] = nz(ndx_s.cummax().iloc[T0])
today["ath_date"] = str(ndx_s.iloc[:T0 + 1].idxmax().date())
today["ema100"] = nz(P["ema100"].iloc[T0])
today["above_ema100"] = bool(ABOVE[T0])
today["strategy_risk_on"] = bool(ON[T0])
_cs = T0 - int(F["age"].iloc[T0]) + 1
today["cluster_start"] = str(idx[_cs].date())
_os = T0 - int(F["off_age"].iloc[T0]) + 1
today["strategy_off_since_mark"] = str(idx[_os].date())
today["held_weights_next_day_F3"] = {c: nz(v) for c, v in tgt.iloc[T0].items() if abs(v) > 1e-12}
today["held_weights_next_day_F2"] = {c: nz(v) for c, v in
                                     wF2.where(P["mark"]).ffill().fillna(0).iloc[T0].items() if abs(v) > 1e-12}
today["daily_ema_recrosses_in_cluster"] = int(np.sum(ABOVE[_cs:T0 + 1][1:] & ~ABOVE[_cs:T0 + 1][:-1]))
# out-of-sample returns since the freeze and since the strategy went risk-off (labelled OOS)
today["oos_since_freeze"] = {c: nz(C.period_return(RET[c], "2026-07-25", idx[T0])) for c in STRATS + LEGS}
today["oos_since_riskoff_mark"] = {c: nz(np.exp(LOGW[c][T0] - LOGW[c][_os]) - 1) for c in STRATS + LEGS}

# percentile of today's features among in-sample dates (all, and risk-off-only)
ins = np.zeros(N, dtype=bool); ins[POS_START:POS_FREEZE + 1] = True
pct = {}
for k in ["dd52", "age", "vol20", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "vix",
          "gold63", "ief63", "corr63_ndx_ief", "ndx63"]:
    v = F[k].iloc[T0]
    a = F[k].values[ins]; a = a[~np.isnan(a)]
    b = F[k].values[ins & ~ON]; b = b[~np.isnan(b)]
    pct[k] = dict(today=nz(v), pct_all_insample=nz((a < v).mean() * 100) if len(a) else None,
                  pct_riskoff_insample=nz((b < v).mean() * 100) if len(b) else None,
                  n_all=int(len(a)), n_riskoff=int(len(b)))
today["percentiles"] = pct
print(f"TODAY {today['date']}: NDX {today['ndx']:.0f}  dd {today['dd_ath']*100:.1f}%  "
      f"cluster age {today['age']} (run {today['run']}, off_age {today['off_age']})  "
      f"vol20 {today['vol20']*100:.1f}%  er20 {today['er20']:.2f}  10y {today['tnx']:.2f} "
      f"(126d {today['tnx_chg126']:+.2f})  curve {today['curve']:+.2f}  dxy126 {today['dxy_chg126']*100:+.1f}%  "
      f"vix {today['vix']:.1f}  gold63 {today['gold63']*100:+.1f}%", flush=True)
for k, v in pct.items():
    print(f"   {k:16s} today {v['today']:+.4f}  pctile all {v['pct_all_insample']:5.1f}  "
          f"risk-off {v['pct_riskoff_insample']:5.1f}", flush=True)


# ------------------------------------------------------------------ per-episode metrics
# Analog outcomes are measured with IN-SAMPLE data only (to 2026-07-24): a horizon that would
# need post-freeze data is NaN and every 'days until' search is censored at the freeze.
END = POS_FREEZE


def fwd(c, i, h, end=END):
    j = i + h
    return np.nan if j > end else float(np.exp(LOGW[c][j] - LOGW[c][i]) - 1)


def path_maxdd(c, i, h, end=END):
    j = i + h
    if j > end:
        return np.nan
    p = np.exp(LOGW[c][i + 1:j + 1] - LOGW[c][i])
    p = np.concatenate([[1.0], p])
    return float((p / np.maximum.accumulate(p) - 1).min())


def episode(i, end=END):
    d = dict(date=str(idx[i].date()), pos=int(i))
    for k in ["dd_ath", "dd52", "age", "run", "off_age", "vol20", "er20", "tnx", "tnx_chg126",
              "curve", "dxy_chg126", "vix", "gold63", "ief63", "corr63_ndx_ief", "ndx63"]:
        d[k] = nz(F[k].iloc[i])
    for c in STRATS + LEGS:
        for hn, h in HZ.items():
            d[f"{c}_{hn}"] = nz(fwd(c, i, h, end))
    for c in ["NDX", "QQQ", "F2", "F3"]:
        d[f"{c}_maxdd12m"] = nz(path_maxdd(c, i, 252, end))
    j = min(i + 252, end)
    d["ndx_further_decline_12m"] = nz(min(0.0, NDX[i + 1:j + 1].min() / NDX[i] - 1)) if j > i else None
    peak = H52.iloc[i]
    d["peak_dd_12m"] = nz(min(F["dd52"].iloc[i], NDX[i + 1:j + 1].min() / peak - 1)) if j > i else None
    d["bear25_12m"] = (d["peak_dd_12m"] <= -0.25) if (d["peak_dd_12m"] is not None and i + 252 <= end) else None
    rec = np.where(NDX[i + 1:end + 1] >= peak)[0]
    d["days_to_new_peak"] = int(rec[0] + 1) if len(rec) else None          # None = not by the freeze
    endc = i + 1 + rec[0] if len(rec) else end
    d["cycle_peak_dd"] = nz(min(F["dd52"].iloc[i], NDX[i + 1:endc + 1].min() / peak - 1))
    d["bear25_cycle"] = bool(d["cycle_peak_dd"] <= -0.25)
    up = np.where(ABOVE[i + 1:end + 1])[0]
    d["days_to_daily_above"] = int(up[0] + 1) if len(up) else None
    on = np.where(ON[i + 1:end + 1])[0]
    d["days_to_strategy_reentry"] = int(on[0] + 1) if len(on) else None
    if i + 126 <= end:
        s = ON[i:i + 127].astype(int)
        d["entries_6m"] = int(np.sum(np.diff(s) == 1))
        d["whipsaws_6m"] = int(np.sum(np.diff(s) == -1))      # re-entries reversed inside 6m
        a = ABOVE[i:i + 127].astype(int)
        d["daily_upcrosses_6m"] = int(np.sum(np.diff(a) == 1))
    else:
        d["entries_6m"] = d["whipsaws_6m"] = d["daily_upcrosses_6m"] = None
    if d["days_to_strategy_reentry"] is not None:
        k = i + d["days_to_strategy_reentry"]
        d["NDX_until_reentry"] = nz(np.exp(LOGW["NDX"][k] - LOGW["NDX"][i]) - 1)
        d["F3_until_reentry"] = nz(np.exp(LOGW["F3"][k] - LOGW["F3"][i]) - 1)
    else:
        d["NDX_until_reentry"] = d["F3_until_reentry"] = None
    d["fwd12m_truncated_at_freeze"] = bool(i + 252 > end)
    return d


# ------------------------------------------------------------------ analog selection
def greedy(cands, score, k=None, radius=RADIUS):
    """pick lowest-score candidates, excluding +-radius positions around picks."""
    order = np.lexsort((cands, score))      # by score, ties -> earliest
    picked = []
    for o in order:
        p = cands[o]
        if all(abs(p - q) >= radius for q in picked):
            picked.append(int(p))
            if k is not None and len(picked) >= k:
                break
    return sorted(picked)


ELIG = ins & ~ON & ~ABOVE          # in-sample, below EMA-100, weekly strategy risk-off
f = F
age_today = int(F["age"].iloc[T0])
rules = {
    "R1": dict(desc="brief spec: DD from ATH in [-20%,-8%], cluster age 10-40d, vol20>25%, risk-off; first hit",
               mask=ELIG & f.dd_ath.between(-0.20, -0.08).values & (f.age.values >= 10) & (f.age.values <= 40)
               & (f.vol20.values > 0.25), pick="first"),
    "R1a": dict(desc="R1 filter + cluster age within +-5d of today's (still below EMA at today's age); "
                     "date in each episode with age closest to today's",
                mask=None, pick="aligned"),
    "R2": dict(desc="R1 with DD from 52-week high instead of ATH; first hit",
               mask=ELIG & f.dd52.between(-0.20, -0.08).values & (f.age.values >= 10) & (f.age.values <= 40)
               & (f.vol20.values > 0.25), pick="first"),
    "R3": dict(desc="R2 + 10y yield up >25bp over 126d (rates-rising sell-off); first hit",
               mask=None, pick="first"),
    "R4": dict(desc="tight band: DD52 [-17%,-10%], cluster age 15-35d, vol20>28%, risk-off; first hit",
               mask=ELIG & f.dd52.between(-0.17, -0.10).values & (f.age.values >= 15) & (f.age.values <= 35)
               & (f.vol20.values > 0.28), pick="first"),
    "R5": dict(desc="gold stress: risk-off, DD52 <= -8%, gold_phys 63d <= -8% (daily-real gold only, 2000-12+); first hit",
               mask=ELIG & (f.dd52.values <= -0.08) & (f.gold63.fillna(0).values <= -0.08), pick="first"),
}
rules["R1a"]["mask"] = rules["R1"]["mask"] & (np.abs(f.age.values - age_today) <= 5)
rules["R3"]["mask"] = rules["R2"]["mask"] & (f.tnx_chg126.values > 0.25)

sets = {}
for name, r in rules.items():
    c = np.where(r["mask"])[0]
    if r["pick"] == "first":
        picks = greedy(c, c.astype(float))
    else:
        picks = greedy(c, np.abs(f.age.values[c] - age_today).astype(float))
    sets[name] = dict(kind="rule", desc=r["desc"], n_qualifying_days=int(len(c)), picks=picks)
    N_VARIANTS += 1

# kNN on standardised features (z-scores over all in-sample dates)
Z = pd.DataFrame(index=idx)
Z["dd52"] = F["dd52"]
Z["lage"] = np.log1p(F["age"])
Z["lvol"] = np.log(F["vol20"])
Z["er20"] = F["er20"]
Z["tnx"] = F["tnx"]
Z["tnx_chg126"] = F["tnx_chg126"]
Z["curve"] = F["curve"]
Z["dxy_chg126"] = F["dxy_chg126"]
Z["lvix"] = np.log(F["vix"])
Z["gold63"] = F["gold63"]
mu = Z[ins].mean(); sd = Z[ins].std()
Zs = (Z - mu) / sd
knn_specs = {
    "K1": dict(desc="kNN price state: dd52, log(1+cluster age), log vol20, er20",
               feats=["dd52", "lage", "lvol", "er20"]),
    "K2": dict(desc="kNN price + macro: K1 + 10y level, 10y 126d chg, curve, dxy 126d chg, log VIX (1990+)",
               feats=["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix"]),
    "K3": dict(desc="kNN K2 + gold 63d return (dates >= 2000-12, daily-real gold only)",
               feats=["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix", "gold63"]),
}
for name, s in knn_specs.items():
    X = Zs[s["feats"]].values
    q = X[T0]
    ok = ELIG & ~np.isnan(X).any(axis=1)
    c = np.where(ok)[0]
    dist = np.sqrt(((X[c] - q) ** 2).sum(axis=1))
    picks = greedy(c, dist, k=KNN_K)
    dmap = dict(zip(c.tolist(), dist.tolist()))
    sets[name] = dict(kind="knn", desc=s["desc"], feats=s["feats"], n_candidate_days=int(len(c)),
                      picks=picks, dist={p: dmap[p] for p in picks},
                      dist_median_all_candidates=nz(np.median(dist)))
    N_VARIANTS += 1


# ------------------------------------------------------------------ summaries
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    ph = k / n
    den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, cen - half), min(1.0, cen + half))


def median_ci(x, conf=0.95):
    """distribution-free order-statistic CI for the median; None if n too small."""
    x = np.sort(x); n = len(x)
    if n < 6:
        return (None, None, None)
    from math import comb
    best = None
    for j in range(1, n // 2 + 1):          # CI = [x_(j), x_(n-j+1)]
        cov = sum(comb(n, i) for i in range(j, n - j + 1)) / 2 ** n
        if cov >= conf:
            best = (float(x[j - 1]), float(x[n - j]), float(cov))
    return best if best else (None, None, None)


def dist_summary(vals):
    x = np.array([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return dict(n=0)
    k = int((x > 0).sum())
    lo, hi = wilson(k, n)
    mlo, mhi, mcov = median_ci(x)
    return dict(n=n, median=nz(np.median(x)), q25=nz(np.percentile(x, 25)), q75=nz(np.percentile(x, 75)),
                mean=nz(x.mean()), min=nz(x.min()), max=nz(x.max()), hit=nz(k / n),
                hit_wilson95=[nz(lo), nz(hi)], median_ci=[nz(mlo), nz(mhi)], median_ci_cov=nz(mcov))


def summarize(rows):
    out = {}
    for c in STRATS + LEGS:
        for hn in HZ:
            out[f"{c}_{hn}"] = dist_summary([r[f"{c}_{hn}"] for r in rows])
    for k in ["ndx_further_decline_12m", "peak_dd_12m", "cycle_peak_dd", "days_to_daily_above",
              "days_to_strategy_reentry", "whipsaws_6m", "entries_6m", "daily_upcrosses_6m",
              "F3_maxdd12m", "F2_maxdd12m", "QQQ_maxdd12m", "NDX_until_reentry", "F3_until_reentry"]:
        out[k] = dist_summary([r[k] for r in rows])
    b = [r["bear25_12m"] for r in rows if r["bear25_12m"] is not None]
    out["bear25_12m_rate"] = dict(k=int(sum(b)), n=len(b), rate=nz(np.mean(b)) if b else None,
                                  wilson95=[nz(v) for v in wilson(int(sum(b)), len(b))])
    bc = [r["bear25_cycle"] for r in rows]
    out["bear25_cycle_rate"] = dict(k=int(sum(bc)), n=len(bc), rate=nz(np.mean(bc)) if bc else None)
    for s in ["F2", "F3"]:
        d = [(r[f"{s}_12m"] - r["QQQ_12m"]) for r in rows if r[f"{s}_12m"] is not None and r["QQQ_12m"] is not None]
        out[f"{s}_minus_QQQ_12m"] = dist_summary(d)
        d6 = [(r[f"{s}_6m"] - r["QQQ_6m"]) for r in rows if r[f"{s}_6m"] is not None and r["QQQ_6m"] is not None]
        out[f"{s}_minus_QQQ_6m"] = dist_summary(d6)
    w = [r["whipsaws_6m"] for r in rows if r["whipsaws_6m"] is not None]
    out["share_with_any_whipsaw_6m"] = nz(np.mean([x > 0 for x in w])) if w else None
    out["n_12m_truncated_at_freeze"] = int(sum(r["fwd12m_truncated_at_freeze"] for r in rows))
    return out


# null distribution: N random de-clustered risk-off dates (full 12m forward window in-sample)
POOL = np.where(ELIG)[0]
POOL = POOL[POOL + 252 <= POS_FREEZE]
rng = np.random.default_rng(SEED)
NULL_REPS = 2000
pool_f = {k: np.array([fwd(c, i, h) for i in POOL]) for k, (c, h) in
          {"NDX_12m": ("NDX", 252), "F3_12m": ("F3", 252), "F2_12m": ("F2", 252), "QQQ_12m": ("QQQ", 252),
           "NDX_6m": ("NDX", 126), "F3_6m": ("F3", 126)}.items()}
pos_in_pool = {int(p): j for j, p in enumerate(POOL)}


def random_declustered(nsel):
    perm = rng.permutation(len(POOL))
    picked = []
    for j in perm:
        p = POOL[j]
        if all(abs(p - POOL[q]) >= RADIUS for q in picked):
            picked.append(j)
            if len(picked) >= nsel:
                break
    return picked


def null_test(rows):
    n = sum(1 for r in rows if r["NDX_12m"] is not None)
    if n < 3:
        return None
    res = {}
    sims = {k: [] for k in pool_f}
    for _ in range(NULL_REPS):
        js = random_declustered(n)
        for k in pool_f:
            sims[k].append(np.median(pool_f[k][js]))
    for k in pool_f:
        obs = np.median([r[k] for r in rows if r[k] is not None])
        a = np.array(sims[k])
        res[k] = dict(observed_median=nz(obs), null_median=nz(np.median(a)),
                      null_p05=nz(np.percentile(a, 5)), null_p95=nz(np.percentile(a, 95)),
                      pctile_of_observed=nz((a < obs).mean() * 100))
    res["n"] = n
    return res


results = {}
for name, s in sets.items():
    rows = [episode(i) for i in s["picks"]]
    if s["kind"] == "knn":
        for r in rows:
            r["distance"] = nz(s["dist"][r["pos"]])
    summ = summarize(rows)
    nt = null_test(rows)
    results[name] = dict(kind=s["kind"], desc=s["desc"], n_episodes=len(rows),
                         n_qualifying_or_candidate_days=s.get("n_qualifying_days", s.get("n_candidate_days")),
                         feats=s.get("feats"), knn_dist_median_all_candidates=s.get("dist_median_all_candidates"),
                         episodes=rows, summary=summ, null_test=nt)
    print(f"\n=== {name}: {s['desc']}  N={len(rows)}", flush=True)
    for r in rows:
        def g(k, m=100, f="{:+6.1f}"):
            v = r.get(k)
            return "   n/a" if v is None else f.format(v * m)
        print(f"  {r['date']} dd52 {r['dd52']*100:6.1f} age {r['age']:3.0f} vol {r['vol20']*100:5.1f} "
              f"10yd {r['tnx_chg126']:+.2f} | NDX 3m {g('NDX_3m')} 12m {g('NDX_12m')} | "
              f"F2 12m {g('F2_12m')} F3 12m {g('F3_12m')} QQQ 12m {g('QQQ_12m')} | "
              f"pkDD12 {g('peak_dd_12m')} reent {r['days_to_strategy_reentry']} whip {r['whipsaws_6m']}"
              + (f" d={r['distance']:.2f}" if 'distance' in r else ""), flush=True)
    sm = summ

    def pm(key, stat="median", m=100.0):
        v = sm.get(key, {}).get(stat)
        return "n/a" if v is None else f"{v*m:+.1f}"
    print(f"  medians: NDX 1/3/6/12m {pm('NDX_1m')} {pm('NDX_3m')} {pm('NDX_6m')} {pm('NDX_12m')} | "
          f"hit12 {pm('NDX_12m', 'hit', 1)} | F2 12m {pm('F2_12m')} F3 12m {pm('F3_12m')} QQQ 12m {pm('QQQ_12m')} | "
          f"bear25 {sm['bear25_12m_rate']['k']}/{sm['bear25_12m_rate']['n']} | reentry med "
          f"{pm('days_to_strategy_reentry', 'median', 1)} | whips med {pm('whipsaws_6m', 'median', 1)}", flush=True)
    if nt:
        print(f"  null pctile of analog median: NDX12 {nt['NDX_12m']['pctile_of_observed']:.0f}  "
              f"F3_12 {nt['F3_12m']['pctile_of_observed']:.0f}  F2_12 {nt['F2_12m']['pctile_of_observed']:.0f}",
              flush=True)


# ------------------------------------------------------------------ unconditional baselines (in-sample)
def baseline(mask):
    ps = np.where(mask)[0]
    ps = ps[ps + 252 <= POS_FREEZE]
    out = {"n_days": int(len(ps))}
    for c in STRATS + ["IEF", "GLD"]:
        for hn, h in HZ.items():
            x = np.array([fwd(c, i, h) for i in ps])
            out[f"{c}_{hn}"] = dict(median=nz(np.median(x)), q25=nz(np.percentile(x, 25)),
                                    q75=nz(np.percentile(x, 75)), hit=nz((x > 0).mean()), mean=nz(x.mean()))
    pk = np.array([min(F["dd52"].iloc[i], NDX[i + 1:i + 253].min() / H52.iloc[i] - 1) for i in ps])
    out["bear25_12m_rate_from_52w_peak"] = nz((pk <= -0.25).mean())
    return out


# number of distinct risk-off spells in-sample (effective sample size of 'risk-off' conditioning)
s_on = ON[POS_START:POS_FREEZE + 1].astype(int)
n_off_spells = int(np.sum(np.diff(s_on) == -1) + (1 - s_on[0]))
base_all = baseline(ins)
base_off = baseline(ELIG)
base_like = baseline(rules["R2"]["mask"])
print(f"\nBASELINES (in-sample, overlapping daily starts): all-days NDX12m median "
      f"{base_all['NDX_12m']['median']*100:+.1f}% hit {base_all['NDX_12m']['hit']:.2f}; risk-off-days "
      f"{base_off['NDX_12m']['median']*100:+.1f}% hit {base_off['NDX_12m']['hit']:.2f} "
      f"(in {n_off_spells} risk-off spells); R2-like days {base_like['NDX_12m']['median']*100:+.1f}%", flush=True)
print(f"   F3 12m median: all {base_all['F3_12m']['median']*100:+.1f}%  risk-off {base_off['F3_12m']['median']*100:+.1f}%"
      f"  R2-like {base_like['F3_12m']['median']*100:+.1f}%   F2: all {base_all['F2_12m']['median']*100:+.1f}%  "
      f"risk-off {base_off['F2_12m']['median']*100:+.1f}%  R2-like {base_like['F2_12m']['median']*100:+.1f}%", flush=True)


# ------------------------------------------------------------------ block bootstrap (in-sample)
BLK, NBLK, REPS = 63, 4, 10000
seg = slice(POS_START, POS_FREEZE + 1)
Rb = {c: RET[c].values[seg] for c in ["F2", "F3", "QQQ", "NDX"]}
on_b = ON[seg]
Tb = len(on_b)
valid = np.arange(0, Tb - BLK)                  # block start s uses returns s+1 .. s+BLK
st_off = valid[~on_b[valid]]
st_on = valid[on_b[valid]]
like = (rules["R2"]["mask"][seg])
st_like = valid[like[valid]]
RATESUP = ELIG & (F["tnx_chg126"].values > 0.25)          # risk-off while the 10y is up >25bp/126d
ratesup_b = RATESUP[seg]
st_ratesup = valid[ratesup_b[valid]]


def bootstrap(first_pool, cont, seed):
    g = np.random.default_rng(seed)
    starts = np.empty((REPS, NBLK), dtype=int)
    starts[:, 0] = g.choice(first_pool, REPS)
    for b in range(1, NBLK):
        if cont == "uniform":
            starts[:, b] = g.choice(valid, REPS)
        elif cont == "state":
            end_state = on_b[starts[:, b - 1] + BLK]
            nxt = np.empty(REPS, dtype=int)
            m = end_state
            nxt[m] = g.choice(st_on, int(m.sum()))
            nxt[~m] = g.choice(st_off, int((~m).sum()))
            starts[:, b] = nxt
    ix = (starts[:, :, None] + 1 + np.arange(BLK)[None, None, :]).reshape(REPS, NBLK * BLK)
    out = {}
    tot = {}
    for c in Rb:
        x = Rb[c][ix]
        w = np.cumprod(1 + x, axis=1)
        tr = w[:, -1] - 1
        wp = np.concatenate([np.ones((REPS, 1)), w], axis=1)
        mdd = (wp / np.maximum.accumulate(wp, axis=1) - 1).min(axis=1)
        tot[c] = tr
        out[c] = dict(p05=nz(np.percentile(tr, 5)), p10=nz(np.percentile(tr, 10)), p25=nz(np.percentile(tr, 25)),
                      median=nz(np.median(tr)), p75=nz(np.percentile(tr, 75)), p90=nz(np.percentile(tr, 90)),
                      p95=nz(np.percentile(tr, 95)), mean=nz(tr.mean()), p_loss=nz((tr < 0).mean()),
                      p_below_m10=nz((tr < -0.10).mean()), p_below_m20=nz((tr < -0.20).mean()),
                      maxdd_median=nz(np.median(mdd)), maxdd_p10=nz(np.percentile(mdd, 10)),
                      maxdd_p05=nz(np.percentile(mdd, 5)))
    for s in ["F2", "F3"]:
        out[f"P_{s}_beats_QQQ"] = nz((tot[s] > tot["QQQ"]).mean())
    out["P_F2_beats_F3"] = nz((tot["F2"] > tot["F3"]).mean())
    out["first_block_pool_days"] = int(len(first_pool))
    return out


boot = {}
schemes = {
    "B1_riskoff_state_matched": (st_off, "state",
                                 "block 1 starts on a risk-off day; blocks 2-4 start on days whose state "
                                 "matches the state at the end of the previous block (keeps regime persistence)"),
    "B2_riskoff_then_uniform": (st_off, "uniform", "block 1 starts on a risk-off day; blocks 2-4 uniform"),
    "B3_likenow_state_matched": (st_like, "state",
                                 "block 1 starts on an R2-like day (risk-off, DD52 8-20%, age 10-40, vol>25%); "
                                 "state-matched continuation"),
    "B4_riskoff_ratesup_state_matched": (st_ratesup, "state",
                                         "block 1 starts on a risk-off day with the 10y yield up >25bp over 126d "
                                         "(bond off-leg under pressure, like now); state-matched continuation"),
    "B0_unconditional": (valid, "uniform", "all 4 blocks uniform (unconditional reference)"),
}
for k, (pool, cont, desc) in schemes.items():
    boot[k] = bootstrap(pool, cont, SEED + len(boot))
    boot[k]["desc"] = desc
    N_VARIANTS += 1
    b = boot[k]
    print(f"\nBOOT {k} (first-block pool {b['first_block_pool_days']} days)", flush=True)
    for c in ["F2", "F3", "QQQ", "NDX"]:
        x = b[c]
        print(f"   {c:4s} 12m: p05 {x['p05']*100:+6.1f}  p25 {x['p25']*100:+6.1f}  med {x['median']*100:+6.1f}  "
              f"p75 {x['p75']*100:+6.1f}  p95 {x['p95']*100:+6.1f}  P(loss) {x['p_loss']:.2f}  "
              f"P(<-20%) {x['p_below_m20']:.2f}  maxDD med {x['maxdd_median']*100:+.1f} p05 {x['maxdd_p05']*100:+.1f}",
              flush=True)
    print(f"   P(F3>QQQ) {b['P_F3_beats_QQQ']:.2f}  P(F2>QQQ) {b['P_F2_beats_QQQ']:.2f}  P(F2>F3) {b['P_F2_beats_F3']:.2f}",
          flush=True)

# empirical (non-bootstrap) 12m outcomes from every risk-off day / R2-like day (overlapping windows)
emp = {}
for k, mask in {"E_riskoff_days": ELIG, "E_R2like_days": rules["R2"]["mask"],
                "E_riskoff_ratesup_days": RATESUP}.items():
    ps = np.where(mask)[0]; ps = ps[ps + 252 <= POS_FREEZE]
    sp = np.where(np.diff(np.concatenate([[0], mask[ps[0]:ps[-1] + 1].astype(int)])) == 1)[0] if len(ps) else []
    e = {"n_days": int(len(ps)), "n_distinct_runs": int(len(sp))}
    for c in ["F2", "F3", "QQQ", "NDX", "IEF", "GLD"]:
        x = np.array([fwd(c, i, 252) for i in ps])
        e[c] = dict(p05=nz(np.percentile(x, 5)), p25=nz(np.percentile(x, 25)), median=nz(np.median(x)),
                    p75=nz(np.percentile(x, 75)), p95=nz(np.percentile(x, 95)), p_loss=nz((x < 0).mean()))
    for c in ["F2", "F3"]:
        x6 = np.array([fwd(c, i, 126) for i in ps])
        e[f"{c}_6m"] = dict(median=nz(np.median(x6)), p05=nz(np.percentile(x6, 5)), p_loss=nz((x6 < 0).mean()))
    emp[k] = e
    N_VARIANTS += 1
    print(f"\nEMP {k} ({e['n_days']} overlapping days): " + "  ".join(
        f"{c} med {e[c]['median']*100:+.1f} p05 {e[c]['p05']*100:+.1f} P(loss) {e[c]['p_loss']:.2f}"
        for c in ["F2", "F3", "QQQ"]), flush=True)

# ------------------------------------------------------------------ consensus across analog sets
cons = {}
for name in results:
    s = results[name]["summary"]
    cons[name] = dict(n=results[name]["n_episodes"],
                      ndx_1m=s["NDX_1m"].get("median"), ndx_3m=s["NDX_3m"].get("median"),
                      ndx_6m=s["NDX_6m"].get("median"), ndx_12m=s["NDX_12m"].get("median"),
                      ndx_12m_hit=s["NDX_12m"].get("hit"), ndx_12m_q25=s["NDX_12m"].get("q25"),
                      ndx_12m_q75=s["NDX_12m"].get("q75"),
                      f2_12m=s["F2_12m"].get("median"), f3_12m=s["F3_12m"].get("median"),
                      qqq_12m=s["QQQ_12m"].get("median"),
                      f2_6m=s["F2_6m"].get("median"), f3_6m=s["F3_6m"].get("median"),
                      f3_minus_qqq_12m=s["F3_minus_QQQ_12m"].get("median"),
                      f2_minus_qqq_12m=s["F2_minus_QQQ_12m"].get("median"),
                      further_decline_med=s["ndx_further_decline_12m"].get("median"),
                      bear25=f"{s['bear25_12m_rate']['k']}/{s['bear25_12m_rate']['n']}",
                      reentry_days_med=s["days_to_strategy_reentry"].get("median"),
                      whipsaws_med=s["whipsaws_6m"].get("median"),
                      share_any_whipsaw=s["share_with_any_whipsaw_6m"],
                      ief_6m=s["IEF_6m"].get("median"), gld_6m=s["GLD_6m"].get("median"),
                      ief_12m=s["IEF_12m"].get("median"), gld_12m=s["GLD_12m"].get("median"),
                      f3_hit_12m=s["F3_12m"].get("hit"), f2_hit_12m=s["F2_12m"].get("hit"),
                      qqq_hit_12m=s["QQQ_12m"].get("hit"),
                      f3_min_12m=s["F3_12m"].get("min"), f2_min_12m=s["F2_12m"].get("min"),
                      qqq_min_12m=s["QQQ_12m"].get("min"),
                      f3_beats_qqq_12m=s["F3_minus_QQQ_12m"].get("hit"),
                      f2_beats_qqq_12m=s["F2_minus_QQQ_12m"].get("hit"),
                      f3_maxdd12m_med=s["F3_maxdd12m"].get("median"), qqq_maxdd12m_med=s["QQQ_maxdd12m"].get("median"),
                      ndx_until_reentry_med=s["NDX_until_reentry"].get("median"),
                      f3_until_reentry_med=s["F3_until_reentry"].get("median"),
                      days_to_daily_above_med=s["days_to_daily_above"].get("median"),
                      null_pct_ndx12=(results[name]["null_test"] or {}).get("NDX_12m", {}).get("pctile_of_observed"),
                      null_pct_f3_12=(results[name]["null_test"] or {}).get("F3_12m", {}).get("pctile_of_observed"),
                      null_pct_f2_12=(results[name]["null_test"] or {}).get("F2_12m", {}).get("pctile_of_observed"))
print("\nCONSENSUS (in-sample analog base rates; medians across episodes)", flush=True)
for name, c in cons.items():
    def q(v, m=100):
        return "  n/a" if v is None else f"{v*m:+5.1f}"
    print(f"  {name:4s} N={c['n']:2d} NDX 1/3/6/12m {q(c['ndx_1m'])} {q(c['ndx_3m'])} {q(c['ndx_6m'])} {q(c['ndx_12m'])}"
          f" hit12 {q(c['ndx_12m_hit'],1)} | F2 {q(c['f2_12m'])} F3 {q(c['f3_12m'])} QQQ {q(c['qqq_12m'])}"
          f" | F3>QQQ {q(c['f3_beats_qqq_12m'],1)} F2>QQQ {q(c['f2_beats_qqq_12m'],1)} | bear25 {c['bear25']}"
          f" | reentry {c['reentry_days_med']} whip>0 {q(c['share_any_whipsaw'],1)}"
          f" | NDX/F3 until re-entry {q(c['ndx_until_reentry_med'])}/{q(c['f3_until_reentry_med'])}"
          f" | IEF6m {q(c['ief_6m'])} GLD6m {q(c['gld_6m'])}", flush=True)
all_dates = sorted({r["date"] for n_ in results for r in results[n_]["episodes"]})
date_count = {d: sum(1 for n_ in results for r in results[n_]["episodes"] if r["date"] == d) for d in all_dates}

# union of every analog set, de-duplicated into unique episodes (earliest date per 60-day cluster)
upos = np.array(sorted({r["pos"] for n_ in results for r in results[n_]["episodes"]}))
upick = greedy(upos, upos.astype(float))
urows = [episode(i) for i in upick]
for r in urows:
    lo, hi = r["pos"] - RADIUS, r["pos"] + RADIUS
    r["in_sets"] = sorted({n_ for n_ in results for e in results[n_]["episodes"] if lo < e["pos"] < hi})
union = dict(desc="union of all analog sets, de-duplicated (earliest date per 60-day cluster)",
             n_episodes=len(urows), episodes=urows, summary=summarize(urows))
N_VARIANTS += 1
us = union["summary"]
neg = [(r["date"], round(r["F3_12m"] * 100, 1), round(r["F2_12m"] * 100, 1), round(r["QQQ_12m"] * 100, 1))
       for r in urows if r["F3_12m"] is not None and r["F3_12m"] < 0]
print(f"\nUNION: {len(urows)} unique episodes  NDX12 med {us['NDX_12m']['median']*100:+.1f} hit {us['NDX_12m']['hit']:.2f} "
      f"| F2 {us['F2_12m']['median']*100:+.1f} (min {us['F2_12m']['min']*100:+.1f}, hit {us['F2_12m']['hit']:.2f}) "
      f"F3 {us['F3_12m']['median']*100:+.1f} (min {us['F3_12m']['min']*100:+.1f}, hit {us['F3_12m']['hit']:.2f}) "
      f"QQQ {us['QQQ_12m']['median']*100:+.1f} (min {us['QQQ_12m']['min']*100:+.1f}, hit {us['QQQ_12m']['hit']:.2f}) "
      f"| bear25 {us['bear25_12m_rate']['k']}/{us['bear25_12m_rate']['n']} | reentry med "
      f"{us['days_to_strategy_reentry']['median']} [{us['days_to_strategy_reentry']['q25']}, "
      f"{us['days_to_strategy_reentry']['q75']}] | daily-above med {us['days_to_daily_above']['median']} "
      f"| whip>0 {us['share_with_any_whipsaw_6m']:.2f} | further decline med {us['ndx_further_decline_12m']['median']*100:+.1f}"
      f" | F3 maxDD12 med {us['F3_maxdd12m']['median']*100:+.1f} QQQ {us['QQQ_maxdd12m']['median']*100:+.1f}", flush=True)
print(f"   negative F3 12m episodes (date, F3, F2, QQQ): {neg}", flush=True)
for r in urows:
    print(f"   {r['date']} sets={','.join(r['in_sets'])}", flush=True)

out = dict(
    meta=dict(script="update_2026_11/analogs.py", panel_last=str(idx[LAST].date()), freeze=str(FREEZE.date()),
              insample=f"{START.date()}..{FREEZE.date()}", horizons_trading_days=HZ, decluster_radius=RADIUS,
              knn_k=KNN_K, cluster_gap_days=GAP, null_reps=NULL_REPS, boot_reps=REPS, boot_block=BLK,
              n_variants_tested=N_VARIANTS,
              notes=["analog dates restricted to 1988-01-01..2026-07-24 (in-sample); today's state is OOS (query only)",
                     "forward windows may extend past the freeze; flagged per episode (fwd12m_crosses_freeze)",
                     "gold63 never computed from monthly-interpolated gold (pre 2000-09 -> NaN)",
                     "F2 = QLD vt 0.30 + 15% gold_phys + IEF; F3 = TQQQ vt 0.20 + 15% gold_phys + IEF; common.py mechanics",
                     "strategy state = target weights at the last Friday mark <= t (held from t+1)",
                     "whipsaws_6m = weekly re-entries reversed within the next 126 trading days"]),
    today=today, variants=results, union=union,
    baselines=dict(all_days=base_all, riskoff_days=base_off, r2like_days=base_like,
                   n_riskoff_spells_insample=n_off_spells),
    bootstrap=boot, empirical_overlapping=emp, consensus=cons, analog_date_frequency=date_count)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return nz(o)
    return o


with open(OUT, "w") as fh:
    json.dump(clean(out), fh, indent=1)
print(f"\nvariants tested: {N_VARIANTS}\nsaved {OUT}", flush=True)
