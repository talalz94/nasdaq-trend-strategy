"""
verify_analogs_integrity_checks.py -- adversarial INTEGRITY checks of update_2026_11/analogs.py

Independent re-implementations (NOT importing analogs.py / common.py mechanics):
  1. today's state from raw panel columns (EMA crosses, cluster, gold peak, percentiles)
  2. loop-based Strategy F2/F3 simulation (Friday mark, t+1 application, 3bp turnover cost)
     compared against common.backtest
  3. forward outcomes for the 29 union episodes from raw levels / raw returns vs analogs.json
  4. bear-market definition probe: analog dates that were ALREADY <= -25% from the 52w high
  5. R1a 'closest age' pick: look-ahead probe + causal alternatives
  6. kNN distance ratios vs feature dimension (is today's macro mismatch unusual?)
  7. null-test power (width of the 5-95 band of random risk-off medians)
  8. whipsaw durations, re-entry conditional on a 23-day cluster, rates-up 6m vs 12m
Writes nothing inside the project except this script; JSON output goes to argv[1] (scratch).
"""
import os, sys, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", HERE)
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
J = json.load(open(os.path.join(HERE, "analogs.json")))
idx = P.index; N = len(idx)
FREEZE = pd.Timestamp("2026-07-24"); START = pd.Timestamp("1988-01-01")
PF = int(idx.searchsorted(FREEZE, side="right") - 1); PS = int(idx.searchsorted(START))
T0 = N - 1
out = {}

# ---------------------------------------------------------------- 1. today's state (raw)
ndx = P["ndx"].values
ema = P["ndx"].ewm(span=100, adjust=False).mean().values
above = ndx > ema
assert np.array_equal(above, P["above"].values.astype(bool)), "panel 'above' != recomputed"
fri = (idx.dayofweek == 4)
assert np.array_equal(fri, P["mark"].values.astype(bool)), "mark != trading Fridays"
recent = pd.DataFrame({"ndx": ndx, "ema": ema, "above": above, "fri": fri}, index=idx).loc["2026-10-01":]
first_below = recent.index[(~recent.above) & recent.above.shift(1, fill_value=True)][0]
cross_runs = []
a = recent.above.values; d = recent.index
i = 0
while i < len(a):
    if a[i] and d[i] > first_below:
        j = i
        while j + 1 < len(a) and a[j + 1]:
            j += 1
        cross_runs.append((str(d[i].date()), str(d[j].date()), int(j - i + 1), bool(recent.fri.values[i:j + 1].any())))
        i = j + 1
    else:
        i += 1
ath_pos = int(np.argmax(ndx[:T0 + 1]))
gw = (1 + P["gold_phys"]).cumprod()
gpk = gw.loc["2025-01-01":].idxmax()
out["today"] = dict(date=str(idx[T0].date()), ndx=float(ndx[T0]), ath=float(ndx[ath_pos]),
                    ath_date=str(idx[ath_pos].date()), dd_ath=float(ndx[T0] / ndx[ath_pos] - 1),
                    first_close_below=str(first_below.date()), recross_runs=cross_runs,
                    gold_peak_date=str(gpk.date()), gold_dd_from_peak=float(gw.iloc[T0] / gw.loc[gpk] - 1),
                    gold63=float(gw.iloc[T0] / gw.iloc[T0 - 63] - 1))
pre2000_high = ndx[:idx.searchsorted(pd.Timestamp("2000-12-31"))].max()
regain = idx[(ndx > pre2000_high) & (idx > pd.Timestamp("2001-01-01"))][0]
out["today"]["ndx_2000_high_first_regained"] = str(regain.date())
print("TODAY", json.dumps(out["today"], indent=0))

# ---------------------------------------------------------------- 2. loop-based strategy sim
vol20 = P["ndx"].pct_change().rolling(20).std().values * np.sqrt(252)
diff20 = np.abs(P["ndx"].diff(20).values)
path20 = P["ndx"].diff().abs().rolling(20).sum().values
with np.errstate(divide="ignore", invalid="ignore"):
    er20 = np.where(path20 > 0, diff20 / path20, np.nan)
month = idx.month.values


def sim(lev, vt, cost=3e-4):
    R = {"lev": P[lev].values, "ief": P["ief"].values, "gold": P["gold_phys"].values}
    held = np.zeros(3); pending = None
    ret = np.zeros(N); state = np.zeros(N, dtype=bool)
    on = False
    for t in range(N):
        turn = 0.0
        if pending is not None:                       # new weights start today (t+1 of the mark)
            turn = np.abs(pending - held).sum(); held = pending; pending = None
        ret[t] = held[0] * R["lev"][t] + held[1] * R["ief"][t] + held[2] * R["gold"][t] - cost * turn
        if fri[t]:
            on = bool(above[t])
            s = 1.0 if (np.isnan(vol20[t]) or vol20[t] == 0) else min(1.0, vt / vol20[t])
            if month[t] in (8, 9):
                s *= 0.5
            if not (er20[t] >= 0.25):
                s *= 0.5
            eq = s if on else 0.0
            pending = np.array([0.85 * eq, 0.85 - 0.85 * eq, 0.15])
        state[t] = on                                  # weekly state at close t (held from t+1)
    return ret, state


f3_sim, ON = sim("tqqq", 0.20)
f2_sim, ON2 = sim("qld", 0.30)
assert np.array_equal(ON, ON2)
sys.path.insert(0, HERE)
import common as C  # only to compare against the shared mechanics
f3_c = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_phys", off="ief")).values
f2_c = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", off="ief")).values
out["sim_vs_common"] = dict(f3_maxabs=float(np.abs(f3_sim - f3_c).max()), f2_maxabs=float(np.abs(f2_sim - f2_c).max()))
print("loop sim vs common.backtest max |diff|:", out["sim_vs_common"])
ABOVE = above


def stats(r, a, b):
    s = pd.Series(r, index=idx)
    s = s[(s.index >= a) & (s.index <= b)]
    e = (1 + s).cumprod(); yrs = (s.index[-1] - s.index[0]).days / 365.25
    return dict(cagr=float(e.iloc[-1] ** (1 / yrs) - 1), maxdd=float((e / e.cummax() - 1).min()))


out["insample_stats"] = {"F3": stats(f3_sim, START, FREEZE), "F2": stats(f2_sim, START, FREEZE)}
print("in-sample F3/F2 (loop sim):", out["insample_stats"])

# ---------------------------------------------------------------- 3. union outcomes from raw data
H52 = pd.Series(ndx, index=idx).rolling(252, min_periods=1).max().values
qqq = P["qqq"].values; ief = P["ief"].values; tlt = P["tlt"].values; gld = P["gold_phys"].values


def comp(r, i, h):
    j = i + h
    return np.nan if j > PF else float(np.prod(1 + r[i + 1:j + 1]) - 1)


def reentry(i):
    on = np.where(ON[i + 1:PF + 1])[0]
    return int(on[0] + 1) if len(on) else None


U = J["union"]["episodes"]
mism = []
rows = []
for e in U:
    i = int(idx.searchsorted(pd.Timestamp(e["date"])))
    r = dict(date=e["date"], i=i,
             NDX_12m=(ndx[i + 252] / ndx[i] - 1) if i + 252 <= PF else np.nan,
             QQQ_12m=comp(qqq, i, 252), F3_12m=comp(f3_sim, i, 252), F2_12m=comp(f2_sim, i, 252),
             IEF_12m=comp(ief, i, 252), TLT_12m=comp(tlt, i, 252), GLD_12m=comp(gld, i, 252),
             dd52=ndx[i] / H52[i] - 1, reent=reentry(i),
             pk=min(ndx[i] / H52[i] - 1, ndx[i + 1:min(i + 252, PF) + 1].min() / H52[i] - 1))
    rows.append(r)
    for k in ["NDX_12m", "QQQ_12m", "F3_12m", "F2_12m", "IEF_12m", "TLT_12m", "GLD_12m"]:
        v = e.get(k)
        if (v is None) != (np.isnan(r[k])) or (v is not None and abs(v - r[k]) > 1e-9):
            mism.append((e["date"], k, v, r[k]))
    if e["days_to_strategy_reentry"] != r["reent"]:
        mism.append((e["date"], "reentry", e["days_to_strategy_reentry"], r["reent"]))
out["union_recompute_mismatches"] = mism
print("union outcome mismatches vs JSON:", len(mism), mism[:5])
R = pd.DataFrame(rows)
full = R[R.NDX_12m.notna()]
out["union_recomputed"] = dict(n12=int(len(full)), ndx_med=float(full.NDX_12m.median()),
                               ndx_hit=float((full.NDX_12m > 0).mean()), f3_med=float(full.F3_12m.median()),
                               f2_med=float(full.F2_12m.median()), qqq_med=float(full.QQQ_12m.median()),
                               f3_min=float(full.F3_12m.min()), f2_min=float(full.F2_12m.min()))
print("union recomputed:", out["union_recomputed"])


# ---------------------------------------------------------------- 4. bear-market definition probe
def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(cen - half, 3), round(cen + half, 3))


full = full.copy()
full["bear"] = full.pk <= -0.25
full["already"] = full.dd52 <= -0.25
k_all = int(full.bear.sum()); n_all = len(full)
new = full[~full.already]
k_new = int(new.bear.sum()); n_new = len(new)
out["bear_probe"] = dict(k_all=k_all, n_all=n_all, already_in_bear_dates=full[full.already].date.tolist(),
                         k_new_excl_already=k_new, n_excl_already=n_new, rate_excl=k_new / n_new,
                         wilson_excl=wilson(k_new, n_new), rate_counting_already_as_nonevents=k_new / n_all)
print("bear probe:", out["bear_probe"])
per_set = {}
for nm, v in J["variants"].items():
    eps = [e for e in v["episodes"] if e["bear25_12m"] is not None]
    k = sum(e["bear25_12m"] for e in eps)
    auto = [e["date"] for e in eps if e["dd52"] <= -0.25]
    kn = sum(e["bear25_12m"] for e in eps if e["dd52"] > -0.25)
    per_set[nm] = dict(reported=f"{k}/{len(eps)}", already_in_bear=auto,
                       excl_already=f"{kn}/{len(eps) - len(auto)}")
out["bear_probe_per_set"] = per_set
print("per set:", per_set)

# ---------------------------------------------------------------- 5. R1a pick look-ahead probe
def cluster_age(ab, gap=5):
    age = np.zeros(len(ab), dtype=int); st, lb = -1, -10 ** 9
    for t in range(len(ab)):
        if not ab[t]:
            if st < 0 or (t - lb - 1) > gap:
                st = t
            lb = t
            age[t] = t - st + 1
    return age


age = cluster_age(ABOVE)
ath = pd.Series(ndx, index=idx).cummax().values
dd_ath = ndx / ath - 1
ins = np.zeros(N, dtype=bool); ins[PS:PF + 1] = True
ELIG = ins & ~ON & ~ABOVE
R1 = ELIG & (dd_ath >= -0.20) & (dd_ath <= -0.08) & (age >= 10) & (age <= 40) & (vol20 > 0.25)
i90 = int(idx.searchsorted(pd.Timestamp("1990-08-16")))
probe90 = []
for t in range(i90 - 2, i90 + 7):
    probe90.append((str(idx[t].date()), int(age[t]), round(float(dd_ath[t]), 4), round(float(vol20[t]), 3),
                    bool(ELIG[t]), bool(R1[t])))
out["r1a_1990_probe"] = probe90
print("R1a 1990 probe (date, age, dd_ath, vol20, ELIG, R1):")
for p in probe90:
    print("   ", p)


def greedy_first(c, radius=60):
    picked = []
    for p in c:
        if all(abs(p - q) >= radius for q in picked):
            picked.append(int(p))
    return picked


def outcomes(picks):
    x = {k: [] for k in ["NDX", "F3", "F2"]}
    re = []
    for i in picks:
        if i + 252 <= PF:
            x["NDX"].append(ndx[i + 252] / ndx[i] - 1)
            x["F3"].append(comp(f3_sim, i, 252)); x["F2"].append(comp(f2_sim, i, 252))
        re.append(reentry(i))
    re = [v for v in re if v is not None]
    return dict(n=len(picks), n12=len(x["NDX"]), ndx12=float(np.median(x["NDX"])), f3_12=float(np.median(x["F3"])),
                f2_12=float(np.median(x["F2"])), ndx_hit=float(np.mean(np.array(x["NDX"]) > 0)),
                reentry_med=float(np.median(re)), reentry_gt63=int(sum(v > 63 for v in re)),
                dates=[str(idx[i].date()) for i in picks])


# (a) causal: first date in each episode reaching age >= 18 with the R1 filter (earliest, de-clustered)
alt_a = greedy_first(np.where(R1 & (age >= 18) & (age <= 28))[0])
# (b) exact alignment: R1 filter holds AT age == 23 (requires the episode to reach today's age in-band)
alt_b = greedy_first(np.where(R1 & (age == 23))[0])
# (c) R1a as published (from JSON)
pub = [int(e["pos"]) for e in J["variants"]["R1a"]["episodes"]]
out["r1a_alternatives"] = dict(published=outcomes(pub), causal_first_age18_28=outcomes(alt_a),
                               exact_age23=outcomes(alt_b))
for k, v in out["r1a_alternatives"].items():
    print(f"R1a {k:24s} n={v['n']} n12={v['n12']} NDX12 {v['ndx12']*100:+.1f} hit {v['ndx_hit']:.2f} "
          f"F3 {v['f3_12']*100:+.1f} F2 {v['f2_12']*100:+.1f} reentry med {v['reentry_med']} >63: {v['reentry_gt63']}")

# ---------------------------------------------------------------- 6. kNN dimension effect
F = pd.DataFrame(index=idx)
F["dd52"] = ndx / H52 - 1
F["lage"] = np.log1p(age)
F["lvol"] = np.log(vol20)
F["er20"] = er20
tnx = P["tnx"].ffill()
F["tnx"] = tnx; F["tnx_chg126"] = tnx - tnx.shift(126)
F["curve"] = (P["tnx"] - P["irx"]).ffill()
dxy = P["dxy"].ffill(); F["dxy_chg126"] = dxy / dxy.shift(126) - 1
F["lvix"] = np.log(P["vix"].ffill())
interp = (P["gold_src"] == "monthly-interp").astype(float)
g63 = gw / gw.shift(63) - 1
g63[interp.rolling(64, min_periods=1).max() > 0] = np.nan       # 64: also excludes the base day
F["gold63"] = g63
mu = F[ins].mean(); sd = F[ins].std()
Z = ((F - mu) / sd)
specs = {"K1": ["dd52", "lage", "lvol", "er20"],
         "K2": ["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix"],
         "K3": ["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix", "gold63"]}
rng = np.random.default_rng(7)
kres = {}
for nm, fs in specs.items():
    X = Z[fs].values
    ok = ELIG & ~np.isnan(X).any(axis=1)
    cand = np.where(ok)[0]
    q = X[T0]
    dt = np.sqrt(((X[cand] - q) ** 2).sum(axis=1))
    today_ratio = float(dt.min() / np.median(dt))
    qs = rng.choice(cand, size=min(400, len(cand)), replace=False)
    ratios = []
    for qi in qs:
        c2 = cand[np.abs(cand - qi) > 82]                # same minimum gap as today -> in-sample
        dd = np.sqrt(((X[c2] - X[qi]) ** 2).sum(axis=1))
        ratios.append(dd.min() / np.median(dd))
    ratios = np.array(ratios)
    # per-feature share of squared distance for the 10 published picks
    picks = [int(e["pos"]) for e in J["variants"][nm]["episodes"]]
    contrib = ((X[picks] - q) ** 2)
    share = contrib.sum(axis=0) / contrib.sum()
    kres[nm] = dict(today_nearest=float(dt.min()), today_median=float(np.median(dt)), today_ratio=today_ratio,
                    insample_query_ratio_median=float(np.median(ratios)),
                    insample_query_ratio_p90=float(np.percentile(ratios, 90)),
                    today_ratio_pctile=float((ratios < today_ratio).mean() * 100),
                    feature_share_of_sqdist={f: round(float(s), 3) for f, s in zip(fs, share)},
                    today_z={f: round(float(v), 2) for f, v in zip(fs, q)})
    print(f"{nm}: today nearest {dt.min():.3f} median {np.median(dt):.3f} ratio {today_ratio:.3f} | in-sample "
          f"query ratio median {np.median(ratios):.3f} p90 {np.percentile(ratios, 90):.3f} -> today pctile "
          f"{kres[nm]['today_ratio_pctile']:.0f}")
    print("    feature share of sq.distance (published picks):", kres[nm]["feature_share_of_sqdist"])
    print("    today z:", kres[nm]["today_z"])
out["knn_dimension"] = kres

# ---------------------------------------------------------------- 7. null-test power
pw = {}
for nm, v in J["variants"].items():
    nt = v["null_test"]
    pw[nm] = {k: dict(obs=round(nt[k]["observed_median"], 3), band=[round(nt[k]["null_p05"], 3), round(nt[k]["null_p95"], 3)],
                      width_pp=round((nt[k]["null_p95"] - nt[k]["null_p05"]) * 100, 1), pct=round(nt[k]["pctile_of_observed"], 1))
              for k in ["NDX_12m", "F3_12m", "F2_12m"]}
    pw[nm]["n"] = nt["n"]
out["null_power"] = pw
print("null 5-95 band widths (pp):", {k: (v["n"], v["NDX_12m"]["width_pp"], v["F3_12m"]["width_pp"]) for k, v in pw.items()})

# ---------------------------------------------------------------- 8. whipsaw durations / re-entry aligned / rates-up
wh = []
for r in rows:
    i = r["i"]
    if i + 126 > PF:
        continue
    s = ON[i:i + 127].astype(int); dd = np.diff(s)
    ups = np.where(dd == 1)[0] + 1; downs = np.where(dd == -1)[0] + 1
    for u in ups:
        later = downs[downs > u]
        if len(later):
            wh.append(dict(date=r["date"], on_days=int(later[0] - u)))
lens = [w["on_days"] for w in wh]
n_ep_whip = len({w["date"] for w in wh})
n_ep_quick = len({w["date"] for w in wh if w["on_days"] <= 20})
out["whipsaw_durations"] = dict(n_reversed_reentries=len(lens), on_days_median=float(np.median(lens)) if lens else None,
                                on_days_list=sorted(lens), episodes_with_any=n_ep_whip,
                                episodes_with_reversal_within_20d=n_ep_quick,
                                n_episodes=sum(1 for r in rows if r["i"] + 126 <= PF))
print("whipsaws:", out["whipsaw_durations"])

# remaining wait for EVERY in-sample risk-off spell that was still risk-off at off-age 23 (like today)
off_age = np.zeros(N, dtype=int); c = 0
for t in range(N):
    c = c + 1 if not ON[t] else 0
    off_age[t] = c
d23 = np.where(ins & (off_age == 23) & ~ABOVE)[0]
rem = [reentry(i) for i in d23]
rem_c = [v for v in rem if v is not None]
out["remaining_wait_at_off_age_23"] = dict(n_spells=len(d23), median=float(np.median(rem_c)),
                                           q25=float(np.percentile(rem_c, 25)), q75=float(np.percentile(rem_c, 75)),
                                           gt63=int(sum(v > 63 for v in rem_c)), censored=int(sum(v is None for v in rem)))
age_union = [e["age"] for e in U]
out["union_age_median"] = float(np.median(age_union))
out["union_off_age_median"] = float(np.median([e["off_age"] for e in U]))
print("union analog-date cluster age median", out["union_age_median"], "off_age median", out["union_off_age_median"])
print("remaining wait from off-age 23 (all in-sample spells):", out["remaining_wait_at_off_age_23"])

# further decline excluding the truncated 2026 episode
fd = [e["ndx_further_decline_12m"] for e in U if not e["fwd12m_truncated_at_freeze"]]
out["further_decline_excl_truncated"] = dict(n=len(fd), median=float(np.median(fd)),
                                             q25=float(np.percentile(fd, 25)), q75=float(np.percentile(fd, 75)))
print("further decline excl truncated:", out["further_decline_excl_truncated"])

E = J["empirical_overlapping"]
out["ratesup_6m_vs_12m"] = {k: dict(F3_6m_ploss=E[k]["F3_6m"]["p_loss"], F3_12m_ploss=E[k]["F3"]["p_loss"],
                                    F3_12m_med=E[k]["F3"]["median"], F3_6m_med=E[k]["F3_6m"]["median"],
                                    F2_6m_ploss=E[k]["F2_6m"]["p_loss"], F2_12m_ploss=E[k]["F2"]["p_loss"],
                                    runs=E[k]["n_distinct_runs"])
                            for k in ["E_riskoff_days", "E_riskoff_ratesup_days"]}
print("rates-up vs all risk-off (empirical overlapping):", out["ratesup_6m_vs_12m"])

json.dump(out, open(os.path.join(SCRATCH, "verify_analogs_integrity_checks.json"), "w"), indent=1, default=str)
print("saved", os.path.join(SCRATCH, "verify_analogs_integrity_checks.json"))
