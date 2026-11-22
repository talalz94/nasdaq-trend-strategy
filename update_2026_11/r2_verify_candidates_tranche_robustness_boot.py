"""
r2_verify_candidates_tranche_robustness_boot.py -- bootstrap robustness checks for r2_candidates_tranche (ROBUSTNESS lens).

Read-only w.r.t. project files; writes only the scratch JSON given on the command line.

  G0  EXACT replay of the specialist's IS88 stationary bootstrap (same algorithm, same seeds 137/263, 5,000 reps)
      on a column subset -> must equal the saved pair statistics (the index draws do not depend on the columns)
  G1  F2 point mass: share of L=252 resamples whose maxDD equals the realized value / keeps the 2022 episode intact
  G2  alternative mean block lengths (63, 504) and alternative seeds at 126/252 (4,000 reps each)
  (G3 start-date sensitivity and L=1008 live in r2_verify_candidates_tranche_robustness_followup.py)
  G4  Martin ratio (CAGR / Ulcer) as a smoother multi-drawdown efficiency metric for the tier comparisons

usage: python -u update_2026_11/r2_verify_candidates_tranche_robustness_boot.py <scratch_out.json>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT = os.path.abspath(sys.argv[1])
assert os.path.dirname(OUT) != HERE
P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
SAVED = json.load(open(os.path.join(HERE, "r2_candidates_tranche.json")))
RES = {}

# ------------------------------------------------------------------ engine (independent, numpy)
SLEEVES = ["qqq", "qld", "tqqq", "ief", "bills", "gold_phys"]
RET = {c: P[c].fillna(0).values.astype(float) for c in SLEEVES}
RFD = P["rf_d"].fillna(0).values.astype(float)
VOL, ER = P["vol20"].values, P["er20"].values
ABOVE = P["above"].values.astype(bool)
MON = P["month"].values
DOW = IDX.dayofweek.values


def my_weights(lev, vt, off):
    v = np.where(np.isnan(VOL), 1.0, VOL)
    sc = np.where(np.isnan(VOL), 1.0, np.clip(vt / v, 0, 1))
    sc = np.clip(sc * np.where((MON == 8) | (MON == 9), 0.5, 1.0), 0, 1)
    ok = np.where(np.isnan(ER), False, ER >= 0.25)
    eng = np.where(ABOVE, np.where(ok, sc, 0.5 * sc), 0.0)
    W = {lev: 0.85 * eng}
    rest = np.clip(0.85 - 0.85 * eng, 0, None)
    if off == "IEF":
        W["ief"] = rest
    else:
        W["ief"], W["bills"] = 0.5 * rest, 0.5 * rest
    W["gold_phys"] = np.full(N, 0.15)
    return W


def book(W, mask, lag, cost):
    li = np.maximum.accumulate(np.where(mask, np.arange(N), -1))
    r = np.zeros(N)
    for c, w in W.items():
        h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
        h = np.r_[np.zeros(lag), h[:-lag]]
        r += h * RET[c] - np.abs(np.r_[0.0, np.diff(h)]) * (cost[c] if isinstance(cost, dict) else cost)
    return r


def tranche(W, lag, cost):
    return np.mean([book(W, DOW == i, lag, cost) for i in range(5)], axis=0)


TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
S = {}
for t, (lev, vt) in TIERS.items():
    for off in ("IEF", "5050"):
        W = my_weights(lev, vt, off)
        S[f"{t}-{off}|us"] = tranche(W, 1, 5e-4)
        S[f"{t}-{off}|ucits"] = tranche(W, 2, {lev: 15e-4, "ief": 8e-4, "bills": 8e-4, "gold_phys": 8e-4})
        if off == "IEF" and t == "F1":
            S["critic:F1-IEF lag2 8/8bp [tranche]"] = tranche(W, 2, {"qqq": 8e-4, "ief": 8e-4, "gold_phys": 8e-4})
S["QQQ B&H"] = P["qqq"].fillna(0).values.astype(float)
COLS = ["F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F1-IEF|ucits", "F2-IEF|ucits", "F3-IEF|ucits",
        "F1-5050|us", "F2-5050|us", "F3-5050|us", "F1-5050|ucits", "F2-5050|ucits", "F3-5050|ucits",
        "critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"]
CI = {c: i for i, c in enumerate(COLS)}


def window(a, b, cols=None):
    cols = COLS if cols is None else cols
    m = (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))
    X = np.vstack([S[c][m] for c in cols])
    d = IDX[m]
    return X, RFD[m], (d[-1] - d[0]).days / 365.25, d


def sb_indices(T, L, rng):
    """identical algorithm to the specialist's sb_indices (so seeded draws replay exactly)"""
    new = rng.random(T) < 1.0 / L
    new[0] = True
    pos = np.flatnonzero(new)
    src = rng.integers(0, T, size=pos.size)
    bid = np.cumsum(new) - 1
    return (src[bid] + np.arange(T) - pos[bid]) % T, pos, src


def bstats(X, rfv, yrs, ix):
    R = X[:, ix]
    e = np.cumprod(1.0 + R, axis=1)
    cagr = e[:, -1] ** (1.0 / yrs) - 1.0
    dd = e / np.maximum.accumulate(e, axis=1) - 1.0
    mdd = dd.min(axis=1)
    ulcer = np.sqrt((dd * dd).mean(axis=1))
    ex = R - rfv[ix][None, :]
    exsh = ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(252)
    rfc = float(np.prod(1.0 + rfv[ix]) ** (1.0 / yrs) - 1.0)   # CAGR of the resampled T-bill series
    return cagr, mdd, ulcer, exsh, rfc


def run_boot(X, rfv, yrs, L, reps, seed, keep_pos=False):
    rng = np.random.default_rng(seed)
    T = X.shape[1]
    D = {m: np.empty((reps, X.shape[0])) for m in ("cagr", "maxdd", "ulcer", "exsh")}
    rfc = np.empty(reps)
    extra = []
    for i in range(reps):
        ix, pos, src = sb_indices(T, L, rng)
        D["cagr"][i], D["maxdd"][i], D["ulcer"][i], D["exsh"][i], rfc[i] = bstats(X, rfv, yrs, ix)
        if keep_pos:
            extra.append((pos, src))
    D["calmar"] = D["cagr"] / np.abs(D["maxdd"])
    D["martin"] = D["cagr"] / D["ulcer"]
    D["xmartin"] = (D["cagr"] - rfc[:, None]) / D["ulcer"]          # Ulcer performance index on excess CAGR
    D["xcalmar"] = (D["cagr"] - rfc[:, None]) / np.abs(D["maxdd"])
    return D, extra


def pair(D, point, a, b, m, ci=None):
    ci = CI if ci is None else ci
    d = D[m][:, ci[a]] - D[m][:, ci[b]]
    return dict(sample=float(point[m][ci[a]] - point[m][ci[b]]), mean=float(d.mean()),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))


X88, rf88, y88, d88 = window("1988-01-01", FREEZE)
T88 = X88.shape[1]
pc, pm, pu, pe, rc = bstats(X88, rf88, y88, np.arange(T88))
POINT = dict(cagr=pc, maxdd=pm, ulcer=pu, exsh=pe, calmar=pc / np.abs(pm), martin=pc / pu,
             xmartin=(pc - rc) / pu, xcalmar=(pc - rc) / np.abs(pm))

# ------------------------------------------------------------------ G0 exact replay of saved IS88 bootstrap
g0 = {}
maxdiff = 0.0
nchk = 0
for L, seed in ((126, 137), (252, 263)):
    t1 = time.time()
    D, extra = run_boot(X88, rf88, y88, L, 5000, seed, keep_pos=(L == 252))
    sv = SAVED["bootstrap"]["IS88"][f"L{L}"]
    for k, e in sv["pairs"].items():
        a, b = k.split(" minus ")
        if a not in CI or b not in CI:
            continue
        for m in ("cagr", "calmar", "ulcer", "exsh"):
            mine = pair(D, POINT, a, b, m)
            for f_ in ("sample", "p_le0"):
                maxdiff = max(maxdiff, abs(mine[f_] - e[m][f_])); nchk += 1
            for j in (0, 1):
                maxdiff = max(maxdiff, abs(mine["ci95"][j] - e[m]["ci95"][j])); nchk += 1
            maxdiff = max(maxdiff, abs(mine["mean"] - e[m]["boot_mean"])); nchk += 1
    for c in COLS:
        own = sv["own"][c]["maxdd"]
        maxdiff = max(maxdiff, abs(float(np.median(D["maxdd"][:, CI[c]])) - own["median"]),
                      abs(float(np.percentile(D["maxdd"][:, CI[c]], 5)) - own["p5"]),
                      abs(float((D["maxdd"][:, CI[c]] < -0.5).mean()) - own["p_worse_than_-50pct"]))
        nchk += 3
    g0[f"L{L}"] = dict(qqq_maxdd_median=float(np.median(D["maxdd"][:, CI["QQQ B&H"]])),
                       qqq_maxdd_p5=float(np.percentile(D["maxdd"][:, CI["QQQ B&H"]], 5)),
                       runtime_s=round(time.time() - t1, 1))
    # Martin-ratio tier comparisons (not in the specialist's output)
    g0[f"L{L}"]["martin"] = {f"{a} minus {b}": {m: pair(D, POINT, a, b, m) for m in ("martin", "xmartin", "xcalmar", "exsh")} for a, b in
                             [("F2-IEF|us", "F1-IEF|us"), ("F2-IEF|us", "F3-IEF|us"), ("F3-IEF|us", "F1-IEF|us"),
                              ("F1-IEF|us", "QQQ B&H"), ("F2-IEF|us", "QQQ B&H"), ("F3-IEF|us", "QQQ B&H"),
                              ("F2-IEF|ucits", "F1-IEF|ucits")]}
    if L == 252:
        # G1 point mass for F2-IEF|us
        j = CI["F2-IEF|us"]
        real = float(POINT["maxdd"][j])
        eq = np.isclose(D["maxdd"][:, j], real, rtol=0, atol=1e-12)
        pk = int(np.searchsorted(d88.values, np.datetime64("2021-11-19")))
        tr = int(np.searchsorted(d88.values, np.datetime64("2022-10-20")))
        intact = 0
        for pos, src in extra:
            ends = np.r_[pos[1:], T88]
            lens = ends - pos
            # a block covers [pk, tr] if its source start s satisfies s <= pk and s + len - 1 >= tr (no wrap needed here)
            ok = (src <= pk) & (src + lens - 1 >= tr)
            intact += bool(ok.any())
        g0["G1_F2_point_mass"] = dict(realized=real, share_equal_realized=float(eq.mean()),
                                      share_episode_intact=intact / len(extra),
                                      share_worse_than_realized=float((D["maxdd"][:, j] < real - 1e-12).mean()),
                                      median=float(np.median(D["maxdd"][:, j])))
RES["G0_exact_replay"] = dict(n_checks=nchk, max_abs_diff_vs_saved=maxdiff, detail=g0,
                              note="saved JSON rounded to 7 dp; <= 5e-8 = exact replay")
print(f"G0 exact replay of saved IS88 bootstrap: {nchk} checks, max |diff| {maxdiff:.1e}")
print(f"   QQQ boot maxDD median L126 {g0['L126']['qqq_maxdd_median']*100:.1f}% L252 {g0['L252']['qqq_maxdd_median']*100:.1f}% (realized {POINT['maxdd'][CI['QQQ B&H']]*100:.1f}%)")
print(f"G1 F2-IEF|us L252: share of resamples with maxDD == realized {g0['G1_F2_point_mass']['share_equal_realized']*100:.1f}%, "
      f"2021-11-19..2022-10-20 inside one block {g0['G1_F2_point_mass']['share_episode_intact']*100:.1f}%, "
      f"worse than realized {g0['G1_F2_point_mass']['share_worse_than_realized']*100:.1f}%")
for L in (126, 252):
    for m in ("martin", "xmartin", "xcalmar"):
        print(f"G4 {m} L{L}: " + "; ".join(
            f"{k}: {v[m]['sample']:+.2f} [{v[m]['ci95'][0]:+.2f},{v[m]['ci95'][1]:+.2f}] P<=0 {v[m]['p_le0']:.3f}" for k, v in g0[f"L{L}"]["martin"].items()))
for m in ("martin", "xmartin", "xcalmar", "calmar"):
    print(f"   point {m}: " + ", ".join(f"{c} {POINT[m][CI[c]]:.3f}" for c in ("F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F1-IEF|ucits", "F2-IEF|ucits", "F3-IEF|ucits", "QQQ B&H")))

# ------------------------------------------------------------------ G2 alternative block lengths and seeds
KEYP = [("F1-IEF|us", "QQQ B&H"), ("F2-IEF|us", "QQQ B&H"), ("F3-IEF|us", "QQQ B&H"),
        ("F1-IEF|ucits", "QQQ B&H"), ("F2-IEF|ucits", "QQQ B&H"), ("F3-IEF|ucits", "QQQ B&H"),
        ("F1-5050|ucits", "QQQ B&H"), ("critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"),
        ("F2-IEF|us", "F1-IEF|us"), ("F2-IEF|us", "F3-IEF|us")]
G2C = ["F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F1-IEF|ucits", "F2-IEF|ucits", "F3-IEF|ucits", "F1-5050|ucits",
       "critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"]
CI2 = {c: i for i, c in enumerate(G2C)}
X2, rf2, y2, _ = window("1988-01-01", FREEZE, G2C)
pc2, pm2, pu2, pe2, rc2 = bstats(X2, rf2, y2, np.arange(X2.shape[1]))
PT2 = dict(cagr=pc2, maxdd=pm2, ulcer=pu2, exsh=pe2, calmar=pc2 / np.abs(pm2), martin=pc2 / pu2,
           xmartin=(pc2 - rc2) / pu2, xcalmar=(pc2 - rc2) / np.abs(pm2))
g2 = {}
print("\nG2 IS88 Calmar diff: P(<=0) and 95% CI by mean block length / seed (4,000 reps)")
for L, seed in ((63, 9001), (126, 9002), (126, 9003), (252, 9004), (252, 9005), (504, 9006)):   # L=1008 -> followup script
    t1 = time.time()
    D, _ = run_boot(X2, rf2, y2, L, 4000, seed)
    row = {f"{a} minus {b}": {m: pair(D, PT2, a, b, m, CI2) for m in ("calmar", "exsh", "cagr")} for a, b in KEYP}
    row["QQQ_maxdd_median"] = float(np.median(D["maxdd"][:, CI2["QQQ B&H"]]))
    row["F2us_maxdd_median_p5"] = [float(np.median(D["maxdd"][:, CI2["F2-IEF|us"]])), float(np.percentile(D["maxdd"][:, CI2["F2-IEF|us"]], 5))]
    row["F3us_maxdd_median_p5"] = [float(np.median(D["maxdd"][:, CI2["F3-IEF|us"]])), float(np.percentile(D["maxdd"][:, CI2["F3-IEF|us"]], 5))]
    row["F1us_maxdd_median_p5"] = [float(np.median(D["maxdd"][:, CI2["F1-IEF|us"]])), float(np.percentile(D["maxdd"][:, CI2["F1-IEF|us"]], 5))]
    g2[f"L{L}|seed{seed}"] = row
    print(f"  L={L:4d} seed {seed} ({time.time()-t1:4.1f}s) QQQ medDD {row['QQQ_maxdd_median']*100:5.1f} | " + " | ".join(
        f"{a.split('|')[0][:7]}{'' if 'QQQ' in b else '-' + b.split('|')[0][:2]}{'E' if 'ucits' in a else ('c' if 'critic' in a else 'U')} "
        f"{row[f'{a} minus {b}']['calmar']['p_le0']:.3f}[{row[f'{a} minus {b}']['calmar']['ci95'][0]:+.2f},{row[f'{a} minus {b}']['calmar']['ci95'][1]:+.2f}]"
        for a, b in KEYP))
RES["G2_block_length_seed"] = g2

# G3 (start-date sensitivity) moved to r2_verify_candidates_tranche_robustness_followup.py (runtime)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


json.dump(clean(RES), open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}  runtime {time.time()-T0:.0f}s")
