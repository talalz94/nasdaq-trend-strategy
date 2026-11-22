"""
INTEGRITY REVIEW of r2_candidates_tranche - bootstrap checks (read-only w.r.t. project files; writes one JSON to the
scratch dir given on the command line).

  1. Series are rebuilt with an independent engine (searchsorted held weights, own weights from the brief).
  2. EXACT regeneration of the specialist's stationary-bootstrap draws (same seeds, same RNG call order) applied to
     the independently built series -> must reproduce the saved CIs / P / maxDD quantiles; and the C15 point-mass
     claim (share of L=252 resamples whose F2-IEF|us maxDD equals the realized one, and whether those resamples
     contain the 2021-11..2022-10 episode intact).
  3. An INDEPENDENT stationary-bootstrap implementation (explicit geometric block lengths, uniform circular starts)
     with different seeds -> Monte-Carlo robustness of the headline P values / CIs.
  4. Drawdown-episode dates for the claims that cite '2022 episode'.
usage: python -u update_2026_11/r2_verify_candidates_tranche_integrity_boot.py <scratch_dir>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = sys.argv[1]
os.makedirs(SCR, exist_ok=True)
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
SAVED = json.load(open(os.path.join(HERE, "r2_candidates_tranche.json")))
START, FREEZE = pd.Timestamp("1988-01-01"), pd.Timestamp("2026-07-24")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
RF = P["rf_d"].fillna(0)


def pr(*a):
    print(*a, flush=True)


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


def book(W, mask, lag, cost):
    n = len(P)
    mpos = np.flatnonzero(mask)
    k = np.searchsorted(mpos, np.arange(n) - lag, side="right") - 1
    valid = k >= 0
    src = np.where(valid, mpos[np.maximum(k, 0)], 0)
    g = np.zeros(n); tc = np.zeros(n)
    for c, w in W.items():
        h = np.where(valid, w[src], 0.0)
        g += h * P[c].fillna(0).values
        dh = np.zeros(n); dh[1:] = np.abs(np.diff(h))
        tc += dh * (cost[c] if isinstance(cost, dict) else cost)
    return g - tc


def tranche(W, lag, cost):
    dow = P.index.dayofweek.values
    return pd.Series(sum(book(W, dow == i, lag, cost) for i in range(5)) / 5.0, P.index)


TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
SER = {}
for t, (lev, vt) in TIERS.items():
    for off in ("IEF", "5050"):
        W = weights(lev, vt, off)
        SER[f"{t}-{off}|us"] = tranche(W, 1, 5e-4)
        SER[f"{t}-{off}|ucits"] = tranche(W, 2, {lev: 15e-4, "ief": 8e-4, "bills": 8e-4, "gold_phys": 8e-4})
SER["critic:F1-IEF lag2 8/8bp [tranche]"] = tranche(weights("qqq", 0.20, "IEF"), 2, {"qqq": 8e-4, "ief": 8e-4, "gold_phys": 8e-4})
SER["QQQ B&H"] = P["qqq"].fillna(0)
COLS = list(SER)
CI = {c: i for i, c in enumerate(COLS)}
WIN = {"IS88": ("1988-01-01", FREEZE), "IS00": ("2000-01-01", FREEZE), "IS07": ("2007-01-01", FREEZE)}
OUT = {}


# ------------------------------------------------------------------ 4. drawdown episodes (IS88)
def dd_episode(r):
    s = r[(r.index >= START) & (r.index <= FREEZE)]
    e = (1 + s).cumprod()
    dd = e / e.cummax() - 1
    tr = dd.idxmin()
    pk = e[:tr].idxmax()
    rec = e[tr:][e[tr:] >= e[pk]]
    return dict(maxdd=float(dd.min()), peak=str(pk.date()), trough=str(tr.date()),
                recovered=str(rec.index[0].date()) if len(rec) else None)


OUT["episodes"] = {c: dd_episode(SER[c]) for c in COLS}
pr("episodes", json.dumps(OUT["episodes"], indent=0))


# ------------------------------------------------------------------ bootstrap helpers
def prep(wn):
    a, b = WIN[wn]
    m = (P.index >= max(pd.Timestamp(a), START)) & (P.index <= b)
    X = np.vstack([SER[c].values[m] for c in COLS])
    rfv = RF.values[m]
    d = P.index[m]
    yrs = (d[-1] - d[0]).days / 365.25
    return X, rfv, yrs, d


def bst(X, rfv, yrs, ix):
    R = X[:, ix]
    e = np.cumprod(1 + R, axis=1)
    cagr = e[:, -1] ** (1 / yrs) - 1
    dd = e / np.maximum.accumulate(e, axis=1) - 1
    mdd = dd.min(axis=1)
    ul = np.sqrt((dd * dd).mean(axis=1))
    ex = R - rfv[ix][None, :]
    sh = ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(252)
    return cagr, mdd, ul, sh


def spec_draw(T, L, rng):          # exact copy of the specialist's RNG call order (for regeneration only)
    new = rng.random(T) < 1.0 / L
    new[0] = True
    pos = np.flatnonzero(new)
    src = rng.integers(0, T, size=pos.size)
    bid = np.cumsum(new) - 1
    return (src[bid] + np.arange(T) - pos[bid]) % T


def my_draw(T, L, rng):            # independent implementation: geometric lengths, uniform circular starts
    out = np.empty(T, dtype=np.int64)
    i = 0
    while i < T:
        ln = int(rng.geometric(1.0 / L))
        st = int(rng.integers(0, T))
        k = min(ln, T - i)
        out[i:i + k] = (st + np.arange(k)) % T
        i += k
    return out


PAIRS = [(f"{t}-IEF|{b}", "QQQ B&H") for b in ("us", "ucits") for t in TIERS] + \
        [(f"{t}-5050|us", "QQQ B&H") for t in TIERS] + \
        [("critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"),
         ("F2-IEF|us", "F3-IEF|us"), ("F2-IEF|us", "F1-IEF|us"), ("F2-IEF|ucits", "F3-IEF|ucits"),
         ("F1-5050|us", "F1-IEF|us"), ("F2-5050|us", "F2-IEF|us"), ("F3-5050|us", "F3-IEF|us")]


def run_boot(wn, L, seed, drawer, reps=5000, want_draw_info=False):
    X, rfv, yrs, d = prep(wn)
    T = X.shape[1]
    rng = np.random.default_rng(seed)
    acc = {m: np.empty((reps, len(COLS))) for m in ("cagr", "maxdd", "ulcer", "exsh")}
    intact = np.zeros(reps, dtype=bool)
    if want_draw_info:
        pk = int(np.flatnonzero(d == pd.Timestamp(OUT["episodes"]["F2-IEF|us"]["peak"]))[0])
        tr = int(np.flatnonzero(d == pd.Timestamp(OUT["episodes"]["F2-IEF|us"]["trough"]))[0])
    for i in range(reps):
        ix = drawer(T, L, rng)
        acc["cagr"][i], acc["maxdd"][i], acc["ulcer"][i], acc["exsh"][i] = bst(X, rfv, yrs, ix)
        if want_draw_info:
            # does some position p hold ix[p]=pk and ix[p+(tr-pk)]=tr with a contiguous run in between?
            ps = np.flatnonzero(ix == pk)
            ok = False
            for p in ps:
                q = p + (tr - pk)
                if q < T and np.all(np.diff(ix[p:q + 1]) == 1):
                    ok = True; break
            intact[i] = ok
    acc["calmar"] = acc["cagr"] / np.abs(acc["maxdd"])
    pt = bst(X, rfv, yrs, np.arange(T))
    point = dict(cagr=pt[0], maxdd=pt[1], ulcer=pt[2], exsh=pt[3], calmar=pt[0] / np.abs(pt[1]))
    res = {}
    for a, b in PAIRS:
        i1, i2 = CI[a], CI[b]
        e = {}
        for m in ("cagr", "calmar", "ulcer", "exsh"):
            dlt = acc[m][:, i1] - acc[m][:, i2]
            e[m] = [float(point[m][i1] - point[m][i2]), float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5)),
                    float((dlt <= 0).mean()), float(dlt.mean())]
        res[f"{a} minus {b}"] = e
    own = {c: dict(maxdd_med=float(np.median(acc["maxdd"][:, CI[c]])), maxdd_p5=float(np.percentile(acc["maxdd"][:, CI[c]], 5)),
                   p_lt_50=float((acc["maxdd"][:, CI[c]] < -0.5).mean())) for c in COLS}
    extra = {}
    if want_draw_info:
        j = CI["F2-IEF|us"]
        real = float(point["maxdd"][j])
        tie = np.isclose(acc["maxdd"][:, j], real, rtol=0, atol=1e-10)
        extra = dict(realized=real, share_equal_realized=float(tie.mean()), share_episode_intact=float(intact.mean()),
                     share_tie_given_intact=float(tie[intact].mean()) if intact.any() else None,
                     share_strictly_worse=float((acc["maxdd"][:, j] < real - 1e-10).mean()),
                     share_strictly_better=float((acc["maxdd"][:, j] > real + 1e-10).mean()))
    return dict(pairs=res, own=own, extra=extra)


def cmp_saved(wn, L, got):
    sv = SAVED["bootstrap"][wn][f"L{L}"]
    worst = 0.0
    for k, e in got["pairs"].items():
        s = sv["pairs"][k]
        for m in ("cagr", "calmar", "ulcer", "exsh"):
            worst = max(worst, abs(e[m][0] - s[m]["sample"]), abs(e[m][1] - s[m]["ci95"][0]), abs(e[m][2] - s[m]["ci95"][1]),
                        abs(e[m][3] - s[m]["p_le0"]), abs(e[m][4] - s[m]["boot_mean"]))
    for c, o in got["own"].items():
        s = sv["own"][c]["maxdd"]
        worst = max(worst, abs(o["maxdd_med"] - s["median"]), abs(o["maxdd_p5"] - s["p5"]),
                    abs(o["p_lt_50"] - s["p_worse_than_-50pct"]))
    return worst


SEEDS = {"IS88": 11, "IS00": 22, "IS07": 33}
RUN_W = sys.argv[2].split(",") if len(sys.argv) > 2 else ["IS88", "IS07"]
RUN_L = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [126, 252]
for wn in RUN_W:
    for L in RUN_L:
        t1 = time.time()
        ex = run_boot(wn, L, SEEDS[wn] + L, spec_draw, want_draw_info=(wn == "IS88" and L == 252))
        OUT[f"exact|{wn}|L{L}"] = dict(max_abs_diff_vs_saved=cmp_saved(wn, L, ex), extra=ex["extra"],
                                      pairs=ex["pairs"], own=ex["own"])
        pr(f"exact {wn} L{L}: max|diff| vs saved {OUT[f'exact|{wn}|L{L}']['max_abs_diff_vs_saved']:.2e}  {time.time()-t1:.0f}s  {ex['extra']}")
        t1 = time.time()
        ind = run_boot(wn, L, 900000 + L + {"IS88": 0, "IS00": 3, "IS07": 7}[wn], my_draw)
        OUT[f"indep|{wn}|L{L}"] = dict(pairs=ind["pairs"], own=ind["own"])
        pr(f"indep {wn} L{L}: {time.time()-t1:.0f}s")
        for k in ind["pairs"]:
            a, b = ex["pairs"][k], ind["pairs"][k]
            pr(f"   {k:48s} Calmar s {a['calmar'][0]:+.3f} exact[{a['calmar'][1]:+.3f},{a['calmar'][2]:+.3f}] P {a['calmar'][3]:.3f}"
               f" | indep[{b['calmar'][1]:+.3f},{b['calmar'][2]:+.3f}] P {b['calmar'][3]:.3f} || CAGR P {a['cagr'][3]:.3f}/{b['cagr'][3]:.3f}"
               f" exSh P {a['exsh'][3]:.3f}/{b['exsh'][3]:.3f} Ulcer P {a['ulcer'][3]:.3f}/{b['ulcer'][3]:.3f}")
        for c in COLS:
            pr(f"   own {c:40s} mdd med {ex['own'][c]['maxdd_med']*100:6.2f}/{ind['own'][c]['maxdd_med']*100:6.2f}"
               f" p5 {ex['own'][c]['maxdd_p5']*100:6.2f}/{ind['own'][c]['maxdd_p5']*100:6.2f}"
               f" P<-50 {ex['own'][c]['p_lt_50']:.4f}/{ind['own'][c]['p_lt_50']:.4f}")
OUT["runtime_s"] = round(time.time() - T0, 1)
json.dump(OUT, open(os.path.join(SCR, f"integrity_boot_{'_'.join(RUN_W)}_{'_'.join(map(str, RUN_L))}.json"), "w"), indent=1)
pr("done", OUT["runtime_s"])
