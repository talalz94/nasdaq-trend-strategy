"""
verify_analogs_integrity_extra.py -- robustness of two inferential claims in analogs.py
  (a) kNN 'macro state has no close analog': today's nearest/median distance ratio vs the same ratio
      for EVERY in-sample risk-off query date (exclusion windows 82 and 252 days), K1/K2/K3
  (b) null-test percentiles near the 95 boundary: re-run with 5 seeds x 4000 reps
Independent of analogs.py; reads panel.parquet + analogs.json only. No project files written.
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
J = json.load(open(os.path.join(HERE, "analogs.json")))
sys.path.insert(0, HERE)
import common as C
idx = P.index; N = len(idx); T0 = N - 1
PF = int(idx.searchsorted(C.FREEZE, side="right") - 1); PS = int(idx.searchsorted(C.START))
ndx = P["ndx"].values
ABOVE = P["above"].values.astype(bool)
wF3 = C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_phys", off="ief")
wF2 = C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", off="ief")
ON = (wF3.where(P["mark"]).ffill().fillna(0)["tqqq"] > 0).values
f3 = C.backtest(P, wF3).values; f2 = C.backtest(P, wF2).values
ins = np.zeros(N, dtype=bool); ins[PS:PF + 1] = True
ELIG = ins & ~ON & ~ABOVE


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
H52 = pd.Series(ndx, index=idx).rolling(252, min_periods=1).max().values
F = pd.DataFrame(index=idx)
F["dd52"] = ndx / H52 - 1; F["lage"] = np.log1p(age); F["lvol"] = np.log(P["vol20"]); F["er20"] = P["er20"]
tnx = P["tnx"].ffill(); F["tnx"] = tnx; F["tnx_chg126"] = tnx - tnx.shift(126)
F["curve"] = (P["tnx"] - P["irx"]).ffill()
dxy = P["dxy"].ffill(); F["dxy_chg126"] = dxy / dxy.shift(126) - 1
F["lvix"] = np.log(P["vix"].ffill())
gw = (1 + P["gold_phys"].fillna(0)).cumprod(); g63 = gw / gw.shift(63) - 1
g63[(P["gold_src"] == "monthly-interp").astype(float).rolling(63, min_periods=1).max() > 0] = np.nan
F["gold63"] = g63
Z = (F - F[ins].mean()) / F[ins].std()
specs = {"K1": ["dd52", "lage", "lvol", "er20"],
         "K2": ["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix"],
         "K2_macro_only": ["tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix"],
         "K3": ["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix", "gold63"],
         "K3_minus_gold": ["dd52", "lage", "lvol", "er20", "tnx", "tnx_chg126", "curve", "dxy_chg126", "lvix"]}
res = {}
for nm, fs in specs.items():
    X = Z[fs].values
    ok = ELIG & ~np.isnan(X).any(axis=1)
    if nm == "K3_minus_gold":                      # same candidate dates as K3 (2000-11+), gold dropped
        ok = ok & ~np.isnan(Z["gold63"].values)
    cand = np.where(ok)[0]
    q = X[T0]
    dt = np.sqrt(((X[cand] - q) ** 2).sum(axis=1))
    tr = dt.min() / np.median(dt)
    D = np.sqrt(((X[cand][:, None, :] - X[cand][None, :, :]) ** 2).sum(axis=2))
    row = dict(n_cand=int(len(cand)), today_nearest=round(float(dt.min()), 3), today_median=round(float(np.median(dt)), 3),
               today_ratio=round(float(tr), 3))
    for excl in (82, 252):
        rat = []
        for a_, qi in enumerate(cand):
            m = np.abs(cand - qi) > excl
            if m.sum() < 50:
                continue
            d = D[a_, m]
            rat.append(d.min() / np.median(d))
        rat = np.array(rat)
        row[f"excl{excl}"] = dict(n_queries=int(len(rat)), ratio_median=round(float(np.median(rat)), 3),
                                  ratio_p75=round(float(np.percentile(rat, 75)), 3),
                                  ratio_p90=round(float(np.percentile(rat, 90)), 3),
                                  today_pctile=round(float((rat < tr).mean() * 100), 1))
    res[nm] = row
    print(nm, row, flush=True)

# (b) null-test percentile stability for the sets nearest the 95 boundary
POOL = np.where(ELIG)[0]; POOL = POOL[POOL + 252 <= PF]


def fwd(r, i, h=252):
    return float(np.prod(1 + r[i + 1:i + h + 1]) - 1)


pool_f2 = np.array([fwd(f2, i) for i in POOL]); pool_f3 = np.array([fwd(f3, i) for i in POOL])


def decl(rng, n):
    perm = rng.permutation(len(POOL)); picked = []
    for j in perm:
        p = POOL[j]
        if all(abs(p - POOL[q]) >= 60 for q in picked):
            picked.append(j)
            if len(picked) >= n:
                break
    return picked


stab = {}
for nm in ["R1", "R1a", "R3", "K2"]:
    eps = [e for e in J["variants"][nm]["episodes"] if e["F2_12m"] is not None]
    obs2 = np.median([e["F2_12m"] for e in eps]); obs3 = np.median([e["F3_12m"] for e in eps])
    out2, out3 = [], []
    for seed in range(5):
        rng = np.random.default_rng(1000 + seed)
        s2, s3 = [], []
        for _ in range(4000):
            js = decl(rng, len(eps))
            s2.append(np.median(pool_f2[js])); s3.append(np.median(pool_f3[js]))
        out2.append(float((np.array(s2) < obs2).mean() * 100)); out3.append(float((np.array(s3) < obs3).mean() * 100))
    stab[nm] = dict(n=len(eps), F2_pctiles=[round(v, 1) for v in out2], F3_pctiles=[round(v, 1) for v in out3])
    print(nm, stab[nm], flush=True)

SCR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", HERE)
json.dump(dict(knn=res, null_stability=stab), open(os.path.join(SCR, "verify_analogs_integrity_extra.json"), "w"), indent=1)
