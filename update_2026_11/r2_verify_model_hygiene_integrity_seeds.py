"""
INTEGRITY REVIEW of r2_model_hygiene.py -- information-only Monte-Carlo check of the flipped bootstrap item
R5|F2|calmar_ci_excl0 (tranche): Calmar(F2 tranche) - Calmar(QQQ) 95% CI lower bound over seeds 0..19, same scheme
(63-day circular blocks, 2000 draws, IS 1988-01-01..FREEZE, rng.integers called once per draw as in the original).
Needs <scratch_dir>/verify_series.npz from r2_verify_model_hygiene_integrity_checks.py. Prints only.
usage: python -u update_2026_11/r2_verify_model_hygiene_integrity_seeds.py <scratch_dir> <panel> [<panel> ...]
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
scratch = sys.argv[1]
panels = sys.argv[2:]
Z = np.load(os.path.join(scratch, "verify_series.npz"))
P = C.panel()
IDX = P.index
M = (IDX >= C.START) & (IDX <= C.FREEZE)
QQQ = P["qqq"].values[M]
N_DRAW, BLOCK, CHUNK = 2000, 63, 100
for pn in panels:
    X = np.column_stack([Z[f"{pn}|tranche|F2"][M], QQQ])
    Lg = np.log1p(X)
    T = len(X)
    nb = int(np.ceil(T / BLOCK))
    ar = np.arange(BLOCK)[None, :]
    los, ps = [], []
    for sd in range(20):
        rng = np.random.default_rng(sd)
        v = []
        for c0 in range(0, N_DRAW, CHUNK):
            ixs = np.stack([(rng.integers(0, T, nb)[:, None] + ar).ravel()[:T] % T for _ in range(min(CHUNK, N_DRAW - c0))])
            cl = np.cumsum(Lg[ixs], axis=1)
            g = np.expm1(cl[:, -1, :] * 252 / T)
            dd = np.expm1((cl - np.maximum.accumulate(cl, axis=1)).min(axis=1))
            cal = g / np.abs(dd)
            v.append(cal[:, 0] - cal[:, 1])
        v = np.concatenate(v)
        los.append(float(np.percentile(v, 2.5)))
        ps.append(float((v > 0).mean()))
    los = np.array(los)
    print(f"[{time.time()-T0:6.1f}s] {pn:8s} tranche F2 Calmar edge vs QQQ: CI excludes 0 in {int((los > 0).sum())}/20 seeds "
          f"(seeds 0-4: {int((los[:5] > 0).sum())}/5); lo95 mean {los.mean():+.4f} sd {los.std(ddof=1):.4f} min {los.min():+.4f} "
          f"max {los.max():+.4f}; P(edge>0) mean {np.mean(ps):.4f}; lo95 seeds 0-4 {np.round(los[:5], 4).tolist()}", flush=True)
    np.save(os.path.join(scratch, f"seeds_lo95_{pn}.npy"), los)
