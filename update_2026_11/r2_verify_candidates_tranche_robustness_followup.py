"""
r2_verify_candidates_tranche_robustness_followup.py -- follow-up robustness checks for r2_candidates_tranche.

Read-only w.r.t. project files; writes only the scratch JSON given on the command line.
Reuses the engine of r2_verify_candidates_tranche_robustness_boot.py (imported by exec of its top part only).

  G3  start-date sensitivity of the Calmar edge vs QQQ ("significant from 2000"), L=252, 2,000 reps
  G2b mean block length 1008 (about the dot-com drawdown length), 2,000 reps
  R1  independent replay of the critic's circular 63-day bootstrap (seed 0, 2,000 reps) for the two
      pairs behind its 'only F1 survives' conclusion, plus seeds 1..4 (Monte Carlo fragility of that verdict)
  X1  average NDX exposure per tier (how much of the planning CAGR is the NDX drift)

usage: python -u update_2026_11/r2_verify_candidates_tranche_robustness_followup.py <scratch_out.json>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1])
assert os.path.dirname(OUT) != HERE
src = open(os.path.join(HERE, "r2_verify_candidates_tranche_robustness_boot.py"), encoding="utf-8").read()
cut = src.index("# ------------------------------------------------------------------ G0 exact replay")
g = {"__file__": os.path.join(HERE, "r2_verify_candidates_tranche_robustness_boot.py"), "__name__": "__lib__"}
sys.argv = [sys.argv[0], OUT]
exec(compile(src[:cut], "boot_engine", "exec"), g)
S, IDX, FREEZE, START, P = g["S"], g["IDX"], g["FREEZE"], g["START"], g["P"]
window, bstats, run_boot, pair = g["window"], g["bstats"], g["run_boot"], g["pair"]
T0 = time.time()
RES = {}

# ------------------------------------------------------------------ R1 critic 63-day circular bootstrap replay
crit = json.load(open(os.path.join(HERE, "critic_out_critic_candidates.json")))["bootstrap_IS"]


def critic_boot(r1, r2, n=2000, block=63, seed=0):
    m = (IDX >= START) & (IDX <= FREEZE)
    a = np.column_stack([r1[m], r2[m]])
    T = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    dk, dd1, dd2 = [], [], []
    for _ in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = a[ix]
        e = np.cumprod(1 + s, axis=0)
        gg = e[-1] ** (252 / T) - 1
        mdd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = gg / np.abs(mdd)
        dk.append(cal[0] - cal[1]); dd1.append(mdd[0]); dd2.append(mdd[1])
    dk = np.array(dk)
    return dict(dcalmar=[float(dk.mean()), float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5))],
                p=float((dk <= 0).mean()), maxdd_median=[float(np.median(dd1)), float(np.median(dd2))])


r1 = {}
print("R1 critic circular 63-day bootstrap (2,000 reps): Calmar-edge mean [CI] P(<=0) | maxDD medians")
for name, key, ser in [("critic F1 8/8bp tranche", "F1-IEF lag2 + UCITS-level costs 8/8bp [tranche] minus QQQ", S["critic:F1-IEF lag2 8/8bp [tranche]"]),
                       ("F2 UCITS 15/8bp tranche", "F2-IEF lag2 + UCITS-level costs 15/8bp [tranche] minus QQQ", S["F2-IEF|ucits"])]:
    r1[name] = {}
    for seed in range(5):
        b = critic_boot(ser, S["QQQ B&H"], seed=seed)
        r1[name][f"seed{seed}"] = b
        extra = ""
        if seed == 0:
            ref = crit[key]
            extra = (f"  | critic JSON {ref['dcalmar']} P {ref['p_dcalmar_le0']} maxDD med {ref['maxdd_median']}"
                     f" -> max|diff| {max(abs(b['dcalmar'][i] - ref['dcalmar'][i]) for i in range(3)):.1e}")
        print(f"  {name:24s} seed {seed}: {b['dcalmar'][0]:+.3f} [{b['dcalmar'][1]:+.3f},{b['dcalmar'][2]:+.3f}] P {b['p']:.4f}"
              f" | medDD {b['maxdd_median'][0]*100:.1f}/{b['maxdd_median'][1]*100:.1f}{extra}")
RES["R1_critic_63d"] = r1

# ------------------------------------------------------------------ G2b L = 1008
KEYP = [("F1-IEF|us", "QQQ B&H"), ("F2-IEF|us", "QQQ B&H"), ("F3-IEF|us", "QQQ B&H"),
        ("F1-IEF|ucits", "QQQ B&H"), ("F2-IEF|ucits", "QQQ B&H"), ("F3-IEF|ucits", "QQQ B&H"),
        ("critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"), ("F2-IEF|us", "F1-IEF|us"), ("F2-IEF|us", "F3-IEF|us")]
C2 = ["F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F1-IEF|ucits", "F2-IEF|ucits", "F3-IEF|ucits",
      "critic:F1-IEF lag2 8/8bp [tranche]", "QQQ B&H"]
CI2 = {c: i for i, c in enumerate(C2)}
X, rfv, yrs, _ = window("1988-01-01", FREEZE, C2)
pc, pm, pu, pe, rc = bstats(X, rfv, yrs, np.arange(X.shape[1]))
PT = dict(cagr=pc, maxdd=pm, ulcer=pu, exsh=pe, calmar=pc / np.abs(pm))
t1 = time.time()
D, _ = run_boot(X, rfv, yrs, 1008, 2000, 9008)
g2b = {f"{a} minus {b}": pair(D, PT, a, b, "calmar", CI2) for a, b in KEYP}
g2b["QQQ_maxdd_median"] = float(np.median(D["maxdd"][:, CI2["QQQ B&H"]]))
RES["G2b_L1008"] = g2b
print(f"\nG2b L=1008 ({time.time()-t1:.0f}s): QQQ medDD {g2b['QQQ_maxdd_median']*100:.1f} | " +
      " | ".join(f"{k.replace(' minus ', '-').replace('QQQ B&H', 'Q')}: {v['p_le0']:.3f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" for k, v in g2b.items() if k != "QQQ_maxdd_median"))

# ------------------------------------------------------------------ G3 start-date sensitivity
g3 = {}
C3 = ["F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F2-IEF|ucits", "QQQ B&H"]
CI3 = {c: i for i, c in enumerate(C3)}
print("\nG3 Calmar edge vs QQQ by start date (to 2026-07-24), L=252, 2,000 reps: sample / P(<=0)")
for st in ("1997-01-01", "1999-01-01", "2000-01-01", "2000-03-27", "2001-01-01", "2002-10-10", "2003-01-01", "2005-01-01"):
    X, rfv, yrs, d = window(st, FREEZE, C3)
    pc_, pm_, pu_, pe_, rc_ = bstats(X, rfv, yrs, np.arange(X.shape[1]))
    pt = dict(cagr=pc_, maxdd=pm_, ulcer=pu_, exsh=pe_, calmar=pc_ / np.abs(pm_))
    D, _ = run_boot(X, rfv, yrs, 252, 2000, 7000 + int(st[:4]) + int(st[5:7]) + int(st[8:10]))
    row = {a: pair(D, pt, a, "QQQ B&H", "calmar", CI3) for a in ("F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F2-IEF|ucits")}
    row["QQQ"] = dict(cagr=float(pc_[CI3["QQQ B&H"]]), maxdd=float(pm_[CI3["QQQ B&H"]]), calmar=float(pt["calmar"][CI3["QQQ B&H"]]))
    g3[st] = row
    print(f"  from {st}: QQQ {row['QQQ']['cagr']*100:5.2f}/{row['QQQ']['maxdd']*100:6.1f}/{row['QQQ']['calmar']:.2f} | " +
          " | ".join(f"{a} {row[a]['sample']:+.2f} [{row[a]['ci95'][0]:+.2f},{row[a]['ci95'][1]:+.2f}] P {row[a]['p_le0']:.3f}"
                     for a in ("F1-IEF|us", "F2-IEF|us", "F3-IEF|us", "F2-IEF|ucits")))
RES["G3_start_date"] = g3

# ------------------------------------------------------------------ X1 average NDX exposure
VOL, ER, ABOVE, MON, DOW, N = g["VOL"], g["ER"], g["ABOVE"], g["MON"], g["DOW"], g["N"]
x1 = {}
m = (IDX >= START) & (IDX <= FREEZE)
for t, (lev, vt, k) in {"F1": ("qqq", 0.20, 1), "F2": ("qld", 0.30, 2), "F3": ("tqqq", 0.20, 3)}.items():
    W = g["my_weights"](lev, vt, "IEF")
    hs = []
    for i in range(5):
        li = np.maximum.accumulate(np.where(DOW == i, np.arange(N), -1))
        h = np.where(li >= 0, W[lev][np.maximum(li, 0)], 0.0)
        hs.append(np.r_[0.0, h[:-1]])
    h = np.mean(hs, axis=0)
    x1[t] = dict(mean_sleeve_weight=float(h[m].mean()), mean_ndx_exposure=float(k * h[m].mean()),
                 share_days_invested=float((h[m] > 0).mean()))
print("\nX1 average held sleeve weight / NDX exposure (tranche, IS88): " +
      ", ".join(f"{t} {v['mean_sleeve_weight']:.2f} / {v['mean_ndx_exposure']:.2f}x (invested {v['share_days_invested']*100:.0f}% of days)" for t, v in x1.items()))
RES["X1_exposure"] = x1


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


json.dump(clean(RES), open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}  runtime {time.time()-T0:.0f}s")
