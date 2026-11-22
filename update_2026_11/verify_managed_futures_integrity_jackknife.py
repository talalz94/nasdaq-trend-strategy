"""
verify_managed_futures_integrity_jackknife.py  (adversarial review, integrity lens)

Leave-one-calendar-year-out jackknife of the Calmar deltas behind C2/C4/C9: for each year Y in the
in-sample window, set BOTH series' returns in Y to zero (neutralise the year, no splicing) and
recompute dCalmar over 1988-01-01..2026-07-24. If a verdict flips only when one year is removed, it
rests on that single episode. Also: paired bootstrap of TSMOM-sleeve minus T-bill-sleeve on the
post-1990-05 window (sensitivity only). Uses series pickled by ..._rerun.py; prints only.
"""
import os, sys, pickle
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "update_2026_11")
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = sys.argv[1] if len(sys.argv) > 1 else "."
S = pickle.load(open(os.path.join(SCR, "mf_series.pkl"), "rb"))
P = C.panel()
FREEZE = C.FREEZE
allp = {**S["port"], **S["rob"]}

def calmar(r):
    return C.stats(r, b=FREEZE)["calmar"]

def jack(r1, r2):
    out = {}
    for y in range(1988, 2027):
        m = r1.index.year == y
        a = r1.copy(); b = r2.copy(); a[m] = 0.0; b[m] = 0.0
        out[y] = calmar(a) - calmar(b)
    return pd.Series(out)

pairs = {}
for eng in ["F3", "F2"]:
    b = S["base"][eng]
    for pk in ["main_12m", "blend_1_3_12m"]:
        pairs[f"{eng} offleg {pk} vs base"] = (S["port"][f"{eng} | replace_ief_offleg | {pk}"], b)
        pairs[f"{eng} sleeve20 {pk} vs base"] = (S["port"][f"{eng} | sleeve20 | {pk}"], b)
        pairs[f"{eng} sleeve20 {pk} vs T-bill sleeve20"] = (S["port"][f"{eng} | sleeve20 | {pk}"],
                                                          S["cash"][f"{eng} | sleeve20 | CONTROL bills (plain de-lever)"])
print("=== leave-one-year-out dCalmar (IS 1988..2026-07-24); full = no year removed ===")
for k, (r1, r2) in pairs.items():
    full = calmar(r1) - calmar(r2)
    j = jack(r1, r2)
    flips = [int(y) for y, v in j.items() if np.sign(v) != np.sign(full)]
    print(f"  {k:42s} full {full:+.3f}  LOYO range [{j.min():+.3f} ({j.idxmin()}), {j.max():+.3f} ({j.idxmax()})]  "
          f"years whose removal flips the sign: {flips}")

# all 22 off-leg variants: removing 1990 only / 1989+1990
print("\n=== off-leg swap, all 22 proxy variants: dCalmar with 1990 (and 1989-90) neutralised ===")
for eng in ["F3", "F2"]:
    b = S["base"][eng]
    rows = {k: v for k, v in allp.items() if k.startswith(f"{eng} | replace_ief_offleg | ") and not k.endswith("control_always_long")}
    for lbl, yrs in (("none", []), ("1990", [1990]), ("1989-90", [1989, 1990])):
        d = []
        for v in rows.values():
            a = v.copy(); bb = b.copy()
            m = a.index.year.isin(yrs)
            a[m] = 0.0; bb[m] = 0.0
            d.append(calmar(a) - calmar(bb))
        d = np.array(d)
        print(f"  {eng} neutralised {lbl:8s}: #dCal<0 {int((d<0).sum())}/22  median {np.median(d):+.3f}  range [{d.min():+.3f},{d.max():+.3f}]")

# bootstrap TSMOM sleeve minus same-size T-bill sleeve on 1990-06-01..freeze
def boot(r1, r2, a, n=2000, block=63, seed=0):
    x = pd.concat([r1, r2], axis=1).dropna()
    x = x[(x.index >= pd.Timestamp(a)) & (x.index <= FREEZE)].values
    T = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); ar = np.arange(block)
    res = np.empty((n, 2))
    for k in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + ar[None, :]).ravel()[:T] % T
        s = x[ix]; e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        c = g / np.abs(dd); res[k] = (c[0] - c[1], g[0] - g[1])
    return [(float(np.percentile(res[:, j], 2.5)), float(np.percentile(res[:, j], 97.5)), float((res[:, j] <= 0).mean())) for j in (0, 1)]

print("\n=== TSMOM sleeve minus same-size T-bill sleeve, window 1990-06-01..2026-07-24 (sensitivity) ===")
for eng in ["F3", "F2"]:
    for s in (10, 20, 30):
        for pk in ["main_12m"]:
            r1 = S["port"][f"{eng} | sleeve{s} | {pk}"]; r2 = S["cash"][f"{eng} | sleeve{s} | CONTROL bills (plain de-lever)"]
            d = C.stats(r1, "1990-06-01", FREEZE, start=None)["calmar"] - C.stats(r2, "1990-06-01", FREEZE, start=None)["calmar"]
            m = (r1.index >= pd.Timestamp("1990-06-01")) & (r1.index <= FREEZE)
            x1 = (r1 - P["rf_d"])[m]; x2 = (r2 - P["rf_d"])[m]
            dsh = x1.mean() / x1.std() * np.sqrt(252) - x2.mean() / x2.std() * np.sqrt(252)
            (cl, ch, cp), (gl, gh, gp) = boot(r1, r2, "1990-06-01")
            print(f"  {eng} sleeve{s} {pk}: dCalmar {d:+.3f} boot [{cl:+.3f},{ch:+.3f}] p<=0 {cp:.2f}   dCAGR boot [{gl*100:+.2f},{gh*100:+.2f}]%   d(rf-adj Sharpe) {dsh:+.3f}")
