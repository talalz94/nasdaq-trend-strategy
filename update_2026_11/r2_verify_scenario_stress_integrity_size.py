"""
Integrity reviewer: distribution of the FIRST Friday re-entry size in the specialist's own primary paths
(A, 30% vol, 40 seeds) -- is the first re-entry typically the clean 0.718 QLD (1.44x NDX) quoted in C3,
or chop-/season-halved? Read-only; writes nothing in the project.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_scenario_stress.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index("# ------------------------------------------------------------------ VALIDATION")
G = {"__file__": SRC, "__name__": "spec_defs"}
exec(compile(code[:cut], SRC, "exec"), G)
evaluate, synth_path, SCEN = G["evaluate"], G["synth_path"], G["SCEN"]
for short in ("S0", "S3", "S4", "S5"):
    spec = [v for k, v in SCEN.items() if k.split(" ")[0] == short][0]
    sz, er, vol, mo = [], [], [], []
    for sd in range(40):
        m = evaluate(*synth_path(spec, 0.30, sd, "A")[:5])
        if m["Fri_reentry_day"]:
            sz.append(m["F2_Fri_size_at_reentry"]); er.append(m["Fri_er20_at_reentry"]); vol.append(m["Fri_vol20_at_reentry"])
            mo.append(G["FWD"][m["Fri_reentry_day"] - 1].month)
    sz, er, vol, mo = map(np.array, (sz, er, vol, mo))
    beta = 2 * sz
    print(f"{short}: n={len(sz)} | share chop-halved (er20<0.25) {np.mean(er < 0.25):.2f} | share in Aug/Sep "
          f"{np.mean(np.isin(mo, (8, 9))):.2f} | F2 size median {np.median(sz):.3f} (NDX beta {np.median(beta):.2f}) "
          f"p25 {np.percentile(sz, 25):.3f} p75 {np.percentile(sz, 75):.3f} | share with beta >= 1.3: {np.mean(beta >= 1.3):.2f}"
          f" | vol20 median {np.median(vol):.3f}")
