"""
verify_analogs_robustness_followup.py -- follow-up checks for the analogs review.
  1  per-feature squared-distance shares across ALL K2/K3 picks (is gold63 the least-matched feature?)  -> C11
  2  which in-sample risk-off spells had a negative F3/F2 12m return (from spell start, and from any day) -> C6/C7
  3  waiting cost (NDX/F3 until re-entry) from the apples-to-apples off_age=23 dates                      -> C8
  4  union 1m/3m/6m medians under alternative representative-date rules                                   -> C4
  5  distinct risk-off spells behind the bootstrap first-block pools B3/B4 and the 'runs' counts         -> C10 caveats
  6  bootstrap tail numbers quoted in C10, read from the committed analogs.json                           -> C10
analogs.json is never written: analogs.py is exec'd with OUT redirected to a scratch file.
"""
import os, sys, json, math, io, contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "analogs.py")
SCR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", HERE)
src = open(SRC, encoding="utf-8").read()
needle = 'OUT = os.path.join(HERE, "analogs.json")'
assert needle in src
src = src.replace(needle, f"OUT = {os.path.join(SCR, 'analogs_rerun_ns2.json')!r}")
g = {"__file__": SRC, "__name__": "analogs_ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, SRC, "exec"), g)

idx, F, ON, ABOVE = g["idx"], g["F"], g["ON"], g["ABOVE"]
episode, greedy, results, urows = g["episode"], g["greedy"], g["results"], g["urows"]
POS_FREEZE, POS_START, T0, Zs, knn_specs, fwd = g["POS_FREEZE"], g["POS_START"], g["T0"], g["Zs"], g["knn_specs"], g["fwd"]

# ---------------------------------------------------------------- 1. feature shares across K picks
print("=== 1. per-feature share of squared distance, averaged over the 10 picks (C11)")
for name in ["K2", "K3"]:
    fs = knn_specs[name]["feats"]
    X = Zs[fs].values
    picks = results[name]["episodes"]
    sh = []
    for r in picks:
        d2 = (X[r["pos"]] - X[T0]) ** 2
        sh.append(d2 / d2.sum())
    sh = np.array(sh)
    meanshare = dict(zip(fs, sh.mean(axis=0)))
    rank = sorted(meanshare.items(), key=lambda kv: -kv[1])
    print(f"  {name}: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in rank))
    # per pick: largest contributor
    tops = [fs[int(np.argmax(s))] for s in sh]
    print(f"      largest contributor per pick: {tops}")
    if "gold63" in fs:
        gi = fs.index("gold63")
        print("      gold63 |z gap| per pick: " + ", ".join(
            f"{r['date']} {abs(X[r['pos'], gi]-X[T0, gi]):.2f}" for r in picks))
        print(f"      today's gold63 z {X[T0, gi]:+.2f}; picks' gold63 z: " +
              ", ".join(f"{X[r['pos'], gi]:+.2f}" for r in picks))

# ---------------------------------------------------------------- 2. negative 12m windows outside 2022
print("\n=== 2. negative F3/F2 12m outcomes from risk-off states (C6/C7)")
off = ~ON
spells = []
for s in range(POS_START, POS_FREEZE + 1):
    if off[s] and not off[s - 1]:
        e = s
        while e + 1 < len(off) and off[e + 1]:
            e += 1
        spells.append((s, e))
neg_start = []
for s, e in spells:
    if s + 252 <= POS_FREEZE:
        f3, f2, q = fwd("F3", s, 252), fwd("F2", s, 252), fwd("QQQ", s, 252)
        if f3 < 0 or f2 < 0:
            neg_start.append((str(idx[s].date()), round(f3 * 100, 1), round(f2 * 100, 1), round(q * 100, 1),
                              round(fwd("IEF", s, 252) * 100, 1)))
print(f"  spell STARTS with F3 or F2 12m < 0 (date, F3, F2, QQQ, IEF): {neg_start}")
ELIG = g["ELIG"]
ps = np.where(ELIG)[0]; ps = ps[ps + 252 <= POS_FREEZE]
x3 = np.array([fwd("F3", i, 252) for i in ps]); x2 = np.array([fwd("F2", i, 252) for i in ps])
ief = np.array([fwd("IEF", i, 252) for i in ps]); qq = np.array([fwd("QQQ", i, 252) for i in ps])
df = pd.DataFrame({"d": idx[ps], "F3": x3, "F2": x2, "IEF": ief, "QQQ": qq})
neg = df[(df.F3 < 0) | (df.F2 < 0)]
by = neg.groupby(neg.d.dt.year).agg(n=("F3", "size"), F3_min=("F3", "min"), F2_min=("F2", "min"),
                                    IEF_med=("IEF", "median"), QQQ_med=("QQQ", "median"))
tot = df.groupby(df.d.dt.year).size()
by["of_riskoff_days"] = tot.reindex(by.index)
print("  risk-off (ELIG) days with a negative F3 or F2 12m return, by start year:")
print((by.assign(F3_min=lambda x: (x.F3_min * 100).round(1), F2_min=lambda x: (x.F2_min * 100).round(1),
                 IEF_med=lambda x: (x.IEF_med * 100).round(1), QQQ_med=lambda x: (x.QQQ_med * 100).round(1)))
      .to_string())
print(f"  overall min over all risk-off days: F3 {x3.min()*100:+.1f}% on {df.d[np.argmin(x3)].date()}, "
      f"F2 {x2.min()*100:+.1f}% on {df.d[np.argmin(x2)].date()}")

# ---------------------------------------------------------------- 3. waiting cost from off_age=23 dates
print("\n=== 3. waiting cost from the apples-to-apples off_age=23 dates (C8)")
age = int(F["off_age"].iloc[T0])
d23 = [s + age - 1 for s, e in spells if (e - s + 1) >= age and s + age - 1 <= POS_FREEZE]
for tag, sel in [("F_a all", d23), ("F_b below EMA", [i for i in d23 if not ABOVE[i]]),
                 ("F_c dd52 band", [i for i in d23 if not ABOVE[i] and -0.20 <= F['dd52'].iloc[i] <= -0.08])]:
    rows = [episode(i) for i in sel]
    nd = [r["NDX_until_reentry"] for r in rows if r["NDX_until_reentry"] is not None]
    f3 = [r["F3_until_reentry"] for r in rows if r["F3_until_reentry"] is not None]
    print(f"  {tag:14s} n={len(rows)}: NDX until re-entry median {np.median(nd)*100:+.1f}%  F3 {np.median(f3)*100:+.1f}%  "
          f"further NDX decline 12m median {np.median([r['ndx_further_decline_12m'] for r in rows])*100:+.1f}%")
print(f"  union (reported): NDX {g['union']['summary']['NDX_until_reentry']['median']*100:+.1f}%  "
      f"F3 {g['union']['summary']['F3_until_reentry']['median']*100:+.1f}%")

# ---------------------------------------------------------------- 4. union short-horizon medians vs representative rule
print("\n=== 4. union 1m/3m/6m/12m NDX medians under alternative representative-date rules (C4)")
upos = np.array(sorted({r["pos"] for n_ in results for r in results[n_]["episodes"]}))
Xk1 = Zs[knn_specs["K1"]["feats"]].values
dK1 = np.sqrt(((Xk1[upos] - Xk1[T0]) ** 2).sum(axis=1))
rules_ = {"earliest (reported)": upos.astype(float), "latest": -upos.astype(float), "nearest K1": dK1,
          "off_age~23": np.abs(F["off_age"].values[upos] - 23).astype(float)}
for k, sc in rules_.items():
    rows = [episode(i) for i in greedy(upos, sc)]
    s = []
    for h in ["1m", "3m", "6m", "12m"]:
        x = np.array([r[f"NDX_{h}"] for r in rows if r[f"NDX_{h}"] is not None])
        s.append(f"{h} {np.median(x)*100:+5.1f} ({(x > 0).mean()*100:.0f}%)")
    print(f"  {k:22s} " + "  ".join(s))

# ---------------------------------------------------------------- 5. distinct spells behind bootstrap pools
print("\n=== 5. distinct risk-off spells behind the first-block pools (C10 caveats)")
spell_id = np.full(len(idx), -1)
for j, (s, e) in enumerate(spells):
    spell_id[s:e + 1] = j
seg0 = POS_START
for nm, arr in [("B3 (R2-like)", g["st_like"]), ("B4 (rates-up)", g["st_ratesup"]), ("B1 (risk-off)", g["st_off"])]:
    absb = np.asarray(arr) + seg0
    sid = spell_id[absb]
    print(f"  {nm}: {len(arr)} start days from {len(set(sid[sid >= 0]))} distinct risk-off spells")
emp = g["emp"]
print(f"  empirical 'n_distinct_runs' reported: riskoff {emp['E_riskoff_days']['n_distinct_runs']}, "
      f"R2like {emp['E_R2like_days']['n_distinct_runs']}, ratesup {emp['E_riskoff_ratesup_days']['n_distinct_runs']}; "
      f"in-sample risk-off spells {g['n_off_spells']}")

# ---------------------------------------------------------------- 6. C10 quoted bootstrap tails
print("\n=== 6. bootstrap tails quoted in C10 (committed analogs.json)")
d = json.load(open(os.path.join(HERE, "analogs.json")))
for k in ["B1_riskoff_state_matched", "B4_riskoff_ratesup_state_matched", "B0_unconditional"]:
    b = d["bootstrap"][k]
    print(f"  {k}: " + "  ".join(
        f"{c} med {b[c]['median']*100:+.1f} p05 {b[c]['p05']*100:+.1f} p95 {b[c]['p95']*100:+.1f} "
        f"P(loss) {b[c]['p_loss']*100:.1f}% P(<-20%) {b[c]['p_below_m20']*100:.1f}% mdd med {b[c]['maxdd_median']*100:+.1f} "
        f"p05 {b[c]['maxdd_p05']*100:+.1f}" for c in ["F2", "F3", "QQQ"]))
