"""
verify_analogs_robustness_checks.py -- adversarial ROBUSTNESS review of update_2026_11/analogs.py

Re-uses analogs.py's exact definitions by exec'ing its source with OUT redirected to a scratch file
(analogs.json is never touched), then runs independent checks:
  A  today's state (gold peak, re-crosses)                      -> C1
  B  OOS framing: returns from the NDX record / own-peak DDs     -> C2
  C  union representative-date + radius sensitivity             -> C3/C4/C6/C8
  D  effective sample size (episodes with overlapping 12m windows)-> C4/C5/C6
  E  bear-rate corrections (already-in-bear starts, today's gap) -> C5
  F  apples-to-apples re-entry: spells already 23 days risk-off  -> C8 (+C4/C6 cross-check)
  G  split halves 1988-2006 / 2007-2026-07                        -> C4/C5/C6/C8
  H  rates-up conditioning: 6m vs 12m, ex-2022, per spell         -> C7
  I  null-test power                                              -> C9
  J  bootstrap block-length sensitivity + per-spell empirical    -> C10
  K  kNN distance-ratio vs dimensionality (historical queries)   -> C11
"""
import os, sys, json, math, io, contextlib, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "analogs.py")
SCR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", HERE)
OUTJ = os.path.join(SCR, "analogs_rerun_ns.json")
RES = os.path.join(SCR, "verify_analogs_robustness_checks.json")

src = open(SRC, encoding="utf-8").read()
needle = 'OUT = os.path.join(HERE, "analogs.json")'
assert needle in src
src = src.replace(needle, f"OUT = {OUTJ!r}")
g = {"__file__": SRC, "__name__": "analogs_ns"}
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, SRC, "exec"), g)
print(f"[exec analogs.py] {time.time()-t0:.0f}s", flush=True)

P, idx, F, ON, ABOVE, NDX = g["P"], g["idx"], g["F"], g["ON"], g["ABOVE"], g["NDX"]
LOGW, RET, H52 = g["LOGW"], g["RET"], g["H52"]
episode, greedy, summarize, wilson = g["episode"], g["greedy"], g["summarize"], g["wilson"]
results, urows = g["results"], g["urows"]
POS_FREEZE, POS_START, T0, END = g["POS_FREEZE"], g["POS_START"], g["T0"], g["END"]
ELIG, ins, rules, Zs, knn_specs = g["ELIG"], g["ins"], g["rules"], g["Zs"], g["knn_specs"]
fwd, RADIUS = g["fwd"], g["RADIUS"]
OUT = {}
VARIANTS = 0          # counts every alternative definition evaluated in THIS review


def pos(d):
    return int(idx.searchsorted(pd.Timestamp(d)))


def pct(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100 * x, 1)


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else None


def keymetrics(rows):
    """compact metrics of a list of episode dicts (as produced by analogs.episode)."""
    r12 = [r for r in rows if r["NDX_12m"] is not None]
    out = dict(n=len(rows), n12=len(r12))
    for c in ["NDX", "QQQ", "F2", "F3"]:
        x = np.array([r[f"{c}_12m"] for r in r12], float)
        out[f"{c}_12m_med"] = pct(np.median(x)) if len(x) else None
        out[f"{c}_12m_hit"] = pct((x > 0).mean()) if len(x) else None
        out[f"{c}_12m_min"] = pct(x.min()) if len(x) else None
    d = np.array([r["F3_12m"] - r["QQQ_12m"] for r in r12], float)
    out["F3_minus_QQQ_med"] = pct(np.median(d)) if len(d) else None
    out["F3_beats_QQQ"] = pct((d > 0).mean()) if len(d) else None
    b = [r["bear25_12m"] for r in rows if r["bear25_12m"] is not None]
    out["bear25"] = f"{int(sum(b))}/{len(b)}"
    re = [r["days_to_strategy_reentry"] for r in rows if r["days_to_strategy_reentry"] is not None]
    out["reentry_med"] = float(np.median(re)) if re else None
    out["reentry_gt63"] = f"{sum(x > 63 for x in re)}/{len(re)}"
    w = [r["whipsaws_6m"] for r in rows if r["whipsaws_6m"] is not None]
    out["any_whipsaw_6m"] = pct(np.mean([x > 0 for x in w])) if w else None
    out["off_age_med_at_date"] = med([r["off_age"] for r in rows])
    return out


def show(tag, m):
    print(f"  {tag:44s} n={m['n']:3d}/{m['n12']:3d}  NDX12 {m['NDX_12m_med']:+6.1f} ({m['NDX_12m_hit']:.0f}%)  "
          f"F2 {m['F2_12m_med']:+6.1f} ({m['F2_12m_hit']:.0f}%, min {m['F2_12m_min']:+.1f})  "
          f"F3 {m['F3_12m_med']:+6.1f} ({m['F3_12m_hit']:.0f}%, min {m['F3_12m_min']:+.1f})  QQQ {m['QQQ_12m_med']:+6.1f}  "
          f"F3-QQQ {m['F3_minus_QQQ_med']:+5.1f} (F3>QQQ {m['F3_beats_QQQ']:.0f}%)  bear {m['bear25']}  "
          f"reentry {m['reentry_med']} (>63: {m['reentry_gt63']})  whip {m['any_whipsaw_6m']}%  offage@date {m['off_age_med_at_date']}",
          flush=True)


# ============================================================ A. today's state (C1)
print("\n=== A. TODAY (C1)", flush=True)
gw = (1 + P["gold_phys"].fillna(0)).cumprod()
y26 = gw[(gw.index >= "2025-07-01") & (gw.index <= idx[T0])]
gpk = y26.idxmax()
A = dict(gold_peak_date=str(gpk.date()), gold_dd_from_peak=float(gw.iloc[T0] / y26.max() - 1),
         gold_63d=float(gw.iloc[T0] / gw.iloc[T0 - 63] - 1))
cs = pos(g["today"]["cluster_start"])
above_days = [str(idx[i].date()) for i in range(cs, T0 + 1) if ABOVE[i]]
above_fridays = [str(idx[i].date()) for i in range(cs, T0 + 1) if ABOVE[i] and bool(P["mark"].iloc[i])]
A["above_days_in_cluster"] = above_days
A["above_days_that_are_marks"] = above_fridays
print(f"  gold peak {A['gold_peak_date']}  dd from peak {A['gold_dd_from_peak']*100:+.1f}%  63d {A['gold_63d']*100:+.1f}%")
print(f"  above-EMA days inside cluster: {above_days}; of which Friday marks: {above_fridays}")
OUT["A_today"] = A

# ============================================================ B. OOS framing (C2)
print("\n=== B. OOS FRAMING (C2)", flush=True)
p_ath, p_mark = pos("2026-10-06"), pos("2026-10-16")


def lr(c, a, b):
    return float(np.exp(LOGW[c][b] - LOGW[c][a]) - 1)


B = {}
for c in ["NDX", "QQQ", "F2", "F3", "IEF", "GLD"]:
    w = np.exp(LOGW[c][POS_FREEZE:T0 + 1] - LOGW[c][POS_FREEZE])
    pk = int(np.argmax(w))
    B[c] = dict(ath_to_mark=lr(c, p_ath, p_mark), ath_to_today=lr(c, p_ath, T0), mark_to_today=lr(c, p_mark, T0),
                freeze_to_today=lr(c, POS_FREEZE, T0), oos_peak_date=str(idx[POS_FREEZE + pk].date()),
                dd_from_oos_peak=float(w[-1] / w.max() - 1),
                oos_maxdd=float((w / np.maximum.accumulate(w) - 1).min()))
    print(f"  {c:4s} 10-06->10-16 {B[c]['ath_to_mark']*100:+6.2f}%  10-06->11-17 {B[c]['ath_to_today']*100:+6.2f}%  "
          f"10-16->11-17 {B[c]['mark_to_today']*100:+6.2f}%  freeze->now {B[c]['freeze_to_today']*100:+6.2f}%  "
          f"OOS peak {B[c]['oos_peak_date']} dd-from-peak {B[c]['dd_from_oos_peak']*100:+6.2f}%  OOS maxDD {B[c]['oos_maxdd']*100:+6.2f}%")
OUT["B_oos"] = B

# ============================================================ C. union sensitivity (C3/C4/C6/C8)
print("\n=== C. UNION REPRESENTATIVE-DATE / RADIUS SENSITIVITY (C3/C4/C6/C8)", flush=True)
upos = np.array(sorted({r["pos"] for n_ in results for r in results[n_]["episodes"]}))
Xk1 = Zs[knn_specs["K1"]["feats"]].values
dK1 = np.sqrt(((Xk1[upos] - Xk1[T0]) ** 2).sum(axis=1))
offage = F["off_age"].values
CU = {}
base = keymetrics(urows)
show("baseline (earliest per 60d, as reported)", base); CU["earliest_r60"] = base
alts = {"latest_r60": (-upos.astype(float), 60),
        "nearest_K1_to_today_r60": (dK1, 60),
        "offage_closest_to_23_r60": (np.abs(offage[upos] - 23).astype(float), 60),
        "earliest_r126": (upos.astype(float), 126),
        "earliest_r252": (upos.astype(float), 252)}
for k, (score, rad) in alts.items():
    pk = greedy(upos, score, radius=rad)
    m = keymetrics([episode(i) for i in pk]); CU[k] = m; VARIANTS += 1
    show(k, m)
rng = np.random.default_rng(7)
sims = {k: [] for k in ["NDX_12m_med", "F2_12m_med", "F3_12m_med", "F3_12m_hit", "reentry_med", "any_whipsaw_6m",
                        "F3_minus_QQQ_med", "n"]}
cache = {}
for _ in range(500):
    pk = greedy(upos, rng.random(len(upos)), radius=60)
    rows = [cache.setdefault(i, episode(i)) for i in pk]
    m = keymetrics(rows)
    for k in sims:
        sims[k].append(m[k])
CU["random_rep_r60_500draws"] = {k: dict(p05=float(np.percentile(v, 5)), med=float(np.median(v)),
                                         p95=float(np.percentile(v, 95))) for k, v in sims.items()}
VARIANTS += 1
print("  random representative per cluster (500 draws), p05/median/p95:")
for k, v in CU["random_rep_r60_500draws"].items():
    print(f"     {k:18s} {v['p05']:+7.1f} {v['med']:+7.1f} {v['p95']:+7.1f}")
# per-set medians vs union
setmed = {n_: (results[n_]["summary"]["F3_12m"]["median"], results[n_]["summary"]["F2_12m"]["median"],
               results[n_]["summary"]["NDX_12m"]["median"]) for n_ in results}
below_f3 = sum(v[0] > base["F3_12m_med"] / 100 for v in setmed.values())
below_f2 = sum(v[1] > base["F2_12m_med"] / 100 for v in setmed.values())
print(f"  sets whose F3 12m median exceeds the union's {base['F3_12m_med']}: {below_f3}/9; F2: {below_f2}/9")
CU["sets_with_F3_median_above_union"] = below_f3
CU["sets_with_F2_median_above_union"] = below_f2
# union rows that satisfy NO price-state rule like today's (age<10 or dd52 outside [-20,-8] or vol<=25%)
loose = [r["date"] for r in urows if not (-0.20 <= r["dd52"] <= -0.08 and 10 <= r["age"] <= 40 and r["vol20"] > 0.25)]
print(f"  union episodes failing the R2 price-state filter at their own date: {len(loose)}/29 {loose}")
CU["union_dates_failing_R2_filter"] = loose
OUT["C_union_sensitivity"] = CU

# ============================================================ D. effective sample size (C4/C5/C6)
print("\n=== D. EFFECTIVE SAMPLE SIZE: clusters of episodes with overlapping 12m windows", flush=True)
r12 = [r for r in urows if r["NDX_12m"] is not None]
clusters, cur = [], [r12[0]]
for r in r12[1:]:
    if r["pos"] - cur[-1]["pos"] < 252:
        cur.append(r)
    else:
        clusters.append(cur); cur = [r]
clusters.append(cur)
D = dict(n_episodes_12m=len(r12), n_clusters=len(clusters),
         clusters=[[r["date"] for r in c] for c in clusters])
for c in ["NDX", "F2", "F3", "QQQ"]:
    cm = np.array([np.mean([r[f"{c}_12m"] for r in cl]) for cl in clusters])
    k = int((cm > 0).sum()); lo, hi = wilson(k, len(cm))
    D[c] = dict(cluster_median=float(np.median(cm)), cluster_hit=k / len(cm), wilson95=[lo, hi])
neg_cl = [cl[0]["date"] for cl in clusters if any(r["F3_12m"] < 0 for r in cl)]
bear_cl = [cl[0]["date"] for cl in clusters if any(r["bear25_12m"] for r in cl)]
D["clusters_with_any_F3_loss"] = neg_cl
D["clusters_with_bear25"] = f"{len(bear_cl)}/{len(clusters)}"
D["bear25_cluster_wilson"] = list(wilson(len(bear_cl), len(clusters)))
print(f"  {len(r12)} episodes with 12m -> {len(clusters)} clusters (start dates <252d apart chained)")
for cl in clusters:
    print("    ", [r["date"] for r in cl])
for c in ["NDX", "F2", "F3", "QQQ"]:
    print(f"  {c}: cluster-mean median {D[c]['cluster_median']*100:+.1f}%  hit {D[c]['cluster_hit']*100:.0f}% "
          f"(Wilson {D[c]['wilson95'][0]*100:.0f}-{D[c]['wilson95'][1]*100:.0f}%)")
print(f"  clusters with a >25% bear: {D['clusters_with_bear25']} (Wilson {D['bear25_cluster_wilson'][0]*100:.0f}-"
      f"{D['bear25_cluster_wilson'][1]*100:.0f}%); with an F3 loss: {neg_cl}")
OUT["D_effective_n"] = D

# ============================================================ E. bear-rate corrections (C5)
print("\n=== E. BEAR RATE CORRECTIONS (C5)", flush=True)
E = {}
already = [r["date"] for r in r12 if r["dd52"] <= -0.25]
fresh = [r for r in r12 if r["dd52"] > -0.25]
kb = sum(bool(r["bear25_12m"]) for r in fresh)
E["already_in_bear_at_date"] = already
E["bear25_excluding_already"] = f"{kb}/{len(fresh)}"
E["bear25_excluding_already_wilson"] = list(wilson(kb, len(fresh)))
need = 0.75 / (1 + g["today"]["dd52"]) - 1          # further fall needed today to reach -25% from the 52w peak
E["today_further_fall_needed"] = need
kf = sum(r["ndx_further_decline_12m"] <= need for r in r12)
E["further_fall_ge_today_gap"] = f"{kf}/{len(r12)}"
print(f"  episodes already <= -25% from 52w peak at the analog date: {already}")
print(f"  bear25 excluding them: {kb}/{len(fresh)} = {kb/len(fresh)*100:.0f}% (Wilson "
      f"{E['bear25_excluding_already_wilson'][0]*100:.0f}-{E['bear25_excluding_already_wilson'][1]*100:.0f}%)")
print(f"  today needs a further {need*100:.1f}% fall to reach -25%; episodes with a further fall at least that big "
      f"within 12m: {kf}/{len(r12)}")
per = {}
for n_ in results:
    rows = [r for r in results[n_]["episodes"] if r["bear25_12m"] is not None]
    a_ = [r for r in rows if r["dd52"] <= -0.25]
    f_ = [r for r in rows if r["dd52"] > -0.25]
    per[n_] = dict(reported=f"{sum(r['bear25_12m'] for r in rows)}/{len(rows)}",
                   already=len(a_), excl=f"{sum(r['bear25_12m'] for r in f_)}/{len(f_)}")
print("  per set (reported -> excluding already-in-bear starts): " +
      ", ".join(f"{k} {v['reported']}->{v['excl']}" for k, v in per.items()))
E["per_set"] = per
OUT["E_bear"] = E

# ============================================================ F. spell-based apples-to-apples (C8)
print("\n=== F. SPELLS ALREADY 23 DAYS RISK-OFF (today's off_age) -- apples-to-apples re-entry", flush=True)
off = ~ON
st = [i for i in range(POS_START, POS_FREEZE + 1) if off[i] and (i == 0 or not off[i - 1])]
spells = []
for s in st:
    e = s
    while e + 1 < len(off) and off[e + 1]:
        e += 1
    spells.append((s, e))
L = np.array([e - s + 1 for s, e in spells])
cens = np.array([e >= POS_FREEZE for s, e in spells])
age_today = int(F["off_age"].iloc[T0])
Fd = dict(n_spells=len(spells), n_censored=int(cens.sum()), today_off_age=age_today,
          share_spells_reaching_today=float((L >= age_today).mean()))
print(f"  {len(spells)} in-sample risk-off spells (start 1988..freeze); {int((L >= age_today).sum())} "
      f"({(L >= age_today).mean()*100:.0f}%) lasted >= {age_today} days")
d23 = [s + age_today - 1 for s, e in spells if (e - s + 1) >= age_today and s + age_today - 1 <= POS_FREEZE]
variants_F = {
    "F_a all spells at off_age=23": d23,
    "F_b + below EMA-100 at that date": [i for i in d23 if not ABOVE[i]],
    "F_c + dd52 in [-20%,-8%]": [i for i in d23 if not ABOVE[i] and -0.20 <= F["dd52"].iloc[i] <= -0.08],
}
for k, ps in variants_F.items():
    rows = [episode(i) for i in ps]
    m = keymetrics(rows); VARIANTS += 1
    re = np.array([r["days_to_strategy_reentry"] for r in rows if r["days_to_strategy_reentry"] is not None])
    m["reentry_q25_q75"] = [float(np.percentile(re, 25)), float(np.percentile(re, 75))]
    m["dates"] = [r["date"] for r in rows]
    Fd[k] = m
    show(k, m)
    print(f"      remaining-days IQR {m['reentry_q25_q75']}  dates: {', '.join(m['dates'])}")
# union analogs: off_age at their own dates
print(f"  union episodes: off_age at the analog date median {base['off_age_med_at_date']} "
      f"(today {age_today}); union episodes whose spell had ALREADY reached {age_today} days at the date: "
      f"{sum(r['off_age'] >= age_today for r in urows)}/29")
# union episodes whose whole spell ended before reaching today's off_age (could never look like today)
short = []
for r in urows:
    i = r["pos"]
    s0 = i - int(r["off_age"]) + 1
    tot = int(r["off_age"]) + (r["days_to_strategy_reentry"] - 1 if r["days_to_strategy_reentry"] else 0)
    if tot < age_today:
        short.append((r["date"], tot))
print(f"  union episodes whose risk-off spell ended before reaching {age_today} days: {len(short)}/29 {short}")
Fd["union_spells_shorter_than_today"] = short
OUT["F_spells"] = Fd

# ============================================================ G. split halves (C4/C5/C6/C8)
print("\n=== G. SPLIT HALVES (1988-2006 vs 2007-2026-07)", flush=True)
G = {}
cut = pd.Timestamp("2007-01-01")
for name, rows in [("union", urows), ("F_b spells@23", [episode(i) for i in variants_F["F_b + below EMA-100 at that date"]])]:
    for half, sel in [("1988-2006", lambda r: pd.Timestamp(r["date"]) < cut),
                      ("2007-2026", lambda r: pd.Timestamp(r["date"]) >= cut)]:
        m = keymetrics([r for r in rows if sel(r)]); G[f"{name} {half}"] = m; VARIANTS += 1
        show(f"{name} {half}", m)
OUT["G_halves"] = G

# ============================================================ H. rates-up conditioning (C7)
print("\n=== H. RATES-UP CONDITIONING (C7)", flush=True)
RATESUP = g["RATESUP"]
spell_id = np.full(len(idx), -1)
for j, (s, e) in enumerate(spells):
    spell_id[s:e + 1] = j


def emp(mask, excl=None):
    ps = np.where(mask)[0]; ps = ps[ps + 252 <= POS_FREEZE]
    if excl is not None:
        ps = np.array([p for p in ps if not (pd.Timestamp(excl[0]) <= idx[p] <= pd.Timestamp(excl[1]))])
    out = dict(n_days=int(len(ps)), n_spells=int(len(set(spell_id[ps]))))
    for c in ["F3", "F2"]:
        for h, hn in [(126, "6m"), (252, "12m")]:
            x = np.array([fwd(c, i, h) for i in ps])
            out[f"{c}_{hn}_med"] = float(np.median(x)); out[f"{c}_{hn}_ploss"] = float((x < 0).mean())
            # per spell: average loss indicator within each spell, then average across spells
            df = pd.DataFrame({"s": spell_id[ps], "l": (x < 0).astype(float)})
            out[f"{c}_{hn}_ploss_spellwt"] = float(df.groupby("s")["l"].mean().mean())
            out[f"{c}_{hn}_spells_with_loss"] = int((df.groupby("s")["l"].max() > 0).sum())
    return out


H = {"riskoff_all": emp(ELIG), "riskoff_ratesup": emp(RATESUP),
     "riskoff_all_ex2022": emp(ELIG, ("2021-06-01", "2022-12-31")),
     "riskoff_ratesup_ex2022": emp(RATESUP, ("2021-06-01", "2022-12-31"))}
VARIANTS += 4
for k, v in H.items():
    print(f"  {k:24s} days {v['n_days']:5d} spells {v['n_spells']:3d} | F3 6m med {v['F3_6m_med']*100:+5.1f} "
          f"P(loss) {v['F3_6m_ploss']*100:4.1f}% (spell-wt {v['F3_6m_ploss_spellwt']*100:4.1f}%, spells w/ loss "
          f"{v['F3_6m_spells_with_loss']}) | F3 12m med {v['F3_12m_med']*100:+5.1f} P(loss) {v['F3_12m_ploss']*100:4.1f}% "
          f"(spell-wt {v['F3_12m_ploss_spellwt']*100:4.1f}%, spells w/ loss {v['F3_12m_spells_with_loss']}) | "
          f"F2 6m P(loss) {v['F2_6m_ploss']*100:4.1f}% 12m {v['F2_12m_ploss']*100:4.1f}%")
# which spells hold the rates-up 6m losses
ps = np.where(RATESUP)[0]; ps = ps[ps + 252 <= POS_FREEZE]
x6 = np.array([fwd("F3", i, 126) for i in ps])
loss_days = pd.Series(x6 < 0, index=idx[ps])
byyear = loss_days.groupby(loss_days.index.year).agg(["sum", "count"])
byyear = byyear[byyear["sum"] > 0]
H["ratesup_6m_loss_days_by_year"] = {int(k): [int(a), int(b)] for k, (a, b) in byyear.iterrows()}
print(f"  rates-up risk-off days with an F3 6m loss, by year (loss days / days): {H['ratesup_6m_loss_days_by_year']}")
OUT["H_ratesup"] = H

# ============================================================ I. null-test power (C9)
print("\n=== I. NULL-TEST POWER (C9)", flush=True)
POOL, pool_f = g["POOL"], g["pool_f"]
rng2 = np.random.default_rng(99)


def rand_decl(nsel):
    perm = rng2.permutation(len(POOL)); picked = []
    for j in perm:
        p = POOL[j]
        if all(abs(p - POOL[q]) >= RADIUS for q in picked):
            picked.append(j)
            if len(picked) >= nsel:
                break
    return picked


I = {}
for n in [8, 10, 12, 16, 21, 28]:
    meds = {k: [] for k in ["NDX_12m", "F3_12m", "F2_12m"]}
    for _ in range(1000):
        js = rand_decl(n)
        for k in meds:
            meds[k].append(np.median(pool_f[k][js]))
    I[n] = {k: dict(p05=float(np.percentile(v, 5)), p50=float(np.median(v)), p95=float(np.percentile(v, 95)))
            for k, v in meds.items()}
    print(f"  n={n:2d}: " + "  ".join(f"{k} null 5-95% [{v['p05']*100:+.1f}, {v['p95']*100:+.1f}] (width "
                                       f"{(v['p95']-v['p05'])*100:.1f}pp)" for k, v in I[n].items()))
rep = {n_: {k: results[n_]["null_test"][k]["pctile_of_observed"] for k in ["NDX_12m", "F3_12m", "F2_12m"]}
       for n_ in results}
print("  reported percentiles per set: " + "; ".join(f"{k} {v['NDX_12m']:.0f}/{v['F3_12m']:.0f}/{v['F2_12m']:.0f}"
                                                    for k, v in rep.items()))
OUT["I_null_power"] = {"bands": I, "reported_pctiles": rep}

# ============================================================ J. bootstrap block length + per-spell empirical (C10)
print("\n=== J. BOOTSTRAP BLOCK-LENGTH SENSITIVITY + PER-SPELL EMPIRICAL (C10)", flush=True)
seg = slice(POS_START, POS_FREEZE + 1)
Rb = {c: RET[c].values[seg] for c in ["F2", "F3", "QQQ"]}
on_b = ON[seg]; Tb = len(on_b)


def boot(blk, first, cont, seed, reps=10000):
    nb = 252 // blk
    valid = np.arange(0, Tb - blk)
    s_off = valid[~on_b[valid]]; s_on = valid[on_b[valid]]
    pool0 = s_off if first == "off" else valid
    gg = np.random.default_rng(seed)
    S = np.empty((reps, nb), dtype=int)
    S[:, 0] = gg.choice(pool0, reps)
    for b in range(1, nb):
        if cont == "uniform":
            S[:, b] = gg.choice(valid, reps)
        else:
            es = on_b[S[:, b - 1] + blk]
            nx = np.empty(reps, dtype=int)
            nx[es] = gg.choice(s_on, int(es.sum())); nx[~es] = gg.choice(s_off, int((~es).sum()))
            S[:, b] = nx
    ix = (S[:, :, None] + 1 + np.arange(blk)[None, None, :]).reshape(reps, nb * blk)
    o = {}
    for c in Rb:
        tr = np.prod(1 + Rb[c][ix], axis=1) - 1
        o[c] = dict(median=float(np.median(tr)), p05=float(np.percentile(tr, 5)), p_loss=float((tr < 0).mean()))
    return o


J = {}
for blk in [21, 63, 126]:
    for nm, (first, cont) in {"B1": ("off", "state"), "B0": ("all", "uniform")}.items():
        J[f"{nm}_blk{blk}"] = boot(blk, first, cont, 1000 + blk); VARIANTS += 1
        o = J[f"{nm}_blk{blk}"]
        print(f"  {nm} block {blk:3d}: " + "  ".join(f"{c} med {o[c]['median']*100:+5.1f} p05 {o[c]['p05']*100:+6.1f} "
                                                   f"P(loss) {o[c]['p_loss']*100:4.1f}%" for c in ["F2", "F3", "QQQ"]))
# per-spell empirical: one observation per risk-off spell (its first off day)
ps0 = [s for s, e in spells if s + 252 <= POS_FREEZE]
Jsp = {}
for c in ["F2", "F3", "QQQ"]:
    x = np.array([fwd(c, i, 252) for i in ps0])
    Jsp[c] = dict(n=len(x), median=float(np.median(x)), p05=float(np.percentile(x, 5)), p_loss=float((x < 0).mean()),
                  min=float(x.min()))
ps_all = [i for i in range(POS_START, POS_FREEZE - 251)]
Jall = {}
for c in ["F2", "F3", "QQQ"]:
    x = np.array([fwd(c, i, 252) for i in ps_all])
    Jall[c] = dict(median=float(np.median(x)), p05=float(np.percentile(x, 5)), p_loss=float((x < 0).mean()))
print("  empirical, one obs per risk-off spell start: " + "  ".join(
    f"{c} med {v['median']*100:+5.1f} p05 {v['p05']*100:+6.1f} P(loss) {v['p_loss']*100:4.1f}% min {v['min']*100:+.1f}"
    for c, v in Jsp.items()) + f"  (n={Jsp['F3']['n']})")
print("  empirical, every in-sample day (overlapping): " + "  ".join(
    f"{c} med {v['median']*100:+5.1f} p05 {v['p05']*100:+6.1f} P(loss) {v['p_loss']*100:4.1f}%" for c, v in Jall.items()))
J["empirical_per_spell_start"] = Jsp
J["empirical_all_days"] = Jall
OUT["J_boot"] = J

# ============================================================ K. kNN distance ratio vs dimensionality (C11)
print("\n=== K. kNN NEAREST/MEDIAN DISTANCE RATIO: today vs historical queries (C11)", flush=True)
K = {}
rngk = np.random.default_rng(5)
for name in ["K1", "K2", "K3"]:
    fs = knn_specs[name]["feats"]
    X = Zs[fs].values
    ok = ELIG & ~np.isnan(X).any(axis=1)
    c = np.where(ok)[0]
    dq = np.sqrt(((X[c] - X[T0]) ** 2).sum(axis=1))
    r_today = float(dq.min() / np.median(dq))
    qs = rngk.choice(c, 400, replace=False)
    ratios, nears = [], []
    for q in qs:
        cc = c[np.abs(c - q) >= 126]
        d = np.sqrt(((X[cc] - X[q]) ** 2).sum(axis=1))
        ratios.append(d.min() / np.median(d)); nears.append(d.min())
    ratios = np.array(ratios); nears = np.array(nears)
    K[name] = dict(n_feats=len(fs), today_nearest=float(dq.min()), today_median=float(np.median(dq)),
                   today_ratio=r_today, hist_ratio_p10=float(np.percentile(ratios, 10)),
                   hist_ratio_med=float(np.median(ratios)), hist_ratio_p90=float(np.percentile(ratios, 90)),
                   today_ratio_pctile=float((ratios < r_today).mean() * 100),
                   today_nearest_pctile=float((nears < dq.min()).mean() * 100))
    v = K[name]
    print(f"  {name} ({v['n_feats']} feats): today nearest {v['today_nearest']:.2f} / median {v['today_median']:.2f} = "
          f"{v['today_ratio']:.2f}; historical risk-off queries ratio p10/med/p90 {v['hist_ratio_p10']:.2f}/"
          f"{v['hist_ratio_med']:.2f}/{v['hist_ratio_p90']:.2f} -> today at {v['today_ratio_pctile']:.0f}th pct "
          f"(nearest-distance pct {v['today_nearest_pctile']:.0f})")
# feature contributions to the nearest K3 match
fs = knn_specs["K3"]["feats"]; X = Zs[fs].values
ok = ELIG & ~np.isnan(X).any(axis=1); c = np.where(ok)[0]
dq = ((X[c] - X[T0]) ** 2)
j = int(np.argmin(dq.sum(axis=1)))
K["K3_nearest_date"] = str(idx[c[j]].date())
K["K3_nearest_sqdist_by_feature"] = {f: float(v) for f, v in zip(fs, dq[j])}
K["today_z"] = {f: float(v) for f, v in zip(fs, X[T0])}
print(f"  nearest K3 match {K['K3_nearest_date']}: squared-distance share by feature " +
      ", ".join(f"{f} {v/dq[j].sum()*100:.0f}%" for f, v in zip(fs, dq[j])))
print("  today's z-scores: " + ", ".join(f"{f} {v:+.2f}" for f, v in K["today_z"].items()))
OUT["K_knn"] = K

OUT["variants_evaluated_in_review"] = VARIANTS
with open(RES, "w") as fh:
    json.dump(OUT, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"\nreview variants evaluated: {VARIANTS}\nsaved {RES}", flush=True)
