"""
r2_verify_scenario_stress_robustness_checks.py -- ROBUSTNESS review of r2_scenario_stress (deterministic and
in-sample checks). Reads the specialist's r2_scenario_stress.json and the panel; writes only to <out.json>.

usage: python -u update_2026_11/r2_verify_scenario_stress_robustness_checks.py <out.json>
  C1  risk units of today's off-leg: standalone vol / correlation / Euler risk shares of 0.85 IEF + 0.15 gold,
      sigma-size and historical frequency of the S1 (+100bp in 21d) and S2 (-15% gold in 42d) shocks
  C3  flat-path threshold ladder recomputed with pandas ewm; bar decline per day; re-entry size if the
      Friday threshold is crossed by a 1-day jump or a 3-day rally (vol20 / er20 actually recomputed)
  C5  F2>QQQ counts by convention x book; start-date jitter (-5..+5 trading days) of every replay; X-day return
  C9  today-state minus own-state per replay: dispersion, sign test
  C10 variance ratios of NDX under the SAME 252-day-window estimator used for the noise; known-mean VR of the
      design-A bridge vs a random walk
  C7  in-sample analogs of the S0 'flat 30%-vol year' from risk-off starts; max in-sample Friday-book whipsaws
      in any 6- or 12-month window
  C6  re-entry cost in S4/S5 measured directly (F2 minus the never-re-enter off-leg) under design A, 'free' and
      block-bootstrap noise
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_verify_scenario_stress_robustness_lib as L  # noqa: E402

OUT = os.path.abspath(sys.argv[1])
assert not OUT.startswith(os.path.abspath(HERE))
g = L.load_specialist()
C = g["C"]
P = g["P"]
idx = P.index
J = json.load(open(os.path.join(HERE, "r2_scenario_stress.json")))
F = L.Fast()
H = g["H"]
res = {}


def pr(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ C1 risk units
c1 = {}
for lab, end in (("to_freeze", C.FREEZE), ("to_2026_11_17", idx[-1])):
    seg = P.loc[:end].iloc[-252:]
    ief, gld = seg["ief"].values, seg["gold_phys"].values
    a, b = 0.85 * ief, 0.15 * gld
    va, vb = a.std(ddof=1) * math.sqrt(252), b.std(ddof=1) * math.sqrt(252)
    rho = float(np.corrcoef(ief, gld)[0, 1])
    cov = np.cov(np.vstack([a, b]))
    tot = cov.sum()
    c1[lab] = dict(vol_085IEF=va, vol_015GLD=vb, corr_IEF_GLD=rho, book_vol=math.sqrt(tot * 252),
                   risk_share_IEF=float((cov[0, 0] + cov[0, 1]) / tot), risk_share_GLD=float((cov[1, 1] + cov[0, 1]) / tot),
                   ief_vol=ief.std(ddof=1) * math.sqrt(252), gold_vol=gld.std(ddof=1) * math.sqrt(252))
    pr(f"C1 {lab}: 0.85 IEF vol {va*100:.2f}% | 0.15 GLD vol {vb*100:.2f}% | corr {rho:+.2f} | risk share IEF "
       f"{c1[lab]['risk_share_IEF']*100:.0f}% GLD {c1[lab]['risk_share_GLD']*100:.0f}%")
SD_Y, SD_G = g["SD_Y"], g["SD_G"]
c1["sigma_S1_100bp_in_21d"] = 1.00 / (SD_Y * math.sqrt(21))
c1["sigma_S2_gold15_in_42d"] = abs(math.log(0.85)) / (SD_G * math.sqrt(42))
ty = P["tnx"].ffill()
ins = (idx >= C.START) & (idx <= C.FREEZE)
d21 = (ty.shift(-21) - ty)[ins].dropna()
c1["hist_10y_21d_change_ge_+100bp_share"] = float((d21 >= 1.0).mean())
c1["hist_10y_21d_change_max_pp"] = float(d21.max())
c1["hist_10y_21d_change_max_date"] = str(d21.idxmax().date())
c1["hist_10y_21d_change_le_-100bp_share"] = float((d21 <= -1.0).mean())
gl = np.log1p(P["gold_phys"].fillna(0)).cumsum()
realg = (idx >= pd.Timestamp("2000-08-30")) & (idx <= C.FREEZE)
g42 = (gl.shift(-42) - gl)[realg].dropna()
c1["hist_gold_42d_le_-15pct_share_2000_freeze"] = float((g42 <= math.log(0.85)).mean())
c1["hist_gold_42d_ge_+15pct_share_2000_freeze"] = float((g42 >= math.log(1.15)).mean())
c1["hist_gold_42d_min"] = float(np.expm1(g42.min()))
pr(f"C1 shock sizes: S1 +100bp/21d = {c1['sigma_S1_100bp_in_21d']:.1f} sigma (hist share of 21d windows >= +100bp "
   f"{c1['hist_10y_21d_change_ge_+100bp_share']*100:.2f}%, max {c1['hist_10y_21d_change_max_pp']:+.2f}pp on "
   f"{c1['hist_10y_21d_change_max_date']}); S2 gold -15%/42d = {c1['sigma_S2_gold15_in_42d']:.1f} sigma "
   f"(hist share {c1['hist_gold_42d_le_-15pct_share_2000_freeze']*100:.1f}%)")
# 1-sigma 12m and 1-month book impact of each leg (risk-equivalent comparison)
c1["one_sigma_12m_book_IEF_leg"] = 0.85 * g["IEF_D"] * SD_Y / 100 * math.sqrt(252)
c1["one_sigma_12m_book_GLD_leg"] = 0.15 * SD_G * math.sqrt(252)
pr(f"C1 1-sigma 12m book impact: IEF leg {c1['one_sigma_12m_book_IEF_leg']*100:.2f}% vs gold leg "
   f"{c1['one_sigma_12m_book_GLD_leg']*100:.2f}%")
res["C1"] = c1

# ------------------------------------------------------------------ C3 ladder and re-entry sizing
L0, EMA0, A_ = g["L0"], g["EMA0"], g["ALPHA"]
FWD = g["FWD"]
flat = pd.concat([P["ndx"], pd.Series(L0, index=FWD[:70])])
ema_full = flat.ewm(span=100, adjust=False).mean()
lad = []
for j, d in enumerate(FWD[:70]):
    if d.dayofweek != 4:
        continue
    thr = float(ema_full.loc[FWD[j - 1]] if j > 0 else ema_full.iloc[len(P) - 1])
    lad.append((str(d.date()), thr, thr / L0 - 1))
spec_lad = J["today"]["friday_threshold_ladder_flat_path"]
c3 = dict(ladder_recomputed=lad[:12],
          max_abs_diff_vs_specialist=float(max(abs(a[1] - b["threshold"]) for a, b in zip(lad, spec_lad))))
dec = [(lad[i][0], (lad[i - 1][1] - lad[i][1]) / L0 / np.busday_count(lad[i - 1][0], lad[i][0],
        holidays=["2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15"])) for i in range(1, len(lad))]
c3["bar_decline_pct_of_L0_per_trading_day"] = [(d, round(v * 100, 3)) for d, v in dec[:11]]
pr(f"C3 ladder max |diff| vs specialist {c3['max_abs_diff_vs_specialist']:.2f} pts; first {lad[0]}; daily bar decline "
   f"(% of today's NDX): {[round(v*100,3) for _, v in dec[:11]]}")


def size_case(ndx_r):
    lv, ema, vol, er, above = F.signals(np.asarray(ndx_r))
    k = 2                                                  # Friday 2026-11-20 = forward day 3 (index 2)
    out = dict(ndx=float(lv[k]), above=bool(above[k]), vol20=float(vol[k]), er20=float(er[k]))
    for tier, (lev, vt, beta) in L.TIER.items():
        sc = min(1.0, vt / vol[k])
        e = sc if er[k] >= 0.25 else 0.5 * sc
        out[f"{tier}_sleeve"] = 0.85 * e
        out[f"{tier}_ndx_beta"] = 0.85 * e * beta
    return out


thr = lad[0][1]
jump = np.zeros(H)
jump[2] = thr * 1.003 / L0 - 1
rally = np.zeros(H)
step = (thr * 1.003 / L0) ** (1 / 3) - 1
rally[:3] = step
c3["reentry_size_if_1day_jump_on_11_20"] = size_case(jump)
c3["reentry_size_if_3day_rally_to_11_20"] = size_case(rally)
for k_ in ("reentry_size_if_1day_jump_on_11_20", "reentry_size_if_3day_rally_to_11_20"):
    v = c3[k_]
    pr(f"C3 {k_}: above {v['above']} vol20 {v['vol20']*100:.1f}% er20 {v['er20']:.2f} -> F2 {v['F2_sleeve']*100:.1f}% QLD "
       f"(beta {v['F2_ndx_beta']:.2f}) F3 {v['F3_sleeve']*100:.1f}% TQQQ (beta {v['F3_ndx_beta']:.2f}) F1 beta {v['F1_ndx_beta']:.2f}")
res["C3"] = c3

# ------------------------------------------------------------------ C5 replay conventions and jitter
R = J["replays"]
cnt = {}
for conv in ("literal", "after", "literal_ief_carry_rebased"):
    for b in ("Fri", "tr"):
        n = sum(1 for d in R if R[d][conv][f"F2_{b}_r12m"] > R[d][conv]["QQQ_r12m"])
        cnt[f"{conv}_{b}"] = n
pr("C5 F2 beats QQQ (12m) by convention/book:", cnt)
xret = {d: float(P.loc[pd.Timestamp(d), "ndx_ret"]) for d in R}
pr("C5 NDX return ON the replay date X (day 1 of the 'literal' path):", {d: round(v * 100, 2) for d, v in xret.items()})
jit = {}
offs = list(range(-5, 6))
for d in R:
    i = idx.get_loc(pd.Timestamp(d))
    rows = []
    for o in offs:
        s0 = i + o
        sg = P.iloc[s0:s0 + H]
        rr = F.run(sg["ndx_ret"].values, sg["ief"].values, sg["gold_phys"].values)
        rows.append(dict(off=o, F2_Fri=L.cum(rr["rets"]["F2"]["Fri"]), F2_tr=L.cum(rr["rets"]["F2"]["tr"]),
                         F1_Fri=L.cum(rr["rets"]["F1"]["Fri"]), F3_Fri=L.cum(rr["rets"]["F3"]["Fri"]),
                         QQQ=L.cum(rr["qqq"])))
    jit[d] = rows
    lit = [r for r in rows if r["off"] == 0][0]
    assert abs(lit["F2_Fri"] - R[d]["literal"]["F2_Fri_r12m"]) < 1e-12
    aft = [r for r in rows if r["off"] == 1][0]
    assert abs(aft["F2_Fri"] - R[d]["after"]["F2_Fri_r12m"]) < 1e-12
    f2 = np.array([r["F2_Fri"] for r in rows])
    f2t = np.array([r["F2_tr"] for r in rows])
    q = np.array([r["QQQ"] for r in rows])
    pr(f"C5 jitter {d}: F2 Fri 12m over offsets -5..+5: min {f2.min()*100:+.1f} med {np.median(f2)*100:+.1f} max "
       f"{f2.max()*100:+.1f} | tranche med {np.median(f2t)*100:+.1f} | QQQ med {np.median(q)*100:+.1f} | F2 Fri>QQQ in "
       f"{int((f2 > q).sum())}/11, tranche>QQQ {int((f2t > q).sum())}/11")
beats_by_off = {o: int(sum(1 for d in R if [r for r in jit[d] if r["off"] == o][0]["F2_Fri"] >
                               [r for r in jit[d] if r["off"] == o][0]["QQQ"])) for o in offs}
beats_by_off_tr = {o: int(sum(1 for d in R if [r for r in jit[d] if r["off"] == o][0]["F2_tr"] >
                                  [r for r in jit[d] if r["off"] == o][0]["QQQ"])) for o in offs}
pr("C5 number of the 9 replays where F2 Fri beats QQQ, by start offset:", beats_by_off)
pr("C5 ... tranche:", beats_by_off_tr)
med_by_off = {o: float(np.median([[r for r in jit[d] if r["off"] == o][0]["F2_Fri"] for d in R])) for o in offs}
pr("C5 replay-median F2 Fri 12m by offset:", {o: round(v * 100, 1) for o, v in med_by_off.items()})
res["C5"] = dict(beats_by_convention=cnt, x_day_ndx_return=xret, jitter=jit, beats_by_offset_Fri=beats_by_off,
                 beats_by_offset_tranche=beats_by_off_tr, replay_median_F2_Fri_by_offset=med_by_off)

# ------------------------------------------------------------------ C9 today-state vs own-state
sf = {d: R[d]["after"]["F2_Fri_r12m"] - R[d]["history_context_close_of_date"]["F2_Fri_r12m_original"] for d in R}
st = {d: R[d]["after"]["F2_tr_r12m"] - R[d]["history_context_close_of_date"]["F2_tr_r12m_original"] for d in R}
gap = {d: R[d]["history_context_close_of_date"]["gap_to_ema"] for d in R}
c9 = {}
for lab, dd in (("Fri", sf), ("tranche", st)):
    v = np.array(list(dd.values()))
    nneg = int((v < 0).sum())
    from math import comb
    p2 = min(1.0, 2 * sum(comb(9, k) for k in range(0, min(nneg, 9 - nneg) + 1)) / 2 ** 9)
    c9[lab] = dict(values={k: round(x * 100, 1) for k, x in dd.items()}, median=float(np.median(v)), mean=float(v.mean()),
                   sd=float(v.std(ddof=1)), t=float(v.mean() / (v.std(ddof=1) / 3)), n_neg=nneg, sign_test_p=p2,
                   min=float(v.min()), max=float(v.max()))
    pr(f"C9 today-state minus own-state F2 {lab}: median {np.median(v)*100:+.1f}pp mean {v.mean()*100:+.1f}pp sd "
       f"{v.std(ddof=1)*100:.1f} t {c9[lab]['t']:.2f} | negative {nneg}/9 (sign p {p2:.2f}) | range {v.min()*100:+.1f}.."
       f"{v.max()*100:+.1f} | {c9[lab]['values']}")
c9["episode_own_gap_to_ema"] = gap
c9["n_episodes_with_deeper_gap_than_today"] = int(sum(1 for v in gap.values() if v < g["L0"] / g["EMA0"] - 1))
pr(f"C9 own-state EMA gap per episode: { {k: round(v*100,1) for k,v in gap.items()} } (today {(g['L0']/g['EMA0']-1)*100:.1f}%); "
   f"deeper than today in {c9['n_episodes_with_deeper_gap_than_today']}/9")
res["C9"] = c9

# ------------------------------------------------------------------ C10 variance-ratio estimator
def vr_window(x, q):
    return pd.Series(x).rolling(q).sum().dropna().var(), np.var(x, ddof=1)


r_ins = np.log1p(P["ndx_ret"][(idx >= C.START) & (idx <= C.FREEZE)].values)
num21, num63, den = [], [], []
for s in range(0, len(r_ins) - 252 + 1, 252):
    w = r_ins[s:s + 252]
    num21.append(pd.Series(w).rolling(21).sum().dropna().var())
    num63.append(pd.Series(w).rolling(63).sum().dropna().var())
    den.append(np.var(w, ddof=1))
c10 = dict(ndx_252d_window_vr21=float(np.mean(num21) / (21 * np.mean(den))),
           ndx_252d_window_vr63=float(np.mean(num63) / (63 * np.mean(den))),
           ndx_full_sample_vr21=J["realism"]["ndx_variance_ratio_1988_freeze"]["vr21"],
           ndx_full_sample_vr63=J["realism"]["ndx_variance_ratio_1988_freeze"]["vr63"],
           n_windows=len(den))
rng = np.random.default_rng(11)
kz = []
for _ in range(4000):
    z = rng.standard_normal(H)
    kz.append(z)
Z = np.array(kz)
br = np.array([g["bridge"](z, [H]) for z in Z])
for lab, X in (("random_walk", Z), ("design_A_bridge", br)):
    for q in (21, 63, 126):
        s = np.array([np.convolve(x, np.ones(q), "valid") for x in X])
        c10[f"known_mean_vr{q}_{lab}"] = float((s ** 2).mean() / (q * (X ** 2).mean()))
pr(f"C10 NDX VR under the specialist's 252-day-window estimator: VR21 {c10['ndx_252d_window_vr21']:.2f} VR63 "
   f"{c10['ndx_252d_window_vr63']:.2f} (full-sample {c10['ndx_full_sample_vr21']:.2f}/{c10['ndx_full_sample_vr63']:.2f}; "
   f"noise A 0.95/0.74)")
pr(f"C10 known-mean VR (no de-meaning) RW vs design-A bridge: q21 {c10['known_mean_vr21_random_walk']:.2f} vs "
   f"{c10['known_mean_vr21_design_A_bridge']:.2f}; q63 {c10['known_mean_vr63_random_walk']:.2f} vs "
   f"{c10['known_mean_vr63_design_A_bridge']:.2f}; q126 {c10['known_mean_vr126_random_walk']:.2f} vs "
   f"{c10['known_mean_vr126_design_A_bridge']:.2f}")
res["C10"] = c10

# ------------------------------------------------------------------ C7 in-sample analogs of S0 / whipsaw windows
w2 = C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
r_fri = C.backtest(P, w2)
r_tr = sum(C.backtest(P, w2, mark=pd.Series(idx.dayofweek == i_, index=idx)) for i_ in range(5)) / 5
st_ = (w2["qld"].where(P["mark"]).ffill().fillna(0) > 0).values
lw = np.cumsum(np.log1p(r_fri.values))
lwt = np.cumsum(np.log1p(r_tr.values))
lnd = np.log(P["ndx"].values)
ndr = P["ndx_ret"].values
insi = np.where((idx >= C.START) & (idx <= C.FREEZE))[0]
a0, b0 = int(insi[0]), int(insi[-1])
rows = []
for i in range(a0, b0 - H + 1):
    if st_[i]:
        continue
    n12 = math.exp(lnd[i + H] - lnd[i]) - 1
    rv = ndr[i + 1:i + H + 1].std(ddof=1) * math.sqrt(252)
    rows.append((i, n12, rv, math.exp(lw[i + H] - lw[i]) - 1, math.exp(lwt[i + H] - lwt[i]) - 1))
A7 = np.array([(r[1], r[2], r[3], r[4]) for r in rows])
I7 = np.array([r[0] for r in rows])
c7 = {}
for vmin in (0.25, 0.20):
    msk = (np.abs(A7[:, 0]) <= 0.10) & (A7[:, 1] >= vmin)
    ii = I7[msk]
    eps = []
    if len(ii):
        start = ii[0]
        prev = ii[0]
        for x in ii[1:]:
            if x - prev > 21:
                eps.append((str(idx[start].date()), str(idx[prev].date())))
                start = x
            prev = x
        eps.append((str(idx[start].date()), str(idx[prev].date())))
    c7[f"flat_abs10_vol_ge{int(vmin*100)}"] = dict(
        n_days=int(msk.sum()), episodes=eps,
        F2_Fri_12m_median=float(np.median(A7[msk, 2])) if msk.any() else None,
        F2_Fri_12m_min=float(A7[msk, 2].min()) if msk.any() else None,
        F2_Fri_12m_max=float(A7[msk, 2].max()) if msk.any() else None,
        F2_tr_12m_median=float(np.median(A7[msk, 3])) if msk.any() else None,
        share_F2_Fri_neg=float((A7[msk, 2] < 0).mean()) if msk.any() else None)
    v = c7[f"flat_abs10_vol_ge{int(vmin*100)}"]
    pr(f"C7 in-sample risk-off starts with |NDX 12m|<=10% and realised vol>={vmin:.0%}: {v['n_days']} days, episodes {v['episodes']}; "
       f"F2 Fri 12m median {v['F2_Fri_12m_median']*100 if v['F2_Fri_12m_median'] is not None else float('nan'):+.1f}% "
       f"(min {v['F2_Fri_12m_min']*100 if v['F2_Fri_12m_min'] is not None else float('nan'):+.1f}, max "
       f"{v['F2_Fri_12m_max']*100 if v['F2_Fri_12m_max'] is not None else float('nan'):+.1f}); tranche median "
       f"{v['F2_tr_12m_median']*100 if v['F2_tr_12m_median'] is not None else float('nan'):+.1f}%; share negative "
       f"{v['share_F2_Fri_neg'] if v['share_F2_Fri_neg'] is not None else float('nan'):.2f}")
sw = st_[a0:b0 + 1]
off_sw = (sw[:-1] & ~sw[1:]).astype(int)
cs = np.concatenate([[0], np.cumsum(off_sw)])
for win, lab in ((126, "6m"), (252, "12m")):
    m_ = np.array([cs[j + win] - cs[j] for j in range(0, len(off_sw) - win + 1)])
    c7[f"max_whipsaws_any_{lab}_window"] = int(m_.max())
    c7[f"share_windows_ge5_{lab}"] = float((m_ >= 5).mean())
    j = int(np.argmax(m_))
    c7[f"max_whipsaws_{lab}_window_start"] = str(idx[a0 + j].date())
pr(f"C7 in-sample F2 Friday-book whipsaws (on->off), any window 1988-freeze: max {c7['max_whipsaws_any_6m_window']} in 6m "
   f"(from {c7['max_whipsaws_6m_window_start']}), max {c7['max_whipsaws_any_12m_window']} in 12m "
   f"(from {c7['max_whipsaws_12m_window_start']}); share of 6m windows with >=5: {c7['share_windows_ge5_6m']*100:.2f}%")
res["C7"] = c7

# ------------------------------------------------------------------ C6 re-entry cost measured directly
SCEN = g["SCEN"]
c6 = {}
for mode, ns in (("A", 500), ("free", 500), ("boot", 300)):
    for nm in [n for n in SCEN if g["SCEN_SHORT"][n] in ("S4", "S5")]:
        diffs, f2s, offs_ = [], [], []
        for sd in range(ns):
            n_, i_, g_ = L.make_path(SCEN[nm], 0.30, sd, mode)
            rr = F.run(n_, i_, g_)
            f2 = L.cum(rr["rets"]["F2"]["Fri"])
            off = L.cum(0.85 * rr["ief_r"] + 0.15 * rr["gold_r"] - 0.0)
            diffs.append(f2 - off)
            f2s.append(f2)
            offs_.append(off)
        sh = g["SCEN_SHORT"][nm]
        c6[f"{sh}_{mode}"] = dict(reentry_cost_median=float(np.median(diffs)), reentry_cost_mean=float(np.mean(diffs)),
                                  F2_median=float(np.median(f2s)), offleg_only_median=float(np.median(offs_)), n=ns)
        pr(f"C6 {sh} noise-{mode} ({ns} seeds): F2 Fri 12m median {np.median(f2s)*100:+.1f}% | never-re-enter off-leg "
           f"{np.median(offs_)*100:+.1f}% | re-entry cost (paired) median {np.median(diffs)*100:+.1f}pp mean {np.mean(diffs)*100:+.1f}pp")
res["C6"] = c6

json.dump(res, open(OUT, "w"), indent=1, default=float)
pr(f"saved {OUT}")
