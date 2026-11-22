"""
r2_verify_scenario_stress_robustness_sims.py -- ROBUSTNESS review of r2_scenario_stress (simulation checks).

usage: python -u update_2026_11/r2_verify_scenario_stress_robustness_sims.py <job> <out.json>
jobs:
  validate : fast numpy evaluator vs the specialist's evaluate() on design-A paths (all scenarios, 3 seeds)
             + 2 historical replays -> max abs differences (must be ~1e-15)
  seeds    : 500 seeds (0..499; the specialist used 0..39) at design A, 30% vol, every scenario:
             median / mean / bootstrap SE of the median of the headline 12m numbers; also seeds 0..39 only
  free     : same 500 seeds, design 'free' = NO day-252 pin (noise is a free random walk after the
             scenario's last own waypoint; S0 = pure random walk)
  boot     : design A pins, NDX noise = circular block bootstrap (21d blocks) of standardised in-sample
             NDX log returns (fat tails, vol clustering, real autocorrelation), 300 seeds
  s6grid   : S6 sensitivity: cycle length T, amplitude A, rising/falling start, pin mode; 30% and 5% noise
Nothing in the specialist's files is modified; all output goes to <out.json>.
"""
from __future__ import annotations
import os, sys, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_verify_scenario_stress_robustness_lib as L  # noqa: E402

job, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
assert not OUT.startswith(os.path.abspath(HERE)), "write reviewer output outside the project folder"
t0 = time.time()
g = L.load_specialist()
F = L.Fast()
SCEN = g["SCEN"]
SHORT = g["SCEN_SHORT"]
H = g["H"]
KEYS = ["NDX_r12m", "QQQ_r12m", "QLD_r12m", "F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F1_tr_r12m", "F2_tr_r12m",
        "F3_tr_r12m", "F2_Fri_mdd", "F2_tr_mdd", "Fri_reentry_day", "Fri_reentry_ndx", "Fri_whip_12m",
        "Fri_time_in_mkt_12m", "F2_Fri_r1m", "OFF_IEF_r1m", "OFF_IEF_r12m", "Fri_participation",
        "Fri_missed_before_invested"]


def agg(rows, keys=KEYS, nboot=400):
    out = {}
    rng = np.random.default_rng(1)
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        if len(v) == 0:
            out[k] = None
            continue
        bs = [np.median(rng.choice(v, len(v))) for _ in range(nboot)]
        out[k] = dict(median=float(np.median(v)), mean=float(v.mean()), p10=float(np.percentile(v, 10)),
                      p90=float(np.percentile(v, 90)), se_median=float(np.std(bs)), n=int(len(v)),
                      share_neg=float((v < 0).mean()))
    return out


res = dict(job=job)

if job == "validate":
    diffs = {}
    for name, spec in SCEN.items():
        for sd in (0, 7, 33):
            n_, i_, g_, t_, b_, _ = g["synth_path"](spec, 0.30, sd, "A")
            ref = g["evaluate"](n_, i_, g_, t_, b_)
            mine = L.summarize(F.run(n_, i_, g_))
            for k in ["F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F1_tr_r12m", "F2_tr_r12m", "F3_tr_r12m",
                      "F2_Fri_mdd", "F2_tr_mdd", "F2_Fri_r1m", "QQQ_r12m", "QLD_r12m", "NDX_r12m", "OFF_IEF_r12m"]:
                diffs.setdefault(k, 0.0)
                diffs[k] = max(diffs[k], abs(ref[k] - mine[k]))
            for k in ["Fri_reentry_day", "Fri_whip_12m"]:
                assert ref[k] == mine[k], (name, sd, k, ref[k], mine[k])
            for k in ["Fri_participation", "Fri_missed_before_invested"]:
                assert (ref[k] is None) == (mine[k] is None), (name, sd, k)
                if ref[k] is not None:
                    assert abs(ref[k] - mine[k]) < 1e-12, (name, sd, k, ref[k], mine[k])
            assert abs(ref["Fri_time_in_mkt_12m"] - mine["Fri_time_in_mkt_12m"]) < 1e-12
            assert abs(ref["tr_whip_12m"] - mine["tr_whip_12m"]) < 1e-12
            if ref["Fri_reentry_day"]:
                assert abs(ref["F2_Fri_size_at_reentry"] - mine["F2_Fri_size_at_reentry"]) < 1e-12
    P = g["P"]
    for dte in ("2020-03-06", "1998-08-31"):
        i = P.index.get_loc(__import__("pandas").Timestamp(dte))
        sg = P.iloc[i:i + H]
        n_, i_, g_ = sg["ndx_ret"].values, sg["ief"].values, sg["gold_phys"].values
        ref = g["evaluate"](n_, i_, g_, sg["tlt"].values, np.full(H, g["RFD"]))
        mine = L.summarize(F.run(n_, i_, g_))
        for k in ["F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F2_tr_r12m"]:
            diffs[k] = max(diffs[k], abs(ref[k] - mine[k]))
        assert ref["Fri_reentry_day"] == mine["Fri_reentry_day"]
    res["max_abs_diff_fast_vs_specialist"] = diffs
    res["n_paths_checked"] = len(SCEN) * 3 + 2
    print("fast vs specialist max |diff|:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-10

elif job in ("seeds", "free", "boot"):
    nseed = 500 if job in ("seeds", "free") else 300
    mode = {"seeds": "A", "free": "free", "boot": "boot"}[job]
    names = list(SCEN)
    out = {}
    paired = {}
    rows_by = {}
    for name in names:
        spec = SCEN[name]
        rows = []
        for sd in range(nseed):
            n_, i_, g_ = L.make_path(spec, 0.30, sd, mode)
            m = L.summarize(F.run(n_, i_, g_))
            m["seed"] = sd
            rows.append(m)
        rows_by[SHORT[name]] = rows
        out[SHORT[name]] = dict(all=agg(rows), first40=agg(rows[:40], nboot=200))
        a = out[SHORT[name]]["all"]
        print(f"{SHORT[name]:4s} {mode:5s} n={nseed}: F2 Fri 12m med {a['F2_Fri_r12m']['median']*100:+6.1f} "
              f"(se {a['F2_Fri_r12m']['se_median']*100:.1f}) mean {a['F2_Fri_r12m']['mean']*100:+6.1f} | "
              f"F2 tr {a['F2_tr_r12m']['median']*100:+6.1f} | F1 {a['F1_Fri_r12m']['median']*100:+6.1f} "
              f"F3 {a['F3_Fri_r12m']['median']*100:+6.1f} | QQQ med {a['QQQ_r12m']['median']*100:+6.1f} "
              f"mean {a['QQQ_r12m']['mean']*100:+6.1f} QLD {a['QLD_r12m']['median']*100:+6.1f} | "
              f"F2mdd {a['F2_Fri_mdd']['median']*100:6.1f} | reentry day med "
              f"{a['Fri_reentry_day']['median'] if a['Fri_reentry_day'] else None} | whip "
              f"{a['Fri_whip_12m']['mean']:.2f} | {time.time()-t0:.0f}s", flush=True)
    s0 = rows_by["S0"]
    for sh in ("S1+", "S1-", "S2-", "S2+"):
        rr = rows_by[sh]
        pd_ = {}
        for k in ["F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F2_tr_r12m", "OFF_IEF_r1m", "F2_Fri_r1m"]:
            v = np.array([a[k] - b[k] for a, b in zip(rr, s0)])
            pd_[k] = dict(median=float(np.median(v)), mean=float(v.mean()), p10=float(np.percentile(v, 10)),
                          p90=float(np.percentile(v, 90)))
        paired[sh] = pd_
        print(f"  paired {sh} - S0: risk-off 1m {pd_['OFF_IEF_r1m']['median']*100:+.2f}pp | F2 12m "
              f"{pd_['F2_Fri_r12m']['median']*100:+.2f}pp (p10 {pd_['F2_Fri_r12m']['p10']*100:+.2f}, p90 "
              f"{pd_['F2_Fri_r12m']['p90']*100:+.2f}) | F1 {pd_['F1_Fri_r12m']['median']*100:+.2f} F3 "
              f"{pd_['F3_Fri_r12m']['median']*100:+.2f}", flush=True)
    # replicate the specialist's sampling error: 12 disjoint batches of 40 seeds -> spread of the 40-seed median
    batch = {}
    for sh, rows in rows_by.items():
        meds = [float(np.median([r["F2_Fri_r12m"] for r in rows[j * 40:(j + 1) * 40]])) for j in range(nseed // 40)]
        batch[sh] = dict(min=min(meds), max=max(meds), sd=float(np.std(meds, ddof=1)), n_batches=len(meds))
    res.update(mode=mode, n_seeds=nseed, scenarios=out, paired_vs_S0=paired, batch40_median_spread_F2_Fri=batch)
    print("40-seed batch medians of F2 Fri 12m (min..max):",
          {k: f"{v['min']*100:+.1f}..{v['max']*100:+.1f}" for k, v in batch.items()})

elif job == "s6grid":
    grid = []
    base = dict(T=42, A=0.08, falling=False, pin="quarter")
    cases = []
    for T in (9, 14, 18, 21, 25.2, 31.5, 42, 63, 126):
        cases.append(dict(base, T=T))
    for A in (0.07, 0.10, 0.12):
        cases.append(dict(base, A=A))
        cases.append(dict(base, T=21, A=A))
    for T in (21, 42):
        cases.append(dict(base, T=T, falling=True))
        cases.append(dict(base, T=T, pin="end"))
        cases.append(dict(base, T=T, pin="none"))
    for c in cases:
        spec = L.osc_spec(c["T"], c["A"], falling=c["falling"], pin_mode=c["pin"])
        row = dict(c)
        for vol, ns in ((0.30, 40), (0.05, 10)):
            rows = []
            for sd in range(ns):
                if c["pin"] == "none" or c["pin"] == "end":
                    n_, i_, g_ = L.make_path(spec, vol, sd, "free") if c["pin"] == "none" else L.make_path(spec, vol, sd, "A")
                else:
                    n_, i_, g_ = L.make_path(spec, vol, sd, "A")
                rows.append(L.summarize(F.run(n_, i_, g_)))
            a = agg(rows, ["F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F2_tr_r12m", "QLD_r12m", "NDX_r12m",
                           "Fri_whip_12m", "F2_Fri_mdd"], nboot=100)
            row[f"vol{int(vol*100)}"] = a
        grid.append(row)
        a = row["vol30"]
        b = row["vol5"]
        print(f"T={c['T']:6.1f} A={c['A']:.2f} {'fall' if c['falling'] else 'rise'} pin={c['pin']:7s} | 30%: F2 "
              f"{a['F2_Fri_r12m']['median']*100:+6.1f} tr {a['F2_tr_r12m']['median']*100:+6.1f} F1 "
              f"{a['F1_Fri_r12m']['median']*100:+6.1f} QLD {a['QLD_r12m']['median']*100:+6.1f} NDX "
              f"{a['NDX_r12m']['median']*100:+6.1f} whip {a['Fri_whip_12m']['mean']:.1f} | 5%: F2 "
              f"{b['F2_Fri_r12m']['median']*100:+6.1f} tr {b['F2_tr_r12m']['median']*100:+6.1f} whip "
              f"{b['Fri_whip_12m']['mean']:.1f} | {time.time()-t0:.0f}s", flush=True)
    res["grid"] = grid

elif job == "decomp":
    # C4 decomposition of F2-Friday's 12m shortfall vs QQQ on the V-rebound (and other) scenarios:
    #   always-on  = same sizing rules with the EMA gate forced ON every day (no entry lag, no exits)
    #   no-exit    = actual gate until the first Friday re-entry, gate forced ON afterwards (entry lag only)
    #   actual     = the rules.  entry-lag cost = always-on - no-exit ; post-entry exit/whipsaw cost = no-exit - actual
    nseed = 500
    out = {}
    for name in [n for n in SCEN if SHORT[n] in ("S0", "S3", "S3c", "S4", "S6a")]:
        spec = SCEN[name]
        rows = []
        for sd in range(nseed):
            n_, i_, g_ = L.make_path(spec, 0.30, sd, "A")
            base = F.run(n_, i_, g_)
            k = L.first_on(base["states"]["Fri"], 1)
            on = F.run(n_, i_, g_, above_override=lambda a: np.ones_like(a, dtype=bool))
            if k:
                def ov(a, k=k):
                    b = a.copy()
                    b[k - 1:] = True
                    return b
                ne = F.run(n_, i_, g_, above_override=ov)
            else:
                ne = base
            rows.append(dict(actual=L.cum(base["rets"]["F2"]["Fri"], H), noexit=L.cum(ne["rets"]["F2"]["Fri"], H),
                             always=L.cum(on["rets"]["F2"]["Fri"], H), qqq=L.cum(base["qqq"], H),
                             qld=L.cum(base["qld"], H)))
        a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        d = dict({k: dict(median=float(np.median(v)), mean=float(v.mean())) for k, v in a.items()},
                 entry_lag_cost=dict(median=float(np.median(a["always"] - a["noexit"])), mean=float((a["always"] - a["noexit"]).mean())),
                 exit_whipsaw_cost=dict(median=float(np.median(a["noexit"] - a["actual"])), mean=float((a["noexit"] - a["actual"]).mean())),
                 sizing_drag_vs_qqq=dict(median=float(np.median(a["qqq"] - a["always"])), mean=float((a["qqq"] - a["always"]).mean())),
                 total_shortfall_vs_qqq=dict(median=float(np.median(a["qqq"] - a["actual"])), mean=float((a["qqq"] - a["actual"]).mean())))
        out[SHORT[name]] = d
        print(f"{SHORT[name]:4s}: QQQ {d['qqq']['mean']*100:+6.1f} | always-on F2 {d['always']['mean']*100:+6.1f} | no-exit "
              f"{d['noexit']['mean']*100:+6.1f} | actual {d['actual']['mean']*100:+6.1f} (means) || shortfall vs QQQ "
              f"{d['total_shortfall_vs_qqq']['mean']*100:+6.1f}pp = sizing/vol-drag {d['sizing_drag_vs_qqq']['mean']*100:+6.1f} + "
              f"entry lag {d['entry_lag_cost']['mean']*100:+6.1f} + exits/whipsaws {d['exit_whipsaw_cost']['mean']*100:+6.1f} | "
              f"medians: lag {d['entry_lag_cost']['median']*100:+.1f} exits {d['exit_whipsaw_cost']['median']*100:+.1f} "
              f"| {time.time()-t0:.0f}s", flush=True)
    res["decomp"] = out

res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(f"saved {OUT} ({time.time()-t0:.0f}s)")
