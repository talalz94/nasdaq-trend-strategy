"""
Integrity reviewer checks for r2_scenario_stress.py (read-only: writes nothing inside the project).
The specialist's definitions (constants, build, run_books, evaluate, synth_path, SCEN) are loaded by
exec'ing its source UP TO the validation section, so none of its heavy runs and no file writes happen.

A  independent full-history engine (pandas ewm/rolling on the whole extended panel, C.strategy_weights,
   C.backtest, own metric code) vs the specialist's evaluate() on synthetic paths + all literal replays
B  look-ahead: perturb the path after day k; decisions and returns up to day k must not change
C  Friday threshold ladder by bisection through the machinery (+ per-day decline of the bar)
D  S1/S2 'pure shock' pairing: specialist pins S1's 10y noise at day 21 (S2's gold at 42) but S0's
   only at 252 -> paired bands contain mismatched noise; rerun with S0's pin set
E  C9 decomposition: today's-state replay minus own-state original = state + financing/real-ETF +
   calendar (weekday & Aug/Sep season mapping); isolate each
F  matched-seed 30% vs 5% NDX noise for S4/S5 (C6 'bear-rally re-entries cost 9-13pp')
G  data checks: ndx_ret vs ndx levels in replay windows, NaNs, freeze, gold source
H  NDX variance ratios under the SAME 252-day-window estimator used for the noise designs
"""
import json, math, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_scenario_stress.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index("# ------------------------------------------------------------------ VALIDATION")
G = {"__file__": SRC, "__name__": "spec_defs"}
exec(compile(code[:cut], SRC, "exec"), G)
P, C, FWD, H, L0, EMA0, RFD = G["P"], G["C"], G["FWD"], G["H"], G["L0"], G["EMA0"], G["RFD"]
build, run_books, evaluate, synth_path, SCEN = G["build"], G["run_books"], G["evaluate"], G["synth_path"], G["SCEN"]
TIERS, DAYS, ALPHA = G["TIERS"], G["DAYS"], G["ALPHA"]
idx = P.index
T0 = time.time()
OUT = {}
STORED = json.load(open(os.path.join(HERE, "r2_scenario_stress.json")))


def sname(short):
    return [k for k in SCEN if k.split(" ")[0] == short][0]


# ------------------------------------------------------------------ A: independent engine
COLS = ["ndx", "qqq", "qld", "tqqq", "ief", "gold_phys"]
HIST = P[COLS].copy()
LEVMULT = {"qqq": 1, "qld": 2, "tqqq": 3}


def indep(ndx_r, ief_r, gold_r, sleeves=None, rfd=None, months=None, dows=None):
    """append forward rows to the WHOLE panel, recompute every signal with pandas, run the rules."""
    ndx_r = np.asarray(ndx_r, float)
    n = len(ndx_r)
    rfd_ = RFD if rfd is None else np.asarray(rfd, float)
    f = pd.DataFrame(index=FWD[:n])
    f["ndx"] = float(P["ndx"].iloc[-1]) * np.cumprod(1 + ndx_r)
    if sleeves is None:
        f["qqq"] = ndx_r - 0.0020 / 252
        f["qld"] = 2 * ndx_r - 0.0095 / 252 - rfd_
        f["tqqq"] = 3 * ndx_r - 0.0095 / 252 - 2 * rfd_
    else:
        for k in ["qqq", "qld", "tqqq"]:
            f[k] = np.asarray(sleeves[k], float)
    f["ief"] = np.asarray(ief_r, float)
    f["gold_phys"] = np.asarray(gold_r, float)
    X = pd.concat([HIST, f])
    nd = X["ndx"]
    X["ema100"] = nd.ewm(span=100, adjust=False).mean()
    X["above"] = nd > X["ema100"]
    X["vol20"] = nd.pct_change().rolling(20).std() * np.sqrt(252)
    X["er20"] = nd.diff(20).abs() / nd.diff().abs().rolling(20).sum().replace(0, np.nan)
    mo = np.asarray(X.index.month).copy()
    dw = np.asarray(X.index.dayofweek).copy()
    if months is not None:
        mo[-n:] = months
    if dows is not None:
        dw[-n:] = dows
    X["month"] = mo
    res = {"lv": np.concatenate([[float(P["ndx"].iloc[-1])], f["ndx"].values])}
    for tier, kw in TIERS.items():
        w = C.strategy_weights(X, lev=kw["lev"], voltgt=kw["voltgt"], gold="gold_phys", gold_w=0.15, off="ief")
        rs = {}
        for i, d in enumerate(DAYS):
            mk = pd.Series(dw == i, index=X.index)
            r = C.backtest(X, w, mark=mk)
            rs[d] = r.values[-n:]
            st = (w[kw["lev"]].where(mk).ffill().fillna(0) > 0).values[-n - 1:]
            res[(tier, d, "st")] = st
        res[(tier, "Fri")] = rs["Fri"]
        res[(tier, "tr")] = sum(rs[d] for d in DAYS) / 5
    res["qqq"] = f["qqq"].values
    return res


def imetrics(res):
    m = {}
    for tier in TIERS:
        for b in ("Fri", "tr"):
            r = res[(tier, b)]
            m[f"{tier}_{b}_r12m"] = float(np.prod(1 + r[:252]) - 1)
            e = np.concatenate([[1.0], np.cumprod(1 + r[:252])])
            m[f"{tier}_{b}_mdd"] = float((e / np.maximum.accumulate(e) - 1).min())
    s = res[("F2", "Fri", "st")]
    on = np.flatnonzero(s[1:])
    k = int(on[0] + 1) if len(on) else None
    m["Fri_reentry_day"] = k
    m["Fri_reentry_ndx"] = float(res["lv"][k]) if k else None
    m["Fri_whip_12m"] = int(np.sum(s[:-1] & ~s[1:]))
    m["Fri_time_in_mkt_12m"] = float(s[:-1].mean())
    firsts = []
    for d in DAYS:
        z = res[("F2", d, "st")]
        o = np.flatnonzero(z[1:])
        if len(o):
            firsts.append(int(o[0] + 1))
    m["tr_first_reentry_day"] = min(firsts) if firsts else None
    a5 = np.logical_and.reduce([res[("F2", d, "st")] for d in DAYS])
    o5 = np.flatnonzero(a5[1:])
    m["tr_all5_day"] = int(o5[0] + 1) if len(o5) else None
    for tier in ("F1", "F3"):
        for d in DAYS:
            assert np.array_equal(res[(tier, d, "st")], res[("F2", d, "st")])
    return m


print("A: independent engine vs specialist evaluate() ...", flush=True)
# sanity: the independent engine reproduces the panel's own signals on the history rows
_X = HIST["ndx"].ewm(span=100, adjust=False).mean()
OUT["A_hist_ema_vs_panel_maxabs"] = float(np.nanmax(np.abs(_X.values - P["ema100"].values)))
KEYS_A = ["F1_Fri_r12m", "F2_Fri_r12m", "F3_Fri_r12m", "F1_tr_r12m", "F2_tr_r12m", "F3_tr_r12m", "F2_Fri_mdd",
          "F2_tr_mdd", "Fri_reentry_day", "Fri_reentry_ndx", "Fri_whip_12m", "Fri_time_in_mkt_12m",
          "tr_first_reentry_day", "tr_all5_day"]
cmpA, worstA = 0, 0.0
mismatchA = []
for short in ["S0", "S1+", "S3", "S3c", "S4", "S5", "S6a", "S6b"]:
    for sd in (0, 7, 23):
        n_, i_, g_, t_, b_, _ = synth_path(SCEN[sname(short)], 0.30, sd, "A")
        ms = evaluate(n_, i_, g_, t_, b_)
        mi = imetrics(indep(n_, i_, g_))
        for k in KEYS_A:
            a, b = ms[k], mi[k]
            cmpA += 1
            if (a is None) != (b is None):
                mismatchA.append((short, sd, k, a, b)); continue
            if a is None:
                continue
            dlt = abs(float(a) - float(b))
            worstA = max(worstA, dlt)
            if dlt > 1e-9:
                mismatchA.append((short, sd, k, a, b))
# replays (literal) through the independent engine vs the stored JSON
for dte in G.get("REPLAYS", ["2018-12-07", "2022-01-21", "2000-04-14", "2011-08-05", "2025-04-04", "2020-03-06",
                             "2015-08-24", "1990-08-10", "1998-08-31"]):
    i = idx.get_loc(pd.Timestamp(dte))
    sg = P.iloc[i:i + H]
    mi = imetrics(indep(sg["ndx_ret"].values, sg["ief"].values, sg["gold_phys"].values))
    st_ = STORED["replays"][dte]["literal"]
    for k in KEYS_A:
        a, b = st_[k], mi[k]
        cmpA += 1
        if (a is None) != (b is None):
            mismatchA.append(("replay", dte, k, a, b)); continue
        if a is None:
            continue
        dlt = abs(float(a) - float(b))
        worstA = max(worstA, dlt)
        if dlt > 1e-9:
            mismatchA.append(("replay", dte, k, a, b))
OUT["A_independent_engine"] = dict(n_compared=cmpA, max_abs_diff=worstA, mismatches=mismatchA[:20])
print(f"  hist EMA vs panel max|d| {OUT['A_hist_ema_vs_panel_maxabs']:.2e}; compared {cmpA}, max|diff| {worstA:.2e}, "
      f"mismatches {len(mismatchA)}  ({time.time()-T0:.0f}s)", flush=True)

# ------------------------------------------------------------------ B: look-ahead perturbation
print("B: look-ahead perturbation ...", flush=True)
lookB = []
for short, sd, k in (("S3", 3, 25), ("S0", 11, 60), ("S6a", 5, 100), ("S4", 2, 140)):
    n_, i_, g_, t_, b_, _ = synth_path(SCEN[sname(short)], 0.30, sd, "A")
    rng = np.random.default_rng(99)
    n2, i2, g2 = n_.copy(), i_.copy(), g_.copy()
    n2[k:] = rng.normal(0, 0.04, H - k)
    i2[k:] = rng.normal(0, 0.01, H - k)
    g2[k:] = rng.normal(0, 0.02, H - k)
    E1, p1 = build(G["POS_LAST"], FWD, n_, i_, g_)
    E2, p2 = build(G["POS_LAST"], FWD, n2, i2, g2)
    r1, s1 = run_books(E1, p1)
    r2, s2 = run_books(E2, p2)
    dec_diff = max(float(np.abs(s1[key][:k + 1] - s2[key][:k + 1]).max()) for key in s1)
    ret_diff = max(float(np.abs(r1[key][:k] - r2[key][:k]).max()) for key in r1)
    after = max(float(np.abs(r1[key][k:] - r2[key][k:]).max()) for key in r1)
    # same through the independent engine
    q1, q2 = indep(n_, i_, g_), indep(n2, i2, g2)
    dec_i = max(float(np.abs(q1[(t, d, "st")][:k + 1].astype(float) - q2[(t, d, "st")][:k + 1].astype(float)).max())
                for t in TIERS for d in DAYS)
    ret_i = max(float(np.abs(q1[(t, b)][:k] - q2[(t, b)][:k]).max()) for t in TIERS for b in ("Fri", "tr"))
    lookB.append(dict(scenario=short, seed=sd, k=k, decisions_upto_k_maxdiff=dec_diff, returns_upto_k_maxdiff=ret_diff,
                      returns_after_k_changed=after, indep_dec=dec_i, indep_ret=ret_i))
OUT["B_lookahead"] = lookB
for x in lookB:
    print("  ", x, flush=True)

# ------------------------------------------------------------------ C: threshold ladder by bisection
print("C: Friday threshold ladder by bisection ...", flush=True)


def fri_on_at(j, level):
    """flat path, then NDX jumps to `level` on forward day j+1 (FWD[j]); F2 Friday decision at that close."""
    r = np.zeros(H)
    r[j] = level / L0 - 1
    E, p0 = build(G["POS_LAST"], FWD, r, np.zeros(H), np.zeros(H))
    w = C.strategy_weights(E, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
    mk = pd.Series(E.index.dayofweek == 4, index=E.index)
    st = (w["qld"].where(mk).ffill().fillna(0) > 0).values[p0:]
    return bool(st[j + 1])


lad = STORED["today"]["friday_threshold_ladder_flat_path"]
ladC = []
for row in lad:
    j = int(np.flatnonzero(FWD == pd.Timestamp(row["friday"]))[0])
    lo, hi = L0, 1.15 * L0
    assert not fri_on_at(j, lo) and fri_on_at(j, hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fri_on_at(j, mid):
            hi = mid
        else:
            lo = mid
    ladC.append(dict(friday=row["friday"], fwd_day=j + 1, bisected=round(hi, 2), stored=row["threshold"],
                     pct=round(hi / L0 - 1, 5)))
# holiday Fridays: is there any Friday mark on 2026-12-25 / 2027-01-01 (no: not trading days)
hol_in_fwd = [d for d in ["2026-12-25", "2027-01-01", "2027-03-26", "2027-06-18"] if pd.Timestamp(d) in FWD]
# per-day decline of the bar (EMA_{t-1} on a flat path), in % of today's level and pp of the threshold
jj = np.arange(0, 80)
bar = L0 + (EMA0 - L0) * (1 - ALPHA) ** jj
dec = -np.diff(bar) / L0
OUT["C_ladder"] = dict(rows=ladC, max_abs_diff=max(abs(r["bisected"] - r["stored"]) for r in ladC),
                       holiday_fridays_in_fwd=hol_in_fwd,
                       bar_decline_pct_per_day=dict(day1=float(dec[0]), day17_dec11=float(dec[16]),
                                                    day35_jan08=float(dec[34]), day64_feb19=float(dec[63]),
                                                    max=float(dec.max())))
print(f"  ladder max |bisected - stored| = {OUT['C_ladder']['max_abs_diff']:.2f} pts; first {ladC[0]}; "
      f"holiday Fridays in FWD: {hol_in_fwd}", flush=True)
print(f"  bar decline per trading day (% of 27,002): {OUT['C_ladder']['bar_decline_pct_per_day']}", flush=True)

# ------------------------------------------------------------------ D: S1/S2 pairing with matched noise pins
print("D: S1/S2 minus S0 with the control's own noise pin set ...", flush=True)
s0 = SCEN[sname("S0")]
BASE = {sd: evaluate(*synth_path(s0, 0.30, sd, "A")[:5]) for sd in range(40)}
D_out = {}
for short, key_over in (("S1+", "wy"), ("S1-", "wy"), ("S2-", "wg"), ("S2+", "wg")):
    spec = dict(SCEN[sname(short)])
    spec_m = dict(spec)
    spec_m[key_over] = []          # identical pin set to S0 -> identical 10y / gold noise path
    diffs = {"orig": {"OFF_IEF_r1m": [], "F2_Fri_r12m": [], "OFF_IEF_r12m": []},
             "matched": {"OFF_IEF_r1m": [], "F2_Fri_r12m": [], "OFF_IEF_r12m": []}}
    for sd in range(40):
        base = BASE[sd]
        for tag, sp in (("orig", spec), ("matched", spec_m)):
            mm = evaluate(*synth_path(sp, 0.30, sd, "A")[:5])
            for k in diffs[tag]:
                diffs[tag][k].append(mm[k] - base[k])
    D_out[short] = {tag: {k: dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                  p90=float(np.percentile(v, 90))) for k, v in dd.items()} for tag, dd in diffs.items()}
    o, mt = D_out[short]["orig"], D_out[short]["matched"]
    print(f"  {short}: OFF 1m orig {o['OFF_IEF_r1m']['median']*100:+.2f} [{o['OFF_IEF_r1m']['p10']*100:+.2f}/"
          f"{o['OFF_IEF_r1m']['p90']*100:+.2f}] matched {mt['OFF_IEF_r1m']['median']*100:+.2f} "
          f"[{mt['OFF_IEF_r1m']['p10']*100:+.2f}/{mt['OFF_IEF_r1m']['p90']*100:+.2f}] | F2 12m orig "
          f"{o['F2_Fri_r12m']['median']*100:+.2f} [{o['F2_Fri_r12m']['p10']*100:+.2f}/{o['F2_Fri_r12m']['p90']*100:+.2f}]"
          f" matched {mt['F2_Fri_r12m']['median']*100:+.2f} [{mt['F2_Fri_r12m']['p10']*100:+.2f}/"
          f"{mt['F2_Fri_r12m']['p90']*100:+.2f}]  ({time.time()-T0:.0f}s)", flush=True)
OUT["D_pairing"] = D_out

# ------------------------------------------------------------------ E: decomposition of C9
print("E: C9 decomposition (state vs financing/real sleeves vs calendar) ...", flush=True)
DOW = {d: pd.Series(idx.dayofweek == i, index=idx) for i, d in enumerate(DAYS)}
wF2 = C.strategy_weights(P, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
rF2 = {d: C.backtest(P, wF2, mark=DOW[d]) for d in DAYS}
E_rows = []
for dte in ["2018-12-07", "2022-01-21", "2000-04-14", "2011-08-05", "2025-04-04", "2020-03-06", "2015-08-24",
            "1990-08-10", "1998-08-31"]:
    i = idx.get_loc(pd.Timestamp(dte))
    s0_ = i + 1                                    # 'after' convention: path = X+1 .. X+H
    sg = P.iloc[s0_:s0_ + H]
    A_fri = float(np.prod(1 + rF2["Fri"].values[s0_:s0_ + H]) - 1)
    A_tr = float(np.prod(1 + (sum(rF2[d] for d in DAYS) / 5).values[s0_:s0_ + H]) - 1)
    own_m, own_d = sg.index.month.values, sg.index.dayofweek.values
    hist_sl = {k: sg[k].values for k in ["qqq", "qld", "tqqq"]}
    nr, ir, gr = sg["ndx_ret"].values, sg["ief"].values, sg["gold_phys"].values
    B = indep(nr, ir, gr, sleeves=hist_sl, months=own_m, dows=own_d)          # today's state only
    Bh = indep(nr, ir, gr, rfd=sg["rf_d"].values, months=own_m, dows=own_d)    # sim at the era's own rf
    Cc = indep(nr, ir, gr, months=own_m, dows=own_d)                           # + today's 4.14% financing
    D = indep(nr, ir, gr)                                                       # + forward calendar
    row = dict(date=dte, A_fri=A_fri, A_tr=A_tr)
    for nm, rr in (("B", B), ("Bh", Bh), ("C", Cc), ("D", D)):
        row[f"{nm}_fri"] = float(np.prod(1 + rr[("F2", "Fri")]) - 1)
        row[f"{nm}_tr"] = float(np.prod(1 + rr[("F2", "tr")]) - 1)
    row["D_matches_stored_after_fri"] = abs(row["D_fri"] - STORED["replays"][dte]["after"]["F2_Fri_r12m"])
    row["D_matches_stored_after_tr"] = abs(row["D_tr"] - STORED["replays"][dte]["after"]["F2_tr_r12m"])
    row["A_matches_stored_fri"] = abs(A_fri - STORED["replays"][dte]["history_context_close_of_date"]["F2_Fri_r12m_original"])
    row["mean_rf_path"] = float(sg["rf"].mean())
    E_rows.append(row)
    print(f"  {dte}: own A Fri {A_fri*100:+6.1f} tr {A_tr*100:+6.1f} | B(state only) Fri {row['B_fri']*100:+6.1f} "
          f"tr {row['B_tr']*100:+6.1f} | Bh(sim@own rf) tr {row['Bh_tr']*100:+6.1f} | C(+4.14% fin) tr {row['C_tr']*100:+6.1f}"
          f" | D(+fwd calendar=spec 'after') Fri {row['D_fri']*100:+6.1f} tr {row['D_tr']*100:+6.1f} | rf {row['mean_rf_path']*100:.2f}%"
          f" | chk D {row['D_matches_stored_after_tr']:.1e} A {row['A_matches_stored_fri']:.1e}", flush=True)
df = pd.DataFrame(E_rows)
dec = {}
for b in ("fri", "tr"):
    dec[b] = dict(total_D_minus_A_median=float((df[f"D_{b}"] - df[f"A_{b}"]).median()),
                  total_D_minus_A_mean=float((df[f"D_{b}"] - df[f"A_{b}"]).mean()),
                  state_B_minus_A_median=float((df[f"B_{b}"] - df[f"A_{b}"]).median()),
                  state_B_minus_A_mean=float((df[f"B_{b}"] - df[f"A_{b}"]).mean()),
                  sleeves_Bh_minus_B_median=float((df[f"Bh_{b}"] - df[f"B_{b}"]).median()),
                  financing_C_minus_Bh_median=float((df[f"C_{b}"] - df[f"Bh_{b}"]).median()),
                  financing_C_minus_Bh_mean=float((df[f"C_{b}"] - df[f"Bh_{b}"]).mean()),
                  calendar_D_minus_C_median=float((df[f"D_{b}"] - df[f"C_{b}"]).median()),
                  calendar_D_minus_C_mean=float((df[f"D_{b}"] - df[f"C_{b}"]).mean()),
                  calendar_D_minus_C_meanabs=float((df[f"D_{b}"] - df[f"C_{b}"]).abs().mean()))
OUT["E_C9_decomposition"] = dict(rows=E_rows, summary=dec)
for b in dec:
    print(f"  F2 {b}: " + ", ".join(f"{k} {v*100:+.2f}pp" for k, v in dec[b].items()), flush=True)

# ------------------------------------------------------------------ F: matched seeds 30% vs 5% (S4, S5, S3)
print("F: matched-seed 30% vs 5% NDX noise ...", flush=True)
F_out = {}
for short in ("S4", "S5", "S3"):
    rows = []
    for sd in range(10):
        a = evaluate(*synth_path(SCEN[sname(short)], 0.30, sd, "A")[:5])
        b = evaluate(*synth_path(SCEN[sname(short)], 0.05, sd, "A")[:5])
        rows.append(dict(seed=sd, F2_30=a["F2_Fri_r12m"], F2_05=b["F2_Fri_r12m"], OFF_30=a["OFF_IEF_r12m"],
                         OFF_05=b["OFF_IEF_r12m"], tim30=a["Fri_time_in_mkt_12m"], tim05=b["Fri_time_in_mkt_12m"]))
    d_ = pd.DataFrame(rows)
    F_out[short] = dict(paired_F2_30_minus_05_median=float((d_.F2_30 - d_.F2_05).median()),
                        paired_F2_30_minus_05_mean=float((d_.F2_30 - d_.F2_05).mean()),
                        F2_30_median_seeds0_9=float(d_.F2_30.median()), F2_05_median=float(d_.F2_05.median()),
                        off_leg_identical=float((d_.OFF_30 - d_.OFF_05).abs().max()),
                        F2_05_minus_OFF_median=float((d_.F2_05 - d_.OFF_05).median()),
                        time_in_mkt_05_median=float(d_.tim05.median()))
    print(f"  {short}: {F_out[short]}", flush=True)
OUT["F_matched_noise"] = F_out

# ------------------------------------------------------------------ G: data checks on replay windows
print("G: data checks ...", flush=True)
G_out = {}
pc = P["ndx"].pct_change()
for dte in ["2018-12-07", "2022-01-21", "2000-04-14", "2011-08-05", "2025-04-04", "2020-03-06", "2015-08-24",
            "1990-08-10", "1998-08-31"]:
    i = idx.get_loc(pd.Timestamp(dte))
    sg = P.iloc[i:i + H + 1]
    G_out[dte] = dict(max_abs_ndxret_minus_pct_change=float((sg["ndx_ret"] - pc.iloc[i:i + H + 1]).abs().max()),
                      end=str(sg.index[-1].date()), before_freeze=bool(sg.index[-1] <= C.FREEZE),
                      gold_interp_share=float((sg["gold_src"].iloc[:H] == "monthly-interp").mean()),
                      nan_any=bool(sg[["ndx_ret", "ief", "gold_phys"]].iloc[:H].isna().any().any()))
OUT["G_data"] = G_out
print("  max |ndx_ret - ndx.pct_change()| over replay windows:",
      max(v["max_abs_ndxret_minus_pct_change"] for v in G_out.values()), flush=True)
print("  all replay windows end before freeze:", all(v["before_freeze"] for v in G_out.values()), flush=True)

# ------------------------------------------------------------------ H: NDX VR under the 252-day-window estimator
print("H: NDX variance ratios, like-for-like estimator ...", flush=True)
x = np.log1p(P["ndx_ret"][(idx >= C.START) & (idx <= C.FREEZE)].values)
wins = [x[s:s + H] for s in range(0, len(x) - H + 1, H)]
vr = {}
for q in (21, 63):
    num = np.mean([pd.Series(w_).rolling(q).sum().dropna().var() for w_ in wins])
    den = q * np.mean([w_.var(ddof=1) for w_ in wins])
    vr[f"vr{q}"] = float(num / den)
OUT["H_ndx_vr_252d_windows"] = dict(n_windows=len(wins), **vr)
print(f"  NDX 1988-freeze in non-overlapping 252d windows: {OUT['H_ndx_vr_252d_windows']}", flush=True)

# ------------------------------------------------------------------ misc: re-entry size / F3 vs F2 / instants
syn = STORED["synthetic"]
misc = {}
for short in ("S0", "S3", "S4", "S5"):
    a = syn[sname(short)]["A_vol0.30"]["agg"]
    misc[short] = dict(F2_size_med=a["F2_Fri_size_at_reentry"]["median"], F3_size_med=a["F3_Fri_size_at_reentry"]["median"],
                       vol20_med=a["Fri_vol20_at_reentry"]["median"], er20_med=a["Fri_er20_at_reentry"]["median"],
                       F2_size_p10=a["F2_Fri_size_at_reentry"]["p10"], F2_size_p90=a["F2_Fri_size_at_reentry"]["p90"])
rep = STORED["replays"]
misc["replays_F3_gt_F2_literal"] = [d for d in rep if rep[d]["literal"]["F3_Fri_r12m"] > rep[d]["literal"]["F2_Fri_r12m"]]
misc["replays_F2_gt_QQQ_literal"] = [d for d in rep if rep[d]["literal"]["F2_Fri_r12m"] > rep[d]["literal"]["QQQ_r12m"]]
D_, Cx = 7.09, 76.9
misc["ief_shock_+100"] = -D_ * 0.01 + 0.5 * Cx * 1e-4
misc["book_+100"] = 0.85 * misc["ief_shock_+100"]
misc["ratio_rate_to_gold"] = abs(misc["book_+100"]) / 0.0225
misc["S3_share_logclimb_at_reentry"] = math.log(29649.92 / L0) / math.log(31224.0 / L0)
# recent empirical IEF duration to the freeze (context for the 7.09 model)
q = P[(idx > C.FREEZE - pd.Timedelta(days=730)) & (idx <= C.FREEZE)]
dy = (q["tnx"] / 100).diff()
car = (q["tnx"] / 100).shift(1) / 252
dfr = pd.concat([q["ief_real"] - car, dy], axis=1).dropna()
dfr.columns = ["y", "dy"]
dfr = dfr[dfr["dy"] != 0]
beta = np.linalg.lstsq(np.column_stack([np.ones(len(dfr)), -dfr.dy.values]), dfr.y.values, rcond=None)[0]
misc["ief_empirical_duration_2y_to_freeze"] = float(beta[1])
OUT["misc"] = misc
print("  misc:", json.dumps(misc, default=float)[:1500], flush=True)

out_path = sys.argv[1] if len(sys.argv) > 1 else None
if out_path:
    json.dump(OUT, open(out_path, "w"), indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
print(f"done {time.time()-T0:.0f}s", flush=True)
