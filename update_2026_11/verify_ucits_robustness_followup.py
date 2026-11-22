"""
ROBUSTNESS REVIEW of update_2026_11/ucits.py  --  step 3: follow-ups to verify_ucits_robustness_checks.py.

  F1  one-session delay A on the 38-year US panel (1988..freeze, both halves) -- a far larger sample than 2011-26
  F2  UCITS-specific drag (B + C) by execution day, sleeve swaps at Tuesday execution, hybrid and 1x control by day
  F3  bootstrap CI of the total gap when executing on Tuesday
  F4  C3 check: daily lead-lag betas (same-day + previous US day) of the European lines
  F5  spikes that survived cleaning (2025-10-24 USD NAV in EUR/GBp lines): effect on the line statistics; same-product
      disagreement scan of the primary 3x line (QQQ3.L vs LQQ3.L / QQQ3.MI)
  F6  chop-flag flips that actually trade (risk-on both weeks) per year, 2011-26 and 1988-freeze
  F7  the specialist's DXY-timing evidence ('0.50 with DXY of D-1 vs 0.40 with D') under several definitions
Every new strategy-level backtest is counted.  Writes ONLY to the scratch dir given as argv[1].
"""
import os
import sys
import io
import json
import contextlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
scratch = sys.argv[1]
code = open(SRC, encoding="utf-8").read()
pat = 'open(os.path.join(HERE, "ucits.json"), "w")'
assert code.count(pat) == 1
code = code.replace(pat, f"open({os.path.join(scratch, 'ucits_rerun_followup.json')!r}, 'w')")
assert "to_parquet" not in code and "yf." not in code
g = {"__name__": "__main__", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, SRC, "exec"), g)
C = g["C"]; P2 = g["P2"]; SERIES = g["SERIES"]; run = g["run"]; CLEAN = g["CLEAN"]; PX = g["PX"]
END = g["END"]; FREEZE = g["FREEZE"]; COST_REAL = g["COST_REAL"]; FX = g["FX"]; WEND = g["WEND"]
N_VAR = 0
OUT = {}


def cg(r, a, b):
    return C.stats(r, a=a, b=b, start=None)["cagr"]


# ---------------------------------------------------------------- F1 delay cost on the long US panel
print("=== F1: one-session delay (US lag2 - US lag1) on the 1988..freeze US panel ===")
F1 = {}
for nm, lev, vt in [("F2", "qld", 0.30), ("F3", "tqqq", 0.20)]:
    w = C.strategy_weights(P2, lev=lev, voltgt=vt, off="ief", gold="gold_phys")
    r1, r2 = C.backtest(P2, w, lag=1), C.backtest(P2, w, lag=2)
    N_VAR += 2
    row = {}
    for lbl, a, b in [("1988-2006", "1988-01-01", "2006-12-31"), ("2007-freeze", "2007-01-01", FREEZE),
                      ("1988-freeze", "1988-01-01", FREEZE), ("2011-04-28..freeze", "2011-04-28", FREEZE)]:
        row[lbl] = cg(r2, a, b) - cg(r1, a, b)
    row["bootstrap_1988_freeze"] = C.block_bootstrap_cagr_diff(r2[:FREEZE], r1[:FREEZE])
    F1[nm] = row
    print(f"   {nm}: " + "  ".join(f"{k} {v*100:+.2f}" for k, v in row.items() if not isinstance(v, dict)) +
          f"  boot 1988-freeze [{row['bootstrap_1988_freeze']['lo']*100:+.2f},{row['bootstrap_1988_freeze']['hi']*100:+.2f}]")
OUT["F1_delay_long_panel"] = F1

# ---------------------------------------------------------------- F2 UCITS-specific drag by execution day
print("\n=== F2: UCITS-specific drag (B+C = UCITS lag-k realistic - US lag-k) by execution day ===")
F2 = {}
US = {"F2 primary": (dict(lev="qld", off="ief", gold="gold_phys"), dict(lev="u_lqq", off="u_cbu0", gold="u_igln"), 0.30),
      "F3 primary": (dict(lev="tqqq", off="ief", gold="gold_phys"), dict(lev="u_qqq3", off="u_cbu0", gold="u_igln"), 0.20)}
for nm, (us, uc, vt) in US.items():
    a, b = SERIES[nm]["a"], SERIES[nm]["b"]
    base = SERIES[nm]["R"]["US lag1 3bp"]
    rows = {}
    for lag in (2, 3, 4, 5):
        if lag == 2:
            rus = SERIES[nm]["R"]["US lag2 3bp"]; rucr = SERIES[nm]["R"]["UCITS lag2 realistic-cost"]
        else:
            rus, _ = run(f"fu {nm} US lag{lag}", us["lev"], vt, us["off"], us["gold"], lag=lag)
            rucr, _ = run(f"fu {nm} UCITS lag{lag} realistic", uc["lev"], vt, uc["off"], uc["gold"], lag=lag, costs=COST_REAL)
            N_VAR += 2
        rows[lag] = dict(B_plus_C=cg(rucr, a, b) - cg(rus, a, b), total_vs_US_lag1=cg(rucr, a, b) - cg(base, a, b))
        if lag == 3:
            rows[lag]["bootstrap_total_vs_US_lag1"] = C.block_bootstrap_cagr_diff(rucr[:b], base[:b], start=a)
            rows[lag]["bootstrap_B_plus_C"] = C.block_bootstrap_cagr_diff(rucr[:b], rus[:b], start=a)
            # sleeve swaps at Tuesday execution (3bp) vs US lag3
            sw = {}
            for lbl, args in [("lev only", (uc["lev"], us["off"], us["gold"])), ("bond only", (us["lev"], uc["off"], us["gold"])),
                              ("gold only", (us["lev"], us["off"], uc["gold"]))]:
                rr, _ = run(f"fu {nm} swap {lbl} lag3", args[0], vt, args[1], args[2], lag=3)
                N_VAR += 1
                sw[lbl] = cg(rr, a, b) - cg(rus, a, b)
            rows[lag]["sleeve_swaps_lag3_vs_US_lag3"] = sw
        print(f"   {nm} lag{lag}: B+C {rows[lag]['B_plus_C']*100:+.2f}  total vs US lag1 {rows[lag]['total_vs_US_lag1']*100:+.2f}"
              + (f"  boot total [{rows[lag]['bootstrap_total_vs_US_lag1']['lo']*100:+.2f},"
                 f"{rows[lag]['bootstrap_total_vs_US_lag1']['hi']*100:+.2f}]  swaps "
                 f"{ {k: round(v*100, 2) for k, v in rows[lag]['sleeve_swaps_lag3_vs_US_lag3'].items()} }" if lag == 3 else ""))
    bc = np.array([rows[k]["B_plus_C"] for k in rows])
    rows["mean_B_plus_C_over_exec_days"] = float(bc.mean())
    F2[nm] = rows
    print(f"   {nm}: mean B+C over Mon..Thu execution {bc.mean()*100:+.2f}%/yr  (specialist's Monday B+C "
          f"{rows[2]['B_plus_C']*100:+.2f})")
# hybrid by day
hyb = {}
for nm, (us, uc, vt) in US.items():
    a, b = SERIES[nm]["a"], SERIES[nm]["b"]
    base = SERIES[nm]["R"]["US lag1 3bp"]
    hyb[nm] = {}
    for lag in (2, 3, 4):
        rh, _ = run(f"fu {nm} hybrid lag{lag}", us["lev"], vt, "u_cbu0", "u_igln", lag=lag, costs={"u_cbu0": 8e-4, "u_igln": 8e-4})
        N_VAR += 1
        hyb[nm][lag] = cg(rh, a, b) - cg(base, a, b)
    print(f"   hybrid {nm} gap vs US lag1 by exec day: " + "  ".join(f"lag{k} {v*100:+.2f}" for k, v in hyb[nm].items()))
F2["hybrid_by_exec_day"] = hyb
# 1x control by day
a1 = P2["u_cndx"].first_valid_index() + pd.Timedelta(days=1)
a1 = P2.index[P2.index >= a1][0]
ctl = {}
for lag in (2, 3, 4, 5):
    rq, _ = run(f"fu 1x QQQ lag{lag}", "qqq", 0.30, "ief", "gold_phys", lag=lag)
    rc, _ = run(f"fu 1x CNDX lag{lag}", "u_cndx", 0.30, "ief", "gold_phys", lag=lag)
    N_VAR += 2
    ctl[lag] = dict(gap=cg(rc, a1, END) - cg(rq, a1, END),
                    gap_2020=C.period_return(rc, "2020-01-01", "2020-12-31") - C.period_return(rq, "2020-01-01", "2020-12-31"))
print("   1x control CNDX-QQQ by exec day: " + "  ".join(f"lag{k} {v['gap']*100:+.2f} (2020 {v['gap_2020']*100:+.1f}pp)"
                                                     for k, v in ctl.items()))
F2["control_1x_by_exec_day"] = ctl
OUT["F2_exec_day"] = F2

# ---------------------------------------------------------------- F4 lead-lag betas
print("\n=== F4: daily lead-lag betas of the European lines on the US analogue (same day, previous US day) ===")
F4 = {}
for tk, col in [("LQQ.PA", "qld_real"), ("QQQ3.L", "tqqq_real"), ("CNDX.L", "qqq_real"), ("CBU0.L", "ief_real"),
                ("IGLN.L", "gld_real")]:
    A = g["us_level"](col)
    cd = CLEAN[tk].index.intersection(A.index)
    cd = cd[cd <= END]
    u = CLEAN[tk].reindex(cd).pct_change(fill_method=None)
    x = A.reindex(cd).pct_change(fill_method=None)
    df = pd.concat([u, x, x.shift(1)], axis=1, keys=["u", "x0", "x1"]).dropna()
    X = np.column_stack([np.ones(len(df)), df.x0, df.x1])
    bta = np.linalg.lstsq(X, df.u.values, rcond=None)[0]
    F4[tk] = dict(beta_same_day=float(bta[1]), beta_prev_us_day=float(bta[2]), sum=float(bta[1] + bta[2]))
    print(f"   {tk:7s} beta same-day {bta[1]:.3f}  prev US day {bta[2]:.3f}  sum {bta[1]+bta[2]:.3f}")
OUT["F4_lead_lag"] = F4

# ---------------------------------------------------------------- F5 surviving spikes
print("\n=== F5: 2025-10-24 USD-NAV spike in non-primary lines (survived cleaning?) ===")
F5 = {}
pair = g["pair"]
overlap = g["overlap"]
for tk in ["QQQ3.MI", "LQQ3.L", "3QQQ.L"]:
    s = CLEAN[tk]
    t0 = pd.Timestamp("2025-10-24")
    if t0 in s.index:
        i = s.index.get_loc(t0)
        jump = float(s.iloc[i] / s.iloc[i - 1] - 1)
        back = float(s.iloc[i + 1] / s.iloc[i] - 1)
    else:
        jump = back = None
    s_fix = s.drop(t0) if t0 in s.index else s
    CLEAN_bak = CLEAN[tk]
    CLEAN[tk] = s_fix
    after = dict(pair_vs_QQQ3L=pair(tk, "QQQ3.L"), overlap_full=overlap(s_fix, "tqqq_real"))
    CLEAN[tk] = CLEAN_bak
    before = dict(pair_vs_QQQ3L=pair(tk, "QQQ3.L"), overlap_full=overlap(s, "tqqq_real"))
    F5[tk] = dict(in_cleaned_series=t0 in s.index, jump_on_2025_10_24=jump, move_next_session=back,
                  te_vs_QQQ3L_before=before["pair_vs_QQQ3L"]["te_ann"], te_vs_QQQ3L_after=after["pair_vs_QQQ3L"]["te_ann"],
                  te_vs_TQQQ_before=before["overlap_full"]["te_weekly_ann"], te_vs_TQQQ_after=after["overlap_full"]["te_weekly_ann"],
                  corr_w_vs_TQQQ_before=before["overlap_full"]["corr_weekly"], corr_w_vs_TQQQ_after=after["overlap_full"]["corr_weekly"],
                  cagr_gap_before=before["overlap_full"]["cagr_gap"], cagr_gap_after=after["overlap_full"]["cagr_gap"])
    v = F5[tk]
    print(f"   {tk}: in cleaned series {v['in_cleaned_series']}  jump {jump*100:+.1f}% then {back*100:+.1f}%  | weekly TE vs QQQ3.L "
          f"{v['te_vs_QQQ3L_before']*100:.2f}% -> {v['te_vs_QQQ3L_after']*100:.2f}%  | vs TQQQ TE {v['te_vs_TQQQ_before']*100:.2f}% -> "
          f"{v['te_vs_TQQQ_after']*100:.2f}%  corr {v['corr_w_vs_TQQQ_before']:.3f} -> {v['corr_w_vs_TQQQ_after']:.3f}  "
          f"CAGR gap {v['cagr_gap_before']*100:+.2f} -> {v['cagr_gap_after']*100:+.2f}")
# same-product disagreement scan: QQQ3.L vs LQQ3.L / QQQ3.MI daily (same close time, USD-converted)
scan = {}
for other in ["LQQ3.L", "QQQ3.MI"]:
    j = pd.concat([CLEAN["QQQ3.L"], CLEAN[other]], axis=1, keys=["p", "o"]).dropna()
    d = np.log(j.p / j.o).diff().dropna()
    big = d[d.abs() > 0.05]
    scan[other] = dict(days=int(len(d)), n_abs_gt_5pct=int(len(big)), sd_daily=float(d.std()),
                       dates=[str(x.date()) for x in big.index[:20]])
    print(f"   QQQ3.L vs {other}: {len(d)} common days, {len(big)} days with |daily log-ratio change| > 5%: "
          f"{[str(x.date()) for x in big.index[:12]]}")
F5["same_product_scan_QQQ3L"] = scan
OUT["F5_surviving_spikes"] = F5

# ---------------------------------------------------------------- F6 chop flips that trade
print("\n=== F6: chop-flag flips per year (all Fridays vs risk-on in both weeks) ===")
mk = P2["mark"].fillna(False)
F6 = {}
for lbl, a, b in [("1988-freeze", C.START, FREEZE), ("2011-04-28..2026-11-16", pd.Timestamp("2011-04-28"), END)]:
    m = mk & (P2.index >= a) & (P2.index <= b)
    cl = (P2["er20"] >= 0.25).fillna(False)[m]
    ron = P2["above"][m].astype(bool)
    flip = (cl != cl.shift()).iloc[1:]
    both_on = (ron & ron.shift(fill_value=False)).iloc[1:]
    yrs = (b - a).days / 365.25
    F6[lbl] = dict(flips_all_fridays_per_year=float(flip.sum() / yrs), flips_risk_on_both_weeks_per_year=float((flip & both_on).sum() / yrs),
                   risk_switches_per_year=float((ron != ron.shift()).iloc[1:].sum() / yrs))
    print(f"   {lbl}: chop flips/yr all Fridays {F6[lbl]['flips_all_fridays_per_year']:.1f}; while risk-on (trading) "
          f"{F6[lbl]['flips_risk_on_both_weeks_per_year']:.1f}; risk on/off switches/yr {F6[lbl]['risk_switches_per_year']:.1f}")
OUT["F6_chop_flips"] = F6

# ---------------------------------------------------------------- F7 DXY timing evidence
print("\n=== F7: specialist's DXY-timing evidence under several definitions (corr of FX bar return with DXY) ===")
F7 = {}
P = C.panel()
for fxn in ["EURUSD=X"]:
    raw = PX[PX.ticker == fxn].set_index("date").sort_index()["close"]
    for lbl, fx in [("cleaned", FX[fxn]), ("raw", raw[raw > 0])]:
        for per in [("2004+", "2004-01-01"), ("2010+", "2010-01-01")]:
            fr = np.log(fx).diff()
            dxr = np.log(P["dxy"]).diff()
            j = pd.concat([fr, dxr, dxr.shift(1)], axis=1, keys=["f", "d0", "d1"]).dropna()
            j = j[j.index >= per[1]]
            F7[f"{fxn} {lbl} {per[0]} daily pearson"] = dict(D_minus_1=float(j.f.corr(j.d1)), D=float(j.f.corr(j.d0)))
            F7[f"{fxn} {lbl} {per[0]} daily spearman"] = dict(D_minus_1=float(j.f.corr(j.d1, method="spearman")),
                                                             D=float(j.f.corr(j.d0, method="spearman")))
for k, v in F7.items():
    print(f"   {k:40s} D-1 {v['D_minus_1']:+.2f}  D {v['D']:+.2f}")
OUT["F7_dxy_timing"] = F7

OUT["n_new_variants_this_script"] = N_VAR
json.dump(OUT, open(os.path.join(scratch, "verify_ucits_robustness_followup.json"), "w"), indent=1, default=float)
print(f"\nnew strategy-level variants in this script: {N_VAR}")
