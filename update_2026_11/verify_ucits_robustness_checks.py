"""
ROBUSTNESS REVIEW of update_2026_11/ucits.py  --  step 2: stress the claims.

Runs ucits.py unchanged in-process (stdout muted, its JSON redirected to the scratch dir) so every check uses the
specialist's own cleaned data, FX conversion, execution calendar and backtest functions, then:
  R1  gap decomposition (A delay / B instruments+EU close / C costs) by half, by year, ex-2020, leave-one-year-out
  R2  start-date sensitivity of the total F2 gap (and the 2009 vs 2011 window difference: window or instruments?)
  R3  execution-day sensitivity: B and the total gap when the decision is executed Tue / Wed / Thu instead of Mon
  R4  block-length sensitivity of the bootstrap CIs (21 / 63 / 126 / 252 days)
  R5  which days make B: daily attribution, share of the top days, data at those days
  R6  robust instrument drift: smoothed-endpoint gaps + log-ratio trend with HAC s.e. (full / last 5y), rolling 5y
  R7  FX bar timing: EURUSD / GBPUSD implied by same-product USD vs EUR/GBp lines at the European close
  R8  withholding drag at historical (not today's) distribution yields
  R9  audit numbers (C16) with common.py
Every new strategy-level backtest is counted in N_VAR.  Writes ONLY to the scratch dir given as argv[1].
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
os.makedirs(scratch, exist_ok=True)

code = open(SRC, encoding="utf-8").read()
pat = 'open(os.path.join(HERE, "ucits.json"), "w")'
assert code.count(pat) == 1
code = code.replace(pat, f"open({os.path.join(scratch, 'ucits_rerun_checks.json')!r}, 'w')")
assert "to_parquet" not in code and "yf." not in code
g = {"__name__": "__main__", "__file__": SRC}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, SRC, "exec"), g)
C = g["C"]; P2 = g["P2"]; SERIES = g["SERIES"]; run = g["run"]; CLEAN = g["CLEAN"]; PX = g["PX"]
END = g["END"]; FREEZE = g["FREEZE"]; OOS_A = g["OOS_A"]; COST_REAL = g["COST_REAL"]; FX = g["FX"]
us_level = g["us_level"]; on_dates = g["on_dates"]; INFO = g["INFO"]
VARIANTS0 = len(g["VARIANTS"])
OUT = {}


def cg(r, a, b):
    return C.stats(r, a=a, b=b, start=None)["cagr"]


def cagr_days(r):
    r = r.dropna()
    return float(np.prod(1 + r.values) ** (252 / len(r)) - 1)


def comps(R, a, b, mask=None):
    """A, B, C, total (CAGR differences) on [a,b]; mask = boolean Series of days to KEEP (252-day annualisation)"""
    keys = ["US lag1 3bp", "US lag2 3bp", "UCITS lag2 3bp", "UCITS lag2 realistic-cost"]
    if mask is None:
        c = {k: cg(R[k], a, b) for k in keys}
    else:
        c = {}
        for k in keys:
            r = R[k][(R[k].index >= a) & (R[k].index <= b)]
            c[k] = cagr_days(r[mask.reindex(r.index).fillna(False).values])
    return dict(A=c["US lag2 3bp"] - c["US lag1 3bp"], B=c["UCITS lag2 3bp"] - c["US lag2 3bp"],
                C=c["UCITS lag2 realistic-cost"] - c["UCITS lag2 3bp"],
                total=c["UCITS lag2 realistic-cost"] - c["US lag1 3bp"], us_lag1=c["US lag1 3bp"],
                ucits_real=c["UCITS lag2 realistic-cost"])


def pp(d):
    return {k: round(v * 100, 2) for k, v in d.items()}


# =========================================================== R1 halves / years / ex-2020 / leave-one-year-out
print("\n=== R1: decomposition stability (CAGR differences, %/yr) ===")
R1 = {}
for nm in ["F2 primary", "F3 primary", "F2 longer history"]:
    R, a, b = SERIES[nm]["R"], SERIES[nm]["a"], SERIES[nm]["b"]
    mid = a + (FREEZE - a) / 2
    mid = P2.index[P2.index >= mid][0]
    blk = {"full": comps(R, a, b), "IS_to_freeze": comps(R, a, FREEZE),
           f"H1 {a.date()}..{(mid - pd.Timedelta(days=1)).date()}": comps(R, a, mid - pd.Timedelta(days=1)),
           f"H2 {mid.date()}..{FREEZE.date()}": comps(R, mid, FREEZE)}
    idx = R["US lag1 3bp"].index
    blk["full ex-2020"] = comps(R, a, b, mask=pd.Series(idx.year != 2020, index=idx))
    blk["full ex-COVID window 2020-02-15..2020-06-30"] = comps(
        R, a, b, mask=pd.Series(~((idx >= "2020-02-15") & (idx <= "2020-06-30")), index=idx))
    # leave-one-calendar-year-out range
    yrs = sorted(set(idx[(idx >= a) & (idx <= b)].year))
    loo = {y: comps(R, a, b, mask=pd.Series(idx.year != y, index=idx)) for y in yrs}
    for comp in ("A", "B", "total"):
        vals = {y: v[comp] for y, v in loo.items()}
        ymin, ymax = min(vals, key=vals.get), max(vals, key=vals.get)
        blk[f"LOYO {comp} range"] = dict(min=vals[ymin], drop_year_min=ymin, max=vals[ymax], drop_year_max=ymax)
    # yearly compounded differences
    yr = {}
    for y in yrs:
        lo_, hi_ = max(a, pd.Timestamp(f"{y}-01-01")), min(b, pd.Timestamp(f"{y}-12-31"))
        f = {k: C.period_return(R[k], lo_, hi_) for k in ["US lag1 3bp", "US lag2 3bp", "UCITS lag2 3bp",
                                                           "UCITS lag2 realistic-cost"]}
        yr[y] = dict(A=f["US lag2 3bp"] - f["US lag1 3bp"], B=f["UCITS lag2 3bp"] - f["US lag2 3bp"],
                     total=f["UCITS lag2 realistic-cost"] - f["US lag1 3bp"])
    Bs = np.array([v["B"] for v in yr.values()])
    As = np.array([v["A"] for v in yr.values()])
    blk["yearly_pp"] = {y: pp(v) for y, v in yr.items()}
    blk["yearly_B_summary"] = dict(n_years=len(Bs), n_negative=int((Bs < 0).sum()), median_pp=float(np.median(Bs) * 100),
                                   mean_pp=float(Bs.mean() * 100), worst_year=int(list(yr)[int(Bs.argmin())]),
                                   worst_pp=float(Bs.min() * 100),
                                   t_stat_mean=float(Bs.mean() / (Bs.std(ddof=1) / np.sqrt(len(Bs)))))
    blk["yearly_A_summary"] = dict(n_negative=int((As < 0).sum()), mean_pp=float(As.mean() * 100),
                                   median_pp=float(np.median(As) * 100))
    R1[nm] = blk
    print(f"\n-- {nm}  window {a.date()}..{b.date()}")
    for k, v in blk.items():
        if k == "yearly_pp":
            print("   yearly B pp: " + ", ".join(f"{y}:{vv['B']:+.1f}" for y, vv in v.items()))
            print("   yearly A pp: " + ", ".join(f"{y}:{vv['A']:+.1f}" for y, vv in v.items()))
            print("   yearly total pp: " + ", ".join(f"{y}:{vv['total']:+.1f}" for y, vv in v.items()))
        elif k.startswith("LOYO") or k.startswith("yearly"):
            print(f"   {k}: {v}")
        else:
            print(f"   {k:45s} A {v['A']*100:+.2f}  B {v['B']*100:+.2f}  C {v['C']*100:+.2f}  total {v['total']*100:+.2f}"
                  f"   (US {v['us_lag1']*100:.2f} / UCITS {v['ucits_real']*100:.2f})")
OUT["R1_decomposition_stability"] = R1

# =========================================================== R2 start-date sensitivity
print("\n=== R2: start-date sensitivity of the F2 total gap ===")
N_VAR = 0
R2 = {}
Rp = SERIES["F2 primary"]["R"]
Rl = SERIES["F2 longer history"]["R"]
for y in range(2009, 2022):
    a_ = pd.Timestamp(f"{y}-01-01")
    row = {}
    if a_ >= SERIES["F2 primary"]["a"] - pd.Timedelta(days=200):
        a2 = max(a_, SERIES["F2 primary"]["a"])
        row["CBU0/IGLN"] = comps(Rp, a2, END)
    a3 = max(a_, SERIES["F2 longer history"]["a"])
    row["IBTM/PHAU"] = comps(Rl, a3, END)
    R2[y] = row
    print(f"   start {y}: " + " | ".join(f"{k}: total {v['total']*100:+.2f} (A {v['A']*100:+.2f} B {v['B']*100:+.2f} "
                                         f"C {v['C']*100:+.2f})" for k, v in row.items()))
aP = SERIES["F2 primary"]["a"]
aL = SERIES["F2 longer history"]["a"]
R2["IBTM/PHAU on the CBU0/IGLN window 2011-04-28..END"] = comps(Rl, aP, END)
R2["IBTM/PHAU 2009-01-21..2011-04-27 only"] = comps(Rl, aL, aP - pd.Timedelta(days=1))
for k in ["IBTM/PHAU on the CBU0/IGLN window 2011-04-28..END", "IBTM/PHAU 2009-01-21..2011-04-27 only"]:
    v = R2[k]
    print(f"   {k}: A {v['A']*100:+.2f} B {v['B']*100:+.2f} C {v['C']*100:+.2f} total {v['total']*100:+.2f}")
OUT["R2_start_date"] = {str(k): v for k, v in R2.items()}

# =========================================================== R3 execution-day sensitivity
print("\n=== R3: execution day (lag k = execute at the close k-1 US sessions after the Friday signal) ===")
R3 = {}
for nm, us, uc, vt in [("F2 primary", dict(lev="qld", off="ief", gold="gold_phys"),
                        dict(lev="u_lqq", off="u_cbu0", gold="u_igln"), 0.30),
                       ("F3 primary", dict(lev="tqqq", off="ief", gold="gold_phys"),
                        dict(lev="u_qqq3", off="u_cbu0", gold="u_igln"), 0.20)]:
    a, b = SERIES[nm]["a"], SERIES[nm]["b"]
    base = SERIES[nm]["R"]["US lag1 3bp"]
    rows = {}
    for lag in (2, 3, 4, 5):
        if lag == 2:
            rus = SERIES[nm]["R"]["US lag2 3bp"]; ruc = SERIES[nm]["R"]["UCITS lag2 3bp"]
            rucr = SERIES[nm]["R"]["UCITS lag2 realistic-cost"]
        else:
            rus, _ = run(f"verify {nm}: US lag{lag} 3bp", us["lev"], vt, us["off"], us["gold"], lag=lag)
            ruc, _ = run(f"verify {nm}: UCITS lag{lag} 3bp", uc["lev"], vt, uc["off"], uc["gold"], lag=lag)
            rucr, _ = run(f"verify {nm}: UCITS lag{lag} realistic", uc["lev"], vt, uc["off"], uc["gold"], lag=lag,
                          costs=COST_REAL)
            N_VAR += 3
        bb = C.block_bootstrap_cagr_diff(ruc[:b], rus[:b], start=a)
        rows[lag] = dict(A=cg(rus, a, b) - cg(base, a, b), B=cg(ruc, a, b) - cg(rus, a, b),
                         total_realistic=cg(rucr, a, b) - cg(base, a, b), B_boot=bb,
                         B_H1=cg(ruc, a, "2018-12-31") - cg(rus, a, "2018-12-31"),
                         B_H2=cg(ruc, "2019-01-01", FREEZE) - cg(rus, "2019-01-01", FREEZE),
                         B_2020=C.period_return(ruc, "2020-01-01", "2020-12-31") - C.period_return(rus, "2020-01-01", "2020-12-31"))
        v = rows[lag]
        print(f"   {nm} lag{lag}: A {v['A']*100:+.2f}  B {v['B']*100:+.2f} [{bb['lo']*100:+.2f},{bb['hi']*100:+.2f}]  "
              f"(B 2011/13-2018 {v['B_H1']*100:+.2f}, 2019-freeze {v['B_H2']*100:+.2f}, 2020 {v['B_2020']*100:+.1f}pp)  "
              f"total realistic vs US lag1 {v['total_realistic']*100:+.2f}")
    R3[nm] = rows
OUT["R3_execution_day"] = R3

# =========================================================== R4 bootstrap block length
print("\n=== R4: bootstrap block-length sensitivity (95% CI, %/yr) ===")
R4 = {}
for nm in ["F2 primary", "F3 primary", "F2 longer history"]:
    R, a, b = SERIES[nm]["R"], SERIES[nm]["a"], SERIES[nm]["b"]
    R4[nm] = {}
    for blen in (21, 63, 126, 252):
        t = C.block_bootstrap_cagr_diff(R["UCITS lag2 realistic-cost"][:b], R["US lag1 3bp"][:b], start=a, block=blen)
        bB = C.block_bootstrap_cagr_diff(R["UCITS lag2 3bp"][:b], R["US lag2 3bp"][:b], start=a, block=blen)
        R4[nm][blen] = dict(total=t, B=bB)
        print(f"   {nm:18s} block {blen:3d}: total {t['mean']*100:+.2f} [{t['lo']*100:+.2f},{t['hi']*100:+.2f}]  "
              f"B {bB['mean']*100:+.2f} [{bB['lo']*100:+.2f},{bB['hi']*100:+.2f}]")
OUT["R4_bootstrap_blocks"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in R4.items()}

# =========================================================== R5 which days make B
print("\n=== R5: daily attribution of B (UCITS lag2 3bp - US lag2 3bp, arithmetic daily differences) ===")
R5 = {}
for nm in ["F2 primary", "F3 primary"]:
    R, a, b = SERIES[nm]["R"], SERIES[nm]["a"], SERIES[nm]["b"]
    d = (R["UCITS lag2 3bp"] - R["US lag2 3bp"])[a:b]
    tot = d.sum()
    top = d.abs().sort_values(ascending=False).index[:10]
    yrs = (b - a).days / 365.25
    covid = d["2020-02-15":"2020-06-30"].sum()
    R5[nm] = dict(sum_daily_diff_pp=float(tot * 100), per_year_pp=float(tot / yrs * 100),
                  covid_feb_jun_2020_pp=float(covid * 100),
                  per_year_ex_covid_pp=float((tot - covid) / yrs * 100),
                  top10_days={str(t.date()): float(d[t] * 100) for t in top},
                  top10_share_of_sum=float(d[top].sum() / tot))
    print(f"   {nm}: sum {tot*100:+.1f}pp = {tot/yrs*100:+.2f}pp/yr; Feb-Jun 2020 {covid*100:+.1f}pp; ex that "
          f"{(tot-covid)/yrs*100:+.2f}pp/yr; top-10 days sum {d[top].sum()*100:+.1f}pp")
    print("     top days: " + ", ".join(f"{t.date()} {d[t]*100:+.2f}" for t in top))
# data at the top F2 days
uc_cols = {"u_lqq": "qld_real", "u_cbu0": "ief_real", "u_igln": "gld_real"}
Rf = SERIES["F2 primary"]["R"]
dF2 = (Rf["UCITS lag2 3bp"] - Rf["US lag2 3bp"])[SERIES["F2 primary"]["a"]:END]
w_uc = g["held_weights_eu"](P2, C.strategy_weights(P2, lev="u_lqq", voltgt=0.30, off="u_cbu0", gold="u_igln"), 2,
                            g["eu_open_mask"](["LQQ.PA", "CBU0.L", "IGLN.L"]))
N_VAR += 0
rows = []
for t in dF2.abs().sort_values(ascending=False).index[:8]:
    i = P2.index.get_loc(t)
    rows.append(dict(date=str(t.date()), dB_pp=round(dF2[t] * 100, 2),
                     w_lev=round(float(w_uc.loc[t, "u_lqq"]), 3), w_prev=round(float(w_uc.iloc[i - 1]["u_lqq"]), 3),
                     LQQ_usd_ret=round(float(P2.loc[t, "u_lqq"]) * 100, 2), QLD_ret=round(float(P2.loc[t, "qld_real"]) * 100, 2),
                     CBU0_ret=round(float(P2.loc[t, "u_cbu0"]) * 100, 2), IEF_ret=round(float(P2.loc[t, "ief_real"]) * 100, 2),
                     IGLN_ret=round(float(P2.loc[t, "u_igln"]) * 100, 2), GLD_ret=round(float(P2.loc[t, "gld_real"]) * 100, 2),
                     weekday=t.day_name()))
    print("     ", rows[-1])
R5["F2 top-day detail"] = rows
OUT["R5_B_attribution"] = R5

# =========================================================== R6 robust instrument drift
print("\n=== R6: instrument drift robust to close-time endpoint noise (log-drift %/yr) ===")


def drift(su, a_col, start=None, end=g["WEND"], k=20, nw_lags=20):
    A = us_level(a_col)
    cd = su.index.intersection(A.index)
    cd = cd[cd <= end]
    if start is not None:
        cd = cd[cd >= pd.Timestamp(start)]
    L = np.log(su.reindex(cd)) - np.log(A.reindex(cd))
    L = L.dropna()
    t = np.array([(x - L.index[0]).days / 365.25 for x in L.index])
    yrs = t[-1]
    raw = (L.iloc[-1] - L.iloc[0]) / yrs
    sm = (L.iloc[-k:].mean() - L.iloc[:k].mean()) / (t[-k:].mean() - t[:k].mean())
    X = np.column_stack([np.ones(len(t)), t])
    bta = np.linalg.lstsq(X, L.values, rcond=None)[0]
    e = L.values - X @ bta
    # Newey-West s.e. of the slope
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, nw_lags + 1):
        wgt = 1 - lag / (nw_lags + 1)
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        S += wgt * (G + G.T)
    se = float(np.sqrt((XtXi @ S @ XtXi)[1, 1]))
    # endpoint noise: sd of daily L deviations from a 20-day centred mean
    noise = float((L - L.rolling(21, center=True).mean()).std())
    return dict(first=str(L.index[0].date()), last=str(L.index[-1].date()), years=float(yrs),
                raw_endpoint_logdrift=float(raw), smoothed_endpoint_logdrift=float(sm),
                ols_trend_logdrift=float(bta[1]), ols_trend_nw_se=se, daily_close_time_noise_sd=noise)


R6 = {}
FIVE_Y = g["WEND"] - pd.DateOffset(years=5)
for tk, col in [("LQQ.PA", "qld_real"), ("QQQ3.L", "tqqq_real"), ("LQQ3.L", "tqqq_real"), ("CNDX.L", "qqq_real"),
                ("CBU0.L", "ief_real"), ("DTLA.L", "tlt_real"), ("IB01.L", "bil_real"), ("IGLN.L", "gld_real"),
                ("IBTM.L", "ief_real"), ("PHAU.L", "gld_real")]:
    R6[tk] = {"full": drift(CLEAN[tk], col), "5y": drift(CLEAN[tk], col, start=FIVE_Y)}
    for lbl, v in R6[tk].items():
        print(f"   {tk:7s} {lbl:4s} {v['first']}: raw {v['raw_endpoint_logdrift']*100:+.2f}  smoothed "
              f"{v['smoothed_endpoint_logdrift']*100:+.2f}  OLS {v['ols_trend_logdrift']*100:+.2f} (NW se "
              f"{v['ols_trend_nw_se']*100:.2f})  daily close-time noise sd {v['daily_close_time_noise_sd']*100:.2f}%")
# rolling 5y endpoint gap for QQQ3.L vs TQQQ and LQQ.PA vs QLD (how much does the 5y number move with the end week?)
roll = {}
for tk, col in [("QQQ3.L", "tqqq_real"), ("LQQ.PA", "qld_real")]:
    vals = []
    for endw in pd.date_range("2025-11-14", g["WEND"], freq="W-FRI"):
        o = g["overlap"](CLEAN[tk], col, start=endw - pd.DateOffset(years=5), end=endw)
        vals.append(o["cagr_gap"])
    roll[tk] = dict(n_end_weeks=len(vals), min=float(min(vals)), max=float(max(vals)), median=float(np.median(vals)),
                    last=float(vals[-1]))
    print(f"   rolling 5y CAGR gap {tk} over the last 52 end-weeks: median {np.median(vals)*100:+.2f}  "
          f"range [{min(vals)*100:+.2f}, {max(vals)*100:+.2f}]")
R6["rolling_5y_gap_last_52_end_weeks"] = roll
OUT["R6_instrument_drift"] = R6

# =========================================================== R7 FX bar timing
print("\n=== R7: Yahoo FX bar timing implied by same-product lines at the European close ===")
R7 = {}


def raw_close(tk):
    s = PX[PX.ticker == tk].set_index("date").sort_index()["close"]
    return s[(s > 0) & (s.index <= END)].dropna()


for usd_tk, loc_tk, fxn, div in [("QQQ3.L", "QQQ3.MI", "EURUSD=X", 1.0), ("IGLN.L", "SGLN.L", "GBPUSD=X", 100.0),
                                 ("QQQ3.L", "LQQ3.L", "GBPUSD=X", 100.0)]:
    u = raw_close(usd_tk)
    l_ = raw_close(loc_tk) / div
    imp = np.log(u / l_).dropna()
    imp = imp[imp.index >= "2018-01-01"]
    fx = np.log(FX[fxn])
    pos = np.searchsorted(fx.index.values, imp.index.values, side="right")
    cur = on_dates(fx, imp.index)
    nxt = pd.Series(fx.values[np.minimum(pos, len(fx) - 1)], index=imp.index)
    df = pd.DataFrame({"imp": imp, "cur": cur, "nxt": nxt}).dropna()
    dd = df.diff().dropna()
    dev = (dd.imp - dd.imp.median()).abs()
    dd = dd[dev < 10 * 1.4826 * dev.median()]                      # drop unit-error jumps
    X = np.column_stack([dd.cur, dd.nxt])
    bta = np.linalg.lstsq(X, dd.imp.values, rcond=None)[0]
    wgrid = np.linspace(0, 1, 101)
    sse = [float(((dd.imp - ((1 - w) * dd.cur + w * dd.nxt)) ** 2).sum()) for w in wgrid]
    wbest = float(wgrid[int(np.argmin(sse))])
    R7[f"{usd_tk}/{loc_tk} ({fxn})"] = dict(n=int(len(dd)), coef_bar_D=float(bta[0]), coef_bar_D1=float(bta[1]),
                                            best_interp_weight_on_next_bar=wbest, specialist_weight=16.5 / 24,
                                            corr_with_bar_D=float(dd.imp.corr(dd.cur)),
                                            corr_with_bar_D1=float(dd.imp.corr(dd.nxt)))
    v = R7[f"{usd_tk}/{loc_tk} ({fxn})"]
    print(f"   {usd_tk}/{loc_tk}: n {v['n']}  coef bar D {v['coef_bar_D']:.3f}  bar D+1 {v['coef_bar_D1']:.3f}  "
          f"best weight on D+1 {wbest:.2f} (specialist {16.5/24:.3f})")
# DXY lead/lag (the specialist's own evidence)
dx = C.panel()["dxy_ret"]
for fxn in ["EURUSD=X", "GBPUSD=X"]:
    fr = np.log(FX[fxn]).diff()
    j = pd.concat([fr, dx, dx.shift(1), dx.shift(-1)], axis=1, keys=["fx", "dxy_D", "dxy_Dm1", "dxy_Dp1"]).dropna()
    j = j[j.index >= "2010-01-01"]
    R7[f"{fxn} corr with DXY"] = {k: float(j.fx.corr(j[k])) for k in ["dxy_Dm1", "dxy_D", "dxy_Dp1"]}
    print(f"   {fxn} daily log return corr with panel DXY: D-1 {R7[f'{fxn} corr with DXY']['dxy_Dm1']:+.2f}  "
          f"D {R7[f'{fxn} corr with DXY']['dxy_D']:+.2f}  D+1 {R7[f'{fxn} corr with DXY']['dxy_Dp1']:+.2f}")
OUT["R7_fx_timing"] = R7

# =========================================================== R8 withholding at historical yields
print("\n=== R8: NRA withholding drag at historical vs current distribution yields ===")
R8 = {}
for tk in ["IEF", "QLD", "TQQQ"]:
    gg = PX[PX.ticker == tk].set_index("date").sort_index()
    gg = gg[(gg.index >= "2011-05-01") & (gg.index <= END)]
    yy = {}
    for y, h in gg.groupby(gg.index.year):
        if len(h) > 200:
            yy[int(y)] = float(h["dividends"].sum() / h["close"].mean())
    R8[tk] = dict(yearly=yy, mean_2011_2025=float(np.mean([v for k, v in yy.items() if k <= 2025])))
f2w = json.load(open(os.path.join(scratch, "ucits_rerun_checks.json")))["strategy"][
    "F2 (2x, vt .30): LQQ.PA + CBU0.L + IGLN.L vs QLD + IEF + GLD"]["avg_weights_US"]
f3w = json.load(open(os.path.join(scratch, "ucits_rerun_checks.json")))["strategy"][
    "F3 (3x, vt .20): QQQ3.L + CBU0.L + IGLN.L vs TQQQ + IEF + GLD"]["avg_weights_US"]
R8["F2 WHT drag at 2011-2025 average yields"] = 0.30 * (f2w["qld"] * R8["QLD"]["mean_2011_2025"] + f2w["ief"] * R8["IEF"]["mean_2011_2025"])
R8["F3 WHT drag at 2011-2025 average yields"] = 0.30 * (f3w["tqqq"] * R8["TQQQ"]["mean_2011_2025"] + f3w["ief"] * R8["IEF"]["mean_2011_2025"])
print(f"   IEF yearly yields: { {k: round(v*100, 2) for k, v in R8['IEF']['yearly'].items()} }")
print(f"   mean 2011-2025: IEF {R8['IEF']['mean_2011_2025']*100:.2f}%  QLD {R8['QLD']['mean_2011_2025']*100:.2f}%  "
      f"TQQQ {R8['TQQQ']['mean_2011_2025']*100:.2f}%")
print(f"   WHT drag at historical-average yields: F2 {R8['F2 WHT drag at 2011-2025 average yields']*100:.2f}%/yr  "
      f"F3 {R8['F3 WHT drag at 2011-2025 average yields']*100:.2f}%/yr  (specialist, today's TTM yields: 0.53 / 0.63)")
OUT["R8_withholding"] = R8

# =========================================================== R9 audit numbers
P = C.panel()
r2 = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30, off="ief", gold="gold_phys"))
r3 = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys"))
N_VAR += 2
s2, s3 = C.stats(r2, b=FREEZE), C.stats(r3, b=FREEZE)
OUT["R9_audit"] = dict(F2=s2, F3=s3)
print(f"\n=== R9: US F2 1988-freeze {C.fmt(s2)} | F3 {C.fmt(s3)}")

OUT["n_new_variants_this_script"] = N_VAR
OUT["specialist_variants"] = VARIANTS0
json.dump(OUT, open(os.path.join(scratch, "verify_ucits_robustness_checks.json"), "w"), indent=1, default=float)
print(f"\nnew strategy-level variants in this script: {N_VAR}; saved to scratch")
