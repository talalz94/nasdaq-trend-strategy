"""
ROUND 2 - dimension 'Sensitivity to model-input corrections'   (KEY = model_hygiene)
Writes update_2026_11/r2_model_hygiene.json.  Read-only use of panel.parquet / common.py.

PRE-REGISTERED DESIGN.  Fixed before ANY strategy result was computed.  The only things inspected
beforehand were calibration diagnostics (real-vs-sim gaps of QLD/TQQQ, real-vs-synthetic fits of
IEF/TLT, the shape of the pre-2000 gold series); none of them contains a strategy return.
Every decision uses data to 2026-07-24 (FREEZE).  2026-07-25..2026-11-17 is out-of-sample (OOS).

(a) FINANCING SPREAD / EXPENSE GAP of the pre-inception leveraged sims
    gap_lit = real - sim, sim = the panel sim = lev*NDX PRICE return - 0.95%/yr - (lev-1)*rf
    gap_dn  = real - [lev*(QQQ_real + 0.20%/yr) - 0.95%/yr - (lev-1)*rf]   ("dividend-neutral":
              QQQ's total return replaces the NDX price index, so the NDX dividend yield drops out)
    Daily OLS  gap = alpha + beta*rf_d  with Newey-West(20) SEs, on QLD 2006-06..FREEZE and
    TQQQ 2010-02..FREEZE.  Regimes 2006-09 (QLD only), 2010-15, 2016-21, 2022-FREEZE.  The pooled
    fit is 'stable' if every regime's mean residual is within 2 HAC SE of zero.
    Applied ONLY on days without a real ETF (QLD before 2006-06-22, TQQQ before 2010-02-12):
      A_div    (primary)     sim + alpha_dn + beta_dn*rf_d   (sims stay price-only, so the NDX
                             dividend omission is kept: a conservative financing correction)
      A_lit    (literal)     sim + alpha_lit + beta_lit*rf_d (the regression the brief suggests; it
                             also credits the 2006/2010-2026 average dividend yield)
      A_stress (bound)       sim - (lev-1)*w, w = worst regime mean gap_dn per unit borrowed
                             (max over funds and regimes), a constant.  A stress, not a calibration.
    F1 holds unlevered QQQ, so (a) cannot change F1.
(b) DURATION of the pre-2002-07-31 synthetic IEF / TLT
    Analytic modified duration D(y) and convexity C(y) of a par bond with semiannual coupons:
    IEF = 7.5y at the 10y yield (tnx), TLT = 20y at the 30y yield (tyx), at y(t-1):
        r = a + y(t-1)/252 - D(y(t-1))*dy + 0.5*C(y(t-1))*dy^2,  only 'a' re-fit (OLS) on
        2002-07-31..FREEZE (one fitted constant instead of the fixed calibration's three).
    FIT GATE vs the fixed calibration, 2002-07-31..FREEZE, for IEF and TLT:
        TE <= 1.10 x fixed TE,  corr >= fixed corr - 0.01,  |CAGR gap| <= |fixed gap| + 0.25pp.
    B   = pure analytic (primary; used in the combination if it passes the gate)
    B_k = level-matched analytic (D = k*D_par, C = k^2*C_par, k by OLS): sensitivity; used in the
          combination only if B fails and B_k passes.
(c) PRE-2000-08-30 GOLD (monthly-average series, log-interpolated)
    C1 month-end step: each month's interpolated gross return booked on its last trading day
       (2000-08-30 for the partial August 2000); the 0.40%/yr fee kept daily.  Same cumulative return.
    C2 gold miners (^XAU, the July proxy) on the same days.
COMBINED:  ALL1 = A_div + B(gate) + C1 (primary combination);  ALL2 = A_div + B(gate) + C2.

STRATEGIES (July rules; IEF off-leg; 15% gold; chop .25/.5; Aug-Sep x0.5; 3 bp; lag 1):
    F1 = QQQ vt .20,  F2 = QLD vt .30,  F3 = TQQQ vt .20.
    Bases: Friday book (P.mark) and the 5-weekday tranche (mean of the Mon..Fri return streams,
    exactly as main_tranche_check.py).
    Configs per tier: default; off=bills; off=tlt; gold slot held in IEF (equal 0.85 core);
    no chop rule (chop_pos 1.0); no vol target (voltgt 99).  Benchmark QQQ buy-and-hold (P.qqq,
    untouched by every correction).

CONCLUSION RULES.  A round-1 conclusion CHANGES under a correction iff its status on the corrected
panel differs from its status on the uncorrected panel (same code, same basis).  Window IS =
1988-01-01..FREEZE; halves H1 1988-2006, H2 2007-FREEZE.  Bootstrap = paired circular blocks of
63 days, 2000 draws, seed 0 (round-1 scheme), on IS.
    R1 2x vs 3x  : (i) excess-Sharpe tie, P(F2 > F3) inside (0.05, 0.95); (ii) F3 - F2 CAGR 95% CI
                   above 0; (iii) F2 maxDD shallower than F3.
    R2 IEF vs bills : sign of Calmar(IEF) - Calmar(bills) per tier, IS, H1, H2.
    R3 gold vs IEF  : no significant gold-specific effect: P(Calmar(gold) - Calmar(IEF in the gold
                      slot) > 0) inside (0.05, 0.95), per tier (CAGR reported alongside).
    R4 chop / vol   : sign of the IS Calmar contribution (default - ablated) and the 'helps in both
                      halves' flag, per tier, for the chop rule and for the vol target.
    R5 edge vs QQQ  : sign of the point Calmar edge and CAGR edge, and whether each 95% CI excludes 0.
    R6 IEF vs TLT   : sign of Calmar(IEF) - Calmar(TLT) per tier, IS, H1, H2 (the scanner fix).
    Flips under A_div, B, C1, C2, ALL1 or ALL2 are conclusion changes.  Flips that appear only under
    the sensitivities A_lit, A_stress or B_k are reported as sensitivity flips.

Run:  cd <project root> && python -u update_2026_11/r2_model_hygiene.py
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
NBOOT = int(os.environ.get("R2_NBOOT", "2000"))
OUT_JSON = os.path.join(HERE, "r2_model_hygiene.json")


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
LATEST = IDX[-1]
OOS_A = pd.Timestamp("2026-07-25")
RF = P["rf_d"].fillna(0).values
COUNT = dict(strategy_variants=0, benchmark_series=0, calibration_fits=0, bootstrap_runs=0)
RES = dict(meta=dict(script="update_2026_11/r2_model_hygiene.py", freeze=str(FREEZE.date()),
                     latest=str(LATEST.date()), is_window=f"{START.date()}..{FREEZE.date()}",
                     oos_window=f"{OOS_A.date()}..{LATEST.date()}", nboot=NBOOT))


# =========================================================================== helpers
def nw_ols(y, X, L=20):
    """OLS with Newey-West (Bartlett, L lags) standard errors."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, np.sqrt(np.diag(V)), e


def cagr_of(r):
    r = np.asarray(r, float)
    return float(np.prod(1 + r) ** (252 / len(r)) - 1)


def fit_stats(real, syn, a, b):
    m = pd.concat([real, syn], axis=1)
    m = m[(m.index >= pd.Timestamp(a)) & (m.index <= pd.Timestamp(b))].dropna()
    m.columns = ["r", "s"]
    return dict(days=int(len(m)), te_ann=float((m.r - m.s).std() * np.sqrt(252)), corr=float(m.r.corr(m.s)),
                cagr_real=cagr_of(m.r), cagr_syn=cagr_of(m.s),
                cagr_gap=cagr_of(m.s) - cagr_of(m.r),
                vol_real=float(m.r.std() * np.sqrt(252)), vol_syn=float(m.s.std() * np.sqrt(252)))


# =========================================================================== (a) financing
log("(a) financing-spread calibration")
REGIMES = {"2006-09": ("2006-01-01", "2009-12-31"), "2010-15": ("2010-01-01", "2015-12-31"),
           "2016-21": ("2016-01-01", "2021-12-31"), "2022-frz": ("2022-01-01", str(FREEZE.date()))}
FIN = {}
qgross = P["qqq_real"] + 0.0020 / 252
for fund, lev in (("qld", 2), ("tqqq", 3)):
    real = P[f"{fund}_real"]
    sims = {"literal": P[f"{fund}_sim"],
            "divneutral": lev * qgross - 0.0095 / 252 - (lev - 1) * P["rf_d"]}
    FIN[fund] = dict(lev=lev, first_real_day=str(real.first_valid_index().date()))
    for spec, sim in sims.items():
        d = pd.concat([real - sim, P["rf_d"]], axis=1).dropna()
        d.columns = ["g", "rf"]
        dis = d[d.index <= FREEZE]
        b, se, e = nw_ols(dis.g.values, np.column_stack([np.ones(len(dis)), dis.rf.values]))
        COUNT["calibration_fits"] += 1
        pooled = dict(n=int(len(dis)), alpha_ann=b[0] * 252, alpha_se_ann=se[0] * 252, beta=b[1], beta_se=se[1],
                      mean_gap_ann=float(dis.g.mean() * 252), mean_rf=float(dis.rf.mean() * 252),
                      alpha_daily=float(b[0]))
        resid = pd.Series(e, index=dis.index)
        regs = {}
        for rk, (a0, b0) in REGIMES.items():
            m = (dis.index >= pd.Timestamp(a0)) & (dis.index <= pd.Timestamp(b0))
            if m.sum() < 100:
                continue
            dd = dis[m]
            br, ser, _ = nw_ols(dd.g.values, np.column_stack([np.ones(len(dd)), dd.rf.values]))
            mr, smr, _ = nw_ols(resid[m].values, np.ones((int(m.sum()), 1)))
            mg, smg, _ = nw_ols(dd.g.values, np.ones((len(dd), 1)))
            COUNT["calibration_fits"] += 1
            regs[rk] = dict(n=int(m.sum()), mean_rf=float(dd.rf.mean() * 252),
                            mean_gap_ann=float(mg[0] * 252), mean_gap_se_ann=float(smg[0] * 252),
                            alpha_ann=float(br[0] * 252), alpha_se_ann=float(ser[0] * 252),
                            beta=float(br[1]), beta_se=float(ser[1]),
                            pooled_pred_ann=float((b[0] + b[1] * dd.rf.mean()) * 252),
                            pooled_resid_mean_ann=float(mr[0] * 252), pooled_resid_se_ann=float(smr[0] * 252),
                            within_2se=bool(abs(mr[0]) <= 2 * smr[0]))
        do = d[d.index >= OOS_A]
        oos = dict(n=int(len(do)), mean_gap_ann=float(do.g.mean() * 252) if len(do) else None,
                   pooled_pred_ann=float((b[0] + b[1] * do.rf.mean()) * 252) if len(do) else None)
        FIN[fund][spec] = dict(pooled=pooled, regimes=regs, stable=bool(all(r["within_2se"] for r in regs.values())),
                               oos_reported_only=oos)
        log(f"   {fund:4s} {spec:10s} pooled alpha {pooled['alpha_ann']*100:+.2f}%/yr (se {pooled['alpha_se_ann']*100:.2f}) "
            f"beta {pooled['beta']:+.3f} (se {pooled['beta_se']:.3f}) | mean gap {pooled['mean_gap_ann']*100:+.2f}%/yr "
            f"| stable={FIN[fund][spec]['stable']} | " +
            " ".join(f"{k}:{v['mean_gap_ann']*100:+.2f}(rf {v['mean_rf']*100:.1f})" for k, v in regs.items()))

# worst regime gap per unit borrowed (dividend-neutral) -> stress bound
per_unit = {f"{f}|{rk}": -FIN[f]["divneutral"]["regimes"][rk]["mean_gap_ann"] / (FIN[f]["lev"] - 1)
            for f in ("qld", "tqqq") for rk in FIN[f]["divneutral"]["regimes"]}
w_key = max(per_unit, key=per_unit.get)
W_STRESS = per_unit[w_key]
FIN["stress"] = dict(per_unit_borrowed_by_regime=per_unit, worst=w_key, worst_per_unit_ann=W_STRESS)
log(f"   stress: worst per-unit-borrowed gap {W_STRESS*100:.2f}%/yr ({w_key})")


def lev_sleeve(fund, adj):
    """real ETF where it exists, else panel sim + adj (adj = daily Series)."""
    return P[f"{fund}_real"].fillna(P[f"{fund}_sim"] + adj).fillna(0).values


ZERO = pd.Series(0.0, index=IDX)
for f in ("qld", "tqqq"):
    assert np.array_equal(lev_sleeve(f, ZERO), P[f].fillna(0).values), f"splice mismatch {f}"


def adj_fit(fund, spec):
    p = FIN[fund][spec]["pooled"]
    return p["alpha_daily"] + p["beta"] * P["rf_d"].fillna(0)


SLEEVES_A = {
    "A_div": {f: lev_sleeve(f, adj_fit(f, "divneutral")) for f in ("qld", "tqqq")},
    "A_lit": {f: lev_sleeve(f, adj_fit(f, "literal")) for f in ("qld", "tqqq")},
    "A_stress": {f: lev_sleeve(f, ZERO - (FIN[f]["lev"] - 1) * W_STRESS / 252) for f in ("qld", "tqqq")},
}
# implied annual drag on the pre-inception sims, at the pre-inception mean rf
FIN["applied"] = {}
for f in ("qld", "tqqq"):
    pre = P[f"{f}_real"].isna() & (IDX >= START)
    rf_pre = float(P["rf_d"][pre].mean() * 252)
    FIN["applied"][f] = dict(pre_inception_days_from_1988=int(pre.sum()), mean_rf_pre=rf_pre,
                             A_div_ann_at_mean_rf=FIN[f]["divneutral"]["pooled"]["alpha_ann"] + FIN[f]["divneutral"]["pooled"]["beta"] * rf_pre,
                             A_lit_ann_at_mean_rf=FIN[f]["literal"]["pooled"]["alpha_ann"] + FIN[f]["literal"]["pooled"]["beta"] * rf_pre,
                             A_stress_ann=-(FIN[f]["lev"] - 1) * W_STRESS,
                             sleeve_cagr_pre_P0=cagr_of(P[f].fillna(0).values[pre.values]),
                             sleeve_cagr_pre_A_div=cagr_of(SLEEVES_A["A_div"][f][pre.values]),
                             sleeve_cagr_pre_A_lit=cagr_of(SLEEVES_A["A_lit"][f][pre.values]),
                             sleeve_cagr_pre_A_stress=cagr_of(SLEEVES_A["A_stress"][f][pre.values]))
    log(f"   applied {f}: pre-inception (1988+) mean rf {rf_pre*100:.2f}% -> A_div "
        f"{FIN['applied'][f]['A_div_ann_at_mean_rf']*100:+.2f}%/yr, A_lit {FIN['applied'][f]['A_lit_ann_at_mean_rf']*100:+.2f}%/yr, "
        f"A_stress {FIN['applied'][f]['A_stress_ann']*100:+.2f}%/yr")
RES["financing"] = FIN


# =========================================================================== (b) duration
log("(b) analytic par-bond duration")


def par_dc(y, T):
    """modified duration (years) and convexity (years^2) of a par bond, semiannual coupons, yield y."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    yy = np.where(ok, y, 0.05)
    n = int(round(2 * T))
    k = np.arange(1, n + 1)[None, :]
    i = (yy / 2)[:, None]
    cf = np.where(k == n, i + 1.0, i)
    disc = (1 + i) ** (-k)
    price = (cf * disc).sum(1)
    dmod = (cf * disc * (k / 2.0)).sum(1) / price / (1 + i[:, 0])
    cvx = (cf * disc * k * (k + 1) / 4.0 / (1 + i) ** 2).sum(1) / price
    return np.where(ok, dmod, np.nan), np.where(ok, cvx, np.nan)


CAL_A, CAL_B = pd.Timestamp("2002-07-31"), FREEZE
DUR = {}
SYN = {}
for name, ycol, T in (("ief", "tnx", 7.5), ("tlt", "tyx", 20.0)):
    yd = P[ycol] / 100.0
    dy = yd.diff()
    y1 = yd.shift(1)
    carry = y1 / 252
    D, Cx = par_dc(y1.values, T)
    D = pd.Series(D, index=IDX)
    Cx = pd.Series(Cx, index=IDX)
    real = P[f"{name}_real"]
    fixed = P[f"{name}_syn"]
    mcal = (IDX >= CAL_A) & (IDX <= CAL_B)
    core = (real - carry + D * dy - 0.5 * Cx * dy ** 2)[mcal].dropna()
    a_B = float(core.mean())
    synB = a_B + carry - D * dy + 0.5 * Cx * dy ** 2
    COUNT["calibration_fits"] += 1
    # level-matched: k by OLS on -D*dy, then convexity k^2, then intercept re-fit
    df = pd.concat([real - carry - 0.5 * Cx * dy ** 2, -D * dy], axis=1)[mcal].dropna()
    bk = np.linalg.lstsq(np.column_stack([np.ones(len(df)), df.iloc[:, 1].values]), df.iloc[:, 0].values, rcond=None)[0]
    kk = float(bk[1])
    a_k = float((real - carry + kk * D * dy - 0.5 * kk ** 2 * Cx * dy ** 2)[mcal].dropna().mean())
    synK = a_k + carry - kk * D * dy + 0.5 * kk ** 2 * Cx * dy ** 2
    COUNT["calibration_fits"] += 1
    fits = {}
    for lab, (a0, b0) in {"2002-frz": (CAL_A, CAL_B), "2002-08": (CAL_A, "2008-12-31"),
                           "2009-21": ("2009-01-01", "2021-12-31"), "2022-frz": ("2022-01-01", CAL_B),
                           "OOS(reported only)": (OOS_A, LATEST)}.items():
        fits[lab] = {"fixed": fit_stats(real, fixed, a0, b0), "analytic_B": fit_stats(real, synB, a0, b0),
                     "level_matched_Bk": fit_stats(real, synK, a0, b0)}
    fx, an, lk = fits["2002-frz"]["fixed"], fits["2002-frz"]["analytic_B"], fits["2002-frz"]["level_matched_Bk"]

    def gate(x):
        return dict(te_ok=bool(x["te_ann"] <= 1.10 * fx["te_ann"]), corr_ok=bool(x["corr"] >= fx["corr"] - 0.01),
                    gap_ok=bool(abs(x["cagr_gap"]) <= abs(fx["cagr_gap"]) + 0.0025))
    gB, gK = gate(an), gate(lk)
    gB["pass"] = all(gB.values())
    gK["pass"] = all(gK.values())
    pre = (IDX >= START) & (IDX < CAL_A)
    DUR[name] = dict(maturity_years=T, yield_col=ycol, alpha_B_ann=a_B * 252, k=kk, alpha_Bk_ann=a_k * 252,
                     fits=fits, gate_B=gB, gate_Bk=gK,
                     pre2002=dict(mean_yield=float(yd[pre].mean()), mean_D_analytic=float(D[pre].mean()),
                                  min_D_analytic=float(D[pre].min()), max_D_analytic=float(D[pre].max()),
                                  mean_C_analytic=float(Cx[pre].mean()),
                                  cagr_fixed=cagr_of(fixed[pre].fillna(0)), cagr_B=cagr_of(synB[pre].fillna(0)),
                                  cagr_Bk=cagr_of(synK[pre].fillna(0)),
                                  vol_fixed=float(fixed[pre].std() * np.sqrt(252)),
                                  vol_B=float(synB[pre].std() * np.sqrt(252)), vol_Bk=float(synK[pre].std() * np.sqrt(252))),
                     analytic_D_today=float(D.iloc[-1]), analytic_C_today=float(Cx.iloc[-1]))
    SYN[name] = dict(B=real.fillna(synB).fillna(0).values, B_k=real.fillna(synK).fillna(0).values)
    assert np.array_equal(real.fillna(fixed).fillna(0).values, P[name].fillna(0).values), f"bond splice mismatch {name}"
    log(f"   {name}: TE fixed {fx['te_ann']*100:.2f} / B {an['te_ann']*100:.2f} / Bk {lk['te_ann']*100:.2f}; corr "
        f"{fx['corr']:.4f}/{an['corr']:.4f}/{lk['corr']:.4f}; CAGR gap {fx['cagr_gap']*100:+.2f}/{an['cagr_gap']*100:+.2f}/"
        f"{lk['cagr_gap']*100:+.2f}pp; gate B {gB['pass']} Bk {gK['pass']} (k {kk:.3f}); pre-2002 mean D "
        f"{DUR[name]['pre2002']['mean_D_analytic']:.2f}, syn CAGR fixed {DUR[name]['pre2002']['cagr_fixed']*100:.2f}% "
        f"B {DUR[name]['pre2002']['cagr_B']*100:.2f}% Bk {DUR[name]['pre2002']['cagr_Bk']*100:.2f}%")
B_PASS = DUR["ief"]["gate_B"]["pass"] and DUR["tlt"]["gate_B"]["pass"]
BK_PASS = DUR["ief"]["gate_Bk"]["pass"] and DUR["tlt"]["gate_Bk"]["pass"]
B_COMB = "B" if B_PASS else ("B_k" if BK_PASS else "B")
DUR["gate_summary"] = dict(B_pass=B_PASS, Bk_pass=BK_PASS, used_in_combination=B_COMB,
                           note="" if (B_PASS or BK_PASS) else "neither passed; B used, flagged")
RES["duration"] = DUR
log(f"   gate: B pass={B_PASS}, B_k pass={BK_PASS} -> combination uses {B_COMB}")

# =========================================================================== (c) gold
log("(c) pre-2000 gold")
interp = P["gold_src"].eq("monthly-interp").values
gp = P["gold_phys"].fillna(0).values
g_gross = np.where(interp, gp + 0.0040 / 252, 0.0)
g_gross[0] = 0.0                                     # panel day 1 has no return
sp = IDX[interp]
last_in_month = pd.Series(sp, index=sp).groupby(sp.to_period("M")).max()
book = np.zeros(N, dtype=bool)
book[IDX.get_indexer(pd.DatetimeIndex(last_in_month.values))] = True
lg = np.log1p(g_gross)
step = np.zeros(N)
acc = 0.0
for t in np.where(interp)[0]:
    acc += lg[t]
    if book[t]:
        step[t] = np.expm1(acc)
        acc = 0.0
assert abs(acc) < 1e-15
gold_c1 = np.where(interp, step - 0.0040 / 252, gp)
gold_c1[0] = gp[0]
gold_c2 = np.where(interp, P["gold_miners"].fillna(0).values, gp)
m88 = interp & (IDX >= START)
GOLD = dict(interp_span=f"{sp.min().date()}..{sp.max().date()}", booking_days=int(book.sum()),
            check_total_interp=float(np.expm1(lg[interp].sum())), check_total_step=float(np.prod(1 + step[interp]) - 1))
for nm, s in (("interp (P0)", gp), ("C1 month-end step", gold_c1), ("C2 miners", gold_c2)):
    x = pd.Series(s[m88], index=IDX[m88])
    GOLD[nm] = dict(cagr_1988_2000=cagr_of(x.values), daily_vol=float(x.std() * np.sqrt(252)),
                    monthly_vol=float(((1 + x).resample("ME").prod() - 1).std() * np.sqrt(12)),
                    corr_ndx_daily=float(x.corr(P["ndx_ret"][x.index])),
                    corr_ndx_monthly=float(((1 + x).resample("ME").prod() - 1).corr((1 + P["ndx_ret"][x.index]).resample("ME").prod() - 1)))
COUNT["calibration_fits"] += 2
RES["gold"] = GOLD
log(f"   gold 1988..2000-08: CAGR interp {GOLD['interp (P0)']['cagr_1988_2000']*100:.2f}% (dvol {GOLD['interp (P0)']['daily_vol']*100:.1f}%), "
    f"step {GOLD['C1 month-end step']['cagr_1988_2000']*100:.2f}% (dvol {GOLD['C1 month-end step']['daily_vol']*100:.1f}%), "
    f"miners {GOLD['C2 miners']['cagr_1988_2000']*100:.2f}% (dvol {GOLD['C2 miners']['daily_vol']*100:.1f}%)")

# =========================================================================== panels
BASE_R = {c: P[c].fillna(0).values for c in ["qqq", "qld", "tqqq", "ief", "tlt", "bills", "gold_phys"]}


def panel_with(**over):
    R = dict(BASE_R)
    R.update(over)
    return R


Bsel = SYN["ief"][B_COMB], SYN["tlt"][B_COMB]
PANELS = {
    "P0": BASE_R,
    "A_div": panel_with(**SLEEVES_A["A_div"]),
    "A_lit": panel_with(**SLEEVES_A["A_lit"]),
    "A_stress": panel_with(**SLEEVES_A["A_stress"]),
    "B": panel_with(ief=SYN["ief"]["B"], tlt=SYN["tlt"]["B"]),
    "B_k": panel_with(ief=SYN["ief"]["B_k"], tlt=SYN["tlt"]["B_k"]),
    "C1": panel_with(gold_phys=gold_c1),
    "C2": panel_with(gold_phys=gold_c2),
    "ALL1": panel_with(**SLEEVES_A["A_div"], ief=Bsel[0], tlt=Bsel[1], gold_phys=gold_c1),
    "ALL2": panel_with(**SLEEVES_A["A_div"], ief=Bsel[0], tlt=Bsel[1], gold_phys=gold_c2),
}
PRIMARY = ["A_div", "B", "C1", "C2", "ALL1", "ALL2"]
SENS = ["A_lit", "A_stress", "B_k"]
PANEL_DESC = {
    "P0": "uncorrected panel (round-1 basis)",
    "A_div": "(a) financing+expense gap, dividend-neutral pooled fit, on pre-inception QLD/TQQQ sims",
    "A_lit": "(a) sensitivity: literal real-minus-sim pooled fit (credits 2006/10-26 dividend yield)",
    "A_stress": "(a) stress: worst regime gap per unit borrowed, constant",
    "B": "(b) analytic par-bond duration/convexity (IEF 7.5y, TLT 20y) pre-2002-07-31",
    "B_k": "(b) sensitivity: level-matched analytic duration (k*D_par)",
    "C1": "(c-i) pre-2000-08-30 gold as month-end step returns",
    "C2": "(c-ii) pre-2000-08-30 gold replaced by gold miners (^XAU)",
    "ALL1": f"combined A_div + {B_COMB} + C1 (primary)",
    "ALL2": f"combined A_div + {B_COMB} + C2",
}
RES["panels"] = PANEL_DESC

# =========================================================================== backtest engine (== common.backtest)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MARKS = {d: (IDX.dayofweek == i) for i, d in enumerate(DAYS)}
assert np.array_equal(MARKS["Fri"], P["mark"].values.astype(bool))
MARKS["Fri"] = P["mark"].values.astype(bool)
LAST = {}
for d in DAYS:
    pos = np.where(MARKS[d], np.arange(N), -1)
    LAST[d] = np.maximum.accumulate(pos)
VOL = P["vol20"].values
ERV = P["er20"].values
ABOVE = P["above"].values.astype(bool)
MONTH = P["month"].values


def weights(lev, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, seasonal=True, off="ief", gold_to="gold_phys"):
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.clip(voltgt / VOL, 0, 1)
    sc = np.where(np.isnan(sc), 1.0, sc)
    if seasonal:
        sc = np.clip(sc * np.where(np.isin(MONTH, (8, 9)), 0.5, 1.0), 0, 1)
    with np.errstate(invalid="ignore"):
        clean = np.where(np.isnan(ERV), False, ERV >= chop_thr)
    eq = (ABOVE & clean) * sc + (ABOVE & ~clean) * sc * chop_pos
    core = 1.0 - gold_w
    W = {lev: eq * core}
    rest = np.clip(core - eq * core, 0, None)
    W[off] = W.get(off, 0.0) + rest
    if gold_w > 0:
        W[gold_to] = W.get(gold_to, 0.0) + gold_w
    return {c: np.broadcast_to(np.asarray(v, float), (N,)).copy() for c, v in W.items()}


def held(W, day, lag=1):
    li = LAST[day]
    H = {}
    for c, w in W.items():
        h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
        H[c] = np.concatenate([np.zeros(lag), h[:-lag]])
    turn = np.zeros(N)
    for h in H.values():
        dh = np.abs(np.diff(h, prepend=np.nan))
        dh[0] = 0.0
        turn += dh
    return H, turn


def ret_of(Ht, R, cost=C.COST):
    H, turn = Ht
    g = np.zeros(N)
    for c, h in H.items():
        g += h * R[c]
    return g - turn * cost


TIERS = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
CFGS = {"default": {}, "bills": dict(off="bills"), "tlt": dict(off="tlt"), "goldIEF": dict(gold_to="ief"),
        "nochop": dict(chop_pos=1.0), "novol": dict(voltgt=99.0)}
HELD = {}
for tn, tk in TIERS.items():
    for cn, ck in CFGS.items():
        kw = dict(tk)
        kw.update(ck)
        W = weights(**kw)
        HELD[(tn, cn)] = {d: held(W, d) for d in DAYS}

# exactness check vs common.backtest (Friday book) for F3 default and F2 bills
for tn, cn in (("F3", "default"), ("F2", "bills")):
    kw = dict(TIERS[tn])
    kw.update(CFGS[cn])
    r_ref = C.backtest(P, C.strategy_weights(P, **{k: v for k, v in kw.items()})).values
    r_me = ret_of(HELD[(tn, cn)]["Fri"], BASE_R)
    assert np.max(np.abs(r_ref - r_me)) < 1e-12, f"engine mismatch {tn} {cn}"
log("engine matches common.backtest to 1e-12")


def series(panel, tn, cn, basis):
    R = PANELS[panel]
    if basis == "Fri":
        return ret_of(HELD[(tn, cn)]["Fri"], R)
    return sum(ret_of(HELD[(tn, cn)][d], R) for d in DAYS) / 5.0


WIN = {"IS": (START, FREEZE), "H1": (START, pd.Timestamp("2006-12-31")), "H2": (pd.Timestamp("2007-01-01"), FREEZE),
       "since2000": (pd.Timestamp("2000-01-01"), FREEZE), "FULL": (START, LATEST)}


def st(r, a, b):
    m = (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))
    rr = r[m]
    d = IDX[m]
    e = np.cumprod(1 + rr)
    yrs = (d[-1] - d[0]).days / 365.25
    cg = e[-1] ** (1 / yrs) - 1
    ddc = e / np.maximum.accumulate(e) - 1
    dd = float(ddc.min())
    ex = rr - RF[m]
    return dict(cagr=float(cg), maxdd=dd, calmar=float(cg / abs(dd)) if dd < 0 else np.nan,
                sharpe=float(rr.mean() / rr.std(ddof=1) * np.sqrt(252)),
                xsharpe=float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)),
                ulcer=float(np.sqrt(np.mean((100 * ddc) ** 2))))


def oos_stats(r):
    m = IDX >= OOS_A
    rr = r[m]
    e = np.cumprod(1 + rr)
    return dict(total=float(e[-1] - 1), maxdd=float((e / np.maximum.accumulate(np.concatenate([[1.0], e]))[1:] - 1).min()),
                days=int(m.sum()))


# =========================================================================== strategy runs
log("strategy runs")
SER = {}
SR = {}
for pn in PANELS:
    SR[pn] = {}
    for basis in ("Fri", "tranche"):
        SR[pn][basis] = {}
        for tn in TIERS:
            for cn in CFGS:
                r = series(pn, tn, cn, basis)
                SER[(pn, basis, tn, cn)] = r
                COUNT["strategy_variants"] += 1
                s = {w: st(r, a, b) for w, (a, b) in WIN.items()}
                s["OOS"] = oos_stats(r)
                SR[pn][basis][f"{tn}|{cn}"] = s
        q = PANELS[pn]["qqq"]
        SER[(pn, basis, "QQQ", "bh")] = q
        SR[pn][basis]["QQQ|bh"] = {w: st(q, a, b) for w, (a, b) in WIN.items()}
COUNT["benchmark_series"] = 1

# reproduction of main_tranche_check.json (round-1 numbers)
ref = json.load(open(os.path.join(HERE, "main_tranche_check.json")))
REPRO = {}
for tn, nm in (("F1", "F1-IEF"), ("F2", "F2-IEF"), ("F3", "F3-IEF")):
    for basis, rk in (("Fri", "Fri"), ("tranche", "tranche")):
        mine = SR["P0"][basis][f"{tn}|default"]["IS"]
        theirs = ref[nm][rk]["to_freeze"]
        diff = max(abs(mine["cagr"] - theirs["cagr"]), abs(mine["maxdd"] - theirs["maxdd"]), abs(mine["calmar"] - theirs["calmar"]))
        REPRO[f"{tn}|{basis}"] = dict(cagr=mine["cagr"], maxdd=mine["maxdd"], calmar=mine["calmar"], max_abs_diff_vs_main=diff)
        assert diff < 1e-9, f"reproduction failed {tn} {basis}: {diff}"
        oos_ref = ref[nm][rk]["oos"]
        assert abs(SR["P0"][basis][f"{tn}|default"]["OOS"]["total"] - oos_ref) < 1e-9
RES["reproduction_of_main_tranche_check"] = REPRO
log("P0 reproduces main_tranche_check.json (IS and OOS) to 1e-9")

# deltas vs P0 (default config) for every correction, tier, basis
DEL = {}
for pn in PANELS:
    if pn == "P0":
        continue
    DEL[pn] = {}
    for tn in TIERS:
        for basis in ("Fri", "tranche"):
            row = {}
            for w in ("IS", "H1", "H2", "since2000"):
                c, b0 = SR[pn][basis][f"{tn}|default"][w], SR["P0"][basis][f"{tn}|default"][w]
                row[w] = dict(cagr=c["cagr"], maxdd=c["maxdd"], calmar=c["calmar"], xsharpe=c["xsharpe"],
                              d_cagr=c["cagr"] - b0["cagr"], d_maxdd=c["maxdd"] - b0["maxdd"],
                              d_calmar=c["calmar"] - b0["calmar"], d_xsharpe=c["xsharpe"] - b0["xsharpe"])
            row["OOS_total_diff"] = SR[pn][basis][f"{tn}|default"]["OOS"]["total"] - SR["P0"][basis][f"{tn}|default"]["OOS"]["total"]
            DEL[pn][f"{tn}|{basis}"] = row
RES["deltas_vs_P0"] = DEL

print("\n  IS 1988-01-01..2026-07-24, default configs.  value (delta vs P0)")
for basis in ("Fri", "tranche"):
    print(f"  --- {basis} ---")
    for tn in TIERS:
        b0 = SR["P0"][basis][f"{tn}|default"]["IS"]
        print(f"  {tn} P0       CAGR {b0['cagr']*100:6.2f}%  maxDD {b0['maxdd']*100:6.1f}%  Calmar {b0['calmar']:.3f}  xSharpe {b0['xsharpe']:.3f}")
        for pn in DEL:
            x = DEL[pn][f"{tn}|{basis}"]["IS"]
            print(f"  {tn} {pn:8s} CAGR {x['cagr']*100:6.2f}% ({x['d_cagr']*100:+.2f})  maxDD {x['maxdd']*100:6.1f}% "
                  f"({x['d_maxdd']*100:+.1f})  Calmar {x['calmar']:.3f} ({x['d_calmar']:+.3f})  xSharpe {x['xsharpe']:.3f} "
                  f"| H1 dCAGR {DEL[pn][f'{tn}|{basis}']['H1']['d_cagr']*100:+.2f} dCal {DEL[pn][f'{tn}|{basis}']['H1']['d_calmar']:+.3f}"
                  f" | OOS diff {DEL[pn][f'{tn}|{basis}']['OOS_total_diff']*100:+.4f}")


# =========================================================================== bootstrap
def boot_matrix(cols, xs_cols, n=NBOOT, block=63, seed=0):
    """paired circular block bootstrap (round-1 scheme: 63-day blocks, seed 0, CAGR on 252/T) of
    CAGR / maxDD / Calmar for every column, excess Sharpe for xs_cols.  Same index draws for all."""
    m = (IDX >= START) & (IDX <= FREEZE)
    names = list(cols)
    X = np.column_stack([cols[k][m] for k in names])
    Lg = np.log1p(X)
    XE = np.column_stack([cols[k][m] - RF[m] for k in xs_cols])
    T = len(X)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    out = {k: np.empty((n, len(names))) for k in ("cagr", "maxdd", "calmar")}
    out["xsharpe"] = np.empty((n, len(xs_cols)))
    ar = np.arange(block)[None, :]
    for i in range(n):
        s0 = rng.integers(0, T, nb)
        ix = (s0[:, None] + ar).ravel()[:T] % T
        cl = np.cumsum(Lg[ix], axis=0)
        g = np.expm1(cl[-1] * 252 / T)
        dd = np.expm1((cl - np.maximum.accumulate(cl, axis=0)).min(axis=0))
        xe = XE[ix]
        out["cagr"][i] = g
        out["maxdd"][i] = dd
        out["calmar"][i] = g / np.abs(dd)
        out["xsharpe"][i] = xe.mean(axis=0) / xe.std(axis=0, ddof=1) * np.sqrt(252)
    res = {k: {nm: out[k][:, j] for j, nm in enumerate(names)} for k in ("cagr", "maxdd", "calmar")}
    res["xsharpe"] = {nm: out["xsharpe"][:, j] for j, nm in enumerate(xs_cols)}
    return res, T


def bdiff(B, metric, a, b):
    v = B[metric][a] - B[metric][b]
    return dict(boot_mean=float(v.mean()), lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)),
                p_gt0=float((v > 0).mean()))


log(f"bootstrap ({NBOOT} draws x {len(PANELS)} panels x 2 bases)")
BOOT = {}
for pn in PANELS:
    BOOT[pn] = {}
    for basis in ("Fri", "tranche"):
        # only R1 (F2 vs F3), R3 (gold vs IEF slot) and R5 (vs QQQ) use the bootstrap
        cols = {f"{tn}|{cn}": SER[(pn, basis, tn, cn)] for tn in TIERS for cn in ("default", "goldIEF")}
        cols["QQQ|bh"] = SER[(pn, basis, "QQQ", "bh")]
        B, T = boot_matrix(cols, ["F2|default", "F3|default"])
        COUNT["bootstrap_runs"] += 1
        BOOT[pn][basis] = B
    log(f"   {pn} done")


# =========================================================================== conclusions
def sgn(x):
    return int(np.sign(x)) if np.isfinite(x) else 0


def conclusions(pn, basis, B=None):
    S = SR[pn][basis]
    B = BOOT[pn][basis] if B is None else B
    out = {}
    # R1 2x vs 3x
    xs = bdiff(B, "xsharpe", "F2|default", "F3|default")
    cg = bdiff(B, "cagr", "F3|default", "F2|default")
    cal = bdiff(B, "calmar", "F2|default", "F3|default")
    f2, f3 = S["F2|default"]["IS"], S["F3|default"]["IS"]
    out["R1_2x_vs_3x"] = dict(
        xsharpe_F2=f2["xsharpe"], xsharpe_F3=f3["xsharpe"], d_xsharpe_F2_minus_F3=f2["xsharpe"] - f3["xsharpe"], boot_xsharpe=xs,
        d_cagr_F3_minus_F2=f3["cagr"] - f2["cagr"], boot_cagr=cg,
        d_calmar_F2_minus_F3=f2["calmar"] - f3["calmar"], boot_calmar=cal,
        maxdd_F2=f2["maxdd"], maxdd_F3=f3["maxdd"],
        status=dict(xsharpe_tie=bool(0.05 < xs["p_gt0"] < 0.95), f3_cagr_sig=bool(cg["lo95"] > 0),
                    f2_shallower=bool(f2["maxdd"] > f3["maxdd"])))
    out["R2_IEF_vs_bills"] = {}
    out["R3_gold_vs_IEF"] = {}
    out["R4_chop_vol"] = {}
    out["R5_edge_vs_QQQ"] = {}
    out["R6_IEF_vs_TLT"] = {}
    for tn in TIERS:
        d = {w: S[f"{tn}|default"][w]["calmar"] - S[f"{tn}|bills"][w]["calmar"] for w in ("IS", "H1", "H2")}
        out["R2_IEF_vs_bills"][tn] = dict(d_calmar=d, d_cagr_IS=S[f"{tn}|default"]["IS"]["cagr"] - S[f"{tn}|bills"]["IS"]["cagr"],
                                          status={w: sgn(v) for w, v in d.items()})
        bc = bdiff(B, "calmar", f"{tn}|default", f"{tn}|goldIEF")
        bg = bdiff(B, "cagr", f"{tn}|default", f"{tn}|goldIEF")
        out["R3_gold_vs_IEF"][tn] = dict(d_calmar_IS=S[f"{tn}|default"]["IS"]["calmar"] - S[f"{tn}|goldIEF"]["IS"]["calmar"],
                                         d_cagr_IS=S[f"{tn}|default"]["IS"]["cagr"] - S[f"{tn}|goldIEF"]["IS"]["cagr"],
                                         d_maxdd_IS=S[f"{tn}|default"]["IS"]["maxdd"] - S[f"{tn}|goldIEF"]["IS"]["maxdd"],
                                         d_calmar_H1=S[f"{tn}|default"]["H1"]["calmar"] - S[f"{tn}|goldIEF"]["H1"]["calmar"],
                                         d_calmar_H2=S[f"{tn}|default"]["H2"]["calmar"] - S[f"{tn}|goldIEF"]["H2"]["calmar"],
                                         boot_calmar=bc, boot_cagr=bg,
                                         status=dict(not_significant=bool(0.05 < bc["p_gt0"] < 0.95)))
        r4 = {}
        for comp, cn in (("chop_rule", "nochop"), ("vol_target", "novol")):
            d = {w: S[f"{tn}|default"][w]["calmar"] - S[f"{tn}|{cn}"][w]["calmar"] for w in ("IS", "H1", "H2")}
            r4[comp] = dict(d_calmar=d, d_cagr_IS=S[f"{tn}|default"]["IS"]["cagr"] - S[f"{tn}|{cn}"]["IS"]["cagr"],
                            d_maxdd_IS=S[f"{tn}|default"]["IS"]["maxdd"] - S[f"{tn}|{cn}"]["IS"]["maxdd"],
                            status=dict(sign_IS=sgn(d["IS"]), helps_both_halves=bool(d["H1"] > 0 and d["H2"] > 0)))
        out["R4_chop_vol"][tn] = r4
        bq = bdiff(B, "calmar", f"{tn}|default", "QQQ|bh")
        bqc = bdiff(B, "cagr", f"{tn}|default", "QQQ|bh")
        dcal = S[f"{tn}|default"]["IS"]["calmar"] - S["QQQ|bh"]["IS"]["calmar"]
        dcg = S[f"{tn}|default"]["IS"]["cagr"] - S["QQQ|bh"]["IS"]["cagr"]
        out["R5_edge_vs_QQQ"][tn] = dict(d_calmar_IS=dcal, d_cagr_IS=dcg, boot_calmar=bq, boot_cagr=bqc,
                                         status=dict(calmar_sign=sgn(dcal), calmar_ci_excl0=bool(bq["lo95"] > 0 or bq["hi95"] < 0),
                                                     cagr_sign=sgn(dcg), cagr_ci_excl0=bool(bqc["lo95"] > 0 or bqc["hi95"] < 0)))
        d = {w: S[f"{tn}|default"][w]["calmar"] - S[f"{tn}|tlt"][w]["calmar"] for w in ("IS", "H1", "H2")}
        out["R6_IEF_vs_TLT"][tn] = dict(d_calmar=d, d_maxdd_IS=S[f"{tn}|default"]["IS"]["maxdd"] - S[f"{tn}|tlt"]["IS"]["maxdd"],
                                        status={w: sgn(v) for w, v in d.items()})
    return out


def flatten_status(c):
    f = {}
    f["R1|xsharpe_tie"] = c["R1_2x_vs_3x"]["status"]["xsharpe_tie"]
    f["R1|f3_cagr_sig"] = c["R1_2x_vs_3x"]["status"]["f3_cagr_sig"]
    f["R1|f2_shallower"] = c["R1_2x_vs_3x"]["status"]["f2_shallower"]
    for tn in TIERS:
        for w, v in c["R2_IEF_vs_bills"][tn]["status"].items():
            f[f"R2|{tn}|{w}"] = v
        f[f"R3|{tn}|not_sig"] = c["R3_gold_vs_IEF"][tn]["status"]["not_significant"]
        for comp in ("chop_rule", "vol_target"):
            for k, v in c["R4_chop_vol"][tn][comp]["status"].items():
                f[f"R4|{tn}|{comp}|{k}"] = v
        for k, v in c["R5_edge_vs_QQQ"][tn]["status"].items():
            f[f"R5|{tn}|{k}"] = v
        for w, v in c["R6_IEF_vs_TLT"][tn]["status"].items():
            f[f"R6|{tn}|{w}"] = v
    return f


CONC = {pn: {b: conclusions(pn, b) for b in ("Fri", "tranche")} for pn in PANELS}
FLIPS = {}
for pn in PANELS:
    if pn == "P0":
        continue
    FLIPS[pn] = {}
    for b in ("Fri", "tranche"):
        s0, s1 = flatten_status(CONC["P0"][b]), flatten_status(CONC[pn][b])
        FLIPS[pn][b] = {k: dict(P0=s0[k], corrected=s1[k]) for k in s0 if s0[k] != s1[k]}
RES["conclusions"] = CONC
RES["status_P0"] = {b: flatten_status(CONC["P0"][b]) for b in ("Fri", "tranche")}
RES["flips_vs_P0"] = FLIPS
prim_flips = {pn: FLIPS[pn] for pn in PRIMARY if any(FLIPS[pn][b] for b in FLIPS[pn])}
sens_flips = {pn: FLIPS[pn] for pn in SENS if any(FLIPS[pn][b] for b in FLIPS[pn])}
RES["verdict"] = dict(any_primary_flip=bool(prim_flips), primary_flips=prim_flips,
                      any_sensitivity_flip=bool(sens_flips), sensitivity_flips=sens_flips,
                      n_status_items_per_basis=len(RES["status_P0"]["Fri"]))

# ---------------------------------------------------------------- INFORMATION ONLY (not a decision input):
# bootstrap Monte-Carlo robustness of every flipped bootstrap-based status item (R1, R3, R5): re-draw with
# seeds 1-4 on P0 and on the corrected panel (same basis) and count how often each panel shows its status.
BOOT_ITEMS = ("R1|", "R3|", "R5|")
SEEDCHK = {}
for pn, fb in FLIPS.items():
    for b, fl in fb.items():
        items = [k for k in fl if k.startswith(BOOT_ITEMS)]
        if not items:
            continue
        res = {k: dict(P0=[], corrected=[]) for k in items}
        for seed in range(5):
            for lab, pp in (("P0", "P0"), ("corrected", pn)):
                if seed == 0:
                    stt = flatten_status(CONC[pp][b])
                else:
                    cols = {f"{tn}|{cn}": SER[(pp, b, tn, cn)] for tn in TIERS for cn in ("default", "goldIEF")}
                    cols["QQQ|bh"] = SER[(pp, b, "QQQ", "bh")]
                    Bs, _ = boot_matrix(cols, ["F2|default", "F3|default"], seed=seed)
                    COUNT["bootstrap_runs_info_seedcheck"] = COUNT.get("bootstrap_runs_info_seedcheck", 0) + 1
                    stt = flatten_status(conclusions(pp, b, Bs))
                for k in items:
                    res[k][lab].append(stt[k])
        for k in items:
            p0, pc = res[k]["P0"], res[k]["corrected"]
            res[k]["seeds_where_status_differs"] = int(sum(x != y for x, y in zip(p0, pc)))
            res[k]["P0_status_seed0_held_in_P0_seeds"] = int(sum(x == p0[0] for x in p0))
        SEEDCHK[f"{pn}|{b}"] = res
        log(f"   seed check {pn} {b}: " + "; ".join(f"{k}: P0 {res[k]['P0']} corrected {res[k]['corrected']}" for k in items))
RES["info_seed_robustness_of_flips"] = SEEDCHK

print("\n  CONCLUSION STATUS (IS).  flips vs P0 (same basis):")
for pn in FLIPS:
    for b in ("Fri", "tranche"):
        fl = FLIPS[pn][b]
        print(f"  {pn:8s} {b:7s}: {len(fl)} flips " + ("; ".join(f"{k}: {v['P0']}->{v['corrected']}" for k, v in fl.items()) if fl else ""))

print("\n  KEY CONCLUSION NUMBERS (IS)")
for pn in ["P0"] + PRIMARY + SENS:
    for b in ("Fri", "tranche"):
        c = CONC[pn][b]
        r1 = c["R1_2x_vs_3x"]
        print(f"  {pn:8s} {b:7s} R1 xSh F2 {r1['xsharpe_F2']:.3f} F3 {r1['xsharpe_F3']:.3f} P(F2>F3) {r1['boot_xsharpe']['p_gt0']:.2f} | "
              f"F3-F2 CAGR {r1['d_cagr_F3_minus_F2']*100:+.2f} [{r1['boot_cagr']['lo95']*100:+.2f},{r1['boot_cagr']['hi95']*100:+.2f}] | "
              f"Cal F2-F3 {r1['d_calmar_F2_minus_F3']:+.3f}")
        print("     R2 IEF-bills dCal IS " + " ".join(f"{tn} {c['R2_IEF_vs_bills'][tn]['d_calmar']['IS']:+.3f}"
                                                   f"(H1 {c['R2_IEF_vs_bills'][tn]['d_calmar']['H1']:+.3f} H2 {c['R2_IEF_vs_bills'][tn]['d_calmar']['H2']:+.3f})" for tn in TIERS))
        print("     R3 gold-IEF dCal " + " ".join(f"{tn} {c['R3_gold_vs_IEF'][tn]['d_calmar_IS']:+.3f} P {c['R3_gold_vs_IEF'][tn]['boot_calmar']['p_gt0']:.2f}"
                                                   f" dCAGR {c['R3_gold_vs_IEF'][tn]['d_cagr_IS']*100:+.2f}" for tn in TIERS))
        print("     R4 chop dCal " + " ".join(f"{tn} {c['R4_chop_vol'][tn]['chop_rule']['d_calmar']['IS']:+.3f}"
                                               f"(H1 {c['R4_chop_vol'][tn]['chop_rule']['d_calmar']['H1']:+.3f} H2 {c['R4_chop_vol'][tn]['chop_rule']['d_calmar']['H2']:+.3f})" for tn in TIERS))
        print("     R4 vol  dCal " + " ".join(f"{tn} {c['R4_chop_vol'][tn]['vol_target']['d_calmar']['IS']:+.3f}"
                                               f"(H1 {c['R4_chop_vol'][tn]['vol_target']['d_calmar']['H1']:+.3f} H2 {c['R4_chop_vol'][tn]['vol_target']['d_calmar']['H2']:+.3f})" for tn in TIERS))
        print("     R5 vs QQQ " + " ".join(f"{tn} dCal {c['R5_edge_vs_QQQ'][tn]['d_calmar_IS']:+.3f} [{c['R5_edge_vs_QQQ'][tn]['boot_calmar']['lo95']:+.2f},"
                                            f"{c['R5_edge_vs_QQQ'][tn]['boot_calmar']['hi95']:+.2f}] dCAGR {c['R5_edge_vs_QQQ'][tn]['d_cagr_IS']*100:+.2f} "
                                            f"[{c['R5_edge_vs_QQQ'][tn]['boot_cagr']['lo95']*100:+.2f},{c['R5_edge_vs_QQQ'][tn]['boot_cagr']['hi95']*100:+.2f}]" for tn in TIERS))
        print("     R6 IEF-TLT dCal " + " ".join(f"{tn} {c['R6_IEF_vs_TLT'][tn]['d_calmar']['IS']:+.3f}"
                                                  f"(H1 {c['R6_IEF_vs_TLT'][tn]['d_calmar']['H1']:+.3f} H2 {c['R6_IEF_vs_TLT'][tn]['d_calmar']['H2']:+.3f})" for tn in TIERS))

# full stats table (all panels, tiers, configs, bases)
RES["stats"] = SR
RES["variant_count"] = dict(COUNT, note="strategy_variants = panels x bases x tiers x configs (each a full backtest; "
                                         "tranche = 5 weekday books averaged); calibration_fits = regressions/synthetic fits; "
                                         "bootstrap_runs = one joint paired bootstrap per panel x basis")
RES["runtime_s"] = time.time() - T0


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(RES), open(OUT_JSON, "w"), indent=1)
log(f"saved {OUT_JSON}; variants {COUNT}")
