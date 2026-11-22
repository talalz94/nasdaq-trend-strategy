"""
INTEGRITY REVIEW of r2_model_hygiene.py -- independent re-implementation (read-only w.r.t. project files).
  1. financing regressions (own HAC), regimes, stability test, applied drags, splice limits
  2. par-bond duration/convexity: closed-form par Macaulay formula + numeric 2nd difference (independent of the
     specialist's cash-flow sums), B synthetic, fit gate; gate re-done vs a fixed calibration re-fit on <= FREEZE only
  3. gold: own month-end step series + a 'spec-exact' version (calendar month avg-to-avg move, reconstructed from the
     interpolation slopes, booked on the month's last trading day)
  4. strategy runs through common.strategy_weights + common.backtest on modified panel copies (Friday P.mark book and
     the 5-weekday tranche), compared with the saved JSON for all 10 panels x 3 tiers x 6 configs
  5. status items R2/R4/R6 recomputed from (4); flips vs P0
Writes nothing except <scratch_dir>/verify_checks.json
usage: python -u update_2026_11/r2_verify_model_hygiene_integrity_checks.py <scratch_dir>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
scratch = sys.argv[1]
P0 = C.panel()
IDX = P0.index
N = len(IDX)
FREEZE, START = C.FREEZE, C.START
OOS_A = pd.Timestamp("2026-07-25")
J = json.load(open(os.path.join(HERE, "r2_model_hygiene.json")))
OUT = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ 1. financing
def hac_ols(y, X, L=20):
    T = len(y)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    u = X * e[:, None]
    S = (u.T @ u) / T
    for l in range(1, L + 1):
        G = (u[l:].T @ u[:-l]) / T
        S = S + (1.0 - l / (L + 1.0)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    V = Q @ S @ Q / T
    return b, np.sqrt(np.diag(V)), e


REG = {"2006-09": ("2006-01-01", "2009-12-31"), "2010-15": ("2010-01-01", "2015-12-31"),
       "2016-21": ("2016-01-01", "2021-12-31"), "2022-frz": ("2022-01-01", "2026-07-24")}
qg = P0["qqq_real"] + 0.0020 / 252
FIN = {}
for fund, lev in (("qld", 2), ("tqqq", 3)):
    real = P0[f"{fund}_real"]
    FIN[fund] = {}
    for spec, sim in (("literal", P0[f"{fund}_sim"]), ("divneutral", lev * qg - 0.0095 / 252 - (lev - 1) * P0["rf_d"])):
        df = pd.DataFrame({"g": real - sim, "rf": P0["rf_d"]}).dropna()
        dis = df[df.index <= FREEZE]
        X = np.column_stack([np.ones(len(dis)), dis.rf.values])
        b, se, e = hac_ols(dis.g.values, X)
        res = pd.Series(e, index=dis.index)
        regs = {}
        for rk, (a0, b0) in REG.items():
            m = (dis.index >= a0) & (dis.index <= b0)
            if m.sum() < 100:
                continue
            br, sr, _ = hac_ols(res[m].values, np.ones((int(m.sum()), 1)))
            regs[rk] = dict(mean_gap_ann=float(dis.g[m].mean() * 252), mean_rf=float(dis.rf[m].mean() * 252),
                            resid_ann=float(br[0] * 252), resid_se_ann=float(sr[0] * 252),
                            within2=bool(abs(br[0]) <= 2 * sr[0]))
        do = df[df.index >= OOS_A]
        FIN[fund][spec] = dict(alpha_ann=float(b[0] * 252), alpha_se_ann=float(se[0] * 252), beta=float(b[1]),
                               beta_se=float(se[1]), alpha_daily=float(b[0]), regimes=regs,
                               oos_gap_ann=float(do.g.mean() * 252), oos_pred_ann=float((b[0] + b[1] * do.rf.mean()) * 252))
        jp = J["financing"][fund][spec]["pooled"]
        log(f"{fund} {spec:10s} alpha {b[0]*252*100:+.3f}% (se {se[0]*252*100:.3f}) beta {b[1]:+.4f} (se {se[1]:.4f}) "
            f"| JSON alpha {jp['alpha_ann']*100:+.3f} se {jp['alpha_se_ann']*100:.3f} beta {jp['beta']:+.4f} se {jp['beta_se']:.4f} | "
            + " ".join(f"{k}: gap {v['mean_gap_ann']*100:+.2f} rf {v['mean_rf']*100:.2f} resid {v['resid_ann']*100:+.2f}"
                       f"(se {v['resid_se_ann']*100:.2f}) ok={v['within2']}" for k, v in regs.items())
            + f" | OOS gap {FIN[fund][spec]['oos_gap_ann']*100:+.2f} pred {FIN[fund][spec]['oos_pred_ann']*100:+.2f}")
# literal minus dividend-neutral mean gap over the fit window = lev * mean(QQQ_real + 0.2% - NDX price)
for fund, lev in (("qld", 2), ("tqqq", 3)):
    real = P0[f"{fund}_real"]
    m = real.notna() & (IDX <= FREEZE)
    dvd = float((qg[m] - P0["ndx_ret"][m]).mean() * 252)
    FIN[fund]["implied_qqq_tr_minus_ndx_price_ann"] = dvd
    FIN[fund]["alpha_lit_minus_alpha_dn"] = FIN[fund]["literal"]["alpha_ann"] - FIN[fund]["divneutral"]["alpha_ann"]
    log(f"{fund}: mean(QQQ_TR+0.2%-NDX_price) over fit window {dvd*100:.3f}%/yr -> lev x = {lev*dvd*100:.2f}%/yr; "
        f"alpha_lit - alpha_dn = {FIN[fund]['alpha_lit_minus_alpha_dn']*100:.2f}%/yr")

per_unit = {f"{f}|{rk}": -FIN[f]["divneutral"]["regimes"][rk]["mean_gap_ann"] / (lv - 1)
            for f, lv in (("qld", 2), ("tqqq", 3)) for rk in FIN[f]["divneutral"]["regimes"]}
W_STRESS = max(per_unit.values())
FIN["stress_w"] = W_STRESS
log(f"stress per unit borrowed: {W_STRESS*100:.3f}% ({max(per_unit, key=per_unit.get)})")


def adj_sleeve(fund, adj):
    real = P0[f"{fund}_real"]
    s = P0[fund].copy()
    pre = real.isna()
    s[pre] = (P0[f"{fund}_sim"] + adj)[pre].fillna(0)
    return s


LEV = {"qld": 2, "tqqq": 3}
SLV = {}
for f in ("qld", "tqqq"):
    dn, lt = FIN[f]["divneutral"], FIN[f]["literal"]
    SLV[("A_div", f)] = adj_sleeve(f, dn["alpha_daily"] + dn["beta"] * P0["rf_d"])
    SLV[("A_lit", f)] = adj_sleeve(f, lt["alpha_daily"] + lt["beta"] * P0["rf_d"])
    SLV[("A_stress", f)] = adj_sleeve(f, pd.Series(-(LEV[f] - 1) * W_STRESS / 252, index=IDX))
    # touched days must be exactly the pre-inception days
    ch = (SLV[("A_div", f)] != P0[f])
    assert ch[P0[f"{f}_real"].notna()].sum() == 0
    pre = P0[f"{f}_real"].isna() & (IDX >= START)
    rfm = float(P0["rf_d"][pre].mean() * 252)
    cg = lambda x: float(np.prod(1 + x.values) ** (252 / len(x)) - 1)
    FIN[f"applied_{f}"] = dict(first_real=str(P0[f"{f}_real"].first_valid_index().date()), last_changed=str(ch[ch].index.max().date()),
                               mean_rf_pre=rfm, A_div_ann=dn["alpha_ann"] + dn["beta"] * rfm,
                               cagr_pre_P0=cg(P0[f][pre]), cagr_pre_A_div=cg(SLV[("A_div", f)][pre]))
    log(f"applied {f}: first real {FIN[f'applied_{f}']['first_real']}, last changed day {FIN[f'applied_{f}']['last_changed']}, "
        f"rf_pre {rfm*100:.2f}% A_div {FIN[f'applied_{f}']['A_div_ann']*100:+.2f}%/yr; pre-incep sleeve CAGR "
        f"{FIN[f'applied_{f}']['cagr_pre_P0']*100:.2f}% -> {FIN[f'applied_{f}']['cagr_pre_A_div']*100:.2f}%")
OUT["financing"] = FIN


# ------------------------------------------------------------------ 2. duration
def par_price(ycoup, yld, T):
    """price per 1 face of a semiannual bond with annual coupon ycoup at annual yield yld, maturity T (closed form)."""
    n = int(round(2 * T))
    i = yld / 2.0
    return (ycoup / 2.0) * (1 - (1 + i) ** (-n)) / i + (1 + i) ** (-n)


def par_dc_closed(y, T, h=1e-4):
    y = np.asarray(y, float)
    n = int(round(2 * T))
    i = y / 2.0
    dmac_periods = (1 + i) / i * (1 - (1 + i) ** (-n))          # Macaulay duration of a par bond, in periods
    dmod = dmac_periods / 2.0 / (1 + i)
    p0 = par_price(y, y, T)
    cvx = (par_price(y, y + h, T) - 2 * p0 + par_price(y, y - h, T)) / h ** 2 / p0
    return dmod, cvx


# check the specialist's cash-flow formula at a few yields (re-implemented verbatim for comparison only)
def spec_par_dc(y, T):
    y = np.asarray(y, float)
    n = int(round(2 * T)); k = np.arange(1, n + 1)[None, :]; i = (y / 2)[:, None]
    cf = np.where(k == n, i + 1.0, i); disc = (1 + i) ** (-k); price = (cf * disc).sum(1)
    return (cf * disc * (k / 2.0)).sum(1) / price / (1 + i[:, 0]), (cf * disc * k * (k + 1) / 4.0 / (1 + i) ** 2).sum(1) / price


ys = np.array([0.01, 0.03, 0.05, 0.08, 0.10])
for T in (7.5, 20.0):
    d1, c1 = par_dc_closed(ys, T)
    d2, c2 = spec_par_dc(ys, T)
    log(f"par {T}y: D closed {np.round(d1,4)} vs spec {np.round(d2,4)}; C numeric {np.round(c1,2)} vs spec {np.round(c2,2)}; "
        f"max|dD| {np.max(np.abs(d1-d2)):.2e} max|dC| {np.max(np.abs(c1-c2)):.2e}")


def fitst(real, syn, a, b):
    m = pd.concat([real, syn], axis=1)
    m = m[(m.index >= pd.Timestamp(a)) & (m.index <= pd.Timestamp(b))].dropna()
    m.columns = ["r", "s"]
    cg = lambda x: float(np.prod(1 + x.values) ** (252 / len(x)) - 1)
    return dict(te=float((m.r - m.s).std() * np.sqrt(252)), corr=float(m.r.corr(m.s)), gap=cg(m.s) - cg(m.r))


CAL_A = pd.Timestamp("2002-07-31")
DUR = {}
BOND = {}
for name, ycol, T in (("ief", "tnx", 7.5), ("tlt", "tyx", 20.0)):
    yd = P0[ycol] / 100.0
    dy = yd.diff(); y1 = yd.shift(1); carry = y1 / 252
    D, Cv = par_dc_closed(y1.fillna(0.05).values, T)
    D = pd.Series(np.where(y1.notna(), D, np.nan), index=IDX); Cv = pd.Series(np.where(y1.notna(), Cv, np.nan), index=IDX)
    real = P0[f"{name}_real"]
    mcal = (IDX >= CAL_A) & (IDX <= FREEZE)
    aB = float((real - carry + D * dy - 0.5 * Cv * dy ** 2)[mcal].dropna().mean())
    synB = aB + carry - D * dy + 0.5 * Cv * dy ** 2
    # level matched (same recipe as the specialist, for the B_k panel)
    df = pd.concat([real - carry - 0.5 * Cv * dy ** 2, -D * dy], axis=1)[mcal].dropna()
    kk = float(np.linalg.lstsq(np.column_stack([np.ones(len(df)), df.iloc[:, 1].values]), df.iloc[:, 0].values, rcond=None)[0][1])
    ak = float((real - carry + kk * D * dy - 0.5 * kk ** 2 * Cv * dy ** 2)[mcal].dropna().mean())
    synK = ak + carry - kk * D * dy + 0.5 * kk ** 2 * Cv * dy ** 2
    fixed = P0[f"{name}_syn"]
    # fixed calibration re-fit on <= FREEZE only (the panel's was fit through 2026-11-17)
    dfx = pd.concat([real, carry, dy], axis=1).dropna(); dfx.columns = ["r", "c", "dy"]
    dfx = dfx[dfx.index <= FREEZE]
    a_, D_, C_ = np.linalg.lstsq(np.column_stack([np.ones(len(dfx)), -dfx.dy.values, 0.5 * dfx.dy.values ** 2]),
                                 (dfx.r - dfx.c).values, rcond=None)[0]
    fixed_frz = a_ + carry - D_ * dy + 0.5 * C_ * dy ** 2
    fx, fz, fb, fk = (fitst(real, s, CAL_A, FREEZE) for s in (fixed, fixed_frz, synB, synK))

    def gate(x, ref):
        return bool(x["te"] <= 1.10 * ref["te"] and x["corr"] >= ref["corr"] - 0.01 and abs(x["gap"]) <= abs(ref["gap"]) + 0.0025)
    pre = (IDX >= START) & (IDX < CAL_A)
    cg = lambda x: float(np.prod(1 + x.fillna(0).values) ** (252 / len(x)) - 1)
    DUR[name] = dict(aB_ann=aB * 252, k=kk, fixed_refit_frz=dict(D=float(D_), C=float(C_), a_ann=float(a_ * 252)),
                     fit_fixed_panel=fx, fit_fixed_refit_frz=fz, fit_B=fb, fit_Bk=fk,
                     gate_B_vs_panel_fixed=gate(fb, fx), gate_B_vs_refit_fixed=gate(fb, fz),
                     gate_Bk_vs_panel_fixed=gate(fk, fx),
                     pre2002=dict(meanD=float(D[pre].mean()), minD=float(D[pre].min()), maxD=float(D[pre].max()),
                                  cagr_fixed=cg(fixed[pre]), cagr_B=cg(synB[pre]), vol_fixed=float(fixed[pre].std() * np.sqrt(252)),
                                  vol_B=float(synB[pre].std() * np.sqrt(252))))
    BOND[("B", name)] = real.fillna(synB).fillna(0)
    BOND[("B_k", name)] = real.fillna(synK).fillna(0)
    jd = J["duration"][name]
    log(f"{name}: aB {aB*252*100:+.3f}%/yr (JSON {jd['alpha_B_ann']*100:+.3f}) k {kk:.4f} (JSON {jd['k']:.4f}) | TE fixed {fx['te']*100:.3f} "
        f"refit {fz['te']*100:.3f} B {fb['te']*100:.3f} | corr {fx['corr']:.4f}/{fz['corr']:.4f}/{fb['corr']:.4f} | gap {fx['gap']*100:+.3f}/"
        f"{fz['gap']*100:+.3f}/{fb['gap']*100:+.3f}pp | gate B vs panel-fixed {DUR[name]['gate_B_vs_panel_fixed']} vs FREEZE-refit fixed "
        f"{DUR[name]['gate_B_vs_refit_fixed']} | refit D {D_:.3f} C {C_:.1f} | pre2002 meanD {DUR[name]['pre2002']['meanD']:.2f} "
        f"[{DUR[name]['pre2002']['minD']:.2f},{DUR[name]['pre2002']['maxD']:.2f}] CAGR fixed {DUR[name]['pre2002']['cagr_fixed']*100:.2f} "
        f"B {DUR[name]['pre2002']['cagr_B']*100:.2f} vol {DUR[name]['pre2002']['vol_fixed']*100:.2f}->{DUR[name]['pre2002']['vol_B']*100:.2f}")
OUT["duration"] = DUR

# ------------------------------------------------------------------ 3. gold
interp = P0["gold_src"].eq("monthly-interp")
FEE = 0.0040 / 252
gp = P0["gold_phys"].fillna(0)
lr = np.log1p(gp + FEE).where(interp, 0.0)
lr.iloc[0] = 0.0
mon = IDX.to_period("M")
iv = IDX[interp.values]
last_td = pd.Series(iv, index=iv).groupby(iv.to_period("M")).max()
bookday = pd.Series(False, index=IDX); bookday[pd.DatetimeIndex(last_td.values)] = True
# (i) the specialist's definition: sum of daily interp log returns of the calendar month, booked on its last trading day
msum = lr[interp].groupby(mon[interp.values]).sum()
c1 = pd.Series(-FEE, index=IDX)
c1[pd.DatetimeIndex(last_td.values)] = np.expm1(msum.values) - FEE
c1[~interp] = gp[~interp]
c1.iloc[0] = gp.iloc[0]
# (ii) spec-exact: average(M)/average(M-1) (knots at calendar month ends), booked on M's last trading day
X = lr.cumsum()                                         # interpolated log price (relative)
prev = pd.Series(IDX, index=IDX).shift(1)
same_month = (prev.dt.to_period("M") == pd.Series(mon, index=IDX)) & interp
gapdays = (pd.Series(IDX, index=IDX) - prev).dt.days
slope = (lr / gapdays)[same_month].groupby(mon[same_month.values]).median()
slope_spread = (lr / gapdays)[same_month].groupby(mon[same_month.values]).agg(lambda v: v.max() - v.min())
print(f"  gold interpolation slope consistency within months: max spread {slope_spread.max():.2e}")
months = list(last_td.index)
knot = {}
for Mp in months:
    tl = last_td[Mp]
    me = Mp.to_timestamp(how="end").normalize()
    if Mp == months[-1]:
        knot[Mp] = X[tl]                                # partial Aug-2000: stop at the last interpolated day
    else:
        knot[Mp] = X[tl] + slope[Mp] * (me - tl).days
c1x = pd.Series(-FEE, index=IDX)
for j, Mp in enumerate(months):
    tl = last_td[Mp]
    base = X[IDX[1]] - lr[IDX[1]] if j == 0 else knot[months[j - 1]]
    c1x[tl] = np.expm1(knot[Mp] - base) - FEE
c1x[~interp] = gp[~interp]
c1x.iloc[0] = gp.iloc[0]
c2 = gp.where(~interp, P0["gold_miners"].fillna(0))
m88 = interp & (IDX >= START)
cgs = lambda x: float(np.prod(1 + x.values) ** (252 / len(x)) - 1)
G = dict(total_interp=float(np.expm1(lr[interp].sum())), total_c1=float(np.prod(1 + (c1[interp] + FEE)) - 1),
         total_c1x=float(np.prod(1 + (c1x[interp] + FEE)) - 1),
         max_abs_monthly_booking_diff_c1_vs_exact=float((c1 - c1x)[bookday].abs().max()),
         mean_abs_monthly_booking_diff=float((c1 - c1x)[bookday].abs().mean()),
         cagr88=dict(interp=cgs(gp[m88]), c1=cgs(c1[m88]), c1x=cgs(c1x[m88]), miners=cgs(c2[m88])),
         dvol88=dict(interp=float(gp[m88].std() * np.sqrt(252)), c1=float(c1[m88].std() * np.sqrt(252)),
                     miners=float(c2[m88].std() * np.sqrt(252))))
OUT["gold"] = G
log(f"gold: total interp {G['total_interp']*100:.2f}% c1 {G['total_c1']*100:.2f}% c1-exact {G['total_c1x']*100:.2f}%; "
    f"booking diff c1 vs exact: max {G['max_abs_monthly_booking_diff_c1_vs_exact']*100:.3f}pp mean {G['mean_abs_monthly_booking_diff']*100:.3f}pp; "
    f"CAGR 88-00 {G['cagr88']} dvol {G['dvol88']}")

# ------------------------------------------------------------------ 4. strategies through common.py
BASECOLS = ["qld", "tqqq", "ief", "tlt", "gold_phys"]
SPEC = {"P0": {}, "A_div": {"qld": SLV[("A_div", "qld")], "tqqq": SLV[("A_div", "tqqq")]},
        "A_lit": {"qld": SLV[("A_lit", "qld")], "tqqq": SLV[("A_lit", "tqqq")]},
        "A_stress": {"qld": SLV[("A_stress", "qld")], "tqqq": SLV[("A_stress", "tqqq")]},
        "B": {"ief": BOND[("B", "ief")], "tlt": BOND[("B", "tlt")]},
        "B_k": {"ief": BOND[("B_k", "ief")], "tlt": BOND[("B_k", "tlt")]},
        "C1": {"gold_phys": c1}, "C2": {"gold_phys": c2}, "C1x": {"gold_phys": c1x}}
SPEC["ALL1"] = {**SPEC["A_div"], **SPEC["B"], **SPEC["C1"]}
SPEC["ALL2"] = {**SPEC["A_div"], **SPEC["B"], **SPEC["C2"]}
SPEC["ALL1x"] = {**SPEC["A_div"], **SPEC["B"], **SPEC["C1x"]}
TIERS = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
CFGS = {"default": {}, "bills": dict(off="bills"), "tlt": dict(off="tlt"), "goldIEF": dict(gold="ief"),
        "nochop": dict(chop_pos=1.0), "novol": dict(voltgt=99.0)}
WKD = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])}
W = {(t, c): C.strategy_weights(P0, **{**TIERS[t], **CFGS[c]}) for t in TIERS for c in CFGS}
WIN = {"IS": (None, FREEZE), "H1": (None, "2006-12-31"), "H2": ("2007-01-01", FREEZE), "since2000": ("2000-01-01", FREEZE)}
SER = {}
STATS = {}
maxdiff = dict(cagr=0.0, maxdd=0.0, calmar=0.0)
worst = None
for pn, rep in SPEC.items():
    Pm = P0.copy()
    for c, s in rep.items():
        Pm[c] = s.values
    for (t, c), w in W.items():
        fri = C.backtest(Pm, w)                                  # P.mark Friday book
        tr = sum(C.backtest(Pm, w, mark=WKD[d]) for d in WKD) / 5.0
        for basis, r in (("Fri", fri), ("tranche", tr)):
            SER[(pn, basis, t, c)] = r
            st = {k: C.stats(r, a, b) for k, (a, b) in WIN.items()}
            st["OOS"] = C.period_return(r, "2026-07-25", "2026-12-31")
            STATS[(pn, basis, t, c)] = st
            if pn in J["stats"]:
                js = J["stats"][pn][basis][f"{t}|{c}"]
                for w_ in ("IS", "H1", "H2", "since2000"):
                    for mtr in ("cagr", "maxdd", "calmar"):
                        dlt = abs(st[w_][mtr] - js[w_][mtr])
                        if dlt > maxdiff[mtr]:
                            maxdiff[mtr] = dlt
                            worst = (pn, basis, t, c, w_, mtr, st[w_][mtr], js[w_][mtr])
    log(f"panel {pn} done")
log(f"max |independent - JSON| over 10 panels x 2 bases x 3 tiers x 6 configs x 4 windows: {maxdiff}  worst {worst}")
OUT["engine_vs_json_maxdiff"] = maxdiff
for b in ("Fri", "tranche"):
    print(f"  --- {b} (independent engine, IS) value (delta vs P0) ---")
    for t in TIERS:
        z = STATS[("P0", b, t, "default")]["IS"]
        line = f"  {t} P0 {z['cagr']*100:.2f}/{z['maxdd']*100:.1f}/{z['calmar']:.3f}"
        for pn in ("A_div", "A_lit", "A_stress", "B", "B_k", "C1", "C1x", "C2", "ALL1", "ALL1x", "ALL2"):
            s = STATS[(pn, b, t, "default")]["IS"]
            line += (f" | {pn} {s['cagr']*100:.2f}({(s['cagr']-z['cagr'])*100:+.2f})/{s['maxdd']*100:.1f}({(s['maxdd']-z['maxdd'])*100:+.1f})"
                     f"/{s['calmar']:.3f}({s['calmar']-z['calmar']:+.3f})")
        print(line)
    for t in TIERS:
        for pn in ("A_div", "ALL1"):
            print(f"  {b} {t} {pn}: dCAGR H1 {(STATS[(pn,b,t,'default')]['H1']['cagr']-STATS[('P0',b,t,'default')]['H1']['cagr'])*100:+.2f} "
                  f"since2000 {(STATS[(pn,b,t,'default')]['since2000']['cagr']-STATS[('P0',b,t,'default')]['since2000']['cagr'])*100:+.2f} "
                  f"H2 {(STATS[(pn,b,t,'default')]['H2']['cagr']-STATS[('P0',b,t,'default')]['H2']['cagr'])*100:+.2f} "
                  f"OOS diff {(STATS[(pn,b,t,'default')]['OOS']-STATS[('P0',b,t,'default')]['OOS'])*100:+.6f}")


# ------------------------------------------------------------------ 5. non-bootstrap statuses and flips
def sgn(x):
    return int(np.sign(x)) if np.isfinite(x) else 0


def status(pn, b):
    f = {}
    S = lambda t, c, w: STATS[(pn, b, t, c)][w]
    f["R1|f2_shallower"] = bool(S("F2", "default", "IS")["maxdd"] > S("F3", "default", "IS")["maxdd"])
    for t in TIERS:
        for w in ("IS", "H1", "H2"):
            f[f"R2|{t}|{w}"] = sgn(S(t, "default", w)["calmar"] - S(t, "bills", w)["calmar"])
            f[f"R6|{t}|{w}"] = sgn(S(t, "default", w)["calmar"] - S(t, "tlt", w)["calmar"])
        for comp, cn in (("chop_rule", "nochop"), ("vol_target", "novol")):
            d = {w: S(t, "default", w)["calmar"] - S(t, cn, w)["calmar"] for w in ("IS", "H1", "H2")}
            f[f"R4|{t}|{comp}|sign_IS"] = sgn(d["IS"])
            f[f"R4|{t}|{comp}|helps_both_halves"] = bool(d["H1"] > 0 and d["H2"] > 0)
        f[f"R5|{t}|calmar_sign"] = sgn(S(t, "default", "IS")["calmar"] - C.stats(P0["qqq"], b=FREEZE)["calmar"])
        f[f"R5|{t}|cagr_sign"] = sgn(S(t, "default", "IS")["cagr"] - C.stats(P0["qqq"], b=FREEZE)["cagr"])
    return f


FL = {}
mismatch_vs_json_status = []
for b in ("Fri", "tranche"):
    s0 = status("P0", b)
    for k, v in s0.items():
        if J["status_P0"][b][k] != v:
            mismatch_vs_json_status.append((b, k, v, J["status_P0"][b][k]))
    for pn in SPEC:
        if pn == "P0":
            continue
        s1 = status(pn, b)
        FL[f"{pn}|{b}"] = {k: (s0[k], s1[k]) for k in s0 if s0[k] != s1[k]}
OUT["nonboot_flips_independent"] = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in FL.items()}
log(f"P0 status mismatches vs JSON (non-bootstrap items): {mismatch_vs_json_status}")
for k, v in FL.items():
    if v:
        print("  NON-BOOT FLIP", k, v)
log("non-bootstrap flips printed above (none printed = none)")
# R6 detail for the claimed flip and the B/B_k H1 shares
for b in ("Fri", "tranche"):
    for t in TIERS:
        row = []
        for pn in ("P0", "B", "B_k", "ALL1", "ALL2", "C2", "A_div"):
            S = lambda c, w: STATS[(pn, b, t, c)][w]["calmar"]
            row.append(f"{pn} H1 {S('default','H1')-S('tlt','H1'):+.3f} IS {S('default','IS')-S('tlt','IS'):+.3f} H2 {S('default','H2')-S('tlt','H2'):+.3f}")
        print(f"  R6 IEF-TLT {b} {t}: " + " | ".join(row))
# R2 / R4 detail
for b in ("Fri", "tranche"):
    for t in TIERS:
        S = lambda pn, c, w: STATS[(pn, b, t, c)][w]["calmar"]
        print(f"  {b} {t}: R2 IEF-bills IS P0 {S('P0','default','IS')-S('P0','bills','IS'):+.3f} ALL1 {S('ALL1','default','IS')-S('ALL1','bills','IS'):+.3f} "
              f"(H2 {S('ALL1','default','H2')-S('ALL1','bills','H2'):+.3f}) | chop P0 {S('P0','default','IS')-S('P0','nochop','IS'):+.3f} ALL1 "
              f"{S('ALL1','default','IS')-S('ALL1','nochop','IS'):+.3f} (H2 {S('ALL1','default','H2')-S('ALL1','nochop','H2'):+.3f}) | vol P0 "
              f"{S('P0','default','IS')-S('P0','novol','IS'):+.3f} ALL1 {S('ALL1','default','IS')-S('ALL1','novol','IS'):+.3f} "
              f"(H2 {S('ALL1','default','H2')-S('ALL1','novol','H2'):+.3f}) | goldIEF dCal ALL1 {S('ALL1','default','IS')-S('ALL1','goldIEF','IS'):+.3f}")

# save the series needed by the bootstrap check
keep = {f"{pn}|{b}|{t}": SER[(pn, b, t, "default")].values for pn in SPEC for b in ("Fri", "tranche") for t in TIERS}
keep.update({f"{pn}|{b}|{t}|goldIEF": SER[(pn, b, t, "goldIEF")].values for pn in SPEC for b in ("Fri", "tranche") for t in TIERS})
np.savez_compressed(os.path.join(scratch, "verify_series.npz"), idx=IDX.values.astype("int64"), qqq=P0["qqq"].values,
                    rf=P0["rf_d"].values, **keep)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(scratch, "verify_checks.json"), "w"), indent=1)
log("done")
