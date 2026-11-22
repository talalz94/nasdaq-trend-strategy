"""
ROBUSTNESS REVIEW of update_2026_11/overlays.py  --  step 2 (fast): independent re-computation + stress tests.
The paired circular block bootstrap is computed exactly from block sums (same draws and truncation as
common.block_bootstrap_cagr_diff / overlays.boot_series). Cross-checked against a direct 3-D resampling
implementation: identical F3 dollar-shock p-values for all 12 block/seed cases (block 63 seed 0 -> 0.006 =
overlays.json). usage: python update_2026_11/verify_overlays_robustness_fast.py <scratch_dir>

Re-builds the key overlay variants with common.py ONLY (no code from overlays.py) and stresses them:
  A  dollar shock (C7): IS/halves stats, where the max drawdowns sit, block-length / seed sensitivity of
     the paired bootstrap, per-half bootstrap, leave-one-year-out and leave-dot-com-out, extra-lag
     sensitivity, a 1000-draw circular-shift placebo, conditional NDX returns on flagged risk-on days,
     multiple-testing arithmetic.
  B  rate shock (C5/C6): per-half Calmar AND Sharpe signs, OOS returns, post-freeze flag dates.
  C  off-leg IEF vs cash (C3): bootstrap, sub-periods incl. the 2020-26 rising-yield era, leave-crisis-out.
  D  curve rules (C4): per-half Sharpe sign, COVID single-week dependence.
  E  credit / VIX data claims (C8/C9): VWEHX zero-return share, HYG-vs-VWEHX flag agreement, lead/lag.
  F  today's read (C10).
Writes <scratch>/verify_overlays_robustness_fast.json  (scratch dir = argv[1]).
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
scratch = sys.argv[1]
os.makedirs(scratch, exist_ok=True)
P = C.panel()
idx = P.index
FREEZE, START, LAST = C.FREEZE, C.START, idx[-1]
IS = (START, FREEZE)
H1 = (pd.Timestamp("1988-01-01"), pd.Timestamp("2006-12-31"))
H2 = (pd.Timestamp("2007-01-01"), FREEZE)
OOS = (pd.Timestamp("2026-07-25"), LAST)
BASES = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
COMMON = dict(gold="gold_phys", gold_w=0.15, off="ief", chop_thr=0.25, chop_pos=0.5, seasonal=True)
OUT = {}
NRUN = [0]


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def run(base, **kw):
    NRUN[0] += 1
    w = C.strategy_weights(P, **{**COMMON, **BASES[base], **kw})
    return C.backtest(P, w), w


def sub(r, win):
    return r[(r.index >= pd.Timestamp(win[0])) & (r.index <= pd.Timestamp(win[1]))].dropna()


def st(r, win):
    s = C.stats(r, win[0], win[1], start=None)
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()}


def sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def dd_info(r, win):
    x = sub(r, win)
    e = (1 + x).cumprod()
    dd = e / e.cummax() - 1
    t = dd.idxmin()
    p = e.loc[:t].idxmax()
    return dict(maxdd=round(float(dd.min()), 4), peak=str(p.date()), trough=str(t.date()))


def lagk(flag, k):
    return flag.reindex(idx).astype(float).shift(k).fillna(0.0).astype(bool)


def mult(flag, m):
    return dict(extra_scale=pd.Series(np.where(flag.values, m, 1.0), index=idx))


def boot_pair(r1, r2, win, n=2000, block=63, seed=0):
    """paired circular block bootstrap (same draws/truncation as common.block_bootstrap_cagr_diff), computed
    exactly from block sums -> p(diff<=0) for Sharpe and CAGR (r1 - r2). Calmar is not computed here."""
    a = pd.concat([r1, r2], axis=1).dropna()
    a = a[(a.index >= pd.Timestamp(win[0])) & (a.index <= pd.Timestamp(win[1]))].values
    T = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    sts = np.stack([rng.integers(0, T, nb) for _ in range(n)])
    Lr = T - (nb - 1) * block
    aa = np.concatenate([a, a], axis=0)
    base_ix = np.arange(T)

    def bsum(v, L):
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
        return cs[base_ix + L] - cs[base_ix]
    tots = []
    for f in (aa, aa ** 2, np.log1p(aa)):
        full, last = bsum(f, block), bsum(f, Lr)
        tots.append(full[sts[:, :nb - 1]].sum(axis=1) + last[sts[:, nb - 1]])
    s1, s2, sl = tots
    mean = s1 / T
    var = (s2 - T * mean ** 2) / (T - 1)
    sh = mean / np.sqrt(var) * np.sqrt(252)
    g = np.expm1(sl * 252 / T)
    dsh, dg = sh[:, 0] - sh[:, 1], g[:, 0] - g[:, 1]
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return dict(T=T, block=block, seed=seed, p_sharpe=float((dsh <= 0).mean()), p_cagr=float((dg <= 0).mean()),
                p_calmar=None, sh_ci=q(dsh), cagr_ci=q(dg), mean_dsh=round(float(dsh.mean()), 4))


def compare(r, rb, win):
    so, sb = st(r, win), st(rb, win)
    return dict(ov=so, base=sb, d_cagr=round(so["cagr"] - sb["cagr"], 5), d_maxdd=round(so["maxdd"] - sb["maxdd"], 5),
                d_sharpe=round(so["sharpe"] - sb["sharpe"], 5), d_calmar=round(so["calmar"] - sb["calmar"], 5))


# ================================================================== baselines
BASE_R, BASE_W, HELD = {}, {}, {}
for b in BASES:
    BASE_R[b], BASE_W[b] = run(b)
    HELD[b] = C.held_weights(P, BASE_W[b])
OUT["baselines"] = {b: dict(IS=st(BASE_R[b], IS), H1=st(BASE_R[b], H1), H2=st(BASE_R[b], H2),
                            OOS_return=round(C.period_return(BASE_R[b], *OOS), 5),
                            dd_IS=dd_info(BASE_R[b], IS), dd_H1=dd_info(BASE_R[b], H1), dd_H2=dd_info(BASE_R[b], H2))
                    for b in BASES}
for b in BASES:
    log(b, "baseline IS", OUT["baselines"][b]["IS"], "dd", OUT["baselines"][b]["dd_IS"], "OOS", OUT["baselines"][b]["OOS_return"])

# ================================================================== A. dollar shock
dxy126 = P["dxy"] / P["dxy"].shift(126) - 1
usd_raw = dxy126 > 0.05
usd = lagk(usd_raw, 1)
A = {}
R_USD = {}
for b in BASES:
    r, w = run(b, **mult(usd, 0.75))
    R_USD[b] = r
    rb = BASE_R[b]
    rec = dict(IS=compare(r, rb, IS), H1=compare(r, rb, H1), H2=compare(r, rb, H2),
               dd_IS_ov=dd_info(r, IS), dd_H1_ov=dd_info(r, H1), dd_H2_ov=dd_info(r, H2),
               dotcom_diff=round(C.period_return(r, "2000-03-01", "2002-10-31") - C.period_return(rb, "2000-03-01", "2002-10-31"), 4),
               y2022_diff=round(C.period_return(r, "2022-01-01", "2022-12-31") - C.period_return(rb, "2022-01-01", "2022-12-31"), 4),
               flag_days_since_freeze=int(usd_raw[usd_raw.index > FREEZE].sum()))
    # bootstrap sensitivity: block length x seed (full IS)
    grid = []
    for blk in (21, 63, 126, 252):
        for sd in (0, 1, 2):
            bp = boot_pair(r, rb, IS, block=blk, seed=sd)
            grid.append(dict(block=blk, seed=sd, p_sharpe=bp["p_sharpe"], p_cagr=bp["p_cagr"], p_calmar=bp["p_calmar"], sh_ci=bp["sh_ci"]))
    rec["boot_grid_IS"] = grid
    rec["boot_H1"] = boot_pair(r, rb, H1)
    rec["boot_H2"] = boot_pair(r, rb, H2)
    # leave-one-year-out Sharpe difference
    loy = {}
    for y in range(1988, 2027):
        m = (r.index >= IS[0]) & (r.index <= IS[1]) & (r.index.year != y)
        loy[y] = round(sharpe(r[m]) - sharpe(rb[m]), 5)
    full = sharpe(sub(r, IS)) - sharpe(sub(rb, IS))
    rec["d_sharpe_full_recomputed"] = round(full, 5)
    rec["leave_one_year_out"] = dict(min=min(loy.values()), argmin=min(loy, key=loy.get), max=max(loy.values()),
                                     drop_effect_top5=sorted(((y, round(full - v, 5)) for y, v in loy.items()), key=lambda t: -t[1])[:5])
    # leave dot-com (2000-03-01..2002-10-31) out: Sharpe diff and bootstrap on the remaining days
    keep = ~((r.index >= "2000-03-01") & (r.index <= "2002-10-31"))
    r_x, rb_x = r[keep], rb[keep]
    rec["ex_dotcom"] = dict(d_sharpe=round(sharpe(sub(r_x, IS)) - sharpe(sub(rb_x, IS)), 5),
                            boot=boot_pair(r_x, rb_x, IS))
    keep2 = ~((r.index >= "1997-01-01") & (r.index <= "2002-12-31"))
    rec["H1a_1988_1996"] = compare(r, rb, ("1988-01-01", "1996-12-31"))
    rec["H1b_1997_2006"] = compare(r, rb, ("1997-01-01", "2006-12-31"))
    rec["boot_H1a"] = boot_pair(r, rb, ("1988-01-01", "1996-12-31"))
    rec["boot_H1b"] = boot_pair(r, rb, ("1997-01-01", "2006-12-31"))
    rec["ex_1997_2002"] = dict(d_sharpe=round(sharpe(sub(r[keep2], IS)) - sharpe(sub(rb[keep2], IS)), 5),
                               boot=boot_pair(r[keep2], rb[keep2], IS))
    # extra-lag sensitivity
    lagres = []
    for k in (1, 2, 3, 5, 10, 21, 42):
        rk, _ = run(b, **mult(lagk(usd_raw, k), 0.75))
        c = compare(rk, rb, IS)
        lagres.append(dict(lag=k, d_sharpe=c["d_sharpe"], d_calmar=c["d_calmar"], d_cagr=c["d_cagr"],
                           p_sharpe=boot_pair(rk, rb, IS)["p_sharpe"],
                           h1_dsh=compare(rk, rb, H1)["d_sharpe"], h2_dsh=compare(rk, rb, H2)["d_sharpe"]))
    rec["lag_sensitivity"] = lagres
    # yearly contribution (sum of daily return differences, simple)
    dd_ = sub(r - rb, IS)
    yc = dd_.groupby(dd_.index.year).sum()
    rec["yearly_sum_diff_top"] = [(int(y), round(float(v) * 100, 2)) for y, v in yc.sort_values().iloc[:4].items()] + \
                                 [(int(y), round(float(v) * 100, 2)) for y, v in yc.sort_values().iloc[-6:].items()]
    rec["sum_diff_H1_pct"] = round(float(sub(r - rb, H1).sum() * 100), 2)
    rec["sum_diff_H2_pct"] = round(float(sub(r - rb, H2).sum() * 100), 2)
    A[b] = rec
    log(f"DOLLAR {b}: IS {rec['IS']['d_calmar']:+.4f} cal {rec['IS']['d_sharpe']:+.4f} sh | H1 {rec['H1']['d_calmar']:+.4f}/{rec['H1']['d_sharpe']:+.4f} "
        f"H2 {rec['H2']['d_calmar']:+.4f}/{rec['H2']['d_sharpe']:+.4f} | ddIS ov {rec['dd_IS_ov']} | pH1 {rec['boot_H1']['p_sharpe']:.3f} pH2 {rec['boot_H2']['p_sharpe']:.3f}")
    log(f"   boot grid p_sharpe: {[(g['block'], g['seed'], g['p_sharpe']) for g in grid]}")
    log(f"   ex-dotcom dSh {rec['ex_dotcom']['d_sharpe']:+.4f} p {rec['ex_dotcom']['boot']['p_sharpe']:.3f} | ex-1997-2002 dSh {rec['ex_1997_2002']['d_sharpe']:+.4f} p {rec['ex_1997_2002']['boot']['p_sharpe']:.3f}")
    log(f"   LOYO {rec['leave_one_year_out']}")
    log(f"   H1a 1988-96 dSh {rec['H1a_1988_1996']['d_sharpe']:+.4f} p {rec['boot_H1a']['p_sharpe']:.3f} | H1b 1997-2006 dSh {rec['H1b_1997_2006']['d_sharpe']:+.4f} p {rec['boot_H1b']['p_sharpe']:.3f}")
    log(f"   lags {[(x['lag'], x['d_sharpe'], x['p_sharpe'], x['d_calmar']) for x in lagres]}")
    log(f"   yearly {rec['yearly_sum_diff_top']}  sum H1 {rec['sum_diff_H1_pct']}%  H2 {rec['sum_diff_H2_pct']}%")

# economic content: NDX returns on held-risk-on days, flag on vs off (flag aligned with the weights actually held)
held_flag = usd.where(P["mark"]).ffill().fillna(False).astype(bool).shift(1).fillna(False).astype(bool)
eco = {}
for b in BASES:
    lev = BASES[b]["lev"]
    on = HELD[b][lev] > 0
    for name, win in (("IS", IS), ("H1", H1), ("H2", H2)):
        m = on & (idx >= win[0]) & (idx <= win[1])
        x1 = P["ndx_ret"][m & held_flag]
        x0 = P["ndx_ret"][m & ~held_flag]
        eco[f"{b}_{name}"] = dict(days_flag=int(len(x1)), days_noflag=int(len(x0)),
                                  ann_mean_flag=round(float(x1.mean() * 252), 4), ann_mean_noflag=round(float(x0.mean() * 252), 4),
                                  ann_vol_flag=round(float(x1.std() * np.sqrt(252)), 4), ann_vol_noflag=round(float(x0.std() * np.sqrt(252)), 4),
                                  sharpe_flag=round(sharpe(x1), 3), sharpe_noflag=round(sharpe(x0), 3))
A["economic_ndx_on_riskon_days"] = eco
log("ECON (NDX on held-risk-on days):", json.dumps(eco))

# 1000-draw circular-shift placebo (same construction as overlays.placebo, different seed / more draws)
def placebo(b, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    a = usd_raw.reindex(idx).fillna(False).values
    T = len(a)
    sb = C.stats(BASE_R[b], *IS, start=None)
    out = []
    for _ in range(n):
        sh = int(rng.integers(252, T - 252))
        f = lagk(pd.Series(np.roll(a, sh), index=idx), 1)
        r, _ = run(b, **mult(f, 0.75))
        s = C.stats(r, *IS, start=None)
        out.append((s["sharpe"] - sb["sharpe"], s["calmar"] - sb["calmar"]))
    return np.array(out)


for b in BASES:
    pl = placebo(b)
    real = A[b]["IS"]
    A[b]["placebo1000"] = dict(p_sharpe=float((pl[:, 0] >= real["d_sharpe"]).mean()), p_calmar=float((pl[:, 1] >= real["d_calmar"]).mean()),
                               sharpe_pct_5_50_95_99=[round(float(np.percentile(pl[:, 0], q)), 4) for q in (5, 50, 95, 99)])
    log(f"PLACEBO1000 {b}: {A[b]['placebo1000']}")
# multiple testing arithmetic on the reported p-values
mt = {}
for b, p in (("F3", 0.006), ("F2", 0.001)):
    mt[b] = dict(p=p, sidak_27=round(1 - (1 - p) ** 27, 4), sidak_54=round(1 - (1 - p) ** 54, 4), bonf_27=round(min(1, p * 27), 4))
A["multiple_testing"] = mt
OUT["A_dollar"] = A

# ================================================================== B. rate shock
d126 = P["tnx"] - P["tnx"].shift(126)
B = {}
for b in BASES:
    rb = BASE_R[b]
    for bp in (50, 75, 100):
        raw = d126 > bp / 100
        f = lagk(raw, 1)
        r, _ = run(b, **mult(f, 0.5))
        c_is, c1, c2 = compare(r, rb, IS), compare(r, rb, H1), compare(r, rb, H2)
        B[f"{b}_x05_{bp}"] = dict(IS_dcal=c_is["d_calmar"], IS_dsh=c_is["d_sharpe"], H1_dcal=c1["d_calmar"], H1_dsh=c1["d_sharpe"],
                                  H2_dcal=c2["d_calmar"], H2_dsh=c2["d_sharpe"],
                                  positive_both_halves_calmar=bool(c1["d_calmar"] > 0 and c2["d_calmar"] > 0),
                                  positive_both_halves_sharpe=bool(c1["d_sharpe"] > 0 and c2["d_sharpe"] > 0),
                                  oos_ov=round(C.period_return(r, *OOS), 4), oos_base=round(C.period_return(rb, *OOS), 4),
                                  p_sharpe=boot_pair(r, rb, IS)["p_sharpe"])
        log(f"RATE {b} {bp}bp x0.5: {B[f'{b}_x05_{bp}']}")
post = d126[d126.index > FREEZE]
B["post_freeze_d126"] = dict(first_over_50=str(post[post > 0.5].index[0].date()), last_over_50=str(post[post > 0.5].index[-1].date()),
                             peak_bp=round(float(post.max() * 100), 1), peak_date=str(post.idxmax().date()),
                             first_over_75=str(post[post > 0.75].index[0].date()), last_over_75=str(post[post > 0.75].index[-1].date()),
                             days_over_50=int((post > 0.5).sum()), n_days=int(len(post)), today_bp=round(float(post.iloc[-1] * 100), 1))
log("RATE post-freeze", B["post_freeze_d126"])
OUT["B_rate"] = B

# ================================================================== C. off-leg IEF vs cash
Cx = {}
for b in BASES:
    rb = BASE_R[b]
    rc, _ = run(b, off="bills")
    rt, _ = run(b, off="tlt")
    rec = dict(IS_cash=st(rc, IS), IS_ief=st(rb, IS), IS_tlt=st(rt, IS),
               H1=compare(rb, rc, H1), H2=compare(rb, rc, H2), boot_ief_minus_cash=boot_pair(rb, rc, IS),
               oos_cash=round(C.period_return(rc, *OOS), 4), oos_ief=round(C.period_return(rb, *OOS), 4))
    subp = {}
    for nm, win in (("1988-1999", ("1988-01-01", "1999-12-31")), ("2000-2009", ("2000-01-01", "2009-12-31")),
                    ("2010-2019", ("2010-01-01", "2019-12-31")), ("2020-01..freeze", ("2020-01-01", FREEZE)),
                    ("2020-08..freeze (yields 0.5%->4.4%)", ("2020-08-01", FREEZE)), ("2021..freeze", ("2021-01-01", FREEZE))):
        c = compare(rb, rc, win)
        subp[nm] = dict(d_cagr=c["d_cagr"], d_sharpe=c["d_sharpe"], d_calmar=c["d_calmar"], d_maxdd=c["d_maxdd"])
    rec["subperiods_ief_minus_cash"] = subp
    yrs = {}
    for y in range(1988, 2027):
        yrs[y] = round(C.period_return(rb, f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), FREEZE)) -
                       C.period_return(rc, f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), FREEZE)), 4)
    rec["years_ief_worse"] = int(sum(v < 0 for v in yrs.values()))
    rec["worst_years"] = sorted(yrs.items(), key=lambda t: t[1])[:5]
    rec["best_years"] = sorted(yrs.items(), key=lambda t: -t[1])[:5]
    # leave-crisis-out geometric d-CAGR
    def gm(x):
        return float(np.prod(1 + x) ** (252 / len(x)) - 1)
    lco = {}
    for nm, (a0, b0) in {"dot-com": ("2000-03-01", "2002-10-31"), "GFC": ("2007-10-01", "2009-03-31"),
                         "COVID": ("2020-02-15", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31")}.items():
        keep = ~((rb.index >= a0) & (rb.index <= b0))
        x1, x0 = sub(rb[keep], IS), sub(rc[keep], IS)
        lco[nm] = round(gm(x1.values) - gm(x0.values), 5)
    keep = np.ones(len(rb), bool)
    for (a0, b0) in (("2000-03-01", "2002-10-31"), ("2007-10-01", "2009-03-31"), ("2020-02-15", "2020-04-30")):
        keep &= ~((rb.index >= a0) & (rb.index <= b0))
    lco["ex dot-com+GFC+COVID"] = round(gm(sub(rb[keep], IS).values) - gm(sub(rc[keep], IS).values), 5)
    lco["full"] = round(gm(sub(rb, IS).values) - gm(sub(rc, IS).values), 5)
    rec["leave_crisis_out_dcagr"] = lco
    Cx[b] = rec
    log(f"OFFLEG {b}: cash {rec['IS_cash']['cagr']}/{rec['IS_cash']['maxdd']}/{rec['IS_cash']['calmar']}  ief {rec['IS_ief']['cagr']}/{rec['IS_ief']['calmar']}  "
        f"tlt {rec['IS_tlt']['cagr']}/{rec['IS_tlt']['maxdd']}/{rec['IS_tlt']['calmar']} | boot p cagr {rec['boot_ief_minus_cash']['p_cagr']:.4f} "
        f"p sh {rec['boot_ief_minus_cash']['p_sharpe']:.4f} | OOS cash {rec['oos_cash']} ief {rec['oos_ief']}")
    log(f"   subperiods {json.dumps(subp)}")
    log(f"   years IEF worse {rec['years_ief_worse']}/39 worst {rec['worst_years']} best {rec['best_years']} | LCO {lco}")
# IEF minus bills total return over the windows (instrument level)
inst = {}
for nm, win in (("IS", IS), ("1988-2006", H1), ("2007-freeze", H2), ("2020-08..freeze", ("2020-08-01", FREEZE)), ("OOS", OOS)):
    x = sub(P["ief"], win); y = sub(P["bills"], win)
    inst[nm] = dict(ief_cagr=round(float((1 + x).prod() ** (252 / len(x)) - 1), 4), bills_cagr=round(float((1 + y).prod() ** (252 / len(y)) - 1), 4))
Cx["instrument_ief_vs_bills"] = inst
tn = P["tnx"]
Cx["tnx_path"] = dict(start=float(tn.loc[tn.index >= START].iloc[0]), freeze=float(tn.loc[:FREEZE].iloc[-1]),
                      min=float(tn.loc[START:FREEZE].min()), min_date=str(tn.loc[START:FREEZE].idxmin().date()))
log("IEF vs bills instrument", inst, Cx["tnx_path"])
OUT["C_offleg"] = Cx

# ================================================================== D. curve rules (sharpe signs, covid week)
curve = P["curve"]
inv = lagk((curve < 0).fillna(False), 1)
D = {}
for b in BASES:
    rb = BASE_R[b]
    r, _ = run(b, risk_on=P["above"].astype(bool) & ~inv)
    c1, c2 = compare(r, rb, H1), compare(r, rb, H2)
    covid_week = C.period_return(r, "2020-02-24", "2020-02-28") - C.period_return(rb, "2020-02-24", "2020-02-28")
    d23 = C.period_return(r, "2023-03-20", "2023-10-20") - C.period_return(rb, "2023-03-20", "2023-10-20")
    D[f"{b}_raw_riskoff"] = dict(IS=compare(r, rb, IS)["d_calmar"], H1_dcal=c1["d_calmar"], H1_dsh=c1["d_sharpe"], H2_dcal=c2["d_calmar"],
                                 H2_dsh=c2["d_sharpe"], covid_week_diff=round(covid_week, 4), mar_oct_2023_diff=round(d23, 4),
                                 inverted_days_feb_mar_2020=int(((curve < 0) & (idx >= "2020-02-01") & (idx <= "2020-03-31")).sum()))
    log(f"CURVE {b} raw risk-off: {D[f'{b}_raw_riskoff']}")
OUT["D_curve"] = D

# ================================================================== E. credit / VIX data claims
EXT = pd.read_parquet(os.path.join(HERE, "overlays_ext.parquet"))
X = EXT[EXT.index <= LAST].reindex(EXT[EXT.index <= LAST].index.union(idx)).ffill().reindex(idx)
vw = EXT["vwehx"].dropna()
E = dict(vwehx_zero_ret_share_own_calendar=round(float((vw.pct_change().dropna() == 0).mean()), 4),
         vwehx_zero_ret_share_since_1988=round(float((vw[vw.index >= "1988-01-01"].pct_change().dropna() == 0).mean()), 4),
         vwehx_zero_ret_share_since_2007=round(float((vw[vw.index >= "2007-01-01"].pct_change().dropna() == 0).mean()), 4))
r_hy = X["hyg"] / X["ief_px"]
ief_tr = (1 + P["ief"]).cumprod()
r_vw = X["vwehx"] / ief_tr
hy_ma, vw_ma = r_hy.rolling(50).mean(), r_vw.rolling(50).mean()
both = hy_ma.notna() & vw_ma.notna() & (idx >= "2007-01-01")
hy50, vw50 = (r_hy < hy_ma), (r_vw < vw_ma)
E["agree_hy50_vw50_since_2007"] = round(float((hy50[both] == vw50[both]).mean()), 4)
E["agree_days"] = int(both.sum())
# coincidence: P(flag on | NDX below its 50d SMA) vs above, and forward 21d NDX return when trend gate on
ndx = P["ndx"]
below50 = ndx < ndx.rolling(50).mean()
m = both & (idx <= FREEZE)
E["p_hy50_given_ndx_below50"] = round(float(hy50[m & below50].mean()), 3)
E["p_hy50_given_ndx_above50"] = round(float(hy50[m & ~below50].mean()), 3)
fwd21 = ndx.shift(-21) / ndx - 1
gate = P["above"].astype(bool) & m
E["fwd21_ndx_gate_on_hy50_on"] = round(float(fwd21[gate & hy50].mean()), 4)
E["fwd21_ndx_gate_on_hy50_off"] = round(float(fwd21[gate & ~hy50].mean()), 4)
# VIX3M coverage
E["vix3m_first"] = str(EXT["vix3m"].dropna().index[0].date())
log("CREDIT/VIX data", E)
OUT["E_credit_vix"] = E

# ================================================================== F. today
L = LAST
vts = X["vix_y"] / X["vix3m"]
d_back = idx[idx.get_loc(L) - 126]
inv_raw = (curve < 0).fillna(False)
hyflag = (r_hy < hy_ma)
on_since = None
if bool(hyflag.loc[L]):
    s = hyflag[::-1]
    k = int(np.argmax(~s.values))      # first False going backwards
    on_since = str(s.index[k - 1].date())
F = dict(date=str(L.date()), ndx=round(float(P["ndx"].loc[L]), 1), ema100=round(float(P["ema100"].loc[L]), 1),
         pct_vs_ema=round(float(P["ndx"].loc[L] / P["ema100"].loc[L] - 1), 4), above=bool(P["above"].loc[L]),
         held_F3={k: round(float(v), 4) for k, v in HELD["F3"].loc[L].items() if abs(v) > 1e-9},
         held_F2={k: round(float(v), 4) for k, v in HELD["F2"].loc[L].items() if abs(v) > 1e-9},
         tnx=float(P["tnx"].loc[L]), irx=float(P["irx"].loc[L]), curve=round(float(curve.loc[L]), 3),
         last_inverted=str(inv_raw[inv_raw].index[-1].date()),
         tnx_126_ago=float(P["tnx"].loc[d_back]), d126_bp=round(float(d126.loc[L] * 100), 1),
         dxy=round(float(P["dxy"].loc[L]), 2), dxy_126_ago=round(float(P["dxy"].loc[d_back]), 2), dxy126_pct=round(float(dxy126.loc[L] * 100), 2),
         hyg_ief_vs_50d_pct=round(float((r_hy.loc[L] / hy_ma.loc[L] - 1) * 100), 3), hy50_on_since=on_since,
         hy200=bool((r_hy < r_hy.rolling(200).mean()).loc[L]),
         lq50=bool(((X["lqd"] / X["ief_px"]) < (X["lqd"] / X["ief_px"]).rolling(50).mean()).loc[L]),
         vw50=bool((r_vw < vw_ma).loc[L]), vw200=bool((r_vw < r_vw.rolling(200).mean()).loc[L]),
         vix=round(float(X["vix_y"].loc[L]), 2), vix3m=round(float(X["vix3m"].loc[L]), 2),
         vts_5d=round(float(vts.rolling(5).mean().loc[L]), 4), backw_days_last20=int((vts > 1).iloc[-20:].sum()))
log("TODAY", F)
OUT["F_today"] = F
OUT["n_backtests_in_this_check"] = NRUN[0]


def jz(o):
    if isinstance(o, dict):
        return {str(k): jz(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jz(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


with open(os.path.join(scratch, "verify_overlays_robustness_fast.json"), "w") as fh:
    json.dump(jz(OUT), fh, indent=1)
log(f"done; backtests {NRUN[0]}")
