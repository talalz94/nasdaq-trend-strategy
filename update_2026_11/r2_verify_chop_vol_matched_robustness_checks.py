"""
r2_verify_chop_vol_matched_robustness_checks.py -- adversarial ROBUSTNESS review of r2_chop_vol_matched.

Independent numpy engine (validated against common.py / main_tranche_check.json), then FRAGILITY checks of
the four pre-registered decisions. Nothing here changes a decision: the pre-registered rule as written governs.
  1. period-boundary sweep (H1/H2 split year 2003..2011; alternatives matched over the full IS, as registered)
  2. single-episode exclusion (both B and the alternative set to zero return inside the episode window)
  3. start date 1986-07-01 (adds the 1987 crash to H1) - original match and re-matched
  4. lag 2 (common.py's conservative execution check) - re-matched, NDX + cross-market
  5. volatility-matched alternatives (match IS realised vol instead of average exposure) - NDX + cross-market
  6. bootstrap design: block length 21..504, seeds, bias (boot mean - point), basic-bootstrap p-value
  7. cross-market alternatives: Friday basis, start 1988 where data allow
  8. per-market sub-period chop differences (1990-99 / 2000-12 / 2013-freeze)
Writes nothing except its own JSON in the scratch path given as argv[1].
usage: python update_2026_11/r2_verify_chop_vol_matched_robustness_checks.py <scratch_out.json>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUTP = os.path.abspath(sys.argv[1])
assert "update_2026_11" not in OUTP.replace("\\", "/").split("/")[-2:][0]
P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
DOW = IDX.dayofweek.values
MARKS = [DOW == i for i in range(5)]
LASTS = [np.maximum.accumulate(np.where(m, np.arange(N), -1)) for m in MARKS]
SEASON = np.where(np.isin(P["month"].values, (8, 9)), 0.5, 1.0)
R_IEF = P["ief"].fillna(0).values
R_GOLD = P["gold_phys"].fillna(0).values
RF_D = P["rf_d"].fillna(0).values
COST = 3e-4
SPEC = json.load(open(os.path.join(HERE, "r2_chop_vol_matched.json")))
REG_MED = {"chop": SPEC["cross_market"]["median_dcalmar"]["chop"]["median"],
           "vol": SPEC["cross_market"]["median_dcalmar"]["vol"]["median"]}
OUT = {}


def log(*a):
    print(*a, flush=True)


def msk(a, b):
    return (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))


def market(mult, r_lev, above, vol20, er20):
    er20 = np.asarray(er20, float)
    return dict(mult=float(mult), r_lev=np.nan_to_num(np.asarray(r_lev, float)), above=np.asarray(above, bool),
                vol20=np.asarray(vol20, float), clean=(~np.isnan(er20)) & (er20 >= 0.25))


def wlev(M, vt=None, chop=True, vol=True, c=1.0):
    if vol:
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.clip(vt / M["vol20"], 0, 1)
        sc = np.where(np.isnan(sc), 1.0, sc)
    else:
        sc = np.ones(N)
    sc = np.clip(np.clip(sc * SEASON, 0, 1) * c, 0, 1)
    if chop:
        eq = np.where(M["above"], np.where(M["clean"], sc, 0.5 * sc), 0.0)
    else:
        eq = np.where(M["above"], sc, 0.0)
    return 0.85 * eq


def held(w, li, lag):
    h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
    out = np.zeros(N)
    out[lag:] = h[:-lag]
    return out


def books(basis):
    return list(range(5)) if basis == "tranche" else [4]


def ret(M, wl, basis="tranche", lag=1, cost=COST):
    bk = books(basis)
    r = np.zeros(N)
    wo = np.clip(0.85 - wl, 0, None)
    wg = np.full(N, 0.15)
    for d in bk:
        li = LASTS[d]
        hl, ho, hg = held(wl, li, lag), held(wo, li, lag), held(wg, li, lag)
        g = hl * M["r_lev"] + ho * R_IEF + hg * R_GOLD
        tn = (np.abs(np.diff(hl, prepend=0.0)) + np.abs(np.diff(ho, prepend=0.0)) + np.abs(np.diff(hg, prepend=0.0)))
        r += (g - tn * cost) / len(bk)
    return r


def expo(M, wl, basis, a, b, lag=1):
    h = sum(held(wl, LASTS[d], lag) for d in books(basis)) / len(books(basis))
    return float(M["mult"] * h[msk(a, b)].mean())


def calmar(r, a, b):
    m = msk(a, b)
    rr = r[m]
    d = IDX[m]
    e = np.cumprod(1 + rr)
    yrs = (d[-1] - d[0]).days / 365.25
    cg = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return float(cg / abs(dd))


def bisect(f, lo, hi, tol=1e-10, it=200):
    flo, fhi = f(lo), f(hi)
    assert flo < 0 <= fhi, (flo, fhi)
    k = 0
    while hi - lo > tol and k < it:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        k += 1
    return 0.5 * (lo + hi)


def match_expo(M, vt_b, basis, a, b, lag=1):
    tgt = expo(M, wlev(M, vt_b), basis, a, b, lag)
    vtA = bisect(lambda v: expo(M, wlev(M, v, chop=False), basis, a, b, lag) - tgt, 1e-4, vt_b)
    cV = tgt / expo(M, wlev(M, None, vol=False), basis, a, b, lag)
    return vtA, cV


def series(M, vt_b, vtA, cV, basis="tranche", lag=1, cost=COST):
    return dict(B=ret(M, wlev(M, vt_b), basis, lag, cost), A=ret(M, wlev(M, vtA, chop=False), basis, lag, cost),
                V=ret(M, wlev(M, None, vol=False, c=cV), basis, lag, cost))


def decide(h1, h2, med):
    both = h1 >= 0.03 and h2 >= 0.03
    if both and med >= 0.02:
        return "RETAINED"
    if (not both) and med <= 0.02:
        return "REPLACEABLE"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ NDX markets + validation
NDXM = {"F2": market(2, P["qld"].values, P["above"].values, P["vol20"].values, P["er20"].values),
        "F3": market(3, P["tqqq"].values, P["above"].values, P["vol20"].values, P["er20"].values)}
VT = {"F2": 0.30, "F3": 0.20}
H1 = (START, pd.Timestamp("2006-12-31"))
H2 = (pd.Timestamp("2007-01-01"), FREEZE)
IS = (START, FREEZE)
ref = json.load(open(os.path.join(HERE, "main_tranche_check.json")))
val = {}
BASE = {}
for t in ["F2", "F3"]:
    M = NDXM[t]
    rB = ret(M, wlev(M, VT[t]))
    val[f"{t}_tranche_calmar_mine_minus_main_tranche_check"] = calmar(rB, *IS) - ref[f"{t}-IEF"]["tranche"]["to_freeze"]["calmar"]
    vtA, cV = match_expo(M, VT[t], "tranche", *IS)
    sp = SPEC["ndx"][t]["tranche"]
    val[f"{t}_vtA_mine_minus_spec"] = vtA - sp["solve_A"]["vt"]
    val[f"{t}_cV_mine_minus_spec"] = cV - sp["solve_V"]["c"]
    S = series(M, VT[t], vtA, cV)
    BASE[t] = dict(vtA=vtA, cV=cV, S=S)
    for comp, x in [("chop", "A"), ("vol", "V")]:
        for w, ab in [("H1", H1), ("H2", H2), ("IS", IS)]:
            mine = calmar(S["B"], *ab) - calmar(S[x], *ab)
            val[f"{t}_{comp}_{w}_dcal_mine_minus_spec"] = mine - sp["dcalmar_B_minus_alt"][comp]["3bps"][w]
OUT["validation"] = val
log("validation max |diff|:", max(abs(v) for v in val.values()))
assert max(abs(v) for v in val.values()) < 1e-8

# ------------------------------------------------------------------ 1. period-boundary sweep
log("\n== 1. boundary sweep (registered full-IS match; registered cross-market median)")
sweep = {}
for t in ["F2", "F3"]:
    S = BASE[t]["S"]
    for Y in range(2003, 2012):
        h1 = (START, pd.Timestamp(f"{Y-1}-12-31"))
        h2 = (pd.Timestamp(f"{Y}-01-01"), FREEZE)
        row = {}
        for comp, x in [("chop", "A"), ("vol", "V")]:
            d1 = calmar(S["B"], *h1) - calmar(S[x], *h1)
            d2 = calmar(S["B"], *h2) - calmar(S[x], *h2)
            row[comp] = dict(H1=d1, H2=d2, decision=decide(d1, d2, REG_MED[comp]))
        sweep[f"{t}|{Y}"] = row
        log(f"  {t} split {Y}: chop H1 {row['chop']['H1']:+.3f} H2 {row['chop']['H2']:+.3f} -> {row['chop']['decision']:12s} | "
            f"vol H1 {row['vol']['H1']:+.3f} H2 {row['vol']['H2']:+.3f} -> {row['vol']['decision']}")
OUT["boundary_sweep"] = sweep
cnt = {}
for t in ["F2", "F3"]:
    for comp in ["chop", "vol"]:
        ds = [sweep[f"{t}|{Y}"][comp]["decision"] for Y in range(2003, 2012)]
        cnt[f"{comp}|{t}"] = {k: ds.count(k) for k in set(ds)}
        cnt[f"{comp}|{t}|2005-2009"] = {k: [sweep[f"{t}|{Y}"][comp]["decision"] for Y in range(2005, 2010)].count(k)
                                        for k in set(ds)}
OUT["boundary_sweep_counts"] = cnt
log("  counts:", cnt)

# ------------------------------------------------------------------ 2. single-episode exclusion
log("\n== 2. episode exclusion (both legs zeroed inside the window)")
EPIS = {"1987-crash(n/a in IS)": None,
        "1989-90": ("1989-07-01", "1990-12-31"), "1998 LTCM": ("1998-07-01", "1998-12-31"),
        "2000 top (Mar-Dec)": ("2000-03-01", "2000-12-31"), "2000-02 bear": ("2000-01-01", "2002-12-31"),
        "GFC 2007-09": ("2007-10-01", "2009-03-31"), "2018": ("2018-01-01", "2018-12-31"),
        "COVID 2020": ("2020-02-15", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31")}
exc = {}
for t in ["F2", "F3"]:
    S = BASE[t]["S"]
    for ep, win in EPIS.items():
        if win is None:
            continue
        z = msk(*win)
        Sx = {k: np.where(z, 0.0, v) for k, v in S.items()}
        row = {}
        for comp, x in [("chop", "A"), ("vol", "V")]:
            d1 = calmar(Sx["B"], *H1) - calmar(Sx[x], *H1)
            d2 = calmar(Sx["B"], *H2) - calmar(Sx[x], *H2)
            di = calmar(Sx["B"], *IS) - calmar(Sx[x], *IS)
            row[comp] = dict(H1=d1, H2=d2, IS=di, decision=decide(d1, d2, REG_MED[comp]))
        exc[f"{t}|{ep}"] = row
        log(f"  {t} excl {ep:20s}: chop H1 {row['chop']['H1']:+.3f} H2 {row['chop']['H2']:+.3f} IS {row['chop']['IS']:+.3f} -> {row['chop']['decision']:12s}"
            f" | vol H1 {row['vol']['H1']:+.3f} H2 {row['vol']['H2']:+.3f} IS {row['vol']['IS']:+.3f} -> {row['vol']['decision']}")
OUT["episode_exclusion"] = exc

# ------------------------------------------------------------------ 3. start 1986-07-01 (adds the 1987 crash)
log("\n== 3. start 1986-07-01 (includes Oct-1987)")
s87 = {}
A86 = pd.Timestamp("1986-07-01")
for t in ["F2", "F3"]:
    M = NDXM[t]
    S = BASE[t]["S"]
    vtA2, cV2 = match_expo(M, VT[t], "tranche", A86, FREEZE)
    S2 = series(M, VT[t], vtA2, cV2)
    row = {}
    for lab, SS in [("original_match", S), ("rematched_1986", S2)]:
        for comp, x in [("chop", "A"), ("vol", "V")]:
            d1 = calmar(SS["B"], A86, H1[1]) - calmar(SS[x], A86, H1[1])
            d2 = calmar(SS["B"], *H2) - calmar(SS[x], *H2)
            row[f"{lab}|{comp}"] = dict(H1_1986=d1, H2=d2, decision=decide(d1, d2, REG_MED[comp]),
                                        B_maxdd_window=None)
        row[f"{lab}|calmar_H1_1986"] = {k: calmar(v, A86, H1[1]) for k, v in SS.items()}
    row["vtA_1986"], row["cV_1986"] = vtA2, cV2
    # 1987 episode on its own
    row["oct1987_total_return"] = {k: float(np.prod(1 + v[msk("1987-10-01", "1987-12-31")]) - 1) for k, v in S.items()}
    s87[t] = row
    log(f"  {t}: orig-match H1' chop {row['original_match|chop']['H1_1986']:+.3f} vol {row['original_match|vol']['H1_1986']:+.3f} | "
        f"re-match chop {row['rematched_1986|chop']['H1_1986']:+.3f} vol {row['rematched_1986|vol']['H1_1986']:+.3f} | "
        f"Q4-1987 total B {row['oct1987_total_return']['B']*100:+.1f}% A {row['oct1987_total_return']['A']*100:+.1f}% "
        f"V {row['oct1987_total_return']['V']*100:+.1f}%")
OUT["start_1986"] = s87

# ------------------------------------------------------------------ cross-market markets (as the specialist builds them)
yfd = pd.read_parquet(os.path.join(HERE, "r2_chop_vol_matched_yf.parquet"))
XM = {"spx": ("panel", "spx"), "rut": ("panel", "rut"), "tsx": ("panel", "tsx"), "hsi": ("panel", "hsi"),
      "dax": ("yfinance", "^GDAXI"), "n225": ("yfinance", "^N225"), "ixic": ("yfinance", "^IXIC")}


def mret(src, col):
    if src == "panel":
        r = P[col].values.astype(float)
        first = IDX[np.argmax((r != 0) & ~np.isnan(r))]
        return np.nan_to_num(r), first
    s = yfd[yfd["ticker"] == col]
    c = pd.Series(s["close"].values, index=pd.DatetimeIndex(s["date"])).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    first = c.index[0]
    lv = c.reindex(c.index.union(IDX)).ffill().reindex(IDX)
    r = lv.pct_change().values
    r[IDX <= first] = np.nan
    return np.nan_to_num(r), first


def sig(r):
    price = pd.Series(np.cumprod(1.0 + r), index=IDX)
    ema = price.ewm(span=100, adjust=False).mean()
    v20 = price.pct_change().rolling(20).std() * np.sqrt(252)
    e20 = price.diff(20).abs() / price.diff().abs().rolling(20).sum().replace(0, np.nan)
    return (price > ema).values, v20.values, e20.values


XMK = {}
XFIRST = {}
for nm, (src, col) in XM.items():
    r, first = mret(src, col)
    ab, v20, e20 = sig(r)
    XMK[nm] = market(2, 2 * r - 0.0095 / 252 - RF_D, ab, v20, e20)
    XFIRST[nm] = first


def xm_dcal(basis="tranche", lag=1, a="1990-01-01", match="expo", starts=None):
    out = {}
    for nm, M in XMK.items():
        aa = pd.Timestamp(starts[nm]) if starts else pd.Timestamp(a)
        if match == "expo":
            vtA, cV = match_expo(M, 0.30, basis, aa, FREEZE, lag)
        else:
            vtA, cV = match_vol(M, 0.30, basis, aa, FREEZE, lag)
        S = series(M, 0.30, vtA, cV, basis, lag)
        out[nm] = dict(chop=calmar(S["B"], aa, FREEZE) - calmar(S["A"], aa, FREEZE),
                       vol=calmar(S["B"], aa, FREEZE) - calmar(S["V"], aa, FREEZE),
                       chop_H1=calmar(S["B"], aa, H1[1]) - calmar(S["A"], aa, H1[1]),
                       chop_H2=calmar(S["B"], *H2) - calmar(S["A"], *H2),
                       vtA=vtA, cV=cV, start=str(aa.date()))
    med = {c: float(np.median([out[m][c] for m in out])) for c in ["chop", "vol", "chop_H1", "chop_H2"]}
    return out, med


def ann_vol(r, a, b):
    return float(r[msk(a, b)].std(ddof=1) * np.sqrt(252))


def match_vol(M, vt_b, basis, a, b, lag=1):
    tgt = ann_vol(ret(M, wlev(M, vt_b), basis, lag), a, b)
    vtA = bisect(lambda v: ann_vol(ret(M, wlev(M, v, chop=False), basis, lag), a, b) - tgt, 1e-4, vt_b, tol=1e-9)
    hiC = 1.0
    f = lambda c: ann_vol(ret(M, wlev(M, None, vol=False, c=c), basis, lag), a, b) - tgt  # noqa: E731
    while f(hiC) < 0 and hiC < 3:
        hiC *= 1.2
    cV = bisect(f, 1e-3, hiC, tol=1e-9)
    return vtA, cV


# validation of the cross-market rebuild against the specialist
xm0, med0 = xm_dcal()
OUT["xm_rebuild_check"] = dict(chop_median_mine=med0["chop"], vol_median_mine=med0["vol"],
                              chop_median_spec=REG_MED["chop"], vol_median_spec=REG_MED["vol"],
                              per_market_chop_diff={m: xm0[m]["chop"] - SPEC["cross_market"]["markets"][m]["dcalmar_B_minus_alt"]["chop"]["FULL"] for m in xm0})
log(f"\ncross-market rebuild: chop median {med0['chop']:+.5f} (spec {REG_MED['chop']:+.5f}) vol {med0['vol']:+.5f} (spec {REG_MED['vol']:+.5f})")
assert abs(med0["chop"] - REG_MED["chop"]) < 1e-9 and abs(med0["vol"] - REG_MED["vol"]) < 1e-9

# ------------------------------------------------------------------ 4. lag 2 (NDX + cross-market, all re-matched)
log("\n== 4. lag 2")
l2 = {}
xm2, med2 = xm_dcal(lag=2)
for t in ["F2", "F3"]:
    M = NDXM[t]
    vtA, cV = match_expo(M, VT[t], "tranche", *IS, lag=2)
    S = series(M, VT[t], vtA, cV, lag=2)
    for comp, x in [("chop", "A"), ("vol", "V")]:
        d1 = calmar(S["B"], *H1) - calmar(S[x], *H1)
        d2 = calmar(S["B"], *H2) - calmar(S[x], *H2)
        l2[f"{comp}|{t}"] = dict(H1=d1, H2=d2, xm_median=med2[comp], decision=decide(d1, d2, med2[comp]),
                                 decision_with_registered_median=decide(d1, d2, REG_MED[comp]))
        log(f"  {comp} {t}: H1 {d1:+.3f} H2 {d2:+.3f} xm-median(lag2) {med2[comp]:+.4f} -> {l2[f'{comp}|{t}']['decision']}")
l2["xm_per_market"] = xm2
OUT["lag2"] = l2

# ------------------------------------------------------------------ 5. volatility-matched alternatives
log("\n== 5. realised-volatility matched alternatives (instead of average exposure)")
vm = {}
xmv, medv = xm_dcal(match="vol")
for t in ["F2", "F3"]:
    M = NDXM[t]
    vtA, cV = match_vol(M, VT[t], "tranche", *IS)
    S = series(M, VT[t], vtA, cV)
    ex = dict(B=expo(M, wlev(M, VT[t]), "tranche", *IS), A=expo(M, wlev(M, vtA, chop=False), "tranche", *IS),
              V=expo(M, wlev(M, None, vol=False, c=cV), "tranche", *IS))
    for comp, x in [("chop", "A"), ("vol", "V")]:
        d1 = calmar(S["B"], *H1) - calmar(S[x], *H1)
        d2 = calmar(S["B"], *H2) - calmar(S[x], *H2)
        vm[f"{comp}|{t}"] = dict(H1=d1, H2=d2, xm_median=medv[comp], decision=decide(d1, d2, medv[comp]),
                                 vtA=vtA, cV=cV, exposure=ex)
        log(f"  {comp} {t}: vtA {vtA:.4f} cV {cV:.4f} expo B/A/V {ex['B']:.3f}/{ex['A']:.3f}/{ex['V']:.3f} | "
            f"H1 {d1:+.3f} H2 {d2:+.3f} xm-median(vol-matched) {medv[comp]:+.4f} -> {vm[f'{comp}|{t}']['decision']}")
vm["xm_per_market"] = xmv
OUT["vol_matched"] = vm

# ------------------------------------------------------------------ 7. cross-market alternatives
log("\n== 7. cross-market alternative constructions (median FULL dCalmar)")
xalt = {"registered (tranche, lag1, 1990)": med0, "lag 2": med2, "vol-matched": medv}
xmF, medF = xm_dcal(basis="Fri")
xalt["Friday basis"] = medF


def start88(first):
    first = pd.Timestamp(first)
    warm = first + pd.Timedelta(days=365)
    jan = pd.Timestamp(year=warm.year + (0 if (warm.month == 1 and warm.day == 1) else 1), month=1, day=1)
    return max(pd.Timestamp("1988-01-01"), jan)


st88 = {m: start88(XFIRST[m]) for m in XMK}
xm88, med88 = xm_dcal(starts=st88)
xalt["start 1988 where data allow"] = med88
for k, v in xalt.items():
    log(f"  {k:32s}: chop {v['chop']:+.4f} vol {v['vol']:+.4f} | chop H1 {v['chop_H1']:+.4f} H2 {v['chop_H2']:+.4f}")
OUT["xm_alternatives"] = dict(medians=xalt, starts_1988={m: str(v.date()) for m, v in st88.items()},
                              per_market_friday=xmF, per_market_1988=xm88)

# ------------------------------------------------------------------ 8. per-market sub-periods (chop)
log("\n== 8. per-market chop dCalmar by sub-period (registered match)")
SUBS = {"1990-99": ("1990-01-01", "1999-12-31"), "2000-12": ("2000-01-01", "2012-12-31"), "2013-frz": ("2013-01-01", FREEZE)}
sub = {}
for nm, M in list(XMK.items()) + [("NDX-F2", NDXM["F2"]), ("NDX-F3", NDXM["F3"])]:
    if nm.startswith("NDX"):
        t = nm[-2:]
        S = BASE[t]["S"]
    else:
        vtA, cV = match_expo(M, 0.30, "tranche", "1990-01-01", FREEZE)
        S = series(M, 0.30, vtA, cV)
    sub[nm] = {k: dict(chop=calmar(S["B"], *ab) - calmar(S["A"], *ab), vol=calmar(S["B"], *ab) - calmar(S["V"], *ab))
               for k, ab in SUBS.items()}
    log(f"  {nm:7s}: " + " | ".join(f"{k} chop {v['chop']:+.3f} vol {v['vol']:+.3f}" for k, v in sub[nm].items()))
OUT["subperiods"] = sub

# ------------------------------------------------------------------ 6. bootstrap design
log("\n== 6. bootstrap block length / bias (IS tranche dCalmar, paired circular blocks, n=1000)")


def boot(r1, r2, a, b, n=1000, block=63, seed=7):
    m = msk(a, b)
    X = np.column_stack([r1[m], r2[m]])
    T = len(X)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        s0 = rng.integers(0, T, nb)
        ix = ((s0[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        e = np.cumprod(1 + X[ix], axis=0)
        cg = e[-1] ** (252.0 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = cg / np.abs(dd)
        out[i] = cal[0] - cal[1]
    return out


bt = {}
for t in ["F2", "F3"]:
    S = BASE[t]["S"]
    for comp, x in [("chop", "A"), ("vol", "V")]:
        pt = calmar(S["B"], *IS) - calmar(S[x], *IS)
        # point estimate on the bootstrap's own CAGR convention (252-day years)
        m = msk(*IS)
        def cal252(r):
            e = np.cumprod(1 + r[m]); cg = e[-1] ** (252.0 / m.sum()) - 1
            return cg / abs((e / np.maximum.accumulate(e) - 1).min())
        pt252 = cal252(S["B"]) - cal252(S[x])
        for blk, seed in [(63, 7), (63, 1), (63, 2), (21, 7), (126, 7), (252, 7), (504, 7)]:
            o = boot(S["B"], S[x], *IS, block=blk, seed=seed)
            bt[f"{comp}|{t}|b{blk}|s{seed}"] = dict(point=pt, point_252conv=pt252, mean=float(o.mean()), bias=float(o.mean() - pt252),
                                                    p05=float(np.percentile(o, 5)), p95=float(np.percentile(o, 95)),
                                                    p_le_0=float((o <= 0).mean()),
                                                    basic_p=float((o >= 2 * pt252).mean()),
                                                    share_draws_below_point=float((o < pt252).mean()))
            b_ = bt[f"{comp}|{t}|b{blk}|s{seed}"]
            log(f"  {comp:4s} {t} block {blk:3d} seed {seed}: point {pt:+.3f} boot mean {b_['mean']:+.3f} (bias {b_['bias']:+.3f}) "
                f"p05 {b_['p05']:+.3f} p95 {b_['p95']:+.3f} P(<=0) {b_['p_le_0']:.2f} | basic-bootstrap p {b_['basic_p']:.3f} | "
                f"draws below point {b_['share_draws_below_point']:.2f}")
OUT["bootstrap"] = bt

json.dump(OUT, open(OUTP, "w"), indent=1, default=float)
log(f"\nsaved {OUTP} | runtime {time.time()-T0:.0f}s")
