"""
INTEGRITY REVIEW of r2_chop_vol_matched - information-only sensitivities and episode checks (writes nothing in
the project; output JSON to the scratch dir given as argv[1]).
  a) the 16 days where panel 'hsi' != Yahoo ^HSI; hsi row + cross-market median recomputed with ^HSI
  b) pre-2000-08-30 gold (monthly interpolation, intra-month look-ahead): replace by 0 / month-end lump; NDX H1/H2
     and cross-market medians
  c) exposure defined on DAILY TARGET weights (not held weights) -> vt_A, c_V, dCal H1/H2 (F3 vol knife edge)
  d) per-weekday books with exposure re-matched PER BOOK (C4 'Friday is the most favourable weekday')
  e) C3 episode details: F3 V drawdown path 2000-2003, vol dCal by sub-era, B-V return gap by year 2000-2003
  f) bootstrap P(dCal<=0) under other seeds (C8)
Own numpy engine (validated in-script against the specialist's B/A/V tranche Calmars).
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = sys.argv[1]
P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
H1E, H2S = pd.Timestamp("2006-12-31"), pd.Timestamp("2007-01-01")
A90 = pd.Timestamp("1990-01-01")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MKB = {d: IDX.dayofweek.values == i for i, d in enumerate(DAYS)}
SEAS = np.where(IDX.month.isin([8, 9]), 0.5, 1.0)
RF = P["rf_d"].fillna(0).values
IEF = P["ief"].fillna(0).values
S = json.load(open(os.path.join(HERE, "r2_chop_vol_matched.json")))
OUT = {}


def msk(a, b):
    return (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))


def held(w, mb):
    li = np.maximum.accumulate(np.where(mb, np.arange(N), -1))
    h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
    o = np.zeros(N)
    o[1:] = h[:-1]
    return o


def tgt(sig, vt, chop=True, vol=True, c=1.0):
    above, v20, e20 = sig
    if vol:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.minimum(vt / v20, 1.0)
        s = np.where(np.isnan(v20), 1.0, s)
    else:
        s = np.ones(N)
    s = s * SEAS * c
    cm = np.where(np.nan_to_num(e20, nan=-1) >= 0.25, 1.0, 0.5) if chop else 1.0
    return 0.85 * np.where(above, s * cm, 0.0)


def bks(basis):
    return DAYS if basis == "tranche" else ([basis] if basis in DAYS else ["Fri"])


def run(wl, rlev, gold, basis, cost=3e-4):
    k = len(bks(basis))
    r = np.zeros(N)
    hlsum = np.zeros(N)
    for d in bks(basis):
        hl = held(wl, MKB[d])
        ho = held(0.85 - wl, MKB[d])
        hg = held(np.full(N, 0.15), MKB[d])
        turn = np.abs(np.diff(hl, prepend=0)) + np.abs(np.diff(ho, prepend=0)) + np.abs(np.diff(hg, prepend=0))
        r += (hl * rlev + ho * IEF + hg * gold - turn * cost) / k
        hlsum += hl / k
    return r, hlsum


def expo_held(wl, mult, basis, a, b):
    h = sum(held(wl, MKB[d]) for d in bks(basis)) / len(bks(basis))
    return float(mult * h[msk(a, b)].mean())


def expo_target(wl, mult, basis, a, b):
    return float(mult * wl[msk(a, b)].mean())


def st(r, a, b):
    m = msk(a, b)
    rr, d = r[m], IDX[m]
    e = np.cumprod(1 + rr)
    yrs = (d[-1] - d[0]).days / 365.25
    cg = e[-1] ** (1 / yrs) - 1
    dd = e / np.maximum.accumulate(e) - 1
    it = int(np.argmin(dd)); ip = int(np.argmax(e[:it + 1]))
    return dict(cagr=float(cg), maxdd=float(dd.min()), calmar=float(cg / abs(dd.min())),
                peak=str(d[ip].date()), trough=str(d[it].date()), dd=dd, e=e, d=d)


def solve(sig, vt, mult, basis, a, b, expo=expo_held):
    eB = expo(tgt(sig, vt), mult, basis, a, b)
    lo, hi = 1e-4, vt
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expo(tgt(sig, mid, chop=False), mult, basis, a, b) > eB:
            hi = mid
        else:
            lo = mid
    vtA = 0.5 * (lo + hi)
    cV = eB / expo(tgt(sig, None, vol=False), mult, basis, a, b)
    return vtA, cV


def dcal_set(sig, vt, mult, rlev, gold, basis, a, b, wins, expo=expo_held, cost=3e-4, params=None):
    vtA, cV = params if params is not None else solve(sig, vt, mult, basis, a, b, expo)
    rB, _ = run(tgt(sig, vt), rlev, gold, basis, cost)
    rA, _ = run(tgt(sig, vtA, chop=False), rlev, gold, basis, cost)
    rV, _ = run(tgt(sig, None, vol=False, c=cV), rlev, gold, basis, cost)
    o = dict(vtA=vtA, cV=cV)
    for w, (x, y) in wins.items():
        sB, sA, sV = st(rB, x, y), st(rA, x, y), st(rV, x, y)
        o[w] = dict(chop=sB["calmar"] - sA["calmar"], vol=sB["calmar"] - sV["calmar"],
                    calB=sB["calmar"], calA=sA["calmar"], calV=sV["calmar"])
    return o, (rB, rA, rV)


def decide(h1, h2, med):
    both = h1 >= 0.03 and h2 >= 0.03
    if both and med >= 0.02:
        return "RETAINED"
    if (not both) and med <= 0.02:
        return "REPLACEABLE"
    return "INCONCLUSIVE"


SIGN = (P["above"].values.astype(bool), P["vol20"].values, P["er20"].values)
TIER = {"F2": ("qld", 2, 0.30), "F3": ("tqqq", 3, 0.20)}
GOLD = P["gold_phys"].fillna(0).values
WN = {"IS": (START, FREEZE), "H1": (START, H1E), "H2": (H2S, FREEZE)}

# validation of this engine against the specialist's JSON
val = {}
for t, (col, mult, vt) in TIER.items():
    o, _ = dcal_set(SIGN, vt, mult, P[col].fillna(0).values, GOLD, "tranche", START, FREEZE, WN)
    sp = S["ndx"][t]["tranche"]["dcalmar_B_minus_alt"]
    val[t] = max(abs(o[w]["chop"] - sp["chop"]["3bps"][w]) for w in WN) + max(abs(o[w]["vol"] - sp["vol"]["3bps"][w]) for w in WN)
print("engine validation max |diff| vs specialist (sum chop+vol):", val, flush=True)
OUT["engine_validation"] = val

# ---------------------------------------------------------------- markets
yf = pd.read_parquet(os.path.join(HERE, "r2_chop_vol_matched_yf.parquet"))
y2 = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))


def yret(df, tk):
    s = df[df.ticker == tk]
    c = pd.Series(s["close"].values, index=pd.DatetimeIndex(s["date"])).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    first = c.index[0]
    lv = c.reindex(IDX, method="ffill")
    r = lv.pct_change().values
    r[IDX <= first] = np.nan
    return np.nan_to_num(r)


def sig_r(r):
    lv = pd.Series(np.cumprod(1 + r), index=IDX)
    ema = lv.ewm(span=100, adjust=False).mean()
    v20 = lv.pct_change().rolling(20).std() * np.sqrt(252)
    e20 = lv.diff(20).abs() / lv.diff().abs().rolling(20).sum().replace(0, np.nan)
    return (lv > ema).values, v20.values, e20.values


MR = {m: P[m].fillna(0).values for m in ["spx", "rut", "tsx", "hsi"]}
MR.update({"dax": yret(yf, "^GDAXI"), "n225": yret(yf, "^N225"), "ixic": yret(yf, "^IXIC")})
WX = {"FULL": (A90, FREEZE), "H1": (A90, H1E), "H2": (H2S, FREEZE)}


def xm_median(MRx, gold, tag):
    rows = {}
    for m, r in MRx.items():
        o, _ = dcal_set(sig_r(r), 0.30, 2, 2 * r - 0.0095 / 252 - RF, gold, "tranche", A90, FREEZE, WX)
        rows[m] = dict(chop=o["FULL"]["chop"], vol=o["FULL"]["vol"])
    med = {c: float(np.median([rows[m][c] for m in rows])) for c in ["chop", "vol"]}
    print(f"  [{tag}] per-market chop " + " ".join(f"{m} {rows[m]['chop']:+.4f}" for m in rows) + f" | median chop {med['chop']:+.4f} vol {med['vol']:+.4f}", flush=True)
    return dict(rows=rows, median=med)


# a) HSI
hs = y2[y2.ticker == "^HSI"]
hc = pd.Series(hs["close"].values, index=pd.DatetimeIndex(hs["date"])).sort_index()
hr = hc.reindex(IDX, method="ffill").pct_change()
m90 = IDX >= A90
diff = (hr - P["hsi"]).abs()
bad = diff[m90 & (diff > 1e-6).values]
OUT["hsi_diff_days"] = {str(k.date()): dict(panel=float(P["hsi"][k]), yahoo=float(hr[k])) for k in bad.index}
print("a) panel hsi vs ^HSI differing days (1990+):", len(bad), flush=True)
for k in bad.index:
    print(f"   {k.date()} panel {P['hsi'][k]:+.4f} yahoo {hr[k]:+.4f}")
MRy = dict(MR)
MRy["hsi"] = np.nan_to_num(hr.values)
OUT["xm_base"] = xm_median(MR, GOLD, "base (as specialist)")
OUT["xm_hsi_yahoo"] = xm_median(MRy, GOLD, "hsi from Yahoo ^HSI")

# b) gold before 2000-08-30
interp = (P["gold_src"] == "monthly-interp").values
g0 = GOLD.copy(); g0[interp] = 0.0
gl = pd.Series(GOLD, index=IDX)
mon = IDX.to_period("M")
lump = gl.copy()
lump[interp] = 0.0
gi = gl[interp]
mret = (1 + gi).groupby(gi.index.to_period("M")).prod() - 1
lastday = pd.Series(gi.index, index=gi.index).groupby(gi.index.to_period("M")).max()
for p_, dlast in lastday.items():
    lump[dlast] = mret[p_]
gL = lump.values
OUT["gold_sens"] = {}
for gname, g in [("zero_pre2000", g0), ("monthend_lump_pre2000", gL)]:
    rr = {}
    for t, (col, mult, vt) in TIER.items():
        o, _ = dcal_set(SIGN, vt, mult, P[col].fillna(0).values, g, "tranche", START, FREEZE, WN)
        rr[t] = {w: dict(chop=o[w]["chop"], vol=o[w]["vol"]) for w in WN}
    xm = xm_median(MR, g, f"gold {gname}")
    dec = {f"{c}|{t}": decide(rr[t]["H1"][c], rr[t]["H2"][c], xm["median"][c]) for t in TIER for c in ["chop", "vol"]}
    OUT["gold_sens"][gname] = dict(ndx=rr, xm_median=xm["median"], decisions=dec)
    print(f"b) gold {gname}: " + " | ".join(f"{t} H1 chop {rr[t]['H1']['chop']:+.3f} vol {rr[t]['H1']['vol']:+.3f} H2 chop {rr[t]['H2']['chop']:+.4f} vol {rr[t]['H2']['vol']:+.4f}" for t in TIER) + f" -> {dec}", flush=True)

# c) exposure on daily target weights
OUT["expo_target_def"] = {}
for t, (col, mult, vt) in TIER.items():
    o, _ = dcal_set(SIGN, vt, mult, P[col].fillna(0).values, GOLD, "tranche", START, FREEZE, WN, expo=expo_target)
    OUT["expo_target_def"][t] = dict(vtA=o["vtA"], cV=o["cV"], H1=o["H1"], H2=o["H2"])
    print(f"c) target-weight exposure {t}: vtA {o['vtA']:.5f} cV {o['cV']:.5f} | chop H1 {o['H1']['chop']:+.4f} H2 {o['H2']['chop']:+.4f} | vol H1 {o['H1']['vol']:+.4f} H2 {o['H2']['vol']:+.4f}", flush=True)

# d) per-weekday books, exposure re-matched per book
OUT["per_weekday_rematched"] = {}
for t, (col, mult, vt) in TIER.items():
    row = {}
    for d in DAYS:
        o, _ = dcal_set(SIGN, vt, mult, P[col].fillna(0).values, GOLD, d, START, FREEZE, {"IS": (START, FREEZE), "H1": (START, H1E), "H2": (H2S, FREEZE)})
        row[d] = dict(chop=o["IS"]["chop"], vol=o["IS"]["vol"], chopH2=o["H2"]["chop"], volH2=o["H2"]["vol"], calB=o["IS"]["calB"])
    OUT["per_weekday_rematched"][t] = row
    print(f"d) per-weekday re-matched {t}: " + " ".join(f"{d} chop {row[d]['chop']:+.3f} vol {row[d]['vol']:+.3f}" for d in DAYS)
          + f" | argmax chop {max(row, key=lambda k: row[k]['chop'])} vol {max(row, key=lambda k: row[k]['vol'])}", flush=True)

# e) C3 episode details
OUT["c3"] = {}
for t, (col, mult, vt) in TIER.items():
    o, (rB, rA, rV) = dcal_set(SIGN, vt, mult, P[col].fillna(0).values, GOLD, "tranche", START, FREEZE,
                               {"T1_88_99": (START, "1999-12-31"), "2000": ("2000-01-01", "2000-12-31"),
                                "2001_06": ("2001-01-01", H1E), "H1_ex2000": (START, "1999-12-31")})
    sV = st(rV, START, H1E)
    sB = st(rB, START, H1E)
    dV = pd.Series(sV["dd"], index=sV["d"])
    dB = pd.Series(sB["dd"], index=sB["d"])
    yr = {}
    for y in range(1998, 2004):
        mm = msk(f"{y}-01-01", f"{y}-12-31")
        yr[y] = dict(B=float(np.prod(1 + rB[mm]) - 1), V=float(np.prod(1 + rV[mm]) - 1), A=float(np.prod(1 + rA[mm]) - 1))
    # V drawdown from its March-2000 peak, measured at 2000-10-04 and at its trough
    pk = sV["peak"]
    ddV_20001004 = float(dV.loc[:"2000-10-04"].iloc[-1])
    OUT["c3"][t] = dict(V_H1_peak=sV["peak"], V_H1_trough=sV["trough"], V_H1_maxdd=sV["maxdd"],
                        V_dd_at_2000_10_04=ddV_20001004, V_min_dd_in_2000=float(dV.loc["2000-01-01":"2000-12-31"].min()),
                        B_H1_peak=sB["peak"], B_H1_trough=sB["trough"], B_H1_maxdd=sB["maxdd"],
                        B_min_dd_2000_2003=float(dB.loc["2000-01-01":"2003-12-31"].min()),
                        vol_dcal_subera={k: o[k]["vol"] for k in ["T1_88_99", "2000", "2001_06"]},
                        chop_dcal_subera={k: o[k]["chop"] for k in ["T1_88_99", "2000", "2001_06"]},
                        yearly_returns=yr)
    print(f"e) {t}: V H1 maxDD {sV['maxdd']*100:.1f}% {sV['peak']}->{sV['trough']}; V dd at 2000-10-04 {ddV_20001004*100:.1f}%, "
          f"V min dd in 2000 {OUT['c3'][t]['V_min_dd_in_2000']*100:.1f}% | B H1 maxDD {sB['maxdd']*100:.1f}% {sB['peak']}->{sB['trough']}, "
          f"B min dd 2000-03 {OUT['c3'][t]['B_min_dd_2000_2003']*100:.1f}% | vol dCal 88-99 {o['T1_88_99']['vol']:+.3f} 2000 {o['2000']['vol']:+.3f} 01-06 {o['2001_06']['vol']:+.3f}", flush=True)
    print("   yearly B/V/A: " + " ".join(f"{y} {v['B']*100:+.1f}/{v['V']*100:+.1f}/{v['A']*100:+.1f}" for y, v in yr.items()), flush=True)

# f) bootstrap seeds (info)
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
    return float(out.mean()), float((out <= 0).mean())


OUT["boot_seeds"] = {}
for t, (col, mult, vt) in TIER.items():
    _, (rB, rA, rV) = dcal_set(SIGN, vt, mult, P[col].fillna(0).values, GOLD, "tranche", START, FREEZE, {"IS": (START, FREEZE)})
    for comp, rx in [("chop", rA), ("vol", rV)]:
        res = [boot(rB, rx, START, FREEZE, seed=s) for s in [7, 11, 123, 2026]]
        OUT["boot_seeds"][f"{comp}|{t}"] = res
        print(f"f) boot {comp} {t}: (mean, P<=0) by seed 7/11/123/2026: " + " ".join(f"({m:+.3f},{p:.2f})" for m, p in res), flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(SCR, "verify_sens.json"), "w"), indent=1)
print("done")
