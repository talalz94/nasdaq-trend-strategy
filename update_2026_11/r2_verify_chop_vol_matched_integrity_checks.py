"""
INTEGRITY REVIEW of r2_chop_vol_matched (independent re-implementation; writes nothing in the project).

1. NDX F2/F3, tranche + Friday: weights built from the SPEC TEXT in pandas (not via the specialist's numpy engine),
   exposures from common.held_weights, vt_A solved by Illinois regula-falsi (not bisection), c_V by ratio,
   returns via common.backtest per weekday (tranche = mean of 5 books), Calmar via common.stats.
2. Look-ahead test: identical computation on the panel TRUNCATED at FREEZE (and yfinance closes truncated).
3. Cross-market (7 indices + NDX-sim ref): own data path (reindex(method='ffill') for yfinance), own signals from
   the level, common.strategy_weights/backtest with substituted columns, own solver; median; rule.
4. yfinance cache vs the independently-fetched r2_friday_effect_yf.parquet (same tickers).

usage: python -u update_2026_11/r2_verify_chop_vol_matched_integrity_checks.py <scratch_dir> [truncate]
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = sys.argv[1]
TRUNC = len(sys.argv) > 2 and sys.argv[2] == "truncate"
P0 = C.panel()
P = P0.loc[:C.FREEZE].copy() if TRUNC else P0.copy()
IDX = P.index
START, FREEZE = C.START, C.FREEZE
H1E, H2S = pd.Timestamp("2006-12-31"), pd.Timestamp("2007-01-01")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MK = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(DAYS)}
TAG = "TRUNC" if TRUNC else "FULL"


def spec_weights(above, vol20, er20, lev_col, vt, chop=True, vol=True, c=1.0):
    """weights exactly as the task text: risk-on iff above; scale=min(1,vt/vol20); x0.5 Aug/Sep;
    engine = scale if er20>=.25 else .5*scale (chop rule) ; sleeve .85*engine; IEF = .85-sleeve; gold .15."""
    idx = above.index
    if vol:
        s = (vt / vol20)
        s = s.where(vol20.notna(), 1.0)          # no vol estimate -> full scale (as July/common)
        s = np.minimum(s, 1.0)
    else:
        s = pd.Series(1.0, index=idx)
    s = s * np.where(idx.month.isin([8, 9]), 0.5, 1.0)
    s = s * c
    cm = np.where(er20.fillna(-1.0) >= 0.25, 1.0, 0.5) if chop else 1.0
    eng = pd.Series(np.where(above.astype(bool), s * cm, 0.0), index=idx)
    w = pd.DataFrame({lev_col: 0.85 * eng, "ief": 0.85 - 0.85 * eng, "gold_phys": 0.15}, index=idx)
    return w


def books(basis):
    return DAYS if basis == "tranche" else ["Fri"]


def expo(PP, w, lev_col, mult, basis, a, b):
    h = sum(C.held_weights(PP, w, mark=MK[d])[lev_col] for d in books(basis)) / len(books(basis))
    m = (h.index >= a) & (h.index <= b)
    return float(mult * h[m].mean())


def rets(PP, w, basis, cost=3e-4):
    return sum(C.backtest(PP, w, cost=cost, mark=MK[d]) for d in books(basis)) / len(books(basis))


def illinois(f, lo, hi, tol=1e-11, it=300):
    flo, fhi = f(lo), f(hi)
    assert flo < 0 < fhi, (flo, fhi)
    side = 0
    for _ in range(it):
        x = (lo * fhi - hi * flo) / (fhi - flo)
        fx = f(x)
        if abs(fx) < tol:
            return x
        if fx > 0:
            hi, fhi = x, fx
            if side == 1:
                flo /= 2
            side = 1
        else:
            lo, flo = x, fx
            if side == -1:
                fhi /= 2
            side = -1
    return x


def cal(r, a, b):
    s = C.stats(r, a, b, start=None)
    return s


def comp_set(PP, sig, lev_col, mult, vt, basis, a, b, wins, costs=(3e-4,)):
    above, vol20, er20 = sig
    wB = spec_weights(above, vol20, er20, lev_col, vt)
    eB = expo(PP, wB, lev_col, mult, basis, a, b)
    fA = lambda v: expo(PP, spec_weights(above, vol20, er20, lev_col, v, chop=False), lev_col, mult, basis, a, b) - eB
    vtA = illinois(fA, 0.01, vt)
    eV0 = expo(PP, spec_weights(above, vol20, er20, lev_col, None, vol=False), lev_col, mult, basis, a, b)
    cV = eB / eV0
    eA0 = expo(PP, spec_weights(above, vol20, er20, lev_col, vt, chop=False), lev_col, mult, basis, a, b)
    cAc = eB / eA0
    W = {"B": wB, "A": spec_weights(above, vol20, er20, lev_col, vtA, chop=False),
         "V": spec_weights(above, vol20, er20, lev_col, None, vol=False, c=cV),
         "A0": spec_weights(above, vol20, er20, lev_col, vt, chop=False),
         "Ac": spec_weights(above, vol20, er20, lev_col, vt, chop=False, c=cAc)}
    out = dict(vtA=vtA, cV=cV, cAc=cAc, expo_B=eB,
               expo_check={k: expo(PP, W[k], lev_col, mult, basis, a, b) for k in ["A", "V", "Ac"]})
    R = {}
    for cost in costs:
        R[cost] = {k: rets(PP, w, basis, cost) for k, w in W.items()}
    out["stats"] = {str(cost): {k: {wn: cal(R[cost][k], *ab) for wn, ab in wins.items()} for k in W} for cost in costs}
    out["dcal"] = {str(cost): {comp: {wn: out["stats"][str(cost)]["B"][wn]["calmar"] - out["stats"][str(cost)][x][wn]["calmar"]
                                      for wn in wins} for comp, x in [("chop", "A"), ("vol", "V"), ("Ac", "Ac")]}
                   for cost in costs}
    return out, R


def decide(h1, h2, med):
    both = h1 >= 0.03 and h2 >= 0.03
    if both and med >= 0.02:
        return "RETAINED"
    if (not both) and med <= 0.02:
        return "REPLACEABLE"
    return "INCONCLUSIVE (retain by default)"


RES = {"mode": TAG}
# ------------------------------------------------------------------ 1. NDX
sigN = (P["above"], P["vol20"], P["er20"])
WN = {"IS": (START, FREEZE), "H1": (START, H1E), "H2": (H2S, FREEZE)}
TIER = {"F2": ("qld", 2, 0.30), "F3": ("tqqq", 3, 0.20)}
RES["ndx"] = {}
for t, (col, mult, vt) in TIER.items():
    for basis in ["tranche", "Fri"]:
        o, _ = comp_set(P, sigN, col, mult, vt, basis, START, FREEZE, WN, costs=(3e-4, 10e-4, 15e-4))
        RES["ndx"][f"{t}|{basis}"] = o
        s = o["stats"]["0.0003"]
        print(f"[{TAG}] {t} {basis:7s} vtA {o['vtA']:.6f} cV {o['cV']:.6f} cAc {o['cAc']:.6f} expoB {o['expo_B']:.6f} "
              f"| Cal IS B {s['B']['IS']['calmar']:.4f} A {s['A']['IS']['calmar']:.4f} V {s['V']['IS']['calmar']:.4f} "
              f"| dChop H1 {o['dcal']['0.0003']['chop']['H1']:+.4f} H2 {o['dcal']['0.0003']['chop']['H2']:+.4f} "
              f"| dVol H1 {o['dcal']['0.0003']['vol']['H1']:+.4f} H2 {o['dcal']['0.0003']['vol']['H2']:+.4f}", flush=True)

# ------------------------------------------------------------------ 2. cross-market
yf = pd.read_parquet(os.path.join(HERE, "r2_chop_vol_matched_yf.parquet"))
rf_d = P["rf_d"].fillna(0)


def mkt(src, col):
    if src == "panel":
        r = P[col].astype(float)
        first = r[(r != 0) & r.notna()].index[0]
        lvl = (1 + r.fillna(0)).cumprod()
        return r.fillna(0), lvl, first
    s = yf[yf.ticker == col]
    c = pd.Series(s["close"].values, index=pd.DatetimeIndex(s["date"])).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    if TRUNC:
        c = c.loc[:FREEZE]
    first = c.index[0]
    lv = c.reindex(IDX, method="ffill")                    # last close at or before each US date
    r = lv.pct_change()
    r[IDX <= first] = np.nan
    lvl = lv.ffill().bfill()                                # flat before first close (warm-up only)
    return r.fillna(0), lvl, first


def sig_from_level(lvl):
    ema = lvl.ewm(span=100, adjust=False).mean()
    v20 = lvl.pct_change().rolling(20).std() * np.sqrt(252)
    e20 = lvl.diff(20).abs() / lvl.diff().abs().rolling(20).sum().replace(0, np.nan)
    return lvl > ema, v20, e20


XM = {"spx": ("panel", "spx"), "rut": ("panel", "rut"), "tsx": ("panel", "tsx"), "hsi": ("panel", "hsi"),
      "dax": ("yf", "^GDAXI"), "n225": ("yf", "^N225"), "ixic": ("yf", "^IXIC"), "ndx_sim2x(ref)": ("panel", "ndx_ret")}
A90 = pd.Timestamp("1990-01-01")
WX = {"FULL": (A90, FREEZE), "H1": (A90, H1E), "H2": (H2S, FREEZE)}
RES["xm"] = {}
for name, (src, col) in XM.items():
    r, lvl, first = mkt(src, col)
    sig = sig_from_level(lvl)
    PP = P.copy()
    PP["lev_x"] = 2 * r - 0.0095 / 252 - rf_d
    o, _ = comp_set(PP, sig, "lev_x", 2, 0.30, "tranche", A90, FREEZE, WX)
    d = o["dcal"]["0.0003"]
    RES["xm"][name] = dict(first=str(first.date()), vtA=o["vtA"], cV=o["cV"], cAc=o["cAc"],
                           calmar_FULL={k: o["stats"]["0.0003"][k]["FULL"]["calmar"] for k in ["B", "A", "V", "A0", "Ac"]},
                           dchop=d["chop"], dvol=d["vol"], dAc=d["Ac"])
    print(f"[{TAG}] {name:15s} first {first.date()} vtA {o['vtA']:.4f} cV {o['cV']:.4f} | dChop FULL {d['chop']['FULL']:+.4f} "
          f"H1 {d['chop']['H1']:+.4f} H2 {d['chop']['H2']:+.4f} | dVol FULL {d['vol']['FULL']:+.4f} | dAc FULL {d['Ac']['FULL']:+.4f}", flush=True)
mk7 = [m for m in XM if "ref" not in m]
med = {c: float(np.median([RES["xm"][m][k]["FULL"] for m in mk7])) for c, k in [("chop", "dchop"), ("vol", "dvol"), ("Ac", "dAc")]}
RES["median"] = med
print(f"[{TAG}] cross-market median: {med}")
RES["decisions"] = {}
for t in ["F2", "F3"]:
    for comp in ["chop", "vol"]:
        d = RES["ndx"][f"{t}|tranche"]["dcal"]["0.0003"][comp]
        RES["decisions"][f"{comp}|{t}"] = dict(H1=d["H1"], H2=d["H2"], median=med[comp], decision=decide(d["H1"], d["H2"], med[comp]))
        print(f"[{TAG}] DECISION {comp} {t}: H1 {d['H1']:+.4f} H2 {d['H2']:+.4f} med {med[comp]:+.4f} -> {decide(d['H1'], d['H2'], med[comp])}")

# ------------------------------------------------------------------ 3. compare with the specialist's JSON
S = json.load(open(os.path.join(HERE, "r2_chop_vol_matched.json")))
cmp = {}
for t in ["F2", "F3"]:
    for basis in ["tranche", "Fri"]:
        mine = RES["ndx"][f"{t}|{basis}"]
        sp = S["ndx"][t][basis]
        cmp[f"{t}|{basis}|vtA"] = mine["vtA"] - sp["solve_A"]["vt"]
        cmp[f"{t}|{basis}|cV"] = mine["cV"] - sp["solve_V"]["c"]
        for comp in ["chop", "vol"]:
            for w in ["IS", "H1", "H2"]:
                for cost, cs in [("0.0003", "3bps"), ("0.001", "10bps"), ("0.0015", "15bps")]:
                    cmp[f"{t}|{basis}|{comp}|{w}|{cs}"] = mine["dcal"][cost][comp][w] - sp["dcalmar_B_minus_alt"][comp][cs][w]
for m in mk7:
    sm = "ixic" if m == "ixic" else m
    for comp, k in [("chop", "dchop"), ("vol", "dvol")]:
        cmp[f"xm|{m}|{comp}|FULL"] = RES["xm"][m][k]["FULL"] - S["cross_market"]["markets"][sm]["dcalmar_B_minus_alt"][comp]["FULL"]
cmp["median_chop"] = med["chop"] - S["cross_market"]["median_dcalmar"]["chop"]["median"]
cmp["median_vol"] = med["vol"] - S["cross_market"]["median_dcalmar"]["vol"]["median"]
RES["diff_vs_specialist"] = cmp
mx = max(abs(v) for v in cmp.values())
print(f"[{TAG}] max |mine - specialist| over {len(cmp)} compared quantities: {mx:.3e}")
for k, v in sorted(cmp.items(), key=lambda kv: -abs(kv[1]))[:8]:
    print(f"      {k}: {v:+.3e}")

# ------------------------------------------------------------------ 4. yfinance data cross-check (FULL mode only)
if not TRUNC:
    y2 = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))
    dchk = {}
    for tk in ["^GDAXI", "^N225", "^IXIC"]:
        a = yf[yf.ticker == tk].set_index("date")["close"].sort_index()
        b = y2[y2.ticker == tk].set_index("date")["close"].sort_index()
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        dchk[tk] = dict(rows_a=len(a), rows_b=len(b), common=len(j),
                        max_rel_diff=float((j.iloc[:, 0] / j.iloc[:, 1] - 1).abs().max()))
    # panel hsi / spx vs ^HSI / ^GSPC closes on the US calendar
    for tk, col in [("^HSI", "hsi"), ("^GSPC", "spx")]:
        b = y2[y2.ticker == tk].set_index("date")["close"].sort_index()
        b.index = pd.DatetimeIndex(b.index)
        lv = b.reindex(IDX, method="ffill").pct_change()
        m = IDX >= "1990-01-01"
        dd = (lv[m] - P[col][m]).abs()
        dchk[f"{tk}_vs_panel_{col}"] = dict(days_absdiff_gt_1e6=int((dd > 1e-6).sum()), max_absdiff=float(dd.max()),
                                            corr=float(np.corrcoef(lv[m].fillna(0), P[col][m].fillna(0))[0, 1]))
    RES["data_crosscheck"] = dchk
    print(f"[{TAG}] data cross-check: {json.dumps(dchk, indent=0)}")


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


json.dump(clean(RES), open(os.path.join(SCR, f"verify_checks_{TAG}.json"), "w"), indent=1)
print(f"[{TAG}] done")
