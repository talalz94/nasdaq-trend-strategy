"""
COMPLETENESS CRITIC - evidence table for the candidate specs on the Friday (research) basis and the
timing-luck-free 5-weekday tranche basis, plus a paired block bootstrap vs QQQ buy-and-hold and F2 vs F3.
All candidate specs are pre-registered July/Nov specs; nothing is tuned. IS stops at 2026-07-24.
Writes update_2026_11/critic_candidates.json.
"""
from __future__ import annotations
import os, sys, json, time, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("CRITIC_OUT_DIR") or os.path.join(tempfile.gettempdir(), "critic_out")
os.makedirs(OUTDIR, exist_ok=True)
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
P = C.panel(); IDX = P.index; N = len(IDX)
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
dates_np = IDX.values
RET = {c: P[c].fillna(0).values for c in ["qqq", "qld", "tqqq", "ief", "tlt", "bills", "gold_phys"]}
VOL = P["vol20"].values; ER = P["er20"].values; ABOVE = P["above"].values.astype(bool); MONTH = P["month"].values
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MARKS = {d: (IDX.dayofweek == i) for i, d in enumerate(DAYS)}
MARKS["Fri"] = P["mark"].values.astype(bool)


def last_idx(mask):
    pos = np.where(mask, np.arange(N), -1)
    return np.maximum.accumulate(pos)


LAST = {d: last_idx(m) for d, m in MARKS.items()}


def weights_np(lev, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, seasonal=True, off=None):
    off = off or {"ief": 1.0}
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.clip(voltgt / VOL, 0, 1)
    sc = np.where(np.isnan(sc), 1.0, sc)
    if seasonal:
        sc = np.clip(sc * np.where(np.isin(MONTH, (8, 9)), 0.5, 1.0), 0, 1)
    clean = np.where(np.isnan(ER), False, ER >= chop_thr)
    eq = (ABOVE & clean) * sc + (ABOVE & ~clean) * sc * chop_pos
    core = 1.0 - gold_w
    W = {lev: eq * core}
    rest = np.clip(core - eq * core, 0, None)
    for c, f in off.items():
        W[c] = W.get(c, 0.0) + rest * f
    if gold_w > 0:
        W["gold_phys"] = W.get("gold_phys", 0.0) + gold_w
    return W


def held_np(W, day, lag=1):
    li = LAST[day]; H = {}
    for c, w in W.items():
        w = np.broadcast_to(np.asarray(w, dtype=float), (N,))
        h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
        H[c] = np.concatenate([np.zeros(lag), h[:-lag]])
    return H


def ret_from_held(H, cost):
    """cost: float (all sleeves) or dict sleeve->cost per unit turnover"""
    gross = np.zeros(N); tc = np.zeros(N)
    for c, h in H.items():
        gross += h * RET[c]
        dh = np.abs(np.diff(h, prepend=np.nan)); dh[0] = 0.0
        tc += dh * (cost[c] if isinstance(cost, dict) else cost)
    return gross - tc


def run(W, mode="Fri", lag=1, cost=C.COST):
    if mode == "tranche":
        Hs = [held_np(W, d, lag) for d in DAYS]
        H = {c: sum(h[c] for h in Hs) / 5.0 for c in Hs[0]}
    else:
        H = held_np(W, mode, lag)
    return ret_from_held(H, cost)


def st(r, a, b):
    m = (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))
    rr = r[m]; d = dates_np[m]
    e = np.cumprod(1 + rr)
    yrs = (pd.Timestamp(d[-1]) - pd.Timestamp(d[0])).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    ex = rr - P["rf_d"].fillna(0).values[m]
    return dict(cagr=float(cagr), maxdd=dd, calmar=float(cagr / abs(dd)),
                ex_sharpe=float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)))


def per(r, a, b):
    m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    return float(np.prod(1 + r[m]) - 1)


WINS = {"IS": ("1988-01-01", FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
        "2000-frz": ("2000-01-01", FREEZE), "2010-frz": ("2010-01-01", FREEZE)}
CRI = {"2022": ("2022-01-01", "2022-12-31"), "dotcom": ("2000-03-01", "2002-10-31"),
       "GFC": ("2007-10-01", "2009-03-31"), "OOS": ("2026-07-25", str(LATEST.date()))}
UC = {"qld": 15e-4, "tqqq": 20e-4, "ief": 8e-4, "gold_phys": 8e-4, "bills": 8e-4}   # ucits.py central one-way costs

SPECS = {
    "F2-IEF (QLD vt.30)": dict(W=weights_np("qld", 0.30), lag=1, cost=C.COST),
    "F2-IEF lag2 (Monday execution)": dict(W=weights_np("qld", 0.30), lag=2, cost=C.COST),
    "F2-IEF lag2 + UCITS-level costs 15/8bp": dict(W=weights_np("qld", 0.30), lag=2, cost=UC),
    "F3-IEF (TQQQ vt.20)": dict(W=weights_np("tqqq", 0.20), lag=1, cost=C.COST),
    "F2 off-leg 50/50 IEF/BIL": dict(W=weights_np("qld", 0.30, off={"ief": 0.5, "bills": 0.5}), lag=1, cost=C.COST),
    "F3 off-leg 50/50 IEF/BIL": dict(W=weights_np("tqqq", 0.20, off={"ief": 0.5, "bills": 0.5}), lag=1, cost=C.COST),
    "F1-IEF (QQQ vt.20)": dict(W=weights_np("qqq", 0.20), lag=1, cost=C.COST),
    "F1-IEF lag2 + UCITS-level costs 8/8bp": dict(W=weights_np("qqq", 0.20), lag=2,
                                                 cost={"qqq": 8e-4, "ief": 8e-4, "gold_phys": 8e-4}),
    "F2-TLT (scanner-style off-leg)": dict(W=weights_np("qld", 0.30, off={"tlt": 1.0}), lag=1, cost=C.COST),
}
OUT = {"meta": dict(script="update_2026_11/critic_candidates.py", ucits_cost_model=UC,
                    note="Friday = research convention; tranche = mean of 5 weekday phases (timing-luck-free)")}
SER = {}
for name, s in SPECS.items():
    row = {}
    for mode in ["Fri", "tranche"]:
        r = run(s["W"], mode, s["lag"], s["cost"])
        SER[(name, mode)] = r
        row[mode] = {w: st(r, *ab) for w, ab in WINS.items()}
        row[mode]["crises"] = {k: per(r, *ab) for k, ab in CRI.items()}
    OUT[name] = row
qqq = P["qqq"].fillna(0).values
OUT["QQQ buy-and-hold"] = {"IS": st(qqq, *WINS["IS"]), "2000-frz": st(qqq, *WINS["2000-frz"])}


# paired circular block bootstrap (63d) of CAGR and Calmar differences, IS
def boot(r1, r2, n=2000, block=63, seed=0):
    m = (IDX >= START) & (IDX <= FREEZE)
    a = np.column_stack([r1[m], r2[m]]); T = len(a); rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block)); dc, dk, dd1, dd2 = [], [], [], []
    for _ in range(n):
        stt = rng.integers(0, T, nb)
        ix = (stt[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = a[ix]; e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        mdd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = g / np.abs(mdd)
        dc.append(g[0] - g[1]); dk.append(cal[0] - cal[1]); dd1.append(mdd[0]); dd2.append(mdd[1])
    dc, dk = np.array(dc), np.array(dk)
    return dict(dcagr=[float(np.mean(dc)), float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
                p_dcagr_le0=float((dc <= 0).mean()),
                dcalmar=[float(np.mean(dk)), float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5))],
                p_dcalmar_le0=float((dk <= 0).mean()),
                maxdd_median=[float(np.median(dd1)), float(np.median(dd2))],
                maxdd_p5=[float(np.percentile(dd1, 5)), float(np.percentile(dd2, 5))])


BO = {}
for nm in ["F2-IEF (QLD vt.30)", "F3-IEF (TQQQ vt.20)", "F1-IEF (QQQ vt.20)", "F2-IEF lag2 + UCITS-level costs 15/8bp",
           "F2 off-leg 50/50 IEF/BIL", "F1-IEF lag2 + UCITS-level costs 8/8bp"]:
    for mode in ["Fri", "tranche"]:
        BO[f"{nm} [{mode}] minus QQQ"] = boot(SER[(nm, mode)], qqq)
for mode in ["Fri", "tranche"]:
    BO[f"F2 minus F3 [{mode}]"] = boot(SER[("F2-IEF (QLD vt.30)", mode)], SER[("F3-IEF (TQQQ vt.20)", mode)])
    BO[f"F2 50/50 minus F2 IEF [{mode}]"] = boot(SER[("F2 off-leg 50/50 IEF/BIL", mode)], SER[("F2-IEF (QLD vt.30)", mode)])
OUT["bootstrap_IS"] = BO


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    return o


json.dump(clean(OUT), open(os.path.join(OUTDIR, "critic_candidates.json"), "w"), indent=1)
for name in SPECS:
    for mode in ["Fri", "tranche"]:
        r = OUT[name][mode]
        print(f"{name:40s} {mode:8s} IS {r['IS']['cagr']*100:5.1f}/{r['IS']['maxdd']*100:5.1f}/{r['IS']['calmar']:.2f} "
              f"exSh {r['IS']['ex_sharpe']:.2f} | H1 {r['H1']['calmar']:.2f} H2 {r['H2']['calmar']:.2f} | "
              f"2000+ {r['2000-frz']['cagr']*100:5.1f}/{r['2000-frz']['maxdd']*100:5.1f}/{r['2000-frz']['calmar']:.2f} | "
              f"2010+ {r['2010-frz']['calmar']:.2f} | 2022 {r['crises']['2022']*100:6.1f} dot {r['crises']['dotcom']*100:6.1f} "
              f"GFC {r['crises']['GFC']*100:5.1f} OOS {r['crises']['OOS']*100:6.1f}")
print("QQQ", OUT["QQQ buy-and-hold"])
for k, v in BO.items():
    print(f"{k:45s} dCAGR {v['dcagr'][0]*100:+5.2f} [{v['dcagr'][1]*100:+5.2f},{v['dcagr'][2]*100:+5.2f}] p<=0 {v['p_dcagr_le0']:.3f} | "
          f"dCalmar {v['dcalmar'][0]:+.3f} [{v['dcalmar'][1]:+.3f},{v['dcalmar'][2]:+.3f}] p<=0 {v['p_dcalmar_le0']:.3f} | "
          f"medDD {v['maxdd_median'][0]*100:.1f}/{v['maxdd_median'][1]*100:.1f} p5DD {v['maxdd_p5'][0]*100:.1f}/{v['maxdd_p5'][1]*100:.1f}")
print(f"runtime {time.time()-T0:.1f}s")
