"""
COMPLETENESS CRITIC - re-run revalidate's pre-registered component ablations (C3) on the timing-luck-free
5-weekday tranche as well as the Friday baseline. Diagnostic only: nothing is chosen; IS stops at 2026-07-24.
Writes update_2026_11/critic_ablation_tranche.json.
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
START, FREEZE = C.START, C.FREEZE
dates_np = IDX.values
RET = {c: P[c].fillna(0).values for c in ["qqq", "qld", "tqqq", "ief", "tlt", "bills", "gold_phys"]}
VOL = P["vol20"].values; ER = P["er20"].values; ABOVE = P["above"].values.astype(bool); MONTH = P["month"].values
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MARKS = {d: (IDX.dayofweek == i) for i, d in enumerate(DAYS)}
MARKS["Fri"] = P["mark"].values.astype(bool)
MARKS["daily"] = np.ones(N, dtype=bool)


def last_idx(mask):
    pos = np.where(mask, np.arange(N), -1)
    return np.maximum.accumulate(pos)


LAST = {d: last_idx(m) for d, m in MARKS.items()}


def weights_np(lev, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, seasonal=True, off="ief",
               gate=True, gold_to="gold_phys"):
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.clip(voltgt / VOL, 0, 1)
    sc = np.where(np.isnan(sc), 1.0, sc)
    if seasonal:
        sc = np.clip(sc * np.where(np.isin(MONTH, (8, 9)), 0.5, 1.0), 0, 1)
    ron = ABOVE if gate else np.ones(N, dtype=bool)
    clean = np.where(np.isnan(ER), False, ER >= chop_thr)
    eq = (ron & clean) * sc + (ron & ~clean) * sc * chop_pos
    core = 1.0 - gold_w
    W = {lev: eq * core}
    rest = np.clip(core - eq * core, 0, None)
    W[off] = W.get(off, 0.0) + rest
    if gold_w > 0:
        W[gold_to] = W.get(gold_to, 0.0) + gold_w
    return W


def held_np(W, day, lag=1):
    li = LAST[day]; H = {}
    for c, w in W.items():
        w = np.broadcast_to(np.asarray(w, dtype=float), (N,))
        h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
        H[c] = np.concatenate([np.zeros(lag), h[:-lag]])
    return H


def ret_from_held(H, cost=C.COST):
    gross = np.zeros(N); turn = np.zeros(N)
    for c, h in H.items():
        gross += h * RET[c]
        dh = np.abs(np.diff(h, prepend=np.nan)); dh[0] = 0.0
        turn += dh
    return gross - turn * cost


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
    ulcer = float(np.sqrt(np.mean((100 * (e / np.maximum.accumulate(e) - 1)) ** 2)))
    return dict(cagr=float(cagr), maxdd=dd, calmar=float(cagr / abs(dd)), ulcer=ulcer)


WINS = {"IS": ("1988-01-01", FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
        "T1_1988-99": ("1988-01-01", "1999-12-31"), "T2_2000-12": ("2000-01-01", "2012-12-31"),
        "T3_2013-frz": ("2013-01-01", FREEZE)}

BASE = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
# (label, kwargs for the 'without' variant, rebalance override for without, description)
ABL = [
    ("EMA gate", dict(gate=False), None),
    ("vol target", dict(voltgt=99.0), None),
    ("chop half vs full", dict(chop_pos=1.0), None),
    ("chop half vs zero", dict(chop_pos=0.0), None),
    ("IEF vs BIL off-leg", dict(off="bills"), None),
    ("15% gold vs none (core 1.0)", dict(gold_w=0.0), None),
    ("gold vs IEF at equal 0.85 core", dict(gold_to="ief"), None),
    ("Aug-Sep half", dict(seasonal=False), None),
    ("weekly vs daily rebalance", dict(), "daily"),
]
OUT = {"meta": dict(script="update_2026_11/critic_ablation_tranche.py",
                    note="value of each component = metric(with) - metric(without); Friday = research baseline; "
                         "tranche = mean of 5 weekday rebalance phases (timing-luck-free). 'daily' rebalance has no "
                         "phase, so its tranche comparison uses daily for the without-variant.")}
for sname, base in BASE.items():
    res = {}
    Wb = weights_np(**base)
    rb = {m: run(Wb, m) for m in ["Fri", "tranche"]}
    sb = {m: {w: st(rb[m], *ab) for w, ab in WINS.items()} for m in rb}
    res["baseline"] = sb
    for label, kw, reb in ABL:
        kk = dict(base); kk.update(kw)
        Wx = weights_np(**kk)
        row = {}
        for m in ["Fri", "tranche"]:
            rx = run(Wx, "daily" if reb == "daily" else m)
            for w, ab in WINS.items():
                sx = st(rx, *ab)
                row[f"{m}|{w}"] = dict(d_calmar=sb[m][w]["calmar"] - sx["calmar"], d_cagr=sb[m][w]["cagr"] - sx["cagr"],
                                       d_maxdd=sb[m][w]["maxdd"] - sx["maxdd"], d_ulcer=sb[m][w]["ulcer"] - sx["ulcer"])
        row["Fri_helps_both_halves"] = bool(row["Fri|H1"]["d_calmar"] > 0 and row["Fri|H2"]["d_calmar"] > 0)
        row["tranche_helps_both_halves"] = bool(row["tranche|H1"]["d_calmar"] > 0 and row["tranche|H2"]["d_calmar"] > 0)
        row["tranche_helps_all_thirds"] = bool(all(row[f"tranche|{t}"]["d_calmar"] > 0
                                                   for t in ["T1_1988-99", "T2_2000-12", "T3_2013-frz"]))
        res[label] = row
    OUT[sname] = res


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(OUTDIR, "critic_ablation_tranche.json"), "w"), indent=1)
for s in BASE:
    print("==", s, "baseline Fri IS", {k: round(v, 3) for k, v in OUT[s]["baseline"]["Fri"]["IS"].items()},
          "tranche IS", {k: round(v, 3) for k, v in OUT[s]["baseline"]["tranche"]["IS"].items()})
    print(f"   {'component':32s} {'Fri dCal IS/H1/H2':>22s}  {'Tr dCal IS/H1/H2':>22s}  {'Tr thirds':>24s}  Tr dCAGR  Tr dDD  Tr dUlcer")
    for label, _, _ in ABL:
        r = OUT[s][label]
        f = [r[f"Fri|{w}"]["d_calmar"] for w in ["IS", "H1", "H2"]]
        t = [r[f"tranche|{w}"]["d_calmar"] for w in ["IS", "H1", "H2"]]
        th = [r[f"tranche|{w}"]["d_calmar"] for w in ["T1_1988-99", "T2_2000-12", "T3_2013-frz"]]
        print(f"   {label:32s} {f[0]:+.3f}/{f[1]:+.3f}/{f[2]:+.3f}  {t[0]:+.3f}/{t[1]:+.3f}/{t[2]:+.3f}  "
              f"{th[0]:+.3f}/{th[1]:+.3f}/{th[2]:+.3f}  {r['tranche|IS']['d_cagr']*100:+6.2f}  "
              f"{r['tranche|IS']['d_maxdd']*100:+5.1f}  {r['tranche|IS']['d_ulcer']:+5.2f}")
print(f"runtime {time.time()-T0:.1f}s")
