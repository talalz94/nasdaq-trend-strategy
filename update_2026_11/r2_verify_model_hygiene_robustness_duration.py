"""
r2_verify_model_hygiene_robustness_duration.py -- ROBUSTNESS-lens checks of r2_model_hygiene, part 2.
Read-only w.r.t. project files; writes only to the scratch dir given on the command line.

  (1) Which pre-2002 duration model does the 2002+ real-ETF evidence support?  By yield bucket of y(t-1), slope k of
      (real - carry) on the model's price component (-D dy + 0.5 C dy^2) for: fixed calibration (panel_validation),
      pure analytic par bond B (IEF 7.5y / TLT 20y), and TLT as a 25y par bond.  If B's k is ~constant (>1) across
      yield buckets, the level-matched B_k is the better extrapolation to the high-yield pre-2002 era.
  (2) R6 (IEF minus TLT Calmar, H1 1988-2006 and the synthetic-only window 1988..2002-07-30) with TLT as a 25y par bond
      (IEF = B), through common.py, Friday + 5-weekday tranche.
  (3) Gold miners (C2) under a weekly-DRIFT convention (holdings drift between Friday marks, reset at marks) vs the
      constant-mix convention, to test the claim that C2's CAGR gain is an artifact of daily constant-mix.
  (4) Episode location of each tier's IS maxDD (P0 and corrected), i.e. which data era drives Calmar.

usage: python -u update_2026_11/r2_verify_model_hygiene_robustness_duration.py <scratch_dir>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = os.path.abspath(sys.argv[1])
os.makedirs(SCR, exist_ok=True)
P = C.panel()
IDX = P.index
START, FREEZE = C.START, C.FREEZE
SAVED = json.load(open(os.path.join(HERE, "r2_model_hygiene.json")))
VAL = json.load(open(os.path.join(HERE, "panel_validation.json")))
OUT = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def par_price(y, c, T):
    n = int(round(2 * T))
    i = y / 2.0
    return (c / 2.0) * (1 - (1 + i) ** (-n)) / i + (1 + i) ** (-n)


def dc(y, T, h=1e-5):
    y = np.asarray(y, float)
    p0, pu, pdn = par_price(y, y, T), par_price(y + h, y, T), par_price(y - h, y, T)
    return -(pu - pdn) / (2 * h * p0), (pu + pdn - 2 * p0) / (h * h * p0)


CAL = (IDX >= pd.Timestamp("2002-07-31")) & (IDX <= FREEZE)
PRE = (IDX >= START) & (IDX < pd.Timestamp("2002-07-31"))

# ------------------------------------------------------------------ (1) yield-bucket slope test
BK = [(0.0, 0.025), (0.025, 0.035), (0.035, 0.045), (0.045, 0.20)]
SYN = {}
for name, real_col, ycol, T in (("ief_B", "ief", "tnx", 7.5), ("tlt_B", "tlt", "tyx", 20.0), ("tlt_25y", "tlt", "tyx", 25.0)):
    y = P[ycol] / 100.0
    dy = y.diff()
    y1 = y.shift(1)
    D, Cx = dc(y1.fillna(0.05).values, T)
    D = pd.Series(np.where(y1.notna(), D, np.nan), index=IDX)
    Cx = pd.Series(np.where(y1.notna(), Cx, np.nan), index=IDX)
    real = P[f"{real_col}_real"]
    comp = -D * dy + 0.5 * Cx * dy ** 2
    cfx = VAL[f"{real_col}_calib"]
    comp_fix = -cfx["duration"] * dy + 0.5 * cfx["convexity"] * dy ** 2
    tgt = real - y1 / 252
    res = {}
    for lo, hi in BK:
        m = CAL & (y1 >= lo) & (y1 < hi)
        d = pd.concat([tgt, comp, comp_fix], axis=1)[m].dropna()
        d.columns = ["t", "c", "f"]
        if len(d) < 100:
            continue
        kb = np.polyfit(d.c.values, d.t.values, 1)[0]
        kf = np.polyfit(d.f.values, d.t.values, 1)[0]
        res[f"{lo*100:.1f}-{hi*100:.1f}%"] = dict(days=int(len(d)), k_analytic=float(kb), k_fixed=float(kf),
                                                   meanD_analytic=float(D[m].mean()), D_fixed=cfx["duration"])
    a = float((tgt - comp)[CAL].dropna().mean())
    syn = a + y1 / 252 + comp
    SYN[name] = P[f"{real_col}_real"].fillna(syn).fillna(0)
    sp = syn[PRE].fillna(0)
    OUT[f"bucket_{name}"] = dict(buckets=res, pre2002_meanD=float(D[PRE].mean()),
                                 pre2002_vol=float(sp.std() * np.sqrt(252)),
                                 pre2002_cagr=float(np.prod(1 + sp.values) ** (252 / len(sp)) - 1))
    log(f"{name}: pre-2002 mean D {D[PRE].mean():.2f}, syn vol {sp.std()*np.sqrt(252)*100:.2f}% | slope of real on model, by y(t-1): " +
        " ".join(f"[{k}] n={v['days']} k_analytic {v['k_analytic']:.3f} (D {v['meanD_analytic']:.2f}) k_fixed {v['k_fixed']:.3f}" for k, v in res.items()))

# ------------------------------------------------------------------ (2) R6 with TLT as 25y par bond
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MK = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(DAYS)}
TIERS = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
WIN = {"IS": (START, FREEZE), "H1": (START, pd.Timestamp("2006-12-31")), "syn_1988_2002.07": (START, pd.Timestamp("2002-07-30"))}


def runs(PP, kw):
    w = C.strategy_weights(PP, **kw)
    return {"Fri": C.backtest(PP, w), "tranche": sum(C.backtest(PP, w, mark=MK[d]) for d in DAYS) / 5}


PB = P.copy()
PB["ief"] = SYN["ief_B"]
PB["tlt"] = SYN["tlt_B"]
P25 = P.copy()
P25["ief"] = SYN["ief_B"]
P25["tlt"] = SYN["tlt_25y"]
R6 = {}
for lab, PP in (("P0", P), ("B (TLT 20y)", PB), ("B, TLT 25y", P25)):
    R6[lab] = {}
    for tn, kw in TIERS.items():
        ri = runs(PP, kw)
        rt = runs(PP, dict(kw, off="tlt"))
        for b in ("Fri", "tranche"):
            R6[lab][f"{tn}|{b}"] = {w: C.stats(ri[b], a, z)["calmar"] - C.stats(rt[b], a, z)["calmar"] for w, (a, z) in WIN.items()}
    log(f"R6 IEF-TLT Calmar [{lab}]: " + " | ".join(f"{k} IS {v['IS']:+.3f} H1 {v['H1']:+.3f} syn-only {v['syn_1988_2002.07']:+.3f}" for k, v in R6[lab].items()))
for k in R6["P0"]:
    p0, b, b25 = R6["P0"][k]["H1"], R6["B (TLT 20y)"][k]["H1"], R6["B, TLT 25y"][k]["H1"]
    print(f"   H1 reduction vs P0 {k:10s}: B {100*(p0-b)/p0:5.1f}%   TLT-25y {100*(p0-b25)/p0:5.1f}%  "
          f"(saved B_k: {100*(p0-SAVED['conclusions']['B_k'][k.split('|')[1]]['R6_IEF_vs_TLT'][k.split('|')[0]]['d_calmar']['H1'])/p0:5.1f}%)")
OUT["R6_tlt25"] = R6

# ------------------------------------------------------------------ (3) gold miners under weekly drift
interp = P["gold_src"].eq("monthly-interp")
PC2 = P.copy()
PC2["gold_phys"] = P["gold_phys"].where(~interp, P["gold_miners"])


def drift_bt(PP, w, mark, cost=C.COST):
    cols = list(w.columns)
    R = np.column_stack([PP[c].fillna(0).values for c in cols])
    W = w[cols].values
    m = mark.values.astype(bool)
    cur = np.zeros(len(cols))
    rp = np.zeros(len(R))
    pend = 0.0
    for t in range(len(R)):
        g = float(cur @ R[t])
        rp[t] = g - pend
        pend = 0.0
        tot = cur.sum() + g
        endw = cur * (1 + R[t]) / tot if tot > 0 else cur
        if m[t]:
            pend = np.abs(W[t] - endw).sum() * cost       # charged on the day the new weights start (as common.backtest)
            cur = W[t].copy()
        else:
            cur = endw
    return pd.Series(rp, index=PP.index)


GD = {}
for tn, kw in TIERS.items():
    w0 = C.strategy_weights(P, **kw)
    out = {}
    for lab, PP in (("P0", P), ("C2", PC2)):
        out[lab] = dict(cmix=C.stats(C.backtest(PP, w0), b=FREEZE)["cagr"], drift=C.stats(drift_bt(PP, w0, P["mark"]), b=FREEZE)["cagr"])
    GD[tn] = dict(cmix_delta_pp=100 * (out["C2"]["cmix"] - out["P0"]["cmix"]), drift_delta_pp=100 * (out["C2"]["drift"] - out["P0"]["drift"]),
                  cagr_P0_cmix=out["P0"]["cmix"], cagr_P0_drift=out["P0"]["drift"])
    log(f"C2 (miners) CAGR effect, Friday {tn}: constant-mix {GD[tn]['cmix_delta_pp']:+.3f}pp vs weekly-drift {GD[tn]['drift_delta_pp']:+.3f}pp "
        f"(P0 CAGR cmix {out['P0']['cmix']*100:.2f}% drift {out['P0']['drift']*100:.2f}%)")
# arithmetic vs geometric sleeve returns 1988..2000-08
m88 = interp & (IDX >= START)
for nm, s in (("gold interp", P["gold_phys"]), ("miners", P["gold_miners"])):
    x = s[m88].values
    log(f"{nm} 1988..2000-08: arithmetic {x.mean()*252*100:+.2f}%/yr, geometric {(np.prod(1+x)**(252/len(x))-1)*100:+.2f}%/yr, vol {x.std()*np.sqrt(252)*100:.1f}%")
OUT["gold_drift"] = GD

# ------------------------------------------------------------------ (4) where is each tier's IS maxDD?
def dd_episode(r):
    r = r[(r.index >= START) & (r.index <= FREEZE)]
    e = (1 + r).cumprod()
    dd = e / e.cummax() - 1
    tr = dd.idxmin()
    pk = e[:tr].idxmax()
    rec = e[tr:][e[tr:] >= e[pk]]
    return str(pk.date()), str(tr.date()), str(rec.index[0].date()) if len(rec) else "none", float(dd.min())


EP = {}
PALL1 = P.copy()
EP["note"] = "peak / trough / recovery / depth"
for tn, kw in TIERS.items():
    rr = runs(P, kw)
    for b in ("Fri", "tranche"):
        EP[f"{tn}|{b}"] = dd_episode(rr[b])
    rq = runs(PB, kw)
    EP[f"{tn}|Fri|B"] = dd_episode(rq["Fri"])
log("IS maxDD episodes (P0): " + " | ".join(f"{k}: {v}" for k, v in EP.items() if k != "note"))
OUT["maxdd_episodes"] = EP
json.dump(OUT, open(os.path.join(SCR, "rob_duration.json"), "w"), indent=1, default=float)
log("saved")
