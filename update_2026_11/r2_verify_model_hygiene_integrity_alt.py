"""
INTEGRITY REVIEW of r2_model_hygiene.py -- are the two formal tranche flips robust to the two modelling choices they
rest on?  (information only; no new tuned parameter: every input is one the specialist already estimated)
  ALL1_lit = A_lit + B + C1      (the brief's literal real-minus-sim regression instead of the dividend-neutral one)
  ALL2_lit = A_lit + B + C2
  ALL1_k   = A_div + B_k + C1    (level-matched duration, which also passed the gate and fits the ETFs better)
  ALL2_k   = A_div + B_k + C2
All 48 status items (same definitions, IS 1988-01-01..FREEZE, bootstrap 63-day blocks, 2000 draws, seed 0) vs P0.
Prints only.  usage: python -u update_2026_11/r2_verify_model_hygiene_integrity_alt.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
P = C.panel()
IDX = P.index
FREEZE, START = C.FREEZE, C.START
J = json.load(open(os.path.join(HERE, "r2_model_hygiene.json")))


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ---- financing point estimates (OLS; identical to the specialist's, verified in the _checks script)
qg = P["qqq_real"] + 0.0020 / 252
SL = {}
for f, lev in (("qld", 2), ("tqqq", 3)):
    for spec, sim in (("lit", P[f"{f}_sim"]), ("div", lev * qg - 0.0095 / 252 - (lev - 1) * P["rf_d"])):
        d = pd.DataFrame({"g": P[f"{f}_real"] - sim, "rf": P["rf_d"]}).dropna()
        d = d[d.index <= FREEZE]
        a, b = np.linalg.lstsq(np.column_stack([np.ones(len(d)), d.rf.values]), d.g.values, rcond=None)[0]
        SL[(spec, f)] = P[f"{f}_real"].fillna(P[f"{f}_sim"] + a + b * P["rf_d"]).fillna(0)
        assert abs(a * 252 - J["financing"][f]["literal" if spec == "lit" else "divneutral"]["pooled"]["alpha_ann"]) < 1e-5


# ---- duration B / B_k
def par_dc(y, T, h=1e-4):
    n = int(round(2 * T)); i = y / 2.0
    price = lambda yy: (y / 2.0) * (1 - (1 + yy / 2) ** (-n)) / (yy / 2) + (1 + yy / 2) ** (-n)
    dmod = (1 + i) / i * (1 - (1 + i) ** (-n)) / 2.0 / (1 + i)
    return dmod, (price(y + h) - 2 * price(y) + price(y - h)) / h ** 2 / price(y)


BD = {}
CAL_A = pd.Timestamp("2002-07-31")
for name, ycol, T in (("ief", "tnx", 7.5), ("tlt", "tyx", 20.0)):
    yd = P[ycol] / 100.0; dy = yd.diff(); y1 = yd.shift(1); carry = y1 / 252
    D, Cv = par_dc(y1.fillna(0.05).values, T)
    D = pd.Series(np.where(y1.notna(), D, np.nan), index=IDX); Cv = pd.Series(np.where(y1.notna(), Cv, np.nan), index=IDX)
    real = P[f"{name}_real"]; mcal = (IDX >= CAL_A) & (IDX <= FREEZE)
    aB = (real - carry + D * dy - 0.5 * Cv * dy ** 2)[mcal].dropna().mean()
    df = pd.concat([real - carry - 0.5 * Cv * dy ** 2, -D * dy], axis=1)[mcal].dropna()
    kk = np.linalg.lstsq(np.column_stack([np.ones(len(df)), df.iloc[:, 1].values]), df.iloc[:, 0].values, rcond=None)[0][1]
    ak = (real - carry + kk * D * dy - 0.5 * kk ** 2 * Cv * dy ** 2)[mcal].dropna().mean()
    assert abs(kk - J["duration"][name]["k"]) < 1e-5
    BD[("B", name)] = real.fillna(aB + carry - D * dy + 0.5 * Cv * dy ** 2).fillna(0)
    BD[("B_k", name)] = real.fillna(ak + carry - kk * D * dy + 0.5 * kk ** 2 * Cv * dy ** 2).fillna(0)

# ---- gold C1 / C2
interp = P["gold_src"].eq("monthly-interp"); FEE = 0.0040 / 252; gp = P["gold_phys"].fillna(0)
lr = np.log1p(gp + FEE).where(interp, 0.0); lr.iloc[0] = 0.0
iv = IDX[interp.values]; last_td = pd.Series(iv, index=iv).groupby(iv.to_period("M")).max()
msum = lr[interp].groupby(IDX.to_period("M")[interp.values]).sum()
c1 = pd.Series(-FEE, index=IDX); c1[pd.DatetimeIndex(last_td.values)] = np.expm1(msum.values) - FEE
c1[~interp] = gp[~interp]; c1.iloc[0] = gp.iloc[0]
c2 = gp.where(~interp, P["gold_miners"].fillna(0))

ALT = {"ALL1_lit": {"qld": SL[("lit", "qld")], "tqqq": SL[("lit", "tqqq")], "ief": BD[("B", "ief")], "tlt": BD[("B", "tlt")], "gold_phys": c1},
       "ALL2_lit": {"qld": SL[("lit", "qld")], "tqqq": SL[("lit", "tqqq")], "ief": BD[("B", "ief")], "tlt": BD[("B", "tlt")], "gold_phys": c2},
       "ALL1_k": {"qld": SL[("div", "qld")], "tqqq": SL[("div", "tqqq")], "ief": BD[("B_k", "ief")], "tlt": BD[("B_k", "tlt")], "gold_phys": c1},
       "ALL2_k": {"qld": SL[("div", "qld")], "tqqq": SL[("div", "tqqq")], "ief": BD[("B_k", "ief")], "tlt": BD[("B_k", "tlt")], "gold_phys": c2},
       "ALL1_rebuilt": {"qld": SL[("div", "qld")], "tqqq": SL[("div", "tqqq")], "ief": BD[("B", "ief")], "tlt": BD[("B", "tlt")], "gold_phys": c1}}
TK = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
CF = {"default": {}, "bills": dict(off="bills"), "tlt": dict(off="tlt"), "goldIEF": dict(gold="ief"),
      "nochop": dict(chop_pos=1.0), "novol": dict(voltgt=99.0)}
WKD = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])}
W = {(t, c): C.strategy_weights(P, **{**TK[t], **CF[c]}) for t in TK for c in CF}
M = (IDX >= START) & (IDX <= FREEZE)
RF = P["rf_d"].values[M]; QQQ = P["qqq"].values[M]
qs = C.stats(P["qqq"], b=FREEZE)
WINS = {"IS": (None, FREEZE), "H1": (None, "2006-12-31"), "H2": ("2007-01-01", FREEZE)}


def boot(cols, xs, seed=0, n=2000, block=63, chunk=25):
    names = list(cols); X = np.column_stack([cols[k] for k in names]); Lg = np.log1p(X)
    XE = np.column_stack([cols[k] - RF for k in xs])
    T = len(X); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); ar = np.arange(block)[None, :]
    G, DD, XS = [], [], []
    for c0 in range(0, n, chunk):
        ixs = np.stack([(rng.integers(0, T, nb)[:, None] + ar).ravel()[:T] % T for _ in range(min(chunk, n - c0))])
        cl = np.cumsum(Lg[ixs], axis=1)
        G.append(np.expm1(cl[:, -1, :] * 252 / T)); DD.append(np.expm1((cl - np.maximum.accumulate(cl, axis=1)).min(axis=1)))
        xe = XE[ixs]; XS.append(xe.mean(1) / xe.std(1, ddof=1) * np.sqrt(252))
    G, DD, XS = np.vstack(G), np.vstack(DD), np.vstack(XS)
    return {"cagr": {k: G[:, j] for j, k in enumerate(names)}, "calmar": {k: G[:, j] / np.abs(DD[:, j]) for j, k in enumerate(names)},
            "xs": {k: XS[:, j] for j, k in enumerate(xs)}}


def bd(B, met, a, b):
    v = B[met][a] - B[met][b]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float((v > 0).mean())


for pn, rep in ALT.items():
    Pm = P.copy()
    for c, s in rep.items():
        Pm[c] = s.values
    SER = {}
    for (t, c), w in W.items():
        SER[("Fri", t, c)] = C.backtest(Pm, w)
        SER[("tranche", t, c)] = sum(C.backtest(Pm, w, mark=WKD[d]) for d in WKD) / 5.0
    for b in ("Fri", "tranche"):
        S = {(t, c, wn): C.stats(SER[(b, t, c)], *WINS[wn]) for t in TK for c in CF for wn in WINS}
        f = {}
        cols = {f"{t}|{c}": SER[(b, t, c)].values[M] for t in TK for c in ("default", "goldIEF")}
        cols["QQQ|bh"] = QQQ
        B = boot(cols, ["F2|default", "F3|default"])
        lo, hi, p = bd(B, "xs", "F2|default", "F3|default"); f["R1|xsharpe_tie"] = bool(0.05 < p < 0.95)
        lo, hi, p = bd(B, "cagr", "F3|default", "F2|default"); f["R1|f3_cagr_sig"] = bool(lo > 0)
        f["R1|f2_shallower"] = bool(S[("F2", "default", "IS")]["maxdd"] > S[("F3", "default", "IS")]["maxdd"])
        r5f2 = None
        for t in TK:
            for wn in WINS:
                f[f"R2|{t}|{wn}"] = int(np.sign(S[(t, "default", wn)]["calmar"] - S[(t, "bills", wn)]["calmar"]))
                f[f"R6|{t}|{wn}"] = int(np.sign(S[(t, "default", wn)]["calmar"] - S[(t, "tlt", wn)]["calmar"]))
            lo, hi, p = bd(B, "calmar", f"{t}|default", f"{t}|goldIEF"); f[f"R3|{t}|not_sig"] = bool(0.05 < p < 0.95)
            for comp, cn in (("chop_rule", "nochop"), ("vol_target", "novol")):
                d = {wn: S[(t, "default", wn)]["calmar"] - S[(t, cn, wn)]["calmar"] for wn in WINS}
                f[f"R4|{t}|{comp}|sign_IS"] = int(np.sign(d["IS"]))
                f[f"R4|{t}|{comp}|helps_both_halves"] = bool(d["H1"] > 0 and d["H2"] > 0)
            lo, hi, p = bd(B, "calmar", f"{t}|default", "QQQ|bh")
            f[f"R5|{t}|calmar_ci_excl0"] = bool(lo > 0 or hi < 0)
            if t == "F2":
                r5f2 = (lo, hi, p)
            f[f"R5|{t}|calmar_sign"] = int(np.sign(S[(t, "default", "IS")]["calmar"] - qs["calmar"]))
            lo, hi, p = bd(B, "cagr", f"{t}|default", "QQQ|bh")
            f[f"R5|{t}|cagr_ci_excl0"] = bool(lo > 0 or hi < 0)
            f[f"R5|{t}|cagr_sign"] = int(np.sign(S[(t, "default", "IS")]["cagr"] - qs["cagr"]))
        s0 = J["status_P0"][b]
        assert set(f) == set(s0), (set(f) ^ set(s0))
        flips = {k: (s0[k], f[k]) for k in f if f[k] != s0[k]}
        is_ = {t: S[(t, "default", "IS")] for t in TK}
        p0 = {t: J["stats"]["P0"][b][f"{t}|default"]["IS"] for t in TK}
        r6 = S[("F2", "default", "H1")]["calmar"] - S[("F2", "tlt", "H1")]["calmar"]
        log(f"{pn:12s} {b:7s} flips vs P0: {flips if flips else 'none'} | R5 F2 Calmar CI [{r5f2[0]:+.4f},{r5f2[1]:+.3f}] P {r5f2[2]:.3f} | "
            f"R6 F2 H1 {r6:+.3f} | dCAGR/dCal " + " ".join(f"{t} {(is_[t]['cagr']-p0[t]['cagr'])*100:+.2f}/{is_[t]['calmar']-p0[t]['calmar']:+.3f}" for t in TK))
        if pn == "ALL1_rebuilt":
            js = J["stats"]["ALL1"][b]
            mx = max(abs(is_[t][m] - js[f"{t}|default"]["IS"][m]) for t in TK for m in ("cagr", "maxdd", "calmar"))
            log(f"   ALL1 rebuilt in this script vs JSON ALL1 IS: max abs diff {mx:.1e}")
log("done")
