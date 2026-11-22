"""
INTEGRITY REVIEW of r2_model_hygiene.py -- bootstrap-based status items (R1, R3, R5), independent numerical path
(cumprod equity curves instead of log-cumsum; same RNG call sequence so seed 0 must reproduce the saved numbers).
Also:
  * extra seeds (0..19, information only) for the flipped item R5|F2|calmar_ci_excl0 on the tranche basis
  * P0' = P0 with the pre-2002 fixed IEF/TLT calibration re-fit on data <= FREEZE only (the panel's was fit through
    2026-11-17): do any P0 statuses, and hence any flips, depend on that post-freeze leakage?
Needs <scratch_dir>/verify_series.npz written by r2_verify_model_hygiene_integrity_checks.py.
Writes nothing except <scratch_dir>/verify_boot.json
usage: python -u update_2026_11/r2_verify_model_hygiene_integrity_boot.py <scratch_dir>
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
scratch = sys.argv[1]
Z = np.load(os.path.join(scratch, "verify_series.npz"))
P = C.panel()
IDX = P.index
FREEZE, START = C.FREEZE, C.START
M = (IDX >= START) & (IDX <= FREEZE)
RF = P["rf_d"].values[M]
QQQ = P["qqq"].values[M]
J = json.load(open(os.path.join(HERE, "r2_model_hygiene.json")))
TIERS = ("F1", "F2", "F3")
PANELS = ["P0", "A_div", "A_lit", "A_stress", "B", "B_k", "C1", "C2", "ALL1", "ALL2"]
OUT = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def boot(cols, xs, seed, n=2000, block=63):
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    XE = np.column_stack([cols[k] - RF for k in xs])
    T = len(X)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    ar = np.arange(block)
    G = np.empty((n, len(names))); DD = np.empty((n, len(names))); XS = np.empty((n, len(xs)))
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + ar[None, :]).ravel()[:T] % T
        eq = np.cumprod(1.0 + X[ix], axis=0)
        G[i] = eq[-1] ** (252.0 / T) - 1
        DD[i] = (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)
        xe = XE[ix]
        XS[i] = xe.mean(0) / xe.std(0, ddof=1) * np.sqrt(252)
    B = {"cagr": {k: G[:, j] for j, k in enumerate(names)}, "maxdd": {k: DD[:, j] for j, k in enumerate(names)}}
    B["calmar"] = {k: G[:, j] / np.abs(DD[:, j]) for j, k in enumerate(names)}
    B["xs"] = {k: XS[:, j] for j, k in enumerate(xs)}
    return B


def bd(B, met, a, b):
    v = B[met][a] - B[met][b]
    return dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p=float((v > 0).mean()))


def cols_for(get):
    c = {}
    for t in TIERS:
        c[f"{t}|default"] = get(t, "default")[M]
        c[f"{t}|goldIEF"] = get(t, "goldIEF")[M]
    c["QQQ|bh"] = QQQ
    return c


def boot_status(B):
    f, num = {}, {}
    xs = bd(B, "xs", "F2|default", "F3|default"); cg = bd(B, "cagr", "F3|default", "F2|default")
    f["R1|xsharpe_tie"] = bool(0.05 < xs["p"] < 0.95); f["R1|f3_cagr_sig"] = bool(cg["lo"] > 0)
    num["R1|xs"] = xs; num["R1|cagr"] = cg
    for t in TIERS:
        g = bd(B, "calmar", f"{t}|default", f"{t}|goldIEF")
        f[f"R3|{t}|not_sig"] = bool(0.05 < g["p"] < 0.95); num[f"R3|{t}"] = g
        q = bd(B, "calmar", f"{t}|default", "QQQ|bh"); qc = bd(B, "cagr", f"{t}|default", "QQQ|bh")
        f[f"R5|{t}|calmar_ci_excl0"] = bool(q["lo"] > 0 or q["hi"] < 0)
        f[f"R5|{t}|cagr_ci_excl0"] = bool(qc["lo"] > 0 or qc["hi"] < 0)
        num[f"R5|{t}|calmar"] = q; num[f"R5|{t}|cagr"] = qc
    return f, num


# ------------------------------------------------------------- 1. seed-0 replication of every bootstrap item
ST, NUM = {}, {}
maxd = 0.0
worst = None
for pn in PANELS:
    for b in ("Fri", "tranche"):
        get = lambda t, c: Z[f"{pn}|{b}|{t}" if c == "default" else f"{pn}|{b}|{t}|goldIEF"]
        B = boot(cols_for(get), ["F2|default", "F3|default"], seed=0)
        f, num = boot_status(B)
        ST[(pn, b)] = f; NUM[(pn, b)] = num
        jc = J["conclusions"][pn][b]
        pairs = [(num["R1|xs"]["p"], jc["R1_2x_vs_3x"]["boot_xsharpe"]["p_gt0"]),
                 (num["R1|cagr"]["lo"], jc["R1_2x_vs_3x"]["boot_cagr"]["lo95"]),
                 (num["R1|cagr"]["hi"], jc["R1_2x_vs_3x"]["boot_cagr"]["hi95"])]
        for t in TIERS:
            pairs += [(num[f"R3|{t}"]["p"], jc["R3_gold_vs_IEF"][t]["boot_calmar"]["p_gt0"]),
                      (num[f"R5|{t}|calmar"]["lo"], jc["R5_edge_vs_QQQ"][t]["boot_calmar"]["lo95"]),
                      (num[f"R5|{t}|calmar"]["hi"], jc["R5_edge_vs_QQQ"][t]["boot_calmar"]["hi95"]),
                      (num[f"R5|{t}|cagr"]["lo"], jc["R5_edge_vs_QQQ"][t]["boot_cagr"]["lo95"]),
                      (num[f"R5|{t}|cagr"]["hi"], jc["R5_edge_vs_QQQ"][t]["boot_cagr"]["hi95"])]
        d = max(abs(x - y) for x, y in pairs)
        if d > maxd:
            maxd, worst = d, (pn, b)
        js = J["status_P0"][b] if pn == "P0" else None
        if js is not None:
            mm = [k for k in f if f[k] != js[k]]
            assert not mm, f"P0 status mismatch {mm}"
    log(f"{pn} bootstraps done")
log(f"seed-0 replication: max |mine - JSON| over all bootstrap summary numbers = {maxd:.2e} (worst {worst})")
OUT["seed0_max_abs_diff"] = maxd
FL = {}
for pn in PANELS[1:]:
    for b in ("Fri", "tranche"):
        fl = {k: (ST[("P0", b)][k], ST[(pn, b)][k]) for k in ST[("P0", b)] if ST[("P0", b)][k] != ST[(pn, b)][k]}
        if fl:
            FL[f"{pn}|{b}"] = fl
log(f"bootstrap-item flips vs P0 (independent): {FL}")
OUT["boot_flips"] = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in FL.items()}
# key numbers quoted in the claims
for pn in ("P0", "ALL1", "ALL2", "A_stress"):
    for b in ("Fri", "tranche"):
        n_ = NUM[(pn, b)]
        print(f"  {pn:8s} {b:7s} R1 P(F2>F3 xs) {n_['R1|xs']['p']:.3f} F3-F2 CAGR CI [{n_['R1|cagr']['lo']*100:+.2f},{n_['R1|cagr']['hi']*100:+.2f}] | "
              + " ".join(f"R3 {t} P {n_[f'R3|{t}']['p']:.2f}" for t in TIERS) + " | "
              + " ".join(f"R5 {t} Cal CI [{n_[f'R5|{t}|calmar']['lo']:+.4f},{n_[f'R5|{t}|calmar']['hi']:+.3f}] P {n_[f'R5|{t}|calmar']['p']:.3f}"
                         f" CAGR CI [{n_[f'R5|{t}|cagr']['lo']*100:+.2f},{n_[f'R5|{t}|cagr']['hi']*100:+.2f}]" for t in TIERS))
r3p = [NUM[(pn, b)][f"R3|{t}"]["p"] for pn in PANELS for b in ("Fri", "tranche") for t in TIERS]
log(f"R3 P(gold>IEF slot) range over 10 panels x 2 bases x 3 tiers: {min(r3p):.3f}..{max(r3p):.3f}")
OUT["R3_p_range"] = [min(r3p), max(r3p)]

# ------------------------------------------------------------- 2. P0' (fixed bond calibration re-fit on <= FREEZE)
syn = {}
for name, ycol in (("ief", "tnx"), ("tlt", "tyx")):
    yd = P[ycol] / 100.0
    dy = yd.diff(); carry = yd.shift(1) / 252
    real = P[f"{name}_real"]
    df = pd.concat([real, carry, dy], axis=1).dropna(); df.columns = ["r", "c", "dy"]
    df = df[df.index <= FREEZE]
    a_, D_, C_ = np.linalg.lstsq(np.column_stack([np.ones(len(df)), -df.dy.values, 0.5 * df.dy.values ** 2]),
                                 (df.r - df.c).values, rcond=None)[0]
    syn[name] = real.fillna(a_ + carry - D_ * dy + 0.5 * C_ * dy ** 2).fillna(0)
    log(f"P0' {name}: refit a {a_*252*100:+.3f}%/yr D {D_:.3f} C {C_:.1f} (panel: see panel_validation.json)")
Pp = P.copy()
Pp["ief"] = syn["ief"].values; Pp["tlt"] = syn["tlt"].values
CF = {"default": {}, "bills": dict(off="bills"), "tlt": dict(off="tlt"), "goldIEF": dict(gold="ief"),
      "nochop": dict(chop_pos=1.0), "novol": dict(voltgt=99.0)}
TK = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
WKD = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])}
SERp = {}
for t in TK:
    for c in CF:
        w = C.strategy_weights(P, **{**TK[t], **CF[c]})
        SERp[("Fri", t, c)] = C.backtest(Pp, w)
        SERp[("tranche", t, c)] = sum(C.backtest(Pp, w, mark=WKD[d]) for d in WKD) / 5.0
qs = C.stats(P["qqq"], b=FREEZE)


def nb_status(b):
    f = {}
    S = lambda t, c, w: C.stats(SERp[(b, t, c)], *{"IS": (None, FREEZE), "H1": (None, "2006-12-31"), "H2": ("2007-01-01", FREEZE)}[w])
    f["R1|f2_shallower"] = bool(S("F2", "default", "IS")["maxdd"] > S("F3", "default", "IS")["maxdd"])
    for t in TK:
        for w in ("IS", "H1", "H2"):
            f[f"R2|{t}|{w}"] = int(np.sign(S(t, "default", w)["calmar"] - S(t, "bills", w)["calmar"]))
            f[f"R6|{t}|{w}"] = int(np.sign(S(t, "default", w)["calmar"] - S(t, "tlt", w)["calmar"]))
        for comp, cn in (("chop_rule", "nochop"), ("vol_target", "novol")):
            d = {w: S(t, "default", w)["calmar"] - S(t, cn, w)["calmar"] for w in ("IS", "H1", "H2")}
            f[f"R4|{t}|{comp}|sign_IS"] = int(np.sign(d["IS"]))
            f[f"R4|{t}|{comp}|helps_both_halves"] = bool(d["H1"] > 0 and d["H2"] > 0)
        f[f"R5|{t}|calmar_sign"] = int(np.sign(S(t, "default", "IS")["calmar"] - qs["calmar"]))
        f[f"R5|{t}|cagr_sign"] = int(np.sign(S(t, "default", "IS")["cagr"] - qs["cagr"]))
    return f


P0p = {}
for b in ("Fri", "tranche"):
    get = lambda t, c: SERp[(b, t, c)].values
    Bp = boot(cols_for(get), ["F2|default", "F3|default"], seed=0)
    fb, nump = boot_status(Bp)
    fs = {**nb_status(b), **fb}
    diff = {k: (J["status_P0"][b][k], fs[k]) for k in fs if fs[k] != J["status_P0"][b][k]}
    P0p[b] = dict(status_diff_vs_P0=diff, R5_F2_calmar=nump["R5|F2|calmar"],
                  IS={t: C.stats(SERp[(b, t, "default")], b=FREEZE) for t in TK},
                  R6_F2_H1=C.stats(SERp[(b, "F2", "default")], b="2006-12-31")["calmar"] - C.stats(SERp[(b, "F2", "tlt")], b="2006-12-31")["calmar"])
    log(f"P0' {b}: status differences vs P0: {diff}; R5 F2 Calmar CI [{nump['R5|F2|calmar']['lo']:+.4f},{nump['R5|F2|calmar']['hi']:+.3f}] "
        f"P {nump['R5|F2|calmar']['p']:.3f}; IS " + " ".join(f"{t} {P0p[b]['IS'][t]['cagr']*100:.2f}/{P0p[b]['IS'][t]['maxdd']*100:.1f}/"
                                                            f"{P0p[b]['IS'][t]['calmar']:.3f}" for t in TK)
        + f"; R6 F2 H1 {P0p[b]['R6_F2_H1']:+.3f}")
OUT["P0_prime"] = P0p

# ------------------------------------------------------------- 3. extra seeds: moved to r2_verify_model_hygiene_integrity_seeds.py
# (the 20-seed loop here exceeded the 9-minute command budget under CPU contention; the chunked version there is faster)
json.dump(OUT, open(os.path.join(scratch, "verify_boot.json"), "w"), indent=1, default=float)
log("done")
