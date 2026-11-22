"""
r2_verify_model_hygiene_robustness_checks.py -- ROBUSTNESS-lens adversarial checks of r2_model_hygiene.
Read-only w.r.t. every project file; writes ONLY into the scratch directory given on the command line.

usage: python -u update_2026_11/r2_verify_model_hygiene_robustness_checks.py <scratch_dir> <part>
parts
  indep : independent re-implementation through common.py (NOT the specialist's engine) of the financing
          regressions (+ HAC-lag / monthly-aggregation SE sensitivity), the analytic par-bond synthetic
          (closed-form price, numerical derivatives), the month-end gold step series, and the default-config
          IS stats of panels A_div, B, C1, C2 and ALL1 on the Friday and 5-weekday tranche bases.
          Also: NDX dividend yield implied by QQQ total return vs NDX price, by year.
  alt   : exec the specialist script UNCHANGED except (i) output redirected to scratch, (ii) extra
          alternative-construction panels injected (reported, never used to overrule the specialist's rule):
            ALL1_lit   = A_lit (the brief's literal real-minus-sim fit) + B + C1
            ALL1_Bk    = A_div + B_k (level-matched duration) + C1 ;  ALL2_Bk = A_div + B_k + C2
            ALL1_divQ  = dividend-neutral model applied with QQQ TOTAL return on 1999-03-11..inception days + B + C1
            ALL1_const = constant dividend-neutral spread (pooled mean gap, no rf slope) + B + C1
          (iii) panel set trimmed to the ones needed and the information-only seed check skipped (runtime; the
          full specialist run was reproduced bit-for-bit by r2_verify_model_hygiene_robustness_rerun.py).
          Reports flips of all 48 status items for the alternative panels under the specialist's own rule code,
          R6 (IEF-TLT) Calmar by sub-window, and saves the bootstrap input series to scratch (npz).
  boot  : bootstrap-design sensitivity on the saved series:
            boot blocks <b1,b2,..>          block length (seed 0) for every bootstrap-based status item
            boot seeds <bases> <panels>     20 seeds (pooled 40k draws) for the R5|F2 Calmar-edge CI
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = os.path.abspath(sys.argv[1])
os.makedirs(SCR, exist_ok=True)
PART = sys.argv[2]
SAVED = json.load(open(os.path.join(HERE, "r2_model_hygiene.json")))
P = C.panel()
IDX = P.index
START, FREEZE = C.START, C.FREEZE
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def nw(y, X, L):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, np.sqrt(np.diag(V))


# =====================================================================================================
if PART == "indep":
    OUT = {}
    # ---------------- dividend yield implied by QQQ total return vs NDX price
    dv = (np.log1p(P["qqq_real"] + 0.0020 / 252) - np.log1p(P["ndx_ret"])).dropna()
    yr = dv.groupby(dv.index.year).sum()
    OUT["ndx_div_yield_by_year_pct"] = {int(k): round(float(v * 100), 2) for k, v in yr.items()}
    for a, b in ((1999, 2005), (2000, 2005), (2002, 2005), (2006, 2009), (2010, 2026), (2006, 2026)):
        OUT[f"ndx_div_yield_mean_{a}_{b}_pct"] = float(yr.loc[a:b].mean() * 100)
    log("NDX dividend yield (QQQ TR - NDX price), mean %/yr: " +
        " ".join(f"{a}-{b}: {OUT[f'ndx_div_yield_mean_{a}_{b}_pct']:.2f}" for a, b in ((1999, 2005), (2002, 2005), (2006, 2009), (2010, 2026))))

    # ---------------- financing regressions (independent code path, + SE sensitivity)
    FIN = {}
    for fund, lev in (("qld", 2), ("tqqq", 3)):
        real = P[f"{fund}_real"]
        sims = {"literal": P[f"{fund}_sim"],
                "divneutral": lev * (P["qqq_real"] + 0.0020 / 252) - 0.0095 / 252 - (lev - 1) * P["rf_d"]}
        FIN[fund] = {}
        for spec, sim in sims.items():
            df = pd.concat([real - sim, P["rf_d"]], axis=1).dropna()
            df.columns = ["g", "rf"]
            df = df[df.index <= FREEZE]
            X = np.column_stack([np.ones(len(df)), df.rf.values])
            res = {}
            for L in (5, 20, 60, 250):
                b, se = nw(df.g.values, X, L)
                res[f"L{L}"] = dict(alpha_ann=b[0] * 252, alpha_se_ann=se[0] * 252, beta=b[1], beta_se=se[1])
            # monthly aggregation: G_m = alpha*n_m + beta*RF_m
            gm = df.g.resample("ME").sum()
            rm = df.rf.resample("ME").sum()
            nm = df.g.resample("ME").count()
            ok = nm > 0
            Xm = np.column_stack([nm[ok].values.astype(float), rm[ok].values])
            bm, sem = nw(gm[ok].values, Xm, 3)
            res["monthly_NW3"] = dict(alpha_ann=bm[0] * 252, alpha_se_ann=sem[0] * 252, beta=bm[1], beta_se=sem[1])
            sv = SAVED["financing"][fund][spec]["pooled"]
            res["diff_vs_saved_alpha_ann"] = res["L20"]["alpha_ann"] - sv["alpha_ann"]
            res["diff_vs_saved_beta"] = res["L20"]["beta"] - sv["beta"]
            res["mean_gap_ann"] = float(df.g.mean() * 252)
            res["alpha_daily"] = float(nw(df.g.values, X, 20)[0][0])
            res["beta_raw"] = float(nw(df.g.values, X, 20)[0][1])
            FIN[fund][spec] = res
            log(f"fin {fund} {spec:10s} alpha {res['L20']['alpha_ann']*100:+.3f}%/yr beta {res['L20']['beta']:+.3f} | "
                f"SE(alpha) L5 {res['L5']['alpha_se_ann']*100:.2f} L20 {res['L20']['alpha_se_ann']*100:.2f} "
                f"L60 {res['L60']['alpha_se_ann']*100:.2f} L250 {res['L250']['alpha_se_ann']*100:.2f} monthly {res['monthly_NW3']['alpha_se_ann']*100:.2f} | "
                f"SE(beta) L20 {res['L20']['beta_se']:.3f} L250 {res['L250']['beta_se']:.3f} monthly {res['monthly_NW3']['beta_se']:.3f} | "
                f"diff vs saved alpha {res['diff_vs_saved_alpha_ann']:+.2e} beta {res['diff_vs_saved_beta']:+.2e}")
    OUT["financing"] = FIN

    # ---------------- analytic par bond: closed-form price, numerical derivatives (independent of cash-flow sums)
    def par_price(y, c, T):
        n = int(round(2 * T))
        i = y / 2.0
        ann = (1 - (1 + i) ** (-n)) / i
        return (c / 2.0) * ann + (1 + i) ** (-n)

    def dc_numeric(y, T, h=1e-5):
        y = np.asarray(y, float)
        p0 = par_price(y, y, T)
        pu = par_price(y + h, y, T)
        pd_ = par_price(y - h, y, T)
        D = -(pu - pd_) / (2 * h * p0)
        Cx = (pu + pd_ - 2 * p0) / (h * h * p0)
        return D, Cx

    SYNB = {}
    for name, ycol, T in (("ief", "tnx", 7.5), ("tlt", "tyx", 20.0)):
        y = P[ycol] / 100.0
        dy = y.diff()
        y1 = y.shift(1)
        D, Cx = dc_numeric(y1.fillna(0.05).values, T)
        D = pd.Series(np.where(y1.notna(), D, np.nan), index=IDX)
        Cx = pd.Series(np.where(y1.notna(), Cx, np.nan), index=IDX)
        real = P[f"{name}_real"]
        m = (IDX >= pd.Timestamp("2002-07-31")) & (IDX <= FREEZE)
        a = float((real - y1 / 252 + D * dy - 0.5 * Cx * dy ** 2)[m].dropna().mean())
        syn = a + y1 / 252 - D * dy + 0.5 * Cx * dy ** 2
        SYNB[name] = real.fillna(syn).fillna(0)
        pre = (IDX >= START) & (IDX < pd.Timestamp("2002-07-31"))
        sp = syn[pre].fillna(0)
        cg = float(np.prod(1 + sp.values) ** (252 / len(sp)) - 1)
        sv = SAVED["duration"][name]["pre2002"]
        log(f"bond {name}: alpha {a*252*100:+.3f}%/yr (saved {SAVED['duration'][name]['alpha_B_ann']*100:+.3f}); pre-2002 mean D "
            f"{D[pre].mean():.3f} (saved {sv['mean_D_analytic']:.3f}) C {Cx[pre].mean():.2f} (saved {sv['mean_C_analytic']:.2f}); "
            f"syn CAGR {cg*100:.3f}% (saved {sv['cagr_B']*100:.3f}%) vol {sp.std()*np.sqrt(252)*100:.2f}% (saved {sv['vol_B']*100:.2f}%)")
        OUT[f"bond_{name}"] = dict(alpha_ann=a * 252, meanD_pre=float(D[pre].mean()), cagr_pre=cg)

    # ---------------- month-end step gold (independent)
    interp = P["gold_src"].eq("monthly-interp")
    gross = (P["gold_phys"] + 0.0040 / 252).where(interp, 0.0)
    gross.iloc[0] = 0.0
    gi = gross[interp]
    mon = gi.groupby(gi.index.to_period("M"))
    mret = mon.apply(lambda s: float(np.prod(1 + s.values) - 1))
    mlast = mon.apply(lambda s: s.index.max())
    step = pd.Series(0.0, index=IDX)
    step.loc[pd.DatetimeIndex(mlast.values)] = mret.values
    gold_c1 = (step - 0.0040 / 252).where(interp, P["gold_phys"])
    gold_c1.iloc[0] = P["gold_phys"].iloc[0]
    tot_i = float(np.prod(1 + gross[interp].values) - 1)
    tot_s = float(np.prod(1 + step[interp].values) - 1)
    log(f"gold step: booking days {len(mlast)}; total interp {tot_i:+.6f} step {tot_s:+.6f} (saved {SAVED['gold']['check_total_step']:+.6f})")

    # ---------------- corrected panels through common.py
    def a_div_sleeve(f, lev):
        r = FIN[f]["divneutral"]
        adj = r["alpha_daily"] + r["beta_raw"] * P["rf_d"].fillna(0)
        return P[f"{f}_real"].fillna(P[f"{f}_sim"] + adj).fillna(0)

    PANS = {}
    base = P.copy()
    PANS["P0"] = base
    x = P.copy(); x["qld"] = a_div_sleeve("qld", 2); x["tqqq"] = a_div_sleeve("tqqq", 3); PANS["A_div"] = x
    x = P.copy(); x["ief"] = SYNB["ief"]; x["tlt"] = SYNB["tlt"]; PANS["B"] = x
    x = P.copy(); x["gold_phys"] = gold_c1; PANS["C1"] = x
    x = P.copy(); x["gold_phys"] = P["gold_phys"].where(~interp, P["gold_miners"]); PANS["C2"] = x
    x = P.copy(); x["qld"] = a_div_sleeve("qld", 2); x["tqqq"] = a_div_sleeve("tqqq", 3)
    x["ief"] = SYNB["ief"]; x["tlt"] = SYNB["tlt"]; x["gold_phys"] = gold_c1; PANS["ALL1"] = x

    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    MK = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(DAYS)}
    TIERS = {"F1": dict(lev="qqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20)}
    rows = []
    maxdiff = 0.0
    for pn, PP in PANS.items():
        for tn, kw in TIERS.items():
            w = C.strategy_weights(PP, **kw)
            rf = C.backtest(PP, w)
            rt = sum(C.backtest(PP, w, mark=MK[d]) for d in DAYS) / 5
            for basis, r in (("Fri", rf), ("tranche", rt)):
                s = C.stats(r, b=FREEZE)
                sv = SAVED["stats"][pn][basis][f"{tn}|default"]["IS"]
                d = max(abs(s["cagr"] - sv["cagr"]), abs(s["maxdd"] - sv["maxdd"]), abs(s["calmar"] - sv["calmar"]))
                maxdiff = max(maxdiff, d)
                rows.append((pn, tn, basis, s["cagr"], s["maxdd"], s["calmar"], d))
    for pn, tn, basis, cg, dd, cal, d in rows:
        print(f"   {pn:6s} {tn} {basis:7s} CAGR {cg*100:6.2f}% maxDD {dd*100:6.1f}% Calmar {cal:.3f}  |diff vs saved| {d:.1e}")
    log(f"independent common.py stats vs saved JSON: max abs diff {maxdiff:.2e} (JSON is rounded to 1e-6)")
    OUT["indep_stats"] = [dict(panel=a, tier=b, basis=c, cagr=d, maxdd=e, calmar=f, absdiff=g) for a, b, c, d, e, f, g in rows]
    OUT["indep_max_abs_diff"] = maxdiff
    json.dump(OUT, open(os.path.join(SCR, "rob_indep.json"), "w"), indent=1, default=float)

# =====================================================================================================
elif PART == "alt":
    import io, contextlib
    SRC = os.path.join(HERE, "r2_model_hygiene.py")
    code = open(SRC, encoding="utf-8").read()
    OUTP = os.path.join(SCR, "rob_alt_rerun.json")
    # runtime: keep only the panels needed (the full specialist set was already reproduced bit-for-bit by
    # r2_verify_model_hygiene_robustness_rerun.py); the information-only seed check is skipped (own 20-seed check
    # is in part 'boot').
    KEEP = ["P0", "A_stress", "B", "B_k", "ALL1", "ALL2", "ALL1_lit", "ALL1_Bk", "ALL2_Bk", "ALL1_divQ", "ALL1_const"]
    rep = [('OUT_JSON = os.path.join(HERE, "r2_model_hygiene.json")', f'OUT_JSON = {OUTP!r}'),
           ('PRIMARY = ["A_div", "B", "C1", "C2", "ALL1", "ALL2"]',
            '_ORIG = list(PANELS)\nPANELS.update(__EXTRA__(globals()))\n'
            f'PANELS = {{k_: v_ for k_, v_ in PANELS.items() if k_ in {KEEP!r}}}\n'
            'PRIMARY = [p_ for p_ in ["A_div", "B", "C1", "C2", "ALL1", "ALL2"] if p_ in PANELS]'),
           ('SENS = ["A_lit", "A_stress", "B_k"]', 'SENS = [p_ for p_ in ["A_lit", "A_stress", "B_k"] if p_ in PANELS]'),
           ('for pn, fb in FLIPS.items():', 'for pn, fb in []:')]
    for old, new in rep:
        assert code.count(old) == 1, old
        code = code.replace(old, new)

    def extra(G):
        PP = G["P"]; SL = G["SLEEVES_A"]; SYN = G["SYN"]; FIN = G["FIN"]; pw = G["panel_with"]
        gc1, gc2 = G["gold_c1"], G["gold_c2"]
        out = {}
        out["ALL1_lit"] = pw(**SL["A_lit"], ief=SYN["ief"]["B"], tlt=SYN["tlt"]["B"], gold_phys=gc1)
        out["ALL1_Bk"] = pw(**SL["A_div"], ief=SYN["ief"]["B_k"], tlt=SYN["tlt"]["B_k"], gold_phys=gc1)
        out["ALL2_Bk"] = pw(**SL["A_div"], ief=SYN["ief"]["B_k"], tlt=SYN["tlt"]["B_k"], gold_phys=gc2)
        sq, sc = {}, {}
        for f, lev in (("qld", 2), ("tqqq", 3)):
            p = FIN[f]["divneutral"]["pooled"]
            adj = p["alpha_daily"] + p["beta"] * PP["rf_d"].fillna(0)
            dnsim = lev * (PP["qqq_real"] + 0.0020 / 252) - 0.0095 / 252 - (lev - 1) * PP["rf_d"]
            basesim = dnsim.fillna(PP[f"{f}_sim"])          # QQQ total return where it exists (1999-03-11+)
            sq[f] = PP[f"{f}_real"].fillna(basesim + adj).fillna(0).values
            sc[f] = G["lev_sleeve"](f, G["ZERO"] + p["mean_gap_ann"] / 252)   # constant spread, no rf slope
        out["ALL1_divQ"] = pw(**sq, ief=SYN["ief"]["B"], tlt=SYN["tlt"]["B"], gold_phys=gc1)
        out["ALL1_const"] = pw(**sc, ief=SYN["ief"]["B"], tlt=SYN["tlt"]["B"], gold_phys=gc1)
        return out

    g = {"__file__": SRC, "__name__": "__rerun__", "__EXTRA__": extra}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, SRC, "exec"), g)
    log("specialist code executed with extra panels")
    open(os.path.join(SCR, "rob_alt_stdout.txt"), "w", encoding="utf-8").write(buf.getvalue())
    FL, CONC, SR, SER, st = g["FLIPS"], g["CONC"], g["SR"], g["SER"], g["st"]
    EXTRA = [k for k in g["PANELS"] if k not in g["_ORIG"]]
    # identity of the (kept) original panels' flips with the saved JSON
    for pn in g["_ORIG"]:
        if pn == "P0" or pn not in g["PANELS"]:
            continue
        for b in ("Fri", "tranche"):
            a1 = sorted(FL[pn][b].keys())
            a2 = sorted(SAVED["flips_vs_P0"][pn][b].keys())
            assert a1 == a2, (pn, b, a1, a2)
    log("original panels' flips identical to saved JSON")
    print("\n  FLIPS (48 items, specialist rule code) for alternative constructions:")
    for pn in EXTRA:
        for b in ("Fri", "tranche"):
            fl = FL[pn][b]
            print(f"   {pn:10s} {b:7s}: {len(fl)} flips " + "; ".join(f"{k}: {v['P0']}->{v['corrected']}" for k, v in fl.items()))
    print("\n  KEY ITEMS for alternative constructions (IS)")
    for pn in ["P0", "ALL1", "ALL2", "A_stress"] + EXTRA:
        for b in ("Fri", "tranche"):
            c = CONC[pn][b]
            r1 = c["R1_2x_vs_3x"]
            r5 = c["R5_edge_vs_QQQ"]
            r6 = c["R6_IEF_vs_TLT"]
            print(f"   {pn:10s} {b:7s} P(xsF2>F3) {r1['boot_xsharpe']['p_gt0']:.3f} | F3-F2 CAGR lo {r1['boot_cagr']['lo95']*100:+.2f} | "
                  f"R5 F2 dCal lo {r5['F2']['boot_calmar']['lo95']:+.4f} P {r5['F2']['boot_calmar']['p_gt0']:.3f} | "
                  f"R5 F3 dCal lo {r5['F3']['boot_calmar']['lo95']:+.4f} | R6 H1 F1 {r6['F1']['d_calmar']['H1']:+.3f} "
                  f"F2 {r6['F2']['d_calmar']['H1']:+.3f} F3 {r6['F3']['d_calmar']['H1']:+.3f} | F2 IS CAGR "
                  f"{SR[pn][b]['F2|default']['IS']['cagr']*100:.2f} F3 {SR[pn][b]['F3|default']['IS']['cagr']*100:.2f} "
                  f"Cal F2 {SR[pn][b]['F2|default']['IS']['calmar']:.3f} F3 {SR[pn][b]['F3|default']['IS']['calmar']:.3f}")
    # R6 IEF - TLT Calmar by sub-window
    SUBW = {"1988-2002.07(syn bonds)": ("1988-01-01", "2002-07-30"), "2002.08-2006": ("2002-07-31", "2006-12-31"),
            "H1 1988-2006": ("1988-01-01", "2006-12-31"), "H2 2007-frz": ("2007-01-01", str(FREEZE.date())),
            "2007-2021 (H2 ex-2022+)": ("2007-01-01", "2021-12-31"), "2022-frz": ("2022-01-01", str(FREEZE.date())),
            "IS ex 2022 (1988-2021)": ("1988-01-01", "2021-12-31")}
    R6W = {}
    print("\n  R6 IEF-minus-TLT Calmar by sub-window (default vs off=tlt)")
    for pn in ["P0", "B", "B_k", "ALL1", "ALL2", "ALL1_Bk", "ALL2_Bk"]:
        for b in ("Fri", "tranche"):
            row = {}
            for tn in ("F1", "F2", "F3"):
                ri, rt = SER[(pn, b, tn, "default")], SER[(pn, b, tn, "tlt")]
                row[tn] = {w: st(ri, a0, b0)["calmar"] - st(rt, a0, b0)["calmar"] for w, (a0, b0) in SUBW.items()}
            R6W[f"{pn}|{b}"] = row
            print(f"   {pn:8s} {b:7s} " + " || ".join(tn + ": " + " ".join(f"{w.split(' ')[0]} {v:+.3f}" for w, v in row[tn].items()) for tn in row))
    # save bootstrap inputs
    keep = list(g["PANELS"])
    arr = {}
    for pn in keep:
        for b in ("Fri", "tranche"):
            for tn in ("F1", "F2", "F3"):
                for cn in ("default", "goldIEF"):
                    arr[f"{pn}__{b}__{tn}|{cn}"] = SER[(pn, b, tn, cn)]
            arr[f"{pn}__{b}__QQQ|bh"] = SER[(pn, b, "QQQ", "bh")]
    np.savez_compressed(os.path.join(SCR, "rob_series.npz"), **arr)
    stat0 = {pn: {b: g["flatten_status"](CONC[pn][b]) for b in ("Fri", "tranche")} for pn in keep}
    json.dump(dict(flips={pn: FL[pn] for pn in EXTRA}, r6_windows=R6W, status=stat0,
                   key={pn: {b: dict(R5F2=CONC[pn][b]["R5_edge_vs_QQQ"]["F2"]["boot_calmar"],
                                     R1xs=CONC[pn][b]["R1_2x_vs_3x"]["boot_xsharpe"]) for b in ("Fri", "tranche")} for pn in keep}),
              open(os.path.join(SCR, "rob_alt.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("saved rob_series.npz / rob_alt.json")

# =====================================================================================================
elif PART == "boot":
    Z = np.load(os.path.join(SCR, "rob_series.npz"))
    RF = P["rf_d"].fillna(0).values
    M = (IDX >= START) & (IDX <= FREEZE)
    PANS = sorted({k.split("__")[0] for k in Z.files})
    NAMES = ["F1|default", "F2|default", "F3|default", "F1|goldIEF", "F2|goldIEF", "F3|goldIEF", "QQQ|bh"]

    def boot(cols, xs, n=2000, block=63, seed=0):
        X = np.column_stack([c[M] for c in cols])
        Lg = np.log1p(X)
        XE = np.column_stack([c[M] - RF[M] for c in xs]) if xs else None
        T = len(X)
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(T / block))
        ar = np.arange(block)[None, :]
        cg = np.empty((n, X.shape[1])); dd = np.empty((n, X.shape[1]))
        xsr = np.empty((n, len(xs))) if xs else None
        for i in range(n):
            s0 = rng.integers(0, T, nb)
            ix = (s0[:, None] + ar).ravel()[:T] % T
            cl = np.cumsum(Lg[ix], axis=0)
            cg[i] = np.expm1(cl[-1] * 252 / T)
            dd[i] = np.expm1((cl - np.maximum.accumulate(cl, axis=0)).min(axis=0))
            if xs:
                xe = XE[ix]
                xsr[i] = xe.mean(axis=0) / xe.std(axis=0, ddof=1) * np.sqrt(252)
        return cg, dd, cg / np.abs(dd), xsr

    def q(v):
        return dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p=float((v > 0).mean()))

    def items(pn, b, block, seed):
        cols = [Z[f"{pn}__{b}__{k}"] for k in NAMES]
        cg, dd, cal, xs = boot(cols, [cols[1], cols[2]], block=block, seed=seed)
        j = {k: i for i, k in enumerate(NAMES)}
        out, num = {}, {}
        x = q(xs[:, 0] - xs[:, 1]); out["R1|xsharpe_tie"] = 0.05 < x["p"] < 0.95; num["R1|xsharpe_P"] = x["p"]
        x = q(cg[:, j["F3|default"]] - cg[:, j["F2|default"]]); out["R1|f3_cagr_sig"] = x["lo"] > 0; num["R1|f3f2_cagr_lo"] = x["lo"]
        for tn in ("F1", "F2", "F3"):
            x = q(cal[:, j[f"{tn}|default"]] - cal[:, j[f"{tn}|goldIEF"]]); out[f"R3|{tn}|not_sig"] = 0.05 < x["p"] < 0.95
            num[f"R3|{tn}|P"] = x["p"]
            x = q(cal[:, j[f"{tn}|default"]] - cal[:, j["QQQ|bh"]]); out[f"R5|{tn}|calmar_ci_excl0"] = x["lo"] > 0 or x["hi"] < 0
            num[f"R5|{tn}|cal_lo"] = x["lo"]
            x = q(cg[:, j[f"{tn}|default"]] - cg[:, j["QQQ|bh"]]); out[f"R5|{tn}|cagr_ci_excl0"] = x["lo"] > 0 or x["hi"] < 0
            num[f"R5|{tn}|cagr_lo"] = x["lo"]
        return out, num

    RES = {}
    which = sys.argv[3] if len(sys.argv) > 3 else "blocks"
    if which == "blocks":
        CORR = ["A_stress", "ALL1", "ALL2", "ALL1_lit", "ALL1_Bk", "ALL1_divQ", "ALL1_const"]
        BLOCKS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [63, 21, 126, 252]
        for block in BLOCKS:
            RES[block] = {}
            for b in ("Fri", "tranche"):
                s0, n0 = items("P0", b, block, 0)
                RES[block][f"P0|{b}"] = dict(status=s0, num=n0)
                for pn in CORR:
                    s1, n1 = items(pn, b, block, 0)
                    fl = {k: (s0[k], s1[k]) for k in s0 if s0[k] != s1[k]}
                    RES[block][f"{pn}|{b}"] = dict(status=s1, num=n1, flips=fl)
                    print(f"  block {block:3d} {b:7s} {pn:10s} flips: {fl if fl else 'none'} | P(xs) {n0['R1|xsharpe_P']:.3f}->{n1['R1|xsharpe_P']:.3f} "
                          f"R5F2 lo {n0['R5|F2|cal_lo']:+.4f}->{n1['R5|F2|cal_lo']:+.4f} R5F3 lo {n0['R5|F3|cal_lo']:+.4f}->{n1['R5|F3|cal_lo']:+.4f} "
                          f"F3-F2 cagr lo {n0['R1|f3f2_cagr_lo']*100:+.2f}->{n1['R1|f3f2_cagr_lo']*100:+.2f}", flush=True)
            log(f"block {block} done")
        json.dump(RES, open(os.path.join(SCR, f"rob_boot_blocks_{'_'.join(map(str, BLOCKS))}.json"), "w"), indent=1, default=float)
    else:
        # 20 seeds, 2 columns only (identical index draws): R5|F2 Calmar edge, pooled 40k draws
        SEEDS = range(20)
        BASES = sys.argv[4].split(",") if len(sys.argv) > 4 else ["tranche"]
        PNS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["P0", "ALL1", "ALL2", "A_stress", "ALL1_lit", "ALL1_Bk"]
        for b in BASES:
            for pn in PNS:
                f2 = Z[f"{pn}__{b}__F2|default"]; qq = Z[f"{pn}__{b}__QQQ|bh"]
                los, pool = [], []
                for sd in SEEDS:
                    cg, dd, cal, _ = boot([f2, qq], [], seed=sd)
                    v = cal[:, 0] - cal[:, 1]
                    los.append(float(np.percentile(v, 2.5)))
                    pool.append(v)
                pool = np.concatenate(pool)
                RES[f"{pn}|{b}"] = dict(lo_by_seed=los, frac_seeds_excl0=float(np.mean(np.array(los) > 0)),
                                        lo_pooled_40k=float(np.percentile(pool, 2.5)), p_gt0_pooled=float((pool > 0).mean()),
                                        lo_seed_sd=float(np.std(los, ddof=1)))
                print(f"  {b:7s} {pn:10s} R5|F2 Calmar edge lo95: seed0 {los[0]:+.4f} | mean over 20 seeds {np.mean(los):+.4f} "
                      f"(sd {np.std(los, ddof=1):.4f}) | seeds with lo>0: {sum(l > 0 for l in los)}/20 | pooled-40k lo {RES[f'{pn}|{b}']['lo_pooled_40k']:+.4f} "
                      f"P(>0) {RES[f'{pn}|{b}']['p_gt0_pooled']:.4f}", flush=True)
        json.dump(RES, open(os.path.join(SCR, f"rob_boot_seeds_{'_'.join(BASES)}_{'_'.join(PNS)}.json"), "w"), indent=1, default=float)
    log("done")
