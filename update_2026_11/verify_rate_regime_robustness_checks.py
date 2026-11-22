"""
verify_rate_regime_robustness_checks.py -- adversarial ROBUSTNESS review of update_2026_11/rate_regime.py

Independent re-implementation of the specialist's off-leg policies (same common.py mechanics), then:
  A  re-derive headline numbers (IS / H1 / H2 / windows) and compare with rate_regime.json
  B  risk-off share of days (C1)
  C  Spearman rank correlation of Calmar H1 vs H2 (C4); two-fold cross-validation regret
  D  alternative splits (thirds, decades) - how often each policy beats static IEF
  E  start-date sensitivity (is F3's verdict decided by the synthetic 1989-90 drawdown?)
  F  leave-one-episode-out (neutralise one crisis window in BOTH series) - single-episode dependence
  G  drawdown metrics that are not a single number: Ulcer / Martin ratio, CDaR95, mean of 5 worst episodes
  H  bootstrap sensitivity to block length and seed (bond trend, TLT, bills vs IEF), incl. P(TLT maxDD worse)
  I  Reality Check on RISK-ADJUSTED metrics (Sharpe, Calmar), not only mean log return
  J  OOS sub-window returns for every main policy (C9 'every policy -17.2%')
  K  rolling 5y windows: share in which bond trend beats IEF

Writes NOTHING inside the project except this script; JSON goes to argv[1] (scratch).
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

T0 = time.time()
OUT = sys.argv[1] if len(sys.argv) > 1 else None
P = C.panel()
idx = P.index
FREEZE = C.FREEZE
LAST = idx[-1]
H1 = ("1988-01-01", "2006-12-31")
H2 = ("2007-01-01", str(FREEZE.date()))
SAVED = json.load(open(os.path.join(HERE, "rate_regime.json")))
RES = {}


def ma(s, n):
    return s.rolling(n, min_periods=n).mean()


tnx = P["tnx"]
ief_px = (1 + P["ief"].fillna(0)).cumprod()
gsci = P["gsci"].ffill()
SIG = {
    "durA_tnx200_bills": tnx > ma(tnx, 200),
    "durB_tnx200_shy": tnx > ma(tnx, 200),
    "corr63_bills": P["ndx_ret"].rolling(63, min_periods=63).corr(P["ief"]) > 0,
    "corr126_bills": P["ndx_ret"].rolling(126, min_periods=126).corr(P["ief"]) > 0,
    "bondtrend200_bills": ief_px < ma(ief_px, 200),
    "infl_gsci_bills": (gsci > ma(gsci, 200)) & (gsci / gsci.shift(252) - 1 > 0),
}
for n in (100, 150, 250, 300):
    SIG[f"bondtrend{n}_bills"] = ief_px < ma(ief_px, n)


def static(**kw):
    return pd.DataFrame({k: pd.Series(float(v), index=idx) for k, v in kw.items()})


def switch(cond, short="bills", long="ief"):
    c = cond.reindex(idx).fillna(False).astype(float)
    return pd.DataFrame({long: 1.0 - c, short: c}, index=idx)


POL = {"static_ief": static(ief=1), "static_tlt": static(tlt=1), "static_shy": static(shy=1),
       "static_bills": static(bills=1), "ief50_bills50": static(ief=0.5, bills=0.5),
       "thirds_bills_shy_ief": static(bills=1/3, shy=1/3, ief=1/3)}
for k, s in SIG.items():
    POL[k] = switch(s, "shy" if k.startswith("durB") else "bills")
MAIN12 = ["static_ief", "static_tlt", "static_shy", "static_bills", "ief50_bills50", "thirds_bills_shy_ief",
          "durA_tnx200_bills", "durB_tnx200_shy", "corr63_bills", "corr126_bills", "bondtrend200_bills",
          "infl_gsci_bills"]
STRATS = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
R, W = {s: {} for s in STRATS}, {s: {} for s in STRATS}
for s in STRATS:
    for p, ow in POL.items():
        w = C.strategy_weights(P, **STRATS[s], gold="gold_phys", gold_w=0.15, off_weights=ow)
        W[s][p] = w
        R[s][p] = C.backtest(P, w, cost=C.COST, lag=1, mark="mark")
NV = len(POL) * 2
print(f"[{time.time()-T0:5.1f}s] {NV} policy backtests rebuilt", flush=True)


def IS(r):
    return r[(r.index >= C.START) & (r.index <= FREEZE)]


# ------------------------------------------------------------------ A  reproduce headline numbers
A = {}
maxabs = 0.0
for s in STRATS:
    A[s] = {}
    for p in POL:
        r = IS(R[s][p])
        f, h1, h2 = C.stats(r), C.stats(r, *H1), C.stats(r, *H2)
        A[s][p] = dict(cagr=f["cagr"], maxdd=f["maxdd"], calmar=f["calmar"], sharpe=f["sharpe"],
                       H1_calmar=h1["calmar"], H2_calmar=h2["calmar"], H1_cagr=h1["cagr"], H2_cagr=h2["cagr"],
                       H1_maxdd=h1["maxdd"], H2_maxdd=h2["maxdd"],
                       y2022=C.period_return(R[s][p], "2022-01-01", "2022-12-31"),
                       OOS=C.period_return(R[s][p], "2026-07-25", str(LAST.date())))
        sv = SAVED["results"][s].get(p)
        if sv:
            for k1, k2 in [("calmar", "calmar"), ("cagr", "cagr"), ("maxdd", "maxdd")]:
                maxabs = max(maxabs, abs(sv["IS_full"][k2] - A[s][p][k1]))
            maxabs = max(maxabs, abs(sv["IS_H1"]["calmar"] - A[s][p]["H1_calmar"]),
                         abs(sv["IS_H2"]["calmar"] - A[s][p]["H2_calmar"]))
RES["A_reproduction_max_abs_diff_vs_saved"] = maxabs
RES["A_metrics"] = A
print("A  max |diff| vs saved JSON (Calmar/CAGR/maxDD, IS/H1/H2):", f"{maxabs:.2e}")
for s in STRATS:
    b = A[s]["static_ief"]
    print(f"   {s}: alternatives with full-IS Calmar > static IEF ({b['calmar']:.4f}):",
          {p: round(A[s][p]["calmar"], 4) for p in MAIN12 if A[s][p]["calmar"] > b["calmar"]})
    print(f"   {s}: alternatives with BOTH-halves Calmar > IEF (any margin):",
          {p: (round(A[s][p]["H1_calmar"] - b["H1_calmar"], 3), round(A[s][p]["H2_calmar"] - b["H2_calmar"], 3))
           for p in POL if p != "static_ief" and A[s][p]["H1_calmar"] > b["H1_calmar"] and A[s][p]["H2_calmar"] > b["H2_calmar"]})
    print(f"   {s}: H1 Calmar ranking:", sorted(MAIN12, key=lambda p: -A[s][p]["H1_calmar"])[:5],
          "| worst H1:", sorted(MAIN12, key=lambda p: A[s][p]["H1_calmar"])[:3])
    print(f"   {s}: H1 maxDD worst 3:", sorted(((p, round(A[s][p]["H1_maxdd"]*100, 1)) for p in MAIN12), key=lambda x: x[1])[:3])
    print(f"   {s}: bond-trend family full-IS Calmar:", {n: round(A[s][f"bondtrend{n}_bills"]["calmar"], 3) for n in (100, 150, 200, 250, 300)},
          "mean", round(np.mean([A[s][f"bondtrend{n}_bills"]["calmar"] for n in (100, 150, 200, 250, 300)]), 3), "vs IEF", round(b["calmar"], 3))
    print(f"   {s}: CAGR diff bond trend - IEF: full {100*(A[s]['bondtrend200_bills']['cagr']-b['cagr']):+.2f}  "
          f"H1 {100*(A[s]['bondtrend200_bills']['H1_cagr']-b['H1_cagr']):+.2f}  H2 {100*(A[s]['bondtrend200_bills']['H2_cagr']-b['H2_cagr']):+.2f} %/yr")

# ------------------------------------------------------------------ B  risk-off share (C1)
B = {}
for s in STRATS:
    h = C.held_weights(P, W[s]["static_ief"])
    h = h[(h.index >= C.START) & (h.index <= FREEZE)]
    lev = STRATS[s]["lev"]
    off = h["ief"]
    B[s] = dict(share_days_offleg_085=float((off > 0.8499).mean()), share_days_lev_zero=float((h[lev] < 1e-12).mean()),
                avg_offleg=float(off.mean()))
RES["B_riskoff_share"] = B
print("B ", B)

# ------------------------------------------------------------------ C  rank correlation / cross-validation
Cc = {}
for s in STRATS:
    h1 = pd.Series({p: A[s][p]["H1_calmar"] for p in MAIN12})
    h2 = pd.Series({p: A[s][p]["H2_calmar"] for p in MAIN12})
    rc = float(h1.rank().corr(h2.rank()))
    pick1, pick2 = h1.idxmax(), h2.idxmax()
    Cc[s] = dict(spearman_H1_H2=rc,
                 pick_on_H1=pick1, its_H2_calmar=float(h2[pick1]), H2_rank=int(h2.rank(ascending=False)[pick1]),
                 H2_regret_vs_best=float(h2.max() - h2[pick1]),
                 pick_on_H2=pick2, its_H1_calmar=float(h1[pick2]), H1_rank=int(h1.rank(ascending=False)[pick2]),
                 H1_regret_vs_best=float(h1.max() - h1[pick2]),
                 half_avg_calmar={p: float((h1[p] + h2[p]) / 2) for p in ["static_ief", "bondtrend200_bills", "ief50_bills50", "static_tlt"]})
RES["C_rank_and_cv"] = Cc
print("C ", json.dumps(Cc, indent=0)[:1500])

# ------------------------------------------------------------------ D  alternative splits
SPLITS = {"thirds": [("1988-01-01", "2000-12-31"), ("2001-01-01", "2013-12-31"), ("2014-01-01", str(FREEZE.date()))],
          "decades": [(a, min(pd.Timestamp(b), FREEZE).strftime("%Y-%m-%d")) for a, b in C.DECADES.values()]}
D = {}
for s in STRATS:
    D[s] = {}
    for sp, wins in SPLITS.items():
        D[s][sp] = {}
        for p in MAIN12[1:] + ["bondtrend100_bills", "bondtrend250_bills"]:
            cw = [C.stats(IS(R[s][p]), a, b, start=None)["calmar"] - C.stats(IS(R[s]["static_ief"]), a, b, start=None)["calmar"] for a, b in wins]
            gw = [C.stats(IS(R[s][p]), a, b, start=None)["cagr"] - C.stats(IS(R[s]["static_ief"]), a, b, start=None)["cagr"] for a, b in wins]
            D[s][sp][p] = dict(calmar_diffs=[round(x, 3) for x in cw], calmar_wins=int(sum(x > 0 for x in cw)),
                               cagr_diffs_pct=[round(100 * x, 2) for x in gw], n=len(wins))
RES["D_alt_splits"] = D
for s in STRATS:
    for sp in SPLITS:
        print(f"D  {s} {sp}: Calmar wins vs IEF / n:",
              {p: f"{v['calmar_wins']}/{v['n']}" for p, v in D[s][sp].items()})
    print(f"   {s} thirds bondtrend200 Calmar diffs", D[s]["thirds"]["bondtrend200_bills"]["calmar_diffs"],
          "CAGR diffs", D[s]["thirds"]["bondtrend200_bills"]["cagr_diffs_pct"])
    print(f"   {s} decades bondtrend200 Calmar diffs", D[s]["decades"]["bondtrend200_bills"]["calmar_diffs"],
          "CAGR diffs", D[s]["decades"]["bondtrend200_bills"]["cagr_diffs_pct"])

# ------------------------------------------------------------------ E  start-date sensitivity
E = {}
for s in STRATS:
    E[s] = {}
    for st in ["1988-01-01", "1990-07-01", "1991-01-01", "1996-01-01", "2002-08-01"]:
        sc = {p: C.stats(IS(R[s][p]), st, None, start=None) for p in MAIN12}
        best = max(MAIN12, key=lambda p: sc[p]["calmar"])
        E[s][st] = dict(best=best, best_calmar=sc[best]["calmar"], ief=sc["static_ief"]["calmar"],
                        ief_maxdd=sc["static_ief"]["maxdd"], bondtrend=sc["bondtrend200_bills"]["calmar"],
                        bondtrend_maxdd=sc["bondtrend200_bills"]["maxdd"], tlt=sc["static_tlt"]["calmar"],
                        bills=sc["static_bills"]["calmar"],
                        n_beating_ief=int(sum(sc[p]["calmar"] > sc["static_ief"]["calmar"] for p in MAIN12[1:])))
        print(f"E  {s} start {st}: IEF Cal {E[s][st]['ief']:.3f} (DD {E[s][st]['ief_maxdd']*100:.1f}) | bondtrend {E[s][st]['bondtrend']:.3f} "
              f"(DD {E[s][st]['bondtrend_maxdd']*100:.1f}) | TLT {E[s][st]['tlt']:.3f} | best {best} {E[s][st]['best_calmar']:.3f} | #beat IEF {E[s][st]['n_beating_ief']}/11")
RES["E_start_date"] = E

# ------------------------------------------------------------------ F  leave-one-episode-out (neutralise in both)
EPIS = {"1989-90 (F3 worst)": ("1989-10-01", "1990-06-30"), "1994": ("1994-01-01", "1994-12-31"),
        "1998 LTCM": ("1998-07-01", "1998-10-31"), "dot-com": ("2000-03-01", "2002-10-31"),
        "GFC": ("2007-10-01", "2009-03-31"), "2011": ("2011-04-01", "2011-12-31"),
        "COVID": ("2020-02-15", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31"),
        "2021-11..2022-12 (whole F2 IEF DD)": ("2021-11-19", "2022-12-28")}
F = {}
for s in STRATS:
    F[s] = {}
    for e, (a, b) in EPIS.items():
        m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        row = {}
        base = C.stats(IS(R[s]["static_ief"].where(~m, 0.0)))
        for p in ["static_tlt", "static_bills", "ief50_bills50", "durA_tnx200_bills", "bondtrend200_bills", "corr126_bills"]:
            x = C.stats(IS(R[s][p].where(~m, 0.0)))
            row[p] = dict(calmar_diff=x["calmar"] - base["calmar"], cagr_diff=x["cagr"] - base["cagr"],
                          maxdd_diff=x["maxdd"] - base["maxdd"])
        row["_ief"] = dict(calmar=base["calmar"], maxdd=base["maxdd"])
        F[s][e] = row
    print(f"F  {s} Calmar diff vs IEF after neutralising one episode (bondtrend / TLT / bills / 50-50):")
    for e in EPIS:
        r_ = F[s][e]
        print(f"     {e:36s} IEF Cal {r_['_ief']['calmar']:.3f} DD {r_['_ief']['maxdd']*100:5.1f} | bt {r_['bondtrend200_bills']['calmar_diff']:+.3f}"
              f" | TLT {r_['static_tlt']['calmar_diff']:+.3f} | bills {r_['static_bills']['calmar_diff']:+.3f} | 50/50 {r_['ief50_bills50']['calmar_diff']:+.3f}")
RES["F_leave_one_episode_out"] = F


# ------------------------------------------------------------------ G  multi-episode drawdown metrics
def dd_metrics(r):
    r = r.dropna()
    e = (1 + r).cumprod()
    dd = e / e.cummax() - 1
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    ulcer = float(np.sqrt((dd ** 2).mean()))
    cdar = float(dd[dd <= dd.quantile(0.05)].mean())
    # distinct episodes: peak-to-recovery segments
    epi, cur = [], 0.0
    for v in dd.values:
        if v < 0:
            cur = min(cur, v)
        elif cur < 0:
            epi.append(cur); cur = 0.0
    if cur < 0:
        epi.append(cur)
    worst5 = sorted(epi)[:5]
    return dict(cagr=float(cagr), ulcer=ulcer, martin=float(cagr / ulcer), cdar95=cdar,
                mean_worst5=float(np.mean(worst5)), cagr_over_worst5=float(cagr / abs(np.mean(worst5))),
                worst5=[round(x * 100, 1) for x in worst5])


G = {}
for s in STRATS:
    G[s] = {}
    for per, (a, b) in {"full": (str(C.START.date()), str(FREEZE.date())), "H1": H1, "H2": H2}.items():
        G[s][per] = {}
        for p in MAIN12:
            r = IS(R[s][p])
            G[s][per][p] = dd_metrics(r[(r.index >= a) & (r.index <= b)])
    for per in ["full", "H1", "H2"]:
        g = G[s][per]
        print(f"G  {s} {per:4s} Martin(CAGR/Ulcer): IEF {g['static_ief']['martin']:.2f} bt {g['bondtrend200_bills']['martin']:.2f} "
              f"TLT {g['static_tlt']['martin']:.2f} bills {g['static_bills']['martin']:.2f} 50/50 {g['ief50_bills50']['martin']:.2f} | "
              f"CAGR/mean-worst5DD: IEF {g['static_ief']['cagr_over_worst5']:.2f} bt {g['bondtrend200_bills']['cagr_over_worst5']:.2f} "
              f"TLT {g['static_tlt']['cagr_over_worst5']:.2f} | CDaR95 IEF {g['static_ief']['cdar95']*100:.1f} bt {g['bondtrend200_bills']['cdar95']*100:.1f}"
              f" | best Martin: {max(MAIN12, key=lambda p: g[p]['martin'])}")
RES["G_multi_dd_metrics"] = G


# ------------------------------------------------------------------ H  bootstrap sensitivity
def boot(strat, pols, n=1000, block=63, seed=0):
    M = pd.concat({p: IS(R[strat][p]) for p in pols}, axis=1).dropna().values
    T, K = M.shape
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    g_, c_, d_ = np.empty((n, K)), np.empty((n, K)), np.empty((n, K))
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        e = np.cumprod(1 + M[ix], axis=0)
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        g = e[-1] ** (252 / T) - 1
        g_[i], c_[i], d_[i] = g, g / np.abs(dd), dd
    return g_, c_, d_


Hh = {}
pols = ["static_ief", "bondtrend200_bills", "static_tlt", "static_bills", "ief50_bills50"]
for s in STRATS:
    Hh[s] = {}
    for blk in (21, 63, 126, 252):
        for seed in (0, 1):
            g_, c_, d_ = boot(s, pols, n=1000, block=blk, seed=seed)
            row = {}
            for k, p in enumerate(pols[1:], start=1):
                dc = c_[:, k] - c_[:, 0]
                dg = g_[:, k] - g_[:, 0]
                row[p] = dict(p_cal_le0=float((dc <= 0).mean()), p_cal_gt007=float((dc > 0.07).mean()),
                              cal_mean=float(dc.mean()), cal_sd=float(dc.std()),
                              cagr_lo=float(np.percentile(dg, 2.5)), cagr_hi=float(np.percentile(dg, 97.5)),
                              p_dd_worse=float((d_[:, k] < d_[:, 0]).mean()))
            Hh[s][f"b{blk}_s{seed}"] = row
    print(f"H  {s}: bootstrap by block/seed  [bond trend P(Cal<=0), P(Cal>.07), sd | TLT P(Cal<=0), P(DD worse) | bills P(Cal<=0)]")
    for k, row in Hh[s].items():
        bt, tl, bi = row["bondtrend200_bills"], row["static_tlt"], row["static_bills"]
        print(f"     {k:9s} bt {bt['p_cal_le0']:.3f} {bt['p_cal_gt007']:.3f} sd {bt['cal_sd']:.3f} mean {bt['cal_mean']:+.3f} | "
              f"TLT {tl['p_cal_le0']:.3f} ddworse {tl['p_dd_worse']:.3f} | bills {bi['p_cal_le0']:.3f} CAGR [{bi['cagr_lo']*100:+.2f},{bi['cagr_hi']*100:+.2f}]")
RES["H_bootstrap_sensitivity"] = Hh
print(f"[{time.time()-T0:5.1f}s] bootstrap sensitivity done", flush=True)


# ------------------------------------------------------------------ I  risk-adjusted Reality Check
def rc_metric(strat, pols, metric, n=1000, block=63, seed=3):
    M = pd.concat({p: IS(R[strat][p]) for p in pols}, axis=1).dropna().values
    T, K = M.shape
    b = 0

    def met(X):
        if metric == "sharpe":
            return X.mean(0) / X.std(0) * np.sqrt(252)
        e = np.cumprod(1 + X, axis=0)
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        return (e[-1] ** (252 / len(X)) - 1) / np.abs(dd)

    obs = met(M); d_obs = obs[1:] - obs[b]
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    D_ = np.empty((n, K - 1))
    for i in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        m_ = met(M[ix]); D_[i] = m_[1:] - m_[b]
    stat = d_obs.max()
    # White-style recentring at the observed differential, and at the bootstrap mean (nonlinear stat)
    p_obs = float(((D_ - d_obs).max(1) >= stat).mean())
    p_mean = float(((D_ - D_.mean(0)).max(1) >= stat).mean())
    best = pols[1:][int(np.argmax(d_obs))]
    return dict(best=best, best_diff=float(stat), p_recentre_obs=p_obs, p_recentre_bootmean=p_mean, K=K - 1)


I = {}
pool_main = MAIN12
pool_bt = ["static_ief"] + [f"bondtrend{n}_bills" for n in (100, 150, 200, 250, 300)]
for s in STRATS:
    I[s] = {f"{metric}_{nm}": rc_metric(s, pool, metric) for metric in ("sharpe", "calmar")
            for nm, pool in (("main12", pool_main), ("bondtrend_family", pool_bt))}
    print(f"I  {s}:", json.dumps(I[s]))
RES["I_riskadj_reality_check"] = I
print(f"[{time.time()-T0:5.1f}s] RC done", flush=True)

# ------------------------------------------------------------------ J  OOS sub-windows for every main policy
OOSW = {"full OOS": ("2026-07-25", str(LAST.date())), "pre-peak Jul25-Oct06": ("2026-07-25", "2026-10-06"),
        "since record Oct07-Nov17": ("2026-10-07", str(LAST.date())), "risk-off Oct19-Nov17": ("2026-10-19", str(LAST.date()))}
J = {s: {p: {k: C.period_return(R[s][p], a, b) for k, (a, b) in OOSW.items()} for p in MAIN12} for s in STRATS}
RES["J_oos_windows"] = J
for s in STRATS:
    v = [J[s][p]["since record Oct07-Nov17"] for p in MAIN12]
    print(f"J  {s} Oct07-Nov17 across 12 main policies: min {min(v)*100:.2f} max {max(v)*100:.2f};",
          {p: round(J[s][p]["since record Oct07-Nov17"] * 100, 2) for p in MAIN12})

# ------------------------------------------------------------------ K  rolling 5-year windows
K_ = {}
for s in STRATS:
    a0 = pd.Timestamp("1988-01-01"); rows = []
    while a0 + pd.DateOffset(years=5) <= FREEZE:
        a1 = a0 + pd.DateOffset(years=5)
        x = C.stats(IS(R[s]["bondtrend200_bills"]), a0, a1, start=None)
        y = C.stats(IS(R[s]["static_ief"]), a0, a1, start=None)
        rows.append((x["calmar"] - y["calmar"], x["cagr"] - y["cagr"], x["maxdd"] - y["maxdd"]))
        a0 = a0 + pd.DateOffset(months=6)
    arr = np.array(rows)
    K_[s] = dict(n_windows=len(rows), share_calmar_better=float((arr[:, 0] > 0).mean()),
                 share_cagr_better=float((arr[:, 1] > 0).mean()), share_maxdd_better=float((arr[:, 2] > 1e-9).mean()),
                 share_maxdd_worse=float((arr[:, 2] < -1e-9).mean()), median_calmar_diff=float(np.median(arr[:, 0])))
    print(f"K  {s} rolling 5y (step 6m) bond trend vs IEF:", K_[s])
RES["K_rolling5y"] = K_

if OUT:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.integer):
            return int(o)
        return o
    json.dump(clean(RES), open(OUT, "w"), indent=1)
print(f"done [{time.time()-T0:.1f}s]")
