"""
ROBUSTNESS REVIEW of update_2026_11/revalidate.py  --  part 1
(single-episode dependence, apples-to-apples instrument/leverage comparisons, rf-adjusted Sharpe,
 lag-2 decomposition, bootstrap seed/block sensitivity, OOS-window influence)

Uses ONLY common.py mechanics (identical to revalidate.py).  Reads revalidate.json for cross-checks.
Writes nothing except stdout.
run:  python -u update_2026_11/verify_revalidate_robustness_checks.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
P = C.panel(); IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
J = json.load(open(os.path.join(HERE, "revalidate.json")))


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def BT(lag=1, mark="mark", **kw):
    return C.backtest(P, C.strategy_weights(P, **kw), lag=lag, mark=mark)


def st(r, a=None, b=FREEZE):
    return C.stats(r, a, b)


def exc_sharpe(r, a=None, b=FREEZE):
    s = r[(r.index >= START)]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    x = (s - P["rf_d"].reindex(s.index)).dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def dd_episode(r, a=None, b=FREEZE):
    s = r[r.index >= START]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    e = (1 + s.fillna(0)).cumprod()
    dd = e / e.cummax() - 1
    t = dd.idxmin(); pk = e[:t].idxmax()
    return f"{dd.min()*100:.1f}% peak {pk.date()} trough {t.date()}"


def zero_window(r, a, b):
    r = r.copy(); m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)); r[m] = 0.0
    return r


def f(x, n=3):
    return f"{x:.{n}f}"


# ------------------------------------------------------------------ series
log("building series")
F3 = BT(lev="tqqq", voltgt=0.20, off="ief")
F3T = BT(lev="tqqq", voltgt=0.20, off="tlt")
F2 = BT(lev="qld", voltgt=0.30, off="ief")
F2T = BT(lev="qld", voltgt=0.30, off="tlt")
QQQ = C.backtest(P, pd.DataFrame({"qqq": 1.0}, index=IDX))
F3ng = BT(lev="tqqq", voltgt=0.20, gold_w=0.0)                       # analysis' "no gold" (core -> 1.0)
F2ng = BT(lev="qld", voltgt=0.30, gold_w=0.0)
F3gi = BT(lev="tqqq", voltgt=0.20, gold="ief", gold_w=0.15)          # 15% IEF instead of gold, same core
F2gi = BT(lev="qld", voltgt=0.30, gold="ief", gold_w=0.15)
F3gb = BT(lev="tqqq", voltgt=0.20, gold="bills", gold_w=0.15)        # 15% bills instead of gold, same core
F2gb = BT(lev="qld", voltgt=0.30, gold="bills", gold_w=0.15)
F3na = BT(lev="tqqq", voltgt=0.20, seasonal=False)
F2na = BT(lev="qld", voltgt=0.30, seasonal=False)
F3L2 = BT(lev="tqqq", voltgt=0.20, lag=2)
F2L2 = BT(lev="qld", voltgt=0.30, lag=2)

# ------------------------------------------------------------------ A. reproduction of headline numbers
log("A. reproduction")
jc = J["candidates"]
for name, r, key in [("F3-IEF", F3, "F3-IEF"), ("F3-TLT", F3T, "F3-TLT (scanner)"), ("F2-IEF", F2, "F2-IEF")]:
    s = st(r); js = jc[key]["windows"]["IS"]
    print(f"  {name}: IS {C.fmt(s)} | json calmar {js['calmar']:.4f} cagr {js['cagr']:.4f} -> "
          f"match={abs(s['calmar']-js['calmar'])<1e-4}")
print(f"  QQQ B&H IS {C.fmt(st(QQQ))}")

# ------------------------------------------------------------------ B1. which episode sets each maxDD
log("B1. drawdown episodes by window")
for name, r in [("F3-IEF", F3), ("F2-IEF", F2), ("F3-TLT", F3T), ("F3 lag2", F3L2), ("F2 lag2", F2L2)]:
    print(f"  {name:8s} IS {dd_episode(r)} | H1 {dd_episode(r, None, '2006-12-31')} | "
          f"H2 {dd_episode(r, '2007-01-01')} | 1991->freeze {dd_episode(r, '1991-01-01')}")

# ------------------------------------------------------------------ B2. F2 vs F3 single-episode dependence
log("B2. F2 vs F3: windows that drop single episodes")
WINS = {"IS 1988-freeze": (None, FREEZE), "1991-freeze (drop 1989-90)": ("1991-01-01", FREEZE),
        "H1 1988-2006": (None, "2006-12-31"), "1991-2006": ("1991-01-01", "2006-12-31"),
        "H2 2007-freeze": ("2007-01-01", FREEZE), "1988-2021 (drop 2022+)": (None, "2021-12-31"),
        "FULL to 11-17": (None, LATEST)}
for wn, (a, b) in WINS.items():
    s3, s2 = st(F3, a, b), st(F2, a, b)
    print(f"  {wn:28s} F3 Cal {f(s3['calmar'],2)} (CAGR {s3['cagr']*100:5.1f} DD {s3['maxdd']*100:5.1f} Sh {f(s3['sharpe'],2)}"
          f" exSh {f(exc_sharpe(F3, a, b),2)}) | F2 Cal {f(s2['calmar'],2)} (CAGR {s2['cagr']*100:5.1f} DD "
          f"{s2['maxdd']*100:5.1f} Sh {f(s2['sharpe'],2)} exSh {f(exc_sharpe(F2, a, b),2)}) | F2-F3 dCal "
          f"{s2['calmar']-s3['calmar']:+.3f}")
print("  leave-one-crisis-out (both series flat inside the window), IS Calmar:")
for cn, (a, b) in C.CRISES.items():
    if pd.Timestamp(a) > FREEZE:
        continue
    s3, s2 = st(zero_window(F3, a, b)), st(zero_window(F2, a, b))
    print(f"    drop {cn:20s} F3 {f(s3['calmar'],3)} F2 {f(s2['calmar'],3)} F2-F3 {s2['calmar']-s3['calmar']:+.3f}")

# ------------------------------------------------------------------ B3. IEF vs TLT single-episode dependence
log("B3. IEF vs TLT off-leg by window")
for wn, (a, b) in {**WINS, "2007-2021": ("2007-01-01", "2021-12-31"), "1988-1993+1995-2021": (None, None)}.items():
    if wn == "1988-1993+1995-2021":
        x3, xT = zero_window(zero_window(F3, "1994-01-01", "1994-12-31"), "2022-01-01", "2030-01-01"), \
            zero_window(zero_window(F3T, "1994-01-01", "1994-12-31"), "2022-01-01", "2030-01-01")
        s3, sT = st(x3), st(xT); wn = "IS with 1994 and 2022+ flat"
    else:
        s3, sT = st(F3, a, b), st(F3T, a, b)
    print(f"  {wn:28s} F3-IEF Cal {f(s3['calmar'],2)} (CAGR {s3['cagr']*100:5.1f} DD {s3['maxdd']*100:5.1f}) | "
          f"F3-TLT Cal {f(sT['calmar'],2)} (CAGR {sT['cagr']*100:5.1f} DD {sT['maxdd']*100:5.1f}) | dCal {s3['calmar']-sT['calmar']:+.3f}")
print("  F3-TLT H1 maxDD episode:", dd_episode(F3T, None, "2006-12-31"), "| F3-IEF H1:", dd_episode(F3, None, "2006-12-31"))
print("  leave-one-crisis-out IS Calmar IEF minus TLT:")
for cn, (a, b) in C.CRISES.items():
    if pd.Timestamp(a) > FREEZE:
        continue
    print(f"    drop {cn:20s} dCal {st(zero_window(F3, a, b))['calmar']-st(zero_window(F3T, a, b))['calmar']:+.3f}")
yr3 = (1 + F3[F3.index >= START]).groupby(F3[F3.index >= START].index.year).prod() - 1
yrT = (1 + F3T[F3T.index >= START]).groupby(F3T[F3T.index >= START].index.year).prod() - 1
dy = (yr3 - yrT)
print(f"  calendar years IEF beats TLT: {(dy > 0).sum()}/{len(dy)}; mean annual diff {dy.mean()*100:+.2f}%;"
      f" worst 3 yrs for TLT vs IEF: {dy.sort_values(ascending=False).head(3).round(4).to_dict()}")

# ------------------------------------------------------------------ B4. gold: apples-to-apples (same core leverage)
log("B4. gold vs IEF/bills at the SAME 85% core (isolates gold from de-levering)")
print(f"  {'variant':34s}{'IS CAGR':>8}{'IS DD':>7}{'IS Cal':>7}{'H1 Cal':>7}{'H2 Cal':>7}{'FULL Cal':>9}{'2022':>8}")
for name, r in [("F3 gold15 (default)", F3), ("F3 no gold (core 1.00, analysis)", F3ng),
                ("F3 15% IEF instead of gold", F3gi), ("F3 15% bills instead of gold", F3gb),
                ("F2 gold15 (default)", F2), ("F2 no gold (core 1.00, analysis)", F2ng),
                ("F2 15% IEF instead of gold", F2gi), ("F2 15% bills instead of gold", F2gb)]:
    s, h1, h2, fu = st(r), st(r, None, "2006-12-31"), st(r, "2007-01-01"), st(r, None, LATEST)
    print(f"  {name:34s}{s['cagr']*100:7.2f}%{s['maxdd']*100:6.1f}%{s['calmar']:7.3f}{h1['calmar']:7.3f}"
          f"{h2['calmar']:7.3f}{fu['calmar']:9.3f}{C.period_return(r,'2022-01-01','2022-12-31')*100:7.1f}%")
for lev, rg, rng, rgi in [("F3", F3, F3ng, F3gi), ("F2", F2, F2ng, F2gi)]:
    print(f"  {lev}: CAGR gold15 - nogold {(st(rg)['cagr']-st(rng)['cagr'])*100:+.2f}%  vs  gold15 - IEF15 "
          f"{(st(rg)['cagr']-st(rgi)['cagr'])*100:+.2f}% ; IS Calmar gold15 - IEF15 {st(rg)['calmar']-st(rgi)['calmar']:+.3f}"
          f" (H1 {st(rg,None,'2006-12-31')['calmar']-st(rgi,None,'2006-12-31')['calmar']:+.3f}, "
          f"H2 {st(rg,'2007-01-01')['calmar']-st(rgi,'2007-01-01')['calmar']:+.3f})")
print("  gold-vs-IEF15 effect by decade (F3, Calmar & CAGR diff):")
for dn, (a, b) in C.DECADES.items():
    sa, sb = C.stats(F3, a, min(pd.Timestamp(b), FREEZE), start=None), C.stats(F3gi, a, min(pd.Timestamp(b), FREEZE), start=None)
    print(f"    {dn:8s} dCal {sa['calmar']-sb['calmar']:+.3f}  dCAGR {(sa['cagr']-sb['cagr'])*100:+.2f}%")
gold_px = (1 + P["gold_phys"].fillna(0))[(IDX >= START) & (IDX <= FREEZE)].prod()
ief_px = (1 + P["ief"].fillna(0))[(IDX >= START) & (IDX <= FREEZE)].prod()
yrs = (FREEZE - START).days / 365.25
print(f"  gold_phys CAGR 1988-freeze {gold_px**(1/yrs)-1:.4f}  vs IEF {ief_px**(1/yrs)-1:.4f}")

# ------------------------------------------------------------------ B5. Aug-Sep: how many years carry it
log("B5. Aug-Sep rule: per-year contribution")
rows = []
for y in range(1988, 2026):
    a, b = f"{y}-08-01", f"{y}-10-10"          # Aug-Sep weights + the 1-week lag tail
    rows.append(dict(y=y, F3=C.period_return(F3, a, b) - C.period_return(F3na, a, b),
                     F2=C.period_return(F2, a, b) - C.period_return(F2na, a, b)))
Y = pd.DataFrame(rows).set_index("y")
for lev in ["F3", "F2"]:
    s = Y[lev]
    print(f"  {lev}: years Aug-Sep-on beats off {int((s>0).sum())}/{len(s)} (ties {int((s.abs()<1e-9).sum())}); "
          f"median {s.median()*100:+.2f}%  top-3 helps {s.nlargest(3).round(4).to_dict()}  top-3 hurts {s.nsmallest(3).round(4).to_dict()}")
for lev, r_on, r_off in [("F3", F3, F3na), ("F2", F2, F2na)]:
    base = st(r_on)["calmar"] - st(r_off)["calmar"]
    jk = []
    for y in range(1988, 2026):
        a, b = f"{y}-01-01", f"{y}-12-31"
        jk.append((y, st(zero_window(r_on, a, b))["calmar"] - st(zero_window(r_off, a, b))["calmar"]))
    jk = pd.Series(dict(jk))
    print(f"  {lev}: IS dCalmar (on-off) {base:+.3f}; leave-one-year-out range [{jk.min():+.3f} ({jk.idxmin()}), "
          f"{jk.max():+.3f} ({jk.idxmax()})]; years whose removal makes it <= 0: {list(jk[jk<=0].index)}")

# ------------------------------------------------------------------ C. rf-adjusted Sharpe
log("C. Sharpe on excess returns (r - rf) vs the analysis' rf-ignored Sharpe")
for wn, (a, b) in [("IS", (None, FREEZE)), ("H1", (None, "2006-12-31")), ("H2", ("2007-01-01", FREEZE)),
                   ("FULL", (None, LATEST))]:
    print(f"  {wn:5s} raw Sharpe F2 {st(F2,a,b)['sharpe']:.3f} F3 {st(F3,a,b)['sharpe']:.3f} (d {st(F2,a,b)['sharpe']-st(F3,a,b)['sharpe']:+.3f})"
          f" | excess Sharpe F2 {exc_sharpe(F2,a,b):.3f} F3 {exc_sharpe(F3,a,b):.3f} (d {exc_sharpe(F2,a,b)-exc_sharpe(F3,a,b):+.3f})"
          f" | QQQ exSh {exc_sharpe(QQQ,a,b):.3f}")

# ------------------------------------------------------------------ D. lag-2 by half
log("D. lag-2 decomposition")
for lev, r1, r2 in [("F3", F3, F3L2), ("F2", F2, F2L2)]:
    for wn, (a, b) in [("IS", (None, FREEZE)), ("H1", (None, "2006-12-31")), ("H2", ("2007-01-01", FREEZE)),
                       ("1991-freeze", ("1991-01-01", FREEZE))]:
        s1, s2 = st(r1, a, b), st(r2, a, b)
        print(f"  {lev} {wn:12s} Cal {s1['calmar']:.3f}->{s2['calmar']:.3f} (d {s2['calmar']-s1['calmar']:+.3f}, "
              f"{(s2['calmar']/s1['calmar']-1)*100:+.0f}%) | CAGR {s1['cagr']*100:.2f}->{s2['cagr']*100:.2f} | "
              f"DD {s1['maxdd']*100:.1f}->{s2['maxdd']*100:.1f} | Sharpe {s1['sharpe']:.3f}->{s2['sharpe']:.3f} "
              f"(d {s2['sharpe']-s1['sharpe']:+.3f}) | exSh d {exc_sharpe(r2,a,b)-exc_sharpe(r1,a,b):+.3f}")

# ------------------------------------------------------------------ E. bootstrap: seeds, block lengths, excess Sharpe
log("E. bootstrap sensitivity (all series resampled jointly)")
SER = {"F3": F3, "F2": F2, "QQQ": QQQ, "F3T": F3T, "F3ng": F3ng, "F2ng": F2ng, "F3gi": F3gi, "F2gi": F2gi,
       "F3na": F3na, "F2na": F2na}
names = list(SER)
X = pd.concat([SER[k] for k in names] + [P["rf_d"]], axis=1)
X = X[(X.index >= START) & (X.index <= LATEST)].dropna().values
RF = X[:, -1:]; X = X[:, :-1]
PAIRS = [("F2", "F3"), ("F2", "QQQ"), ("F3", "QQQ"), ("F3", "F3T"), ("F3", "F3ng"), ("F2", "F2ng"),
         ("F3", "F3gi"), ("F2", "F2gi"), ("F3", "F3na"), ("F2", "F2na")]
ix_of = {k: i for i, k in enumerate(names)}


def boot_all(block, seed, n):
    T = len(X); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    out = {p: {m: np.empty(n) for m in ("cagr", "calmar", "sharpe", "exsharpe", "maxdd")} for p in PAIRS}
    for i in range(n):
        st_ = rng.integers(0, T, nb)
        ix = (st_[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = X[ix]; rf = RF[ix]
        e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        cal = g / np.abs(dd)
        sh = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
        xs = s - rf
        exs = xs.mean(axis=0) / xs.std(axis=0, ddof=1) * np.sqrt(252)
        for p in PAIRS:
            a, b = ix_of[p[0]], ix_of[p[1]]
            out[p]["cagr"][i] = g[a] - g[b]; out[p]["calmar"][i] = cal[a] - cal[b]
            out[p]["sharpe"][i] = sh[a] - sh[b]; out[p]["exsharpe"][i] = exs[a] - exs[b]
            out[p]["maxdd"][i] = dd[a] - dd[b]
    return out


summ = {}
for block in (21, 63, 126, 252):
    for seed in ((0, 1, 2) if block == 63 else (0,)):
        o = boot_all(block, seed, 1000 if (block, seed) != (63, 0) else 2000)
        for p in PAIRS:
            for m, v in o[p].items():
                summ.setdefault((p, m), []).append((block, seed, float(v.mean()), float(np.percentile(v, 2.5)),
                                                    float(np.percentile(v, 97.5)), float((v > 0).mean())))
        log(f"  block {block} seed {seed} done")
for p in PAIRS:
    for m in ("cagr", "calmar", "sharpe", "exsharpe", "maxdd"):
        if p in [("F3", "F3T"), ("F3", "F3na"), ("F2", "F2na")] and m in ("exsharpe",):
            continue
        rows_ = summ[(p, m)]
        txt = "  ".join(f"b{b}s{s}:{mu:+.3f}[{lo:+.3f},{hi:+.3f}]P{pp:.2f}" for b, s, mu, lo, hi, pp in rows_)
        print(f"  {p[0]}-{p[1]:5s} {m:8s} {txt}")

# ------------------------------------------------------------------ F. exposure-matched 2x vs 3x
log("F. exposure-matched 2x vs 3x")
LEVX = {"tqqq": 3.0, "qld": 2.0}
for lev, vt in [("qld", 0.30), ("tqqq", 0.20), ("tqqq", 0.15), ("tqqq", 0.14), ("tqqq", 0.13)]:
    w = C.strategy_weights(P, lev=lev, voltgt=vt)
    hw = C.held_weights(P, w)[lev] * LEVX[lev]
    hw = hw[(hw.index >= START) & (hw.index <= FREEZE)]
    r = C.backtest(P, w)
    s, h1, h2 = st(r), st(r, None, "2006-12-31"), st(r, "2007-01-01")
    print(f"  {lev} vt{vt:.2f}: mean NDX notional IS {hw.mean():.3f} | IS {C.fmt(s)} exSh {exc_sharpe(r):.3f} | "
          f"H1 Cal {h1['calmar']:.3f} H2 Cal {h2['calmar']:.3f}")

# ------------------------------------------------------------------ G. OOS window influence on walk-forward test (H2 vs H2_FULL)
log("G. walk-forward early->late picks: test on 2007-freeze vs 2007-latest")
for lev, pk in [("tqqq", dict(voltgt=0.20, chop_thr=0.30, chop_pos=0.35, gold_w=0.25)),
                ("qld", dict(voltgt=0.35, chop_thr=0.25, chop_pos=0.35, gold_w=0.25))]:
    rp = BT(lev=lev, **pk)
    rd = F3 if lev == "tqqq" else F2
    for wn, b in [("H2 2007-freeze", FREEZE), ("H2_FULL 2007-11-17", LATEST)]:
        sp, sd = st(rp, "2007-01-01", b), st(rd, "2007-01-01", b)
        print(f"  {lev} pick {pk} {wn}: pick Cal {sp['calmar']:.3f} (DD {sp['maxdd']*100:.1f}%, {dd_episode(rp,'2007-01-01',b)})"
              f" vs default {sd['calmar']:.3f} (DD {sd['maxdd']*100:.1f}%) gap {sp['calmar']-sd['calmar']:+.3f}")
log("done")
