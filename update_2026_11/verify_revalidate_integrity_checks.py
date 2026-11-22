"""
INTEGRITY REVIEW of update_2026_11/revalidate.py  --  step 2: independent checks.

Read-only: loads panel.parquet + revalidate.json, writes nothing except stdout.
  K1  signal causality: recompute ema100/above/vol20/er20/mark from ndx; truncation test
  K2  independent numpy re-implementation of the backtest (no common.py) vs saved headline stats
  K3  true weekly rebalancing (holdings drift between Friday marks, cost on drift trades)
      vs common.backtest's implicit daily constant-mix
  K4  pre-2000 gold: is the 'monthly' series month-END or monthly-AVERAGE?  + sensitivity of H1
  K5  excess-return (rf-adjusted) Sharpe for F2 vs F3: point, paired grid, block bootstrap
  K6  grid / plateau claims (81-config Aug-Sep-on box, one-step details, F2 box >= 0.70 share)
  K7  walk-forward: pick/default test-half percentiles; does gold 0.25 help the picks in the test half
  K8  1989-90 F3 max-drawdown decomposition (the IS/H1 max-DD episode)
  K9  OOS narrative timing (C10) and current-drawdown split (C12)
  K10 pre-2002 synthetic-treasury alpha sensitivity
  K11 ablation "helps both halves" count under July instruments; IS-only F2-vs-F3 bootstrap
usage: python update_2026_11/verify_revalidate_integrity_checks.py
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402  (used only for panel() and for re-creating the specialist's own runs)

P = C.panel()
IDX = P.index
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
J = json.load(open(os.path.join(HERE, "revalidate.json")))
CAND = J["candidates"]


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100, flush=True)


def st(r, a=None, b=None):
    """independent stats (same definitions as the specialist: CAGR by calendar days, Sharpe rf-ignored)."""
    s = r[(r.index >= START)]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    e = np.cumprod(1 + s.values)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return dict(cagr=cagr, maxdd=dd, calmar=cagr / abs(dd), sharpe=s.mean() / s.std() * np.sqrt(252))


def f(d):
    return f"CAGR {d['cagr']*100:6.2f}%  DD {d['maxdd']*100:6.1f}%  Cal {d['calmar']:.3f}  Sh {d['sharpe']:.3f}"


# =====================================================================================  K1
hdr("K1  signal causality")
ndx = P["ndx"]
ema = ndx.ewm(span=100, adjust=False).mean()
vol = ndx.pct_change().rolling(20).std() * np.sqrt(252)
er = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum().replace(0, np.nan)
mk = pd.Series(IDX.dayofweek == 4, index=IDX)
print("max|ema100 - recomputed|", float((P.ema100 - ema).abs().max()))
print("above mismatches", int((P.above != (ndx > ema)).sum()))
print("max|vol20 - recomputed|", float((P.vol20 - vol).abs().max()))
print("max|er20 - recomputed|", float((P.er20 - er).abs().max(skipna=True)))
print("mark == (weekday is Friday) mismatches", int((P["mark"].astype(bool) != mk).sum()),
      "| month mismatches", int((P["month"] != IDX.month).sum()))
rng = np.random.default_rng(1)
cut_dates = IDX[(IDX >= START)][np.sort(rng.choice(np.arange(((IDX >= START)).sum()), 40, replace=False))]
bad = 0
for D in cut_dates:
    q = ndx[ndx.index <= D]
    e2 = q.ewm(span=100, adjust=False).mean().iloc[-1]
    v2 = (q.pct_change().rolling(20).std() * np.sqrt(252)).iloc[-1]
    r2 = (q.diff(20).abs() / q.diff().abs().rolling(20).sum()).iloc[-1]
    if not (np.isclose(e2, P.at[D, "ema100"]) and np.isclose(v2, P.at[D, "vol20"]) and np.isclose(r2, P.at[D, "er20"])):
        bad += 1
print(f"truncation test (signals from data <= D equal the panel value at D) on 40 random dates: {40-bad}/40 equal")
w3 = C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_phys", off="ief")
hw3 = C.held_weights(P, w3)
chg = hw3.diff().abs().sum(axis=1) > 1e-12
prev_is_mark = P["mark"].astype(bool).shift(1).fillna(False)
print("held-weight changes on a day NOT following a Friday mark:", int((chg & ~prev_is_mark).sum()),
      "(should be 0; lag=1, weights set at Friday close, earn from next session)")

# =====================================================================================  K2
hdr("K2  independent re-implementation (pure numpy loop; no common.py mechanics)")
RET = {c: P[c].fillna(0).values for c in ["qqq", "qld", "tqqq", "ief", "tlt", "bills", "gold_phys"]}
mark = P["mark"].astype(bool).values
above = (ndx > ema).values
volv = vol.values
erv = er.values
months = IDX.month.values


def target(i, lev, vt, thr, pos, seas, gw, off):
    """target weights computed from information at close of day i only."""
    if not above[i]:
        eqw = 0.0
    else:
        sc = 1.0 if not np.isfinite(volv[i]) else min(1.0, vt / volv[i])
        if seas and months[i] in (8, 9):
            sc *= 0.5
        clean = np.isfinite(erv[i]) and erv[i] >= thr
        eqw = sc if clean else sc * pos
    core = 1.0 - gw
    return {lev: eqw * core, off: core - eqw * core, "gold_phys": gw}


def indep(lev, vt, thr=0.25, pos=0.5, seas=True, gw=0.15, off="ief", cost=3e-4, drift=False):
    n = len(IDX)
    out = np.zeros(n)
    assets = [lev, off, "gold_phys"]
    held = {a: 0.0 for a in assets}
    pending = None
    for i in range(n):
        # 1) today's return on yesterday's closing holdings (+ cost if holdings were re-set at yesterday's close)
        r = sum(held[a] * RET[a][i] for a in assets)
        if pending is not None:
            r -= pending
            pending = None
        out[i] = r
        if drift:                                    # holdings drift with prices
            tot = 1 + r if (1 + r) != 0 else 1e-12
            gross = sum(held[a] * (1 + RET[a][i]) for a in assets)
            if gross > 0:
                held = {a: held[a] * (1 + RET[a][i]) / gross * sum(held.values()) for a in assets}
        # 2) at today's close: if Friday mark, trade to target computed from today's close
        if mark[i]:
            t = target(i, lev, vt, thr, pos, seas, gw, off)
            tt = {a: t.get(a, 0.0) for a in assets}
            pending = cost * sum(abs(tt[a] - held[a]) for a in assets)
            held = tt
    return pd.Series(out, index=IDX)


CHK = {"F3-IEF": dict(lev="tqqq", vt=0.20, off="ief"), "F2-IEF": dict(lev="qld", vt=0.30, off="ief"),
       "F3-TLT (scanner)": dict(lev="tqqq", vt=0.20, off="tlt"),
       "A (TQQQ vt.25, cash)": dict(lev="tqqq", vt=0.25, pos=1.0, seas=False, gw=0.0, off="bills")}
INDEP, DRIFT = {}, {}
for k, kw in CHK.items():
    INDEP[k] = indep(**kw)
    DRIFT[k] = indep(**kw, drift=True)
    for wn, (a, b) in {"IS": (None, FREEZE), "FULL": (None, LATEST), "H1": ("1988-01-01", "2006-12-31"),
                       "H2": ("2007-01-01", FREEZE)}.items():
        mine = st(INDEP[k], a, b)
        saved = CAND[k]["windows"][wn]
        print(f"  {k:22s} {wn:4s} indep {f(mine)} | saved CAGR {saved['cagr']*100:6.2f}% DD {saved['maxdd']*100:6.1f}%"
              f" Cal {saved['calmar']:.3f} | dCAGR {(mine['cagr']-saved['cagr'])*1e4:+.1f}bp")
    o = INDEP[k][INDEP[k].index > FREEZE]
    print(f"  {k:22s} OOS total indep {((1+o).prod()-1)*100:+.2f}% vs saved {CAND[k]['windows']['OOS']['total']*100:+.2f}%")

# =====================================================================================  K3
hdr("K3  true weekly rebalancing with drift (cost on actual trades incl. drift-back) vs constant-mix")
for k in CHK:
    for wn, (a, b) in {"IS": (None, FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
                       "FULL": (None, LATEST)}.items():
        cm, dr = st(INDEP[k], a, b), st(DRIFT[k], a, b)
        print(f"  {k:22s} {wn:4s} const-mix {f(cm)} | drift {f(dr)} | dCAGR {(dr['cagr']-cm['cagr'])*100:+.2f}pp"
              f" dDD {(dr['maxdd']-cm['maxdd'])*100:+.2f}pp dCal {dr['calmar']-cm['calmar']:+.3f}")
    o = DRIFT[k][DRIFT[k].index > FREEZE]
    print(f"  {k:22s} OOS drift {((1+o).prod()-1)*100:+.2f}%")

# =====================================================================================  K4
hdr("K4  pre-2000 gold: month-end vs monthly-average labelling, and H1 sensitivity")
g = P["gold_phys"]
post = g[g.index >= "2000-09-01"]
Gx = (1 + post).cumprod()
true_m = (1 + post).resample("ME").prod() - 1
pidx = post.index


def interp_from(points):
    grid = points.reindex(points.index.union(pidx)).apply(np.log).interpolate(method="time").apply(np.exp)
    d = grid.reindex(pidx).pct_change()
    return (1 + d).resample("ME").prod() - 1


for lab, pts in {"month-END points": Gx.resample("ME").last(), "monthly-AVERAGE points": Gx.resample("ME").mean()}.items():
    m = pd.concat([interp_from(pts), true_m], axis=1).dropna().iloc[2:]
    m.columns = ["interp", "true"]
    print(f"  {lab:24s} corr with true month returns {m.interp.corr(m.true):.3f} | vol {m.interp.std()*np.sqrt(12):.4f}"
          f" vs true {m.true.std()*np.sqrt(12):.4f} (ratio {m.interp.std()/m.true.std():.3f})")
V = json.load(open(os.path.join(HERE, "panel_validation.json")))["gold_monthly_interp_vs_gcf_monthly"]
print(f"  panel_validation (the panel's actual monthly.csv series vs GC=F): corr {V['corr']} vol ratio "
      f"{V['vol_interp']/V['vol_gcf']:.3f}   | theory for averages: corr sqrt(3/8)={np.sqrt(3/8):.3f}, vol ratio sqrt(2/3)={np.sqrt(2/3):.3f}")

pre = (g.index < "2000-08-30")
alt = {}
alt["as built"] = g.copy()
s = g.copy(); s[pre] = g.shift(-11)[pre]; alt["advanced 11d (undo half-month lag)"] = s
s = g.copy()
gm = g[pre]; mret = (1 + gm).groupby(gm.index.to_period("M")).prod() - 1
last = pd.Series(gm.index, index=gm.index).groupby(gm.index.to_period("M")).max()
s[pre] = 0.0
for per, d in last.items():
    s[d] = mret[per]
alt["month-step (book month on last day)"] = s
s = g.copy(); s[pre] = P["bills"][pre]; alt["bills instead of gold pre-2000"] = s
Pg = P.copy()
for lab, ser in alt.items():
    Pg["gold_alt"] = ser.fillna(0)
    res = {}
    for fam, kw in {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}.items():
        r1 = C.backtest(Pg, C.strategy_weights(Pg, gold="gold_alt", off="ief", **kw))
        r0 = C.backtest(Pg, C.strategy_weights(Pg, gold="gold_alt", gold_w=0.0, off="ief", **kw))
        h1, h10 = st(r1, "1988-01-01", "2006-12-31"), st(r0, "1988-01-01", "2006-12-31")
        is1 = st(r1, None, FREEZE)
        res[fam] = f"{fam} H1 Cal {h1['calmar']:.3f} DD {h1['maxdd']*100:5.1f}% | gold dCal_H1 {h1['calmar']-h10['calmar']:+.3f} | IS Cal {is1['calmar']:.3f}"
    print(f"  {lab:38s} " + " || ".join(res.values()))

# =====================================================================================  K5
hdr("K5  Sharpe with the risk-free rate subtracted (specialist's Sharpe ignores rf)")
rf = P["rf_d"].fillna(0)
R3 = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20))
R2 = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30))


def xsh(r, a=None, b=None):
    s = (r - rf)[(r.index >= START)]
    if a is not None:
        s = s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    return s.mean() / s.std() * np.sqrt(252)


for wn, (a, b) in {"IS": (None, FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
                   "FULL": (None, LATEST)}.items():
    print(f"  {wn:4s} raw Sharpe F2 {st(R2,a,b)['sharpe']:.3f} F3 {st(R3,a,b)['sharpe']:.3f} | excess Sharpe "
          f"F2 {xsh(R2,a,b):.3f} F3 {xsh(R3,a,b):.3f}")
# paired grid: 108 nuisance combos, excess Sharpe
GV = dict(chop_thr=[0.20, 0.25, 0.30], chop_pos=[0.35, 0.50, 0.65], gold_w=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
          seasonal=[False, True])
wins = {"H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE), "IS": (None, FREEZE)}
cnt = {w: [0, 0] for w in wins}
mind = {w: [] for w in wins}
for th, po, gw, se in itertools.product(*GV.values()):
    a2 = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30, chop_thr=th, chop_pos=po, gold_w=gw, seasonal=se))
    a3 = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20, chop_thr=th, chop_pos=po, gold_w=gw, seasonal=se))
    for w, (a, b) in wins.items():
        d = xsh(a2, a, b) - xsh(a3, a, b)
        cnt[w][0] += d > 0; cnt[w][1] += 1; mind[w].append(d)
print("  paired F2 vt.30 vs F3 vt.20, share with higher EXCESS Sharpe:",
      {w: f"{c[0]}/{c[1]} (median d {np.median(mind[w]):+.3f})" for w, c in cnt.items()})


def boot_xsh(r1, r2, b=None, n=2000, block=63, seed=0):
    x = pd.concat([r1 - rf, r2 - rf], axis=1)
    if b is not None:
        x = x[x.index <= pd.Timestamp(b)]
    x = x[x.index >= START].values
    T = len(x); rg = np.random.default_rng(seed); nb = int(np.ceil(T / block)); out = np.empty(n)
    for i in range(n):
        st_ = rg.integers(0, T, nb)
        ix = (st_[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = x[ix]
        sh = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
        out[i] = sh[0] - sh[1]
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), (out > 0).mean()


for lab, b in {"FULL": None, "IS": FREEZE}.items():
    m_, lo, hi, p = boot_xsh(R2, R3, b)
    print(f"  bootstrap F2-F3 EXCESS Sharpe {lab}: mean {m_:+.3f} [{lo:+.3f},{hi:+.3f}] P>0 {p:.3f}"
          f"  (saved raw-Sharpe P>0 FULL {J['bootstrap']['F2-IEF minus F3-IEF']['FULL']['sharpe']['p_gt0']:.3f})")

# =====================================================================================  K6
hdr("K6  grid / plateau claims")
G = pd.DataFrame(J["grid"])
for lev, dflt in {"tqqq": dict(voltgt=0.20, chop_thr=0.25, chop_pos=0.50, gold_w=0.15),
                  "qld": dict(voltgt=0.30, chop_thr=0.25, chop_pos=0.50, gold_w=0.15)}.items():
    df = G[G.lev == lev].copy()
    lv = dict(voltgt=[0.15, 0.20, 0.25, 0.30, 0.35], chop_thr=[0.20, 0.25, 0.30], chop_pos=[0.35, 0.50, 0.65],
              gold_w=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    for k, vals in lv.items():
        df[k + "_i"] = df[k].apply(lambda v, vals=vals: int(np.argmin(np.abs(np.array(vals) - v))))
    di = {k: int(np.argmin(np.abs(np.array(lv[k]) - dflt[k]))) for k in lv}
    D = np.abs(np.column_stack([df[k + "_i"] - di[k] for k in lv]))
    box_on = df[(D.max(axis=1) <= 1) & df.seasonal]
    box_all = df[(D.max(axis=1) <= 1)]
    print(f"  {lev}: IS Calmar grid min {df.IS_calmar.min():.3f} median {df.IS_calmar.median():.3f} max {df.IS_calmar.max():.3f}"
          f" | box Aug-Sep ON n={len(box_on)} median {box_on.IS_calmar.median():.3f} share>=0.70 {(box_on.IS_calmar>=0.70).mean():.2f}"
          f" | box both n={len(box_all)} median {box_all.IS_calmar.median():.3f} share>=0.70 {(box_all.IS_calmar>=0.70).mean():.2f}")
    d = J["plateau"][f"{lev}|IS"]["one_step"]["detail"]
    print("      one-step:", ", ".join(f"{x['change']}:{x['calmar']:.3f}" for x in d))
    for wn in ["H1", "H2"]:
        mm = J["plateau"][f"{lev}|{wn}"]["marginal_median"]["gold_w"]
        print(f"      {wn} marginal median Calmar by gold_w: " + " ".join(f"{a}:{b:.3f}" for a, b in mm.items()))
print("  all 1080 configs IS Calmar >= ", round(G.IS_calmar.min(), 3), "| QQQ B&H IS Calmar",
      round(CAND["QQQ B&H"]["windows"]["IS"]["calmar"], 3))

# =====================================================================================  K7
hdr("K7  walk-forward: honest out-of-sample read and the gold=0.25 edge pick")
for key in ["F3 5-param | early->late", "F3 5-param | late->early", "F2 5-param | early->late", "F2 5-param | late->early"]:
    w = J["walk_forward"][key]
    p = w["pick"]
    for t, tt in w["tests"].items():
        dl = list(tt["defaults"].keys())[0]
        print(f"  {key:28s} test {t:7s} pick Cal {tt['pick']['calmar']:.3f} (test pctile {tt['pick']['pctile']*100:4.1f}%)"
              f" | default {tt['defaults'][dl]['calmar']:.3f} (pctile {tt['defaults'][dl]['pctile']*100:4.1f}%; chosen in July"
              f" on 1988-2026 = hindsight) | test-grid median {tt['grid_median_calmar']:.3f}")
        sub = G[(G.lev == p["lev"]) & np.isclose(G.voltgt, p["voltgt"]) & np.isclose(G.chop_thr, p["chop_thr"]) &
                np.isclose(G.chop_pos, p["chop_pos"]) & (G.seasonal == p["seasonal"])].sort_values("gold_w")
        print("        pick knobs, gold_w sweep in TEST window:",
              " ".join(f"g{r.gold_w:.2f}:{r[t + '_calmar']:.3f}" for _, r in sub.iterrows()))

# =====================================================================================  K8
hdr("K8  the F3-IEF max drawdown 1989-10-09 -> 1990-05-02 (sets the IS/H1/FULL max DD of -30.0%)")
hw = C.held_weights(P, w3)
a, b = pd.Timestamp("1989-10-10"), pd.Timestamp("1990-05-02")
sl = (IDX >= a) & (IDX <= b)
contrib = {c: float((hw[c] * P[c].fillna(0))[sl].sum()) for c in hw.columns}
print("  sum of daily contributions (approx.):", {k: f"{v*100:+.1f}%" for k, v in contrib.items()},
      "| total compounding", f"{((1 + R3[sl]).prod() - 1) * 100:+.1f}%")
print(f"  NDX over window {(P.ndx[b] / P.ndx[pd.Timestamp('1989-10-09')] - 1) * 100:+.1f}% | TQQQ-sim compounding "
      f"{((1 + P.tqqq[sl]).prod() - 1) * 100:+.1f}% | mean rf {P.rf[sl].mean()*100:.2f}%")
wk = P[sl & P["mark"].astype(bool)]
for d in wk.index:
    nxt = IDX[IDX > d]
    e = nxt[nxt <= d + pd.Timedelta(days=7)]
    print(f"    {d.date()} above {bool(P.at[d,'above'])} vol {P.at[d,'vol20']:.2f} er {P.at[d,'er20']:.2f} -> tqqq w "
          f"{w3.at[d,'tqqq']:.2f} | next wk NDX {((1+P.ndx_ret[e]).prod()-1)*100:+5.1f}% F3 {((1+R3[e]).prod()-1)*100:+5.1f}%")
big = P.loc[sl, "ndx_ret"].abs().sort_values(ascending=False).head(5)
print("  largest |NDX| daily moves in window:", {str(k.date()): f"{P.at[k,'ndx_ret']*100:+.2f}%" for k in big.index})

# =====================================================================================  K9
hdr("K9  OOS narrative (C10) and current drawdown split (C12)")
for o in J["oos_weekly_path"]:
    if o["friday"] in ("2026-07-24", "2026-07-31", "2026-08-07", "2026-09-25", "2026-10-02", "2026-10-09"):
        print(f"  {o['friday']} month {pd.Timestamp(o['friday']).month} F3 tqqq {o['F3_tqqq']:.2f} er {o['er20']:.2f}"
              f" vol {o['vol20']:.2f} | following week F3 {o['next_week']['F3']*100:+.1f}% A {o['next_week']['A']*100:+.1f}%"
              f" NDX {o['next_week']['NDX']*100:+.1f}%")
for k, r in {"F3-IEF": R3, "F2-IEF": R2}.items():
    e = (1 + r[r.index >= START]).cumprod()
    pk = e[e.index <= LATEST].idxmax()
    print(f"  {k}: peak {pk.date()} | DD at freeze {(e[FREEZE]/e[pk]-1)*100:+.1f}% | DD now {(e.iloc[-1]/e[pk]-1)*100:+.1f}%"
          f" | OOS {((e.iloc[-1]/e[FREEZE])-1)*100:+.1f}%")

# =====================================================================================  K10
hdr("K10 pre-2002 synthetic IEF carries a full-sample-calibrated alpha (+0.305%/yr, fitted on 2002-2026)")
Pa = P.copy()
syn = Pa["ief_real"].isna()
Pa["ief_noalpha"] = Pa["ief"] - np.where(syn, 0.00305 / 252, 0.0)
for fam, kw in {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}.items():
    r0 = C.backtest(Pa, C.strategy_weights(Pa, off="ief", **kw))
    r1 = C.backtest(Pa, C.strategy_weights(Pa, off="ief_noalpha", **kw))
    print(f"  {fam}: H1 {f(st(r0,'1988-01-01','2006-12-31'))} -> no-alpha {f(st(r1,'1988-01-01','2006-12-31'))}"
          f" | IS Cal {st(r0,None,FREEZE)['calmar']:.3f} -> {st(r1,None,FREEZE)['calmar']:.3f}")

# =====================================================================================  K11
hdr("K11 ablation: 'helps both halves' under July instruments; exact small margins; IS-only F2-F3 bootstrap")
for fam in ["F3", "F2"]:
    for ins in ["faithful", "july"]:
        tab = J["ablation"][f"{fam}|{ins}"]
        rows = [(k, v["dCal_H1"], v["dCal_H2"]) for k, v in tab.items() if k != "default"]
        n_ok = sum(1 for _, h1, h2 in rows if h1 > 0 and h2 > 0)
        fails = [f"{k} (H1 {h1:+.4f}, H2 {h2:+.4f})" for k, h1, h2 in rows if not (h1 > 0 and h2 > 0)]
        small = sorted(rows, key=lambda t: min(t[1], t[2]))[:3]
        print(f"  {fam}|{ins:8s}: helps both halves {n_ok}/9; fails: {fails}")
        print(f"        smallest margins: " + "; ".join(f"{k} H1 {h1:+.4f} H2 {h2:+.4f}" for k, h1, h2 in small))
bi = J["bootstrap"]["F2-IEF minus F3-IEF"]["IS"]
print(f"  IS-only bootstrap F2-F3: dCAGR {bi['cagr']['mean']*100:+.2f}% [{bi['cagr']['lo95']*100:+.2f},{bi['cagr']['hi95']*100:+.2f}]"
      f" | dCalmar {bi['calmar']['mean']:+.3f} [{bi['calmar']['lo95']:+.3f},{bi['calmar']['hi95']:+.3f}] P>0 {bi['calmar']['p_gt0']:.3f}"
      f" | Sharpe P>0 {bi['sharpe']['p_gt0']:.3f} | shallower DD P {bi['maxdd']['p_gt0']:.3f}")
gb = J["bootstrap"]
for k in ["F3 gold15 minus no-gold", "F2 gold15 minus no-gold", "F3 IEF minus TLT off-leg"]:
    x = gb[k]["FULL"]
    print(f"  {k:28s} dCAGR {x['cagr']['mean']*100:+.2f}% P>0 {x['cagr']['p_gt0']:.3f} | dCal {x['calmar']['mean']:+.3f}"
          f" [{x['calmar']['lo95']:+.3f},{x['calmar']['hi95']:+.3f}] P>0 {x['calmar']['p_gt0']:.3f} | DD P {x['maxdd']['p_gt0']:.3f}")
print("\ndone", flush=True)
