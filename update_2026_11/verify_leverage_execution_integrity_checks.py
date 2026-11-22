"""
INTEGRITY REVIEW of leverage_execution  --  step 3: independent re-implementation + targeted checks.
Read-only (prints only).  usage: python update_2026_11/verify_leverage_execution_integrity_checks.py

 K0  fast numpy backtester; must equal common.backtest exactly (Fri, Mon, tranche)
 K1  Friday effect vs a timing-luck NULL:
       (a) non-calendar 5-trading-day cycles (offsets 0..4, drift through weekdays at holidays)
       (b) random weekday per calendar YEAR (fixed spacing within a year), 1000 draws
       (c) random weekday per calendar WEEK, 300 draws
     + on/off switch counts for each schedule family
 K2  OOS anatomy: when 'above' flipped after the freeze and how each weekday phase was exposed
 K3  mechanism robustness (median / trimmed mean / excluding 2008 & 2020)
 K4  planning numbers: tranche bootstrap of F2/F3 (recommendation (7) quotes the FRIDAY bootstrap)
 K5  tranche matched-risk: first-crossing interpolation vs envelope, monotonicity
 K6  gold pre-2000-08-31 monthly interpolation: step (no in-month path) sensitivity
 K7  IEF/TLT synthetic alpha (full-sample calibration 2002-2026) sensitivity pre-2002-07-30
 K8  instrument effect TQQQx2/3 - QLD using 2023..FREEZE only (claim used 2023..latest = includes OOS)
 K9  pre-close signal noise (lag-1 realism): perturb the Friday signal price by N(0, s)
 K10 current state
"""
import sys, os, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

T0 = time.time()
P = C.panel(); IDX = P.index; T = len(IDX)
START, FREEZE, LATEST = C.START, C.FREEZE, IDX[-1]
IS = (IDX >= START) & (IDX <= FREEZE)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def W_of(lev, vt, off="ief", **kw):
    w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off=off, **kw)
    return w


def held_np(Wm, marks, lag=1):
    ii = np.where(marks, np.arange(T), -1)
    lm = np.maximum.accumulate(ii)
    src = np.full(T, -1)
    src[lag:] = lm[:-lag]
    return np.where((src >= 0)[:, None], Wm[np.maximum(src, 0)], 0.0)


def bt_from_held(H, R, cost=C.COST):
    gross = (H * R).sum(1)
    turn = np.r_[0.0, np.abs(np.diff(H, axis=0)).sum(1)]
    return gross - turn * cost


def fast_bt(w, marks, lag=1, cost=C.COST, R=None):
    Wm = w.to_numpy(float)
    if R is None:
        R = np.nan_to_num(P[list(w.columns)].to_numpy(float))
    return bt_from_held(held_np(Wm, marks, lag), R, cost)


def st(r, mask=IS):
    x = r[mask]
    e = np.cumprod(1 + x)
    d = IDX[mask]
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return cagr, dd


def switches(w, lev, marks, lag=1):
    H = held_np(w.to_numpy(float), marks, lag)
    on = (H[:, list(w.columns).index(lev)] > 1e-12).astype(int)
    sw = np.abs(np.diff(on, prepend=on[0]))
    return int(sw[IS].sum())


OUT = {}
F2 = W_of("qld", 0.30); F3 = W_of("tqqq", 0.20)
MK = {d: (IDX.dayofweek == i) for i, d in enumerate(DAYS)}

# ------------------------------------------------------------- K0: exactness of the fast engine
r_c = C.backtest(P, F2).to_numpy(); r_f = fast_bt(F2, MK["Fri"])
r_cm = C.backtest(P, F2, mark=pd.Series(MK["Mon"], index=IDX)).to_numpy(); r_fm = fast_bt(F2, MK["Mon"])
print(f"K0 fast engine vs common.backtest: Fri max|diff| {np.abs(r_c - r_f).max():.2e}  Mon {np.abs(r_cm - r_fm).max():.2e}")
for nm, w in (("F2", F2), ("F3", F3)):
    rs = {d: fast_bt(w, MK[d]) for d in DAYS}
    print(f"   {nm} weekday IS CAGR: " + " ".join(f"{d} {st(rs[d])[0]*100:5.2f}" for d in DAYS))

# ------------------------------------------------------------- K1: timing-luck null
print("\nK1 FRIDAY EFFECT vs TIMING-LUCK NULL (IS 1988-01-01..2026-07-24, lag 1, 3 bps)")
rng = np.random.default_rng(2026)
k1 = {}
years = IDX.year.values
week_id = IDX.to_period("W-FRI").astype(str).values
for nm, w, lev in (("F2", F2, "qld"), ("F3", F3, "tqqq")):
    R = np.nan_to_num(P[list(w.columns)].to_numpy(float))
    res = {}
    # weekday phases
    wk = {d: st(fast_bt(w, MK[d], R=R))[0] for d in DAYS}
    wk_sw = {d: switches(w, lev, MK[d]) for d in DAYS}
    # (a) non-calendar 5-trading-day cycle offsets
    cyc = {}
    for k in range(5):
        m = (np.arange(T) % 5) == k
        c, dd = st(fast_bt(w, m, R=R))
        cyc[k] = dict(cagr=c, maxdd=dd, switches=switches(w, lev, m),
                      dow_share={d: float((IDX[m & IS].dayofweek == i).mean()) for i, d in enumerate(DAYS)})
    cc = np.array([cyc[k]["cagr"] for k in range(5)])
    best = int(cc.argmax())
    res["cycle5"] = cyc
    res["cycle5_spread_pp"] = float((cc.max() - cc.min()) * 100)
    res["cycle5_best_minus_mean_others_pp"] = float((cc[best] - np.delete(cc, best).mean()) * 100)
    wkv = np.array([wk[d] for d in DAYS])
    res["weekday_spread_pp"] = float((wkv.max() - wkv.min()) * 100)
    res["weekday_fri_minus_mean_others_pp"] = float((wk["Fri"] - wkv[:4].mean()) * 100)
    # (b) random weekday per calendar year (regular spacing inside a year)
    uy = np.unique(years)
    dows = IDX.dayofweek.values
    yb = []
    sw_b = []
    for i in range(1000):
        pick = rng.integers(0, 5, len(uy))
        dmap = dict(zip(uy, pick))
        m = dows == np.vectorize(dmap.get)(years)
        yb.append(st(fast_bt(w, m, R=R))[0])
        if i < 200:
            sw_b.append(switches(w, lev, m))
    yb = np.array(yb)
    # (c) random trading day per calendar week
    wb = []
    codes, uniq = pd.factorize(week_id)
    order = np.arange(T)
    for i in range(300):
        u = rng.random(T)
        # pick, in every week, the day with the smallest random key
        df = pd.DataFrame({"g": codes, "u": u})
        first = df.groupby("g")["u"].idxmin().to_numpy()
        m = np.zeros(T, bool); m[first] = True
        wb.append(st(fast_bt(w, m, R=R))[0])
    wb = np.array(wb)
    res["weekday"] = wk; res["weekday_switches"] = wk_sw
    res["rand_year_weekday"] = dict(mean=float(yb.mean()), sd=float(yb.std()), p5=float(np.percentile(yb, 5)),
                                    p95=float(np.percentile(yb, 95)), max=float(yb.max()),
                                    pct_ge_fri=float((yb >= wk["Fri"]).mean()), pct_le_mon=float((yb <= wk["Mon"]).mean()),
                                    switches_mean=float(np.mean(sw_b)), switches_min=int(np.min(sw_b)))
    res["rand_week_day"] = dict(mean=float(wb.mean()), sd=float(wb.std()), p5=float(np.percentile(wb, 5)),
                                p95=float(np.percentile(wb, 95)), max=float(wb.max()),
                                pct_ge_fri=float((wb >= wk["Fri"]).mean()))
    k1[nm] = res
    print(f"  {nm}: weekday CAGR {[round(wk[d]*100,2) for d in DAYS]} spread {res['weekday_spread_pp']:.2f}pp; "
          f"Fri - mean(others) {res['weekday_fri_minus_mean_others_pp']:+.2f}pp; switches {wk_sw}")
    print(f"      5-day cycle offsets CAGR {[round(cyc[k]['cagr']*100,2) for k in range(5)]} maxDD "
          f"{[round(cyc[k]['maxdd']*100,1) for k in range(5)]} switches {[cyc[k]['switches'] for k in range(5)]}; "
          f"spread {res['cycle5_spread_pp']:.2f}pp; best - mean(others) {res['cycle5_best_minus_mean_others_pp']:+.2f}pp")
    print(f"      random weekday per YEAR (1000): mean {yb.mean()*100:.2f}% sd {yb.std()*100:.2f}pp "
          f"[p5 {np.percentile(yb,5)*100:.2f}, p95 {np.percentile(yb,95)*100:.2f}, max {yb.max()*100:.2f}]  "
          f"share >= Fri {res['rand_year_weekday']['pct_ge_fri']:.3f}; share <= Mon {res['rand_year_weekday']['pct_le_mon']:.3f}; "
          f"switches mean {np.mean(sw_b):.0f} min {np.min(sw_b)}")
    print(f"      random day per WEEK (300):      mean {wb.mean()*100:.2f}% sd {wb.std()*100:.2f}pp "
          f"[p5 {np.percentile(wb,5)*100:.2f}, p95 {np.percentile(wb,95)*100:.2f}, max {wb.max()*100:.2f}]  "
          f"share >= Fri {res['rand_week_day']['pct_ge_fri']:.3f}")
OUT["K1"] = k1
print(f"  [{time.time()-T0:.0f}s]")

# ------------------------------------------------------------- K2: OOS anatomy
print("\nK2 OOS anatomy (2026-07-24..latest): NDX vs EMA-100 at each close and weekday-phase QLD weight (F2)")
oos = P.loc[P.index >= "2026-07-20", ["ndx", "ema100", "above", "vol20", "er20"]].copy()
for d in DAYS:
    H = held_np(F2.to_numpy(float), MK[d])
    oos[f"{d}_qld"] = pd.Series(H[:, 0], index=IDX).reindex(oos.index)
oos["ndx_ret"] = P["ndx_ret"].reindex(oos.index)
flips = oos.index[oos["above"].astype(int).diff().fillna(0) != 0]
print("   above-state flips after freeze:", [(str(d.date()), d.day_name()[:3], bool(oos.loc[d, 'above'])) for d in flips])
sub = oos.loc["2026-10-01":"2026-10-30"]
for d, row in sub.iterrows():
    print(f"   {d.date()} {d.day_name()[:3]} ndx {row.ndx:9.1f} ema {row.ema100:9.1f} above {int(row.above)} "
          f"ret {row.ndx_ret*100:+5.2f}% | QLD held  " + " ".join(f"{x[:3]} {row[f'{x}_qld']:.2f}" for x in DAYS))
oo = (IDX > FREEZE)
for nm, w in (("F2", F2), ("F3", F3)):
    tot = {d: float(np.prod(1 + fast_bt(w, MK[d])[oo]) - 1) for d in DAYS}
    print(f"   {nm} OOS total by phase: " + " ".join(f"{d} {tot[d]*100:+.1f}%" for d in DAYS))
    # window-by-window contribution: up to 2026-10-09 vs after
    for a, b in (("2026-07-25", "2026-10-09"), ("2026-10-10", "2026-11-17")):
        mm = (IDX >= a) & (IDX <= b)
        print(f"      {a}..{b}: " + " ".join(f"{d} {(np.prod(1 + fast_bt(w, MK[d])[mm]) - 1)*100:+.1f}%" for d in DAYS))

# ------------------------------------------------------------- K3: mechanism robustness
print("\nK3 mechanism: next-session NDX return after a close BELOW EMA-100 (bp): mean / median / 1%-trimmed / ex-2008&2020 / n")
nxt = P["ndx_ret"].shift(-1)
abv = P["above"].astype(bool).to_numpy()
dow = IDX.dayofweek
yr = IDX.year
for wn, a, b in (("H1", "1988-01-01", "2006-12-31"), ("H2", "2007-01-01", str(FREEZE.date()))):
    sel = (IDX >= a) & (IDX <= b)
    parts = []
    for i, d in enumerate(DAYS):
        x = nxt[sel & (dow == i) & ~abv].dropna()
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        tr = x[(x >= lo) & (x <= hi)]
        ex = nxt[sel & (dow == i) & ~abv & ~np.isin(yr, [2008, 2020])].dropna()
        parts.append(f"{d} {x.mean()*1e4:+6.1f}/{x.median()*1e4:+5.1f}/{tr.mean()*1e4:+5.1f}/{ex.mean()*1e4:+5.1f} n{len(x)}")
    print(f"   {wn}: " + " | ".join(parts))

# ------------------------------------------------------------- K4/K5: tranche frontier, bootstrap, envelope
print("\nK4/K5 tranche frontier (5 weekday phases averaged), IS")
FINE_T = [round(float(x), 3) for x in np.arange(0.05, 0.8001, 0.025)]


def tranche(w, lag=1, cost=C.COST):
    Wm = w.to_numpy(float)
    H = sum(held_np(Wm, MK[d], lag) for d in DAYS) / 5.0
    R = np.nan_to_num(P[list(w.columns)].to_numpy(float))
    return bt_from_held(H, R, cost)


TF = {}
for lev in ("qld", "tqqq"):
    TF[lev] = []
    for vt in FINE_T:
        r = tranche(W_of(lev, vt))
        c, dd = st(r)
        TF[lev].append((vt, c, -dd, r))
for lev in ("qld", "tqqq"):
    risk = np.array([x[2] for x in TF[lev]])
    nonmono = int((np.diff(risk) < -1e-9).sum())
    print(f"   {lev}: maxDD non-monotone steps along vt grid: {nonmono} of {len(risk)-1}; "
          f"min DD {risk.min()*100:.1f}% at vt {TF[lev][int(risk.argmin())][0]}, max DD {risk.max()*100:.1f}%")
for tgt in (0.25, 0.28, 0.30, 0.325):
    row = []
    for lev in ("qld", "tqqq"):
        risk = np.array([x[2] for x in TF[lev]]); cg = np.array([x[1] for x in TF[lev]])
        ok = risk <= tgt
        env = cg[ok].max() if ok.any() else np.nan
        envvt = TF[lev][int(np.where(ok, cg, -9).argmax())][0] if ok.any() else None
        hit = np.where(risk >= tgt)[0]
        if len(hit) and hit[0] > 0:
            i = hit[0]; f = (tgt - risk[i-1]) / (risk[i] - risk[i-1])
            fc = cg[i-1] + f * (cg[i] - cg[i-1])
        else:
            fc = np.nan
        row.append(f"{lev} first-cross {fc*100:5.2f}% envelope {env*100:5.2f}% (vt {envvt})")
    print(f"   DD<= {tgt:.3f}: " + " | ".join(row))

# tranche bootstrap of the two incumbents (same machinery as the specialist: circular 63d, paired)
def paired_boot(M, n=1000, block=63, seed=11):
    Tn, k = M.shape
    rng2 = np.random.default_rng(seed); nb = int(np.ceil(Tn / block))
    cg = np.empty((n, k)); dd = np.empty((n, k))
    for i in range(n):
        s0 = rng2.integers(0, Tn, nb)
        ix = ((s0[:, None] + np.arange(block)[None, :]).ravel()[:Tn]) % Tn
        e = np.vstack([np.ones((1, k)), np.cumprod(1 + M[ix], axis=0)])
        cg[i] = e[-1] ** (252.0 / Tn) - 1
        dd[i] = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
    return cg, dd

rF2 = fast_bt(F2, MK["Fri"]); rF3 = fast_bt(F3, MK["Fri"])
tF2 = tranche(F2); tF3 = tranche(F3)
M = np.column_stack([rF2, rF3, tF2, tF3])[IS]
cg, dd = paired_boot(M)
names = ["F2_fri", "F3_fri", "F2_tranche", "F3_tranche"]
K4 = {}
for j, nmb in enumerate(names):
    K4[nmb] = dict(cagr_med=float(np.median(cg[:, j])), maxdd_med=float(np.median(dd[:, j])),
                   maxdd_p5=float(np.percentile(dd[:, j], 5)), p_dd_worse_50=float((dd[:, j] <= -0.5).mean()),
                   p_dd_worse_40=float((dd[:, j] <= -0.4).mean()))
    print(f"   boot {nmb:11s} median CAGR {K4[nmb]['cagr_med']*100:5.2f}%  median maxDD {K4[nmb]['maxdd_med']*100:5.1f}%  "
          f"1-in-20 {K4[nmb]['maxdd_p5']*100:5.1f}%  P(DD<-40%) {K4[nmb]['p_dd_worse_40']*100:4.1f}%  P(DD<-50%) {K4[nmb]['p_dd_worse_50']*100:4.1f}%")
print(f"   P(Calmar F2_tranche > F3_tranche) {float(((cg[:,2]/-dd[:,2]) > (cg[:,3]/-dd[:,3])).mean()):.2f}")
OUT["K4"] = K4
print(f"  [{time.time()-T0:.0f}s]")

# ------------------------------------------------------------- K6: gold interpolation sensitivity
print("\nK6 gold before 2000-08-31: monthly log-interpolation (path uses month-end price) vs month-end STEP (no in-month path)")
g = P["gold_phys"].copy()
pre = (P["gold_src"] == "monthly-interp").to_numpy()
gl = np.log1p(g[pre])
mon = pd.Series(gl.values, index=IDX[pre]).groupby(IDX[pre].to_period("M")).transform("sum")
last = pd.Series(IDX[pre], index=IDX[pre]).groupby(IDX[pre].to_period("M")).transform("max")
step = np.where(IDX[pre] == last.values, np.expm1(mon.values), 0.0)
g2 = g.copy(); g2[pre] = step
R_alt = None
for nm, w in (("F2", F2), ("F3", F3)):
    cols = list(w.columns)
    Rb = np.nan_to_num(P[cols].to_numpy(float))
    Ra = Rb.copy(); Ra[:, cols.index("gold_phys")] = g2.to_numpy()
    for lab, R in (("interp (as used)", Rb), ("month-end step", Ra)):
        r = fast_bt(w, MK["Fri"], R=R); rt = bt_from_held(sum(held_np(w.to_numpy(float), MK[d]) for d in DAYS) / 5.0, R)
        c, d_ = st(r); ct, dt = st(rt)
        h1 = (IDX >= START) & (IDX <= pd.Timestamp("2006-12-31"))
        c1, d1 = st(r, h1)
        e = np.cumprod(1 + r[IS]); ddp = e / np.maximum.accumulate(e) - 1
        tr_i = int(ddp.argmin()); tr_d = IDX[IS][tr_i]; pk_d = IDX[IS][int(np.argmax(e[:tr_i+1]))]
        print(f"   {nm} {lab:17s} Fri IS {c*100:6.2f}% / {d_*100:5.1f}% (peak {pk_d.date()} trough {tr_d.date()}) | "
              f"H1 {c1*100:5.2f}% / {d1*100:5.1f}% | tranche IS {ct*100:6.2f}% / {dt*100:5.1f}%")

# ------------------------------------------------------------- K7: bond synthetic alpha pre-inception
print("\nK7 IEF synthetic carries a fitted alpha (+0.305%/yr, fitted 2002-07..2026-11 incl. OOS). Remove it pre-2002-07-31:")
pre_ief = P["ief_real"].isna().to_numpy()
for nm, w in (("F2", F2), ("F3", F3)):
    cols = list(w.columns)
    R = np.nan_to_num(P[cols].to_numpy(float)); R2 = R.copy()
    R2[pre_ief, cols.index("ief")] -= 0.00305 / 252
    c0, d0 = st(fast_bt(w, MK["Fri"], R=R)); c1, d1 = st(fast_bt(w, MK["Fri"], R=R2))
    print(f"   {nm}: IS CAGR {c0*100:.2f}% -> {c1*100:.2f}% ({(c1-c0)*100:+.2f}pp); maxDD {d0*100:.1f}% -> {d1*100:.1f}%")

# ------------------------------------------------------------- K8: instrument effect without OOS data
print("\nK8 TQQQ x2/3 vs QLD (real ETFs), CAGR difference by window")
w_inst = W_of("tqqq", 0.30, extra_scale=pd.Series(2.0 / 3.0, index=IDX))
a3 = pd.Series(fast_bt(w_inst, MK["Fri"]), index=IDX); a2 = pd.Series(rF2, index=IDX)
for a, b in (("2010-03-01", str(FREEZE.date())), ("2023-01-01", str(FREEZE.date())), ("2023-01-01", str(LATEST.date())),
             ("2026-07-25", str(LATEST.date()))):
    s3 = C.stats(a3, a, b, start=None); s2 = C.stats(a2, a, b, start=None)
    print(f"   {a}..{b}: TQQQx2/3 {s3['cagr']*100:6.2f}%  QLD {s2['cagr']*100:6.2f}%  diff {(s3['cagr']-s2['cagr'])*100:+.2f}pp "
          f"(total {s3['total']*100:+.1f}% vs {s2['total']*100:+.1f}%)")

# ------------------------------------------------------------- K9: pre-close signal noise
print("\nK9 lag-1 realism: Friday decision computed on a pre-close price = close x (1+e), e ~ N(0, s); trade at the true close")
ndx = P["ndx"].to_numpy(); ema = P["ema100"].to_numpy()
ema_prev = np.r_[ema[0], ema[:-1]]
fri = MK["Fri"]
for s in (0.002, 0.004):
    res = {"F2": [], "F3": []}
    flips = []
    for i in range(200):
        e = rng.normal(0, s, T)
        pc = ndx * (1 + e)
        above_p = pd.Series(np.where(fri, pc > ema_prev, P["above"].to_numpy()), index=IDX)
        flips.append(int((above_p.to_numpy() != P["above"].to_numpy())[fri & IS].sum()))
        for nm, lev, vt in (("F2", "qld", 0.30), ("F3", "tqqq", 0.20)):
            w = W_of(lev, vt, risk_on=above_p)
            res[nm].append(st(fast_bt(w, fri))[0])
    print(f"   s={s*100:.1f}%: Friday decisions flipped per run {np.mean(flips):.1f} of {int((fri & IS).sum())}; "
          + "  ".join(f"{nm} CAGR mean {np.mean(v)*100:.2f}% [p5 {np.percentile(v,5)*100:.2f}, p95 {np.percentile(v,95)*100:.2f}]"
                      for nm, v in res.items()))
dist = np.abs(ndx / ema - 1)
print("   Friday closes within 0.25% / 0.5% of EMA-100 (IS):", int((fri & IS & (dist < 0.0025)).sum()),
      "/", int((fri & IS & (dist < 0.005)).sum()))

# ------------------------------------------------------------- K10: current state
last = P.iloc[-1]
print(f"\nK10 {IDX[-1].date()}: ndx {last.ndx:.1f} ema100 {last.ema100:.1f} ({(last.ndx/last.ema100-1)*100:+.2f}%; "
      f"re-entry needs {(last.ema100/last.ndx-1)*100:+.2f}%) above {bool(last.above)} vol20 {last.vol20*100:.1f}% er20 {last.er20:.3f}")
print(f"   re-entry weights at today's vol: F2 QLD {0.85*min(1,0.30/last.vol20):.3f} (chop {0.85*min(1,0.30/last.vol20)*0.5:.3f}); "
      f"F3 TQQQ {0.85*min(1,0.20/last.vol20):.3f}")
lastfri = IDX[MK["Fri"]][-1]
print(f"   last Friday mark {lastfri.date()}: above {bool(P.loc[lastfri,'above'])}; held F2 last row:",
      {c: round(float(v), 3) for c, v in C.held_weights(P, F2).iloc[-1].items()})
below = P["above"].astype(bool)
lastabove = IDX[below.to_numpy()][-1]
print(f"   last close above EMA-100: {lastabove.date()}")
print(f"\n[{time.time()-T0:.0f}s] done")
