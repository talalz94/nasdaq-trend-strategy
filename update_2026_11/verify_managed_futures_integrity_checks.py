"""
verify_managed_futures_integrity_checks.py  (adversarial review, integrity lens)

Loads ONLY the function definitions of update_2026_11/managed_futures.py (exec of the source up to
the '1. PROXIES' section, GSG fetch disabled -> read-only cache), then runs:

  A. future-shock lookahead test: replace EVERY panel value after a cut date T with garbage and
     check that proxy positions/returns and F+TSMOM portfolio returns up to T are bit-identical.
  B. independent from-spec re-implementation of the main_12m proxy (separate code path).
  C. gold pre-2000 information-time audit (monthly averages anchored at month-end).
  D. intra-month drift: implicit daily-rebalancing turnover the proxy does not charge, and a
     'hold contracts within the month' variant.
  E. C3 vol numbers / off-leg weight on risk-on days.
  F. episode dependence of the off-leg verdict (C2/C9) using the pickled series from the rerun.
Nothing in the project is written; prints only.
"""
import os, sys, json, pickle, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "update_2026_11")
SCR = sys.argv[1] if len(sys.argv) > 1 else "."
SRC = os.path.join(HERE, "managed_futures.py")
src = open(SRC, encoding="utf-8").read()
src = src.replace('h = yf.Ticker("GSG").history(period="max", auto_adjust=True)["Close"].dropna()',
                  'raise RuntimeError("offline by verifier")')
cut = src.index("# ================================================================== 1. PROXIES")
NS = {"__file__": SRC, "__name__": "mf_defs"}
exec(compile(src[:cut], SRC, "exec"), NS)
C = NS["C"]
P0 = NS["P"].copy()
idx = P0.index
START, FREEZE = C.START, C.FREEZE
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# =====================================================================================
# A. future-shock lookahead test
# =====================================================================================
log("A. future-shock lookahead test")
KEEP = {"mark", "mark_wk", "month", "gold_src"}
def shocked(P, T, seed):
    rng = np.random.default_rng(seed)
    P2 = P.copy()
    after = P2.index > pd.Timestamp(T)
    n = int(after.sum())
    for c in P2.columns:
        if c in KEEP:
            continue
        if P2[c].dtype == bool:
            P2.loc[after, c] = rng.random(n) > 0.5
        elif np.issubdtype(P2[c].dtype, np.number):
            v = P2.loc[after, c].values.astype(float)
            P2.loc[after, c] = v * rng.lognormal(0, 0.5, n) + rng.normal(0, 0.03, n)
    return P2

def build_with_panel(Pp, **kw):
    NS["P"] = Pp
    try:
        return NS["build_proxy"]("x", **kw)
    finally:
        NS["P"] = P0

proxy_specs = {"main_12m": {}, "blend_1_3_12m": dict(lbs=(21, 63, 252)), "ex_equities": dict(mkts=("ief", "tlt", "gold", "dxy", "gsci")),
               "realized_vol_scaling": dict(scaling="realized")}
base_proxies = {k: build_with_panel(P0, **kw) for k, kw in proxy_specs.items()}
cuts = ["1989-12-29", "1990-03-30", "1999-12-31", "2000-08-31", "2000-09-29", "2001-03-30", "2008-10-31",
        "2018-12-14", "2022-06-30", "2026-07-24", "2026-10-30"]
lookA = []
for j, T in enumerate(cuts):
    Pp = shocked(P0, T, 100 + j)
    Tm = pd.Timestamp(T)
    for k, kw in proxy_specs.items():
        b = base_proxies[k]
        s = build_with_panel(Pp, **kw)
        r_diff = float((b["ret"][idx <= Tm] - s["ret"][idx <= Tm]).abs().max())
        rb = b["POS"].index[b["POS"].index <= Tm]
        pos_diff = float((b["POS"].loc[rb] - s["POS"].loc[rb]).abs().max().max()) if len(rb) else 0.0
        nxt = idx[idx > Tm][:25]
        r_after = float((b["ret"].reindex(nxt) - s["ret"].reindex(nxt)).abs().max()) if len(nxt) else np.nan
        lookA.append((T, k, r_diff, pos_diff, r_after))
    # portfolio level: F3/F2 x off-leg replacement and 20% sleeve with the proxy built on the shocked panel
    s_main = build_with_panel(Pp)
    for eng, (lev, vt) in NS["ENGINES"].items():
        for st in ["replace_ief_offleg", "sleeve20", "offleg_50ief_50tsmom", "replace_gold"]:
            Pa = P0.copy(); Pa["ts_main_12m"] = base_proxies["main_12m"]["ret"]
            Pb = Pp.copy(); Pb["ts_main_12m"] = s_main["ret"]
            NS["P"] = Pa; wa = NS["structure_weights"](lev, vt, "ts_main_12m", st); ra = C.backtest(Pa, wa)
            NS["P"] = Pb; wb = NS["structure_weights"](lev, vt, "ts_main_12m", st); rb_ = C.backtest(Pb, wb)
            NS["P"] = P0
            d = float((ra[idx <= Tm] - rb_[idx <= Tm]).abs().max())
            lookA.append((T, f"{eng}|{st}", d, np.nan, np.nan))
mx = max(x[2] for x in lookA); mxp = np.nanmax([x[3] for x in lookA])
print(f"  cuts tested: {len(cuts)}; series/portfolios per cut: {len(lookA)//len(cuts)}")
print(f"  MAX |return diff| up to T across all = {mx:.3e}   MAX |position diff| at rebalances <= T = {mxp:.3e}")
print("  (sanity: max |return diff| in the 25 sessions after T, proxies) =",
      f"{np.nanmin([x[4] for x in lookA if np.isfinite(x[4])]):.3e} .. {np.nanmax([x[4] for x in lookA if np.isfinite(x[4])]):.3e}")

# =====================================================================================
# B. independent from-spec re-implementation of main_12m
# =====================================================================================
log("B. independent re-implementation of main_12m from the stated spec")
P = P0
rf = P["rf_d"].values
X = np.column_stack([
    (P["spx"] - P["rf_d"]).values, (P["ndx_ret"] - P["rf_d"]).values, (P["intl"] - P["rf_d"]).values,
    (P["ief"] - P["rf_d"]).values, (P["tlt"] - P["rf_d"]).values, (P["gold_phys"] - P["rf_d"]).values,
    P["dxy_ret"].values, P["gsci_ret"].values])
X = np.nan_to_num(X, nan=0.0)
N = X.shape[1]; G = 5
TV = 0.40 / np.sqrt(N)
dates = idx
pos_of = {d: i for i, d in enumerate(dates)}
# complete calendar months only
mon = dates.to_period("M")
last_in_month = pd.Series(np.arange(len(dates)), index=dates).groupby(mon).max().values
if dates[-1] < dates[-1] + pd.offsets.BMonthEnd(0):
    last_in_month = last_in_month[:-1]
first_real_gold = int(np.where(P["gold_src"].values != "monthly-interp")[0][0])
gold_daily_ok = first_real_gold + 126
sig_switch = pd.Timestamp("2000-09-01")
logcum = np.vstack([np.zeros(N), np.cumsum(np.log1p(X), axis=0)])     # logcum[i+1] = sum_{0..i}
# monthly gold returns (calendar months), 36m std, lagged one month, / smoothing ratio
gmon = pd.Series(X[:, G], index=dates).add(1).groupby(mon).prod() - 1
SMOOTH = NS["SMOOTH"]
gstd = (gmon.rolling(36, min_periods=12).std() * np.sqrt(12) / SMOOTH).shift(1)
my_pos = {}
for i in last_in_month:
    if i < 252 + 21 + 5:
        continue
    t = dates[i]
    sig = np.sign(np.exp(logcum[i + 1] - logcum[i + 1 - 252]) - 1)
    if t < sig_switch:
        j = i - 21
        sig[G] = np.sign(np.exp(logcum[j + 1, G] - logcum[j + 1 - 252, G]) - 1) if j + 1 - 252 >= 0 else 0.0
    vol = X[i - 62:i + 1].std(axis=0, ddof=1) * np.sqrt(252)
    early = i < gold_daily_ok
    if early:
        vol[G] = gstd.loc[t.to_period("M")]
    vol = np.maximum(vol, 0.01)
    raw = np.nan_to_num(sig * TV / vol, nan=0.0)
    cols = [c for c in range(N) if not (early and c == G)]
    hs = X[i - 125:i + 1][:, cols] @ raw[cols]
    var = hs.var(ddof=1) * 252
    if early:
        var += (raw[G] * vol[G]) ** 2
    k = min(0.12 / np.sqrt(var), 1.5) if var > 0 else 0.0
    my_pos[i] = raw * k
held = np.zeros_like(X)
cur = np.zeros(N)
for i in range(len(dates)):
    held[i] = cur
    if i in my_pos:
        cur = my_pos[i]
turn = np.r_[0.0, np.abs(np.diff(held, axis=0)).sum(axis=1)]
my_ret = rf + (held * X).sum(axis=1) - turn * C.COST - 0.0075 / 252
my_ret = pd.Series(my_ret, index=dates)
theirs = base_proxies["main_12m"]["ret"]
dB = float((my_ret - theirs).abs().max())
print(f"  max |daily return diff| independent vs specialist main_12m = {dB:.3e}")
s1, s2 = C.stats(my_ret, b=FREEZE), C.stats(theirs, b=FREEZE)
print(f"  independent IS: CAGR {s1['cagr']*100:.2f}% maxDD {s1['maxdd']*100:.1f}%   specialist: CAGR {s2['cagr']*100:.2f}% maxDD {s2['maxdd']*100:.1f}%")

# =====================================================================================
# C. gold pre-2000 information-time audit
# =====================================================================================
log("C. gold pre-2000 information-time audit")
# a monthly-average anchored at calendar month-end: value at trading day d depends on avg(month(d)),
# i.e. on prices up to the LAST TRADING DAY of month(d).  info_time(d) = last trading day of month(d).
interp = P["gold_src"].values == "monthly-interp"
ltd = pd.Series(dates, index=dates).groupby(mon).transform("max")
viol = []
for i in last_in_month:
    t = dates[i]
    if t >= sig_switch or i < 252 + 21 + 5:
        continue
    # signal uses gold levels at i-21 and i-21-252, and every daily return in between
    used = np.arange(i - 21 - 252, i - 21 + 1)
    used = used[used >= 0]
    it = ltd.values[used][interp[used]]
    if len(it) and it.max() > t:
        viol.append((t, it.max()))
# switch months (unlagged signal from 2000-09-01): uses levels up to t; interp part ends 2000-08-30
for i in last_in_month:
    t = dates[i]
    if not (sig_switch <= t < pd.Timestamp("2001-09-30")):
        continue
    used = np.arange(i - 252, i + 1)
    it = ltd.values[used][interp[used]]
    if len(it) and it.max() > t:
        viol.append((t, it.max()))
print(f"  rebalance dates whose gold signal uses information dated after the decision: {len(viol)}")
# share of rebalance months where the 21d lag lands inside the decision month (uses avg of the
# current month, which is legitimately known at its last trading day)
inside = sum(1 for i in last_in_month if dates[i] < sig_switch and i >= 300 and mon[i - 21] == mon[i])
tot = sum(1 for i in last_in_month if dates[i] < sig_switch and i >= 300)
print(f"  pre-2000 rebalances where t-21 falls inside the decision month: {inside}/{tot}")
# empirical: autocorrelation of monthly interp-gold returns (smoothing) at lag 1/2
gm_pre = gmon[gmon.index < pd.Period("2000-08", "M")]
print(f"  interp gold monthly-return autocorr lag1 {gm_pre.autocorr(1):+.3f} lag2 {gm_pre.autocorr(2):+.3f} "
      f"(RW + monthly averaging predicts +0.25 / 0)")
# sensitivity: gold leg flat until 126 real daily obs exist (no use of interpolated gold at all)
def gold_flat_proxy(base_kw=None):
    kw = base_kw or {}
    b = build_with_panel(P0, **kw)
    return b
NS["P"] = P0
pr_gf = NS["build_proxy"]("gold_flat_pre2001")   # rebuild then zero the gold leg manually below
POS = pr_gf["POS"].copy()
early_rb = POS.index < NS["GOLD_DAILY_OK"]
# recompute K without gold in early months: simplest faithful variant -> drop gold from book pre-2001-03
X_df = NS["excess_returns"]()[NS["MKTS"]]
S_df = NS["make_signals"](X_df, (252,))
S_df.loc[S_df.index < NS["GOLD_DAILY_OK"], "gold"] = 0.0
V_df = NS["make_vols"](X_df)
RB = POS.index
raw = (S_df.loc[RB] * (0.40 / np.sqrt(8)) / V_df.loc[RB]).fillna(0.0)
Kl = []
for t in RB:
    i = idx.get_loc(t)
    W = X_df.iloc[i - 125:i + 1]
    var = float((W @ raw.loc[t]).var() * 252)
    Kl.append(min(0.12 / np.sqrt(var), 1.5) if var > 0 else 0.0)
POSg = raw.mul(pd.Series(Kl, index=RB), axis=0)
heldg = POSg.reindex(idx).ffill().shift(1).fillna(0.0)
retg = P0["rf_d"] + (heldg * X_df).sum(axis=1) - heldg.diff().abs().sum(axis=1).fillna(0) * C.COST - 0.0075 / 252
P0["ts_goldflat"] = retg
NS["P"] = P0
outC = {}
for eng, (lev, vt) in NS["ENGINES"].items():
    base = C.backtest(P0, NS["F"](lev, vt))
    for st in ["sleeve20", "replace_ief_offleg"]:
        r = C.backtest(P0, NS["structure_weights"](lev, vt, "ts_goldflat", st))
        outC[f"{eng}|{st}"] = C.stats(r, b=FREEZE)["calmar"] - C.stats(base, b=FREEZE)["calmar"]
sg = C.stats(retg, b=FREEZE)
exg = (retg - P0["rf_d"])[(idx >= START) & (idx <= FREEZE)]
print(f"  gold-flat-until-2001-03 proxy: IS exSharpe {exg.mean()/exg.std()*np.sqrt(252):.3f}  CAGR {sg['cagr']*100:.2f}%; dCalmar:",
      {k: round(v, 3) for k, v in outC.items()})

# =====================================================================================
# D. intra-month drift: implicit daily rebalancing of constant notional weights
# =====================================================================================
log("D. drift / implicit daily rebalancing")
pm = base_proxies["main_12m"]
held_c = pm["held"]; Xm = pm["X"]
rp = pm["ret"]
# with constant weights, the book must trade back to target every day: |w (1+x_i)/(1+r_p) - w|
drift_turn = (held_c.abs() * ((Xm.add(1)).div(1 + rp, axis=0) - 1).abs()).sum(axis=1)
m_is = (idx >= START) & (idx <= FREEZE)
print(f"  implicit daily-rebalance turnover {drift_turn[m_is].sum()/(m_is.sum()/252):.2f}x NAV/yr "
      f"-> uncharged cost at 3bp = {drift_turn[m_is].mean()*252*C.COST*100:.3f}%/yr  (charged monthly turnover {pm['turn'][m_is].sum()/(m_is.sum()/252):.2f}x)")
# hold-contracts variant: within the month weights drift with (1+x_i)/(1+r_p)
POSm = pm["POS"]
W = np.zeros((len(idx), Xm.shape[1])); cur = np.zeros(Xm.shape[1]); rets = np.zeros(len(idx)); turn_d = np.zeros(len(idx))
rbset = set(idx.get_indexer(POSm.index))
Xv = Xm.values; rfv = P0["rf_d"].values
for i in range(len(idx)):
    w_now = cur.copy()
    g = (w_now * Xv[i]).sum()
    r_i = rfv[i] + g
    rets[i] = r_i
    cur = w_now * (1 + Xv[i]) / (1 + r_i) if (1 + r_i) != 0 else w_now
    if i in rbset:
        tgt = POSm.iloc[list(POSm.index).index(idx[i])].values
        turn_d[min(i + 1, len(idx) - 1)] += np.abs(tgt - cur).sum()
        cur = tgt.copy()
rets = pd.Series(rets - turn_d * C.COST - 0.0075 / 252, index=idx)
e1 = (rets - P0["rf_d"])[m_is]; e0 = (rp - P0["rf_d"])[m_is]
s_d, s_c = C.stats(rets, b=FREEZE), C.stats(rp, b=FREEZE)
print(f"  hold-contracts-within-month variant: IS exSharpe {e1.mean()/e1.std()*np.sqrt(252):.3f} vs constant-weight {e0.mean()/e0.std()*np.sqrt(252):.3f};"
      f" CAGR {s_d['cagr']*100:.2f}% vs {s_c['cagr']*100:.2f}%; 2022 {C.period_return(rets,'2022-01-01','2022-12-31')*100:.1f}% vs {C.period_return(rp,'2022-01-01','2022-12-31')*100:.1f}%")
P0["ts_drift"] = rets
NS["P"] = P0
for eng, (lev, vt) in NS["ENGINES"].items():
    base = C.backtest(P0, NS["F"](lev, vt))
    vals = {}
    for st in ["sleeve20", "replace_ief_offleg"]:
        r = C.backtest(P0, NS["structure_weights"](lev, vt, "ts_drift", st))
        vals[st] = round(C.stats(r, b=FREEZE)["calmar"] - C.stats(base, b=FREEZE)["calmar"], 4)
    print(f"  {eng} dCalmar with drift proxy: {vals}")

# =====================================================================================
# E. C3: vols on F risk-off days; off-leg weight while risk-on
# =====================================================================================
log("E. C3 regime numbers")
ron = P0["above"].astype(float).where(P0["mark"]).ffill().shift(1).fillna(0.0) > 0.5
for lbl, mask in (("risk_off", (~ron) & m_is), ("risk_on", ron & m_is), ("all IS", pd.Series(m_is, index=idx))):
    ep = (rp - P0["rf_d"])[mask]; ei = (P0["ief"] - P0["rf_d"])[mask]
    print(f"  {lbl:8s} days {int(mask.sum()):5d}  proxy ex-ret {ep.mean()*252*100:5.2f}%/yr vol {ep.std()*np.sqrt(252)*100:5.1f}%  "
          f"IEF ex-ret {ei.mean()*252*100:5.2f}%/yr vol {ei.std()*np.sqrt(252)*100:5.1f}%   proxy/IEF Sharpe {ep.mean()/ep.std()*np.sqrt(252):.2f}/{ei.mean()/ei.std()*np.sqrt(252):.2f}")
for eng, (lev, vt) in NS["ENGINES"].items():
    hw = C.held_weights(P0, NS["F"](lev, vt))
    off_on = hw["ief"][ron & m_is].mean(); off_off = hw["ief"][(~ron) & m_is].mean()
    share = (hw["ief"][ron & m_is].sum()) / hw["ief"][m_is].sum()
    print(f"  {eng}: mean IEF off-leg weight risk-on days {off_on:.3f}, risk-off days {off_off:.3f}; share of off-leg exposure-days that are risk-on {share:.2f}")

# =====================================================================================
# F. episode dependence of the off-leg verdict (pickled series from the rerun)
# =====================================================================================
log("F. episode dependence")
pk = os.path.join(SCR, "mf_series.pkl")
if os.path.exists(pk):
    S = pickle.load(open(pk, "rb"))
    allp = {**S["port"], **S["rob"]}
    for eng in ["F3", "F2"]:
        b = S["base"][eng]
        for st in ["replace_ief_offleg", "sleeve20"]:
            rows = {k: v for k, v in allp.items() if k.startswith(f"{eng} | {st} | ") and not k.endswith("control_always_long")}
            out = {}
            for lbl, a in (("1988-start", "1988-01-01"), ("1990-06-start", "1990-06-01"), ("1991-start", "1991-01-01"), ("1996-start", "1996-01-01")):
                bc = C.stats(b, a, FREEZE, start=None)["calmar"]
                d = np.array([C.stats(v, a, FREEZE, start=None)["calmar"] - bc for v in rows.values()])
                out[lbl] = (int((d < 0).sum()), len(d), round(float(np.median(d)), 3), round(float(d.min()), 3), round(float(d.max()), 3))
            # where is each variant's IS maxDD trough?
            tr = []
            for v in rows.values():
                s = v[(v.index >= START) & (v.index <= FREEZE)]
                e = (1 + s).cumprod(); dd = e / e.cummax() - 1
                tr.append(str(dd.idxmin().date())[:7])
            vc = pd.Series(tr).value_counts().to_dict()
            print(f"  {eng} {st:20s} (#neg, n, median, min, max) by window: {out}")
            print(f"       maxDD trough month across the {len(rows)} variants: {vc}")
        e = (1 + b[(b.index >= START) & (b.index <= FREEZE)]).cumprod(); dd = e / e.cummax() - 1
        print(f"  {eng} baseline maxDD trough {dd.idxmin().date()}  depth {dd.min()*100:.1f}%")
else:
    print("  (run verify_managed_futures_integrity_rerun.py first)")
log("done")
