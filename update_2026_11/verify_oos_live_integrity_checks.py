"""
INTEGRITY REVIEW of oos_live (adversarial reviewer) -- step 2: independent re-implementation + probes.

Does NOT import common.py or oos_live.py.  Reads panel.parquet / oos_live_yf.parquet / oos_live.json only.
Writes nothing (prints only).

 A. independent weekly engine (own EMA/vol/ER, own Friday calendar, own holdings loop), OOS on real ETFs
 B. independent 81-day window percentiles
 C. NDX date-alignment probe (would a shifted NDX create look-ahead in lag-1?)
 D. signature of the 4 worst historical F3 weeks (claim C3)
 E. day-level decomposition of lag-2 / Monday-open vs Friday-close OOS (claim C8)
 F. pre-2000 gold interpolation sensitivity (IS stats + percentiles)
 G. open-price quality (Monday-open variant)
 H. misc: yields path, maxDD dates, TLT-IEF gap
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
J = json.load(open(os.path.join(HERE, "oos_live.json")))
idx = P.index
FREEZE = pd.Timestamp("2026-07-24")
START = pd.Timestamp("1988-01-01")
OOS = idx[idx > FREEZE]
COST = 3e-4

# ------------------------------------------------------------------ A. independent engine
ndx = P["ndx"].astype(float)
ema = ndx.ewm(span=100, adjust=False).mean()
vol = ndx.pct_change().rolling(20).std() * np.sqrt(252)
er = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum()
above = ndx > ema
fri = pd.Series(idx.weekday == 4, index=idx)
print("A0 Friday calendar == panel mark:", bool((fri == P["mark"]).all()),
      "| signals equal panel:", float((ema - P.ema100).abs().max()), float((vol - P.vol20).abs().max()),
      float((er - P.er20).abs().max()))


def targets(lev, voltgt, chop_pos=0.5, seasonal=True, gold_w=0.15, off="ief", gold="gold_phys"):
    rows = {}
    for t in idx[fri.values]:
        if above[t]:
            v = vol[t]
            s = 1.0 if not np.isfinite(v) else min(1.0, voltgt / v)
            if seasonal and t.month in (8, 9):
                s *= 0.5
            e = er[t]
            if not (np.isfinite(e) and e >= 0.25):
                s *= chop_pos
        else:
            s = 0.0
        core = 1 - gold_w
        d = {lev: core * s, off: core * (1 - s)}
        if gold_w > 0:
            d[gold] = gold_w
        rows[t] = d
    return pd.DataFrame(rows).T.fillna(0.0)


def run(T, R, lag=1):
    """constant-mix holdings: weights decided at Friday t are held from the (lag)-th session after t."""
    cols = list(T.columns)
    pos = {t: i for i, t in enumerate(idx)}
    H = pd.DataFrame(np.nan, index=idx, columns=cols)
    for t, row in T.iterrows():
        j = pos[t] + lag
        if j < len(idx):
            H.iloc[j] = row.values
    H = H.ffill().fillna(0.0)
    gross = (H * R[cols].fillna(0.0)).sum(axis=1)
    turn = H.diff().abs().sum(axis=1).fillna(0.0)
    return gross - COST * turn, H


REALCOL = {"tqqq": "tqqq_real", "qld": "qld_real", "qqq": "qqq_real", "ief": "ief_real", "tlt": "tlt_real",
           "bills": "bil_real", "gold_phys": "gld_real"}
RR = P.copy()
for k, c in REALCOL.items():
    RR[k] = P[c]
for c in REALCOL.values():
    assert P.loc[OOS, c].notna().all(), c

SPEC = {"a": dict(lev="tqqq", voltgt=0.20, off="tlt"), "b": dict(lev="tqqq", voltgt=0.20, off="ief"),
        "c": dict(lev="qld", voltgt=0.30, off="ief"),
        "d": dict(lev="tqqq", voltgt=0.25, chop_pos=1.0, seasonal=False, gold_w=0.0, off="bills"),
        "ref": dict(lev="tqqq", voltgt=0.20, off="bond10_old", gold="gold_miners")}
KEY = {"a": "a_F_scanner_TQQQ_TLT_GLD", "b": "b_F3_TQQQ_IEF_GLD", "c": "c_F2_QLD_IEF_GLD_vt30",
       "d": "d_A_TQQQ_BIL_vt25", "ref": "ref_Fjuly_TQQQ_bond10syn_XAUminers"}


def stats(r, a=START, b=FREEZE):
    s = r[(r.index >= a) & (r.index <= b)].dropna()
    e = (1 + s).cumprod()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    dd = e / e.cummax() - 1
    return dict(cagr=cagr, maxdd=dd.min(), calmar=cagr / -dd.min(), dd_date=str(dd.idxmin().date()))


TG, RES = {}, {}
print("\nA1 independent engine vs published (OOS real ETFs, lag 1 / lag 2; IS model lag 1 / lag 2)")
for k, sp in SPEC.items():
    T = targets(**sp)
    TG[k] = T
    r1, H1 = run(T, RR, 1)
    r2, _ = run(T, RR, 2)
    m1, _ = run(T, P, 1)
    m2, _ = run(T, P, 2)
    RES[k] = dict(r1=r1, r2=r2, m1=m1, m2=m2, H1=H1)
    o1 = (1 + r1.loc[OOS]).prod() - 1
    o2 = (1 + r2.loc[OOS]).prod() - 1
    s1, s2 = stats(m1), stats(m2)
    pub = J["results_table"][KEY[k]]
    print(f"  {k:4s} OOS lag1 {o1*100:7.3f}% (pub {pub['oos_lag1']*100:7.3f})  lag2 {o2*100:7.3f}% "
          f"(pub {pub['oos_lag2']*100:7.3f}) | IS lag1 {s1['cagr']*100:5.2f}%/{s1['maxdd']*100:6.2f}%/{s1['calmar']:.3f} "
          f"[DD {s1['dd_date']}]  lag2 {s2['cagr']*100:5.2f}%/{s2['maxdd']*100:6.2f}%/{s2['calmar']:.3f} [DD {s2['dd_date']}]"
          f"  (pub IS {pub['is_cagr']*100:5.2f}/{pub['is_maxdd']*100:6.2f}/{pub['is_calmar']:.3f})")

# independent share-drift OOS (Friday close) for b,c
print("\nA2 independent share-level drift, trade at Friday close (OOS):")
for k in ["a", "b", "c", "d"]:
    T = TG[k]
    cols = list(T.columns)
    d0 = idx[(idx >= pd.Timestamp("2026-01-02"))]
    first_dec = [t for t in d0 if fri[t]][0]
    h = T.loc[first_dec].values.astype(float).copy()
    V = 1.0
    eq = {}
    for i in range(list(idx).index(first_dec) + 1, len(idx)):
        d, prevd = idx[i], idx[i - 1]
        if fri[prevd]:
            tgt = T.loc[prevd].values
            Vt = h.sum()
            c_ = COST * np.abs(tgt - h / Vt).sum()
            h = Vt * (1 - c_) * tgt
        h = h * (1 + RR.loc[d, cols].values.astype(float))
        eq[d] = h.sum()
    eq = pd.Series(eq)
    print(f"  {k}: {(eq.loc[OOS[-1]] / eq.loc[FREEZE] - 1)*100:7.3f}%  (pub drift {J['results_table'][KEY[k]]['oos_drift_fri_close']*100:7.3f})")

# ------------------------------------------------------------------ B. percentiles
print("\nB  independent 81-day window percentiles (model, lag 1, 1988-01-04..freeze)")
N = len(OOS)
for k in ["a", "b", "c", "d"]:
    m = RES[k]["m1"]
    s = m[(m.index >= START) & (m.index <= FREEZE)]
    e = np.concatenate([[1.0], (1 + s.values).cumprod()])
    w = e[N:] / e[:-N] - 1
    o = (1 + RES[k]["r1"].loc[OOS]).prod() - 1
    s10 = m[(m.index >= pd.Timestamp("2010-02-12")) & (m.index <= FREEZE)]
    e10 = np.concatenate([[1.0], (1 + s10.values).cumprod()])
    w10 = e10[N:] / e10[:-N] - 1
    sg = m[(m.index >= pd.Timestamp("2000-09-01")) & (m.index <= FREEZE)]
    eg = np.concatenate([[1.0], (1 + sg.values).cumprod()])
    wg = eg[N:] / eg[:-N] - 1
    print(f"  {k}: pct {100*(w <= o).mean():5.2f} (pub {J['percentiles'][KEY[k]]['pct_rank_return']:5.2f}) | 2010-02-12+ "
          f"{100*(w10 <= o).mean():5.2f} | post-interp-gold 2000-09+ {100*(wg <= o).mean():5.2f} | 1988-1999 only "
          f"{100*(w[:len(s[s.index<='1999-12-31'])] <= o).mean():5.2f} | n={len(w)} median {np.median(w)*100:5.2f}% 5th {np.percentile(w,5)*100:6.2f}%")

# ------------------------------------------------------------------ C. NDX alignment probe
print("\nC  date alignment: corr(ndx_ret_t, X_{t+k}), k=-1,0,+1")
for nm, (a, b) in {"1985-1989": ("1985-10-01", "1989-12-31"), "1990-1994": ("1990-01-01", "1994-12-31"),
                   "1995-1999": ("1995-01-01", "1999-12-31"), "2000-2009": ("2000-01-01", "2009-12-31"),
                   "2010-2026": ("2010-01-01", "2026-11-17")}.items():
    x = P.loc[a:b]
    out = []
    for col in ["spx", "qqq_real"]:
        if x[col].notna().sum() < 200 or (col == "qqq_real" and a < "1999"):
            continue
        cs = [x["ndx_ret"].corr(x[col].shift(-kk)) for kk in (-1, 0, 1)]
        out.append(f"{col}: " + " ".join(f"{c:+.3f}" for c in cs))
    print(f"  {nm}: " + " | ".join(out))
print("  ndx_ret == ndx.pct_change():", float((P.ndx_ret - P.ndx.pct_change()).abs().max()))
# NDX weekly (Fri-Fri) autocorrelation by era, and corr(sign of Friday weight change, next-day NDX return)
fr = ndx[fri.values]
wk = fr.pct_change()
for nm, (a, b) in {"1988-1999": ("1988", "1999"), "2000-2009": ("2000", "2009"), "2010-2026": ("2010", "2026-07-24")}.items():
    print(f"  weekly NDX autocorr {nm}: {wk.loc[a:b].autocorr(1):+.3f}")

# lag-1 minus lag-2 gap decomposition (model, F3): Monday-after-decision P&L of the weight change
print("\nC2 F3 model: where does lag1 - lag2 come from?  sum over Mondays of (dW_TQQQ x r_TQQQ) etc.")
m1, m2 = RES["b"]["m1"], RES["b"]["m2"]
diff = np.log1p(m1) - np.log1p(m2)
for nm, (a, b) in {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-2009": ("2000-01-01", "2009-12-31"),
                   "2010-freeze": ("2010-01-01", "2026-07-24")}.items():
    dd = diff.loc[a:b]
    yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    T = TG["b"]
    gate = (T["tqqq"] > 0).astype(int).diff().fillna(0)
    flips = set(T.index[gate.values != 0])
    pos = {t: i for i, t in enumerate(idx)}
    on_flip = sum(dd.get(idx[pos[t] + 1], 0.0) for t in flips if pos[t] + 1 < len(idx) and a <= str(idx[pos[t] + 1].date()) <= b)
    print(f"  {nm}: total log gap {dd.sum():+.3f} ({dd.sum()/yrs*100:+.2f}%/yr) ; on day after gate flips {on_flip:+.3f} "
          f"({on_flip/yrs*100:+.2f}%/yr), rest {dd.sum()-on_flip:+.3f}; n flips {sum(1 for t in flips if a <= str(t.date()) <= b)}")

# ------------------------------------------------------------------ D. four worst weeks (claim C3)
print("\nD  decision-time signature of the worst F3 held weeks (model lag 1)")
eq = (1 + RES["b"]["m1"]).cumprod()
marks = idx[fri.values]
wkr = (eq.loc[marks] / eq.loc[marks].shift(1) - 1).dropna()
h = wkr[(wkr.index >= "1988-01-04") & (wkr.index <= FREEZE)]
worst = h.nsmallest(8)
pos = {t: i for i, t in enumerate(marks)}
for t, v in list(worst.items()) + [(pd.Timestamp("2026-10-16"), wkr.loc["2026-10-16"])]:
    dec = marks[pos[t] - 1]
    print(f"  week ending {t.date()} {v*100:7.2f}% | decided {dec.date()}: TQQQ {TG['b'].loc[dec,'tqqq']:.3f} "
          f"vol20 {vol[dec]:.3f} ER {er[dec]:.3f} month {dec.month} | NDX week {ndx[t]/ndx[dec]-1:+.4f}")
print(f"  hist weeks {len(h)}, worse than 2026-10-16: {(h < wkr.loc['2026-10-16']).sum()}")

# ------------------------------------------------------------------ E. lag-2 / Monday-open day decomposition (OOS)
print("\nE  OOS F3: per-day contribution (log) of lag2 - lag1 (constant mix) -- top days")
r1, r2 = RES["b"]["r1"].loc[OOS], RES["b"]["r2"].loc[OOS]
dl = (np.log1p(r2) - np.log1p(r1))
print("   total", round(float(np.expm1(np.log1p(r2).sum()) - np.expm1(np.log1p(r1).sum())) * 100, 3), "pp;",
      {str(k.date()): round(v * 100, 3) for k, v in dl[dl.abs() > 0.002].items()})
print("   r1 vs r2 on those days:", {str(k.date()): (round(r1[k]*100, 2), round(r2[k]*100, 2)) for k in dl[dl.abs() > 0.002].index})

Y = pd.read_parquet(os.path.join(HERE, "oos_live_yf.parquet"))
YC = Y.pivot(index="date", columns="ticker", values="close")
YO = Y.pivot(index="date", columns="ticker", values="open")
for t in ["TQQQ", "QLD", "QQQ", "IEF", "GLD", "^NDX"]:
    on = YO[t] / YC[t].shift(1) - 1
    print(f"   {t:5s} overnight 2026-10-19 {on.loc['2026-10-19']*100:+.2f}%  intraday {(YC[t]/YO[t]-1).loc['2026-10-19']*100:+.2f}%"
          f" | 2026-10-05 close-close {(YC[t].pct_change()).loc['2026-10-05']*100:+.2f}%")

# ------------------------------------------------------------------ F. gold interpolation sensitivity
print("\nF  pre-2000-09 gold: monthly-interpolated (uses next month-end price). Alternatives with no foresight:")
src = P["gold_src"]
interp = src == "monthly-interp"
g = P["gold_phys"].copy()
# (i) step: whole month's return booked on the last trading day of the month (known at that close)
mon = idx.to_period("M")
gm = (1 + g[interp]).groupby(mon[interp.values]).prod() - 1
g_step = g.copy()
g_step[interp] = 0.0
last_days = pd.Series(idx[interp.values], index=idx[interp.values]).groupby(mon[interp.values]).max()
for per, d in last_days.items():
    g_step[d] = gm[per]
# (ii) the July proxy gold miners pre-2000 (real daily, but miners not gold)
g_min = g.copy()
g_min[interp] = P.loc[interp, "gold_miners"]
for nm, gs in [("as published (interp)", g), ("step at month-end", g_step), ("miners pre-2000-09", g_min)]:
    PP = P.copy()
    PP["gold_phys"] = gs
    for k in ["b", "c"]:
        m, _ = run(TG[k], PP, 1)
        s = stats(m)
        sw = m[(m.index >= START) & (m.index <= FREEZE)]
        e = np.concatenate([[1.0], (1 + sw.values).cumprod()])
        w = e[N:] / e[:-N] - 1
        o = (1 + RES[k]["r1"].loc[OOS]).prod() - 1
        s9 = stats(m, START, pd.Timestamp("1999-12-31"))
        print(f"  {nm:24s} {k}: IS {s['cagr']*100:5.2f}% / {s['maxdd']*100:6.2f}% [{s['dd_date']}] / Calmar {s['calmar']:.3f} | "
              f"1988-99 maxDD {s9['maxdd']*100:6.2f}% | OOS pct {100*(w<=o).mean():5.2f}")
print(f"  interp days in 1988..freeze: {int(interp.loc[START:FREEZE].sum())}; daily vol interp {g[interp].loc[START:].std()*np.sqrt(252):.3f}"
      f" vs step {g_step[interp].loc[START:].std()*np.sqrt(252):.3f} vs GLD era {P.gld_real.std()*np.sqrt(252):.3f}")

# ------------------------------------------------------------------ G. open-price quality 2010..2026
print("\nG  Monday-open data quality (adjusted OHLC), 2010-02-19..2026-11-17")
on = {}
for t in ["TQQQ", "QLD", "QQQ", "IEF", "TLT", "GLD", "BIL"]:
    c = YC[t].dropna()
    o = YO[t].reindex(c.index)
    x = (o / c.shift(1) - 1).loc["2010-02-19":]
    on[t] = x
    same = (o == c.shift(1)).loc["2010-02-19":].mean()
    oc = ((o == c) | (o == YC[t].reindex(c.index))).loc["2010-02-19":].mean()
    print(f"  {t:5s} open==prev close {same*100:5.2f}%  open==close {((o==c).loc['2010-02-19':]).mean()*100:5.2f}%  "
          f"|overnight| mean {x.abs().mean()*100:.3f}%")
a = pd.concat([on["TQQQ"], on["QQQ"], on["QLD"]], axis=1).dropna()
a.columns = ["t", "q", "l"]
print(f"  corr(TQQQ overnight, QQQ overnight) {a.t.corr(a.q):.3f}  slope {np.polyfit(a.q, a.t, 1)[0]:.2f};  "
      f"corr(QLD, QQQ) {a.l.corr(a.q):.3f} slope {np.polyfit(a.q, a.l, 1)[0]:.2f}")
bad = a[(a.t - 3 * a.q).abs() > 0.03]
print(f"  days with |TQQQ_on - 3 QQQ_on| > 3pp: {len(bad)}", [str(d.date()) for d in bad.index[:12]])
# calendar mismatch between ETF data and the NDX calendar
for t in ["TQQQ", "QLD", "IEF", "GLD", "BIL"]:
    c = YC[t].dropna().loc["2010-02-19":"2026-11-17"]
    print(f"  {t}: ETF dates not in NDX calendar {len(c.index.difference(idx))}, NDX dates missing in ETF "
          f"{len(idx[(idx >= '2010-02-19')].difference(c.index))}")

# ------------------------------------------------------------------ H. misc
print("\nH  yields (tnx) path:", {d: round(float(P.at[pd.Timestamp(d), 'tnx']), 3) for d in
                                 ["2026-06-30", "2026-07-24", "2026-08-25", "2026-10-02", "2026-10-05", "2026-10-09",
                                  "2026-10-16", "2026-11-17"]},
      "max in risk-on phase", round(float(P.loc["2026-07-27":"2026-10-16", "tnx"].max()), 3),
      P.loc["2026-07-27":"2026-10-16", "tnx"].idxmax().date())
print("   TLT-IEF strategy gap (a - b) OOS pp:", round((J['results_table'][KEY['a']]['oos_lag1'] - J['results_table'][KEY['b']]['oos_lag1']) * 100, 3))
print("   QQQ / TQQQ 10-19..11-17:", round(((1 + P.loc['2026-10-19':, 'qqq_real']).prod() - 1) * 100, 2),
      round(((1 + P.loc['2026-10-19':, 'tqqq_real']).prod() - 1) * 100, 2))
