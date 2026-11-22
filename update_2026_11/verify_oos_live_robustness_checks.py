"""
verify_oos_live_robustness_checks.py -- ADVERSARIAL REVIEW of oos_live (ROBUSTNESS lens).

Independent re-implementation (own window code, own weekly bookkeeping) of the statistics behind
claims C2, C3, C4, C6, C8, plus sensitivity tests the specialist did not run:
  1. percentile sensitivity: era, non-overlapping windows (all 81 offsets), conditioning bandwidth,
     NDX PATH-shape conditioning (rally-then-fall), lag-2-consistent history
  2. power / likelihood ratio of the OOS draw under "backtest edge" vs "no edge" (shifted) distributions
  3. tail week: base rate of the claimed "signature", conditional frequency, real-ETF-era rank,
     clustering of the worse weeks
  4. overlay cost (-13pp) vs its own historical 81-day distribution; overlay benefit by halves
  5. TLT-vs-IEF 81-day difference vs its own historical distribution
  6. lag: where the lag-2 maxDD sits; 1988-93 vs 1994-99; Friday->Monday autocorrelation;
     OOS day-level decomposition of lag1 vs lag2 and of the Monday-open gap
  7. un-rounded independent-yfinance diffs; 10y yield path
Writes nothing except stdout. Reads panel.parquet, oos_live.json, oos_live_yf.parquet (all read-only).
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel().copy()
IDX = P.index
FREEZE = C.FREEZE
OOS = IDX[IDX > FREEZE]
A0, END = OOS[0], OOS[-1]
N = len(OOS)
HIST_A = IDX[IDX >= C.START][0]
J = json.load(open(os.path.join(HERE, "oos_live.json")))
COUNT = []          # every variant this reviewer evaluates


def reg(x):
    if x not in COUNT:
        COUNT.append(x)


def tot(r, a, b):
    s = r.loc[a:b]
    return float((1 + s).prod() - 1)


# ---------------------------------------------------------------- strategies (pre-registered specs)
SPEC = {
    "a": dict(lev="tqqq", voltgt=0.20, off="tlt", gold="gold_phys", gold_w=0.15),
    "b": dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15),
    "c": dict(lev="qld", voltgt=0.30, off="ief", gold="gold_phys", gold_w=0.15),
    "d": dict(lev="tqqq", voltgt=0.25, chop_pos=1.0, seasonal=False, gold_w=0.0, off="bills"),
    "b_noseas": dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15, seasonal=False),
    "b_nochop": dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15, chop_pos=1.0),
    "b_neither": dict(lev="tqqq", voltgt=0.20, off="ief", gold="gold_phys", gold_w=0.15, chop_pos=1.0,
                      seasonal=False),
    "c_neither": dict(lev="qld", voltgt=0.30, off="ief", gold="gold_phys", gold_w=0.15, chop_pos=1.0,
                      seasonal=False),
    "static8515": None,
}
W, R1, R2 = {}, {}, {}
for k, kw in SPEC.items():
    if kw is None:
        W[k] = pd.DataFrame({"ief": 0.85, "gold_phys": 0.15}, index=IDX)
    else:
        W[k] = C.strategy_weights(P, **kw)
    R1[k] = C.backtest(P, W[k], lag=1)
    R2[k] = C.backtest(P, W[k], lag=2)
    reg(f"{k} lag1"); reg(f"{k} lag2")
R1["ndx"] = P["ndx_ret"].fillna(0)

oos = {k: tot(r, A0, END) for k, r in R1.items()}
oos2 = {k: tot(r, A0, END) for k, r in R2.items()}
print("OOS lag1 totals (own rebuild):", {k: round(v * 100, 3) for k, v in oos.items()})
print("OOS lag2 totals (own rebuild):", {k: round(v * 100, 3) for k, v in oos2.items()})


# ---------------------------------------------------------------- own window machinery (cumprod based)
def wins(r, a=HIST_A, b=FREEZE, n=N):
    s = r.loc[a:b].fillna(0).values
    e = np.concatenate([[1.0], np.cumprod(1 + s)])
    ret = e[n:] / e[:-n] - 1
    return ret


def ndx_path_feats(a=HIST_A, b=FREEZE, n=N):
    s = P["ndx_ret"].loc[a:b].fillna(0).values
    e = np.concatenate([[1.0], np.cumprod(1 + s)])
    from numpy.lib.stride_tricks import sliding_window_view as swv
    Wl = swv(e, n + 1)                      # each row: equity path start..end (n+1 points)
    base = Wl[:, :1]
    rel = Wl / base
    peak = rel.max(axis=1) - 1              # best point vs start
    endr = rel[:, -1] - 1
    from_peak = rel[:, -1] / rel.max(axis=1) - 1
    argpk = rel.argmax(axis=1)
    return endr, peak, from_peak, argpk


def pct(h, x):
    return float(100 * (h <= x).mean())


print("\n================ 1. PERCENTILE SENSITIVITY (C2)")
hb = {k: wins(R1[k]) for k in ["a", "b", "c", "d", "ndx", "static8515"]}
endr, peak, from_peak, argpk = ndx_path_feats()
assert np.allclose(endr, hb["ndx"])
o_end = oos["ndx"]
e_oos = (1 + P.loc[OOS, "ndx_ret"]).cumprod()
o_peak = float(e_oos.max() - 1)
o_from_peak = float(e_oos.iloc[-1] / e_oos.max() - 1)
o_argpk = int(np.argmax(np.concatenate([[1.0], e_oos.values])))
print(f"OOS NDX path: end {o_end*100:.2f}%  peak gain {o_peak*100:.2f}% (day {o_argpk}/{N})  end vs peak {o_from_peak*100:.2f}%")
for k in ["a", "b", "c", "d"]:
    h = hb[k]
    print(f"  [{k}] all-windows pct {pct(h, oos[k]):5.2f}  (n={len(h)})   median {np.median(h)*100:5.2f}%")
for era, a in [("2000+", "2000-01-01"), ("2007+", "2007-01-01"), ("2010+ real TQQQ", "2010-02-12")]:
    print(f"  era {era:16s}: " + "  ".join(f"{k} {pct(wins(R1[k], a=pd.Timestamp(a)), oos[k]):5.2f}" for k in "abcd"))

# non-overlapping windows: every possible offset
print("  non-overlapping 81d windows (each of 81 offsets, ~118 windows each): pct range over offsets")
for k in ["a", "b", "c"]:
    h = hb[k]
    ps = [pct(h[o::N], oos[k]) for o in range(N)]
    print(f"    [{k}] min {min(ps):5.2f}  p10 {np.percentile(ps,10):5.2f}  median {np.median(ps):5.2f}  p90 {np.percentile(ps,90):5.2f}  max {max(ps):5.2f}  (n per offset {len(h[0::N])})")

# conditioning bandwidth
print("  conditional on NDX 81d return within +-bw of OOS (-4.0%): pct of F3 / F2 / scanner-F, #windows, ~episodes")


def episodes(mask, gap=N):
    pos = np.flatnonzero(mask)
    return 0 if len(pos) == 0 else int(1 + (np.diff(pos) > gap).sum())


for bw in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    m = np.abs(endr - o_end) <= bw
    print(f"    bw +-{bw*100:4.1f}pp: b {pct(hb['b'][m], oos['b']):5.1f}  c {pct(hb['c'][m], oos['c']):5.1f}  "
          f"a {pct(hb['a'][m], oos['a']):5.1f}  d {pct(hb['d'][m], oos['d']):5.1f}  n={m.sum():5d} ep~{episodes(m)}")

# PATH conditioning (the specialist conditions only on the END return; OOS path = rally +11% then -13.5%)
print("  PATH-shape conditioning (rally then fall):")
conds = {
    "end+-4pp & peak>=+6%": (np.abs(endr - o_end) <= 0.04) & (peak >= 0.06),
    "end+-4pp & peak>=+8%": (np.abs(endr - o_end) <= 0.04) & (peak >= 0.08),
    "end+-6pp & peak>=+8% & fromPeak<=-10%": (np.abs(endr - o_end) <= 0.06) & (peak >= 0.08) & (from_peak <= -0.10),
    "peak>=+8% & fromPeak<=-10% (any end)": (peak >= 0.08) & (from_peak <= -0.10),
    "peak>=+6% & fromPeak<=-10% & peak in 2nd half": (peak >= 0.06) & (from_peak <= -0.10) & (argpk >= N // 2),
    "end+-4pp & fromPeak<=-12%": (np.abs(endr - o_end) <= 0.04) & (from_peak <= -0.12),
}
PATHRES = {}
for nm, m in conds.items():
    if m.sum() == 0:
        print(f"    {nm:48s} no windows"); continue
    rowp = {k: pct(hb[k][m], oos[k]) for k in "abcd"}
    PATHRES[nm] = dict(n=int(m.sum()), ep=episodes(m), **rowp, med_b=float(np.median(hb["b"][m])))
    starts = P.loc[HIST_A:FREEZE].index[:len(endr)][m]
    yrs = sorted(set(starts.year))
    print(f"    {nm:48s} b {rowp['b']:5.1f}  c {rowp['c']:5.1f}  a {rowp['a']:5.1f}  d {rowp['d']:5.1f}  "
          f"n={m.sum():4d} ep~{episodes(m):2d}  F3 median {np.median(hb['b'][m])*100:6.2f}%  years {yrs}")

# lag-2 consistent percentile (live user runs the scanner on the weekend -> trades Monday)
print("  lag-2 consistent (OOS lag2 vs lag2 history):",
      {k: round(pct(wins(R2[k]), oos2[k]), 2) for k in "abcd"},
      " OOS lag2:", {k: round(oos2[k] * 100, 2) for k in "abcd"})

print("\n================ 2. POWER / LIKELIHOOD RATIO of the OOS draw (C2: 'not evidence the strategy has broken')")


def kde_pdf(h, x, n_eff):
    sd = h.std()
    iqr = np.subtract(*np.percentile(h, [75, 25]))
    bw = 0.9 * min(sd, iqr / 1.34) * n_eff ** (-0.2)
    return float(np.mean(np.exp(-0.5 * ((x - h) / bw) ** 2)) / (bw * np.sqrt(2 * np.pi)))


n_eff = len(hb["b"]) / N
for k in ["b", "c"]:
    h = hb[k]
    x = oos[k]
    f1 = kde_pdf(h, x, n_eff)
    out = []
    for nm, target in [("median->0 (no excess over cash-ish 0)", 0.0),
                       ("median->static 85IEF/15GLD median", float(np.median(hb["static8515"]))),
                       ("median->half the backtest edge", float(np.median(h)) / 2)]:
        h0 = h - np.median(h) + target
        f0 = kde_pdf(h0, x, n_eff)
        out.append(f"{nm}: pct {pct(h0, x):5.1f}, density ratio f0/f1 = {f0 / f1:4.2f}, "
                   f"tail ratio P0(<=x)/P1(<=x) = {(h0 <= x).mean() / (h <= x).mean():4.2f}")
    print(f"  [{k}] OOS {x*100:.2f}%  backtest pct {pct(h, x):.2f}")
    for o in out:
        print("      ", o)

print("\n================ 3. TAIL WEEK (C3)")
marks = IDX[P["mark"].values]
wk_rows = []
for k in ["b", "c", "d"]:
    eq = (1 + R1[k].fillna(0)).cumprod()
    lev = SPEC[k]["lev"]
    em = eq.loc[marks]
    wr = (em / em.shift(1) - 1)
    df = pd.DataFrame({"ret": wr.values, "dec": list(marks[:-1].insert(0, pd.NaT))[:len(marks)]}, index=marks)
    df["dec"] = pd.Series(marks, index=marks).shift(1)
    df = df.dropna()
    df["w_lev"] = W[k][lev].reindex(df["dec"]).values
    df["vol"] = P["vol20"].reindex(df["dec"]).values
    df["er"] = P["er20"].reindex(df["dec"]).values
    df["ndx_wk"] = (P["ndx"].reindex(df.index).values / P["ndx"].reindex(df["dec"]).values - 1)
    hist = df[(df.index >= HIST_A) & (df.index <= FREEZE)]
    o = df[df.index > FREEZE]
    worst_o = o["ret"].min()
    worse = hist[hist["ret"] < worst_o]
    print(f"  [{k}] worst OOS week {worst_o*100:.2f}% ({o['ret'].idxmin().date()}); hist weeks {len(hist)}; worse {len(worse)}")
    for d_, rw in worse.iterrows():
        print(f"      {d_.date()} ret {rw.ret*100:6.2f}%  w_{lev} {rw.w_lev:.3f} vol {rw.vol:.3f} er {rw.er:.3f}  NDX wk {rw.ndx_wk*100:6.2f}%")
    re_ = hist[hist.index >= pd.Timestamp("2010-02-12")]
    print(f"      real-ETF era (2010-02..freeze): {len(re_)} weeks; worse than OOS week: {(re_['ret'] < worst_o).sum()}; "
          f"5 worst: " + ", ".join(f"{d_.date()} {v*100:.2f}%" for d_, v in re_['ret'].nsmallest(5).items()))
    pre = hist[hist.index < pd.Timestamp("2008-01-01")]; post = hist[hist.index >= pd.Timestamp("2008-01-01")]
    print(f"      worse weeks 1988-2007: {(pre['ret'] < worst_o).sum()} of {len(pre)};  2008-freeze: {(post['ret'] < worst_o).sum()} of {len(post)}")
    if k == "b":
        full = hist["w_lev"] >= 0.76
        sig = full & (hist["vol"] >= 0.10) & (hist["vol"] <= 0.22) & (hist["er"] >= 0.33)
        sig2 = full & (hist["vol"] <= 0.22)
        print(f"      BASE RATE: weeks at >=76% TQQQ: {full.mean()*100:.1f}%;  'signature' (>=76%, vol 10-22%, ER>=0.33): "
              f"{sig.mean()*100:.1f}% of weeks ({sig.sum()});  >=76% & vol<=22%: {sig2.mean()*100:.1f}%")
        for nm, m in [("signature", sig), ("not signature", ~sig), (">=76% TQQQ", full), ("<76% TQQQ", ~full)]:
            sub = hist[m]
            print(f"        P(week <= {worst_o*100:.2f}% | {nm:14s}) = {(sub['ret'] <= worst_o).mean()*100:.3f}%  "
                  f"(n={len(sub)});  P(week <= -10%) = {(sub['ret'] <= -0.10).mean()*100:.2f}%;  "
                  f"P(NDX wk <= -5.49%) = {(sub['ndx_wk'] <= -0.0549).mean()*100:.2f}%")
        # NDX -5.49% weeks while at full size: what did F3 do
        bad = hist[(hist["ndx_wk"] <= -0.0549)]
        print(f"        NDX weeks <= -5.49% in history: {len(bad)}; of which F3 at >=76% TQQQ: {(bad['w_lev']>=0.76).sum()}; "
              f"F3 mean weight in those weeks {bad['w_lev'].mean():.3f}")
        yrs = (FREEZE - HIST_A).days / 365.25
        print(f"        rate of weeks <= OOS week: hist {len(worse)}/{yrs:.1f}y; incl. OOS {len(worse)+1}/{yrs+0.32:.1f}y "
              f"= 1 per {(yrs+0.32)/(len(worse)+1):.1f}y;  but 1988-2007 {(pre['ret'] < worst_o).sum()}/20y and 2008-2026 "
              f"{(post['ret'] < worst_o).sum()+1}/18.9y (incl. OOS)")

print("\n================ 4. OVERLAY COST vs ITS OWN DISTRIBUTION (C4)")
for lab, a_, b_ in [("spec - neither", "b", "b_neither"), ("spec - noChop", "b", "b_nochop"),
                    ("spec - noSeason", "b", "b_noseas"), ("F2 spec - neither", "c", "c_neither")]:
    dh = wins(R1[a_]) - wins(R1[b_])
    do = oos[a_] - oos[b_]
    print(f"  {lab:18s} OOS diff {do*100:6.2f}pp | hist pct {pct(dh, do):5.2f} | hist median {np.median(dh)*100:5.2f}pp "
          f"| share |diff|>=|OOS| {(np.abs(dh) >= abs(do)).mean()*100:5.1f}% | p5 {np.percentile(dh,5)*100:6.2f} p95 {np.percentile(dh,95)*100:6.2f}")
print("  halves (IS, lag 1): CAGR / maxDD (date) / Calmar")
for half, (a_, b_) in {"1988-2006": ("1988-01-01", "2006-12-31"), "2007-freeze": ("2007-01-01", str(FREEZE.date()))}.items():
    for k in ["b", "b_noseas", "b_nochop", "b_neither", "c", "c_neither"]:
        s = R1[k].loc[a_:b_]
        e = (1 + s).cumprod(); dd = e / e.cummax() - 1
        st = C.stats(R1[k], a_, b_, start=None)
        print(f"    {half:11s} {k:10s} {st['cagr']*100:5.1f}%  {st['maxdd']*100:6.1f}% ({dd.idxmin().date()})  {st['calmar']:.2f}")
for k in ["b", "b_neither", "b_nochop"]:
    s = R1[k].loc[HIST_A:FREEZE]; e = (1 + s).cumprod(); dd = e / e.cummax() - 1
    print(f"  full-IS maxDD {k:10s} {dd.min()*100:6.1f}% trough {dd.idxmin().date()} peak {e.loc[:dd.idxmin()].idxmax().date()}")
# maxDD excluding the single worst episode: drawdown stats in 2000-2002 vs elsewhere
for k in ["b", "b_neither"]:
    for nm, (a_, b_) in [("ex dot-com: 2003-freeze", ("2003-01-01", str(FREEZE.date()))),
                         ("1988-1999", ("1988-01-01", "1999-12-31")), ("2000-2002", ("2000-01-01", "2002-12-31"))]:
        st = C.stats(R1[k], a_, b_, start=None)
        print(f"    {k:10s} {nm:24s} CAGR {st['cagr']*100:5.1f}%  maxDD {st['maxdd']*100:6.1f}%  Calmar {st['calmar']:.2f}")

print("\n================ 5. TLT vs IEF (C6) over 81-day windows")
dh = wins(R1["a"]) - wins(R1["b"])
do = oos["a"] - oos["b"]
print(f"  OOS (a-b) {do*100:.2f}pp; hist pct {pct(dh, do):.1f}; share of windows with a-b <= OOS {(dh <= do).mean()*100:.1f}%; "
      f"median {np.median(dh)*100:.2f}pp; p5/p95 {np.percentile(dh,5)*100:.2f}/{np.percentile(dh,95)*100:.2f}pp")
dh2 = wins(R1["a"], a=pd.Timestamp("2002-07-30")) - wins(R1["b"], a=pd.Timestamp("2002-07-30"))
print(f"  real TLT/IEF era only (2002-07+): pct {pct(dh2, do):.1f}; share <= OOS {(dh2 <= do).mean()*100:.1f}%")

print("\n================ 6. LAG (C8)")
for k in ["b", "c"]:
    for lag, R in [(1, R1), (2, R2)]:
        s = R[k].loc[HIST_A:FREEZE]; e = (1 + s).cumprod(); dd = e / e.cummax() - 1
        st = C.stats(R[k], b=FREEZE)
        print(f"  [{k}] lag{lag} full IS CAGR {st['cagr']*100:5.2f}% maxDD {st['maxdd']*100:6.2f}% (trough {dd.idxmin().date()}, "
              f"peak {e.loc[:dd.idxmin()].idxmax().date()}) Calmar {st['calmar']:.3f}")
    for nm, (a_, b_) in [("1988-1993", ("1988-01-01", "1993-12-31")), ("1994-1999", ("1994-01-01", "1999-12-31")),
                         ("2000-2009", ("2000-01-01", "2009-12-31")), ("2010-freeze", ("2010-01-01", str(FREEZE.date()))),
                         ("2000-freeze", ("2000-01-01", str(FREEZE.date())))]:
        s1, s2 = C.stats(R1[k], a_, b_, start=None), C.stats(R2[k], a_, b_, start=None)
        print(f"     {nm:12s} lag1 {s1['cagr']*100:5.1f}% / {s1['maxdd']*100:6.1f}% / {s1['calmar']:.2f}   "
              f"lag2 {s2['cagr']*100:5.1f}% / {s2['maxdd']*100:6.1f}% / {s2['calmar']:.2f}")
# Friday->Monday autocorrelation (the return that lag1 captures and lag2 misses follows the Friday decision)
nr = P["ndx_ret"]
fri = IDX[P["mark"].values]
nxt_day = pd.Series(IDX[1:], index=IDX[:-1])
for nm, (a_, b_) in [("1988-1993", ("1988-01-01", "1993-12-31")), ("1994-1999", ("1994-01-01", "1999-12-31")),
                     ("2000-2009", ("2000-01-01", "2009-12-31")), ("2010-freeze", ("2010-01-01", str(FREEZE.date())))]:
    f_ = fri[(fri >= a_) & (fri <= b_)]
    f_ = f_[f_.isin(nxt_day.index)]
    x = nr.loc[f_].values; y = nr.loc[nxt_day.loc[f_].values].values
    print(f"  NDX corr(Friday ret, next-session ret) {nm}: {np.corrcoef(x, y)[0,1]:+.3f}   all-days lag-1 autocorr "
          f"{nr.loc[a_:b_].autocorr(1):+.3f}")
# OOS day-level decomposition of lag1 vs lag2 (F3)
d12 = (R2["b"] - R1["b"]).loc[OOS]
print("  OOS F3 lag2-lag1 daily return differences, largest |diff| days:",
      ", ".join(f"{d_.date()} {v*100:+.2f}pp" for d_, v in d12.reindex(d12.abs().sort_values(ascending=False).index)[:5].items()),
      f"| sum of all {d12.sum()*100:+.2f}pp; sum excl. top-2 {d12.drop(d12.abs().nlargest(2).index).sum()*100:+.2f}pp")
# Monday-open gap on 10-19
yfp = os.path.join(HERE, "oos_live_yf.parquet")
if os.path.exists(yfp):
    y = pd.read_parquet(yfp)
    YC = y.pivot(index="date", columns="ticker", values="close")
    YO = y.pivot(index="date", columns="ticker", values="open")
    for t in ["TQQQ", "QLD", "IEF", "GLD"]:
        on = YO.loc["2026-10-19", t] / YC.loc["2026-10-16", t] - 1
        cc = YC.loc["2026-10-19", t] / YC.loc["2026-10-16", t] - 1
        on5 = YO.loc["2026-10-05", t] / YC.loc["2026-10-02", t] - 1
        print(f"    {t}: overnight 10-16->10-19 open {on*100:+.2f}%, close-to-close 10-19 {cc*100:+.2f}%  | overnight 10-02->10-05 open {on5*100:+.2f}%")
    # un-rounded independent check
    MAP = {"QQQ": "qqq_real", "QLD": "qld_real", "TQQQ": "tqqq_real", "TLT": "tlt_real", "IEF": "ief_real",
           "SHY": "shy_real", "BIL": "bil_real", "GLD": "gld_real"}
    mx = {}
    for t, c in MAP.items():
        r = YC[t].dropna().reindex(IDX).ffill().pct_change(fill_method=None)
        mx[t] = float((r.loc[OOS] - P.loc[OOS, c]).abs().max())
    print("  un-rounded max |yf - panel| daily return diff in OOS:", {k: f"{v:.1e}" for k, v in mx.items()})

print("\n================ 7. 10y yield path (C6)")
for d_ in [FREEZE, pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-16"), END]:
    print(f"  {d_.date()} tnx {P.at[d_, 'tnx']:.3f}")
t = P.loc[A0:END, "tnx"]
print(f"  OOS tnx max {t.max():.3f} on {t.idxmax().date()}, min {t.min():.3f} on {t.idxmin().date()}")
print(f"  IEF return risk-on phase 07-27..10-16 {tot(P['ief_real'], A0, '2026-10-16')*100:.2f}%, risk-off 10-19..end "
      f"{tot(P['ief_real'], '2026-10-19', END)*100:.2f}%;  TLT {tot(P['tlt_real'], A0, '2026-10-16')*100:.2f}% / "
      f"{tot(P['tlt_real'], '2026-10-19', END)*100:.2f}%;  GLD {tot(P['gld_real'], A0, '2026-10-16')*100:.2f}% / "
      f"{tot(P['gld_real'], '2026-10-19', END)*100:.2f}%")
print(f"  TQQQ freeze->record {tot(P['tqqq_real'], A0, '2026-10-06')*100:.2f}%; F3 avg held TQQQ freeze->record "
      f"{C.held_weights(P, W['b']).loc[A0:'2026-10-06','tqqq'].mean():.3f}")
print(f"\nreviewer variants evaluated in this script: {len(COUNT)}")
