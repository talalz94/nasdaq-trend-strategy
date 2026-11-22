"""
INTEGRITY REVIEW of leverage_execution  --  step 2: data alignment / splicing / weekday artefacts.
Read-only: prints only.  usage: python update_2026_11/verify_leverage_execution_integrity_data.py
"""
import sys, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

P = C.panel(); IDX = P.index
print("rows", len(P), IDX[0].date(), IDX[-1].date(), "dup", IDX.duplicated().sum(),
      "weekend rows", int((IDX.dayofweek >= 5).sum()), "monotonic", IDX.is_monotonic_increasing)
print("dtypes mark/mark_wk/above:", P["mark"].dtype, P["mark_wk"].dtype, P["above"].dtype)

# ---- A1: weekday x era statistics of ndx_ret (interpolation / stale-price artefacts would show here)
r = P["ndx_ret"]
eras = [("1985-1989", "1985-10-01", "1989-12-31"), ("1990-1999", "1990-01-01", "1999-12-31"),
        ("2000-2006", "2000-01-01", "2006-12-31"), ("2007-2016", "2007-01-01", "2016-12-31"),
        ("2017-2026", "2017-01-01", "2026-11-17")]
print("\nA1 ndx_ret by weekday x era: mean bp | std % | zero-share | n")
for nm, a, b in eras:
    s = r[(r.index >= a) & (r.index <= b)]
    row = []
    for d in range(5):
        x = s[s.index.dayofweek == d]
        row.append(f"{'MTWRF'[d]} {x.mean()*1e4:+6.1f} {x.std()*100:4.2f} {(x == 0).mean()*100:3.1f}% {len(x)}")
    print(f"  {nm}: " + " | ".join(row))

# ---- A2: known event dates (alignment of the NDX calendar)
print("\nA2 known dates (ndx_ret %):")
for d in ["1987-10-16", "1987-10-19", "1987-10-20", "2000-04-14", "2001-01-03", "2008-09-29", "2008-10-13",
          "2008-10-28", "2020-03-12", "2020-03-16", "2020-03-24", "2025-04-03", "2025-04-04", "2025-04-09"]:
    t = pd.Timestamp(d)
    print(f"   {d} {t.day_name()[:3]}  ndx_ret {P['ndx_ret'].get(t, np.nan)*100:+6.2f}%  "
          f"qld {P['qld'].get(t, np.nan)*100:+6.2f}%  tqqq {P['tqqq'].get(t, np.nan)*100:+6.2f}%")

# ---- A3: lead/lag alignment of REAL sleeves vs ndx_ret and of real bonds vs yield changes
print("\nA3 cross-correlation (real sleeve vs signal series) at lags -1/0/+1  (only lag 0 should be high)")
for c, m in [("qqq_real", 1), ("qld_real", 2), ("tqqq_real", 3)]:
    s = pd.concat([P[c], P["ndx_ret"]], axis=1).dropna()
    print(f"   {c:10s} " + "  ".join(f"lag{k:+d}: {s[c].corr(s['ndx_ret'].shift(k)):+.3f}" for k in (-1, 0, 1)))
for c, y in [("ief_real", "tnx"), ("tlt_real", "tyx"), ("gld_real", None)]:
    if y is None:
        continue
    dy = P[y].diff()
    s = pd.concat([P[c], dy], axis=1).dropna()
    print(f"   {c:10s} vs d{y}: " + "  ".join(f"lag{k:+d}: {s[c].corr(s[y].shift(k)):+.3f}" for k in (-1, 0, 1)))
# the spliced columns used in the backtests equal the real ETF after inception?
for c in ["qld", "tqqq", "ief", "tlt"]:
    rr = P[f"{c}_real"]
    m = rr.notna()
    print(f"   spliced {c:5s} == {c}_real after inception: max|diff| {float((P[c][m] - rr[m]).abs().max()):.2e}; "
          f"first real day {rr.first_valid_index().date()}")

# ---- A4: zero-return runs / NaN-to-zero fills in the spliced sleeves 1988..freeze
print("\nA4 zero-return days per sleeve, 1988-01-01..freeze (by era), and longest zero run")
cols = ["qqq", "qld", "tqqq", "ief", "tlt", "bills", "gold_phys", "ndx_ret"]
S = P.loc[(IDX >= C.START) & (IDX <= C.FREEZE), cols]
for c in cols:
    z = (S[c] == 0)
    runs = (z != z.shift()).cumsum()
    longest = int(z.groupby(runs).sum().max())
    by = {nm: int(z[(z.index >= a) & (z.index <= b)].sum()) for nm, a, b in eras}
    print(f"   {c:9s} zero days {int(z.sum()):4d}  longest run {longest:2d}  by era {by}  NaN {int(S[c].isna().sum())}")

# ---- A5: gold sources and daily vol by source (interpolation smoothness)
print("\nA5 gold_phys source counts (1988..freeze) and annualised daily vol by source")
g = P.loc[(IDX >= C.START) & (IDX <= C.FREEZE)]
for src, grp in g.groupby("gold_src"):
    print(f"   {src:15s} days {len(grp):5d}  {grp.index[0].date()}..{grp.index[-1].date()}  "
          f"daily vol {grp['gold_phys'].std()*np.sqrt(252)*100:5.1f}%  corr w/ ndx {grp['gold_phys'].corr(grp['ndx_ret']):+.3f}")

# ---- A6: mark vs mark_wk
mm = P["mark"].astype(bool); mw = P["mark_wk"].astype(bool)
print("\nA6 mark vs mark_wk differing days:", int((mm != mw).sum()), " mark&~wk:", int((mm & ~mw).sum()),
      " wk&~mark:", int((mw & ~mm).sum()), " last 3 mark_wk dates:", [str(d.date()) for d in IDX[mw][-3:]],
      " last panel day weekday:", IDX[-1].day_name())
print("   Friday marks == dayofweek 4:", bool((mm == pd.Series(IDX.dayofweek == 4, index=IDX)).all()))

# ---- A7: signal construction re-derived independently
ndx = P["ndx"]
ema = ndx.ewm(span=100, adjust=False).mean()
v20 = ndx.pct_change().rolling(20).std() * np.sqrt(252)
er = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum()
print("\nA7 signals re-derived: max|ema diff| %.3e  above mismatches %d  max|vol20 diff| %.3e  max|er20 diff| %.3e" % (
    float((ema - P["ema100"]).abs().max()), int(((ndx > ema) != P["above"].astype(bool)).sum()),
    float((v20 - P["vol20"]).abs().max()), float((er - P["er20"]).abs().max())))
print("   ndx_ret == ndx.pct_change():", float((ndx.pct_change() - P["ndx_ret"]).abs().max()))
