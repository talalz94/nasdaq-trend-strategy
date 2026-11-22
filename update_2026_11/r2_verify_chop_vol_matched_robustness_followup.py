"""
r2_verify_chop_vol_matched_robustness_followup.py -- follow-up numeric checks for the ROBUSTNESS review of
r2_chop_vol_matched (reads the specialist's JSON; recomputes drawdown-shape metrics with an episode removed).
  a. C2: NDX T1/T2/T3 chop and vol differences as saved
  b. C5 / recommendation: constant-multiplier Ac vs B at 3/10/15 bps, IS and per half
  c. C6: average leveraged-sleeve trade size (B vs A), per book
  d. C7: Ulcer / CDaR95 / maxDD of B vs V with the 2000-02 bear removed (both legs zero inside the window)
  e. C10: number of OOS trading days
  f. C2/C8 sign check: cross-market chop by half, NDX-sim vs IXIC
usage: python update_2026_11/r2_verify_chop_vol_matched_robustness_followup.py <scratch_out.json>
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OUTP = os.path.abspath(sys.argv[1])
S = json.load(open(os.path.join(HERE, "r2_chop_vol_matched.json")))
P = C.panel()
IDX = P.index
N = len(IDX)
OUT = {}

# a. saved T1/T2/T3 differences
a = {}
for t in ["F2", "F3"]:
    d = S["ndx"][t]["tranche"]["dcalmar_B_minus_alt"]
    a[t] = {comp: {w: d[comp]["3bps"][w] for w in ["T1_1988-99", "T2_2000-12", "T3_2013-frz"]} for comp in ["chop", "vol"]}
    print(t, "T-windows", {c: {w: round(v, 3) for w, v in x.items()} for c, x in a[t].items()})
OUT["a_Twindows"] = a

# b. Ac vs B by cost and half
b = {}
for t in ["F2", "F3"]:
    for basis in ["tranche", "Fri"]:
        v = S["ndx"][t][basis]["variants"]
        row = {}
        for c in ["3bps", "10bps", "15bps"]:
            row[c] = {k: {w: v[k][c][w]["calmar"] for w in ["IS", "H1", "H2"]} for k in ["B", "Ac", "A"]}
            row[c]["B_minus_Ac"] = {w: row[c]["B"][w] - row[c]["Ac"][w] for w in ["IS", "H1", "H2"]}
        b[f"{t}|{basis}"] = row
        print(f"{t} {basis:7s} " + " | ".join(
            f"{c}: B {row[c]['B']['IS']:.3f} Ac {row[c]['Ac']['IS']:.3f} (B-Ac IS {row[c]['B_minus_Ac']['IS']:+.3f} H1 {row[c]['B_minus_Ac']['H1']:+.3f} H2 {row[c]['B_minus_Ac']['H2']:+.3f})"
            for c in ["3bps", "15bps"]))
OUT["b_Ac"] = b

# c. average leveraged-sleeve trade size per book
c_ = {}
for t in ["F2", "F3"]:
    for k in ["B", "A", "Ac", "V"]:
        ac = S["ndx"][t]["tranche"]["variants"][k]["activity"]
        per_book_trades = ac["lev_sleeve_trades_per_yr_all_books"] / ac["books"]
        c_[f"{t}|{k}"] = dict(turn_lev=ac["turnover_lev_sleeve_per_yr"], lev_trades_per_book=per_book_trades,
                             avg_trade=ac["turnover_lev_sleeve_per_yr"] / per_book_trades,
                             reb_per_book=ac["rebalances_per_yr_per_book"], turnover=ac["turnover_per_yr"])
        print(f"{t} {k:2s}: lev-sleeve turnover {c_[f'{t}|{k}']['turn_lev']:.2f}/yr, lev trades/book {per_book_trades:.1f}/yr, "
              f"avg lev trade {c_[f'{t}|{k}']['avg_trade']*100:.1f}% NAV | total turnover {ac['turnover_per_yr']:.2f} reb/book {ac['rebalances_per_yr_per_book']:.1f}")
OUT["c_tradesize"] = c_

# d. drawdown shape of B vs V, with the 2000-02 bear removed
DOW = IDX.dayofweek.values
LASTS = [np.maximum.accumulate(np.where(DOW == i, np.arange(N), -1)) for i in range(5)]
SEASON = np.where(np.isin(P["month"].values, (8, 9)), 0.5, 1.0)
R_IEF, R_GOLD = P["ief"].fillna(0).values, P["gold_phys"].fillna(0).values


def held(w, li):
    h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
    o = np.zeros(N); o[1:] = h[:-1]
    return o


def tranche(r_lev, wl):
    r = np.zeros(N)
    wo, wg = np.clip(0.85 - wl, 0, None), np.full(N, 0.15)
    for li in LASTS:
        hl, ho, hg = held(wl, li), held(wo, li), held(wg, li)
        tn = np.abs(np.diff(hl, prepend=0.0)) + np.abs(np.diff(ho, prepend=0.0)) + np.abs(np.diff(hg, prepend=0.0))
        r += (hl * r_lev + ho * R_IEF + hg * R_GOLD - tn * 3e-4) / 5
    return r


def shape(r, m):
    e = np.cumprod(1 + r[m]); dd = e / np.maximum.accumulate(e) - 1
    q = np.quantile(dd, 0.05)
    return dict(ulcer=float(np.sqrt(np.mean((100 * dd) ** 2))), cdar95=float(dd[dd <= q].mean()), maxdd=float(dd.min()))


d_ = {}
above, vol20, er20 = P["above"].values, P["vol20"].values, P["er20"].values
clean = (~np.isnan(er20)) & (er20 >= 0.25)
for t, col, vt in [("F2", "qld", 0.30), ("F3", "tqqq", 0.20)]:
    rl = np.nan_to_num(P[col].values)
    sc = np.clip(np.where(np.isnan(np.clip(vt / vol20, 0, 1)), 1.0, np.clip(vt / vol20, 0, 1)) * SEASON, 0, 1)
    wB = 0.85 * np.where(above, np.where(clean, sc, 0.5 * sc), 0.0)
    cV = S["ndx"][t]["tranche"]["solve_V"]["c"]
    scV = np.clip(SEASON * cV, 0, 1)
    wV = 0.85 * np.where(above, np.where(clean, scV, 0.5 * scV), 0.0)
    rB, rV = tranche(rl, wB), tranche(rl, wV)
    m = (IDX >= C.START) & (IDX <= C.FREEZE)
    z = (IDX >= "2000-01-01") & (IDX <= "2002-12-31")
    full = dict(B=shape(rB, m), V=shape(rV, m))
    ex = dict(B=shape(np.where(z, 0, rB), m), V=shape(np.where(z, 0, rV), m))
    d_[t] = dict(full=full, excl_2000_02=ex)
    print(f"{t} B vs V  full: Ulcer {full['B']['ulcer']:.2f}/{full['V']['ulcer']:.2f} CDaR {full['B']['cdar95']*100:.1f}/{full['V']['cdar95']*100:.1f} "
          f"maxDD {full['B']['maxdd']*100:.1f}/{full['V']['maxdd']*100:.1f} | excl 2000-02: Ulcer {ex['B']['ulcer']:.2f}/{ex['V']['ulcer']:.2f} "
          f"CDaR {ex['B']['cdar95']*100:.1f}/{ex['V']['cdar95']*100:.1f} maxDD {ex['B']['maxdd']*100:.1f}/{ex['V']['maxdd']*100:.1f}")
    assert abs(full["B"]["ulcer"] - S["ndx"][t]["tranche"]["variants"]["B"]["3bps"]["IS"]["ulcer"]) < 1e-9
    assert abs(full["V"]["cdar95"] - S["ndx"][t]["tranche"]["variants"]["V"]["3bps"]["IS"]["cdar95"]) < 1e-9
OUT["d_shape_excl2000"] = d_

# e. OOS day count
oos = (IDX >= "2026-07-25")
OUT["e_oos_days"] = dict(n=int(oos.sum()), first=str(IDX[oos][0].date()), last=str(IDX[oos][-1].date()))
print("OOS days", OUT["e_oos_days"])

json.dump(OUT, open(OUTP, "w"), indent=1, default=float)
print("saved", OUTP)
