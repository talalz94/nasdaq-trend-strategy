"""
Integrity reviewer checks for r2_friday_effect (independent code paths; writes only to a scratch dir).
  K1  F2/F1/F3 weekday books + tranche via common.backtest/common.stats (pandas path)
  K2  tranche convention: mean of daily returns (used) vs mean of 5 sub-account equity curves
  K3  placebo engine: same seed, masks through common.backtest -> distribution summary vs JSON
  K4  placebo percentile stability under 3 other seeds (H1/H2 Calmar, rule-A inputs)
  K5  switch counts: Friday 189 vs others - where does it sit among 1,000 random schedules / offset cycles
  K6  lag-2 split of the H1 gap: 1988-1999 vs 2000-2006 (C6/C7 mechanism attribution)
  K7  panel NDX close vs fresh Yahoo ^NDX close by year (E4 1988 anomaly) + IXIC/GSPC Wednesday 1985-99
  K8  F1/F3 decade gaps (C2 'every tier and every decade')
  K9  fixed-day books' own maxDD/CAGR/Calmar medians vs the block-null planning numbers (C12)
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

SCR = os.environ.get("VERIFY_SCRATCH", os.path.join(os.path.expanduser("~"), "r2_verify_scratch"))
os.makedirs(SCR, exist_ok=True)
J = json.load(open(os.path.join(HERE, "r2_friday_effect.json")))
P = C.panel(); IDX = P.index
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
WIN = {"IS": ("1988-01-01", "2026-07-24"), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", "2026-07-24")}
TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
OUT = {}


def st(r, a, b):
    return C.stats(r, a=a, b=b)


# ---------------- K1 + K2
books = {}
for t, (lev, vt) in TIERS.items():
    w = C.strategy_weights(P, lev=lev, voltgt=vt)
    rs = {d: C.backtest(P, w, mark=pd.Series(IDX.dayofweek == i, index=IDX)) for i, d in enumerate(DAYS)}
    rs["tranche"] = sum(rs[d] for d in DAYS) / 5
    books[t] = (w, rs)
    row = {}
    for wn, (a, b) in WIN.items():
        sf, stt = st(rs["Fri"], a, b), st(rs["tranche"], a, b)
        row[wn] = dict(fri=[round(sf["cagr"] * 100, 2), round(sf["maxdd"] * 100, 1), round(sf["calmar"], 3)],
                       tranche=[round(stt["cagr"] * 100, 2), round(stt["maxdd"] * 100, 1), round(stt["calmar"], 3)],
                       d_cagr_pp=round((sf["cagr"] - stt["cagr"]) * 100, 3), d_calmar=round(sf["calmar"] - stt["calmar"], 4))
    # K2 sub-account tranche: each book starts with 1/5 at 1988-01-01 and is never rebalanced across books
    eqs = []
    for wn, (a, b) in WIN.items():
        m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
        E = sum((1 + rs[d][m]).cumprod() for d in DAYS) / 5
        rr = E.pct_change(); rr.iloc[0] = E.iloc[0] - 1
        s = C.stats(rr, start=None)
        row[wn]["tranche_subaccount"] = [round(s["cagr"] * 100, 2), round(s["maxdd"] * 100, 1), round(s["calmar"], 3)]
        fixed = [st(rs[d], a, b) for d in DAYS]
        row[wn]["fixed_day_median"] = [round(np.median([x["cagr"] for x in fixed]) * 100, 2),
                                       round(np.median([x["maxdd"] for x in fixed]) * 100, 1),
                                       round(np.median([x["calmar"] for x in fixed]), 3)]
        row[wn]["fixed_day_mean_cagr"] = round(np.mean([x["cagr"] for x in fixed]) * 100, 2)
    OUT.setdefault("K1_K2_K9", {})[t] = row
    print("K1/K2/K9", t, json.dumps(row), flush=True)

# ---------------- K8 decade gaps all tiers
dec = {t: {k: round(v["d_cagr_pp"], 2) for k, v in J["a_decomposition"][t]["by_decade"].items()} for t in TIERS}
dec_cal = {t: {k: round(v["d_calmar"], 3) for k, v in J["a_decomposition"][t]["by_decade"].items()} for t in TIERS}
OUT["K8_decade_gap_cagr_pp"] = dec; OUT["K8_decade_gap_calmar"] = dec_cal
print("K8 decade CAGR gaps", json.dumps(dec), "\n   calmar gaps", json.dumps(dec_cal), flush=True)

# ---------------- K3 placebo engine check (F2, same seed, pandas path)
w2, rs2 = books["F2"]
wk = pd.factorize(IDX.to_period("W-FRI"))[0]
dow = IDX.dayofweek.values


def masks_for(seed, S, block=1):
    rng = np.random.default_rng(seed)
    g = wk // block
    draws = rng.integers(0, 5, size=(S, g.max() + 1))
    return dow[None, :] == draws[:, g]


def pl_stats(masks, wobj):
    res = {wn: {"cagr": [], "calmar": [], "maxdd": []} for wn in WIN}
    for s in range(masks.shape[0]):
        r = C.backtest(P, wobj, mark=pd.Series(masks[s], index=IDX))
        for wn, (a, b) in WIN.items():
            x = st(r, a, b)
            for k in ("cagr", "calmar", "maxdd"):
                res[wn][k].append(x[k])
    return {wn: {k: np.array(v) for k, v in d.items()} for wn, d in res.items()}


M0 = masks_for(20261118, 1000)
ps = pl_stats(M0, w2)
k3 = {}
for wn in WIN:
    ref = J["d_placebo_ndx"]["F2"][wn]
    fri = J["d_placebo_ndx"]["F2"][wn]["Fri"]
    k3[wn] = dict(mine_cagr_mean=float(ps[wn]["cagr"].mean()), json_cagr_mean=ref["placebo_cagr"]["mean"],
                  mine_calmar_p50=float(np.median(ps[wn]["calmar"])), json_calmar_p50=ref["placebo_calmar"]["p50"],
                  mine_maxdd_p50=float(np.median(ps[wn]["maxdd"])), json_maxdd_p50=ref["placebo_maxdd"]["p50"],
                  mine_pct_fri_calmar=float(100 * (ps[wn]["calmar"] < fri["calmar"]).mean()),
                  json_pct_fri_calmar=fri["pct_calmar"],
                  mine_pct_fri_cagr=float(100 * (ps[wn]["cagr"] < fri["cagr"]).mean()), json_pct_fri_cagr=fri["pct_cagr"])
OUT["K3_placebo_engine_same_seed"] = k3
print("K3", json.dumps(k3), flush=True)

# ---------------- K4 percentile stability (other seeds)
k4 = {}
for seed in (1, 2, 3):
    p_ = pl_stats(masks_for(seed, 1000), w2)
    k4[seed] = {wn: dict(pct_fri_calmar=float(100 * (p_[wn]["calmar"] < J["d_placebo_ndx"]["F2"][wn]["Fri"]["calmar"]).mean()),
                         pct_fri_cagr=float(100 * (p_[wn]["cagr"] < J["d_placebo_ndx"]["F2"][wn]["Fri"]["cagr"]).mean()))
                for wn in WIN}
    print("K4 seed", seed, json.dumps(k4[seed]), flush=True)
OUT["K4_placebo_other_seeds"] = k4

# ---------------- K5 switch counts
ABOVE = P["above"].values.astype(bool)


def nsw(mask, a, b):
    m = mask & (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    s = ABOVE[np.where(m)[0]]
    return int((s[1:] != s[:-1]).sum())


k5 = {}
for wn, (a, b) in WIN.items():
    fixed = {d: nsw(dow == i, a, b) for i, d in enumerate(DAYS)}
    pl = np.array([nsw(M0[s], a, b) for s in range(M0.shape[0])])
    off = {o: nsw((np.arange(len(IDX)) % 5) == o, a, b) for o in range(5)}
    k5[wn] = dict(fixed=fixed, placebo_mean=float(pl.mean()), placebo_sd=float(pl.std()), placebo_min=int(pl.min()),
                  placebo_p1=float(np.percentile(pl, 1)), share_placebo_le_fri=float((pl <= fixed["Fri"]).mean()),
                  offset_cycles=off)
OUT["K5_switch_counts"] = k5
print("K5", json.dumps(k5), flush=True)

# ---------------- K6 lag-2 split inside H1 (F2)
rs2_l2 = {d: C.backtest(P, w2, mark=pd.Series(IDX.dayofweek == i, index=IDX), lag=2) for i, d in enumerate(DAYS)}
rs2_l2["tranche"] = sum(rs2_l2[d] for d in DAYS) / 5
k6 = {}
for wn, (a, b) in {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-2006": ("2000-01-01", "2006-12-31"),
                   "H1": WIN["H1"], "H2": WIN["H2"], "IS": WIN["IS"]}.items():
    g1 = (st(rs2["Fri"], a, b)["cagr"] - st(rs2["tranche"], a, b)["cagr"]) * 100
    g2 = (st(rs2_l2["Fri"], a, b)["cagr"] - st(rs2_l2["tranche"], a, b)["cagr"]) * 100
    gr = (rs2["Fri"] - rs2["tranche"])
    m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    yrs = (IDX[m][-1] - IDX[m][0]).days / 365.25
    # first holding day of the Friday book (session after a Friday mark)
    fm = (IDX.dayofweek == 4)
    k1 = np.r_[False, fm[:-1]]
    k6[wn] = dict(gap_lag1_pp=round(g1, 3), gap_lag2_pp=round(g2, 3),
                  share_removed=round(1 - g2 / g1, 3) if abs(g1) > 1e-9 else None,
                  first_day_share=round(float(gr[m & k1].sum() / gr[m].sum()), 3),
                  monday_share=round(float(gr[m & (IDX.dayofweek == 0)].sum() / gr[m].sum()), 3),
                  arith_pp_yr=round(float(gr[m].sum() / yrs * 100), 3))
OUT["K6_lag2_split"] = k6
print("K6", json.dumps(k6), flush=True)

# ---------------- K7 data: panel NDX vs fresh Yahoo ^NDX; Wednesday effect in IXIC/GSPC
Y = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))


def ys(t, col="close"):
    s = Y[Y.ticker == t].set_index("date")[col].sort_index()
    return s[~s.index.duplicated(keep="last")].astype(float)


ndx_y = ys("^NDX").reindex(IDX)
rel = (P["ndx"] / ndx_y - 1).abs()
k7 = {"panel_vs_yahoo_ndx_close": {str(y): dict(max_abs_rel_diff=float(rel[IDX.year == y].max()),
                                                share_equal_1e6=float((rel[IDX.year == y] < 1e-6).mean()))
                                   for y in range(1985, 2000)}}
o_ = ys("^NDX", "open").reindex(IDX)
pc_y = ndx_y.shift(1)
k7["yahoo_open_eq_yahoo_prev_close_share"] = {str(y): float((np.abs(o_[IDX.year == y] / pc_y[IDX.year == y] - 1) < 1e-9).mean())
                                              for y in range(1985, 1999)}
k7["yahoo_open_eq_prev_close_within_0.1pct_share"] = {str(y): float((np.abs(o_[IDX.year == y] / pc_y[IDX.year == y] - 1) < 1e-3).mean())
                                                      for y in range(1985, 1999)}
wed = {}
for t in ("^IXIC", "^GSPC", "^NDX"):
    s = ys(t)
    r = s.pct_change()
    r = r[(r.index >= "1985-10-03") & (r.index <= "1999-12-31")]
    for i, d in enumerate(DAYS):
        x = r[r.index.dayofweek == i].dropna()
        wed.setdefault(t, {})[d] = [round(float(x.mean() * 1e4), 1), round(float(x.mean() / (x.std() / np.sqrt(len(x)))), 2)]
k7["dow_mean_bp_t_1985_1999_yahoo"] = wed
OUT["K7_data"] = k7
print("K7", json.dumps(k7), flush=True)

json.dump(OUT, open(os.path.join(SCR, "r2_verify_friday_effect_integrity_checks.json"), "w"), indent=1, default=float)
print("saved")
