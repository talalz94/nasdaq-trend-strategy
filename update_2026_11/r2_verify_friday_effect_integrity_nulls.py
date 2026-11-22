"""
Integrity reviewer: (N1) Monte-Carlo stability of the 13-week-block null (H2 Friday Calmar pct 95.6 in the JSON),
(N2) where the fixed Mon..Thu books sit in the per-week placebo (bias of the per-week null for regular schedules),
(N3) OOS: which weeks carry the Friday-minus-tranche gap (claim: 'rests on 2 whipsaw events'), OOS day count.
Writes only to the scratch dir.
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

SCR = os.environ.get("VERIFY_SCRATCH", os.path.join(os.path.expanduser("~"), "r2_verify_scratch"))
J = json.load(open(os.path.join(HERE, "r2_friday_effect.json")))
P = C.panel(); IDX = P.index; N = len(IDX)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
WIN = {"IS": ("1988-01-01", "2026-07-24"), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", "2026-07-24")}
w = C.strategy_weights(P, lev="qld", voltgt=0.30)
cols = list(w.columns); W = w.values; R = P[cols].fillna(0).values
wk = pd.factorize(IDX.to_period("W-FRI"))[0]
dow = IDX.dayofweek.values


def run_masks(masks, lag=1, cost=C.COST, chunk=100):
    res = {wn: {"cagr": [], "calmar": [], "maxdd": []} for wn in WIN}
    ar = np.arange(N); k = W.shape[1]
    for i in range(0, masks.shape[0], chunk):
        mk = masks[i:i + chunk]; s = mk.shape[0]
        li = np.maximum.accumulate(np.where(mk, ar[None, :], -1), axis=1)
        H = np.where((li >= 0)[..., None], W[np.maximum(li, 0)], 0.0)
        H = np.concatenate([np.zeros((s, lag, k)), H[:, :-lag]], axis=1)
        Rm = (H * R[None]).sum(2) - np.abs(np.diff(H, axis=1, prepend=np.zeros((s, 1, k)))).sum(2) * cost
        for wn, (a, b) in WIN.items():
            m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
            X = Rm[:, m]; d = IDX[m]
            E = np.cumprod(1 + X, axis=1); yrs = (d[-1] - d[0]).days / 365.25
            cg = E[:, -1] ** (1 / yrs) - 1; dd = (E / np.maximum.accumulate(E, axis=1) - 1).min(axis=1)
            res[wn]["cagr"].append(cg); res[wn]["maxdd"].append(dd); res[wn]["calmar"].append(cg / np.abs(dd))
    return {wn: {k_: np.concatenate(v) for k_, v in d_.items()} for wn, d_ in res.items()}


def masks_for(seed, S, block=1):
    rng = np.random.default_rng(seed)
    g = wk // block
    draws = rng.integers(0, 5, size=(S, g.max() + 1))
    return dow[None, :] == draws[:, g]


def pct(dist, x):
    return float(100 * ((dist < x).mean() + 0.5 * (dist == x).mean()))


fri = {wn: J["d_placebo_ndx"]["F2"][wn]["Fri"] for wn in WIN}
out = {"N1_block13_other_seeds": {}}
# sanity: the specialist's block seed reproduces 95.6
for seed in (20261118 + 100, 11, 12, 13, 14, 15):
    pb = run_masks(masks_for(seed, 1000, 13))
    out["N1_block13_other_seeds"][seed] = {wn: dict(pct_calmar=pct(pb[wn]["calmar"], fri[wn]["calmar"]),
                                                    pct_cagr=pct(pb[wn]["cagr"], fri[wn]["cagr"]),
                                                    maxdd_p50=float(np.median(pb[wn]["maxdd"])),
                                                    cagr_p50=float(np.median(pb[wn]["cagr"])))
                                           for wn in WIN}
    print("N1 seed", seed, json.dumps({wn: (round(v["pct_cagr"], 1), round(v["pct_calmar"], 1)) for wn, v in out["N1_block13_other_seeds"][seed].items()}), flush=True)

# N2 fixed books' percentiles in the per-week placebo (from the JSON)
n2 = {wn: {d: (J["d_placebo_ndx"]["F2"][wn][d]["pct_cagr"], J["d_placebo_ndx"]["F2"][wn][d]["pct_calmar"]) for d in DAYS + ["tranche"]}
      for wn in WIN}
out["N2_fixed_books_pct_in_per_week_null_(cagr,calmar)"] = n2
print("N2", json.dumps(n2), flush=True)

# N3 OOS weekly decomposition of the F2 Friday-minus-tranche gap
rs = {d: C.backtest(P, w, mark=pd.Series(dow == i, index=IDX)) for i, d in enumerate(DAYS)}
tr = sum(rs[d] for d in DAYS) / 5
m = IDX >= pd.Timestamp("2026-07-25")
g = (rs["Fri"] - tr)[m]
wkg = g.groupby(IDX[m].to_period("W-FRI")).sum() * 100
held = {d: C.held_weights(P, w, mark=pd.Series(dow == i, index=IDX))["qld"][m] for i, d in enumerate(DAYS)}
H = pd.DataFrame(held)
out["N3_oos"] = dict(oos_days=int(m.sum()), first=str(IDX[m][0].date()), last=str(IDX[m][-1].date()),
                     arith_gap_pp=float(g.sum() * 100),
                     weekly_gap_pp={str(k): round(float(v), 3) for k, v in wkg.items()},
                     top2_weeks_share=float(wkg.sort_values(ascending=False).head(2).sum() / wkg.sum()))
print("N3", json.dumps(out["N3_oos"]), flush=True)
wk_q = H.groupby(IDX[m].to_period("W-FRI")).mean().round(3)
print(wk_q.to_string(), flush=True)
json.dump(out, open(os.path.join(SCR, "r2_verify_friday_effect_integrity_nulls.json"), "w"), indent=1, default=float)
print("saved")
