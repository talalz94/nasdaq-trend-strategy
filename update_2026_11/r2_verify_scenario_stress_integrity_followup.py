"""
Integrity reviewer follow-ups (read-only; writes nothing in the project).
 (a) C9: split the 'calendar' part of today's-state-minus-own-state into weekday phase vs the Aug/Sep
     seasonal-halving months (own months + forward weekdays, and forward months + own weekdays)
 (b) design-A endpoint pin: S0 and S3 with the day-252 pin removed (free random-walk noise after the
     scenario's last real waypoint) vs the specialist's pinned paths, same seeds and same z draws
 (c) tranche rebound participation in the replays (C5 'strong trends captured, choppy not')
"""
import json, math, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_scenario_stress.py")
code = open(SRC, encoding="utf-8").read()
cut = code.index("# ------------------------------------------------------------------ VALIDATION")
G = {"__file__": SRC, "__name__": "spec_defs"}
exec(compile(code[:cut], SRC, "exec"), G)
P, C, FWD, H, L0, RFD = G["P"], G["C"], G["FWD"], G["H"], G["L0"], G["RFD"]
evaluate, synth_path, SCEN, bridge = G["evaluate"], G["synth_path"], G["SCEN"], G["bridge"]
TIERS, DAYS = G["TIERS"], G["DAYS"]
idx = P.index
STORED = json.load(open(os.path.join(HERE, "r2_scenario_stress.json")))
T0 = time.time()
HIST = P[["ndx", "qqq", "qld", "tqqq", "ief", "gold_phys"]].copy()


def indep(ndx_r, ief_r, gold_r, sleeves=None, rfd=None, months=None, dows=None):
    ndx_r = np.asarray(ndx_r, float); n = len(ndx_r)
    rfd_ = RFD if rfd is None else np.asarray(rfd, float)
    f = pd.DataFrame(index=FWD[:n])
    f["ndx"] = float(P["ndx"].iloc[-1]) * np.cumprod(1 + ndx_r)
    if sleeves is None:
        f["qqq"] = ndx_r - 0.0020 / 252; f["qld"] = 2 * ndx_r - 0.0095 / 252 - rfd_
        f["tqqq"] = 3 * ndx_r - 0.0095 / 252 - 2 * rfd_
    else:
        for k in ["qqq", "qld", "tqqq"]:
            f[k] = np.asarray(sleeves[k], float)
    f["ief"] = np.asarray(ief_r, float); f["gold_phys"] = np.asarray(gold_r, float)
    X = pd.concat([HIST, f]); nd = X["ndx"]
    X["ema100"] = nd.ewm(span=100, adjust=False).mean(); X["above"] = nd > X["ema100"]
    X["vol20"] = nd.pct_change().rolling(20).std() * np.sqrt(252)
    X["er20"] = nd.diff(20).abs() / nd.diff().abs().rolling(20).sum().replace(0, np.nan)
    mo = np.asarray(X.index.month).copy(); dw = np.asarray(X.index.dayofweek).copy()
    if months is not None: mo[-n:] = months
    if dows is not None: dw[-n:] = dows
    X["month"] = mo
    out = {}
    w = C.strategy_weights(X, lev="qld", voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
    rs = {d: C.backtest(X, w, mark=pd.Series(dw == i, index=X.index)).values[-n:] for i, d in enumerate(DAYS)}
    out["Fri"] = float(np.prod(1 + rs["Fri"]) - 1)
    out["tr"] = float(np.prod(1 + sum(rs[d] for d in DAYS) / 5) - 1)
    return out


# ---------------- (a) calendar split
print("(a) C9 calendar split: weekday phase vs Aug/Sep season months", flush=True)
rows = []
for dte in ["2018-12-07", "2022-01-21", "2000-04-14", "2011-08-05", "2025-04-04", "2020-03-06", "2015-08-24",
            "1990-08-10", "1998-08-31"]:
    i = idx.get_loc(pd.Timestamp(dte)); s0 = i + 1
    sg = P.iloc[s0:s0 + H]
    om, od = sg.index.month.values, sg.index.dayofweek.values
    a = (sg["ndx_ret"].values, sg["ief"].values, sg["gold_phys"].values)
    Cc = indep(*a, months=om, dows=od)          # own months + own weekdays
    Cm = indep(*a, months=om)                   # own months, forward weekdays
    Cd = indep(*a, dows=od)                     # forward months, own weekdays
    D = indep(*a)                               # forward months + forward weekdays (= spec 'after')
    fm = FWD.month.values
    n_season_own = int(np.isin(om, (8, 9)).sum()); n_season_fwd = int(np.isin(fm, (8, 9)).sum())
    r = dict(date=dte, own_season_days=n_season_own, fwd_season_days=n_season_fwd,
             own_season_pathdays=[int(np.flatnonzero(np.isin(om, (8, 9)))[0]) + 1, int(np.flatnonzero(np.isin(om, (8, 9)))[-1]) + 1] if n_season_own else None,
             fwd_season_pathdays=[int(np.flatnonzero(np.isin(fm, (8, 9)))[0]) + 1, int(np.flatnonzero(np.isin(fm, (8, 9)))[-1]) + 1])
    for b in ("Fri", "tr"):
        r[f"{b}_total_cal"] = D[b] - Cc[b]
        r[f"{b}_season_only"] = Cd[b] - Cc[b]          # move the months, keep own weekdays
        r[f"{b}_weekday_only"] = Cm[b] - Cc[b]         # move the weekdays, keep own months
    rows.append(r)
    print(f"  {dte}: season path-days own {r['own_season_pathdays']} fwd {r['fwd_season_pathdays']} | tranche: total "
          f"{r['tr_total_cal']*100:+6.1f} season-only {r['tr_season_only']*100:+6.1f} weekday-only {r['tr_weekday_only']*100:+6.1f}"
          f" | Fri: total {r['Fri_total_cal']*100:+6.1f} season-only {r['Fri_season_only']*100:+6.1f} weekday-only "
          f"{r['Fri_weekday_only']*100:+6.1f}", flush=True)
df = pd.DataFrame(rows)
for b in ("Fri", "tr"):
    print(f"  {b}: mean|total| {df[f'{b}_total_cal'].abs().mean()*100:.2f}pp, mean|season-only| "
          f"{df[f'{b}_season_only'].abs().mean()*100:.2f}pp, mean|weekday-only| {df[f'{b}_weekday_only'].abs().mean()*100:.2f}pp",
          flush=True)

# ---------------- (b) endpoint pin
print("\n(b) day-252 endpoint pin: pinned (spec) vs free after the last real waypoint", flush=True)


def free_bridge(z, pins_real):
    """bridge only up to the scenario's last real waypoint; raw (unpinned) noise afterwards."""
    out = np.array(z, float)
    if pins_real:
        last = max(pins_real)
        out[:last] = bridge(np.asarray(z, float)[:last], sorted(pins_real))
    return out


def path_free(spec, vol, seed):
    rng = np.random.default_rng([20261118, seed])
    zn, zy, zg = rng.standard_normal(H), rng.standard_normal(H), rng.standard_normal(H)
    dn = np.diff(np.concatenate([[0.0], spec["ndx"]])) + free_bridge(zn * vol / math.sqrt(252), [int(x) for x in spec["wn"]])
    ndx_r = np.expm1(dn)
    _n, ief_r, gold_r, tlt_r, bil_r, _y = synth_path(spec, vol, seed, "A")     # off-leg legs unchanged
    return ndx_r, ief_r, gold_r, tlt_r, bil_r


res_b = {}
for short in ("S0", "S3"):
    spec = [v for k, v in SCEN.items() if k.split(" ")[0] == short][0]
    pin, free = [], []
    for sd in range(40):
        pin.append(evaluate(*synth_path(spec, 0.30, sd, "A")[:5]))
        free.append(evaluate(*path_free(spec, 0.30, sd)))
    def med(rows_, k):
        v = [r_[k] for r_ in rows_ if r_[k] is not None]
        return float(np.median(v)) if v else None
    res_b[short] = {tag: dict(NDX_r12m=med(rr, "NDX_r12m"), NDX_r12m_p10=float(np.percentile([r_["NDX_r12m"] for r_ in rr], 10)),
                              NDX_r12m_p90=float(np.percentile([r_["NDX_r12m"] for r_ in rr], 90)),
                              QQQ_r12m=med(rr, "QQQ_r12m"), QLD_r12m=med(rr, "QLD_r12m"), F1=med(rr, "F1_Fri_r12m"),
                              F2=med(rr, "F2_Fri_r12m"), F3=med(rr, "F3_Fri_r12m"), F2tr=med(rr, "F2_tr_r12m"),
                              whip=float(np.mean([r_["Fri_whip_12m"] for r_ in rr])),
                              F2_minus_QQQ=float(np.median([r_["F2_Fri_r12m"] - r_["QQQ_r12m"] for r_ in rr])))
                    for tag, rr in (("pinned_spec", pin), ("free_after_last_waypoint", free))}
    for tag in res_b[short]:
        x = res_b[short][tag]
        print(f"  {short} {tag:26s} NDX12 med {x['NDX_r12m']*100:+6.1f} [p10 {x['NDX_r12m_p10']*100:+6.1f} p90 "
              f"{x['NDX_r12m_p90']*100:+6.1f}] QQQ {x['QQQ_r12m']*100:+6.1f} QLD {x['QLD_r12m']*100:+6.1f} | F1 {x['F1']*100:+6.1f}"
              f" F2 {x['F2']*100:+6.1f} F3 {x['F3']*100:+6.1f} F2tr {x['F2tr']*100:+6.1f} | whip {x['whip']:.2f} | "
              f"F2-QQQ med {x['F2_minus_QQQ']*100:+6.1f}  ({time.time()-T0:.0f}s)", flush=True)

# ---------------- (c) tranche participation in replays
print("\n(c) replay rebound participation (literal): Friday vs tranche; F2/QQQ capture", flush=True)
for d, r in STORED["replays"].items():
    m = r["literal"]
    print(f"  {d}: Fri part {m['Fri_participation']} tr part {m['tr_participation']} | F2 Fri/QQQ capture "
          f"{m['F2_Fri_capture_vs_qqq']} tr {m['F2_tr_capture_vs_qqq']} | F2tr r12 {m['F2_tr_r12m']*100:+.1f} QQQ "
          f"{m['QQQ_r12m']*100:+.1f} | after: F2 Fri {r['after']['F2_Fri_r12m']*100:+.1f} QQQ {r['after']['QQQ_r12m']*100:+.1f}",
          flush=True)
print(f"done {time.time()-T0:.0f}s")
