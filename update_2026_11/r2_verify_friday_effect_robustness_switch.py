"""
r2_verify_friday_effect_robustness_switch.py -- follow-up ROBUSTNESS checks of r2_friday_effect.
Read-only w.r.t. every project file; numeric output -> argv[1] (scratch).

S1  Is 'Friday samples fewer EMA on/off switches' (189 vs 214-217) a stable feature? Switch counts per weekday
    by half and EMA span on NDX, and per weekday in every cross-market series (own-price EMA-100, spec windows).
S2  Monday-worst (claim C9) robustness: NDX F2 Monday rank with single crash episodes removed; cross-market
    Monday worst-Calmar count under holiday substitution and in each half.
S3  Next-session continuation by weekday on every market, by half (is Friday the least-reverting day?).
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT_PATH = os.path.abspath(sys.argv[1])
assert not OUT_PATH.startswith(os.path.abspath(HERE))
P = C.panel(); IDX = P.index; N = len(IDX)
START, FREEZE, LAST = C.START, C.FREEZE, P.index[-1]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
NB = {"books": 0}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def last_idx(mask):
    pos = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(pos)


def run_book(W, R, mask, lag=1, cost=C.COST):
    li = last_idx(mask)
    H = np.where((li >= 0)[:, None], W[np.maximum(li, 0)], 0.0)
    if lag:
        H = np.vstack([np.zeros((lag, W.shape[1])), H[:-lag]])
    NB["books"] += 1
    return (H * R).sum(1) - np.abs(np.diff(H, axis=0, prepend=np.zeros((1, H.shape[1])))).sum(1) * cost


def wm(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def st1(dates, r, a, b):
    m = wm(dates, a, b)
    rr = r[m]; d = dates[m]
    e = np.cumprod(1 + rr)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    return dict(cagr=float(cagr), maxdd=dd, calmar=float(cagr / abs(dd)))


def f_weights(lvl, month, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, span=100):
    lvl = pd.Series(lvl)
    ema = lvl.ewm(span=span, adjust=False).mean()
    above = (lvl > ema)
    vol20 = lvl.pct_change().rolling(20).std() * np.sqrt(252)
    er20 = lvl.diff(20).abs() / lvl.diff().abs().rolling(20).sum().replace(0, np.nan)
    scale = (voltgt / vol20).clip(0, 1).fillna(1.0)
    scale = (scale * np.where(np.isin(month, (8, 9)), 0.5, 1.0)).clip(0, 1)
    clean = (er20 >= chop_thr).fillna(False)
    eq = (above & clean) * scale + (above & ~clean) * scale * chop_pos
    core = 1.0 - gold_w
    lev = (eq * core).values
    return np.column_stack([lev, np.clip(core - lev, 0, None), np.full(len(lvl), gold_w)]), above.values.astype(bool)


def switch_counts(above, dates, a, b):
    dow = pd.DatetimeIndex(dates).dayofweek.values
    m = wm(dates, a, b)
    out = {}
    for i, d in enumerate(DAYS):
        mk = np.where(m & (dow == i))[0]
        out[d] = int((above[mk][1:] != above[mk][:-1]).sum())
    return out


OUT = {}
# ------------------------------------------------------------------ S1 NDX by half x span
H1 = (START, pd.Timestamp("2006-12-31")); H2 = (pd.Timestamp("2007-01-01"), FREEZE)
s1 = {}
for span in [50, 75, 100, 125, 150, 200, 250]:
    _, ab = f_weights(P["ndx"], IDX.month.values, 0.30, span=span)
    s1[span] = {wn: switch_counts(ab, IDX, *w) for wn, w in {"IS": (START, FREEZE), "H1": H1, "H2": H2}.items()}
OUT["S1_ndx_switches_by_span_half"] = s1
for span, v in s1.items():
    log(f"NDX span {span:3d} switches IS {v['IS']}  H1 {v['H1']}  H2 {v['H2']}")
OUT["S1_ndx_fri_fewest"] = {f"{s}_{wn}": bool(v[wn]["Fri"] == min(v[wn].values())) for s, v in s1.items() for wn in v}
OUT["S1_ndx_fri_rank_fewest"] = {f"{s}_{wn}": int(sorted(v[wn].values()).index(v[wn]["Fri"]) + 1) for s, v in s1.items() for wn in v}

# ------------------------------------------------------------------ cross-market (spec construction, copied)
Y = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))


def yser(t, col="close"):
    s = Y[Y.ticker == t].set_index("date")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index <= LAST].astype(float)


def asof(s, dates):
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


irx, tnx = yser("^IRX"), yser("^TNX")
rf_us = pd.concat([irx[irx.index < IDX[0]] / 100.0, P["rf"]]).sort_index()
yv = P["tnx"] / 100.0; dy = yv.diff(); carry = yv.shift(1) / 252
cal = pd.concat([P["ief_syn"] - carry, dy], axis=1).dropna(); cal.columns = ["y", "dy"]
Xc = np.column_stack([np.ones(len(cal)), -cal["dy"].values, 0.5 * cal["dy"].values ** 2])
a_, D_, Cx_ = np.linalg.lstsq(Xc, cal["y"].values, rcond=None)[0]
yt = tnx / 100.0
ief_pre = (a_ + yt.shift(1) / 252 - D_ * yt.diff() + 0.5 * Cx_ * yt.diff() ** 2)
ief_pre = ief_pre[ief_pre.index < IDX[0]].fillna(0)
IEF_IDX = (1 + pd.concat([ief_pre, P["ief"].fillna(0)]).sort_index()).cumprod()
GOLD_IDX = (1 + pd.concat([irx[irx.index < IDX[0]] / 100.0 / 252, P["gold_phys"].fillna(0)]).sort_index()).cumprod()


def market_panel(col):
    r = P[col].fillna(0).values.astype(float)
    nz = np.where(r != 0)[0]
    return dict(dates=IDX, lvl=np.cumprod(1 + r), ret=r, rf_d=P["rf_d"].values, ief=P["ief"].fillna(0).values,
                gold=P["gold_phys"].fillna(0).values, start=max(START, IDX[min(nz[0] + 250, N - 1)]))


def market_yf(t, fixed=None):
    c = yser(t); c = c[c > 0]; dates = c.index
    return dict(dates=dates, lvl=c.values, ret=c.pct_change().fillna(0).values, rf_d=(asof(rf_us, dates).bfill() / 252).values,
                ief=asof(IEF_IDX, dates).pct_change().fillna(0).values, gold=asof(GOLD_IDX, dates).pct_change().fillna(0).values,
                start=pd.Timestamp(fixed) if fixed else max(START, dates[250]))


MK = {"NDX-2xsim": market_panel("ndx_ret"), "SPX": market_panel("spx"), "RUT": market_panel("rut"), "TSX": market_panel("tsx"),
      "HSI": market_panel("hsi"), "DAX": market_yf("^GDAXI"), "N225": market_yf("^N225", "1975-01-01"),
      "IXIC": market_yf("^IXIC", "1975-01-01"), "HSI_own": market_yf("^HSI")}


def subst_masks(dates):
    dates = pd.DatetimeIndex(dates)
    wk = pd.factorize(dates.to_period("W-FRI"))[0]
    df = pd.DataFrame({"wk": wk, "dow": dates.dayofweek.values, "pos": np.arange(len(dates))})
    out = {d: np.zeros(len(dates), bool) for d in DAYS}
    for _, gg in df.groupby("wk"):
        for i, d in enumerate(DAYS):
            c = gg[gg.dow >= i]
            out[d][(c.pos.iloc[0] if (len(c) and i < 4) else gg.pos.iloc[-1])] = True
    return out


s1x, s2x, s3x = {}, {}, {}
for name, M in MK.items():
    dates = pd.DatetimeIndex(M["dates"]); dow = dates.dayofweek.values
    W, ab = f_weights(pd.Series(M["lvl"], index=dates), dates.month.values, 0.30)
    R = np.column_stack([2 * M["ret"] - 0.0095 / 252 - M["rf_d"], M["ief"], M["gold"]])
    st, en = M["start"], FREEZE
    mid = pd.Timestamp((st + (en - st) / 2).date())
    wins = {"IS": (st, en), "H1": (st, mid), "H2": (mid + pd.Timedelta(days=1), en)}
    s1x[name] = {wn: switch_counts(ab, dates, *w) for wn, w in wins.items()}
    sk = {d: run_book(W, R, dow == i) for i, d in enumerate(DAYS)}
    sm = subst_masks(dates)
    su = {d: run_book(W, R, sm[d]) for d in DAYS}
    row = {}
    for hol, bk in (("skip", sk), ("subst", su)):
        for wn, w in wins.items():
            cm = {d: st1(dates, bk[d], *w)["calmar"] for d in DAYS}
            row[f"{hol}_{wn}"] = dict(worst=min(cm, key=cm.get), best=max(cm, key=cm.get),
                                      mon_rank=int(sorted(cm.values(), reverse=True).index(cm["Mon"]) + 1),
                                      fri_rank=int(sorted(cm.values(), reverse=True).index(cm["Fri"]) + 1))
    s2x[name] = row
    # S3 next-session correlation by weekday, by half
    r = M["ret"]; rn = np.r_[r[1:], np.nan]
    c3 = {}
    for wn, w in wins.items():
        m = wm(dates, *w) & np.r_[wm(dates, *w)[1:], False] & np.isfinite(rn)
        c3[wn] = {d: round(float(pd.Series(r[m & (dow == i)]).corr(pd.Series(rn[m & (dow == i)]), method="spearman")), 3)
                  for i, d in enumerate(DAYS)}
    s3x[name] = c3
    log(f"{name:10s} switches IS {s1x[name]['IS']} H1 {s1x[name]['H1']} H2 {s1x[name]['H2']}")
    log(f"{'':10s} worst-Calmar skip IS/H1/H2 {row['skip_IS']['worst']}/{row['skip_H1']['worst']}/{row['skip_H2']['worst']}"
        f"  subst IS/H1/H2 {row['subst_IS']['worst']}/{row['subst_H1']['worst']}/{row['subst_H2']['worst']}"
        f"  | Mon rank skip IS {row['skip_IS']['mon_rank']} subst IS {row['subst_IS']['mon_rank']}")
    log(f"{'':10s} next-session Spearman IS {c3['IS']}  H1 {c3['H1']}  H2 {c3['H2']}")
OUT["S1_xmkt_switches"] = s1x
OUT["S2_xmkt_worst_calmar"] = s2x
OUT["S3_xmkt_next_session_spearman"] = s3x
SEVEN = ["SPX", "RUT", "TSX", "HSI", "DAX", "N225", "IXIC"]
OUT["S1_xmkt_fri_fewest_switches_IS"] = [m for m in SEVEN if s1x[m]["IS"]["Fri"] == min(s1x[m]["IS"].values())]
OUT["S1_xmkt_fri_switch_rank_IS"] = {m: int(sorted(s1x[m]["IS"].values()).index(s1x[m]["IS"]["Fri"]) + 1) for m in SEVEN}
OUT["S2_mon_worst_counts"] = {k: int(sum(s2x[m][k]["worst"] == "Mon" for m in SEVEN)) for k in s2x["SPX"]}
OUT["S3_fri_highest_spearman_count"] = {wn: [m for m in SEVEN if s3x[m][wn]["Fri"] == max(s3x[m][wn].values())] for wn in ("IS", "H1", "H2")}
OUT["S3_mon_lowest_spearman_count"] = {wn: [m for m in SEVEN if s3x[m][wn]["Mon"] == min(s3x[m][wn].values())] for wn in ("IS", "H1", "H2")}
log("Friday fewest switches IS in:", OUT["S1_xmkt_fri_fewest_switches_IS"], "ranks", OUT["S1_xmkt_fri_switch_rank_IS"])
log("Monday worst-Calmar counts (7 markets):", OUT["S2_mon_worst_counts"])
log("Friday highest next-session Spearman:", OUT["S3_fri_highest_spearman_count"])
log("Monday lowest next-session Spearman:", OUT["S3_mon_lowest_spearman_count"])

# ------------------------------------------------------------------ S2 NDX F2 Monday penalty with single episodes removed
w = C.strategy_weights(P, lev="qld", voltgt=0.30)
W2 = w.values; R2 = P[list(w.columns)].fillna(0).values
DOW = IDX.dayofweek.values
bk = {d: run_book(W2, R2, DOW == i) for i, d in enumerate(DAYS)}
EP = {"none": None, "GFC 2008-09..2009-03": ("2008-09-01", "2009-03-31"), "COVID 2020-02..04": ("2020-02-15", "2020-04-30"),
      "2000-02": ("2000-03-01", "2002-10-31"), "1998": ("1998-01-01", "1998-12-31"), "2022": ("2022-01-01", "2022-12-31")}
s2n = {}
for nm, ab in EP.items():
    z = wm(IDX, *ab) if ab else np.zeros(N, bool)
    cm = {d: st1(IDX, np.where(z, 0.0, bk[d]), START, FREEZE)["calmar"] for d in DAYS}
    cg = {d: st1(IDX, np.where(z, 0.0, bk[d]), START, FREEZE)["cagr"] for d in DAYS}
    s2n[nm] = dict(calmar={d: round(v, 3) for d, v in cm.items()}, cagr_pct={d: round(v * 100, 2) for d, v in cg.items()},
                   mon_rank_calmar=int(sorted(cm.values(), reverse=True).index(cm["Mon"]) + 1),
                   fri_rank_calmar=int(sorted(cm.values(), reverse=True).index(cm["Fri"]) + 1))
    log(f"NDX F2 drop {nm:22s} Calmar {s2n[nm]['calmar']}  Mon rank {s2n[nm]['mon_rank_calmar']} Fri rank {s2n[nm]['fri_rank_calmar']}")
OUT["S2_ndx_monday_episode_removal"] = s2n
OUT["books"] = NB["books"]


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(OUT_PATH, "w"), indent=1)
log("saved", OUT_PATH, "books", NB["books"])
