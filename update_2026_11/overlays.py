"""
update_2026_11/overlays.py  --  MACRO OVERLAYS RE-CHECK  (Nov-2026 refresh, KEY = overlays)

 (1) Prove the July 'confirmed yield-curve' rule is a logical no-op (truth table + code + backtest
     equality) and re-attribute its claimed '+1.0%/yr edge' to the bond off-leg: 2x2 grid
     {cash | bond off-leg} x {rule on | off}, on the ARCHIVED July inputs with the exact July
     frontier.py mechanics, and on the corrected Nov-2026 panel (F3/F2 + GLD + off-leg).
 (2) GENUINE curve rules: inversion caps the engine at 0.5 ; un-inversion (curve crosses back above 0
     within 24 months after an inversion lasting >= 63 days) multiplies the engine by 0.5 for 252 days.
     Also their risk-off (risk_on override) forms, an alternative inversion definition, and the combo.
 (3) RATE SHOCK: 10y yield up > 50/75/100 bp over 126d -> engine x0.5 ; separately off-leg IEF -> bills.
 (4) DOLLAR SHOCK: DXY 126d change > +5% -> engine x0.75.
 (5) CREDIT: HYG/IEF and LQD/IEF total-return ratio below its moving average -> engine x0.5 (2007+/2002+);
     long-history extension VWEHX (Vanguard HY fund, 1980+) / IEF (spliced) -> both halves testable.
     (FRED Moody's BAA/AAA was tried and is unreachable from this machine.)
 (6) VIX TERM STRUCTURE: VIX > VIX3M (backwardation) -> engine x0.5 (^VIX3M exists from 2006-07).

No look-ahead: every macro input is lagged ONE EXTRA trading day (the Friday-close decision uses
Thursday's macro close), then common.backtest applies weights from the next session (lag=1).
Baselines (shared mechanics, common.py): F3 = TQQQ vt .20 ; F2 = QLD vt .30 ; both + 15% GLD + IEF off-leg.
Every overlay variant is compared with its baseline on: full in-sample (1988-01-01..2026-07-24), both halves
(1988-2006 / 2007-freeze; for 2006+ signals the signal window split at its midpoint), crises, OOS
(2026-07-25..latest), paired block bootstrap (CAGR / Sharpe / Calmar differences), and a VOL-MATCHED DIAL
(the baseline with its engine uniformly scaled to the same in-sample volatility = "just hold less leverage").
Decision rule (pre-registered): INCLUDE only if Calmar AND Sharpe improve in BOTH halves, the in-sample
Calmar gain exceeds the 0.15 selection-cost haircut, and the gain is significant (bootstrap p<0.05 on the
Sharpe difference vs base, or on the CAGR difference vs the vol-matched dial).  MARGINAL = both halves and
gain > 0.07.  Otherwise REJECT.
Writes update_2026_11/overlays.json (+ cache update_2026_11/overlays_ext.parquet for fetched series).
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
P = C.panel()
idx = P.index
FREEZE, START = C.FREEZE, C.START
LAST = idx[-1]
IS = (START, FREEZE)
H1 = (pd.Timestamp("1988-01-01"), pd.Timestamp("2006-12-31"))
H2 = (pd.Timestamp("2007-01-01"), FREEZE)
OOS = (pd.Timestamp("2026-07-25"), LAST)
OUT_JSON = os.path.join(HERE, "overlays.json")
EXT_PATH = os.path.join(HERE, "overlays_ext.parquet")
HAIRCUT_LO, HAIRCUT_HI = 0.07, 0.15
N_BOOT = 2000

BASES = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
COMMON = dict(gold="gold_phys", gold_w=0.15, off="ief", chop_thr=0.25, chop_pos=0.5, seasonal=True)
COUNT = dict(backtests=0, candidate_variants=0, reproduction_runs=0)


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ core helpers
def W(base, **kw):
    return C.strategy_weights(P, **{**COMMON, **BASES[base], **kw})


def run(base, **kw):
    COUNT["backtests"] += 1
    w = W(base, **kw)
    return C.backtest(P, w), w


def st(r, win):
    a, b = pd.Timestamp(win[0]), pd.Timestamp(win[1])
    s = C.stats(r, a, b, start=None)
    x = r[(r.index >= a) & (r.index <= b)].dropna()
    s["vol"] = float(x.std() * np.sqrt(252)) if len(x) > 20 else float("nan")
    return s


def halves_for(win):
    a, b = pd.Timestamp(win[0]), pd.Timestamp(win[1])
    if a <= pd.Timestamp("1988-01-01"):
        return [H1, H2]
    mid = (a + (b - a) / 2).normalize()
    return [(a, mid), (mid + pd.Timedelta(days=1), b)]


_DRAWS, _BS = {}, {}


def draws(T, n=N_BOOT, block=63, seed=0):
    """block start indices: the SAME sequence common.block_bootstrap_cagr_diff draws (seed 0)."""
    key = (T, n, block, seed)
    if key not in _DRAWS:
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(T / block))
        _DRAWS[key] = np.stack([rng.integers(0, T, nb) for _ in range(n)])
    return _DRAWS[key]


def boot_series(r, win, cache_key=None, n=N_BOOT, block=63, seed=0, batch=250):
    """per-resample CAGR / Sharpe / Calmar of one series under the shared circular block draws.
    All strategy series share the panel calendar, so equal windows => identical (paired) resamples."""
    ck = None if cache_key is None else (cache_key, str(win[0]), str(win[1]))
    if ck is not None and ck in _BS:
        return _BS[ck]
    x = r[(r.index >= pd.Timestamp(win[0])) & (r.index <= pd.Timestamp(win[1]))].dropna().values
    T = len(x)
    sts = draws(T, n, block, seed)
    ar = np.arange(block)
    lr = np.log1p(x)
    G, SH, DD = np.empty(n), np.empty(n), np.empty(n)
    for s0 in range(0, n, batch):
        sb = sts[s0:s0 + batch]
        ix = (sb[:, :, None] + ar[None, None, :]).reshape(len(sb), -1)[:, :T] % T
        cs = np.cumsum(lr[ix], axis=1)
        G[s0:s0 + len(sb)] = np.expm1(cs[:, -1] * 252 / T)                       # = prod(1+r)^(252/T)-1
        DD[s0:s0 + len(sb)] = np.expm1((cs - np.maximum.accumulate(cs, axis=1)).min(axis=1))   # = min(e/cummax(e))-1
        xs = x[ix]
        SH[s0:s0 + len(sb)] = xs.mean(axis=1) / xs.std(axis=1, ddof=1) * np.sqrt(252)
    res = dict(g=G, sh=SH, cal=G / np.abs(DD), T=T, n=n, block=block)
    if ck is not None:
        _BS[ck] = res
    return res


def pair(a, b):
    assert a["T"] == b["T"], "unpaired bootstrap windows"

    def summ(x):
        return dict(mean=float(x.mean()), lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)),
                    p_le0=float((x <= 0).mean()))
    return dict(cagr=summ(a["g"] - b["g"]), sharpe=summ(a["sh"] - b["sh"]), calmar=summ(a["cal"] - b["cal"]),
                n=a["n"], block=a["block"], days=a["T"])


def boot(r1, r2, win, key1=None, key2=None):
    """paired circular block bootstrap of the CAGR / Sharpe / Calmar differences r1 - r2."""
    return pair(boot_series(r1, win, key1), boot_series(r2, win, key2))


def lag1(flag):
    """macro flag known at the close of t-1 -> usable for the decision at the close of t."""
    return flag.reindex(idx).astype(float).shift(1).fillna(0.0).astype(bool)


def mult_kw(flag, m):
    return dict(extra_scale=pd.Series(np.where(flag.values, m, 1.0), index=idx))


def cap_scale(flag, cap, base):
    eq0 = BASE_EQ[base]
    es = np.where(flag.values & (eq0.values > cap), cap / eq0.replace(0, np.nan).values, 1.0)
    return pd.Series(es, index=idx).fillna(1.0)


def riskoff_kw(flag):
    return dict(risk_on=P["above"].astype(bool) & ~flag)


def offleg_kw(flag, to="bills", frm="ief"):
    f = flag.values.astype(float)
    return dict(off_weights=pd.DataFrame({frm: 1.0 - f, to: f}, index=idx))


def episodes(differ, r, rb, win, top=8):
    m = pd.Series((idx >= pd.Timestamp(win[0])) & (idx <= pd.Timestamp(win[1])), index=idx)
    d = (differ & m)
    g = (d != d.shift()).cumsum()
    out = []
    for _, s in d.groupby(g):
        if not s.iloc[0]:
            continue
        a, b = s.index[0], s.index[-1]
        ro, rbb = C.period_return(r, a, b), C.period_return(rb, a, b)
        out.append(dict(start=str(a.date()), end=str(b.date()), days=int(len(s)), ov=ro, base=rbb, diff=ro - rbb))
    tot = float(sum(e["diff"] for e in out))
    out.sort(key=lambda e: -abs(e["diff"]))
    return dict(n=len(out), sum_simple_diff=tot, top=out[:top])


# ------------------------------------------------------------------ baselines
log(f"panel {idx[0].date()}..{LAST.date()}  rows={len(P)}")
BASE_R, BASE_W, BASE_EQ, HELD_BASE = {}, {}, {}, {}
for b in BASES:
    r, w = run(b)
    BASE_R[b], BASE_W[b] = r, w
    BASE_EQ[b] = w[BASES[b]["lev"]] / (1 - COMMON["gold_w"])
    HELD_BASE[b] = C.held_weights(P, w)
baselines = {}
for b in BASES:
    baselines[b] = dict(spec={**COMMON, **BASES[b]}, IS=st(BASE_R[b], IS), H1=st(BASE_R[b], H1), H2=st(BASE_R[b], H2),
                        OOS=st(BASE_R[b], OOS), OOS_return=C.period_return(BASE_R[b], *OOS),
                        to_latest=st(BASE_R[b], (START, LAST)), crises=C.crises(BASE_R[b]))
    log(f"BASE {b}: IS {C.fmt(baselines[b]['IS'])} | OOS {baselines[b]['OOS_return']*100:+.1f}%")

# bootstrap implementation check vs common.py (same draws => identical CAGR-diff distribution)
_chk_mine = boot(BASE_R["F2"], BASE_R["F3"], IS, ("base", "F2"), ("base", "F3"))["cagr"]
_chk_c = C.block_bootstrap_cagr_diff(BASE_R["F2"][:FREEZE], BASE_R["F3"][:FREEZE])
log(f"bootstrap check F2-F3 CAGR diff: mine mean {_chk_mine['mean']:.6f} [{_chk_mine['lo']:.4f},{_chk_mine['hi']:.4f}]"
    f"  common.py mean {_chk_c['mean']:.6f} [{_chk_c['lo']:.4f},{_chk_c['hi']:.4f}]")

# vol-matched dial grid: baseline with engine uniformly scaled by k
KGRID = [round(float(k), 2) for k in np.arange(0.30, 1.0001, 0.01)]
DIAL = {b: {k: run(b, extra_scale=pd.Series(k, index=idx))[0] for k in KGRID} for b in BASES}
_DV = {}


def dial_table(b, win):
    key = (b, str(win[0]), str(win[1]))
    if key not in _DV:
        _DV[key] = {k: st(DIAL[b][k], win) for k in KGRID}
    return _DV[key]


def dial_match_vol(b, target, win):
    t = dial_table(b, win)
    k = min(t, key=lambda kk: abs(t[kk]["vol"] - target))
    return k, DIAL[b][k], t[k]


def dial_cagr_at_dd(b, dd, win):
    t = dial_table(b, win)
    arr = sorted((t[k]["maxdd"], t[k]["cagr"]) for k in t)
    return float(np.interp(dd, [a for a, _ in arr], [c for _, c in arr]))


log(f"dial grid built ({len(KGRID)} k-values x 2 bases)")

# ================================================================== (1) NO-OP PROOF
above = P["above"].astype(bool)
inv_raw = (P["curve"] < 0).fillna(False)
inv_recent = inv_raw.astype(float).rolling(252, min_periods=1).max().astype(bool)
confirmed = inv_recent & ~above
ron_july = above & ~confirmed
truth = []
for A in (False, True):
    for I in (False, True):
        conf = I and not A
        ron = A and not conf
        truth.append(dict(above=A, inv_recent=I, confirmed=conf, risk_on_july=ron, risk_on_equals_above=(ron == A)))
noop = dict(
    algebra="A & ~(I & ~A) = A & (~I | A) = (A & ~I) | (A & A) = (A & ~I) | A = A   (absorption law)",
    truth_table=truth,
    days_total=int(len(idx)), days_inv_recent=int(inv_recent.sum()), days_confirmed_flag=int(confirmed.sum()),
    days_rule_changes_risk_on=int((ron_july != above).sum()),
)
eq_runs = {}
for b in BASES:
    r_rule, w_rule = run(b, risk_on=ron_july)
    eq_runs[b] = dict(max_abs_daily_return_diff=float((r_rule - BASE_R[b]).abs().max()),
                      max_abs_weight_diff=float((w_rule - BASE_W[b]).abs().max().max()))
noop["backtest_equality"] = eq_runs
log(f"NO-OP: rule changes risk-on on {noop['days_rule_changes_risk_on']} of {len(idx)} days "
    f"(confirmed flag was ON {noop['days_confirmed_flag']} days, always while already risk-off); "
    f"max |dr| F3={eq_runs['F3']['max_abs_daily_return_diff']:.1e}, F2={eq_runs['F2']['max_abs_daily_return_diff']:.1e}")

# 2x2 on the corrected panel: off-leg {bills(cash), ief, tlt, bond10_old} x rule {off, on}
grid_now = {}
for b in BASES:
    grid_now[b] = {}
    for off in ["bills", "ief", "tlt", "bond10_old"]:
        row = {}
        for rule in ["off", "on"]:
            kw = dict(off=off)
            if rule == "on":
                kw["risk_on"] = ron_july
            r, _ = run(b, **kw)
            COUNT["reproduction_runs"] += 1
            row[rule] = dict(IS=st(r, IS), H1=st(r, H1), H2=st(r, H2), y2022=C.period_return(r, "2022-01-01", "2022-12-31"),
                             dotcom=C.period_return(r, "2000-03-01", "2002-10-31"), _r=r)
        row["rule_effect_max_abs_dr"] = float((row["on"]["_r"] - row["off"]["_r"]).abs().max())
        grid_now[b][off] = row
    cash = grid_now[b]["bills"]["off"]["IS"]
    for off in ["ief", "tlt", "bond10_old"]:
        o = grid_now[b][off]["off"]["IS"]
        grid_now[b][off]["offleg_effect_vs_cash"] = dict(d_cagr=o["cagr"] - cash["cagr"], d_maxdd=o["maxdd"] - cash["maxdd"],
                                                         d_calmar=o["calmar"] - cash["calmar"], d_sharpe=o["sharpe"] - cash["sharpe"])
    log(f"2x2 {b}: cash {cash['cagr']*100:.2f}%/{cash['maxdd']*100:.1f}%/{cash['calmar']:.2f}  "
        f"IEF {grid_now[b]['ief']['off']['IS']['cagr']*100:.2f}%/{grid_now[b]['ief']['off']['IS']['maxdd']*100:.1f}%/"
        f"{grid_now[b]['ief']['off']['IS']['calmar']:.2f}  rule effect max|dr| = "
        f"{max(grid_now[b][o]['rule_effect_max_abs_dr'] for o in grid_now[b]):.1e}")
for b in grid_now:
    for off in grid_now[b]:
        for rule in ["off", "on"]:
            grid_now[b][off][rule].pop("_r", None)
noop["grid_corrected_panel"] = grid_now


# ---- exact July frontier.py mechanics on the ARCHIVED July inputs
def july_repro():
    A = os.path.join(ROOT, "data_letf", "archive_2026-07-24")
    d = pd.read_parquet(os.path.join(A, "letf_daily.parquet")).set_index("date")
    m = pd.read_parquet(os.path.join(A, "macro_daily.parquet")).reindex(d.index)
    ix = d.index
    rf_d = (d["rf"] / 252).fillna(0)
    bond = pd.Series(np.where(ix >= pd.Timestamp("2002-08-01"), m["tlt_ret"].values, rf_d.values), index=ix).fillna(0)
    R = {"tqqq": d["tqqq_ret"].fillna(0), "qqq": d["qqq_ret"].fillna(0), "cash": rf_d, "bond": bond}
    ndx = d["ndx"]
    ema = ndx.ewm(span=100, adjust=False).mean()
    vol = ndx.pct_change().rolling(20).std() * np.sqrt(252)
    er = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum().replace(0, np.nan)
    curve = m["curve"]
    mark = pd.Series(False, index=ix)
    mark[ndx.resample("W-FRI").last().index.intersection(ix)] = True
    ab = ndx > ema
    cinv = (curve < 0).fillna(False)
    inv_r = cinv.astype(float).rolling(252, min_periods=1).max().astype(bool)
    conf = inv_r & ~ab

    def build(voltgt=0.25, chop_thr=None, chop_pos=0.0, curve_mode=None, off="cash"):   # == frontier.py (no boost)
        scale = (voltgt / vol).clip(0, 1).fillna(1.0)
        ron = ab.copy()
        if curve_mode == "raw":
            ron = ron & ~cinv
        elif curve_mode == "confirm":
            ron = ron & ~conf
        w = pd.DataFrame(0.0, index=ix, columns=list(R))
        if chop_thr is None:
            w["tqqq"] = ron * scale
        else:
            clean = (er >= chop_thr).fillna(False)
            w["tqqq"] = (ron & clean) * scale + (ron & ~clean) * scale * chop_pos
            w["qqq"] = (ron & ~clean).astype(float) * (1.0 if chop_pos == 0 else 0.0)
        w["cash"] += (1 - w[["tqqq", "qqq"]].sum(axis=1)).clip(lower=0) * (1 if off == "cash" else 0)
        if off == "bond":
            w["bond"] = (1 - w[["tqqq", "qqq"]].sum(axis=1)).clip(lower=0)
        w = w.where(mark).ffill().fillna(0).shift(1).fillna(0)
        COUNT["reproduction_runs"] += 1
        return sum(w[a] * R[a] for a in R) - w.diff().abs().sum(axis=1).fillna(0) * C.COST

    def stats(r):        # == frontier.py stats (whole series, from 1985-10)
        e = (1 + r).cumprod()
        yrs = (r.index[-1] - r.index[0]).days / 365.25
        cagr = e.iloc[-1] ** (1 / yrs) - 1
        dd = (e / e.cummax() - 1).min()
        return dict(cagr=float(cagr), maxdd=float(dd), sharpe=float(r.mean() / r.std() * np.sqrt(252)),
                    calmar=float(cagr / abs(dd)))

    dial = {v: stats(build(voltgt=v)) for v in [0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]}
    dd_s = sorted((s["maxdd"], s["cagr"]) for s in dial.values())

    def edge(s):
        return s["cagr"] - float(np.interp(s["maxdd"], [a for a, _ in dd_s], [c for _, c in dd_s]))

    rows = {}
    cfgs = {
        "vol25 | cash | no rule  (July 'A: champion')": dict(voltgt=0.25),
        "vol25 | cash | confirmed rule": dict(voltgt=0.25, curve_mode="confirm"),
        "vol25 | bond | no rule": dict(voltgt=0.25, off="bond"),
        "vol25 | bond | confirmed rule  (July 'curve:confirm' row, '+1.0%/yr')": dict(voltgt=0.25, curve_mode="confirm", off="bond"),
        "vol25 | cash | RAW inversion risk-off": dict(voltgt=0.25, curve_mode="raw"),
        "vol25 | bond | RAW inversion risk-off  (July 'curve:raw' row)": dict(voltgt=0.25, curve_mode="raw", off="bond"),
        "chop<.25->50% | cash | no rule": dict(voltgt=0.25, chop_thr=0.25, chop_pos=0.5),
        "chop<.25->50% | cash | confirmed rule": dict(voltgt=0.25, chop_thr=0.25, chop_pos=0.5, curve_mode="confirm"),
        "chop<.25->50% | bond | no rule": dict(voltgt=0.25, chop_thr=0.25, chop_pos=0.5, off="bond"),
        "chop<.25->50% | bond | confirmed rule": dict(voltgt=0.25, chop_thr=0.25, chop_pos=0.5, curve_mode="confirm", off="bond"),
    }
    rets = {}
    for k, kw in cfgs.items():
        r = build(**kw)
        rets[k] = r
        s = stats(r)
        s["edge_vs_cash_dial"] = edge(s)
        rows[k] = s
    k_cc, k_cr = "vol25 | cash | confirmed rule", "vol25 | cash | no rule  (July 'A: champion')"
    k_bc, k_bn = "vol25 | bond | confirmed rule  (July 'curve:confirm' row, '+1.0%/yr')", "vol25 | bond | no rule"
    ident = dict(cash_rule_vs_norule_max_abs_dr=float((rets[k_cc] - rets[k_cr]).abs().max()),
                 bond_rule_vs_norule_max_abs_dr=float((rets[k_bc] - rets[k_bn]).abs().max()),
                 chop_cash_rule_vs_norule=float((rets["chop<.25->50% | cash | confirmed rule"] - rets["chop<.25->50% | cash | no rule"]).abs().max()),
                 chop_bond_rule_vs_norule=float((rets["chop<.25->50% | bond | confirmed rule"] - rets["chop<.25->50% | bond | no rule"]).abs().max()))
    attrib = dict(
        total_edge_of_july_curve_confirm_row=rows[k_bc]["edge_vs_cash_dial"],
        part_from_confirmed_rule=rows[k_bc]["edge_vs_cash_dial"] - rows[k_bn]["edge_vs_cash_dial"],
        part_from_bond_offleg=rows[k_bn]["edge_vs_cash_dial"] - rows[k_cr]["edge_vs_cash_dial"],
        july_reported_row=dict(cagr=0.321409, maxdd=-0.646101, calmar=0.497460, source="July frontier.csv 'curve:confirm'"),
        raw_rule_effect_with_bond_cagr=rows["vol25 | bond | RAW inversion risk-off  (July 'curve:raw' row)"]["cagr"] - rows[k_bn]["cagr"],
    )
    return dict(dial={f"vol{int(v*100)}": s for v, s in dial.items()}, rows=rows, identity=ident, attribution=attrib,
                data=f"data_letf/archive_2026-07-24 ({ix[0].date()}..{ix[-1].date()}), frontier.py mechanics, stats from data start")


ARCHIVE = os.path.join(ROOT, "data_letf", "archive_2026-07-24", "letf_daily.parquet")
if not os.path.exists(ARCHIVE):
    # the July input snapshot is not in the public repository (market data cannot be redistributed and the
    # data builders always fetch up to today); everything else in this script runs without it
    log(f"JULY REPRO skipped: {ARCHIVE} not found (the July data snapshot is not published)")
    noop["july_framework_reproduction"] = "skipped: data_letf/archive_2026-07-24 (July input snapshot) not available"
    jr = None
else:
    jr = july_repro()
    noop["july_framework_reproduction"] = jr
if jr is not None:
    log(f"JULY REPRO: curve:confirm(bond) edge {jr['attribution']['total_edge_of_july_curve_confirm_row']*100:+.2f}%/yr = "
        f"rule {jr['attribution']['part_from_confirmed_rule']*100:+.2f} + bond off-leg {jr['attribution']['part_from_bond_offleg']*100:+.2f}; "
        f"identity max|dr| cash {jr['identity']['cash_rule_vs_norule_max_abs_dr']:.1e} bond {jr['identity']['bond_rule_vs_norule_max_abs_dr']:.1e}")
    for k, s in jr["rows"].items():
        log(f"   {k:72s} CAGR {s['cagr']*100:6.2f}%  DD {s['maxdd']*100:6.1f}%  Cal {s['calmar']:.3f}  edge {s['edge_vs_cash_dial']*100:+.2f}%")

# ================================================================== external series (cached)
def load_ext():
    if os.path.exists(EXT_PATH):
        return pd.read_parquet(EXT_PATH), "cache"
    import yfinance as yf
    out = {}
    for t, k, adj in [("HYG", "hyg", True), ("IEF", "ief_px", True), ("LQD", "lqd", True), ("VWEHX", "vwehx", True),
                      ("^VIX", "vix_y", False), ("^VIX3M", "vix3m", False)]:
        h = yf.Ticker(t).history(period="max", auto_adjust=adj)["Close"].dropna()
        h.index = h.index.tz_localize(None).normalize()
        out[k] = h[~h.index.duplicated()]
        log(f"  fetched {t:7s} {h.index[0].date()}..{h.index[-1].date()} n={len(h)}")
    df = pd.DataFrame(out)
    df.to_parquet(EXT_PATH)
    return df, "yfinance"


EXT, ext_src = load_ext()
X = EXT[EXT.index <= LAST].reindex(EXT[EXT.index <= LAST].index.union(idx)).ffill().reindex(idx)
ext_cov = {c: [str(EXT[c].dropna().index[0].date()), str(EXT[c].dropna().index[-1].date())] for c in EXT.columns}
log(f"external series from {ext_src}: {ext_cov}")

# ================================================================== signals (raw = as of close t ; flags lagged 1d)
curve = P["curve"]


def uninversion_window(inv, defn="contig", min_len=63, within=504, hold=252):
    """trigger = curve crosses back above 0 (inverted at t-1, not at t) within `within` trading days after a
    qualifying inversion; qualifying = 'contig': a spell of >= min_len consecutive inverted days
    (measured to the last day of that spell); 'cum252': >= min_len inverted days in the trailing 252.
    window = trigger day + next hold-1 days."""
    a = inv.fillna(False).astype(bool).values
    n = len(a)
    runl = np.zeros(n, dtype=int)
    for i in range(n):
        runl[i] = ((runl[i - 1] + 1) if i > 0 else 1) if a[i] else 0
    if defn == "contig":
        qual = runl >= min_len
    else:
        qual = pd.Series(a.astype(float)).rolling(252, min_periods=1).sum().values >= min_len
    last_q = np.full(n, -10 ** 9, dtype=np.int64)
    lq = -10 ** 9
    for i in range(n):
        if qual[i]:
            lq = i
        last_q[i] = lq
    cross = np.zeros(n, dtype=bool)
    cross[1:] = a[:-1] & ~a[1:]
    trig = cross & ((np.arange(n) - last_q) <= within)
    win = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(trig):
        win[i:i + hold] = True
    return pd.Series(win, index=inv.index), pd.Series(trig, index=inv.index)


uw_c, trig_c = uninversion_window(inv_raw, "contig")
uw_k, trig_k = uninversion_window(inv_raw, "cum252")


def spans(s):
    s = s.astype(bool)
    g = (s != s.shift()).cumsum()
    return [[str(x.index[0].date()), str(x.index[-1].date()), int(len(x))] for _, x in s.groupby(g) if x.iloc[0]]


spells = [sp for sp in spans(inv_raw) if sp[2] >= 3]
curve_info = dict(inversion_spells_ge3d=spells,
                  uninv_contig_triggers=[str(d.date()) for d in trig_c[trig_c].index],
                  uninv_contig_windows=spans(uw_c),
                  uninv_cum252_triggers=[str(d.date()) for d in trig_k[trig_k].index],
                  uninv_cum252_windows=spans(uw_k))
log(f"un-inversion (contig>=63d) triggers: {curve_info['uninv_contig_triggers']}")
log(f"un-inversion windows contig: {curve_info['uninv_contig_windows']}")
log(f"un-inversion windows cum252: {curve_info['uninv_cum252_windows']}")

d126 = P["tnx"] - P["tnx"].shift(126)                       # pct points
dxy126 = P["dxy"] / P["dxy"].shift(126) - 1
r_hy = X["hyg"] / X["ief_px"]
r_lq = X["lqd"] / X["ief_px"]
ief_tr = (1 + P["ief"]).cumprod()                           # spliced IEF total-return index (real from 2002-07)
r_vw = X["vwehx"] / ief_tr
vix_y, vix3m = X["vix_y"], X["vix3m"]
vts = vix_y / vix3m

RAW = {
    "inv": inv_raw, "uninv_contig": uw_c, "uninv_cum252": uw_k,
    "rs50": d126 > 0.50, "rs75": d126 > 0.75, "rs100": d126 > 1.00,
    "usd5": dxy126 > 0.05,
    "hy50": r_hy < r_hy.rolling(50).mean(), "hy200": r_hy < r_hy.rolling(200).mean(),
    "lq50": r_lq < r_lq.rolling(50).mean(),
    "vw50": r_vw < r_vw.rolling(50).mean(), "vw200": r_vw < r_vw.rolling(200).mean(),
    "vts": vts > 1.0, "vts5": vts.rolling(5).mean() > 1.0,
}
FLAG = {k: lag1(v) for k, v in RAW.items()}


def first_valid(*series):
    return max(s.dropna().index[0] for s in series) + pd.Timedelta(days=1)


WIN_HY = (max(first_valid(r_hy.rolling(200).mean()), pd.Timestamp("2008-01-01")), FREEZE)
WIN_LQ = (first_valid(r_lq.rolling(50).mean()), FREEZE)
WIN_VIX = (first_valid(vts.rolling(5).mean()), FREEZE)
WIN_VW = IS
log(f"windows: HYG {WIN_HY[0].date()}  LQD {WIN_LQ[0].date()}  VIX3M {WIN_VIX[0].date()}  VWEHX {WIN_VW[0].date()}")

# ================================================================== evaluation
RESULTS = []
RET_STORE = {}


def evaluate(name, family, base, kw, flag, win=IS, note="", raw=None):
    COUNT["candidate_variants"] += 1
    r, w = run(base, **kw)
    rb = BASE_R[base]
    hw = C.held_weights(P, w)
    cols = hw.columns.union(HELD_BASE[base].columns)
    differ = (hw.reindex(columns=cols, fill_value=0) - HELD_BASE[base].reindex(columns=cols, fill_value=0)).abs().sum(axis=1) > 1e-9
    m = (idx >= pd.Timestamp(win[0])) & (idx <= pd.Timestamp(win[1]))
    s_o, s_b = st(r, win), st(rb, win)
    hv = []
    for h in halves_for(win):
        so, sb = st(r, h), st(rb, h)
        hv.append(dict(window=[str(pd.Timestamp(h[0]).date()), str(pd.Timestamp(h[1]).date())], ov=so, base=sb,
                       d_cagr=so["cagr"] - sb["cagr"], d_maxdd=so["maxdd"] - sb["maxdd"],
                       d_sharpe=so["sharpe"] - sb["sharpe"], d_calmar=so["calmar"] - sb["calmar"]))
    bo = boot_series(r, win)
    bb = pair(bo, boot_series(rb, win, ("base", base)))
    k, rd, sd = dial_match_vol(base, s_o["vol"], win)
    bd = pair(bo, boot_series(rd, win, ("dial", base, k)))
    dd_edge = s_o["cagr"] - dial_cagr_at_dd(base, s_o["maxdd"], win)
    cro, crb = C.crises(r), C.crises(rb)
    d_cal, d_sh = s_o["calmar"] - s_b["calmar"], s_o["sharpe"] - s_b["sharpe"]
    halves_ok = all((h["d_calmar"] > 0) and (h["d_sharpe"] > 0) for h in hv)
    sig = (bb["sharpe"]["p_le0"] < 0.05) or (bd["cagr"]["p_le0"] < 0.05)
    reasons = []
    if not halves_ok:
        bad = [f"{h['window'][0][:4]}-{h['window'][1][:4]} dCal {h['d_calmar']:+.2f} dSh {h['d_sharpe']:+.2f}" for h in hv
               if not (h["d_calmar"] > 0 and h["d_sharpe"] > 0)]
        reasons.append("fails a half: " + "; ".join(bad))
    if d_cal <= HAIRCUT_HI:
        reasons.append(f"in-sample Calmar gain {d_cal:+.3f} <= haircut {HAIRCUT_HI}")
    if not sig:
        reasons.append(f"not significant (p Sharpe-diff vs base {bb['sharpe']['p_le0']:.2f}; p CAGR-diff vs vol-matched dial {bd['cagr']['p_le0']:.2f})")
    verdict = "INCLUDE" if (halves_ok and d_cal > HAIRCUT_HI and sig) else ("MARGINAL" if (halves_ok and d_cal > HAIRCUT_LO) else "REJECT")
    rec = dict(
        name=name, family=family, base=base, note=note, window=[str(pd.Timestamp(win[0]).date()), str(pd.Timestamp(win[1]).date())],
        flag_on_frac=float(flag[m].mean()) if flag is not None else None,
        weights_differ_frac=float(differ[m].mean()),
        ov=s_o, base_stats=s_b, d_cagr=s_o["cagr"] - s_b["cagr"], d_maxdd=s_o["maxdd"] - s_b["maxdd"],
        d_sharpe=d_sh, d_calmar=d_cal, d_calmar_after_haircut=[d_cal - HAIRCUT_LO, d_cal - HAIRCUT_HI],
        halves=hv, boot_vs_base=bb,
        dial=dict(k=k, stats=sd, d_cagr_vs_volmatched_dial=s_o["cagr"] - sd["cagr"], d_calmar_vs_dial=s_o["calmar"] - sd["calmar"],
                  boot_vs_dial=bd, edge_vs_dial_at_equal_maxdd=dd_edge),
        crises=dict(ov=cro, base=crb, diff={kk: cro[kk] - crb[kk] for kk in cro}),
        oos=dict(ov=C.period_return(r, *OOS), base=C.period_return(rb, *OOS), ov_stats=st(r, OOS), base_stats=st(rb, OOS)),
        to_latest=dict(ov=st(r, (win[0], LAST)), base=st(rb, (win[0], LAST))),
        episodes=episodes(differ, r, rb, win),
        today_flag_raw=(bool(raw.reindex(idx).fillna(False).loc[LAST]) if raw is not None else None),
        verdict=verdict, reasons=reasons,
    )
    RESULTS.append(rec)
    RET_STORE[(name, base)] = r
    log(f"{base} {name:58s} Cal {s_o['calmar']:.3f} vs {s_b['calmar']:.3f} ({d_cal:+.3f})  Sh {d_sh:+.3f}  "
        f"halves dCal {hv[0]['d_calmar']:+.2f}/{hv[1]['d_calmar']:+.2f}  dial dCAGR {(s_o['cagr']-sd['cagr'])*100:+.2f}% "
        f"(p {bd['cagr']['p_le0']:.2f})  2022 {rec['crises']['diff']['2022 inflation']*100:+.1f}  OOS {(rec['oos']['ov']-rec['oos']['base'])*100:+.1f}  -> {verdict}")
    return rec


for b in BASES:
    # (2) genuine curve rules
    evaluate("curve: inversion caps engine at 0.5", "curve", b,
             dict(extra_scale=cap_scale(FLAG["inv"], 0.5, b)), FLAG["inv"], note="raw key inv")
    evaluate("curve: inversion -> risk-off (July raw rule)", "curve", b, riskoff_kw(FLAG["inv"]), FLAG["inv"])
    evaluate("curve: un-inversion window x0.5 (>=63d spell)", "curve", b, mult_kw(FLAG["uninv_contig"], 0.5), FLAG["uninv_contig"])
    evaluate("curve: un-inversion window -> risk-off", "curve", b, riskoff_kw(FLAG["uninv_contig"]), FLAG["uninv_contig"])
    evaluate("curve: un-inversion x0.5 (alt: >=63 inv days in 252)", "curve", b, mult_kw(FLAG["uninv_cum252"], 0.5), FLAG["uninv_cum252"])
    es_combo = cap_scale(FLAG["inv"], 0.5, b) * pd.Series(np.where(FLAG["uninv_contig"].values, 0.5, 1.0), index=idx)
    evaluate("curve: inversion cap 0.5 + un-inversion x0.5", "curve", b, dict(extra_scale=es_combo),
             FLAG["inv"] | FLAG["uninv_contig"])
    # (3) rate shock
    for bp in (50, 75, 100):
        f = FLAG[f"rs{bp}"]
        evaluate(f"rate shock 10y +{bp}bp/126d: engine x0.5", "rate", b, mult_kw(f, 0.5), f)
        evaluate(f"rate shock 10y +{bp}bp/126d: off-leg IEF->bills", "rate", b, offleg_kw(f), f)
    # (4) dollar shock
    evaluate("dollar shock DXY +5%/126d: engine x0.75", "dollar", b, mult_kw(FLAG["usd5"], 0.75), FLAG["usd5"])
    # (5) credit
    evaluate("credit HYG/IEF < 50d SMA: engine x0.5", "credit", b, mult_kw(FLAG["hy50"], 0.5), FLAG["hy50"], win=WIN_HY)
    evaluate("credit HYG/IEF < 200d SMA: engine x0.5", "credit", b, mult_kw(FLAG["hy200"], 0.5), FLAG["hy200"], win=WIN_HY)
    evaluate("credit LQD/IEF < 50d SMA: engine x0.5", "credit", b, mult_kw(FLAG["lq50"], 0.5), FLAG["lq50"], win=WIN_LQ)
    evaluate("credit VWEHX/IEF < 50d SMA: engine x0.5 (1988+)", "credit", b, mult_kw(FLAG["vw50"], 0.5), FLAG["vw50"], win=WIN_VW)
    evaluate("credit VWEHX/IEF < 200d SMA: engine x0.5 (1988+)", "credit", b, mult_kw(FLAG["vw200"], 0.5), FLAG["vw200"], win=WIN_VW)
    # (6) VIX term structure
    evaluate("VIX term structure: VIX>VIX3M -> engine x0.5", "vix_ts", b, mult_kw(FLAG["vts"], 0.5), FLAG["vts"], win=WIN_VIX)
    evaluate("VIX term structure: 5d-avg VIX/VIX3M>1 -> x0.5", "vix_ts", b, mult_kw(FLAG["vts5"], 0.5), FLAG["vts5"], win=WIN_VIX)

# ================================================================== dollar-shock robustness (diagnostic only)
# The pre-specified dollar rule was the only variant with a bootstrap-significant Sharpe gain.  Pre-declared
# check: is it a plateau or a lucky point?  Neighbours are NOT candidates for selection (picking the best
# neighbour would add selection cost); they are counted as variants anyway.  Plus a circular-shift placebo:
# the same x0.75 overlay driven by the DXY flag rotated by a random offset (keeps on-fraction and episode
# lengths, destroys timing) -> where does the real flag's Sharpe / Calmar gain sit in that null distribution?
DOLLAR_NEIGH = [(126, 0.03, 0.75), (126, 0.04, 0.75), (126, 0.06, 0.75), (126, 0.07, 0.75),
                (63, 0.05, 0.75), (252, 0.05, 0.75), (126, 0.05, 0.50)]
for b in BASES:
    for lb, thr, mlt in DOLLAR_NEIGH:
        raw_n = (P["dxy"] / P["dxy"].shift(lb) - 1) > thr
        f_n = lag1(raw_n)
        evaluate(f"dollar robustness: DXY +{int(round(thr*100))}%/{lb}d -> engine x{mlt}", "dollar_robust", b,
                 mult_kw(f_n, mlt), f_n, raw=raw_n)


def placebo(base, raw_flag, mult, n=200, seed=1):
    rng = np.random.default_rng(seed)
    a = raw_flag.reindex(idx).fillna(False).values
    T = len(a)
    sb = st(BASE_R[base], IS)
    out = []
    for _ in range(n):
        sh = int(rng.integers(252, T - 252))
        f = lag1(pd.Series(np.roll(a, sh), index=idx))
        r, _ = run(base, **mult_kw(f, mult))
        s = st(r, IS)
        out.append((s["sharpe"] - sb["sharpe"], s["calmar"] - sb["calmar"], s["cagr"] - sb["cagr"]))
    COUNT["placebo_runs"] = COUNT.get("placebo_runs", 0) + n
    return np.array(out)


dollar_robust = {}
for b in BASES:
    real = next(r for r in RESULTS if r["base"] == b and r["name"].startswith("dollar shock"))
    pl = placebo(b, RAW["usd5"], 0.75)
    neigh = [r for r in RESULTS if r["base"] == b and r["family"] == "dollar_robust"]
    dollar_robust[b] = dict(
        real=dict(d_sharpe=real["d_sharpe"], d_calmar=real["d_calmar"], d_cagr=real["d_cagr"]),
        placebo=dict(n=len(pl), p_sharpe=float((pl[:, 0] >= real["d_sharpe"]).mean()),
                     p_calmar=float((pl[:, 1] >= real["d_calmar"]).mean()),
                     sharpe_pct=[float(np.percentile(pl[:, 0], q)) for q in (5, 50, 95)],
                     calmar_pct=[float(np.percentile(pl[:, 1], q)) for q in (5, 50, 95)]),
        neighbours=[dict(name=r["name"], d_calmar=r["d_calmar"], d_sharpe=r["d_sharpe"],
                         h1=[r["halves"][0]["d_calmar"], r["halves"][0]["d_sharpe"]],
                         h2=[r["halves"][1]["d_calmar"], r["halves"][1]["d_sharpe"]],
                         p_sharpe=r["boot_vs_base"]["sharpe"]["p_le0"]) for r in neigh],
        neighbours_positive_both_halves=int(sum(all(h["d_calmar"] > 0 and h["d_sharpe"] > 0 for h in r["halves"]) for r in neigh)),
        neighbours_total=len(neigh),
        h2_only=dict(d_sharpe=real["halves"][1]["d_sharpe"], d_calmar=real["halves"][1]["d_calmar"], d_cagr=real["halves"][1]["d_cagr"]),
        dotcom_contribution=real["crises"]["diff"]["dot-com 2000-02"],
    )
    log(f"DOLLAR ROBUSTNESS {b}: real dSh {real['d_sharpe']:+.3f} dCal {real['d_calmar']:+.3f} | placebo p(Sh) "
        f"{dollar_robust[b]['placebo']['p_sharpe']:.3f} p(Cal) {dollar_robust[b]['placebo']['p_calmar']:.3f} | "
        f"neighbours positive in both halves {dollar_robust[b]['neighbours_positive_both_halves']}/{len(neigh)}")

# ================================================================== TODAY
L = LAST
mk = P.index[P["mark"].astype(bool)]
last_mark = mk[mk <= L][-1]


def val(s, d=L):
    v = s.reindex(idx).loc[d]
    return None if pd.isna(v) else float(v)


d_back = idx[idx.get_loc(L) - 126]
last_trig_c = trig_c[trig_c].index[-1] if trig_c.any() else None
last_trig_k = trig_k[trig_k].index[-1] if trig_k.any() else None
today = dict(
    date=str(L.date()), last_rebalance_mark=str(last_mark.date()),
    ndx=val(P["ndx"]), ema100=val(P["ema100"]), above_ema100=bool(P["above"].loc[L]),
    ndx_pct_vs_ema=val(P["ndx"]) / val(P["ema100"]) - 1, vol20=val(P["vol20"]), er20=val(P["er20"]),
    engine_target_F3=float(BASE_EQ["F3"].loc[L]), engine_target_F2=float(BASE_EQ["F2"].loc[L]),
    held_F3={k: round(float(v), 4) for k, v in HELD_BASE["F3"].loc[L].items() if abs(v) > 1e-9},
    held_F2={k: round(float(v), 4) for k, v in HELD_BASE["F2"].loc[L].items() if abs(v) > 1e-9},
    curve=dict(tnx=val(P["tnx"]), irx=val(P["irx"]), curve_10y_3m=val(curve), inverted=bool(inv_raw.loc[L]),
               last_inverted_day=str(inv_raw[inv_raw].index[-1].date()),
               uninv_contig_last_trigger=str(last_trig_c.date()) if last_trig_c is not None else None,
               uninv_contig_window_active=bool(uw_c.loc[L]),
               uninv_contig_window_end=spans(uw_c)[-1][1] if spans(uw_c) else None,
               uninv_cum252_last_trigger=str(last_trig_k.date()) if last_trig_k is not None else None,
               uninv_cum252_window_active=bool(uw_k.loc[L])),
    rate_shock=dict(tnx_now=val(P["tnx"]), tnx_126d_ago=val(P["tnx"], d_back), date_126d_ago=str(d_back.date()),
                    change_bp=val(d126) * 100, flag50=bool(RAW["rs50"].loc[L]), flag75=bool(RAW["rs75"].loc[L]),
                    flag100=bool(RAW["rs100"].loc[L]),
                    max_change_bp_since_freeze=float(d126[d126.index > FREEZE].max() * 100),
                    first_day_over_50bp_since_freeze=(str(d126[(d126.index > FREEZE) & (d126 > 0.5)].index[0].date())
                                                      if ((d126.index > FREEZE) & (d126 > 0.5)).any() else None)),
    dollar=dict(dxy_now=val(P["dxy"]), dxy_126d_ago=val(P["dxy"], d_back), change_pct=val(dxy126) * 100,
                flag=bool(RAW["usd5"].loc[L])),
    credit=dict(hyg_ief_vs_50d_pct=(val(r_hy) / val(r_hy.rolling(50).mean()) - 1) * 100, flag_hy50=bool(RAW["hy50"].loc[L]),
                hyg_ief_vs_200d_pct=(val(r_hy) / val(r_hy.rolling(200).mean()) - 1) * 100, flag_hy200=bool(RAW["hy200"].loc[L]),
                lqd_ief_vs_50d_pct=(val(r_lq) / val(r_lq.rolling(50).mean()) - 1) * 100, flag_lq50=bool(RAW["lq50"].loc[L]),
                vwehx_ief_vs_50d_pct=(val(r_vw) / val(r_vw.rolling(50).mean()) - 1) * 100, flag_vw50=bool(RAW["vw50"].loc[L]),
                vwehx_ief_vs_200d_pct=(val(r_vw) / val(r_vw.rolling(200).mean()) - 1) * 100, flag_vw200=bool(RAW["vw200"].loc[L])),
    vix_ts=dict(vix=val(vix_y), vix3m=val(vix3m), ratio=val(vts), ratio_5d=val(vts.rolling(5).mean()),
                backwardation=bool(RAW["vts"].loc[L]), flag_5d=bool(RAW["vts5"].loc[L]),
                backwardation_days_last_20=int(RAW["vts"].iloc[-20:].sum())),
)
# recent history of each raw flag since the freeze (for the OOS narrative)
since = {}
for k, s in RAW.items():
    s2 = s.reindex(idx).fillna(False).astype(bool)
    oo = s2[s2.index > FREEZE]
    since[k] = dict(days_on_since_freeze=int(oo.sum()), n_days=int(len(oo)),
                    first_on=str(oo[oo].index[0].date()) if oo.any() else None,
                    last_on=str(oo[oo].index[-1].date()) if oo.any() else None)
today["flags_since_freeze"] = since
_rs_on = [bp for bp in (50, 75, 100) if today["rate_shock"][f"flag{bp}"]]
if not today["above_ema100"]:
    today["implication"] = ("Trend gate is OFF (NDX below EMA-100): engine weight is 0, so every engine-multiplier overlay "
                            "(curve, rate-shock x0.5, dollar, credit, VIX) changes NOTHING in the current portfolio. "
                            + (f"The rate-shock off-leg rule IS active at {_rs_on} bp thresholds and would move the off-leg "
                               "from IEF to bills today." if _rs_on else
                               "The rate-shock off-leg rule is not active at any threshold, so it would not change holdings either."))
else:
    today["implication"] = "Trend gate is ON: engine-multiplier overlays whose flag is on would scale the leveraged sleeve."
for rec in RESULTS:
    key = {"curve: inversion caps engine at 0.5": "inv", "curve: inversion -> risk-off (July raw rule)": "inv",
           "curve: un-inversion window x0.5 (>=63d spell)": "uninv_contig", "curve: un-inversion window -> risk-off": "uninv_contig",
           "curve: un-inversion x0.5 (alt: >=63 inv days in 252)": "uninv_cum252",
           "dollar shock DXY +5%/126d: engine x0.75": "usd5",
           "credit HYG/IEF < 50d SMA: engine x0.5": "hy50", "credit HYG/IEF < 200d SMA: engine x0.5": "hy200",
           "credit LQD/IEF < 50d SMA: engine x0.5": "lq50", "credit VWEHX/IEF < 50d SMA: engine x0.5 (1988+)": "vw50",
           "credit VWEHX/IEF < 200d SMA: engine x0.5 (1988+)": "vw200",
           "VIX term structure: VIX>VIX3M -> engine x0.5": "vts", "VIX term structure: 5d-avg VIX/VIX3M>1 -> x0.5": "vts5"}.get(rec["name"])
    if key is None and rec["name"].startswith("rate shock"):
        key = "rs" + rec["name"].split("+")[1].split("bp")[0]
    if rec["today_flag_raw"] is not None:
        continue
    if key is None and rec["name"].startswith("curve: inversion cap 0.5 + un"):
        rec["today_flag_raw"] = bool(RAW["inv"].loc[L] or RAW["uninv_contig"].loc[L])
    elif key is not None:
        rec["today_flag_raw"] = bool(RAW[key].reindex(idx).fillna(False).loc[L])
log(f"TODAY {today['date']}: above={today['above_ema100']} ({today['ndx_pct_vs_ema']*100:+.1f}% vs EMA)  curve {today['curve']['curve_10y_3m']:+.2f}  "
    f"10y d126 {today['rate_shock']['change_bp']:+.0f}bp  DXY d126 {today['dollar']['change_pct']:+.1f}%  "
    f"HYG/IEF vs50 {today['credit']['hyg_ief_vs_50d_pct']:+.2f}%  VIX/VIX3M {today['vix_ts']['ratio']:.3f}")

# ================================================================== summary
summary_rows = []
for rec in RESULTS:
    summary_rows.append(dict(name=rec["name"], base=rec["base"], family=rec["family"], window=rec["window"],
                             cagr=rec["ov"]["cagr"], maxdd=rec["ov"]["maxdd"], calmar=rec["ov"]["calmar"], sharpe=rec["ov"]["sharpe"],
                             base_calmar=rec["base_stats"]["calmar"], d_calmar=rec["d_calmar"], d_sharpe=rec["d_sharpe"],
                             d_cagr=rec["d_cagr"], d_maxdd=rec["d_maxdd"],
                             h1_dcal=rec["halves"][0]["d_calmar"], h2_dcal=rec["halves"][1]["d_calmar"],
                             p_sharpe_vs_base=rec["boot_vs_base"]["sharpe"]["p_le0"],
                             d_cagr_vs_dial=rec["dial"]["d_cagr_vs_volmatched_dial"], p_vs_dial=rec["dial"]["boot_vs_dial"]["cagr"]["p_le0"],
                             d2022=rec["crises"]["diff"]["2022 inflation"], d_oos=rec["oos"]["ov"] - rec["oos"]["base"],
                             today_flag=rec["today_flag_raw"], verdict=rec["verdict"]))
verdicts = pd.Series([r["verdict"] for r in RESULTS]).value_counts().to_dict()
best = max(RESULTS, key=lambda r: r["d_calmar"])
summary = dict(counts=COUNT, verdict_counts=verdicts,
               best_by_calmar_gain=dict(name=best["name"], base=best["base"], d_calmar=best["d_calmar"], verdict=best["verdict"]),
               n_candidate_variants=COUNT["candidate_variants"],
               rows=summary_rows)

out = dict(
    meta=dict(created=time.strftime("%Y-%m-%d %H:%M"), panel_end=str(LAST.date()), freeze=str(FREEZE.date()),
              in_sample=[str(IS[0].date()), str(IS[1].date())], halves=[[str(H1[0].date()), str(H1[1].date())], [str(H2[0].date()), str(H2[1].date())]],
              oos=[str(OOS[0].date()), str(OOS[1].date())], macro_lag="1 extra trading day on every macro input; weights lag 1 (common.backtest)",
              haircut=[HAIRCUT_LO, HAIRCUT_HI], n_boot=N_BOOT, block=63,
              decision_rule="INCLUDE iff Calmar&Sharpe up in both halves AND IS Calmar gain > 0.15 AND (p(Sharpe diff<=0)<0.05 or p(CAGR diff vs vol-matched dial<=0)<0.05); MARGINAL iff both halves and gain>0.07",
              external_series=dict(source=ext_src, coverage=ext_cov,
                                   fred="FRED (DBAA/DAAA/BAMLH0A0HYM2) unreachable from this machine (connection reset / timeout) -> VWEHX/IEF used as the long-history credit proxy"),
              bootstrap_check=dict(mine=_chk_mine, common_py=_chk_c)),
    baselines=baselines, noop=noop, curve_info=curve_info, variants=RESULTS, dollar_robustness=dollar_robust,
    today=today, summary=summary)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


with open(OUT_JSON, "w") as fh:
    json.dump(jsonable(out), fh, indent=1)

print("\n================ SUMMARY (in-sample 1988-01-01..2026-07-24 unless window says otherwise) ================")
print(f"{'base':4s} {'variant':58s} {'window':9s} {'CAGR':>6s} {'maxDD':>6s} {'Cal':>5s} {'dCal':>6s} {'dSh':>6s} {'H1dCal':>6s} {'H2dCal':>6s} "
      f"{'pSh':>4s} {'dDial':>6s} {'pDial':>5s} {'d2022':>6s} {'dOOS':>5s} {'today':>5s} verdict")
for s in summary_rows:
    print(f"{s['base']:4s} {s['name']:58s} {s['window'][0][:4]}+     {s['cagr']*100:5.1f}% {s['maxdd']*100:5.1f}% {s['calmar']:5.2f} {s['d_calmar']:+6.3f} "
          f"{s['d_sharpe']:+6.3f} {s['h1_dcal']:+6.2f} {s['h2_dcal']:+6.2f} {s['p_sharpe_vs_base']:4.2f} {s['d_cagr_vs_dial']*100:+5.2f}% "
          f"{s['p_vs_dial']:5.2f} {s['d2022']*100:+5.1f}% {s['d_oos']*100:+4.1f}% {str(s['today_flag']):>5s} {s['verdict']}")
print(f"\nverdicts: {verdicts}   counts: {COUNT}")
print(f"saved {OUT_JSON}   total {time.time()-T0:.0f}s")
