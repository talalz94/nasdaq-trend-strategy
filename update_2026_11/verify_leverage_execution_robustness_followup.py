"""
ROBUSTNESS REVIEW of update_2026_11/leverage_execution.py  --  step 3: follow-up checks.

  F1 is the NDX weekday dispersion unusual for a trend-gated index book in general?  For NDX, S&P 500,
     Russell 2000 and TSX (same F2 machinery: EMA-100 gate, vt .30, chop halving, Aug/Sep halving, synthetic
     2x sleeve, IEF off-leg, 15% gold) compare the dispersion of the 5 weekday phases with a timing-luck null
     of 200 random schedules (one random trading day per week); also on/off switch counts by weekday.
  F2 direction of the regime switches skipped by the pure 10pp band (F3), using the specialist's own
     drift_loop source (extracted verbatim from leverage_execution.py, not re-implemented).
usage: python update_2026_11/verify_leverage_execution_robustness_followup.py <scratch_dir>
"""
import os, sys, json, ast, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
scratch = sys.argv[1]
OUTF = os.path.join(scratch, "verify_leverage_execution_robustness_followup.json")
P = C.panel()
IDX = P.index
T = len(IDX)
START, FREEZE = C.START, C.FREEZE
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DOW = IDX.dayofweek.values
MK = {d: pd.Series(DOW == i, index=IDX) for i, d in enumerate(DAYS)}
OUT = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def cg(r):
    return C.stats(r, None, FREEZE)["cagr"] * 100


def market_panel(ret_col):
    PP = P.copy()
    lvl = (1 + PP[ret_col].fillna(0)).cumprod()
    PP["above"] = lvl > lvl.ewm(span=100, adjust=False).mean()
    PP["vol20"] = lvl.pct_change().rolling(20).std() * np.sqrt(252)
    PP["er20"] = lvl.diff(20).abs() / lvl.diff().abs().rolling(20).sum().replace(0, np.nan)
    PP[f"{ret_col}_2x"] = 2 * PP[ret_col].fillna(0) - 0.0095 / 252 - PP["rf_d"]
    return PP


wk_code, _ = pd.factorize(IDX.to_period("W-FRI"))
starts = np.r_[0, np.where(np.diff(wk_code) != 0)[0] + 1]
lens = np.diff(np.r_[starts, T])
nwk = len(starts)
rng = np.random.default_rng(777)
NS = 200
marks = []
for _ in range(NS):
    pick = starts + np.floor(rng.random(nwk) * lens).astype(int)
    m = np.zeros(T, bool); m[pick] = True
    marks.append(pd.Series(m, index=IDX))


def nswitch(PP, w, lev, mark):
    H = C.held_weights(PP, w, mark=mark)
    sw = (H[lev] > 1e-12).astype(int).diff().abs()
    return int(sw[(sw.index >= START) & (sw.index <= FREEZE)].sum())


F1 = {}
for mkt in ("ndx_ret", "spx", "rut", "tsx"):
    PP = market_panel(mkt)
    lev = f"{mkt}_2x"
    w = C.strategy_weights(PP, lev=lev, voltgt=0.30, gold="gold_phys", gold_w=0.15, off="ief")
    ph = {d: cg(C.backtest(PP, w, mark=MK[d])) for d in DAYS}
    sw = {d: nswitch(PP, w, lev, MK[d]) for d in DAYS}
    null = np.array([cg(C.backtest(PP, w, mark=m)) for m in marks])
    nsw = np.array([nswitch(PP, w, lev, m) for m in marks])
    v = np.array(list(ph.values()))
    F1[mkt] = dict(weekday_IS_cagr={d: round(x, 2) for d, x in ph.items()},
                   weekday_sd=round(float(v.std(ddof=1)), 2), weekday_spread=round(float(v.max() - v.min()), 2),
                   null_mean=round(float(null.mean()), 2), null_sd=round(float(null.std(ddof=1)), 2),
                   ratio_weekday_sd_to_null_sd=round(float(v.std(ddof=1) / null.std(ddof=1)), 2),
                   z_by_day={d: round(float((x - null.mean()) / null.std(ddof=1)), 2) for d, x in ph.items()},
                   switches_by_day=sw, null_switch_mean=round(float(nsw.mean()), 1),
                   null_switch_sd=round(float(nsw.std(ddof=1)), 1),
                   z_switch_fri=round(float((sw["Fri"] - nsw.mean()) / nsw.std(ddof=1)), 2))
    log("F1", mkt, json.dumps(F1[mkt]))
OUT["F1_weekday_dispersion_vs_null_by_market"] = F1

# ---------------------------------------------------------------- F2 band-skipped switches (verbatim drift_loop)
src = open(os.path.join(HERE, "leverage_execution.py"), encoding="utf-8").read()
tree = ast.parse(src)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "drift_loop"][0]
code = ast.get_source_segment(src, fn)
ns = {"np": np, "pd": pd, "P": P, "IDX": IDX, "C": C}
exec(code, ns)
drift_loop = ns["drift_loop"]
w3 = C.strategy_weights(P, lev="tqqq", voltgt=0.20, gold="gold_phys", gold_w=0.15, off="ief")
o = drift_loop(w3, "tqqq", band=0.10, cost=10 / 1e4, override=False)
sk = o["skip_regime"]
sk = sk[sk & (sk.index >= START) & (sk.index <= FREEZE)]
rows = []
for t in sk.index:
    i = IDX.get_loc(t)
    held_before = float(o["held_lev"].iloc[i - 1])
    tgt_now = float(w3["tqqq"].loc[t]) if P["mark"].loc[t] else None
    rows.append(dict(date=str(t.date()), held_tqqq_before=round(held_before, 4), new_target=tgt_now,
                     direction=("risk-off skipped (held>0, target 0)" if (tgt_now is not None and tgt_now <= 1e-12)
                                else "risk-on skipped (held 0, target>0)"),
                     ndx_vs_ema=round(float(P["ndx"].loc[t] / P["ema100"].loc[t] - 1) * 100, 2)))
OUT["F2_band10_skipped_switches_F3"] = rows
log("F2", json.dumps(rows))
OUT["runtime_sec"] = time.time() - T0
json.dump(OUT, open(OUTF, "w"), indent=1)
log("saved", OUTF)
