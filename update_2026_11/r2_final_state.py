"""
r2 FINAL CRITIC - quick check of today's state and the exact re-entry trigger.
Read-only on project data; writes JSON only to the path given as argv[1] (scratchpad).
Uses the panel's canonical signal definitions (panel.py / common.py):
  ema100 = ewm(span=100, adjust=False); vol20 = std(pct_change, 20, ddof=1)*sqrt(252);
  er20 = |C_t - C_t-20| / sum_20 |dC|;   risk-on iff C_t > EMA_t  <=>  C_t > EMA_{t-1}.
No parameter is tuned; this only evaluates the frozen July rules on hypothetical next closes.
"""
import sys, os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

OUT = sys.argv[1] if len(sys.argv) > 1 else None
P = C.panel()
ndx = P["ndx"].dropna()
last = ndx.index[-1]
A = 2.0 / 101.0
TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}

res = {"panel_last_day": str(last.date()), "ndx_last": float(ndx.iloc[-1]),
       "ema100_last": float(P["ema100"].iloc[-1]), "vol20_last": float(P["vol20"].iloc[-1]),
       "er20_last": float(P["er20"].iloc[-1]), "tnx_last": float(P["tnx"].ffill().iloc[-1])}

# current held weights (Friday book, lag 1) for each tier
hold = {}
for t, (lev, vt) in TIERS.items():
    w = C.strategy_weights(P, lev=lev, voltgt=vt)
    hw = C.held_weights(P, w)
    row = hw.iloc[-1]
    hold[t] = {k: round(float(v), 4) for k, v in row.items() if abs(v) > 1e-9}
res["held_weights_today"] = hold
mk = P.index[P["mark"]]
res["last_marks"] = [str(d.date()) for d in mk[-6:]]
res["above_at_last_marks"] = {str(d.date()): bool(P.loc[d, "above"]) for d in mk[-8:]}
res["ndx_at_last_marks"] = {str(d.date()): round(float(P.loc[d, "ndx"]), 2) for d in mk[-8:]}
res["ema_at_last_marks"] = {str(d.date()): round(float(P.loc[d, "ema100"]), 2) for d in mk[-8:]}

# future NYSE sessions (2026-11-18 .. 2027-03-05), NYSE holidays in the window
hol = pd.to_datetime(["2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15"])
fut = pd.bdate_range("2026-11-18", "2027-03-05")
fut = fut[~fut.isin(hol)]


def extend(path_closes):
    """append hypothetical closes and recompute signals exactly as panel.py"""
    s = pd.concat([ndx, pd.Series(path_closes, index=fut[:len(path_closes)])])
    ema = s.ewm(span=100, adjust=False).mean()
    vol = s.pct_change().rolling(20).std() * np.sqrt(252)
    er = s.diff(20).abs() / s.diff().abs().rolling(20).sum()
    return s, ema, vol, er


def targets(vol, er, month, above):
    out = {}
    for t, (lev, vt) in TIERS.items():
        if not above:
            out[t] = {lev: 0.0, "ief": 0.85, "gold": 0.15}
            continue
        sc = min(1.0, vt / vol)
        if month in (8, 9):
            sc *= 0.5
        eng = sc if er >= 0.25 else 0.5 * sc
        lw = 0.85 * eng
        beta = lw * {"qqq": 1, "qld": 2, "tqqq": 3}[lev]
        out[t] = {lev: round(lw, 4), "ief": round(0.85 - lw, 4), "gold": 0.15, "ndx_beta": round(beta, 3),
                  "chop_halved": bool(er < 0.25)}
    return out


# (1) exact trigger for Fri 2026-11-20: risk-on iff C_Fri > EMA_Thu
flat = [float(ndx.iloc[-1])] * 2
s, ema, vol, er = extend(flat)
ema_thu = float(ema.loc["2026-11-19"])
res["trigger_2026_11_20_if_wed_thu_flat"] = round(ema_thu, 2)
res["trigger_pct_above_last"] = round((ema_thu / float(ndx.iloc[-1]) - 1) * 100, 3)
e_tue = float(P["ema100"].iloc[-1]); c0 = float(ndx.iloc[-1])
res["trigger_formula"] = (f"EMA_Thu = {e_tue:.2f}*(1-a)^2 + a*(1-a)*C_Wed + a*C_Thu, a=2/101 "
                          f"= {ema_thu:.1f} + {A*(1-A):.4f}*(C_Wed-{c0:.1f}) + {A:.4f}*(C_Thu-{c0:.1f})")
res["c_t_minus_20_for_2026_11_20"] = {"date": str(ndx.index[-18].date()), "close": round(float(ndx.iloc[-18]), 2)}  # 11-20 is 3 sessions after the last panel day
res["c_t_minus_20_for_2026_11_27"] = {"date": str(ndx.index[-14].date()), "close": round(float(ndx.iloc[-14]), 2)}

# (2) sizing under hypothetical crossing paths (just above trigger, and a bigger rally)
paths = {}
def run(name, closes, at="2026-11-20"):
    s, ema, vol, er = extend(closes)
    d = pd.Timestamp(at)
    ab = bool(s.loc[d] > ema.loc[d])
    paths[name] = {"ndx_fri": round(float(s.loc[d]), 1), "ema_fri": round(float(ema.loc[d]), 1),
                   "ema_thu": round(float(ema.shift(1).loc[d]), 1),
                   "risk_on": ab, "vol20": round(float(vol.loc[d]), 4), "er20": round(float(er.loc[d]), 4),
                   "targets": targets(float(vol.loc[d]), float(er.loc[d]), d.month, ab)}
thr = ema_thu
run("flat Wed/Thu, Fri jump to trigger+0.1%", [c0, c0, thr * 1.001])
g = (thr * 1.001 / c0) ** (1 / 3)
run("3-day rally to trigger+0.1% (bar moves up)", [c0 * g, c0 * g * g, thr * 1.001])
g2 = (28960.0 / c0) ** (1 / 3)
run("3-day rally to 28,960 (+7.25%)", [c0 * g2, c0 * g2 * g2, 28960.0])
run("3-day rally to 29,700 (+10%)", [c0 * (29700 / c0) ** (1 / 3), c0 * (29700 / c0) ** (2 / 3), 29700.0])
run("Fri close just below trigger (-0.1%)", [c0, c0, thr * 0.999])
res["crossing_paths_2026_11_20"] = paths

# (3) threshold ladder if NDX stays flat at today's close: P.mark (holiday-Friday weeks skipped)
#     and the live-faithful last-session-of-week convention (mark_wk)
flat_all = [c0] * len(fut)
s, ema, vol, er = extend(flat_all)
lad = {}
for d in fut:
    wk_end = d + pd.offsets.Week(weekday=4) if d.weekday() != 4 else d
    is_fri = d.weekday() == 4
    nxt = fut[fut > d]
    last_of_week = (len(nxt) == 0) or (nxt[0] > wk_end)
    if is_fri or last_of_week:
        lad[str(d.date())] = {"weekday": d.strftime("%a"), "threshold_ema_prev": round(float(ema.shift(1).loc[d]), 1),
                              "pct_above_27002": round((float(ema.shift(1).loc[d]) / c0 - 1) * 100, 2),
                              "is_friday_mark": bool(is_fri), "is_last_session_of_week": bool(last_of_week)}
    if d > pd.Timestamp("2027-02-26"):
        break
res["flat_path_threshold_ladder"] = lad

# (4) daily decline rate of the bar under a flat path
ks = list(lad.keys())
res["bar_decline_pct_of_c0_per_session"] = {
    "today_to_11_20": round((float(P["ema100"].iloc[-1]) - lad[ks[0]]["threshold_ema_prev"]) / c0 / 2 * 100, 3),
    "11_20_to_11_27 (4 sessions)": round((lad[ks[0]]["threshold_ema_prev"] - lad[ks[1]]["threshold_ema_prev"]) / c0 / 4 * 100, 3),
    "12_11_to_12_18": round((lad["2026-12-11"]["threshold_ema_prev"] - lad["2026-12-18"]["threshold_ema_prev"]) / c0 / 5 * 100, 3),
    "02_12_to_02_19 (4 sessions)": round((lad["2027-02-12"]["threshold_ema_prev"] - lad["2027-02-19"]["threshold_ema_prev"]) / c0 / 4 * 100, 3)}

txt = json.dumps(res, indent=1, default=str)
print(txt)
if OUT:
    with open(OUT, "w") as f:
        f.write(txt)
