"""
LIVE WEEKLY SCANNER v2 (Nov 2026) - research tool, NOT investment advice.

Implements EXACTLY the rules backtested in update_2026_11/common.py::strategy_weights
(run `python scanner.py --check` to prove it on the full 1985-2026 history):

  risk-on  : NDX close > EMA-100            (pandas ewm, span=100, adjust=False)
  scale    : min(1, VOL_TARGET / vol20)     (vol20 = stdev of last 20 daily returns x sqrt(252))
  seasonal : scale x 0.5 when the decision date is in August or September
  chop     : engine = scale if ER20 >= 0.25 else 0.5 x scale ; engine = 0 when risk-off
             (ER20 = |close - close 20 days ago| / sum of the last 20 absolute daily changes)
  weights  : leveraged ETF = 0.85 x engine ; IEF = 0.85 - that ; GLD = 0.15 always

Tiers (identical rules, different sleeve; they tie on risk-adjusted return - pick by tolerable drawdown):
  F2  QLD  (2x)  vol target 0.30   <- reference tier
  F1  QQQ  (1x)  vol target 0.20   <- lower risk; best in every stress scenario
  F3  TQQQ (3x)  vol target 0.20   <- higher risk; US-listed only (no UCITS 3x fund)

Planning numbers (Nov-2026 re-validation; average of all five rebalance weekdays, 5 bps costs,
in-sample to 2026-07-24; base case = since-2000 value, upside = 1988/2007 starts):
  F1  CAGR  9.1-10.6%   worst drawdown -21% (1-in-20: -24%)
  F2  CAGR 13.1-17.0%   worst drawdown -28% (1-in-20: -37% to -40%)
  F3  CAGR 16.7-21.6%   worst drawdown -33% (1-in-20: -46% to -51%)
  UCITS route about 1 point a year lower (possibly 2-3). The edge over QQQ is drawdown avoidance,
  not extra return: since 2010 F2 made 15.8%/yr vs QQQ's 18.8%.

Changes from the July scanner (kept as scanner_v1_jul2026.py):
  * IEF replaces TLT. The July backtest's bond was a synthetic 10-year (IEF-like); TLT (20y+)
    turns the 3x strategy into a -40.7% max-drawdown strategy (Calmar 0.60 vs 0.80).
  * The 'confirmed yield-curve' clause is deleted: above AND NOT (inverted AND NOT above) == above,
    so it never changed a position. Macro readings are printed as context only.
  * F2 is primary; every run can append its decision to scanner_log.csv (--log).

Execution (from the Nov-2026 re-validation):
  * Run on Friday about 15:40-15:45 ET and send market-on-close orders. If Friday is a market
    holiday, run on the last trading day of that week. A Monday-close fallback costs ~0 since 2000.
  * No-trade band: skip a sleeve whose weight is within 2 percentage points of target, but ALWAYS
    execute a risk-on/off switch.
  * Planning numbers must not assume the in-sample 'Friday premium' (see RESEARCH_LOG.md).

Run:  python scanner.py            today's allocation
      python scanner.py --json     also write scanner_state.json
      python scanner.py --log      also append this decision to scanner_log.csv
      python scanner.py --check    offline parity test vs the backtest engine (needs update_2026_11/)
"""
import os, sys, json, csv
import numpy as np
import pandas as pd

EMA_N, VOL_N, ER_N = 100, 20, 20
CHOP_THR, CHOP_POS = 0.25, 0.5
USE_CHOP = True                    # pre-registered chop-rule test result goes here (round 2)
GOLD_W, CORE = 0.15, 0.85
SEASON_MONTHS, SEASON_MULT = (8, 9), 0.5
TIERS = {   # name: (sleeve ticker, vol target, label) - leverage is a risk dial, not an efficiency choice
    "F2": ("QLD", 0.30, "reference tier - 2x Nasdaq-100"),
    "F1": ("QQQ", 0.20, "lower risk - 1x Nasdaq-100, held up best in stress tests"),
    "F3": ("TQQQ", 0.20, "higher risk - 3x Nasdaq-100, US-listed only"),
}
OFF, GOLD = "IEF", "GLD"
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ signals (single source of truth)
def signals(ndx: pd.Series) -> pd.DataFrame:
    """Daily signal frame from a series of NDX closes (index = dates)."""
    s = pd.DataFrame(index=ndx.index)
    s["ndx"] = ndx
    s["ema100"] = ndx.ewm(span=EMA_N, adjust=False).mean()
    s["above"] = ndx > s["ema100"]
    s["vol20"] = ndx.pct_change().rolling(VOL_N).std() * np.sqrt(252)
    s["er20"] = ndx.diff(ER_N).abs() / ndx.diff().abs().rolling(ER_N).sum().replace(0, np.nan)
    return s


def engine(row, voltgt, month):
    """Leveraged-sleeve engine weight (before the 0.85 core) for one decision row."""
    if not bool(row["above"]):
        return 0.0
    v = row["vol20"]
    scale = 1.0 if (pd.isna(v) or v <= 0) else min(1.0, voltgt / v)
    if month in SEASON_MONTHS:
        scale = min(1.0, scale * SEASON_MULT)
    if USE_CHOP:
        er = row["er20"]
        clean = (not pd.isna(er)) and er >= CHOP_THR
        return scale if clean else scale * CHOP_POS
    return scale


def tier_weights(row, month, tier):
    sleeve, vt, _ = TIERS[tier]
    e = engine(row, vt, month)
    return {sleeve: CORE * e, OFF: CORE - CORE * e, GOLD: GOLD_W}


def reentry_projection(ndx: pd.Series, decision_date: pd.Timestamp, close_at_decision: float):
    """Weights each tier would take if NDX stays flat until the decision day and closes there at
    `close_at_decision` (e.g. just above the re-entry bar). Uses the same signal code."""
    days = pd.bdate_range(ndx.index[-1] + pd.Timedelta(days=1), decision_date)
    if len(days) == 0:
        return None
    path = pd.Series([float(ndx.iloc[-1])] * (len(days) - 1) + [close_at_decision], index=days)
    sig = signals(pd.concat([ndx, path]))
    row = sig.iloc[-1]
    return dict(vol20=float(row["vol20"]), er20=float(row["er20"]), above=bool(row["above"]),
                tiers={t: tier_weights(row, decision_date.month, t) for t in TIERS})


def reentry_threshold(s: pd.Series, decision_date: pd.Timestamp) -> float:
    """NDX close needed at the next decision for risk-on, assuming NDX stays at today's close
    until the day before (risk-on at a close C requires C > EMA of the previous day)."""
    ema, last = float(s["ema100"].iloc[-1]), float(s["ndx"].iloc[-1])
    alpha = 2 / (EMA_N + 1)
    days = pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), decision_date)
    for _ in range(max(len(days) - 1, 0)):
        ema = alpha * last + (1 - alpha) * ema
    return ema


# ------------------------------------------------------------------ data
def fetch(ticker, period="5y"):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period=period, auto_adjust=False)["Close"].dropna()
    h.index = h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()]


def next_decision_day(asof: pd.Timestamp) -> pd.Timestamp:
    """The Friday whose close is the NEXT decision after the data in hand (holidays ignored)."""
    if asof.dayofweek == 4:
        return asof + pd.Timedelta(days=7)
    return asof + pd.Timedelta(days=(4 - asof.dayofweek) % 7)


# ------------------------------------------------------------------ parity test (offline)
def check():
    sys.path.insert(0, os.path.join(HERE, "update_2026_11"))
    import common as C
    if not os.path.exists(C.PANEL_PATH):
        print(f"--check needs the research panel {C.PANEL_PATH}: build it with\n"
              f"  python letf_data.py; python macro_data.py; python multi_panel.py; python geo_panel.py; "
              f"python update_2026_11/panel.py\n(see README, Data). Nothing was checked.")
        return "no-panel"
    P = C.panel()
    s = signals(P["ndx"])
    sleeve_col = {"QLD": "qld", "QQQ": "qqq", "TQQQ": "tqqq"}
    worst = 0.0
    for tier, (sleeve, vt, _) in TIERS.items():
        ref = C.strategy_weights(P, lev=sleeve_col[sleeve], voltgt=vt, off="ief", gold="gold_phys",
                                 chop_pos=(CHOP_POS if USE_CHOP else 1.0))
        days = P.index[P["mark"]]
        diffs = []
        for d in days:
            w = tier_weights(s.loc[d], d.month, tier)
            diffs.append(max(abs(w[sleeve] - ref.loc[d, sleeve_col[sleeve]]),
                             abs(w[OFF] - ref.loc[d, "ief"]), abs(w[GOLD] - ref.loc[d, "gold_phys"])))
        m = float(np.nanmax(diffs)); worst = max(worst, m)
        print(f"  {tier}: {len(days)} weekly decisions 1985-{days[-1].date()}  max |weight diff| = {m:.2e}")
    ok = worst < 1e-9
    print(f"  PARITY {'PASSED' if ok else 'FAILED'}: scanner rules {'==' if ok else '!='} backtest engine")
    return ok


# ------------------------------------------------------------------ main
def main(as_json=False, log=False):
    ndx = fetch("^NDX")
    s = signals(ndx)
    row = s.iloc[-1]
    asof = s.index[-1]
    now_et = pd.Timestamp.now(tz="America/New_York")
    partial = (asof.date() == now_et.date()) and (now_et.hour < 16)
    ctx = {}
    for t, k in [("^TNX", "us10y_pct"), ("^IRX", "us3m_pct"), ("DX-Y.NYB", "dxy"), ("^VIX", "vix"), ("GLD", "gld")]:
        try:
            x = fetch(t, "1y")
            ctx[k] = float(x.iloc[-1]); ctx[k + "_6m_chg"] = float(x.iloc[-1] - x.iloc[-min(len(x), 127)])
        except Exception:
            ctx[k] = None
    if ctx.get("us10y_pct") is not None and ctx.get("us3m_pct") is not None:
        ctx["curve_10y_3m_pct"] = ctx["us10y_pct"] - ctx["us3m_pct"]
    ath = float(ndx.cummax().iloc[-1])
    below_streak = int((~s["above"])[::-1].cumprod().sum())
    W = {t: tier_weights(row, asof.month, t) for t in TIERS}
    nxt = next_decision_day(asof)
    thr = reentry_threshold(s, nxt) if not bool(row["above"]) else None
    proj = reentry_projection(ndx, nxt, thr * 1.001) if thr else None

    state = dict(asof=str(asof.date()), bar_is_partial=partial, ndx=round(float(row["ndx"]), 2),
                 ema100=round(float(row["ema100"]), 2), above_ema=bool(row["above"]),
                 pct_from_ema=round(float(row["ndx"] / row["ema100"] - 1), 4),
                 pct_from_record=round(float(row["ndx"] / ath - 1), 4), days_below_ema=below_streak,
                 realized_vol20=round(float(row["vol20"]), 4), efficiency_ratio20=round(float(row["er20"]), 4),
                 trend="CLEAN" if row["er20"] >= CHOP_THR else "CHOPPY", chop_rule=USE_CHOP,
                 seasonal_half=asof.month in SEASON_MONTHS,
                 reentry_close_needed_next_friday=(round(thr, 0) if thr else None), next_friday=str(nxt.date()),
                 reentry_if_triggered=({"vol20": round(proj["vol20"], 4), "er20": round(proj["er20"], 4),
                                        "tiers": {t: {k: round(v, 4) for k, v in w.items()} for t, w in proj["tiers"].items()}}
                                       if proj else None),
                 context_only={k: (round(v, 3) if isinstance(v, float) else v) for k, v in ctx.items()},
                 tiers={t: {k: round(v, 4) for k, v in W[t].items()} for t in TIERS})

    if as_json:
        json.dump(state, open(os.path.join(HERE, "scanner_state.json"), "w"), indent=1)
    if log:
        path = os.path.join(HERE, "scanner_log.csv")
        rows = []
        if os.path.exists(path):
            rows = [r for r in csv.DictReader(open(path, newline="")) if r["asof"] != state["asof"]]
        rec = {"asof": state["asof"], "bar_is_partial": partial, "ndx": state["ndx"], "ema100": state["ema100"],
               "above": state["above_ema"], "vol20": state["realized_vol20"], "er20": state["efficiency_ratio20"],
               **{f"{t}_{k}": v for t in TIERS for k, v in state["tiers"][t].items()}}
        rows.append(rec)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rec.keys())); wr.writeheader(); wr.writerows(rows)

    bar = "=" * 72
    print(f"\n{bar}\n WEEKLY SCANNER v2  -  data to {asof.date()}{'  (PARTIAL BAR: intraday price)' if partial else ''}"
          f"\n research tool, not investment advice\n{bar}")
    print(f" Nasdaq-100  {row['ndx']:>10,.0f}   EMA-100 {row['ema100']:>10,.0f}   "
          f"{'ABOVE -> risk-on' if row['above'] else 'BELOW -> risk-off'}  ({state['pct_from_ema']*100:+.1f}%)")
    print(f" from record {state['pct_from_record']*100:>+9.1f}%   below EMA for {below_streak} trading days")
    print(f" vol (20d)   {row['vol20']*100:>9.1f}%   efficiency {row['er20']:.2f} -> "
          f"{'CLEAN (full size)' if row['er20'] >= CHOP_THR else 'CHOPPY (half size)'}")
    print(f" seasonality {'Aug/Sep: half size' if state['seasonal_half'] else 'normal month'}")
    if thr:
        print(f" RE-ENTRY: risk-on at the {nxt.date()} close needs NDX above ~{thr:,.0f} "
              f"({(thr/row['ndx']-1)*100:+.1f}% from now, if NDX is flat until then)")
        if proj:
            sz = "  ".join(f"{t} {TIERS[t][0]} {proj['tiers'][t][TIERS[t][0]]*100:.0f}%" for t in TIERS)
            print(f"   if it closes just above: vol {proj['vol20']*100:.0f}%, efficiency {proj['er20']:.2f} "
                  f"({'full size' if proj['er20'] >= CHOP_THR else 'chop rule halves it'}) -> {sz}")
    for t, (sleeve, vt, label) in TIERS.items():
        print(f"\n {'-'*8} {t}: {label} (vol target {vt:.2f}) {'-'*8}")
        for k, v in W[t].items():
            if v > 0.0005:
                print(f"   {k:<5}{v*100:>7.1f}%   {'#'*int(round(v*40))}")
    c = state["context_only"]
    print(f"\n context only (NOT used by the rules): 10y {c.get('us10y_pct')}%  "
          f"10y-3m {round(c['curve_10y_3m_pct'], 2) if c.get('curve_10y_3m_pct') is not None else 'n/a'}pp  "
          f"DXY {c.get('dxy')}  VIX {c.get('vix')}")
    print(f"\n Run Friday ~15:40-15:45 ET; market-on-close orders; 2pp no-trade band, always trade on/off switches.\n{bar}\n")
    return state


if __name__ == "__main__":
    if "--check" in sys.argv:
        res = check()
        sys.exit(2 if res == "no-panel" else 0 if res else 1)
    main(as_json="--json" in sys.argv, log="--log" in sys.argv)
