"""
LIVE WEEKLY SCANNER — what to hold this week (research tool, NOT investment advice).

Implements the FINAL researched strategy plus the aggressive variant, using exactly
the rules validated in the backtests (Phases A-C + final validation).

  ── STRATEGY F — "FINAL / BALANCED"   (CAGR ~24%, maxDD ~-31%, Calmar 0.79)
     85% ENGINE + 15% GOLD, rebalanced weekly (Friday close):
       risk-on = NDX > EMA-100  AND NOT (curve inverted in last 12m AND NDX < EMA-100)
       scale   = min(1, 0.20 / realized_vol_20d)
       seasonal: x0.5 during August & September            (drawdown reducer)
       clean trend (efficiency ratio >= 0.25) -> TQQQ x scale
       choppy      (efficiency ratio <  0.25) -> TQQQ x scale x 0.5
       capital not in TQQQ -> LONG TREASURIES (TLT)
     Permanent 15% GOLD (GLD) sleeve.

  ── STRATEGY A — "AGGRESSIVE GROWTH"  (CAGR ~31%, but maxDD ~-67%)
       risk-on = NDX > EMA-100 ; TQQQ = risk_on x min(1, 0.25/vol) ; remainder CASH.

Run:  python scanner.py           (today's allocation)
      python scanner.py --json    (machine readable, writes scanner_state.json)
"""
import sys, json
import numpy as np, pandas as pd, yfinance as yf

EMA_N, ER_N = 100, 20
VOL_F, CHOP_THR, CHOP_POS, GOLD_W = 0.20, 0.25, 0.5, 0.15
VOL_A = 0.25
SEASONAL_MONTHS = (8, 9)
SEASONAL_MULT = 0.5


def close(t, period="5y"):
    h = yf.Ticker(t).history(period=period)["Close"]
    h.index = h.index.tz_localize(None).normalize()
    return h[~h.index.duplicated()].dropna()


def main(as_json=False):
    ndx = close("^NDX"); tnx = close("^TNX")/100; irx = close("^IRX")/100
    ema = ndx.ewm(span=EMA_N, adjust=False).mean()
    vol = float(ndx.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252))
    er = float(abs(ndx.diff(ER_N).iloc[-1]) / ndx.diff().abs().rolling(ER_N).sum().iloc[-1])
    curve = (tnx.reindex(ndx.index).ffill() - irx.reindex(ndx.index).ffill())
    inv_recent = bool((curve.iloc[-252:] < 0).any())
    above = bool(ndx.iloc[-1] > ema.iloc[-1])
    confirmed = inv_recent and not above
    today = ndx.index[-1]; month = today.month

    # ---- Strategy F ----
    scale = min(1.0, VOL_F/vol) if vol > 0 else 1.0
    raw_scale = scale
    seasonal_on = month in SEASONAL_MONTHS
    if seasonal_on: scale *= SEASONAL_MULT
    risk_on = above and not confirmed
    clean = er >= CHOP_THR
    eq_w = 0.0 if not risk_on else (scale if clean else scale*CHOP_POS)
    core = 1.0-GOLD_W
    F = {"TQQQ": eq_w*core, "TLT": core-eq_w*core, "GLD": GOLD_W}

    # ---- Strategy A ----
    sa = min(1.0, VOL_A/vol) if vol > 0 else 1.0
    A = {"TQQQ": sa if above else 0.0, "CASH": (1-sa) if above else 1.0}

    state = dict(asof=str(today.date()), ndx=round(float(ndx.iloc[-1]), 1),
                 ema100=round(float(ema.iloc[-1]), 1), above_ema=above,
                 realized_vol=round(vol, 4), vol_scale=round(raw_scale, 3),
                 efficiency_ratio=round(er, 3), trend="CLEAN" if clean else "CHOPPY",
                 curve_10y_3mo=round(float(curve.iloc[-1]), 2),
                 inverted_last_12m=inv_recent, recession_signal=confirmed,
                 seasonal_derisk=seasonal_on,
                 strategy_F={k: round(v, 3) for k, v in F.items()},
                 strategy_A={k: round(v, 3) for k, v in A.items()})

    if as_json:
        json.dump(state, open("scanner_state.json", "w"), indent=1)
        print(json.dumps(state, indent=1)); return state

    print(f"\n{'='*68}\n WEEKLY SCANNER  —  {today.date()}    (research tool, not advice)\n{'='*68}")
    print(f" Nasdaq-100   {ndx.iloc[-1]:>11,.0f}    EMA-100 {ema.iloc[-1]:>10,.0f}   "
          f"{'ABOVE = risk-on' if above else 'BELOW = risk-off'}")
    print(f" realized vol {vol*100:>10.1f}%    position scale {raw_scale*100:.0f}%")
    print(f" efficiency   {er:>11.2f}    {'CLEAN trend (full size)' if clean else 'CHOPPY (half size)'}")
    print(f" yield curve  {float(curve.iloc[-1]):>+11.2f}    inverted last 12m: {'YES' if inv_recent else 'no'}"
          f"{'   >> RECESSION SIGNAL ON' if confirmed else ''}")
    print(f" seasonality  {'AUG/SEP: half size' if seasonal_on else 'normal month':>11}")
    print(f"\n {'-'*20} STRATEGY F  (balanced, recommended) {'-'*10}")
    for k, v in F.items():
        if v > 0.001: print(f"   {k:<6}{v*100:>7.1f}%   {'#'*int(v*40)}")
    cf = 1-sum(F.values())
    if cf > 0.001: print(f"   {'CASH':<6}{cf*100:>7.1f}%")
    print(f"\n {'-'*20} STRATEGY A  (aggressive) {'-'*21}")
    for k, v in A.items():
        if v > 0.001: print(f"   {k:<6}{v*100:>7.1f}%   {'#'*int(v*40)}")
    print(f"\n Rebalance weekly at Friday close.")
    print(f" Expect drawdowns of ~-31% (F) to ~-67% (A). Research only.\n{'='*68}\n")
    return state


if __name__ == "__main__":
    main("--json" in sys.argv)
