"""
The weekly decision, computed ONLY through scanner.py (signals / tier_weights), which is
parity-tested against the backtest engine (`python scanner.py --check`). Nothing here
re-implements a rule; it maps the scanner's tickers to the chosen route and records the inputs.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import asdict, dataclass, field

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import scanner  # noqa: E402

ROUTE_MAP = {
    "US":    {"QQQ": "QQQ",    "QLD": "QLD",    "TQQQ": "TQQQ", "IEF": "IEF",    "GLD": "GLD"},
    "UCITS": {"QQQ": "CNDX.L", "QLD": "LQQ.PA", "TQQQ": None,   "IEF": "CBU0.L", "GLD": "IGLN.L"},
}
MIN_HISTORY = 500          # EMA-100 seed error below 0.01% after 500 sessions


@dataclass
class Decision:
    decision_date: dt.date
    signal_bar: dt.date
    tier: str
    route: str
    ndx: float
    ema100: float
    ema100_prev: float
    vol20: float
    er20: float
    risk_on: bool
    chop_half: bool
    seasonal_half: bool
    engine: float
    targets: dict = field(default_factory=dict)      # instrument -> weight (sums to 1)

    @property
    def sleeve_symbol(self) -> str:
        return instruments(self.tier, self.route)[0]

    def as_row(self) -> dict:
        d = asdict(self)
        d["decision_date"] = str(self.decision_date)
        d["signal_bar"] = str(self.signal_bar)
        return d


def instruments(tier: str, route: str) -> list:
    """[Nasdaq sleeve, bond, gold] for the tier/route."""
    sleeve = scanner.TIERS[tier][0]
    m = ROUTE_MAP[route]
    out = [m[sleeve], m[scanner.OFF], m[scanner.GOLD]]
    if any(x is None for x in out):
        raise ValueError(f"{tier} has no {route} instruments")
    return out


def decide(ndx: pd.Series, decision_date, tier: str, route: str) -> Decision:
    """Decision from NDX closes whose LAST bar is the signal bar for `decision_date`.
    Raises if the series is too short or its last bar is not on decision_date."""
    decision_date = pd.Timestamp(decision_date).date()
    if len(ndx) < MIN_HISTORY:
        raise ValueError(f"only {len(ndx)} NDX bars; need {MIN_HISTORY} for a stable EMA-100")
    last = ndx.index[-1].date()
    if last != decision_date:
        raise ValueError(f"NDX last bar {last} is not the decision date {decision_date}")
    s = scanner.signals(ndx)
    row, prev = s.iloc[-1], s.iloc[-2]
    if pd.isna(row["vol20"]) or pd.isna(row["er20"]):
        raise ValueError("signal inputs are NaN")
    raw = scanner.tier_weights(row, decision_date.month, tier)
    m = ROUTE_MAP[route]
    targets = {m[k]: float(v) for k, v in raw.items()}
    if any(k is None for k in targets):
        raise ValueError(f"{tier} has no {route} instruments")
    sleeve = scanner.TIERS[tier][0]
    risk_on = bool(row["above"])
    return Decision(
        decision_date=decision_date, signal_bar=last, tier=tier, route=route,
        ndx=float(row["ndx"]), ema100=float(row["ema100"]), ema100_prev=float(prev["ema100"]),
        vol20=float(row["vol20"]), er20=float(row["er20"]), risk_on=risk_on,
        chop_half=bool(risk_on and scanner.USE_CHOP and row["er20"] < scanner.CHOP_THR),
        seasonal_half=decision_date.month in scanner.SEASON_MONTHS,
        engine=float(raw[sleeve] / scanner.CORE) if scanner.CORE else 0.0,
        targets=targets)


def reentry_bar(ndx: pd.Series, next_decision) -> float:
    """NDX close needed at the next decision for risk-on if NDX stays flat until then (scanner logic)."""
    s = scanner.signals(ndx)
    return float(scanner.reentry_threshold(s, pd.Timestamp(next_decision)))
