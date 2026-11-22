"""
Shadow 5-weekday tranche (no orders): the planning basis from the Nov-2026 re-check.

Five notional books follow the same rules; book k decides on every trading day that is weekday k
(Mon..Fri) from that day's NDX close and holds the weights until its next rebalance. The tranche is the
equal-weight average of the five books' daily returns. Returns are total-return (Yahoo Adj Close), costs
are cost_bps per unit traded. The live account (one fixed weekday, Friday) should scatter around this
line; judge results against it, not against the lucky in-sample Friday backtest.

Execution follows the research conventions exactly:
  US route     trade at the decision day's own close (market-on-close), hold from the next session;
  UCITS route  European listings close (~11:30 ET) BEFORE the US-close signal exists, so the trade is at
               the close of the first later NYSE session on which every listing of the route trades
               (update_2026_11/ucits.py held_weights_eu with lag 2 - "the earliest honest execution").
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import strategy
from .market import CURRENCY

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
WARMUP_DAYS = 400


def held_weights(sessions: pd.DatetimeIndex, decisions: dict, syms: list, lag: int, ok=None) -> pd.DataFrame:
    """Weights held on each session (decided targets take effect the session AFTER execution).
    decisions: {session: {symbol: weight}}. A decision on session i executes at the close of the first
    session j >= i + lag - 1 with ok[j] (all listings trading). Mirrors held_weights_eu in the research."""
    n = len(sessions)
    ok = np.ones(n, dtype=bool) if ok is None else np.asarray(ok, dtype=bool)
    out = np.zeros((n, len(syms)))
    cur = np.zeros(len(syms))
    pending, earliest = None, None
    for i, day in enumerate(sessions):
        out[i] = cur
        t = decisions.get(day)
        if t is not None:
            pending, earliest = np.array([t[s] for s in syms]), i + lag - 1
        if pending is not None and i >= earliest and ok[i]:
            cur, pending = pending, None
    return pd.DataFrame(out, index=sessions, columns=syms)


def compute(md, cal, tier, route, start, end, cost_bps=5.0) -> pd.DataFrame:
    """Tranche from `start` (NAV 100) to `end`. Books are warmed up over the prior ~year."""
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    syms = strategy.instruments(tier, route)
    sessions = pd.DatetimeIndex([pd.Timestamp(d) for d in cal.sessions(start - dt.timedelta(days=WARMUP_DAYS), end)])
    ndx = md.ndx_closes(upto=end)
    sig = strategy.scanner.signals(ndx).reindex(sessions)
    raw = md.adj_closes(syms, sessions[0] - pd.Timedelta(days=10), end)
    first_needed = sessions[sessions >= pd.Timestamp(start)]
    if not len(first_needed):
        return pd.DataFrame(columns=["tranche_nav"])
    for s in syms:                                  # refuse to fill missing history with zero returns
        col = raw[s].dropna() if s in raw else pd.Series(dtype=float)
        if not len(col) or col.index[0] > first_needed[0] or col.index[-1] < pd.Timestamp(end) - pd.Timedelta(days=10):
            have = f"{col.index[0].date()}..{col.index[-1].date()}" if len(col) else "none"
            raise ValueError(f"shadow tranche needs {s} prices from {first_needed[0].date()} to {end} (have {have})")
    adj = raw.reindex(sessions).ffill()
    for s in syms:                                  # non-USD listings -> USD
        ccy = CURRENCY.get(s, "USD")
        if ccy != "USD":
            fx = pd.Series({t: md.fx_to_usd(ccy, t.date()) for t in sessions}, dtype=float).ffill()
            adj[s] = adj[s] * fx
    rets = adj.pct_change(fill_method=None).fillna(0.0)
    mapping = strategy.ROUTE_MAP[route]
    if route == "US":
        lag, ok = 1, None
    else:
        lag = 2
        bars = [md.bar_dates(s) for s in syms]
        ok = np.array([all(t in b for b in bars) for t in sessions])

    def targets(day):
        row = sig.loc[day]
        if row.isna()[["ndx", "vol20", "er20"]].any():
            return None
        raw_w = strategy.scanner.tier_weights(row, day.month, tier)
        return {mapping[k]: v for k, v in raw_w.items()}

    books, book_targets = {}, {}
    for k, name in enumerate(WEEKDAYS):
        dec = {}
        for day in sessions[sessions.dayofweek == k]:
            t = targets(day)
            if t is not None:
                dec[day] = t
                book_targets[day] = (name, t)
        held = held_weights(sessions, dec, syms, lag, ok)
        turn = held.diff().abs().sum(axis=1).fillna(0.0)
        books[name] = (held * rets).sum(axis=1) - turn * cost_bps / 1e4
    B = pd.DataFrame(books)
    B["tranche"] = B[WEEKDAYS].mean(axis=1)
    keep = B.index >= pd.Timestamp(start)
    navs = (1 + B).cumprod()
    navs = 100 * navs[keep] / navs[keep].iloc[0]           # = 100 at the close of the start day
    out = pd.DataFrame(index=navs.index)
    out["tranche_nav"] = navs["tranche"]
    for name in WEEKDAYS:
        out[f"nav_{name}"] = navs[name]
    out["rebalanced_book"] = [book_targets[d][0] if d in book_targets else "" for d in out.index]
    out["book_targets"] = [json.dumps({s: round(v, 4) for s, v in book_targets[d][1].items()})
                           if d in book_targets else "" for d in out.index]
    out.index.name = "date"
    return out
