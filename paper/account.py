"""
The simulated paper account (no broker, no credentials).

Orders are queued at decision time and filled at the official closing price of their execution
session, published by Yahoo shortly after the bell:
  * US route: market-on-close on the decision day (or the fallback day): `execute_on` is that date.
  * Orders on European listings (the UCITS route) carry no date. All orders of one decision fill
    TOGETHER at the close of the first NYSE session after the decision day on which every instrument
    of the decision trades and whose close comes after the decision was made (`not_before`). This is
    the research's lag-2 execution (update_2026_11/ucits.py, held_weights_eu), so a London-only
    holiday can no longer fill the Paris buy before the London sale that pays for it.
Costs = cost_bps per unit traded + an optional commission per order.

process_session(d) books, in this order: dividends whose ex-date falls on or before d (paid to whoever
held at the previous close), fills due on d (sells first), then the NAV mark at d's close. It checks that
every price it needs exists BEFORE changing anything, so a missing price (data not yet published) leaves
the account untouched (NotReady) and the next run retries.

Splits. Yahoo back-adjusts every earlier close when a split takes effect. rebase() compares Yahoo's
current close for a reference day with the close the account recorded for that day: a change by a
round factor (2:1, 1:10, 200:1 ...) means the share basis changed, and positions, pending quantities and
the dividend history are converted before anything is priced. This works whenever the account catches
up, and it also catches splits Yahoo never records (LQQ.PA's 200:1 in July 2026). A close that jumps
inconsistently with the market without such a basis change stops booking (SuspectPrice) until the data
is fixed or the user confirms it (`runner.py split` / `runner.py accept-move`).
"""
from __future__ import annotations

import copy
import datetime as dt
import math

import pandas as pd

from .market import CURRENCY

DIV_LOOKBACK_DAYS = 10      # Yahoo sometimes posts a dividend a day or two after its ex-date
HISTORY_KEEP = 60           # sessions of end-of-day positions kept for late dividends
NAV_KEEP = 800              # sessions of NAV kept for the 12-month flag and the 3-year review check
BASIS_TOL = 1.25            # a reference close that changed by more than this factor = new share basis
ORDER_LEAD = pd.Timedelta(minutes=10)   # an order must exist this long before the close it fills at


class NotReady(RuntimeError):
    """A price needed for this session is not published yet; nothing was changed."""

    def __init__(self, msg, symbol=None, day=None):
        super().__init__(msg)
        self.symbol, self.day = symbol, day


class SuspectPrice(NotReady):
    """A close moved inconsistently with the market (an unrecorded split or a bad print); nothing changed."""


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) and not isinstance(x, dt.datetime) else dt.date.fromisoformat(str(x)[:10])


def _floor(q: float, fractional: bool) -> float:
    """Whole shares, or 4 decimals rounded DOWN (rounding up could overspend)."""
    return math.floor(q * 1e4 + 1e-6) / 1e4 if fractional else float(math.floor(q + 1e-9))


def snap_ratio(r: float) -> float:
    """Snap a measured basis change to the nearest plain split ratio (n:1, 1:n, 3:2, 2:3, 5:4, 4:5) within 3%."""
    cands = [float(n) for n in range(2, 1001)] + [1.0 / n for n in range(2, 1001)] + [1.5, 2 / 3, 1.25, 0.8]
    best = min(cands, key=lambda c: abs(math.log(r / c)))
    return best if abs(r / best - 1) < 0.03 else r


class SimAccount:
    def __init__(self, cash, positions=None, pending=None, last_processed=None, start_date=None,
                 start_nav=None, high_water=None, high_water_date=None, credited=None, pos_history=None,
                 last_nav=None, refs=None, accepted=None, nav_hist=None):
        self.cash = float(cash)
        self.positions = {k: float(v) for k, v in (positions or {}).items() if v}
        self.pending = list(pending or [])
        self.last_processed = last_processed            # ISO date string of the last booked session
        self.start_date = start_date
        self.start_nav = float(start_nav if start_nav is not None else cash)
        self.high_water = float(high_water if high_water is not None else self.start_nav)
        self.high_water_date = high_water_date or start_date
        self.credited = set(tuple(x) for x in (credited or []))   # {(symbol, exdate)}
        self.pos_history = dict(pos_history or {})      # ISO date -> end-of-day positions
        self.last_nav = float(last_nav if last_nav is not None else self.start_nav)
        self.refs = {k: list(v) for k, v in (refs or {}).items()}  # symbol -> [ISO date, close on today's basis]
        self.accepted = [list(x) for x in (accepted or [])]       # [[symbol, ISO date]] moves the user confirmed
        self.nav_hist = dict(nav_hist or {})            # ISO date -> NAV (last NAV_KEEP sessions)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return dict(cash=self.cash, positions=self.positions, pending=self.pending,
                    last_processed=self.last_processed, start_date=self.start_date,
                    start_nav=self.start_nav, high_water=self.high_water,
                    high_water_date=self.high_water_date, credited=sorted(list(x) for x in self.credited),
                    pos_history=self.pos_history, last_nav=self.last_nav, refs=self.refs,
                    accepted=self.accepted, nav_hist=self.nav_hist)

    @classmethod
    def from_dict(cls, d: dict) -> "SimAccount":
        return cls(**d)

    def copy(self) -> "SimAccount":
        return SimAccount.from_dict(copy.deepcopy(self.to_dict()))

    # ------------------------------------------------------------------ orders
    def queue(self, orders, decision_date, execute_on=None, *, not_before=None, group=None,
              px_date=None, tag="") -> list:
        """Queue one decision's orders; every order still pending from an earlier decision is superseded.
        execute_on: fill date (US market-on-close) or None = fill together with the rest of the decision
        at the first session on which every symbol in `group` trades after `not_before`."""
        dd = str(decision_date)
        cancelled, self.pending = self.pending, []
        for o in orders:
            pxd = px_date.get(o.symbol) if isinstance(px_date, dict) else px_date
            self.pending.append(dict(
                order_id=f"{dd}{tag}-{o.symbol}-{o.side}", decision_date=dd,
                execute_on=str(execute_on) if execute_on else None,
                not_before=pd.Timestamp(not_before).isoformat() if (not_before is not None and not execute_on) else None,
                group=sorted(set(group or []) | {x.symbol for x in orders}) if not execute_on else None,
                px_date=str(pxd) if pxd else None,
                symbol=o.symbol, side=o.side, qty=float(o.qty), est_price=float(o.est_price),
                reason=o.reason))
        return cancelled

    def due_orders(self, d: dt.date, md, close_ts=None) -> list:
        due = [p for p in self.pending if p["execute_on"] and _d(p["execute_on"]) == d]
        floating = {}
        for p in self.pending:
            if not p["execute_on"]:
                floating.setdefault(p["decision_date"], []).append(p)
        for dd, ps in floating.items():
            if _d(dd) >= d:
                continue
            syms = set(ps[0].get("group") or []) | {p["symbol"] for p in ps}
            if not all(md.has_bar(s, d) for s in syms):
                continue                                    # some listing is closed today: all legs wait
            nb = ps[0].get("not_before")
            if nb and close_ts is not None:
                t0 = pd.Timestamp(nb)
                if any(close_ts(s, d) < t0 + ORDER_LEAD for s in syms):
                    continue                                # that close printed before the decision existed
            due += ps
        return due

    # ------------------------------------------------------------------ prices
    def _mark_price(self, md, symbol, d, ffill_ok=()):
        """Close on d; a non-US listing closed on d (local holiday), or a symbol whose bar Yahoo never
        published (ffill_ok), marks at its last earlier close."""
        px = md.close_on(symbol, d)
        if px is None and ("." in symbol or symbol in CURRENCY or symbol in ffill_ok):
            prev = md.last_close_before(symbol, d, max_days=10)
            if prev is not None:
                return prev[1]
        return px

    def _fx(self, md, symbol, d):
        ccy = CURRENCY.get(symbol, "USD")
        fx = md.fx_to_usd(ccy, d)
        if fx is None:
            raise NotReady(f"no {ccy}USD rate for {d}", symbol, d)
        return fx

    # ------------------------------------------------------------------ splits / share basis
    def rebase(self, md, upto, fractional: bool = False) -> list:
        """Convert positions, pending orders and dividend history to Yahoo's current share basis.
        Returns [(level, kind, detail)] for every change made."""
        out = []
        syms = set(self.positions) | {p["symbol"] for p in self.pending} | set(self.refs)
        for s in sorted(syms):
            ref = self.refs.get(s)
            if ref is not None and len(ref) > 2:
                continue        # a split the user entered by hand: Yahoo's history is not on the new basis yet
            if ref is None:
                ps = [p for p in self.pending if p["symbol"] == s and p.get("px_date")]
                if not ps:
                    continue
                ref = [ps[0]["px_date"], ps[0]["est_price"]]
            rdate, rclose = _d(ref[0]), float(ref[1])
            try:
                now = md.close_on(s, rdate)
            except Exception:
                continue
            if now is None or not now > 0 or not rclose > 0:
                continue
            r = rclose / now
            if 1 / BASIS_TOL < r < BASIS_TOL:
                continue
            rec = md.splits_between(s, rdate, upto)
            prod = math.prod(x for _, x in rec) if rec else None
            k = prod if prod and abs(prod / r - 1) < 0.05 else snap_ratio(r)
            lieu = 0.0
            old = self.positions.get(s, 0.0)
            if old:
                new = old * k
                if not fractional:
                    whole = math.floor(new + 1e-9)
                    if new - whole > 1e-9:
                        fx = self._fx(md, s, rdate)
                        lieu = (new - whole) * now * fx
                        self.cash += lieu
                    new = float(whole)
                self.positions[s] = new
                if not new:
                    self.positions.pop(s)
            for p in self.pending:
                if p["symbol"] == s:
                    p["qty"] = _floor(p["qty"] * k, fractional)
                    p["est_price"] = p["est_price"] / k
            for h in self.pos_history.values():
                if s in h:
                    h[s] = h[s] * k
            self.refs[s] = [rdate.isoformat(), now]
            how = (f"Yahoo recorded split(s) {[(str(a), b) for a, b in rec]}" if prod and k == prod
                   else "NOT recorded by Yahoo: inferred from the back-adjusted closes - please check")
            detail = (f"{s}: share basis changed by {k:g}:1 ({how}); {s} {old:g} -> {self.positions.get(s, 0.0):g} "
                      f"shares" + (f", ${lieu:,.2f} cash in lieu of a fractional share" if lieu else ""))
            out.append(("INFO" if prod and k == prod else "WARN", "split", detail))
        return out

    def manual_split(self, symbol, ratio: float, fractional: bool = False) -> str:
        """User-confirmed split the data does not show: shares x ratio, prices / ratio."""
        old = self.positions.get(symbol, 0.0)
        if old:
            self.positions[symbol] = old * ratio if fractional else float(math.floor(old * ratio + 1e-9))
        for p in self.pending:
            if p["symbol"] == symbol:
                p["qty"] = _floor(p["qty"] * ratio, fractional)
                p["est_price"] = p["est_price"] / ratio
        for h in self.pos_history.values():
            if symbol in h:
                h[symbol] = h[symbol] * ratio
        if symbol in self.refs:          # the reference close on the new basis, marked so rebase() leaves it alone
            self.refs[symbol] = [self.refs[symbol][0], self.refs[symbol][1] / ratio, "manual"]
        return f"{symbol}: manual split {ratio:g}:1, {old:g} -> {self.positions.get(symbol, 0.0):g} shares"

    # ------------------------------------------------------------------ the daily booking
    def process_session(self, d, md, *, cost_bps: float, commission: float = 0.0, withholding: float = 0.0,
                        fractional: bool = False, close_ts=None, guard=None, ffill_ok=(), roll_to=None,
                        apply_splits=None) -> dict:
        """Book session d. close_ts(symbol, d) -> the close's time (for not_before checks); guard(...) -> a
        message if a close looks wrong; ffill_ok = symbols whose missing bar on d may be marked at the last
        earlier close (an order due on d for them moves to roll_to)."""
        d = _d(d)
        if self.last_processed and _d(self.last_processed) >= d:
            raise ValueError(f"session {d} already processed (last {self.last_processed})")
        due = self.due_orders(d, md, close_ts)
        stale = [p for p in self.pending if p["execute_on"] and _d(p["execute_on"]) < d]
        events, fills, divs, cuts = [], [], [], []

        # ---- 1. readiness: every price we need must exist before anything changes
        rolled = []
        for p in list(due):
            if md.close_on(p["symbol"], d) is None:
                if p["symbol"] in ffill_ok and roll_to is not None:
                    rolled.append(p)
                    due.remove(p)
                else:
                    raise NotReady(f"no fill price for {p['symbol']} on {d}", p["symbol"], d)
        need = set(self.positions) | {p["symbol"] for p in due}
        marks = {}
        for s in need:
            px = self._mark_price(md, s, d, ffill_ok)
            if px is None or not px > 0:
                raise NotReady(f"no close for {s} on {d}", s, d)
            marks[s] = px
        fxs = {s: self._fx(md, s, d) for s in need}
        if guard is not None:
            for s in sorted(need):
                ref = self.refs.get(s)
                if ref is None or _d(ref[0]) >= d or [s, str(d)] in self.accepted:
                    continue
                msg = guard(s, _d(ref[0]), float(ref[1]), d, marks[s])
                if msg:
                    raise SuspectPrice(msg, s, d)
        for p in rolled:
            events.append(("rolled", f"{p['order_id']}: no {p['symbol']} close on {d}; moved to {roll_to}"))
            p["execute_on"] = str(roll_to)

        # ---- 2. dividends (ex-date in the look-back window, not yet credited)
        prev_sessions = sorted(self.pos_history)
        for s in set(self.positions) | {s for h in self.pos_history.values() for s in h}:
            try:
                rows = md.dividends_between(s, d - dt.timedelta(days=DIV_LOOKBACK_DAYS), d)
            except Exception:
                rows = []
            for exd, amt in rows:
                key = (s, str(exd))
                if key in self.credited:
                    continue
                holder = None                                   # shares held at the close before exd
                earlier = [h for h in prev_sessions if _d(h) < exd]
                if earlier:
                    holder = self.pos_history[earlier[-1]].get(s, 0.0)
                elif self.start_date and exd > _d(self.start_date):
                    holder = self.positions.get(s, 0.0)
                if not holder:
                    self.credited.add(key)
                    continue
                fx = fxs.get(s) or self._fx(md, s, d)
                cash_in = holder * amt * fx * (1.0 - withholding)
                self.cash += cash_in
                self.credited.add(key)
                divs.append(dict(symbol=s, ex_date=str(exd), per_share=amt, shares=holder, usd=round(cash_in, 2)))

        # ---- 3. fills at the official close (sells first, then buys)
        c = cost_bps / 1e4
        for p in sorted(due, key=lambda p: 0 if p["side"] == "SELL" else 1):
            s, px = p["symbol"], md.close_on(p["symbol"], d)
            fx = fxs.get(s) or self._fx(md, s, d)
            qty, note = float(p["qty"]), ""
            if p["side"] == "SELL":
                qty = min(qty, self.positions.get(s, 0.0))
                notional = qty * px * fx
                cost = notional * c + (commission if qty else 0.0)
                self.cash += notional - cost
                self.positions[s] = self.positions.get(s, 0.0) - qty
            else:
                unit = px * fx * (1 + c)
                afford = max(self.cash - commission, 0.0) / unit if unit > 0 else 0.0
                if qty > afford + 1e-9:
                    planned, qty = qty, _floor(afford, fractional)
                    note = f"reduced from {planned:g} (cash)"
                    cuts.append(dict(order_id=p["order_id"], symbol=s, planned=planned, filled=qty))
                notional = qty * px * fx
                cost = notional * c + (commission if qty else 0.0)
                self.cash -= notional + cost
                self.positions[s] = self.positions.get(s, 0.0) + qty
            if abs(self.positions.get(s, 0.0)) < 1e-9:
                self.positions.pop(s, None)
            if qty <= 0:
                events.append(("unfilled", f"{p['order_id']}: {p['side']} {p['qty']:g} {s} not filled ({note or 'nothing to sell'})"))
                continue
            fills.append(dict(fill_date=str(d), decision_date=p["decision_date"], order_id=p["order_id"],
                              symbol=s, side=p["side"], qty=qty, price=px, price_ccy=CURRENCY.get(s, "USD"),
                              fx_to_usd=fx, notional_usd=round(qty * px * fx, 2), cost_usd=round(cost, 2),
                              cash_after=round(self.cash, 2), note=note))
        done = {p["order_id"] for p in due}
        expired = []
        for p in stale:
            if p["order_id"] not in done:
                events.append(("expired", f"{p['order_id']} was due {p['execute_on']} but never filled"))
                expired.append(p["order_id"])
                done.add(p["order_id"])
        self.pending = [p for p in self.pending if p["order_id"] not in done]

        # ---- 4. mark to market at d's close
        pos_val = 0.0
        for s, q in self.positions.items():
            pos_val += q * marks[s] * fxs[s]
        nav = self.cash + pos_val
        prev_nav = self.last_nav
        if nav > self.high_water:
            self.high_water, self.high_water_date = nav, str(d)
        weights = {s: round(q * marks[s] * fxs[s] / nav, 4) for s, q in self.positions.items()} if nav else {}
        nav_row = dict(date=str(d), nav=round(nav, 2), cash=round(self.cash, 2), positions_value=round(pos_val, 2),
                       positions=dict(self.positions), weights=weights,
                       dividends_usd=round(sum(x["usd"] for x in divs), 2),
                       daily_return=(nav / prev_nav - 1) if prev_nav else 0.0,
                       cum_return=nav / self.start_nav - 1 if self.start_nav else 0.0,
                       high_water=round(self.high_water, 2),
                       drawdown=nav / self.high_water - 1 if self.high_water > 0 else 0.0)
        self.last_nav = nav
        self.last_processed = str(d)
        self.pos_history[str(d)] = dict(self.positions)
        for k in sorted(self.pos_history)[:-HISTORY_KEEP]:
            del self.pos_history[k]
        self.nav_hist[str(d)] = round(nav, 2)
        for k in sorted(self.nav_hist)[:-NAV_KEEP]:
            del self.nav_hist[k]
        refs = {}
        for s in self.positions:
            px = md.close_on(s, d)
            if px is not None:
                refs[s] = [str(d), px]
            elif s in self.refs:
                refs[s] = self.refs[s]
            else:
                prev = md.last_close_before(s, d, max_days=10)
                if prev is not None:
                    refs[s] = [str(prev[0]), prev[1]]
        self.refs = refs
        self.accepted = [x for x in self.accepted if _d(x[1]) > d]
        return dict(fills=fills, nav_row=nav_row, dividends=divs, events=events, cuts=cuts, expired=expired)
