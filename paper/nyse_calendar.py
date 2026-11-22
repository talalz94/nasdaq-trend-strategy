"""
Rules-based NYSE trading calendar (no network, no third-party packages).

Full-day holidays: New Year's Day, Martin Luther King Jr. Day, Washington's Birthday, Good Friday,
Memorial Day, Juneteenth (2022+), Independence Day, Labor Day, Thanksgiving, Christmas.
Weekend rule: a Saturday holiday is observed on Friday and a Sunday holiday on Monday, EXCEPT
New Year's Day on a Saturday, which NYSE does not observe (Friday 31 Dec stays open).

1:00 pm ET early closes: 3 July when it falls Monday-Thursday, the day after Thanksgiving, and
24 December when it falls Monday-Thursday.

Unscheduled closures (e.g. a national day of mourning) cannot be predicted by rules: list them in
config [schedule] extra_holidays / extra_early_closes.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

REGULAR_CLOSE = dt.time(16, 0)
EARLY_CLOSE = dt.time(13, 0)
ONE_DAY = dt.timedelta(days=1)


def _easter(y: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(y, month, day)


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> dt.date:
    first = dt.date(y, m, 1)
    return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(y: int, m: int, weekday: int) -> dt.date:
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    last = nxt - ONE_DAY
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - ONE_DAY
    if d.weekday() == 6:
        return d + ONE_DAY
    return d


@lru_cache(maxsize=None)
def rule_holidays(y: int) -> frozenset:
    hs = set()
    nyd = dt.date(y, 1, 1)
    if nyd.weekday() == 6:
        hs.add(nyd + ONE_DAY)
    elif nyd.weekday() < 5:
        hs.add(nyd)
    hs.add(_nth_weekday(y, 1, 0, 3))          # Martin Luther King Jr. Day
    hs.add(_nth_weekday(y, 2, 0, 3))          # Washington's Birthday
    hs.add(_easter(y) - 2 * ONE_DAY)          # Good Friday
    hs.add(_last_weekday(y, 5, 0))            # Memorial Day
    if y >= 2022:
        hs.add(_observed(dt.date(y, 6, 19)))  # Juneteenth
    hs.add(_observed(dt.date(y, 7, 4)))       # Independence Day
    hs.add(_nth_weekday(y, 9, 0, 1))          # Labor Day
    hs.add(_nth_weekday(y, 11, 3, 4))         # Thanksgiving
    hs.add(_observed(dt.date(y, 12, 25)))     # Christmas
    return frozenset(hs)


@lru_cache(maxsize=None)
def rule_early_closes(y: int) -> frozenset:
    ec = set()
    j3 = dt.date(y, 7, 3)
    if j3.weekday() <= 3:
        ec.add(j3)
    ec.add(_nth_weekday(y, 11, 3, 4) + ONE_DAY)
    d24 = dt.date(y, 12, 24)
    if d24.weekday() <= 3:
        ec.add(d24)
    return frozenset(ec - rule_holidays(y))


def _as_date(x) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x)[:10])


class NYSECalendar:
    def __init__(self, extra_holidays=(), extra_early_closes=()):
        self.extra_holidays = frozenset(_as_date(d) for d in extra_holidays)
        self.extra_early = frozenset(_as_date(d) for d in extra_early_closes)

    # ---------------------------------------------------------------- sessions
    def is_holiday(self, d) -> bool:
        d = _as_date(d)
        return d in rule_holidays(d.year) or d in self.extra_holidays

    def is_session(self, d) -> bool:
        d = _as_date(d)
        return d.weekday() < 5 and not self.is_holiday(d)

    def is_early_close(self, d) -> bool:
        d = _as_date(d)
        return self.is_session(d) and (d in rule_early_closes(d.year) or d in self.extra_early)

    def close_time(self, d) -> dt.time:
        return EARLY_CLOSE if self.is_early_close(d) else REGULAR_CLOSE

    def sessions(self, start, end) -> list:
        d, end = _as_date(start), _as_date(end)
        out = []
        while d <= end:
            if self.is_session(d):
                out.append(d)
            d += ONE_DAY
        return out

    def prev_session(self, d) -> dt.date:
        d = _as_date(d) - ONE_DAY
        while not self.is_session(d):
            d -= ONE_DAY
        return d

    def next_session(self, d) -> dt.date:
        d = _as_date(d) + ONE_DAY
        while not self.is_session(d):
            d += ONE_DAY
        return d

    # ---------------------------------------------------------------- weekly decision days
    def week_decision_day(self, d):
        """Last trading session of the Monday-Friday week containing d (None if the week is closed)."""
        d = _as_date(d)
        monday = d - dt.timedelta(days=d.weekday())
        for k in range(4, -1, -1):
            day = monday + dt.timedelta(days=k)
            if self.is_session(day):
                return day
        return None

    def is_decision_day(self, d) -> bool:
        d = _as_date(d)
        return self.week_decision_day(d) == d

    def latest_decision_day(self, d) -> dt.date:
        """Most recent weekly decision day on or before d."""
        d = _as_date(d)
        for _ in range(60):
            wd = self.week_decision_day(d)
            if wd is not None and wd <= d:
                return wd
            d = d - dt.timedelta(days=d.weekday() + 1)      # Sunday of the previous week
        raise RuntimeError("no decision day found")

    def next_decision_day(self, d) -> dt.date:
        """First weekly decision day strictly after d."""
        d = _as_date(d)
        for _ in range(60):
            wd = self.week_decision_day(d)
            if wd is not None and wd > d:
                return wd
            d = d + dt.timedelta(days=7 - d.weekday())      # Monday of the next week
        raise RuntimeError("no decision day found")
