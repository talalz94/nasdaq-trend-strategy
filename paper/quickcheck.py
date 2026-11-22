"""
Is anything due? A standard-library-only check (no pandas, no network) that lets an idle scheduled run finish
in a fraction of a second; importing pandas alone costs ~3 s of CPU on a laptop, and the scheduler starts a
run every 5 minutes.

It mirrors what Engine.tick() can do and is deliberately conservative: whenever something might be due it
answers with a reason and the full engine runs. Work that just failed (a booking waiting for Yahoo, a CSV held
open in Excel, a shadow refresh or the monthly QQQ check) is retried only after the engine's back-off time, here
as in the engine. tests/test_quickcheck.py replays the engine over weeks of simulated time and asserts that
every run this module calls idle really changes nothing.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from .nyse_calendar import NYSECalendar

ET = ZoneInfo("America/New_York")
THREE_YEARS = 757                     # sessions of NAV history the 3-years-behind-QQQ check needs


def _d(x) -> dt.date:
    return dt.date.fromisoformat(str(x)[:10])


def _later(iso, now) -> bool:
    return bool(iso) and now < dt.datetime.fromisoformat(str(iso))


def reason_to_run(cfg: dict, state: dict | None, now: dt.datetime) -> str | None:
    """The reason the full engine must run now, or None if a run would change nothing."""
    if not state or not isinstance(state, dict) or state.get("account") is None:
        return "no account yet"
    r, a = state.get("runner") or {}, state["account"]
    s, sch = cfg["strategy"], cfg["schedule"]
    backoff = r.get("backoff") or {}
    waiting = lambda key: _later(backoff.get(key), now)
    if r.get("log_backlog") and not waiting("logs"):
        return "log rows waiting to be written"
    if r.get("shadow_dirty") and not waiting("shadow"):
        return "shadow tranche to refresh"
    if (r.get("tier"), r.get("route")) != (s["tier"], s["route"]):
        return "tier or route changed"
    segs = r.get("shadow_segments") or []
    if not segs or (segs[-1].get("tier"), segs[-1].get("route"), segs[-1].get("cost_bps")) != (
            s["tier"], s["route"], cfg["account"]["cost_bps"]):
        return "settings changed"
    sent = r.get("alerts_sent") or {}
    cal = NYSECalendar(list(sch["extra_holidays"]) + list(r.get("detected_closures") or []), sch["extra_early_closes"])
    today = now.date()
    close = lambda d: dt.datetime.combine(d, cal.close_time(d), tzinfo=ET)
    settle = dt.timedelta(minutes=sch["settle_minutes_after_close"])
    last_complete = today if cal.is_session(today) and now >= close(today) + settle else cal.prev_session(today)
    last_booked = _d(a["last_processed"])
    if last_complete > last_booked:
        if not waiting("book"):
            return "a session to book"
        if len(cal.sessions(last_booked + dt.timedelta(days=1), last_complete)) >= 2 \
                and f"booking-lag:{today}" not in sent:
            return "booking-lag alert"
    D = cal.latest_decision_day(today)
    last = r.get("last_decision_date")
    if not (last and _d(last) >= D and r.get("redo") != str(D)):
        if s["route"] == "US":
            start = close(D) - dt.timedelta(minutes=sch["signal_minutes_before_close"])
            if today == D and start <= now < start + dt.timedelta(minutes=sch["window_minutes"]):
                return "decision window"
            ready = close(D) + settle
        else:
            ready = close(D) + dt.timedelta(minutes=sch["ucits_decide_minutes_after_close"])
        if now >= ready and not _later(r.get("retry_after"), now):
            return "weekly decision due"
    hw = a.get("high_water") or 0
    trig = cfg["review"]["drawdown_trigger"].get(s["tier"], 0.35)
    if hw > 0 and a["last_nav"] / hw - 1 <= -trig and f"dd:{a.get('high_water_date')}" not in sent:
        return "drawdown review trigger"
    if len(a.get("nav_hist") or {}) >= THREE_YEARS and r.get("qqq3y_checked") != f"{now:%Y-%m}" \
            and not waiting("qqq3y"):
        return "monthly review check"
    return None
