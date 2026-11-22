"""
Orchestration of the paper-trading loop (all times US Eastern).

tick()   is what the scheduler runs every few minutes; it is idempotent and does only what is due:
  1. book every completed session not yet booked (dividends, market-on-close fills at the official
     close, NAV mark) once Yahoo has published its closes; split basis changes are applied first;
  2. make the weekly decision when it is due:
       US route    in the decision day's window (default 15:40-15:48 ET; 12:40-12:48 on 1 pm closes)
                   from the live NDX price, orders filled at that day's close (market-on-close);
                   if the PC was off or asleep then: CATCH-UP - as soon as the PC is on after the
                   decision day's official close is published, from that close, filled at the next close
                   the market-on-close order can still reach (1-3 trading days late barely changes the
                   backtest return; a week late costs 5-7 points a year);
       UCITS route after the decision day's US close (whenever the PC is on), from the official close;
                   all orders fill together at the close of the first later NYSE session on which every
                   listing trades and whose close comes after the decision (the research's lag-2 execution);
  3. health checks: skipped weeks, booking stuck on missing prices, the pre-committed review triggers
     (drawdown on every booked session, three years behind QQQ) and the 12-month flag;
  4. refresh the shadow 5-weekday tranche after new sessions are booked.

State is saved FIRST and the CSV logs are written from a backlog kept inside state.json, so a log held
open (e.g. in Excel) or a crash can neither block the run nor leave a log describing something that did
not happen; unwritten rows are written by a later run.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil

import pandas as pd

from . import orders as ordmod
from . import shadow as shadowmod
from . import strategy
from .account import NotReady, SimAccount, SuspectPrice
from .alerts import Alerter
from .config import MOC_CUTOFF_MINUTES
from .market import CURRENCY, DataError, YahooMarketData, now_et
from .nyse_calendar import NYSECalendar
from .storage import (DECISION_FIELDS, DECISION_KEY, EVENT_FIELDS, FILL_FIELDS, NAV_FIELDS, RunLock,
                      append_unique_csv, load_json, read_csv, save_json_atomic, write_csv)

ET = "America/New_York"
DAY = dt.timedelta(days=1)
LOGS = {"decisions": (DECISION_FIELDS, DECISION_KEY), "fills": (FILL_FIELDS, ("order_id",)),
        "nav": (NAV_FIELDS, ("date",)), "events": (EVENT_FIELDS, ("time_utc", "kind", "detail"))}
LEVERAGE = {"QQQ": 1, "QLD": 2, "TQQQ": 3, "CNDX.L": 1, "LQQ.PA": 2, "IEF": 0, "CBU0.L": 0, "GLD": 0, "IGLN.L": 0}
EU_CLOSE = dt.time(16, 30)              # London 16:30 = Paris 17:30: the cash close of every UCITS listing
SUSPECT_LO, SUSPECT_HI = 0.6, 1.6       # a close this far off the market move (and no basis change) is suspect
DECIDE_RETRY = pd.Timedelta(minutes=30) # outside the live window a failed decision is retried at most this often
LATE_WARN_SESSIONS = 4                  # a decision that trades this many sessions late (or more) raises a WARN
# how long to wait before retrying work that just failed, so a stuck booking, a CSV held open in Excel or a
# Yahoo outage does not turn every 5-minute run into a full run (paper/quickcheck.py honours the same times)
BACKOFF = {"book": pd.Timedelta(minutes=15), "book_suspect": pd.Timedelta(minutes=60),
           "logs": pd.Timedelta(minutes=30), "shadow": pd.Timedelta(minutes=60), "qqq3y": pd.Timedelta(minutes=60)}
REPLAY_MARKER = "REPLAY_OUTPUT.txt"


class NotInitialized(RuntimeError):
    pass


class StateCorrupt(RuntimeError):
    pass


def _date(x) -> dt.date:
    return dt.date.fromisoformat(str(x)[:10])


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _us_listed(symbol) -> bool:
    return "." not in symbol and symbol not in CURRENCY


def same_dir(a, b) -> bool:
    na = os.path.normcase(os.path.realpath(os.path.abspath(a)))
    nb = os.path.normcase(os.path.realpath(os.path.abspath(b)))
    if na == nb:
        return True
    try:
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return False


class Engine:
    def __init__(self, cfg: dict, md=None, now_fn=None, data_dir: str | None = None,
                 quiet: bool = False, apply_splits: bool = True):
        self.cfg = cfg
        self._md = md
        self.now_fn = now_fn or now_et
        self.data_dir = data_dir or cfg["paths"]["data_dir"]
        sch = cfg["schedule"]
        self._extra_holidays = list(sch.get("extra_holidays", []))
        self._extra_early = list(sch.get("extra_early_closes", []))
        self.cal = NYSECalendar(self._extra_holidays, self._extra_early)
        self.alerter = Alerter(cfg, self.data_dir, quiet=quiet)
        self.apply_splits = apply_splits            # live mode: share-basis checks and the price guard
        self.tier, self.route = cfg["strategy"]["tier"], cfg["strategy"]["route"]
        p = lambda n: os.path.join(self.data_dir, n)
        self.paths = dict(state=p("state.json"), decisions=p("decisions.csv"), fills=p("fills.csv"),
                          nav=p("nav.csv"), shadow=p("shadow_tranche.csv"), events=p("events.csv"),
                          heartbeat=p("heartbeat.json"), lock=p("run.lock"), log=p("runner.log"))
        self._outbox = []

    # ------------------------------------------------------------------ market data (lazy)
    @property
    def md(self):
        if self._md is None:
            self._md = YahooMarketData(start=self._data_start(), ndx_period="max")
        return self._md

    @md.setter
    def md(self, value):
        self._md = value

    def _data_start(self) -> dt.date:
        """Download from the account start minus the shadow warm-up, so the shadow tranche and the review
        checks keep the account's whole history however old it gets."""
        start = self.now_fn().date()
        try:
            st = load_json(self.paths["state"], None)
            if st and st.get("account"):
                start = min(start, _date(st["account"]["start_date"]))
        except Exception:
            pass
        return start - dt.timedelta(days=shadowmod.WARMUP_DAYS + 30)

    # ------------------------------------------------------------------ state and logs
    def _empty_state(self) -> dict:
        return {"account": None,
                "runner": {"last_decision_date": None, "last_risk_on": None, "last_decision": None,
                           "alerts_sent": {}, "last_tick_utc": None, "tier": self.tier, "route": self.route,
                           "log_backlog": []}}

    def load_state(self) -> dict:
        try:
            st = load_json(self.paths["state"], None)
        except (ValueError, OSError) as e:
            raise StateCorrupt(f"{self.paths['state']} cannot be read ({e}). Restore it from state.json.bak "
                               f"in the same folder, or start over with: python paper/runner.py init --force")
        if st is None:
            return self._empty_state()
        if not isinstance(st, dict) or "runner" not in st or "account" not in st:
            raise StateCorrupt(f"{self.paths['state']} is not a paper-account state file")
        st["runner"].setdefault("alerts_sent", {})
        st["runner"].setdefault("log_backlog", [])
        return st

    def save_state(self, st: dict):
        save_json_atomic(self.paths["state"], st, backup=True)

    def _log_row(self, name: str, row: dict):
        self._outbox.append([name, json.loads(json.dumps(row, default=str))])

    def _event(self, kind: str, detail: str):
        self._log_row("events", {"time_utc": _utc_now(), "kind": kind, "detail": detail})

    def commit(self, st) -> dict:
        """Save state (holding every unwritten log row), then write the CSV logs. Rows that cannot be
        written (file open in another program) stay in state.json and the next run retries them.
        Returns {log name: error} for the logs that could not be written."""
        r = st["runner"]
        r["log_backlog"] = r.get("log_backlog", []) + self._outbox
        self._outbox = []
        self.save_state(st)
        if not r["log_backlog"]:
            return {}
        blocked, left = {}, []
        for name, row in r["log_backlog"]:
            if name in blocked:
                left.append([name, row])
                continue
            fields, key = LOGS[name]
            try:
                append_unique_csv(self.paths[name], row, fields, key)
            except OSError as e:
                blocked[name] = e
                left.append([name, row])
        r["log_backlog"] = left
        self.save_state(st)
        return blocked

    @staticmethod
    def backing_off(st, key: str, now) -> bool:
        until = (st["runner"].get("backoff") or {}).get(key)
        return bool(until) and now < pd.Timestamp(until)

    @staticmethod
    def set_backoff(st, key: str, now, delay: pd.Timedelta):
        st["runner"].setdefault("backoff", {})[key] = (now + delay).isoformat()

    @staticmethod
    def clear_backoff(st, key: str):
        b = st["runner"].get("backoff")
        if b and key in b:
            del b[key]
            if not b:
                st["runner"].pop("backoff")

    def _alert_once(self, st, key: str, subject: str, body: str = "", level: str = "WARN") -> bool:
        sent = st["runner"].setdefault("alerts_sent", {})
        if key in sent:
            return False
        sent[key] = _utc_now()
        self.alerter.alert(subject, body, level)
        return True

    # ------------------------------------------------------------------ unscheduled closures
    def apply_closures(self, st):
        """Rebuild the calendar with closures detected from market data (e.g. a national day of mourning)."""
        found = st["runner"].get("detected_closures", [])
        self.cal = NYSECalendar(self._extra_holidays + found, self._extra_early)

    def is_unscheduled_closure(self, d) -> bool:
        """True if ^NDX has no bar on d but has bars after it, and no strategy ETF traded on d."""
        try:
            ndx = self.md.ndx_closes()
        except DataError:
            return False
        ts = pd.Timestamp(d)
        if ts in ndx.index or not (ndx.index > ts).any():
            return False
        for s in strategy.instruments(self.tier, "US"):
            try:
                if self.md.close_on(s, d) is not None:
                    return False
            except DataError:
                pass
        return True

    def _later_bars_possible(self, d, now) -> bool:
        """A closure can only be confirmed once the next session has opened (its bars then exist)."""
        nxt = self.cal.next_session(d)
        opened = pd.Timestamp(dt.datetime.combine(nxt, dt.time(9, 45))).tz_localize(ET)
        return now >= opened

    def record_closure(self, st, d, now=None):
        r = st["runner"]
        lst = r.setdefault("detected_closures", [])
        if str(d) in lst:
            return
        was_decision_day = self.cal.is_decision_day(d)
        lst.append(str(d))
        self.apply_closures(st)
        acct = st["account"]
        lost = [p for p in acct["pending"] if p.get("execute_on") == str(d)]
        if lost:                                     # orders that were to fill on the closed day
            acct["pending"] = [p for p in acct["pending"] if p.get("execute_on") != str(d)]
            dd = lost[0]["decision_date"]
            r["redo"] = dd                           # the decision is made again at the next opportunity
            r["last_risk_on"] = None                 # nothing was traded: re-infer from holdings
            for p in lost:
                self._event("cancelled", f"{p['order_id']}: {d} turned out to be a market closure")
        body = f"no trading on {d} (an unscheduled NYSE closure, e.g. a day of mourning); treated as a holiday"
        if was_decision_day:
            new_d = self.cal.week_decision_day(d)
            if new_d is not None:
                r.setdefault("closure_shift", {})[str(new_d)] = str(d)
                body += (f". That week's decision day becomes {new_d}: it is made from {new_d}'s official close "
                         f"right away and trades at the next close")
            self.alerter.withdraw(f"weekly decision for {d}", f"{d} turned out to be a market closure")
        if lost:
            body += (f". The {len(lost)} order(s) queued for {d} could not fill, so the decision for "
                     f"{lost[0]['decision_date']} is made again right away and trades at the next close.")
        self._event("closure", body)
        self.alerter.alert(f"unscheduled market closure on {d}", body + " Announced closures can be added to "
                           "schedule.extra_holidays in paper/config.toml.", level="WARN" if lost else "INFO")

    # ------------------------------------------------------------------ timing
    def close_dt(self, d) -> pd.Timestamp:
        return pd.Timestamp(dt.datetime.combine(_date(d), self.cal.close_time(d))).tz_localize(ET)

    @staticmethod
    def eu_close_dt(d) -> pd.Timestamp:
        return pd.Timestamp(dt.datetime.combine(_date(d), EU_CLOSE)).tz_localize("Europe/London").tz_convert(ET)

    def close_ts(self, symbol, d) -> pd.Timestamp:
        return self.close_dt(d) if _us_listed(symbol) else self.eu_close_dt(d)

    def window(self, d):
        sch = self.cfg["schedule"]
        start = self.close_dt(d) - pd.Timedelta(minutes=sch["signal_minutes_before_close"])
        return start, start + pd.Timedelta(minutes=sch["window_minutes"])

    def in_window(self, now, d) -> bool:
        if not self.cal.is_session(d):
            return False
        s, e = self.window(d)
        return s <= now < e

    def ucits_ready(self, d) -> pd.Timestamp:
        return self.close_dt(d) + pd.Timedelta(minutes=self.cfg["schedule"]["ucits_decide_minutes_after_close"])

    def official_ready(self, d) -> pd.Timestamp:
        """From when d's decision can be made from its official close (catch-up, or the UCITS route)."""
        if self.route == "US":
            return self.close_dt(d) + pd.Timedelta(minutes=self.cfg["schedule"]["settle_minutes_after_close"])
        return self.ucits_ready(d)

    def decision_deadline(self, d) -> pd.Timestamp:
        """When the decision for d is due: the end of its window (US) or its UCITS decision time."""
        return self.window(d)[1] if self.route == "US" else self.ucits_ready(d)

    def next_moc_session(self, now) -> dt.date:
        """The first session whose market-on-close cut-off (close - 10 min) is still ahead of `now`."""
        d = now.date()
        if not self.cal.is_session(d):
            d = self.cal.next_session(d)
        while now > self.close_dt(d) - pd.Timedelta(minutes=MOC_CUTOFF_MINUTES):
            d = self.cal.next_session(d)
        return d

    def last_complete_session(self, now) -> dt.date:
        today = now.date()
        delay = pd.Timedelta(minutes=self.cfg["schedule"]["settle_minutes_after_close"])
        if self.cal.is_session(today) and now >= self.close_dt(today) + delay:
            return today
        return self.cal.prev_session(today)

    def _existed_before(self, st, ts) -> bool:
        r = st["runner"]
        init = r.get("initialized_et")
        t0 = (pd.Timestamp(init) if init else
              pd.Timestamp(dt.datetime.combine(_date(st["account"]["start_date"]), dt.time(0, 0))).tz_localize(ET))
        return t0 < ts

    # ------------------------------------------------------------------ init
    def init_account(self, cash: float | None = None, force: bool = False, start=None) -> dict:
        os.makedirs(self.data_dir, exist_ok=True)
        with RunLock(self.paths["lock"]):
            if os.path.exists(self.paths["state"]):
                try:
                    exists = self.load_state()["account"] is not None
                except StateCorrupt:
                    exists = True                    # unreadable: --force may still archive it
                if exists:
                    if not force:
                        raise RuntimeError("a paper account already exists here; use --force to archive it and start over")
                    arch = os.path.join(self.data_dir, f"archive_{dt.datetime.now():%Y%m%d_%H%M%S}")
                    os.makedirs(arch)
                    for n in os.listdir(self.data_dir):
                        src = os.path.join(self.data_dir, n)
                        if os.path.isfile(src) and n != "run.lock":
                            shutil.move(src, os.path.join(arch, n))
            now = self.now_fn()
            start = _date(start) if start else now.date()
            cash = float(cash if cash is not None else self.cfg["account"]["starting_cash"])
            acct = SimAccount(cash=cash, start_date=str(start), last_processed=str(self.cal.prev_session(start)))
            st = self._empty_state()
            st["account"] = acct.to_dict()
            st["runner"].update(initialized_utc=_utc_now(), initialized_et=now.isoformat(),
                                shadow_segments=[self._segment(start)])
            self._event("init", f"paper account started with ${cash:,.2f} on {start} ({self.tier}, {self.route} route)")
            self.commit(st)
            return st

    def _segment(self, start) -> dict:
        return dict(start=str(start), tier=self.tier, route=self.route, cost_bps=self.cfg["account"]["cost_bps"])

    # ------------------------------------------------------------------ the scheduled entry point
    def tick(self) -> dict:
        now = self.now_fn()
        out = {"now_et": now.isoformat(), "actions": [], "booked": []}
        with RunLock(self.paths["lock"]):
            st = self.load_state()
            if st["account"] is None:
                raise NotInitialized("no paper account yet: run `python paper/runner.py init`")
            r = st["runner"]
            snapshot = lambda: json.dumps({**st, "runner": {k: v for k, v in r.items() if k != "last_tick_utc"}},
                                          sort_keys=True, default=str)
            before = snapshot()
            self.apply_closures(st)
            self._config_change(st, now, out)
            self._book_sessions(st, now, out, force=self.decision_due(st, now))
            try:
                self._maybe_decide(st, now, out)
            except (DataError, ValueError) as e:
                d = self.cal.latest_decision_day(now.date())
                out["actions"].append(f"decision for {d} not made: {e}")
                retry = ("It is retried every 5 minutes inside the decision window, otherwise every 30 minutes, and "
                         "made from the official close once that is published." if self.route == "US"
                         else "It is retried every 30 minutes.")
                self._alert_once(st, f"decide-error:{d}:{now.date()}", f"weekly decision for {d} failed",
                                 f"{type(e).__name__}: {e}. {retry}")
            self._health(st, now, out)
            if out["booked"] or (r.get("shadow_dirty") and not self.backing_off(st, "shadow", now)):
                self._update_shadow(st, now, out)
            r["last_tick_utc"] = _utc_now()
            flush_due = bool(r.get("log_backlog")) and not self.backing_off(st, "logs", now)
            if not (self._outbox or flush_due or snapshot() != before):
                out["idle"] = True                   # nothing changed: only the heartbeat is written
                blocked = {}
            else:
                blocked = self.commit(st)
                if not r.get("log_backlog") and "logs" in (r.get("backoff") or {}):
                    self.clear_backoff(st, "logs")
                    self.save_state(st)
            if blocked:
                for name, e in blocked.items():
                    self._alert_once(st, f"log-blocked:{name}:{now.date()}", f"cannot write {name}.csv",
                                     f"{type(e).__name__} on {os.path.basename(getattr(e, 'filename', '') or name + '.csv')}"
                                     f". Is it open in Excel? The rows are kept in state.json and written once the "
                                     f"file is closed (retried every 30 minutes). To look at a log while the runner "
                                     f"is active, open a copy of it.")
                out["actions"].append("log files blocked: " + ", ".join(blocked))
                self.set_backoff(st, "logs", now, BACKOFF["logs"])
                self.save_state(st)
            try:
                save_json_atomic(self.paths["heartbeat"], {"last_tick_utc": r["last_tick_utc"],
                                                           "last_tick_et": now.isoformat(),
                                                           "last_processed": st["account"]["last_processed"],
                                                           "last_decision_date": r["last_decision_date"]})
            except OSError as e:                     # the watchdog will notice a stale heartbeat
                out["actions"].append(f"heartbeat not written: {e}")
        return out

    def _config_change(self, st, now, out):
        r, acct = st["runner"], st["account"]
        segs = r.get("shadow_segments")
        if not segs:
            segs = r["shadow_segments"] = [dict(start=acct["start_date"], tier=r.get("tier") or self.tier,
                                                route=r.get("route") or self.route,
                                                cost_bps=self.cfg["account"]["cost_bps"])]
        if (r.get("tier"), r.get("route")) != (self.tier, self.route):
            self._event("config", f"tier/route changed {r.get('tier')}/{r.get('route')} -> {self.tier}/{self.route}")
            self.alerter.alert("tier/route changed",
                               f"{r.get('tier')}/{r.get('route')} -> {self.tier}/{self.route}; the next decision "
                               f"trades into the new instruments. The shadow tranche keeps its history and "
                               f"follows the new settings from {acct['last_processed']}.", level="INFO")
            r["tier"], r["route"] = self.tier, self.route
            r["last_risk_on"] = None          # re-infer from holdings so the next decision trades fully
            out["actions"].append("config change recorded")
        last = segs[-1]
        cur = self._segment(acct["last_processed"])
        if (last["tier"], last["route"], last["cost_bps"]) != (cur["tier"], cur["route"], cur["cost_bps"]):
            if _date(acct["last_processed"]) < _date(acct["start_date"]):
                segs[-1] = self._segment(acct["start_date"])      # nothing booked yet: just replace it
            else:
                segs.append(cur)
            r["shadow_dirty"] = True

    # ------------------------------------------------------------------ 1. daily booking
    def _price_guard(self, s, ref_date, ref_close, d, px):
        """Message if s moved inconsistently with ^NDX x its leverage since ref_date (else None)."""
        if (d - ref_date).days > 31 or not ref_close > 0:
            return None
        move = px / ref_close
        m = 0.0
        try:
            ndx = self.md.ndx_closes(upto=d)
            base = ndx[ndx.index <= pd.Timestamp(ref_date)]
            if len(base) and len(ndx):
                m = float(ndx.iloc[-1] / base.iloc[-1] - 1)
        except DataError:
            pass
        expected = max(1 + LEVERAGE.get(s, 1) * m, 0.05)
        if not SUSPECT_LO < move / expected < SUSPECT_HI:
            return (f"{s} closed at {px:,.4f} on {d} against {ref_close:,.4f} on {ref_date} ({move - 1:+.1%}) "
                    f"while ^NDX moved {m:+.1%}")
        return None

    def _rebase(self, st, now, out):
        if not self.apply_splits:
            return
        acct = SimAccount.from_dict(st["account"])
        try:
            changes = acct.rebase(self.md, now.date(), fractional=self.cfg["account"]["fractional_shares"])
        except (DataError, NotReady) as e:
            out["actions"].append(f"split check skipped: {e}")
            return
        if not changes:
            return
        st["account"] = acct.to_dict()
        for level, kind, detail in changes:
            self._event(kind, detail)
            out["actions"].append(detail)
            self.alerter.alert("share basis change (split)", detail, level=level)

    def book_session(self, st, d, ffill=()) -> dict:
        acct = SimAccount.from_dict(st["account"])
        a = self.cfg["account"]
        trial = acct.copy()
        rec = trial.process_session(d, self.md, cost_bps=a["cost_bps"], commission=a["commission_per_order"],
                                    withholding=a["dividend_withholding"], fractional=a["fractional_shares"],
                                    close_ts=self.close_ts, guard=self._price_guard if self.apply_splits else None,
                                    ffill_ok=set(ffill), roll_to=self.cal.next_session(d) if ffill else None)
        for f in rec["fills"]:
            self._log_row("fills", f)
        self._log_row("nav", rec["nav_row"])
        for x in rec["dividends"]:
            self._event("dividend", f"{x['symbol']} ex {x['ex_date']}: {x['shares']} x {x['per_share']} = ${x['usd']}")
        for kind, detail in rec["events"]:
            self._event(kind, detail)
        st["account"] = trial.to_dict()
        st["runner"].pop("waiting_for", None)
        r = rec["nav_row"]["daily_return"]
        if abs(r) > 0.40:
            self._alert_once(st, f"jump:{d}", f"NAV moved {r:+.1%} on {d}",
                             "check for a stock split or a bad price in data/nav.csv")
        if rec["expired"]:
            self._alert_once(st, f"expired:{d}", f"{len(rec['expired'])} order(s) expired unfilled on {d}",
                             f"{', '.join(rec['expired'])}; the account keeps its previous holdings until the next decision")
        for c in rec["cuts"]:
            if c["filled"] < 0.95 * c["planned"]:
                self._alert_once(st, f"cut:{c['order_id']}", f"buy of {c['symbol']} cut for lack of cash on {d}",
                                 f"planned {c['planned']:g}, filled {c['filled']:g}; the rest of the cash stays "
                                 f"idle until the next decision")
        self._review_checks(st, d, rec["nav_row"])
        return rec

    def _book_sessions(self, st, now, out, force: bool = False):
        """Book every completed session not yet booked. After a failure it waits (back-off) unless `force`:
        a decision due in this run always tries to book first, so it never plans on stale holdings."""
        r = st["runner"]
        rebased = False
        while True:
            last = _date(st["account"]["last_processed"])
            pending_days = self.cal.sessions(last + DAY, self.last_complete_session(now))
            if not pending_days:
                break
            d = pending_days[0]
            if self.backing_off(st, "book", now) and not force:
                break                                 # the last attempt failed recently: retried later
            if not rebased:
                self._rebase(st, now, out)
                rebased = True
            try:
                self.book_session(st, d)
            except SuspectPrice as e:
                self._suspect(st, e, d, now, out)
                break
            except NotReady as e:
                if self._later_bars_possible(d, now) and self.is_unscheduled_closure(d):
                    self.record_closure(st, d, now)          # calendar rebuilt; continue with the next day
                    out["actions"].append(f"{d}: unscheduled market closure detected")
                    continue
                if e.symbol and self._missing_for_good(e.symbol, d, now):
                    try:
                        self.book_session(st, d, ffill={e.symbol})
                    except SuspectPrice as e2:
                        self._suspect(st, e2, d, now, out)
                        break
                    except (NotReady, DataError) as e2:
                        r["waiting_for"] = str(e2)
                        self.set_backoff(st, "book", now, BACKOFF["book"])
                        out["actions"].append(f"waiting for {d} closing prices ({e2})")
                        break
                    self._alert_once(st, f"no-bar:{e.symbol}:{d}", f"Yahoo has no {e.symbol} close for {d}",
                                     f"later closes exist, so {d} was marked at {e.symbol}'s previous close and any "
                                     f"order for it moved to the next session")
                    self.clear_backoff(st, "book")
                    out["booked"].append(str(d))
                    continue
                r["waiting_for"] = str(e)
                self.set_backoff(st, "book", now, BACKOFF["book"])
                out["actions"].append(f"waiting for {d} closing prices ({e}); retried in 15 minutes")
                break
            except DataError as e:
                r["waiting_for"] = str(e)
                self.set_backoff(st, "book", now, BACKOFF["book"])
                out["actions"].append(f"price data unavailable for {d}: {e}; retried in 15 minutes")
                break
            self.clear_backoff(st, "book")
            out["booked"].append(str(d))
        if out["booked"]:
            out["actions"].append(f"booked sessions {out['booked'][0]}..{out['booked'][-1]}")

    def _suspect(self, st, e, d, now, out):
        st["runner"]["waiting_for"] = str(e)
        self.set_backoff(st, "book", now, BACKOFF["book_suspect"])
        self._alert_once(st, f"suspect:{e.symbol}:{d}", f"booking paused: suspicious {e.symbol} close on {d}",
                         f"{e}. If {e.symbol} had a split Yahoo has not recorded, run: python paper/runner.py "
                         f"split {e.symbol} RATIO (e.g. 2 for 2-for-1). If the move is real, run: python "
                         f"paper/runner.py accept-move {e.symbol} {d}", level="ERROR")
        out["actions"].append(f"booking paused at {d}: {e}")

    def decision_due(self, st, now) -> bool:
        """Would _maybe_decide attempt a decision now?"""
        D = self.cal.latest_decision_day(now.date())
        if self._decided(st, D):
            return False
        if self.route == "US" and now.date() == D and self.in_window(now, D):
            return True
        ra = st["runner"].get("retry_after")
        return now >= self.official_ready(D) and not (ra and now < pd.Timestamp(ra))

    def _missing_for_good(self, symbol, d, now) -> bool:
        """Yahoo skipped d for this symbol: it has later bars and two later sessions are complete."""
        try:
            lb = self.md.last_bar_date(symbol)
        except DataError:
            return False
        return bool(lb and lb > d and len(self.cal.sessions(d + DAY, self.last_complete_session(now))) >= 2)

    # ------------------------------------------------------------------ 2. the weekly decision
    def _decided(self, st, D) -> bool:
        r = st["runner"]
        last = r["last_decision_date"]
        return bool(last) and _date(last) >= D and r.get("redo") != str(D)

    def _maybe_decide(self, st, now, out):
        """US route: inside the decision day's window, from the live NDX - the planned market-on-close trade.
        Otherwise (PC off or asleep in the window), and always on the UCITS route: as soon as the PC is on
        after the decision day's official close has been published, from that close, trading at the next
        close the order can still reach (catch-up). A failed attempt outside the window is retried every
        30 minutes."""
        today, r = now.date(), st["runner"]
        D = self.cal.latest_decision_day(today)
        if self._decided(st, D):
            return
        if self.route == "US" and today == D and self.in_window(now, D):
            try:
                self._decide_or_shift(st, D, live=True, execute_on=D, now=now, out=out)
            except (DataError, ValueError) as e:
                r["live_failed"] = {"date": str(D), "reason": f"{type(e).__name__}: {e}"[:200]}
                raise
            return
        if now < self.official_ready(D):
            return
        ra = r.get("retry_after")
        if ra and now < pd.Timestamp(ra):
            return
        try:
            self._decide_or_shift(st, D, live=False, now=now, out=out, fallback=True,
                                  execute_on=self.next_moc_session(now) if self.route == "US" else None)
            r.pop("retry_after", None)
        except (DataError, ValueError) as e:
            r["retry_after"] = (now + DECIDE_RETRY).isoformat()
            df = r.get("decide_failed") or {}
            if df.get("date") != str(D):
                df = {"date": str(D), "first": f"{now:%a %H:%M}"}
            df.update(last=f"{now:%a %H:%M}", reason=f"{type(e).__name__}: {e}"[:160])
            r["decide_failed"] = df
            raise

    def sessions_late(self, D, execute_on, now) -> int:
        """Trading days between the planned execution and the actual one (0 = on time). US: the decision
        day's own close; UCITS: the next session's European close."""
        if self.route == "US":
            ex = _date(execute_on) if execute_on else D
            return len(self.cal.sessions(D + DAY, ex))
        ex = self.cal.next_session(D)
        while self.eu_close_dt(ex) < now + pd.Timedelta(minutes=MOC_CUTOFF_MINUTES):
            ex = self.cal.next_session(ex)
        return len(self.cal.sessions(D + DAY, ex)) - 1

    def _decide_or_shift(self, st, D, *, live, execute_on, now, out, fallback=False):
        try:
            return self.decide(st, D, live=live, execute_on=execute_on, now=now, out=out, fallback=fallback)
        except ValueError as e:
            # the decision day itself may have been an unscheduled closure (no bar for D, later bars exist)
            if live or "is not the decision date" not in str(e) or not self.is_unscheduled_closure(D):
                raise
            self.record_closure(st, D, now)
            D2 = self.cal.latest_decision_day(now.date())
            if self._decided(st, D2):
                return None
            return self.decide(st, D2, live=False, execute_on=execute_on, now=now, out=out, fallback=fallback)

    def _latest_price(self, symbol, day):
        """(date, close) of the latest bar on or before `day` (within 10 days), else None."""
        for back in range(0, 11):
            d = day - dt.timedelta(days=back)
            px = self.md.close_on(symbol, d)
            if px is not None:
                return d, px
        return None

    def _skipped_days(self, st, D) -> list:
        last = st["runner"].get("last_decision_date")
        if not last:
            return []
        out, d = [], self.cal.next_decision_day(_date(last))
        while d < D:
            out.append(str(d))
            d = self.cal.next_decision_day(d)
        return out

    def decide(self, st, D, *, live: bool, execute_on, now, out, fallback: bool = False):
        r = st["runner"]
        self._rebase(st, now, out)
        acct = SimAccount.from_dict(st["account"])
        if live:
            ndx = self.md.ndx_closes()
            if ndx.index[-1].date() != D:
                raise DataError(f"today's live ^NDX bar is not available yet (last bar {ndx.index[-1].date()})")
            source = f"live {now:%H:%M} ET"
        else:
            ndx = self.md.ndx_closes(upto=D)
            source = f"official close {D}"
        if acct.pending and self.cal.sessions(_date(acct.last_processed) + DAY, self.last_complete_session(now)):
            raise DataError(f"orders from the {acct.pending[0]['decision_date']} decision may already have filled at a "
                            f"close that is not booked yet ({r.get('waiting_for') or 'closing prices not in'}); the "
                            f"new decision waits until they are booked")
        dec = strategy.decide(ndx, D, self.tier, self.route)
        today = now.date()
        syms = sorted(set(dec.targets) | set(acct.positions))
        prices, fx, px_dates = {}, {}, {}
        for s in syms:
            got = self._latest_price(s, today)
            if got is None or not got[1] > 0:
                raise DataError(f"no recent price for {s}")
            f = self.md.fx_to_usd(CURRENCY.get(s, "USD"), today)
            if f is None:
                raise DataError(f"no FX rate for {s}")
            px_dates[s], prices[s], fx[s] = got[0], got[1], f
        if self.apply_splits:                        # an unrecorded split would distort the whole plan
            for s in acct.positions:
                ref = acct.refs.get(s)
                if ref and _date(ref[0]) < px_dates[s] and [s, str(px_dates[s])] not in acct.accepted:
                    msg = self._price_guard(s, _date(ref[0]), float(ref[1]), px_dates[s], prices[s])
                    if msg:
                        raise DataError(f"suspicious price, decision postponed: {msg}")
        prices_usd = {s: prices[s] * fx[s] for s in syms}
        _, cur0 = ordmod.portfolio_weights(acct.positions, prices_usd, acct.cash)
        prev = r["last_risk_on"]
        inferred = prev is None
        if inferred:
            prev = ordmod.infer_risk_on(cur0, dec.sleeve_symbol)
        switch = bool(prev) != dec.risk_on
        sc, ac = self.cfg["strategy"], self.cfg["account"]
        orders, skipped, nav, cur = ordmod.plan_orders(
            dec.targets, acct.positions, prices, fx, acct.cash, band=sc["band"], switch=switch,
            cash_buffer=sc["cash_buffer"], fractional=ac["fractional_shares"], cost_bps=ac["cost_bps"],
            commission=ac["commission_per_order"])
        self._sanity(orders, nav, acct, today)
        group = None
        if execute_on is not None and any(not _us_listed(o.symbol) for o in orders):
            execute_on = None        # European listings (e.g. after a route change): all legs fill together later
        if execute_on is None:
            group = strategy.instruments(self.tier, self.route) if self.route == "UCITS" else []
        when = f"executed at the {execute_on} close" if execute_on else "executed at the next close on which all listings trade"
        shifted = r.get("closure_shift", {}).get(str(D))
        existed = self._existed_before(st, self.decision_deadline(D))
        redo = r.get("redo") == str(D)
        n_late = self.sessions_late(D, execute_on, now)
        lf, dfl = r.get("live_failed") or {}, r.get("decide_failed") or {}
        causes = []
        if self.route == "US":
            causes.append(f"the live decision in the window failed ({lf.get('reason')})" if lf.get("date") == str(D)
                          else "no run during the decision window (PC off or asleep)")
        if dfl.get("date") == str(D):
            causes.append(f"attempts from {dfl['first']} to {dfl['last']} ET failed ({dfl.get('reason')})")
        elif self.route != "US":
            causes.append("no run before the next European close (PC off or asleep)")
        why_late = "; ".join(causes)
        if not existed:
            source += f" (initial entry of a new account, {when})"
        elif shifted:
            source += f" (decision day moved here: {shifted} was an unscheduled closure; {when})"
        elif redo:
            source += f" (made again: the earlier orders could not fill; {when})"
        elif fallback and n_late:
            source += f" (catch-up: {why_late}; {when}, {n_late} trading day(s) late)"
        attempts = r.setdefault("attempts", {})
        n = attempts.get(str(D), 0) + 1
        attempts[str(D)] = n
        for k in sorted(attempts)[:-10]:
            del attempts[k]
        cancelled = acct.queue(orders, D, execute_on, not_before=now if execute_on is None else None, group=group,
                               px_date={s: str(v) for s, v in px_dates.items()}, tag="" if n == 1 else f"r{n}")
        skipped_days = self._skipped_days(st, D) if self._existed_before(st, self.decision_deadline(D)) else []
        row = dict(decision_date=str(D), run_time_et=now.isoformat(timespec="seconds"), signal_source=source,
                   tier=self.tier, route=self.route, ndx_signal=round(dec.ndx, 2), ema100_prev=round(dec.ema100_prev, 2),
                   ema100=round(dec.ema100, 2), pct_vs_ema=round(dec.ndx / dec.ema100 - 1, 5), vol20=round(dec.vol20, 5),
                   er20=round(dec.er20, 5), risk_on=dec.risk_on, prev_risk_on=bool(prev) if not inferred else f"{bool(prev)} (inferred)",
                   switch=switch, chop_half=dec.chop_half, seasonal_half=dec.seasonal_half, engine=round(dec.engine, 5),
                   targets={k: round(v, 5) for k, v in dec.targets.items()}, nav_est=round(nav, 2),
                   current_weights={k: round(v, 4) for k, v in cur.items()},
                   orders=[o.as_dict() for o in orders], skipped_in_band=skipped,
                   execute_on=str(execute_on) if execute_on else "next close on which all listings trade",
                   status="queued" if orders else "no trades needed",
                   notes="; ".join(x for x in [f"superseded {len(cancelled)} unfilled orders" if cancelled else "",
                                               f"skipped decision days {skipped_days}" if skipped_days else ""] if x))
        self._log_row("decisions", row)
        st["account"] = acct.to_dict()
        r.update(last_decision_date=str(D), last_risk_on=dec.risk_on, last_decision=row)
        if r.get("redo") == str(D):
            r.pop("redo")
        r.pop("live_failed", None)
        r.pop("decide_failed", None)
        state = "RISK-ON" if dec.risk_on else "RISK-OFF"
        msg = f"decision {D}: {state} ({source}) -> {len(orders)} order(s) for {row['execute_on']}"
        self._event("decision", msg)
        out["actions"].append(msg)
        if fallback and n_late and existed and not shifted and not redo:
            warn = n_late >= LATE_WARN_SESSIONS
            target = (f"the {execute_on} close" if execute_on else
                      "the next European close on which all its listings trade")
            self.alerter.alert(f"weekly decision for {D} made {n_late} trading day(s) late",
                               f"{why_late[0].upper() + why_late[1:]}, so it trades at {target}. "
                               + ("Trades this late follow the strategy noticeably less closely (in the backtest a "
                                  "week's delay cost 5-7 points a year)." if warn else
                                  "In the backtest a delay of 1-3 trading days barely changed the return."),
                               level="WARN" if warn else "INFO", notify=warn)
        if skipped_days:
            self._event("skipped", f"weekly decisions {skipped_days} were never made (no run in time); the account "
                                   f"moved straight to the decision for {D}")
            self._alert_once(st, f"skipped:{D}", f"{len(skipped_days)} weekly decision(s) were skipped",
                             f"{', '.join(skipped_days)} passed with no run (PC off?). The account follows the latest "
                             f"decision ({D}); the rules' trades in between did not happen.")
        if switch:
            self.alerter.alert(f"{state} switch on {D}",
                               f"NDX {dec.ndx:,.0f} vs EMA-100 {dec.ema100:,.0f}; targets {row['targets']}", level="INFO")
        self.alerter.ping()
        return row

    @staticmethod
    def _sanity(orders, nav, acct=None, today=None):
        buys = sum(o.est_notional_usd for o in orders if o.side == "BUY")
        for o in orders:
            if not (o.qty > 0 and o.est_notional_usd <= 1.10 * nav):
                raise ValueError(f"order failed sanity check: {o.as_dict()} (NAV {nav:,.2f})")
        if buys > 1.05 * nav:
            raise ValueError(f"buys {buys:,.2f} exceed NAV {nav:,.2f}")
        if acct is not None and today is not None and acct.positions and acct.last_nav > 0 and acct.last_processed \
                and (today - _date(acct.last_processed)).days <= 10 and not 0.5 < nav / acct.last_nav < 1.6:
            raise ValueError(f"the planning NAV ${nav:,.2f} is far from the last booked NAV ${acct.last_nav:,.2f}: "
                             f"prices or positions look inconsistent (a split?); decision postponed")

    # ------------------------------------------------------------------ 3. health
    def _review_checks(self, st, d, nav_row):
        """Pre-committed review trigger (drawdown) and the 12-month flag, on every booked session."""
        acct, r = st["account"], st["runner"]
        trig = self.cfg["review"]["drawdown_trigger"].get(self.tier, 0.35)
        dd = nav_row["drawdown"]
        if dd <= -trig:
            self._alert_once(st, f"dd:{acct['high_water_date']}", f"REVIEW TRIGGER: drawdown {dd:.1%} on {d}",
                             f"beyond the pre-committed {-trig:.0%} for {self.tier}; review, do not re-tune.", level="ERROR")
        hist = acct.get("nav_hist") or {}
        keys = sorted(hist)
        if len(keys) > 252:
            r12 = hist[keys[-1]] / hist[keys[-253]] - 1
            if r12 <= -self.cfg["review"]["rolling_12m_flag"]:
                self._alert_once(st, f"r12:{_date(d):%Y-%m}", f"12-month return {r12:.1%}",
                                 "monitoring flag (not a trigger)", level="WARN")

    def _health(self, st, now, out):
        """Missed windows are not errors: the decision is caught up as soon as the PC is on (an INFO line, or a
        WARN when it trades 4+ days late) and a failed attempt raises 'decision failed'. Here: booking stuck,
        the drawdown review trigger and the 3-years-behind-QQQ check."""
        today, r = now.date(), st["runner"]
        last_booked = _date(st["account"]["last_processed"])
        lag = len(self.cal.sessions(last_booked + DAY, self.last_complete_session(now)))
        if lag >= 2:
            self._alert_once(st, f"booking-lag:{today}", f"{lag} sessions not booked",
                             f"nothing booked since {last_booked}: {r.get('waiting_for') or 'closing prices unavailable'}")
        acct = st["account"]
        hw = acct["high_water"]
        dd = acct["last_nav"] / hw - 1 if hw else 0.0
        trig = self.cfg["review"]["drawdown_trigger"].get(self.tier, 0.35)
        if dd <= -trig:
            self._alert_once(st, f"dd:{acct['high_water_date']}", f"REVIEW TRIGGER: drawdown {dd:.1%}",
                             f"beyond the pre-committed {-trig:.0%} for {self.tier}; review, do not re-tune.", level="ERROR")
        self._three_year_check(st, now)

    def _three_year_check(self, st, now):
        """Rulebook review trigger 'three years behind QQQ with no bear market to explain it' (checked monthly)."""
        acct, r = st["account"], st["runner"]
        hist = acct.get("nav_hist") or {}
        keys = sorted(hist)
        month = f"{now:%Y-%m}"
        if len(keys) < 757 or r.get("qqq3y_checked") == month or self.backing_off(st, "qqq3y", now):
            return
        d0, d1 = keys[-757], keys[-1]
        acc = hist[d1] / hist[d0] - 1
        try:
            q = self.md.adj_closes(["QQQ"], d0, d1)["QQQ"].dropna()
            qret = float(q.iloc[-1] / q.iloc[0] - 1)
        except Exception as e:                       # e.g. Yahoo unreachable right after a wake: retry in an hour
            self.set_backoff(st, "qqq3y", now, BACKOFF["qqq3y"])
            out_msg = f"3-year QQQ review check postponed: {type(e).__name__}"
            self._event("review-check", out_msg)
            return
        r["qqq3y_checked"] = month
        self.clear_backoff(st, "qqq3y")
        if acc < qret:
            self._alert_once(st, f"qqq3y:{month}", "REVIEW CHECK: three years behind QQQ",
                             f"account {acc:+.1%} vs QQQ {qret:+.1%} (total return) from {d0} to {d1}. The rulebook "
                             f"trigger is 'three years behind QQQ with no bear market to explain it': check whether a "
                             f"bear market explains it; do not re-tune.", level="WARN")

    # ------------------------------------------------------------------ 4. shadow tranche
    def _update_shadow(self, st, now, out):
        acct, r = st["account"], st["runner"]
        end = _date(acct["last_processed"])
        segs = r.get("shadow_segments") or [self._segment(acct["start_date"])]
        if end < _date(segs[0]["start"]):
            r.pop("shadow_dirty", None)
            return
        try:
            parts, levels = [], {}
            for i, sg in enumerate(segs):
                s0 = _date(sg["start"])
                s1 = min(_date(segs[i + 1]["start"]), end) if i + 1 < len(segs) else end
                if s0 > end or s1 < s0:
                    continue
                df = shadowmod.compute(self.md, self.cal, sg["tier"], sg["route"], s0, s1, cost_bps=sg["cost_bps"])
                if not len(df):
                    continue
                navcols = [c for c in df.columns if c == "tranche_nav" or c.startswith("nav_")]
                for c in navcols:
                    df[c] = df[c] * levels.get(c, 100.0) / 100.0
                if parts:
                    df = df.iloc[1:]                 # the boundary day belongs to the previous segment
                if len(df):
                    levels = {c: float(df[c].iloc[-1]) for c in navcols}
                    parts.append(df.assign(settings=f"{sg['tier']}/{sg['route']}/{sg['cost_bps']:g}bp"))
            if not parts:
                return
            df = pd.concat(parts)
        except DataError as e:                       # transient: retried within the hour
            r["shadow_dirty"] = True
            self.set_backoff(st, "shadow", now, BACKOFF["shadow"])
            out["actions"].append(f"shadow tranche not updated: {e}")
            return
        except Exception as e:
            r["shadow_dirty"] = True
            self.set_backoff(st, "shadow", now, BACKOFF["shadow"])
            out["actions"].append(f"shadow tranche not updated: {e}")
            self._alert_once(st, f"shadow-error:{now:%Y-%m}", "shadow tranche not updated",
                             f"{type(e).__name__}: {e}", level="WARN")
            return
        rows = [dict(date=str(i.date()), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rr.items()})
                for i, rr in df.iterrows()]
        try:
            write_csv(self.paths["shadow"], rows, ["date"] + list(df.columns))
        except OSError as e:
            r["shadow_dirty"] = True
            self.set_backoff(st, "shadow", now, BACKOFF["shadow"])
            self._alert_once(st, f"log-blocked:shadow:{now.date()}", "cannot write shadow_tranche.csv",
                             f"{type(e).__name__}. Is it open in Excel? It is rewritten once the file is closed "
                             f"(retried hourly).")
            return
        r.pop("shadow_dirty", None)
        self.clear_backoff(st, "shadow")
        r["shadow_last"] = {"date": rows[-1]["date"], "tranche_return": float(df["tranche_nav"].iloc[-1]) / 100 - 1,
                            "friday_book_return": float(df["nav_Fri"].iloc[-1]) / 100 - 1,
                            "account_return": acct["last_nav"] / acct["start_nav"] - 1}

    # ------------------------------------------------------------------ manual corrections (runner commands)
    def manual_split(self, symbol: str, ratio: float, force: bool = False) -> str:
        if not (math.isfinite(ratio) and 1 / 1000 <= ratio <= 1000) or ratio == 1:
            raise ValueError("the split ratio must be a number between 0.001 and 1000 other than 1 (e.g. 2 for 2-for-1)")
        with RunLock(self.paths["lock"]):
            st = self.load_state()
            if st["account"] is None:
                raise NotInitialized("no paper account yet")
            acct = SimAccount.from_dict(st["account"])
            if not force and symbol not in acct.positions and symbol not in {p["symbol"] for p in acct.pending}:
                raise ValueError(f"the account neither holds {symbol} nor has an order for it; nothing to split "
                                 f"(use --force if you really mean it)")
            msg = acct.manual_split(symbol, ratio, fractional=self.cfg["account"]["fractional_shares"])
            st["account"] = acct.to_dict()
            self.clear_backoff(st, "book")          # the next run books right away
            self._event("split", msg + " (entered by the user)")
            self.commit(st)
            return msg

    def accept_move(self, symbol: str, day, force: bool = False) -> str:
        with RunLock(self.paths["lock"]):
            st = self.load_state()
            if st["account"] is None:
                raise NotInitialized("no paper account yet")
            day = _date(day)
            if not force and f"suspect:{symbol}:{day}" not in st["runner"].get("alerts_sent", {}):
                raise ValueError(f"no suspicious {symbol} price on {day} has been flagged; accept-move only confirms "
                                 f"a move named in a 'booking paused' alert (use --force if you really mean it)")
            acct = SimAccount.from_dict(st["account"])
            acct.accepted.append([symbol, str(_date(day))])
            st["account"] = acct.to_dict()
            self.clear_backoff(st, "book")          # the next run books right away
            self._event("accept-move", f"{symbol} {day}: the user confirmed the price move is real")
            self.commit(st)
            return f"{symbol} on {day} accepted; the next run books it"

    # ------------------------------------------------------------------ status
    def status(self, fetch_signal: bool = True) -> str:
        st = self.load_state()
        if st["account"] is None:
            return "No paper account yet. Run: python paper/runner.py init"
        self.apply_closures(st)
        a, r = st["account"], st["runner"]
        now = self.now_fn()
        lines = [f"NDX trend paper account ({self.tier}, {self.route} route) - {now:%Y-%m-%d %H:%M} ET", ""]
        lines.append(f"  NAV ${a['last_nav']:,.2f} on {a['last_processed']}  |  since {a['start_date']}: "
                     f"{a['last_nav'] / a['start_nav'] - 1:+.2%}  |  drawdown {a['last_nav'] / a['high_water'] - 1:+.2%}")
        lines.append(f"  cash ${a['cash']:,.2f}")
        for s, q in sorted(a["positions"].items()):
            lines.append(f"  {s:<7} {q:>10,.2f} shares")
        if a["pending"]:
            lines.append("  pending orders:")
            for p in a["pending"]:
                lines.append(f"    {p['side']:<4} {p['qty']:>9,.2f} {p['symbol']:<7} at the close of "
                             f"{p['execute_on'] or 'the next session on which all its listings trade'}  ({p['reason']})")
        if r.get("shadow_last"):
            sl = r["shadow_last"]
            fri = sl.get("friday_book_return")
            lines.append(f"  to {sl['date']}: account {sl['account_return']:+.2%} | same rules traded on time (Friday "
                         f"book) {fri:+.2%} | 5-weekday tranche (planning basis) {sl['tranche_return']:+.2%}"
                         if fri is not None else
                         f"  vs shadow 5-weekday tranche to {sl['date']}: account {sl['account_return']:+.2%}, "
                         f"tranche {sl['tranche_return']:+.2%}")
        if r.get("waiting_for"):
            lines.append(f"  booking is waiting: {r['waiting_for']}")
        if r.get("log_backlog"):
            lines.append(f"  {len(r['log_backlog'])} log row(s) not yet written (a CSV file was open elsewhere)")
        lines.append("")
        ld = r.get("last_decision")
        if ld:
            lines.append(f"  last decision {ld['decision_date']}: {'RISK-ON' if str(ld['risk_on']) == 'True' else 'RISK-OFF'} "
                         f"(NDX {ld['ndx_signal']:,} vs EMA {ld['ema100']:,}); targets {ld['targets']}")
        today = now.date()
        latest = self.cal.latest_decision_day(today)
        local = lambda ts: ts.to_pydatetime().astimezone().strftime("%a %d %b %H:%M")
        catch_up = lambda d: (f"if the PC is off then, it is made as soon as the PC is on after "
                              f"{local(self.official_ready(d))} your time and trades at the next close")
        if self._decided(st, latest):
            nd = self.cal.next_decision_day(today)
            ws, we = self.window(nd)
            lines.append(f"  next decision {nd} ({nd:%a}): " + (
                f"window {ws:%H:%M}-{we:%H:%M} ET (= {local(ws)} your time); {catch_up(nd)}" if self.route == "US"
                else f"made as soon as the PC is on after {local(self.ucits_ready(nd))} your time; fills at the next "
                     f"European close"))
        elif self.route == "US" and latest == today and now < self.window(latest)[1]:
            ws, we = self.window(latest)
            lines.append(f"  decision due today {latest} ({latest:%a}): window {ws:%H:%M}-{we:%H:%M} ET "
                         f"(= {local(ws)} your time); {catch_up(latest)}")
            nd = latest
        else:
            ready = self.official_ready(latest)
            if now < ready:
                lines.append(f"  decision for {latest} waits for the official close: it is made at the first run after "
                             f"{local(ready)} your time" + (f" and trades at the {self.next_moc_session(ready)} close"
                                                            if self.route == "US" else ""))
            else:
                ra = r.get("retry_after")
                when = f"after {local(pd.Timestamp(ra))} your time (retrying)" if ra and now < pd.Timestamp(ra) \
                    else "at the next run (within 5 minutes)"
                lines.append(f"  decision for {latest} is PENDING: made {when} from {latest}'s official close" +
                             (f", trading at the {self.next_moc_session(now)} close" if self.route == "US" else ""))
            nd = self.cal.next_decision_day(today)
        if fetch_signal:
            try:
                ndx = self.md.ndx_closes()
                sig = strategy.scanner.signals(ndx)
                last = float(ndx.iloc[-1])
                s = sig.iloc[-1]
                if ndx.index[-1].date() == nd:
                    bar = float(sig["ema100"].iloc[-2])     # C > EMA_t  <=>  C > EMA_{t-1}
                else:
                    bar = strategy.reentry_bar(ndx, pd.Timestamp(nd))
                state = "above" if s["above"] else "below"
                lines.append(f"  NDX {last:,.0f} ({ndx.index[-1].date()}) is {state} its EMA-100 ({s['ema100']:,.0f}); "
                             + (f"risk-on at {nd} needs a close above ~{bar:,.0f} ({bar / last - 1:+.1%})" if not s["above"]
                                else f"it stays risk-on while the {nd} close is above ~{bar:,.0f}"))
            except Exception as e:
                lines.append(f"  (live signal unavailable: {e})")
        act = self.alerter.active()
        if act:
            lines += ["", f"  ACTIVE ALERTS ({len(act)}) - clear with: python paper/runner.py ack"] + [f"    {l}" for l in act[-5:]]
        hb = load_json(self.paths["heartbeat"], {})
        lines += ["", f"  last scheduled run: {hb.get('last_tick_utc', 'never')} (UTC)", f"  files: {self.data_dir}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------- historical replay
def replay(cfg: dict, start, end, cash: float, out_dir: str, md=None) -> dict:
    """Run the production code day by day over history (decisions at each decision day's official
    close, fills at that close; UCITS fills at the next close on which all listings trade). Writes the
    same files as live mode into out_dir, which must not be the live data folder."""
    start, end = _date(start), _date(end)
    if same_dir(out_dir, cfg["paths"]["data_dir"]):
        raise ValueError("replay output must not be the live data directory")
    files = ("state.json", "state.json.bak", "decisions.csv", "fills.csv", "nav.csv", "shadow_tranche.csv",
             "events.csv", "alerts.log", "ALERT_ACTIVE.txt", "heartbeat.json", "replay_summary.json")
    if os.path.isdir(out_dir):
        has_state = os.path.exists(os.path.join(out_dir, "state.json"))
        if has_state and not os.path.exists(os.path.join(out_dir, REPLAY_MARKER)):
            raise ValueError(f"{out_dir} holds a paper account that was not written by a replay; choose another --out")
        for n in files:
            if os.path.exists(os.path.join(out_dir, n)):
                os.remove(os.path.join(out_dir, n))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, REPLAY_MARKER), "w", encoding="utf-8") as f:
        f.write("This folder holds a historical replay (python paper/runner.py replay); it is not a live account.\n")
    if md is None:
        md = YahooMarketData(start=start - dt.timedelta(days=shadowmod.WARMUP_DAYS + 30), ndx_period="max")
    clock = {"now": None}
    eng = Engine(cfg, md=md, now_fn=lambda: clock["now"], data_dir=out_dir, quiet=True, apply_splits=False)
    first = eng.cal.sessions(start, end)
    if not first:
        raise ValueError("no sessions in range")
    clock["now"] = eng.close_dt(first[0]) - pd.Timedelta(hours=8)
    st = eng.init_account(cash=cash, start=first[0])
    out = {"actions": [], "booked": []}
    d = first[0]
    while d <= first[-1]:
        if not eng.cal.is_session(d):
            d += DAY
            continue
        clock["now"] = eng.close_dt(d) + pd.Timedelta(minutes=1)
        if eng.md.close_on(strategy.instruments(eng.tier, "US")[0], d) is None and eng.is_unscheduled_closure(d):
            eng.record_closure(st, d, clock["now"])         # e.g. 5 Dec 2018, 9 Jan 2025
            d += DAY
            continue
        D = eng.cal.latest_decision_day(d)
        if not eng._decided(st, D):
            # decision at D's official close; if D itself had no session data (closure), the next session
            # executes it from D's official close, exactly as a live catch-up does
            eng.decide(st, D, live=False, execute_on=(d if eng.route == "US" else None), now=clock["now"],
                       out=out, fallback=(D < d))
        try:
            eng.book_session(st, d)
        except NotReady as e:
            raise DataError(f"replay stopped at {d}: {e}")
        eng.commit(st)
        d += DAY
    eng._update_shadow(st, clock["now"], out)
    eng.commit(st)
    navs = read_csv(eng.paths["nav"])
    vals = pd.Series([float(r["nav"]) for r in navs], index=pd.to_datetime([r["date"] for r in navs]))
    yrs = max((vals.index[-1] - vals.index[0]).days / 365.25, 1 / 365.25)
    summary = dict(start=str(first[0]), end=str(first[-1]), sessions=len(vals), start_nav=cash,
                   end_nav=float(vals.iloc[-1]), total_return=float(vals.iloc[-1] / cash - 1),
                   cagr=float((vals.iloc[-1] / cash) ** (1 / yrs) - 1), max_drawdown=float((vals / vals.cummax() - 1).min()),
                   decisions=len(read_csv(eng.paths["decisions"])), fills=len(read_csv(eng.paths["fills"])),
                   shadow_tranche_return=st["runner"].get("shadow_last", {}).get("tranche_return"),
                   tier=eng.tier, route=eng.route, cost_bps=cfg["account"]["cost_bps"])
    with open(os.path.join(out_dir, "replay_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary
