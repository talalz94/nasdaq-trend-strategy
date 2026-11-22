"""
NDX trend strategy - local paper-trading runner (no broker account, no credentials).

    python paper/runner.py init [--cash 10000] [--force]   start a simulated paper account
    python paper/runner.py tick                            what the scheduler runs every 5 minutes
    python paper/runner.py status                          account, pending orders, next decision
    python paper/runner.py replay --start 2026-07-24 [--end YYYY-MM-DD] [--cash 10000] [--out DIR]
    python paper/runner.py ack                             clear active alerts
    python paper/runner.py test-alert                      send a test alert and report each channel
    python paper/runner.py split SYMBOL RATIO              record a split Yahoo has not (e.g. QLD 2)
    python paper/runner.py accept-move SYMBOL YYYY-MM-DD   confirm a price move the guard flagged

Options before the command: --config PATH (default paper/config.toml), --data-dir DIR.
Exit codes: 0 ok, 1 error (logged to data/runner.log and alerted), 2 usage error, 3 another run busy,
4 no account yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paper.alerts import Alerter                             # noqa: E402
from paper.config import DEFAULTS, load_config               # noqa: E402
from paper.storage import LockBusy, load_json, save_json_atomic   # noqa: E402

ERROR_REPEAT_HOURS = 6       # the same error is alerted at most this often (it is logged every time)


def _log(data_dir: str, text: str):
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "runner.log"), "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%SZ} {text}\n")
    except OSError:
        pass


def _report_error(data_dir: str, cfg: dict | None, cmd: str, e: Exception):
    """Log every error; alert (log + ALERT_ACTIVE + toast) at most once per ERROR_REPEAT_HOURS per error."""
    _log(data_dir, f"ERROR in '{cmd}': {type(e).__name__}: {e}\n{traceback.format_exc()}")
    sig = f"{cmd}|{type(e).__name__}|{str(e)[:160]}"
    path = os.path.join(data_dir, "runner_errors.json")
    try:
        seen = load_json(path, {})
    except Exception:
        seen = {}
    now = dt.datetime.now(dt.timezone.utc)
    last = seen.get(sig)
    if last and now - dt.datetime.fromisoformat(last) < dt.timedelta(hours=ERROR_REPEAT_HOURS):
        return
    seen = {k: v for k, v in seen.items() if now - dt.datetime.fromisoformat(v) < dt.timedelta(days=7)}
    seen[sig] = now.isoformat()
    try:
        save_json_atomic(path, seen)
    except OSError:
        pass
    hint = " (check paper/config.toml)" if cfg is None else ""
    try:
        Alerter(cfg or DEFAULTS, data_dir).alert(f"runner error in '{cmd}'{hint}", f"{type(e).__name__}: {e}. "
                                                 f"Details in {os.path.join(data_dir, 'runner.log')}", level="ERROR")
    except Exception:
        pass


def _quick_idle(cfg: dict, data_dir: str) -> bool:
    """True if nothing is due (see paper/quickcheck.py): then only the heartbeat is written. This keeps an idle
    scheduled run at a fraction of a second of CPU; the full engine (and pandas) loads only when needed."""
    from paper.quickcheck import ET, reason_to_run
    now = dt.datetime.now(ET)
    try:
        state = load_json(os.path.join(data_dir, "state.json"), None)
        if reason_to_run(cfg, state, now) is not None:
            return False
        save_json_atomic(os.path.join(data_dir, "heartbeat.json"), {
            "last_tick_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_tick_et": now.isoformat(timespec="seconds"), "quick": True,
            "last_processed": state["account"]["last_processed"],
            "last_decision_date": state["runner"].get("last_decision_date")})
        return True
    except Exception:
        return False                               # anything unusual: let the full engine handle and report it


WAKE_TRIGGERS = 46        # Task Scheduler allows 48 triggers per task: ~10 months of decision days


def _install_info(cfg: dict, data_dir: str) -> dict:
    """What install_task.ps1 needs: the data folder, the window settings and, for -WakeToRun, the exact UTC
    instants one minute into each of the next decision windows (1 pm half-days included)."""
    from zoneinfo import ZoneInfo
    from paper.nyse_calendar import NYSECalendar
    sch = cfg["schedule"]
    cal = NYSECalendar(sch["extra_holidays"], sch["extra_early_closes"])
    et, wakes = ZoneInfo("America/New_York"), []
    now = dt.datetime.now(dt.timezone.utc)
    d = cal.latest_decision_day(now.astimezone(et).date())
    into = 1 if sch["window_minutes"] >= 2 else 0
    while len(wakes) < WAKE_TRIGGERS:
        close = dt.datetime.combine(d, cal.close_time(d), tzinfo=et)
        if cfg["strategy"]["route"] == "UCITS":      # the UCITS decision is due after the US close
            t = close + dt.timedelta(minutes=sch["ucits_decide_minutes_after_close"] + 1)
        else:
            t = close - dt.timedelta(minutes=sch["signal_minutes_before_close"] - into)
        if t > now:
            wakes.append(t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        d = cal.next_decision_day(d)
    return {"data_dir": data_dir, "state_exists": os.path.exists(os.path.join(data_dir, "state.json")),
            "signal_minutes_before_close": sch["signal_minutes_before_close"], "window_minutes": sch["window_minutes"],
            "wake_times_utc": wakes}


def _last_complete_session_et(cfg) -> dt.date:
    """The last NYSE session whose close is settled (US Eastern), whatever this PC's own time zone."""
    from paper.market import now_et
    from paper.nyse_calendar import NYSECalendar
    import pandas as pd
    sch = cfg["schedule"]
    cal = NYSECalendar(sch["extra_holidays"], sch["extra_early_closes"])
    now = now_et()
    today = now.date()
    if cal.is_session(today):
        close = pd.Timestamp(dt.datetime.combine(today, cal.close_time(today))).tz_localize("America/New_York")
        if now >= close + pd.Timedelta(minutes=sch["settle_minutes_after_close"]):
            return today
    return cal.prev_session(today)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python paper/runner.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default paper/config.toml)")
    ap.add_argument("--data-dir", help="where state and logs live (default paper/data)")
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{init,tick,status,replay,ack,test-alert,split,accept-move}")
    p = sub.add_parser("init", help="start a simulated paper account")
    p.add_argument("--cash", type=float)
    p.add_argument("--force", action="store_true", help="archive the existing account and start over")
    sub.add_parser("tick", help="scheduled run: book closes, decide when due, health checks")
    sub.add_parser("status", help="show the account and the next decision")
    p = sub.add_parser("replay", help="run the same code over history")
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--out", default=os.path.join(ROOT, "paper", "samples", "replay"))
    sub.add_parser("ack", help="clear active alerts (ALERT_ACTIVE.txt)")
    sub.add_parser("test-alert", help="send a test alert and report each channel")
    p = sub.add_parser("split", help="record a split that Yahoo's data does not show (when an alert asks for it)")
    p.add_argument("symbol")
    p.add_argument("ratio", type=float, help="new shares per old share, e.g. 2 for 2-for-1, 0.1 for 1-for-10")
    p.add_argument("--force", action="store_true", help="allow a symbol the account neither holds nor has orders for")
    p = sub.add_parser("accept-move", help="confirm a price move the guard flagged (when an alert asks for it)")
    p.add_argument("symbol")
    p.add_argument("date")
    p.add_argument("--force", action="store_true", help="accept a move that has not been flagged")
    sub.add_parser("install-info")                 # internal: used by install_task.ps1 (kept out of the usage line)
    args = ap.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir) if args.data_dir else DEFAULTS["paths"]["data_dir"]
    cfg = None
    try:
        overrides = {"paths": {"data_dir": data_dir}} if args.data_dir else None
        cfg = load_config(args.config, overrides, must_exist=bool(args.config))
        data_dir = cfg["paths"]["data_dir"]
        if args.cmd == "tick" and _quick_idle(cfg, data_dir):
            return 0                               # nothing due: finished without loading pandas
        from paper.engine import Engine, NotInitialized, replay

        if args.cmd == "install-info":
            print(json.dumps(_install_info(cfg, data_dir)))
            return 0
        if args.cmd == "replay":
            end = args.end or str(_last_complete_session_et(cfg))
            s = replay(cfg, args.start, end, args.cash, os.path.abspath(args.out))
            print(f"replay {s['start']} .. {s['end']} ({cfg['strategy']['tier']}, {cfg['strategy']['route']}): "
                  f"${s['start_nav']:,.0f} -> ${s['end_nav']:,.2f} ({s['total_return']:+.2%}), "
                  f"max drawdown {s['max_drawdown']:.2%}, {s['decisions']} decisions, {s['fills']} fills")
            if s.get("shadow_tranche_return") is not None:
                print(f"shadow 5-weekday tranche over the same period: {s['shadow_tranche_return']:+.2%}")
            print(f"files: {os.path.abspath(args.out)}")
            return 0

        eng = Engine(cfg)
        if args.cmd == "init":
            eng.init_account(cash=args.cash, force=args.force)
            print(eng.status(fetch_signal=True))
        elif args.cmd == "tick":
            out = eng.tick()
            if out["actions"]:
                _log(data_dir, "; ".join(out["actions"]))
                print("\n".join(out["actions"]))
        elif args.cmd == "status":
            text = eng.status(fetch_signal=True)
            print(text)
            if text.startswith("No paper account yet"):
                return 4
        elif args.cmd == "ack":
            print("alerts cleared" if eng.alerter.acknowledge() else "no active alerts")
        elif args.cmd == "test-alert":
            res = eng.alerter.alert("test notification", "if you can read this, desktop alerts work", level="INFO")
            a = cfg["alerts"]
            print("alerts.log: " + ("written" if res.get("alerts.log") else "FAILED"))
            print("desktop notification: " + ("off (alerts.desktop_notifications = false)" if not a["desktop_notifications"]
                                              else "shown - check the Windows notification area" if res.get("desktop")
                                              else "FAILED (PowerShell could not show it)"))
            print("webhook: " + ("not configured" if not a["webhook_url"] else "delivered" if res.get("webhook")
                                 else "FAILED - see alerts.log for the reason"))
            hc = eng.alerter.ping()
            print("healthcheck ping: " + ("not configured" if hc is None else "ok" if hc else "FAILED - see alerts.log"))
        elif args.cmd == "split":
            print(eng.manual_split(args.symbol, args.ratio, force=args.force))
        elif args.cmd == "accept-move":
            print(eng.accept_move(args.symbol, args.date, force=args.force))
        return 0
    except LockBusy as e:
        if args.cmd == "tick":
            _log(data_dir, f"tick skipped: {e}")
            return 0                               # the scheduler simply tries again in 5 minutes
        print(f"{e}; try again in a minute", file=sys.stderr)
        return 3
    except Exception as e:                         # never fail silently
        try:
            from paper.engine import NotInitialized
        except Exception:
            NotInitialized = ()
        if NotInitialized and isinstance(e, NotInitialized):
            print(e)
            return 4
        if isinstance(e, ValueError) and args.cmd in ("split", "accept-move", "init"):
            print(f"error: {e}", file=sys.stderr)      # a typo at the keyboard: shown, not alerted (DELTA-8)
            return 1
        _report_error(data_dir, cfg, args.cmd, e)
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
