"""Crash-safe persistence: atomic JSON state, append-if-absent CSV logs, and a run lock."""
from __future__ import annotations

import csv
import json
import os
import shutil
import time

DECISION_FIELDS = ["decision_date", "run_time_et", "signal_source", "tier", "route", "ndx_signal",
                   "ema100_prev", "ema100", "pct_vs_ema", "vol20", "er20", "risk_on", "prev_risk_on",
                   "switch", "chop_half", "seasonal_half", "engine", "targets", "nav_est",
                   "current_weights", "orders", "skipped_in_band", "execute_on", "status", "notes"]
DECISION_KEY = ("decision_date", "run_time_et")      # a decision re-made later (e.g. after a closure) is a new row
FILL_FIELDS = ["fill_date", "decision_date", "order_id", "symbol", "side", "qty", "price", "price_ccy",
               "fx_to_usd", "notional_usd", "cost_usd", "cash_after", "note"]
NAV_FIELDS = ["date", "nav", "cash", "positions_value", "positions", "weights", "dividends_usd",
              "daily_return", "cum_return", "high_water", "drawdown"]
EVENT_FIELDS = ["time_utc", "kind", "detail"]


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, obj, backup: bool = False):
    """Write a temp file, fsync it, then atomically replace `path`. With backup=True the previous file is
    first copied to <path>.bak (a last-good copy to recover from a bad hand edit)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        if backup and os.path.exists(path):
            try:
                shutil.copyfile(path, f"{path}.bak.tmp")
                os.replace(f"{path}.bak.tmp", f"{path}.bak")
            except OSError:
                pass                                    # the backup is best effort
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def read_csv(path) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_unique_csv(path, row: dict, fields: list, key_fields: tuple) -> bool:
    """Append `row` unless a row with the same key already exists. Returns True if written.
    Raises OSError (e.g. PermissionError while the file is open in Excel); callers keep the row and retry."""
    existing = read_csv(path)
    key = tuple(str(_fmt(row.get(k, ""))) for k in key_fields)
    if any(tuple(r.get(k, "") for k in key_fields) == key for r in existing):
        return False
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in fields})
    return True


def write_csv(path, rows: list, fields: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: _fmt(r.get(k, "")) for k in fields})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") if v == v else ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    return v


class RunLock:
    """Exclusive lock file so two runs never overlap.

    The holder keeps the file open for the whole run. On Windows an open file cannot be deleted, so the
    lock of a run that died (killed by the task time limit, log-off, power loss) is removed and retaken
    at once, while a live holder's lock cannot be broken. On other systems a lock older than `stale_s`
    (15 minutes = the scheduled task's time limit) is treated as abandoned."""

    def __init__(self, path, stale_s=900):
        self.path, self.stale_s, self.fd = path, stale_s, None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        for _ in range(3):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
                return self
            except FileExistsError:
                if os.name == "nt":
                    try:
                        os.remove(self.path)            # fails while a live process holds it open
                    except FileNotFoundError:
                        pass
                    except OSError:
                        raise LockBusy(f"another run is in progress ({self._holder()})")
                    continue
                try:
                    age = time.time() - os.path.getmtime(self.path)
                except OSError:
                    continue
                if age > self.stale_s:
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                raise LockBusy(f"another run is in progress ({self._holder()}, {age:.0f}s)")
        raise LockBusy(f"could not acquire {self.path}")

    def _holder(self) -> str:
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.read(80).strip() or "holder unknown"
        except OSError:
            return "holder unknown"

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            try:
                os.remove(self.path)
            except OSError:
                pass
        return False


class LockBusy(RuntimeError):
    pass
