"""Configuration: defaults merged with an optional paper/config.toml (copy config.example.toml).

Every value is checked (type, finite, range, known key) so a typo fails loudly at once - the runner then
logs it to data/runner.log and shows an alert - instead of silently changing behaviour."""
from __future__ import annotations

import copy
import datetime as dt
import math
import os
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "strategy": {
        "tier": "F2",            # F1 (QQQ) | F2 (QLD, reference) | F3 (TQQQ, US route only)
        "route": "US",           # US = US-listed ETFs ; UCITS = LQQ.PA/CNDX.L + CBU0.L + IGLN.L
        "band": 0.02,            # skip a sleeve whose weight is within 2 percentage points of target
        "cash_buffer": 0.005,    # keep 0.5% in cash for rounding and costs
    },
    "account": {
        "starting_cash": 10000.0,
        "fractional_shares": False,
        "cost_bps": 5.0,               # spread/slippage per unit traded (the planning basis)
        "commission_per_order": 0.0,
        "dividend_withholding": 0.0,   # e.g. 0.30 to model US withholding on a non-US account
    },
    "schedule": {
        "signal_minutes_before_close": 20,   # 15:40 ET on a normal day, 12:40 ET on a 1 pm close
        "window_minutes": 8,                 # act between 15:40 and 15:48 (MOC cut-off is 15:50)
        "settle_minutes_after_close": 45,    # fill MOC orders once the official close is published
        "ucits_decide_minutes_after_close": 30,
        "extra_holidays": [],
        "extra_early_closes": [],
    },
    "review": {
        "drawdown_trigger": {"F1": 0.25, "F2": 0.35, "F3": 0.45},
        "rolling_12m_flag": 0.25,
    },
    "alerts": {
        "desktop_notifications": True,   # Windows toast via PowerShell (no account needed)
        "webhook_url": "",               # optional: your own webhook (POST JSON {"text": ...}); empty = off
        "healthcheck_url": "",           # optional: dead-man ping URL; empty = off
    },
    "paths": {
        "data_dir": os.path.join(HERE, "data"),
    },
}

VALID_TIERS = ("F1", "F2", "F3")
VALID_ROUTES = ("US", "UCITS")
MOC_CUTOFF_MINUTES = 10          # NYSE market-on-close orders must be in 10 minutes before the close


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None, must_exist: bool = False) -> dict:
    """Defaults <- config file <- overrides. `must_exist` (set when --config is given) makes a missing
    file an error instead of silently running on defaults."""
    path = path or os.path.join(HERE, "config.toml")
    user = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                user = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise ValueError(f"{path} is not valid TOML: {e}") from None
    elif must_exist:
        raise ValueError(f"config file not found: {path}")
    _check_keys(user, DEFAULTS, os.path.basename(path))
    cfg = _merge(DEFAULTS, user)
    cfg = _merge(cfg, overrides or {})
    validate(cfg)
    return cfg


def _check_keys(user, ref, where, prefix=""):
    for k, v in user.items():
        name = f"{prefix}{k}"
        if k not in ref:
            raise ValueError(f"unknown setting '{name}' in {where} (check the spelling; see config.example.toml)")
        if isinstance(ref[k], dict) and name != "review.drawdown_trigger":
            if not isinstance(v, dict):
                raise ValueError(f"'{name}' in {where} must be a [section]")
            _check_keys(v, ref[k], where, name + ".")


def _num(x, name, lo=None, hi=None, lo_open=False, hi_open=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number (got {x!r})")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f"{name} must be {'>' if lo_open else '>='} {lo} (got {x})")
    if hi is not None and (x >= hi if hi_open else x > hi):
        raise ValueError(f"{name} must be {'<' if hi_open else '<='} {hi} (got {x})")
    return x


def _bool(x, name):
    if not isinstance(x, bool):
        raise ValueError(f"{name} must be true or false without quotes (got {x!r})")


def _dates(x, name):
    if not isinstance(x, list):
        raise ValueError(f"{name} must be a list of dates like [\"2025-01-09\"] (got {x!r})")
    for d in x:
        try:
            if not isinstance(d, (str, dt.date)) or (isinstance(d, str) and len(d) != 10):
                raise ValueError
            dt.date.fromisoformat(str(d))
        except ValueError:
            raise ValueError(f"{name}: '{d}' is not a date in the form YYYY-MM-DD") from None


def validate(cfg):
    s, a, sch, rv, al = cfg["strategy"], cfg["account"], cfg["schedule"], cfg["review"], cfg["alerts"]
    if s["tier"] not in VALID_TIERS:
        raise ValueError(f"strategy.tier must be one of {VALID_TIERS} (got {s['tier']!r})")
    if s["route"] not in VALID_ROUTES:
        raise ValueError(f"strategy.route must be one of {VALID_ROUTES} (got {s['route']!r})")
    if s["route"] == "UCITS" and s["tier"] == "F3":
        raise ValueError("F3 has no UCITS route (no UCITS 3x fund exists)")
    _num(s["band"], "strategy.band", 0, 0.2)
    _num(s["cash_buffer"], "strategy.cash_buffer", 0, 0.05)
    _num(a["starting_cash"], "account.starting_cash", 0, lo_open=True)
    _bool(a["fractional_shares"], "account.fractional_shares")
    _num(a["cost_bps"], "account.cost_bps", 0, 200)
    _num(a["commission_per_order"], "account.commission_per_order", 0, 1000)
    _num(a["dividend_withholding"], "account.dividend_withholding", 0, 1, hi_open=True)
    for k in ("signal_minutes_before_close", "window_minutes", "settle_minutes_after_close",
              "ucits_decide_minutes_after_close"):
        if isinstance(sch[k], bool) or not isinstance(sch[k], int):
            raise ValueError(f"schedule.{k} must be a whole number of minutes (got {sch[k]!r})")
    _num(sch["window_minutes"], "schedule.window_minutes", 1, 60)
    _num(sch["signal_minutes_before_close"], "schedule.signal_minutes_before_close", 1, 180)
    if sch["signal_minutes_before_close"] - sch["window_minutes"] < MOC_CUTOFF_MINUTES:
        raise ValueError("the decision window must end at least 10 minutes before the close (the market-on-close "
                         "cut-off): signal_minutes_before_close - window_minutes must be >= 10")
    _num(sch["settle_minutes_after_close"], "schedule.settle_minutes_after_close", 15, 720)
    _num(sch["ucits_decide_minutes_after_close"], "schedule.ucits_decide_minutes_after_close", 0, 720)
    _dates(sch["extra_holidays"], "schedule.extra_holidays")
    _dates(sch["extra_early_closes"], "schedule.extra_early_closes")
    trig = rv["drawdown_trigger"]
    if not isinstance(trig, dict):
        raise ValueError("review.drawdown_trigger must be a table like { F1 = 0.25, F2 = 0.35, F3 = 0.45 }")
    for k, v in trig.items():
        if k not in VALID_TIERS:
            raise ValueError(f"review.drawdown_trigger: unknown tier {k!r}")
        _num(v, f"review.drawdown_trigger.{k} (a positive fraction, e.g. 0.35 for -35%)", 0, 1, True, True)
    _num(rv["rolling_12m_flag"], "review.rolling_12m_flag (a positive fraction)", 0, 1, True, True)
    _bool(al["desktop_notifications"], "alerts.desktop_notifications")
    for k in ("webhook_url", "healthcheck_url"):
        u = al[k]
        if not isinstance(u, str) or (u and not u.lower().startswith(("http://", "https://"))):
            raise ValueError(f"alerts.{k} must be empty or an http(s) URL")
    if not isinstance(cfg["paths"]["data_dir"], str) or not cfg["paths"]["data_dir"].strip():
        raise ValueError("paths.data_dir must be a folder path")
