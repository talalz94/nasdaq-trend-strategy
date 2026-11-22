"""The stdlib-only pre-check (paper/quickcheck.py) must never skip work: whenever it calls a run idle, the full
engine run at that moment must change nothing. Checked over weeks of simulated time, with the PC on and off,
on both routes and across an unscheduled closure."""
import numpy as np
import pandas as pd
import pytest

from conftest import et, frame, synthetic
from paper.quickcheck import reason_to_run


def boom_then_crash(idx, rng):
    r = rng.normal(0.0015, 0.006, len(idx))
    crash = (idx >= "2026-10-05") & (idx <= "2026-11-06")
    r[crash] = rng.normal(-0.012, 0.006, crash.sum())
    return r


def ucits(frames, eur=1.1):
    f = dict(frames)
    f["LQQ.PA"] = frame((frames["QLD"]["Close"] / eur).values, frames["QLD"].index)
    for us, uc in [("IEF", "CBU0.L"), ("GLD", "IGLN.L")]:
        f[uc] = frames[us].copy()
    f["EURUSD=X"] = frame(np.full(len(frames["QLD"]), eur), frames["QLD"].index)
    return f


def timeline():
    """Runs every 45 minutes, except when the 'PC' is off (some nights and two whole days), plus the windows."""
    t, end, out = et(2026, 9, 28, 8, 0), et(2026, 10, 24, 12, 0), []
    while t < end:
        off = (t.hour >= 23 or t.hour < 7) and t.day % 3 == 0 or t.date() in (pd.Timestamp("2026-10-15").date(),
                                                                              pd.Timestamp("2026-10-16").date())
        if not off:
            out.append(t)
        t += pd.Timedelta(minutes=45)
    for d in ("2026-10-02", "2026-10-09", "2026-10-23"):
        out += [et(*map(int, d.split("-")), 15, 41), et(*map(int, d.split("-")), 15, 46)]
    return sorted(set(out))


@pytest.mark.parametrize("route,closure", [("US", None), ("UCITS", None), ("US", "2026-10-09")])
def test_quick_check_never_skips_work(make_engine, route, closure):
    cal, frames, ndx = synthetic(ndx_returns=boom_then_crash)
    f = ucits(frames) if route == "UCITS" else frames
    if closure:
        ts = pd.Timestamp(closure)
        f = {k: (v.drop(ts) if ts in v.index else v) for k, v in f.items()}
        ndx = ndx.drop(ts)
    eng, clock = make_engine(f, ndx, et(2026, 9, 28, 7, 30), strategy={"route": route})
    eng.init_account(cash=10000)
    quick = 0
    for t in timeline():
        clock["now"] = t
        why = reason_to_run(eng.cfg, eng.load_state(), t)
        out = eng.tick()
        if why is None:
            quick += 1
            assert out.get("idle"), f"{t}: the pre-check said idle but the engine did {out['actions']}"
    assert quick > 0.6 * len(timeline())          # and it does skip the bulk of the runs


@pytest.mark.parametrize("trouble", ["csv-open-in-excel", "bar-never-published"])
def test_quick_check_with_failures_backs_off_and_never_skips_work(make_engine, monkeypatch, trouble):
    """A CSV held open, or a close Yahoo never publishes, is retried after a pause instead of on every run."""
    import paper.engine as E
    cal, frames, ndx = synthetic(ndx_returns=boom_then_crash)
    if trouble == "bar-never-published":
        ts = pd.Timestamp("2026-10-06")
        frames = dict(frames, GLD=frames["GLD"].drop(ts))
    eng, clock = make_engine(frames, ndx, et(2026, 9, 28, 7, 30))
    eng.init_account(cash=10000)
    real = E.append_unique_csv

    def locked(path, row, fields, key):
        if pd.Timestamp("2026-10-05") <= clock["now"].tz_localize(None) < pd.Timestamp("2026-10-07 18:00") \
                and path.endswith("nav.csv"):
            raise PermissionError(13, "Permission denied", path)
        return real(path, row, fields, key)

    if trouble == "csv-open-in-excel":
        monkeypatch.setattr(E, "append_unique_csv", locked)
    full_runs, quick, t = 0, 0, et(2026, 10, 5, 0, 0)
    while t < et(2026, 10, 9, 0, 0):                      # every 5 minutes, like the scheduler
        clock["now"] = t
        why = reason_to_run(eng.cfg, eng.load_state(), t)
        out = eng.tick()
        if why is None:
            quick += 1
            assert out.get("idle"), f"{t}: the pre-check said idle but the engine did {out['actions']}"
        else:
            full_runs += 1
        t += pd.Timedelta(minutes=5)
    st = eng.load_state()
    assert st["account"]["last_processed"] == "2026-10-08"          # it recovered
    assert full_runs < 0.2 * (full_runs + quick)       # retried every 15-30 min, not on every 5-minute run


def test_quick_check_reports_a_drawdown_trigger_after_a_config_change(make_engine, tmp_path):
    cal, frames, ndx = synthetic(ndx_returns=boom_then_crash)
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    st = eng.init_account(cash=10000)
    st["account"].update(high_water=10000.0, high_water_date="2026-09-30", last_nav=8000.0,   # -20%
                         last_processed="2026-10-02")
    st["runner"]["last_decision_date"] = "2026-10-02"              # booked and decided: nothing else is due
    eng.save_state(st)
    now = et(2026, 10, 3, 10, 0)                                    # a Saturday: nothing else is due
    assert reason_to_run(eng.cfg, eng.load_state(), now) is None    # -20% is inside the default -35%
    cfg = dict(eng.cfg, review={"drawdown_trigger": {"F2": 0.15}, "rolling_12m_flag": 0.25})
    assert reason_to_run(cfg, eng.load_state(), now) == "drawdown review trigger"
