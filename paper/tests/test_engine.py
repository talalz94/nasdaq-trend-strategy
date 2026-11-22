import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ClockedMarketData, et, frame, synthetic
from paper.storage import LockBusy, RunLock, read_csv


def boom_then_crash(idx, rng):
    r = rng.normal(0.0015, 0.006, len(idx))
    crash = (idx >= "2026-10-05") & (idx <= "2026-11-06")
    r[crash] = rng.normal(-0.012, 0.006, crash.sum())
    return r


@pytest.fixture
def market():
    return synthetic(ndx_returns=boom_then_crash)


def run(eng, clock, *times):
    outs = []
    for t in times:
        clock["now"] = t
        outs.append(eng.tick())
    return outs


def test_live_decision_in_window_is_idempotent_and_settles_at_close(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 2, 15, 30))
    assert read_csv(eng.paths["decisions"]) == []                       # before the window
    run(eng, clock, et(2026, 10, 2, 15, 41), et(2026, 10, 2, 15, 45))
    dec = read_csv(eng.paths["decisions"])
    assert len(dec) == 1 and dec[0]["decision_date"] == "2026-10-02" and dec[0]["risk_on"] == "True"
    assert dec[0]["signal_source"].startswith("live 15:41")
    st = eng.load_state()
    assert st["account"]["pending"] and all(p["execute_on"] == "2026-10-02" for p in st["account"]["pending"])
    run(eng, clock, et(2026, 10, 2, 16, 50))
    st = eng.load_state()
    assert st["account"]["pending"] == []
    fills = read_csv(eng.paths["fills"])
    assert fills and {f["fill_date"] for f in fills} == {"2026-10-02"}
    assert st["account"]["positions"].get("QLD", 0) > 0
    assert st["account"]["last_processed"] == "2026-10-02"
    navs = read_csv(eng.paths["nav"])
    assert [r["date"] for r in navs] == ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
    run(eng, clock, et(2026, 10, 2, 18, 0))                              # nothing new to do
    assert len(read_csv(eng.paths["fills"])) == len(fills)


def test_missed_window_is_caught_up_as_soon_as_the_pc_is_on(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 5, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 9, 16, 30))             # window missed; Friday's official close not settled yet
    st = eng.load_state()
    assert st["runner"]["last_decision_date"] is None and st["runner"]["alerts_sent"] == {}
    run(eng, clock, et(2026, 10, 10, 11, 0))             # the PC is switched on on Saturday
    dec = read_csv(eng.paths["decisions"])
    assert len(dec) == 1 and dec[0]["decision_date"] == "2026-10-09"
    assert "catch-up" in dec[0]["signal_source"] and dec[0]["execute_on"] == "2026-10-12"
    assert float(dec[0]["ndx_signal"]) == pytest.approx(round(ndx[pd.Timestamp("2026-10-09")], 2))
    assert "made 1 trading day(s) late" in open(eng.alerter.log_path, encoding="utf-8").read()
    assert eng.alerter.active() == []                     # routine information, not an alarm
    run(eng, clock, et(2026, 10, 13, 9, 0))              # off over Monday's close: booked the next morning
    assert {f["fill_date"] for f in read_csv(eng.paths["fills"])} == {"2026-10-12"}


def test_catch_up_after_the_market_on_close_cut_off_trades_the_next_day(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 5, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 12, 15, 55))            # first time on: Monday after 15:50 ET
    dec = read_csv(eng.paths["decisions"])
    assert dec[-1]["decision_date"] == "2026-10-09" and dec[-1]["execute_on"] == "2026-10-13"


def test_a_week_late_decision_warns(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 5, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 15, 10, 0))             # off from Friday until Thursday
    assert any("trading day(s) late" in l for l in eng.alerter.active())


def test_new_account_mid_week_joins_with_latest_decision(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 14, 8, 0))
    eng.init_account(cash=5000)
    run(eng, clock, et(2026, 10, 14, 8, 5))
    assert eng.load_state()["runner"]["alerts_sent"] == {}             # not "missed": the account is new
    run(eng, clock, et(2026, 10, 14, 15, 41))
    dec = read_csv(eng.paths["decisions"])
    assert dec[0]["decision_date"] == "2026-10-09" and dec[0]["execute_on"] == "2026-10-14"


def test_risk_off_switch_trades_every_sleeve(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    t = []
    for f in cal.sessions("2026-10-01", "2026-11-20"):
        if cal.is_decision_day(f):
            s = eng.window(f)[0] + pd.Timedelta(minutes=1)
            t += [s, eng.close_dt(f) + pd.Timedelta(minutes=50)]
    run(eng, clock, *t)
    dec = read_csv(eng.paths["decisions"])
    off = [r for r in dec if r["risk_on"] == "False"]
    assert off, "the synthetic crash should produce a risk-off decision"
    first_off = off[0]
    assert first_off["switch"] == "True"
    orders = json.loads(first_off["orders"])
    assert any(o["symbol"] == "QLD" and o["side"] == "SELL" for o in orders)
    assert all(o["reason"] == "switch" for o in orders)
    st = eng.load_state()
    assert st["account"]["positions"].get("QLD", 0) == 0


def test_ucits_route_decides_after_close_and_fills_next_bar(make_engine, market):
    cal, frames, ndx = market
    f = dict(frames)
    eur = 1.1
    f["LQQ.PA"] = frame((frames["QLD"]["Close"] / eur).values, frames["QLD"].index)
    for us, uc in [("IEF", "CBU0.L"), ("GLD", "IGLN.L")]:
        f[uc] = frames[us].copy()
    f["EURUSD=X"] = frame(np.full(len(frames["QLD"]), eur), frames["QLD"].index)
    eng, clock = make_engine(f, ndx, et(2026, 9, 28, 9, 0), strategy={"route": "UCITS"})
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 2, 15, 41))
    assert read_csv(eng.paths["decisions"]) == []                       # UCITS waits for the US close
    run(eng, clock, et(2026, 10, 2, 16, 40))
    dec = read_csv(eng.paths["decisions"])
    assert dec and dec[0]["decision_date"] == "2026-10-02"
    run(eng, clock, et(2026, 10, 5, 17, 0))
    fills = read_csv(eng.paths["fills"])
    assert fills and {x["fill_date"] for x in fills} == {"2026-10-05"}
    lqq = [x for x in fills if x["symbol"] == "LQQ.PA"]
    assert lqq and float(lqq[0]["fx_to_usd"]) == pytest.approx(eur)


def test_concurrent_run_is_blocked(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 2, 9, 0))
    eng.init_account(cash=10000)
    with RunLock(eng.paths["lock"]):
        with pytest.raises(LockBusy):
            eng.tick()


def test_missing_live_bar_alerts_and_does_not_decide(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    stale = ClockedMarketData(frames, ndx[ndx.index < "2026-10-02"], clock)
    eng.md = stale
    run(eng, clock, et(2026, 10, 2, 15, 41))
    st = eng.load_state()
    assert st["runner"]["last_decision_date"] is None
    assert any(k.startswith("decide-error:2026-10-02") for k in st["runner"]["alerts_sent"])


def test_drawdown_review_trigger_alerts_once(make_engine, market):
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    st = eng.init_account(cash=10000)
    st["account"].update(high_water=10000.0, high_water_date="2026-09-30", last_nav=6000.0)
    eng.save_state(st)
    run(eng, clock, et(2026, 10, 1, 10, 0), et(2026, 10, 1, 10, 5))
    keys = [k for k in eng.load_state()["runner"]["alerts_sent"] if k.startswith("dd:")]
    assert keys == ["dd:2026-09-30"]
    assert "REVIEW TRIGGER" in open(eng.paths["log"].replace("runner.log", "alerts.log"), encoding="utf-8").read()
