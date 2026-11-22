"""Regression tests for the adversarial review of the paper package (Nov 2026); each names its finding(s)."""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import ClockedMarketData, et, frame, synthetic
from paper import runner, shadow
from paper.config import load_config
from paper.engine import Engine, replay
from paper.market import FrameMarketData
from paper.storage import read_csv
import scanner

US_ETFS = ["QQQ", "QLD", "TQQQ", "IEF", "GLD"]


def boom_then_crash(idx, rng):
    r = rng.normal(0.0015, 0.006, len(idx))
    crash = (idx >= "2026-10-05") & (idx <= "2026-11-06")
    r[crash] = rng.normal(-0.012, 0.006, crash.sum())
    return r


@pytest.fixture
def market():
    return synthetic(ndx_returns=boom_then_crash)      # F2: QLD .425 on 10-02, .85 on 10-09, risk-off 10-16


def run(eng, clock, *times):
    for t in times:
        clock["now"] = t
        eng.tick()


def without(frames, ndx, day, symbols=None, ndx_too=True):
    ts = pd.Timestamp(day)
    f = {k: (v.drop(ts) if (symbols is None or k in symbols) and ts in v.index else v) for k, v in frames.items()}
    return f, (ndx.drop(ts) if ndx_too else ndx)


def ucits(frames, eur=1.1):
    f = dict(frames)
    f["LQQ.PA"] = frame((frames["QLD"]["Close"] / eur).values, frames["QLD"].index)
    for us, uc in [("IEF", "CBU0.L"), ("GLD", "IGLN.L")]:
        f[uc] = frames[us].copy()
    f["EURUSD=X"] = frame(np.full(len(frames["QLD"]), eur), frames["QLD"].index)
    return f


def make(dirpath, frames, ndx, now, md_cls=ClockedMarketData, md_kw=None, **cfg_over):
    clock = {"now": now}
    over = {"paths": {"data_dir": str(dirpath)}, "alerts": {"desktop_notifications": False}}
    for k, v in cfg_over.items():
        over.setdefault(k, {}).update(v)
    cfg = load_config(os.path.join(str(dirpath), "none.toml"), over)
    md = md_cls(frames, ndx, clock, **(md_kw or {}))
    return Engine(cfg, md=md, now_fn=lambda: clock["now"], data_dir=str(dirpath), quiet=True), clock


def active(eng):
    return "\n".join(eng.alerter.active())


# ---------------------------------------------------------------------------------------------- alerts / timing
def test_account_created_after_the_window_is_not_told_it_missed(make_engine, market):          # SD-4 T9 ROB-04
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 2, 20, 0))          # Saturday 05:00 in Pakistan
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 2, 20, 5), et(2026, 10, 3, 10, 0))
    assert not [k for k in eng.load_state()["runner"]["alerts_sent"] if k.startswith("missed")]
    assert active(eng) == ""
    run(eng, clock, et(2026, 10, 5, 15, 41))
    dec = read_csv(eng.paths["decisions"])
    assert dec[0]["decision_date"] == "2026-10-02" and "initial entry" in dec[0]["signal_source"]


def test_status_after_a_missed_window_points_to_the_next_session(make_engine, market):          # T10
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    clock["now"] = et(2026, 10, 2, 17, 0)
    s = eng.status(fetch_signal=False)
    assert "decision for 2026-10-02 is PENDING: made at the next run (within 5 minutes) from 2026-10-02's "            "official close, trading at the 2026-10-05 close" in s


def test_reentry_bar_on_the_decision_day_uses_yesterdays_ema(make_engine, market):               # FID-5
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    clock["now"] = et(2026, 10, 2, 15, 30)
    s = eng.status(fetch_signal=True)
    bar = scanner.signals(ndx[ndx.index <= "2026-10-02"])["ema100"].iloc[-2]
    assert f"~{bar:,.0f}" in s


# ---------------------------------------------------------------------------------------------- Excel / crash safety
def test_log_file_open_in_excel_blocks_neither_booking_nor_the_decision(make_engine, market, monkeypatch):  # SD-3 ROB-02
    import paper.engine as E
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    real = E.append_unique_csv

    def locked(path, row, fields, key):
        if path.endswith(("nav.csv", "decisions.csv")):
            raise PermissionError(13, "Permission denied", path)
        return real(path, row, fields, key)

    monkeypatch.setattr(E, "append_unique_csv", locked)
    run(eng, clock, et(2026, 10, 1, 16, 50), et(2026, 10, 2, 15, 41), et(2026, 10, 2, 16, 50))
    st = eng.load_state()
    assert st["runner"]["last_decision_date"] == "2026-10-02" and st["account"]["last_processed"] == "2026-10-02"
    assert st["account"]["positions"] and read_csv(eng.paths["fills"])
    assert read_csv(eng.paths["nav"]) == [] and any(n == "nav" for n, _ in st["runner"]["log_backlog"])
    assert any(k.startswith("log-blocked:nav") for k in st["runner"]["alerts_sent"])
    monkeypatch.setattr(E, "append_unique_csv", real)
    run(eng, clock, et(2026, 10, 2, 17, 0))                  # within the 30-minute retry pause: still waiting
    assert eng.load_state()["runner"]["log_backlog"]
    run(eng, clock, et(2026, 10, 2, 17, 25))                 # the pause is over: written
    st = eng.load_state()
    assert st["runner"]["log_backlog"] == []
    dec = read_csv(eng.paths["decisions"])                # the new account's entry (caught up) and the live 10-02 one
    assert read_csv(eng.paths["nav"])[-1]["date"] == "2026-10-02" and [r["decision_date"] for r in dec] == [
        "2026-09-25", "2026-10-02"]


def test_crash_before_the_state_save_leaves_no_phantom_decision_row(make_engine, market, monkeypatch):  # ROB-10
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    real = eng.commit

    def die(st):
        raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(eng, "commit", die)
    with pytest.raises(KeyboardInterrupt):
        run(eng, clock, et(2026, 10, 2, 15, 46))
    monkeypatch.setattr(eng, "commit", real)
    eng._outbox = []                                      # the process died: its memory is gone
    assert read_csv(eng.paths["decisions"]) == [] and eng.load_state()["runner"]["last_decision_date"] is None
    run(eng, clock, et(2026, 10, 5, 15, 42))
    dec = read_csv(eng.paths["decisions"])
    assert len(dec) == 1 and dec[0]["execute_on"] == "2026-10-05"


@pytest.mark.skipif(os.name != "nt", reason="Windows file-lock semantics")
def test_lock_left_by_a_killed_run_is_taken_over_at_once(make_engine, market):                    # ROB-12 T7
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    with open(eng.paths["lock"], "w") as f:
        f.write("pid 999999 since earlier")              # a dead holder: nobody keeps the file open
    clock["now"] = et(2026, 10, 1, 10, 0)
    eng.tick()
    assert not os.path.exists(eng.paths["lock"])


# ---------------------------------------------------------------------------------------------- closures
def test_friday_closure_moves_the_decision_without_false_alarms(make_engine, market):              # SD-5 T11 T4
    cal, frames, ndx = market
    f, n = without(frames, ndx, "2026-10-09")
    eng, clock = make_engine(f, n, et(2026, 10, 5, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 5, 15, 41), et(2026, 10, 5, 17, 0),
        et(2026, 10, 9, 15, 41), et(2026, 10, 9, 15, 46), et(2026, 10, 9, 16, 0), et(2026, 10, 9, 17, 0),
        et(2026, 10, 10, 10, 0), et(2026, 10, 12, 10, 0))
    st = eng.load_state()
    assert st["runner"]["detected_closures"] == ["2026-10-09"]
    assert "weekly decision for 2026-10-09" not in active(eng)          # the Friday alerts were withdrawn
    fresh, _ = make_engine(f, n, et(2026, 10, 12, 11, 0))               # what `runner.py status` builds
    assert "2026-10-09 is PENDING" not in fresh.status(fetch_signal=False)
    run(eng, clock, et(2026, 10, 12, 15, 41), et(2026, 10, 12, 17, 0))
    dec = read_csv(eng.paths["decisions"])
    assert [r["decision_date"] for r in dec] == ["2026-10-02", "2026-10-08"]
    assert "2026-10-09 was an unscheduled closure" in dec[1]["signal_source"] and dec[1]["execute_on"] == "2026-10-12"
    assert not [k for k in eng.load_state()["runner"]["alerts_sent"] if k == "missed:2026-10-08"]


def test_orders_for_a_day_that_turns_out_closed_are_made_again(make_engine, market):               # ACC-6 ROB-07c
    cal, frames, ndx = market
    f, n = without(frames, ndx, "2026-10-12")                            # Monday closed, nobody knew
    eng, clock = make_engine(f, n, et(2026, 10, 5, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 5, 15, 41), et(2026, 10, 5, 17, 0))     # initial entry (decision 10-02)
    run(eng, clock, et(2026, 10, 12, 15, 42), et(2026, 10, 12, 17, 0), et(2026, 10, 13, 10, 0))   # Friday missed
    st = eng.load_state()
    assert st["runner"]["detected_closures"] == ["2026-10-12"] and "redo" not in st["runner"]   # re-made at once
    assert st["account"]["pending"] and {p["execute_on"] for p in st["account"]["pending"]} == {"2026-10-13"}
    run(eng, clock, et(2026, 10, 13, 17, 0))
    dec = [r for r in read_csv(eng.paths["decisions"]) if r["decision_date"] == "2026-10-09"]
    assert len(dec) == 2 and dec[1]["execute_on"] == "2026-10-13" and "made again" in dec[1]["signal_source"]
    fills = [x for x in read_csv(eng.paths["fills"]) if x["decision_date"] == "2026-10-09"]
    assert fills and {x["fill_date"] for x in fills} == {"2026-10-13"} and all("r2-" in x["order_id"] for x in fills)


def test_ucits_route_detects_a_us_closure_and_decides_the_session_before(make_engine, market):     # ROB-07a
    cal, frames, ndx = market
    f, n = without(ucits(frames), ndx, "2026-10-09", symbols=US_ETFS)    # Europe traded that Friday
    eng, clock = make_engine(f, n, et(2026, 10, 5, 9, 0), strategy={"route": "UCITS"})
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 5, 9, 5), et(2026, 10, 6, 17, 0), et(2026, 10, 9, 16, 40), et(2026, 10, 9, 17, 20),
        et(2026, 10, 12, 10, 0), et(2026, 10, 13, 17, 0))
    dec = [r["decision_date"] for r in read_csv(eng.paths["decisions"])]
    assert dec == ["2026-10-02", "2026-10-08"]


# ---------------------------------------------------------------------------------------------- UCITS execution
def test_ucits_orders_wait_for_a_session_on_which_every_listing_trades(make_engine, market):       # FID-2 ACC-3
    cal, frames, ndx = market
    f, n = without(ucits(frames), ndx, "2026-10-05", symbols=["CBU0.L", "IGLN.L"], ndx_too=False)   # UK holiday
    eng, clock = make_engine(f, n, et(2026, 10, 2, 9, 0), strategy={"route": "UCITS"})
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 2, 16, 40), et(2026, 10, 5, 17, 0), et(2026, 10, 6, 17, 0))
    fills = read_csv(eng.paths["fills"])
    assert {x["symbol"] for x in fills} == {"LQQ.PA", "CBU0.L", "IGLN.L"} and {x["fill_date"] for x in fills} == {"2026-10-06"}


def test_ucits_decision_made_after_the_european_close_fills_the_next_day(make_engine, market):      # SD-6 FID-7 T3
    cal, frames, ndx = market
    eng, clock = make_engine(ucits(frames), ndx, et(2026, 10, 3, 10, 0), strategy={"route": "UCITS"})
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 5, 12, 30), et(2026, 10, 5, 17, 0), et(2026, 10, 6, 17, 0))
    fills = read_csv(eng.paths["fills"])
    assert fills and {x["fill_date"] for x in fills} == {"2026-10-06"}  # Monday's 11:30 ET close printed first


def test_skipped_weekly_decisions_are_recorded_and_alerted(make_engine, market):                   # ROB-13 T5
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 9, 28, 15, 41), et(2026, 9, 28, 17, 0), et(2026, 10, 21, 15, 41), et(2026, 10, 21, 17, 0))
    st = eng.load_state()
    assert [r["decision_date"] for r in read_csv(eng.paths["decisions"])] == ["2026-09-25", "2026-10-16"]
    assert "skipped:2026-10-16" in st["runner"]["alerts_sent"]
    assert any(e["kind"] == "skipped" and "2026-10-02" in e["detail"] for e in read_csv(eng.paths["events"]))


# ---------------------------------------------------------------------------------------------- review triggers
def test_drawdown_breach_inside_a_catch_up_batch_is_alerted(tmp_path):                            # ACC-5
    cal, frames, ndx = synthetic(seed=7)                    # risk-on throughout
    f = {k: v.copy() for k, v in frames.items()}
    n = ndx.copy()
    day = pd.Timestamp("2026-10-14")
    f["QLD"].loc[day, ["Close", "Adj Close"]] *= 0.6        # a one-day 40% fall that recovers the next day ...
    n.loc[day] *= 0.8                                       # ... consistent with a 20% NDX day (not a bad print)
    eng, clock = make(tmp_path, f, n, et(2026, 10, 5, 9, 0), review={"drawdown_trigger": {"F2": 0.10}})
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 5, 15, 41), et(2026, 10, 5, 17, 0), et(2026, 10, 15, 17, 0))   # one catch-up run
    st = eng.load_state()
    assert st["account"]["last_nav"] / st["account"]["high_water"] - 1 > -0.10       # recovered by the end ...
    assert [k for k in st["runner"]["alerts_sent"] if k.startswith("dd:")]           # ... but the breach was seen


def test_three_year_review_check_compares_with_qqq(make_engine, market):                             # SD-13
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    st = eng.init_account(cash=10000)
    days = [str(d) for d in cal.sessions("2023-09-25", "2026-09-30")][-757:]
    st["account"]["nav_hist"] = {d: 10000.0 for d in days}                 # flat for 3 years: behind QQQ
    eng.save_state(st)
    run(eng, clock, et(2026, 10, 1, 10, 0))
    assert any(k.startswith("qqq3y:") for k in eng.load_state()["runner"]["alerts_sent"])


# ---------------------------------------------------------------------------------------------- splits
class SplitMD(ClockedMarketData):
    """Serves a symbol the way Yahoo does around a split: from the effective date on, every earlier close is
    back-adjusted (adjust=True) and, if recorded, a 'Stock Splits' row appears; before it nothing changes."""

    def __init__(self, frames, ndx, clock, symbol="QLD", ex="2026-10-07", ratio=2.0, recorded=True, adjust=True):
        traded = frames[symbol].copy()
        traded.loc[traded.index >= pd.Timestamp(ex), ["Close", "Adj Close"]] /= ratio
        super().__init__(dict(frames, **{symbol: traded}), ndx, clock)
        self.sym, self.ex, self.ratio, self.recorded, self.adjust = symbol, pd.Timestamp(ex), ratio, recorded, adjust

    def _frame(self, symbol):
        h = super()._frame(symbol)
        if symbol != self.sym or self._cut() < self.ex:
            return h
        h = h.copy()
        if self.adjust:
            h.loc[h.index < self.ex, ["Close", "Adj Close"]] /= self.ratio
        if self.recorded and self.ex in h.index:
            h.loc[self.ex, "Stock Splits"] = self.ratio
        return h


def nav_on(eng, day):
    return float([r for r in read_csv(eng.paths["nav"]) if r["date"] == day][0]["nav"])


def pair(tmp_path, frames, ndx, start, **split_kw):
    ctrl, c1 = make(tmp_path / "control", frames, ndx, start)
    test, c2 = make(tmp_path / "split", frames, ndx, start, md_cls=SplitMD, md_kw=split_kw)
    ctrl.init_account(cash=100000)
    test.init_account(cash=100000)
    return (ctrl, c1), (test, c2)


@pytest.mark.parametrize("recorded", [True, False])
def test_catch_up_across_a_split_books_no_phantom_profit(tmp_path, recorded):                      # ACC-2 ROB-01a
    cal, frames, ndx = synthetic(seed=7)
    (ctrl, c1), (test, c2) = pair(tmp_path, frames, ndx, et(2026, 9, 28, 9, 0), ex="2026-10-01", recorded=recorded)
    for eng, clock in ((ctrl, c1), (test, c2)):             # queue the entry, then the PC sleeps for 10 days
        run(eng, clock, et(2026, 9, 28, 15, 41), et(2026, 10, 8, 17, 0))
    assert nav_on(test, "2026-10-08") == pytest.approx(nav_on(ctrl, "2026-10-08"), rel=1e-9)
    assert test.load_state()["account"]["positions"]["QLD"] == 2 * ctrl.load_state()["account"]["positions"]["QLD"]
    assert ("NOT recorded" in open(os.path.join(test.data_dir, "alerts.log"), encoding="utf-8").read()) == (not recorded)


def test_split_on_the_decision_day_is_applied_before_planning(tmp_path):                            # ACC-1 ROB-01b
    cal, frames, ndx = synthetic(seed=7)
    (ctrl, c1), (test, c2) = pair(tmp_path, frames, ndx, et(2026, 9, 28, 9, 0), ex="2026-10-16")
    times = [et(2026, 9, 28, 15, 41)] + [et(2026, 9, 28, 17, 0) + pd.Timedelta(days=k) for k in range(19)]
    times = sorted(times + [et(2026, 10, 2, 15, 41), et(2026, 10, 9, 15, 41), et(2026, 10, 16, 15, 41)])
    for eng, clock in ((ctrl, c1), (test, c2)):
        run(eng, clock, *times)
    dc, dt_ = read_csv(ctrl.paths["decisions"])[-1], read_csv(test.paths["decisions"])[-1]
    assert dc["decision_date"] == dt_["decision_date"] == "2026-10-16"
    assert float(dt_["nav_est"]) == pytest.approx(float(dc["nav_est"]), rel=1e-9)
    assert nav_on(test, "2026-10-16") == pytest.approx(nav_on(ctrl, "2026-10-16"), rel=1e-9)


def test_split_yahoo_never_adjusts_pauses_booking_until_recorded(tmp_path):                         # ACC-4
    cal, frames, ndx = synthetic(seed=7)
    (ctrl, c1), (test, c2) = pair(tmp_path, frames, ndx, et(2026, 9, 28, 9, 0), ex="2026-10-01", recorded=False,
                                  adjust=False)
    for eng, clock in ((ctrl, c1), (test, c2)):
        run(eng, clock, et(2026, 9, 28, 15, 41), et(2026, 9, 28, 17, 0), et(2026, 9, 29, 17, 0),
            et(2026, 9, 30, 17, 0), et(2026, 10, 1, 17, 0))
    st = test.load_state()
    assert st["account"]["last_processed"] == "2026-09-30" and "suspect:QLD:2026-10-01" in st["runner"]["alerts_sent"]
    test.manual_split("QLD", 2)
    run(test, c2, et(2026, 10, 1, 17, 5))
    assert nav_on(test, "2026-10-01") == pytest.approx(nav_on(ctrl, "2026-10-01"), rel=1e-9)


def test_a_bar_yahoo_never_publishes_does_not_freeze_the_account(make_engine, market):              # ROB-11
    cal, frames, ndx = market
    f, n = without(frames, ndx, "2026-10-06", symbols=["GLD"], ndx_too=False)
    eng, clock = make_engine(f, n, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 9, 28, 15, 41), et(2026, 9, 28, 17, 0), et(2026, 10, 6, 17, 0))
    assert eng.load_state()["account"]["last_processed"] == "2026-10-05"
    run(eng, clock, et(2026, 10, 8, 17, 0))
    st = eng.load_state()
    assert st["account"]["last_processed"] == "2026-10-08" and "no-bar:GLD:2026-10-06" in st["runner"]["alerts_sent"]


# ---------------------------------------------------------------------------------------------- shadow tranche
def test_held_weights_lags_and_waits_for_all_listings():                                           # FID-1
    idx = pd.DatetimeIndex(pd.bdate_range("2026-01-05", periods=6))
    dec = {idx[1]: {"A": 1.0}}
    assert list(shadow.held_weights(idx, dec, ["A"], lag=1)["A"]) == [0, 0, 1, 1, 1, 1]      # US: trade at the close
    assert list(shadow.held_weights(idx, dec, ["A"], lag=2)["A"]) == [0, 0, 0, 1, 1, 1]      # UCITS: next close
    ok = [True, True, False, True, True, True]
    assert list(shadow.held_weights(idx, dec, ["A"], lag=2, ok=ok)["A"]) == [0, 0, 0, 0, 1, 1]


def test_ucits_shadow_does_not_earn_moves_the_signal_already_saw():                                # FID-1
    cal, frames, ndx = synthetic(seed=3)
    f = ucits(frames, eur=1.0)
    f["LQQ.PA"] = frames["QLD"].copy()
    for s in ["LQQ.PA", "CBU0.L", "IGLN.L"]:              # a European close only knows the previous US close
        f[s] = f[s].shift(1).dropna()
    md = FrameMarketData(f, ndx)
    us = shadow.compute(md, cal, "F2", "US", "2025-01-06", "2026-06-30")["tranche_nav"].pct_change()
    uc = shadow.compute(md, cal, "F2", "UCITS", "2025-01-06", "2026-06-30")["tranche_nav"].pct_change()
    # trading these European closes at lag 2 earns exactly the US lag-1 returns, one session later; the
    # old lag-1 shadow would have earned them on the same day, i.e. a move the signal had already seen
    assert np.allclose(uc.values[2:], us.values[1:-1], atol=1e-12)
    assert not np.allclose(uc.values[1:], us.values[1:], atol=1e-6)


def test_shadow_refuses_to_fill_missing_history_with_zero_returns():                               # FID-3 ROB-03
    cal, frames, ndx = synthetic()
    f = dict(frames)
    f["GLD"] = frames["GLD"][frames["GLD"].index >= "2026-06-01"]
    with pytest.raises(ValueError):
        shadow.compute(FrameMarketData(f, ndx), cal, "F2", "US", "2026-03-02", "2026-09-30")


def test_live_market_data_reaches_back_to_the_account_start(make_engine, market):                   # FID-3 ROB-03
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    st = eng.init_account(cash=10000, start="2023-05-01")
    eng2 = Engine(eng.cfg, now_fn=lambda: et(2026, 11, 19, 9, 0), data_dir=eng.data_dir, quiet=True)
    assert str(eng2._data_start()) <= "2022-03-27" and eng2.md.ndx_period == "max"


def test_changing_tier_keeps_the_shadow_history(make_engine, market):                                # ROB-09
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, *[et(2026, 9, 28, 17, 0) + pd.Timedelta(days=k) for k in range(18)])
    before = {r["date"]: r["tranche_nav"] for r in read_csv(eng.paths["shadow"])}
    eng2, clock2 = make_engine(frames, ndx, et(2026, 10, 19, 9, 0), strategy={"tier": "F1"})
    run(eng2, clock2, et(2026, 10, 19, 17, 0), et(2026, 10, 20, 17, 0))
    after = {r["date"]: r["tranche_nav"] for r in read_csv(eng2.paths["shadow"])}
    assert len(after) > len(before) and all(after[d] == v for d, v in before.items())
    assert [s["tier"] for s in eng2.load_state()["runner"]["shadow_segments"]] == ["F2", "F1"]


# ---------------------------------------------------------------------------------------------- runner / config
def test_config_error_under_the_scheduler_is_logged_and_alerted(tmp_path):                          # SD-1 ROB-05
    bad = tmp_path / "bad.toml"
    bad.write_text('[strategy]\ntier = "f2"\n', encoding="utf-8")
    data = tmp_path / "data"
    assert runner.main(["--config", str(bad), "--data-dir", str(data), "tick"]) == 1
    assert "strategy.tier" in (data / "runner.log").read_text(encoding="utf-8")
    assert "runner error" in (data / "ALERT_ACTIVE.txt").read_text(encoding="utf-8")


def test_repeated_errors_are_logged_every_time_but_alerted_once(tmp_path):                         # ROB-06
    bad = tmp_path / "bad.toml"
    bad.write_text('[schedule]\nextra_holidays = ["2025/11/27"]\n', encoding="utf-8")
    data = tmp_path / "data"
    for _ in range(3):
        assert runner.main(["--config", str(bad), "--data-dir", str(data), "tick"]) == 1
    assert (data / "runner.log").read_text(encoding="utf-8").count("ERROR in 'tick'") == 3
    assert len((data / "ALERT_ACTIVE.txt").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_corrupt_state_is_reported_and_init_force_still_recovers(tmp_path):                        # ROB-06
    data = tmp_path / "data"
    data.mkdir()
    (data / "state.json").write_text('{"account": {"cash": 1', encoding="utf-8")
    cfg = load_config(str(tmp_path / "none.toml"), {"paths": {"data_dir": str(data)},
                                                     "alerts": {"desktop_notifications": False}})
    eng = Engine(cfg, md=FrameMarketData({}, None), now_fn=lambda: et(2026, 10, 1, 9, 0), quiet=True)
    with pytest.raises(Exception) as e:
        eng.tick()
    assert "state.json.bak" in str(e.value)
    eng.init_account(cash=5000, force=True)
    assert eng.load_state()["account"]["cash"] == 5000
    assert any(p.startswith("archive_") for p in os.listdir(data))


def test_busy_lock_makes_interactive_commands_say_so(tmp_path):                                      # ROB-12
    from paper.storage import RunLock
    data = tmp_path / "data"
    with RunLock(str(data / "run.lock")):
        assert runner.main(["--data-dir", str(data), "init", "--cash", "1000"]) == 3
        assert runner.main(["--data-dir", str(data), "tick"]) == 0
    assert "tick skipped" in (data / "runner.log").read_text(encoding="utf-8")


def test_replay_never_touches_a_live_account(tmp_path):                                              # SD-7
    live = tmp_path / "data"
    live.mkdir()
    (live / "state.json").write_text("{}", encoding="utf-8")
    cfg = load_config(str(tmp_path / "none.toml"), {"paths": {"data_dir": str(live)}})
    md = FrameMarketData({}, None)
    if os.name == "nt":
        with pytest.raises(ValueError):
            replay(cfg, "2026-01-05", "2026-01-09", 1000.0, str(tmp_path / "DATA"), md=md)
    other = tmp_path / "account"
    other.mkdir()
    (other / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        replay(cfg, "2026-01-05", "2026-01-09", 1000.0, str(other), md=md)
    assert (live / "state.json").exists() and (other / "state.json").exists()


# ---------------------------------------------------------------------------------------------- pre-publication audit
def test_monthly_qqq_check_is_retried_when_yahoo_fails(make_engine, market, monkeypatch):            # CODE-2
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    st = eng.init_account(cash=10000)
    days = [str(d) for d in cal.sessions("2023-09-25", "2026-09-30")][-757:]
    st["account"]["nav_hist"] = {d: 10000.0 for d in days}
    eng.save_state(st)
    real = eng.md.adj_closes

    def down(*a, **k):
        raise OSError("Yahoo unreachable")

    monkeypatch.setattr(eng.md, "adj_closes", down)
    run(eng, clock, et(2026, 10, 1, 10, 0))
    st = eng.load_state()
    assert st["runner"].get("qqq3y_checked") is None and "qqq3y" in st["runner"]["backoff"]
    monkeypatch.setattr(eng.md, "adj_closes", real)
    run(eng, clock, et(2026, 10, 1, 10, 30))                       # still backing off
    assert eng.load_state()["runner"].get("qqq3y_checked") is None
    run(eng, clock, et(2026, 10, 1, 11, 5))                        # an hour later it runs
    st = eng.load_state()
    assert st["runner"]["qqq3y_checked"] == "2026-10" and any(k.startswith("qqq3y:") for k in st["runner"]["alerts_sent"])


def test_late_label_says_the_live_attempt_failed_rather_than_the_pc_was_off(make_engine, market):    # CODE-4
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    eng.md = ClockedMarketData(frames, ndx[ndx.index < "2026-10-02"], clock)   # today's live bar never arrives
    run(eng, clock, et(2026, 10, 2, 15, 41))
    eng.md = ClockedMarketData(frames, ndx, clock)
    run(eng, clock, et(2026, 10, 2, 16, 50))
    dec = read_csv(eng.paths["decisions"])[-1]
    assert dec["decision_date"] == "2026-10-02" and "the live decision in the window failed" in dec["signal_source"]
    assert "PC off" not in dec["signal_source"]


def test_accept_move_and_split_refuse_what_was_not_flagged(make_engine, market):                      # FC-9
    cal, frames, ndx = market
    eng, clock = make_engine(frames, ndx, et(2026, 10, 1, 9, 0))
    eng.init_account(cash=10000)
    with pytest.raises(ValueError):
        eng.accept_move("QLD", "2026-12-04")
    with pytest.raises(ValueError):
        eng.manual_split("QLD", 2)
    assert "accepted" in eng.accept_move("QLD", "2026-12-04", force=True)


def test_status_without_an_account_exits_4(tmp_path):                                                 # FC-10 DELTA-9
    assert runner.main(["--data-dir", str(tmp_path / "empty"), "status"]) == 4


def test_webhook_text_never_carries_local_paths(tmp_path):                                            # PRV-04
    import os as _os
    from paper.alerts import local_paths_removed
    data = str(tmp_path / "data")
    line = f"[WARN] cannot write nav.csv | PermissionError: '{_os.path.join(data, 'nav.csv')}' (in {_os.path.expanduser('~')})"
    out = local_paths_removed(line, data)
    assert data not in out and _os.path.expanduser("~") not in out and "<data>" in out
    user = _os.path.basename(_os.path.expanduser("~")).lower()          # DELTA-2: escaped forms too
    for text in (str(PermissionError(13, "denied", _os.path.join(data, "state.json"))),
                 str([_os.path.join(_os.path.expanduser("~"), "x.txt")])):
        assert user not in local_paths_removed(text, data).lower()



class OutageMD(ClockedMarketData):
    """Yahoo unreachable while the clock is inside [start, end)."""

    def __init__(self, frames, ndx, clock, start, end):
        super().__init__(frames, ndx, clock)
        self.start, self.end = start, end

    def _down(self):
        if self.start <= self.clock["now"] < self.end:
            from paper.market import DataError
            raise DataError("Yahoo unreachable")

    def _frame(self, symbol):
        self._down()
        return super()._frame(symbol)

    def _ndx_series(self):
        self._down()
        return super()._ndx_series()


def test_window_decision_never_erases_fills_that_are_not_booked_yet(tmp_path):                       # DELTA-1
    cal, frames, ndx = synthetic(ndx_returns=boom_then_crash)
    eng, clock = make(tmp_path, frames, ndx, et(2026, 10, 5, 9, 0), md_cls=OutageMD,
                      md_kw=dict(start=et(2026, 10, 16, 15, 40), end=et(2026, 10, 16, 15, 42)))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 9, 15, 41))                   # the PC is on only for this window ...
    assert read_csv(eng.paths["decisions"])[-1]["decision_date"] == "2026-10-09"
    run(eng, clock, et(2026, 10, 16, 15, 41), et(2026, 10, 16, 15, 42), et(2026, 10, 16, 17, 0))   # ... and the next
    fills = read_csv(eng.paths["fills"])
    assert {f["fill_date"] for f in fills if f["decision_date"] == "2026-10-09"} == {"2026-10-09"}
    dec = read_csv(eng.paths["decisions"])
    assert dec[-1]["decision_date"] == "2026-10-16" and "superseded" not in dec[-1]["notes"]


def test_suspicious_close_met_while_filling_a_missing_bar_pauses_with_an_alert(tmp_path):            # DELTA-3
    cal, frames, ndx = synthetic(seed=7)
    f = {k: v.copy() for k, v in frames.items()}
    f["GLD"] = f["GLD"].drop(pd.Timestamp("2026-10-06"))
    f["QLD"].loc[pd.Timestamp("2026-10-06"), ["Close", "Adj Close"]] *= 0.55
    eng, clock = make(tmp_path, f, ndx, et(2026, 9, 28, 9, 0))
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 9, 28, 15, 41), et(2026, 9, 28, 17, 0), et(2026, 10, 8, 17, 0))
    st = eng.load_state()
    assert st["account"]["last_processed"] == "2026-10-05" and "suspect:QLD:2026-10-06" in st["runner"]["alerts_sent"]


def test_ucits_late_label_names_failed_attempts_when_the_pc_was_on(tmp_path):                       # DELTA-4
    cal, frames, ndx = synthetic(ndx_returns=boom_then_crash)
    eng, clock = make(tmp_path, ucits(frames), ndx, et(2026, 10, 1, 9, 0), md_cls=OutageMD,
                      md_kw=dict(start=et(2026, 10, 9, 16, 0), end=et(2026, 10, 12, 12, 0)),
                      strategy={"route": "UCITS"})
    eng.init_account(cash=10000)
    run(eng, clock, et(2026, 10, 1, 9, 5), et(2026, 10, 2, 17, 0), et(2026, 10, 9, 16, 40), et(2026, 10, 10, 10, 0),
        et(2026, 10, 12, 12, 30))
    dec = read_csv(eng.paths["decisions"])[-1]
    assert dec["decision_date"] == "2026-10-09" and "attempts from" in dec["signal_source"]
    assert "PC off" not in dec["signal_source"]
