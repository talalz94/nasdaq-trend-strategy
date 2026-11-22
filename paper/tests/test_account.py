import datetime as dt

import pandas as pd
import pytest

from paper.account import NotReady, SimAccount, SuspectPrice, snap_ratio
from paper.market import FrameMarketData
from paper.orders import Order

D = dt.date
DAYS = pd.DatetimeIndex(["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"])


def md(qld=(100, 101, 102, 103, 104, 105), ief=(90, 90, 90, 90, 90, 90), ief_div=None, qld_split=None, gld=None):
    def f(p, div=None, split=None):
        n = len(p)
        return pd.DataFrame({"Close": list(p), "Adj Close": list(p),
                             "Dividends": div or [0.0] * n, "Stock Splits": split or [0.0] * n},
                            index=DAYS[:n])
    frames = {"QLD": f(qld, split=qld_split), "IEF": f(ief, div=ief_div)}
    if gld is not None:
        frames["GLD"] = f(gld)
    return FrameMarketData(frames, pd.Series(range(len(DAYS)), index=DAYS, dtype=float))


def acct(**kw):
    base = dict(cash=10000.0, start_date="2026-11-16", last_processed="2026-11-13")
    base.update(kw)
    return SimAccount(**base)


def test_fill_at_close_with_costs_and_nav():
    a = acct()
    a.queue([Order("QLD", "BUY", 50, 100.0, 1.0, "switch")], "2026-11-16", "2026-11-16")
    rec = a.process_session(D(2026, 11, 16), md(), cost_bps=5.0)
    f = rec["fills"][0]
    assert f["qty"] == 50 and f["price"] == 100.0 and f["cost_usd"] == pytest.approx(2.5)
    assert a.positions == {"QLD": 50.0}
    assert a.cash == pytest.approx(10000 - 5000 - 2.5)
    assert rec["nav_row"]["nav"] == pytest.approx(9997.5)
    assert a.pending == [] and a.last_processed == "2026-11-16"


def test_missing_price_leaves_account_untouched():
    a = acct(positions={"QLD": 10.0})
    before = a.to_dict()
    with pytest.raises(NotReady):
        a.process_session(D(2026, 11, 16), md(qld=(None, 101, 102, 103, 104, 105)), cost_bps=5.0)
    assert a.to_dict() == before


def test_dividend_paid_to_holder_at_previous_close_before_same_day_sell():
    a = acct(positions={"IEF": 100.0}, last_processed="2026-11-16",
             pos_history={"2026-11-16": {"IEF": 100.0}})
    a.queue([Order("IEF", "SELL", 100, 90.0, 1.0, "switch")], "2026-11-17", "2026-11-17")
    rec = a.process_session(D(2026, 11, 17), md(ief_div=[0, 0.3, 0, 0, 0, 0]), cost_bps=0.0)
    assert rec["dividends"][0]["usd"] == pytest.approx(30.0)
    assert a.positions == {}
    assert a.cash == pytest.approx(10000 + 30 + 9000)


def test_dividend_not_paid_on_shares_bought_on_ex_date():
    a = acct(last_processed="2026-11-16", pos_history={"2026-11-16": {}})
    a.queue([Order("IEF", "BUY", 10, 90.0, 1.0, "switch")], "2026-11-17", "2026-11-17")
    rec = a.process_session(D(2026, 11, 17), md(ief_div=[0, 0.3, 0, 0, 0, 0]), cost_bps=0.0)
    assert rec["dividends"] == []


def test_late_posted_dividend_is_credited_once():
    a = acct(positions={"IEF": 10.0}, last_processed="2026-11-17",
             pos_history={"2026-11-16": {"IEF": 10.0}, "2026-11-17": {"IEF": 10.0}})
    data = md(ief_div=[0, 0.3, 0, 0, 0, 0])
    rec = a.process_session(D(2026, 11, 18), data, cost_bps=0.0)      # ex-date was the 17th
    assert rec["dividends"][0]["usd"] == pytest.approx(3.0)
    rec = a.process_session(D(2026, 11, 19), data, cost_bps=0.0)
    assert rec["dividends"] == []


def test_recorded_split_rebases_positions_before_booking():
    # Yahoo after a 2:1 split on the 17th: the 16th's close is back-adjusted to 50 and a split row is shown
    a = acct(positions={"QLD": 10.0}, last_processed="2026-11-16", pos_history={"2026-11-16": {"QLD": 10.0}},
             refs={"QLD": ["2026-11-16", 100.0]})
    data = md(qld=(50, 51, 52, 53, 54, 55), qld_split=[0, 2.0, 0, 0, 0, 0])
    ch = a.rebase(data, D(2026, 11, 17))
    assert a.positions["QLD"] == 20.0 and ch[0][0] == "INFO" and "recorded" in ch[0][2]
    assert a.pos_history["2026-11-16"]["QLD"] == 20.0 and a.refs["QLD"] == ["2026-11-16", 50.0]
    rec = a.process_session(D(2026, 11, 17), data, cost_bps=0.0)
    assert rec["nav_row"]["nav"] == pytest.approx(10000 + 20 * 51)
    assert a.rebase(data, D(2026, 11, 17)) == []                   # applied once only


def test_unrecorded_split_is_inferred_from_back_adjusted_closes_and_warned():
    a = acct(positions={"QLD": 10.0}, last_processed="2026-11-16", refs={"QLD": ["2026-11-16", 100.0]})
    ch = a.rebase(md(qld=(0.5, 0.51, 0.52, 0.53, 0.54, 0.55)), D(2026, 11, 17))   # a 200:1 unit change
    assert a.positions["QLD"] == 2000.0 and ch[0][0] == "WARN" and "NOT recorded" in ch[0][2]


def test_reverse_split_credits_cash_in_lieu():
    a = acct(cash=0.0, start_nav=75.0, positions={"QLD": 15.0}, last_processed="2026-11-16", refs={"QLD": ["2026-11-16", 5.0]})
    a.rebase(md(qld=(50, 50, 50, 50, 50, 50), qld_split=[0, 0.1, 0, 0, 0, 0]), D(2026, 11, 17))
    assert a.positions["QLD"] == 1.0 and a.cash == pytest.approx(25.0)     # 1.5 shares -> 1 + 0.5 x $50


def test_pending_orders_follow_a_split():
    a = acct(last_processed="2026-11-16")
    a.queue([Order("QLD", "BUY", 8, 100.0, 1.0, "switch")], "2026-11-16", "2026-11-18", px_date="2026-11-16")
    a.rebase(md(qld=(50, 51, 52, 53, 54, 55), qld_split=[0, 2.0, 0, 0, 0, 0]), D(2026, 11, 17))
    assert a.pending[0]["qty"] == 16.0 and a.pending[0]["est_price"] == pytest.approx(50.0)


def test_suspicious_close_stops_booking_until_accepted():
    a = acct(positions={"QLD": 10.0}, last_processed="2026-11-16", refs={"QLD": ["2026-11-16", 100.0]})
    data = md(qld=(100, 50, 52, 53, 54, 55))          # halved with no back-adjustment and no split row
    guard = lambda s, d0, p0, d, p: "jump" if not 0.6 < p / p0 < 1.6 else None
    before = a.to_dict()
    with pytest.raises(SuspectPrice):
        a.process_session(D(2026, 11, 17), data, cost_bps=0.0, guard=guard)
    assert a.to_dict() == before
    a.accepted.append(["QLD", "2026-11-17"])
    a.process_session(D(2026, 11, 17), data, cost_bps=0.0, guard=guard)
    assert a.last_processed == "2026-11-17" and a.accepted == []


def ucits_md(london_closed=()):
    idx = DAYS
    def f(p, closed=()):
        df = pd.DataFrame({"Close": [float(p)] * len(idx), "Adj Close": [float(p)] * len(idx),
                           "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
        return df.drop([pd.Timestamp(x) for x in closed])
    frames = {"LQQ.PA": f(10.0), "CBU0.L": f(100.0, london_closed), "IGLN.L": f(50.0, london_closed),
              "EURUSD=X": f(1.0)}
    return FrameMarketData(frames, pd.Series(1.0, index=idx))


GROUP = ["CBU0.L", "IGLN.L", "LQQ.PA"]


def test_ucits_legs_fill_together_when_london_is_closed():
    a = acct(cash=0.0, start_nav=10000.0, positions={"CBU0.L": 100.0}, last_processed="2026-11-16")
    a.queue([Order("CBU0.L", "SELL", 100, 100.0, 1.0, "switch"), Order("LQQ.PA", "BUY", 990, 10.0, 1.0, "switch")],
            "2026-11-16", None, group=GROUP)
    data = ucits_md(london_closed=["2026-11-17"])
    rec = a.process_session(D(2026, 11, 17), data, cost_bps=0.0)
    assert rec["fills"] == [] and len(a.pending) == 2                     # Paris open, London shut: nobody trades
    rec = a.process_session(D(2026, 11, 18), data, cost_bps=0.0)
    assert {f["symbol"]: f["qty"] for f in rec["fills"]} == {"CBU0.L": 100.0, "LQQ.PA": 990.0}


def test_order_never_fills_at_a_close_printed_before_the_decision():
    a = acct(last_processed="2026-11-16")
    eu_close = lambda s, d: pd.Timestamp(f"{d} 11:30").tz_localize("America/New_York")
    a.queue([Order("IGLN.L", "BUY", 10, 50.0, 1.0, "switch")], "2026-11-16", None, group=GROUP,
            not_before=pd.Timestamp("2026-11-17 12:30").tz_localize("America/New_York"))
    data = ucits_md()
    assert a.process_session(D(2026, 11, 17), data, cost_bps=0.0, close_ts=eu_close)["fills"] == []
    assert a.process_session(D(2026, 11, 18), data, cost_bps=0.0, close_ts=eu_close)["fills"][0]["fill_date"] == "2026-11-18"


def test_buy_reduced_to_nothing_is_not_written_as_a_fill():
    a = acct(cash=50.0)
    a.queue([Order("QLD", "BUY", 5, 100.0, 1.0, "switch")], "2026-11-16", "2026-11-16")
    rec = a.process_session(D(2026, 11, 16), md(), cost_bps=5.0)
    assert rec["fills"] == [] and rec["cuts"][0]["filled"] == 0 and rec["events"][0][0] == "unfilled"


def test_fractional_buy_never_overspends():
    a = acct(cash=1000.0)
    a.queue([Order("QLD", "BUY", 50, 100.0, 1.0, "switch")], "2026-11-16", "2026-11-16")
    a.process_session(D(2026, 11, 16), md(qld=(99.99, 101, 102, 103, 104, 105)), cost_bps=5.0, commission=1.0,
                      fractional=True)
    assert a.cash >= 0 and a.positions["QLD"] > 9.9


def test_missing_bar_for_good_marks_previous_close_and_rolls_the_order():
    a = acct(positions={"QLD": 10.0}, last_processed="2026-11-16", refs={"QLD": ["2026-11-16", 100.0]})
    a.queue([Order("IEF", "BUY", 5, 90.0, 1.0, "switch")], "2026-11-17", "2026-11-17")
    data = md(qld=(100, None, 102, 103, 104, 105))
    with pytest.raises(NotReady) as e:
        a.process_session(D(2026, 11, 17), data, cost_bps=0.0)
    assert e.value.symbol == "QLD"
    a2 = acct(positions={"QLD": 10.0}, last_processed="2026-11-16", refs={"QLD": ["2026-11-16", 100.0]})
    a2.queue([Order("QLD", "BUY", 5, 100.0, 1.0, "switch")], "2026-11-17", "2026-11-17")
    rec = a2.process_session(D(2026, 11, 17), data, cost_bps=0.0, ffill_ok={"QLD"}, roll_to=D(2026, 11, 18))
    assert rec["nav_row"]["nav"] == pytest.approx(10000 + 10 * 100) and a2.pending[0]["execute_on"] == "2026-11-18"


def test_next_bar_execution_for_ucits_style_orders():
    a = acct(last_processed="2026-11-16")
    a.queue([Order("QLD", "BUY", 5, 101.0, 1.0, "switch")], "2026-11-16", None)
    rec = a.process_session(D(2026, 11, 17), md(), cost_bps=0.0)
    assert rec["fills"][0]["price"] == 101 and a.positions == {"QLD": 5.0}


def test_buy_reduced_when_cash_short():
    a = acct(cash=1000.0)
    a.queue([Order("QLD", "BUY", 50, 100.0, 1.0, "switch")], "2026-11-16", "2026-11-16")
    rec = a.process_session(D(2026, 11, 16), md(), cost_bps=5.0)
    assert rec["fills"][0]["qty"] == 9.0 and "reduced" in rec["fills"][0]["note"]
    assert a.cash >= 0


def test_new_decision_supersedes_unfilled_orders():
    a = acct()
    a.queue([Order("QLD", "BUY", 5, 100.0, 1.0, "switch")], "2026-11-13", None)
    cancelled = a.queue([Order("IEF", "BUY", 5, 90.0, 1.0, "switch")], "2026-11-20", "2026-11-20")
    assert len(cancelled) == 1 and [p["symbol"] for p in a.pending] == ["IEF"]


def test_cannot_process_same_session_twice():
    a = acct()
    a.process_session(D(2026, 11, 16), md(), cost_bps=0.0)
    with pytest.raises(ValueError):
        a.process_session(D(2026, 11, 16), md(), cost_bps=0.0)


def test_snap_ratio_finds_plain_split_ratios():
    assert snap_ratio(199.97) == 200.0 and snap_ratio(2.01) == 2.0 and snap_ratio(0.1002) == pytest.approx(0.1)
    assert snap_ratio(1.5004) == 1.5 and snap_ratio(1.37) == 1.37
