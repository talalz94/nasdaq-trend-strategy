import pytest

from paper.orders import infer_risk_on, plan_orders, portfolio_weights

PX = {"QLD": 100.0, "IEF": 100.0, "GLD": 100.0}
FX = {"QLD": 1.0, "IEF": 1.0, "GLD": 1.0}
KW = dict(cash_buffer=0.0, fractional=False, cost_bps=5.0)


def test_inside_band_is_skipped_outside_is_traded():
    pos = {"QLD": 42.0, "IEF": 43.0, "GLD": 15.0}            # 42 / 43 / 15 % of a 10,000 book
    tgt = {"QLD": 0.425, "IEF": 0.425, "GLD": 0.15}
    orders, skipped, nav, _ = plan_orders(tgt, pos, PX, FX, 0.0, band=0.02, switch=False, **KW)
    assert nav == pytest.approx(10000)
    assert orders == [] and set(skipped) == {"QLD", "IEF", "GLD"}
    tgt = {"QLD": 0.2125, "IEF": 0.6375, "GLD": 0.15}
    orders, skipped, _, _ = plan_orders(tgt, pos, PX, FX, 0.0, band=0.02, switch=False, **KW)
    sides = {(o.symbol, o.side, o.qty, o.reason) for o in orders}
    assert ("QLD", "SELL", 21.0, "outside band") in sides and ("IEF", "BUY", 20.0, "outside band") in sides
    assert "GLD" in skipped


def test_switch_trades_every_sleeve_even_inside_band():
    pos = {"QLD": 41.0, "IEF": 44.0, "GLD": 14.0}             # every sleeve within 2pp of target
    orders, skipped, _, _ = plan_orders({"QLD": 0.425, "IEF": 0.425, "GLD": 0.15}, pos, PX, FX, 100.0,
                                        band=0.02, switch=True, **KW)
    assert skipped == {}
    assert {(o.symbol, o.side, o.qty) for o in orders} == {("QLD", "BUY", 1.0), ("IEF", "SELL", 2.0), ("GLD", "BUY", 1.0)}
    assert all(o.reason == "switch" for o in orders)
    orders, skipped, _, _ = plan_orders({"QLD": 0.425, "IEF": 0.425, "GLD": 0.15}, pos, PX, FX, 100.0,
                                        band=0.02, switch=False, **KW)
    assert orders == [] and set(skipped) == {"QLD", "IEF", "GLD"}


def test_symbols_outside_the_strategy_are_sold():
    px = dict(PX, TLT=50.0)
    fx = dict(FX, TLT=1.0)
    orders, _, _, _ = plan_orders({"QLD": 0.0, "IEF": 0.85, "GLD": 0.15}, {"TLT": 100.0}, px, fx, 5000.0,
                                  band=0.02, switch=False, **KW)
    tlt = [o for o in orders if o.symbol == "TLT"]
    assert tlt and tlt[0].side == "SELL" and tlt[0].qty == 100.0 and tlt[0].reason == "not in strategy"
    assert orders[0].side == "SELL"                       # sells are listed first


def test_whole_shares_cash_buffer_and_affordability():
    orders, _, _, _ = plan_orders({"QLD": 0.0, "IEF": 0.85, "GLD": 0.15}, {}, {"QLD": 100.0, "IEF": 93.0, "GLD": 371.9},
                                  {"QLD": 1, "IEF": 1, "GLD": 1}, 10000.0, band=0.02, switch=True,
                                  cash_buffer=0.005, fractional=False, cost_bps=5.0)
    q = {o.symbol: o.qty for o in orders}
    assert q == {"IEF": 90.0, "GLD": 4.0}                 # floor(0.85*0.995*10000/93), floor(0.15*0.995*10000/371.9)
    spent = sum(o.est_notional_usd for o in orders) * 1.0005
    assert spent <= 10000.0


def test_buys_scaled_when_cash_short():
    orders, _, _, _ = plan_orders({"QLD": 0.5, "IEF": 0.35, "GLD": 0.15}, {"QLD": 0.0}, PX, FX, 1000.0,
                                  band=0.0, switch=True, cash_buffer=0.0, fractional=False, cost_bps=100.0)
    assert sum(o.est_notional_usd for o in orders) * 1.01 <= 1000.0 + 1e-6


def test_bad_targets_and_missing_prices_raise():
    with pytest.raises(ValueError):
        plan_orders({"QLD": 0.5, "IEF": 0.4}, {}, PX, FX, 1000.0, band=0.02, switch=False, **KW)
    with pytest.raises(ValueError):
        portfolio_weights({"XYZ": 1.0}, PX, 0.0)


def test_infer_risk_on():
    assert infer_risk_on({"QLD": 0.3}, "QLD") is True
    assert infer_risk_on({"IEF": 0.85}, "QLD") is False


def test_commission_counts_in_affordability():
    # $1,000, three buys and a $5 commission each: the plan must leave room for $15 of commissions
    orders, _, _, _ = plan_orders({"QLD": 0.5, "IEF": 0.35, "GLD": 0.15}, {}, PX, FX, 1000.0, band=0.0, switch=True,
                                  cash_buffer=0.0, fractional=True, cost_bps=0.0, commission=5.0)
    assert sum(o.est_notional_usd for o in orders) + 5.0 * len(orders) <= 1000.0 + 1e-9


def test_fractional_quantities_round_down():
    orders, _, _, _ = plan_orders({"QLD": 0.0, "IEF": 1.0, "GLD": 0.0}, {}, {"QLD": 1.0, "IEF": 3.0, "GLD": 1.0},
                                  {"QLD": 1, "IEF": 1, "GLD": 1}, 1000.0, band=0.0, switch=True, cash_buffer=0.0,
                                  fractional=True, cost_bps=0.0)
    assert orders[0].qty == 333.3333 and orders[0].qty * 3.0 <= 1000.0
