"""
Order planning (pure functions, no I/O).

Rules (STRATEGY_RULEBOOK.md, Execution):
  * targets are scaled by (1 - cash_buffer) so a little cash absorbs rounding and costs
  * a sleeve whose current weight is within `band` of its target is skipped ...
  * ... unless the decision switches risk-on <-> risk-off: then EVERY sleeve trades to target
  * anything held that is not part of the strategy is sold in full
  * whole shares unless fractional_shares (then 4 decimals, rounded down); sells are listed before buys
  * buys are scaled down if the cash after sells (net of costs and commissions) cannot pay for them
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class Order:
    symbol: str
    side: str            # "BUY" or "SELL"
    qty: float
    est_price: float     # instrument currency
    fx: float            # instrument currency -> USD
    reason: str          # "switch" | "outside band" | "not in strategy"

    @property
    def est_notional_usd(self) -> float:
        return self.qty * self.est_price * self.fx

    def as_dict(self) -> dict:
        d = asdict(self)
        d["est_notional_usd"] = round(self.est_notional_usd, 2)
        return d


def portfolio_weights(positions: dict, prices_usd: dict, cash: float):
    """(nav_usd, {symbol: weight}) for the non-zero positions."""
    held = {s: q for s, q in positions.items() if q}
    missing = [s for s in held if s not in prices_usd or not prices_usd[s] > 0]
    if missing:
        raise ValueError(f"no price for held symbols {missing}")
    nav = cash + sum(q * prices_usd[s] for s, q in held.items())
    if not nav > 0:
        raise ValueError(f"non-positive NAV {nav}")
    return nav, {s: q * prices_usd[s] / nav for s, q in held.items()}


def infer_risk_on(weights: dict, sleeve_symbol: str, threshold: float = 0.01) -> bool:
    return weights.get(sleeve_symbol, 0.0) > threshold


def _round_qty(q: float, fractional: bool) -> float:
    """Whole shares, or 4 decimals rounded DOWN (rounding up could overspend)."""
    return math.floor(q * 1e4 + 1e-6) / 1e4 if fractional else float(math.floor(q + 1e-9))


def plan_orders(targets: dict, positions: dict, prices: dict, fx: dict, cash: float, *,
                band: float, switch: bool, cash_buffer: float, fractional: bool,
                cost_bps: float, commission: float = 0.0, min_notional_usd: float = 1.0):
    """Returns (orders, skipped, nav_usd, current_weights). `prices` are in instrument currency."""
    prices_usd = {s: prices[s] * fx[s] for s in prices}
    nav, cur = portfolio_weights(positions, prices_usd, cash)
    if abs(sum(targets.values()) - 1.0) > 1e-6:
        raise ValueError(f"targets sum to {sum(targets.values()):.6f}, expected 1")
    eff = {s: w * (1.0 - cash_buffer) for s, w in targets.items()}
    for s in eff:
        if s not in prices_usd or not prices_usd[s] > 0:
            raise ValueError(f"no price for target symbol {s}")
    symbols = sorted(set(eff) | {s for s, q in positions.items() if q})
    orders, skipped = [], {}
    for s in symbols:
        tw, cw = eff.get(s, 0.0), cur.get(s, 0.0)
        if s not in eff:
            reason = "not in strategy"
        elif switch:
            reason = "switch"
        elif abs(tw - cw) > band + 1e-12:
            reason = "outside band"
        else:
            skipped[s] = {"current": round(cw, 4), "target": round(tw, 4)}
            continue
        target_qty = _round_qty(tw * nav / prices_usd[s], fractional) if tw > 0 else 0.0
        dq = target_qty - positions.get(s, 0.0)
        if abs(dq) * prices_usd[s] < min_notional_usd:
            continue
        orders.append(Order(s, "BUY" if dq > 0 else "SELL", abs(dq), prices[s], fx[s], reason))

    sells = [o for o in orders if o.side == "SELL"]
    buys = [o for o in orders if o.side == "BUY"]
    c = cost_bps / 1e4
    avail = cash + sum(o.est_notional_usd for o in sells) * (1 - c) - commission * (len(sells) + len(buys))
    need = sum(o.est_notional_usd for o in buys) * (1 + c)
    if buys and need > avail:
        scale = max(avail, 0.0) / need
        for o in buys:
            o.qty = _round_qty(o.qty * scale, fractional)
        buys = [o for o in buys if o.qty * prices_usd[o.symbol] >= min_notional_usd]
    return sells + buys, skipped, nav, cur
