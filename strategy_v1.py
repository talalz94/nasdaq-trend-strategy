"""
FROZEN baseline — the proven winner from phase 1. Do not change this file; new
work must beat it without altering it (idea #6: don't break what works).
"""
STRATEGY_V1 = dict(
    name="Confluence_oversold",     # WT bull cross while wt2<-53 AND squeeze momentum rising
    stop=0.02,                      # 2% hard stop
    exit="bearish WT cross",        # ride the full bounce
    cost_bps_side=2.0,              # stocks; gold uses ~1.5
)
# Phase-1 result (1h S&P 500, 42 days): +48 bps/trade net, PF 1.58, win 43%,
# 850 trades, bootstrap p(mean<=0)=0.0000, portfolio +7.5% vs +0.76% buy&hold.
