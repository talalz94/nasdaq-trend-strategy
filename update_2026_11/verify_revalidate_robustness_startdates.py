"""
ROBUSTNESS REVIEW of update_2026_11/revalidate.py  --  part 4
Is the '1989-90 single episode' finding a knife-edge of the chosen start date?
For stats start years 1988..1995 (end = 2026-07-24) report F3/F2 Calmar, the F2-F3 gap, and the
lag-2 Calmar drop of each.  Also zero out ONLY the 1989-10-01..1991-01-31 episode.
run: python -u update_2026_11/verify_revalidate_robustness_startdates.py
"""
import sys, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel(); FREEZE = C.FREEZE
F3 = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20))
F2 = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30))
F3L = C.backtest(P, C.strategy_weights(P, lev="tqqq", voltgt=0.20), lag=2)
F2L = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30), lag=2)
print(f"{'start':>6} {'F3 Cal':>7} {'F2 Cal':>7} {'F2-F3':>7} | {'F3 lag2 drop':>12} {'F2 lag2 drop':>12}")
for y in range(1988, 1996):
    a = f"{y}-01-01"
    c3, c2 = C.stats(F3, a, FREEZE)["calmar"], C.stats(F2, a, FREEZE)["calmar"]
    l3, l2 = C.stats(F3L, a, FREEZE)["calmar"] - c3, C.stats(F2L, a, FREEZE)["calmar"] - c2
    print(f"{y:>6} {c3:7.3f} {c2:7.3f} {c2-c3:+7.3f} | {l3:+12.3f} {l2:+12.3f}")


def z(r):
    r = r.copy(); r[(r.index >= "1989-10-01") & (r.index <= "1991-01-31")] = 0.0
    return r


c3, c2, l3, l2 = (C.stats(z(x), None, FREEZE)["calmar"] for x in (F3, F2, F3L, F2L))
print(f"\nIS with ONLY 1989-10..1991-01 flattened: F3 {c3:.3f} F2 {c2:.3f} (F2-F3 {c2-c3:+.3f}) | "
      f"lag-2 drop F3 {l3-c3:+.3f} F2 {l2-c2:+.3f}")
