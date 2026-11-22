import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, synthetic
from paper import shadow, strategy
from paper.market import FrameMarketData


def test_shadow_tranche_is_average_of_weekday_books():
    cal, frames, ndx = synthetic()
    md = FrameMarketData(frames, ndx)
    df = shadow.compute(md, cal, "F2", "US", "2026-03-02", "2026-09-30", cost_bps=5.0)
    assert df["tranche_nav"].iloc[0] == pytest.approx(100.0)
    for d, r in df.iterrows():
        if r["rebalanced_book"]:
            assert shadow.WEEKDAYS[d.dayofweek] == r["rebalanced_book"]
            assert abs(sum(json.loads(r["book_targets"]).values()) - 1) < 1e-9


def test_shadow_tranche_daily_return_equals_mean_book_return():
    cal, frames, ndx = synthetic(seed=7)
    md = FrameMarketData(frames, ndx)
    df = shadow.compute(md, cal, "F2", "US", "2026-01-05", "2026-06-30")
    # rebuild book daily returns from their NAVs (all rebased to 100 on the same day)
    book_r = df[[f"nav_{w}" for w in shadow.WEEKDAYS]].pct_change().iloc[1:]
    tr_r = df["tranche_nav"].pct_change().iloc[1:]
    assert np.allclose(tr_r.values, book_r.mean(axis=1).values, atol=1e-12)


PANEL = os.path.join(ROOT, "update_2026_11", "panel.parquet")


@pytest.mark.skipif(not os.path.exists(PANEL), reason="research panel not available")
def test_decisions_match_the_backtest_engine_exactly():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "update_2026_11"))
    import common as C
    P = C.panel()
    ndx = P["ndx"]
    for tier, (lev, vt) in {"F2": ("qld", 0.30), "F1": ("qqq", 0.20), "F3": ("tqqq", 0.20)}.items():
        ref = C.strategy_weights(P, lev=lev, voltgt=vt, off="ief", gold="gold_phys")
        days = P.index[P["mark"]]
        sample = days[np.linspace(400, len(days) - 1, 60).astype(int)]
        for d in sample:
            dec = strategy.decide(ndx[ndx.index <= d], d, tier, "US")
            sleeve = strategy.instruments(tier, "US")[0]
            assert dec.targets[sleeve] == pytest.approx(ref.loc[d, lev], abs=1e-12)
            assert dec.targets["IEF"] == pytest.approx(ref.loc[d, "ief"], abs=1e-12)
            assert dec.targets["GLD"] == pytest.approx(ref.loc[d, "gold_phys"], abs=1e-12)
