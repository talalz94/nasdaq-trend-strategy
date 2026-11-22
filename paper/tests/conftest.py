import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("NDX_PAPER_NO_TOAST", "1")        # tests never show real desktop notifications

from paper.config import load_config            # noqa: E402
from paper.engine import Engine                 # noqa: E402
from paper.market import FrameMarketData        # noqa: E402
from paper.nyse_calendar import NYSECalendar    # noqa: E402

ET = "America/New_York"


def frame(prices, index=None, dividends=None, splits=None):
    prices = np.asarray(prices, dtype=float)
    df = pd.DataFrame({"Close": prices, "Adj Close": prices}, index=index)
    df["Dividends"] = 0.0 if dividends is None else np.asarray(dividends, dtype=float)
    df["Stock Splits"] = 0.0 if splits is None else np.asarray(splits, dtype=float)
    return df


def synthetic(start="2023-01-03", end="2027-03-31", seed=1, ndx_returns=None):
    """NDX random walk plus leveraged sleeves derived from it, IEF with monthly dividends, GLD."""
    cal = NYSECalendar()
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in cal.sessions(start, end)])
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.011, len(idx)) if ndx_returns is None else ndx_returns(idx, rng)
    ndx = pd.Series(10000 * np.cumprod(1 + r), index=idx)
    frames = {}
    for sym, lev, base in [("QQQ", 1, 300.0), ("QLD", 2, 60.0), ("TQQQ", 3, 40.0)]:
        frames[sym] = frame(base * np.cumprod(1 + lev * r), idx)
    ief = 100 * np.cumprod(1 + rng.normal(0.0001, 0.003, len(idx)))
    div = pd.Series(0.0, index=idx)
    firsts = pd.Series(idx, index=idx).groupby(idx.to_period("M")).min()
    div[pd.DatetimeIndex(firsts.values)] = 0.30
    frames["IEF"] = frame(ief, idx, dividends=div.values)
    frames["GLD"] = frame(200 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(idx))), idx)
    return cal, frames, ndx


class ClockedMarketData(FrameMarketData):
    """Serves only data up to the simulated clock's date (today's bar included), like the live feed."""

    def __init__(self, frames, ndx, clock):
        super().__init__(frames, ndx)
        self.clock = clock

    def _cut(self):
        return pd.Timestamp(self.clock["now"].date())

    def _frame(self, symbol):
        h = super()._frame(symbol)
        return h[h.index <= self._cut()]

    def _ndx_series(self):
        s = super()._ndx_series()
        return s[s.index <= self._cut()]


def et(y, m, d, hh, mm):
    return pd.Timestamp(dt.datetime(y, m, d, hh, mm)).tz_localize(ET)


@pytest.fixture
def make_engine(tmp_path):
    def _make(frames, ndx, now, **cfg_over):
        clock = {"now": now}
        over = {"paths": {"data_dir": str(tmp_path)}, "alerts": {"desktop_notifications": False}}
        for k, v in cfg_over.items():
            over.setdefault(k, {}).update(v)
        cfg = load_config(os.path.join(str(tmp_path), "none.toml"), over)
        md = ClockedMarketData(frames, ndx, clock)
        eng = Engine(cfg, md=md, now_fn=lambda: clock["now"], data_dir=str(tmp_path), quiet=True)
        return eng, clock
    return _make
