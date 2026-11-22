# Nasdaq-100 trend strategy: research and paper trading

A weekly, rules-based, leveraged Nasdaq-100 trend-following strategy; the research that produced and stress-tested
it (2026); and a local paper-trading system that runs it automatically on a Windows PC without any broker account,
API key or password.

> **Research only, not investment advice.** All results are backtests or simulations on free public data. Leveraged
> ETFs can lose most of their value quickly, past performance does not predict future returns, and taxes are not
> modelled. Nothing in this repository places real orders.

The project began as a study of LazyBear's WaveTrend and Squeeze Momentum indicators on S&P 500 stocks. On 42 days
of hourly data an oversold-bounce entry showed a real edge, but on ten years of daily data the apparent profits of
the indicator signals were mostly market drift: they did not beat simply holding for the same period. So the research
moved to the trend of the Nasdaq-100 itself, leverage and diversifiers, and ended with the strategy below. The full
story, including everything that failed, is in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## The strategy

Once a week, at the close of the week's last trading day (normally Friday), from Nasdaq-100 (^NDX) daily closes:

1. **Trend:** risk-on when the NDX closes above its 100-day exponential moving average; otherwise risk-off.
2. **Volatility scale:** `min(1, target vol / 20-day realised vol)`, halved in August and September.
3. **Choppiness:** full size if the 20-day efficiency ratio is at least 0.25, half size otherwise.
4. **Weights:** Nasdaq sleeve = 0.85 × the result of steps 1–3 (0 when risk-off); 7–10-year Treasuries (IEF) =
   0.85 − sleeve; physical gold (GLD) = 0.15 always.

| Tier | Nasdaq sleeve | Target vol | Note |
|---|---|---|---|
| F1 | QQQ (1×) | 0.20 | lower risk; held up best in the stress tests |
| **F2** | **QLD (2×)** | **0.30** | **the reference tier** |
| F3 | TQQQ (3×) | 0.20 | US-listed only (no UCITS 3× fund) |

Execution: market-on-close at the decision day's close, a 2-percentage-point no-trade band, and a risk-on/off switch
always trades in full. A non-US investor can use a UCITS route (LQQ.PA or CNDX.L, CBU0.L, IGLN.L) traded the next
European session. Full specification: [`STRATEGY_RULEBOOK.md`](STRATEGY_RULEBOOK.md).

## What to expect (honest planning numbers)

From the November 2026 re-check: the average of five books that rebalance on each weekday, so no lucky rebalance
day; 5 bps per trade; in-sample to 2026-07-24; ranges = since 2000 (base) to the better of the 1988 and 2007 starts
(upside).

| Tier | CAGR | Worst drawdown / 1-in-20 | QQQ buy and hold, same period |
|---|---|---|---|
| F1 | 9.1–10.6% | −21% / −24% | 8.6–16.1%, worst −83% |
| F2 | 13.1–17.0% | −28% / −37% | |
| F3 | 16.7–21.6% | −33% / −46% | |

The edge over simply holding QQQ is **avoiding deep drawdowns, not extra return**: there is no statistically
significant return edge, and since 2010 F2 made 15.8% a year against QQQ's 18.8%. Out of sample since the rules
were frozen in July 2026 (to 2026-11-17, real ETFs): F2 −6.0%, QQQ −3.9%. Details, caveats and every rejected idea
(the November re-check alone rejected more than 1,600 variants) are in the research log and the reports.

## Repository map

| Path | What it is |
|---|---|
| [`scanner.py`](scanner.py) | This week's allocation for each tier: `python scanner.py`; `--check` proves it makes exactly the backtest's decisions (2,072 weekly decisions since 1985). |
| [`STRATEGY_RULEBOOK.md`](STRATEGY_RULEBOOK.md) | The rules, execution and the pre-committed review triggers. |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | The complete research history and findings. Start at section 0. |
| [`paper/`](paper/README.md) | Local paper trading: a simulated account, Windows Task Scheduler, desktop alerts; no broker or credentials. 117 tests. |
| [`update_2026_11/`](update_2026_11/) | The November 2026 re-check: scripts plus JSON results of every analysis and its adversarial reviews. |
| `*.html` | The reports of each research phase; open them in a browser (`Recheck_Nov2026.html` is the latest, `Final_Report.html` the July summary). |
| `*.py`, `*.json` (top level) | The 2026 research phases and their results. [`RESEARCH_LOG.md`](RESEARCH_LOG.md) section 2 describes the main modules and section 3 names each phase's scripts; a few one-off helpers are not documented individually. |

## Quick start

```powershell
pip install -r requirements.txt
python scanner.py                                                   # this week's allocation (Yahoo data)
python -m pytest paper/tests -q -p no:cacheprovider --rootdir paper # paper-trading tests (a few minutes)
python paper/runner.py replay --start 2026-07-24 --end 2026-11-17   # replay the paper account over history
```

To paper-trade the strategy on your own PC, see [`paper/README.md`](paper/README.md): one command starts a simulated
account and one registers the scheduled tasks. The PC does not need to be on at a fixed time; a decision missed
while it is off is made as soon as it is back on, and an idle check costs well under a second of CPU.

## Data

Every input is free. Daily prices come from Yahoo Finance through `yfinance` (no account). Hourly US stock data for
the early phases came from Alpaca's free market-data API, which needs your own keys in `.env` (see `.env.example`).
**No raw price datasets are included**: Yahoo's and Alpaca's terms do not allow redistribution. The repository holds
derived results only (equity curves, statistics, the sample replays with their fill prices). Rebuild the data with the
scripts (a few minutes; this also refreshes the published `update_2026_11/panel_validation.json`):

```powershell
python letf_data.py; python macro_data.py; python multi_panel.py; python geo_panel.py
python update_2026_11/panel.py
```

`scanner.py --check` and one parity test need `update_2026_11/panel.parquet` (built above); the test is skipped
without it. Some November 2026 analyses need extra downloads first (`update_2026_11/*_fetch.py`) and re-runs use
data up to today rather than the published 2026-11-17 cut-off; [`RESEARCH_LOG.md`](RESEARCH_LOG.md) section 7 lists
the order and the exceptions.

The first research phases (WaveTrend and Squeeze on S&P 500 stocks and on gold) are **not reproducible from this
repository alone**: they read two separate local datasets that are not published, an hourly S&P 500 price store
(`<STOCK_PRICES_DIR>/<interval>/*.parquet` with columns ticker, datetime, date, open, high, low, close, volume, turned
into an indicator cache by `build_cache.py`) and Dukascopy XAUUSD gold (`<GOLD_DATA_DIR>/dukascopy_cfd_XAUUSD_<tf>.parquet`
with an open_time column). Their results and reports are included.

## How the research was checked

- No look-ahead: signals use closes up to the decision day, trades happen at or after that close.
- Benchmarks against buy-and-hold drift over the same holding period and against an equal-exposure leverage dial,
  so beta is not mistaken for skill.
- Out-of-sample periods, other markets (8 indices), 99 years of S&P 500 history, a 29-asset search, published
  strategies re-implemented on the same data, and walk-forward re-optimisation (which did worse than frozen rules).
- The November 2026 re-check ran 13 analyses, each attacked by an integrity reviewer and a robustness reviewer, with
  pre-registered decision rules for the second round. The paper-trading code was reviewed the same way (54 findings,
  all fixed with regression tests).

## Requirements

Python 3.11 or newer (developed on 3.13). Packages: [`requirements.txt`](requirements.txt). Paper trading's
scheduling is Windows-specific (Task Scheduler); everything else runs anywhere.
