# WaveTrend × Squeeze — Trading Research Log

Master documentation of everything learned across this project, a Python research
repository. All source data directories are **read-only** and were
never modified.

Last updated: 2026-11-21 (paper trading: catch-up mode; repository published).  **STATUS: re-validated on data to 2026-11-17; rules unchanged except three corrections; planning numbers re-based. Paper-trading automation BUILT in `paper/` (local simulated account, no broker or credentials; adversarially reviewed), not yet started. Next: start the paper account + confirm SBP/tax before funding.**

---

## 0. TL;DR — READ THIS FIRST (current as of 2026-11-18, after the Nov-2026 re-check)

**THE STRATEGY** (`scanner.py` v2 — `python scanner.py`; `python scanner.py --check` proves it makes exactly the
backtest's decisions on all 2,072 weekly decisions since 1985). Weekly at the Friday close (holiday Friday -> that
week's last session), from ^NDX daily closes:
> 1. risk-on = NDX > EMA-100 (ewm span 100, adjust=False). The July "confirmed yield-curve" clause was a logical
>    no-op (above AND NOT(inverted AND NOT above) == above) and is DELETED.
> 2. scale = min(1, VT / vol20); x0.5 when the decision date is in Aug or Sep
> 3. engine = scale if ER20 >= 0.25 else 0.5 x scale; 0 when risk-off
> 4. leveraged sleeve = 0.85 x engine; **IEF** = 0.85 - sleeve (**NOT TLT**); **physical gold (GLD) 15% always**
>
> Tiers (leverage is a RISK DIAL; they tie risk-adjusted): **F2 = QLD, VT 0.30 (reference)** · F1 = QQQ, VT 0.20
> (lower risk; best in every stress scenario) · F3 = TQQQ, VT 0.20 (US-listed only; no UCITS 3x fund).
> Execution: signal 15:40-15:45 ET Friday, market-on-close orders, 2pp no-trade band, always trade on/off switches;
> fallback = next session's close. UCITS route: trade Monday's European session (LQQ.PA or CNDX.L, CBU0.L, IGLN.L).

**PLANNING NUMBERS** (5-weekday tranche = no rebalance-day luck; 5 bps; in-sample to 2026-07-24; base case = since 2000):

| Tier | US CAGR base-upside | worst DD / 1-in-20 DD | Calmar | UCITS CAGR |
|---|---|---|---|---|
| F1 1x | 9.1-10.6% | -21% / -24% | 0.44-0.51 | 8.3-9.7% |
| F2 2x | 13.1-17.0% | -28% / -37% (to -40%) | 0.47-0.61 | 12.4-15.9% |
| F3 3x | 16.7-21.6% | -33% / -46% (to -51%) | 0.53-0.68 | n/a |
| QQQ buy & hold | 8.6-16.1% | -83% / -88% | 0.10-0.30 | |

The July headline (F3 24.2% / -30.7% / 0.79) was the Friday book with a gold-MINERS sleeve and an IEF-like bond, and
the live scanner actually held TLT (that version: -40.7% maxDD, 2022 -33.9%). Not in the plan: the Friday premium
(upside), a parameter-selection haircut, real UCITS frictions (measured 1-3pp), taxes, NDX drift below history.

**WHAT THE NOV-2026 RE-CHECK ESTABLISHED** (2 rounds, 13 analyses, each checked by an integrity reviewer and a
robustness reviewer plus critics; the 5 round-2 tests were pre-registered; plus a separate fact-checked market sweep):
1. **Implementation mismatch fixed.** The backtest bond was a 10y synthetic (IEF-like) but the scanner said TLT (20y+)
   -> IEF now. The backtest "gold" was ^XAU gold MINERS -> physical gold now (long-run results barely change).
2. **The yield-curve rule was a no-op**; its "+1%/yr" was 100% the bond off-leg. Genuine overlays (curve, rate shock,
   dollar, credit, VIX term structure; 54 variants) all rejected. The lesson "only leading indicators help" is withdrawn.
3. **Rebalance-day luck.** Friday beat the 5-weekday average by +2.4pp/yr (F2) in-sample AND out-of-sample, but FAILED
   the pre-registered structural test (only 3 of 7 other markets; 5 needed); its 1988-99 mechanism (stale Nasdaq
   closes -> Friday-to-Monday continuation) has expired; it peaks at the Friday-tuned EMA-100. Plan on the tranche;
   keep Friday as the live day; never use a Monday-close signal (worst book in 6 of 7 markets).
4. **The edge over QQQ is drawdown avoidance, not return.** No significant CAGR edge; the Calmar edge is borderline
   (P 0.02-0.06), rests on 2000-02, and is not significant since 2007. Since 2010 F2 made 15.8%/yr vs QQQ 18.8%.
5. **Chop rule = exposure dial** (pre-registered REPLACEABLE for F2/F3: no matched-exposure edge since 2007 or across
   7 markets; its edge was a 1990s era effect; it drives ~2/3 of turnover). **Vol target** REPLACEABLE (F2) /
   INCONCLUSIVE (F3; its value is the 2000 crash). BOTH KEPT UNCHANGED: replacing either needs a new tuned parameter.
6. **Nothing re-tuned.** Off-leg switch rules (98 variants), managed-futures sleeves (319) and parameter re-tuning
   (1,133; walk-forward picks land mid-pack, oracle-minus-pick 0.18 Calmar) were all rejected. Gold ties IEF in the
   same 15% slot (de-levering only); kept.
7. **True out-of-sample** (2026-07-27..11-17, real ETFs): F2 -6.0% (Friday) / -10.7% (tranche), F3 -9.0%, scanner-TLT
   -10.85%, QQQ -3.9%, TQQQ -18.9%. One full-size week after the 6 Oct record (vol targeting maxed out at 15% vol)
   cost F2 -9.4% / F3 -13.9%; the EMA exit then saved ~14pp. 9th pct (Friday) / 2nd pct (tranche) of 81-day windows:
   weak evidence against the edge, not used for any decision.
8. **Model-input corrections** (financing spread on pre-2010 LETF sims, yield-dependent duration, month-end gold) cost
   at most 0.1 / 0.35 / 0.7 pp/yr (F1/F2/F3) and change no conclusion.
9. **Weekly drift vs the backtest's daily constant mix** is immaterial (-0.05 to -0.13pp CAGR).

**CURRENT STATE (2026-11-17 close):** risk-off since the 16 Oct mark; all tiers 85% IEF + 15% GLD. Re-entry at Fri
2026-11-20 needs an NDX close above Thursday's EMA-100 (~28,891, +7%); it would be CHOP-HALVED (ER20 ~0.00) to F2 ~30%
QLD. Exposure now = the off-leg only: 10y +100bp ~ -5.7%; gold -15% ~ -2.25%. Optional 50/50 IEF/BIL off-leg halves
the rate shock (-2.85%) for ~0.2-1pp/yr: a pre-specified option to decide before the 25 Nov PCE release, or never.

**MARKET CONTEXT (fact-checked, Nov 2026; `update_2026_11/market_context.json`):** Fed hiking under Chair Warsh
(3.75-4.00%; Dec hike ~priced); US-Iran war oil shock; core CPI +0.3%/m three months running; 10y 5.10% (5.31% close
on 5 Oct, the highest since 2002), real 10y ~2.9%; NDX -13.5% from the 6 Oct record on AI-capex worries (5 Nov -6.36%,
TQQQ -18.9%; Nvidia's CEO resigned 12 Nov); Q3 GDP +3.4%, payrolls +165k, HY spread 3.34% (no recession signal);
gold -32% from its January record; DXY 103.5. Read: 2022-type inflation/real-rate correction, 2018-Q4 shape,
2000-02 tail risk.

**INVESTOR GO/NO-GO (UNCONFIRMED):** SBP FE Manual ch.20 para 13 (one designated bank
branch, US$25k/yr, repatriation of dividends AND sale proceeds) may apply even with the UAE account; if repatriation
covers in-account sales, weekly rebalancing may be infeasible. Pakistani tax on foreign ETF gains: 15% (s.37A) vs
slab rates up to 45%, unconfirmed. US estate tax above ~$60k of US-situs assets -> UCITS route.

**STILL TRUE FROM JULY:** benchmark against drift and against an equal-exposure dial; static beats timing; never
re-optimise; never short via SQQQ; never fully de-lever; the rules generalize across markets but the magnitude does
not; the Nasdaq survived a 29-asset search; on the S&P a parameter-free 200-day rule beats our Nasdaq-tuned settings.

**NEXT:** the paper-trading automation is BUILT (`paper/`: a local simulated account, no broker, API key or password;
adversarially reviewed, 117 tests; section 3 "PAPER-TRADING BUILD") but NOT started. Start it and paper-trade for
months. The PC need not be on at a fixed time: a decision missed while it is off is made as soon as it is on and
trades at the next close (1-3 trading days late barely changes the backtest return; a week late costs 5-7 pp/yr), so
switch it on at least once between Saturday and Wednesday. Confirm SBP, tax and residency with a Pakistan-qualified
cross-border adviser before funding anything.

---

## 1. Data sources (all FREE, all read-only)

| Source | What | History | Access | Caveats |
|---|---|---|---|---|
| **S&P 500 price store** | OHLCV 15m/1h/4h/1d, 503 names | ~42 days (May–Jul 2026) | `../../Stock Screener` (a separate local project, not included; yfinance-built); `dataset.py` API | tiny window; `daily_update.py --days 730 --intervals 1h` extends to 2yr |
| **Dukascopy gold** | XAUUSD 1m/5m/15m/1h/4h CFD | **2005–2026 (21yr) hourly** | `../../Binance/data/dukascopy_cfd_XAUUSD_*.parquet` (a separate local project, not included) | 24/5; leverage-friendly |
| **Alpaca Market Data** | US equity/ETF bars | daily **2016+** (SIP) | REST `data.alpaca.markets/v2/stocks/bars`, keys in `.env` | **SIP free gives full history + volume**; IEX only 2020+; **hourly served in ~291-bar pages → 10yr×500 names ≈ 6–7h** (daily = 1 req/ticker, fast) |
| **yfinance** | indices + ETFs, daily | **^NDX 1985+, ^IXIC 1971+, QQQ 1999+, TQQQ/SQQQ 2010+, ^IRX** | `yfinance` | best free *long* daily history; intraday capped (1h≤730d) |

**Simulated LETFs (for max leverage history):** real TQQQ/SQQQ start 2010. To stress-test through
the dot-com/GFC crashes we simulate them from ^NDX daily returns with the true daily mechanics:
`TQQQ_ret = 3·ndx_ret − 0.95%/252 − 2·rf_daily` (SQQQ = −3·…), rf from ^IRX, daily-compounded.
Validated vs real 2010–26: sim x248 vs real x322 over 16yr (**conservative undershoot ~1.3%/yr,
tracking error 3.5%/yr**) — accurate on daily mechanics/decay, understates the 2010s trend "leverage
bonus". Fine for strategy-vs-benchmark comparison (same sim both sides) and for showing crash decay.
Built by `letf_data.py` → `data_letf/letf_daily.parquet` (1985-10-02 … 2026-07-24, 10,281 days).

---

## 2. The engine (reusable code)

Backtest data-flow: **raw OHLC → `indicators.compute_all(df)` → cache dict `{ticker:{col:np.array}}`
→ `bt.simulate(...)` → `bt.metrics(...)`**. `sweep.run_spec(cache, spec)` builds parameterised
signals via `sweep.make_signal` and reuses the same `bt.simulate`.

| File | Role / key funcs |
|---|---|
| `indicators.py` | LazyBear WaveTrend (10/21), Squeeze (BB20×2 / KC20×1.5, TrueRange) + colored EMA/SMA slope (Nance) + MA-Shift (ChartPrime, HMA + rolling-percentile). `compute_all(df)`; linreg validated vs numpy.polyfit. |
| `bt.py` | `simulate()` (no-lookahead state machine, stop-before-target intrabar), `metrics()`, `run_strategy()`, `build_signals()`, `load_cache()`. `CACHE` = env `CACHE_DIR` (default `cache/`). |
| `sweep.py` | overfit-guarded grid: `make_signal(d,spec)`, `run_spec`, `split_eval` (train/test), `build_configs`. Env `SWEEP_CACHE / SWEEP_MODE / SWEEP_TAG`. |
| `regime.py` | market regime = equal-weight index vs **200-day SMA (trend) + realised vol**. BULL / BEAR / CHOPPY. `regime_series`, `tag_trades`. |
| `alpaca_fetch.py` | threaded SIP fetch, split-adjusted, **RateLimiter(170/min)**, robust never-truncate pagination, RTH filter (ET 9–15). |
| `assets.py` | gold/multi-asset loaders → bt cache dicts. |
| `letf_data.py` | build max-history NDX/QQQ/TQQQ/SQQQ + simulated LETFs. |
| `letf_opt.py` | holistic leveraged-ETF optimizer (signal×tf×rebalance×allocation×vol-target×scale-in). |
| `tqqq.py` | first TQQQ/SQQQ timing study (2016–26). |
| **`engine.py`** | **side-effect-free strategy engine of the July phases** (new work: `update_2026_11/common.py`). `data()`, `weights()`, `backtest()`, `stats()`, `by_decade()`, `proj1000()`. All later scripts build on it. |
| **`scanner.py`** | **live weekly scanner — what to hold now (v2: tiers F2 / F1 / F3).** `python scanner.py [--json] [--log]`; `--check` = parity test with the backtest. Prints allocations; does NOT place orders. |
| `STRATEGY_RULEBOOK.md` | plain-English spec of the strategies. |
| `letf_data.py` / `macro_data.py` / `multi_panel.py` / `geo_panel.py` | data builders -> `data_letf/*.parquet` (LETF sims, macro, multi-asset, geopolitical flags). |
| `attribution.py` / `frontier.py` / `frontier2.py` / `validate.py` | Phase B: gap attribution, efficient-frontier tuning, walk-forward + stability. |
| `rotation.py` / `rotation2.py` / `phaseC_final.py` | Phase C: multi-asset rotation (failed) then static diversification (worked). |
| `geo_test.py` / `indicator_test.py` / `final_validate.py` | Phase D: geopolitical adaptation (all failed), seasonality/breadth/vol-term, full validation. |
| `tier1.py` / `crossmarket.py` | 2x vs 3x, ensemble, duration-matched bonds; and the 8-market generality test. |
| `longrun.py` / `fees_reopt.py` / `ucits.py` | Nikkei crash + S&P 1927 validation; cost stack + re-optimisation test; non-US/UCITS route. |

> Column-list constants that must stay in sync when adding indicators: `bt.load_cache` inline `cols`,
> `sweep.CACHE_COLS`, `assets.CACHE_COLS`.

---

## 3. Findings by phase

### PAPER-TRADING BUILD (Nov 2026) — `paper/` — a local simulated account (no broker, no credentials)
No broker account is used: the planned "IBKR paper automation" was built as a **local simulated paper account**:
Yahoo prices (as the scanner), state in a JSON file, Windows Task Scheduler (runs as the Windows user only while logged on,
no stored password), local alerts (alerts.log, ALERT_ACTIVE.txt, Windows toast; an own webhook / healthcheck URL is
optional and empty by default). Nothing in it can place a real order. Usage: `paper/README.md`.
- **What it runs:** the frozen rules through `scanner.py`'s own functions (parity-tested). US route: decision in the
  15:40-15:48 ET window from the live NDX, market-on-close fills at the official close; if the PC is off or asleep in the
  window, the decision is made from the official close as soon as it is on and trades at the next reachable close. UCITS route: decision after the US close; all legs fill together at
  the close of the first later NYSE session on which every listing trades and that closes after the decision
  (research lag 2). 2pp band, switches always trade in full, 5 bps, whole shares, 0.5% cash buffer, dividends on the
  ex-date, splits from Yahoo's back-adjusted closes. A shadow 5-weekday tranche is kept as the planning benchmark.
- **Validation (replays through the production code, 5 bps):** 24 Jul-17 Nov 2026 -5.97% vs the backtest's -6.10%
  (-6.03% at 3 bps); 2016-01-04..2026-07-24 CAGR 21.93% vs 22.24%, maxDD -24.27% vs -24.10%, daily corr 0.997 (the
  gaps are holiday-Friday weeks: the rules decide on Thursday, the backtest skipped them). All 552 replay decisions
  equal the research weights (max diff 4.7e-6 = CSV rounding). The review fixes changed no US-route trade: both
  replays are row-for-row identical before and after.
- **Adversarial review** (5 lenses: fidelity, safety/docs, accounting, time, robustness; every non-low finding
  re-verified independently): 54 findings; 30 of the 31 non-low ones confirmed (1 refuted); all fixed, each with a
  regression test (101 tests then; 117 after catch-up mode and two pre-publication audits). The material ones:
  - UCITS shadow traded European closes at lag 1, a look-ahead that inflated the UCITS benchmark (F2 2016-26 tranche
    22.3% -> 19.0% CAGR after the fix); UCITS legs filled per exchange (on a UK-only holiday the Paris buy ran before
    the London sale and sat in cash for a week); a late UCITS decision filled at a close printed before it.
  - Live price data covered only 2 years, so the shadow tranche would have silently lost its early returns.
  - Splits: a split on a decision day, or a catch-up booking across a split, gave wrong trades or phantom P&L
    (+20% NAV in a test). Now detected from Yahoo's back-adjusted closes (this also catches unrecorded splits such
    as LQQ.PA's 200:1), with a price guard and `split` / `accept-move` commands if Yahoo never adjusts.
  - A CSV open in Excel blocked booking and the decision; a config typo crashed every scheduled run silently; a crash
    could leave a phantom decision row. Now state is saved first, logs are written from a backlog kept in
    state.json, and startup errors are logged and alerted (de-duplicated).
  - A sleeping PC missed the window (Saturday 00:40/01:40 Pakistan time) and its fallback. First answered with a
    power-settings check and an optional -WakeToRun task, then replaced by catch-up mode (below).
  - Unscheduled closures: false "missed" alerts, a lost week on the UCITS route, fallback orders expiring silently.
    Closures are now detected on both routes, alerts withdrawn, lost orders re-made.
  - Smaller: false "missed" alert for an account created after the window; the drawdown trigger was checked only
    on the last session of a catch-up; a tier change rewrote the shadow history; config validation; a replay could
    overwrite the live folder via a case-variant path; the watchdog's grace ignored sleep and Fast Startup; a stale
    lock skipped 30 min of runs; the 3-years-behind-QQQ review trigger is now monitored.
- **Catch-up mode (2026-11-21).** The PC does not have to be on at a fixed time: if it is off or asleep in the Friday
  window, the decision is made as soon as it is on (once the official close is published) from that close and trades
  at the next close a market-on-close order can still reach; the scheduled task also runs on log-on and one minute
  after a wake from sleep, and a run missed while the PC was off starts when it is back. Cost of lateness in the
  research backtest (F2, 5 bps, Friday decision traded k days late): 2000-2026 CAGR 14.8% on time, 15.0 / 15.6 /
  15.1% at Mon / Tue / Wed, 11.9% Thu, 9.8% a week late; worst drawdown -24.1% -> -25.2 / -26.7 / -27.0 / -29.6 /
  -30.3%. So 1-3 days late is noise in return but 1-3 points of drawdown; a week late is expensive. `status` shows
  the account next to the same rules traded on time (the Friday book).
- **Compute cost (measured on an i7-10510U laptop):** a run with nothing due - almost every 5-minute run - takes
  ~0.7 s CPU and ~22 MB for about a second, with no network: a standard-library pre-check (`paper/quickcheck.py`)
  decides whether anything is due before pandas is loaded (pandas alone costs ~3 s CPU to import on that PC); a
  test replays the engine over weeks of simulated time and proves the pre-check never skips work. Booking a close or
  deciding: ~8.5 s CPU, ~125 MB for 15-25 s, 1-5 MB download, about once per trading day. Watchdog: ~2.5 s hourly.
  Total ~4-5 min CPU per day, well under 1% of one core.
- **Not started:** `paper/data` is empty (no account initialised) and no scheduled task is registered; both wait for
  the go-ahead (section 8).

### NOV-2026 RE-CHECK — refresh, audit, two adversarially reviewed rounds — `update_2026_11/`
**Data refresh** to 2026-11-17 (July inputs archived in `data_letf/archive_2026-07-24/`; `geo_panel.py` patched
because ^BCOM was withdrawn from Yahoo). **Shared corrected panel** `update_2026_11/panel.py` -> `panel.parquet`:
real QQQ/QLD/TQQQ spliced over sims (QLD sim corr 0.996, TE 4.2%/yr); IEF/TLT/SHY synthetic treasuries calibrated
on the real ETFs (duration 7.09 / 15.8 / 1.82; TE 2.2 / 5.0 / 0.9%/yr) spliced under the real ETFs; BIL; physical
gold = GLD (2004+) <- GC=F (2000+) <- monthly gold (datasets/gold-prices, log-interpolated, never used for signals);
gold miners kept for comparison; DBMF/KMLM/CTA; yields ^TNX/^TYX/^FVX/^IRX. `common.py` holds the shared mechanics
(strategy_weights, backtest, stats, crises, bootstrap). `main_drift_check.py`: weekly drift vs constant mix -0.1pp.
**Audit:** (1) the scanner held TLT while the backtest held a 10y-like bond (F3: TLT maxDD -40.7% / Calmar 0.60 vs
IEF -30.0% / 0.80); (2) "gold" was ^XAU miners (physical gold gives about the same long-run result); (3) the
confirmed-curve rule is a no-op; (4) Friday rebalance premium (`main_tranche_check.py`: F2 Friday 19.71% / -24.0% /
0.82 vs 5-weekday tranche 17.29% / -27.7% / 0.62; since 2000 15.58% vs 13.40%).
**Round 1** (8 dimensions x 2 reviewers + critic; `quant_results.json`; scripts `oos_live, revalidate, rate_regime,
managed_futures, overlays, analogs, leverage_execution, ucits`): no July conclusion overturned on the Friday basis;
TLT -> IEF; off-leg switches, managed futures, macro overlays and re-tuning rejected; 2x vs 3x is a risk choice (the
"Sharpe edge" was killed: raw Sharpe ignored the risk-free rate); execution = Friday pre-close MOC + 2pp band; the
UCITS route LQQ.PA + CBU0.L + IGLN.L tracks within 0.1-0.3%/yr per instrument, with ~1.1-1.4pp/yr structural extra
cost and +/-1-2pp timing noise; analog base rates (NDX 12m median ~+21%, >25% bear ~21-24%, strategy 12m positive
from 92-96% of risk-off starts). The critic surfaced the Friday premium and re-based planning on the tranche.
**Round 2** (5 PRE-REGISTERED dimensions x 2 reviewers + final decision critic; `r2_results.json`; scripts `r2_*.py`):
chop rule REPLACEABLE (F2 H1 +0.244 / H2 -0.012; F3 +0.155 / +0.007; 7-market median +0.011); vol target REPLACEABLE
(F2) / INCONCLUSIVE (F3, knife-edge); Friday premium NOT structural (criterion A passes, B 3/7 fails); planning table
(see TL;DR) with stationary-bootstrap tails; model hygiene at most 0.7pp/yr with no flips on the Friday basis;
scenario stress from today's state (descriptive only): a V-rebound is only ~half captured (re-entry ~11 Dec near
29,650), a 2022-like bear gives F2 -9.9% vs QQQ -25%, a deflationary bust F2 +3.7% vs QQQ -35%, and a violent +/-8%
range around the EMA is the blind spot (F2 -25% to -38% with pinned noise, -12% to -19% with free noise). F1 was the
best tier in every adverse scenario.
**Market research** (5 dimensions x fact-checker + synthesis; `market_context.json`; 69 findings, 61 confirmed or
corrected).
**Scanner v2** (`scanner.py`; the July version is kept as `scanner_v1_jul2026.py`): IEF, no curve clause, F2/F1/F3
tiers, re-entry bar plus a chop-aware projection of the re-entry size, `--log` CSV, `--check` parity test (0.00 max
weight difference).
**Report:** `Recheck_Nov2026.html` (built by `update_2026_11/build_report.py` from `report_template.html`).


### Phase 1 — WaveTrend + Squeeze on S&P 500 (42-day, 1h)
- Winner: **Confluence_oversold** (WT bull cross & wt2<−53 & squeeze rising) + 2% stop, exit next
  bearish cross → **+48 bps/trade, PF 1.58, win 43%, 850 trades, bootstrap p(mean≤0)=0.0000**;
  edge ~6× matched drift. Portfolio (10 pos, 10% risk): **+7.5% in 42 days vs +0.76% buy&hold**.
- **Scalping HURTS** (quick-exit/trailing cut the edge 50–100%). **Sector matters**: Health Care/
  Utilities/Staples/Financials/Tech work; **Comm Services loses** (PF 0.49). Options amplify the edge
  (+11.9%/trade) but same PF, 35% win — leverage not better edge. Report: `WaveTrend_Squeeze_Report.html`.

### Phase 2 — gold + new indicators + leverage/ruin
- **Regime-specific:** oversold-bounce **fails on gold** (1h −5.8 bps, walk-forward confirms).
  Gold wants momentum: SQZ zero-cross / SQZ-colour / **MAShift_regime_long** (buy dip in uptrend,
  +8 bps edge, p=0.02). Champions pass true 21-yr walk-forward (train<2019/test≥2019).
- **Filter ladder** ("eliminate bad moves"): stocks 48→69 bps via sector whitelist; gold 18→38 bps
  via price>MA40. The *right* filter is asset-specific (quality for mean-rev, trend for momentum).
- **Leverage/risk-of-ruin (gold, $1000):** full-Kelly ~9–10× but 34–49% ruin; **safe ~3× → ~$9.7k,
  <3% ruin; 5× (½-Kelly) → ~$27k but ~45% DD**. Report: `WaveTrend_Squeeze_Phase2.html`.

### Phase 3 — 10-year daily sweep (498 names, 462 survivors) & the DRIFT TRAP
- Sweep champion (MA-Shift+trend) **+211 bps OOS looks great but edge vs 48-day drift ≈ −4 bps**;
  portfolios (CAGR 11–15%) **trail buy&hold 15.8%**. Weekly amplifies the illusion. **On daily holds
  the numbers are beta.** Real edge = intraday.
- **Regime playbook (OOS):** bull→MA-Shift (+320 bps), bear→SQZ-colour (+448 bps, thin sample),
  choppy→WT-−70 (+286–326 bps). Regime-combo Sharpe **1.93**, maxDD −16% (a better *ride*, not better
  return). Momentum's best sectors (Tech/Financials/Energy) invert phase-1's mean-rev profile;
  Utilities loses. Report: `WaveTrend_Squeeze_Phase3.html`.

### Phase 4 — TQQQ/SQQQ timing (2016–26)
- **YES timing beats leveraged buy&hold** (on this bull decade): hold TQQQ when QQQ>200-SMA else CASH
  → +3,101% vs +2,765%, maxDD −55% vs −82%; MA-Shift version −45% DD, Sharpe 1.01. **Cash ≫ SQQQ**
  (every short-leg variant lost). Report: `TQQQ_Timing.html`. *(Superseded/contextualised by Phase 5.)*

### BEST-ASSET SEARCH — is the Nasdaq even the right underlying? Files: `bestasset.py`, `assetcombo.py`
Report: `BestAsset_Report.html`. NOTE: the earlier 8-market test asked "do the RULES generalize"; this asks the
DIFFERENT question "is there a BETTER ASSET". Identical rules on **29 assets** (global indices, crypto, semis,
sector ETFs, mega-cap stocks), each with its own simulated leveraged sleeve, engine-only (no gold sleeve).

**Raw ranking (own full history) — two apparent winners:** Bitcoin 3x Calmar **1.28** (64% CAGR, -50% DD),
Bitcoin 2x 1.23, then **Nasdaq Comp 2x 0.81 / Nasdaq-100 2x 0.78 / Nasdaq-100 3x 0.77**, Apple 1x 0.74,
Nvidia 0.62-0.66, Tech ETF 0.53-0.54. Everything else (semis ^SOX, India Nifty, Korea, Taiwan, Brazil, Mexico,
EuroStoxx, DAX, Nikkei, HSI, gold, gold miners, biotech, ARKK) **failed to beat the Nasdaq**.

**Candidate 1 — BITCOIN: edge DECAYED ~90%.** Split sample: 2015-20 Calmar **2.52** (87.7%/yr) ->
2021-26 Calmar **0.23** (7.7%/yr), vs Nasdaq 3x holding 1.31 -> 0.79 (24.6%/yr). Blends look superb on
2015-2026 (50/50 Nasdaq3x+BTC2x = Calmar 1.40, Sharpe 1.54) but are **powered by the one-off adoption boom** —
do NOT act on them. Crypto sleeve = conviction call, not backtest-supported.

**Candidate 2 — MEGA-CAP STOCKS: hindsight trap, and the trap backfires.** Honest rule-based test (monthly rotate
to the trailing-12m momentum leader among 12 mega-caps) vs hindsight fixed picks, 2x:
rule-picked 31.0% CAGR / -43% / Calmar **0.72**; **hindsight always-NVDA 21.8% / -45% / Calmar 0.49 (WORSE than
the index!)**; always-AAPL 0.66; **Nasdaq-100 index 20.2% / -26% / Calmar 0.78 = BEST**. NVDA suffered -62%
drawdowns through dot-com/2008/2018/2022 — the "unbeatable" impression is only the recent AI era.
**Why the index wins: it is cap-weighted and reconstituted, so it ACCUMULATES the winners automatically** — you get
the Nvidia effect without having to pick it, with drawdowns diluted across 100 names.

**Stability (the deciding test — searching 29 assets manufactures false champions):** split each candidate's history.
**Nasdaq-100 3x: 0.87 -> 0.72 = most stable in the universe** (41 yrs). Bitcoin 2.27 -> 0.59 (collapse),
Semis ETF 3x 0.14 -> 0.73 (erratic), India weak in both (0.29/0.35). NVDA/AAPL improve — which is what
survivorship looks like from the winners' side.

**ANSWER: nothing beats it reliably.** The Nasdaq-100 wasn't chosen by searching — it's where we started, and it
**survived a search designed to dethrone it**, which is the strongest confirmation available. More return is
available (small BTC sleeve, mega-cap momentum rotation) but you'd be **buying return with drawdown, not finding a
better strategy**. Side-finding: **across almost every asset 2x beat 3x on Calmar — the THIRD independent
confirmation** of the lower-leverage result.

### EXTERNAL BENCHMARKING — vs published & community strategies. Files: `external.py`, `fairtest.py`
Report: `External_Report.html`. Searched literature/forums, then **re-implemented 11 strategies on OUR data,
period (1988-2026) and cost model** (published CAGRs across different eras/assets are not comparable).

**Head-to-head by Calmar (identical terms):** **OURS 2x 0.79 (#1)**, **OURS 3x 0.72 (#2)**, vol-target-12% 0.33,
VAA-G4 0.31, Gayed-on-Nasdaq 0.28, Faber GTAA 0.27, Antonacci GEM 0.25, Gayed LRS S&P 0.24, leveraged 60/40 0.23,
**HFEA 0.20** (13.9% CAGR, **-70% DD, -63% in 2022** — confirmed fragile), Gayed weekly 0.15.
Cleanest comparison = Gayed's rule on the SAME asset (Nasdaq, 3x): 22.7% CAGR but **-82% DD vs our -33%** —
our vol-targeting/half-size-chop/bond+gold sleeves buy the drawdown reduction while keeping the return.

**CRITICAL FAIRNESS TEST (`fairtest.py`) — is it the rules or the asset?** Ran our ruleset vs Gayed's on the SAME
asset/leverage/off-leg/costs:
- **Nasdaq: OURS WINS BIG** — 3x Calmar **0.68 vs 0.30**; 1x **0.79 vs 0.37** (>2x better).
- **S&P 500: GAYED'S SIMPLER RULE WINS** — 3x **0.27 vs our 0.22**; 1x **0.34 vs our 0.27**.
=> Our parameters (vol .20, ER .25, EMA-100) are **tuned to Nasdaq's volatility signature**; Gayed's rule has no
parameters to misfit. **The head-to-head win is genuine but CONDITIONAL on the Nasdaq.** If ever applying this to
another index, START FROM THE SIMPLE 200-DAY RULE, not our parameters.

**Other findings:** our core idea has independent academic validation — Gayed's **2016 Charles H. Dow Award** paper
is the same foundation (leverage above 200d MA, T-bills below); what we added on top is where our edge lives.
Keller's **BAA** (published Calmar ~0.78, 11.2% CAGR / -14.4% DD) is genuinely competitive risk-adjusted but targets
a different risk level — worth a look if a smoother ride is wanted. **2022 hurt EVERY leveraged strategy tested**
(HFEA -63%, Gayed -53%, vol-target -24%, ours -25%) — nobody in the literature has solved it either.
**Multiple independent researchers converge near Calmar ~0.80** — likely a real ceiling for single-index trend
following; anyone claiming much better is probably overfitting or measuring a favourable era.
**CAVEAT:** my multi-asset TAA re-implementations (VAA/GTAA/GEM) are APPROXIMATIONS with asset substitutions
(no REITs/EM ETFs pre-2003) — my VAA scored 0.31 vs BAA's published 0.78, so that gap is likely MY implementation.
**NOT TESTABLE:** managed futures (short free history — still the best untested candidate for the 2022 regime),
stat-arb, options overlays, anything needing paid data.

### PRACTICAL / IMPLEMENTATION — crash scenarios, 99-yr validation, costs, re-optimisation, jurisdiction
Files: `longrun.py`, `fees_reopt.py`, `ucits.py`. Report: `Practical_Report.html`.

**"What if the Nasdaq crashes and never recovers?"** The NIKKEI is the natural experiment (peaked 38,900 in 1989,
then **-5.6%/yr for 23 years**). Strategy result 1990-2012: **+3.6%/yr at 1x (maxDD -17%)** vs index -5.6% and
**3x buy&hold -34.6%/yr (total wipeout)**. It survives a generational bear market. TWO COSTS: leverage stops helping
(1x 3.6% > 2x 3.5% > 3x 2.8%), and it **badly lags the recovery** (Japan 2013-26: strategy 4.8% vs index 14.4%) —
the trend gate re-engages slowly after long bears.

**PRE-1985 VALIDATION EXISTS** — ^GSPC from **1927** (99 years), ^IXIC from 1971, ^N225 from 1965.
S&P full 99yr at 1x: **6.1% CAGR / -25% maxDD** vs buy&hold 6.3% / **-86%** (Calmar 0.24 vs 0.07 — same return,
one third the pain). Best relative era = **1966-82 STAGFLATION: 7.3%/yr vs index 2.5%**, DD -11% vs -48%.
1927-54 (Crash/Depression/WWII): 4.7% vs 2.7%. It **LAGS strong bulls** (1983-99: 10.4% vs 14.8%; S&P 2000-26:
3.0% vs 6.3%). **Confirms the leverage warning independently: on the S&P, 1x beats 3x on Calmar in EVERY era**
(stagflation 0.64 vs 0.18). Nasdaq Composite 1971-2026 full: 1x Calmar 0.86, 2x 0.73, 3x 0.69.

**SHOULD YOU RE-OPTIMISE ANNUALLY? NO — FREEZE THE RULES.** 108-point grid re-fit each January, all modes scored on
1998-2026: **frozen Calmar 0.82**; re-fit on all history 0.69; on last 10y **0.55**; on last 5y **0.55** (20-24
parameter changes). Shorter lookback = worse. Re-optimisation chases noise; the parameter plateau exists precisely
because exact values don't matter. Only revisit on STRUCTURAL failure (DD worse than -35%, or 3+ yrs lagging
buy&hold with no bear market to explain it) — that's a break check, not an optimisation.

**FULL COST ACCOUNTING** — turnover 11.9x/yr. Idealised 24.17% -> +ETF expense ratios (TLT .15%, GLD .40%) -0.16%
-> 6bps spreads -0.45% -> 10bps -0.60% -> pessimistic 20bps **21.46%**. Worst-case total drag **2.7%/yr — NOT
fee-fragile**. Commissions ~$0 (Alpaca/IBKR/Schwab). **TAXES DOMINATE: weekly rebalancing realises SHORT-TERM gains;
~24% gross -> ~16% net at a 35% rate. A US person should use a tax-advantaged account.**

**AUTOMATION** — very well suited: trades once/week, 3 liquid ETFs, no intraday data. Weekly cron Fri 15:45 ET ->
fetch -> scanner.py target weights -> reconcile broker positions -> order the delta (skip drift <1%) -> log + alert.
$5/mo VM or free GitHub Actions; idempotent; user only deposits cash since targets are percentages.
**Risks are OPERATIONAL not analytical** (stale feed, API outage, silent bug) -> paper mode first,
dead-man's-switch alert, monthly human check. NOT YET BUILT.

**NON-US INVESTOR ROUTE (case studied: a Pakistan-based investor with a UAE bank account)**
Broker: **IBKR**. Non-resident aliens generally owe **no US capital gains tax** (this strategy is ~all cap gains =
favourable); dividends withheld 30% (UAE has no comprehensive US treaty; Irish UCITS get 15% via US-Ireland treaty).
**US ESTATE TAX is the real trap: only $60k exemption on US-situs assets (US-listed ETFs count), up to 40% above —
and it follows ASSET SITUS, not residency or bank location.** Quantified UCITS route cost (`ucits.py`):
UCITS 3x ETP 21.6% (vs US 24.0%) = **-2.4%/yr**; UCITS 2x 17.6% (vs US 19.4%) = **-1.7%/yr**; pessimistic -4.3%/yr.
=> **under ~$60k use US-listed; above ~$60k switch to Irish UCITS** (LQQ-style 2x, DTLA, SGLN). Prefer **2x** also
because most European 3x are **ETPs/ETNs with issuer credit risk** (a risk absent from all backtests).
NOTE: a UAE bank account is NOT UAE tax residency — if still a Pakistani tax resident, Pakistan taxes worldwide
income and CRS reports the account. Requires a local cross-border accountant; not settled.

### ADDENDUM — Tier 1 tests + CROSS-MARKET VALIDATION — `tier1.py`, `crossmarket.py`
**Tier 1 (3 untested ideas).** Nothing beat 0.79 Calmar; three configs TIE with different risk profiles:
1. **FINAL 3x** — CAGR 24.2%, DD -31%, Sharpe 0.94. Best in deflationary crashes (dot-com +4.4%, GFC -0.3%).
2. **2x QLD @ vol-target 0.30** — CAGR 19.5%, DD **-25%**, Sharpe **0.97**, 2011 -16.2%. (2x sim validated vs real
   QLD: corr 0.996, 4.3%/yr trackErr, conservative.) Same Calmar, better Sharpe, 6pp less drawdown, ~5pp less CAGR.
3. **3x + DURATION-MATCHED off-leg** (short 2y bills when 10y yield > its 200d MA, else long 10y) — same CAGR/Calmar
   but **2022: -21.9% -> -14.1%**. Trade-off: gives back deflation protection (dot-com +4.4%->-2.3%, GFC -0.3%->-5.2%).
   A regime swap, not a pure win.
**FAILED: signal ensemble** (mean of price>EMA 50/100/150/200 as continuous exposure) — Calmar 0.69-0.72. It DID
soften 2011 (-18.2%) but dilution cost more than the whipsaw it saved. My hypothesis was wrong.

**CROSS-MARKET VALIDATION (the most informative test done).** Identical rules, own 3x-sim sleeve, engine only,
8 markets. Strategy Calmar beat **1x buy&hold in 7/8** and **3x buy&hold in 8/8**. Median strategy Calmar **0.23 vs
0.12** (1x BH) — the rules roughly DOUBLE risk-adjusted return across unrelated markets = NOT a curve fit.
BUT: Nasdaq **0.73** is a 3x OUTLIER. Canada 0.33, Russell 0.25, Nikkei 0.24, DAX 0.22, S&P **0.16**, HSI 0.11,
gold miners **-0.01** (fails). The strategy needs a strongly/persistently trending market; Nasdaq trended hardest.
**Median Calmar across markets, params perturbed:** as-specified 0.23 | **1x leverage 0.29** | **2x 0.26** | 3x 0.23 |
NO chop rule 0.16 | NO seasonal 0.20 | vol .15 0.25 | chop_thr .30 0.24.
=> **Both Nasdaq-derived rules GENERALIZE** (chop rule worth +0.07, seasonal +0.03 on out-of-market evidence — much
stronger than the in-sample case). **BUT the LEVERAGE CHOICE DOES NOT: less leverage is better risk-adjusted across
markets (1x 0.29 > 2x 0.26 > 3x 0.23).** Nasdaq is the ONLY market where 3x was right — the most exciting part of the
strategy is the part least supported by evidence.
**REVISED EXPECTATIONS:** design is sound, but 24%/yr assumes Nasdaq keeps out-trending everything. If the future
resembles the median market, the same rules give ~**9-14%/yr with -40% to -55% DD**. Prefer **2x** (ties Calmar, wins
Sharpe, and is the leverage level cross-market evidence supports). ~230 variants tested total; remaining gains are in
IMPLEMENTATION (taxes, futures vs LETFs, sizing vs net worth, paper trading), not more searching.
Report: `Addendum_Report.html` — supersedes the expectations section of `Final_Report.html`.

### Phase D (FINAL) — Geopolitics, new indicators, full validation — `engine.py`, `geo_panel.py`, `geo_test.py`, `indicator_test.py`, `final_validate.py`
Built `engine.py` (side-effect-free strategy engine) + `geo_panel.py` (long-history geopolitical regime flags from
^XOI oil 1983+, ^SPGSCI commodities 1984+, DX-Y.NYB dollar 1971+, ^HSI, ^VIX; flags: oil_shock, commod_bull,
dollar_weak, gold_lead, china_lead, stagflation, geo_score 0-4). Flags validate historically — the 2000s show
commod_bull 63%, gold_lead 59%, dollar_weak 51% = exactly the stagflation decade.
**GEOPOLITICAL ADAPTATION: 0 of 18 variants beat the static baseline** (best 0.71 vs baseline 0.72 Calmar).
Adaptive gold by geo_score, gold-instead-of-bonds in stagflation (Calmar 0.46!), geo de-risking, oil-shock veto,
China/EM sleeve — **all failed**. Reason: the flags are trailing momentum, not leading.
**NEW INDICATORS:** only **Aug-Sep half-size** survived (Calmar 0.72→0.79, DD −33→−31%, CAGR unchanged).
Failed: sell-in-May full exit (0.49), Nov-Apr boost, turn-of-month, election cycle, breadth (RSP/SPY & RUT/SPX,
0.27–0.68), VIX term structure/VXN (≈0).
**FINAL VALIDATION:** walk-forward **both** directions → the specified config scored **Calmar 0.80 out-of-sample in
each** (higher than the grid-optimized pick; selection cost only 0.07–0.12). Aug-Sep improved drawdown in **5/5
decades** but block-bootstrap says its return effect is **statistically zero** (−0.4%/yr, 95% CI −1.7% to +0.9%) →
it is a **risk reducer, not alpha**. Parameter perturbation: Calmar 0.68–0.80 (no knife-edges). Cost stress 3→20 bps:
Calmar 0.79→0.68. **Note: with bonds+gold present the yield-curve rule is now redundant** (0.79 either way) —
harmless insurance, not a driver.
**FINAL STRATEGY F** = 85% engine (NDX>EMA100 & not confirmed-recession · vol-target 0.20 · ER≥0.25 full else HALF ·
**Aug-Sep ×0.5** · off-leg **long treasuries**) + **15% gold**, weekly → **CAGR 24.2%, maxDD −30.7%, Calmar 0.79,
Sharpe 0.94, $1→$4,215 since 1988**; decades 21.7/38.3/10.9/21.5/30.3%. $1000→~$8,595 typical/10yr.
**CRISIS STRESS:** dot-com **+4.4%** (NDX −77%), GFC **−0.3%** (NDX −41%, SPX −48%), COVID −0.0%, 1998 LTCM +11.2%.
**WEAKNESSES (honest):** 2011 EU crisis **−21.6%** (V-shaped whipsaw) and 2022 **−21.9%** (stocks AND bonds fell —
no clean hedge). `scanner.py` UPDATED to Strategy F (+ A aggressive). Report: `Final_Report.html`.

### Phase C — Solving the flat decades + benchmarks — `multi_panel.py`, `rotation.py`, `rotation2.py`, `phaseC_final.py`
**Two different diseases.** 1992–95: NDX rose **+14.8%/yr** yet the strategy took −39% DD → self-inflicted whipsaw
(already fixed by Phase B's half-leverage). 2000–2010: NDX itself **−4.6%/yr** → nothing to harvest; structural.
**Research first.** Built a long-history multi-asset panel (gold miners ^XAU 1983+, synthetic 10y UST from ^TNX
validated **corr 0.95 / identical 6.8% vol vs IEF**, ^RUT, ^HSI, ^GSPTSE, intl). What trended in 2000–2010:
**gold miners +11.6%/yr, bonds +6.4%, Canada +4.4%, small caps +4.1%** vs NDX −4.6%. Opportunity existed.
**FAILED — multi-asset momentum rotation.** Core-sat (rotate free capital to best-trending alt) gave 2000-10 only
**3.7%** (vs 7.9% baseline) with DD −58%; pure ROTATE top-N gave CAGR 9–12.6%. Momentum-rotating the *defensive*
leg was worst (2000-10 = 2.4%). **Why:** the alt pool was equity-like (correlated in crises) and gold miners carry
their own −83% DD; **timing the diversifiers destroys their value.**
**WORKED — static, boring diversification (two changes, no timing):**
1. **Bonds as the off-leg** (instead of cash): strictly better — CAGR 24.5→26.1%, same −39% DD, 2000s **6.2→9.2%**.
2. **15% permanent gold sleeve**: 2000s **9.2→11.0%**, DD **−39→−33%**, Calmar **0.63→0.72**. Sweep 10–25% all
   give Calmar 0.71–0.72 (plateau, not a lucky point).
**FINAL (Phase C):** 85% engine (B-v2 rules, **bond** off-leg) + 15% gold, weekly →
**CAGR 24.1%, maxDD −33%, Calmar 0.72, Sharpe 0.90**; decades: 21.8 / 38.4 / **11.0** / 21.2 / 29.8% — all
positive & double-digit. $1000 → ~$8,302 typical/10yr.
**HONESTY:** the gold sleeve is **insurance, not alpha** — it helped only the 2000s (+4.8pp) and cost ~1–4%/yr in
every other decade; its Calmar gain is mostly drawdown reduction. Not walk-forward validated.
**Benchmarks (item #4, from 1988):** FINAL Calmar **0.72** (best of all) / Sharpe 0.90; A champion 30.7% CAGR
but Calmar 0.46; BH Nasdaq 14.4%/−83%/0.17; S&P 9.2%/0.16; 60/40 8.1%/0.24; Berkshire (1996+) 10.7%/0.20;
BH gold miners **2.8%/−83%/0.03** (terrible standalone); Bitcoin (2014+) 51.8% CAGR, Sharpe 0.97 but −83% DD, 1 regime.
Report: `PhaseC_Report.html`.

### Phase B — Diagnose & tune (why the hybrid trailed) — `attribution.py`, `frontier.py`, `frontier2.py`, `validate.py`
**Diagnosis.** A and B differ in only 3 mutually-exclusive states, so the 10.1%/yr gap decomposes exactly:
CHOP de-lever **−8.0%/yr** (43% of days, 80% of the gap), off-leg −2.6%/yr, curve −1.7%/yr.
**The chop rule's premise is false**: forward-20d returns by efficiency-ratio bucket show TQQQ beats QQQ in
*every* bucket (choppiest +2.2pp, and it had *higher* forward returns than the "clean" bucket). Episode analysis:
only 46% good calls, bad calls (−7.8%) worse than good (+6.0%). It wasn't timing — just de-risking.
**Acid test.** Scored every variant against a *pure vol-target dial at matched drawdown*. Old hybrid B is
**−2.2%/yr WORSE than simply holding less leverage** (21.6% vs 23.8% at −43% DD).
**What genuinely beat the dial:** (1) **partial de-lever — half TQQQ in chop, not zero** (+3.6%/yr edge);
(2) **"confirmed" curve** (inverted in last 12m AND trend broken) instead of raw inversion (+1.0%/yr).
**Rejected:** full de-lever to QQQ (−1.3 to −5.5%), raw inversion (one 2023 episode cost 82%), **boosting
leverage in strong trends (−0.0 to −0.2%, literally no effect)**, credit-spread overlay (Calmar unchanged 0.65).
**STRATEGY B-v2** = vol-target 0.20 · chop<0.25 → **50%** TQQQ · confirmed-curve risk-off:
CAGR **25.9%**, maxDD **−39%**, Calmar **0.67**, 2000-09 +6.2%. **Dominates old B on both axes** (21.6%/−43%/0.50).
$1000 → ~$9,236 typical/10yr. **Validation:** walk-forward both directions — beat A champion in both
(0.53 vs 0.51; 0.63 vs 0.49), beat old B in one (0.53 vs 0.39), lost slightly in the other (0.63 vs 0.67).
Parameter surface is a broad plateau (thr .20–.30 × pos .35–.65 all 0.56–0.67; the no-filter column is
uniformly 0.51) = robust, not a lucky cell. **Selection cost ≈0.15 Calmar → expect ~0.53–0.63 live.**
Report: `PhaseB_Report.html`.

### Phase A (post-research) — Rulebook + live weekly scanner — `scanner.py`, `STRATEGY_RULEBOOK.md`
Plain-English spec of the two production strategies + `scanner.py` which fetches live yfinance data and prints
exactly what to hold this week. Verified the scanner's rules reproduce the backtest allocations (weights sum to 1;
A in TQQQ 72% of days; B = 22% TQQQ / 43% QQQ / 34% bonds). Dashboard: `Weekly_Scanner.html`.
**Strategy A "GROWTH"**: NDX>EMA100 → TQQQ at min(1, 0.25/vol20) size, else cash. **Strategy B "DURABLE"**:
NDX>EMA100 AND curve≥0 → (efficiency-ratio≥0.30 ? TQQQ at vol-target : 100% QQQ), else 100% TLT.

### Phase 6 — Macro-aware HYBRID (fixing the 2000–2012 dead decade) — `macro_data.py`, `hybrid.py`
Ablation on 1985–2026 (focus: the choppy, twice-crashed 2000–09). Macro from yfinance (yields 1960+, dollar 1971+,
VIX 1990+, credit 2007+). **WORKED:** (1) **bonds off-leg** (TLT when risk-off) +2–3%/yr in bad decades; (2) **de-lever
in chop** (hold 1× QQQ not 3× TQQQ when Kaufman efficiency-ratio low) cut maxDD −67%→−47%; (3) **yield-curve overlay**
(go defensive when 10y−3mo inverts) — the standout, because the curve **leads** recessions ~12–18mo. Winner
(bonds + de-lever + curve): full CAGR 21.6% / maxDD −43% / Calmar 0.50; **2000–09 +8.4%/yr −39%DD** (base +4.3%/−67%;
BH QQQ −6.4%; BH TQQQ −47.5%). $1000→~$6,100 typical over 10yr. **FAILED (honest negatives):** VIX overlay (lagging —
sells the bottom), dollar overlay (noisy), chop mean-reversion (worse than de-levering), **faster/smaller EMA in chop**
(backfires — more whipsaw, −78%DD), **selective SQQQ** (still loses even in strong downtrends). **Lesson: only LEADING
macro signals (yield curve) help; coincident ones (VIX, dollar) hurt.** Report: `Hybrid_Macro.html`.

### Phase 5 — HOLISTIC leveraged optimizer on 40 years (1985–2026) — `letf_opt.py`
Grid: signal (SMA/EMA 100–250, crossovers) × timeframe (D/W) × rebalance (D/W/M) ×
allocation (on/off ∈ TQQQ/QQQ/SQQQ/cash) × vol-target (0 / 25%) × scale-in (1 / 5-day).
- **Benchmarks (full 41yr incl. dot-com):** BH QQQ CAGR 14.9% / maxDD −83% / Calmar 0.18;
  **BH TQQQ CAGR 14.8% / maxDD −100% (wipeout) / Calmar 0.15 → leveraged buy&hold does NOT beat QQQ.**
- **Best by Calmar (leveraged):** `EMA100 · daily signal · weekly rebalance · TQQQ/cash · vol-target 25%`
  → **CAGR 32%, maxDD −67%, Sharpe 0.83, Calmar 0.47, +74,484%.**
- **Best by Sharpe (unleveraged):** `EMA100 · weekly rebalance · QQQ/cash` → **CAGR 15%, maxDD −43%,
  Sharpe 0.91** (~QQQ return, half the drawdown, no leverage).
- **Sub-period stress (the honest test):** the vol-targeted champion **survives 2000–09 dot-com/GFC
  positive (+4.3%/yr, −67% DD)**; the *naive* 200-SMA/cash TQQQ timer (great on 2016–26) **fails
  it: −15%/yr, −95% DD.** → the 2016–26 result was regime-luck; **vol-targeting is what saves leverage.**
- **Optimization takeaways:** EMA≈SMA (EMA100 best); daily≈weekly signal; **rebalance weekly** (less
  whipsaw); **allocation = TQQQ/cash for max return, QQQ/cash for best Sharpe, SQQQ never**;
  **vol-targeting is the single most valuable refinement for leverage**; scale-in minor.

---

## 4. Key lessons (portable)

1. **Benchmark against drift.** Rank by raw P&L and you optimise into leveraged beta and call it alpha.
2. **Edge lives on short holds; beta on long ones.** Intraday oversold-bounce beat drift; daily didn't.
3. **Match the tactic to the regime/asset.** Mean-reversion for choppy/defensive; momentum for trend.
4. **Don't cut winners.** Stops/scalps trade return for risk; the profit is in the full move.
5. **Leverage rewards drawdown-avoidance, not prediction** — and only over favourable cycles unless
   you **vol-target**. Over a full cycle, unmanaged leverage (buy&hold TQQQ) is a trap.
6. **Cash ≫ SQQQ.** Inverse leveraged ETFs decay to zero; be flat, never short-hold them.
7. **Walk-forward / out-of-sample or it isn't real.** The dot-com/GFC decade is the acid test for any
   Nasdaq/leverage strategy.

---

## 5. Caveats & open items

- **Survivorship:** the 10-yr equity sweep uses *today's* 462 survivors — delisted names absent
  (flatters long-only). Point-in-time membership study is a planned separate analysis.
- **Intraday depth:** Alpaca free hourly is ~6–7h to pull for 10yr×500; intraday work stays on the
  42-day store or gold. Time-of-day not covered on daily.
- **Simulated LETFs** are conservative approximations (see §1) — good for relative/crash analysis,
  not a promise of absolute returns. 1985–2010 TQQQ/SQQQ is fully simulated.
- **1985–2026 was a historic tech bull.** Don't extrapolate 30%+ CAGRs; expect −45% to −67% leveraged
  drawdowns.
- All backtests: costs modelled, no lookahead (signal t → act t+1), but real fills/slippage/borrow/
  funding will lower live results, most on leverage and lower timeframes.

---

## 6. Reports

Each report is a self-contained HTML file in this repository; open it in a browser.

| Phase | Report file |
|---|---|
| 1 | `WaveTrend_Squeeze_Report.html` |
| 2 | `WaveTrend_Squeeze_Phase2.html` |
| 3 | `WaveTrend_Squeeze_Phase3.html` |
| 4 | `TQQQ_Timing.html` |
| 5 | `LETF_Optimizer.html` |
| 6 | `Hybrid_Macro.html` |
| B | `PhaseB_Report.html` |
| C | `PhaseC_Report.html` |
| **FINAL** | **`Final_Report.html`** |
| **ADDENDUM** | **`Addendum_Report.html`** |
| **PRACTICAL** | **`Practical_Report.html`** |
| **VS WORLD** | **`External_Report.html`** |
| **BEST ASSET** | **`BestAsset_Report.html`** |
| **NOV-2026 RE-CHECK** | **`Recheck_Nov2026.html`** |

## 7. How to reproduce / resume

**Fastest way back in:** `python scanner.py` (today's allocation; `--check` for the parity test once the data below
is built), then section 0. The commands work in Windows PowerShell and in bash.

```powershell
# --- refresh data (all free; a few minutes; also refreshes the published update_2026_11/panel_validation.json) ---
python letf_data.py; python macro_data.py; python multi_panel.py; python geo_panel.py
python update_2026_11/panel.py              # corrected shared panel (physical gold, IEF/TLT/SHY/BIL, real ETFs)

# --- the strategy, as live ---
python scanner.py --check                   # proves scanner == backtest engine (common.strategy_weights)
python scanner.py --log                     # weekly decision, appended to scanner_log.csv

# --- Nov-2026 re-check analyses (each writes update_2026_11/<name>.json) ---
python update_2026_11/main_tranche_check.py # weekday books + 5-weekday tranche (planning basis)
python update_2026_11/main_drift_check.py   # weekly drift vs constant mix
# extra market data some analyses need first (each writes a parquet next to it, not published; ucits_fetch.py
# also rewrites the published ucits_info.json):
python update_2026_11/oos_live_fetch.py; python update_2026_11/ucits_fetch.py
python update_2026_11/r2_chop_vol_matched_fetch.py; python update_2026_11/r2_friday_effect_fetch.py
# round 1: oos_live, revalidate, rate_regime, managed_futures, overlays, analogs, leverage_execution, ucits
# round 2: r2_chop_vol_matched, r2_candidates_tranche, r2_friday_effect, r2_scenario_stress, r2_model_hygiene
#   python update_2026_11/<script>.py   (verify_* / r2_verify_* = reviewers' checks; critic_* = critics)
# Notes: the data builders always fetch up to today, so re-runs use newer data than the published results
# (data to 2026-11-17) and overwrite the published JSON files (git shows what changed). Pinned to the published
# cut-off: r2_scenario_stress.py and build_report.py refuse a newer panel; ucits.py uses data to 2026-11-16; the
# re-entry block of oos_live.py assumes the 2026-11-17 cut-off. overlays.py skips its July reproduction without the
# unpublished July snapshot data_letf/archive_2026-07-24/, and reuses its own cache overlays_ext.parquet (delete it
# to refetch). The critic_* scripts write to %TEMP%/critic_out (or CRITIC_OUT_DIR); round 2 reads copies saved
# as update_2026_11/critic_out_*.json.

# --- rebuild the Nov-2026 report (only on the 2026-11-17 panel; see the notes above) ---
python update_2026_11/main_report_series.py; python update_2026_11/main_report_series2.py
python update_2026_11/report_round2_content.py; python update_2026_11/report_content.py
python update_2026_11/build_report.py       # -> Recheck_Nov2026.html
```
```powershell
# --- paper trading: local simulated account, no broker or credentials (paper/README.md) ---
python -m pytest paper/tests -q -p no:cacheprovider --rootdir paper     # 117 tests, a few minutes
python paper/runner.py replay --start 2026-07-24 --end 2026-11-17 --out paper/samples/replay_2026_oos
python paper/runner.py init --cash 10000    # start a simulated account; then: status, install_task.ps1
```
Notes: new work should `import common` from `update_2026_11/` (side-effect-free; mechanics identical to engine.py).
`bt.CACHE` reads env `CACHE_DIR` (default `cache/`, created automatically); the phase-1 indicator scripts also need
`build_cache.py` run first on an hourly S&P 500 price store (not published; see README, Data). Use `PYTHONIOENCODING=utf-8` on Windows.
Full agent outputs, including both reviews of every claim: `update_2026_11/quant_results.json` and `r2_results.json`;
fact-checked market research: `update_2026_11/market_context.json`.

## 8. Open items
1. **Paper-trading automation: BUILT** as a local simulated account in `paper/` (no broker account, API key or
   password), adversarially reviewed and fixed (section 3, PAPER-TRADING BUILD). **Not started.**
   To start it: `python paper/runner.py init --cash 10000`, `python paper/runner.py status`,
   `powershell -ExecutionPolicy Bypass -File paper\install_task.ps1`. The PC need not be on at a fixed time (catch-up
   mode); switch it on at least once between Saturday and Wednesday each week. Paper-trade for several months and
   judge it against the shadow 5-weekday tranche, not the Friday backtest.
2. **Before funding (go/no-go):** a Pakistan-qualified cross-border adviser must confirm the SBP rule's reach, the tax
   rate on foreign ETF gains, tax residency, and the US estate-tax position.
3. **Optional, decide before 2026-11-25 or never:** the 50/50 IEF/BIL off-leg (insurance against a 2022-type
   stock-and-bond selloff; costs ~0.2-1pp/yr).
4. Untested (only with genuinely new data or a pre-registered design): a lower-turnover constant multiplier in place of
   the chop rule; managed futures on a real CTA index from ~1990; a TIPS off-leg; tranche-basis re-runs of the July
   cross-market leverage test; multi-drawdown decision metrics.
