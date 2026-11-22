# Strategy Rulebook (Nov-2026 re-validated)

Plain-English spec of exactly what the strategy holds and why. It is a **weekly** strategy: decide once a week at
the Friday close, and only trade what changed. Re-validated on Nasdaq-100 data from 1985 to 17 Nov 2026 (leveraged
ETFs simulated before they existed; real ETF prices after). Research only, not investment advice.

Run `python scanner.py` for this week's answer. `python scanner.py --check` proves the scanner makes exactly the
same decisions as the backtest (all 2,072 weekly decisions since 1985, zero difference).

> This replaces the July "Strategy A / Strategy B / Strategy F" descriptions. Strategy B (raw yield-curve veto, 1×
> QQQ in chop) was superseded in August; Strategy A (TQQQ/cash, −67% drawdowns) is not recommended.

---

## The three inputs (computed on ^NDX daily closes)

| Input | Formula | What it means |
|---|---|---|
| **Trend** | close vs **EMA-100** (`ewm(span=100, adjust=False)`) | Above = risk-on. The core switch, and the one rule that helped in every test. |
| **Volatility** | 20-day realised vol = `stdev(daily returns) × √252` | Sizes the position: high volatility, smaller position. |
| **Efficiency ratio** | `|close − close 20 days ago| ÷ Σ|daily changes| over 20 days` | 1.0 = straight-line trend, 0 = pure noise. Below 0.25 counts as choppy. |

No macro inputs. The July yield-curve clause never changed a position (it simplifies to "above the EMA") and was
deleted. Every macro overlay tested in Nov 2026 (rate shock, dollar, credit spreads, VIX term structure, curve
inversion and un-inversion; 54 variants) failed.

---

## The rule (all tiers)

```
EVERY FRIDAY at the close  (holiday Friday -> the last trading day of that week)
  1. risk-on?   NDX close above its EMA-100
                (equivalently: above the previous day's EMA-100, so the bar is known in advance)
  2. scale    = min(1, VOL_TARGET / vol20)
                x 0.5 if the decision date is in August or September
  3. engine   = 0                 if risk-off
              = scale             if efficiency ratio >= 0.25
              = 0.5 x scale       if efficiency ratio <  0.25   (half size in a choppy trend, never zero)
  4. targets  = Nasdaq sleeve       0.85 x engine
                IEF (US Treasuries 7-10y)  0.85 - Nasdaq sleeve
                physical gold (GLD)         0.15 always
```

| Tier | Nasdaq sleeve | Vol target | Character |
|---|---|---|---|
| **F2 (reference)** | QLD (2×) | 0.30 | middle of the risk dial |
| F1 | QQQ (1×) | 0.20 | lowest risk; held up best in every stress scenario |
| F3 | TQQQ (3×) | 0.20 | highest risk; US-listed only (no UCITS 3× fund exists) |

The three tiers tie on risk-adjusted return. Choose by the drawdown you can live with, not by which "wins".

---

## What to expect (planning numbers, not the backtest headline)

Basis: the average of all five possible rebalance weekdays (Friday was luckily the best day in-sample and that
luck is not planned), 5 bps per unit traded, in-sample 1988 to 24 Jul 2026. **Base case** = the result since 2000;
upside = the better of the 1988 and 2007 starts.

| Tier | CAGR base case – upside | Worst drawdown | 1-in-20 drawdown | UCITS route CAGR |
|---|---|---|---|---|
| F1 | 9.1% – 10.6% | −21% | −24% | 8.3% – 9.7% |
| **F2** | **13.1% – 17.0%** | **−28%** | **−37% (to −40%)** | 12.4% – 15.9% |
| F3 | 16.7% – 21.6% | −33% | −46% (to −51%) | not available |
| QQQ buy & hold | 8.6% – 16.1% | −83% | −88% | |

**The edge is drawdown avoidance, not extra return.** No tier beats QQQ's return by a statistically significant
margin; the big wins came from sidestepping crashes like 2000–02. In a strong bull market like 2010–26 expect F2 to
trail QQQ (15.8% vs 18.8% a year) while taking about 80% of its worst drawdown.

Not in these numbers: taxes, real UCITS frictions beyond the modelled 1 point (measured 1–3 points a year), a
parameter-selection haircut, and any future Nasdaq growth below its history.

---

## Execution

| | US route (US-situs holdings below ~$60k) | UCITS route (above ~$60k; non-US investor) |
|---|---|---|
| Instruments | QQQ / QLD / TQQQ, IEF, GLD | CNDX.L (F1) or LQQ.PA (F2), CBU0.L, IGLN.L |
| When | Friday 15:40–15:45 ET signal, market-on-close orders | Signal at the Friday US close; trade Monday 14:30–16:30 London (limit orders) |
| Fallback | the next close after the missed signal (1–3 sessions late: ~nothing in return, +1–3 points of drawdown; a week late: −5 to −7 points a year) | next overlapping session |
| Trade filter | skip a sleeve within 2 percentage points of target; **always** trade a risk-on/off switch in full | same |

- Never move the signal to a Monday close (the worst rebalance day in 6 of 7 markets tested).
- LQQ.PA had a 200:1 split on 2026-07-09: check share maths.
- Between decisions positions drift freely; this costs nothing measurable (−0.05 points a year for F2).

---

## Review triggers (pre-committed; never re-tune in response)

- Drawdown worse than **−35% (F2)**, **−45% (F3)** or **−25% (F1)**.
- Three years behind QQQ with no bear market to explain it.
- A 12-month loss worse than −25% is a monitoring flag, not a trigger.

---

## Why each rule is there, honestly

- **EMA-100 trend gate**: leveraged ETFs decay in downtrends. The gate is what kept the strategy out of most of
  2000–02 and 2008, and what limited the Oct–Nov 2026 loss (without it F3 would have lost ~23% instead of 9%).
- **Bond off-leg (IEF)**: Treasuries usually rally in flight-to-safety selloffs. IEF, not TLT: the 20-year bond adds
  about 10 points of worst drawdown and doubles the damage when stocks and bonds fall together (2022).
- **15% physical gold**: diversification and de-levering. It does not beat the same 15% held in IEF.
- **Volatility target**: an exposure dial whose measurable value came from the 2000 crash. Kept unchanged.
- **Half size in choppy trends**: an exposure dial with no edge since 2007 beyond holding less; it drives about
  two-thirds of turnover. Kept unchanged because replacing it would mean tuning a new parameter.
- **Half size in Aug–Sep**: a drawdown reducer; its effect on returns is statistically zero.

## Known weak spots

- **Violent, trendless markets** that swing across the EMA: repeated false re-entries can cost F2 a quarter or more.
- **Fast V-shaped rebounds**: the strategy re-enters late and typically earns about half of the move.
- **Stocks and bonds falling together** (2022): the off-leg loses too. A pre-specified 50/50 IEF/T-bill off-leg
  halves that exposure for about 0.2–1 point a year; decide on it in advance, not after a move.
- **Low volatility at a market top**: the vol target allows full size right before a drop (Oct 2026).

## Caveats

- Pre-inception leveraged ETFs and pre-2002 Treasuries are simulated; real ETFs tracked the simulations within
  0.3 points in the live window. Gold before 2000 is a monthly series.
- Results hinge on a few episodes (1989–90, 2000–02, 2022); the 81-day live window is too short to judge.
- Taxes are not modelled; for a Pakistan-based investor the SBP foreign-investment rule and the tax rate on foreign
  ETF gains must be confirmed by a qualified adviser before funding (see RESEARCH_LOG.md, section 0).
