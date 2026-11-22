# Paper trading for the NDX trend strategy

A local, fully automatic paper-trading loop for the weekly strategy in `scanner.py` / `STRATEGY_RULEBOOK.md`.
**It needs no broker account, no API key and no password.** The "broker" is a simulated account kept in a file on
this PC; prices come from Yahoo Finance, the same free source the scanner uses. Nothing in this folder can place a
real order.

## What it does

Every 5 minutes (and right after the PC starts or wakes) Windows runs `python paper/runner.py tick`. Each run does
only what is due, and **the PC does not have to be on at any particular time**:

| When (New York time) | What happens |
|---|---|
| Decision day (normally Friday) **15:40–15:48** (12:40–12:48 on 1 pm half-days) | Computes the decision from the live Nasdaq-100 with the scanner's own code, plans orders (2-point no-trade band; a risk-on/off switch always trades in full) and queues them as **market-on-close** orders. |
| ~45 minutes after each close | Fills queued orders at the **official closing price**, charges 5 bps per unit traded, credits ETF dividends, applies stock splits, marks the account to market and logs everything. |
| If the PC was off or asleep in the window | **Catch-up:** as soon as the PC is on again (and the decision day's official close has been published, ~16:45 New York time), the decision is made from that close and queued for the next close the market-on-close order can still reach. A PC switched on any time between Saturday morning and Monday night (Pakistan time) trades at Monday's close. Weeks skipped entirely are logged and alerted. |
| Holiday Friday | The week's last trading day becomes the decision day. |
| Unscheduled closure (e.g. a national day of mourning) | Recognised from the data once the next session has opened. The week's decision moves to the previous session, and the "decision failed" alerts already sent for the closed day are withdrawn. Closures announced in advance can be listed in `schedule.extra_holidays`. |

**UCITS route:** the decision is made after the US close from the official close. All of a decision's orders then
fill together, at the close of the first later NYSE session on which every listing trades and whose close comes after
the decision was made. That is the research's lag-2 execution: a Friday decision normally fills at Monday's European
close; a decision made after Monday's European close (about 11:30 New York time) fills on Tuesday; on a UK-only bank
holiday everything waits a day, so the Paris buy is never made before the London sale that pays for it. A European
close on a US holiday is not used (the research convention).

**Splits** are detected from Yahoo's back-adjusted prices, so they are applied correctly whenever the account catches
up, even when Yahoo never records the split (LQQ.PA's 200:1 change in July 2026 was such a case). If a price jumps in
a way the market move cannot explain and the data shows no split, booking pauses with an alert telling you which of
two commands to run (`split` or `accept-move`, below).

It also keeps a **shadow 5-weekday tranche**: the same rules run on all five weekdays and averaged, with the research's
execution (US: at the decision close; UCITS: at the next European close on which all listings trade). That is the
planning basis from the Nov-2026 re-check. The live account (Friday only) should scatter around that line, so judge
results against it rather than the lucky in-sample Friday backtest. If you change the tier or route later, the
tranche keeps its history and follows the new settings from the day of the change.

## Set up (once)

```powershell
python paper/runner.py init --cash 10000
python paper/runner.py status
powershell -ExecutionPolicy Bypass -File paper\install_task.ps1
```

- `init` starts the simulated account (default $10,000; settings in `paper/config.toml`, copied from
  `config.example.toml`). The first run after it invests according to the latest weekly decision.
- `install_task.ps1` registers two Windows scheduled tasks that run **as you, only while you are logged on**, so
  Windows stores no password:
  - `NDX-Trend-Paper-Tick`: every 5 minutes, at log-on and one minute after the PC wakes from sleep or modern standby, it runs
    `pythonw paper\runner.py tick`; a run missed while the PC was off starts as soon as it is back. There is no
    console window. `-EveryMinutes` (1–8) changes the interval; it must not exceed the 8-minute window.
  - `NDX-Trend-Paper-Watchdog`: hourly, it runs `paper\watchdog.ps1`, which warns you if the runner has stopped.
    It does not use Python.
  - Only with `-WakeToRun`: `NDX-Trend-Paper-Wake` wakes a sleeping PC once a week, one minute into the decision
    window, for the next ~10 months (Task Scheduler allows 48 triggers per task); re-run the installer to extend it.
- Remove them with `powershell -ExecutionPolicy Bypass -File paper\uninstall_task.ps1` (your data is kept).

### When does the PC need to be on?

Never at a fixed time. The planned trade is at the Friday close (the window is **Saturday 01:40–01:48 in Pakistan**,
00:40–00:48 while US daylight saving time is in effect, mid-March to early November; on 1 pm half-days it is 22:40
the same evening, 21:40 on 3 July). If the PC is asleep or off then, the decision is caught up the next time it is on
and trades at the next close. What trading late costs in the research backtest (F2, 5 bps, the Friday decision traded
k trading days late):

| Trade at | 2000–2026 CAGR | 2016–2026 CAGR | Worst drawdown |
|---|---|---|---|
| Friday close (on time) | 14.8% | 20.9% | −24.1% |
| Monday close | 15.0% | 20.8% | −25.2% |
| Tuesday close | 15.6% | 20.8% | −26.7% |
| Wednesday close | 15.1% | 20.8% | −27.0% |
| Thursday close | 11.9% | 15.3% | −29.6% |
| A week late | 9.8% | 13.8% | −30.3% |

So 1–3 trading days late changes the return by less than the noise and deepens the worst drawdown by 1–3 points; a
week late is expensive. In practice: **switch the PC on at least once between Saturday morning and Wednesday each
week.** A decision made 1–3 days late is written to `alerts.log` only; 4 or more days late raises a desktop alert.
`status` shows the account next to the same rules traded on time (the "Friday book"), so you can see roughly what the
delays cost (the Friday book skips holiday-Friday weeks, which the account decides on the Thursday, so part of the
gap is those weeks). For the exact Friday-close trade, keep the PC awake at the window, or install with `-WakeToRun` and set **Allow
wake timers** to Enable (Control Panel > Power Options > Change plan settings > Change advanced power settings >
Sleep); the installer only reads your power settings, it never changes them.

### What it costs your PC

Measured on the author's laptop (Intel i7-10510U):

| Run | How often | CPU | Memory | Disk / network |
|---|---|---|---|---|
| Nothing due (almost every run) | every 5 min while on | ~0.7 s | ~22 MB for ~1 s | 0.2 KB heartbeat, no network |
| Book a close or make a decision | ~once per trading day | ~8.5 s | ~125 MB for 15–25 s | ~30 KB, 1–5 MB download |
| Watchdog (PowerShell) | hourly | ~2.5 s | ~80 MB for ~5 s | none |

That is about 4–5 minutes of CPU per day spread over ~300 short runs, well under 1% of one core, with no lasting
memory use and no noticeable effect on battery or other programs. An idle run never loads pandas or touches the
network: a standard-library check (`paper/quickcheck.py`) decides whether anything is due, and a test replays the
engine over weeks of simulated time to prove that the check never skips work. When something fails - a close that
Yahoo has not published yet, no internet, a CSV held open in Excel - the run retries after 15-60 minutes rather than
every 5, so even then the cost stays small.

## Daily use

| Command | Purpose |
|---|---|
| `python paper/runner.py status` | Account value, positions, pending orders, next decision time (in New York and your local time), the re-entry level, alerts. |
| `python paper/runner.py ack` | Clear active alerts after reading them. |
| `python paper/runner.py test-alert` | Sends a test alert and reports each channel (log, desktop, webhook, healthcheck). |
| `python paper/runner.py replay --start 2026-07-24` | Run the same code over history into `paper/samples/replay` (no effect on your account; it refuses to write into a live account folder). |
| `python paper/runner.py split QLD 2` | Only if a "booking paused" alert asks for it: record a split Yahoo's data does not show (2 = two-for-one). It refuses a symbol the account neither holds nor has orders for. |
| `python paper/runner.py accept-move QLD 2026-12-04` | Only if a "booking paused" alert asks for it: confirm that the flagged price move is real. It accepts only a move that was flagged (`--force` overrides both checks). |

## Files (`paper/data/`)

| File | Contents |
|---|---|
| `decisions.csv` | One row per weekly decision: NDX, EMA-100 (and the prior day's, which is the re-entry bar), vol20, ER20, risk-on, switch, chop and seasonal flags, targets, current weights, orders, skipped sleeves. A decision that had to be made again (after a closure) gets a second row. |
| `fills.csv` | Every simulated fill: price, cost, cash after. |
| `nav.csv` | Daily account value, positions, weights, dividends, drawdown. |
| `shadow_tranche.csv` | The 5-weekday tranche and its five books (planning benchmark), with the settings each part used. |
| `events.csv`, `runner.log` | Decisions, dividends, splits, closures, configuration changes, errors. |
| `alerts.log`, `ALERT_ACTIVE.txt` | Alerts; the second stays until you run `ack`. |
| `state.json`, `state.json.bak` | The account and runner state (written atomically; `.bak` is the previous version). |
| `heartbeat.json` | The time of the last run (read by the watchdog). |

**Open a copy of a CSV file, not the file itself, while the runner is active.** Excel locks a file it has open. The
runner still books and decides, keeps the rows it could not write in `state.json` and writes them once the file is
closed; you get one alert per day that names the file.

## Alerts and the dead-man's switch (no accounts needed)

- **Desktop notifications** (Windows toast) for risk-on/off switches, decisions made 4+ trading days late, skipped
  weeks, data problems,
  errors (including a mistake in `config.toml`; each different error is alerted at most every 6 hours and always
  logged), and the pre-committed **review triggers**:
  - drawdown worse than −35% (F2), −45% (F3) or −25% (F1), checked on every booked day;
  - three years behind QQQ (checked monthly; the rulebook adds "with no bear market to explain it", which is your
    judgement);
  - and a 12-month loss flag (a monitoring flag, not a trigger).

  These are review prompts. Never re-tune in response.
- **Watchdog task:** alerts if no run has completed in the last hour. It stays quiet for 20 minutes after the PC
  starts or wakes, so it does not raise a false alarm while the runner catches up.
- Optional and off by default: `alerts.webhook_url` (your own webhook; it receives JSON `{"text": ...}`, the Slack
  format) and `alerts.healthcheck_url` (an external dead-man ping after each weekly decision, which also catches a
  PC that is off). Both need you to set up a service yourself, so they are left empty. Delivery failures are written
  to `alerts.log` (without the URL).

## Settings (`paper/config.toml`)

`tier` F1 / F2 / F3 (default F2), `route` US / UCITS (UCITS = LQQ.PA or CNDX.L, CBU0.L and IGLN.L; F3 is US only),
`band`, `cash_buffer`, `starting_cash`, `cost_bps`, `commission_per_order`, `fractional_shares`,
`dividend_withholding` (set 0.30 to model US withholding on a non-US account), schedule offsets and extra holidays.
Every key is optional. Every value is checked when the runner starts: an unknown key, a wrong type or an
out-of-range value stops the run, writes the reason to `runner.log` and raises an alert.

## How faithful is it?

- **Parity:** the decision code is `scanner.py`'s own functions. A test checks that the targets equal the research
  backtest engine's on 180 historical decisions.
- **Replays through this exact code (5 bps per unit traded, the same basis as the research figures quoted):**
  - 24 Jul–17 Nov 2026: −5.97% vs the backtest's −6.10% (−6.03% at the research's 3 bps).
  - 2016–Jul 2026: CAGR 21.9% vs 22.2%, max drawdown −24.3% vs −24.1%, daily correlation 0.997.
  - The only months more than 1 point apart follow holiday-Friday weeks. The rules decide on that Thursday; the
    research backtest skipped those weeks. Against the research variant that also decides on those Thursdays, the
    daily correlation is 0.9997 and no month differs by more than 1 point.
  - UCITS route, 2016–Jul 2026: CAGR 19.5%, max drawdown −27.0% (2.4 points below the US route, inside the
    research's 1–3 point range). Every decision's legs filled on the same day.
- **What the simulation simplifies:**
  - Fills happen exactly at the official close: no partial fills, no auction imbalance.
  - The live signal at 15:40 uses Yahoo's latest NDX value, which can lag by a few minutes; a value that close to the
    EMA can flip by the close.
  - Dividends are credited on the ex-date rather than the pay date.
  - UCITS orders fill at the European close rather than in the 14:30–16:30 London window; European half-days are
    not modelled. Their quantities are sized from the prices at decision time, so a large move before the fill can
    leave a little cash idle until the next decision (a cut of more than 5% raises an alert; 5 times in 2016–2026).
  - No taxes.

## Tests

```powershell
python -m pytest paper/tests -q -p no:cacheprovider --rootdir paper
```

These cover the calendar (holidays, half-days, holiday Fridays), configuration checks, order planning (band, switch,
rounding, cash, commissions), account booking (fills, costs, dividends incl. late-posted ones, splits recorded or
not, cash in lieu, UCITS legs filling together, no change when a price is missing or suspicious), the engine (live
window, idempotent re-runs, catch-up after a missed window, joining mid-week or after the window, risk-off switch, UCITS
timing, unscheduled closures, skipped weeks, run lock, data errors, logs held open in Excel, crash safety, review
triggers), the runner (config errors, repeated errors, busy lock, replay safety), the shadow tranche (UCITS lag 2,
missing history, tier changes), the idle pre-check (it never skips work, also while it backs off after a
failure) and parity with the backtest. The suite takes a few minutes.

## Adding a real broker later

Everything broker-specific is confined to the simulated account (`paper/account.py`). A real broker adapter would
replace queueing and filling with the broker's API, and the decision, planning, logging and alerts would stay as
they are. Nothing here needs one today.
