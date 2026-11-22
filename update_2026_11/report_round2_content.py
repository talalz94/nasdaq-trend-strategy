"""Round-2 report content (verified numbers from r2_results.json) -> update_2026_11/r2_dynamic.json"""
import json

def pct(x, d=1): return f"{x:+.{d}f}%"

lede = ("Since the July freeze the Nasdaq-100 set a record on 6 October, then fell 13.5% in an inflation-and-rates "
        "correction. The rules did what they say: they exited on 16 October and have held bonds and gold since. Two rounds "
        "of re-testing, with every result checked by two independent reviewers, found the rules sound but three things "
        "wrong with what we were quoting: the scanner held the wrong bond, one rule never did anything, and a lucky choice "
        "of rebalance day inflated every number. Corrected, this is a <b>drawdown-control strategy, not a way to beat the "
        "Nasdaq</b>: plan on about <b>13% a year for the 2× tier, with a worst drawdown near −28%</b> (one year in twenty, −37%).")

spec_html = """<span class="k">EVERY FRIDAY</span> at 15:40–15:45 ET  (a holiday Friday → that week's last trading day)
  from ^NDX daily closes
    EMA-100   ewm(span=100, adjust=False)
    vol20     stdev of the last 20 daily returns × √252
    ER20      |close − close 20 days ago| ÷ sum of the last 20 absolute daily changes
  <b>risk-on</b>    NDX close above its EMA-100
  <b>scale</b>      min(1, VT ÷ vol20), halved in August and September
  <b>engine</b>     risk-off → 0 · ER20 ≥ 0.25 → scale · ER20 &lt; 0.25 → scale × 0.5
  <b>targets</b>    Nasdaq sleeve = 0.85 × engine · IEF = 0.85 − sleeve · physical gold = 0.15 always

<span class="k">TIERS</span>     F2  QLD   VT 0.30   reference tier
          F1  QQQ   VT 0.20   lower risk; best in every stress scenario
          F3  TQQQ  VT 0.20   higher risk; US-listed only

<span class="k">EXECUTE</span>   market-on-close orders for the changes · skip a sleeve within 2 points of target
          · always trade a risk-on/off switch in full · fallback: next session's close
<span class="k">UCITS</span>     signal at the Friday US close; trade Monday 14:30–16:30 London:
          LQQ.PA (F2) or CNDX.L (F1), CBU0.L, IGLN.L   (no UCITS route for F3)
<span class="k">REVIEW</span>    if drawdown is worse than −35% (F2), −45% (F3), −25% (F1),
          or after 3 years behind QQQ with no bear market to explain it
<span class="k">NEVER</span>     re-tune a parameter · move the signal to a Monday close · hold TLT"""

rows = [
    ("F1 · 1×", "US: QQQ + IEF + GLD", "9.1%", "10.6%", "−20.9%", "−23.7%", "0.44–0.51", False),
    ("F1 · 1×", "UCITS: CNDX.L + CBU0.L + IGLN.L", "8.3%", "9.7%", "−21.5%", "−24.7%", "0.39–0.45", False),
    ("F2 · 2×", "US: QLD + IEF + GLD", "13.1%", "17.0%", "−27.8%", "−36.7%", "0.47–0.61", True),
    ("F2 · 2×", "UCITS: LQQ.PA + CBU0.L + IGLN.L", "12.4%", "15.9%", "−28.6%", "−38.2%", "0.43–0.55", False),
    ("F3 · 3×", "US: TQQQ + IEF + GLD", "16.7%", "21.6%", "−32.6%", "−46.0%", "0.53–0.68", False),
    ("F3 · 3×", "UCITS: not available", "–", "–", "–", "–", "–", False),
    ("QQQ buy & hold", "for comparison", "8.6%", "16.1%", "−83.0%", "−87.6%", "0.10–0.30", False),
]
HL = ' class="hl"'
body = "".join(
    f"<tr{HL if hl else ''}><td>{t}</td><td style='text-align:left;font-family:var(--sans);font-size:12.5px'>{r}</td>"
    f"<td>{b}</td><td>{u}</td><td class='neg'>{dd}</td><td class='neg'>{dd20}</td><td>{cal}</td></tr>"
    for t, r, b, u, dd, dd20, cal, hl in rows)
table_html = ("<table><thead><tr><th>Tier</th><th style='text-align:left'>Route</th><th>CAGR base case</th>"
              "<th>CAGR upside</th><th>Worst drawdown</th><th>1-in-20 drawdown</th><th>Calmar</th></tr></thead>"
              f"<tbody>{body}</tbody></table>")
table_note = ("Base case = the result since 2000; upside = the better of the 1988 and 2007 starts. Average of all five "
              "rebalance weekdays, 5 bps per unit traded (UCITS: one session later and 15/8 bps), in-sample to 24 Jul 2026. "
              "Worst drawdown = realised 1988–2026; 1-in-20 = 5th percentile of a 5,000-path block bootstrap. Not included: "
              "the Friday premium (upside), a parameter-selection haircut, real UCITS frictions beyond the 1-point proxy "
              "(measured 1–3 points), taxes, and any future Nasdaq growth below its history.")

plan_sub = ("Nothing was re-tuned. The July rules stand with three corrections: IEF instead of TLT, physical gold, and the "
            "do-nothing curve rule deleted. Two pre-registered tests found that the chop rule and the volatility target act as "
            "exposure dials rather than sources of edge; both stay unchanged, because replacing them would add a new tuned "
            "parameter. One optional change is yours to make: holding the bond leg as half IEF, half T-bills (BIL) halves "
            "today's rate exposure (−2.9% instead of −5.7% on a 1-point rise in yields) for about 0.2–1 point a year. "
            "If you want it, decide before the 25 Nov inflation data, not after.")

scen_rows = [
    ("Flat, volatile year (30% vol)", "−0.2", "−13.2", "+1.2", "−7.4", "−7.8", "re-entry ~mid-Jan"),
    ("10-year yield +1 point within a month, stocks flat", "−0.2", "−13.2", "−3.8", "−12.4", "−12.3", "book −5.7% at once"),
    ("Gold −15% over two months", "−0.2", "−13.2", "−1.2", "−9.6", "−10.0", "book −2.3% at once"),
    ("V-rebound to the record in 6 weeks", "+22.5", "+30.7", "+4.2", "−1.2", "−2.2", "back in ~11 Dec near 29,650"),
    ("2022-like bear: NDX −25%, yields +0.75 pt, gold −10%", "−25.2", "−51.2", "−3.7", "−9.9", "−11.1", "losses from bonds and gold"),
    ("Deflationary bust: NDX −35%, yields −1.5 pts", "−35.1", "−63.4", "+11.2", "+3.7", "+2.8", "bonds rally"),
    ("Violent range: ±8% around the EMA", "−0.2", "−14.0", "−9.0", "−32.7", "−31.8", "the blind spot"),
    ("Replay of 2000–01 from today", "−53.7", "−86.2", "+0.6", "−9.6", "−13.9", "actual path"),
    ("Replay of 2022 from today", "−20.2", "−45.3", "−8.4", "−12.7", "−14.8", "actual path"),
    ("Replay of the Q4-2018 rebound", "+22.5", "+38.3", "+8.6", "+9.2", "+9.4", "actual path"),
    ("Replay of the 2025 tariff rebound", "+30.4", "+54.9", "+18.3", "+31.9", "+40.3", "actual path"),
    ("Replay of the 2020 COVID rebound", "+45.8", "+77.7", "+16.3", "+44.9", "+46.8", "actual path"),
]
def c(v):
    return "neg" if v.startswith("−") else ("pos" if v.startswith("+") else "")
sbody = "".join(
    f"<tr><td style='white-space:normal;min-width:220px'>{s}</td><td class='{c(q)}'>{q}%</td><td class='{c(l)}'>{l}%</td>"
    f"<td class='{c(f1)}'>{f1}%</td><td class='{c(f2)}'><b>{f2}%</b></td><td class='{c(f3)}'>{f3}%</td>"
    f"<td style='text-align:left;font-family:var(--sans);font-size:12.5px;color:var(--muted);white-space:normal'>{n}</td></tr>"
    for s, q, l, f1, f2, f3, n in scen_rows)
scen_table = ("<table><thead><tr><th>12 months from today</th><th>QQQ</th><th>QLD</th><th>F1</th><th>F2</th><th>F3</th>"
              "<th style='text-align:left'>Note</th></tr></thead>" f"<tbody>{sbody}</tbody></table>")
scen_sub = ("Today all three tiers hold 85% IEF and 15% gold, so the account moves with interest rates and gold, not with the "
            "Nasdaq. Each row runs the unchanged rules forward from today's exact state: the synthetic cases are the median of "
            "40 random paths; the replays apply one real historical path to today's levels. Descriptive only; nothing was "
            "chosen from these results.")
scen_note = ("The strategy protects in bear markets in relative terms, not absolute ones: on the 2000 path F2 loses about 10% "
             "while QQQ halves, but a 2022-style grind still costs F2 10–13% because bonds and gold fall too. In a fast V-shaped "
             "rebound it re-enters late and earns roughly half the move. Its blind spot is a violent, trendless market: "
             "repeated swings across the EMA cost F2 a quarter to a third in these simulations, more than buying QLD and holding "
             "(about −12% to −19% with less regular swings). Re-entry needs a Friday close above ~28,890; the first re-entry "
             "would be halved by the chop rule (F2 about 30% QLD) because the market is still choppy. Single replay paths "
             "carry about ±4.5 points of luck from the weekday and the timing of the Aug–Sep halving.")

ideas_extra = [
    dict(idea="Chop rule (half size in choppy trends)",
         evidence="Pre-registered test against a version without it, matched to the same average exposure: no edge since 2007 (Calmar −0.01) or across 7 other markets (median +0.01). Its old edge was a 1990s effect. It drives about two-thirds of turnover. Kept unchanged, credited only as an exposure dial; a lower-turnover replacement would need its own test.",
         n="157", verdict="Replaceable"),
    dict(idea="Volatility target",
         evidence="Its value comes almost entirely from the 2000 crash; since 2007 it changes almost nothing for F2 and is a knife-edge for F3. Kept: it costs no extra trading and the version without it had much deeper drawdowns.",
         n="157", verdict="Kept"),
    dict(idea="Friday rebalance premium",
         evidence="Real on the Nasdaq in both halves of history, but it did not repeat in other markets (3 of 7 where 5 were required), and its early-years cause (stale Nasdaq closing prices) has expired. Friday stays the decision day; planning uses the five-day average.",
         n="13,409", verdict="Plan on average"),
    dict(idea="Model-input corrections",
         evidence="Adding financing costs to the pre-2010 leveraged simulations, yield-dependent bond duration and month-end gold costs at most 0.1 / 0.35 / 0.7 points a year (F1 / F2 / F3) and changes no conclusion.",
         n="360", verdict="Corrected"),
    dict(idea="Edge over QQQ buy-and-hold",
         evidence="On the honest basis no tier has a significant return edge. The drawdown advantage is large (Ulcer index 4.5–13.7 vs 36.8) but the Calmar edge is borderline and rests on 2000–02. Since 2010 F2 made 15.8% a year against QQQ's 18.8%.",
         n="34", verdict="Risk choice"),
]

changes = [
    "<b>Bond: TLT → IEF.</b> The scanner now holds what the backtest held. TLT would also more than double today's rate exposure (−12.9% vs −5.7% for a 1-point rise in yields).",
    "<b>Gold: miners → physical gold,</b> and the do-nothing yield-curve rule is deleted. Neither changes the long-run results much.",
    "<b>Expectations re-based.</b> July quoted F3 at 24% a year with a −31% worst drawdown, from Friday-only backtests. Planning numbers now: F2 13–17% a year (base case 13%), worst −28%, one year in twenty −37%; F3 17–22% (base 17%), worst −33%, one in twenty −46%; F1 9–11%, worst −21%.",
    "<b>The edge is drawdown avoidance, not extra return.</b> No tier beats QQQ's return significantly. Since 2010 F2 trailed QQQ by 3 points a year while taking about 80% of its worst drawdown. The big wins come from sidestepping crashes like 2000–02.",
    "<b>Leverage is a risk dial.</b> 1×, 2× and 3× tie on risk-adjusted return. 1× held up best in every stress scenario. F2 stays the reference tier; choose by the drawdown you can live with.",
    "<b>Two rules reclassified.</b> The chop rule and the volatility target add nothing beyond holding less exposure since 2007. Both stay unchanged.",
    "<b>Execution defined.</b> Signal at Friday 15:40–15:45 ET, market-on-close orders, a 2-point no-trade band, risk-on/off switches always traded. UCITS trades in Monday's European session.",
    "<b>Your route needs confirming first.</b> The SBP rule and the Pakistani tax rate may decide whether weekly rebalancing is workable at all. That is now a go/no-go question, not a detail.",
]

caveats_extra = [
    "The planning ranges leave out a parameter-selection haircut (the July settings rank at the 82nd–84th percentile of all settings tried), real UCITS frictions beyond the 1-point proxy, taxes, and any future Nasdaq growth below its 14.8%-a-year history. Use the floor of each range as the base case.",
    "Statistical tests of the edge over QQQ flip with bootstrap block length and seed. “Borderline” means borderline.",
]

you_extra = [
    dict(h="This may be go/no-go, not a detail", kind="bad", p=[
        "If the SBP requirement to bring sale proceeds back to Pakistan covers sales inside the brokerage account, a strategy that rebalances every week may not be workable under that regime. If the US$25,000 annual cap applies, it would take two to three years of contributions to pass ~$60k, so the US route would serve first, and a later switch to UCITS would realise taxable gains.",
        "Paper trading needs no funding, so the IBKR automation can be built and tested while a Pakistan-qualified adviser confirms these points."]),
]

out = dict(lede=lede,
           plan=dict(tile="~13% a year", tileSub="F2 base case · worst drawdown ≈ −28%", sub=plan_sub, spec_html=spec_html,
                     table_html=table_html + f"<p style='font-size:12.5px;color:var(--muted);margin:10px 12px 12px;max-width:none'>{table_note}</p>"),
           scen=dict(sub=scen_sub, table_html=scen_table, note=scen_note),
           ideas_extra=ideas_extra, changes=changes, caveats_extra=caveats_extra, you_extra=you_extra)
json.dump(out, open("update_2026_11/r2_dynamic.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote update_2026_11/r2_dynamic.json")
