"""
Narrative content for the Nov-2026 re-check report -> update_2026_11/r2_report_content.json
Static sections come from the fact-checked market brief and the verified round-1 results.
Round-2 sections (plan, scenarios, ideas rows for round 2, changes, lede) are filled from ROUND2 below.
"""
import json, os

facts = [
  dict(html="<b>The Fed is raising rates, not cutting.</b> It hiked to 3.75–4.00% on 16 Sep (its first hike since 2023), held on 28 Oct with two votes for another hike, and a December hike is close to fully priced.",
       src="Federal Reserve, 28 Oct", url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm"),
  dict(html="<b>Inflation has re-accelerated</b> after the oil shock from the US–Iran war: September CPI 3.7% y/y, core CPI up 0.3% a month for three months, October producer prices +6.5% y/y.",
       src="BLS, 10 Nov", url="https://www.bls.gov/news.release/cpi.htm"),
  dict(html="<b>Long yields are at a 24-year high.</b> The 10-year closed at 5.31% on 5 Oct, the highest close since 2002, and at 5.10% on 17 Nov. The real yield (~2.9%) is near its highest since 2008.",
       src="US Treasury, 17 Nov", url="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve"),
  dict(html="<b>The sell-off centred on AI mega-caps.</b> NDX fell 6.36% on 5 Nov, its worst day since March 2020 (TQQQ −18.9%). Drivers cited: AI-capex doubts, Alphabet's negative cash flow after capex, a new AI-chip export rule, and Nvidia's CEO resigning on 12 Nov.",
       src="Yahoo Finance, 5 Nov", url="https://finance.yahoo.com/markets/live/stock-market-today-thursday-november-5-dow-sp-500-nasdaq-plunge-chip-stocks-melt-down-193726491.html"),
  dict(html="<b>The economy is not in recession.</b> Q3 GDP grew 3.4% annualised, October payrolls rose 165k, unemployment is 4.1%, and the high-yield credit spread is a calm 3.34%.",
       src="BEA, 29 Oct", url="https://www.bea.gov/news/2026/gdp-advance-estimate-3rd-quarter-2026"),
  dict(html="<b>Earnings are strong while valuations fall.</b> S&amp;P 500 Q3 earnings grew 31.9%; tech's forward P/E (17.1) slipped below the market's (17.9) for the first time since 2017.",
       src="FactSet, 13 Nov", url="https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/EarningsInsight_111326.pdf"),
  dict(html="<b>Treasuries hedged part of the fall.</b> TLT rose 6.45% from 6 Oct to 6 Nov while NDX fell 13.6%, then drew no safe-haven demand in mid-November as yields climbed again.",
       src="TotalRealReturns, 17 Nov", url="https://totalrealreturns.com/n/TLT,IEF,GLD"),
  dict(html="<b>Gold has been the weak link.</b> Spot gold is about $3,767, roughly 32% below its 28 Jan record, after a 15% fall in 12 weeks.",
       src="BullionVault, 16 Nov", url="https://www.bullionvault.com/gold-news/gold-price-news/gold-price-reasons-sell-111620261"),
  dict(html="<b>Oil has eased, the dollar has firmed.</b> Brent is about $92–93 on a projected supply surplus, but the Iran war is unresolved. The dollar index is at a 19-month high, mostly because the euro fell.",
       src="Trading Economics, 16 Nov", url="https://tradingeconomics.com/commodity/brent-crude-oil/news/601607"),
  dict(html="<b>Politics.</b> Democrats won both chambers on 3 Nov; markets read it partly as a regulatory risk for AI and data centres.",
       src="CNBC, 4 Nov", url="https://www.cnbc.com/2026/11/04/2026-midterm-election-results-live-updates.html"),
]

rules_now = [
  ["Trend gate", "<span class='neg'>Off</span> · NDX 27,002 is 6.8% below its EMA-100 (28,968); off since the 16 Oct close"],
  ["Re-entry trigger", "a Friday close above <b>~28,890</b> (+7.0%) if NDX is flat until then"],
  ["Volatility (20-day)", "35.5% · a full-size F2 re-entry would be 72% QLD (vol target 0.30 ÷ 0.355 × 85%)"],
  ["Efficiency ratio", "0.26 today. A jump back above the EMA this Friday would leave the 20-day change near zero, so the ratio falls to ~0.00 and the chop rule halves the re-entry: <b>F2 about 30% QLD</b>"],
  ["Season", "November · no Aug–Sep halving"],
  ["Holding (all tiers)", "85% IEF · 15% GLD"],
  ["10-year yield", "5.10%, +44bp over six months · context only, not a rule input"],
  ["Yield curve (10y−3m)", "+0.96pp, no inversion in the last year · context only"],
]

regime = ("An inflation and real-rate shock combined with a sell-off in AI leaders, while the economy is still growing. "
          "Closest matches: 2022 for the macro mechanics (Fed hikes, Treasuries an unreliable hedge, gold pressured by real yields); "
          "Q4 2018 for the market shape (a tightening-driven tech correction that reversed when the rate path softened); "
          "and 2000–02 as the tail risk (a capex-financed tech boom unwinding). Both a fast V-shaped rebound and a 2022-style grind lower are live.")

dates = [["25 Nov", "October PCE inflation, Q3 GDP revision"], ["4 Dec", "November jobs report"], ["8–9 Dec", "FOMC decision and projections"],
         ["10 Dec", "November CPI"], ["11 Dec", "US government funding deadline"], ["14 Dec", "Hearing on Fed Governor Cook"],
         ["10 Jan", "US–China tariff truce expires"]]

analogs_text = ("After states like today's (about 15 independent episodes since 1988, in-sample), the Nasdaq-100 was higher 12 months later "
  "in 82–87% of cases, with a median gain of about 21%. A bear market deeper than 25% followed within a year in about 21–24% "
  "(17% before 2007, 38% after). From a risk-off start the strategy's 12-month return was positive about 92–96% of the time; the worst "
  "nearby start, January 2022, gave F3 −15% and F2 −14%. From today, re-entry typically took another 22–30 trading days; about 1 in 5 "
  "waits ran past three months, and 61–65% of episodes saw at least one false re-entry within six months. Conditioning on today's "
  "specific mix (high yields, strong dollar, falling gold) adds no information beyond simply being risk-off.")

findings = [
  dict(t="The scanner held the wrong bond", n="−40.7% <span style='font-size:13px;color:var(--muted)'>vs</span> −30.0%",
       p="The July backtest's bond was a 10-year Treasury (like IEF), but the scanner told you to buy TLT (20-year+). With TLT the same rules have a −40.7% worst drawdown instead of −30.0%, and lost 33.9% in 2022 instead of 22.1%. The scanner now holds IEF."),
  dict(t="“Gold” was gold-mining stocks", n="≈ no effect",
       p="The backtest's gold sleeve was the ^XAU gold-miners index, not physical gold. With real GLD the long-run results barely move (Calmar 0.80 vs 0.79), so the July numbers stand on this point. In the live window the gap was large: miners +2.1%, GLD −5.2%."),
  dict(t="The yield-curve rule never did anything", n="0 position changes",
       p="“Risk-on unless the curve inverted recently and the trend is broken” reduces to “risk-on if above the EMA”. It never changed a position in 41 years. Its claimed +1%/yr came entirely from holding bonds instead of cash. It is deleted."),
  dict(t="Friday was a lucky rebalance day", n="+2.4 pts/yr",
       p="Run on any other weekday, the same rules earned 2–5 points a year less. Friday was never chosen; it was a software default. Nothing says it will keep winning, so the planning numbers now use the average of all five weekdays."),
]

live_sub = ("The rules were frozen on 24 July; everything after is a genuine out-of-sample test on real ETF prices. "
            "The Nasdaq-100 rallied to a record on 6 October, then fell 13.5% by 17 November.")

loss_where = ("Through August and September the rules held F2 at only 21% QLD (the Aug–Sep halving plus the chop halving) while the Nasdaq rallied. "
  "Both cuts lifted at the 2 Oct close and volatility was low (15%), so F2 went to its maximum of 85% QLD right at the top. "
  "The week of 12–16 Oct cost F2 9.4% and F3 13.9%, the worst F3 week of the real-TQQQ era. F3 weeks of −12% or worse come "
  "about every five years; 2026 has had two. Low volatility at a peak is a known weak spot of volatility targeting.")

what_worked = ("The EMA-100 exit took effect on 19 Oct. Since then F2 and F3 are each down 0.9% while QQQ fell 7.5% and TQQQ 24%. "
  "Without the exit F3 would be down about 23% instead of 9%. Real ETFs matched the model to within 0.33 points, so the backtest "
  "mechanics are faithful.")

you = [
  dict(h="Pakistan's foreign-investment rule may apply", kind="warn", p=[
    "The State Bank of Pakistan's Foreign Exchange Manual (ch. 20, para 13; circular of July 2024, no 2026 change found) lets resident individuals buy foreign listed shares only through <b>one designated bank branch</b>, up to <b>US$25,000 per calendar year</b>, if they are on the Active Taxpayers' List. A V-100 return is due within a month, and <b>sale proceeds and dividends must be brought back to Pakistan</b>.",
    "No source confirmed that funding IBKR from your UAE account sidesteps this. I said in August that it did; that was an unverified assumption on my part, and I have corrected it."]),
  dict(h="The tax rate is not settled", kind="warn", p=[
    "Pakistan's 15% rate (section 37A) applies to listed securities for filers. It is not clear that foreign-listed ETFs qualify; if they don't, gains could be taxed at slab rates up to 45%. Weekly rebalancing realises gains often, so this decides a large part of your net return.",
    "A UAE bank account is not UAE tax residency. Pakistan taxes residents on worldwide income, and the UAE lists Pakistan for automatic exchange of account information (CRS)."]),
  dict(h="US estate tax above about $60k", kind="bad", p=[
    "US-listed ETFs are US-situs assets. A non-US person's estate has an exemption of only about $60,000; above it US estate tax can reach 40%. (This threshold was not re-verified in this round.)",
    "Above that level the plan is the UCITS route: <b>LQQ.PA</b> (Amundi Nasdaq-100 2×; it levers the US-dollar index, so no euro exposure), <b>CBU0.L</b> (US Treasuries 7–10y) and <b>IGLN.L</b> (physical gold). Each tracks its US twin within 0.1–0.3%/yr; the whole strategy costs about 1.2 points a year more in European trading costs, plus ±1–2 points of execution-day noise. Signal at the Friday US close, trade the Monday European session while US markets are open."]),
]

caveats = [
  "The live window is 81 trading days. It is too short to confirm or reject the strategy; it mainly tests that the rules and data behave as modelled.",
  "Before each ETF existed the leveraged sleeves are simulated (QLD before 2006, TQQQ before 2010), and Treasuries before 2002 are built from yields. Real ETFs tracked the simulations within 0.3 points in the live window, but the early decades are models.",
  "Calmar comparisons often hinge on one or two episodes (1989–90 and 2022). Gold before 2000 is a monthly series interpolated to daily.",
  "Costs assume 3 bps per unit traded on US ETFs; turnover is about 12× a year, so each extra 10 bps costs about 1.5 points a year. Taxes are not modelled.",
  "These are rules tested on history. They are not a forecast and not investment advice.",
]

method = ("<b>Data:</b> Yahoo Finance daily closes to 17 Nov 2026 (^NDX from 1985; QQQ, QLD, TQQQ; IEF, TLT, SHY, BIL; GLD and GC=F; Treasury yields); "
  "monthly gold before 2000 (datasets/gold-prices). Leveraged sleeves before inception = L × NDX − 0.95%/yr − (L−1) × T-bill; "
  "Treasuries before 2002 from yields, with duration and convexity fitted to the real ETFs. Weekly decisions at the close, applied from the next session, "
  "3 bps per unit traded. <b>Process:</b> round 1 ran 8 analyses (over 2,000 variants); round 2 ran 5 pre-registered tests; every result was "
  "reviewed by two independent adversarial agents and a final critic; a separate sourced sweep of market conditions had every finding fact-checked. "
  "<b>Code:</b> update_2026_11/ (panel.py, common.py, one script per analysis); the live scanner passes an exact parity test against the backtest engine.")

sources_html = ("Sources (fact-checked): "
  + " · ".join(f"<a href='{u}' target='_blank' rel='noopener'>{t}</a>" for t, u in [
    ("Fed statement 28 Oct", "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm"),
    ("Fed projections Sep", "https://www.federalreserve.gov/monetarypolicy/fomcprojtabl20260916.htm"),
    ("BLS CPI", "https://www.bls.gov/news.release/cpi.htm"),
    ("Treasury yield curve", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve"),
    ("BEA GDP Q3", "https://www.bea.gov/news/2026/gdp-advance-estimate-3rd-quarter-2026"),
    ("FactSet Earnings Insight", "https://advantage.factset.com/hubfs/Website/Resources%20Section/Research%20Desk/Earnings%20Insight/EarningsInsight_111326.pdf"),
    ("FRED HY spread", "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"),
    ("Nvidia Q3 FY27", "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2027"),
    ("SBP FE Manual ch. 20", "https://www.sbp.org.pk/assets/document/Chapter-20-foreign-exchange-manual.pdf"),
    ("PwC Pakistan individual tax", "https://taxsummaries.pwc.com/pakistan/individual"),
    ("UAE CRS jurisdictions", "https://mof.gov.ae/wp-content/uploads/2026/06/CRS_List-of-Participating-Jurisdictions-23-June-2026.pdf"),
  ]))

ideas_round1 = [
  dict(idea="Bond off-leg: TLT vs IEF", evidence="TLT adds about 10 points of worst drawdown and lost 34% in 2022 (IEF 22%). The gap is not statistically significant, but IEF is what the backtest actually held.", n="98", verdict="Changed"),
  dict(idea="Gold miners vs physical gold", evidence="Physical gold (GLD) gives almost the same long-run result. Gold itself adds no measurable value over the same 15% in IEF; it is kept as pre-registered diversification.", n="—", verdict="Corrected"),
  dict(idea="Yield-curve rule", evidence="A logical no-op; never changed a position. Genuine curve rules (inversion, un-inversion) all failed.", n="54", verdict="Removed"),
  dict(idea="Switch bonds to cash when rates rise", evidence="Duration, stock–bond correlation, bond-trend and inflation switches each fix 2022 but give back protection in deflationary crashes. None beats static IEF in both halves of history.", n="98", verdict="Rejected"),
  dict(idea="Managed-futures (trend) sleeve", evidence="A trend-following proxy and real funds (DBMF, KMLM) fix 2022 but deepen other drawdowns; 0 of 24 pre-specified structures passed.", n="319", verdict="Rejected"),
  dict(idea="Macro overlays", evidence="Rate shock, dollar surge, credit spreads and VIX term structure: none survived both halves after the selection-cost haircut. Credit and VIX move with stocks rather than ahead of them.", n="54", verdict="Rejected"),
  dict(idea="Re-tuning the parameters", evidence="Walk-forward picks land mid-pack out of sample (45th–59th percentile); the July defaults sit near the top of the grid, so expect less than their in-sample score.", n="1,133", verdict="Rejected"),
  dict(idea="2× vs 3×", evidence="No difference in risk-adjusted return. 3× earns about 4 points a year more and carries far fatter drawdown tails; 2× is the risk-level choice.", n="432", verdict="Risk choice"),
  dict(idea="Execution", evidence="Signal at Friday 15:40–15:45 ET, market-on-close orders. A 2-point no-trade band cuts orders from about 132 to 56 a year with no return cost; always trade on/off switches.", n="—", verdict="Added"),
  dict(idea="UCITS route (non-US investor)", evidence="LQQ.PA + CBU0.L + IGLN.L is workable at about 1.2 points a year extra cost plus timing noise. No UCITS 3× fund exists.", n="36", verdict="Route"),
]

# ---------------- ROUND 2 (filled after the pre-registered tests) ----------------
ROUND2 = json.load(open("update_2026_11/r2_dynamic.json", encoding="utf-8")) if os.path.exists("update_2026_11/r2_dynamic.json") else {}

out = dict(facts=facts, rules_now=rules_now, regime=regime, dates=dates, analogs_text=analogs_text, findings=findings,
           live_sub=live_sub, loss_where=loss_where, what_worked=what_worked, you=you + ROUND2.get("you_extra", []),
           caveats=caveats + ROUND2.get("caveats_extra", []), method=method, sources_html=sources_html,
           ideas=ideas_round1 + ROUND2.get("ideas_extra", []),
           lede=ROUND2.get("lede", ""), plan=ROUND2.get("plan", dict(tile="pending", tileSub="round 2", sub="", spec_html="", table_html="")),
           scen=ROUND2.get("scen", dict(sub="", table_html="", note="")), changes=ROUND2.get("changes", []))
json.dump(out, open("update_2026_11/r2_report_content.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote update_2026_11/r2_report_content.json")
