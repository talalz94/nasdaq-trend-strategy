"""
Assemble the Nov-2026 re-check report: update_2026_11/report_template.html + data -> Recheck_Nov2026.html
Round-2 content is read from update_2026_11/r2_report_content.json (written after round 2).
"""
import sys, os, json
sys.path.insert(0, "update_2026_11")
import numpy as np, pandas as pd
import common as C

P = C.panel(); idx = P.index
# The report's text (labels, live-window dates, the re-entry level) is written for data up to 2026-11-17.
# On a newer panel it would mix fresh numbers with those fixed labels, so it refuses instead.
_END = str(C.panel().index[-1].date())
if _END != "2026-11-17" and os.environ.get("REPORT_ANY_DATE") != "1":
    sys.exit(f"build_report.py rebuilds the November 2026 report, whose text is fixed to data up to 2026-11-17; "
             f"the panel now ends {_END}. Recheck_Nov2026.html was left unchanged "
             f"(set REPORT_ANY_DATE=1 to build it anyway).")
S1 = json.load(open("update_2026_11/report_series.json"))
S2 = json.load(open("update_2026_11/report_series2.json"))
R2PATH = "update_2026_11/r2_report_content.json"
R2 = json.load(open(R2PATH, encoding="utf-8")) if os.path.exists(R2PATH) else {}
M = S1["market"]

def esc(t): return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

# ---------------- OOS ranks: percentile of the 81-day OOS return among all rolling 81-day windows to the freeze
oos_days = idx[idx > C.FREEZE]; L = len(oos_days)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
marks = {d: pd.Series(idx.dayofweek == i, index=idx) for i, d in enumerate(DAYS)}
def rank(r):
    eq = (1 + r.fillna(0)).cumprod()
    roll = (eq / eq.shift(L) - 1)
    hist = roll[(roll.index >= C.START + pd.Timedelta(days=130)) & (roll.index <= C.FREEZE)].dropna()
    v = float((1 + r.loc[oos_days].fillna(0)).prod() - 1)
    return v, float((hist < v).mean() * 100)
cfg = {"F2": dict(lev="qld", voltgt=0.30), "F3": dict(lev="tqqq", voltgt=0.20), "F3T": dict(lev="tqqq", voltgt=0.20, off="tlt")}
ret = {k: C.backtest(P, C.strategy_weights(P, **v)) for k, v in cfg.items()}
w2 = C.strategy_weights(P, **cfg["F2"])
ret["F2tr"] = sum(C.backtest(P, w2, mark=marks[d]) for d in DAYS) / 5
ret["QQQ"] = P["qqq"].fillna(0); ret["TQQQ"] = P["tqqq"].fillna(0)
RK = {k: rank(r) for k, r in ret.items()}
def mdd(r):
    e = (1 + r.loc[oos_days].fillna(0)).cumprod(); e = pd.concat([pd.Series([1.0]), e.reset_index(drop=True)])
    return float((e / e.cummax() - 1).min())
def ordinal(p):
    p = int(round(p)); suf = "th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
    return f"{p}{suf} percentile"

# ---------------- OOS weekly log (F2): held during the week ending each Friday = decision of the previous Friday
log = []
fridays = [d for d in idx[P["mark"]] if d > C.FREEZE]
hw = C.held_weights(P, w2)
dec_days = list(idx[P["mark"]])
for f in fridays:
    prev = max(d for d in dec_days if d < f)
    row = P.loc[prev]
    if not row["above"]:
        why = f"risk-off: NDX {row['ndx']:,.0f} below EMA {row['ema100']:,.0f} at the {prev:%d %b} close"
    else:
        parts = [f"vol {row['vol20']*100:.1f}%"]
        if prev.month in (8, 9): parts.append("Aug-Sep half")
        parts.append("chop half (ER %.2f)" % row["er20"] if row["er20"] < 0.25 else "clean trend (ER %.2f)" % row["er20"])
        why = "risk-on: " + ", ".join(parts)
    h = hw.loc[f]
    log.append(dict(date=f.strftime("%d %b"), qld=float(h["qld"]), ief=float(h["ief"]), gold=float(h["gold_phys"]), why=why))

# ---------------- OOS chart series (prepend 24 Jul = 100)
x = ["2026-07-24"] + [d for d, _ in S1["oos"]["F2-IEF"]]
def ser(key): return [100.0] + [v for _, v in S1["oos"][key]]
oos = dict(x=x, series=[
    dict(name="F2 (2×, IEF)", short="F2", color="s1", v=ser("F2-IEF")),
    dict(name="F3 (3×, IEF)", short="F3", color="s2", v=ser("F3-IEF")),
    dict(name="F3 as the scanner had it (TLT)", short="F3-TLT", color="s3", v=ser("F3-TLT (as scanner)")),
    dict(name="QQQ buy & hold", short="QQQ", color="bench", v=ser("QQQ buy & hold")),
    dict(name="TQQQ buy & hold", short="TQQQ", color="bench2", v=ser("TQQQ buy & hold"))],
    marks=[dict(x="2026-10-06", label="6 Oct record"), dict(x="2026-10-16", label="16 Oct exit")],
    stats=[dict(name="F2 (2×, IEF), Friday", ret=RK["F2"][0], dd=mdd(ret["F2"]), rank=ordinal(RK["F2"][1]), hl=True),
           dict(name="F2, average of five rebalance days", ret=RK["F2tr"][0], dd=mdd(ret["F2tr"]), rank=ordinal(RK["F2tr"][1])),
           dict(name="F3 (3×, IEF), Friday", ret=RK["F3"][0], dd=mdd(ret["F3"]), rank=ordinal(RK["F3"][1])),
           dict(name="F3 as the scanner had it (TLT)", ret=RK["F3T"][0], dd=mdd(ret["F3T"]), rank=ordinal(RK["F3T"][1])),
           dict(name="QQQ buy & hold", ret=RK["QQQ"][0], dd=mdd(ret["QQQ"]), rank=ordinal(RK["QQQ"][1])),
           dict(name="TQQQ buy & hold", ret=RK["TQQQ"][0], dd=mdd(ret["TQQQ"]), rank=ordinal(RK["TQQQ"][1]))],
    log=log)

# ---------------- long-run chart (tranche basis)
TL = S2["tranche_long"]
xl = [d for d, _ in TL["F2"]]
long_series = [dict(name="F2 (2×)", short="F2", color="s1", v=[v for _, v in TL["F2"]]),
               dict(name="F3 (3×)", short="F3", color="s2", v=[v for _, v in TL["F3"]]),
               dict(name="F1 (1×)", short="F1", color="s3", v=[v for _, v in TL["F1"]]),
               dict(name="QQQ buy & hold", short="QQQ", color="bench", v=[v for _, v in TL["QQQ buy & hold"]])]
yearly = []
for i, d in enumerate(xl):
    if d.endswith("-12") or i == len(xl) - 1:
        yearly.append([d[:4] if d.endswith("-12") else d] + [f"{s['v'][i]:,.2f}" for s in long_series])
longd = dict(x=xl, series=long_series, marks=[], yearly=yearly)

# ---------------- decades / crises table (tranche basis)
TS, QS = S2["tranche_stats"], S2["qqq_stats"]
dec_rows = []
for k in ["1988-89", "1990s", "2000s", "2010s", "2020s"]:
    lab = k if k != "2020s" else "2020s (to 17 Nov 2026)"
    dec_rows.append([lab] + [f"{TS[t]['decades'][k]['cagr']*100:+.1f}%/yr" for t in ["F1", "F2", "F3"]] + [f"{QS['decades'][k]['cagr']*100:+.1f}%/yr"])
for k, lab in [("dot-com 2000-02", "Dot-com bust (Mar 2000–Oct 2002)"), ("GFC 2007-09", "Financial crisis (Oct 2007–Mar 2009)"),
               ("2011 EU", "2011 euro crisis (Apr–Dec)"), ("COVID 2020", "COVID crash (15 Feb–30 Apr 2020)"),
               ("2022 inflation", "2022 inflation bear (calendar year)"), ("2025 tariff shock", "2025 tariff shock (15 Feb–31 May)"),
               ("OOS Jul-Nov 2026", "Live window (27 Jul–17 Nov 2026)")]:
    dec_rows.append([lab] + [f"{TS[t]['crises'][k]*100:+.1f}%" for t in ["F1", "F2", "F3"]] + [f"{QS['crises'][k]*100:+.1f}%"])
decades = dict(head=["Period", "F1 (1×)", "F2 (2×)", "F3 (3×)", "QQQ buy & hold"], rows=dec_rows)

# ---------------- instrument audit (F3, Friday book, to 24 Jul)
def st(**kw):
    r = C.backtest(P, C.strategy_weights(P, **kw)); s = C.stats(r, b=C.FREEZE)
    return dict(cagr=s["cagr"], dd=s["maxdd"], cal=s["calmar"], y22=C.period_return(r, "2022-01-01", "2022-12-31"))
instr = [dict(name="July backtest (as published)", gold="gold-miners index", bond="synthetic 10-year", **st(lev="tqqq", voltgt=0.2, gold="gold_miners", off="bond10_old")),
         dict(name="What the scanner told you to buy", gold="GLD", bond="TLT (20y+)", **st(lev="tqqq", voltgt=0.2, off="tlt")),
         dict(name="Corrected, now in the scanner", gold="GLD", bond="IEF (7–10y)", hl=True, **st(lev="tqqq", voltgt=0.2, off="ief"))]

wk = {t: S2["weekday"][t] for t in ["F1", "F2", "F3"]}
reentry = 28891.0

D = dict(
    asof_label="17 Nov 2026",
    lede=R2.get("lede", ""),
    market=dict(ndx=M["ndx"], reentry=reentry),
    live=dict(F2=RK["F2"][0], QQQ=RK["QQQ"][0], TQQQ=RK["TQQQ"][0]),
    plan=R2.get("plan", dict(tile="pending", tileSub="round 2", sub="", spec_html="", table_html="")),
    facts=R2.get("facts", []), rules_now=R2.get("rules_now", []), regime=R2.get("regime", ""), dates=R2.get("dates", []),
    analogs_text=R2.get("analogs_text", ""), findings=R2.get("findings", []), instr=instr, weekday=wk,
    live_sub=R2.get("live_sub", ""), oos=oos, loss_where=R2.get("loss_where", ""), what_worked=R2.get("what_worked", ""),
    ideas=R2.get("ideas", []), long=longd, decades=decades, scen=R2.get("scen", dict(sub="", table_html="", note="")),
    you=R2.get("you", []), changes=R2.get("changes", []), caveats=R2.get("caveats", []),
    method=R2.get("method", ""), sources_html=R2.get("sources_html", ""))

def clean(o):
    if isinstance(o, float): return None if (o != o) else round(o, 6)
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
data = json.dumps(clean(D), ensure_ascii=False)
tpl = open("update_2026_11/report_template.html", encoding="utf-8").read()
assert "const D = __DATA__;" in tpl
open("Recheck_Nov2026.html", "w", encoding="utf-8").write(tpl.replace("const D = __DATA__;", "const D = " + data + ";"))
print("built Recheck_Nov2026.html", len(data), "bytes of data")
print("OOS ranks:", {k: (round(v[0]*100, 2), round(v[1], 1)) for k, v in RK.items()})
for r in log: print(r)
print("instr:", [(i["name"], round(i["cagr"]*100, 2), round(i["dd"]*100, 1), round(i["cal"], 2), round(i["y22"]*100, 1)) for i in instr])
