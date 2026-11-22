"""
verify_rate_regime_robustness_followup.py -- follow-up checks for the rate_regime robustness review.
  1  TLT vs IEF by decade and by thirds (Calmar, maxDD) and Ulcer index by half  (claim C8)
  2  which drawdown sets F3 / F2 static-IEF maxDD for later start dates           (claim C7)
  3  real-ETF era (2002-08..2026-07) with and without calendar 2022             (claims C2/C7)
  4  selection bias in the quoted 'cost' of the chosen switch rule               (recommendation)
  5  OOS full-window instrument returns and stock-bond sign                       (claim C9)
Prints only; writes nothing.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

P = C.panel(); idx = P.index; FREEZE = C.FREEZE; LAST = idx[-1]


def ma(s, n):
    return s.rolling(n, min_periods=n).mean()


tnx = P["tnx"]; ief_px = (1 + P["ief"].fillna(0)).cumprod(); gsci = P["gsci"].ffill()
SIG = {"durA_tnx200_bills": tnx > ma(tnx, 200), "durB_tnx200_shy": tnx > ma(tnx, 200),
       "corr63_bills": P["ndx_ret"].rolling(63, min_periods=63).corr(P["ief"]) > 0,
       "corr126_bills": P["ndx_ret"].rolling(126, min_periods=126).corr(P["ief"]) > 0,
       "bondtrend200_bills": ief_px < ma(ief_px, 200),
       "infl_gsci_bills": (gsci > ma(gsci, 200)) & (gsci / gsci.shift(252) - 1 > 0)}


def static(**kw):
    return pd.DataFrame({k: pd.Series(float(v), index=idx) for k, v in kw.items()})


def switch(cond, short="bills"):
    c = cond.reindex(idx).fillna(False).astype(float)
    return pd.DataFrame({"ief": 1.0 - c, short: c}, index=idx)


POL = {"static_ief": static(ief=1), "static_tlt": static(tlt=1), "static_shy": static(shy=1),
       "static_bills": static(bills=1), "ief50_bills50": static(ief=0.5, bills=0.5),
       "thirds_bills_shy_ief": static(bills=1/3, shy=1/3, ief=1/3)}
for k, s in SIG.items():
    POL[k] = switch(s, "shy" if k.startswith("durB") else "bills")
STRATS = {"F3": dict(lev="tqqq", voltgt=0.20), "F2": dict(lev="qld", voltgt=0.30)}
R = {s: {p: C.backtest(P, C.strategy_weights(P, **STRATS[s], gold="gold_phys", gold_w=0.15, off_weights=ow))
         for p, ow in POL.items()} for s in STRATS}


def IS(r):
    return r[(r.index >= C.START) & (r.index <= FREEZE)]


def ulcer(r):
    e = (1 + r).cumprod(); dd = e / e.cummax() - 1
    return float(np.sqrt((dd ** 2).mean()))


def ddloc(r):
    e = (1 + r).cumprod(); dd = e / e.cummax() - 1; t = dd.idxmin(); pk = e.loc[:t].idxmax()
    return f"{dd.min()*100:.1f}% {pk.date()}->{t.date()}"


print("1) TLT vs IEF")
DEC = [("1988-01-01", "1989-12-31"), ("1990-01-01", "1999-12-31"), ("2000-01-01", "2009-12-31"),
       ("2010-01-01", "2019-12-31"), ("2020-01-01", str(FREEZE.date()))]
THIRDS = [("1988-01-01", "2000-12-31"), ("2001-01-01", "2013-12-31"), ("2014-01-01", str(FREEZE.date()))]
for s in STRATS:
    for lab, wins in (("decades", DEC), ("thirds", THIRDS)):
        row = []
        for a, b in wins:
            t = C.stats(IS(R[s]["static_tlt"]), a, b, start=None); i = C.stats(IS(R[s]["static_ief"]), a, b, start=None)
            row.append(f"{a[:4]}-{b[:4]}: Cal {t['calmar']:.2f} vs {i['calmar']:.2f}, DD {t['maxdd']*100:.1f} vs {i['maxdd']*100:.1f}")
        print(f"  {s} {lab}: " + " | ".join(row))
    for lab, (a, b) in {"full": ("1988-01-01", str(FREEZE.date())), "H1": ("1988-01-01", "2006-12-31"),
                        "H2": ("2007-01-01", str(FREEZE.date()))}.items():
        u = {p: ulcer(IS(R[s][p])[a:b]) for p in POL}
        print(f"  {s} {lab} Ulcer index (lower=better): " + ", ".join(f"{p} {v*100:.2f}" for p, v in sorted(u.items(), key=lambda x: x[1])))

print("\n2) static-IEF maxDD location by start date")
for s in STRATS:
    for st in ["1988-01-01", "1990-07-01", "1991-01-01", "1996-01-01", "2002-08-01"]:
        r = IS(R[s]["static_ief"]); b = IS(R[s]["bondtrend200_bills"])
        print(f"  {s} from {st}: IEF {ddloc(r[st:])} | bond trend {ddloc(b[st:])}")

print("\n3) real-ETF era 2002-08-01..2026-07-24, with / without calendar 2022 (both series neutralised)")
m22 = (idx >= "2022-01-01") & (idx <= "2022-12-31")
for s in STRATS:
    for lab, f in (("with 2022", lambda r: r), ("2022 neutralised", lambda r: r.where(~m22, 0.0))):
        sc = {p: C.stats(IS(f(R[s][p])), "2002-08-01", None, start=None)["calmar"] for p in POL}
        beat = sorted([p for p in POL if p != "static_ief" and sc[p] > sc["static_ief"]])
        print(f"  {s} {lab:17s}: IEF {sc['static_ief']:.3f}  bond trend {sc['bondtrend200_bills']:.3f}  bills {sc['static_bills']:.3f}"
              f"  -> {len(beat)}/11 beat IEF: {beat}")

print("\n4) selection bias in the quoted cost of the chosen switch rule (full-IS CAGR diff vs IEF, %/yr)")
for s in STRATS:
    base = C.stats(IS(R[s]["static_ief"]))["cagr"]
    d = {k: (C.stats(IS(R[s][k]))["cagr"] - base) * 100 for k in SIG}
    print(f"  {s}: " + ", ".join(f"{k} {v:+.2f}" for k, v in d.items()) + f" | mean of 6 {np.mean(list(d.values())):+.2f}, chosen bond trend {d['bondtrend200_bills']:+.2f}")

print("\n5) OOS instruments (2026-07-25..latest) and since risk-off")
for a, b in (("2026-07-25", str(LAST.date())), ("2026-10-19", str(LAST.date()))):
    v = {c: C.period_return(P[c], a, b) * 100 for c in ["ndx_ret", "ief", "tlt", "bills", "gold_phys"]}
    print(f"  {a}..{b}: " + " ".join(f"{k} {x:+.2f}%" for k, x in v.items()),
          f" d10y {P['tnx'].loc[b] - P['tnx'].loc[:pd.Timestamp(a) - pd.Timedelta(days=1)].iloc[-1]:+.3f}pp")
    x = P.loc[a:b]
    print(f"     daily corr(ndx_ret, ief) {x['ndx_ret'].corr(x['ief']):+.2f}; weekly corr "
          f"{((1+x['ndx_ret']).resample('W-FRI').prod()-1).corr((1+x['ief']).resample('W-FRI').prod()-1):+.2f}")

print("\n6) TLT vs IEF with BOTH 2022 and the 1989-90 drawdown neutralised (both series)")
m8990 = (idx >= "1989-10-01") & (idx <= "1990-06-30")
for s in STRATS:
    for lab, m in (("2022 only", m22), ("1989-90 only", m8990), ("2022 + 1989-90", m22 | m8990)):
        t = C.stats(IS(R[s]["static_tlt"].where(~m, 0.0))); i = C.stats(IS(R[s]["static_ief"].where(~m, 0.0)))
        print(f"  {s} {lab:15s}: TLT Cal {t['calmar']:.3f} DD {t['maxdd']*100:.1f} CAGR {t['cagr']*100:.2f} | "
              f"IEF Cal {i['calmar']:.3f} DD {i['maxdd']*100:.1f} CAGR {i['cagr']*100:.2f}")

print("\n7) expanding walk-forward (saved picks): log-return contribution vs static IEF by picked policy")
import json
SV = json.load(open(os.path.join(HERE, "rate_regime.json")))
for s in STRATS:
    picks = SV["walkforward"][s]["expanding_main12"]["picks"]
    contrib = {}
    for y, p in picks.items():
        y = int(y)
        if y > 2026:
            continue
        a, b = f"{y}-01-01", (f"{y}-12-31" if y < 2026 else str(FREEZE.date()))
        d = np.log1p(C.period_return(R[s][p], a, b)) - np.log1p(C.period_return(R[s]["static_ief"], a, b))
        contrib[p] = contrib.get(p, 0.0) + d
    print(f"  {s}: " + ", ".join(f"{p}: {v*100:+.2f}% log" for p, v in contrib.items()) +
          f" | total {sum(contrib.values())*100:+.2f}% log over 1996..2026-07")

print("\n8) post-2022 record of the switch rules vs static IEF (log-return difference, %)")
for s in STRATS:
    for a, b in (("2023-01-01", str(FREEZE.date())), ("2026-07-25", str(LAST.date())), ("2023-01-01", str(LAST.date()))):
        d = {p: (np.log1p(C.period_return(R[s][p], a, b)) - np.log1p(C.period_return(R[s]["static_ief"], a, b))) * 100
             for p in ["bondtrend200_bills", "durA_tnx200_bills", "corr126_bills", "static_bills", "static_tlt"]}
        print(f"  {s} {a}..{b}: " + ", ".join(f"{p} {v:+.2f}" for p, v in d.items()))
