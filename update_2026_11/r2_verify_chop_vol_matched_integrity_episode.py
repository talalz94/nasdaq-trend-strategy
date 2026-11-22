"""INTEGRITY REVIEW helper (read-only): where does the B-vs-V (vol target) and B-vs-A (chop) gap arise?
Monthly B/V/A tranche returns 2000-2002, vol20/above at key dates, sub-era Calmar differences (2001-02, 2003-06),
plus N225 / IXIC half-period rows used to test the 'one era and one market' wording. Prints only."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
P = C.panel(); IDX = P.index
S = json.load(open(os.path.join(HERE, "r2_chop_vol_matched.json")))
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MK = {d: pd.Series(IDX.dayofweek == i, index=IDX) for i, d in enumerate(DAYS)}
def tr(w): return sum(C.backtest(P, w, mark=MK[d]) for d in DAYS) / 5
for t, col, vt in [("F2", "qld", 0.30), ("F3", "tqqq", 0.20)]:
    sp = S["ndx"][t]["tranche"]
    vtA, cV = sp["solve_A"]["vt"], sp["solve_V"]["c"]
    rB = tr(C.strategy_weights(P, lev=col, voltgt=vt))
    rA = tr(C.strategy_weights(P, lev=col, voltgt=vtA, chop_pos=1.0))
    rV = tr(C.strategy_weights(P, lev=col, voltgt=1e9, extra_scale=pd.Series(cV, index=IDX)))
    print(f"\n{t}: sub-era Calmar B-V / B-A")
    for a, b in [("1988-01-01", "1999-12-31"), ("2000-01-01", "2000-12-31"), ("2001-01-01", "2002-12-31"),
                 ("2003-01-01", "2006-12-31"), ("2001-01-01", "2006-12-31"), ("2007-01-01", "2026-07-24")]:
        sB, sV, sA = (C.stats(r, a, b, start=None) for r in (rB, rV, rA))
        print(f"  {a[:4]}-{b[:4]}: B Cal {sB['calmar']:.3f} DD {sB['maxdd']*100:6.1f}% | V Cal {sV['calmar']:.3f} DD {sV['maxdd']*100:6.1f}% | A Cal {sA['calmar']:.3f} | B-V {sB['calmar']-sV['calmar']:+.3f} B-A {sB['calmar']-sA['calmar']:+.3f}")
    m = (1 + pd.concat([rB, rV, rA], axis=1, keys=["B", "V", "A"])).loc["2000-01-01":"2002-12-31"].resample("ME").prod() - 1
    m["V-B"] = m["V"] - m["B"]
    print("  monthly 2000-2002 (V-B gap, %), months with |gap| > 1pp:")
    for k, r in m.iterrows():
        if abs(r["V-B"]) > 0.01:
            print(f"    {k.date()} B {r['B']*100:+6.1f} V {r['V']*100:+6.1f} A {r['A']*100:+6.1f} gap {r['V-B']*100:+5.1f}")
    # cumulative V-B gap in 2000 split: Mar-Apr vs rest
    g = (m["V"] - m["B"])
    print(f"  sum of monthly V-B gaps: 2000-03..04 {g.loc['2000-03':'2000-04'].sum()*100:+.1f}pp, 2000-05..12 {g.loc['2000-05':'2000-12'].sum()*100:+.1f}pp, 2001 {g.loc['2001'].sum()*100:+.1f}pp, 2002 {g.loc['2002'].sum()*100:+.1f}pp")
# NDX state at key 2000 dates
print("\nNDX state 2000 (Fridays):")
for d in pd.date_range("2000-03-03", "2000-10-27", freq="W-FRI"):
    if d in IDX:
        print(f"  {d.date()} above {bool(P.loc[d,'above'])} vol20 {P.loc[d,'vol20']:.2f} er20 {P.loc[d,'er20']:.2f} scale.30 {min(1,0.30/P.loc[d,'vol20']):.2f} scale.20 {min(1,0.20/P.loc[d,'vol20']):.2f}")
cm = S["cross_market"]["markets"]
print("\ncross-market chop dCal H1/H2/FULL:", {m: tuple(round(cm[m]['dcalmar_B_minus_alt']['chop'][w], 3) for w in ['H1', 'H2', 'FULL']) for m in cm})
print("cross-market vol  dCal H1/H2/FULL:", {m: tuple(round(cm[m]['dcalmar_B_minus_alt']['vol'][w], 3) for w in ['H1', 'H2', 'FULL']) for m in cm})
print("n225 B/A FULL & H1 calmar:", {k: (round(cm['n225']['variants'][k]['3bps']['FULL']['calmar'], 3), round(cm['n225']['variants'][k]['3bps']['H1']['calmar'], 3), cm['n225']['variants'][k]['3bps']['H1']['maxdd']) for k in ['B', 'A']})
rm = S["information_only"]["within_half_rematch_tranche"]
print("within-half rematch:", {k: (round(v['dcalmar_chop'], 4), round(v['dcalmar_vol'], 4)) for k, v in rm.items()})
for t in ["F2", "F3"]:
    b = S["ndx"][t]["tranche"]["variants"]
    print(t, "A0 vs B IS Calmar", round(b["A0"]["3bps"]["IS"]["calmar"], 4), round(b["B"]["3bps"]["IS"]["calmar"], 4))
