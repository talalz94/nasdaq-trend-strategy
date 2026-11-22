"""
Item 3: $1000 + leverage, "how much without blowing up?"
Constant-leverage model: equity_{t+1} = equity_t * (1 + L * r_t), r = per-trade net return.
Uses the walk-forward-validated GOLD champion (single instrument, one position at a time,
so sequential compounding is realistic; 21 years of trades; gold CFD offers real leverage).
Kelly gives the growth-optimal L; Monte-Carlo gives probability of ruin.
"""
import json, numpy as np, pandas as pd
import bt, assets

CACHE = bt.CACHE
START = 1000.0
RUIN = 0.50          # "blown up" = equity ever falls >=50% below start (<= $500)

def champion_trades():
    """Gold champion with a 3% protective stop (essential under leverage)."""
    gc1 = assets.gold_cache("1h")
    def feat(cache, tr, col):
        return np.array([bool(cache[t["ticker"]][col][max(int(t.entry_i)-1,0)]) for _,t in tr.iterrows()])
    tr = bt.run_strategy(gc1, "SQZ_zero_cross", stop=0.03, cost_bps=1.5)
    keep = feat(gc1,tr,"mashift_regime_up") & feat(gc1,tr,"mashift_up")
    a = tr[keep].sort_values("entry_time")
    gc4 = assets.gold_cache("4h")
    tr4 = bt.run_strategy(gc4, "SQZ_color_long", stop=0.03, cost_bps=1.5)
    keep4 = feat(gc4,tr4,"mashift_regime_up")
    b = tr4[keep4].sort_values("entry_time")
    return a, b

def compound(r, L):
    mult = 1 + L*r
    mult = np.where(mult <= 0, 0.0, mult)          # wiped out on that trade
    eq = START*np.cumprod(mult)
    return eq

def maxdd(eq):
    eq = np.concatenate([[START], eq]); peak=np.maximum.accumulate(eq)
    return float((eq/peak-1).min())

def monte(r, L, npaths=5000, seed=7):
    rng=np.random.default_rng(seed); n=len(r)
    finals=[]; ruined=0
    idx=rng.integers(0,n,size=(npaths,n))
    for p in range(npaths):
        rr=r[idx[p]]; mult=1+L*rr; mult=np.where(mult<=0,0.0,mult)
        eq=START*np.cumprod(mult)
        emin=np.min(np.concatenate([[START],eq]))
        if emin <= START*RUIN: ruined+=1
        finals.append(eq[-1])
    finals=np.array(finals)
    return dict(p_ruin=ruined/npaths, med=float(np.median(finals)),
                p05=float(np.percentile(finals,5)), p95=float(np.percentile(finals,95)))

def kelly(r):
    m=r.mean(); v=r.var()
    return float(m/v) if v>0 else np.nan

def study(name, tr):
    r=tr["ret"].to_numpy()
    yrs=(pd.to_datetime(tr.entry_time).max()-pd.to_datetime(tr.entry_time).min()).days/365.25
    kf=kelly(r)
    print(f"\n===== LEVERAGE STUDY — {name} =====")
    print(f"trades={len(r)} over {yrs:.1f}y | avg={r.mean()*1e4:.1f}bps win={(r>0).mean():.2f} "
          f"worst={r.min()*100:.1f}% best={r.max()*100:.1f}% | full-Kelly L*~{kf:.1f} (half~{kf/2:.1f})")
    print(f"{'lev':>4}{'final$(actual)':>16}{'CAGR':>9}{'maxDD':>8}{'P(ruin50%)':>12}{'MC median$':>12}{'MC p05$':>10}{'MC p95$':>12}")
    rows=[]
    for L in [1,2,3,5,8,10,15,20,30]:
        eq=compound(r,L); final=eq[-1]; cagr=(final/START)**(1/yrs)-1 if final>0 else -1
        mc=monte(r,L)
        print(f"{L:>4}{final:>16,.0f}{cagr*100:>8.0f}%{maxdd(eq)*100:>7.0f}%{mc['p_ruin']*100:>11.1f}%"
              f"{mc['med']:>12,.0f}{mc['p05']:>10,.0f}{mc['p95']:>12,.0f}")
        rows.append(dict(L=L, final=float(final), cagr=float(cagr), maxdd=maxdd(eq), **mc))
    return dict(name=name, kelly=kf, years=yrs, avg_bps=float(r.mean()*1e4),
                worst=float(r.min()), rows=rows)

a,b = champion_trades()
out={}
out["gold_1h"]=study("Gold SQZ_zero_cross 1h +filters +3% stop", a)
out["gold_4h"]=study("Gold SQZ_color_long 4h +trend +3% stop", b)

# recommendation: highest L with P(ruin)<5%
for k,v in out.items():
    ok=[row for row in v["rows"] if row["p_ruin"]<0.05]
    best=max(ok, key=lambda x:x["med"]) if ok else None
    if best:
        print(f"\n[{k}] safe pick (P(ruin)<5%): L={best['L']}x -> $1000 becomes ~${best['med']:,.0f} "
              f"(median), maxDD {best['maxdd']*100:.0f}%")
json.dump(out, open(f"{CACHE}/leverage.json","w"), indent=1)
print("\nsaved leverage.json")
