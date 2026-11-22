"""
PHASE B1 — Why does the hybrid (B) trail the champion (A)?

A: NDX>EMA100 -> TQQQ x volscale, else CASH
B: NDX>EMA100 AND curve>=0 -> (ER>=0.30 ? TQQQ x volscale : 100% QQQ), else BONDS

The two differ ONLY in three mutually-exclusive states:
  CURVE  : A in TQQQ, B in bonds        (curve inverted while trend up)
  CHOP   : A in TQQQ, B in QQQ          (trend up, curve ok, but choppy)
  OFFLEG : both risk-off; A cash, B bonds
Everything else is identical. So we can attribute the entire gap exactly.

Also: leave-one-out counterfactuals + conditional forward returns by ER bucket
(the latter tells us where the chop threshold SHOULD be -> feeds Phase B2).
"""
import json, numpy as np, pandas as pd
import bt
CACHE = bt.CACHE
d = pd.read_parquet("data_letf/letf_daily.parquet").set_index("date")
m = pd.read_parquet("data_letf/macro_daily.parquet")
idx = d.index; rf_d = (d["rf"]/252).fillna(0)
bond = pd.Series(np.where(idx >= pd.Timestamp("2002-08-01"), m["tlt_ret"].values, rf_d.values),
                 index=idx).fillna(0)
R = {"tqqq": d["tqqq_ret"].fillna(0), "qqq": d["qqq_ret"].fillna(0), "cash": rf_d, "bond": bond}
ndx = d["ndx"]
ema = ndx.ewm(span=100, adjust=False).mean()
vol = ndx.pct_change().rolling(20).std()*np.sqrt(252)
er = (ndx.diff(20).abs()/ndx.diff().abs().rolling(20).sum().replace(0, np.nan))
curve = m["curve"].reindex(idx)
scale = (0.25/vol).clip(0, 1).fillna(1.0)
mark = pd.Series(False, index=idx); mark[ndx.resample("W-FRI").last().index.intersection(idx)] = True
COST = 3.0/1e4
CHOP_THR = 0.30

above = ndx > ema; curve_ok = (curve >= 0).fillna(True); clean = (er >= CHOP_THR).fillna(False)

def weights(cfg):
    w = pd.DataFrame(0.0, index=idx, columns=list(R))
    use_curve = cfg.get("curve", True); use_chop = cfg.get("chop", True)
    off = cfg.get("off", "bond")
    ron = above & (curve_ok if use_curve else True)
    if use_chop:
        w["tqqq"] = (ron & clean)*scale; w["cash"] = (ron & clean)*(1-scale)
        w["qqq"] = (ron & ~clean).astype(float)
    else:
        w["tqqq"] = ron*scale; w["cash"] = ron*(1-scale)
    w[off] += (~ron).astype(float)
    return w.where(mark).ffill().fillna(0).shift(1).fillna(0)

def ret_of(w):
    return sum(w[a]*R[a] for a in R) - w.diff().abs().sum(axis=1).fillna(0)*COST

def stats(r, msk=None):
    r = r if msk is None else r[msk]
    e = (1+r).cumprod(); yrs = (r.index[-1]-r.index[0]).days/365.25
    cagr = e.iloc[-1]**(1/yrs)-1; dd = (e/e.cummax()-1).min()
    return dict(cagr=float(cagr), maxdd=float(dd), sharpe=float(r.mean()/r.std()*np.sqrt(252)),
                calmar=float(cagr/abs(dd)) if dd < 0 else np.nan, total=float(e.iloc[-1]-1))

wA = weights(dict(curve=False, chop=False, off="cash"))
wB = weights(dict(curve=True, chop=True, off="bond"))
rA, rB = ret_of(wA), ret_of(wB)
sA, sB = stats(rA), stats(rB)
print(f"A GROWTH : CAGR {sA['cagr']*100:5.1f}%  maxDD {sA['maxdd']*100:4.0f}%  Calmar {sA['calmar']:.2f}")
print(f"B HYBRID : CAGR {sB['cagr']*100:5.1f}%  maxDD {sB['maxdd']*100:4.0f}%  Calmar {sB['calmar']:.2f}")
print(f"  gap = {(sB['cagr']-sA['cagr'])*100:+.1f}%/yr CAGR, {(sB['maxdd']-sA['maxdd'])*100:+.0f}pp drawdown\n")

# ---------- 1. state-by-state attribution of the daily return difference ----------
pos = wB.shift(0)  # already shifted
stA = above.shift(1).fillna(False)
st_curve = (stA & ~curve_ok.shift(1).fillna(True))
st_chop  = (stA & curve_ok.shift(1).fillna(True) & ~clean.shift(1).fillna(False))
st_off   = (~stA)
st_same  = (stA & curve_ok.shift(1).fillna(True) & clean.shift(1).fillna(False))
diff = rB - rA
print("=== WHERE THE GAP COMES FROM (sum of daily return differences) ===")
print(f"{'state':28s}{'days':>7}{'%time':>7}{'B−A total':>12}{'B−A ann.':>11}")
rows=[]
for nm, msk in [("CURVE: A=TQQQ, B=bonds", st_curve), ("CHOP: A=TQQQ, B=QQQ", st_chop),
                ("OFFLEG: A=cash, B=bonds", st_off), ("identical (both TQQQ)", st_same)]:
    tot = diff[msk].sum(); yrs=(idx[-1]-idx[0]).days/365.25
    print(f"{nm:28s}{int(msk.sum()):7d}{msk.mean()*100:6.0f}%{tot*100:11.0f}%{tot/yrs*100:10.1f}%")
    rows.append(dict(state=nm, days=int(msk.sum()), pct=float(msk.mean()), total=float(tot), ann=float(tot/yrs)))

# ---------- 2. leave-one-out counterfactuals ----------
print("\n=== LEAVE-ONE-OUT: remove one hybrid rule at a time ===")
print(f"{'variant':34s}{'CAGR':>8}{'maxDD':>8}{'Calmar':>8}{'2000-09':>9}")
m0009 = (idx>=pd.Timestamp("2000-01-01"))&(idx<=pd.Timestamp("2009-12-31"))
loo=[]
for nm,cfg in [("B: full hybrid",dict(curve=True,chop=True,off="bond")),
               ("  − curve overlay",dict(curve=False,chop=True,off="bond")),
               ("  − chop de-lever",dict(curve=True,chop=False,off="bond")),
               ("  − bonds (use cash)",dict(curve=True,chop=True,off="cash")),
               ("A: no rules at all",dict(curve=False,chop=False,off="cash"))]:
    r=ret_of(weights(cfg)); s=stats(r); s09=stats(r,m0009)
    print(f"{nm:34s}{s['cagr']*100:7.1f}%{s['maxdd']*100:7.0f}%{s['calmar']:8.2f}{s09['cagr']*100:8.1f}%")
    loo.append(dict(name=nm,**s,cagr0009=s09["cagr"],maxdd0009=s09["maxdd"]))

# ---------- 3. episode analysis: were the interventions good calls? ----------
def episodes(msk, a_asset, b_asset, label):
    grp = (msk != msk.shift()).cumsum()[msk]
    good=bad=0; gsum=bsum=0.0; eps=[]
    for _, ix in pd.Series(grp.index, index=grp.values).groupby(level=0):
        per = pd.DatetimeIndex(ix.values)
        if len(per) < 5: continue
        a = (1+R[a_asset].loc[per]*(scale.loc[per] if a_asset=="tqqq" else 1)).prod()-1
        b = (1+R[b_asset].loc[per]).prod()-1
        (good := good) ;
        if b > a: good += 1; gsum += (b-a)
        else: bad += 1; bsum += (b-a)
        eps.append(dict(start=str(per[0].date()), days=len(per), a=float(a), b=float(b)))
    n=good+bad
    print(f"\n{label}: {n} episodes | GOOD (B beat A) {good} ({good/max(n,1)*100:.0f}%) avg +{gsum/max(good,1)*100:.1f}%"
          f" | BAD {bad} avg {bsum/max(bad,1)*100:.1f}%")
    worst=sorted(eps,key=lambda e:e["b"]-e["a"])[:3]; best=sorted(eps,key=lambda e:e["a"]-e["b"])[:3]
    print("   worst calls (B gave up most):", ", ".join(f"{e['start']} ({(e['b']-e['a'])*100:+.0f}%)" for e in worst))
    print("   best  calls (B saved most):  ", ", ".join(f"{e['start']} ({(e['b']-e['a'])*100:+.0f}%)" for e in best))
    return dict(n=n,good=good,bad=bad,worst=worst,best=best)
print("\n=== EPISODE ANALYSIS (was each intervention the right call?) ===")
ep_chop = episodes(st_chop, "tqqq", "qqq", "CHOP de-lever")
ep_curve= episodes(st_curve, "tqqq", "bond", "CURVE risk-off")

# ---------- 4. conditional forward returns by ER bucket -> where should the threshold be? ----------
print("\n=== FORWARD 20d TQQQ RETURN BY EFFICIENCY-RATIO BUCKET (uptrend days only) ===")
fwd = (d["tqqq_ret"].fillna(0).add(1).rolling(20).apply(np.prod, raw=True).shift(-20)-1)
up = above & curve_ok
buck = pd.cut(er[up], [0,0.15,0.25,0.30,0.40,0.55,1.01],
              labels=["0-.15",".15-.25",".25-.30",".30-.40",".40-.55",".55+"])
tab = pd.DataFrame({"er_bucket":buck,"fwd":fwd[up],"qqq_fwd":(d["qqq_ret"].fillna(0).add(1).rolling(20).apply(np.prod,raw=True).shift(-20)-1)[up]}).dropna()
g = tab.groupby("er_bucket", observed=True).agg(n=("fwd","size"), tqqq_fwd=("fwd","mean"),
                                                qqq_fwd=("qqq_fwd","mean"), winrate=("fwd", lambda x:(x>0).mean()))
g["tqqq_minus_qqq"] = g["tqqq_fwd"]-g["qqq_fwd"]
print((g*100).round(2).to_string())
print("\n-> buckets where tqqq_minus_qqq > 0 are where LEVERAGE PAYS; current threshold de-levers below 0.30")

json.dump({"A":sA,"B":sB,"attrib":rows,"loo":loo,"ep_chop":ep_chop,"ep_curve":ep_curve,
           "er_table":g.reset_index().astype({"er_bucket":str}).to_dict("records")},
          open(f"{CACHE}/attribution.json","w"), indent=1, default=str)
print("\nsaved attribution.json")
