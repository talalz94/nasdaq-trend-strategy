"""
Does GEOPOLITICAL/MACRO regime adaptation beat the static Phase-C allocation?

Economic hypotheses (each simple + motivated, to resist overfitting):
 H1 adaptive gold     : hold more gold when debasement/commodity/dollar flags fire
 H2 stagflation off-leg: bonds FAIL in stagflation (2022 proved it) -> use gold when
                         stagflation is on, bonds otherwise
 H3 geo de-risk       : cut leveraged exposure when geopolitical stress is high
 H4 china sleeve      : allocate to HK/EM when China is leading
 H5 combos of the survivors
Baseline = Phase C final (bond off-leg, static 15% gold).
"""
import json, numpy as np, pandas as pd
import engine as E
D = E.data()
G = D.geo
idx = D.idx
CACHE = E.CACHE

def build(gold_series=None, gold_w=0.15, off_series=None, off="bond",
          veto=None, chop_pos=0.5, voltgt=0.20, extra_sleeve=None):
    """weights with optional time-varying gold weight and time-varying off-leg."""
    scale = (voltgt/D.vol).clip(0,1).fillna(1.0)
    ron = D.above & ~D.confirmed
    if veto is not None: ron = ron & ~veto.reindex(idx).fillna(False)
    clean = (D.er >= 0.25).fillna(False)
    eq = (ron&clean)*scale + (ron&~clean)*scale*chop_pos
    gw = (gold_series.reindex(idx).fillna(0) if gold_series is not None
          else pd.Series(gold_w, index=idx)).clip(0,0.6)
    core = 1.0-gw
    cols = ["tqqq","bond","gold","cash","hsi"]
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w["tqqq"] = eq*core
    free = (core - eq*core).clip(lower=0)
    if extra_sleeve is not None:                 # carve a slice of free capital
        s = extra_sleeve.reindex(idx).fillna(0).clip(0,1)
        w["hsi"] += free*s; free = free*(1-s)
    if off_series is None:
        w[off] += free
    else:                                        # off_series: True -> gold, else bond
        m = off_series.reindex(idx).fillna(False)
        w["gold"] += free*m.astype(float); w["bond"] += free*(~m).astype(float)
    w["gold"] += gw
    return E.backtest(w)

res=[]
def add(name, grp, r):
    s = E.stats(r); dec = E.by_decade(r)
    w2 = E.stats(r,*E.WINDOWS["2000-2010"]); w1 = E.stats(r,*E.WINDOWS["1992-95"])
    res.append(dict(name=name, grp=grp, **s,
                    dec={k:round(v["cagr"],4) for k,v in dec.items()},
                    w0010=w2["cagr"], dd0010=w2["maxdd"], w9295=w1["cagr"]))
    return res[-1]

# ---- baselines ----
add("BASELINE Phase C (bond, 15% gold)","base", build())
add("  no gold (bond off-leg only)","base", build(gold_w=0.0))

# ---- H1 adaptive gold from geo_score ----
gs = G["geo_score"].reindex(idx).ffill().fillna(0)
for lo,hi in [(0.05,0.30),(0.0,0.30),(0.10,0.35)]:
    gser = lo + (hi-lo)*(gs/4.0)
    add(f"H1 gold {int(lo*100)}-{int(hi*100)}% by geo_score","H1", build(gold_series=gser))
# gold weight driven by gold_lead flag alone
for base_g,up_g in [(0.05,0.25),(0.10,0.30)]:
    gser = pd.Series(np.where(G["gold_lead"].reindex(idx).fillna(False), up_g, base_g), index=idx)
    add(f"H1 gold {int(base_g*100)}/{int(up_g)*1}%->{int(up_g*100)}% on gold_lead","H1", build(gold_series=gser))

# ---- H2 stagflation-aware off-leg (bonds fail in stagflation) ----
stag = G["stagflation"].reindex(idx).fillna(False)
add("H2 off-leg = gold when stagflation","H2", build(off_series=stag))
add("H2 off-leg gold-in-stag + 10% gold","H2", build(off_series=stag, gold_w=0.10))
cb = G["commod_bull"].reindex(idx).fillna(False)
add("H2b off-leg = gold when commod_bull","H2", build(off_series=cb))

# ---- H3 de-risk on geopolitical stress ----
for thr in [3,4]:
    add(f"H3 veto equity when geo_score>={thr}","H3", build(veto=(gs>=thr)))
for cp in [0.25,0.35]:
    # lower chop position when stress high: approximate via two runs blended by flag
    r_hi = build(chop_pos=cp); r_lo = build(chop_pos=0.5)
    mask = (gs>=3).astype(float).reindex(idx).fillna(0)
    add(f"H3 chop_pos {cp} when geo_score>=3","H3", r_hi*mask + r_lo*(1-mask))
add("H3 veto equity when oil_shock","H3", build(veto=G["oil_shock"].reindex(idx).fillna(False)))

# ---- H4 China/EM sleeve when China leads ----
cl = G["china_lead"].reindex(idx).fillna(False)
for share in [0.3,0.5]:
    add(f"H4 {int(share*100)}% of free capital to HK when china_lead","H4",
        build(extra_sleeve=pd.Series(np.where(cl,share,0.0),index=idx)))

# ---- H5 combos of the promising ones ----
gser_best = 0.05 + 0.25*(gs/4.0)
add("H5 adaptive gold + gold-in-stagflation","H5", build(gold_series=gser_best, off_series=stag))
add("H5 adaptive gold + stag off-leg + oil veto","H5",
    build(gold_series=gser_best, off_series=stag, veto=G["oil_shock"].reindex(idx).fillna(False)))

df = pd.DataFrame(res)
base = df[df.name.str.startswith("BASELINE")].iloc[0]
df["d_cagr"] = df["cagr"]-base["cagr"]; df["d_calmar"] = df["calmar"]-base["calmar"]
df["d_0010"] = df["w0010"]-base["w0010"]

print(f"{'variant':46s}{'CAGR':>7}{'maxDD':>7}{'Calmar':>7}{'2000-10':>9}{'dCal':>7}{'d00-10':>8}")
for _,r in df.sort_values("calmar",ascending=False).iterrows():
    print(f"{r['name']:46s}{r['cagr']*100:6.1f}%{r['maxdd']*100:6.0f}%{r['calmar']:7.2f}"
          f"{r['w0010']*100:8.1f}%{r['d_calmar']:+7.2f}{r['d_0010']*100:+7.1f}%")

print("\n=== DECADE VIEW: baseline vs best 3 by Calmar ===")
top = df.sort_values("calmar",ascending=False).head(4)
print(f"{'variant':46s}"+"".join(f"{k:>9s}" for k in E.DECADES))
for _,r in pd.concat([df[df.name.str.startswith('BASELINE')],top]).drop_duplicates("name").iterrows():
    print(f"{r['name']:46s}"+"".join(f"{r['dec'][k]*100:8.1f}%" for k in E.DECADES))

json.dump(df.to_dict("records"), open(f"{CACHE}/geo_test.json","w"), indent=1, default=str)
print("\nsaved geo_test.json")
