"""
NON-US INVESTOR ROUTE — what the European/UCITS version costs.
US-listed: TQQQ 0.95% / QLD 0.95% / TLT 0.15% / GLD 0.40%, spreads ~1-3bps, $0 commission.
UCITS/ETP equivalents are pricier and thinner:
  leveraged Nasdaq ETPs (Leverage Shares 3x, Amundi LQQ 2x): ~0.75-1.00% + wider spreads
  DTLA (20y+ UST UCITS) ~0.07% ; IGLN/SGLN physical gold ~0.12-0.25%
  plus per-trade commission at most non-US brokers (~$1-5 or 0.05-0.10%)
Model the realistic drag so the decision is informed.
"""
import numpy as np, pandas as pd, json
import engine as E
D=E.data(); idx=D.idx; CACHE=E.CACHE
mo=pd.Series(idx.month,index=idx); AS=pd.Series(np.where(mo.isin([8,9]),0.5,1.0),index=idx)
EXP_L=0.0095
qld=(2*D.d["ndx_ret"]-EXP_L/252-1*D.rf_d).fillna(0)
R=dict(D.R); R["qld"]=qld

def build(lev="tqqq",voltgt=0.20,gold_w=0.15,cost=3/1e4,extra=None,commission=0.0):
    scale=((voltgt/D.vol).clip(0,1).fillna(1.0)*AS).clip(0,1)
    ron=D.above & ~D.confirmed; clean=(D.er>=0.25).fillna(False)
    eq=(ron&clean)*scale+(ron&~clean)*scale*0.5
    gw=pd.Series(gold_w,index=idx); core=1.0-gw
    w=pd.DataFrame(0.0,index=idx,columns=[lev,"bond","gold","cash"])
    w[lev]=eq*core; w["bond"]+=(core-eq*core).clip(lower=0); w["gold"]+=gw
    ww=w.where(D.mark).ffill().fillna(0).shift(1).fillna(0)
    turn=ww.diff().abs().sum(axis=1).fillna(0)
    r=sum(ww[c]*R[c].fillna(0) for c in ww.columns)-turn*cost
    if commission: r=r-(turn>0).astype(float)*commission     # per-rebalance ticket cost
    if extra:
        for c,er in extra.items():
            if c in ww: r=r-ww[c]*(er/252)
    return r

rows=[]
def add(lbl,**kw):
    r=build(**kw); s=E.stats(r); rows.append(dict(label=lbl,**s)); return s

print("="*78); print(" US-LISTED vs UCITS/EUROPEAN ROUTE — realistic all-in comparison"); print("="*78)
print(f"{'route':52s}{'CAGR':>8}{'maxDD':>7}{'Calmar':>7}")
s=add("US 3x (TQQQ/TLT/GLD, 3bps, $0 comm)",lev="tqqq",cost=3/1e4,extra={"bond":0.0015,"gold":0.0040})
print(f"{'US 3x (TQQQ/TLT/GLD, 3bps, $0 comm)':52s}{s['cagr']*100:7.2f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}")
s=add("US 2x (QLD, vol .30)",lev="qld",voltgt=0.30,cost=3/1e4,extra={"bond":0.0015,"gold":0.0040})
print(f"{'US 2x (QLD, vol .30)':52s}{s['cagr']*100:7.2f}%{s['maxdd']*100:6.0f}%{s['calmar']:7.2f}")
# UCITS: +0.30% extra on the leveraged sleeve (0.75-1.0% vs 0.95% baseline is ~similar; ETPs often ~1.0-1.25%)
for lbl,lev,vt,extra_lev,cst,comm in [
  ("UCITS 3x ETP (+0.30% fee, 15bps sprd, $2/trade)","tqqq",0.20,0.0030,15/1e4,0.0002),
  ("UCITS 2x (LQQ-style, +0.05% fee, 12bps, $2)","qld",0.30,0.0005,12/1e4,0.0002),
  ("UCITS 2x pessimistic (25bps sprd, $5/trade)","qld",0.30,0.0005,25/1e4,0.0005)]:
    ex={"bond":0.0007,"gold":0.0020}; ex[lev]=extra_lev
    r=build(lev=lev,voltgt=vt,cost=cst,extra=ex,commission=comm); st=E.stats(r)
    rows.append(dict(label=lbl,**st))
    print(f"{lbl:52s}{st['cagr']*100:7.2f}%{st['maxdd']*100:6.0f}%{st['calmar']:7.2f}")

base3=rows[0]["cagr"]; base2=rows[1]["cagr"]
print(f"\n UCITS 3x route costs ~{(base3-rows[2]['cagr'])*100:.2f}%/yr vs US 3x")
print(f" UCITS 2x route costs ~{(base2-rows[3]['cagr'])*100:.2f}%/yr vs US 2x  (pessimistic {(base2-rows[4]['cagr'])*100:.2f}%/yr)")

# tax comparison for a non-US resident (illustrative, capital-gains driven strategy)
print("\n=== WHY THE TAX PICTURE IS DIFFERENT FOR A NON-US RESIDENT ===")
print(" This strategy's return is ~all CAPITAL GAINS, not dividends:")
gross=rows[1]["cagr"]
for lbl,rate in [("US person, taxable acct, 35% short-term",0.35),
                 ("US person, IRA/401k (sheltered)",0.0),
                 ("Non-US resident: US capital-gains tax on NRA*",0.0),
                 ("...but home-country tax at 15%",0.15),
                 ("...home-country tax at 30%",0.30)]:
    print(f"   {lbl:48s} net ~{gross*(1-rate)*100:5.1f}%/yr")
print(" *Non-resident aliens generally owe NO US tax on US capital gains; dividends are")
print("  withheld (30%, or less by treaty). Home-country tax still applies. VERIFY LOCALLY.")
json.dump(rows,open(f"{CACHE}/ucits.json","w"),indent=1,default=str)
print("\nsaved ucits.json")
