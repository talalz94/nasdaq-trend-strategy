"""
verify_managed_futures_robustness_followup.py -- follow-ups to the ROBUSTNESS review of managed_futures.py

  1  real managed-futures funds vs the proxy on identical windows: CAGR, excess Sharpe; 20% sleeve of each real fund
     vs a same-size T-bill sleeve (is a REAL TSMOM sleeve more than de-levering?); real-fund off-leg CAGR delta
  2  C10 ex-equity sleeve: windows excluding 1989-90 / 2022 and leave-one-episode-out; vs T-bill sleeve
  3  TSMOM sleeve vs same-size T-bill sleeve: paired block bootstrap of rf-adjusted Sharpe and CAGR/Ulcer (C4)
  4  1991-2006 (H1 without the 1989-90 episode)

usage: python update_2026_11/verify_managed_futures_robustness_followup.py <scratch_dir>
(needs the scratch files written by ..._rerun.py and ..._realfunds.py)
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = os.path.abspath(sys.argv[1])
D = pd.read_parquet(os.path.join(SCR, "managed_futures_rerun_series.parquet"))
PXR = pd.read_parquet(os.path.join(SCR, "verify_mf_realfunds_prices.parquet"))
P = C.panel().copy()
idx = P.index
rf = P["rf_d"]
START, FREEZE = C.START, C.FREEZE
ENG = {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}
P["ts_main_12m"] = D["prox::main_12m"]
NV = 0


def F(lev, vt, **kw):
    return C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, **kw)


def sl(w, col, s):
    w2 = w * (1 - s)
    w2[col] = w2.get(col, 0.0) + s
    return w2


def exsh(r, a, b):
    x = (r - rf)[(idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))].dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def stt(r, a, b):
    return C.stats(r, a, b, start=None)


print("1. REAL FUNDS vs PROXY on identical windows (start = first month after inception, end 2026-07-24)")
for t in PXR.columns:
    s = PXR[t].dropna()
    r = s.reindex(idx).ffill().pct_change(); r[idx <= s.index[0]] = np.nan
    col = t.lower(); P[col] = r
    a = max((s.index[0] + pd.offsets.MonthBegin(1)).normalize(), pd.Timestamp("2007-04-01"))
    sf, sp = stt(r, a, FREEZE), stt(D["prox::main_12m"], a, FREEZE)
    print(f"\n  {t} {a.date()}..: fund CAGR {sf['cagr']*100:5.1f}% exSh {exsh(r, a, FREEZE):+.2f} maxDD {sf['maxdd']*100:5.1f}% | "
          f"proxy CAGR {sp['cagr']*100:5.1f}% exSh {exsh(D['prox::main_12m'], a, FREEZE):+.2f} maxDD {sp['maxdd']*100:5.1f}% | "
          f"T-bills CAGR {stt(P['bills'], a, FREEZE)['cagr']*100:4.1f}%")
    for e, (lev, vt) in ENG.items():
        rb = C.backtest(P, F(lev, vt, off="ief")); sb = stt(rb, a, FREEZE)
        out = []
        for lbl, w in (("+20% T-bills", sl(F(lev, vt, off="ief"), "bills", 0.2)), (f"+20% {t}", sl(F(lev, vt, off="ief"), col, 0.2)),
                       ("+20% proxy", sl(F(lev, vt, off="ief"), "ts_main_12m", 0.2)), (f"off={t}", F(lev, vt, off=col))):
            rr = C.backtest(P, w)
            if lbl.startswith("+20% T") or col in lbl.lower():
                NV += 1
            s2 = stt(rr, a, FREEZE)
            out.append(f"{lbl} dCal {s2['calmar']-sb['calmar']:+.2f} dCAGR {(s2['cagr']-sb['cagr'])*100:+.1f}% 2022 {C.period_return(rr,'2022-01-01','2022-12-31')*100:+.1f}%")
        print(f"   {e} base Cal {sb['calmar']:.2f} CAGR {sb['cagr']*100:.1f}% | " + " | ".join(out))

print("\n2. C10 ex-equity TSMOM sleeve (KMLM-like): episode dependence")
WINS = {"IS": (START, FREEZE), "H1": ("1988-01-01", "2006-12-31"), "H2": ("2007-01-01", FREEZE),
        "1991-2026.07": ("1991-01-01", FREEZE), "1988-2021": ("1988-01-01", "2021-12-31"),
        "2007-2021": ("2007-01-01", "2021-12-31"), "1991-2006": ("1991-01-01", "2006-12-31")}
for e in ENG:
    rb = D[f"base::{e}"]
    rt = D[f"rob::{e} | sleeve20 | ex_equities"]
    rc = D[f"cash::{e} | sleeve20 | CONTROL bills (plain de-lever)"]
    for wn, (a, b) in WINS.items():
        sb, s2, sc = stt(rb, a, b), stt(rt, a, b), stt(rc, a, b)
        print(f"   {e} {wn:13s} base Cal {sb['calmar']:.2f} | +20% ex-eq Cal {s2['calmar']:.2f} (DD {s2['maxdd']*100:5.1f}%) dCal {s2['calmar']-sb['calmar']:+.3f} "
              f"| vs +20% T-bills ({sc['calmar']:.2f}) {s2['calmar']-sc['calmar']:+.3f}")
    for en, (a, b) in {"drop 1989-10..1990-12": ("1989-10-01", "1990-12-31"), "drop 2022": ("2022-01-01", "2022-12-31")}.items():
        keep = ~((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))
        d = C.stats(rt[keep], b=FREEZE)["calmar"] - C.stats(rb[keep], b=FREEZE)["calmar"]
        print(f"   {e} {en}: dCal vs baseline {d:+.3f}")
    print(f"   {e} ex-eq off-leg swap dCal IS {C.stats(D[f'rob::{e} | replace_ief_offleg | ex_equities'], b=FREEZE)['calmar'] - C.stats(rb, b=FREEZE)['calmar']:+.3f}")

print("\n3. TSMOM sleeve vs same-size T-bill sleeve: paired block bootstrap (63d, 2000, seed 0, IS) of rf-adjusted quantities")


def boot(r1, r2, n=2000, block=63, seed=0):
    a = pd.concat([r1, r2, rf], axis=1).dropna()
    a = a[(a.index >= START) & (a.index <= FREEZE)].values
    T = len(a); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); ar = np.arange(block)
    res = np.empty((n, 2))
    for k in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + ar[None, :]).ravel()[:T] % T
        s = a[ix]
        x = s[:, :2] - s[:, 2:3]
        sh = x.mean(0) / x.std(0, ddof=1) * np.sqrt(252)
        e = np.cumprod(1 + s[:, :2], axis=0); g = e[-1] ** (252 / T) - 1
        ul = np.sqrt(((e / np.maximum.accumulate(e, axis=0) - 1) ** 2).mean(0))
        res[k] = (sh[0] - sh[1], g[0] / ul[0] - g[1] / ul[1])
    return [(float(np.percentile(res[:, j], 2.5)), float(np.percentile(res[:, j], 97.5)), float((res[:, j] <= 0).mean())) for j in range(2)]


for e in ENG:
    for s in (10, 20, 30):
        for pk in ("main_12m", "blend_1_3_12m"):
            b = boot(D[f"port::{e} | sleeve{s} | {pk}"], D[f"cash::{e} | sleeve{s} | CONTROL bills (plain de-lever)"])
            print(f"   {e} sleeve{s} {pk:14s} TSMOM - T-bills: dExSharpe [{b[0][0]:+.3f},{b[0][1]:+.3f}] p<=0 {b[0][2]:.3f} | "
                  f"d(CAGR/Ulcer) [{b[1][0]:+.2f},{b[1][1]:+.2f}] p<=0 {b[1][2]:.3f}")

print("\n4. 1991-2006 (H1 without 1989-90): dCalmar")
for e in ENG:
    rb = D[f"base::{e}"]
    for nm, rt in (("+20% TSMOM", D[f"port::{e} | sleeve20 | main_12m"]), ("+20% T-bills", D[f"cash::{e} | sleeve20 | CONTROL bills (plain de-lever)"]),
                   ("off-leg swap main", D[f"port::{e} | replace_ief_offleg | main_12m"]), ("off-leg swap blend", D[f"port::{e} | replace_ief_offleg | blend_1_3_12m"])):
        print(f"   {e} {nm:20s} 1991-2006 dCal {stt(rt,'1991-01-01','2006-12-31')['calmar']-stt(rb,'1991-01-01','2006-12-31')['calmar']:+.3f}   "
              f"H1 dCal {stt(rt,'1988-01-01','2006-12-31')['calmar']-stt(rb,'1988-01-01','2006-12-31')['calmar']:+.3f}")
print(f"\nreviewer variants evaluated in this script (new portfolios): {NV}")
