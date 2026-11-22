"""
verify_managed_futures_robustness_checks.py -- adversarial ROBUSTNESS checks of managed_futures.py claims.

Reads the series dumped by verify_managed_futures_robustness_rerun.py (scratch parquet) + the shared panel.
Writes nothing inside the project (prints only; optional JSON dump to the scratch dir).

Checks
  A  episode dependence of every Calmar verdict: which drawdown episode sets maxDD; Calmar on windows that
     exclude 1989-90 and/or 2022; leave-one-episode-out; rolling 10y windows
  B  path-robust risk metrics (Ulcer, CDaR95, mean of 5 deepest DDs) and rf-adjusted Sharpe
  C  paired block bootstrap (63d blocks, 2000 draws, seed 0, IS only) of rf-ADJUSTED Sharpe / CAGR / Calmar
  D  vol-matched off-leg (50% TSMOM + 50% T-bills ~ IEF vol) and an all-bills off-leg control
  E  mechanism (C3): risk-on/off excess returns by half; stocks-and-bonds-down months vs IEF
  F  beta vs timing decomposition of the proxy P&L (C7)
  G  the 1989-10..1990-05 episode: proxy attribution

usage: python update_2026_11/verify_managed_futures_robustness_checks.py <scratch_dir>
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = os.path.abspath(sys.argv[1])
D = pd.read_parquet(os.path.join(SCR, "managed_futures_rerun_series.parquet"))
P = C.panel().copy()
idx = P.index
START, FREEZE = C.START, C.FREEZE
rf = P["rf_d"]
OUT = {}
NV = 0          # reviewer variants (new portfolios) evaluated


def win(r, a, b):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
    return s


def st(r, a=START, b=FREEZE):
    s = win(r, a, b)
    e = (1 + s).cumprod()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    dd = e / e.cummax() - 1
    tr = dd.idxmin()
    pk = e[:tr].idxmax()
    x = (s - rf.reindex(s.index))
    return dict(cagr=float(cagr), maxdd=float(dd.min()), calmar=float(cagr / abs(dd.min())),
                peak=str(pk.date()), trough=str(tr.date()),
                ulcer=float(np.sqrt((dd ** 2).mean())),
                cdar95=float(dd[dd <= dd.quantile(0.05)].mean()),
                exsharpe=float(x.mean() / x.std() * np.sqrt(252)))


def top_dd(r, a=START, b=FREEZE, n=5):
    s = win(r, a, b)
    e = (1 + s).cumprod()
    dd = e / e.cummax() - 1
    out, used = [], pd.Series(False, index=dd.index)
    for _ in range(n):
        d2 = dd[~used]
        if len(d2) == 0 or d2.min() >= 0:
            break
        tr = d2.idxmin(); pk = e[:tr].idxmax()
        rec = e[tr:][e[tr:] >= e[pk]]
        end = rec.index[0] if len(rec) else dd.index[-1]
        out.append((float(dd[tr]), str(pk.date()), str(tr.date())))
        used[(dd.index >= pk) & (dd.index <= end)] = True
    return out


def drop_window(r, a, b):
    return r[~((r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)))]


def fmt(d):
    return (f"CAGR {d['cagr']*100:5.1f}% maxDD {d['maxdd']*100:6.1f}% ({d['peak']}->{d['trough']}) "
            f"Calmar {d['calmar']:.2f} exSh {d['exsharpe']:.2f} Ulcer {d['ulcer']*100:4.1f} CDaR95 {d['cdar95']*100:5.1f}")


base = {"F3": D["base::F3"], "F2": D["base::F2"]}
PX = ["main_12m", "blend_1_3_12m"]
ALLPX = [c.split("::")[1] for c in D.columns if c.startswith("prox::") and not c.endswith("control_always_long")]
assert len(ALLPX) == 22, len(ALLPX)


def port(e, stc, pk):
    key = f"port::{e} | {stc} | {pk}"
    return D[key] if key in D else D[f"rob::{e} | {stc} | {pk}"]


WINS = {"IS 1988-2026.07": (START, FREEZE), "H1 1988-2006": ("1988-01-01", "2006-12-31"),
        "H2 2007-2026.07": ("2007-01-01", FREEZE), "ex 1988-90 (1991-2026.07)": ("1991-01-01", FREEZE),
        "ex 2022+ (1988-2021)": ("1988-01-01", "2021-12-31"), "H2 ex 2022+ (2007-2021)": ("2007-01-01", "2021-12-31"),
        "1991-2021": ("1991-01-01", "2021-12-31")}

# ============================================================ A. episode dependence
print("=" * 110)
print("A. EPISODE DEPENDENCE -- which drawdown sets maxDD, and the Calmar verdict on windows excluding 1989-90 / 2022")
A = {}
pairs = []
for e in ("F3", "F2"):
    for pk in PX:
        pairs.append((f"{e} off-leg swap [{pk}]", base[e], port(e, "replace_ief_offleg", pk)))
    pairs.append((f"{e} +20% TSMOM sleeve [main]", base[e], port(e, "sleeve20", "main_12m")))
    pairs.append((f"{e} +20% T-bill sleeve", base[e], D[f"cash::{e} | sleeve20 | CONTROL bills (plain de-lever)"]))
    pairs.append((f"{e} +20% TSMOM vs +20% T-bills", D[f"cash::{e} | sleeve20 | CONTROL bills (plain de-lever)"],
                  port(e, "sleeve20", "main_12m")))
for nm, rb, rt in pairs:
    A[nm] = {}
    print(f"\n-- {nm}  (first = reference, second = candidate)")
    for wn, (a, b) in WINS.items():
        sb, s2 = st(rb, a, b), st(rt, a, b)
        A[nm][wn] = dict(ref=sb, cand=s2, d_calmar=s2["calmar"] - sb["calmar"], d_cagr=s2["cagr"] - sb["cagr"])
        print(f"  {wn:27s} ref Cal {sb['calmar']:.2f} (DD {sb['maxdd']*100:5.1f}% trough {sb['trough']})  "
              f"cand Cal {s2['calmar']:.2f} (DD {s2['maxdd']*100:5.1f}% trough {s2['trough']})  dCal {s2['calmar']-sb['calmar']:+.3f}  dCAGR {(s2['cagr']-sb['cagr'])*100:+.2f}%")
OUT["A_windows"] = A

print("\n-- top-5 IS drawdowns")
for nm, r in [("F3 baseline", base["F3"]), ("F3 off-leg swap main", port("F3", "replace_ief_offleg", "main_12m")),
              ("F2 baseline", base["F2"]), ("F2 off-leg swap main", port("F2", "replace_ief_offleg", "main_12m")),
              ("F2 +20% TSMOM", port("F2", "sleeve20", "main_12m")),
              ("F2 +20% T-bills", D["cash::F2 | sleeve20 | CONTROL bills (plain de-lever)"]),
              ("proxy main_12m", D["prox::main_12m"])]:
    print(f"  {nm:22s} " + " | ".join(f"{x[0]*100:5.1f}% {x[1]}->{x[2]}" for x in top_dd(r)))

# across all 22 proxies: sign of dCalmar for the off-leg swap on windows
print("\n-- off-leg swap across ALL 22 TSMOM proxy variants: # variants with dCalmar > 0 per window")
A2 = {}
for e in ("F3", "F2"):
    A2[e] = {}
    for wn, (a, b) in WINS.items():
        sb = st(base[e], a, b)
        ds = np.array([st(port(e, "replace_ief_offleg", pk), a, b)["calmar"] - sb["calmar"] for pk in ALLPX])
        A2[e][wn] = dict(n_pos=int((ds > 0).sum()), median=float(np.median(ds)), min=float(ds.min()), max=float(ds.max()))
        print(f"  {e} {wn:27s} dCal>0 in {int((ds>0).sum()):2d}/22  median {np.median(ds):+.3f}  range [{ds.min():+.3f},{ds.max():+.3f}]")
OUT["A_offleg_all_proxies"] = A2

# path-robust metrics across all 22 proxies (IS)
print("\n-- off-leg swap, IS, path-robust measures across 22 proxies: # variants better than baseline")
A3 = {}
for e in ("F3", "F2"):
    sb = st(base[e])
    rows = [st(port(e, "replace_ief_offleg", pk)) for pk in ALLPX]
    upi_b = sb["cagr"] / sb["ulcer"]
    cd_b = sb["cagr"] / abs(sb["cdar95"])
    t5b = np.mean([x[0] for x in top_dd(base[e])])
    better = dict(
        cagr_over_ulcer=sum(r["cagr"] / r["ulcer"] > upi_b for r in rows),
        cagr_over_cdar95=sum(r["cagr"] / abs(r["cdar95"]) > cd_b for r in rows),
        cagr_over_avgtop5dd=sum(r["cagr"] / abs(np.mean([x[0] for x in top_dd(port(e, "replace_ief_offleg", pk))])) > sb["cagr"] / abs(t5b)
                                for r, pk in zip(rows, ALLPX)),
        exsharpe=sum(r["exsharpe"] > sb["exsharpe"] for r in rows),
        cagr=sum(r["cagr"] > sb["cagr"] for r in rows))
    A3[e] = better
    m = st(port(e, "replace_ief_offleg", "main_12m"))
    t5m = np.mean([x[0] for x in top_dd(port(e, "replace_ief_offleg", "main_12m"))])
    print(f"  {e}: baseline {fmt(sb)}  avgTop5DD {t5b*100:.1f}%")
    print(f"  {e}: swap main {fmt(m)}  avgTop5DD {t5m*100:.1f}%")
    print(f"  {e}: CAGR/Ulcer base {upi_b:.2f} swap-main {m['cagr']/m['ulcer']:.2f} | CAGR/CDaR95 base {cd_b:.2f} swap-main {m['cagr']/abs(m['cdar95']):.2f} | "
          f"CAGR/avgTop5 base {sb['cagr']/abs(t5b):.2f} swap-main {m['cagr']/abs(t5m):.2f}")
    print(f"  {e}: # of 22 proxies where swap beats baseline: {better}")
OUT["A_pathrobust_offleg"] = A3

# leave-one-episode-out (drop the days of one episode from both series)
EPIS = {"1989-10..1990-12": ("1989-10-01", "1990-12-31"), "1994": ("1994-01-01", "1994-12-31"),
        "1998 LTCM": ("1998-07-01", "1998-10-31"), "dot-com 2000-02": ("2000-03-01", "2002-10-31"),
        "GFC 2007-09": ("2007-10-01", "2009-03-31"), "2011 EU": ("2011-04-01", "2011-12-31"),
        "2018 Q4": ("2018-10-01", "2018-12-31"), "COVID 2020": ("2020-02-15", "2020-04-30"),
        "2022": ("2022-01-01", "2022-12-31")}
print("\n-- leave-one-episode-out (IS, days of the episode dropped from both series): dCalmar candidate - reference")
LOEO = {}
for nm, rb, rt in pairs:
    LOEO[nm] = {}
    row = []
    for en, (a, b) in EPIS.items():
        d = st(drop_window(rt, a, b))["calmar"] - st(drop_window(rb, a, b))["calmar"]
        LOEO[nm][en] = d
        row.append(f"{en}:{d:+.3f}")
    full = st(rt)["calmar"] - st(rb)["calmar"]
    print(f"  {nm:34s} full {full:+.3f} | " + " ".join(row))
OUT["A_leave_one_episode_out"] = LOEO

# rolling 10y windows
print("\n-- rolling 10-year windows (Jan y .. Dec y+9, y=1988..2016): share of windows where candidate Calmar > reference")
ROLL = {}
for nm, rb, rt in pairs:
    ds = []
    for y in range(1988, 2017):
        a, b = f"{y}-01-01", f"{y+9}-12-31"
        ds.append(st(rt, a, b)["calmar"] - st(rb, a, b)["calmar"])
    ds = np.array(ds)
    ROLL[nm] = dict(n=len(ds), share_pos=float((ds > 0).mean()), median=float(np.median(ds)))
    print(f"  {nm:34s} {int((ds>0).sum()):2d}/{len(ds)} windows positive, median dCal {np.median(ds):+.3f}")
OUT["A_rolling10y"] = ROLL

# ============================================================ C. bootstrap on rf-adjusted quantities
print("\n" + "=" * 110)
print("C. PAIRED BLOCK BOOTSTRAP (63d, 2000 draws, seed 0, IS) -- rf-ADJUSTED Sharpe vs the C.stats rf-ignoring Sharpe")


def boot(r1, r2, n=2000, block=63, seed=0):
    a = pd.concat([r1, r2, rf], axis=1).dropna()
    a = a[(a.index >= START) & (a.index <= FREEZE)].values
    T = len(a); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); ar = np.arange(block)
    res = np.empty((n, 5))
    for k in range(n):
        stt = rng.integers(0, T, nb)
        ix = (stt[:, None] + ar[None, :]).ravel()[:T] % T
        s = a[ix]
        x = s[:, :2] - s[:, 2:3]
        sh_ex = x.mean(0) / x.std(0, ddof=1) * np.sqrt(252)
        sh_raw = s[:, :2].mean(0) / s[:, :2].std(0, ddof=1) * np.sqrt(252)
        e = np.cumprod(1 + s[:, :2], axis=0)
        g = e[-1] ** (252 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1)
        ul = np.sqrt((dd ** 2).mean(0))
        res[k] = (sh_ex[0] - sh_ex[1], sh_raw[0] - sh_raw[1], g[0] - g[1],
                  g[0] / abs(dd.min(0)[0]) - g[1] / abs(dd.min(0)[1]), g[0] / ul[0] - g[1] / ul[1])
    out = {}
    for j, kk in enumerate(["d_exsharpe", "d_sharpe_rawC", "d_cagr", "d_calmar", "d_cagr_over_ulcer"]):
        x = res[:, j]
        out[kk] = dict(lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)), mean=float(x.mean()),
                       p_le0=float((x <= 0).mean()))
    return out


BOOT = {}
for e in ("F3", "F2"):
    for stc in ("sleeve10", "sleeve20", "sleeve30", "replace_ief_offleg"):
        for pk in PX:
            nm = f"{e} {stc} {pk} vs baseline"
            b = boot(port(e, stc, pk), base[e])
            BOOT[nm] = b
            print(f"  {nm:44s} dExSharpe [{b['d_exsharpe']['lo']:+.3f},{b['d_exsharpe']['hi']:+.3f}] p<=0 {b['d_exsharpe']['p_le0']:.3f} | "
                  f"dSharpe(rf-ignored) [{b['d_sharpe_rawC']['lo']:+.3f},{b['d_sharpe_rawC']['hi']:+.3f}] | dCAGR [{b['d_cagr']['lo']*100:+.2f},{b['d_cagr']['hi']*100:+.2f}]% | "
                  f"d(CAGR/Ulcer) [{b['d_cagr_over_ulcer']['lo']:+.2f},{b['d_cagr_over_ulcer']['hi']:+.2f}] p<=0 {b['d_cagr_over_ulcer']['p_le0']:.2f}")
    nm = f"{e} sleeve20 T-bills vs baseline"
    b = boot(D[f"cash::{e} | sleeve20 | CONTROL bills (plain de-lever)"], base[e])
    BOOT[nm] = b
    print(f"  {nm:44s} dExSharpe [{b['d_exsharpe']['lo']:+.3f},{b['d_exsharpe']['hi']:+.3f}] | dSharpe(rf-ignored) [{b['d_sharpe_rawC']['lo']:+.3f},{b['d_sharpe_rawC']['hi']:+.3f}]")
OUT["C_boot"] = BOOT

# ============================================================ D. vol-matched off-leg
print("\n" + "=" * 110)
print("D. APPLES-TO-APPLES RISK: IEF vol vs proxy vol, and a vol-matched off-leg (50% TSMOM + 50% T-bills)")
m_is = (idx >= START) & (idx <= FREEZE)
for c, s in (("ief", P["ief"]), ("proxy main_12m", D["prox::main_12m"]), ("proxy blend", D["prox::blend_1_3_12m"])):
    print(f"  IS vol {c:16s} {s[m_is].std()*np.sqrt(252)*100:5.1f}%")
for pk in PX:
    P[f"ts_{pk}"] = D[f"prox::{pk}"]
VM = {}
ENG = {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}
for e, (lev, vt) in ENG.items():
    rb = base[e]
    sb = st(rb)
    rows = {}
    for nm, ow in [("off = 50% TSMOM main + 50% bills", {"ts_main_12m": 0.5, "bills": 0.5}),
                   ("off = 50% TSMOM blend + 50% bills", {"ts_blend_1_3_12m": 0.5, "bills": 0.5}),
                   ("off = 100% bills (control)", {"bills": 1.0})]:
        w = C.strategy_weights(P, lev=lev, voltgt=vt, gold="gold_phys", gold_w=0.15, off=None,
                               off_weights=pd.DataFrame(ow, index=idx))
        r = C.backtest(P, w); NV += 1
        s, h1, h2 = st(r), st(r, *WINS["H1 1988-2006"]), st(r, *WINS["H2 2007-2026.07"])
        y22 = C.period_return(r, "2022-01-01", "2022-12-31"); oos = C.period_return(r, "2026-07-25", idx[-1])
        rows[nm] = dict(IS=s, H1=h1, H2=h2, y2022=y22, OOS=oos,
                        d_calmar=s["calmar"] - sb["calmar"], d_H1=h1["calmar"] - st(rb, *WINS["H1 1988-2006"])["calmar"],
                        d_H2=h2["calmar"] - st(rb, *WINS["H2 2007-2026.07"])["calmar"])
        print(f"  {e} {nm:36s} {fmt(s)}  dCal {rows[nm]['d_calmar']:+.3f} (H1 {rows[nm]['d_H1']:+.3f}, H2 {rows[nm]['d_H2']:+.3f})  2022 {y22*100:+.1f}%  OOS {oos*100:+.1f}%")
    print(f"  {e} baseline (IEF off-leg)                {fmt(sb)}")
    VM[e] = rows
OUT["D_volmatched"] = VM

# ============================================================ E. mechanism (C3)
print("\n" + "=" * 110)
print("E. MECHANISM (C3): proxy vs IEF excess return by F state and by half")
ron = P["above"].astype(float).where(P["mark"]).ffill().shift(1).fillna(0.0) > 0.5
E = {}
for pk in PX:
    r = D[f"prox::{pk}"]
    for hn, (a, b) in (("IS", (START, FREEZE)), ("H1", WINS["H1 1988-2006"]), ("H2", WINS["H2 2007-2026.07"])):
        mm = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        for lbl, mask in (("risk-off", (~ron) & mm), ("risk-on", ron & mm)):
            xp = (r - rf)[mask]; xi = (P["ief"] - rf)[mask]
            E[f"{pk} {hn} {lbl}"] = dict(days=int(mask.sum()), proxy=float(xp.mean() * 252), ief=float(xi.mean() * 252),
                                          proxy_vol=float(xp.std() * np.sqrt(252)), ief_vol=float(xi.std() * np.sqrt(252)),
                                          proxy_sh=float(xp.mean() / xp.std() * np.sqrt(252)), ief_sh=float(xi.mean() / xi.std() * np.sqrt(252)))
            v = E[f"{pk} {hn} {lbl}"]
            print(f"  {pk:14s} {hn} {lbl:8s} days {v['days']:5d}  proxy {v['proxy']*100:+5.1f}%/yr (vol {v['proxy_vol']*100:4.1f}, Sh {v['proxy_sh']:+.2f})  "
                  f"IEF {v['ief']*100:+5.1f}%/yr (vol {v['ief_vol']*100:4.1f}, Sh {v['ief_sh']:+.2f})")
# attribution of the main proxy's risk-on P&L
held = {mk: D[f"held::main_12m::{mk}"] for mk in ["spx", "ndx", "intl", "ief", "tlt", "gold", "dxy", "gsci"]}
X = pd.DataFrame({"spx": P["spx"] - rf, "ndx": P["ndx_ret"] - rf, "intl": P["intl"] - rf, "ief": P["ief"] - rf,
                  "tlt": P["tlt"] - rf, "gold": P["gold_phys"] - rf, "dxy": P["dxy_ret"], "gsci": P["gsci_ret"]}).fillna(0.0)
contrib = pd.DataFrame({mk: held[mk] * X[mk] for mk in held})
for lbl, mask in (("risk-on", ron & m_is), ("risk-off", (~ron) & m_is)):
    c = contrib[mask].mean() * 252
    eq = c[["spx", "ndx", "intl"]].sum()
    print(f"  main_12m gross P&L {lbl}: total {c.sum()*100:+.2f}%/yr, equity legs {eq*100:+.2f}%/yr ({eq/c.sum()*100:.0f}%), "
          f"bonds {c[['ief','tlt']].sum()*100:+.2f}, gold {c['gold']*100:+.2f}, dxy {c['dxy']*100:+.2f}, gsci {c['gsci']*100:+.2f}")
    E[f"attrib main {lbl}"] = {k: float(v) for k, v in c.items()}
# stocks-and-bonds-down months: proxy vs IEF
def mo(s):
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1
Mx = pd.DataFrame({"spx": mo(P["spx"][m_is]), "ief": mo(P["ief"][m_is]), "proxy": mo(D["prox::main_12m"][m_is]),
                   "blend": mo(D["prox::blend_1_3_12m"][m_is]), "rf": mo(rf[m_is])})
sb_ = Mx[(Mx.spx < 0) & (Mx.ief < 0)]
print(f"  SBD months {len(sb_)}: proxy avg {sb_.proxy.mean()*100:+.2f}%/mo vs IEF {sb_.ief.mean()*100:+.2f}%; proxy>IEF in "
      f"{(sb_.proxy > sb_.ief).mean()*100:.0f}% of them; proxy>T-bill {(sb_.proxy > sb_.rf).mean()*100:.0f}%; "
      f"blend>IEF {(sb_.blend > sb_.ief).mean()*100:.0f}%")
E["sbd"] = dict(n=int(len(sb_)), proxy_gt_ief=float((sb_.proxy > sb_.ief).mean()), proxy_gt_rf=float((sb_.proxy > sb_.rf).mean()),
                proxy_avg=float(sb_.proxy.mean()), ief_avg=float(sb_.ief.mean()))
OUT["E_mechanism"] = E

# ============================================================ F. beta vs timing (C7)
print("\n" + "=" * 110)
print("F. BETA vs TIMING DECOMPOSITION of the main_12m proxy (gross, IS)")
Fd = {}
H = pd.DataFrame(held)[m_is]; XX = X[m_is]
static = sum(H[mk].mean() * XX[mk].mean() for mk in H) * 252
timing = sum(((H[mk] - H[mk].mean()) * (XX[mk] - XX[mk].mean())).mean() for mk in H) * 252
total = (H * XX).sum(axis=1).mean() * 252
print(f"  gross excess P&L {total*100:.2f}%/yr = static (avg position x avg return) {static*100:.2f}%/yr + timing (cov) {timing*100:.2f}%/yr")
for hn, (a, b) in (("H1", WINS["H1 1988-2006"]), ("H2", WINS["H2 2007-2026.07"])):
    mm = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    H_ = pd.DataFrame(held)[mm]; X_ = X[mm]
    s_ = sum(H_[mk].mean() * X_[mk].mean() for mk in H_) * 252
    t_ = sum(((H_[mk] - H_[mk].mean()) * (X_[mk] - X_[mk].mean())).mean() for mk in H_) * 252
    print(f"    {hn}: static {s_*100:.2f}%/yr  timing {t_*100:.2f}%/yr")
    Fd[hn] = dict(static=float(s_), timing=float(t_))
Fd["IS"] = dict(static=float(static), timing=float(timing), total=float(total))
# regression of TSMOM excess on the always-long control's excess
y = (D["prox::main_12m"] - rf)[m_is]; xx = (D["prox::control_always_long"] - rf)[m_is]
beta = np.cov(y, xx)[0, 1] / xx.var()
alpha = (y.mean() - beta * xx.mean()) * 252
print(f"  regression TSMOM_ex = a + b*control_ex (daily IS): beta {beta:.2f}, alpha {alpha*100:.2f}%/yr, R2 {np.corrcoef(y,xx)[0,1]**2:.2f}; "
      f"TSMOM ex mean {y.mean()*252*100:.2f}%/yr -> share explained by control beta {beta*xx.mean()/y.mean()*100:.0f}%")
ym, xm = mo(D["prox::main_12m"][m_is]) - mo(rf[m_is]), mo(D["prox::control_always_long"][m_is]) - mo(rf[m_is])
bm = np.cov(ym, xm)[0, 1] / xm.var()
print(f"  monthly: beta {bm:.2f}, alpha {(ym.mean()-bm*xm.mean())*1200:.2f}%/yr, R2 {np.corrcoef(ym,xm)[0,1]**2:.2f}")
Fd["regression_on_control"] = dict(beta=float(beta), alpha_ann=float(alpha), r2=float(np.corrcoef(y, xx)[0, 1] ** 2),
                                   share_explained=float(beta * xx.mean() / y.mean()), beta_monthly=float(bm),
                                   alpha_monthly_ann=float((ym.mean() - bm * xm.mean()) * 12))
OUT["F_beta_timing"] = Fd

# ============================================================ G. the 1989-90 episode
print("\n" + "=" * 110)
print("G. 1989-10-09..1990-05-02 (the IS maxDD episode of F3 baseline and of every off-leg swap)")
a, b = "1989-10-10", "1990-05-02"
mm = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
for nm, r in (("proxy main_12m", D["prox::main_12m"]), ("proxy blend", D["prox::blend_1_3_12m"]),
              ("proxy ex_equities", D["prox::ex_equities"]), ("control always-long", D["prox::control_always_long"]),
              ("IEF", P["ief"]), ("T-bills", P["bills"]), ("TQQQ sim", P["tqqq"]), ("QLD sim", P["qld"]), ("gold_phys", P["gold_phys"])):
    print(f"  {nm:20s} {C.period_return(r, a, b)*100:+6.1f}%")
c = contrib[mm].sum()
print("  main_12m gross contribution by leg (sum of daily): " + ", ".join(f"{k} {v*100:+.1f}" for k, v in c.items()))
held_ep = pd.DataFrame(held)[mm].mean()
print("  avg held notional in the episode: " + ", ".join(f"{k} {v:+.2f}" for k, v in held_ep.items()))
yr = {}
for y_ in (1989, 1990, 1994, 2011, 2016, 2018):
    yr[y_] = C.period_return(D["prox::main_12m"], f"{y_}-01-01", f"{y_}-12-31")
print("  proxy main_12m calendar years: " + ", ".join(f"{k} {v*100:+.1f}%" for k, v in yr.items()))
w_ = P.loc[mm, "gold_src"].value_counts().to_dict()
print("  gold_src in episode:", w_)
OUT["G_1989_90"] = dict(proxy_years=yr, contrib={k: float(v) for k, v in c.items()})

# OOS proxy maxDD (C11)
print("\nOOS maxDD of proxy main_12m 2026-07-25..:",
      round(float(((1 + win(D['prox::main_12m'], '2026-07-25', idx[-1])).cumprod().pipe(lambda e: (e / e.cummax().clip(lower=1.0) - 1).min())) * 100), 1))
print(f"\nreviewer variants evaluated in this script (new portfolios): {NV}")
json.dump(OUT, open(os.path.join(SCR, "verify_mf_checks.json"), "w"), indent=1, default=str)
