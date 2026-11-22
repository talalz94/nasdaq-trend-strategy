"""
r2_verify_model_hygiene_robustness_divq.py -- ROBUSTNESS-lens check, part 3 (read-only; writes only to scratch).

Diagnoses the Friday R6 H1 flips seen under the reviewer's alternative ALL1_divQ (dividend-neutral model applied with
QQQ TOTAL return on 1999-03-11..inception days):
  (1) daily tracking noise of QQQ_real vs NDX price (+0.20% fee) by period -- QQQ closed on AMEX (4:15pm) until 2004-12;
  (2) R6 H1 (IEF minus TLT Calmar, 1988-2006) and H1 maxDD episodes for F2/F3, Friday + tranche, under
        ALL1      (specialist primary: A_div + B + C1)
        ALL1_divQ (QQQ TR base on 1999-03-11+ pre-inception days + A_div spread + B + C1)  -- imports QQQ's noise
        ALL1_div50 (A_div + a SMOOTH lev x 0.50%/yr dividend credit on all pre-inception days + B + C1; 0.50% sits
                    between the measured 2002-05 (0.42%) and 2006-09 (0.61%) NDX yields; no daily noise imported)
        ALL1_divYQ (A_div + SMOOTH lev x 0.10%/yr credit on 1999-03-11+ pre-inception days = the measured 1999-2005 mean)
usage: python -u update_2026_11/r2_verify_model_hygiene_robustness_divq.py <scratch_dir>
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = os.path.abspath(sys.argv[1])
P = C.panel()
IDX = P.index
START, FREEZE = C.START, C.FREEZE
S = json.load(open(os.path.join(HERE, "r2_model_hygiene.json")))
OUT = {}

# (1) QQQ vs NDX daily noise by period
d = (P["qqq_real"] + 0.0020 / 252 - P["ndx_ret"]).dropna()
for a, b in (("1999-03-11", "2004-11-30"), ("2004-12-01", "2008-12-31"), ("2009-01-01", str(FREEZE.date()))):
    x = d[(d.index >= a) & (d.index <= b)]
    ac = float(x.autocorr(1))
    OUT[f"qqq_minus_ndx_{a}_{b}"] = dict(te_ann=float(x.std() * np.sqrt(252)), lag1_autocorr=ac, mean_ann=float(x.mean() * 252))
    print(f"QQQ_real+fee - NDX price, {a}..{b}: daily TE {x.std()*np.sqrt(252)*100:.2f}%/yr, lag-1 autocorr {ac:+.2f}, mean {x.mean()*252*100:+.2f}%/yr")

# (2) panels
FIN = S["financing"]
interp = P["gold_src"].eq("monthly-interp")


def syn_B():
    out = {}
    for name, ycol, T in (("ief", "tnx", 7.5), ("tlt", "tyx", 20.0)):
        y = P[ycol] / 100.0
        dy = y.diff()
        y1 = y.shift(1)
        n = int(round(2 * T))
        k = np.arange(1, n + 1)[None, :]
        yy = y1.fillna(0.05).values
        i = (yy / 2)[:, None]
        cf = np.where(k == n, i + 1.0, i)
        disc = (1 + i) ** (-k)
        pr = (cf * disc).sum(1)
        D = (cf * disc * (k / 2.0)).sum(1) / pr / (1 + i[:, 0])
        Cx = (cf * disc * k * (k + 1) / 4.0 / (1 + i) ** 2).sum(1) / pr
        D = pd.Series(np.where(y1.notna(), D, np.nan), index=IDX)
        Cx = pd.Series(np.where(y1.notna(), Cx, np.nan), index=IDX)
        real = P[f"{name}_real"]
        m = (IDX >= pd.Timestamp("2002-07-31")) & (IDX <= FREEZE)
        a = float((real - y1 / 252 + D * dy - 0.5 * Cx * dy ** 2)[m].dropna().mean())
        out[name] = real.fillna(a + y1 / 252 - D * dy + 0.5 * Cx * dy ** 2).fillna(0)
    return out


def gold_step():
    gross = (P["gold_phys"] + 0.0040 / 252).where(interp, 0.0)
    gross.iloc[0] = 0.0
    gi = gross[interp]
    mon = gi.groupby(gi.index.to_period("M"))
    step = pd.Series(0.0, index=IDX)
    step.loc[pd.DatetimeIndex(mon.apply(lambda s: s.index.max()).values)] = mon.apply(lambda s: float(np.prod(1 + s.values) - 1)).values
    g = (step - 0.0040 / 252).where(interp, P["gold_phys"])
    g.iloc[0] = P["gold_phys"].iloc[0]
    return g


SB = syn_B()
G1 = gold_step()


def lev_series(f, lev, mode):
    p = FIN[f]["divneutral"]["pooled"]
    adj = p["alpha_ann"] / 252 + p["beta"] * P["rf_d"].fillna(0)
    sim = P[f"{f}_sim"]
    if mode == "A_div":
        base = sim
    elif mode == "divQ":
        base = (lev * (P["qqq_real"] + 0.0020 / 252) - 0.0095 / 252 - (lev - 1) * P["rf_d"]).fillna(sim)
    elif mode == "div50":
        base = sim + lev * 0.0050 / 252
    elif mode == "divYQ":
        base = sim + np.where(P["qqq_real"].notna(), lev * 0.0010 / 252, 0.0)
    return P[f"{f}_real"].fillna(base + adj).fillna(0)


def make(mode):
    x = P.copy()
    x["qld"] = lev_series("qld", 2, mode)
    x["tqqq"] = lev_series("tqqq", 3, mode)
    x["ief"], x["tlt"], x["gold_phys"] = SB["ief"], SB["tlt"], G1
    return x


DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MK = {dd: pd.Series(IDX.dayofweek == i, index=IDX) for i, dd in enumerate(DAYS)}
H1 = ("1988-01-01", "2006-12-31")


def ep(r, a, b):
    r = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    e = (1 + r).cumprod()
    dd = e / e.cummax() - 1
    tr = dd.idxmin()
    pk = e[:tr].idxmax()
    return f"{pk.date()}..{tr.date()} {dd.min()*100:.1f}%"


for mode in ("A_div", "divQ", "div50", "divYQ"):
    PP = make(mode)
    OUT[mode] = {}
    for tn, kw in (("F2", dict(lev="qld", voltgt=0.30)), ("F3", dict(lev="tqqq", voltgt=0.20))):
        wi = C.strategy_weights(PP, **kw)
        wt = C.strategy_weights(PP, **dict(kw, off="tlt"))
        for basis in ("Fri", "tranche"):
            if basis == "Fri":
                ri, rt = C.backtest(PP, wi), C.backtest(PP, wt)
            else:
                ri = sum(C.backtest(PP, wi, mark=MK[x]) for x in DAYS) / 5
                rt = sum(C.backtest(PP, wt, mark=MK[x]) for x in DAYS) / 5
            si, stt = C.stats(ri, *H1), C.stats(rt, *H1)
            isi = C.stats(ri, b=FREEZE)
            row = dict(H1_cal_ief=si["calmar"], H1_cal_tlt=stt["calmar"], H1_diff=si["calmar"] - stt["calmar"],
                       H1_dd_ief=ep(ri, *H1), H1_dd_tlt=ep(rt, *H1), IS_cagr=isi["cagr"], IS_maxdd=isi["maxdd"],
                       IS_calmar=isi["calmar"], IS_dd_episode=ep(ri, "1988-01-01", str(FREEZE.date())))
            OUT[mode][f"{tn}|{basis}"] = row
            print(f"ALL1[{mode:5s}] {tn} {basis:7s} IS {isi['cagr']*100:6.2f}% / {isi['maxdd']*100:6.1f}% / {isi['calmar']:.3f} (IS maxDD {row['IS_dd_episode']}) | "
                  f"R6 H1 IEF-TLT {row['H1_diff']:+.3f}  [IEF H1 Cal {si['calmar']:.3f} DD {row['H1_dd_ief']}; TLT H1 Cal {stt['calmar']:.3f} DD {row['H1_dd_tlt']}]")
json.dump(OUT, open(os.path.join(SCR, "rob_divq.json"), "w"), indent=1, default=float)
