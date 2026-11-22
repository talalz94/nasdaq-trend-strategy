"""
r2_verify_candidates_tranche_robustness_checks.py -- adversarial ROBUSTNESS review of r2_candidates_tranche.

Read-only w.r.t. project files: reads panel.parquet, common.py, r2_candidates_tranche.json, oos_live_yf.parquet;
writes only the scratch JSON given on the command line.

  A  independent numpy engine (own weights from panel signals, own weekday books, own per-sleeve costs)
     vs the specialist's saved metrics
  B  drawdown-episode dating (which episode sets each candidate's maxDD; simulated-data dependence)
  C  start-year sensitivity (every Jan-1 start 1988..2016 -> 2026-07-24) and NON-nested sub-periods,
     compared with the pre-registered planning ranges (nested 1988/2000/2007 starts)
  D  rolling 10-year CAGR / maxDD distribution (planning horizon dispersion)
  E  single-episode dependence of the Calmar edge vs QQQ (drop dot-com / 2022 / 1989-90 days from both series)
  F  basic (reverse-percentile) vs percentile bootstrap CIs, from the specialist's own saved bootstrap output
  H  off-leg cost of 50/50 vs IEF: IEF-minus-bills by sub-period, average off-leg weight, current term spread
  I  out-of-sample weekday result: where does the Friday-book OOS advantage come from (week decomposition)
  J  multiple-testing (Holm) on the C7/C8 Calmar-edge P values
  K  60/40 benchmark with SPY (adjusted close) instead of price-only SPX, 2002-07..2026-07-24

usage: python -u update_2026_11/r2_verify_candidates_tranche_robustness_checks.py <scratch_out.json>
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OUT = os.path.abspath(sys.argv[1])
assert os.path.dirname(OUT) != HERE, "write the scratch output outside update_2026_11"
P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
SAVED = json.load(open(os.path.join(HERE, "r2_candidates_tranche.json")))
RES = {}

# ------------------------------------------------------------------ A. independent engine
SLEEVES = ["qqq", "qld", "tqqq", "ief", "bills", "gold_phys"]
RET = {c: P[c].fillna(0).values.astype(float) for c in SLEEVES}
RFD = P["rf_d"].fillna(0).values.astype(float)
VOL = P["vol20"].values
ER = P["er20"].values
ABOVE = P["above"].values.astype(bool)
MON = P["month"].values
DOW = IDX.dayofweek.values
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def my_weights(lev, vt, off):
    sc = np.where(np.isnan(VOL), 1.0, np.clip(vt / np.where(np.isnan(VOL), 1.0, VOL), 0, 1))
    sc = np.clip(sc * np.where((MON == 8) | (MON == 9), 0.5, 1.0), 0, 1)
    ok = np.where(np.isnan(ER), False, ER >= 0.25)
    eng = np.where(ABOVE, np.where(ok, sc, 0.5 * sc), 0.0)
    W = {lev: 0.85 * eng}
    rest = np.clip(0.85 - 0.85 * eng, 0, None)
    if off == "IEF":
        W["ief"] = rest
    else:
        W["ief"] = 0.5 * rest
        W["bills"] = 0.5 * rest
    W["gold_phys"] = np.full(N, 0.15)
    return W


def book(W, mask, lag, cost):
    # position index of the most recent mark day (inclusive); -1 before the first mark
    pos = np.where(mask, np.arange(N), -1)
    li = np.maximum.accumulate(pos)
    r = np.zeros(N)
    for c, w in W.items():
        h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
        h = np.r_[np.zeros(lag), h[:-lag]]
        dh = np.abs(np.r_[0.0, np.diff(h)])
        k = cost[c] if isinstance(cost, dict) else cost
        r += h * RET[c] - dh * k
    return r


def tranche(W, lag, cost):
    return np.mean([book(W, DOW == i, lag, cost) for i in range(5)], axis=0)


TIERS = {"F1": ("qqq", 0.20), "F2": ("qld", 0.30), "F3": ("tqqq", 0.20)}
SER = {}
WTS = {}
for t, (lev, vt) in TIERS.items():
    for off in ("IEF", "5050"):
        W = my_weights(lev, vt, off)
        WTS[f"{t}-{off}"] = W
        uc = {lev: 15e-4, "ief": 8e-4, "bills": 8e-4, "gold_phys": 8e-4}
        SER[(f"{t}-{off}", "fri")] = book(W, P["mark"].values.astype(bool), 1, 3e-4)
        SER[(f"{t}-{off}", "tranche")] = tranche(W, 1, 3e-4)
        SER[(f"{t}-{off}", "us")] = tranche(W, 1, 5e-4)
        SER[(f"{t}-{off}", "ucits")] = tranche(W, 2, uc)
QQQ = P["qqq"].fillna(0).values.astype(float)
B6040 = 0.6 * P["spx"].fillna(0).values + 0.4 * P["ief"].fillna(0).values
CRITIC_F1 = tranche(WTS["F1-IEF"], 2, {"qqq": 8e-4, "ief": 8e-4, "gold_phys": 8e-4})

# weights vs common.strategy_weights
wdev = 0.0
for t, (lev, vt) in TIERS.items():
    for off in ("IEF", "5050"):
        ow = None if off == "IEF" else pd.DataFrame({"ief": 0.5, "bills": 0.5}, index=IDX)
        wc = C.strategy_weights(P, lev=lev, voltgt=vt, off_weights=ow)
        for c in wc.columns:
            wdev = max(wdev, float(np.max(np.abs(wc[c].values - WTS[f"{t}-{off}"][c]))))


def mets(r, a, b):
    m = (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))
    x = r[m]
    d = IDX[m]
    e = np.cumprod(1 + x)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = e / np.maximum.accumulate(e) - 1
    ex = x - RFD[m]
    return dict(cagr=float(cagr), maxdd=float(dd.min()), calmar=float(cagr / abs(dd.min())),
                exsh=float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)), ulcer=float(np.sqrt(np.mean(dd ** 2))),
                total=float(e[-1] - 1), years=float(yrs))


WIN = {"IS88": ("1988-01-01", FREEZE), "IS00": ("2000-01-01", FREEZE), "IS07": ("2007-01-01", FREEZE)}
cmp = 0.0
nchk = 0
for (cfg, basis), r in SER.items():
    for w, (a, b) in WIN.items():
        mm = mets(r, a, b)
        sv = SAVED["candidates"][cfg][basis][w]
        for k in ("cagr", "maxdd", "calmar", "ulcer"):
            cmp = max(cmp, abs(mm[k] - sv[k]))
            nchk += 1
        cmp = max(cmp, abs(mm["exsh"] - sv["ex_sharpe"]))
    oo = float(np.prod(1 + r[IDX >= pd.Timestamp("2026-07-25")]) - 1)
    cmp = max(cmp, abs(oo - SAVED["candidates"][cfg][basis]["OOS"]["total"]))
RES["A_independent_engine"] = dict(weights_max_abs_dev_vs_common=wdev, n_metric_checks=nchk,
                                   max_abs_diff_vs_saved_metrics=cmp,
                                   note="saved JSON is rounded to 7 dp, so <=5e-8 = exact")
print(f"A. own engine vs saved: weights dev {wdev:.1e}, {nchk} metric checks, max |diff| {cmp:.1e}")

# ------------------------------------------------------------------ B. drawdown episodes
def episodes(r, a, b, k=4):
    m = (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))
    x = r[m]
    d = IDX[m]
    e = np.cumprod(1 + x)
    pk = np.maximum.accumulate(e)
    dd = e / pk - 1
    under = dd < 0
    out = []
    i = 0
    n = len(x)
    while i < n:
        if under[i]:
            j = i
            while j < n and under[j]:
                j += 1
            seg = dd[i:j]
            t = i + int(np.argmin(seg))
            out.append(dict(depth=float(seg.min()), peak=str(d[i - 1].date()) if i > 0 else str(d[i].date()),
                            trough=str(d[t].date()), recovered=str(d[j].date()) if j < n else "not by end",
                            days_to_trough=int(t - i + 1)))
            i = j
        else:
            i += 1
    out.sort(key=lambda z: z["depth"])
    return out[:k]


EP = {}
for cfg in ["F1-IEF", "F1-5050", "F2-IEF", "F2-5050", "F3-IEF", "F3-5050"]:
    for basis in ("us", "ucits"):
        EP[f"{cfg}|{basis}"] = episodes(SER[(cfg, basis)], *WIN["IS88"])
EP["QQQ"] = episodes(QQQ, *WIN["IS88"])
RES["B_drawdown_episodes_IS88_top4"] = EP
print("\nB. worst IS88 drawdown episodes (depth, peak -> trough -> recovered)")
for k, v in EP.items():
    print(f"  {k:16s} " + " | ".join(f"{e['depth']*100:6.1f} {e['peak']}->{e['trough']}" for e in v[:3]))

# ------------------------------------------------------------------ C. start-year and non-nested sub-periods
PLAN = {(p["tier"], p["route"], p["off_leg"]): p for p in SAVED["planning_table"]}
grid = {}
print("\nC. start-year grid (Jan-1 start .. 2026-07-24), US/UCITS IEF and 50/50: min..max CAGR / Calmar vs plan range")
for t in TIERS:
    for route, basis in (("US", "us"), ("UCITS", "ucits")):
        for off in ("IEF", "5050"):
            r = SER[(f"{t}-{off}", basis)]
            rows = {y: mets(r, f"{y}-01-01", FREEZE) for y in range(1988, 2017)}
            cg = {y: v["cagr"] for y, v in rows.items()}
            ca = {y: v["calmar"] for y, v in rows.items()}
            pl = PLAN[(t, route, off)]
            out_c = [y for y in cg if cg[y] < pl["cagr_range"][0] - 1e-12 or cg[y] > pl["cagr_range"][1] + 1e-12]
            out_k = [y for y in ca if ca[y] < pl["calmar_range"][0] - 1e-12 or ca[y] > pl["calmar_range"][1] + 1e-12]
            grid[f"{t}|{route}|{off}"] = dict(
                cagr_min=[min(cg.values()), min(cg, key=cg.get)], cagr_max=[max(cg.values()), max(cg, key=cg.get)],
                calmar_min=[min(ca.values()), min(ca, key=ca.get)], calmar_max=[max(ca.values()), max(ca, key=ca.get)],
                plan_cagr=pl["cagr_range"], plan_calmar=pl["calmar_range"],
                n_starts=len(cg), starts_outside_plan_cagr=out_c, starts_outside_plan_calmar=out_k,
                cagr_by_start={str(y): v for y, v in cg.items()}, calmar_by_start={str(y): v for y, v in ca.items()})
            g = grid[f"{t}|{route}|{off}"]
            print(f"  {t} {route:5s} {off:4s} CAGR {g['cagr_min'][0]*100:5.2f}({g['cagr_min'][1]})..{g['cagr_max'][0]*100:5.2f}({g['cagr_max'][1]})"
                  f" plan {pl['cagr_range'][0]*100:5.2f}..{pl['cagr_range'][1]*100:5.2f} out {len(out_c)}/29 | Calmar "
                  f"{g['calmar_min'][0]:.2f}({g['calmar_min'][1]})..{g['calmar_max'][0]:.2f}({g['calmar_max'][1]}) plan "
                  f"{pl['calmar_range'][0]:.2f}..{pl['calmar_range'][1]:.2f} out {len(out_k)}/29")
qg = {y: mets(QQQ, f"{y}-01-01", FREEZE) for y in range(1988, 2017)}
grid["QQQ"] = dict(cagr_by_start={str(y): v["cagr"] for y, v in qg.items()}, calmar_by_start={str(y): v["calmar"] for y, v in qg.items()})
RES["C_start_year_grid"] = grid

SUB = {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-2009": ("2000-01-01", "2009-12-31"),
       "2010-2026.07": ("2010-01-01", FREEZE), "1988-2006": ("1988-01-01", "2006-12-31"),
       "2000-2006": ("2000-01-01", "2006-12-31")}
subp = {}
print("\nC2. NON-nested sub-periods (CAGR / maxDD / Calmar / exSharpe)")
for name, r in [(f"{t}-{off}|{b}", SER[(f"{t}-{off}", b)]) for t in TIERS for off in ("IEF", "5050") for b in ("us", "ucits")] + \
               [("QQQ", QQQ), ("60/40 price", B6040)]:
    subp[name] = {s: mets(r, a, b) for s, (a, b) in SUB.items()}
    print(f"  {name:16s} " + " | ".join(f"{s} {v['cagr']*100:6.2f}/{v['maxdd']*100:6.1f}/{v['calmar']:5.2f}/{v['exsh']:5.2f}"
                                        for s, v in subp[name].items() if s in ("1988-1999", "2000-2009", "2010-2026.07")))
RES["C2_subperiods"] = subp

# ------------------------------------------------------------------ D. rolling 10-year windows
def rolling(r, yrs=10, step=21):
    m = (IDX >= START) & (IDX <= FREEZE)
    x = r[m]
    d = IDX[m]
    lx = np.r_[0.0, np.cumsum(np.log1p(x))]
    out_c, out_dd, starts = [], [], []
    for i in range(0, len(x), step):
        end_date = d[i] + pd.DateOffset(years=yrs)
        j = int(np.searchsorted(d.values, end_date.to_datetime64(), side="right"))
        if j > len(x) or end_date > d[-1]:
            break
        seg = x[i:j]
        e = np.cumprod(1 + seg)
        out_c.append(float(np.exp((lx[j] - lx[i]) / ((d[j - 1] - d[i]).days / 365.25)) - 1))
        out_dd.append(float((e / np.maximum.accumulate(e) - 1).min()))
        starts.append(str(d[i].date()))
    c, dd = np.array(out_c), np.array(out_dd)
    return dict(n=len(c), cagr_pct={p: float(np.percentile(c, p)) for p in (0, 5, 25, 50, 75, 95, 100)},
                maxdd_pct={p: float(np.percentile(dd, p)) for p in (0, 5, 50, 95)},
                worst_start=starts[int(np.argmin(c))], share_cagr_below_plan_min=None)


roll = {}
print("\nD. rolling 10-year windows (monthly steps, 1988..2026-07): CAGR pct 0/5/50/95 | maxDD pct 0/5/50")
for name, r, key in [("F1-IEF|us", SER[("F1-IEF", "us")], ("F1", "US", "IEF")),
                     ("F2-IEF|us", SER[("F2-IEF", "us")], ("F2", "US", "IEF")),
                     ("F3-IEF|us", SER[("F3-IEF", "us")], ("F3", "US", "IEF")),
                     ("F2-IEF|ucits", SER[("F2-IEF", "ucits")], ("F2", "UCITS", "IEF")),
                     ("QQQ", QQQ, None)]:
    rr = rolling(r)
    if key:
        m = (IDX >= START) & (IDX <= FREEZE)
        # recompute share below the plan's CAGR minimum
        x = r[m]
        d = IDX[m]
        lx = np.r_[0.0, np.cumsum(np.log1p(x))]
        cs = []
        for i in range(0, len(x), 21):
            end_date = d[i] + pd.DateOffset(years=10)
            if end_date > d[-1]:
                break
            j = int(np.searchsorted(d.values, end_date.to_datetime64(), side="right"))
            cs.append(np.exp((lx[j] - lx[i]) / ((d[j - 1] - d[i]).days / 365.25)) - 1)
        cs = np.array(cs)
        rr["share_cagr_below_plan_min"] = float((cs < PLAN[key]["cagr_range"][0]).mean())
        rr["share_cagr_above_plan_max"] = float((cs > PLAN[key]["cagr_range"][1]).mean())
        rr["plan_cagr_range"] = PLAN[key]["cagr_range"]
    roll[name] = rr
    cp, dp = rr["cagr_pct"], rr["maxdd_pct"]
    extra = (f" | share of 10y windows below plan min {rr['share_cagr_below_plan_min']*100:.0f}%, above plan max "
             f"{rr['share_cagr_above_plan_max']*100:.0f}%") if key else ""
    print(f"  {name:14s} n={rr['n']} CAGR {cp[0]*100:5.1f}/{cp[5]*100:5.1f}/{cp[50]*100:5.1f}/{cp[95]*100:5.1f}"
          f" (worst start {rr['worst_start']}) | maxDD {dp[0]*100:6.1f}/{dp[5]*100:6.1f}/{dp[50]*100:6.1f}{extra}")
RES["D_rolling_10y"] = roll

# ------------------------------------------------------------------ E. single-episode dependence of the Calmar edge
EPIS = {"dot-com 2000-03-10..2002-10-09": ("2000-03-10", "2002-10-09"),
        "2022 (2022-01-01..2022-12-31)": ("2022-01-01", "2022-12-31"),
        "1989-90 (1989-10-01..1990-10-31)": ("1989-10-01", "1990-10-31"),
        "GFC 2007-10-31..2009-03-09": ("2007-10-31", "2009-03-09")}


def mets_drop(r, a, b, drop):
    m = (IDX >= max(pd.Timestamp(a), START)) & (IDX <= pd.Timestamp(b))
    if drop is not None:
        m &= ~((IDX >= pd.Timestamp(drop[0])) & (IDX <= pd.Timestamp(drop[1])))
    x = r[m]
    e = np.cumprod(1 + x)
    yrs = len(x) / 252.0
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return float(cagr), float(dd), float(cagr / abs(dd))


edge = {}
print("\nE. Calmar edge vs QQQ (IS88, 252-day years) with one episode's days removed from BOTH series")
for drop_name, drop in [("none", None)] + list(EPIS.items()):
    q = mets_drop(QQQ, *WIN["IS88"], drop)
    row = {"QQQ": q}
    for cfg in ("F1-IEF", "F2-IEF", "F3-IEF", "F2-5050"):
        for basis in ("us", "ucits"):
            s = mets_drop(SER[(cfg, basis)], *WIN["IS88"], drop)
            row[f"{cfg}|{basis}"] = dict(cagr=s[0], maxdd=s[1], calmar=s[2], calmar_edge=s[2] - q[2])
    edge[drop_name] = row
    print(f"  drop {drop_name:34s} QQQ {q[0]*100:5.2f}/{q[1]*100:6.1f}/{q[2]:.2f} | " +
          " ".join(f"{k.split('|')[0]}{'U' if k.endswith('us') else 'E'} {v['calmar']:.2f}({v['calmar_edge']:+.2f})"
                   for k, v in row.items() if k != "QQQ"))
RES["E_episode_drop_calmar_edge"] = edge

# ------------------------------------------------------------------ F. basic vs percentile bootstrap CIs (saved output)
flip = {}
print("\nF. percentile CI (pre-registered) vs basic CI = [2*sample - hi, 2*sample - lo] for Calmar differences")
for w in ("IS88", "IS00", "IS07"):
    for L in ("L126", "L252"):
        pr = SAVED["bootstrap"][w][L]["pairs"]
        for k, e in pr.items():
            if "|fri" in k:
                continue
            for met in ("calmar", "exsh", "cagr"):
                s, (lo, hi) = e[met]["sample"], e[met]["ci95"]
                blo, bhi = 2 * s - hi, 2 * s - lo
                p_ex = lo > 0 or hi < 0
                b_ex = blo > 0 or bhi < 0
                flip[f"{w}|{L}|{k}|{met}"] = dict(sample=s, pct_ci=[lo, hi], basic_ci=[blo, bhi], pct_excl0=p_ex,
                                                  basic_excl0=b_ex, p_le0=e[met]["p_le0"], boot_mean=e[met]["boot_mean"])
RES["F_basic_vs_percentile"] = flip
keys_show = ["F1-IEF|us minus QQQ B&H", "F2-IEF|us minus QQQ B&H", "F3-IEF|us minus QQQ B&H",
             "F1-IEF|ucits minus QQQ B&H", "F2-IEF|ucits minus QQQ B&H", "F3-IEF|ucits minus QQQ B&H",
             "F1-5050|ucits minus QQQ B&H", "F2-5050|ucits minus QQQ B&H", "F3-5050|ucits minus QQQ B&H",
             "critic:F1-IEF lag2 8/8bp [tranche] minus QQQ B&H",
             "F2-IEF|us minus F1-IEF|us", "F2-IEF|us minus F3-IEF|us", "F2-IEF|ucits minus F1-IEF|ucits"]
for w in ("IS88", "IS07"):
    for L in ("L126", "L252"):
        print(f"  {w} {L}")
        for k in keys_show:
            f_ = flip[f"{w}|{L}|{k}|calmar"]
            print(f"    {k:52s} sample {f_['sample']:+.3f} pct [{f_['pct_ci'][0]:+.3f},{f_['pct_ci'][1]:+.3f}]"
                  f"{'*' if f_['pct_excl0'] else ' '} basic [{f_['basic_ci'][0]:+.3f},{f_['basic_ci'][1]:+.3f}]"
                  f"{'*' if f_['basic_excl0'] else ' '}")
cnt = {}
for k, v in flip.items():
    w, L, pair, met = k.split("|", 3)[0], k.split("|", 3)[1], "|".join(k.split("|")[2:-1]), k.split("|")[-1]
    if met != "calmar":
        continue
    fam = "vsQQQ" if "minus QQQ" in k else "other"
    key = f"{w}|{L}|{fam}"
    c_ = cnt.setdefault(key, dict(n=0, pct_excl=0, basic_excl=0, flips=0))
    c_["n"] += 1
    c_["pct_excl"] += v["pct_excl0"]
    c_["basic_excl"] += v["basic_excl0"]
    c_["flips"] += v["pct_excl0"] != v["basic_excl0"]
RES["F_counts_calmar"] = cnt
print("  Calmar verdict counts (CI excludes 0): " + "; ".join(f"{k} pct {v['pct_excl']}/{v['n']} basic {v['basic_excl']}/{v['n']} flips {v['flips']}"
                                                         for k, v in cnt.items()))

# ------------------------------------------------------------------ H. off-leg: IEF vs bills by sub-period
SUBH = {"1988-1999": ("1988-01-01", "1999-12-31"), "2000-2009": ("2000-01-01", "2009-12-31"),
        "2010-2019": ("2010-01-01", "2019-12-31"), "2020-2026.07": ("2020-01-01", FREEZE),
        "IS88": ("1988-01-01", FREEZE), "IS07": ("2007-01-01", FREEZE)}
offleg = {}
print("\nH. IEF vs bills and the 50/50 drag (US basis), by sub-period")
for s, (a, b) in SUBH.items():
    m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    yrs = (IDX[m][-1] - IDX[m][0]).days / 365.25
    ief = float(np.prod(1 + RET["ief"][m]) ** (1 / yrs) - 1)
    bil = float(np.prod(1 + RET["bills"][m]) ** (1 / yrs) - 1)
    row = dict(ief_cagr=ief, bills_cagr=bil, ief_minus_bills=ief - bil,
               tnx_change_pp=float(P["tnx"][m].dropna().iloc[-1] - P["tnx"][m].dropna().iloc[0]),
               mean_curve_tnx_minus_irx=float((P["tnx"] - P["irx"])[m].mean()))
    for t in TIERS:
        # average held off-leg weight in the tranche (mean of the 5 books' held weights)
        W = WTS[f"{t}-IEF"]
        offw = []
        for i in range(5):
            pos = np.where(DOW == i, np.arange(N), -1)
            li = np.maximum.accumulate(pos)
            h = np.where(li >= 0, W["ief"][np.maximum(li, 0)], 0.0)
            offw.append(np.r_[0.0, h[:-1]])
        offw = np.mean(offw, axis=0)
        c5 = mets(SER[(f"{t}-5050", "us")], a, b)["cagr"]
        ci = mets(SER[(f"{t}-IEF", "us")], a, b)["cagr"]
        row[t] = dict(mean_offleg_weight=float(offw[m].mean()), cagr_5050_minus_ief=c5 - ci,
                      naive_drag=float(-0.5 * offw[m].mean() * (ief - bil)))
    offleg[s] = row
    print(f"  {s:13s} IEF {ief*100:5.2f} bills {bil*100:5.2f} diff {(ief-bil)*100:+5.2f}pp (10y chg {row['tnx_change_pp']:+.2f}pp,"
          f" mean curve {row['mean_curve_tnx_minus_irx']:+.2f}) | 50/50-IEF CAGR F1 {row['F1']['cagr_5050_minus_ief']*100:+.2f}"
          f" F2 {row['F2']['cagr_5050_minus_ief']*100:+.2f} F3 {row['F3']['cagr_5050_minus_ief']*100:+.2f}pp; mean off-leg w "
          f"F1 {row['F1']['mean_offleg_weight']:.2f} F2 {row['F2']['mean_offleg_weight']:.2f} F3 {row['F3']['mean_offleg_weight']:.2f}")
cur = dict(freeze=dict(tnx=float(P.loc[:FREEZE, "tnx"].dropna().iloc[-1]), irx=float(P.loc[:FREEZE, "irx"].dropna().iloc[-1])),
           latest=dict(tnx=float(P["tnx"].dropna().iloc[-1]), irx=float(P["irx"].dropna().iloc[-1])))
offleg["current_yields"] = cur
print(f"  yields at freeze: 10y {cur['freeze']['tnx']:.2f} 3m {cur['freeze']['irx']:.2f}; latest 10y {cur['latest']['tnx']:.2f} 3m {cur['latest']['irx']:.2f}")
RES["H_offleg"] = offleg

# ------------------------------------------------------------------ I. OOS weekday decomposition
oos = IDX >= pd.Timestamp("2026-07-25")
W = WTS["F2-IEF"]
books = {d: book(W, DOW == i, 1, 5e-4) for i, d in enumerate(DAYS)}
tr = np.mean([books[d] for d in DAYS], axis=0)
diff = books["Fri"] - tr
wk = pd.Series(diff[oos], index=IDX[oos]).groupby(IDX[oos].to_period("W-FRI")).sum()
tot = float(np.prod(1 + books["Fri"][oos]) - np.prod(1 + tr[oos]))
heldF = {}
for i, d in enumerate(DAYS):
    pos = np.where(DOW == i, np.arange(N), -1)
    li = np.maximum.accumulate(pos)
    h = np.where(li >= 0, W["qld"][np.maximum(li, 0)], 0.0)
    heldF[d] = np.r_[0.0, h[:-1]]
hq = pd.DataFrame({d: heldF[d][oos] for d in DAYS}, index=IDX[oos])
sig = pd.DataFrame({"above": ABOVE[oos], "qld_target": W["qld"][oos], "ndx": P["ndx"].values[oos],
                    "ema100": P["ema100"].values[oos]}, index=IDX[oos])
top = wk.abs().sort_values(ascending=False)
RES["I_oos_weekday"] = dict(
    F2_us_book_totals={d: float(np.prod(1 + books[d][oos]) - 1) for d in DAYS},
    tranche_total=float(np.prod(1 + tr[oos]) - 1),
    fri_minus_tranche_sum_daily=float(diff[oos].sum()),
    weekly_fri_minus_tranche={str(k): float(v) for k, v in wk.items()},
    top2_weeks_share_of_abs_sum=float(top.iloc[:2].sum() / wk.abs().sum()),
    risk_off_flips=[str(d.date()) for d, a, b_ in zip(sig.index[1:], sig["above"].values[:-1], sig["above"].values[1:]) if a != b_])
print("\nI. OOS F2-IEF US: book totals", {d: round(v * 100, 2) for d, v in RES["I_oos_weekday"]["F2_us_book_totals"].items()},
      "tranche", round(RES["I_oos_weekday"]["tranche_total"] * 100, 2))
print("   weekly (Fri book - tranche) daily-return sums, largest 4:",
      {str(k): round(float(wk[k]) * 100, 2) for k in top.index[:4]},
      f"; top-2 weeks = {RES['I_oos_weekday']['top2_weeks_share_of_abs_sum']*100:.0f}% of |sum|")
print("   EMA-100 regime flips in OOS (day above changes):", RES["I_oos_weekday"]["risk_off_flips"])
print("   held QLD weight by book around the flips:")
for dflip in RES["I_oos_weekday"]["risk_off_flips"]:
    j = hq.index.get_loc(pd.Timestamp(dflip))
    print(hq.iloc[max(0, j - 2):j + 6].round(3).to_string())

# ------------------------------------------------------------------ J. Holm on Calmar edges
def holm(ps, alpha):
    o = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for r_, i in enumerate(o):
        if ps[i] <= alpha / (m - r_):
            rej[i] = True
        else:
            break
    return rej


hj = {}
print("\nJ. Holm step-down on P(Calmar edge <= 0) treated as one-sided p-values (alpha 0.025 = two-sided 5%)")
for w in ("IS88", "IS00"):
    for L in ("L126", "L252"):
        pr = SAVED["bootstrap"][w][L]["pairs"]
        for fam_name, keys in {"US-IEF 3 tiers": [f"F{i}-IEF|us minus QQQ B&H" for i in (1, 2, 3)],
                               "US all 6": [f"F{i}-{o}|us minus QQQ B&H" for i in (1, 2, 3) for o in ("IEF", "5050")],
                               "US+UCITS 12": [f"F{i}-{o}|{b} minus QQQ B&H" for i in (1, 2, 3) for o in ("IEF", "5050") for b in ("us", "ucits")]}.items():
            ps = np.array([pr[k]["calmar"]["p_le0"] for k in keys])
            rj = holm(ps, 0.025)
            hj[f"{w}|{L}|{fam_name}"] = dict(p=ps.tolist(), raw_reject=int((ps <= 0.025).sum()), holm_reject=int(rj.sum()), n=len(ps))
            print(f"  {w} {L} {fam_name:15s} raw p<=0.025: {int((ps <= 0.025).sum())}/{len(ps)}  Holm: {int(rj.sum())}/{len(ps)}  min p {ps.min():.3f}")
RES["J_holm"] = hj

# ------------------------------------------------------------------ K. 60/40 with SPY adjusted close (2002-07 on)
try:
    Y = pd.read_parquet(os.path.join(HERE, "oos_live_yf.parquet"))
    spy = Y[Y.ticker == "SPY"].set_index("date")["close"]
    spy.index = pd.to_datetime(spy.index)
    spy_r = spy.pct_change().reindex(IDX)
    m = (IDX >= pd.Timestamp("2002-08-01")) & (IDX <= FREEZE) & spy_r.notna().values
    yrs = (IDX[m][-1] - IDX[m][0]).days / 365.25
    spx_c = float(np.prod(1 + P["spx"].fillna(0).values[m]) ** (1 / yrs) - 1)
    spy_c = float(np.prod(1 + spy_r.values[m]) ** (1 / yrs) - 1)
    a = 0.6 * P["spx"].fillna(0).values[m] + 0.4 * RET["ief"][m]
    bq = 0.6 * spy_r.values[m] + 0.4 * RET["ief"][m]

    def cm(x):
        e = np.cumprod(1 + x)
        c = e[-1] ** (1 / yrs) - 1
        d = (e / np.maximum.accumulate(e) - 1)
        return dict(cagr=float(c), maxdd=float(d.min()), calmar=float(c / abs(d.min())), ulcer=float(np.sqrt(np.mean(d ** 2))))
    RES["K_6040_spy"] = dict(window=[str(IDX[m][0].date()), str(IDX[m][-1].date())], spx_price_cagr=spx_c, spy_adj_cagr=spy_c,
                             spy_minus_spx=spy_c - spx_c, b6040_price=cm(a), b6040_spy=cm(bq))
    k_ = RES["K_6040_spy"]
    print(f"\nK. 2002-08..2026-07-24: SPX price CAGR {spx_c*100:.2f}% vs SPY adj {spy_c*100:.2f}% (+{(spy_c-spx_c)*100:.2f}pp);"
          f" 60/40 price {k_['b6040_price']['cagr']*100:.2f}%/{k_['b6040_price']['maxdd']*100:.1f}%/U {k_['b6040_price']['ulcer']*100:.1f}"
          f" vs SPY {k_['b6040_spy']['cagr']*100:.2f}%/{k_['b6040_spy']['maxdd']*100:.1f}%/U {k_['b6040_spy']['ulcer']*100:.1f}")
except Exception as ex:  # pragma: no cover
    RES["K_6040_spy"] = dict(error=str(ex))
    print("K. failed", ex)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


json.dump(clean(RES), open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")
