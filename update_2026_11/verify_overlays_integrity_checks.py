"""
INTEGRITY REVIEW of update_2026_11/overlays.py  --  step 2: independent checks.

  A. independent engine (NO common.py): signals re-derived from P.ndx, own weights/backtest/stats;
     baselines + key overlays vs the saved overlays.json numbers.
  B. causality: every RAW macro flag recomputed on data TRUNCATED at ~30 cut-off dates must equal
     the full-sample flag at the cut-off (incl. the un-inversion state machine and the ext ffill).
  C. extra-lag sensitivity (macro lag 0 / 1 / 2 trading days) for the key variants.
  D. pre-2000 gold leak: monthly-interpolated gold replaced by month-end-booked gold (no look-ahead).
  E. bootstrap / placebo seed stability for the dollar-shock significance claims.
  F. un-inversion rule interpretations (re-triggers allowed [specialist] / first cross only /
     crossing within 504d of spell START).
  G. today: d126 path, current holdings of the rate-shock->bills variants, flag continuity,
     VWEHX-vs-HYG flag agreement.
Writes nothing except <scratch>/verify_overlays_integrity_checks.json.
usage: python update_2026_11/verify_overlays_integrity_checks.py <scratch_dir>
"""
import os, sys, json, ast, time
import numpy as np
import pandas as pd

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 2:
    sys.exit("usage: python update_2026_11/verify_overlays_integrity_checks.py <scratch_dir>  (output never goes to update_2026_11/)")
scratch = sys.argv[1]
P = pd.read_parquet(os.path.join(HERE, "panel.parquet"))
EXT = pd.read_parquet(os.path.join(HERE, "overlays_ext.parquet"))
J = json.load(open(os.path.join(HERE, "overlays.json")))
idx = P.index
LAST = idx[-1]
START, FREEZE = pd.Timestamp("1988-01-01"), pd.Timestamp("2026-07-24")
H1 = (START, pd.Timestamp("2006-12-31")); H2 = (pd.Timestamp("2007-01-01"), FREEZE)
OOS = (pd.Timestamp("2026-07-25"), LAST)
OUT = {}


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ A. independent engine
ndx = P["ndx"]
ema = ndx.ewm(span=100, adjust=False).mean()
above = ndx > ema
vol = ndx.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
er = ndx.diff(20).abs() / ndx.diff().abs().rolling(20).sum().replace(0, np.nan)
mark = pd.Series(False, index=idx)
mark[ndx.resample("W-FRI").last().index.intersection(idx)] = True
sig_chk = dict(ema_maxabs=float((ema - P["ema100"]).abs().max()), above_mismatch=int((above != P["above"].astype(bool)).sum()),
               vol_maxabs=float((vol - P["vol20"]).abs().max()), er_maxabs=float((er - P["er20"]).abs().max()),
               mark_mismatch=int((mark != P["mark"].astype(bool)).sum()))
log("signals re-derived vs panel:", sig_chk)
OUT["signals_vs_panel"] = sig_chk
month = pd.Series(idx.month, index=idx)
BASES = {"F3": ("tqqq", 0.20), "F2": ("qld", 0.30)}
GOLD = P["gold_phys"].copy()


def weights(base, mult=None, veto=None, bills_flag=None, gold=None):
    lev, vt = BASES[base]
    sc = (vt / vol).clip(0, 1).fillna(1.0)
    sc = sc * np.where(month.isin([8, 9]), 0.5, 1.0)
    if mult is not None:
        sc = sc * mult
    ron = above.copy() if veto is None else (above & ~veto)
    clean = (er >= 0.25).fillna(False)
    eq = np.where(ron, np.where(clean, sc, 0.5 * sc), 0.0)
    w = pd.DataFrame(0.0, index=idx, columns=[lev, "ief", "bills", "gold"])
    w[lev] = 0.85 * eq
    rest = 0.85 - 0.85 * eq
    bf = np.zeros(len(idx)) if bills_flag is None else bills_flag.values.astype(float)
    w["ief"] = rest * (1 - bf)
    w["bills"] = rest * bf
    w["gold"] = 0.15
    return w


def backtest(w, gold=None, lag=1):
    g = GOLD if gold is None else gold
    held = w.where(mark).ffill().fillna(0).shift(lag).fillna(0)
    R = pd.DataFrame({c: (g if c == "gold" else P[c]) for c in w.columns})
    gross = (held * R.fillna(0)).sum(axis=1)
    return gross - held.diff().abs().sum(axis=1).fillna(0) * 3e-4, held


def stats(r, a, b):
    x = r[(r.index >= a) & (r.index <= b)]
    e = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = e.iloc[-1] ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    return dict(cagr=float(cagr), maxdd=float(dd), sharpe=float(x.mean() / x.std() * np.sqrt(252)), calmar=float(cagr / -dd))


def pret(r, a, b):
    x = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    return float((1 + x).prod() - 1)


def lagk(raw, k=1):
    return raw.reindex(idx).astype(float).shift(k).fillna(0.0).astype(bool)


# macro raw flags (independent construction)
X = EXT[EXT.index <= LAST].reindex(EXT[EXT.index <= LAST].index.union(idx)).ffill().reindex(idx)
inv = (P["tnx"] - P["irx"] < 0)
d126 = P["tnx"] - P["tnx"].shift(126)
dxy126 = P["dxy"] / P["dxy"].shift(126) - 1
r_hy = X["hyg"] / X["ief_px"]
vts = X["vix_y"] / X["vix3m"]
ief_tr = (1 + P["ief"]).cumprod()
r_vw = X["vwehx"] / ief_tr

# un-inversion state machine: extract the specialist's function verbatim (exec in isolation)
src = open(os.path.join(HERE, "overlays.py"), encoding="utf-8").read()
fn_node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "uninversion_window")
ns = {"np": np, "pd": pd}
exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "uninv", "exec"), ns)
uninversion_window = ns["uninversion_window"]

RAW = {"inv": inv, "uninv_contig": uninversion_window(inv, "contig")[0],
       "rs50": d126 > 0.5, "rs75": d126 > 0.75, "rs100": d126 > 1.0, "usd5": dxy126 > 0.05,
       "hy50": r_hy < r_hy.rolling(50).mean(), "vw50": r_vw < r_vw.rolling(50).mean(), "vts": vts > 1.0}

V = {(v["base"], v["name"]): v for v in J["variants"]}
KEY = [  # (base, json name, kind, raw key, param, window)
    ("F3", "dollar shock DXY +5%/126d: engine x0.75", "mult", "usd5", 0.75, (START, FREEZE)),
    ("F2", "dollar shock DXY +5%/126d: engine x0.75", "mult", "usd5", 0.75, (START, FREEZE)),
    ("F3", "rate shock 10y +50bp/126d: engine x0.5", "mult", "rs50", 0.5, (START, FREEZE)),
    ("F2", "rate shock 10y +75bp/126d: off-leg IEF->bills", "bills", "rs75", None, (START, FREEZE)),
    ("F2", "rate shock 10y +100bp/126d: off-leg IEF->bills", "bills", "rs100", None, (START, FREEZE)),
    ("F3", "curve: inversion -> risk-off (July raw rule)", "veto", "inv", None, (START, FREEZE)),
    ("F3", "curve: un-inversion window x0.5 (>=63d spell)", "mult", "uninv_contig", 0.5, (START, FREEZE)),
    ("F3", "credit HYG/IEF < 50d SMA: engine x0.5", "mult", "hy50", 0.5, None),
    ("F3", "credit VWEHX/IEF < 50d SMA: engine x0.5 (1988+)", "mult", "vw50", 0.5, (START, FREEZE)),
    ("F3", "VIX term structure: VIX>VIX3M -> engine x0.5", "mult", "vts", 0.5, None),
]


def variant(base, kind, rawkey, param, lag=1, gold=None, raw=None):
    f = lagk(RAW[rawkey] if raw is None else raw, lag)
    if kind == "mult":
        w = weights(base, mult=pd.Series(np.where(f, param, 1.0), index=idx))
    elif kind == "veto":
        w = weights(base, veto=f)
    else:
        w = weights(base, bills_flag=f)
    return backtest(w, gold=gold)


BASE_R = {b: backtest(weights(b))[0] for b in BASES}
indep = {}
for b in BASES:
    s = stats(BASE_R[b], START, FREEZE)
    js = J["baselines"][b]["IS"]
    indep[f"base_{b}"] = dict(mine=s, json=dict(cagr=js["cagr"], maxdd=js["maxdd"], calmar=js["calmar"], sharpe=js["sharpe"]),
                              oos_mine=pret(BASE_R[b], *OOS), oos_json=J["baselines"][b]["OOS_return"])
    log(f"BASE {b}: mine CAGR {s['cagr']*100:.3f}% DD {s['maxdd']*100:.2f}% Cal {s['calmar']:.4f} Sh {s['sharpe']:.4f} | json "
        f"{js['cagr']*100:.3f}% {js['maxdd']*100:.2f}% {js['calmar']:.4f} {js['sharpe']:.4f} | OOS mine {indep[f'base_{b}']['oos_mine']*100:+.2f}% json {indep[f'base_{b}']['oos_json']*100:+.2f}%")
for base, name, kind, rk, prm, win in KEY:
    v = V[(base, name)]
    win = win or (pd.Timestamp(v["window"][0]), pd.Timestamp(v["window"][1]))
    r, _ = variant(base, kind, rk, prm)
    so, sb = stats(r, *win), stats(BASE_R[base], *win)
    h = [stats(r, *hh)["calmar"] - stats(BASE_R[base], *hh)["calmar"] for hh in ([H1, H2] if win[0] <= START else
         [(win[0], (win[0] + (win[1] - win[0]) / 2).normalize()), ((win[0] + (win[1] - win[0]) / 2).normalize() + pd.Timedelta(days=1), win[1])])]
    rec = dict(mine=dict(cagr=so["cagr"], maxdd=so["maxdd"], calmar=so["calmar"], d_calmar=so["calmar"] - sb["calmar"],
                         d_sharpe=so["sharpe"] - sb["sharpe"], h_dcal=h, d2022=pret(r, "2022-01-01", "2022-12-31") - pret(BASE_R[base], "2022-01-01", "2022-12-31"),
                         d_oos=pret(r, *OOS) - pret(BASE_R[base], *OOS)),
               json=dict(cagr=v["ov"]["cagr"], maxdd=v["ov"]["maxdd"], calmar=v["ov"]["calmar"], d_calmar=v["d_calmar"], d_sharpe=v["d_sharpe"],
                         h_dcal=[v["halves"][0]["d_calmar"], v["halves"][1]["d_calmar"]], d2022=v["crises"]["diff"]["2022 inflation"],
                         d_oos=v["oos"]["ov"] - v["oos"]["base"]))
    indep[f"{base} {name}"] = rec
    m, j = rec["mine"], rec["json"]
    log(f"{base} {name[:48]:48s} mine Cal {m['calmar']:.4f} dCal {m['d_calmar']:+.4f} dSh {m['d_sharpe']:+.4f} H {m['h_dcal'][0]:+.3f}/{m['h_dcal'][1]:+.3f} "
        f"| json {j['calmar']:.4f} {j['d_calmar']:+.4f} {j['d_sharpe']:+.4f} H {j['h_dcal'][0]:+.3f}/{j['h_dcal'][1]:+.3f}")
OUT["independent_engine"] = indep

# ------------------------------------------------------------------ B. causality (truncation test)
rng = np.random.default_rng(7)
cut_ix = sorted(set(rng.integers(400, len(idx) - 1, 24).tolist()))
special = ["2001-01-16", "2001-01-22", "2007-05-16", "2007-08-09", "2008-08-05", "2008-08-07", "2019-09-17", "2024-11-13",
           "2025-05-02", "2026-04-30", "2026-05-01", "2026-05-04", "2026-10-14", "2026-11-13"]
cuts = sorted(set([idx[i] for i in cut_ix] + [idx[idx.get_indexer([pd.Timestamp(s)], method="bfill")[0]] for s in special]))
mism = {}
for c in cuts:
    Pc = P[P.index <= c]
    Ec = EXT[EXT.index <= c]
    ic = Pc.index
    Xc = Ec.reindex(Ec.index.union(ic)).ffill().reindex(ic)
    invc = (Pc["tnx"] - Pc["irx"] < 0)
    dc = Pc["tnx"] - Pc["tnx"].shift(126)
    rhy = Xc["hyg"] / Xc["ief_px"]
    rvw = Xc["vwehx"] / (1 + Pc["ief"]).cumprod()
    v = Xc["vix_y"] / Xc["vix3m"]
    rawc = {"inv": invc, "uninv_contig": uninversion_window(invc, "contig")[0],
            "rs50": dc > 0.5, "rs75": dc > 0.75, "rs100": dc > 1.0, "usd5": (Pc["dxy"] / Pc["dxy"].shift(126) - 1) > 0.05,
            "hy50": rhy < rhy.rolling(50).mean(), "vw50": rvw < rvw.rolling(50).mean(), "vts": v > 1.0}
    for k in rawc:
        if bool(rawc[k].iloc[-1]) != bool(RAW[k].reindex(idx).fillna(False).loc[c]):
            mism.setdefault(k, []).append(str(c.date()))
OUT["causality_truncation"] = dict(n_cutoffs=len(cuts), mismatches=mism)
log(f"CAUSALITY: {len(cuts)} cut-offs x {len(RAW)} flags -> mismatches: {mism if mism else 'none'}")

# ------------------------------------------------------------------ C. extra-lag sensitivity
lagsens = {}
for base, name, kind, rk, prm, win in KEY[:7]:
    row = {}
    for L in (0, 1, 2):
        r, _ = variant(base, kind, rk, prm, lag=L)
        so, sb = stats(r, START, FREEZE), stats(BASE_R[base], START, FREEZE)
        row[f"lag{L}"] = dict(d_calmar=so["calmar"] - sb["calmar"], d_sharpe=so["sharpe"] - sb["sharpe"],
                              h1=stats(r, *H1)["calmar"] - stats(BASE_R[base], *H1)["calmar"],
                              h2=stats(r, *H2)["calmar"] - stats(BASE_R[base], *H2)["calmar"])
    lagsens[f"{base} {name}"] = row
    log(f"LAG {base} {name[:46]:46s} " + "  ".join(f"L{L}: dCal {row[f'lag{L}']['d_calmar']:+.3f} dSh {row[f'lag{L}']['d_sharpe']:+.3f} "
                                                  f"H {row[f'lag{L}']['h1']:+.3f}/{row[f'lag{L}']['h2']:+.3f}" for L in (0, 1, 2)))
OUT["extra_lag_sensitivity"] = lagsens

# ------------------------------------------------------------------ D. pre-2000 gold leak
gi = P["gold_src"] == "monthly-interp"
g_mi = P["gold_phys"].where(gi)
ym = pd.Series(idx.to_period("M"), index=idx)
mret = (1 + g_mi).groupby(ym).prod() - 1                         # compounded month return of the interpolated days
last_day = g_mi.dropna().groupby(ym[gi]).apply(lambda s: s.index[-1])
gold_nl = P["gold_phys"].copy()
gold_nl[gi] = 0.0
for per, d in last_day.items():
    gold_nl.loc[d] = mret.loc[per]
chk_total = float((1 + P["gold_phys"][gi]).prod() / (1 + gold_nl[gi]).prod() - 1)
gl = dict(interp_days=int(gi.sum()), last_interp_day=str(idx[gi][-1].date()), total_return_mismatch=chk_total)
BASE_NL = {b: backtest(weights(b), gold=gold_nl)[0] for b in BASES}
for b in BASES:
    gl[f"base_{b}"] = dict(leaky=stats(BASE_R[b], START, FREEZE), noleak=stats(BASE_NL[b], START, FREEZE),
                           leaky_h1=stats(BASE_R[b], *H1), noleak_h1=stats(BASE_NL[b], *H1))
    log(f"GOLD {b}: IS Cal leaky {gl[f'base_{b}']['leaky']['calmar']:.4f} -> no-leak {gl[f'base_{b}']['noleak']['calmar']:.4f} | "
        f"H1 Cal {gl[f'base_{b}']['leaky_h1']['calmar']:.4f} -> {gl[f'base_{b}']['noleak_h1']['calmar']:.4f}  "
        f"DD {gl[f'base_{b}']['leaky']['maxdd']*100:.2f} -> {gl[f'base_{b}']['noleak']['maxdd']*100:.2f}")
gl["deltas"] = {}
for base, name, kind, rk, prm, win in KEY[:7]:
    r_l, _ = variant(base, kind, rk, prm)
    r_n, _ = variant(base, kind, rk, prm, gold=gold_nl)
    rec = {}
    for tag, r, rb in (("leaky", r_l, BASE_R[base]), ("noleak", r_n, BASE_NL[base])):
        rec[tag] = dict(d_calmar=stats(r, START, FREEZE)["calmar"] - stats(rb, START, FREEZE)["calmar"],
                        d_sharpe=stats(r, START, FREEZE)["sharpe"] - stats(rb, START, FREEZE)["sharpe"],
                        h1=stats(r, *H1)["calmar"] - stats(rb, *H1)["calmar"], h1_sh=stats(r, *H1)["sharpe"] - stats(rb, *H1)["sharpe"])
    gl["deltas"][f"{base} {name}"] = rec
    log(f"GOLD {base} {name[:46]:46s} leaky dCal {rec['leaky']['d_calmar']:+.4f} H1 {rec['leaky']['h1']:+.4f} | no-leak dCal "
        f"{rec['noleak']['d_calmar']:+.4f} H1 {rec['noleak']['h1']:+.4f} dSh {rec['noleak']['d_sharpe']:+.4f}")
OUT["gold_leak"] = gl

# ------------------------------------------------------------------ E. bootstrap / placebo seed stability (dollar)


def boot_sharpe_p(r1, r2, a, b, seed, n=2000, block=63):
    x = pd.concat([r1, r2], axis=1)
    x = x[(x.index >= a) & (x.index <= b)].values
    T = len(x); g = np.random.default_rng(seed); nb = int(np.ceil(T / block)); out = np.empty(n)
    for i in range(n):
        st = g.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = x[ix]
        sh = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(252)
        out[i] = sh[0] - sh[1]
    return float((out <= 0).mean())


def boot_cagr_p(r1, r2, a, b, seed, n=2000, block=63):
    x = pd.concat([r1, r2], axis=1)
    x = x[(x.index >= a) & (x.index <= b)].values
    T = len(x); g = np.random.default_rng(seed); nb = int(np.ceil(T / block)); out = np.empty(n)
    lx = np.log1p(x)
    for i in range(n):
        st = g.integers(0, T, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = lx[ix].sum(axis=0)
        out[i] = np.expm1(s[0] * 252 / T) - np.expm1(s[1] * 252 / T)
    return float((out <= 0).mean())


bs = {}
for b in BASES:
    r, _ = variant(b, "mult", "usd5", 0.75)
    k = V[(b, "dollar shock DXY +5%/126d: engine x0.75")]["dial"]["k"]
    rd = backtest(weights(b, mult=pd.Series(k, index=idx)))[0]
    bs[b] = dict(sharpe_p={s: boot_sharpe_p(r, BASE_R[b], START, FREEZE, s) for s in (0, 1, 2, 3)},
                 dial_k=k, dial_cagr_p={s: boot_cagr_p(r, rd, START, FREEZE, s) for s in (0, 1, 2, 3)},
                 json_sharpe_p=V[(b, "dollar shock DXY +5%/126d: engine x0.75")]["boot_vs_base"]["sharpe"]["p_le0"],
                 json_dial_p=V[(b, "dollar shock DXY +5%/126d: engine x0.75")]["dial"]["boot_vs_dial"]["cagr"]["p_le0"])
    # placebo: circular shift of the RAW flag, new seed, 400 draws
    a = RAW["usd5"].reindex(idx).fillna(False).values
    g = np.random.default_rng(11)
    sb = stats(BASE_R[b], START, FREEZE)
    real = stats(r, START, FREEZE)
    pl = []
    for _ in range(400):
        sh = int(g.integers(252, len(a) - 252))
        f = lagk(pd.Series(np.roll(a, sh), index=idx), 1)
        rp = backtest(weights(b, mult=pd.Series(np.where(f, 0.75, 1.0), index=idx)))[0]
        sp = stats(rp, START, FREEZE)
        pl.append((sp["sharpe"] - sb["sharpe"], sp["calmar"] - sb["calmar"]))
    pl = np.array(pl)
    bs[b]["placebo_seed11_n400"] = dict(p_sharpe=float((pl[:, 0] >= real["sharpe"] - sb["sharpe"]).mean()),
                                         p_calmar=float((pl[:, 1] >= real["calmar"] - sb["calmar"]).mean()))
    log(f"BOOT {b} dollar: Sharpe-diff p by seed {bs[b]['sharpe_p']} (json {bs[b]['json_sharpe_p']}); dial(k={k}) CAGR p {bs[b]['dial_cagr_p']} "
        f"(json {bs[b]['json_dial_p']}); placebo seed11 n400 p(Sh) {bs[b]['placebo_seed11_n400']['p_sharpe']:.4f} p(Cal) {bs[b]['placebo_seed11_n400']['p_calmar']:.4f}")
OUT["dollar_significance_stability"] = bs

# ------------------------------------------------------------------ F. un-inversion interpretations
a_inv = inv.values
runl = np.zeros(len(a_inv), dtype=int)
for i in range(len(a_inv)):
    runl[i] = (runl[i - 1] + 1 if i > 0 else 1) if a_inv[i] else 0
cross = np.zeros(len(a_inv), dtype=bool); cross[1:] = a_inv[:-1] & ~a_inv[1:]
spell_end = [i for i in range(len(a_inv)) if a_inv[i] and (i + 1 == len(a_inv) or not a_inv[i + 1])]
q_spells = [(i - runl[i] + 1, i) for i in spell_end if runl[i] >= 63]


def window_from(trigs, hold=252):
    w = np.zeros(len(a_inv), dtype=bool)
    for t in trigs:
        w[t:t + hold] = True
    return pd.Series(w, index=idx)


first_only = [e + 1 for s, e in q_spells if e + 1 < len(a_inv)]
from_start = [i for i in np.flatnonzero(cross) if any(0 < i - s <= 504 for s, e in q_spells if e < i)]
interp = {"specialist (re-triggers within 504d of spell END)": RAW["uninv_contig"], "first crossing after each >=63d spell only": window_from(first_only),
          "crossings within 504d of spell START": window_from(from_start)}
unv = {"qualifying_spells": [[str(idx[s].date()), str(idx[e].date()), int(e - s + 1)] for s, e in q_spells],
       "first_only_triggers": [str(idx[t].date()) for t in first_only], "from_start_triggers": [str(idx[t].date()) for t in from_start]}
for nm, rawwin in interp.items():
    for b in BASES:
        for kind, prm in (("mult", 0.5), ("veto", None)):
            r, _ = variant(b, kind, None, prm, raw=rawwin)
            so, sb = stats(r, START, FREEZE), stats(BASE_R[b], START, FREEZE)
            h1 = (stats(r, *H1)["calmar"] - stats(BASE_R[b], *H1)["calmar"], stats(r, *H1)["sharpe"] - stats(BASE_R[b], *H1)["sharpe"])
            h2 = (stats(r, *H2)["calmar"] - stats(BASE_R[b], *H2)["calmar"], stats(r, *H2)["sharpe"] - stats(BASE_R[b], *H2)["sharpe"])
            key = f"{nm} | {b} | {'x0.5' if kind == 'mult' else 'risk-off'}"
            unv[key] = dict(d_calmar=so["calmar"] - sb["calmar"], d_sharpe=so["sharpe"] - sb["sharpe"], h1=h1, h2=h2,
                            gfc=pret(r, "2007-10-01", "2009-03-31") - pret(BASE_R[b], "2007-10-01", "2009-03-31"),
                            y2025_26=pret(r, "2025-01-01", FREEZE) - pret(BASE_R[b], "2025-01-01", FREEZE))
            log(f"UNINV {key:90s} dCal {unv[key]['d_calmar']:+.4f} dSh {unv[key]['d_sharpe']:+.4f} H1 {h1[0]:+.3f}/{h1[1]:+.3f} H2 {h2[0]:+.3f}/{h2[1]:+.3f} "
                f"GFC {unv[key]['gfc']*100:+.1f} 2025-26 {unv[key]['y2025_26']*100:+.1f}")
log(f"qualifying spells {unv['qualifying_spells']}; first-only triggers {unv['first_only_triggers']}; from-start triggers {unv['from_start_triggers']}")
OUT["uninversion_interpretations"] = unv

# ------------------------------------------------------------------ G. today
tail = pd.DataFrame({"tnx": P["tnx"], "d126_bp": d126 * 100, "rs50_raw": RAW["rs50"], "rs50_lag1": lagk(RAW["rs50"]), "mark": mark}).loc["2026-11-05":]
print(tail.to_string())
held_now = {}
for b in BASES:
    for bp, k in ((50, "rs50"), (75, "rs75"), (100, "rs100")):
        _, h = variant(b, "bills", k, None)
        held_now[f"{b} rs{bp}->bills"] = {c: round(float(v), 4) for c, v in h.loc[LAST].items() if abs(v) > 1e-9}
    _, h = backtest(weights(b))
    held_now[f"{b} base"] = {c: round(float(v), 4) for c, v in h.loc[LAST].items() if abs(v) > 1e-9}
log("held at", LAST.date(), held_now)
# HYG 50d flag continuity since 2026-10-14
hy = RAW["hy50"].reindex(idx).fillna(False)
off_days_since_1014 = int((~hy[(hy.index >= "2026-10-14")]).sum())
# VWEHX vs HYG 50d agreement, restricted to days where the HYG 50d average exists
valid = r_hy.rolling(50).mean().notna() & (idx >= pd.Timestamp("2007-01-01"))
agree = float((RAW["vw50"][valid] == RAW["hy50"][valid]).mean())
agree_naive = float((lagk(RAW["vw50"])[idx >= pd.Timestamp("2007-01-01")] == lagk(RAW["hy50"])[idx >= pd.Timestamp("2007-01-01")]).mean())
OUT["today"] = dict(tail={str(k.date()): {c: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for c, v in row.items()} for k, row in tail.iterrows()},
                    held_now=held_now, hy50_off_days_since_2026_10_14=off_days_since_1014,
                    vw50_vs_hy50_agreement_valid_days_2007plus=agree, vw50_vs_hy50_agreement_incl_pre_hyg_days=agree_naive,
                    n_valid=int(valid.sum()))
log(f"HYG50 off-days since 2026-10-14: {off_days_since_1014};  VWEHX50 vs HYG50 agreement (valid HYG days 2007+): {agree:.3f} (naive incl. pre-HYG: {agree_naive:.3f})")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    return o


with open(os.path.join(scratch, "verify_overlays_integrity_checks.json"), "w") as fh:
    json.dump(jsonable(OUT), fh, indent=1)
log("done")
