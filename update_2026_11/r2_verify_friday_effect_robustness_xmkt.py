"""
r2_verify_friday_effect_robustness_xmkt.py -- adversarial ROBUSTNESS checks of criterion B (cross-market) of
r2_friday_effect.  Read-only w.r.t. every project file; numeric output -> argv[1] (scratch); optional argv[2] =
scratch cache path for a ^GSPTSE (TSX own-calendar) download (skipped if the download fails).

Market construction is copied from r2_friday_effect.py (F2 rules: 2x sim sleeve 2r - 0.95%/252 - rf_d, own-price
EMA-100/vol20/er20, vt .30, chop .25 -> x.5, Aug-Sep x.5, IEF off-leg, 15% gold_phys; IEF rebuilt from ^TNX and gold
-> T-bills before 1985-10-02). First the spec B tally is reproduced, then alternative constructions are scored:
  metric      : both CAGR & Calmar (spec) | Calmar only | CAGR only | Sharpe only | both with materiality
  HSI         : panel/US calendar (spec) | own calendar
  TSX         : panel/US calendar (spec) | own calendar (^GSPTSE, if downloadable)
  window      : spec | first data + 500 | all from 1990-01-01 | common 1988-12-28.. | N225/IXIC from 1988
  holidays    : skip (spec) | substitute next trading day of the same week (Fri: last trading day)
  base rate   : B tally for every weekday, not just Friday
"""
from __future__ import annotations
import os, sys, json, time, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

T0 = time.time()
OUT_PATH = os.path.abspath(sys.argv[1])
TSX_CACHE = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
for p_ in [OUT_PATH] + ([TSX_CACHE] if TSX_CACHE else []):
    assert not p_.startswith(os.path.abspath(HERE)), "scratch outputs only"
P = C.panel(); IDX = P.index; N = len(IDX)
START, FREEZE = C.START, C.FREEZE
LAST = IDX[-1]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
NV = {"books": 0}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def last_idx(mask):
    pos = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(pos)


def run_book(W, R, mask, lag=1, cost=C.COST):
    li = last_idx(mask)
    H = np.where((li >= 0)[:, None], W[np.maximum(li, 0)], 0.0)
    if lag:
        H = np.vstack([np.zeros((lag, W.shape[1])), H[:-lag]])
    gross = (H * R).sum(1)
    turn = np.abs(np.diff(H, axis=0, prepend=np.zeros((1, H.shape[1])))).sum(1)
    NV["books"] += 1
    return gross - turn * cost


def wm(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def st1(dates, r, a, b):
    m = wm(dates, a, b)
    rr = r[m]; d = dates[m]
    e = np.cumprod(1 + rr)
    yrs = (d[-1] - d[0]).days / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    sd = rr.std(ddof=1)
    return dict(cagr=float(cagr), maxdd=dd, calmar=float(cagr / abs(dd)), sharpe=float(rr.mean() / sd * np.sqrt(252)))


def f_weights(lvl, month, voltgt, chop_thr=0.25, chop_pos=0.5, gold_w=0.15, span=100):
    lvl = pd.Series(lvl)
    ema = lvl.ewm(span=span, adjust=False).mean()
    above = (lvl > ema)
    vol20 = lvl.pct_change().rolling(20).std() * np.sqrt(252)
    er20 = lvl.diff(20).abs() / lvl.diff().abs().rolling(20).sum().replace(0, np.nan)
    scale = (voltgt / vol20).clip(0, 1).fillna(1.0)
    scale = (scale * np.where(np.isin(month, (8, 9)), 0.5, 1.0)).clip(0, 1)
    clean = (er20 >= chop_thr).fillna(False)
    eq = (above & clean) * scale + (above & ~clean) * scale * chop_pos
    core = 1.0 - gold_w
    lev = (eq * core).values
    off = np.clip(core - lev, 0, None)
    gold = np.full(len(lvl), gold_w)
    return np.column_stack([lev, off, gold])


Y = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))


def yser(t, col="close", src=None):
    src = Y if src is None else src
    s = src[src.ticker == t].set_index("date")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index <= LAST].astype(float)


def asof(s, dates):
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


# ---- off-leg / rf construction (verbatim logic of the specialist)
irx, tnx = yser("^IRX"), yser("^TNX")
rf_us = pd.concat([irx[irx.index < IDX[0]] / 100.0, P["rf"]]).sort_index()
yv = P["tnx"] / 100.0; dy = yv.diff(); carry = yv.shift(1) / 252
cal = pd.concat([P["ief_syn"] - carry, dy], axis=1).dropna(); cal.columns = ["y", "dy"]
Xc = np.column_stack([np.ones(len(cal)), -cal["dy"].values, 0.5 * cal["dy"].values ** 2])
a_, D_, Cx_ = np.linalg.lstsq(Xc, cal["y"].values, rcond=None)[0]
yt = tnx / 100.0
ief_pre = (a_ + yt.shift(1) / 252 - D_ * yt.diff() + 0.5 * Cx_ * yt.diff() ** 2)
ief_pre = ief_pre[ief_pre.index < IDX[0]].fillna(0)
ief_us = pd.concat([ief_pre, P["ief"].fillna(0)]).sort_index()
gold_pre = (irx[irx.index < IDX[0]] / 100.0 / 252)
gold_us = pd.concat([gold_pre, P["gold_phys"].fillna(0)]).sort_index()
IEF_IDX = (1 + ief_us).cumprod(); GOLD_IDX = (1 + gold_us).cumprod()


def market_panel(col):
    r = P[col].fillna(0).values.astype(float)
    nz = np.where(r != 0)[0]
    return dict(dates=IDX, lvl=np.cumprod(1 + r), ret=r, rf_d=P["rf_d"].values, ief=P["ief"].fillna(0).values,
                gold=P["gold_phys"].fillna(0).values, first=nz[0])


def market_closes(c):
    c = c[c > 0]
    dates = c.index
    r = c.pct_change().fillna(0).values
    rf_d = (asof(rf_us, dates).bfill() / 252).values
    ief = asof(IEF_IDX, dates).pct_change().fillna(0).values
    gold = asof(GOLD_IDX, dates).pct_change().fillna(0).values
    return dict(dates=dates, lvl=c.values, ret=r, rf_d=rf_d, ief=ief, gold=gold, first=0)


MK = {"SPX": market_panel("spx"), "RUT": market_panel("rut"), "TSX": market_panel("tsx"), "HSI": market_panel("hsi"),
      "DAX": market_closes(yser("^GDAXI")), "N225": market_closes(yser("^N225")), "IXIC": market_closes(yser("^IXIC")),
      "HSI_own": market_closes(yser("^HSI"))}
TSX_OWN = False
if TSX_CACHE:
    try:
        if not os.path.exists(TSX_CACHE):
            import yfinance as yf
            h = yf.Ticker("^GSPTSE").history(period="max", auto_adjust=False)
            h.index = h.index.tz_localize(None).normalize()
            h = h[~h.index.duplicated(keep="last")]
            pd.DataFrame({"date": h.index, "ticker": "^GSPTSE", "close": h["Close"].values}).to_parquet(TSX_CACHE)
        Tq = pd.read_parquet(TSX_CACHE)
        ts = yser("^GSPTSE", src=Tq)
        log("TSX own calendar:", ts.index[0].date(), "..", ts.index[-1].date(), len(ts))
        MK["TSX_own"] = market_closes(ts)
        TSX_OWN = True
    except Exception as ex:  # noqa: BLE001
        log("TSX own-calendar download failed, skipped:", repr(ex)[:200])


def spec_start(name, M):
    if name in ("N225", "IXIC"):
        return pd.Timestamp("1975-01-01")
    d = pd.DatetimeIndex(M["dates"])
    return max(START, d[min(M["first"] + 250, len(d) - 1)])


def subst_masks(dates):
    dates = pd.DatetimeIndex(dates)
    wk = pd.factorize(dates.to_period("W-FRI"))[0]
    dow = dates.dayofweek.values
    out = {d: np.zeros(len(dates), bool) for d in DAYS}
    df = pd.DataFrame({"wk": wk, "dow": dow, "pos": np.arange(len(dates))})
    for _, gg in df.groupby("wk"):
        for i, d in enumerate(DAYS):
            c = gg[gg.dow >= i]
            out[d][(c.pos.iloc[0] if (len(c) and i < 4) else gg.pos.iloc[-1])] = True
    return out


BOOKS = {}
for name, M in MK.items():
    dates = pd.DatetimeIndex(M["dates"])
    W = f_weights(pd.Series(M["lvl"], index=dates), dates.month.values, 0.30)
    R = np.column_stack([2 * M["ret"] - 0.0095 / 252 - M["rf_d"], M["ief"], M["gold"]])
    dow = dates.dayofweek.values
    sk = {d: run_book(W, R, dow == i) for i, d in enumerate(DAYS)}
    sm = subst_masks(dates)
    su = {d: run_book(W, R, sm[d]) for d in DAYS}
    l2 = {d: run_book(W, R, dow == i, lag=2) for i, d in enumerate(DAYS)}
    for bk in (sk, su, l2):
        bk["tranche"] = sum(bk[d] for d in DAYS) / 5
    BOOKS[name] = dict(dates=dates, skip=sk, subst=su, lag2=l2, first=dates[min(M["first"], len(dates) - 1)],
                       spec_start=spec_start(name, M), first250=dates[min(M["first"] + 250, len(dates) - 1)],
                       first500=dates[min(M["first"] + 500, len(dates) - 1)])
    log(f"{name:8s} first {BOOKS[name]['first'].date()} spec start {BOOKS[name]['spec_start'].date()}")


def score(name, hol="skip", start=None, day="Fri"):
    b = BOOKS[name]
    a = b["spec_start"] if start is None else start
    s = {k: st1(b["dates"], b[hol][k], a, FREEZE) for k in DAYS + ["tranche"]}
    f, t = s[day], s["tranche"]
    return dict(d_cagr_pp=(f["cagr"] - t["cagr"]) * 100, d_calmar=f["calmar"] - t["calmar"], d_sharpe=f["sharpe"] - t["sharpe"],
                start=str(pd.Timestamp(a).date()))


METRICS = {"both (spec)": lambda x: x["d_cagr_pp"] > 0 and x["d_calmar"] > 0,
           "calmar_only": lambda x: x["d_calmar"] > 0,
           "cagr_only": lambda x: x["d_cagr_pp"] > 0,
           "sharpe_only": lambda x: x["d_sharpe"] > 0,
           "both_material (>0.25pp & >0.02)": lambda x: x["d_cagr_pp"] > 0.25 and x["d_calmar"] > 0.02}


def window_start(name, wv):
    b = BOOKS[name]
    if wv == "spec":
        return b["spec_start"]
    if wv == "first+500":
        return b["spec_start"] if name in ("N225", "IXIC") else max(START, b["first500"])
    if wv == "all_from_1990":
        return max(pd.Timestamp("1990-01-01"), b["first250"])
    if wv == "common_1988-12-28":
        return max(pd.Timestamp("1988-12-28"), b["first250"])
    if wv == "N225_IXIC_from_1988":
        return max(START, b["first250"]) if name in ("N225", "IXIC") else b["spec_start"]
    if wv == "all_from_1993":
        return max(pd.Timestamp("1993-01-01"), b["first250"])
    raise ValueError(wv)


OUT = {}
# ---- 1. reproduce the spec tally
SPEC = ["SPX", "RUT", "TSX", "HSI", "DAX", "N225", "IXIC"]
rep = {m: score(m) for m in SPEC + ["HSI_own"] + (["TSX_own"] if TSX_OWN else [])}
OUT["spec_reproduction"] = rep
for m, v in rep.items():
    log(f"  spec {m:8s} start {v['start']} Fri-tr CAGR {v['d_cagr_pp']:+.3f}pp Calmar {v['d_calmar']:+.4f} Sharpe {v['d_sharpe']:+.4f}")
OUT["spec_B_count"] = int(sum(METRICS["both (spec)"](rep[m]) for m in SPEC))
log("spec B count:", OUT["spec_B_count"], "(spec says 3)")

# ---- 2. base rate: the B tally for every weekday (spec construction)
base = {}
for d in DAYS:
    sc = {m: score(m, day=d) for m in SPEC}
    base[d] = {mn: int(sum(f(sc[m]) for m in SPEC)) for mn, f in METRICS.items()}
OUT["weekday_B_tallies_spec_construction"] = base
log("weekday B tallies (spec construction):", json.dumps(base))

# ---- 3. alternative constructions
HSI_OPTS = ["HSI", "HSI_own"]
TSX_OPTS = ["TSX"] + (["TSX_own"] if TSX_OWN else [])
WIN_OPTS = ["spec", "first+500", "all_from_1990", "common_1988-12-28", "N225_IXIC_from_1988", "all_from_1993"]
HOL_OPTS = ["skip", "subst"]
rows = []
for hs, ts, wv, ho in itertools.product(HSI_OPTS, TSX_OPTS, WIN_OPTS, HOL_OPTS):
    mk = ["SPX", "RUT", ts, hs, "DAX", "N225", "IXIC"]
    sc = {m: score(m, hol=ho, start=window_start(m, wv)) for m in mk}
    row = dict(HSI=hs, TSX=ts, window=wv, holidays=ho)
    for mn, f in METRICS.items():
        row[mn] = int(sum(f(sc[m]) for m in mk))
        row[mn + "_markets"] = [m for m in mk if f(sc[m])]
    row["per_market"] = {m: (round(sc[m]["d_cagr_pp"], 3), round(sc[m]["d_calmar"], 4)) for m in mk}
    rows.append(row)
OUT["alternatives"] = rows
df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_markets") and k != "per_market"} for r in rows])
log("\n" + df.to_string())
OUT["alternatives_summary"] = {mn: dict(min=int(df[mn].min()), max=int(df[mn].max()), n_constructions=int(len(df)),
                                        n_ge5=int((df[mn] >= 5).sum()))
                               for mn in METRICS}
OUT["spec_metric_constructions_reaching_5"] = [
    {k: v for k, v in r.items() if k in ("HSI", "TSX", "window", "holidays", "both (spec)", "both (spec)_markets", "per_market")}
    for r in rows if r["both (spec)"] >= 5]
log("summary:", json.dumps(OUT["alternatives_summary"]))
log("spec-metric constructions reaching 5:", json.dumps(OUT["spec_metric_constructions_reaching_5"])[:3000])

# ---- 4. DAX / TSX / RUT start-date sweep (the knife-edge markets)
sweep = {}
for m in ["DAX", "TSX", "RUT", "HSI_own", "SPX"] + (["TSX_own"] if TSX_OWN else []):
    b = BOOKS[m]; out = {}
    for y in range(1988, 1996):
        a = max(pd.Timestamp(f"{y}-01-01"), b["first250"])
        s = score(m, start=a)
        out[str(a.date())] = (round(s["d_cagr_pp"], 3), round(s["d_calmar"], 4), bool(METRICS["both (spec)"](s)))
    sweep[m] = out
OUT["start_year_sweep"] = sweep
for m, v in sweep.items():
    log(f"  sweep {m:8s}", v)

# ---- 5. lag-2 B tally (first holding day neutralised in every market)
l2 = {m: score(m, hol="lag2") for m in SPEC}
OUT["lag2_B_count"] = {mn: int(sum(f(l2[m]) for m in SPEC)) for mn, f in METRICS.items()}
log("lag2 B tallies:", OUT["lag2_B_count"])

OUT["n_books_evaluated"] = NV["books"]
OUT["runtime_s"] = time.time() - T0


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(OUT_PATH, "w"), indent=1)
log("books evaluated", NV["books"], "saved", OUT_PATH)
