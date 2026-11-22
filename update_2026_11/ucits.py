"""
ucits.py -- NON-US (UCITS / Irish-ETP) IMPLEMENTATION CHECK            (KEY = ucits)

Question: can a Pakistan-based investor (UAE bank account, IBKR) run Strategy F2 / F3 through non-US-situs
instruments (US estate tax applies above ~$60k of US-situs assets such as US-listed ETFs), and what does it
cost versus the US-ETF version?

Steps
  1. VERIFY candidates on Yahoo: ucits_fetch.py stores Ticker.info (longName, currency, exchange, TER, AUM)
     and full daily history for every probed ticker, including the ones Yahoo does not know.
  2. CLEAN Yahoo's European histories with logged, rule-based fixes (first 10 sessions trimmed; split / unit
     changes snapped to round factors; analogue-relative spikes that revert next session dropped; persistent
     unexplained level breaks truncate history; FX series: reverting spikes only).
     USD conversion: Yahoo's FX bar dated D is the rate at ~00:00 London of D (verified: it correlates 0.50 with
     DXY of D-1 vs 0.40 with DXY of D; the 2015-01-22 ECB-QE drop shows in the 01-23 bar). A European close
     (~16:35 London) therefore uses log-FX interpolated 16.5/24 of the way from bar D to the next bar. This is a
     valuation estimate of the FX rate at the close, never an input to a trading signal.
  3. OVERLAP each candidate with its US analogue (QLD/TQQQ/QQQ/IEF/TLT/SHY/BIL/GLD = the shared panel's
     total-return *_real columns): weekly corr / beta / tracking error / CAGR gap, plus daily same-day and
     US-lagged correlations and 4-week correlation to show the close-time effect; full common history and last
     5 years; FX-residual regressions (interpolated vs same-date FX; USD-line placebos); same-product checks.
  4. STRATEGY: F2 (2x, vt .30) and F3 (3x, vt .20), off-leg = 7-10y UST, permanent 15% gold, on the PRIMARY
     UCITS mapping (pre-specified by structure, USD line where one exists, liquidity -- NOT by backtest result)
     vs the US-ETF version over the identical window.
       lag 1 = trade at the signal close (July convention; feasible for US ETFs with MOC orders).
       lag 2 = trade at the next session's close. For European listings Friday's close (~11:35 ET) PRECEDES the
               Friday US-close signal, so lag 2 (Monday European close) is the earliest honest execution; a lag-1
               backtest on European closes lets the new weights earn Friday-afternoon moves the signal already
               saw (look-ahead) -- it is run once only to measure that artifact.
     Attribution at lag 2 (sleeve swaps), turnover attribution, trading-cost and NRA dividend-withholding drag,
     a 1x control (CNDX vs QQQ) and ONE pre-specified implementation mitigation (5pp no-trade band on the
     leveraged sleeve), checked on 1988+ US history (both halves) before being applied to the UCITS window.
Every strategy-level backtest is appended to VARIANTS and counted.

Inputs : update_2026_11/ucits_prices.parquet + ucits_info.json (run ucits_fetch.py first), shared panel (read-only)
Output : update_2026_11/ucits.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

END = pd.Timestamp("2026-11-16")     # last session with European closes on Yahoo (2026-11-17 rows are empty)
WEND = pd.Timestamp("2026-11-13")    # last complete Mon-Fri week (weekly statistics)
FREEZE = C.FREEZE
OOS_A = FREEZE + pd.Timedelta(days=1)
P = C.panel()
PX = pd.read_parquet(os.path.join(HERE, "ucits_prices.parquet"))
INFO = json.load(open(os.path.join(HERE, "ucits_info.json")))["tickers"]
FETCHED = json.load(open(os.path.join(HERE, "ucits_info.json")))["fetched_utc"]
VARIANTS = []
OUT = {}

# ------------------------------------------------------------------ catalogue (roles fixed a priori)
# Yahoo supplies name / currency / exchange / TER; the legal-form notes are issuer-documentation knowledge
# (flagged 'verify on the KID' in the report).
CAT = {
    "LQQ.PA":  dict(us="qld_real",  us_tk="QLD",  sleeve="2x Nasdaq-100", primary=True,
                    structure="UCITS ETF (Amundi, French-domiciled, swap-based), accumulating; EUR line, Euronext Paris"),
    "LQQ.MI":  dict(us="qld_real",  us_tk="QLD",  sleeve="2x Nasdaq-100", primary=False,
                    structure="same fund as LQQ.PA; EUR line, Borsa Italiana"),
    "QQQ3.L":  dict(us="tqqq_real", us_tk="TQQQ", sleeve="3x Nasdaq-100", primary=True,
                    structure="ETP / collateralised note (WisdomTree Multi Asset Issuer PLC) - NOT a UCITS fund; USD line, LSE"),
    "LQQ3.L":  dict(us="tqqq_real", us_tk="TQQQ", sleeve="3x Nasdaq-100", primary=False,
                    structure="same ETP as QQQ3.L; GBp line, LSE"),
    "QQQ3.MI": dict(us="tqqq_real", us_tk="TQQQ", sleeve="3x Nasdaq-100", primary=False,
                    structure="same ETP as QQQ3.L; EUR line, Borsa Italiana"),
    "3QQQ.L":  dict(us="tqqq_real", us_tk="TQQQ", sleeve="3x Nasdaq-100", primary=False,
                    structure="ETP (Leverage Shares 3x Long Nasdaq 100) - NOT a UCITS fund; GBp line, LSE"),
    "CNDX.L":  dict(us="qqq_real",  us_tk="QQQ",  sleeve="1x Nasdaq-100 (control)", primary=False,
                    structure="UCITS ETF (iShares, Irish), physical, accumulating; USD line, LSE"),
    "EQQQ.L":  dict(us="qqq_real",  us_tk="QQQ",  sleeve="1x Nasdaq-100 (control)", primary=False,
                    structure="UCITS ETF (Invesco, Irish), physical, distributing; GBp line, LSE"),
    "CBU0.L":  dict(us="ief_real",  us_tk="IEF",  sleeve="UST 7-10y", primary=True,
                    structure="UCITS ETF (iShares, Irish), accumulating; USD line, LSE"),
    "IBTM.L":  dict(us="ief_real",  us_tk="IEF",  sleeve="UST 7-10y", primary=False,
                    structure="same fund as CBU0 (distributing share class); GBP line, LSE"),
    "DTLA.L":  dict(us="tlt_real",  us_tk="TLT",  sleeve="UST 20y+", primary=True,
                    structure="UCITS ETF (iShares, Irish), accumulating; USD line, LSE"),
    "IDTL.L":  dict(us="tlt_real",  us_tk="TLT",  sleeve="UST 20y+", primary=False,
                    structure="same fund as DTLA (distributing share class); USD line, LSE"),
    "IBTS.L":  dict(us="shy_real",  us_tk="SHY",  sleeve="UST 1-3y", primary=False,
                    structure="UCITS ETF (iShares, Irish), distributing; GBP line, LSE"),
    "IB01.L":  dict(us="bil_real",  us_tk="BIL",  sleeve="UST 0-1y", primary=True,
                    structure="UCITS ETF (iShares, Irish), accumulating; USD line, LSE"),
    "IGLN.L":  dict(us="gld_real",  us_tk="GLD",  sleeve="physical gold", primary=True,
                    structure="ETC (physically backed note, iShares Physical Metals plc) - gold cannot be a UCITS fund; USD line, LSE"),
    "SGLN.L":  dict(us="gld_real",  us_tk="GLD",  sleeve="physical gold", primary=False,
                    structure="same ETC as IGLN; GBp line, LSE"),
    "PHAU.L":  dict(us="gld_real",  us_tk="GLD",  sleeve="physical gold", primary=False,
                    structure="ETC (WisdomTree Physical Gold, Jersey issuer); USD line, LSE"),
    "SGLD.L":  dict(us="gld_real",  us_tk="GLD",  sleeve="physical gold", primary=False,
                    structure="ETC (Invesco Physical Gold); USD line, LSE"),
}


def rnd(x, n=5):
    """recursively round floats / convert numpy for JSON"""
    if isinstance(x, dict):
        return {str(k): rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ------------------------------------------------------------------ helpers
def us_level(col):
    r = P[col].dropna()
    return (1 + r).cumprod()


def on_dates(level, dates):
    """last available value on or before each date (NaN before the first observation)"""
    return level.reindex(level.index.union(dates)).ffill().reindex(dates)


NICE = np.array([2, 3, 4, 5, 8, 10, 20, 25, 50, 100, 200, 250, 500, 1000], float)


def clean(s, a=None, trim=10, allow_breaks=True, max_iter=300):
    """rule-based cleaning of a price series against an analogue level a (US ETF) -> (series, log)"""
    log = []
    s = s.dropna()
    s = s[s > 0].copy()
    if trim and len(s) > trim + 50:
        log.append(f"trimmed first {trim} sessions ({s.index[0].date()}..{s.index[trim - 1].date()}) as launch-period data")
        s = s.iloc[trim:]
    for _ in range(max_iter):
        lg = np.log(s).diff()
        e = (np.log(on_dates(a, s.index)).diff() if a is not None else pd.Series(0.0, index=s.index)).fillna(0)
        u = (lg - e).fillna(0)
        sig = 1.4826 * (u - u.median()).abs().median()
        ae = e.abs()
        thr = max(8 * sig, 0.04) + 1.5 * (ae + ae.shift(1).fillna(0) + ae.shift(-1).fillna(0))
        fixed = False
        for t in u.index[(u.abs() > thr)]:
            i = s.index.get_loc(t)
            if i == 0 or i >= len(s) - 1:
                continue
            ut, un = u.iloc[i], u.iloc[i + 1]
            if np.sign(un) == -np.sign(ut) and abs(un) > 0.5 * abs(ut):          # spike that reverts
                log.append(f"{t.date()} spike {np.expm1(lg.iloc[i]):+.1%} vs analogue {np.expm1(e.iloc[i]):+.1%}, "
                           f"reverted next session -> dropped")
                s = s.drop(t)
                fixed = True
                break
            if not allow_breaks:
                continue
            nxt = s.iloc[i + 1:i + 6]
            if len(nxt) and abs(np.log(nxt.median() / s.iloc[i])) < np.log(1.5) and abs(ut) > 0.10:
                ratio = float(np.exp(-ut))
                k = ratio if ratio >= 1 else 1 / ratio
                j = int(np.argmin(np.abs(np.log(NICE / k))))
                if k > 1.8 and abs(np.log(NICE[j] / k)) < 0.10:
                    f = NICE[j] if ratio >= 1 else 1 / NICE[j]
                    s.iloc[:i] = s.iloc[:i] / f
                    log.append(f"{t.date()} split/unit change: raw move {np.expm1(lg.iloc[i]):+.2%}, implied factor "
                               f"{k:.1f} -> snapped {NICE[j]:g}; earlier prices divided by {f:g}")
                else:
                    log.append(f"{t.date()} persistent level break {np.expm1(lg.iloc[i]):+.1%} vs analogue "
                               f"{np.expm1(e.iloc[i]):+.1%} (factor {k:.2f}, not a round split) -> earlier history discarded")
                    s = s.iloc[i:]
                fixed = True
                break
        if not fixed:
            break
    return s, log


FX, FXLOG = {}, {}
for nm in ["EURUSD=X", "GBPUSD=X"]:
    fs = PX[PX.ticker == nm].set_index("date").sort_index()["close"]
    FX[nm], FXLOG[nm] = clean(fs, None, trim=0, allow_breaks=False)      # keep data after END: needed for interpolation

FX_W = 16.5 / 24      # European close (~16:35 London) lies 16.5h after Yahoo's FX bar D (~00:00 London)


def fx_at_close(fx, dates, interp=True):
    lf = np.log(fx)
    cur = on_dates(lf, dates)
    if not interp:
        return np.exp(cur)
    pos = np.searchsorted(lf.index.values, dates.values, side="right")
    nxt = pd.Series(np.where(pos < len(lf), lf.values[np.minimum(pos, len(lf) - 1)], np.nan), index=dates)
    return np.exp((1 - FX_W) * cur + FX_W * nxt.fillna(cur))


def fx_for(ccy):
    if ccy in ("GBp", "GBX", "GBP"):
        return "GBPUSD=X", (100.0 if ccy in ("GBp", "GBX") else 1.0)
    if ccy == "EUR":
        return "EURUSD=X", 1.0
    return None, 1.0


def to_usd(tk, s, interp=True):
    fxn, div = fx_for(INFO[tk]["currency"])
    if fxn is None:
        return s / div
    return s / div * fx_at_close(FX[fxn], s.index, interp)


def ann_cagr(level):
    yrs = (level.index[-1] - level.index[0]).days / 365.25
    return float((level.iloc[-1] / level.iloc[0]) ** (1 / yrs) - 1)


# ------------------------------------------------------------------ 1. verification
verify, nonexist = {}, []
for tk, rec in INFO.items():
    if not rec.get("exists"):
        nonexist.append(tk)
        continue
    verify[tk] = {k: rec.get(k) for k in ["role", "longName", "currency", "exchange", "fullExchangeName", "quoteType",
                                         "legalType", "fundFamily", "netExpenseRatio", "totalAssets",
                                         "averageDailyVolume3Month", "first", "last", "rows", "n_dividends"]}
    if tk in CAT:
        verify[tk].update(structure_note=CAT[tk]["structure"], us_analogue=CAT[tk]["us_tk"],
                          primary_mapping=CAT[tk]["primary"])
OUT["verification"] = verify
OUT["probed_but_not_on_yahoo"] = nonexist
print(f"verified on Yahoo: {len(verify)} tickers; not found: {nonexist}")

# ------------------------------------------------------------------ 2. cleaning (USD, interpolated FX)
CLEAN, CLOG = {}, {}
for tk, meta in CAT.items():
    if not INFO.get(tk, {}).get("exists"):
        continue
    g = PX[PX.ticker == tk].set_index("date").sort_index()
    s = g["adjclose"].where(g["adjclose"] > 0).dropna()
    s = s[s.index <= END]
    CLEAN[tk], CLOG[tk] = clean(to_usd(tk, s), us_level(meta["us"]))
    print(f"  cleaned {tk:8s} {len(CLEAN[tk]):5d} obs from {CLEAN[tk].index[0].date()}  log entries={len(CLOG[tk])}")
OUT["cleaning_log"] = {**CLOG, **{f"FX {k}": v for k, v in FXLOG.items()}}


# ------------------------------------------------------------------ 3. overlap statistics
def overlap(su, a_col, start=None, end=WEND):
    a = us_level(a_col)
    W = pd.concat([su.resample("W-FRI").last(), a.resample("W-FRI").last()], axis=1, keys=["u", "a"]).dropna()
    W = W[W.index <= end]
    if start is not None:
        W = W[W.index >= pd.Timestamp(start)]
    if len(W) < 30:
        return None
    rw = W.pct_change().dropna()
    d = rw.u - rw.a
    r4 = W.iloc[::4].pct_change().dropna()
    cd = su.index.intersection(a.index)
    cd = cd[(cd >= W.index[0]) & (cd <= end)]
    du = su.reindex(cd).pct_change(fill_method=None)
    da = a.reindex(cd).pct_change(fill_method=None)
    return dict(first_week=str(W.index[0].date()), last_week=str(W.index[-1].date()), weeks=int(len(rw)),
                years=(W.index[-1] - W.index[0]).days / 365.25,
                corr_weekly=float(rw.u.corr(rw.a)), beta_weekly=float(np.cov(rw.u, rw.a)[0, 1] / rw.a.var()),
                corr_4week=float(r4.u.corr(r4.a)),
                te_weekly_ann=float(d.std() * np.sqrt(52)),
                cagr_ucits_usd=ann_cagr(W.u), cagr_us=ann_cagr(W.a), cagr_gap=ann_cagr(W.u) - ann_cagr(W.a),
                vol_ucits=float(rw.u.std() * np.sqrt(52)), vol_us=float(rw.a.std() * np.sqrt(52)),
                corr_daily_same_day=float(du.corr(da)), corr_daily_vs_us_prev_day=float(du.corr(da.shift(1))))


FIVE_Y = WEND - pd.DateOffset(years=5)
ov_full, ov_5y = {}, {}
for tk, meta in CAT.items():
    if tk not in CLEAN:
        continue
    ov_full[tk] = overlap(CLEAN[tk], meta["us"])
    ov_5y[tk] = overlap(CLEAN[tk], meta["us"], start=FIVE_Y)
    for lbl, o in [("full", ov_full[tk]), ("5y", ov_5y[tk])]:
        if o:
            print(f"  {tk:8s} vs {meta['us_tk']:4s} [{lbl:4s} {o['first_week']}] corr_w {o['corr_weekly']:.3f} "
                  f"beta {o['beta_weekly']:.2f} corr_4w {o['corr_4week']:.3f} TE {o['te_weekly_ann']*100:5.2f}%  "
                  f"CAGR {o['cagr_ucits_usd']*100:6.2f}% vs {o['cagr_us']*100:6.2f}% gap {o['cagr_gap']*100:+.2f}%  "
                  f"daily corr same {o['corr_daily_same_day']:.2f} / US-prev {o['corr_daily_vs_us_prev_day']:.2f}")
OUT["overlap_full_history"] = ov_full
OUT["overlap_last_5y"] = ov_5y


def fx_reg(su, a_col, fxname, end=WEND):
    a = us_level(a_col)
    W = pd.concat([su.resample("W-FRI").last(), a.resample("W-FRI").last(),
                   FX[fxname][FX[fxname].index <= END].resample("W-FRI").last()],
                  axis=1, keys=["u", "a", "f"]).dropna()
    W = W[W.index <= end]
    r = W.pct_change().dropna()
    X = np.column_stack([np.ones(len(r)), r.a, r.f])
    b = np.linalg.lstsq(X, r.u.values, rcond=None)[0]
    e = r.u.values - X @ b
    se = np.sqrt(np.diag(e.var(ddof=3) * np.linalg.inv(X.T @ X)))
    return dict(fx=fxname, weeks=int(len(r)), coef_us_analogue=float(b[1]), coef_fx=float(b[2]),
                t_fx=float(b[2] / se[2]), r2=float(1 - e.var() / r.u.var()))


fxr = {}
for tk in ["LQQ.PA", "LQQ.MI", "QQQ3.MI", "LQQ3.L", "3QQQ.L", "EQQQ.L", "IBTM.L", "IBTS.L", "SGLN.L"]:
    if tk in CLEAN:
        fxn, div = fx_for(INFO[tk]["currency"])
        s_int = CLEAN[tk]
        s_raw = s_int / fx_at_close(FX[fxn], s_int.index, True) * fx_at_close(FX[fxn], s_int.index, False)
        fxr[tk] = {"interpolated_fx (used)": fx_reg(s_int, CAT[tk]["us"], fxn),
                   "same_date_yahoo_fx": fx_reg(s_raw, CAT[tk]["us"], fxn)}
for tk in ["QQQ3.L", "CNDX.L", "CBU0.L", "IGLN.L"]:                      # USD-line placebos
    if tk in CLEAN:
        fxr[f"{tk} (USD-line placebo vs EURUSD)"] = {"interpolated_fx (used)": fx_reg(CLEAN[tk], CAT[tk]["us"], "EURUSD=X")}
OUT["fx_residual_regressions"] = {
    "note": ("weekly USD return of the line regressed on its US analogue and on the FX rate of its quote currency. "
             "If a 2x product applied its leverage to the EUR-translated index, the USD-converted line would load "
             "-(L-1) = -1.0 on EURUSD; ~0 means the quote currency is only a wrapper. Same-date Yahoo FX is ~16.5h "
             "stale at a European close, which by itself creates a positive loading; USD-line placebos show what "
             "the European close time alone produces."), **fxr}
for k, v in fxr.items():
    for kk, vv in v.items():
        print(f"  FX reg {k:36s} {kk:24s} coef_us {vv['coef_us_analogue']:.3f}  coef_fx {vv['coef_fx']:+.3f} "
              f"(t {vv['t_fx']:+.1f})  R2 {vv['r2']:.3f}")


def pair(t1, t2):
    W = pd.concat([CLEAN[t1].resample("W-FRI").last(), CLEAN[t2].resample("W-FRI").last()], axis=1).dropna()
    W = W[W.index <= WEND]
    r = W.pct_change().dropna()
    return dict(first_week=str(W.index[0].date()), weeks=int(len(r)), corr_weekly=float(r.iloc[:, 0].corr(r.iloc[:, 1])),
                te_ann=float((r.iloc[:, 0] - r.iloc[:, 1]).std() * np.sqrt(52)),
                cagr_gap=ann_cagr(W.iloc[:, 0]) - ann_cagr(W.iloc[:, 1]))


same_lines = {}
for t1, t2, why in [("LQQ.MI", "LQQ.PA", "same fund, Milan vs Paris"), ("LQQ3.L", "QQQ3.L", "same ETP, GBp vs USD line"),
                    ("QQQ3.MI", "QQQ3.L", "same ETP, EUR vs USD line"), ("SGLN.L", "IGLN.L", "same ETC, GBp vs USD line"),
                    ("IBTM.L", "CBU0.L", "same fund, Dist (GBP line, Yahoo dividend-adjusted) vs Acc (USD line)"),
                    ("IDTL.L", "DTLA.L", "same fund, Dist vs Acc (USD lines)"),
                    ("3QQQ.L", "QQQ3.L", "Leverage Shares vs WisdomTree 3x"),
                    ("PHAU.L", "IGLN.L", "WisdomTree vs iShares gold ETC"), ("SGLD.L", "IGLN.L", "Invesco vs iShares gold ETC"),
                    ("EQQQ.L", "CNDX.L", "Invesco Dist vs iShares Acc 1x Nasdaq")]:
    if t1 in CLEAN and t2 in CLEAN:
        same_lines[f"{t1} vs {t2}"] = dict(what=why, **pair(t1, t2))
OUT["line_and_issuer_checks"] = same_lines
for k, v in same_lines.items():
    print(f"  {k:20s} from {v['first_week']} corr {v['corr_weekly']:.4f} TE {v['te_ann']*100:5.2f}% "
          f"CAGR gap {v['cagr_gap']*100:+.2f}%  ({v['what']})")


# ------------------------------------------------------------------ liquidity / spreads / expense ratios
def abdi_ranaldo(g):
    g = g[(g.volume > 0) & (g.high > g.low) & (g.close > 0)]
    c = np.log(g.close)
    eta = (np.log(g.high) + np.log(g.low)) / 2
    s2 = 4 * (c - eta) * (c - eta.shift(-1))
    m = s2.groupby(s2.index.to_period("M")).mean().clip(lower=0)
    return float(np.sqrt(m).median()) if len(m) else np.nan


liq = {}
for tk in list(CAT) + ["QLD", "TQQQ", "QQQ", "IEF", "TLT", "SHY", "BIL", "GLD"]:
    if not INFO.get(tk, {}).get("exists"):
        continue
    g = PX[PX.ticker == tk].set_index("date").sort_index()
    g = g[(g.index <= END)].dropna(subset=["close"])
    g6 = g[g.index > END - pd.Timedelta(days=183)]
    g12 = g[g.index > END - pd.Timedelta(days=365)]
    pu = to_usd(tk, g6["close"]) if tk in CAT else g6["close"]
    rec = INFO[tk]
    bid, ask, last = rec.get("bid") or 0, rec.get("ask") or 0, rec.get("regularMarketPrice") or 0
    snap = None
    if ask > bid > 0 and last > 0 and abs((ask + bid) / 2 / last - 1) < 0.02:
        snap = (ask - bid) / ((ask + bid) / 2)
    rt = rec.get("regularMarketTime")
    liq[tk] = dict(median_daily_traded_value_usd_6m=float((g6.volume * pu).median()),
                   zero_volume_share_6m=float((g6.volume == 0).mean()),
                   yahoo_quote_spread_snapshot=snap,
                   snapshot_time_utc=str(pd.Timestamp(int(rt), unit="s")) if rt else None,
                   snapshot_note=("taken around the 16:35 London closing auction / after hours - indicative only"
                                  if snap is not None else "bid/ask missing, inverted or stale vs last price"),
                   abdi_ranaldo_spread_12m_UNRELIABLE=abdi_ranaldo(g12),
                   ter_yahoo_pct=rec.get("netExpenseRatio"), aum_yahoo=rec.get("totalAssets"))
OUT["liquidity_and_costs"] = {"note": ("Abdi-Ranaldo high/low spread estimator was tried and is UNINFORMATIVE here "
                                       "(it returns 35-107bp for QQQ/QLD/TQQQ/GLD whose quoted spreads are ~1bp); "
                                       "on-exchange volume understates European ETF liquidity (much trades OTC/RFQ)."),
                              **liq}
for tk, v in liq.items():
    sn = v["yahoo_quote_spread_snapshot"]
    print(f"  liq {tk:8s} ADV ${v['median_daily_traded_value_usd_6m']/1e6:9.2f}m  zero-vol {v['zero_volume_share_6m']:.0%}  "
          f"snap {sn*1e4 if sn is not None else float('nan'):6.1f}bp  TER {v['ter_yahoo_pct']}  AUM {v['aum_yahoo']}")

# ------------------------------------------------------------------ 4. strategy on UCITS instruments
P2 = P.copy()
COLMAP = {"u_lqq": "LQQ.PA", "u_qqq3": "QQQ3.L", "u_cbu0": "CBU0.L", "u_igln": "IGLN.L",
          "u_ibtm": "IBTM.L", "u_phau": "PHAU.L", "u_cndx": "CNDX.L"}
for col, tk in COLMAP.items():
    if tk in CLEAN:
        P2[col] = on_dates(CLEAN[tk], P2.index).pct_change(fill_method=None)


_PXC = PX.dropna(subset=["close"])
EXCH_CAL = {sfx: pd.DatetimeIndex(sorted(set(_PXC[_PXC.ticker.str.endswith(sfx)].date))) for sfx in (".L", ".PA", ".MI")}


def eu_open_mask(tickers):
    """True on panel (US) dates on which the exchanges of all listed European instruments are open. The exchange
    calendar is the union of all Yahoo lines on that exchange (a thin line's missing row is not a holiday)."""
    m = pd.Series(True, index=P2.index)
    for tk in tickers:
        sfx = "." + tk.split(".")[-1]
        m &= P2.index.isin(EXCH_CAL[sfx])
    return m


def held_weights_eu(Pp, w, lag, open_mask, mark="mark"):
    """C.held_weights, except that a decision taken on a mark is executed at the close of the first session that is
    >= lag-1 sessions after the mark AND on which all the European listings trade (holdings change the next day).
    With an all-True mask this reproduces C.held_weights exactly (asserted below)."""
    m = Pp[mark].fillna(False).values
    tgt = w.where(Pp[mark]).ffill().fillna(0).values
    ok = open_mask.reindex(Pp.index).fillna(False).values
    out = np.zeros_like(tgt)
    cur = np.zeros(tgt.shape[1])
    pending, earliest = None, None
    for i in range(len(tgt)):
        out[i] = cur
        if m[i]:
            pending, earliest = tgt[i].copy(), i + lag - 1
        if pending is not None and i >= earliest and ok[i]:
            cur, pending = pending, None
    return pd.DataFrame(out, index=Pp.index, columns=w.columns)


def bt_pc(w, costs, lag, Pp=None, eu_mask=None):
    """C.backtest with per-column one-way costs (identical to C.backtest when every cost = C.COST and no mask);
    eu_mask -> execution only on sessions where the European listings trade. Returns (returns, held weights)."""
    Pp = P2 if Pp is None else Pp
    ww = C.held_weights(Pp, w, lag=lag) if eu_mask is None else held_weights_eu(Pp, w, lag, eu_mask)
    gross = sum(ww[c] * Pp[c].fillna(0) for c in ww.columns)
    tc = sum(ww[c].diff().abs().fillna(0) * costs.get(c, C.COST) for c in ww.columns)
    return gross - tc, ww


def band_weights(w, lev, band=0.05):
    """5pp no-trade band (pre-specified, not tuned): at each rebalance mark keep the held weights unless the risk
    state flips (lev weight zero <-> non-zero) or the target leveraged-sleeve weight moved > band from the held one."""
    m = P2["mark"].fillna(False).values
    tgt = w.values
    out = np.full(tgt.shape, np.nan)
    li = list(w.columns).index(lev)
    held = None
    for i in np.where(m)[0]:
        t = tgt[i]
        if held is None or ((t[li] == 0) != (held[li] == 0)) or abs(t[li] - held[li]) > band:
            held = t.copy()
        out[i] = held
    return pd.DataFrame(out, index=w.index, columns=w.columns)


COL2TK = {"u_lqq": "LQQ.PA", "u_qqq3": "QQQ3.L", "u_cbu0": "CBU0.L", "u_igln": "IGLN.L",
          "u_ibtm": "IBTM.L", "u_phau": "PHAU.L", "u_cndx": "CNDX.L"}


def run(label, lev, vt, off, gold, lag=1, costs=None, band=None, Pp=None, eu_exec=True):
    """returns (daily returns, held weights).  Any run holding a European listing executes only on sessions where
    its European listings trade (eu_exec=False only for the documented lag-1 look-ahead artifact)."""
    Pp = P2 if Pp is None else Pp
    w = C.strategy_weights(Pp, lev=lev, voltgt=vt, off=off, gold=gold)
    if band is not None:
        w = band_weights(w, lev, band)
    eu = [COL2TK[c] for c in (lev, off, gold) if c in COL2TK]
    mask = eu_open_mask(eu) if (eu and eu_exec) else None
    r, ww = bt_pc(w, costs or {}, lag, Pp, mask)
    VARIANTS.append(label)
    return r, ww


def monthly_te(r1, r2, a, b):
    m1 = (1 + r1[(r1.index >= a) & (r1.index <= b)]).resample("ME").prod() - 1
    m2 = (1 + r2[(r2.index >= a) & (r2.index <= b)]).resample("ME").prod() - 1
    return float((m1 - m2).std() * np.sqrt(12))


def yearly(r, a, b):
    rr = r[(r.index >= a) & (r.index <= b)]
    return {int(y): float((1 + g).prod() - 1) for y, g in rr.groupby(rr.index.year)}


def turnover_attribution(lev, vt, a, b):
    """hierarchical cause of each weekly change of the TARGET leveraged weight (US version, descriptive only)"""
    w = C.strategy_weights(P2, lev=lev, voltgt=vt, off="ief", gold="gold_phys")[lev]
    mk = P2.index[P2["mark"].fillna(False).values]
    mk = mk[(mk >= a - pd.Timedelta(days=7)) & (mk <= b)]
    ron = P2["above"].reindex(mk)
    cln = (P2["er20"] >= 0.25).fillna(False).reindex(mk)
    sea = P2["month"].isin([8, 9]).reindex(mk)
    dw = w.reindex(mk).diff().abs()
    ron_p = ron.shift(fill_value=False).astype(bool)
    cause = np.where(ron != ron_p, "risk on/off switch",
                     np.where(~ron & ~ron_p, "none (risk-off)",
                              np.where(cln != cln.shift(fill_value=False), "chop flag flip (ER 0.25)",
                                       np.where(sea != sea.shift(fill_value=False), "Aug/Sep season switch",
                                                "vol-target rescaling"))))
    yrs = (b - a).days / 365.25
    df = pd.DataFrame({"dw": dw, "cause": cause}).iloc[1:]
    return {k: float(v / yrs) for k, v in df.groupby("cause")["dw"].sum().items()}


_w = C.strategy_weights(P2, lev="qld", voltgt=0.30, off="ief", gold="gold_phys")
_chk = float((bt_pc(_w, {}, 1)[0] - C.backtest(P2, _w)).abs().max())
assert _chk < 1e-12, _chk
_all = pd.Series(True, index=P2.index)
for _lag in (1, 2):
    _chk = float((held_weights_eu(P2, _w, _lag, _all) - C.held_weights(P2, _w, lag=_lag)).abs().max().max())
    assert _chk < 1e-12, (_lag, _chk)

# central realistic one-way costs (assumptions): ~0.05% commission typical of IBKR on LSE/Euronext + half-spread
# (+~1bp FX conversion for the EUR line).  US ETFs keep the July 3bp.  Pessimistic = roughly double.
COST_REAL = {"u_lqq": 15e-4, "u_qqq3": 20e-4, "u_cbu0": 8e-4, "u_igln": 8e-4, "u_ibtm": 8e-4, "u_phau": 8e-4, "u_cndx": 8e-4}
COST_PESS = {"u_lqq": 30e-4, "u_qqq3": 40e-4, "u_cbu0": 12e-4, "u_igln": 12e-4, "u_ibtm": 12e-4, "u_phau": 12e-4, "u_cndx": 12e-4}


def ttm_yield(tk):
    g = PX[PX.ticker == tk].set_index("date").sort_index()
    g = g[g.index <= END]
    return float(g.loc[g.index > END - pd.Timedelta(days=365), "dividends"].sum() / g["close"].dropna().iloc[-1])


YIELD = {tk: ttm_yield(tk) for tk in ["QLD", "TQQQ", "IEF", "TLT", "BIL", "GLD"]}
OUT["us_etf_ttm_distribution_yield"] = YIELD
print(f"  US ETF trailing-12m distribution yields: {({k: round(v*100, 2) for k, v in YIELD.items()})}")

# instrument-level gaps (weekly, full overlap) used for the 'product only' estimate
INSTR_GAP = {"u_lqq": ov_full["LQQ.PA"]["cagr_gap"], "u_qqq3": ov_full["QQQ3.L"]["cagr_gap"],
             "u_cbu0": ov_full["CBU0.L"]["cagr_gap"], "u_igln": ov_full["IGLN.L"]["cagr_gap"],
             "u_ibtm": ov_full["IBTM.L"]["cagr_gap"], "u_phau": ov_full["PHAU.L"]["cagr_gap"]}
INSTR_GAP_5Y = {"u_lqq": ov_5y["LQQ.PA"]["cagr_gap"], "u_qqq3": ov_5y["QQQ3.L"]["cagr_gap"],
                "u_cbu0": ov_5y["CBU0.L"]["cagr_gap"], "u_igln": ov_5y["IGLN.L"]["cagr_gap"]}


def stats3(r, a, b):
    return {"window": C.stats(r, a=a, b=b, start=None), "in_sample_to_freeze": C.stats(r, a=a, b=FREEZE, start=None),
            "oos_total_return_after_freeze": C.period_return(r, OOS_A, b)}


def strategy_block(name, us, uc, vt, full=True):
    cols = [uc["lev"], uc["off"], uc["gold"]]
    a = max(P2[c].first_valid_index() for c in cols) + pd.Timedelta(days=1)
    a = P2.index[P2.index >= a][0]
    b = END
    R, W = {}, {}
    R["US lag1 3bp"], W["us"] = run(f"{name}: US lag1 3bp", us["lev"], vt, us["off"], us["gold"], lag=1)
    R["US lag2 3bp"], _ = run(f"{name}: US lag2 3bp", us["lev"], vt, us["off"], us["gold"], lag=2)
    R["UCITS lag2 3bp"], W["uc"] = run(f"{name}: UCITS lag2 3bp", uc["lev"], vt, uc["off"], uc["gold"], lag=2)
    R["UCITS lag2 realistic-cost"], _ = run(f"{name}: UCITS lag2 realistic cost", uc["lev"], vt, uc["off"], uc["gold"],
                                            lag=2, costs=COST_REAL)
    if full:
        R["UCITS lag1 3bp (LOOK-AHEAD artifact)"], _ = run(f"{name}: UCITS lag1 3bp (look-ahead artifact)",
                                                           uc["lev"], vt, uc["off"], uc["gold"], lag=1)
        R["swap lev sleeve only (lag2)"], _ = run(f"{name}: swap lev only lag2", uc["lev"], vt, us["off"], us["gold"], lag=2)
        R["swap bond sleeve only (lag2)"], _ = run(f"{name}: swap bond only lag2", us["lev"], vt, uc["off"], us["gold"], lag=2)
        R["swap gold sleeve only (lag2)"], _ = run(f"{name}: swap gold only lag2", us["lev"], vt, us["off"], uc["gold"], lag=2)
        R["US lag1 3bp + 5pp band"], W["band"] = run(f"{name}: US lag1 3bp + 5pp band", us["lev"], vt, us["off"],
                                                     us["gold"], lag=1, band=0.05)
        R["UCITS lag2 realistic-cost + 5pp band"], _ = run(f"{name}: UCITS lag2 realistic + 5pp band", uc["lev"], vt,
                                                           uc["off"], uc["gold"], lag=2, costs=COST_REAL, band=0.05)
    SERIES[name] = dict(R=R, a=a, b=b)
    S = {k: stats3(r, a, b) for k, r in R.items()}
    base = S["US lag1 3bp"]["window"]["cagr"]
    cg = {k: S[k]["window"]["cagr"] for k in S}
    gaps = {k: cg[k] - base for k in S}
    boots = {"UCITS lag2 realistic - US lag1 3bp (total)":
             C.block_bootstrap_cagr_diff(R["UCITS lag2 realistic-cost"][:b], R["US lag1 3bp"][:b], start=a),
             "UCITS lag2 3bp - US lag2 3bp (instrument + European close, same session)":
             C.block_bootstrap_cagr_diff(R["UCITS lag2 3bp"][:b], R["US lag2 3bp"][:b], start=a),
             "US lag2 - US lag1 (one-session execution delay)":
             C.block_bootstrap_cagr_diff(R["US lag2 3bp"][:b], R["US lag1 3bp"][:b], start=a)}
    te_strategy = monthly_te(R["UCITS lag2 3bp"], R["US lag2 3bp"], a, b)
    # weights / turnover (US lag-1 positions are the strategy's own, instrument independent)
    yrs = (b - a).days / 365.25
    ww = W["us"][(W["us"].index >= a) & (W["us"].index <= b)]
    avg_w = ww.mean().to_dict()
    turn = (ww.diff().abs().sum() / yrs).to_dict()
    wwu = W["uc"][(W["uc"].index >= a) & (W["uc"].index <= b)]
    turn_uc = (wwu.diff().abs().sum() / yrs).to_dict()
    extra_central = sum(turn_uc[c] * (COST_REAL.get(c, C.COST) - C.COST) for c in turn_uc)
    extra_pess = sum(turn_uc[c] * (COST_PESS.get(c, C.COST) - C.COST) for c in turn_uc)
    lev_c, oth = uc["lev"], [c for c in turn_uc if c != uc["lev"]]
    cost_grid = {f"lev {cl}bp / bond+gold {cb}bp one-way": turn_uc[lev_c] * (cl - 3) / 1e4 +
                 sum(turn_uc[c] for c in oth) * (cb - 3) / 1e4
                 for cl in (8, 10, 15, 20, 30, 40) for cb in (5, 8, 12)}
    uc_avg = {c: wwu[c].mean() for c in wwu.columns}
    instr_only = sum(uc_avg[c] * INSTR_GAP.get(c, 0.0) for c in uc_avg)
    instr_only_5y = sum(uc_avg[c] * INSTR_GAP_5Y.get(c, INSTR_GAP.get(c, 0.0)) for c in uc_avg)
    lev_tk = {"qld": "QLD", "tqqq": "TQQQ"}[us["lev"]]
    wht_all = 0.30 * (avg_w.get(us["lev"], 0) * YIELD[lev_tk] + avg_w.get(us["off"], 0) * YIELD["IEF"])
    wht_lev = 0.30 * avg_w.get(us["lev"], 0) * YIELD[lev_tk]
    res = dict(window=[str(a.date()), str(b.date())], years=yrs, stats=S, cagr_gap_vs_US_lag1_3bp=gaps,
               bootstrap=boots, strategy_level_monthly_TE_ucits_vs_us_lag2=te_strategy,
               analytic_extra_cost_drag_grid_per_year=cost_grid,
               yearly={k: yearly(R[k], a, b) for k in ("US lag1 3bp", "US lag2 3bp", "UCITS lag2 3bp", "UCITS lag2 realistic-cost")},
               avg_weights_US=avg_w, annual_turnover_US_lag1=turn, annual_turnover_UCITS_lag2=turn_uc,
               turnover_attribution_lev_sleeve=turnover_attribution(us["lev"], vt, a, b),
               extra_trading_cost_drag_analytic={"central": extra_central, "pessimistic": extra_pess,
                                                 "one_way_cost_assumptions": {"central": COST_REAL, "pessimistic": COST_PESS,
                                                                              "US": C.COST}},
               instrument_only_drag_from_weekly_gaps={"full_overlap": instr_only, "last_5y_gaps": instr_only_5y},
               potential_nra_withholding_drag_of_US_version={
                   "30pct_on_all_US_ETF_distributions": wht_all,
                   "30pct_on_leveraged_ETF_only (Treasury-ETF interest exempt)": wht_lev,
                   "basis": "trailing-12m distribution yields x average weights; 0 if the broker applies the "
                            "interest-related-dividend exemption to everything"})
    if full:
        dec = {"A. one-session execution delay on US ETFs (US lag2 - US lag1)": cg["US lag2 3bp"] - base,
               "B. instruments + European close, same session (UCITS lag2 - US lag2)": cg["UCITS lag2 3bp"] - cg["US lag2 3bp"],
               "   B1 leveraged sleeve swap (lag2)": cg["swap lev sleeve only (lag2)"] - cg["US lag2 3bp"],
               "   B2 bond sleeve swap (lag2)": cg["swap bond sleeve only (lag2)"] - cg["US lag2 3bp"],
               "   B3 gold sleeve swap (lag2)": cg["swap gold sleeve only (lag2)"] - cg["US lag2 3bp"],
               "   memo: instrument-only estimate from weekly CAGR gaps x avg weights": instr_only,
               "C. higher European trading costs (UCITS realistic - UCITS 3bp, lag2)": cg["UCITS lag2 realistic-cost"] - cg["UCITS lag2 3bp"],
               "TOTAL measured (UCITS lag2 realistic - US lag1 3bp) = A+B+C": cg["UCITS lag2 realistic-cost"] - base,
               "TOTAL with 5pp band on UCITS (vs US lag1 3bp, no band)": cg["UCITS lag2 realistic-cost + 5pp band"] - base,
               "memo: look-ahead artifact if European closes are backtested at lag1 (UCITS lag1 - UCITS lag2)":
                   cg["UCITS lag1 3bp (LOOK-AHEAD artifact)"] - cg["UCITS lag2 3bp"],
               "memo: avoided 30% WHT on all US ETF distributions (adds back if the broker withholds)": wht_all}
        res["gap_decomposition_cagr"] = dec
        # band check on the long US history (1988+) and both halves -- must not hurt before being recommended
        rb, r0 = R["US lag1 3bp + 5pp band"], R["US lag1 3bp"]
        bw = W["band"][(W["band"].index >= C.START) & (W["band"].index <= FREEZE)]
        res["band_check_US_1988_to_freeze"] = {
            "no_band_1988_2006": C.stats(r0, "1988-01-01", "2006-12-31", start=None),
            "band_1988_2006": C.stats(rb, "1988-01-01", "2006-12-31", start=None),
            "no_band_2007_freeze": C.stats(r0, "2007-01-01", FREEZE, start=None),
            "band_2007_freeze": C.stats(rb, "2007-01-01", FREEZE, start=None),
            "no_band_1988_freeze": C.stats(r0, "1988-01-01", FREEZE, start=None),
            "band_1988_freeze": C.stats(rb, "1988-01-01", FREEZE, start=None),
            "oos_total_no_band": C.period_return(r0, OOS_A, b), "oos_total_band": C.period_return(rb, OOS_A, b),
            "bootstrap_band_minus_noband_1988_freeze": C.block_bootstrap_cagr_diff(rb[:FREEZE], r0[:FREEZE]),
            "lev_turnover_per_year_1988_freeze_band": float(bw[us["lev"]].diff().abs().sum() /
                                                            ((FREEZE - C.START).days / 365.25))}
        ww0 = W["us"][(W["us"].index >= C.START) & (W["us"].index <= FREEZE)]
        res["band_check_US_1988_to_freeze"]["lev_turnover_per_year_1988_freeze_no_band"] = float(
            ww0[us["lev"]].diff().abs().sum() / ((FREEZE - C.START).days / 365.25))
    print(f"\n=== {name}: window {a.date()} .. {b.date()} ({yrs:.1f}y) ===")
    for k in S:
        s = S[k]["window"]
        print(f"  {k:40s} {C.fmt(s)}  gap {gaps[k]*100:+.2f}% | IS CAGR {S[k]['in_sample_to_freeze']['cagr']*100:6.2f}% "
              f"OOS total {S[k]['oos_total_return_after_freeze']*100:+6.2f}%")
    for k, v in boots.items():
        print(f"  bootstrap {k}: mean {v['mean']*100:+.2f}%  95% [{v['lo']*100:+.2f}, {v['hi']*100:+.2f}]  P(<=0) {v['p_le0']:.2f}")
    print(f"  strategy-level monthly TE (UCITS lag2 vs US lag2) {te_strategy*100:.2f}%/yr")
    print(f"  avg weights {({k: round(v, 3) for k, v in avg_w.items()})}  US turnover/yr {({k: round(v, 2) for k, v in turn.items()})}")
    print(f"  lev turnover attribution/yr {({k: round(v, 2) for k, v in res['turnover_attribution_lev_sleeve'].items()})}")
    print(f"  extra trading-cost drag (analytic) central {extra_central*100:.2f}%/yr  pessimistic {extra_pess*100:.2f}%/yr; "
          f"instrument-only drag {instr_only*100:+.2f}%/yr (5y gaps {instr_only_5y*100:+.2f}%); "
          f"US WHT drag 30% all {wht_all*100:.2f}%/yr, lev-only {wht_lev*100:.2f}%/yr")
    if full:
        for k, v in res["gap_decomposition_cagr"].items():
            print(f"    {k:100s} {v*100:+.2f}%")
        bc = res["band_check_US_1988_to_freeze"]
        print(f"  band check US 1988-2006: {C.fmt(bc['no_band_1988_2006'])}  ->  {C.fmt(bc['band_1988_2006'])}")
        print(f"  band check US 2007-frz : {C.fmt(bc['no_band_2007_freeze'])}  ->  {C.fmt(bc['band_2007_freeze'])}")
        bb = bc["bootstrap_band_minus_noband_1988_freeze"]
        print(f"  band - no band 1988-freeze CAGR {bb['mean']*100:+.2f}% [{bb['lo']*100:+.2f},{bb['hi']*100:+.2f}]; "
              f"lev turnover {bc['lev_turnover_per_year_1988_freeze_no_band']:.2f} -> {bc['lev_turnover_per_year_1988_freeze_band']:.2f}/yr")
    return res


US2 = dict(lev="qld", off="ief", gold="gold_phys")
US3 = dict(lev="tqqq", off="ief", gold="gold_phys")
SERIES = {}
OUT["strategy"] = {}
OUT["strategy"]["F2 (2x, vt .30): LQQ.PA + CBU0.L + IGLN.L vs QLD + IEF + GLD"] = strategy_block(
    "F2 primary", US2, dict(lev="u_lqq", off="u_cbu0", gold="u_igln"), 0.30)
OUT["strategy"]["F3 (3x, vt .20): QQQ3.L + CBU0.L + IGLN.L vs TQQQ + IEF + GLD"] = strategy_block(
    "F3 primary", US3, dict(lev="u_qqq3", off="u_cbu0", gold="u_igln"), 0.20)
OUT["strategy"]["F2 longer history: LQQ.PA + IBTM.L + PHAU.L vs QLD + IEF + GLD"] = strategy_block(
    "F2 longer history", US2, dict(lev="u_lqq", off="u_ibtm", gold="u_phau"), 0.30, full=False)

# 1x control: same F2 rules, unlevered Nasdaq sleeve, only the equity sleeve differs (bonds/gold US in both)
a1 = P2["u_cndx"].first_valid_index() + pd.Timedelta(days=1)
a1 = P2.index[P2.index >= a1][0]
r_q, _ = run("1x control: QQQ lag2 3bp (F2 rules)", "qqq", 0.30, "ief", "gold_phys", lag=2)
r_c, _ = run("1x control: CNDX.L lag2 3bp (F2 rules)", "u_cndx", 0.30, "ief", "gold_phys", lag=2)
sq, sc = C.stats(r_q, a1, END, start=None), C.stats(r_c, a1, END, start=None)
yq, yc = yearly(r_q, a1, END), yearly(r_c, a1, END)
OUT["control_1x_CNDX_vs_QQQ"] = dict(window=[str(a1.date()), str(END.date())], us_qqq_lag2=sq, ucits_cndx_lag2=sc,
                                     cagr_gap=sc["cagr"] - sq["cagr"],
                                     instrument_weekly_gap_CNDX_vs_QQQ=ov_full["CNDX.L"]["cagr_gap"],
                                     yearly_gap={y: yc[y] - yq[y] for y in yq},
                                     bootstrap=C.block_bootstrap_cagr_diff(r_c[:END], r_q[:END], start=a1))
print("  1x control yearly gap (CNDX - QQQ, pp): " + ", ".join(f"{y}:{(yc[y]-yq[y])*100:+.1f}" for y in yq))
print(f"\n=== 1x control {a1.date()}..{END.date()}: QQQ lag2 {C.fmt(sq)} | CNDX lag2 {C.fmt(sc)} | gap "
      f"{(sc['cagr']-sq['cagr'])*100:+.2f}%  (instrument gap {ov_full['CNDX.L']['cagr_gap']*100:+.2f}%)")

# ------------------------------------------------------------------ 5. ONE pre-specified cost mitigation: ER hysteresis
# ~2/3 of leveraged-sleeve turnover is the chop flag (ER20 vs 0.25) flipping. Pre-specified variant (NOT tuned):
# weekly-sampled hysteresis -- enter 'chop' only when ER20 < 0.20, leave it only when ER20 >= 0.30.
# Pre-stated adoption rule: at UCITS-level costs on 1988..freeze US history, CAGR AND Calmar must improve in BOTH
# halves (1988-2006, 2007-freeze) and the block-bootstrap 95% CI of the full-period CAGR difference must exclude 0.
def er_hysteresis(lo=0.20, hi=0.30):
    mk = P2["mark"].fillna(False).values
    er = P2["er20"].values
    st = np.full(len(P2), np.nan)
    cur = None
    for i in np.where(mk)[0]:
        e = er[i]
        if not np.isfinite(e):
            continue
        if cur is None:
            cur = 1.0 if e >= 0.25 else 0.0
        elif e >= hi:
            cur = 1.0
        elif e < lo:
            cur = 0.0
        st[i] = cur
    return pd.Series(st, index=P2.index).ffill()


P3 = P2.copy()
P3["er20"] = er_hysteresis()
_mk = P2["mark"].fillna(False) & (P2.index >= C.START) & (P2.index <= FREEZE)
_yrs = (FREEZE - C.START).days / 365.25
_f0 = (P2["er20"] >= 0.25).fillna(False)[_mk]
_f1 = (P3["er20"] >= 0.25).fillna(False)[_mk]
H1, H2 = ("1988-01-01", "2006-12-31"), ("2007-01-01", str(FREEZE.date()))
hyst = {"rule": "weekly ER20 hysteresis: chop when < 0.20, clean when >= 0.30, else keep previous state (pre-specified)",
        "adoption_rule": ("at UCITS-level costs on 1988..freeze US history: CAGR and Calmar better in BOTH halves AND "
                          "bootstrap 95% CI of the CAGR difference excludes 0"),
        "chop_flag_changes_per_year_1988_freeze": {"baseline": float((_f0 != _f0.shift()).iloc[1:].sum() / _yrs),
                                                   "hysteresis": float((_f1 != _f1.shift()).iloc[1:].sum() / _yrs)}}
for nm, us, uc, vt, lev_cost in [("F2 primary", US2, dict(lev="u_lqq", off="u_cbu0", gold="u_igln"), 0.30, 15e-4),
                                 ("F3 primary", US3, dict(lev="u_qqq3", off="u_cbu0", gold="u_igln"), 0.20, 20e-4)]:
    c_us_hi = {us["lev"]: lev_cost, us["off"]: 8e-4, us["gold"]: 8e-4}
    r0 = SERIES[nm]["R"]["US lag1 3bp"]
    rh, wh = run(f"{nm}: US lag1 3bp + ER hysteresis", us["lev"], vt, us["off"], us["gold"], lag=1, Pp=P3)
    r0c, _ = run(f"{nm}: US lag1 at UCITS-level costs", us["lev"], vt, us["off"], us["gold"], lag=1, costs=c_us_hi)
    rhc, _ = run(f"{nm}: US lag1 at UCITS-level costs + ER hysteresis", us["lev"], vt, us["off"], us["gold"], lag=1,
                 costs=c_us_hi, Pp=P3)
    ru0 = SERIES[nm]["R"]["UCITS lag2 realistic-cost"]
    ruh, _ = run(f"{nm}: UCITS lag2 realistic + ER hysteresis", uc["lev"], vt, uc["off"], uc["gold"], lag=2,
                 costs=COST_REAL, Pp=P3)
    a, b = SERIES[nm]["a"], SERIES[nm]["b"]
    blk = {}
    for tag, x0, x1 in [("US 3bp", r0, rh), ("US at UCITS-level costs", r0c, rhc)]:
        blk[tag] = {"1988_2006": {"base": C.stats(x0, *H1, start=None), "hyst": C.stats(x1, *H1, start=None)},
                    "2007_freeze": {"base": C.stats(x0, *H2, start=None), "hyst": C.stats(x1, *H2, start=None)},
                    "1988_freeze": {"base": C.stats(x0, H1[0], H2[1], start=None), "hyst": C.stats(x1, H1[0], H2[1], start=None)},
                    "oos_total": {"base": C.period_return(x0, OOS_A, b), "hyst": C.period_return(x1, OOS_A, b)},
                    "bootstrap_hyst_minus_base_1988_freeze": C.block_bootstrap_cagr_diff(x1[:FREEZE], x0[:FREEZE])}
    blk["UCITS window lag2 realistic"] = {"window": [str(a.date()), str(b.date())],
                                          "base": C.stats(ru0, a, b, start=None), "hyst": C.stats(ruh, a, b, start=None),
                                          "bootstrap_hyst_minus_base": C.block_bootstrap_cagr_diff(ruh[:b], ru0[:b], start=a)}
    hw = wh[(wh.index >= C.START) & (wh.index <= FREEZE)]
    blk["lev_turnover_per_year_1988_freeze_hyst"] = float(hw[us["lev"]].diff().abs().sum() / _yrs)
    hi_blk = blk["US at UCITS-level costs"]
    ok = all(hi_blk[h]["hyst"]["cagr"] > hi_blk[h]["base"]["cagr"] and hi_blk[h]["hyst"]["calmar"] > hi_blk[h]["base"]["calmar"]
             for h in ("1988_2006", "2007_freeze")) and hi_blk["bootstrap_hyst_minus_base_1988_freeze"]["lo"] > 0
    blk["passes_pre_stated_rule"] = bool(ok)
    hyst[nm] = blk
    print(f"\n=== ER hysteresis test {nm} (chop flips/yr {hyst['chop_flag_changes_per_year_1988_freeze']['baseline']:.1f} -> "
          f"{hyst['chop_flag_changes_per_year_1988_freeze']['hysteresis']:.1f}; lev turnover/yr hyst "
          f"{blk['lev_turnover_per_year_1988_freeze_hyst']:.2f}) ===")
    for tag in ("US 3bp", "US at UCITS-level costs"):
        for h in ("1988_2006", "2007_freeze", "1988_freeze"):
            print(f"  {tag:24s} {h:12s} base {C.fmt(blk[tag][h]['base'])}  |  hyst {C.fmt(blk[tag][h]['hyst'])}")
        bb = blk[tag]["bootstrap_hyst_minus_base_1988_freeze"]
        print(f"  {tag:24s} bootstrap hyst-base CAGR {bb['mean']*100:+.2f}% [{bb['lo']*100:+.2f},{bb['hi']*100:+.2f}]  "
              f"OOS {blk[tag]['oos_total']['base']*100:+.2f}% -> {blk[tag]['oos_total']['hyst']*100:+.2f}%")
    u = blk["UCITS window lag2 realistic"]
    print(f"  UCITS window: base {C.fmt(u['base'])} | hyst {C.fmt(u['hyst'])} | "
          f"boot {u['bootstrap_hyst_minus_base']['mean']*100:+.2f}% [{u['bootstrap_hyst_minus_base']['lo']*100:+.2f},"
          f"{u['bootstrap_hyst_minus_base']['hi']*100:+.2f}]   passes rule: {ok}")
OUT["er_hysteresis_test"] = hyst

# ------------------------------------------------------------------ 6. hybrid route (partial estate-tax protection)
# US leveraged ETF + UCITS bond/gold sleeves; every leg traded on Monday (lag 2) on sessions where London is open,
# US 3bp on the leveraged leg, 8bp on the UCITS legs.  Compared with US lag1 3bp over the same windows.
hyb = {}
for nm, us, vt, a_ in [("F2", US2, 0.30, SERIES["F2 primary"]["a"]), ("F3", US3, 0.20, SERIES["F3 primary"]["a"])]:
    rh_, wh_ = run(f"{nm} hybrid: US {us['lev'].upper()} + CBU0.L + IGLN.L, lag2, realistic", us["lev"], vt, "u_cbu0",
                   "u_igln", lag=2, costs={"u_cbu0": 8e-4, "u_igln": 8e-4})
    r0_ = SERIES[f"{nm} primary"]["R"]["US lag1 3bp"]
    hw_ = wh_[(wh_.index >= a_) & (wh_.index <= END)]
    hyb[nm] = dict(window=[str(a_.date()), str(END.date())], hybrid=C.stats(rh_, a_, END, start=None),
                   us_lag1_3bp=C.stats(r0_, a_, END, start=None),
                   cagr_gap=C.stats(rh_, a_, END, start=None)["cagr"] - C.stats(r0_, a_, END, start=None)["cagr"],
                   bootstrap=C.block_bootstrap_cagr_diff(rh_[:END], r0_[:END], start=a_),
                   us_situs_weight_avg=float(hw_[us["lev"]].mean()), us_situs_weight_max=float(hw_[us["lev"]].max()),
                   oos_total=C.period_return(rh_, OOS_A, END))
    print(f"\n=== hybrid {nm} {a_.date()}..{END.date()}: {C.fmt(hyb[nm]['hybrid'])}  vs US lag1 {C.fmt(hyb[nm]['us_lag1_3bp'])}"
          f"  gap {hyb[nm]['cagr_gap']*100:+.2f}% [{hyb[nm]['bootstrap']['lo']*100:+.2f},{hyb[nm]['bootstrap']['hi']*100:+.2f}]"
          f"  US-situs weight avg {hyb[nm]['us_situs_weight_avg']:.2f} max {hyb[nm]['us_situs_weight_max']:.2f}")
OUT["hybrid_US_leveraged_plus_UCITS_bonds_gold"] = hyb

# ------------------------------------------------------------------ 7. instruments since the freeze (out-of-sample)
oos_inst = {}
for tk in ["LQQ.PA", "QQQ3.L", "CNDX.L", "CBU0.L", "DTLA.L", "IB01.L", "IGLN.L", "SGLD.L", "PHAU.L"]:
    s = CLEAN[tk]
    a_lvl = us_level(CAT[tk]["us"])
    u0, u1 = on_dates(s, pd.DatetimeIndex([FREEZE, END])).values
    v0, v1 = on_dates(a_lvl, pd.DatetimeIndex([FREEZE, END])).values
    oos_inst[tk] = {"ucits_usd": float(u1 / u0 - 1), CAT[tk]["us_tk"]: float(v1 / v0 - 1),
                    "gap_pp": float((u1 / u0) - (v1 / v0))}
OUT["instruments_since_freeze_2026_07_24_to_2026_11_16"] = {
    "note": "European close (~11:35 ET) vs US close at both ends; indicative only", **oos_inst}
print("\n  since freeze: " + " | ".join(f"{k} {v['ucits_usd']*100:+.1f}% vs {CAT[k]['us_tk']} "
                                     f"{v[CAT[k]['us_tk']]*100:+.1f}%" for k, v in oos_inst.items()))

last = P.loc[:END].iloc[-1]
OUT["current_state"] = dict(asof=str(P.loc[:END].index[-1].date()), above_ema100=bool(last["above"]),
                            ndx=float(last["ndx"]), ema100=float(last["ema100"]),
                            ucits_allocation_F2_F3=({"CBU0.L (UST 7-10y, USD Acc)": 0.85, "IGLN.L (physical gold ETC, USD)": 0.15}
                                                    if not bool(last["above"]) else "risk-on: see scanner"))
_f2 = OUT["strategy"]["F2 (2x, vt .30): LQQ.PA + CBU0.L + IGLN.L vs QLD + IEF + GLD"]
_f3 = OUT["strategy"]["F3 (3x, vt .20): QQQ3.L + CBU0.L + IGLN.L vs TQQQ + IEF + GLD"]
_fl = OUT["strategy"]["F2 longer history: LQQ.PA + IBTM.L + PHAU.L vs QLD + IEF + GLD"]
OUT["summary"] = {
    "viable_mapping": {"2x Nasdaq-100": "LQQ.PA (only 2x Nasdaq-100 UCITS found; leverage on the USD index, EUR is only the quote currency)",
                       "UST 7-10y": "CBU0.L", "UST 20y+": "DTLA.L", "UST 0-1y": "IB01.L",
                       "gold": "IGLN.L (ETC; SGLD.L / PHAU.L equivalent)",
                       "3x Nasdaq-100": "QQQ3.L tradable but NOT recommended (non-UCITS ETP issuer risk, thin, last-5y drag)",
                       "not viable / not verifiable": "3QQQ.L (tiny AUM), IBTS.L (Yahoo data unusable), Dist GBP lines for backtests"},
    "instrument_cagr_gap_full_history": {k: ov_full[k]["cagr_gap"] for k in ("LQQ.PA", "QQQ3.L", "CBU0.L", "DTLA.L", "IB01.L", "IGLN.L")},
    "instrument_cagr_gap_last_5y": {k: ov_5y[k]["cagr_gap"] for k in ("LQQ.PA", "QQQ3.L", "CBU0.L", "DTLA.L", "IB01.L", "IGLN.L")},
    "F2_total_measured_gap": {"2011-2026 (CBU0/IGLN)": _f2["gap_decomposition_cagr"]["TOTAL measured (UCITS lag2 realistic - US lag1 3bp) = A+B+C"],
                              "2009-2026 (IBTM/PHAU)": _fl["cagr_gap_vs_US_lag1_3bp"]["UCITS lag2 realistic-cost"]},
    "F3_total_measured_gap_2013_2026": _f3["gap_decomposition_cagr"]["TOTAL measured (UCITS lag2 realistic - US lag1 3bp) = A+B+C"],
    "F2_extra_trading_cost_central": _f2["extra_trading_cost_drag_analytic"]["central"],
    "F3_extra_trading_cost_central": _f3["extra_trading_cost_drag_analytic"]["central"],
    "hybrid_gap": {k: v["cagr_gap"] for k, v in hyb.items()},
    "mitigations_tested": {"5pp no-trade band": "no effect (turnover is chop flips / risk switches, not vol rescaling)",
                           "ER hysteresis 0.20/0.30": "fails pre-stated both-halves rule"},
    "look_ahead_warning": "European closes backtested at lag1 overstate CAGR by "
                          f"{_f2['gap_decomposition_cagr']['memo: look-ahead artifact if European closes are backtested at lag1 (UCITS lag1 - UCITS lag2)']*100:.1f} (F2) / "
                          f"{_f3['gap_decomposition_cagr']['memo: look-ahead artifact if European closes are backtested at lag1 (UCITS lag1 - UCITS lag2)']*100:.1f} (F3) pts/yr",
}
OUT["meta"] = dict(script="update_2026_11/ucits.py", helper="update_2026_11/ucits_fetch.py", fetched_utc=FETCHED,
                   end=str(END.date()), weekly_end=str(WEND.date()), freeze=str(FREEZE.date()),
                   fx_conversion="log-interpolated 16.5/24 between Yahoo FX bar D (~00:00 London) and the next bar",
                   execution=("lag1 = trade at the signal (Friday US) close, July convention; lag2 = trade at the next "
                              "session close. European listings close ~11:35 ET, before the signal exists, so lag2 "
                              "(Monday European close) is the earliest honest execution."),
                   variants_tested=len(VARIANTS), variants=VARIANTS,
                   overlap_pairs_measured=int(sum(1 for v in ov_full.values() if v)))
json.dump(rnd(OUT), open(os.path.join(HERE, "ucits.json"), "w"), indent=1)
print(f"\nvariants_tested (strategy-level backtests) = {len(VARIANTS)}")
print("saved update_2026_11/ucits.json")
