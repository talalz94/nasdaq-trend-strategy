"""
r2_verify_chop_vol_matched_robustness_lib.py -- side-effect-free numpy engine for the ROBUSTNESS review of
r2_chop_vol_matched (same mechanics as common.py / main_tranche_check.py; validated to 3e-11 against the
specialist's saved results in r2_verify_chop_vol_matched_robustness_checks.py). Import only; writes nothing.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P = C.panel()
IDX = P.index
N = len(IDX)
START, FREEZE = C.START, C.FREEZE
DOW = IDX.dayofweek.values
LASTS = [np.maximum.accumulate(np.where(DOW == i, np.arange(N), -1)) for i in range(5)]
SEASON = np.where(np.isin(P["month"].values, (8, 9)), 0.5, 1.0)
R_IEF = P["ief"].fillna(0).values
R_GOLD = P["gold_phys"].fillna(0).values
RF_D = P["rf_d"].fillna(0).values


def msk(a, b):
    return (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))


def market(mult, r_lev, above, vol20, er20):
    er20 = np.asarray(er20, float)
    return dict(mult=float(mult), r_lev=np.nan_to_num(np.asarray(r_lev, float)), above=np.asarray(above, bool),
                vol20=np.asarray(vol20, float), clean=(~np.isnan(er20)) & (er20 >= 0.25))


def wlev(M, vt=None, chop=True, vol=True, c=1.0):
    if vol:
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.clip(vt / M["vol20"], 0, 1)
        sc = np.where(np.isnan(sc), 1.0, sc)
    else:
        sc = np.ones(N)
    sc = np.clip(np.clip(sc * SEASON, 0, 1) * c, 0, 1)
    eq = np.where(M["above"], np.where(M["clean"] | (not chop), sc, 0.5 * sc), 0.0)
    return 0.85 * eq


def held(w, li, lag=1):
    h = np.where(li >= 0, w[np.maximum(li, 0)], 0.0)
    out = np.zeros(N)
    out[lag:] = h[:-lag]
    return out


def books(basis):
    return list(range(5)) if basis == "tranche" else [4]


def ret(M, wl, basis="tranche", lag=1, cost=3e-4):
    bk = books(basis)
    r = np.zeros(N)
    wo, wg = np.clip(0.85 - wl, 0, None), np.full(N, 0.15)
    for d in bk:
        li = LASTS[d]
        hl, ho, hg = held(wl, li, lag), held(wo, li, lag), held(wg, li, lag)
        tn = np.abs(np.diff(hl, prepend=0.0)) + np.abs(np.diff(ho, prepend=0.0)) + np.abs(np.diff(hg, prepend=0.0))
        r += (hl * M["r_lev"] + ho * R_IEF + hg * R_GOLD - tn * cost) / len(bk)
    return r


def expo(M, wl, basis, a, b, lag=1):
    h = sum(held(wl, LASTS[d], lag) for d in books(basis)) / len(books(basis))
    return float(M["mult"] * h[msk(a, b)].mean())


def calmar(r, a, b):
    m = msk(a, b)
    rr, d = r[m], IDX[m]
    e = np.cumprod(1 + rr)
    cg = e[-1] ** (1 / ((d[-1] - d[0]).days / 365.25)) - 1
    return float(cg / abs((e / np.maximum.accumulate(e) - 1).min()))


def bisect(f, lo, hi, tol=1e-10, it=200):
    flo, fhi = f(lo), f(hi)
    assert flo < 0 <= fhi, (flo, fhi)
    k = 0
    while hi - lo > tol and k < it:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        k += 1
    return 0.5 * (lo + hi)


def match_expo(M, vt_b, basis, a, b, lag=1):
    tgt = expo(M, wlev(M, vt_b), basis, a, b, lag)
    vtA = bisect(lambda v: expo(M, wlev(M, v, chop=False), basis, a, b, lag) - tgt, 1e-4, vt_b)
    cV = tgt / expo(M, wlev(M, None, vol=False), basis, a, b, lag)
    return vtA, cV


def series(M, vt_b, vtA, cV, basis="tranche", lag=1, cost=3e-4):
    return dict(B=ret(M, wlev(M, vt_b), basis, lag, cost), A=ret(M, wlev(M, vtA, chop=False), basis, lag, cost),
                V=ret(M, wlev(M, None, vol=False, c=cV), basis, lag, cost))


NDXM = {"F2": market(2, P["qld"].values, P["above"].values, P["vol20"].values, P["er20"].values),
        "F3": market(3, P["tqqq"].values, P["above"].values, P["vol20"].values, P["er20"].values)}


def cross_markets(mult=2):
    """the 7 registered cross-market indices, built exactly as r2_chop_vol_matched.py builds them."""
    yfd = pd.read_parquet(os.path.join(HERE, "r2_chop_vol_matched_yf.parquet"))
    XM = {"spx": ("panel", "spx"), "rut": ("panel", "rut"), "tsx": ("panel", "tsx"), "hsi": ("panel", "hsi"),
          "dax": ("yfinance", "^GDAXI"), "n225": ("yfinance", "^N225"), "ixic": ("yfinance", "^IXIC")}
    out = {}
    for nm, (src, col) in XM.items():
        if src == "panel":
            r = np.nan_to_num(P[col].values.astype(float))
        else:
            s = yfd[yfd["ticker"] == col]
            c = pd.Series(s["close"].values, index=pd.DatetimeIndex(s["date"])).sort_index()
            c = c[~c.index.duplicated(keep="last")]
            lv = c.reindex(c.index.union(IDX)).ffill().reindex(IDX)
            r = lv.pct_change().values
            r[IDX <= c.index[0]] = np.nan
            r = np.nan_to_num(r)
        price = pd.Series(np.cumprod(1.0 + r), index=IDX)
        ema = price.ewm(span=100, adjust=False).mean()
        v20 = price.pct_change().rolling(20).std() * np.sqrt(252)
        e20 = price.diff(20).abs() / price.diff().abs().rolling(20).sum().replace(0, np.nan)
        out[nm] = market(mult, mult * r - 0.0095 / 252 - (mult - 1) * RF_D, (price > ema).values, v20.values, e20.values)
    return out
