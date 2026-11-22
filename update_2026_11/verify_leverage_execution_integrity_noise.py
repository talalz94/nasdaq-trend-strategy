"""
INTEGRITY REVIEW of leverage_execution  --  step 4: is the Friday premium carried by marginal near-EMA decisions?
 N1  pre-close signal noise N(0,s) applied to EVERY weekday phase (not only Friday) and to the tranche
 N2  hysteresis: at a mark, if |NDX/EMA-1| < x keep the previous on/off state (marginal decisions removed)
     -> Friday minus mean(other 4) and Friday minus tranche as x grows
Read-only (prints only).  usage: python update_2026_11/verify_leverage_execution_integrity_noise.py
"""
import sys, os, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

T0 = time.time()
P = C.panel(); IDX = P.index; T = len(IDX)
IS = (IDX >= C.START) & (IDX <= C.FREEZE)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MK = {d: (IDX.dayofweek == i) for i, d in enumerate(DAYS)}
ndx = P["ndx"].to_numpy(); ema = P["ema100"].to_numpy(); ema_prev = np.r_[ema[0], ema[:-1]]
above = P["above"].to_numpy().astype(bool)
vol = P["vol20"].to_numpy(); er = P["er20"].to_numpy(); mon = P["month"].to_numpy()
clean = np.nan_to_num(er, nan=-1) >= 0.25


def weights_np(ron, lev, vt):
    scale = np.clip(np.nan_to_num(vt / vol, nan=1.0), 0, 1)
    scale = np.clip(scale * np.where(np.isin(mon, (8, 9)), 0.5, 1.0), 0, 1)
    eq = ron * scale * np.where(clean, 1.0, 0.5)
    W = np.column_stack([0.85 * eq, np.clip(0.85 - 0.85 * eq, 0, None), np.full(T, 0.15)])
    R = np.nan_to_num(P[[lev, "ief", "gold_phys"]].to_numpy(float))
    return W, R


def held_np(Wm, marks, lag=1):
    ii = np.where(marks, np.arange(T), -1)
    lm = np.maximum.accumulate(ii)
    src = np.full(T, -1); src[lag:] = lm[:-lag]
    return np.where((src >= 0)[:, None], Wm[np.maximum(src, 0)], 0.0)


def bt(H, R, cost=C.COST):
    return (H * R).sum(1) - np.r_[0.0, np.abs(np.diff(H, axis=0)).sum(1)] * cost


def cagr(r):
    x = r[IS]; d = IDX[IS]
    return np.prod(1 + x) ** (365.25 / (d[-1] - d[0]).days) - 1


# sanity: reproduce common for F2 Friday
W, R = weights_np(above.astype(float), "qld", 0.30)
ref = C.backtest(P, C.strategy_weights(P, lev="qld", voltgt=0.30)).to_numpy()
print(f"sanity numpy weights vs common: max|diff| {np.abs(bt(held_np(W, MK['Fri']), R) - ref).max():.2e}")

rng = np.random.default_rng(7)
for nm, lev, vt in (("F2", "qld", 0.30), ("F3", "tqqq", 0.20)):
    W0, R = weights_np(above.astype(float), lev, vt)
    base = {d: cagr(bt(held_np(W0, MK[d]), R)) for d in DAYS}
    Htr0 = sum(held_np(W0, MK[d]) for d in DAYS) / 5
    base["tranche"] = cagr(bt(Htr0, R))
    print(f"\n{nm} exact-close CAGR: " + " ".join(f"{k} {v*100:.2f}" for k, v in base.items()))
    # ---------------- N1 noise on every phase
    for s in (0.002, 0.004):
        acc = {k: [] for k in base}
        for i in range(150):
            e = rng.normal(0, s, T)
            ab = (ndx * (1 + e)) > ema_prev        # decision on a pre-close price, every day independently
            Wn, _ = weights_np(ab.astype(float), lev, vt)
            Hs = {}
            for d in DAYS:
                # only that phase's mark-day decisions use the noisy price
                Hs[d] = held_np(Wn, MK[d])
                acc[d].append(cagr(bt(Hs[d], R)))
            acc["tranche"].append(cagr(bt(sum(Hs.values()) / 5, R)))
        print(f"  N1 s={s*100:.1f}%  mean CAGR change vs exact close (pp): " +
              " ".join(f"{k} {(np.mean(v)-base[k])*100:+.2f}" for k, v in acc.items()) +
              f" | Fri - tranche under noise {(np.mean(acc['Fri'])-np.mean(acc['tranche']))*100:+.2f}pp"
              f" (exact {(base['Fri']-base['tranche'])*100:+.2f})")
    # ---------------- N2 hysteresis
    dist = ndx / ema - 1
    for x in (0.0, 0.0025, 0.005, 0.01, 0.02):
        res = {}
        Hs = {}
        for d in DAYS:
            ron = np.zeros(T)
            state = 0.0
            idx_m = np.where(MK[d])[0]
            st_series = np.full(T, np.nan)
            for t in idx_m:
                if abs(dist[t]) >= x or t == idx_m[0]:
                    state = 1.0 if above[t] else 0.0
                st_series[t] = state
            st_series = pd.Series(st_series).ffill().fillna(0).to_numpy()
            Wh, _ = weights_np(st_series, lev, vt)
            Hs[d] = held_np(Wh, MK[d])
            res[d] = cagr(bt(Hs[d], R))
        res["tranche"] = cagr(bt(sum(Hs.values()) / 5, R))
        oth = np.mean([res[d] for d in DAYS[:4]])
        print(f"  N2 hysteresis {x*100:4.2f}%: " + " ".join(f"{k} {v*100:5.2f}" for k, v in res.items()) +
              f" | Fri-mean(others) {(res['Fri']-oth)*100:+.2f}pp  Fri-tranche {(res['Fri']-res['tranche'])*100:+.2f}pp")
print(f"\n[{time.time()-T0:.0f}s] done")
