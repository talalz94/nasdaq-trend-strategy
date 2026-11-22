"""
Integrity reviewer: independent re-implementation of criterion-B markets through common.strategy_weights /
common.backtest (pandas), with
  X1  IEF pre-1985 rebuild calibrated ONLY on real IEF <= 2026-07-24 (specialist recovered the panel's full-sample fit)
  X2  sensitivity: pre-1985 gold stand-in = T-bills (specialist) vs zero-return cash
  X3  per-market Fri-minus-tranche (CAGR pp, Calmar) and pass/fail vs the JSON
Writes only to the scratch dir.
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa

SCR = os.environ.get("VERIFY_SCRATCH", os.path.join(os.path.expanduser("~"), "r2_verify_scratch"))
J = json.load(open(os.path.join(HERE, "r2_friday_effect.json")))
P = C.panel(); IDX = P.index; FREEZE = C.FREEZE
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
Y = pd.read_parquet(os.path.join(HERE, "r2_friday_effect_yf.parquet"))


def ys(t):
    s = Y[Y.ticker == t].set_index("date")["close"].sort_index()
    s = s[~s.index.duplicated(keep="last")].astype(float)
    return s[s.index <= IDX[-1]]


# ---- X1: IEF calibration on real IEF up to the freeze only
yd = P["tnx"] / 100.0
df = pd.concat([P["ief_real"], yd.shift(1) / 252, yd.diff()], axis=1).dropna()
df.columns = ["r", "carry", "dy"]
df = df[df.index <= FREEZE]
X = np.column_stack([np.ones(len(df)), -df.dy.values, 0.5 * df.dy.values ** 2])
a_f, D_f, C_f = np.linalg.lstsq(X, (df.r - df.carry).values, rcond=None)[0]
cal_full = J["c_meta"]["ief_calibration_recovered"]
print("X1 calib <=FREEZE alpha %.5f dur %.3f conv %.1f | specialist (full panel) alpha %.5f dur %.3f conv %.1f"
      % (a_f * 252, D_f, C_f, cal_full["alpha_ann"], cal_full["duration"], cal_full["convexity"]), flush=True)

irx, tnx = ys("^IRX"), ys("^TNX")
yt = tnx / 100.0
pre = IDX[0]


def us_streams(gold_mode="bills"):
    ief_pre = (a_f + yt.shift(1) / 252 - D_f * yt.diff() + 0.5 * C_f * yt.diff() ** 2)
    ief_pre = ief_pre[ief_pre.index < pre].fillna(0)
    ief = pd.concat([ief_pre, P["ief"].fillna(0)]).sort_index()
    if gold_mode == "bills":
        gpre = irx[irx.index < pre] / 100.0 / 252
    else:
        gpre = irx[irx.index < pre] * 0.0
    gold = pd.concat([gpre, P["gold_phys"].fillna(0)]).sort_index()
    rf = pd.concat([irx[irx.index < pre] / 100.0, P["rf"]]).sort_index()
    return (1 + ief).cumprod(), (1 + gold).cumprod(), rf


def asof(s, dates):
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def build(dates, ret, lvl, rf_d, ief, gold):
    Q = pd.DataFrame(index=dates)
    Q["lev"] = 2 * ret - 0.0095 / 252 - rf_d
    Q["ief"] = ief; Q["gold_phys"] = gold
    l = pd.Series(lvl, index=dates)
    ema = l.ewm(span=100, adjust=False).mean()
    Q["above"] = l > ema
    Q["vol20"] = l.pct_change().rolling(20).std() * np.sqrt(252)
    Q["er20"] = l.diff(20).abs() / l.diff().abs().rolling(20).sum().replace(0, np.nan)
    Q["month"] = dates.month
    return Q


def evaluate(Q, start):
    w = C.strategy_weights(Q, lev="lev", voltgt=0.30, gold="gold_phys", off="ief")
    rs = {d: C.backtest(Q, w, mark=pd.Series(Q.index.dayofweek == i, index=Q.index)) for i, d in enumerate(DAYS)}
    rs["tranche"] = sum(rs[d] for d in DAYS) / 5
    s = {k: C.stats(v, a=start, b=FREEZE, start=None) for k, v in rs.items()}
    return dict(d_cagr_pp=(s["Fri"]["cagr"] - s["tranche"]["cagr"]) * 100,
                d_calmar=s["Fri"]["calmar"] - s["tranche"]["calmar"],
                pass_both=bool(s["Fri"]["cagr"] > s["tranche"]["cagr"] and s["Fri"]["calmar"] > s["tranche"]["calmar"]),
                fri=[round(s["Fri"]["cagr"] * 100, 2), round(s["Fri"]["maxdd"] * 100, 1), round(s["Fri"]["calmar"], 4)],
                tranche=[round(s["tranche"]["cagr"] * 100, 2), round(s["tranche"]["maxdd"] * 100, 1), round(s["tranche"]["calmar"], 4)])


res = {}
for gold_mode in ("bills", "zero"):
    IEF_IDX, GOLD_IDX, RF = us_streams(gold_mode)
    out = {}
    for col in ("spx", "rut", "tsx", "hsi"):
        r = P[col].fillna(0)
        nz = np.where(r.values != 0)[0]
        start = max(C.START, IDX[min(nz[0] + 250, len(IDX) - 1)])
        Q = build(IDX, r.values, np.cumprod(1 + r.values), P["rf_d"].values, P["ief"].fillna(0).values,
                  P["gold_phys"].fillna(0).values)
        out[col.upper()] = dict(start=str(start.date()), **evaluate(Q, start))
    for name, t, fixed in (("DAX", "^GDAXI", None), ("N225", "^N225", "1975-01-01"), ("IXIC", "^IXIC", "1975-01-01"),
                           ("HSI-own", "^HSI", None)):
        c = ys(t); c = c[c > 0]
        dates = c.index
        start = pd.Timestamp(fixed) if fixed else max(C.START, dates[250])
        r = c.pct_change().fillna(0).values
        rf_d = (asof(RF, dates).bfill() / 252).values
        ief = asof(IEF_IDX, dates).pct_change().fillna(0).values
        gold = asof(GOLD_IDX, dates).pct_change().fillna(0).values
        Q = build(dates, r, c.values, rf_d, ief, gold)
        out[name] = dict(start=str(start.date()), **evaluate(Q, start))
    res[gold_mode] = out
    print("gold pre-1985 =", gold_mode, flush=True)
    for k, v in out.items():
        jk = {"HSI-own": "HSI-own-calendar (robustness)"}.get(k, k)
        jw = J["c_cross_market"][jk]["windows"]["IS"]
        print(f"  {k:8s} start {v['start']}  mine Fri-tr {v['d_cagr_pp']:+.3f}pp {v['d_calmar']:+.5f} pass {v['pass_both']} | "
              f"JSON {jw['fri_minus_tranche_cagr_pp']:+.3f}pp {jw['fri_minus_tranche_calmar']:+.5f} pass {jw['fri_beats_tranche_both']}",
              flush=True)
    n = sum(out[m]["pass_both"] for m in ("SPX", "RUT", "TSX", "HSI", "DAX", "N225", "IXIC"))
    print("  criterion-B count:", n, flush=True)
    res[gold_mode]["B_count"] = int(n)

json.dump(dict(calib_le_freeze=dict(alpha_ann=a_f * 252, duration=D_f, convexity=C_f), results=res),
          open(os.path.join(SCR, "r2_verify_friday_effect_integrity_xmkt.json"), "w"), indent=1, default=float)
print("saved")
