"""
verify_managed_futures_integrity_episode.py  (adversarial review, integrity lens)

How much of the managed_futures verdicts rests on the single 1989-10..1990-05 episode that sits at
the very start of the in-sample window (where TQQQ/QLD are simulated, gold is interpolated, and the
proxy trades markets with no futures contract at the time)?
Uses the series pickled by verify_managed_futures_integrity_rerun.py; prints only.
NOTE: alternative start dates are a SENSITIVITY, not a proposal (choosing a start date ex post is
itself a selection).
"""
import os, sys, pickle
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "update_2026_11")
sys.path.insert(0, HERE)
import common as C  # noqa: E402

SCR = sys.argv[1] if len(sys.argv) > 1 else "."
S = pickle.load(open(os.path.join(SCR, "mf_series.pkl"), "rb"))
P = C.panel()
idx = P.index
FREEZE = C.FREEZE
allp = {**S["port"], **S["rob"]}

def cal(r, a, b=FREEZE):
    return C.stats(r, a, b, start=None)["calmar"]

def boot(r1, r2, a, b=FREEZE, n=2000, block=63, seed=0):
    x = pd.concat([r1, r2], axis=1).dropna()
    x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].values
    T = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); ar = np.arange(block)
    out = np.empty(n)
    for k in range(n):
        st = rng.integers(0, T, nb)
        ix = (st[:, None] + ar[None, :]).ravel()[:T] % T
        s = x[ix]; e = np.cumprod(1 + s, axis=0)
        g = e[-1] ** (252 / T) - 1
        dd = (e / np.maximum.accumulate(e, axis=0) - 1).min(axis=0)
        c = g / np.abs(dd); out[k] = c[0] - c[1]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())

def ulcer_martin(r, a="1988-01-01", b=FREEZE):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    e = (1 + s).cumprod(); dd = e / e.cummax() - 1
    ui = float(np.sqrt((dd ** 2).mean()))
    cagr = C.stats(r, a, b, start=None)["cagr"]
    return ui, cagr / ui

def exsharpe(r, a="1988-01-01", b=FREEZE):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    x = (r - P["rf_d"])[m]
    return float(x.mean() / x.std() * np.sqrt(252))

def top3(r, a="1988-01-01", b=FREEZE):
    s = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    e = (1 + s).cumprod(); dd = e / e.cummax() - 1
    used = pd.Series(False, index=dd.index); out = []
    for _ in range(3):
        d2 = dd[~used]
        tr = d2.idxmin(); pk = e[:tr].idxmax(); after = e[tr:]; rec = after[after >= e[pk]]
        end = rec.index[0] if len(rec) else dd.index[-1]
        out.append(float(dd[tr])); used[(dd.index >= pk) & (dd.index <= end)] = True
    return out

# ---------------------------------------------------------------- 1. off-leg swap: robust metrics, full IS
print("=== 1. off-leg swap across the 22 proxy variants, FULL IS 1988..2026-07-24: metrics less dominated by one episode ===")
for eng in ["F3", "F2"]:
    b = S["base"][eng]
    ub, mb = ulcer_martin(b); sb = exsharpe(b); t3b = np.mean(top3(b))
    rows = {k: v for k, v in allp.items() if k.startswith(f"{eng} | replace_ief_offleg | ") and not k.endswith("control_always_long")}
    dm, ds, dt3, du = [], [], [], []
    for k, v in rows.items():
        u, m = ulcer_martin(v); dm.append(m - mb); ds.append(exsharpe(v) - sb); du.append(u - ub)
        dt3.append(np.mean(top3(v)) - t3b)
    dm, ds, dt3, du = map(np.array, (dm, ds, dt3, du))
    print(f"  {eng}: dMartin(CAGR/Ulcer) median {np.median(dm):+.3f} (#<0 {int((dm<0).sum())}/22)  "
          f"dUlcer median {np.median(du)*100:+.2f}pp (#worse {int((du>0).sum())}/22)  "
          f"d(rf-adj Sharpe) median {np.median(ds):+.3f} (#<0 {int((ds<0).sum())}/22)  "
          f"d(avg of 3 deepest DDs) median {np.median(dt3)*100:+.2f}pp (#deeper {int((dt3<0).sum())}/22)")

# ---------------------------------------------------------------- 2. start-date sensitivity incl. bootstrap + T-bill control
print("\n=== 2. start-date sensitivity (main-grid rows), dCalmar vs baseline; H1' = start..2006, H2 = 2007..freeze ===")
STARTS = ["1988-01-01", "1990-06-01", "1991-01-01"]
for a in STARTS:
    print(f"-- comparison window {a} .. 2026-07-24")
    for eng in ["F3", "F2"]:
        b = S["base"][eng]
        bc, bh1, bh2 = cal(b, a), cal(b, a, "2006-12-31"), cal(b, "2007-01-01")
        for pk in ["main_12m", "blend_1_3_12m"]:
            for st in ["sleeve10", "sleeve20", "sleeve30", "replace_gold", "replace_ief_offleg", "offleg_50ief_50tsmom"]:
                r = S["port"][f"{eng} | {st} | {pk}"]
                d = cal(r, a) - bc; h1 = cal(r, a, "2006-12-31") - bh1; h2 = cal(r, "2007-01-01") - bh2
                if st.startswith("sleeve") or st == "replace_ief_offleg":
                    lo, hi, p = boot(r, b, a)
                    verdict = "PASS" if (d > 0.15 and h1 > 0 and h2 > 0 and lo > 0) else ("MARG" if (d > 0.07 and h1 > 0 and h2 > 0) else "FAIL")
                    extra = ""
                    if st.startswith("sleeve"):
                        cash = S["cash"][f"{eng} | {st} | CONTROL bills (plain de-lever)"]
                        extra = f"  vs same-size T-bill sleeve {cal(r, a) - cal(cash, a):+.3f}"
                    print(f"   {eng} {st:20s} {pk:14s} dCal {d:+.3f}  H1' {h1:+.3f} H2 {h2:+.3f}  boot [{lo:+.3f},{hi:+.3f}] p<=0 {p:.2f} -> {verdict}{extra}")

# ---------------------------------------------------------------- 3. what happened in the 1989-10-09..1990-05-02 episode
print("\n=== 3. the 1989-10-09 .. 1990-05-02 episode (F3 baseline peak->trough) ===")
a, b = "1989-10-10", "1990-05-02"
for nm in ["main_12m", "blend_1_3_12m", "ex_equities", "control_always_long", "gold_sig_miners_pre2000", "drop_intl", "drop_ndx"]:
    r = S["proxies"][nm]
    print(f"  proxy {nm:24s} {C.period_return(r, a, b)*100:+6.1f}%")
for c in ["ief", "tlt", "bills", "spx", "ndx_ret", "intl", "gold_phys", "dxy_ret", "gsci_ret"]:
    print(f"  {c:10s} {C.period_return(P[c], a, b)*100:+6.1f}%")
for eng in ["F3", "F2"]:
    for k in [f"{eng} | replace_ief_offleg | main_12m"]:
        print(f"  {eng} baseline {C.period_return(S['base'][eng], a, b)*100:+6.1f}%   off-leg TSMOM {C.period_return(S['port'][k], a, b)*100:+6.1f}%")
# per-market contribution of main_12m in the episode (positions x excess returns)
pos = S["proxy_pos"]["main_12m"]
held = pos.reindex(idx).ffill().shift(1).fillna(0.0)
X = pd.DataFrame({"spx": P["spx"] - P["rf_d"], "ndx": P["ndx_ret"] - P["rf_d"], "intl": P["intl"] - P["rf_d"],
                  "ief": P["ief"] - P["rf_d"], "tlt": P["tlt"] - P["rf_d"], "gold": P["gold_phys"] - P["rf_d"],
                  "dxy": P["dxy_ret"], "gsci": P["gsci_ret"]}).fillna(0.0)
m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
contrib = (held * X)[m].sum()
print("  main_12m gross contribution by market in the episode (sum, %):", {k: round(v * 100, 1) for k, v in contrib.items()})
print("  main_12m average held notional in the episode:", {k: round(v, 2) for k, v in held[m].mean().items()})

# ---------------------------------------------------------------- 4. C3 mechanism: where the risk-on excess return comes from
print("\n=== 4. C3: main_12m gross contribution by market on F risk-on vs risk-off days (IS, %/yr) ===")
ron = P["above"].astype(float).where(P["mark"]).ffill().shift(1).fillna(0.0) > 0.5
mis = (idx >= C.START) & (idx <= FREEZE)
for lbl, mask in (("risk-on", ron & mis), ("risk-off", (~ron) & mis)):
    c = (held * X)[mask].mean() * 252
    eq = c[["spx", "ndx", "intl"]].sum()
    print(f"  {lbl:8s} total gross {c.sum()*100:5.2f}%/yr  equity legs {eq*100:5.2f}%/yr ({eq/c.sum()*100:4.0f}%)  "
          + "  ".join(f"{k} {v*100:+.2f}" for k, v in c.items()))
