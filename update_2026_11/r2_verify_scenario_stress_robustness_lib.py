"""
r2_verify_scenario_stress_robustness_lib.py -- helper for the ROBUSTNESS review of r2_scenario_stress.

* load_specialist(): exec()s r2_scenario_stress.py only up to its '# ---- VALIDATION' marker (definitions,
  calibration, calendar, scenario skeletons, synth_path, evaluate) -- nothing is written, no JSON touched.
* Fast: a numpy re-implementation of the forward books (3 tiers x 5 weekday books, lag 1, 3 bp,
  85% IEF / 15% gold_phys off-leg, July rules), used for the large seed / design / grid runs. It is
  validated against the specialist's own evaluate() (pandas, common.strategy_weights/backtest) before use.
* path generators that reproduce synth_path() exactly for design A/B and add reviewer variants:
  'free'  = pins only at the scenario's own waypoints, NO day-252 pin (free random walk after the last pin)
  'boot'  = design-A bridges, but NDX noise = circular block bootstrap of standardised in-sample NDX log returns
"""
from __future__ import annotations
import os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_scenario_stress.py")
_G = None


def load_specialist():
    global _G
    if _G is not None:
        return _G
    code = open(SRC, encoding="utf-8").read()
    marker = "# ------------------------------------------------------------------ VALIDATION"
    assert code.count(marker) == 1
    head = code.split(marker)[0]
    assert "json.dump" not in head and "open(OUT" not in head
    g = {"__file__": SRC, "__name__": "__spec_head__"}
    exec(compile(head, SRC, "exec"), g)
    _G = g
    return g


TIER = {"F1": ("qqq", 0.20, 1), "F2": ("qld", 0.30, 2), "F3": ("tqqq", 0.20, 3)}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


class Fast:
    def __init__(self):
        g = load_specialist()
        self.g = g
        P = g["P"]
        self.L0, self.EMA0, self.RFD, self.EXP, self.ALPHA = g["L0"], g["EMA0"], g["RFD"], g["EXP"], g["ALPHA"]
        self.H = g["H"]
        self.FWD = g["FWD"]
        self.dow = self.FWD.dayofweek.values
        self.season = np.isin(self.FWD.month.values, (8, 9))
        self.hist_lv = P["ndx"].values[-21:].astype(float)            # last 21 closes (20 returns)
        H = self.H
        self.lastmark = {}
        for i, d in enumerate(DAYS):
            m = self.dow == i
            self.lastmark[d] = np.maximum.accumulate(np.where(m, np.arange(H), -1))

    def signals(self, ndx_r):
        H = self.H
        lv = self.L0 * np.cumprod(1 + ndx_r)
        ema = np.empty(H)
        prev = self.EMA0
        a = self.ALPHA
        for t in range(H):
            prev = a * lv[t] + (1 - a) * prev
            ema[t] = prev
        full = np.concatenate([self.hist_lv, lv])
        pc = full[1:] / full[:-1] - 1
        win = np.lib.stride_tricks.sliding_window_view(pc, 20)[1:]     # windows ending at each fwd day
        vol = win.std(axis=1, ddof=1) * math.sqrt(252)
        ad = np.abs(np.diff(full))
        den = np.lib.stride_tricks.sliding_window_view(ad, 20)[1:].sum(axis=1)
        num = np.abs(full[21:] - full[1:-20])
        with np.errstate(invalid="ignore", divide="ignore"):
            er = np.where(den > 0, num / den, np.nan)
        above = lv > ema
        return lv, ema, vol, er, above

    def run(self, ndx_r, ief_r, gold_r, above_override=None):
        ndx_r = np.asarray(ndx_r, float)
        ief_r = np.asarray(ief_r, float)
        gold_r = np.asarray(gold_r, float)
        lv, ema, vol, er, above = self.signals(ndx_r)
        if above_override is not None:
            above = above_override(above)
        H = self.H
        clean = np.where(np.isnan(er), False, er >= 0.25)
        sleeves = {"qqq": ndx_r - 0.0020 / 252,
                   "qld": 2 * ndx_r - self.EXP / 252 - self.RFD,
                   "tqqq": 3 * ndx_r - self.EXP / 252 - 2 * self.RFD}
        out, states, sizes = {}, {}, {}
        for tier, (lev, vt, beta) in TIER.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                sc = np.clip(vt / vol, 0, 1)
            sc = np.where(np.isnan(sc), 1.0, sc)
            sc = np.clip(sc * np.where(self.season, 0.5, 1.0), 0, 1)
            eq = above * clean * sc + above * (~clean) * sc * 0.5
            tgt = eq * 0.85
            rets = {}
            for d in DAYS:
                lm = self.lastmark[d]
                held_tgt = np.concatenate([[0.0], np.where(lm >= 0, tgt[np.maximum(lm, 0)], 0.0)])  # after close of day k
                hw = held_tgt[:H]
                prevw = np.concatenate([[0.0], hw[:-1]])
                turn = 2 * np.abs(hw - prevw)
                rets[d] = hw * sleeves[lev] + (0.85 - hw) * ief_r + 0.15 * gold_r - turn * 3e-4
                if tier == "F2":
                    states[d] = held_tgt > 0
                if d == "Fri":
                    sizes[tier] = held_tgt
            rets["tr"] = sum(rets[d] for d in DAYS) / 5
            out[tier] = rets
        return dict(L0=self.L0, lv=lv, ema=ema, vol=vol, er=er, above=above, rets=out, states=states,
                    sizes=sizes, qqq=sleeves["qqq"], qld=sleeves["qld"], ndx_r=ndx_r, ief_r=ief_r, gold_r=gold_r)


def cum(r, h=None):
    r = r if h is None else r[:h]
    return float(np.prod(1 + r) - 1)


def mdd(r):
    e = np.concatenate([[1.0], np.cumprod(1 + r)])
    return float((e / np.maximum.accumulate(e) - 1).min())


def first_on(st, start=1):
    w = np.flatnonzero(st[start:])
    return int(w[0] + start) if len(w) else None


def summarize(res, H=252):
    """compact per-path metrics with the specialist's definitions."""
    m = {}
    for tier in TIER:
        for b in ("Fri", "tr"):
            r = res["rets"][tier][b]
            m[f"{tier}_{b}_r12m"] = cum(r, H)
            m[f"{tier}_{b}_r1m"] = cum(r, 21)
            m[f"{tier}_{b}_r6m"] = cum(r, 126)
            m[f"{tier}_{b}_mdd"] = mdd(r[:H])
    m["QQQ_r12m"] = cum(res["qqq"], H)
    m["QLD_r12m"] = cum(res["qld"], H)
    m["NDX_r12m"] = float(res["lv"][H - 1] / res["L0"] - 1)
    m["OFF_IEF_r12m"] = cum(0.85 * res["ief_r"] + 0.15 * res["gold_r"], H)
    m["OFF_IEF_r1m"] = cum(0.85 * res["ief_r"] + 0.15 * res["gold_r"], 21)
    s = res["states"]["Fri"]
    k = first_on(s, 1)
    lv = np.concatenate([[res["L0"]], res["lv"]])
    m["Fri_reentry_day"] = k
    m["Fri_reentry_ndx"] = float(lv[k]) if k else None
    m["Fri_whip_12m"] = int(np.sum(s[:H] & ~s[1:H + 1]))
    m["Fri_time_in_mkt_12m"] = float(s[:H].mean())
    m["tr_whip_12m"] = float(np.mean([np.sum(res["states"][d][:H] & ~res["states"][d][1:H + 1]) for d in DAYS]))
    m["F2_Fri_size_at_reentry"] = float(res["sizes"]["F2"][k]) if k else None
    # rebound participation / missed share (specialist's definitions, Friday gate)
    kt = int(np.argmin(lv))
    kp = kt + int(np.argmax(lv[kt:]))
    reb = float(lv[kp] / lv[kt] - 1)
    m["rebound_ndx"] = reb
    if kp > kt and reb >= 0.10:
        logmove = math.log(lv[kp] / lv[kt])
        ndx_log = np.log1p(res["ndx_r"])
        held = s[:H].astype(float)
        m["Fri_participation"] = float(np.sum(held[kt:kp] * ndx_log[kt:kp]) / logmove)
        if s[kt]:
            m["Fri_missed_before_invested"] = 0.0
        else:
            kin = first_on(s, kt + 1)
            m["Fri_missed_before_invested"] = 1.0 if (kin is None or kin >= kp) else math.log(lv[kin] / lv[kt]) / logmove
    else:
        m["Fri_participation"] = None
        m["Fri_missed_before_invested"] = None
    return m


# ------------------------------------------------------------------ path generators
def bridge_free(z, pins):
    """bridge segments up to the last pin exactly as the specialist's bridge(); raw noise after the last pin."""
    g = load_specialist()
    out = g["bridge"](z, pins) if pins else np.zeros_like(z)
    last = pins[-1] if pins else 0
    out[last:] = z[last:]
    return out


_BOOT = None


def boot_pool():
    """standardised (de-meaned, unit-variance) in-sample NDX daily log returns, 1988-01-01..freeze."""
    global _BOOT
    if _BOOT is None:
        g = load_specialist()
        P, C = g["P"], g["C"]
        idx = P.index
        x = np.log1p(P["ndx_ret"][(idx >= C.START) & (idx <= C.FREEZE)].values)
        _BOOT = (x - x.mean()) / x.std(ddof=1)
    return _BOOT


def boot_noise(rng, H, block):
    x = boot_pool()
    T = len(x)
    nb = int(np.ceil(H / block))
    st = rng.integers(0, T, nb)
    ix = ((st[:, None] + np.arange(block)[None, :]).ravel()[:H]) % T
    return x[ix]


def make_path(spec, vol, seed, mode="A", block=21):
    """mode 'A'/'B' reproduce the specialist's synth_path exactly; 'free' drops the day-252 pin;
    'boot' = design-A pins with block-bootstrapped NDX noise (same seeds for the y / gold noise)."""
    g = load_specialist()
    H = g["H"]
    if mode in ("A", "B"):
        out = g["synth_path"](spec, vol, seed, mode)
        return out[0], out[1], out[2]
    rng = np.random.default_rng([20261118, seed])
    zn, zy, zg = rng.standard_normal(H), rng.standard_normal(H), rng.standard_normal(H)
    sig = np.full(H, vol)
    if "calm_after" in spec:
        sig[spec["calm_after"]:] = g["CALM_VOL"]
    if mode == "free":
        pn = sorted({int(x) for x in spec["wn"] if 0 < int(x) <= H})
        py = sorted({int(x) for x in spec["wy"] if 0 < int(x) <= H})
        pg = sorted({int(x) for x in spec["wg"] if 0 < int(x) <= H})
        bn, by, bg = (lambda z: bridge_free(z, pn)), (lambda z: bridge_free(z, py)), (lambda z: bridge_free(z, pg))
    elif mode == "boot":
        rngb = np.random.default_rng([777, seed, block])
        zn = boot_noise(rngb, H, block)
        pn, py, pg = g["pins_for"](spec["wn"], "A"), g["pins_for"](spec["wy"], "A"), g["pins_for"](spec["wg"], "A")
        bn, by, bg = (lambda z: g["bridge"](z, pn)), (lambda z: g["bridge"](z, py)), (lambda z: g["bridge"](z, pg))
    else:
        raise ValueError(mode)
    dn = np.diff(np.concatenate([[0.0], spec["ndx"]])) + bn(zn * sig / math.sqrt(252))
    ndx_r = np.expm1(dn)
    Y0, IEF_D, IEF_C, IEF_A = g["Y0"], g["IEF_D"], g["IEF_C"], g["IEF_A"]
    ycum = spec["y"] + np.cumsum(by(zy * g["SD_Y"]))
    y = Y0 + ycum
    yprev = np.concatenate([[Y0], y[:-1]])
    dy = ycum / 100.0
    f = 1 - IEF_D * dy + 0.5 * IEF_C * dy ** 2
    fprev = np.concatenate([[1.0], f[:-1]])
    ief_r = f / fprev - 1 + (yprev / 100 + IEF_A) / 252
    gcum = spec["g"] + np.cumsum(bg(zg * g["SD_G"]))
    gold_r = np.expm1(np.diff(np.concatenate([[0.0], gcum])))
    return ndx_r, ief_r, gold_r


def osc_spec(T, A=0.08, n=126, falling=False, pin_mode="quarter"):
    """S6-type skeleton (specialist's osc() generalised): c = today's EMA, path starts at today's level."""
    g = load_specialist()
    c, L0, tt, H = g["EMA0"], g["L0"], g["tt"], g["H"]
    phi = math.asin((L0 / c - 1) / A)
    if falling:
        phi = math.pi - phi
    s = math.log(c / L0) + np.log(1 + A * np.sin(2 * np.pi * tt / T + phi))
    s[tt > n] = s[n - 1]
    if pin_mode == "quarter":
        wn = g["quarter_pins"](T, n)
    elif pin_mode == "end":
        wn = [n]
    elif pin_mode == "none":
        wn = []
    else:
        raise ValueError(pin_mode)
    Z = np.zeros(H)
    return dict(ndx=s, y=Z, g=Z, wn=wn, wy=[], wg=[])
