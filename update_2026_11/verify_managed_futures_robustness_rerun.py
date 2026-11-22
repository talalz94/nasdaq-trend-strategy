"""
verify_managed_futures_robustness_rerun.py -- adversarial reproducibility check (ROBUSTNESS lens).

Re-executes update_2026_11/managed_futures.py UNCHANGED except that
  * its JSON output is redirected to a scratch path (managed_futures.json is never touched),
  * the GSG yfinance refresh is disabled so the specialist's cached managed_futures_gsg.parquet is only READ
    (never rewritten) -> exact reproduction of the saved run,
then diffs every numeric leaf against the saved managed_futures.json and dumps all return series
(proxies, baselines, main grid, cash controls, robustness grid) + main proxy holdings to a scratch parquet
for the follow-up checks (verify_managed_futures_robustness_checks.py).

usage: python update_2026_11/verify_managed_futures_robustness_rerun.py <scratch_dir>
"""
import os, sys, json, time, io, contextlib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "managed_futures.py")
SAVED = os.path.join(HERE, "managed_futures.json")
SCR = os.path.abspath(sys.argv[1])
os.makedirs(SCR, exist_ok=True)
OUTJ = os.path.join(SCR, "managed_futures_rerun.json")
OUTP = os.path.join(SCR, "managed_futures_rerun_series.parquet")
assert os.path.abspath(OUTJ) != os.path.abspath(SAVED)

code = open(SRC, encoding="utf-8").read()
n1 = 'with open(os.path.join(HERE, "managed_futures.json"), "w") as f:'
n2 = 'import yfinance as yf'
n3 = 's.to_frame("gsg").to_parquet(GSG_CACHE)'
for n in (n1, n2, n3):
    assert code.count(n) == 1, n
code = code.replace(n1, f'with open({OUTJ!r}, "w") as f:')
code = code.replace(n2, 'raise RuntimeError("reviewer rerun: use cached GSG (read-only)")')
code = code.replace(n3, 'pass')
g = {"__file__": SRC, "__name__": "__rerun__"}
t0 = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, SRC, "exec"), g)
txt = buf.getvalue()
open(os.path.join(SCR, "managed_futures_rerun_console.txt"), "w", encoding="utf-8").write(txt)
print(f"[rerun] finished in {time.time()-t0:.0f}s")
print("\n".join(l for l in txt.splitlines() if "GSG" in l or "variants" in l or "bootstrap check" in l))

new = json.load(open(OUTJ))
old = json.load(open(SAVED))
diffs = []


def walk(a, b, path):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            diffs.append((path, "type")); return
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append((path + "/" + k, "missing")); continue
            walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            diffs.append((path, "len")); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            if a != b:
                diffs.append((path, f"{a} vs {b}"))
            return
        if abs(a - b) > 1e-9 * max(1, abs(a)):
            diffs.append((path, f"{a} vs {b}"))
    else:
        if a != b and path.find("run_date") < 0:
            diffs.append((path, f"{a} vs {b}"))


walk(old, new, "")
print(f"[rerun] leaves differing between saved managed_futures.json and re-run: {len(diffs)}")
for d in diffs[:30]:
    print("   ", d)

# ---- dump series for follow-up checks
S = {}
for k, v in g["PROX"].items():
    S[f"prox::{k}"] = v["ret"]
for k, v in g["BASE"].items():
    S[f"base::{k}"] = v["r"]
for k, v in g["PORT"].items():
    S[f"port::{k}"] = v["r"]
for k, v in g["CASH"].items():
    S[f"cash::{k}"] = v["r"]
for k, v in g["ROB"].items():
    S[f"rob::{k}"] = v["r"]
for pk in ["main_12m", "blend_1_3_12m", "ex_equities", "control_always_long"]:
    pr = g["PROX"][pk]
    for mk in pr["mkts"]:
        S[f"held::{pk}::{mk}"] = pr["held"][mk]
    S[f"turn::{pk}"] = pr["turn"]
D = pd.DataFrame(S)
D.to_parquet(OUTP)
print(f"[rerun] saved {D.shape[1]} series x {D.shape[0]} days -> {OUTP}")
