"""
verify_managed_futures_integrity_rerun.py  (adversarial review, integrity lens)

Re-executes update_2026_11/managed_futures.py WITHOUT touching any project file:
  * the json.dump target is redirected to the scratchpad (argv[1] dir)
  * the GSG yfinance fetch is disabled -> the existing cache (read-only) is used
Then diffs every numeric leaf of the fresh output against the saved managed_futures.json,
and pickles the key return series for follow-up checks.
"""
import os, sys, json, re, pickle
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "update_2026_11")
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", ".")
os.makedirs(OUTDIR, exist_ok=True)
SRC = os.path.join(HERE, "managed_futures.py")
src = open(SRC, encoding="utf-8").read()

# 1) never write managed_futures.json in the project
n1 = src.count('open(os.path.join(HERE, "managed_futures.json"), "w")')
assert n1 == 1, n1
src = src.replace('open(os.path.join(HERE, "managed_futures.json"), "w")',
                  'open(os.path.join(%r, "managed_futures_rerun.json"), "w")' % OUTDIR)
# 2) never refresh the GSG cache (no network, no write): force the offline branch
n2 = src.count('h = yf.Ticker("GSG").history(period="max", auto_adjust=True)["Close"].dropna()')
assert n2 == 1, n2
src = src.replace('h = yf.Ticker("GSG").history(period="max", auto_adjust=True)["Close"].dropna()',
                  'raise RuntimeError("offline by verifier")')
assert "to_parquet(" in src  # only reachable after a successful fetch, which is now impossible

NS = {"__file__": SRC, "__name__": "mf_rerun"}
exec(compile(src, SRC, "exec"), NS)

# ---------------- diff against saved JSON
saved = json.load(open(os.path.join(HERE, "managed_futures.json")))
fresh = json.load(open(os.path.join(OUTDIR, "managed_futures_rerun.json")))

diffs = []
def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append((path + "/" + str(k), "missing", a.get(k, "<none>") if isinstance(a, dict) else None))
                continue
            walk(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, "len", (len(a), len(b))))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > 1e-6 * max(1, abs(a)):
            diffs.append((path, a, b))
    else:
        if a != b and path not in ("/meta/run_date",):
            diffs.append((path, a, b))
walk(saved, fresh)
print("\n=== RERUN DIFF vs saved managed_futures.json ===")
print("number of differing leaves:", len(diffs))
for d in diffs[:40]:
    print("  ", d)

# ---------------- pickle series for follow-ups
keep = dict(
    proxies={k: v["ret"] for k, v in NS["PROX"].items()},
    proxy_pos={k: v["POS"] for k, v in NS["PROX"].items()},
    proxy_K={k: v["K"] for k, v in NS["PROX"].items()},
    base={k: v["r"] for k, v in NS["BASE"].items()},
    port={k: v["r"] for k, v in NS["PORT"].items()},
    rob={k: v["r"] for k, v in NS["ROB"].items()},
    cash={k: v["r"] for k, v in NS["CASH"].items()},
)
with open(os.path.join(OUTDIR, "mf_series.pkl"), "wb") as f:
    pickle.dump(keep, f)
print("pickled series to", os.path.join(OUTDIR, "mf_series.pkl"))
