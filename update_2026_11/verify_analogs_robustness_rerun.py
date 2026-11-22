"""
verify_analogs_robustness_rerun.py -- re-run analogs.py WITHOUT touching analogs.json.
The source of analogs.py is exec'd with OUT redirected to a scratch path, then the fresh
output is diffed against the committed analogs.json (numeric tolerance 1e-9).
"""
import os, sys, json, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "analogs.py")
ORIG = os.path.join(HERE, "analogs.json")
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", HERE), "analogs_rerun.json")

src = open(SRC, encoding="utf-8").read()
needle = 'OUT = os.path.join(HERE, "analogs.json")'
assert needle in src
src = src.replace(needle, f"OUT = {SCRATCH!r}")
g = {"__file__": SRC, "__name__": "analogs_rerun"}
t0 = time.time()
exec(compile(src, SRC, "exec"), g)
print(f"\n[rerun] finished in {time.time()-t0:.0f}s -> {SCRATCH}")

a = json.load(open(ORIG))
b = json.load(open(SCRATCH))
diffs = []


def walk(x, y, path):
    if isinstance(x, dict):
        if not isinstance(y, dict):
            diffs.append((path, "type")); return
        for k in set(x) | set(y):
            if k not in x or k not in y:
                diffs.append((path + "/" + k, "missing")); continue
            walk(x[k], y[k], path + "/" + k)
    elif isinstance(x, list):
        if not isinstance(y, list) or len(x) != len(y):
            diffs.append((path, "len")); return
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{path}[{i}]")
    elif isinstance(x, float) or isinstance(y, float):
        if x is None or y is None:
            if x != y:
                diffs.append((path, f"{x} vs {y}"))
            return
        if not (abs(x - y) <= 1e-9 * max(1, abs(x))):
            diffs.append((path, f"{x} vs {y}"))
    else:
        if x != y:
            diffs.append((path, f"{x} vs {y}"))


walk(a, b, "")
print(f"[rerun] differences vs committed analogs.json: {len(diffs)}")
for d in diffs[:40]:
    print("   ", d)
