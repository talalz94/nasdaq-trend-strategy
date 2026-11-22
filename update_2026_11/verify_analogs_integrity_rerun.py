"""
verify_analogs_integrity_rerun.py -- INTEGRITY REVIEW of update_2026_11/analogs.py

Re-executes analogs.py UNCHANGED except that its output path is redirected to the
reviewer's scratch directory (so update_2026_11/analogs.json is never modified), then
deep-compares every numeric leaf of the fresh result against the saved analogs.json.
"""
import os, sys, json, math, io, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "analogs.py")
SAVED = os.path.join(HERE, "analogs.json")
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", HERE)
RERUN = os.path.join(SCRATCH, "analogs_rerun.json")
LOG = os.path.join(SCRATCH, "analogs_rerun_stdout.txt")

src = open(SRC, encoding="utf-8").read()
needle = 'OUT = os.path.join(HERE, "analogs.json")'
assert src.count(needle) == 1
src = src.replace(needle, f"OUT = r'{RERUN}'")
g = {"__file__": SRC, "__name__": "__analogs_rerun__"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, SRC, "exec"), g)
open(LOG, "w", encoding="utf-8").write(buf.getvalue())
print(f"re-run done; stdout saved to {LOG}")

A = json.load(open(SAVED))
B = json.load(open(RERUN))
diffs, n = [], [0]


def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append((path + "/" + k, "missing", k in a, k in b)); continue
            walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, "len", len(a), len(b))); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        n[0] += 1
        if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)):
            diffs.append((path, "num", a, b))
    else:
        n[0] += 1
        if a != b:
            diffs.append((path, "val", a, b))


walk(A, B, "")
print(f"compared {n[0]} leaves; {len(diffs)} differences")
for d in diffs[:40]:
    print("  DIFF", d)
