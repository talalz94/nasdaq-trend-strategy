"""
INTEGRITY REVIEW of oos_live (adversarial reviewer) -- step 1: reproducibility.

Re-executes update_2026_11/oos_live.py UNCHANGED except that its JSON output path is redirected to
a scratch file (so the specialist's oos_live.json is never touched), then compares every numeric leaf
of the fresh output with the published oos_live.json.

Writes nothing inside the project (scratch JSON goes to the path given as argv[1]).
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "oos_live.py")
PUB = os.path.join(HERE, "oos_live.json")
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "."), "oos_live_rerun.json")

code = open(SRC, encoding="utf-8").read()
old = 'OUTJ = os.path.join(HERE, "oos_live.json")'
assert code.count(old) == 1
code = code.replace(old, f"OUTJ = {SCRATCH!r}")
g = {"__file__": SRC, "__name__": "__main__"}
exec(compile(code, SRC, "exec"), g)

A = json.load(open(PUB))
B = json.load(open(SCRATCH))


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + "/" + str(k), "missing", None))
                continue
            walk(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "len", (len(a), len(b))))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=0, abs_tol=2e-6)):
            out.append((path, "num", (a, b)))
    else:
        if a != b:
            out.append((path, "val", (a, b)))


diffs = []
walk(A, B, "", diffs)
print("\n================ REPRODUCIBILITY ================")
print("leaf differences between published oos_live.json and fresh rerun:", len(diffs))
for d in diffs[:40]:
    print("  ", d)
