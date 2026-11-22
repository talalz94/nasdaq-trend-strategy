"""
INTEGRITY REVIEW of update_2026_11/revalidate.py  --  step 1: reproduce.

Executes revalidate.py UNCHANGED except that its output path is redirected to a scratch
file (so the specialist's revalidate.json is never overwritten), then compares every
numeric leaf of the fresh output with the saved revalidate.json.

usage: python update_2026_11/verify_revalidate_integrity_rerun.py <scratch_dir>
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "revalidate.py")
SAVED = os.path.join(HERE, "revalidate.json")
scratch = sys.argv[1]
NEW = os.path.join(scratch, "revalidate_rerun.json")

code = open(SRC, encoding="utf-8").read()
pat = 'OUT_JSON = os.path.join(HERE, "revalidate.json")'
assert code.count(pat) == 1, "output-path line not found exactly once"
code = code.replace(pat, f"OUT_JSON = {NEW!r}")
g = {"__name__": "__main__", "__file__": SRC}
exec(compile(code, SRC, "exec"), g)

a = json.load(open(SAVED))
b = json.load(open(NEW))


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


la, lb = dict(leaves(a)), dict(leaves(b))
missing = set(la) ^ set(lb)
diffs = []
for k in la.keys() & lb.keys():
    x, y = la[k], lb[k]
    if k.endswith("runtime_s"):
        continue
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if abs(x - y) > 1e-6:
            diffs.append((k, x, y))
    elif x != y:
        diffs.append((k, x, y))
print(f"\nREPRODUCTION: {len(la)} saved leaves, {len(lb)} rerun leaves, "
      f"{len(missing)} keys differ in presence, {len(diffs)} values differ (>1e-6)")
for d in sorted(diffs)[:40]:
    print("   DIFF", d)
for m in sorted(missing)[:20]:
    print("   KEY-MISMATCH", m)
