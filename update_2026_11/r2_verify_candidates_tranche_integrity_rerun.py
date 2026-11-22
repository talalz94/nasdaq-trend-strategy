"""
INTEGRITY REVIEW helper (read-only w.r.t. project files): re-run r2_candidates_tranche.py UNCHANGED except that
its output JSON path is redirected to a scratch directory (so the specialist's r2_candidates_tranche.json is not
overwritten), then diff every numeric leaf of the re-run against the saved JSON.

usage: python -u update_2026_11/r2_verify_candidates_tranche_integrity_rerun.py <scratch_dir>
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_candidates_tranche.py")
SAVED = os.path.join(HERE, "r2_candidates_tranche.json")
scratch = sys.argv[1]
os.makedirs(scratch, exist_ok=True)
OUTP = os.path.join(scratch, "rerun_r2_candidates_tranche.json").replace("\\", "/")

code = open(SRC, encoding="utf-8").read()
old = 'OUT_JSON = os.path.join(HERE, "r2_candidates_tranche.json")'
assert code.count(old) == 1, "output line not found exactly once"
code = code.replace(old, f'OUT_JSON = r"{OUTP}"')
assert code.count("open(OUT_JSON") == 1
# the script must not write anything else
assert code.count('"w")') == 1, "unexpected extra write in source"
g = {"__file__": SRC, "__name__": "__main__"}
exec(compile(code, SRC, "exec"), g)

a = json.load(open(SAVED))
b = json.load(open(OUTP))


def walk(x, y, path, out):
    if isinstance(x, dict):
        if not isinstance(y, dict):
            out.append((path, "type", x, y)); return
        for k in set(x) | set(y):
            if k not in x or k not in y:
                out.append((path + "/" + k, "missing", k in x, k in y)); continue
            walk(x[k], y[k], path + "/" + k, out)
    elif isinstance(x, list):
        if not isinstance(y, list) or len(x) != len(y):
            out.append((path, "len", len(x) if isinstance(x, list) else None, len(y) if isinstance(y, list) else None)); return
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{path}[{i}]", out)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if not (math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-13)):
            out.append((path, "num", x, y))
    else:
        if x != y:
            out.append((path, "val", x, y))


diffs = []
walk(a, b, "", diffs)
diffs = [d for d in diffs if not d[0].startswith("/meta/run_utc") and "runtime" not in d[0]]
print(f"\nRERUN-vs-SAVED: {len(diffs)} differing leaves (excluding run_utc / runtime fields)")
for d in diffs[:60]:
    print("  ", d)
