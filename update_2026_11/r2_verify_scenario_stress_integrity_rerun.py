"""
Integrity reviewer: re-run r2_scenario_stress.py UNMODIFIED except that its output JSON is
redirected (so the specialist's r2_scenario_stress.json is never overwritten), then deep-compare
every numeric leaf against the stored JSON.
usage: python update_2026_11/r2_verify_scenario_stress_integrity_rerun.py <out_json>
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_scenario_stress.py")
STORED = os.path.join(HERE, "r2_scenario_stress.json")
out = sys.argv[1]
code = open(SRC, encoding="utf-8").read()
old = 'OUT = os.path.join(HERE, "r2_scenario_stress.json")'
assert code.count(old) == 1
code = code.replace(old, f"OUT = {out!r}")
g = {"__file__": SRC, "__name__": "__main__"}
exec(compile(code, SRC, "exec"), g)

a = json.load(open(STORED))
b = json.load(open(out))
n_cmp, n_diff, worst = 0, 0, []
SKIP = {"runtime_sec"}


def walk(x, y, path):
    global n_cmp, n_diff
    if isinstance(x, dict):
        if set(x) != set(y):
            worst.append((float("inf"), path + " KEYSET", str(set(x) ^ set(y))[:200]))
        for k in x:
            if k in y and k not in SKIP:
                walk(x[k], y[k], path + "/" + k)
    elif isinstance(x, list):
        if len(x) != len(y):
            worst.append((float("inf"), path + " LEN", f"{len(x)} vs {len(y)}"))
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)):
        n_cmp += 1
        d = abs(x - y)
        if d > 1e-12 * max(1.0, abs(x)):
            n_diff += 1
            worst.append((d, path, f"{x} vs {y}"))
    else:
        n_cmp += 1
        if x != y:
            n_diff += 1
            worst.append((float("inf"), path, f"{x!r} vs {y!r}"))


walk(a, b, "")
worst.sort(key=lambda t: -t[0] if math.isfinite(t[0]) else -1e300)
print(f"\nDEEP COMPARE stored vs rerun: {n_cmp} leaves compared, {n_diff} differ (tol 1e-12 rel)")
for w in worst[:25]:
    print("   ", w)
