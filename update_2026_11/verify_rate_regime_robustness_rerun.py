"""
verify_rate_regime_robustness_rerun.py -- adversarial reproducibility check (ROBUSTNESS lens).

Re-executes update_2026_11/rate_regime.py UNCHANGED except that its output path is redirected to
the reviewer's scratch file (so the specialist's rate_regime.json is never touched), then diffs
every numeric leaf of the fresh output against the saved rate_regime.json.

usage: python update_2026_11/verify_rate_regime_robustness_rerun.py <scratch_out.json>
"""
import os, sys, json, math, runpy, io, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "rate_regime.py")
SAVED = os.path.join(HERE, "rate_regime.json")
OUT = os.path.abspath(sys.argv[1])
assert os.path.abspath(OUT) != os.path.abspath(SAVED)

code = open(SRC, encoding="utf-8").read()
needle = 'OUT = os.path.join(HERE, "rate_regime.json")'
assert code.count(needle) == 1
code = code.replace(needle, f"OUT = {OUT!r}")
g = {"__file__": SRC, "__name__": "__rerun__"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, SRC, "exec"), g)
print("\n".join(buf.getvalue().splitlines()[:3]))

new = json.load(open(OUT))
old = json.load(open(SAVED))


def walk(a, b, path, out):
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing", None))
            else:
                walk(a[k], b[k], path + "/" + k, out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path, "len", (len(a), len(b))))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-6 * max(1, abs(a))):
            out.append((path, "num", (a, b)))
    else:
        if a != b:
            out.append((path, "val", (a, b)))


diffs = []
walk(old, new, "", diffs)
print(f"numeric/value leaves differing between saved and re-run: {len(diffs)}")
for d in diffs[:40]:
    print("  ", d)
