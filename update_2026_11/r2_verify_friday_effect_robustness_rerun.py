"""
r2_verify_friday_effect_robustness_rerun.py -- adversarial reproducibility check (ROBUSTNESS lens).

Re-executes update_2026_11/r2_friday_effect.py UNCHANGED except that its output path is redirected to a
scratch file (so the specialist's r2_friday_effect.json is never touched), then diffs every numeric leaf of
the fresh output against the saved JSON (relative tolerance 1e-9; runtime fields ignored).

usage: python -u update_2026_11/r2_verify_friday_effect_robustness_rerun.py <scratch_out.json>
"""
import os, sys, json, io, contextlib, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_friday_effect.py")
SAVED = os.path.join(HERE, "r2_friday_effect.json")
OUT = os.path.abspath(sys.argv[1])
assert os.path.abspath(OUT) != os.path.abspath(SAVED)

code = open(SRC, encoding="utf-8").read()
needle = 'OUT_JSON = os.path.join(HERE, "r2_friday_effect.json")'
assert code.count(needle) == 1, code.count(needle)
code = code.replace(needle, f'OUT_JSON = {OUT!r}')
g = {"__file__": SRC, "__name__": "__rerun__"}
buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    exec(compile(code, SRC, "exec"), g)
print(buf.getvalue())
print(f"[rerun] finished in {time.time()-t0:.0f}s -> {OUT}")

new = json.load(open(OUT))
old = json.load(open(SAVED))
IGNORE = {"run_utc", "runtime_s"}


def walk(a, b, path, out):
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k in IGNORE:
                continue
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
        if abs(a - b) > 1e-9 * max(1, abs(a)):
            out.append((path, "num", (a, b)))
    else:
        if a != b:
            out.append((path, "val", (a, b)))


diffs = []
walk(old, new, "", diffs)
print(f"[rerun] numeric/value leaves differing between saved and re-run: {len(diffs)}")
for d in diffs[:40]:
    print("  ", d)
