"""
Integrity reviewer: re-run r2_friday_effect.py UNCHANGED (exec of its source) with only the output path
redirected to a scratch location, then diff every numeric leaf against the stored r2_friday_effect.json.
Writes nothing inside the project except this script.
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r2_friday_effect.py")
SCR = os.environ.get("VERIFY_SCRATCH", os.path.join(os.path.expanduser("~"), "r2_verify_scratch"))
os.makedirs(SCR, exist_ok=True)
NEW = os.path.join(SCR, "r2_friday_effect_rerun.json")

code = open(SRC, encoding="utf-8").read()
old_line = 'OUT_JSON = os.path.join(HERE, "r2_friday_effect.json")'
assert code.count(old_line) == 1
code = code.replace(old_line, f'OUT_JSON = r"{NEW}"')
g = {"__file__": SRC, "__name__": "__rerun__"}
exec(compile(code, SRC, "exec"), g)

A = json.load(open(os.path.join(HERE, "r2_friday_effect.json")))
B = json.load(open(NEW))


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing", None)); continue
            walk(a[k], b[k], path + "/" + k, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "len", (len(a), len(b)))); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, abs_tol=1e-9))):
            out.append((path, "num", (a, b)))
    else:
        if a != b:
            out.append((path, "val", (a, b)))


diffs = []
walk(A, B, "", diffs)
diffs = [d for d in diffs if not d[0].startswith("/runtime_s")]
print("n_diffs (excluding runtime):", len(diffs))
for d in diffs[:40]:
    print(d)
