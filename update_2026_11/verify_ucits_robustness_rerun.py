"""
ROBUSTNESS REVIEW of update_2026_11/ucits.py  --  step 1: re-run the specialist's code unchanged.

ucits.py is executed with only its output path redirected to a scratch file (so update_2026_11/ucits.json is
never overwritten; ucits.py only READS ucits_prices.parquet / ucits_info.json), every numeric leaf of the fresh
output is compared with the saved ucits.json, and the daily return series the specialist built (SERIES, the
cleaned USD price lines, P2's UCITS return columns) are pickled to the scratch dir for the follow-up checks.

usage: python update_2026_11/verify_ucits_robustness_rerun.py <scratch_dir>
"""
import os
import sys
import json
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ucits.py")
SAVED = os.path.join(HERE, "ucits.json")
scratch = sys.argv[1]
os.makedirs(scratch, exist_ok=True)
NEW = os.path.join(scratch, "ucits_rerun.json")

code = open(SRC, encoding="utf-8").read()
pat = 'open(os.path.join(HERE, "ucits.json"), "w")'
assert code.count(pat) == 1, "output-path expression not found exactly once"
code = code.replace(pat, f"open({NEW!r}, 'w')")
assert "to_parquet" not in code and "yf." not in code, "ucits.py would write parquet / fetch -> abort"
g = {"__name__": "__main__", "__file__": SRC}
t0 = time.time()
exec(compile(code, SRC, "exec"), g)
print(f"\n[rerun took {time.time() - t0:.0f}s]")

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

# ---- persist the specialist's own objects for the follow-up robustness checks (scratch only)
dump = {"SERIES": {k: {"R": v["R"], "a": v["a"], "b": v["b"]} for k, v in g["SERIES"].items()},
        "CLEAN": g["CLEAN"],
        "P2_ucits_cols": g["P2"][[c for c in g["P2"].columns if c.startswith("u_")]],
        "EXCH_CAL": g["EXCH_CAL"],
        "COST_REAL": g["COST_REAL"], "COST_PESS": g["COST_PESS"],
        "FX": g["FX"]}
with open(os.path.join(scratch, "ucits_objects.pkl"), "wb") as f:
    pickle.dump(dump, f)
print("saved", os.path.join(scratch, "ucits_objects.pkl"))
