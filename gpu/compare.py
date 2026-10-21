"""Compare two output files (`kitec run` / run_gpu.py format) with tolerance.

    python3 gpu/compare.py expected.out actual.out [--exact]
Exit code 0 if equal (ints exact, floats within rtol=1e-5, atol=1e-6), else 1.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kite_spec import compare  # noqa: E402

if len(sys.argv) < 3:
    sys.exit("usage: compare.py EXPECTED ACTUAL [--exact]")
exact = "--exact" in sys.argv
problems = compare(open(sys.argv[1]).read(), open(sys.argv[2]).read(),
                   *((0.0, 0.0) if exact else ()))
for p in problems:
    print(p)
print("MATCH" if not problems else f"{len(problems)} MISMATCH(ES)")
sys.exit(1 if problems else 0)
