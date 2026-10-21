#!/usr/bin/env python3
"""Kite test runner (see docs/TESTING.md section 4).

    python3 tests/run_tests.py                 # run everything, exit 1 on failure
    python3 tests/run_tests.py -k softmax      # only tests whose name contains "softmax"
    python3 tests/run_tests.py --update        # rewrite golden files from current output
    python3 tests/run_tests.py --kitec build/kitec

Categories implemented so far:
  golden  tests/golden/NAME.ast  vs  `kitec NAME.kite --emit=ast`
          source is tests/golden/NAME.kite, else examples/NAME.kite
  errors  tests/errors/*.kite whose first line is `// ERROR: LINE:COL: message`
          kitec must exit 1 and print `...:LINE:COL: error: message` on stderr
Later milestones add golden IR/PTX files and the exec (differential) category.
"""
import argparse
import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(kitec, *args):
    return subprocess.run([str(kitec), *map(str, args)], capture_output=True, text=True)


def golden_cases(pattern):
    for expected in sorted((ROOT / "tests/golden").glob("*.ast")):
        name = expected.stem
        src = ROOT / "tests/golden" / f"{name}.kite"
        if not src.exists():
            src = ROOT / "examples" / f"{name}.kite"
        if pattern in f"golden/{name}":
            yield f"golden/{name}", src, expected


def sources_without_golden():
    have = {p.stem for p in (ROOT / "tests/golden").glob("*.ast")}
    srcs = list((ROOT / "examples").glob("*.kite")) + list((ROOT / "tests/golden").glob("*.kite"))
    return [p for p in sorted(srcs) if p.stem not in have]


def check_golden(kitec, src, expected, update):
    r = run(kitec, src, "--emit=ast")
    if r.returncode != 0:
        return f"kitec exited {r.returncode}:\n{r.stderr}"
    if update:
        expected.write_text(r.stdout)
        return None
    want = expected.read_text()
    if r.stdout == want:
        return None
    diff = difflib.unified_diff(want.splitlines(True), r.stdout.splitlines(True), "expected", "actual")
    return "".join(diff)


def check_error(kitec, src):
    lines = src.read_text().splitlines()
    first = lines[0] if lines else ""
    if not first.startswith("// ERROR:"):
        return "first line must be `// ERROR: LINE:COL: message`"
    spec = first[len("// ERROR:"):].strip()
    line, col, msg = spec.split(":", 2)
    needle = f":{line.strip()}:{col.strip()}: error: {msg.strip()}"
    r = run(kitec, src, "--emit=ast")
    if r.returncode != 1:
        return f"expected exit code 1, got {r.returncode}"
    if needle not in r.stderr:
        return f"expected stderr to contain `{needle}`, got:\n{r.stderr}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="run only tests whose name contains this")
    ap.add_argument("--update", action="store_true", help="rewrite golden files (review the diff!)")
    ap.add_argument("--kitec", default=str(ROOT / "build/kitec"))
    args = ap.parse_args()
    kitec = Path(args.kitec)
    if not kitec.exists():
        sys.exit(f"{kitec} not found; build first (cmake --build build)")

    if args.update:  # create a golden file for every source that lacks one
        for src in sources_without_golden():
            (ROOT / "tests/golden" / f"{src.stem}.ast").touch()

    failures, total = [], 0
    for name, src, expected in golden_cases(args.k):
        total += 1
        err = check_golden(kitec, src, expected, args.update)
        print(f"{'FAIL' if err else 'ok  '} {name}" + (" (updated)" if args.update else ""))
        if err:
            failures.append((name, err))

    for src in sorted((ROOT / "tests/errors").glob("*.kite")):
        name = f"errors/{src.stem}"
        if args.k not in name:
            continue
        total += 1
        err = check_error(kitec, src)
        print(f"{'FAIL' if err else 'ok  '} {name}")
        if err:
            failures.append((name, err))

    for name, err in failures:
        print(f"\n--- {name} ---\n{err}")
    print(f"\n{total - len(failures)}/{total} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
