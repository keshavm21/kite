#!/usr/bin/env python3
"""Kite test runner (see docs/TESTING.md section 4).

    python3 tests/run_tests.py                 # run everything, exit 1 on failure
    python3 tests/run_tests.py -k softmax      # only tests whose name contains "softmax"
    python3 tests/run_tests.py --update        # rewrite golden files from current output
    python3 tests/run_tests.py --kitec build/kitec

Categories implemented so far:
  golden  tests/golden/NAME.ast  vs  `kitec NAME.kite --emit=ast`
          tests/golden/NAME.ir   vs  `kitec NAME.kite --emit=ir -O0 --verify-each`
          tests/golden/NAME.ptx  vs  `kitec NAME.kite --emit=ptx -O0`
          the source is tests/golden/NAME.kite, else tests/exec/NAME.kite, else examples/NAME.kite
  errors  tests/errors/*.kite whose first line is `// ERROR: LINE:COL: message`
          kitec must exit 1 and print `...:LINE:COL: error: message` on stderr
  exec    every examples/*.kite and tests/exec/*.kite: `kitec run -O0` and `kitec run -O1`
          (both with --verify-each) must succeed and print byte-for-byte identical output
  ptx     every source: `kitec --emit=ptx` must assemble with `ptxas -arch=sm_75` when ptxas is on PATH;
          otherwise tests/ptx_lint.py does a structural check (undeclared registers, bad labels, ...)
Later milestones add golden .O1.ir / .O1.ptx files.
"""
import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptx_lint import lint  # noqa: E402
SOURCE_DIRS = [ROOT / "tests/golden", ROOT / "tests/exec", ROOT / "examples"]

# golden file extension -> kitec arguments that produce its contents
GOLDEN_KINDS = {
    "ast": ["--emit=ast"],
    "ir": ["--emit=ir", "-O0", "--verify-each"],
    "ptx": ["--emit=ptx", "-O0"],
}


def run(kitec, *args):
    return subprocess.run([str(kitec), *map(str, args)], capture_output=True, text=True)


def find_source(name):
    for d in SOURCE_DIRS:
        if (d / f"{name}.kite").exists():
            return d / f"{name}.kite"
    return None


def golden_cases(pattern):
    for ext in GOLDEN_KINDS:
        for expected in sorted((ROOT / "tests/golden").glob(f"*.{ext}")):
            name = expected.stem
            src = find_source(name)
            if src and pattern in f"golden/{name}.{ext}":
                yield f"golden/{name}.{ext}", ext, src, expected


def sources_without_golden():
    """Every compilable source: examples, golden inputs and exec tests (no error tests)."""
    srcs = [p for d in SOURCE_DIRS for p in sorted(d.glob("*.kite"))]
    return [(ext, p) for ext in GOLDEN_KINDS for p in srcs if not (ROOT / "tests/golden" / f"{p.stem}.{ext}").exists()]


def check_golden(kitec, ext, src, expected, update):
    r = run(kitec, src, *GOLDEN_KINDS[ext])
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


def exec_sources():
    return sorted(p for d in (ROOT / "examples", ROOT / "tests/exec") for p in d.glob("*.kite"))


def check_exec(kitec, src):
    results = {}
    for level in ("-O0", "-O1"):
        r = run(kitec, "run", src, level, "--verify-each")
        if r.returncode != 0:
            return f"`kitec run {level}` exited {r.returncode}:\n{r.stderr}"
        if not r.stdout.strip():
            return f"`kitec run {level}` printed nothing (does the @spec have a @check line?)"
        results[level] = r.stdout
    if results["-O0"] != results["-O1"]:
        diff = difflib.unified_diff(results["-O0"].splitlines(True), results["-O1"].splitlines(True), "-O0", "-O1")
        return "".join(diff)
    return None


def check_ptx(kitec, src):
    r = run(kitec, src, "--emit=ptx", "-O0")
    if r.returncode != 0:
        return f"kitec exited {r.returncode}:\n{r.stderr}"
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return "\n".join(lint(r.stdout)) or None
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "out.ptx"
        f.write_text(r.stdout)
        a = subprocess.run([ptxas, "-arch=sm_75", str(f), "-o", "/dev/null"], capture_output=True, text=True)
    return None if a.returncode == 0 else f"ptxas failed:\n{a.stderr}"


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
        for ext, src in sources_without_golden():
            if args.k in src.stem:
                (ROOT / "tests/golden" / f"{src.stem}.{ext}").touch()

    failures, total = [], 0

    def report(name, err, suffix=""):
        nonlocal total
        total += 1
        print(f"{'FAIL' if err else 'ok  '} {name}{suffix}")
        if err:
            failures.append((name, err))

    for name, ext, src, expected in golden_cases(args.k):
        report(name, check_golden(kitec, ext, src, expected, args.update), " (updated)" if args.update else "")

    for src in sorted((ROOT / "tests/errors").glob("*.kite")):
        name = f"errors/{src.stem}"
        if args.k in name:
            report(name, check_error(kitec, src))

    for src in exec_sources():
        name = f"exec/{src.stem}"
        if args.k in name:
            report(name, check_exec(kitec, src))

    for src in exec_sources() + sorted((ROOT / "tests/golden").glob("*.kite")):
        name = f"ptx/{src.stem}"
        if args.k in name:
            report(name, check_ptx(kitec, src))

    for name, err in failures:
        print(f"\n--- {name} ---\n{err}")
    print(f"\n{total - len(failures)}/{total} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
