#!/usr/bin/env python3
"""Differential fuzzer for kitec (docs/TESTING.md section 6).

    python3 tests/fuzz/fuzz.py                      # 200 seeds from 0
    python3 tests/fuzz/fuzz.py -n 5000 --start 1000 -j 8
    python3 tests/fuzz/fuzz.py --seed 1234 --keep   # one seed; keep it in tests/exec even if it passes

For every seed tests/fuzz/gen.py makes a kernel. `kitec run -O0` is the oracle. If it reports undefined
behaviour (division by zero, out of bounds, step limit...) the program is *invalid* and skipped. Otherwise
each of these must print exactly the same output, with --verify-each:
    -O1, --passes=ssa, --passes=ssa,out-of-ssa, and -O1 with each pass of the pipeline taken out in turn
(the last one is what pinpoints which pass is wrong), and `--emit=ptx -O0/-O1` must succeed
(and assemble with ptxas when it is on PATH).
A failing program is saved as tests/exec/fuzz_<seed>.kite, which then also fails tests/run_tests.py until the
compiler is fixed, so the bug stays pinned as a regression test.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen import generate  # noqa: E402

TIMEOUT = 60
# A runtime error from the interpreter: "error: <what> in kernel K, block B, thread T, ..."
UB = re.compile(r"^error: .* in kernel \w+, block \d+", re.M)

PIPELINE = ["constfold", "copyprop", "gvn", "dce", "simplifycfg", "licm", "strength"]


def kitec(binary, *args):
    try:
        return subprocess.run([binary, *map(str, args)], capture_output=True, text=True, timeout=TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timeout")


def check_seed(args):
    binary, seed, ptxas = args
    src = generate(seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fuzz.kite"
        path.write_text(src)
        base = kitec(binary, "run", path, "-O0", "--verify-each")
        if base.returncode != 0:
            if UB.search(base.stderr):
                return seed, "invalid", base.stderr.strip().splitlines()[0]
            return seed, "FAIL", "-O0 failed to compile or run (generator or compiler bug):\n" + base.stderr
        variants = {
            "-O1": ["-O1"],
            "ssa": ["--passes=ssa"],
            "ssa,out-of-ssa": ["--passes=ssa,out-of-ssa"],
            "ssa,licm,out-of-ssa": ["--passes=remove-unreachable,ssa,licm,out-of-ssa"],
            "ssa,strength,out-of-ssa": ["--passes=remove-unreachable,ssa,strength,out-of-ssa"],
            "ssa,gvn,out-of-ssa": ["--passes=remove-unreachable,ssa,gvn,out-of-ssa"],
            "ssa,constfold,dce,out-of-ssa": ["--passes=remove-unreachable,ssa,constfold,copyprop,dce,out-of-ssa"],
            "ssa,simplifycfg,out-of-ssa": ["--passes=remove-unreachable,ssa,simplifycfg,out-of-ssa"],
        }
        problems = []
        for name, flags in variants.items():
            r = kitec(binary, "run", path, *flags, "--verify-each")
            if r.returncode != 0:
                problems.append(f"[{name}] failed although -O0 succeeded:\n{r.stderr.strip()}")
            elif r.stdout != base.stdout:
                problems.append(f"[{name}] output differs from -O0")
        for level in ("-O0", "-O1"):
            r = kitec(binary, path, "--emit=ptx", level, "-o", Path(tmp) / "k.ptx")
            if r.returncode != 0:
                problems.append(f"[ptx {level}] kitec failed:\n{r.stderr.strip()}")
            elif ptxas:
                a = subprocess.run([ptxas, "-arch=sm_75", str(Path(tmp) / "k.ptx"), "-o", "/dev/null"], capture_output=True, text=True)
                if a.returncode != 0:
                    problems.append(f"[ptx {level}] ptxas rejected it:\n{a.stderr.strip()}")
        if problems:
            return seed, "FAIL", "\n".join(problems)
        zero = all(tok == "0" for line in base.stdout.splitlines() for tok in line.split()[1:])
        return seed, "trivial" if zero else "ok", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=200, help="number of seeds")
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--seed", type=int, help="run only this seed")
    ap.add_argument("--keep", action="store_true", help="save the program even if it passes")
    ap.add_argument("-j", type=int, default=4, help="parallel jobs")
    ap.add_argument("--kitec", default=str(ROOT / "build/kitec"))
    ap.add_argument("--save-dir", default=str(ROOT / "tests/exec"))
    opts = ap.parse_args()

    seeds = [opts.seed] if opts.seed is not None else range(opts.start, opts.start + opts.n)
    ptxas = shutil.which("ptxas")
    counts = {"ok": 0, "trivial": 0, "invalid": 0, "FAIL": 0}
    failures = []
    with ProcessPoolExecutor(opts.j) as pool:
        for seed, status, detail in pool.map(check_seed, [(opts.kitec, s, ptxas) for s in seeds], chunksize=4):
            counts[status] += 1
            if status == "FAIL" or opts.keep:
                dest = Path(opts.save_dir) / f"fuzz_{seed}.kite"
                dest.write_text(generate(seed))
                if status == "FAIL":
                    failures.append(seed)
                    print(f"FAIL seed {seed} (saved {dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}):\n{detail}\n")
    total = sum(counts.values())
    print(f"{total} seeds: {counts['ok']} ok, {counts['trivial']} ok but all-zero output, "
          f"{counts['invalid']} skipped (UB at -O0), {counts['FAIL']} FAILED"
          f"{' (ptxas checked)' if ptxas else ' (ptxas not on PATH: PTX not assembled)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
