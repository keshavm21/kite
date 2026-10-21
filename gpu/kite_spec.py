"""Shared test-spec utilities for Kite (see docs/TESTING.md).

Parses the `// @...` header of a .kite file, builds input buffers with the
xorshift32 generator that the C++ interpreter must reproduce bit-for-bit, and
reads/writes/compares the output format printed by `kitec run`.

Pure Python + NumPy; no GPU needed. Used by run_gpu.py, bench.py and
tests/run_tests.py.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- RNG ------
class XorShift32:
    """Marsaglia xorshift32. Seed 0 is replaced by 0x9E3779B9 (state must be != 0)."""

    def __init__(self, seed: int):
        self.s = (seed & 0xFFFFFFFF) or 0x9E3779B9

    def next_u32(self) -> int:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s
        return s

    def next_f32(self) -> float:
        """Uniform in [-1, 1): 24 random bits, exact in f32."""
        return ((self.next_u32() >> 8) / 16777216.0) * 2.0 - 1.0

    def next_i32(self) -> int:
        """Uniform integer in [-100, 100]."""
        return int((self.next_u32() >> 8) % 201) - 100


# --------------------------------------------------------------- spec ------
@dataclass
class Arg:
    name: str
    elem: str               # "i32" | "f32"
    count: int | None       # None => scalar
    init: str               # scalar literal, or zeros|ones|iota|rand(S)|fill(V)

    @property
    def is_buffer(self) -> bool:
        return self.count is not None


@dataclass
class Spec:
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[Arg] = field(default_factory=list)
    check: list[str] = field(default_factory=list)


_ARG_TY = re.compile(r"^(i32|f32)(?:\[(\d+)\])?$")


def _dims(words: list[str], what: str, line: int) -> tuple[int, int, int]:
    if not 1 <= len(words) <= 3:
        raise ValueError(f"line {line}: @{what} takes 1-3 integers")
    d = [int(w) for w in words] + [1] * (3 - len(words))
    return d[0], d[1], d[2]


def parse_spec(source: str) -> Spec:
    kernel = None
    grid = block = None
    args: list[Arg] = []
    check: list[str] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        m = re.match(r"\s*//\s*@(\w+)\s*(.*)$", line)
        if not m:
            continue
        key, rest = m.group(1), m.group(2).split()
        if key == "kernel":
            kernel = rest[0]
        elif key == "grid":
            grid = _dims(rest, "grid", lineno)
        elif key == "block":
            block = _dims(rest, "block", lineno)
        elif key == "arg":
            if len(rest) != 3:
                raise ValueError(f"line {lineno}: @arg NAME TYPE INIT")
            tm = _ARG_TY.match(rest[1])
            if not tm:
                raise ValueError(f"line {lineno}: bad type {rest[1]!r}")
            count = int(tm.group(2)) if tm.group(2) else None
            args.append(Arg(rest[0], tm.group(1), count, rest[2]))
        elif key == "check":
            check.extend(rest)
        else:
            raise ValueError(f"line {lineno}: unknown directive @{key}")
    if kernel is None or grid is None or block is None:
        raise ValueError("spec needs @kernel, @grid and @block")
    names = {a.name for a in args}
    for c in check:
        if c not in names:
            raise ValueError(f"@check {c}: no such @arg")
    return Spec(kernel, grid, block, args, check)


def make_input(arg: Arg):
    """Scalar -> Python int/float; buffer -> np.ndarray (int32 or float32)."""
    dtype = np.float32 if arg.elem == "f32" else np.int32
    if not arg.is_buffer:
        return float(arg.init) if arg.elem == "f32" else int(arg.init)
    n, init = arg.count, arg.init
    if init == "zeros":
        return np.zeros(n, dtype)
    if init == "ones":
        return np.ones(n, dtype)
    if init == "iota":
        return np.arange(n, dtype=dtype)
    m = re.fullmatch(r"fill\((.+)\)", init)
    if m:
        return np.full(n, float(m.group(1)) if arg.elem == "f32" else int(m.group(1)), dtype)
    m = re.fullmatch(r"rand\((\d+)\)", init)
    if m:
        g = XorShift32(int(m.group(1)))
        gen = g.next_f32 if arg.elem == "f32" else g.next_i32
        return np.array([gen() for _ in range(n)], dtype)
    raise ValueError(f"bad initializer {init!r} for {arg.name}")


# ------------------------------------------------------------- output ------
def format_outputs(spec: Spec, buffers: dict[str, np.ndarray]) -> str:
    """One line per checked buffer: `name: v0 v1 ...` (%.9g for f32, %d for i32)."""
    lines = []
    for a in spec.args:
        if a.name in spec.check:
            vals = buffers[a.name]
            fmt = (lambda v: "%.9g" % float(v)) if a.elem == "f32" else (lambda v: "%d" % int(v))
            lines.append(f"{a.name}: " + " ".join(fmt(v) for v in vals))
    return "\n".join(lines) + "\n"


def parse_outputs(text: str) -> dict[str, list[float]]:
    out = {}
    for line in text.strip().splitlines():
        name, _, rest = line.partition(":")
        out[name.strip()] = [float(v) for v in rest.split()]
    return out


def compare(expected: str, actual: str, rtol=1e-5, atol=1e-6) -> list[str]:
    """Return a list of human-readable mismatches (empty == equal within tolerance)."""
    e, a = parse_outputs(expected), parse_outputs(actual)
    problems = []
    if e.keys() != a.keys():
        return [f"buffers differ: {sorted(e)} vs {sorted(a)}"]
    for name in e:
        if len(e[name]) != len(a[name]):
            problems.append(f"{name}: length {len(e[name])} vs {len(a[name])}")
            continue
        for i, (x, y) in enumerate(zip(e[name], a[name])):
            same = (math.isnan(x) and math.isnan(y)) or abs(x - y) <= atol + rtol * abs(x)
            if not same:
                problems.append(f"{name}[{i}]: expected {x!r}, got {y!r}")
                if len(problems) > 10:
                    return problems
    return problems


if __name__ == "__main__":
    # Known-answer test: the C++ interpreter must print exactly these.
    g = XorShift32(1)
    print([g.next_u32() for _ in range(3)])        # [270369, 67634689, 2647435461]
    g = XorShift32(1)
    print([g.next_f32() for _ in range(3)])
    g = XorShift32(1)
    print([g.next_i32() for _ in range(3)])        # [-49, -16, -6]
