#!/usr/bin/env python3
"""Random valid Kite kernels for differential fuzzing (docs/TESTING.md section 6).

    python3 tests/fuzz/gen.py SEED          # print the kernel for SEED

The generator is deterministic in SEED. Programs avoid undefined behaviour by construction:
  * integer divisors are `abs(e % 5) + 1` (1..5) or a constant other than 0 and -1,
  * every array index is `((e % N) + N) % N`, and each thread owns a private slice of `buf`,
  * loops have a bounded trip count (at most 11 iterations, nesting depth at most 3),
  * f32 -> i32 casts clamp first (`min`/`max` return the non-NaN operand, so NaN becomes -1000.0).
It does use integer wraparound, f32 inf/NaN, short-circuit `&&`/`||`, `break`, `continue` and early `return`.
Anything the generator still gets wrong is caught by the driver, which discards programs whose `-O0` run
reports UB (tests/fuzz/fuzz.py).
"""
import random
import sys

THREADS = 4          # @block
N_INT, N_FLT = 5, 3  # per-thread result slots written at the end
BUF = 8              # per-thread scratch slice of `buf`
INTS = [0, 1, 2, 3, 4, 5, 7, 8, 16, 100, -1, -2, -8, 2147483647, -2147483648]
FLTS = ["0.0", "1.0", "0.5", "2.5", "3.0", "-1.5", "0.1", "100.0"]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.counter = 0
        self.ints = []    # readable i32 names in scope (innermost scopes last)
        self.flts = []
        self.wints = []   # assignable i32 names (never loop counters)
        self.wflts = []
        self.loops = 0    # loop nesting depth
        self.pool = []    # int expressions already generated in the current scope, reused to create common subexpressions
        self.budget = self.r.randint(10, 40)  # statements

    def fresh(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    # ------------------------------------------------------------- expressions --
    def idx(self, e, n):
        return f"((({e}) % {n} + {n}) % {n})"

    def int_lit(self):
        v = self.r.choice(INTS) if self.r.random() < 0.6 else self.r.randint(-20, 20)
        return f"({v})" if v < 0 else str(v)

    def slot(self, e):
        """Index into this thread's slice of buf: often a fixed slot, so loads and stores of one address meet."""
        if self.r.random() < 0.5:
            return f"t * {BUF} + {self.r.randint(0, 3)}"
        return f"t * {BUF} + {self.idx(e, BUF)}"

    def int_expr(self, d):
        if d > 0 and self.pool and self.r.random() < 0.12:
            return self.r.choice(self.pool)
        e = self._int_expr(d)
        if d > 0 and len(self.pool) < 30:
            self.pool.append(e)
        return e

    def _int_expr(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.2:
            return self.int_lit() if r.random() < 0.35 else r.choice(self.ints)
        k = r.random()
        a = lambda: self.int_expr(d - 1)
        if k < 0.40:
            return f"({a()} {r.choice('+-*')} {a()})"
        if k < 0.50:  # multiply by a constant, in either order (strength reduction)
            c = self.r.choice([2, 4, 8, 16, 3, 6, 1, 0])
            return f"({a()} * {c})" if r.random() < 0.5 else f"({c} * {a()})"
        if k < 0.58:
            if r.random() < 0.5:  # constant divisor, so strength reduction and folding see `x / 2^k`
                return f"({a()} {r.choice('/%')} {r.choice(['2', '3', '4', '8', '16', '(-2)', '(-4)'])})"
            return f"({a()} {r.choice('/%')} (abs({a()} % 5) + 1))"
        if k < 0.63:
            return f"(-{a()})"
        if k < 0.70:
            return f"{r.choice(['min', 'max'])}({a()}, {a()})"
        if k < 0.73:
            return f"abs({a()})"
        if k < 0.80:
            return f"({self.bool_expr(d - 1)} as i32)"
        if k < 0.86:
            return f"a[{self.idx(a(), 32)}]"
        if k < 0.92:
            return f"buf[{self.slot(a())}]"
        if k < 0.97:
            return f"(min(max({self.flt_expr(d - 1)}, -1000.0), 1000.0) as i32)"
        return f"{r.choice(['thread_idx.x', 'block_dim.x', 'block_idx.x', 'grid_dim.x'])}"

    def flt_expr(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.2:
            if r.random() < 0.4 or not self.flts:
                v = r.choice(FLTS)
                return f"({v})" if v.startswith("-") else v
            return r.choice(self.flts)
        k = r.random()
        a = lambda: self.flt_expr(d - 1)
        if k < 0.50:
            return f"({a()} {r.choice('+-*/')} {a()})"
        if k < 0.58:
            return f"(-{a()})"
        if k < 0.66:
            return f"{r.choice(['min', 'max'])}({a()}, {a()})"
        if k < 0.70:
            return f"abs({a()})"
        if k < 0.76:
            return f"sqrt({a()})"
        if k < 0.80:
            return f"exp(min(max({a()}, -20.0), 20.0))"
        if k < 0.90:
            return f"({self.int_expr(d - 1)} as f32)"
        return f"fa[{self.idx(self.int_expr(d - 1), 16)}]"

    def bool_expr(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.15:
            return r.choice(["true", "false"]) if r.random() < 0.2 else f"({self.int_expr(0)} {r.choice(['<', '>', '==', '!='])} {self.int_expr(0)})"
        k = r.random()
        if k < 0.45:
            return f"({self.int_expr(d - 1)} {r.choice(['<', '<=', '>', '>=', '==', '!='])} {self.int_expr(d - 1)})"
        if k < 0.58:
            return f"({self.flt_expr(d - 1)} {r.choice(['<', '<=', '>', '>=', '==', '!='])} {self.flt_expr(d - 1)})"
        if k < 0.80:
            return f"({self.bool_expr(d - 1)} {r.choice(['&&', '||'])} {self.bool_expr(d - 1)})"
        if k < 0.90:
            return f"(!{self.bool_expr(d - 1)})"
        return f"({self.bool_expr(d - 1)} == {self.bool_expr(d - 1)})"

    # -------------------------------------------------------------- statements --
    def block(self, depth, ind):
        """Statements for one `{ ... }`; names declared inside leave scope at the end."""
        mark = (len(self.ints), len(self.flts), len(self.wints), len(self.wflts), len(self.pool))
        out = []
        for _ in range(self.r.randint(1, 4)):
            if self.budget <= 0:
                break
            out += self.stmt(depth, ind)
        del self.ints[mark[0]:], self.flts[mark[1]:], self.wints[mark[2]:], self.wflts[mark[3]:], self.pool[mark[4]:]
        return out

    def stmt(self, depth, ind):
        r = self.r
        self.budget -= 1
        p = "    " * ind
        k = r.random()
        e = r.randint(1, 3)
        if k < 0.25 and self.wints:
            v = r.choice(self.wints)
            op = r.choice(["=", "=", "+=", "-=", "*="])
            return [f"{p}{v} {op} {self.int_expr(e)};"]
        if k < 0.35 and self.wflts:
            v = r.choice(self.wflts)
            return [f"{p}{v} {r.choice(['=', '=', '+=', '-=', '*=', '/='])} {self.flt_expr(e)};"]
        if k < 0.42:
            return [f"{p}buf[{self.slot(self.int_expr(e))}] = {self.int_expr(e)};"]
        if k < 0.48:
            name = self.fresh("x")
            line = f"{p}var {name}: i32 = {self.int_expr(e)};"
            self.ints.append(name)
            self.wints.append(name)
            return [line]
        if k < 0.52:
            name = self.fresh("g")
            line = f"{p}var {name}: f32 = {self.flt_expr(e)};"
            self.flts.append(name)
            self.wflts.append(name)
            return [line]
        if k < 0.68 and depth < 3:
            lines = [f"{p}if ({self.bool_expr(e)}) {{"] + self.block(depth + 1, ind + 1)
            if r.random() < 0.5:
                lines += [f"{p}}} else {{"] + self.block(depth + 1, ind + 1)
            return lines + [f"{p}}}"]
        if k < 0.84 and depth < 3:
            c = self.fresh("c")
            bound = str(r.randint(0, 8)) if r.random() < 0.6 else self.idx(self.int_expr(2), 6)
            self.ints.append(c)  # readable, not assignable
            self.loops += 1
            body = self.block(depth + 1, ind + 1)
            self.loops -= 1
            self.ints.remove(c)
            if r.random() < 0.7:
                return [f"{p}for (var {c}: i32 = 0; {c} < {bound}; {c} += 1) {{"] + body + [f"{p}}}"]
            # while: step first so `continue` cannot skip it
            return [f"{p}var {c}: i32 = 0;", f"{p}while ({c} < {bound}) {{", f"{p}    {c} += 1;"] + body + [f"{p}}}"]
        if k < 0.92 and self.loops:
            return [f"{p}if ({self.bool_expr(e)}) {{ {r.choice(['break', 'continue'])}; }}"]
        if k < 0.935:
            return [f"{p}if ({self.bool_expr(e)}) {{ return; }}"]
        return [f"{p}buf[{self.slot(self.int_expr(e))}] = {self.int_expr(e)};"]

    def program(self, seed):
        self.ints = ["n", "t", "m"]
        header = [
            "// Generated by tests/fuzz/gen.py, seed %d" % seed,
            "// @kernel fuzz",
            "// @grid 1",
            f"// @block {THREADS}",
            f"// @arg n i32 {self.r.randint(-5, 40)}",
            f"// @arg m i32 {self.r.choice([0, 1, -1, 7, 2147483647, -2147483648])}",
            "// @arg a i32[32] rand(%d)" % self.r.randint(1, 1000),
            "// @arg fa f32[16] rand(%d)" % self.r.randint(1, 1000),
            f"// @arg out i32[{THREADS * N_INT}] zeros",
            f"// @arg fout f32[{THREADS * N_FLT}] zeros",
            f"// @arg buf i32[{THREADS * BUF}] zeros",
            "// @check out fout buf",
            "",
            "kernel fuzz(n: i32, m: i32, a: *i32, fa: *f32, out: *i32, fout: *f32, buf: *i32) {",
            "    var t: i32 = thread_idx.x;",
        ]
        body = []
        for j in range(N_INT):  # slots first: later statements may read earlier ones
            body.append(f"    var v{j}: i32 = {self.int_expr(2)};")
            self.ints.append(f"v{j}")
            self.wints.append(f"v{j}")
        for j in range(N_FLT):
            body.append(f"    var f{j}: f32 = {self.flt_expr(2)};")
            self.flts.append(f"f{j}")
            self.wflts.append(f"f{j}")
        while self.budget > 0:
            body += self.stmt(0, 1)
        tail = [f"    out[t * {N_INT} + {j}] = v{j};" for j in range(N_INT)]
        tail += [f"    fout[t * {N_FLT} + {j}] = f{j};" for j in range(N_FLT)]
        return "\n".join(header + body + tail + ["}", ""])


def generate(seed: int) -> str:
    return Gen(seed).program(seed)


if __name__ == "__main__":
    sys.stdout.write(generate(int(sys.argv[1])))
