# Testing Strategy

A compiler is only as good as its tests. Kite's testing story is also one of its strongest interview talking points: **"every optimization is checked by an IR verifier and by differential testing against an interpreter, and the GPU output is checked against the same oracle."**

## 1. The four layers

| Layer | What | Tool | Runs where |
|---|---|---|---|
| **Unit** | Lexer tokens, parser shapes, dominator trees, folding rules, parallel-copy sequencing | doctest (C++) | Mac + CI |
| **Golden** | `--emit=ir` / `--emit=ptx` output for small inputs matches checked-in files | `tests/run_tests.py` | Mac + CI |
| **Error** | Invalid programs produce the right `line:col: error:` message | `tests/run_tests.py` | Mac + CI |
| **Differential** | Same program, same inputs: interpreter `-O0` == interpreter `-O1` == GPU | `tests/run_tests.py`, `gpu/run_gpu.py` | Mac + CI (interp), Colab (GPU) |

Plus: **the IR verifier** runs after lowering always, and after every pass with `--verify-each`. The test runner always passes `--verify-each`.

## 2. The `@spec` header

Every executable test and example carries its launch configuration and inputs as comments at the top, so the interpreter and the GPU runner use **identical inputs**:

```kite
// @kernel saxpy                 kernel to launch
// @grid 4                       1-3 integers (x [y [z]]), default 1
// @block 256                    1-3 integers
// @arg n i32 1000               scalar: name, type, value
// @arg a f32 2.5
// @arg x f32[1000] rand(1)      buffer: name, elem[count], initializer
// @arg y f32[1000] rand(2)
// @check y                      buffers to print after the run (one or more)
```

- `@arg` lines must be in the same order as the kernel's parameters. The C++ side should check names and types against the kernel signature and fail clearly on a mismatch.
- Buffer initializers: `zeros`, `ones`, `iota` (0, 1, 2, ...), `fill(V)`, `rand(SEED)`.
- The reference parser is `gpu/kite_spec.py`. Your C++ `src/interp/spec.cpp` must accept the same syntax.

### 2.1 The RNG (must match bit-for-bit in C++ and Python)

```cpp
struct XorShift32 {
  uint32_t s;
  explicit XorShift32(uint32_t seed) : s(seed ? seed : 0x9E3779B9u) {}
  uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
  float nextF32() { return ((next() >> 8) / 16777216.0f) * 2.0f - 1.0f; }   // [-1, 1)
  int32_t nextI32() { return int32_t((next() >> 8) % 201) - 100; }           // [-100, 100]
};
```
**Known-answer test** (put this in a unit test):

| seed 1 | call 1 | call 2 | call 3 |
|---|---|---|---|
| `next()` | 270369 | 67634689 | 2647435461 |
| `nextF32()` | -0.99987411499 | -0.96850514412 | 0.23280811310 |
| `nextI32()` | -49 | -16 | -6 |

`nextF32` is exact in f32 (24 random bits, multiply by 2, subtract 1), so C++ and Python agree exactly.

### 2.2 Output format

`kitec run FILE.kite [-O0|-O1]` prints one line per `@check` buffer, in `@arg` order:
```
y: 1.98765433 -0.123456791 ...
```
f32 values use `%.9g` (round-trips exactly); i32 values use `%d`. `gpu/run_gpu.py` prints the same format, and `gpu/compare.py` compares two files (integers exact; floats within `rtol=1e-5, atol=1e-6`; `--exact` for bitwise).

## 3. The interpreter (your oracle)

Spec: `src/interp/interp.{h,cpp}`. It executes **IR** (any form: lowered, SSA or post-SSA), so it can check every stage of the pipeline.

- **Threads:** for each block of the grid, create `block_dim.x*y*z` thread states `{ regs, curBlock, prevBlock, ip, done }`.
- **Scheduling:** run thread 0 until it hits `barrier` or `ret`, then thread 1, and so on. When every thread is waiting at a barrier, release them all and repeat. If some threads are at a barrier and the rest have returned, report **"divergent barrier"** (UB #4). Without barriers this reduces to running each thread to completion.
- **Phis:** when control enters block `B` from `P`, read *all* phi inputs for edge `P→B` first, then write them all (phis are parallel assignments; sequential evaluation is a real bug).
- **Memory:** a pointer register holds a buffer id. `load`/`store` check `0 <= idx < size` (UB #1). One set of shared arrays per block.
- **Arithmetic:** i32 ops in `uint32_t` (wraparound without C++ UB), division checks for zero and `INT_MIN / -1`. f32 ops in `float` (not `double`!), `exp` → `std::exp(float)`, `min/max` → `std::fmin/std::fmax`, `ftoi` checks range and NaN (UB #3).
- **Step limit:** 10⁷ instructions per thread by default; fail with "step limit exceeded".
- Errors print the kernel, block, thread, IR instruction and the reason, e.g. `error: out-of-bounds load x[1000] (size 1000) in block 3 thread 232 at if.then.1:0`.

Floating-point note: compile the interpreter with default floating-point flags (no `-ffast-math`), and keep f32 operations separate (`float t = a * b; t = t + c;`). Compilers may contract `a*b+c` into an FMA: Apple Clang does it within an expression by default, and GCC does it more aggressively on ARM64. Add `-ffp-contract=off` to the interpreter's compile flags so its results match `mul.rn` + `add.rn` on the GPU exactly.

## 4. The test runner (`tests/run_tests.py`)

Plain Python 3, no dependencies beyond NumPy. Structure:

```
tests/
  golden/   foo.kite + foo.ir (expected --emit=ir -O0) + foo.O1.ir + foo.ptx
  errors/   bar.kite whose first line is   // ERROR: 3:15: cannot assign f32 to i32
  exec/     extra executable tests with @spec headers (examples/ are included automatically)
```

```
python3 tests/run_tests.py                 # run everything, print a summary, exit 1 on failure
python3 tests/run_tests.py -k softmax      # filter by name
python3 tests/run_tests.py --update        # rewrite golden files from current output (review the diff!)
python3 tests/run_tests.py --kitec build/kitec
```

Checks per category:
- **golden:** run `kitec --emit=ir|ptx`, diff against the file. Show a unified diff on failure.
- **errors:** run `kitec`, expect exit code 1, and expect the `line:col: message` from the `// ERROR:` line to appear in stderr.
- **exec:** run `kitec run -O0` and `kitec run -O1`, and require **exact** equality (the optimizer must not change any result). If `ptxas` is on `PATH`, also compile to PTX at both levels and assemble with `ptxas -arch=sm_75`.

## 5. What to unit-test (minimum list)

- Lexer: every token kind, `<=` vs `<`, float vs int, comments, line:col tracking, an unterminated comment error.
- Parser: precedence (`1 + 2 * 3`, `-x as f32`, `a || b && c`), else-if chains, `for` with assignment init.
- Dominators on hand-built CFGs: diamond, loop, nested loops, a CFG with an unreachable block.
- Dominance frontiers for the same CFGs (compare with a hand-computed answer).
- SSA: build `row_sum` IR, construct SSA, check that the verifier passes and that the phis sit where §8.2 of IR.md says.
- Parallel copy sequencing: `{a←b, b←a}` (swap needs a temp), `{a←b, b←c}` (order matters), `{a←a}` (dropped).
- Constant folding: `i32` wraparound (`2147483647 + 1`), no folding of division by zero, `x * 0` folds for i32 but **not** for f32 (NaN and -0.0), `x + 0.0` doesn't fold for f32 either (`-0.0 + 0.0 = +0.0`).

## 6. Fuzzing (stretch, high value)

`tests/fuzz/gen.py` generates random valid kernels: random expressions over a few i32 variables, nested ifs and bounded loops (`for (var k: i32 = 0; k < 8; k += 1)`), with stores of every variable to an output buffer at the end. It must avoid UB: guard divisions (`d != 0 && !(n == -2147483648 && d == -1)`), keep indices in range, and use only `i32` for exact comparison. Then for N seeds: `kitec run -O0` vs `kitec run -O1` must be identical; save any failing program to `tests/exec/fuzz_<seed>.kite`. One evening of this will find bugs that hand-written tests miss, and "I wrote a fuzzer that found N optimizer bugs" is a great resume line if it's true.

## 7. CI (`.github/workflows/ci.yml`)

On every push (Ubuntu): configure and build with CMake, run `ctest` (unit tests), `pip install numpy nvidia-cuda-nvcc-cu12` (for `ptxas`), then `python3 tests/run_tests.py --kitec build/kitec`. GPU tests are not in CI; run them on Colab and commit the results to `results/`.

## 8. Running on the GPU (Colab)

1. Colab → *Runtime → Change runtime type → T4 GPU*.
2. In a cell:
   ```bash
   !git clone https://github.com/keshavm21/kite && cd kite && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
   !nvidia-smi && ptxas --version
   !python3 gpu/run_hello.py                      # Milestone 0 check
   ```
3. Per example: `kitec -O1 X.kite -o X.ptx`, then `ptxas -arch=sm_75 -v X.ptx -o X.cubin`, then `run_gpu.py X.kite X.ptx > gpu.out`, then `kitec run -O1 X.kite > interp.out`, then `compare.py interp.out gpu.out`.
4. Wrap step 3 in `gpu/check_all.sh` so one command checks every example.

If `cp.RawModule(path=...)` gives you trouble, the PyCUDA equivalent is `pycuda.driver.module_from_buffer(open(p,'rb').read()).get_function(name)`.
