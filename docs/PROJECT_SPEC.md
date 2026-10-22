# Kite: Project Specification

**One line:** Kite is an optimizing compiler, written in C++17, for a small GPU kernel language. It compiles to **NVIDIA PTX**, runs on real NVIDIA GPUs, and is checked by an IR interpreter used as a correctness oracle.

**Full title:** *Kite — Optimizing Compiler for a GPU Kernel Language Targeting NVIDIA PTX*

---

## 1. The problem

NVIDIA's CUDA toolchain works like this:

```
CUDA C++ ──► NVVM (LLVM-based front/middle end) ──► PTX ──► ptxas ──► SASS (machine code)
                                                     ▲
                                        virtual ISA, JIT-compiled by the driver
```

PTX is NVIDIA's *virtual* instruction set. It has unlimited typed virtual registers, explicit memory spaces (global, shared, param) and thread-index special registers. Any compiler that wants to run code on an NVIDIA GPU emits PTX. Examples are NVCC, Triton, Numba, JAX/XLA and Halide.

Kite is a small, complete version of that pipeline:

```
.kite source ─► Lexer ─► Parser ─► AST ─► Semantic analysis ─► IR (CFG of basic blocks)
                                                                 │
                     ┌──────────── -O1 ──────────────────────────┤ -O0
                     ▼                                           │
             SSA construction                                    │
             optimization passes                                 │
             SSA destruction                                     │
                     └───────────────────┬───────────────────────┘
                                         ▼
                         PTX backend ─► .ptx ─► GPU (driver JIT)
                                         │
              IR interpreter ◄───────────┘ (oracle: same inputs, same outputs?)
```

The question the project answers is: **can I take a kernel language from source text all the way to code that runs correctly on an NVIDIA GPU, optimize it, and prove each optimization preserves behaviour?**

## 2. What Kite demonstrates

| Area | Where Kite shows it |
|---|---|
| C/C++ | The whole compiler is modern C++17, with no parser generators or LLVM dependency. |
| Compiler knowledge | Hand-written lexer and recursive-descent parser, type checker, CFG IR, dominators, SSA, dataflow, optimization passes. |
| PTX / GPU | A real PTX backend that `ptxas` accepts and that runs on a GPU; shared memory and barriers. |
| Computer architecture | Thread/block model, memory spaces, coalescing, register pressure (`ptxas -v`), SIMT divergence. |
| ML exposure | Example kernels are ML ops (ReLU, softmax, reductions, matmul), benchmarked against the libraries. |
| Engineering discipline | IR verifier, differential testing against the interpreter, golden tests, CI. |

Kite has a **hand-written** front end (no parser generators), an **SSA optimizer**, and a **real GPU backend**, and every stage is checked against an interpreter that serves as a correctness oracle.

## 3. Goals

1. **G1 — End to end:** every kernel in `examples/` compiles to PTX that `ptxas` accepts, runs on an NVIDIA GPU (Colab T4 is enough), and matches the interpreter's output.
2. **G2 — Real optimizer:** SSA-based passes (constant folding, copy propagation, dead-code elimination, CFG simplification, dominator-based CSE, LICM). Their effect is *measured*: IR and PTX instruction counts before and after, plus `ptxas -v` register counts.
3. **G3 — Proven correct:** every pass is checked by (a) the IR verifier and (b) differential testing (interpreter `-O0` vs `-O1` vs GPU).
4. **G4 — Explainable:** every design decision is documented with its reasoning, not only its result.

## 4. Non-goals

- Not a CUDA C++ compiler. The language is deliberately tiny (see `LANGUAGE.md`).
- **No register allocation.** PTX has unlimited virtual registers; `ptxas` allocates physical registers. This is a real design property of PTX.
- No LLVM and no parser generators. Building these yourself is the point.
- No device functions, structs, 64-bit types, unsigned types or atomics (atomics are a possible stretch goal).
- Not competing with nvcc/cuBLAS on speed. The benchmarks are honest measurements, not marketing.

## 5. Scope tiers

Build in this order. Every tier leaves a working project.

### Tier 1 — MVP (Milestones 0–4): "a kernel language that runs on the GPU"
- Lexer, recursive-descent parser, AST, AST printer
- Semantic analysis: scopes, types, all errors in `LANGUAGE.md §8`, with `file:line:col` diagnostics
- IR: CFG of basic blocks, three-address instructions, printer, **verifier**
- AST → IR lowering, including short-circuit `&&`/`||`, loops, `break`/`continue`/`return`
- **IR interpreter** that simulates a grid of threads, with bounds-checked memory
- **PTX backend** (no optimization): all examples except the shared-memory ones run on the GPU
- Test spec format, golden tests, differential tests, CI

### Tier 2 — Optimizer (Milestones 5–6): "an *optimizing* compiler"
- Analyses: RPO, dominator tree (Cooper–Harvey–Kennedy), dominance frontiers, liveness
- SSA construction (Cytron et al.: phi insertion on iterated dominance frontiers, renaming) and SSA destruction (critical-edge splitting, parallel copies)
- Passes: `constfold` (constant propagation and folding plus algebraic identities), `copyprop`, `dce`, `simplifycfg`, `gvn` (dominator-scoped value numbering)
- Pass manager with `-O0`/`-O1`, `--passes=`, `--print-after-all`, `--stats`

### Tier 3 — GPU features (Milestone 7)
- `shared` arrays and `sync()` → `.shared` and `bar.sync`; the interpreter simulates barriers
- `block_sum` and `matmul_tiled` run correctly

### Tier 4 — Results (Milestone 8)
- `licm` (natural loops, preheaders, hoisting) and `strength-reduce` (`x * 2^k` → `shl`)
- Benchmarks on the GPU: Kite `-O0` vs `-O1` vs NVRTC-compiled CUDA C vs library (CuPy/cuBLAS)
- README with results table, `docs/architecture.md`, a demo GIF or asciinema recording

### Stretch (only after Tier 4)
- A random program generator (Csmith-style fuzzer) for differential testing of the optimizer
- `atomic_add` → `atom.global.add.f32`
- Peephole: `mul`+`add` (i32) → `mad.lo.s32`; constant offsets folded into `[%rd+off]` addressing
- A C++ runner using the CUDA Driver API (`cuModuleLoadDataEx`, `cuLaunchKernel`)
- Device functions with an inlining pass

## 6. Deliverables

1. `kitec`, a command-line compiler:
   ```
   kitec FILE.kite [-o OUT.ptx] [-O0|-O1] [--emit=tokens|ast|ir|ptx]
         [--passes=a,b,c] [--print-after-all] [--verify-each] [--stats] [--target=sm_75]
   kitec run FILE.kite [-O0|-O1]        # interpret using the @spec header, print outputs
   ```
2. Test suites: C++ unit tests, golden IR/PTX tests, error tests, differential tests; GitHub Actions CI (CPU-only).
3. `gpu/` scripts: run any example on a GPU from its spec header, compare with the interpreter, benchmark.
4. `results/`: measured tables (instruction counts, registers, timings) produced by scripts, never typed by hand.
5. Docs: README, architecture, language reference, IR reference.

## 7. Success criteria (definition of done)

- [ ] `ctest` and `python3 tests/run_tests.py` pass in CI on every push
- [ ] Every example: interpreter `-O0` output == interpreter `-O1` output (exactly)
- [ ] Every example: GPU output matches the interpreter (integers exact; floats within `rtol=1e-5, atol=1e-6`)
- [ ] `--verify-each` passes for every example and every test
- [ ] `results/opt_stats.md` shows per-kernel instruction counts at `-O0` vs `-O1`
- [ ] You can draw the pipeline on a whiteboard and explain SSA construction without notes

## 8. Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | C++17 | The role tests C++. No exotic features needed. |
| Build | CMake ≥ 3.16 | Standard; works on macOS and Linux CI. |
| Unit tests | doctest (single header, vendored) | No dependency hassle. |
| Test runner | Python 3 script | Golden files, error tests and differential tests are easiest in Python. |
| GPU runs | Google Colab (free T4, `sm_75`) + CuPy | You develop on a Mac; NVIDIA GPUs are only needed for GPU tests and benchmarks. |
| PTX validation | `ptxas` (in the CUDA toolkit on Colab; or `pip install nvidia-cuda-nvcc-cu12` on Linux) | Rejects invalid PTX with exact line numbers. |

Everything except GPU execution runs on your Mac. The interpreter is what makes this possible: it is your "GPU" during development.

## 9. Repository layout

```
kite/
├── CMakeLists.txt
├── README.md
├── docs/            PROJECT_SPEC, LANGUAGE, IR, PTX_BACKEND, IMPLEMENTATION_PLAN, TESTING
├── src/
│   ├── main.cpp                 driver: flags, pipeline, exit codes
│   ├── support/                 diagnostics.{h,cpp}, source.{h,cpp} (file + line/col)
│   ├── frontend/                token.h, lexer.{h,cpp}, ast.h, parser.{h,cpp}, ast_printer.cpp, sema.{h,cpp}
│   ├── ir/                      ir.{h,cpp}, builder.{h,cpp}, printer.cpp, verifier.cpp, lower.{h,cpp}
│   ├── analysis/                cfg.{h,cpp} (preds/succs/RPO), dominators.{h,cpp}, liveness.{h,cpp}, loops.{h,cpp}
│   ├── transform/               ssa_construct.cpp, ssa_destruct.cpp, constfold.cpp, copyprop.cpp, dce.cpp,
│   │                            simplifycfg.cpp, gvn.cpp, licm.cpp, strength_reduce.cpp, pass_manager.{h,cpp}
│   ├── interp/                  interp.{h,cpp}, spec.{h,cpp} (@spec header parser + xorshift RNG)
│   └── codegen/                 ptx.{h,cpp}
├── tests/
│   ├── unit/                    doctest files: lexer, parser, dominators, ssa, folding ...
│   ├── golden/                  *.kite + *.ir / *.ptx expected outputs
│   ├── errors/                  *.kite with `// ERROR:` expectations
│   ├── ptx_reference/           hand-checked PTX (provided)
│   └── run_tests.py
├── examples/                    10 kernels with @spec headers (provided)
├── gpu/                         hello.ptx, run_hello.py, run_gpu.py, compare.py, kite_spec.py (provided);
│                                bench.py, sass_opcodes.py, check_all.sh
├── results/                     generated tables
└── .github/workflows/ci.yml
```
