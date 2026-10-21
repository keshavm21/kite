# Interview Prep: Kite and the NVIDIA System Software (Compiler) Role

## 1. Resume entry (use the version that matches what is pushed)

**Title line**
> **Kite: Optimizing Compiler for a GPU Kernel Language** | *C++17, CMake, NVIDIA PTX, CUDA, Python* | Source Code

**After Milestone 4** (front end + backend)
- Built a compiler in C++17 for a CUDA-like kernel language, with a hand-written lexer, recursive-descent parser and type checker, lowering to a CFG-based three-address IR and emitting **NVIDIA PTX** that runs on GPUs through the driver JIT.
- Wrote an **IR interpreter** that simulates the GPU thread grid, with bounds-checked memory, as a correctness oracle; GPU outputs match it on [N] kernels (vector ops, softmax, matmul).
- Emitted `.rn`-rounded float ops to stop ptxas from contracting into FMA, making GPU and interpreter results **bit-exact** for `+ − × ÷ √`.

**After Milestone 6** (replace or extend the bullets above)
- Built an **SSA** middle end: Cooper–Harvey–Kennedy dominators, dominance frontiers, Cytron phi placement and renaming, and out-of-SSA translation with critical-edge splitting and parallel-copy sequentialization.
- Implemented constant folding, copy propagation, dominator-based **GVN**, mark-and-sweep DCE and CFG simplification behind a pass manager; reduced PTX instruction count by [X]% (geomean over [N] kernels).
- Checked every pass with an IR **verifier** and **differential testing** (interpreter `-O0` vs `-O1` vs GPU) in GitHub Actions CI.

**After Milestone 8** (add)
- Added shared memory and barriers (`bar.sync`) and loop-invariant code motion; a tiled matmul kernel reached [X] GFLOP/s on a T4, [Y]× the naive kernel, benchmarked against NVRTC and cuBLAS.

Fill in [X]/[N] only from `results/`. Never estimate them.

## 2. The 60-second pitch (practice saying it out loud)

> "Kite is a compiler I wrote in C++ for a small CUDA-like kernel language. It targets PTX, NVIDIA's virtual ISA, so the output runs on real GPUs through the driver's JIT. The front end is hand-written: lexer, recursive-descent parser, type checker. It lowers to a CFG-based three-address IR. At -O1 the IR goes into SSA using dominance frontiers, runs constant folding, GVN, DCE, CFG simplification and LICM, and comes back out of SSA with parallel copies. For correctness I wrote an interpreter that simulates the thread grid, and I check every pass by differential testing: same inputs at -O0, at -O1 and on the GPU must agree. One thing I learned: since PTX has virtual registers, register allocation belongs to ptxas, so I measured my optimizations at both the PTX and the SASS level and found [what you found]."

## 3. Questions you should expect, and where your answer lives

**About Kite**
| Question | Your answer is in |
|---|---|
| Walk me through what happens to `y[i] = a * x[i] + y[i]`. | IR.md §7, PTX_BACKEND.md §5, saxpy_O0.ptx |
| Why didn't you do register allocation? | PROJECT_SPEC §4, PTX_BACKEND §1 |
| How do you construct SSA? What is a dominance frontier? Why do phis go there? | IMPLEMENTATION_PLAN M5 |
| What are the lost-copy and swap problems? | IMPLEMENTATION_PLAN M5.4 |
| Why is `x + 0.0 → x` wrong? Why can't signed `x / 2` become `x >> 1`? | IMPLEMENTATION_PLAN M6, M8.2 |
| Why `mul.rn.f32` instead of `mul.f32`? | PTX_BACKEND.md §7 |
| Why `setp.neu.f32` for `!=`? | PTX_BACKEND.md §5 |
| How do you know your optimizer is correct? | TESTING.md §1, §6 |
| Why can't LICM hoist a division? Or a load? | IMPLEMENTATION_PLAN M8.1 |
| What happens if only some threads reach `sync()`? | LANGUAGE.md §5 |
| Did your optimizations make the kernel faster? Why or why not? | PTX_BACKEND.md §11 + your results |
| What would you do next? | IMPLEMENTATION_PLAN stretch goals |

**General compiler questions (from the job description)**
- The phases of a compiler. Front end vs middle end vs back end. What an IR is and why compilers use one.
- LL vs LR parsing (your course project used Bison's LALR(1); Kite uses recursive descent; compare them).
- Basic blocks, CFG, dominators, natural loops, reducible CFGs.
- Dataflow analysis: liveness (backward), reaching definitions (forward), the lattice and the fixed point.
- SSA: phi, dominance frontier, why SSA makes optimizations simpler.
- Optimizations: constant propagation, copy propagation, CSE/GVN, DCE, LICM, strength reduction, inlining, loop unrolling, vectorization.
- Back end: instruction selection, **register allocation** (graph coloring, linear scan, spilling), instruction scheduling.
- LLVM basics: IR, `opt` passes, `mem2reg`, `-O2` pipeline. NVIDIA's NVVM is LLVM-based, so expect at least one LLVM question.

**Linker and loader (the JD names "Compiler/Linker"; Kite doesn't cover this, so study it)**
- Object files and ELF sections (`.text .data .bss .rodata .symtab .rel*`).
- Symbol resolution: strong/weak symbols, duplicate and undefined symbols, static library ordering.
- Relocation: what it is, absolute vs PC-relative, why it exists.
- Static vs dynamic linking; shared libraries; PIC; **GOT and PLT**, lazy binding.
- Tools: `nm`, `objdump -d -r`, `readelf -a`, `ldd`. Compile a two-file C program and inspect every step.
- Reading: *CS:APP* (Bryant & O'Hallaron), chapter 7 (Linking). The best single resource.

**Computer architecture (assessment topic)**
- Pipelining, hazards, branch prediction; the memory hierarchy, caches (direct-mapped, set-associative, write policies), locality, TLB.
- **GPU:** SIMT, warps (32 threads), divergence, occupancy, register pressure, coalesced memory access, shared memory banks and bank conflicts, latency hiding. Kite gives you concrete examples of each.
- Reading: *CS:APP* chapters 4–6; NVIDIA *CUDA C++ Programming Guide* chapters 1–5 and the Performance Guidelines chapter.

**C and C++ (assessment topic)**
- Pointers, memory layout (stack/heap/static), undefined behaviour (signed overflow, strict aliasing, uninitialized reads).
- `const`, references, RAII, smart pointers, move semantics, rule of 3/5/0, virtual functions and vtables, templates, STL containers and their complexities.
- Bit manipulation (HackerRank loves it).
- You're using all of this in Kite, so point to real code when you can.

**ML exposure (assessment topic)**
- What happens on a GPU in a forward pass: matmul/GEMM, elementwise ops, reductions, softmax, why kernel fusion matters (memory bandwidth).
- How ML compilers work at a high level: Triton, XLA, TVM, `torch.compile`. All lower to PTX on NVIDIA GPUs, which is exactly where Kite sits.

## 4. Before every interview

- Rebuild Kite from a clean clone and run the tests, so nothing is broken when you share your screen.
- Re-read your own `README.md` and the pass you're least sure about.
- Be able to draw on a whiteboard: the pipeline, a CFG with its dominator tree, and the SSA form of `row_sum`.
- Know your numbers (`results/`) and your known limitations.
