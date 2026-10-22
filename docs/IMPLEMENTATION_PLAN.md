# Kite: Implementation Plan

Nine milestones, about **13 focused days** in total. Each milestone ends with something that runs, a passing test suite, and a commit. **Milestone 4 is the minimum end-to-end version** (about 6.5 days); everything after it upgrades the project.

| # | Milestone | Days | Running total | What exists at this point |
|---|---|---|---|---|
| 0 | Setup + run hand-written PTX on a GPU | 0.5 | 0.5 | — |
| 1 | Lexer + parser + AST printer | 1.5 | 2 | — |
| 2 | Semantic analysis | 1 | 3 | — |
| 3 | IR, lowering, verifier, interpreter | 2 | 5 | — |
| 4 | **PTX backend: examples run on the GPU** | 1.5 | **6.5** | a compiler for a GPU kernel language targeting PTX |
| 5 | CFG analyses + SSA construction/destruction | 2 | 8.5 | + SSA-based IR |
| 6 | Optimization passes + pass manager | 2 | 10.5 | + optimizer, with measured results |
| 7 | Shared memory + barriers | 1 | 11.5 | + shared-memory tiled matmul |
| 8 | LICM, benchmarks, README | 1.5 | 13 | LICM, benchmarks with numbers, documentation |

**Rules for the whole project**
1. Read the relevant doc section before starting a milestone: `LANGUAGE.md` for M1–2, `IR.md` for M3 and M5–6, `PTX_BACKEND.md` for M4 and M7, `TESTING.md` throughout.
2. Write tests in the same sitting as the code. A milestone isn't done until its tests pass.
3. Commit at least once per milestone with a clear message.
4. Claim only what is pushed, working and measured.
5. Stuck for more than 45 minutes on one bug? Shrink the input to the smallest failing kernel, then dump `--emit=ir` and `--print-after-all`. Check the verifier first.

---

## Milestone 0 — Setup and a first GPU run (0.5 day)

**Goal:** prove the toolchain end to end *before* writing any compiler code.

1. Create the repo `keshavm21/kite`. Copy in everything from this starter pack (`docs/`, `examples/`, `gpu/`, `tests/ptx_reference/`, `CMakeLists.txt`, `.gitignore`, `.github/`).
2. Mac setup:
   ```bash
   xcode-select --install
   brew install cmake python
   pip3 install numpy
   mkdir -p third_party/doctest
   curl -L https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h -o third_party/doctest/doctest.h
   ```
3. Create the empty skeleton: `src/main.cpp` printing `kitec 0.1`, plus `tests/unit/test_main.cpp`:
   ```cpp
   #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
   #include "doctest/doctest.h"
   TEST_CASE("skeleton") { CHECK(1 + 1 == 2); }
   ```
   (Only one test file defines `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`; the others just `#include "doctest/doctest.h"`.)
4. `cmake -B build && cmake --build build && ./build/kitec && ctest --test-dir build` should all work. The provided `CMakeLists.txt` was tested with exactly this skeleton, sanitizers on.
5. Open Google Colab with a T4 GPU, clone the repo, and run `python3 gpu/run_hello.py`. You should see `OK`.
6. Read `gpu/hello.ptx` line by line next to `PTX_BACKEND.md §2–5` until every line makes sense. This file is the target your compiler must produce.

**Done when:** CI is green on GitHub with the skeleton, and `run_hello.py` prints `OK` on Colab.

---

## Milestone 1 — Lexer and parser (1.5 days)

**Files:** `src/support/source.{h,cpp}`, `src/support/diagnostics.{h,cpp}`, `src/frontend/token.h`, `lexer.{h,cpp}`, `ast.h`, `parser.{h,cpp}`, `ast_printer.cpp`

### 1.1 Diagnostics (do this first; everything uses it)
```cpp
struct SourceLoc { int line = 1, col = 1; };
class Diagnostics {
 public:
  explicit Diagnostics(std::string filename);
  void error(SourceLoc loc, const std::string& msg);    // prints "file:line:col: error: msg"
  bool hasErrors() const;
};
```

### 1.2 Lexer
```cpp
enum class Tok { Ident, IntLit, FloatLit, KwKernel, KwVar, /* ...all keywords... */,
                 Plus, Minus, Star, Slash, Percent, Assign, PlusAssign, /* ... */, Eof, Error };
struct Token { Tok kind; std::string text; SourceLoc loc; };
class Lexer { public: Lexer(const std::string& src, Diagnostics&); Token next(); };
```
- Hand-written loop over characters; track line and column.
- Keywords: lex an identifier, then look it up in a `std::unordered_map<std::string, Tok>`.
- Numbers: digits, then a `.` followed by a digit means float; otherwise int. Convert ints with `std::stoll` and range-check later in sema (E13); convert floats with `std::strtof`.
- Two-character operators: check the next character (`<` then `=` gives `<=`).
- Comments: skip `//` to newline; `/* */` with an error for an unterminated comment.
- Add `--emit=tokens` to the driver: one token per line, `line:col KIND 'text'`.

### 1.3 AST (`ast.h`)
```cpp
enum class Type { Error, Void, Bool, I32, F32, PtrI32, PtrF32, SharedI32, SharedF32 };

struct Expr { enum Kind { IntLit, FloatLit, BoolLit, Name, Index, Unary, Binary, Cast, Call, Builtin } kind;
              SourceLoc loc; Type type = Type::Error; /* set by sema */ };
struct BinaryExpr : Expr { BinOp op; std::unique_ptr<Expr> lhs, rhs; };
// ...one struct per kind (or a single struct with a union of fields; your choice)

struct Stmt { enum Kind { VarDecl, SharedDecl, Assign, If, While, For, Break, Continue, Return, Sync, Block } kind;
              SourceLoc loc; };
struct Kernel { std::string name; std::vector<Param> params; std::unique_ptr<BlockStmt> body; SourceLoc loc; };
struct Program { std::vector<Kernel> kernels; };
```
`std::unique_ptr` everywhere; the AST is a tree.

### 1.4 Parser (recursive descent)
One function per grammar rule in `LANGUAGE.md §3`:
```cpp
class Parser {
  Token cur_;                                // one token of lookahead is enough
  void advance();  bool check(Tok);  bool accept(Tok);  Token expect(Tok, const char* what);
  std::unique_ptr<Expr> parseOr();           // and_expr { "||" and_expr }
  std::unique_ptr<Expr> parseAnd();          // ...one per precedence level
  std::unique_ptr<Expr> parseCast();         // unary { "as" type }
  std::unique_ptr<Expr> parseUnary();
  std::unique_ptr<Expr> parsePrimary();
  std::unique_ptr<Stmt> parseStmt();
  ...
};
```
- A left-associative level is a loop: `lhs = parseNext(); while (check(op)) { advance(); rhs = parseNext(); lhs = Binary(op, lhs, rhs); }`.
- **Desugar compound assignment here**: `x += e` becomes `Assign(x, Binary(+, Name(x), e))`; for `p[i] += e`, clone the index expression. Write a small `cloneExpr`.
- On an error: report `expected ';' after statement` at the current token and stop (throw a `ParseError` caught in `main`). No error recovery needed.
- Add `--emit=ast`: an indented tree print. Make it deterministic; golden tests use it.

### 1.5 Tests
Unit-test the lexer on every token kind. Golden-test `--emit=ast` for every example. Add error tests for missing `;`, an unclosed `{`, and `1.` (a bad float).

**Done when:** all 10 examples parse; precedence tests pass (`1 + 2 * 3`, `-x as f32`, `a || b && c`); error tests pass.

**Pitfall:** `thread_idx.x`. The lexer emits `Ident(thread_idx) Dot Ident(x)`. Recognize builtins in `parsePrimary` by name.

---

## Milestone 2 — Semantic analysis (1 day)

**Files:** `src/frontend/sema.{h,cpp}`

```cpp
struct Symbol { enum Kind { Local, Param, Shared } kind; std::string name; Type type; int index; SourceLoc loc; };
class Scope {   // a stack of maps
  std::vector<std::unordered_map<std::string, Symbol*>> frames_;
 public:
  void push(); void pop();
  bool declare(Symbol*);                  // false if already in the innermost frame (E2)
  Symbol* lookup(const std::string&);     // innermost first
};
class Sema { public: Sema(Diagnostics&); void check(Program&); };
```
- Walk every kernel: push a scope, declare params, walk the body.
- `checkExpr(Expr&) -> Type`: compute the type bottom-up, store it in `expr.type`, and report errors E1, E3, E6, E9–E11, E13, E14. If an operand has type `Error`, return `Error` silently (no error cascades).
- Store the resolved `Symbol*` on each `Name` node; lowering needs it.
- `checkStmt`: E4, E5, E7, E8 (keep a loop-depth counter), E12.
- Write **one error test per row** in `LANGUAGE.md §8` (14 files in `tests/errors/`).

**Done when:** all examples pass sema; all 14 error tests produce the right line:col and message.

---

## Milestone 3 — IR, lowering, verifier, interpreter (2 days)

**Files:** `src/ir/ir.{h,cpp}`, `builder.{h,cpp}`, `printer.cpp`, `verifier.cpp`, `lower.{h,cpp}`, `src/analysis/cfg.{h,cpp}`, `src/interp/interp.{h,cpp}`, `spec.{h,cpp}`

### 3.1 IR data structures
Implement `IR.md §9`. Then:
- `Builder`: `setBlock(bb)`, `emitBinary(op, ty, a, b) -> Operand`, `emitBr(c, t, f)`, … Each `emit*` creates the destination register.
- `cfg::rebuild(fn)`: recompute `preds`/`succs` from terminators.
- `printer`: `IR.md §5` format. This is the golden output.

### 3.2 Lowering (AST → IR)
```cpp
class Lowerer {
  Function* fn; Builder b;
  std::unordered_map<const Symbol*, Operand> vars;            // var/param -> register, shared -> @arr
  struct Loop { BasicBlock* breakTo; BasicBlock* continueTo; };
  std::vector<Loop> loops;
  Operand lowerExpr(const Expr&);
  void lowerCond(const Expr&, BasicBlock* T, BasicBlock* F);  // branch on a bool expression
  void lowerStmt(const Stmt&);
};
```
| Construct | Lowering |
|---|---|
| `var x: T = e` | `r = newReg(T, "x")`; `copy r, lowerExpr(e)` |
| `x = e` | `copy vars[x], lowerExpr(e)` (same register again, hence not SSA) |
| `p[i] = e` | `store T global vars[p][lowerExpr(i)], lowerExpr(e)` |
| `if (c) {A} else {B}` | `lowerCond(c, then, else)`; then: A, `jmp end`; else: B, `jmp end`; continue in `end` |
| `while (c) {A}` | `jmp cond`; cond: `lowerCond(c, body, end)`; body: A, `jmp cond`; continue in `end` |
| `for (init; c; step) {A}` | init; `jmp cond`; cond: `lowerCond(c, body, end)`; body: A, `jmp step`; step: step, `jmp cond`; continue in `end`. **`continue` → `step`** |
| `break` / `continue` | `jmp loops.back().breakTo / continueTo` |
| `return` | `ret` |
| `sync()` | `barrier` |
| end of kernel | `ret` |

`lowerCond(e, T, F)` gives short-circuit evaluation:
- `a && b`: `mid = newBlock`; `lowerCond(a, mid, F)`; in `mid`: `lowerCond(b, T, F)`
- `a || b`: `mid = newBlock`; `lowerCond(a, T, mid)`; in `mid`: `lowerCond(b, T, F)`
- `!a`: `lowerCond(a, F, T)`
- otherwise: `br lowerExpr(e), T, F`

A `&&`/`||` in **value** position (e.g. `var f: bool = a || b`): `r = newReg(bool)`; `lowerCond(e, Tb, Fb)`; `Tb: copy r, true; jmp J`; `Fb: copy r, false; jmp J`; continue in `J`. The optimizer cleans this up later.

**Code after a terminator** (`return; x = 1;`) still needs a block. After emitting any terminator, switch to a fresh block named `dead.N`. It has no predecessors, and `simplifycfg` deletes it at `-O1`. At `-O0` it is harmless. The verifier allows unreachable blocks; dominator analysis must ignore them (run `removeUnreachable` before SSA).

### 3.3 Verifier
Implement every "Always" invariant in `IR.md §6` now. Add the SSA ones in M5. Report as `verifier: kernel saxpy, block if.then.1, inst 2: <problem>` and abort.

### 3.4 Interpreter
Follow `TESTING.md §3` exactly. Wire up `kitec run FILE.kite [-O0|-O1]`: parse the spec header (`src/interp/spec.cpp`, same syntax as `gpu/kite_spec.py`, same RNG; add its known-answer unit test), allocate buffers, run, and print the output format.

### 3.5 Test runner
Write `tests/run_tests.py` (`TESTING.md §4`): golden, error and exec categories. Seed golden files with `--update`, then **read every one** and check it by hand against `IR.md §7–8`.

**Done when:** `kitec run` prints correct results for every example **except** `block_sum` and `matmul_tiled` (shared memory comes in M7). Check them by hand against NumPy once with a short Python snippet (e.g. saxpy: `2.5*x + y`).

---

## Milestone 4 — PTX backend (1.5 days)

**Files:** `src/codegen/ptx.{h,cpp}`

```cpp
class PtxEmitter {
 public:
  PtxEmitter(const Module&, std::string target = "sm_75");
  std::string emit();
 private:
  std::string reg(int id) const;          // "%r7" / "%f7" / "%p7" / "%rd7" from fn.regTy[id]
  std::string operand(const Operand&);    // register, int, 0fXXXXXXXX, 1/0 for bool
  int newTemp(Ty);                        // backend temps, ids >= fn.regTy.size()
  void emitInst(const Inst&, const BasicBlock* next);
  std::ostringstream out_;
};
```
Steps:
1. Header and `.entry` signature (`PTX_BACKEND.md §2–4`).
2. Emit the body into a separate string first. Track the highest id used per register class, then write the `.reg` declarations, then the body.
3. One `case` per opcode, from the table in §5.
4. Branches and fall-through per §6.
5. `--emit=ptx` and `-o out.ptx`.

Test loop:
- Compare `kitec -O0 examples/saxpy.kite` against `tests/ptx_reference/saxpy_O0.ptx` by eye.
- Run `ptxas -arch=sm_75` on every example's PTX (in CI through `nvidia-cuda-nvcc-cu12`).
- On Colab: `run_gpu.py` + `compare.py` for every example except the shared-memory ones.

**Done when:** 8 examples produce identical results on the GPU and the interpreter (`exp` within tolerance), and CI assembles all PTX with ptxas. This is a complete end-to-end compiler.

---

## Milestone 5 — CFG analyses and SSA (2 days)

**Files:** `src/analysis/dominators.{h,cpp}`, `liveness.{h,cpp}`, `src/transform/ssa_construct.cpp`, `ssa_destruct.cpp`

### 5.1 Reverse postorder and dominators (Cooper, Harvey, Kennedy, *A Simple, Fast Dominance Algorithm*, 2001)
```
rpo = reverse postorder from entry          // reachable blocks only
idom[entry] = entry
repeat until no change:
  for b in rpo, skipping entry:
    new = the first pred p of b with idom[p] defined
    for every other pred p with idom[p] defined:
      new = intersect(p, new)
    if idom[b] != new: idom[b] = new; changed = true

intersect(a, b):
  while a != b:
    while rpoIndex[a] > rpoIndex[b]: a = idom[a]
    while rpoIndex[b] > rpoIndex[a]: b = idom[b]
  return a
```
Also build the dominator tree (children lists) and a `dominates(a, b)` helper (walk `idom` up from b; fine at this size).

### 5.2 Dominance frontiers (same paper)
```
for b with >= 2 preds:
  for p in preds(b):
    runner = p
    while runner != idom[b]:
      DF[runner].insert(b); runner = idom[runner]
```

### 5.3 SSA construction (Cytron et al. 1991)
Variables = registers with more than one definition (in lowered IR, exactly the `var`s that get reassigned).
```
for v in multiDefRegs:                       // 1. place phis
  work = blocks defining v; placed = {}
  while work not empty:
    X = work.pop()
    for Y in DF[X]:
      if Y not in placed:
        insert "v = phi" at top of Y (one operand slot per pred); placed += Y
        if Y does not define v: work.push(Y)

rename(entry)                                // 2. rename via dominator tree DFS
rename(B):
  for phi in B (for var v): d = newReg(type(v)); phi.dst = d; stack[v].push(d)
  for inst in B (non-phi):
    replace each use of a variable v with stack[v].top()   (or undef if empty)
    if inst defines variable v: d = newReg(...); inst.dst = d; stack[v].push(d)
  for S in succs(B):
    for phi in S for var v: set the operand for edge B to stack[v].top() (or undef)
  for C in domTreeChildren(B): rename(C)
  pop everything pushed in this call
```
Keep a map `phi → original variable` during construction. Run `removeUnreachable` first. Afterwards, `copyprop` + `dce` make the IR look like `IR.md §8.2`.

### 5.4 SSA destruction
1. Split every edge `P → S` where `P` has more than one successor and `S` has phis: new block `E` with `jmp S`; retarget `P`'s terminator; in `S`'s phis, rename incoming block `P` to `E`.
2. For each block `S` with phis, for each pred `P`: build the **parallel copy** `{phi.dst ← phi.incoming(P)}` (skip `undef`), **sequentialize** it, and insert it before `P`'s terminator.
3. Delete all phis.

Sequentializing a parallel copy (it must behave as if all sources are read first):
```
pending = [(d, s) for each copy, if d != s]
while pending:
  if some (d, s) in pending where d is not the source of any other pending copy:
    emit "d = copy s"; remove it
  else:                                       // only cycles remain, e.g. {a<-b, b<-a}
    (d, s) = pending[0]
    t = newReg(type(d)); emit "t = copy d"
    replace every pending source d with t
```
This is where the classic **lost-copy** and **swap** problems are solved. Both are worth knowing by name (Briggs et al. 1998; Sreedhar et al. 1999).

### 5.5 Tests
Unit tests for dominators and DF on hand-built CFGs (diamond, loop, nested loop). Then a big one: for **every example**, `lowered → SSA → verify (SSA rules) → destruct → verify → interpret` must match `-O0` exactly. Add the hidden flag `--passes=ssa,out-of-ssa` for this.

**Done when:** every example survives the SSA round trip with identical outputs, and the verifier's SSA checks pass.

---

## Milestone 6 — Optimizations and pass manager (2 days)

**Files:** `src/transform/pass_manager.{h,cpp}`, `constfold.cpp`, `copyprop.cpp`, `dce.cpp`, `simplifycfg.cpp`, `gvn.cpp`

```cpp
struct Pass { const char* name; bool (*run)(Function&); };    // returns true if it changed anything
class PassManager {
 public:
  void add(const Pass&);
  void run(Module&, bool verifyEach, bool printAfterAll, Stats*);
};
```
**`-O1` pipeline:** `remove-unreachable, ssa, { constfold, copyprop, gvn, dce, simplifycfg } until no change (max 4 rounds), licm, strength-reduce, constfold, dce, simplifycfg, out-of-ssa`. LICM and strength-reduce arrive in M8; leave them out until then.

| Pass | Algorithm | Watch out for |
|---|---|---|
| `constfold` | Any pure instruction whose operands are all immediates: compute, replace all uses with the immediate, delete. A phi whose incoming values are all the same (ignoring itself) → replace with that value. Algebraic: i32 `x+0, x-0, x*1, x/1 → x`, `x*0 → 0`, `x-x → 0`; f32 `x*1.0 → x`, `x-0.0 → x`. | i32 math in `uint32_t` (wraparound). **Never fold** div/rem by 0 or `INT_MIN/-1` (leave them for runtime). **Don't** fold f32 `x+0.0` (−0.0) or `x*0.0` (NaN, −0.0). Fold f32 with `float`, not `double`. |
| `copyprop` | For `%d = copy a` in SSA: replace all uses of `%d` with `a`; delete. | SSA only. In lowered IR a copy target has several definitions. |
| `dce` | Mark-and-sweep: roots are `store`, `barrier` and terminators; mark the definitions of their operands transitively; delete everything unmarked (including dead phis and phi cycles). | Simple "delete if unused" misses dead phi cycles; mark-and-sweep handles them. |
| `simplifycfg` | (1) `br true/false` → `jmp`, removing the phi entries on the dropped edge; (2) `br c, X, X` → `jmp X`; (3) delete unreachable blocks and their phi entries in successors; (4) merge `B` into its unique pred `P` when `P`'s only successor is `B` (B's phis are trivial: replace each with its single operand; in B's successors' phis rename B to P). | Always fix up successor phis. Call `cfg::rebuild` after each change. |
| `gvn` | Walk the dominator tree with a scoped hash table `key → register`; key = `(op, ty, operands, special reg/dim)`, with operands sorted for commutative ops (`add mul eq ne and or xor min max`). On a hit, replace uses and delete; pop the scope after the children. | **Never** number `load`s or phis. `special` is fine (same value for the whole thread). |

**Stats:** `--stats` prints a table: instruction count per kernel after each pass. Generate `results/opt_stats.md` with a script: for each example, `-O0` vs `-O1` IR and PTX instruction counts.

**Done when:** every example gives `-O0 == -O1` exactly in the interpreter; `--verify-each` passes; GPU results still match; `opt_stats.md` is generated.

---

## Milestone 7 — Shared memory and barriers (1 day)

1. Sema already accepts `shared` (E12). Lowering: `vars[sym] = Operand::shared(index)`; indexing becomes `load/store ... shared @name[...]`.
2. Interpreter: per-block shared buffers; barrier scheduling and divergent-barrier detection (`TESTING.md §3`).
3. Backend: module-scope `.shared` declaration, `mov.u32` of the array address in the prologue, `ld/st.shared` with 32-bit addresses, `bar.sync 0` (`PTX_BACKEND.md §8`; reference `tests/ptx_reference/shared_reverse.ptx`).
4. Optimizer: `barrier` is a root in DCE and is never moved; shared loads are never CSE'd or hoisted.
5. Write a test that **must** fail: a kernel calling `sync()` inside `if (thread_idx.x < 10)`. The interpreter must report "divergent barrier".

**Done when:** `block_sum` and `matmul_tiled` match on the interpreter (`-O0`/`-O1`) and on the GPU. `matmul_tiled` should equal `matmul_naive` **bitwise** for these inputs (the summation order is the same).

---

## Milestone 8 — LICM, strength reduction, benchmarks, README (1.5 days)

### 8.1 LICM (`src/analysis/loops.cpp`, `src/transform/licm.cpp`)
1. **Natural loops:** each edge `T → H` where `H` dominates `T` is a back edge. Loop body = `H` + every block that reaches `T` without passing through `H` (reverse DFS from `T`, stopping at `H`).
2. **Preheader:** use `H`'s unique predecessor outside the loop if that block has exactly one successor (`H`). Kite's lowering always produces one (the block that does `jmp for.cond`). If there isn't one, skip the loop and count it in `--stats`.
3. **Hoist** (repeat until nothing moves): a non-phi instruction in the loop is invariant if it is pure, **not** `div`/`rem` (they could trap if speculated), **not** a load, and every operand is an immediate or defined outside the loop. Move it to the end of the preheader, before its `jmp`.
4. Process inner loops before outer ones (sort by body size).

### 8.2 Strength reduction
`mul i32 x, 2^k` → `shl i32 x, k` (either operand order). **Don't** turn signed `div` by `2^k` into a shift: `-7 / 2 = -3`, but `-7 >> 1 = -4`. Knowing *why* matters: it is the classic strength-reduction trap.

### 8.3 Benchmarks (`gpu/bench.py`, on Colab)
For saxpy, relu, softmax_rows, matmul_naive and matmul_tiled at realistic sizes (e.g. `n = 2^24`; matmul 1024³):
- time with `cp.cuda.Event` (warm up 3 runs, take the median of 20)
- variants: Kite `-O0`, Kite `-O1`, the same kernel in CUDA C via `cp.RawKernel` (NVRTC, i.e. NVIDIA's own compiler), and the library version (`2.5*x+y` in CuPy; `cp.matmul`, which is cuBLAS)
- metrics: time, GB/s for memory-bound kernels, GFLOP/s for matmul; `ptxas -v` registers; SASS instruction count from `cuobjdump -sass | grep -c '^ */\*[0-9a-f]*\*/'`
- write `results/bench_T4.md` from the script

Expect: memory-bound kernels near NVRTC (bandwidth-limited); `matmul_tiled` clearly faster than `matmul_naive`; both far behind cuBLAS. **Report what you measure, including when -O1 doesn't change runtime** (see `PTX_BACKEND.md §11`).

### 8.4 README
Fill in `README.md`: what it is, the pipeline diagram, a 30-second quick start, the example kernel → IR → PTX, the results tables, the testing approach, and "what I'd do next". Add `docs/architecture.md` (one page; reuse the diagram from `PROJECT_SPEC.md §1`).

---

## Stretch goals (only after M8, ranked by value)

1. **Fuzzer** (`TESTING.md §6`): high value; finds real bugs; great story.
2. **SCCP** (sparse conditional constant propagation; Wegman–Zadeck): replaces `constfold` + branch folding and handles constants through phis and loops.
3. **Load CSE within a block** (invalidate on store or barrier).
4. **`mad.lo.s32` peephole** and `[reg+imm]` addressing.
5. **atomics:** `atomic_add(p, i, v)` → `atom.global.add.f32`.
6. **`-ffast-math`**: emit `fma.rn.f32` for `a*b+c`; show the precision difference.
7. **C++ runner on the CUDA Driver API** (`cuModuleLoadDataEx` with JIT error log, `cuLaunchKernel`).

---

## If time runs out

- **Short on time and at M3?** Finish M4 first. A backend that runs on a GPU beats a half-done optimizer.
- **Stuck in SSA (M5)?** Fallback: skip SSA and do *local* (per-block) constant folding + copy propagation, plus DCE driven by liveness on the lowered IR. Less impressive, but still real and honest.
