# Kite architecture (one page)

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

| Stage | Where | Notes |
|---|---|---|
| Front end | `src/frontend` | Hand-written lexer and recursive-descent parser; sema resolves names and checks types. Errors carry `file:line:col`. |
| IR | `src/ir` | Three-address instructions in basic blocks. Locals are virtual registers (no memory, no alloca). Memory ops index by element (`base[idx]`). The verifier checks the invariants in `docs/IR.md` section 6. |
| Analyses | `src/analysis` | CFG (preds/succs, reverse postorder, unreachable removal), dominator tree and dominance frontiers (Cooper-Harvey-Kennedy), natural loops with preheaders. |
| SSA | `src/transform/ssa_*` | Phis at iterated dominance frontiers, renaming along the dominator tree. Out-of-SSA turns phis into copies in the predecessors. |
| Optimizer | `src/transform`, `pass_manager.*` | `constfold`, `copyprop`, `gvn`, `dce`, `simplifycfg`, `unroll`, `licm`, `strength`. The pass manager tracks the IR form (lowered, SSA, post-SSA), verifies after each pass on request, and prints per-pass statistics. |
| Backend | `src/codegen/ptx` | Accepts any IR form. `mul.rn`/`add.rn` (no fma), float immediates as hex bits, `mul.wide` + `add` address math, `.reg` declarations sized from the highest id per class. No register allocation: ptxas does it. |
| Interpreter | `src/interp` | Simulates the thread grid, shared memory and barriers; reports UB (out of bounds, divide by zero, divergent barrier, bad `ftoi`). It is the correctness oracle. |

## The IR forms

The same data structure is used in three forms: *lowered* (a local may be assigned many times), *SSA* (every register has one definition; phis at joins) and *post-SSA* (phis replaced by copies). `-O0` goes lowered to PTX. `-O1` goes lowered, SSA, optimized, post-SSA, PTX.

## -O1

`remove-unreachable, ssa, {constfold copyprop gvn dce simplifycfg}` until stable (at most 4 rounds), then `{unroll constfold copyprop gvn dce simplifycfg}` (until stable, at most 4 rounds), then `licm, gvn, strength, constfold, dce, simplifycfg, out-of-ssa`.

LICM hoists only what cannot trap or read memory, so it is safe even when the loop runs zero times. The extra `gvn` after it merges the identical computations LICM collects in a preheader.

## How correctness is checked

1. The IR verifier (after lowering, and after every pass with `--verify-each`).
2. Differential tests: `kitec run` at `-O0`, `-O1`, `--passes=ssa` and `--passes=ssa,out-of-ssa` must print identical output, for every example and test program.
3. Golden IR and PTX files; ptxas assembles every PTX file.
4. GPU runs on a T4 compared with the interpreter (`gpu/check_all.sh`).

Details: `docs/TESTING.md`.
