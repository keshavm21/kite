# Kite IR Reference

The IR is the center of the compiler. The front end produces it; the interpreter, the optimizer and the PTX backend all consume it. Get this right and everything else is easier.

## 1. Design decisions (and why)

| Decision | Why |
|---|---|
| **Control-flow graph of basic blocks.** Each block is a list of instructions ending in exactly one terminator. | The standard structure every analysis needs (dominators, liveness, loops). |
| **Three-address code on typed virtual registers** (`%7 = add i32 %3, %5`). Unlimited registers. | Matches PTX almost 1:1, so the backend is mostly a table lookup. No register allocation needed, because ptxas does it. |
| **The same IR in three forms:** *lowered* (local variables are registers assigned multiple times), *SSA* (each register defined exactly once, with `phi`), *post-SSA* (phis replaced by copies). | `-O0` emits PTX straight from lowered IR. `-O1` goes lowered → SSA → passes → post-SSA → PTX. PTX registers are mutable, so the backend accepts any non-SSA form. |
| **Memory ops take a base and an element index** (`load f32 global %x[%i]`), not raw byte addresses. | Keeps the interpreter trivially bounds-checkable and keeps address arithmetic out of the optimizer. The backend expands it to `mul.wide` + `add` + `ld`. |
| **Locals are registers, not stack slots.** | There are no pointers to locals, so nothing can alias a local. This lets SSA construction work directly on registers (no LLVM-style `alloca` / mem2reg needed). |

## 2. Types

`void`, `bool`, `i32`, `f32`, `ptr.f32`, `ptr.i32`. Pointer-typed registers exist only as kernel parameters.

## 3. Operands

| Operand | Printed as | Notes |
|---|---|---|
| Virtual register | `%7` or `%acc.7` | The number is the identity; `.acc` is only a name hint for readability. Printed as `%<hint>.<id>` if there is a hint, else `%<id>`. |
| i32 immediate | `42`, `-1` | |
| f32 immediate | `2.5`, `0.0` | Printed with `%.9g` so it round-trips exactly. If the result has no `.`, `e`, `inf` or `nan`, append `.0`. |
| bool immediate | `true`, `false` | |
| `undef` | `undef` | **Only in phis**, in SSA form: "no definition reaches along this edge". Out-of-SSA emits no copy for it. |
| Shared array | `@buf` | Only as the base of a shared load or store. |

## 4. Instructions

`T` is the operation type printed after the opcode. For comparisons `T` is the **operand** type and the result is `bool`.

### 4.1 Pure instructions (no side effects; safe to delete if unused, safe to CSE/hoist)

| Opcode | Form | Types | Semantics |
|---|---|---|---|
| `copy` | `%d = copy T a` | any scalar | `d = a` |
| `add` `sub` `mul` | `%d = add T a, b` | i32, f32 | i32 wraps; f32 round-to-nearest |
| `div` | `%d = div T a, b` | i32, f32 | i32 truncates. **i32 div is not hoistable** (it can trap in the interpreter). |
| `rem` | `%d = rem i32 a, b` | i32 | C semantics. Not hoistable. |
| `neg` | `%d = neg T a` | i32, f32 | |
| `shl` | `%d = shl i32 a, b` | i32 | Produced only by strength reduction. |
| `eq` `ne` `lt` `le` `gt` `ge` | `%d = lt T a, b` | i32, f32 (`eq`/`ne` also bool) | result `bool` |
| `and` `or` `xor` `not` | `%d = and bool a, b` | bool | Not from `&&`/`||` (those become branches); available to passes. |
| `itof` `ftoi` `btoi` | `%d = itof a` | | casts per LANGUAGE §4.5 |
| `sqrt` `exp` `abs` | `%d = sqrt f32 a` | | intrinsics |
| `min` `max` | `%d = min T a, b` | i32, f32 | |
| `special` | `%d = special i32 tid.x` | | reads `tid ntid ctaid nctaid` `.x/.y/.z` |
| `phi` | `%d = phi T [a, bb1], [b, bb2]` | any | SSA only; must be at the start of a block; one entry per predecessor |

### 4.2 Instructions with effects

| Opcode | Form | Semantics |
|---|---|---|
| `load` | `%d = load T global %p[%i]` / `%d = load T shared @s[%i]` | read element `i` |
| `store` | `store T global %p[%i], v` / `store T shared @s[%i], v` | write element `i` |
| `barrier` | `barrier` | `sync()` |

Loads are not "pure" for CSE purposes: two identical loads separated by a store may differ. Keep it simple: **never CSE or hoist loads.** (A good stretch: CSE loads within a block when no store or barrier lies between them.)

### 4.3 Terminators (exactly one, always last)

| Opcode | Form |
|---|---|
| `br` | `br %c, then.1, else.2` (`%c` is bool; may be an immediate after folding) |
| `jmp` | `jmp loop.3` |
| `ret` | `ret` |

## 5. Text format

```
kernel @NAME(T %p.0, T %q.1, ...) {
  shared f32 @buf[256]                 ; zero or more, before the first block
BLOCKNAME:                             ; first block is the entry block
  INSTRUCTION
  ...
  TERMINATOR
BLOCKNAME:
  ...
}
```

- Block names are `<hint>.<id>` (e.g. `for.cond.3`), except the entry block, which is named `entry`.
- `;` starts a comment.
- Print blocks in their storage order, and keep that order deterministic (golden tests depend on it).

You only need the **printer**. A parser for the text IR is optional (nice for writing pass unit tests, but you can build IR in C++ with the builder instead).

## 6. Invariants (the verifier checks these)

Run the verifier after lowering and, with `--verify-each`, after every pass. **It will save you days.** A pass that breaks an invariant fails loudly at the pass that broke it, not three passes later in the backend.

**Always:**
1. Every block is non-empty and ends with exactly one terminator; no terminator appears earlier.
2. Branch targets are blocks of this function. `preds`/`succs` match the terminators (recompute and compare).
3. The entry block has no predecessors.
4. Every instruction's operand types match its opcode (e.g. `add f32` has f32 operands; `br` has a bool condition).
5. Every register used has a type, and the destination type matches the result type.
6. Shared-array bases refer to declared arrays.

**Additionally in SSA form:**
7. Each register has exactly one definition (parameters count as defined in `entry`).
8. Each definition **dominates** each use. For a phi operand, the definition must dominate the end of the corresponding predecessor block.
9. Phis appear only at the start of a block, with exactly one entry per predecessor, and each predecessor appears exactly once.
10. `undef` appears only as a phi operand.

**Additionally in post-SSA form:** no phis.

## 7. Worked example 1: `saxpy` at `-O0`

Source: `examples/saxpy.kite`. Lowered IR:

```
kernel @saxpy(i32 %n.0, f32 %a.1, ptr.f32 %x.2, ptr.f32 %y.3) {
entry:
  %4 = special i32 ctaid.x
  %5 = special i32 ntid.x
  %6 = mul i32 %4, %5
  %7 = special i32 tid.x
  %8 = add i32 %6, %7
  %i.9 = copy i32 %8
  %10 = lt i32 %i.9, %n.0
  br %10, if.then.1, if.end.2
if.then.1:
  %11 = load f32 global %x.2[%i.9]
  %12 = mul f32 %a.1, %11
  %13 = load f32 global %y.3[%i.9]
  %14 = add f32 %12, %13
  store f32 global %y.3[%i.9], %14
  jmp if.end.2
if.end.2:
  ret
}
```

Notes:
- Parameters are registers `0..n-1`, in order.
- Every `var` gets one register (here `%i.9`), and its initializer becomes a `copy` into it. Each later assignment to the variable is another `copy` into the **same** register, which is why lowered IR is not SSA.
- At `-O1`, `copyprop` replaces uses of `%i.9` with `%8` and `dce` deletes the copy.

## 8. Worked example 2: `row_sum` through the pipeline

Register and block numbers below are illustrative; yours will differ, and that's fine.

### 8.1 Lowered (`--emit=ir -O0`)
```
kernel @row_sum(i32 %rows.0, i32 %cols.1, ptr.f32 %m.2, ptr.f32 %out.3) {
entry:
  %4 = special i32 ctaid.x
  %5 = special i32 ntid.x
  %6 = mul i32 %4, %5
  %7 = special i32 tid.x
  %8 = add i32 %6, %7
  %r.9 = copy i32 %8
  %10 = ge i32 %r.9, %rows.0
  br %10, if.then.1, if.end.2
if.then.1:
  ret
if.end.2:
  %acc.11 = copy f32 0.0
  %c.12 = copy i32 0
  jmp for.cond.3
for.cond.3:
  %13 = lt i32 %c.12, %cols.1
  br %13, for.body.4, for.end.6
for.body.4:
  %14 = mul i32 %r.9, %cols.1
  %15 = add i32 %14, %c.12
  %16 = load f32 global %m.2[%15]
  %17 = add f32 %acc.11, %16
  %acc.11 = copy f32 %17                ; second definition of %acc.11
  jmp for.step.5
for.step.5:
  %18 = add i32 %c.12, 1
  %c.12 = copy i32 %18                  ; second definition of %c.12
  jmp for.cond.3
for.end.6:
  store f32 global %out.3[%r.9], %acc.11
  ret
}
```

### 8.2 After SSA construction and `copyprop`
`%acc.11` and `%c.12` have two definitions each, so they need phis. Both are defined in `if.end.2` and in the loop, and `for.cond.3` is in the dominance frontier of the loop blocks, so the phis go at `for.cond.3`:
```
if.end.2:
  jmp for.cond.3
for.cond.3:
  %acc.19 = phi f32 [0.0, if.end.2], [%17, for.step.5]
  %c.20 = phi i32 [0, if.end.2], [%18, for.step.5]
  %13 = lt i32 %c.20, %cols.1
  br %13, for.body.4, for.end.6
for.body.4:
  %14 = mul i32 %8, %cols.1
  %15 = add i32 %14, %c.20
  %16 = load f32 global %m.2[%15]
  %17 = add f32 %acc.19, %16
  jmp for.step.5
for.step.5:
  %18 = add i32 %c.20, 1
  jmp for.cond.3
for.end.6:
  store f32 global %out.3[%8], %acc.19
  ret
```

### 8.3 After `simplifycfg` and `licm` (`-O1`)
`for.step.5` has one predecessor, and `for.body.4` has one successor, so the two blocks merge. `%14 = mul %8, %cols.1` uses only values defined outside the loop, so LICM hoists it to the preheader `if.end.2`:
```
if.end.2:
  %14 = mul i32 %8, %cols.1             ; hoisted
  jmp for.cond.3
for.cond.3:
  %acc.19 = phi f32 [0.0, if.end.2], [%17, for.body.4]
  %c.20 = phi i32 [0, if.end.2], [%18, for.body.4]
  %13 = lt i32 %c.20, %cols.1
  br %13, for.body.4, for.end.6
for.body.4:
  %15 = add i32 %14, %c.20
  %16 = load f32 global %m.2[%15]
  %17 = add f32 %acc.19, %16
  %18 = add i32 %c.20, 1
  jmp for.cond.3
```

### 8.4 After SSA destruction
Each phi becomes a copy at the end of each predecessor:
```
if.end.2:
  %14 = mul i32 %8, %cols.1
  %acc.19 = copy f32 0.0
  %c.20 = copy i32 0
  jmp for.cond.3
...
for.body.4:
  ...
  %18 = add i32 %c.20, 1
  %acc.19 = copy f32 %17
  %c.20 = copy i32 %18
  jmp for.cond.3
```
The PTX for this is `tests/ptx_reference/row_sum_O1.ptx`, which ptxas accepts.

## 9. C++ data structures (suggested)

```cpp
enum class Ty : uint8_t { Void, Bool, I32, F32, PtrF32, PtrI32 };

enum class Op : uint8_t {
  Copy, Add, Sub, Mul, Div, Rem, Neg, Shl,
  Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, Not,
  IToF, FToI, BToI, Sqrt, Exp, Abs, Min, Max,
  Special, Phi,
  Load, Store, Barrier,
  Br, Jmp, Ret,
};
enum class Space : uint8_t { Global, Shared };
enum class SReg : uint8_t { Tid, Ntid, Ctaid, Nctaid };   // + dimension 0/1/2

struct Operand {
  enum Kind : uint8_t { Reg, ImmI32, ImmF32, ImmBool, Undef, SharedArr } kind;
  int reg = -1;          // Reg
  int32_t i = 0;         // ImmI32 / ImmBool (0/1) / SharedArr index
  float f = 0.0f;        // ImmF32
  static Operand r(int id);  static Operand imm(int32_t);  static Operand imm(float);  // ...
};

struct BasicBlock;
struct Inst {
  Op op;
  Ty ty = Ty::Void;                 // operation type (operand type for compares)
  int dst = -1;                     // destination register, or -1
  std::vector<Operand> args;        // load: {base, idx}; store: {base, idx, val}; phi: incoming values
  std::vector<BasicBlock*> blocks;  // br: {then, else}; jmp: {target}; phi: incoming blocks (parallel to args)
  Space space = Space::Global;      // load/store
  SReg sreg; int dim = 0;           // special
  bool isTerminator() const;  bool isPure() const;  bool hasSideEffects() const;
  Ty resultType() const;
};

struct BasicBlock {
  int id; std::string hint;
  std::vector<Inst> insts;
  std::vector<BasicBlock*> preds, succs;    // derived; rebuild with cfg::rebuild(fn)
};

struct Function {
  std::string name;
  int numParams;
  std::vector<SharedArray> shared;                  // {name, elemTy, size}
  std::vector<std::unique_ptr<BasicBlock>> blocks;  // blocks[0] is entry
  std::vector<Ty> regTy;                            // type of every register
  std::vector<std::string> regHint;
  int newReg(Ty t, std::string hint = "");
  BasicBlock* newBlock(std::string hint);
};
struct Module { std::vector<Function> kernels; };
```

Practical tips:
- **Keep `preds`/`succs` derived.** Passes edit terminators; then call `cfg::rebuild(fn)`. Don't hand-maintain edges.
- **Deleting instructions:** mark them dead in a pass, then `erase_if` once at the end. Never erase while iterating.
- **Replacing all uses** (`replaceAllUses(fn, oldReg, newOperand)`): a linear scan over all instructions is fine at this size. You don't need use-lists.
- **Removing a block:** you must also remove the matching entries from phis in its successors. Forgetting this is the #1 SSA bug, and the verifier catches it (invariant 9).
