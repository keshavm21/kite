# Kite Language Reference

Kite is a small, statically typed language for writing GPU kernels. It looks like a cut-down CUDA C with Rust-style declarations. A file contains one or more `kernel`s. Each kernel is launched over a grid of thread blocks, and every thread runs the kernel body.

```kite
kernel saxpy(n: i32, a: f32, x: *f32, y: *f32) {
    var i: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

Design rule: **anything that makes the compiler harder without teaching something new is left out.**

---

## 1. Lexical structure

| Token | Rule | Examples |
|---|---|---|
| Identifier | `[A-Za-z_][A-Za-z0-9_]*`, not a keyword | `x`, `row_sum`, `_tmp1` |
| Integer literal | `[0-9]+`, value must fit in `i32` (≤ 2147483647) | `0`, `256` |
| Float literal | `[0-9]+ "." [0-9]+ ([eE] [+-]? [0-9]+)?` | `0.0`, `2.5`, `1.0e-3` |
| Comment | `// ...` to end of line, or `/* ... */` (not nested) | |
| Whitespace | space, tab, `\r`, `\n` (ignored, but tracked for line:col) | |

A float literal **must** have digits on both sides of the dot (`1.0`, not `1.` or `.5`). That keeps the lexer simple and avoids ambiguity with `.x`.

**Keywords:** `kernel var shared if else while for break continue return sync true false as i32 f32 bool`

**Builtin names** (reserved; not usable as identifiers): `thread_idx block_idx block_dim grid_dim sqrt exp min max abs`

**Operators and punctuation:**
```
+  -  *  /  %  =  +=  -=  *=  /=  ==  !=  <  <=  >  >=  &&  ||  !
(  )  {  }  [  ]  ,  ;  :  .
```
Lexing uses maximal munch: `<=` is one token, `< =` is two.

## 2. Types

| Type | Meaning | Where allowed |
|---|---|---|
| `i32` | 32-bit signed integer, two's-complement wraparound | everywhere |
| `f32` | IEEE-754 single precision | everywhere |
| `bool` | `true` / `false` | locals, params, conditions |
| `*f32`, `*i32` | pointer to a global-memory array | **kernel parameters only** |
| `f32[N]`, `i32[N]` | shared-memory array of N elements | **`shared` declarations only** |

There are **no implicit conversions**. `1 + 2.0` is a type error; write `1 as f32 + 2.0`.

## 3. Grammar (EBNF)

`{ x }` means zero or more, `[ x ]` means optional. Braces around bodies are **mandatory**, so there is no dangling-else problem in this grammar by construction.

```ebnf
program      = { kernel } EOF ;
kernel       = "kernel" IDENT "(" [ param { "," param } ] ")" block ;
param        = IDENT ":" ( scalar_type | ptr_type ) ;
scalar_type  = "i32" | "f32" | "bool" ;
ptr_type     = "*" ( "i32" | "f32" ) ;

block        = "{" { stmt } "}" ;
stmt         = var_decl ";"
             | shared_decl
             | assign ";"
             | if_stmt
             | "while" "(" expr ")" block
             | "for" "(" for_init ";" expr ";" assign ")" block
             | "break" ";"
             | "continue" ";"
             | "return" ";"
             | "sync" "(" ")" ";"
             | block ;

var_decl     = "var" IDENT ":" scalar_type "=" expr ;
shared_decl  = "shared" IDENT ":" ( "i32" | "f32" ) "[" INT "]" ";" ;
assign       = lvalue ( "=" | "+=" | "-=" | "*=" | "/=" ) expr ;
lvalue       = IDENT [ "[" expr "]" ] ;
for_init     = var_decl | assign ;
if_stmt      = "if" "(" expr ")" block [ "else" ( block | if_stmt ) ] ;

expr         = or_expr ;
or_expr      = and_expr { "||" and_expr } ;
and_expr     = eq_expr  { "&&" eq_expr } ;
eq_expr      = rel_expr { ( "==" | "!=" ) rel_expr } ;
rel_expr     = add_expr { ( "<" | "<=" | ">" | ">=" ) add_expr } ;
add_expr     = mul_expr { ( "+" | "-" ) mul_expr } ;
mul_expr     = cast_expr { ( "*" | "/" | "%" ) cast_expr } ;
cast_expr    = unary { "as" scalar_type } ;
unary        = ( "-" | "!" ) unary | primary ;
primary      = INT | FLOAT | "true" | "false"
             | IDENT "[" expr "]"                      (* index *)
             | INTRINSIC "(" [ expr { "," expr } ] ")"  (* sqrt exp min max abs *)
             | BUILTIN "." ( "x" | "y" | "z" )         (* thread_idx.x ... *)
             | IDENT
             | "(" expr ")" ;
```

Every `{ ... }` loop in the grammar becomes a `while` loop in a recursive-descent parser. Each rule becomes one function: `parseOrExpr()`, `parseAndExpr()`, and so on.

### Precedence (low → high)

| Level | Operators | Assoc |
|---|---|---|
| 1 | `\|\|` | left |
| 2 | `&&` | left |
| 3 | `==` `!=` | left |
| 4 | `<` `<=` `>` `>=` | left |
| 5 | `+` `-` | left |
| 6 | `*` `/` `%` | left |
| 7 | `as` | left (postfix) |
| 8 | unary `-` `!` | right |
| 9 | indexing, intrinsic call, `.x` | — |

So `-x as f32` is `(-x) as f32`, and `t * 2 + flag as i32` is `(t * 2) + (flag as i32)`.

## 4. Semantics

### 4.1 Execution model
A launch has `grid_dim.x * grid_dim.y * grid_dim.z` blocks, each with `block_dim.x * block_dim.y * block_dim.z` threads. Every thread runs the kernel body with its own locals. Threads in different blocks can't communicate. Threads in the same block can communicate through `shared` arrays, separated by `sync()`.

### 4.2 Builtins (all `i32`, read-only)
| Kite | CUDA | PTX special register |
|---|---|---|
| `thread_idx.{x,y,z}` | `threadIdx` | `%tid.{x,y,z}` |
| `block_idx.{x,y,z}` | `blockIdx` | `%ctaid.{x,y,z}` |
| `block_dim.{x,y,z}` | `blockDim` | `%ntid.{x,y,z}` |
| `grid_dim.{x,y,z}` | `gridDim` | `%nctaid.{x,y,z}` |

### 4.3 Variables and scope
- `var x: T = e;` declares a **mutable** local. The initializer is required, so a variable can never be read before it is written. This guarantee simplifies SSA construction.
- Scope is lexical: a variable is visible from its declaration to the end of its enclosing block. A `for` init variable is scoped to the loop.
- Redeclaring a name **in the same scope** is an error. Shadowing a name from an outer scope is allowed.
- Parameters are in the kernel's outermost scope and are **read-only**.

### 4.4 Expressions
| Operator | Operand types | Result | Notes |
|---|---|---|---|
| `+ - *` | `i32,i32` / `f32,f32` | same | i32 wraps around |
| `/` | `i32,i32` / `f32,f32` | same | i32 truncates toward zero (like C) |
| `%` | `i32,i32` | `i32` | sign follows the dividend (like C) |
| unary `-` | `i32` / `f32` | same | |
| `< <= > >=` | `i32,i32` / `f32,f32` | `bool` | |
| `== !=` | same type, any scalar | `bool` | |
| `&& \|\|` | `bool,bool` | `bool` | **short-circuit**: the right side runs only if needed |
| `!` | `bool` | `bool` | |
| `e as T` | see §4.5 | `T` | |
| `p[i]` | pointer param or shared array; `i: i32` | element type | |

Expressions have **no side effects** (no calls with effects, no `++`). That makes `p[i] += e` → `p[i] = p[i] + e` a safe desugaring, and lets the optimizer move pure expressions freely.

### 4.5 Casts
| From → To | Meaning | PTX |
|---|---|---|
| `i32 → f32` | round to nearest | `cvt.rn.f32.s32` |
| `f32 → i32` | truncate toward zero (like C) | `cvt.rzi.s32.f32` |
| `bool → i32` | `true` → 1, `false` → 0 | `selp.s32` |
| `T → T` | no-op | — |
| anything else | **error** (`i32 → bool`: write `x != 0`) | |

### 4.6 Intrinsics
| Call | Types | Result | Notes |
|---|---|---|---|
| `sqrt(x)` | `f32` | `f32` | correctly rounded |
| `exp(x)` | `f32` | `f32` | **approximate** on GPU (`ex2.approx`); compare with tolerance |
| `abs(x)` | `i32` or `f32` | same | |
| `min(a,b)`, `max(a,b)` | both `i32` or both `f32` | same | f32: if one is NaN, returns the other (like `fminf`) |

### 4.7 Statements
- **Assignment:** `x = e`, `p[i] = e`, `s[i] = e`. The target and the value must have the same type. `+= -= *= /=` desugar in the parser to `x = x op e`.
- **if / else / else if:** the condition must be `bool`.
- **while (cond) { }**, **for (init; cond; step) { }**: `continue` in a `for` jumps to the **step**, not the condition.
- **break / continue:** only inside a loop; they apply to the innermost loop.
- **return;** ends the thread. Kernels return nothing.
- **sync();** is a block-wide barrier (`__syncthreads()`, PTX `bar.sync 0`).

### 4.8 Shared arrays
```kite
shared tile: f32[256];
```
- Allowed only as a **direct child of the kernel body**, not inside `if` or loops.
- One instance per block, uninitialized, shared by all threads in the block.
- Total shared memory per kernel ≤ 48 KiB (49152 bytes).
- Indexed like pointers: `tile[ty * 16 + tx]`.

## 5. Undefined behaviour

The compiler may assume these never happen. The **interpreter detects and reports** each one, which makes it a great debugging tool.

1. Out-of-bounds pointer or shared-array index.
2. Integer division or remainder by zero, and `-2147483648 / -1`.
3. `f32 → i32` of NaN, infinity, or a value outside the i32 range.
4. `sync()` reached by only some threads of a block (divergent barrier).
5. A data race: two threads of the same block access the same shared element without a `sync()` between them, at least one access being a write. (The interpreter does not need to detect this one.)
6. A kernel that runs forever (the interpreter enforces a step limit).

Integer overflow is **not** UB: it wraps. Constant folding must wrap too: fold with `uint32_t` arithmetic, then cast.

## 6. Example programs

See `examples/`. In rough order of difficulty: `vector_add`, `saxpy`, `relu`, `casts`, `row_sum`, `collatz_steps`, `softmax_rows`, `matmul_naive`, then `block_sum` and `matmul_tiled` (these two need shared memory).

## 7. What is deliberately missing

Functions other than kernels, structs, unsigned and 64-bit types, bitwise operators in the source (the *IR* has `shl` for strength reduction), `++`/`--`, a ternary operator, `do-while`, `switch`, string literals, atomics, a preprocessor. If you have time at the end, add atomics (§ Stretch in PROJECT_SPEC).

## 8. Static errors (what `sema` must check)

Each error is printed as `FILE:LINE:COL: error: MESSAGE` and the compiler exits with code 1. The parser may stop at its first error. `sema` should report **all** errors it finds.

| # | Error | Message (suggested) |
|---|---|---|
| E1 | Use of an undeclared name | `use of undeclared identifier 'foo'` |
| E2 | Redeclaration in the same scope (including params, shared arrays, kernel names) | `redefinition of 'x'` |
| E3 | Operand types don't match the operator | `invalid operands to '+': i32 and f32` |
| E4 | Init or assignment type mismatch | `cannot assign f32 to i32` |
| E5 | Non-bool condition in `if`/`while`/`for` | `condition must be bool, found i32` |
| E6 | Indexing something that isn't a pointer or shared array, or a non-i32 index | `cannot index 'n' of type i32` / `index must be i32` |
| E7 | Assignment to a parameter or builtin | `cannot assign to parameter 'n'` |
| E8 | `break`/`continue` outside a loop | `'break' outside of a loop` |
| E9 | Invalid cast | `cannot cast i32 to bool` |
| E10 | Wrong intrinsic arity or argument types | `'min' expects 2 arguments of the same numeric type` |
| E11 | Pointer or shared array used as a value (`x + 1`, `var p: ... = x`) | `'x' is an array; index it` |
| E12 | `shared` not at the top level of a kernel, size ≤ 0, or total > 48 KiB | `shared arrays must be declared at kernel top level` |
| E13 | Integer literal out of range | `integer literal too large for i32` |
| E14 | `%` on f32 | `'%' requires i32 operands` |

Write one `tests/errors/*.kite` file per row.
