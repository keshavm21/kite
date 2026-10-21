# PTX Backend Guide

Everything you need to turn Kite IR into PTX that `ptxas` accepts and the GPU runs. Every PTX snippet in this document comes from files in `tests/ptx_reference/` or `gpu/hello.ptx`, and **all of them were assembled successfully with `ptxas` 12.9 for `sm_75`.**

Official reference (bookmark it): *PTX ISA* — https://docs.nvidia.com/cuda/parallel-thread-execution/

## 1. What PTX is

- A **virtual** ISA. NVIDIA publishes it; the real machine code (SASS) changes with every GPU generation and is not documented.
- **Unlimited typed virtual registers.** `ptxas` (or the driver's JIT) does register allocation, scheduling and many optimizations, then emits SASS.
- **Explicit state spaces:** `.param` (kernel arguments), `.global` (device memory), `.shared` (per-block on-chip memory), `.reg` (registers), and others.
- **Forward compatible:** PTX written for `sm_75` is JIT-compiled by the driver for any newer GPU. That is why we target `sm_75` (Colab's T4) and it also runs on A100/H100/L4.

So Kite's job ends at valid, sensible PTX. Register allocation and instruction scheduling belong to ptxas. Knowing where that line sits is exactly what the PTX compiler team works on.

## 2. Module structure

```ptx
.version 7.0                 // PTX ISA version (CUDA 11.0+; any modern driver accepts it)
.target sm_75                // minimum GPU architecture
.address_size 64

.shared .align 4 .f32 block_reverse_buf[256];   // shared arrays at module scope

.visible .entry saxpy(       // one .entry per kernel
    .param .u32 saxpy_param_0,
    .param .f32 saxpy_param_1,
    .param .u64 saxpy_param_2,
    .param .u64 saxpy_param_3
)
{
    .reg .pred  %p<11>;      // declares %p0 .. %p10
    .reg .b32   %r<10>;      // declares %r0 .. %r9
    .reg .f32   %f<15>;
    .reg .b64   %rd<21>;

    ... instructions and labels ...
}
```

## 3. Naming scheme (simple and deterministic)

| IR type | PTX reg class | Declaration | Name for IR register `%17` |
|---|---|---|---|
| `bool` | predicate | `.reg .pred %p<N>` | `%p17` |
| `i32` | 32-bit | `.reg .b32 %r<N>` | `%r17` |
| `f32` | float | `.reg .f32 %f<N>` | `%f17` |
| `ptr.*` / address temps | 64-bit | `.reg .b64 %rd<N>` | `%rd17` |

- **Register name = class prefix + IR register id.** Every register has one type, so names never clash.
- The backend needs extra temporaries for addresses. Take them from a counter that starts at `fn.regTy.size()`.
- Declare each class as `<maxIdUsedInThatClass + 1>`. Unused declared registers cost nothing; ptxas ignores them.
- **Labels:** `$L_` + block name with `.` replaced by `_` (`for.cond.3` → `$L_for_cond_3`). PTX identifiers may not contain `.`. ptxas reports `Unknown field '.cond'` if you forget.
- **Params:** `<kernel>_param_<i>`. **Shared arrays:** `<kernel>_<name>`.

## 4. Prologue

For each parameter `i` (IR register `i`):

| Param type | PTX |
|---|---|
| `i32` | `.param .u32` → `ld.param.u32 %ri, [K_param_i];` |
| `f32` | `.param .f32` → `ld.param.f32 %fi, [K_param_i];` |
| `bool` | `.param .u32` (host passes 0/1) → `ld.param.u32 %rT, [...]; setp.ne.s32 %pi, %rT, 0;` |
| `ptr.*` | `.param .u64` → `ld.param.u64 %rdi, [...]; cvta.to.global.u64 %rdi, %rdi;` |

`cvta.to.global` converts the generic pointer the host passes into a global-space address, so later accesses can use `ld.global` / `st.global`. nvcc does the same thing.

For each shared array, emit `mov.u32 %rS, K_name;` once, at the start of the entry block. That puts its 32-bit shared-window address in a register. Shared addresses are **32-bit** (`.b32`), not 64-bit.

## 5. Lowering table

`d` = destination register, `a`, `b` = operands (register or immediate; **ptxas accepts an immediate in any source operand position**, which was tested).

| IR | PTX |
|---|---|
| `special i32 tid.x` | `mov.u32 d, %tid.x;` (`%ntid`, `%ctaid`, `%nctaid`; `.x/.y/.z`) |
| `copy i32/f32/bool` | `mov.u32` / `mov.f32` / `mov.pred` |
| `add/sub i32` | `add.s32` / `sub.s32` |
| `mul i32` | `mul.lo.s32` (low 32 bits of the product) |
| `div i32`, `rem i32` | `div.s32`, `rem.s32` |
| `neg i32`, `shl i32` | `neg.s32`, `shl.b32` |
| `add/sub/mul f32` | `add.rn.f32` / `sub.rn.f32` / `mul.rn.f32` ← **see §7, the `.rn` matters** |
| `div f32` | `div.rn.f32` (IEEE-correct division) |
| `neg/abs f32` | `neg.f32` / `abs.f32` |
| `min/max/abs i32` | `min.s32` / `max.s32` / `abs.s32` |
| `min/max f32` | `min.f32` / `max.f32` |
| `sqrt f32` | `sqrt.rn.f32` |
| `exp f32` | `mul.rn.f32 t, a, 0f3FB8AA3B;` then `ex2.approx.f32 d, t;` (e^x = 2^(x·log₂e)) |
| `eq/ne/lt/le/gt/ge i32` | `setp.eq.s32` … `setp.ge.s32` |
| `eq/lt/le/gt/ge f32` | `setp.eq.f32` … (ordered: false if either input is NaN) |
| `ne f32` | **`setp.neu.f32`** (unordered: true if either is NaN, matching C's `!=`) |
| `eq bool` / `ne bool` | `xor.pred t, a, b; not.pred d, t;` / `xor.pred d, a, b;` |
| `and/or/xor/not bool` | `and.pred` / `or.pred` / `xor.pred` / `not.pred` |
| `itof` | `cvt.rn.f32.s32` |
| `ftoi` | `cvt.rzi.s32.f32` (round toward zero, like C) |
| `btoi` | `selp.s32 d, 1, 0, a;` |
| `load T global %p[i]` | `mul.wide.s32 t1, i, 4; add.s64 t2, %rdP, t1; ld.global.f32 d, [t2];` (`ld.global.u32` for i32) |
| `store T global %p[i], v` | same address, then `st.global.f32 [t2], v;` |
| `load T shared @s[i]` | `shl.b32 t1, i, 2; add.s32 t2, %rS, t1; ld.shared.f32 d, [t2];` |
| `store T shared @s[i], v` | same, then `st.shared.f32 [t2], v;` |
| `barrier` | `bar.sync 0;` |
| `ret` | `ret;` |
| `phi` | **must not exist**; the backend asserts on it |

`mul.wide.s32` multiplies two 32-bit values into a 64-bit result: it sign-extends the index and scales it by `sizeof(float)` in one instruction.

Constant index (optional peephole): `ld.global.f32 d, [%rdP+12];` for `%p[3]`. ptxas accepts `[reg+imm]`.

## 6. Control flow

1. **Block order:** emit blocks in storage order (or reverse postorder; either works if consistent).
2. **Fall-through:** if a jump's target is the next block emitted, emit nothing.
3. **`br c, T, F`**, with `N` = next block:

| Case | Emit |
|---|---|
| `c` is an immediate (`true`/`false`) | `bra.uni $L_taken;` (or nothing if taken == N) |
| `F == N` | `@%pc bra $L_T;` |
| `T == N` | `@!%pc bra $L_F;` |
| neither | `@%pc bra $L_T;` then `bra.uni $L_F;` |

4. **`jmp T`**: `bra.uni $L_T;` unless `T == N`.

`bra.uni` tells ptxas the branch is non-divergent (all threads take it). Use it only for unconditional jumps. The guarded `@%p bra` may diverge, and the hardware handles that. Interviewers like asking what divergence costs: both paths execute serially for the warp.

## 7. Floating point: the `.rn` rule (an interview-worthy detail)

PTX's plain `add.f32` / `mul.f32` (with no rounding modifier) **may be contracted by ptxas into a fused multiply-add** (`fma`), which rounds once instead of twice. That changes results in the last bit. Kite's interpreter computes `a*b` then `+c` with two roundings, so the GPU and the interpreter would disagree.

Emitting **`mul.rn.f32` and `add.rn.f32`** (explicit round-to-nearest) forbids contraction, per the PTX ISA. With that, `+ - * /` and `sqrt` are **bit-exact** between the interpreter and the GPU. Only `exp` (approximate) needs a tolerance. A future `-ffast-math` flag could deliberately emit `fma.rn.f32` for speed. That's a nice stretch, and a good thing to bring up in interviews.

**Float immediates** are written as hex bit patterns, `0f` + 8 hex digits:
```cpp
uint32_t bits; std::memcpy(&bits, &f, 4);
snprintf(buf, sizeof buf, "0f%08X", bits);      // 1.0f -> 0f3F800000, -0.5f -> 0fBF000000
```
This is exact, unlike printing decimal.

## 8. Shared memory and barriers

From `tests/ptx_reference/shared_reverse.ptx`:
```ptx
.shared .align 4 .f32 block_reverse_buf[256];
...
    mov.u32             %r8, block_reverse_buf; // 32-bit shared-window address
    shl.b32             %r9, %r2, 2;            // t * 4
    add.s32             %r10, %r8, %r9;
    st.shared.f32       [%r10], %f1;
    bar.sync            0;
```
`ptxas -v` reports `1024 bytes smem, used 1 barriers` for it.

## 9. Complete examples (expected output shapes)

- `gpu/hello.ptx`: hand-optimized SAXPY. **Run this first (Milestone 0).**
- `tests/ptx_reference/saxpy_O0.ptx`: what `kitec -O0 examples/saxpy.kite` should look like. Note the redundant `mov.u32 %r9, %r8` and the three identical `mul.wide.s32` address computations. That is the waste `-O1` (copyprop, GVN) removes.
- `tests/ptx_reference/row_sum_O1.ptx`: the loop after SSA, LICM and out-of-SSA. Note the phi copies at the end of the loop body and the hoisted `mul.lo.s32 %r14`.
- `tests/ptx_reference/opcode_table.ptx`: one instance of every instruction in §5. If you change the table, add the instruction here and re-run ptxas.

Your output doesn't need to match these byte for byte. Golden tests compare against **your** compiler's output once you've checked it by eye and with ptxas.

## 10. Validating and inspecting

On Colab (or any machine with the CUDA toolkit):
```bash
ptxas -arch=sm_75 -v out.ptx -o out.cubin   # validate; prints registers / smem / spills
cuobjdump -sass out.cubin                    # see the real machine code ptxas produced
```
On Linux without a GPU (e.g. CI), `pip install nvidia-cuda-nvcc-cu12` provides `ptxas`, so CI can validate every generated `.ptx` file without a GPU.

**ptxas errors you will see, and what they mean** (all reproduced while writing this guide):

| Error | Cause |
|---|---|
| `Unknown field '.done'` / `Label expected for argument 0 of instruction 'bra'` | A label contains `.`; sanitize names. |
| `Unexpected instruction types specified for 'mul'` | Wrong type suffix, e.g. `mul.rn.s32` (`.rn` is for floats). |
| `Arguments mismatch for instruction 'mad.lo'` | A register is used but not declared (e.g. `%r4` with `.reg .b32 %r<4>`), or a register of the wrong class. |

## 11. Honest notes for interviews

- ptxas is an optimizing compiler itself. For simple kernels, `-O1` vs `-O0` may produce **similar SASS**, because ptxas cleans up the redundancy anyway. Measure and report both: PTX instruction counts (what Kite controls) and SASS instruction counts and registers (`cuobjdump -sass`, `ptxas -v`). Saying "my optimizations cut PTX size by X%, but ptxas already removed most of it in SASS, so runtime didn't change for memory-bound kernels" shows real understanding. It is a strong answer, not a weak one.
- Memory-bound kernels (saxpy, relu) are limited by DRAM bandwidth. Report achieved GB/s against the GPU's peak. Compute-bound ones (matmul) show the effect of shared-memory tiling.
