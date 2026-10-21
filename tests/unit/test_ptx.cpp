#include <stdexcept>

#include "analysis/cfg.h"
#include "codegen/ptx.h"
#include "doctest/doctest.h"
#include "ir/builder.h"
#include "test_util.h"

using namespace kite;
using namespace kite::test;

namespace {

std::string ptxFor(const std::string& src, const std::string& target = "sm_75") {
    return codegen::PtxEmitter(compile(src), target).emit();
}

bool has(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

}  // namespace

TEST_CASE("ptx: header, signature and register declarations (saxpy)") {
    std::string p = ptxFor(R"(
kernel saxpy(n: i32, a: f32, x: *f32, y: *f32) {
    var i: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (i < n) { y[i] = a * x[i] + y[i]; }
}
)");
    CHECK(p.rfind(".version 7.0\n.target sm_75\n.address_size 64\n", 0) == 0);
    CHECK(has(p, ".visible .entry saxpy(\n    .param .u32 saxpy_param_0,\n    .param .f32 saxpy_param_1,\n"
                 "    .param .u64 saxpy_param_2,\n    .param .u64 saxpy_param_3\n)"));
    // Same register classes and counts as tests/ptx_reference/saxpy_O0.ptx.
    CHECK(has(p, ".reg .pred  %p<11>;"));
    CHECK(has(p, ".reg .b32   %r<10>;"));
    CHECK(has(p, ".reg .f32   %f<15>;"));
    CHECK(has(p, ".reg .b64   %rd<21>;"));
    CHECK(has(p, "ld.param.u64        %rd2, [saxpy_param_2];\n    cvta.to.global.u64  %rd2, %rd2;"));
    CHECK(has(p, "mul.rn.f32          %f12, %f1, %f11;"));
    CHECK(has(p, "add.rn.f32          %f14, %f12, %f13;"));
    CHECK(has(p, "mul.wide.s32        %rd15, %r9, 4;"));
    CHECK(has(p, "@!%p10 bra          $L_if_end_2;"));  // false target is not next
    CHECK_FALSE(has(p, "jmp"));
}

TEST_CASE("ptx: target flag and labels never contain dots") {
    std::string p = ptxFor("kernel k(n: i32, p: *i32) { for (var i: i32 = 0; i < n; i += 1) { p[i] = i; } }", "sm_80");
    CHECK(has(p, ".target sm_80"));
    CHECK(has(p, "$L_for_cond_1:"));
    CHECK(has(p, "bra.uni             $L_for_cond_1;"));
}

TEST_CASE("ptx: float immediates are exact hex bits, f32 ops never contract") {
    std::string p = ptxFor("kernel k(x: f32, p: *f32) { p[0] = x * 1.0 + -0.5; p[1] = exp(x); }");
    CHECK(has(p, "0f3F800000"));
    CHECK(has(p, "0f3F000000"));  // 0.5; `-0.5` is neg(0.5) at -O0
    CHECK(has(p, "neg.f32"));
    CHECK(has(p, "mul.rn.f32"));
    CHECK(has(p, "add.rn.f32"));
    CHECK(has(p, "0f3FB8AA3B"));
    CHECK(has(p, "ex2.approx.f32"));
    CHECK_FALSE(has(p, "fma"));
    CHECK_FALSE(has(p, "add.f32"));
    CHECK_FALSE(has(p, "mul.f32"));
}

TEST_CASE("ptx: constant indices become [reg+offset]") {
    std::string p = ptxFor("kernel k(p: *i32, q: *f32) { p[0] = 1; p[3] = 2; q[1] = 1.5; }");
    CHECK(has(p, "st.global.u32       [%rd0], %r"));
    CHECK(has(p, "[%rd0+12]"));
    CHECK(has(p, "[%rd1+4]"));
    CHECK_FALSE(has(p, "mul.wide"));
}

TEST_CASE("ptx: shared arrays, barriers") {
    std::string p = ptxFor("kernel k(p: *f32) { shared buf: f32[64]; buf[thread_idx.x] = 1.0; sync(); p[0] = buf[2]; }");
    CHECK(has(p, ".shared .align 4 .f32 k_buf[64];"));
    CHECK(has(p, "mov.u32             %r"));
    CHECK(has(p, ", k_buf;"));
    CHECK(has(p, "shl.b32"));
    CHECK(has(p, "st.shared.f32"));
    CHECK(has(p, "bar.sync            0;"));
    CHECK(has(p, "ld.shared.f32"));
}

TEST_CASE("ptx: comparisons") {
    std::string p = ptxFor(R"(
kernel k(x: f32, n: i32, p: *i32) {
    p[0] = (x != x) as i32 + (x == x) as i32 + (x < 1.0) as i32 + (n >= 3) as i32;
    var a: bool = n < 1;
    var b: bool = n > 5;
    p[1] = (a == b) as i32 + (a != b) as i32;
}
)");
    CHECK(has(p, "setp.neu.f32"));  // C's != is true for NaN
    CHECK(has(p, "setp.eq.f32"));
    CHECK(has(p, "setp.lt.f32"));
    CHECK(has(p, "setp.ge.s32"));
    CHECK(has(p, "selp.s32"));
    CHECK(has(p, "not.pred"));  // bool ==
    CHECK(has(p, "xor.pred"));
}

TEST_CASE("ptx: branch forms (fall-through, constant conditions)") {
    using namespace ir;
    Function fn;
    fn.name = "k";
    int c = fn.newReg(Ty::Bool, "c");
    fn.numParams = 1;
    Builder b(fn);
    BasicBlock *e = fn.newBlock("entry"), *t = fn.newBlock("t"), *f = fn.newBlock("f"), *z = fn.newBlock("z"),
               *w = fn.newBlock("w");
    b.setBlock(e); b.emitBr(Operand::r(c), t, f);           // true target is next
    b.setBlock(t); b.emitBr(Operand::r(c), z, w);           // neither target is next (f is)
    b.setBlock(f); b.emitBr(Operand::boolean(true), z, t);  // constant: taken target z is next
    b.setBlock(z); b.emitRet();
    b.setBlock(w); b.emitRet();
    cfg::rebuild(fn);
    Module m;
    m.kernels.push_back(std::move(fn));
    std::string p = codegen::PtxEmitter(m).emit();
    CHECK(has(p, "setp.ne.s32"));  // bool param: the host passes u32 0/1
    CHECK(has(p, "$L_entry:\n    @!%p0 bra "));
    CHECK(has(p, "$L_f_2;\n\n$L_t_1:"));
    CHECK(has(p, "$L_t_1:\n    @%p0 bra "));
    CHECK(has(p, "$L_z_3;\n    bra.uni             $L_w_4;\n"));
    CHECK(has(p, "$L_f_2:\n\n$L_z_3:"));  // constant branch to the next block emits nothing
}

TEST_CASE("ptx: immediates that PTX may reject are moved into registers first") {
    std::string p = ptxFor("kernel k(p: *f32, q: *i32) { p[0] = 5 as f32; q[0] = 2 * 3; p[1] = sqrt(4.0); }");
    CHECK(has(p, "mov.u32"));
    CHECK(has(p, "cvt.rn.f32.s32"));
    CHECK_FALSE(has(p, "cvt.rn.f32.s32      %f1, 5"));  // cvt source must be a register
    CHECK_FALSE(has(p, "mul.lo.s32          %r1, 2, 3"));
    CHECK_FALSE(has(p, "sqrt.rn.f32         %f2, 0f"));
}

TEST_CASE("ptx: phi is rejected") {
    using namespace ir;
    Function fn;
    fn.name = "k";
    Builder b(fn);
    BasicBlock* e = fn.newBlock("entry");
    b.setBlock(e);
    b.emitPhi(Ty::I32, {{Operand::i32(1), e}});
    b.emitRet();
    Module m;
    m.kernels.push_back(std::move(fn));
    CHECK_THROWS_AS(codegen::PtxEmitter(m).emit(), std::logic_error);
}
