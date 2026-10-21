#include "analysis/cfg.h"
#include "doctest/doctest.h"
#include "ir/builder.h"
#include "test_util.h"

using namespace kite;
using namespace kite::ir;

namespace {

// kernel k(i32 %n.0, f32 %x.1, ptr.f32 %p.2) { entry: ...; } with the builder positioned in `entry`.
struct Fixture {
    Function fn;
    Builder b{fn};
    Operand n, x, p;
    Fixture() {
        fn.name = "k";
        n = Operand::r(fn.newReg(Ty::I32, "n"));
        x = Operand::r(fn.newReg(Ty::F32, "x"));
        p = Operand::r(fn.newReg(Ty::PtrF32, "p"));
        fn.numParams = 3;
        b.setBlock(fn.newBlock("entry"));
    }
    std::vector<std::string> check(Form form = Form::Lowered) {
        cfg::rebuild(fn);
        return verify(fn, form);
    }
};

bool mentions(const std::vector<std::string>& problems, const std::string& needle) {
    for (const std::string& s : problems) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("verifier accepts a well-formed function") {
    Fixture f;
    Operand sum = f.b.emitBinary(Op::Add, Ty::I32, f.n, Operand::i32(1));
    Operand v = f.b.emitLoad(Ty::F32, Space::Global, f.p, sum);
    f.b.emitStore(Ty::F32, Space::Global, f.p, f.n, f.b.emitBinary(Op::Mul, Ty::F32, v, f.x));
    f.b.emitRet();
    CHECK(f.check().empty());
}

TEST_CASE("verifier: message format names kernel, block and instruction") {
    Fixture f;
    f.b.emitBinary(Op::Add, Ty::I32, f.n, f.x);
    f.b.emitRet();
    auto problems = f.check();
    REQUIRE(problems.size() == 1);
    CHECK(problems[0] == "verifier: kernel k, block entry, inst 0: operand 1 (%x.1) has type f32, expected i32");
}

TEST_CASE("verifier: block structure") {
    SUBCASE("missing terminator") {
        Fixture f;
        f.b.emitCopy(Ty::I32, f.n);
        CHECK(mentions(f.check(), "does not end with a terminator"));
    }
    SUBCASE("empty block") {
        Fixture f;
        f.b.emitRet();
        f.fn.newBlock("orphan");
        CHECK(mentions(f.check(), "is empty"));
    }
    SUBCASE("terminator in the middle") {
        Fixture f;
        f.b.emitRet();
        Inst extra;
        extra.op = Op::Copy;
        extra.ty = Ty::I32;
        extra.dst = f.fn.newReg(Ty::I32);
        extra.args = {Operand::i32(1)};
        f.fn.blocks[0]->insts.push_back(extra);  // an instruction after the ret
        CHECK(mentions(f.check(), "terminator before the end of the block"));
    }
    SUBCASE("branch to a block of another function") {
        Fixture f;
        Function other;
        BasicBlock* foreign = other.newBlock("entry");
        f.b.emitJmp(foreign);
        CHECK(mentions(f.check(), "branch target is not a block of this function"));
    }
    SUBCASE("stale preds/succs") {
        Fixture f;
        BasicBlock* next = f.fn.newBlock("next");
        f.b.emitJmp(next);
        f.b.setBlock(next);
        f.b.emitRet();
        // no cfg::rebuild
        CHECK(mentions(verify(f.fn), "succs do not match"));
        CHECK(mentions(verify(f.fn), "preds do not match"));
    }
    SUBCASE("the entry block must have no predecessors") {
        Fixture f;
        f.b.emitJmp(f.fn.blocks[0].get());
        CHECK(mentions(f.check(), "entry block has a predecessor"));
    }
}

TEST_CASE("verifier: operand and result types") {
    SUBCASE("br needs a bool") {
        Fixture f;
        BasicBlock* t = f.fn.newBlock("t");
        f.b.emitBr(f.n, t, t);
        f.b.setBlock(t);
        f.b.emitRet();
        CHECK(mentions(f.check(), "has type i32, expected bool"));
    }
    SUBCASE("compare yields bool, not the operand type") {
        Fixture f;
        Inst i;
        i.op = Op::Lt;
        i.ty = Ty::I32;
        i.dst = f.fn.newReg(Ty::I32);
        i.args = {f.n, Operand::i32(3)};
        f.b.append(i);
        f.b.emitRet();
        CHECK(mentions(f.check(), "destination"));
    }
    SUBCASE("rem is i32 only") {
        Fixture f;
        f.b.emitBinary(Op::Rem, Ty::F32, f.x, f.x);
        f.b.emitRet();
        CHECK(mentions(f.check(), "rem does not support type f32"));
    }
    SUBCASE("copy to a register of the wrong type") {
        Fixture f;
        f.b.emitCopyTo(f.x.reg, Ty::I32, Operand::i32(1));
        f.b.emitRet();
        CHECK(mentions(f.check(), "destination %x.1 has type f32"));
    }
    SUBCASE("pointer registers are not arithmetic operands") {
        Fixture f;
        f.b.emitBinary(Op::Add, Ty::I32, f.p, Operand::i32(1));
        f.b.emitRet();
        CHECK(mentions(f.check(), "has type ptr.f32, expected i32"));
    }
    SUBCASE("load through a pointer of the wrong element type") {
        Fixture f;
        f.b.emitLoad(Ty::I32, Space::Global, f.p, Operand::i32(0));
        f.b.emitRet();
        CHECK(mentions(f.check(), "has type ptr.f32, expected ptr.i32"));
    }
    SUBCASE("index must be i32") {
        Fixture f;
        f.b.emitLoad(Ty::F32, Space::Global, f.p, f.x);
        f.b.emitRet();
        CHECK(mentions(f.check(), "operand 1"));
    }
    SUBCASE("store value type") {
        Fixture f;
        f.b.emitStore(Ty::F32, Space::Global, f.p, Operand::i32(0), Operand::i32(1));
        f.b.emitRet();
        CHECK(mentions(f.check(), "operand 2"));
    }
    SUBCASE("nonexistent register") {
        Fixture f;
        f.b.emitCopy(Ty::I32, Operand::r(99));
        f.b.emitRet();
        CHECK(mentions(f.check(), "nonexistent register 99"));
    }
    SUBCASE("undef outside a phi") {
        Fixture f;
        f.b.emitCopy(Ty::I32, Operand::undef());
        f.b.emitRet();
        CHECK(mentions(f.check(), "undef is only allowed as a phi operand"));
    }
}

TEST_CASE("verifier: shared arrays") {
    SUBCASE("undeclared array") {
        Fixture f;
        f.b.emitLoad(Ty::F32, Space::Shared, Operand::shared(0), Operand::i32(0));
        f.b.emitRet();
        CHECK(mentions(f.check(), "declared shared array"));
    }
    SUBCASE("element type mismatch") {
        Fixture f;
        f.fn.shared.push_back({"buf", Ty::I32, 4});
        f.b.emitLoad(Ty::F32, Space::Shared, Operand::shared(0), Operand::i32(0));
        f.b.emitRet();
        CHECK(mentions(f.check(), "holds i32, accessed as f32"));
    }
    SUBCASE("a declared array is fine") {
        Fixture f;
        f.fn.shared.push_back({"buf", Ty::F32, 4});
        f.b.emitStore(Ty::F32, Space::Shared, Operand::shared(0), Operand::i32(0), f.x);
        f.b.emitRet();
        CHECK(f.check().empty());
    }
}

TEST_CASE("verifier: phis only exist in SSA form") {
    Fixture f;
    BasicBlock* loop = f.fn.newBlock("loop");
    f.b.emitJmp(loop);
    f.b.setBlock(loop);
    f.b.emitPhi(Ty::I32, {{Operand::i32(0), f.fn.blocks[0].get()}, {f.n, loop}});
    f.b.emitJmp(loop);
    CHECK(mentions(f.check(Form::Lowered), "phi outside SSA form"));
    CHECK(mentions(f.check(Form::PostSSA), "phi outside SSA form"));
    CHECK_FALSE(mentions(f.check(Form::SSA), "phi outside SSA form"));
}

TEST_CASE("tricky control-flow shapes lower to IR that verifies") {
    for (const char* src : {
             "kernel k(n: i32, p: *i32) { var s: i32 = 0; while (s < n) { s += 1; if (s == 5) { break; } } p[0] = s; }",
             "kernel k(n: i32, p: *i32) { for (var i: i32 = 0; i < n && i < 4 || n == 9; i += 1) { p[i] = i; } }",
             "kernel k(p: *i32) { while (true) { return; } }",
         }) {
        ir::Module m = test::compile(src);  // compile() fails the test if the verifier complains
        CHECK(m.kernels.size() == 1);
    }
}
