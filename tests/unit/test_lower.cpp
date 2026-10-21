#include "doctest/doctest.h"
#include "test_util.h"

using namespace kite;
using namespace kite::test;

namespace {

bool contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

}  // namespace

TEST_CASE("lowering saxpy matches docs/IR.md section 7 exactly") {
    ir::Function fn = compileKernel(R"(
kernel saxpy(n: i32, a: f32, x: *f32, y: *f32) {
    var i: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
)");
    CHECK(ir::toString(fn) == R"(kernel @saxpy(i32 %n.0, f32 %a.1, ptr.f32 %x.2, ptr.f32 %y.3) {
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
)");
}

TEST_CASE("lowering row_sum matches docs/IR.md section 8.1 exactly") {
    ir::Function fn = compileKernel(R"(
kernel row_sum(rows: i32, cols: i32, m: *f32, out: *f32) {
    var r: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (r >= rows) {
        return;
    }
    var acc: f32 = 0.0;
    for (var c: i32 = 0; c < cols; c += 1) {
        acc += m[r * cols + c];
    }
    out[r] = acc;
}
)");
    CHECK(ir::toString(fn) == R"(kernel @row_sum(i32 %rows.0, i32 %cols.1, ptr.f32 %m.2, ptr.f32 %out.3) {
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
  %acc.11 = copy f32 %17
  jmp for.step.5
for.step.5:
  %18 = add i32 %c.12, 1
  %c.12 = copy i32 %18
  jmp for.cond.3
for.end.6:
  store f32 global %out.3[%r.9], %acc.11
  ret
}
)");
}

TEST_CASE("&& and || in a condition become branches, in program order") {
    std::string ir = ir::toString(compileKernel(R"(
kernel k(a: i32, b: i32, p: *i32) {
    if (a < 1 && b < 2 || a == b) {
        p[0] = 1;
    }
}
)"));
    CHECK_FALSE(contains(ir, " and "));
    CHECK_FALSE(contains(ir, " or "));
    // (a < 1 && b < 2) || a == b: and.rhs sits between entry and the or.rhs block, before the if body.
    size_t andRhs = ir.find("and.rhs.");
    size_t orRhs = ir.find("or.rhs.");
    size_t body = ir.find("if.then.");
    REQUIRE(andRhs != std::string::npos);
    REQUIRE(orRhs != std::string::npos);
    CHECK(andRhs < orRhs);
    CHECK(orRhs < body);
}

TEST_CASE("! in a condition swaps the targets instead of emitting not") {
    std::string ir = ir::toString(compileKernel("kernel k(a: i32, p: *i32) { if (!(a < 1)) { p[0] = 1; } }"));
    CHECK_FALSE(contains(ir, " not "));
    CHECK(contains(ir, "br %2, if.end.2, if.then.1"));
}

TEST_CASE("&& as a value materialises through branches") {
    std::string ir = ir::toString(compileKernel(R"(
kernel k(a: i32, p: *i32) {
    var f: bool = a < 1 && a > -5;
    p[0] = f as i32;
}
)"));
    CHECK(contains(ir, "sc.true"));
    CHECK(contains(ir, "copy bool true"));
    CHECK(contains(ir, "copy bool false"));
    CHECK(contains(ir, "btoi"));
}

TEST_CASE("! as a value emits not") {
    CHECK(contains(ir::toString(compileKernel("kernel k(a: i32, p: *i32) { var f: bool = !(a < 1); p[0] = f as i32; }")),
                   "not bool"));
}

TEST_CASE("code after a terminator goes into a dead block") {
    ir::Function fn = compileKernel(R"(
kernel k(p: *i32) {
    return;
    p[0] = 1;
}
)");
    REQUIRE(fn.blocks.size() == 2);
    CHECK(fn.blocks[1]->name() == "dead.1");
    CHECK(fn.blocks[1]->preds.empty());
    CHECK(fn.blocks[1]->terminator().op == ir::Op::Ret);
}

TEST_CASE("no dead block is created when nothing follows a terminator") {
    ir::Function fn = compileKernel("kernel k(a: i32, p: *i32) { if (a < 1) { return; } p[0] = 1; }");
    for (const auto& bb : fn.blocks) CHECK(bb->hint != "dead");
}

TEST_CASE("continue in a for loop jumps to the step block, break to the end") {
    std::string ir = ir::toString(compileKernel(R"(
kernel k(n: i32, p: *i32) {
    for (var i: i32 = 0; i < n; i += 1) {
        if (i == 3) { continue; }
        if (i == 7) { break; }
        p[i] = i;
    }
}
)"));
    CHECK(contains(ir, "jmp for.step.3"));
    CHECK(contains(ir, "jmp for.end.4"));
}

TEST_CASE("continue in a while loop jumps to the condition") {
    std::string ir = ir::toString(compileKernel(R"(
kernel k(n: i32, p: *i32) {
    var i: i32 = 0;
    while (i < n) {
        i += 1;
        if (i == 3) { continue; }
        p[i] = i;
    }
}
)"));
    CHECK(contains(ir, "jmp while.cond.1"));
}

TEST_CASE("shared arrays lower to @arrays and shared loads and stores") {
    std::string ir = ir::toString(compileKernel(R"(
kernel k(p: *f32) {
    shared buf: f32[8];
    buf[thread_idx.x] = 1.5;
    sync();
    p[0] = buf[0];
}
)"));
    CHECK(contains(ir, "  shared f32 @buf[8]\n"));
    CHECK(contains(ir, "store f32 shared @buf["));
    CHECK(contains(ir, "barrier"));
    CHECK(contains(ir, "load f32 shared @buf[0]"));
}

TEST_CASE("casts, intrinsics and shadowed variables lower to distinct registers") {
    std::string ir = ir::toString(compileKernel(R"(
kernel k(x: f32, p: *i32, q: *f32) {
    var v: i32 = x as i32;
    { var v: f32 = sqrt(abs(x)) + min(v, 3) as f32; q[0] = v; }
    p[0] = v;
}
)"));
    CHECK(contains(ir, "ftoi %x.0"));
    CHECK(contains(ir, "sqrt f32"));
    CHECK(contains(ir, "abs f32"));
    CHECK(contains(ir, "min i32"));
    CHECK(contains(ir, "itof"));
    CHECK(contains(ir, "%v.4 = copy i32"));
    CHECK(contains(ir, "%v.10 = copy f32"));
}

TEST_CASE("f32 immediates print so they round-trip") {
    std::string ir = ir::toString(compileKernel("kernel k(p: *f32) { p[0] = 0.1; p[1] = 3.0; p[2] = 1.0e10; }"));
    CHECK(contains(ir, "0.100000001"));
    CHECK(contains(ir, ", 3.0\n"));
    CHECK(contains(ir, "1e+10"));
}

TEST_CASE("the minimum i32 literal lowers to one immediate") {
    CHECK(contains(ir::toString(compileKernel("kernel k(p: *i32) { p[0] = -2147483648; }")), ", -2147483648\n"));
}
