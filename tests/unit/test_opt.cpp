#include <sstream>

#include "analysis/cfg.h"
#include "doctest/doctest.h"
#include "ir/builder.h"
#include "test_util.h"
#include "transform/opt.h"
#include "transform/pass_manager.h"

using namespace kite;
using namespace kite::ir;
using namespace kite::test;

namespace {

bool has(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

int count(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
    return n;
}

// Compiles `src`, runs `names` (verifying after each), and returns the printed IR.
std::string after(const std::string& src, std::vector<std::string> names) {
    Module m = compile(src);
    Form form = Form::Lowered;
    std::string err;
    transform::PassManager pm({true, false, false});
    REQUIRE_MESSAGE(pm.run(m, transform::flatPipeline(names), form, err), err);
    return toString(m.kernels[0]);
}

// ssa + the five optimization passes once each (not out-of-ssa, so the IR stays readable).
std::string optimized(const std::string& src) {
    return after(src, {"ssa", "constfold", "copyprop", "gvn", "dce", "simplifycfg", "constfold", "copyprop", "dce"});
}

std::string o1(const std::string& src) {
    Module m = compile(src);
    Form form = Form::Lowered;
    std::string err;
    transform::PassManager pm({true, false, false});
    REQUIRE_MESSAGE(pm.run(m, transform::o1Pipeline(), form, err), err);
    CHECK(form == Form::PostSSA);
    return toString(m.kernels[0]);
}

}  // namespace

TEST_CASE("constfold: i32 wraps, and division by zero or INT_MIN / -1 is left alone") {
    std::string ir = optimized(R"(
kernel k(p: *i32) {
    p[0] = 2147483647 + 1;
    p[1] = 7 / 0;
    p[2] = -2147483648 / -1;
    p[3] = 7 % 0;
    p[4] = 7 / 2 + 7 % 4;
    p[5] = -7 / 2;
    p[6] = 2147483647 * 2;
}
)");
    CHECK(has(ir, "store i32 global %p.0[0], -2147483648"));
    CHECK(has(ir, "div i32 7, 0"));
    CHECK(has(ir, "div i32 -2147483648, -1"));
    CHECK(has(ir, "rem i32 7, 0"));
    CHECK(has(ir, "store i32 global %p.0[4], 6"));
    CHECK(has(ir, "store i32 global %p.0[5], -3"));
    CHECK(has(ir, "store i32 global %p.0[6], -2"));
}

TEST_CASE("constfold: i32 identities") {
    std::string ir = optimized(R"(
kernel k(n: i32, p: *i32) {
    p[0] = n + 0;
    p[1] = 0 + n;
    p[2] = n - 0;
    p[3] = n * 1;
    p[4] = 1 * n;
    p[5] = n / 1;
    p[6] = n * 0;
    p[7] = n - n;
}
)");
    for (int k = 0; k <= 5; ++k) CHECK(has(ir, "store i32 global %p.1[" + std::to_string(k) + "], %n.0"));
    CHECK(has(ir, "[6], 0"));
    CHECK(has(ir, "[7], 0"));
    CHECK_FALSE(has(ir, "add i32"));
    CHECK_FALSE(has(ir, "mul i32"));
}

TEST_CASE("constfold: f32 x*0.0 and x+0.0 are NOT folded, x*1.0 and x-0.0 are") {
    std::string ir = optimized(R"(
kernel k(x: f32, p: *f32) {
    p[0] = x * 0.0;
    p[1] = x + 0.0;
    p[2] = 0.0 + x;
    p[3] = x * 1.0;
    p[4] = x - 0.0;
    p[5] = x / 1.0;
    p[6] = x - x;
}
)");
    CHECK(has(ir, "mul f32 %x.0, 0.0"));  // NaN * 0 is NaN, -3 * 0 is -0
    CHECK(has(ir, "add f32 %x.0, 0.0"));  // -0.0 + 0.0 is +0.0
    CHECK(count(ir, "add f32") == 1);  // p[1] and p[2] are the same value, so gvn shares one add
    CHECK(has(ir, "store f32 global %p.1[3], %x.0"));
    CHECK(has(ir, "store f32 global %p.1[4], %x.0"));
    CHECK(has(ir, "store f32 global %p.1[5], %x.0"));
    CHECK(has(ir, "sub f32 %x.0, %x.0"));  // inf - inf is NaN
}

TEST_CASE("constfold: f32 arithmetic is single precision; NaN, exp and out-of-range casts are not folded") {
    std::string ir = optimized(R"(
kernel k(p: *f32, q: *i32) {
    var z: f32 = 0.0;
    p[0] = 0.1 * 3.0;
    p[1] = 16777216.0 + 1.0;
    p[2] = z / z;
    p[3] = exp(1.0);
    p[4] = 1.0 / z;
    q[0] = 3000000000.0 as i32;
    q[1] = -2.9 as i32;
    p[5] = -0.0 + 0.0;
}
)");
    CHECK(has(ir, "store f32 global %p.0[0], 0.300000012"));
    CHECK(has(ir, "store f32 global %p.0[1], 16777216.0"));
    CHECK(has(ir, "div f32 0.0, 0.0"));
    CHECK(has(ir, "exp f32 1.0"));
    CHECK(has(ir, "store f32 global %p.0[4], inf"));
    CHECK(has(ir, "ftoi 3e+09"));
    CHECK(has(ir, "store i32 global %q.1[1], -2"));
    CHECK(has(ir, "store f32 global %p.0[5], 0.0"));
}

TEST_CASE("constfold: comparisons, casts, intrinsics and bool logic") {
    std::string ir = optimized(R"(
kernel k(p: *i32, q: *f32) {
    p[0] = (3 < 5) as i32 + (2.5 >= 2.5) as i32 + (true != false) as i32;
    p[1] = min(3, -4) + max(3, -4) + abs(-9);
    q[0] = sqrt(16.0) + 2 as f32;
}
)");
    CHECK(has(ir, "store i32 global %p.0[0], 3"));
    CHECK(has(ir, "store i32 global %p.0[1], 8"));  // -4 + 3 + 9
    CHECK(has(ir, "store f32 global %q.1[0], 6.0"));
}

TEST_CASE("constfold: a phi whose inputs agree (ignoring itself) is replaced") {
    // a never changes in the loop: a = phi [5, entry], [a, ...] -> 5
    std::string ir = o1(R"(
kernel k(n: i32, p: *i32) {
    var a: i32 = 5;
    for (var i: i32 = 0; i < n; i += 1) { a = a; }
    p[0] = a;
}
)");
    CHECK(has(ir, "store i32 global %p.1[0], 5"));
}

TEST_CASE("copyprop and dce remove copies and dead code") {
    std::string ir = optimized(R"(
kernel k(n: i32, p: *i32) {
    var a: i32 = n;
    var b: i32 = a;
    var unused: i32 = b * 99;
    p[0] = b + 1;
}
)");
    CHECK_FALSE(has(ir, "copy"));
    CHECK_FALSE(has(ir, "99"));
    CHECK(has(ir, "add i32 %n.0, 1"));
}

TEST_CASE("dce removes a dead phi cycle but keeps stores and barriers") {
    Function fn;
    fn.name = "k";
    int out = fn.newReg(Ty::PtrI32, "out");
    int again = fn.newReg(Ty::Bool, "again");
    fn.numParams = 2;
    Builder b(fn);
    BasicBlock *e = fn.newBlock("entry"), *l = fn.newBlock("loop"), *done = fn.newBlock("done");
    b.setBlock(e); b.emitJmp(l);
    int x = fn.newReg(Ty::I32, "x"), x2 = fn.newReg(Ty::I32, "x2");
    b.setBlock(l);
    Inst phi;
    phi.op = Op::Phi; phi.ty = Ty::I32; phi.dst = x;
    phi.args = {Operand::i32(0), Operand::r(x2)};
    phi.blocks = {e, l};
    b.append(phi);
    Inst add;
    add.op = Op::Add; add.ty = Ty::I32; add.dst = x2; add.args = {Operand::r(x), Operand::i32(1)};
    b.append(add);
    b.emitBr(Operand::r(again), l, done);  // the exit does not depend on x
    b.setBlock(done);
    b.emitLoad(Ty::I32, Space::Global, Operand::r(out), Operand::i32(0));  // unused load
    b.emitBarrier();
    b.emitStore(Ty::I32, Space::Global, Operand::r(out), Operand::i32(1), Operand::i32(7));
    b.emitRet();
    cfg::rebuild(fn);
    REQUIRE(verify(fn, Form::SSA).empty());
    CHECK(transform::dce(fn));
    std::string ir = toString(fn);
    CHECK_FALSE(has(ir, "phi"));      // x and x2 only feed each other
    CHECK_FALSE(has(ir, "add i32"));
    CHECK(has(ir, "barrier"));
    CHECK(has(ir, "store i32"));
    CHECK_FALSE(has(ir, "load"));
    CHECK(verify(fn, Form::SSA).empty());
}

TEST_CASE("simplifycfg: constant branches, unreachable blocks, merging") {
    std::string ir = optimized(R"(
kernel k(n: i32, p: *i32) {
    if (true) { p[0] = 1; } else { p[0] = 2; }
    while (false) { p[1] = 3; }
    if (n < 1) { p[2] = 4; } else { p[2] = 4; }
}
)");
    CHECK_FALSE(has(ir, "store i32 global %p.1[0], 2"));
    CHECK_FALSE(has(ir, "[1], 3"));
    CHECK(has(ir, "store i32 global %p.1[0], 1"));
    CHECK_FALSE(has(ir, "while"));
    CHECK(count(ir, "br ") == 1);  // only `n < 1` still branches
}

TEST_CASE("simplifycfg: br c, X, X becomes jmp and phi entries follow the merged blocks") {
    // diamond whose arms produce the same value, then a loop: the phis must stay consistent through merges
    std::string src = R"(
kernel k(n: i32, p: *i32) {
    var s: i32 = 0;
    for (var i: i32 = 0; i < n; i += 1) {
        if (i % 2 == 0) { s += i; } else { s += i; }
    }
    p[0] = s;
}
)";
    std::string ir = optimized(src);  // --verify-each inside `after` checks every intermediate form
    CHECK(has(ir, "phi i32"));
}

TEST_CASE("simplifycfg: a block with a lone predecessor is merged into it, phis fixed") {
    Function fn = compileKernel("kernel k(n: i32, p: *i32) { var x: i32 = 1; { x = 2; } p[0] = x + n; }");
    size_t before = fn.blocks.size();
    CHECK_FALSE(transform::simplifycfg(fn));  // straight-line code: one block already
    CHECK(fn.blocks.size() == before);

    Function loop = compileKernel("kernel k(n: i32, p: *i32) { for (var i: i32 = 0; i < n; i += 1) { p[i] = i; } }");
    size_t blocks = loop.blocks.size();
    CHECK(transform::simplifycfg(loop));  // body + step merge
    CHECK(loop.blocks.size() == blocks - 1);
    CHECK(verify(loop).empty());
}

TEST_CASE("gvn: commutative operations share one value, loads never do") {
    std::string ir = optimized(R"(
kernel k(a: i32, b: i32, p: *i32, q: *i32) {
    var x: i32 = a + b;
    var y: i32 = b + a;
    p[0] = x * y;
    p[1] = q[0] + q[0];
    p[2] = a - b;
    p[3] = b - a;
}
)");
    CHECK(count(ir, "add i32 %a.0, %b.1") + count(ir, "add i32 %b.1, %a.0") == 1);
    CHECK(count(ir, "load i32") == 2);
    CHECK(count(ir, "sub i32") == 2);  // sub is not commutative
}

TEST_CASE("gvn: a value is shared only with dominated code") {
    std::string ir = optimized(R"(
kernel k(a: i32, b: i32, c: bool, p: *i32) {
    var t: i32 = a * b;
    if (c) { p[0] = a * b + t; } else { p[1] = a * b; }
    var u: i32 = a * 7;
    if (c) { p[2] = u; } else { p[3] = a * 7; }
}
)");
    CHECK(count(ir, "mul i32 %a.0, %b.1") == 1);  // t dominates both arms
    CHECK(count(ir, "mul i32 %a.0, 7") == 1);
}

TEST_CASE("gvn: siblings do not share, and special registers are numbered") {
    std::string ir = optimized(R"(
kernel k(a: i32, c: bool, p: *i32) {
    if (c) { p[0] = a * 3; } else { p[1] = a * 3; }
    p[2] = thread_idx.x + thread_idx.x;
}
)");
    CHECK(count(ir, "mul i32 %a.0, 3") == 2);  // neither arm dominates the other
    CHECK(count(ir, "special i32 tid.x") == 1);
}

TEST_CASE("-O1 pipeline: row_sum has no copies left except the loop-carried phi copies") {
    std::string ir = o1(R"(
kernel row_sum(rows: i32, cols: i32, m: *f32, out: *f32) {
    var r: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (r >= rows) { return; }
    var acc: f32 = 0.0;
    for (var c: i32 = 0; c < cols; c += 1) { acc += m[r * cols + c]; }
    out[r] = acc;
}
)");
    CHECK(has(ir, "if.end.2:\n  %14 = mul i32 %8, %cols.1"));  // hoisted by licm
    CHECK_FALSE(has(ir, "for.body.4:\n  %14"));
    CHECK_FALSE(has(ir, "for.step"));
    CHECK_FALSE(has(ir, "phi"));
    CHECK(count(ir, "copy") == 4);  // acc and c initialised in if.end.2 and updated in for.body.4
}

TEST_CASE("pass manager: stats, print-after-all and the fixpoint limit") {
    Module m = compile("kernel k(n: i32, p: *i32) { var a: i32 = n; p[0] = a + 0; }");
    Form form = Form::Lowered;
    std::string err;
    std::ostringstream log;
    transform::Options opts;
    opts.collectStats = true;
    opts.printAfterAll = true;
    transform::PassManager pm(opts, &log);
    REQUIRE(pm.run(m, transform::o1Pipeline(), form, err));
    std::ostringstream stats;
    pm.printStats(stats);
    CHECK(has(stats.str(), "constfold (round 1)"));
    CHECK(has(stats.str(), "out-of-ssa"));
    CHECK(has(stats.str(), "lowered"));
    CHECK(has(log.str(), "; --- after copyprop (round 1) ---"));
    CHECK(has(log.str(), "kernel @k("));
    CHECK_FALSE(has(stats.str(), "round 5"));
}

TEST_CASE("licm: r * cols leaves row_sum's loop, the load and the loop-carried adds stay") {
    std::string ir = after(R"(
kernel row_sum(rows: i32, cols: i32, m: *f32, out: *f32) {
    var r: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (r >= rows) { return; }
    var acc: f32 = 0.0;
    for (var c: i32 = 0; c < cols; c += 1) { acc += m[r * cols + c]; }
    out[r] = acc;
}
)", {"ssa", "copyprop", "simplifycfg", "licm"});
    CHECK(has(ir, "if.end.2:\n  %14 = mul i32 %8, %cols.1\n  jmp for.cond.3"));
    CHECK(has(ir, "%16 = load f32 global %m.2[%15]"));
    CHECK(has(ir, "%15 = add i32 %14, %c."));
}

TEST_CASE("licm: div, rem, ftoi and loads are never hoisted, nested invariants climb out of both loops") {
    std::string ir = after(R"(
kernel k(n: i32, d: i32, x: f32, p: *i32, q: *f32) {
    for (var i: i32 = 0; i < n; i += 1) {
        for (var j: i32 = 0; j < n; j += 1) {
            p[i * n + j] = (n / d) + (n % d) + (x as i32) + p[0] + (n * 3 + d);
        }
    }
}
)", {"ssa", "copyprop", "licm"});
    // Find the innermost loop body by locating its store; everything speculatable above must be outside it.
    size_t store = ir.find("store i32");
    REQUIRE(store != std::string::npos);
    std::string head = ir.substr(0, store);
    size_t body = head.rfind("for.body");
    std::string inner = head.substr(body);
    CHECK(has(inner, "div i32"));
    CHECK(has(inner, "rem i32"));
    CHECK(has(inner, "ftoi"));
    CHECK(has(inner, "load i32"));
    CHECK_FALSE(has(inner, "mul i32 %n.0, 3"));
}

TEST_CASE("licm: a loop whose header is the entry block has no preheader and is left alone") {
    // Lowering always emits a preheader, so build the shape by hand: the entry block is its own loop.
    Function g;
    g.name = "g";
    int n = g.newReg(Ty::I32, "n");
    g.numParams = 1;
    BasicBlock* e = g.newBlock("entry");
    int t = g.newReg(Ty::I32);
    Inst mul;
    mul.op = Op::Mul; mul.ty = Ty::I32; mul.dst = t;
    mul.args = {Operand::r(n), Operand::i32(3)};
    e->insts.push_back(mul);
    Inst jmp;
    jmp.op = Op::Jmp; jmp.blocks = {e};
    e->insts.push_back(jmp);
    CHECK_FALSE(transform::licm(g));
    CHECK(g.blocks[0]->insts.size() == 2);
}

TEST_CASE("strength: mul by a power of two becomes shl, in either operand order; other constants stay") {
    std::string ir = after(R"(
kernel k(a: i32, p: *i32) {
    p[0] = a * 8;
    p[1] = 4 * a;
    p[2] = a * 6;
    p[3] = a * 1;
    p[4] = a * 0 - 6 * a;
}
)", {"ssa", "strength"});
    CHECK(has(ir, "shl i32 %a.0, 3"));
    CHECK(has(ir, "shl i32 %a.0, 2"));
    CHECK(has(ir, "mul i32 %a.0, 6"));
    CHECK(count(ir, "shl") == 2);
}

TEST_CASE("strength: signed div by a power of two is NOT a shift (-7 / 2 is -3, -7 >> 1 is -4)") {
    std::string src = R"(
kernel k(a: i32, p: *i32) {
    p[0] = a / 2;
    p[1] = a % 4;
}
)";
    std::string ir = after(src, {"ssa", "strength"});
    CHECK(has(ir, "div i32 %a.0, 2"));
    CHECK(has(ir, "rem i32 %a.0, 4"));
    CHECK_FALSE(has(ir, "shl"));
    // and the end-to-end value: -7 / 2 == -3 at -O1
    Module m = compile("kernel k(p: *i32) { var a: i32 = 0 - 7; p[0] = a / 2; p[1] = a * 2; }");
    Form form = Form::Lowered;
    std::string err;
    transform::PassManager pm({true, false, false});
    REQUIRE_MESSAGE(pm.run(m, transform::o1Pipeline(), form, err), err);
    CHECK(has(toString(m.kernels[0]), "-3"));
    CHECK(has(toString(m.kernels[0]), "-14"));
}
