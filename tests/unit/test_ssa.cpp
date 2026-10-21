#include <map>

#include "analysis/cfg.h"
#include "analysis/dominators.h"
#include "doctest/doctest.h"
#include "interp/interp.h"
#include "ir/builder.h"
#include "test_util.h"
#include "transform/pass_manager.h"
#include "transform/ssa.h"

using namespace kite;
using namespace kite::ir;
using namespace kite::test;

namespace {

// A CFG skeleton: blocks named by letter, `edges` maps a block to its successors (0, 1 or 2 of them).
struct Graph {
    Function fn;
    std::map<char, BasicBlock*> b;
    Graph(const std::string& names, const std::map<char, std::string>& edges) {
        fn.name = "g";
        for (char c : names) b[c] = fn.newBlock(std::string(1, c));  // the first becomes the entry
        Builder bld(fn);
        for (char c : names) {
            bld.setBlock(b[c]);
            const std::string& to = edges.count(c) ? edges.at(c) : "";
            if (to.empty()) bld.emitRet();
            else if (to.size() == 1) bld.emitJmp(b[to[0]]);
            else bld.emitBr(Operand::boolean(true), b[to[0]], b[to[1]]);
        }
        cfg::rebuild(fn);
    }
    std::string idom(analysis::DomTree& dt, char c) {
        BasicBlock* i = dt.idom(b[c]);
        return i ? i->hint : "-";
    }
    std::string df(const analysis::DomFrontier& d, char c) {
        std::string s;
        for (BasicBlock* x : d.at(b[c])) s += x->hint;
        return s;
    }
};

}  // namespace

TEST_CASE("dominators and frontiers: diamond") {
    Graph g("ABCD", {{'A', "BC"}, {'B', "D"}, {'C', "D"}});
    analysis::DomTree dt(g.fn);
    CHECK(g.idom(dt, 'A') == "-");
    CHECK(g.idom(dt, 'B') == "A");
    CHECK(g.idom(dt, 'C') == "A");
    CHECK(g.idom(dt, 'D') == "A");
    CHECK(dt.dominates(g.b['A'], g.b['D']));
    CHECK_FALSE(dt.dominates(g.b['B'], g.b['D']));
    CHECK(dt.dominates(g.b['B'], g.b['B']));
    auto df = analysis::dominanceFrontiers(g.fn, dt);
    CHECK(g.df(df, 'A') == "");
    CHECK(g.df(df, 'B') == "D");
    CHECK(g.df(df, 'C') == "D");
    CHECK(g.df(df, 'D') == "");
    CHECK(dt.children(g.b['A']).size() == 3);
}

TEST_CASE("dominators and frontiers: loop") {
    // A -> B; B -> C | D; C -> B; D exit
    Graph g("ABCD", {{'A', "B"}, {'B', "CD"}, {'C', "B"}});
    analysis::DomTree dt(g.fn);
    CHECK(g.idom(dt, 'B') == "A");
    CHECK(g.idom(dt, 'C') == "B");
    CHECK(g.idom(dt, 'D') == "B");
    auto df = analysis::dominanceFrontiers(g.fn, dt);
    CHECK(g.df(df, 'B') == "B");  // a loop header is in its own frontier
    CHECK(g.df(df, 'C') == "B");
    CHECK(g.df(df, 'A') == "");
    CHECK(g.df(df, 'D') == "");
}

TEST_CASE("dominators and frontiers: nested loops") {
    // A -> B; B -> C | G; C -> D | F; D -> E; E -> D | F... built as:
    // outer header B, inner header D, inner latch E, outer latch F, exit G
    Graph g("ABCDEFG", {{'A', "B"}, {'B', "CG"}, {'C', "D"}, {'D', "E"}, {'E', "DF"}, {'F', "B"}});
    analysis::DomTree dt(g.fn);
    CHECK(g.idom(dt, 'C') == "B");
    CHECK(g.idom(dt, 'D') == "C");
    CHECK(g.idom(dt, 'E') == "D");
    CHECK(g.idom(dt, 'F') == "E");
    CHECK(g.idom(dt, 'G') == "B");
    auto df = analysis::dominanceFrontiers(g.fn, dt);
    CHECK(g.df(df, 'E') == "BD");  // E ends the inner loop (D) and, via F, the outer loop (B)
    CHECK(g.df(df, 'D') == "BD");
    CHECK(g.df(df, 'F') == "B");
    CHECK(g.df(df, 'B') == "B");
}

TEST_CASE("dominators ignore unreachable blocks") {
    // A -> C; B (unreachable) -> C; C exit
    Graph g("ABC", {{'A', "C"}, {'B', "C"}});
    analysis::DomTree dt(g.fn);
    CHECK_FALSE(dt.reachable(g.b['B']));
    CHECK(g.idom(dt, 'C') == "A");  // the unreachable predecessor does not count
    CHECK(g.idom(dt, 'B') == "-");
    CHECK_FALSE(dt.dominates(g.b['B'], g.b['C']));
    CHECK_FALSE(dt.dominates(g.b['A'], g.b['B']));
    auto df = analysis::dominanceFrontiers(g.fn, dt);
    CHECK(df.count(g.b['B']) == 0);
}

TEST_CASE("removeUnreachable deletes blocks and their phi entries") {
    Function fn;
    fn.name = "k";
    Builder b(fn);
    BasicBlock *e = fn.newBlock("entry"), *dead = fn.newBlock("dead"), *join = fn.newBlock("join");
    b.setBlock(e); b.emitJmp(join);
    b.setBlock(dead); b.emitJmp(join);
    b.setBlock(join);
    b.emitPhi(Ty::I32, {{Operand::i32(1), e}, {Operand::i32(2), dead}});
    b.emitRet();
    cfg::rebuild(fn);
    CHECK(cfg::removeUnreachable(fn));
    CHECK(fn.blocks.size() == 2);
    CHECK(fn.blocks[1]->insts[0].blocks.size() == 1);
    CHECK(verify(fn, Form::SSA).empty());
    CHECK_FALSE(cfg::removeUnreachable(fn));
}

TEST_CASE("SSA construction on row_sum: phis for acc and c at the loop header only") {
    Function fn = compileKernel(R"(
kernel row_sum(rows: i32, cols: i32, m: *f32, out: *f32) {
    var r: i32 = block_idx.x * block_dim.x + thread_idx.x;
    if (r >= rows) { return; }
    var acc: f32 = 0.0;
    for (var c: i32 = 0; c < cols; c += 1) { acc += m[r * cols + c]; }
    out[r] = acc;
}
)");
    transform::constructSSA(fn);
    CHECK(verify(fn, Form::SSA).empty());
    int phis = 0;
    for (const auto& bb : fn.blocks) {
        for (const Inst& in : bb->insts) {
            if (in.op != Op::Phi) continue;
            ++phis;
            CHECK(bb->name() == "for.cond.3");
            REQUIRE(in.blocks.size() == 2);
            std::set<std::string> from = {in.blocks[0]->name(), in.blocks[1]->name()};
            CHECK(from == std::set<std::string>{"if.end.2", "for.step.5"});
        }
    }
    CHECK(phis == 2);
    // every register is defined exactly once now (verify checked it); the original variables are gone
    std::string text = toString(fn);
    CHECK(text.find("%acc.11") == std::string::npos);
    CHECK(text.find("%c.12") == std::string::npos);
}

TEST_CASE("SSA construction: if/else merge and value-position ||") {
    Function fn = compileKernel(R"(
kernel k(n: i32, p: *i32) {
    var x: i32 = 0;
    if (n > 3) { x = 1; } else { x = 2; }
    var f: bool = n < 1 || n > 9;
    p[0] = x;
    p[1] = f as i32;
}
)");
    transform::constructSSA(fn);
    CHECK(verify(fn, Form::SSA).empty());
    int phis = 0;
    for (const auto& bb : fn.blocks) {
        for (const Inst& in : bb->insts) phis += in.op == Op::Phi;
    }
    CHECK(phis == 2);  // x at if.end, the bool at sc.end
}

TEST_CASE("SSA construction removes unreachable blocks first") {
    Function fn = compileKernel("kernel k(p: *i32) { p[0] = 1; return; p[0] = 2; }");
    CHECK(fn.blocks.size() == 2);
    transform::constructSSA(fn);
    CHECK(fn.blocks.size() == 1);
    CHECK(verify(fn, Form::SSA).empty());
}

TEST_CASE("verifier: SSA invariants") {
    SUBCASE("a register defined twice") {
        Function fn;
        fn.name = "k";
        Builder b(fn);
        b.setBlock(fn.newBlock("entry"));
        int r = fn.newReg(Ty::I32);
        b.emitCopyTo(r, Ty::I32, Operand::i32(1));
        b.emitCopyTo(r, Ty::I32, Operand::i32(2));
        b.emitRet();
        cfg::rebuild(fn);
        CHECK(verify(fn, Form::Lowered).empty());
        auto problems = verify(fn, Form::SSA);
        REQUIRE(problems.size() == 1);
        CHECK(problems[0].find("defined more than once") != std::string::npos);
    }
    SUBCASE("a use not dominated by its definition") {
        Function fn;
        fn.name = "k";
        int c = fn.newReg(Ty::Bool, "c");
        fn.numParams = 1;
        Builder b(fn);
        BasicBlock *e = fn.newBlock("entry"), *l = fn.newBlock("l"), *r = fn.newBlock("r"), *j = fn.newBlock("j");
        b.setBlock(e); b.emitBr(Operand::r(c), l, r);
        b.setBlock(l);
        Operand x = b.emitCopy(Ty::I32, Operand::i32(1));
        b.emitJmp(j);
        b.setBlock(r); b.emitJmp(j);
        b.setBlock(j);
        b.emitBinary(Op::Add, Ty::I32, x, Operand::i32(1));  // x is only defined on one path
        b.emitRet();
        cfg::rebuild(fn);
        auto problems = verify(fn, Form::SSA);
        REQUIRE(problems.size() == 1);
        CHECK(problems[0].find("not dominated") != std::string::npos);
    }
    SUBCASE("use before definition in the same block") {
        Function fn;
        fn.name = "k";
        Builder b(fn);
        b.setBlock(fn.newBlock("entry"));
        int late = fn.newReg(Ty::I32);
        b.emitBinary(Op::Add, Ty::I32, Operand::r(late), Operand::i32(1));
        b.emitCopyTo(late, Ty::I32, Operand::i32(1));
        b.emitRet();
        cfg::rebuild(fn);
        CHECK(!verify(fn, Form::SSA).empty());
    }
    SUBCASE("phi entries must match the predecessors") {
        Function fn;
        fn.name = "k";
        Builder b(fn);
        BasicBlock *e = fn.newBlock("entry"), *j = fn.newBlock("j");
        b.setBlock(e); b.emitJmp(j);
        b.setBlock(j);
        b.emitPhi(Ty::I32, {{Operand::i32(1), e}, {Operand::i32(2), e}});  // e listed twice
        b.emitRet();
        cfg::rebuild(fn);
        auto problems = verify(fn, Form::SSA);
        REQUIRE(problems.size() == 1);
        CHECK(problems[0].find("exactly one entry per predecessor") != std::string::npos);
    }
    SUBCASE("phi after a non-phi") {
        Function fn;
        fn.name = "k";
        Builder b(fn);
        BasicBlock *e = fn.newBlock("entry"), *j = fn.newBlock("j");
        b.setBlock(e); b.emitJmp(j);
        b.setBlock(j);
        b.emitCopy(Ty::I32, Operand::i32(1));
        b.emitPhi(Ty::I32, {{Operand::i32(1), e}});
        b.emitRet();
        cfg::rebuild(fn);
        bool found = false;
        for (const std::string& p : verify(fn, Form::SSA)) found |= p.find("phi after a non-phi") != std::string::npos;
        CHECK(found);
    }
    SUBCASE("a phi operand must be available at the end of its predecessor") {
        Function fn;
        fn.name = "k";
        int c = fn.newReg(Ty::Bool, "c");
        fn.numParams = 1;
        Builder b(fn);
        BasicBlock *e = fn.newBlock("entry"), *l = fn.newBlock("l"), *r = fn.newBlock("r"), *j = fn.newBlock("j");
        b.setBlock(e); b.emitBr(Operand::r(c), l, r);
        b.setBlock(l);
        Operand x = b.emitCopy(Ty::I32, Operand::i32(1));
        b.emitJmp(j);
        b.setBlock(r); b.emitJmp(j);
        b.setBlock(j);
        b.emitPhi(Ty::I32, {{x, r}, {x, l}});  // x does not reach the end of r
        b.emitRet();
        cfg::rebuild(fn);
        auto problems = verify(fn, Form::SSA);
        REQUIRE(problems.size() == 1);
        CHECK(problems[0].find("phi operand") != std::string::npos);
    }
    SUBCASE("undef is fine as a phi operand") {
        Function fn;
        fn.name = "k";
        Builder b(fn);
        BasicBlock *e = fn.newBlock("entry"), *j = fn.newBlock("j");
        b.setBlock(e); b.emitJmp(j);
        b.setBlock(j);
        b.emitPhi(Ty::I32, {{Operand::undef(), e}});
        b.emitRet();
        cfg::rebuild(fn);
        CHECK(verify(fn, Form::SSA).empty());
    }
}

namespace {

// Applies the copies in order to a register file and returns the result.
std::map<int, int> apply(const std::vector<transform::Copy>& copies, std::map<int, int> regs) {
    for (const auto& c : copies) regs[c.dst] = c.src.isReg() ? regs[c.src.reg] : c.src.i;
    return regs;
}

std::vector<transform::Copy> sequentialize(std::vector<transform::Copy> parallel, int firstTemp = 100) {
    int next = firstTemp;
    return transform::sequentializeParallelCopy(std::move(parallel), [&](int) { return next++; });
}

}  // namespace

TEST_CASE("parallel copy: swap needs a temporary") {
    auto out = sequentialize({{1, Operand::r(2)}, {2, Operand::r(1)}});
    CHECK(out.size() == 3);
    auto regs = apply(out, {{1, 10}, {2, 20}});
    CHECK(regs[1] == 20);
    CHECK(regs[2] == 10);
}

TEST_CASE("parallel copy: order matters for a chain") {
    auto out = sequentialize({{1, Operand::r(2)}, {2, Operand::r(3)}});
    CHECK(out.size() == 2);  // no temporary
    auto regs = apply(out, {{1, 10}, {2, 20}, {3, 30}});
    CHECK(regs[1] == 20);
    CHECK(regs[2] == 30);
}

TEST_CASE("parallel copy: self copies vanish, immediates never conflict") {
    CHECK(sequentialize({{1, Operand::r(1)}}).empty());
    auto out = sequentialize({{1, Operand::i32(7)}, {2, Operand::r(1)}});
    auto regs = apply(out, {{1, 10}, {2, 20}});
    CHECK(regs[1] == 7);
    CHECK(regs[2] == 10);  // reads the old value of 1
}

TEST_CASE("parallel copy: three-cycle and a cycle with a tail") {
    auto out = sequentialize({{1, Operand::r(2)}, {2, Operand::r(3)}, {3, Operand::r(1)}});
    auto regs = apply(out, {{1, 10}, {2, 20}, {3, 30}});
    CHECK(regs[1] == 20);
    CHECK(regs[2] == 30);
    CHECK(regs[3] == 10);
    out = sequentialize({{1, Operand::r(2)}, {2, Operand::r(1)}, {3, Operand::r(1)}});
    regs = apply(out, {{1, 10}, {2, 20}, {3, 30}});
    CHECK(regs[1] == 20);
    CHECK(regs[2] == 10);
    CHECK(regs[3] == 10);
}

namespace {

// entry: jmp loop
// loop:  a = phi [1, entry], [b, loop]; b = phi [2, entry], [a, loop]; i = phi [0, entry], [i2, loop]
//        i2 = i + 1; c = i2 < 3; br c, loop, done
// done:  out[0] = a; out[1] = b; out[2] = i; ret
Function swapLoop() {
    Function fn;
    fn.name = "swap";
    int out = fn.newReg(Ty::PtrI32, "out");
    fn.numParams = 1;
    Builder bld(fn);
    BasicBlock *entry = fn.newBlock("entry"), *loop = fn.newBlock("loop"), *done = fn.newBlock("done");
    bld.setBlock(entry);
    bld.emitJmp(loop);
    int a = fn.newReg(Ty::I32, "a"), b = fn.newReg(Ty::I32, "b"), i = fn.newReg(Ty::I32, "i");
    int i2 = fn.newReg(Ty::I32, "i2"), c = fn.newReg(Ty::Bool, "c");
    auto phi = [&](int dst, Operand init, Operand back) {
        Inst p;
        p.op = Op::Phi; p.ty = Ty::I32; p.dst = dst;
        p.args = {init, back};
        p.blocks = {entry, loop};
        return p;
    };
    bld.setBlock(loop);
    bld.append(phi(a, Operand::i32(1), Operand::r(b)));
    bld.append(phi(b, Operand::i32(2), Operand::r(a)));
    bld.append(phi(i, Operand::i32(0), Operand::r(i2)));
    Inst add;
    add.op = Op::Add; add.ty = Ty::I32; add.dst = i2; add.args = {Operand::r(i), Operand::i32(1)};
    bld.append(add);
    Inst lt;
    lt.op = Op::Lt; lt.ty = Ty::I32; lt.dst = c; lt.args = {Operand::r(i2), Operand::i32(3)};
    bld.append(lt);
    bld.emitBr(Operand::r(c), loop, done);
    bld.setBlock(done);
    bld.emitStore(Ty::I32, Space::Global, Operand::r(out), Operand::i32(0), Operand::r(a));
    bld.emitStore(Ty::I32, Space::Global, Operand::r(out), Operand::i32(1), Operand::r(b));
    bld.emitStore(Ty::I32, Space::Global, Operand::r(out), Operand::i32(2), Operand::r(i));  // the lost-copy hazard
    bld.emitRet();
    cfg::rebuild(fn);
    return fn;
}

std::vector<int32_t> runSwap(const Function& fn) {
    std::vector<interp::Buffer> bufs = {intBuffer({0, 0, 0})};
    interp::Launch l;
    interp::Result r = interp::run(fn, l, {0}, bufs);
    REQUIRE_MESSAGE(r.ok, r.error);
    return {static_cast<int32_t>(bufs[0].data[0]), static_cast<int32_t>(bufs[0].data[1]), static_cast<int32_t>(bufs[0].data[2])};
}

}  // namespace

TEST_CASE("out-of-SSA: the swap and lost-copy problems") {
    Function fn = swapLoop();
    REQUIRE(verify(fn, Form::SSA).empty());
    // Phis run from entry, then twice from loop: (a,b) = (1,2) -> (2,1) -> (1,2); i is 2 when the loop exits.
    CHECK(runSwap(fn) == std::vector<int32_t>{1, 2, 2});

    transform::destructSSA(fn);
    CHECK(verify(fn, Form::PostSSA).empty());
    CHECK(runSwap(fn) == std::vector<int32_t>{1, 2, 2});
    // The back edge loop->loop is critical (loop branches, loop has phis), so it got its own block, and the
    // swap of a and b needed a temporary.
    bool split = false, swapTemp = false;
    for (const auto& bb : fn.blocks) split |= bb->hint == "split";
    for (int r = 0; r < fn.numRegs(); ++r) swapTemp |= fn.regHint[r] == "swap";
    CHECK(split);
    CHECK(swapTemp);
}

TEST_CASE("out-of-SSA: no copy for undef operands, phis are gone") {
    Function fn;
    fn.name = "k";
    int out = fn.newReg(Ty::PtrI32, "out");
    fn.numParams = 1;
    Builder b(fn);
    BasicBlock *e = fn.newBlock("entry"), *j = fn.newBlock("j");
    b.setBlock(e); b.emitJmp(j);
    b.setBlock(j);
    Operand v = b.emitPhi(Ty::I32, {{Operand::undef(), e}});
    b.emitStore(Ty::I32, Space::Global, Operand::r(out), Operand::i32(0), v);
    b.emitRet();
    cfg::rebuild(fn);
    transform::destructSSA(fn);
    for (const auto& bb : fn.blocks) {
        for (const Inst& in : bb->insts) {
            CHECK(in.op != Op::Phi);
            CHECK(in.op != Op::Copy);
        }
    }
}

TEST_CASE("SSA round trip preserves results (interpreter)") {
    struct Case {
        const char* src;
        std::vector<int32_t> in;
    };
    const Case cases[] = {
        {"kernel k(n: i32, p: *i32) { var s: i32 = 0; var t: i32 = 1; for (var i: i32 = 0; i < 10; i += 1) { var u: i32 = s; s = t; t = u + t; } p[0] = s; p[1] = t; p[2] = n; }", {7, 0, 0}},
        {"kernel k(n: i32, p: *i32) { var a: i32 = 1; var b: i32 = 2; for (var i: i32 = 0; i < n; i += 1) { var t: i32 = a; a = b; b = t; } p[0] = a; p[1] = b; }", {5, 0, 0}},
        {"kernel k(n: i32, p: *i32) { var x: i32 = 0; while (true) { x += 3; if (x > n) { break; } if (x % 2 == 0) { continue; } x += 1; } p[0] = x; }", {40, 0, 0}},
        {"kernel k(n: i32, p: *i32) { var f: bool = n < 3 || n > 8 && n != 10; var x: i32 = 5; if (f) { x = 6; } else if (n == 5) { x = 7; } p[0] = x; p[1] = f as i32; }", {10, 0, 0}},
        {"kernel k(n: i32, p: *i32) { for (var i: i32 = 0; i < 3; i += 1) { for (var j: i32 = 0; j < 3; j += 1) { if (j == 2) { break; } p[i] = p[i] + j + n; } } }", {1, 1, 1}},
    };
    for (const Case& c : cases) {
        Module m = compile(c.src);
        Function& fn = m.kernels[0];
        auto run = [&](const Function& f) {
            std::vector<interp::Buffer> bufs = {intBuffer(c.in)};
            interp::Launch l;
            // `n` is the first parameter; give it the buffer's first element so each case has its own input
            interp::Result r = interp::run(f, l, {static_cast<uint32_t>(c.in[0]), 0}, bufs);
            REQUIRE_MESSAGE(r.ok, r.error);
            return bufs[0].data;
        };
        auto expected = run(fn);
        transform::constructSSA(fn);
        REQUIRE_MESSAGE(verify(fn, Form::SSA).empty(), c.src);
        CHECK(run(fn) == expected);
        transform::destructSSA(fn);
        REQUIRE_MESSAGE(verify(fn, Form::PostSSA).empty(), c.src);
        CHECK(run(fn) == expected);
    }
}

TEST_CASE("pass driver: forms are tracked and wrong orders are rejected") {
    Module m = compile("kernel k(p: *i32) { p[0] = 1; }");
    Form form = Form::Lowered;
    std::string err;
    auto run = [&](std::vector<std::string> names) {
        transform::PassManager pm({true, false, false});
        return pm.run(m, transform::flatPipeline(names), form, err);
    };
    CHECK_FALSE(run({"out-of-ssa"}));
    CHECK(err.find("needs SSA IR but the IR is lowered") != std::string::npos);
    CHECK_FALSE(run({"nope"}));
    CHECK(err.find("unknown pass 'nope'") != std::string::npos);
    CHECK_FALSE(run({"constfold"}));  // the optimizer passes need SSA
    CHECK(run({"ssa"}));
    CHECK(form == Form::SSA);
    CHECK_FALSE(run({"ssa"}));
    CHECK(run({"out-of-ssa"}));
    CHECK(form == Form::PostSSA);
}
