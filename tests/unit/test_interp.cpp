#include <cmath>
#include <limits>

#include "analysis/cfg.h"
#include "doctest/doctest.h"
#include "interp/spec.h"
#include "ir/builder.h"
#include "test_util.h"

using namespace kite;
using namespace kite::interp;
using namespace kite::test;

namespace {

Result runOn(const ir::Function& fn, std::vector<Buffer>& bufs, std::vector<uint32_t> args, int grid = 1, int block = 1,
             uint64_t stepLimit = kDefaultStepLimit) {
    Launch l;
    l.grid[0] = grid;
    l.block[0] = block;
    return run(fn, l, args, bufs, stepLimit);
}

// Runs a one-thread kernel `kernel k(p: *i32 ...)` over a 1-element i32 buffer; returns the result.
Result runSingle(const std::string& body, std::vector<Buffer>& bufs, std::vector<uint32_t> args = {0}) {
    ir::Function fn = compileKernel(body);
    return runOn(fn, bufs, args);
}

bool mentions(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

}  // namespace

TEST_CASE("xorshift32 known-answer test (docs/TESTING.md section 2.1)") {
    XorShift32 a(1);
    CHECK(a.next() == 270369u);
    CHECK(a.next() == 67634689u);
    CHECK(a.next() == 2647435461u);

    XorShift32 f(1);
    CHECK(static_cast<double>(f.nextF32()) == doctest::Approx(-0.99987411499).epsilon(1e-9));
    CHECK(static_cast<double>(f.nextF32()) == doctest::Approx(-0.96850514412).epsilon(1e-9));
    CHECK(static_cast<double>(f.nextF32()) == doctest::Approx(0.23280811310).epsilon(1e-9));

    XorShift32 i(1);
    CHECK(i.nextI32() == -49);
    CHECK(i.nextI32() == -16);
    CHECK(i.nextI32() == -6);

    XorShift32 zero(0);  // seed 0 is replaced
    CHECK(zero.s == 0x9E3779B9u);
}

TEST_CASE("spec parsing") {
    Spec s;
    std::string err;
    REQUIRE(parseSpec(R"(// header
// @kernel saxpy
// @grid 4
// @block 16 16
// @arg n i32 1000
// @arg a f32 2.5
// @arg x f32[1000] rand(1)
//   @check x
kernel saxpy() {}
)", s, err));
    CHECK(s.kernel == "saxpy");
    CHECK(s.grid[0] == 4);
    CHECK(s.grid[1] == 1);
    CHECK(s.grid[2] == 1);
    CHECK(s.block[0] == 16);
    CHECK(s.block[1] == 16);
    REQUIRE(s.args.size() == 3);
    CHECK_FALSE(s.args[0].isBuffer);
    CHECK(s.args[1].elem == ir::Ty::F32);
    CHECK(s.args[2].isBuffer);
    CHECK(s.args[2].count == 1000);
    CHECK(s.args[2].init == "rand(1)");
}

TEST_CASE("spec parsing rejects bad input") {
    Spec s;
    std::string err;
    CHECK_FALSE(parseSpec("// @grid 1\n// @block 1\n", s, err));
    CHECK(mentions(err, "needs @kernel"));
    CHECK_FALSE(parseSpec("// @kernel k\n// @grid 1 2 3 4\n// @block 1\n", s, err));
    CHECK(mentions(err, "line 2"));
    CHECK_FALSE(parseSpec("// @kernel k\n// @grid 0\n// @block 1\n", s, err));
    CHECK_FALSE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg x f64 1\n", s, err));
    CHECK(mentions(err, "bad type"));
    CHECK_FALSE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg x f32[] zeros\n", s, err));
    CHECK_FALSE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @bogus 1\n", s, err));
    CHECK(mentions(err, "unknown directive @bogus"));
    CHECK_FALSE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @check nope\n", s, err));
    CHECK(mentions(err, "no such"));
}

TEST_CASE("spec inputs and output format") {
    Spec s;
    std::string err;
    REQUIRE(parseSpec(R"(// @kernel k
// @grid 1
// @block 1
// @arg n i32 -7
// @arg f f32 0.1
// @arg a f32[4] iota
// @arg b i32[3] fill(5)
// @arg c f32[2] ones
// @arg r i32[4] rand(1)
// @arg z f32[2] zeros
// @check a b c r z
)", s, err));
    Inputs in;
    REQUIRE_MESSAGE(makeInputs(s, in, err), err);
    CHECK(static_cast<int32_t>(in.args[0]) == -7);
    CHECK(floatOf(in.args[1]) == 0.1f);
    REQUIRE(in.buffers.size() == 5);
    CHECK(in.args[2] == 0u);  // pointer arguments are buffer indices
    CHECK(in.args[3] == 1u);
    CHECK(floatOf(in.buffers[0].data[3]) == 3.0f);
    CHECK(static_cast<int32_t>(in.buffers[1].data[2]) == 5);
    CHECK(floatOf(in.buffers[2].data[1]) == 1.0f);
    CHECK(static_cast<int32_t>(in.buffers[3].data[0]) == -49);  // rand(1): the known-answer table
    CHECK(static_cast<int32_t>(in.buffers[3].data[1]) == -16);
    CHECK(static_cast<int32_t>(in.buffers[3].data[2]) == -6);
    CHECK(formatOutputs(s, in) ==
          "a: 0 1 2 3\n"
          "b: 5 5 5\n"
          "c: 1 1\n"
          "r: -49 -16 -6 " + std::to_string(static_cast<int32_t>(in.buffers[3].data[3])) + "\n"
          "z: 0 0\n");
}

TEST_CASE("spec rand(1) f32 buffers match the known-answer table") {
    Spec s;
    std::string err;
    REQUIRE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg x f32[3] rand(1)\n", s, err));
    Inputs in;
    REQUIRE(makeInputs(s, in, err));
    CHECK(static_cast<double>(floatOf(in.buffers[0].data[0])) == doctest::Approx(-0.99987411499).epsilon(1e-9));
    CHECK(static_cast<double>(floatOf(in.buffers[0].data[2])) == doctest::Approx(0.23280811310).epsilon(1e-9));
}

TEST_CASE("spec signature check") {
    ir::Function fn = compileKernel("kernel k(n: i32, x: *f32) {}");
    Spec s;
    std::string err;
    REQUIRE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg n i32 1\n// @arg x f32[2] zeros\n", s, err));
    CHECK(checkSignature(s, fn, err));
    REQUIRE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg n i32 1\n// @arg x i32[2] zeros\n", s, err));
    CHECK_FALSE(checkSignature(s, fn, err));
    REQUIRE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg m i32 1\n// @arg x f32[2] zeros\n", s, err));
    CHECK_FALSE(checkSignature(s, fn, err));
    REQUIRE(parseSpec("// @kernel k\n// @grid 1\n// @block 1\n// @arg n i32 1\n", s, err));
    CHECK_FALSE(checkSignature(s, fn, err));
}

TEST_CASE("interpreter: thread and block indices follow x-fastest order") {
    ir::Function fn = compileKernel(R"(
kernel k(out: *i32) {
    var lin: i32 = (block_idx.y * grid_dim.x + block_idx.x) * (block_dim.x * block_dim.y) + thread_idx.y * block_dim.x + thread_idx.x;
    out[lin] = block_idx.x * 1000 + block_idx.y * 100 + thread_idx.y * 10 + thread_idx.x;
}
)");
    std::vector<Buffer> bufs = {intBuffer(std::vector<int32_t>(2 * 2 * 3 * 2, -1))};
    Launch l;
    l.grid[0] = 2; l.grid[1] = 2;
    l.block[0] = 3; l.block[1] = 2;
    REQUIRE(run(fn, l, {0}, bufs).ok);
    // Every slot written exactly once, with the expected id.
    for (int by = 0; by < 2; ++by)
        for (int bx = 0; bx < 2; ++bx)
            for (int ty = 0; ty < 2; ++ty)
                for (int tx = 0; tx < 3; ++tx) {
                    int lin = (by * 2 + bx) * 6 + ty * 3 + tx;
                    CHECK(static_cast<int32_t>(bufs[0].data[lin]) == bx * 1000 + by * 100 + ty * 10 + tx);
                }
}

TEST_CASE("interpreter: i32 wraps, f32 stays f32") {
    std::vector<Buffer> bufs = {intBuffer({0, 0, 0}), floatBuffer({0, 0})};
    ir::Function fn = compileKernel(R"(
kernel k(p: *i32, q: *f32) {
    var big: i32 = 2147483647;
    p[0] = big + 1;
    p[1] = -2147483648 - 1;
    p[2] = big * 2;
    var f: f32 = 16777216.0;
    q[0] = f + 1.0;
    q[1] = 0.1 * 3.0;
}
)");
    REQUIRE(runOn(fn, bufs, {0, 1}).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == std::numeric_limits<int32_t>::min());
    CHECK(static_cast<int32_t>(bufs[0].data[1]) == std::numeric_limits<int32_t>::max());
    CHECK(static_cast<int32_t>(bufs[0].data[2]) == -2);
    CHECK(floatOf(bufs[1].data[0]) == 16777216.0f);          // 2^24 + 1 is not representable in f32
    CHECK(bufs[1].data[1] == bitsOf(0.1f * 3.0f));           // single-precision product
}

TEST_CASE("interpreter: division and remainder follow C") {
    std::vector<Buffer> bufs = {intBuffer({0, 0, 0, 0})};
    ir::Function fn = compileKernel("kernel k(p: *i32) { p[0] = -7 / 2; p[1] = -7 % 2; p[2] = 7 % -2; p[3] = 7 / -2; }");
    REQUIRE(runOn(fn, bufs, {0}).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == -3);
    CHECK(static_cast<int32_t>(bufs[0].data[1]) == -1);
    CHECK(static_cast<int32_t>(bufs[0].data[2]) == 1);
    CHECK(static_cast<int32_t>(bufs[0].data[3]) == -3);
}

TEST_CASE("interpreter: casts, min/max, abs, sqrt") {
    std::vector<Buffer> bufs = {intBuffer({0, 0, 0, 0}), floatBuffer({0, 0, 0})};
    ir::Function fn = compileKernel(R"(
kernel k(p: *i32, q: *f32) {
    p[0] = -2.9 as i32;
    p[1] = (3 < 5) as i32;
    p[2] = abs(-2147483648);
    p[3] = min(-3, 4) + max(-3, 4);
    q[0] = 5 as f32;
    q[1] = sqrt(2.0);
    q[2] = max(1.0, 2.0) - min(1.0, 2.0);
}
)");
    REQUIRE(runOn(fn, bufs, {0, 1}).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == -2);
    CHECK(static_cast<int32_t>(bufs[0].data[1]) == 1);
    CHECK(static_cast<int32_t>(bufs[0].data[2]) == std::numeric_limits<int32_t>::min());
    CHECK(static_cast<int32_t>(bufs[0].data[3]) == 1);
    CHECK(floatOf(bufs[1].data[0]) == 5.0f);
    CHECK(floatOf(bufs[1].data[1]) == std::sqrt(2.0f));
    CHECK(floatOf(bufs[1].data[2]) == 1.0f);
}

TEST_CASE("interpreter: && and || short-circuit") {
    // p has 1 element, so evaluating p[5] would be out-of-bounds.
    std::vector<Buffer> bufs = {intBuffer({0})};
    ir::Function fn = compileKernel(R"(
kernel k(p: *i32) {
    var i: i32 = 5;
    if (i < 1 && p[i] > 0) { p[0] = 1; }
    if (i > 1 || p[i] > 0) { p[0] = p[0] + 10; }
}
)");
    REQUIRE_MESSAGE(runOn(fn, bufs, {0}).ok, "right-hand sides must not run");
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == 10);
}

TEST_CASE("interpreter: loops, break and continue (continue in a for runs the step)") {
    std::vector<Buffer> bufs = {intBuffer({0, 0})};
    ir::Function fn = compileKernel(R"(
kernel k(p: *i32) {
    var s: i32 = 0;
    for (var i: i32 = 0; i < 100; i += 1) {
        if (i % 2 == 1) { continue; }
        if (i > 10) { break; }
        s += i;
    }
    p[0] = s;
    var j: i32 = 0;
    while (true) { j += 3; if (j > 20) { break; } }
    p[1] = j;
}
)");
    REQUIRE(runOn(fn, bufs, {0}).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == 0 + 2 + 4 + 6 + 8 + 10);
    CHECK(static_cast<int32_t>(bufs[0].data[1]) == 21);
}

TEST_CASE("interpreter: shared memory and barriers (block reduction)") {
    std::vector<Buffer> bufs = {intBuffer({1, 2, 3, 4, 5, 6, 7, 8}), intBuffer({0})};
    ir::Function fn = compileKernel(R"(
kernel k(x: *i32, out: *i32) {
    shared buf: i32[8];
    var t: i32 = thread_idx.x;
    buf[t] = x[t];
    sync();
    var s: i32 = block_dim.x / 2;
    while (s > 0) {
        if (t < s) { buf[t] = buf[t] + buf[t + s]; }
        sync();
        s = s / 2;
    }
    if (t == 0) { out[0] = buf[0]; }
}
)");
    REQUIRE(runOn(fn, bufs, {0, 1}, 1, 8).ok);
    CHECK(static_cast<int32_t>(bufs[1].data[0]) == 36);
}

TEST_CASE("interpreter: each block gets its own shared memory") {
    std::vector<Buffer> bufs = {intBuffer({0, 0})};
    ir::Function fn = compileKernel(R"(
kernel k(out: *i32) {
    shared s: i32[1];
    s[0] = s[0] + 1;          // zero-filled at block start, so this is 1 in every block
    sync();
    out[block_idx.x] = s[0];
}
)");
    REQUIRE(runOn(fn, bufs, {0}, 2, 1).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == 1);
    CHECK(static_cast<int32_t>(bufs[0].data[1]) == 1);
}

TEST_CASE("interpreter: undefined behaviour is reported") {
    std::vector<Buffer> bufs = {intBuffer({0})};

    SUBCASE("out-of-bounds store") {
        Result r = runSingle("kernel k(p: *i32) { p[1] = 0; }", bufs);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "error: out-of-bounds store p[1] (size 1) in kernel k, block 0, thread 0, at entry:"));
    }
    SUBCASE("negative index on a load") {
        Result r = runSingle("kernel k(p: *i32) { p[0] = p[0 - 1]; }", bufs);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "out-of-bounds load p[-1]"));
    }
    SUBCASE("out-of-bounds shared access") {
        Result r = runSingle("kernel k(p: *i32) { shared s: i32[4]; s[4] = 1; }", bufs);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "out-of-bounds store s[4] (size 4)"));
    }
    SUBCASE("division by zero") {
        Result r = runSingle("kernel k(p: *i32) { p[0] = 1 / p[0]; }", bufs);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "integer division by zero"));
    }
    SUBCASE("remainder by zero") {
        Result r = runSingle("kernel k(p: *i32) { p[0] = 1 % p[0]; }", bufs);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "integer division by zero"));
    }
    SUBCASE("INT_MIN / -1") {
        bufs[0].data[0] = static_cast<uint32_t>(-1);
        Result r = runSingle("kernel k(p: *i32) { p[0] = -2147483648 / p[0]; }", bufs);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "overflow"));
    }
    SUBCASE("ftoi of NaN") {
        std::vector<Buffer> fb = {floatBuffer({std::nanf("")}), intBuffer({0})};
        Result r = runSingle("kernel k(f: *f32, p: *i32) { p[0] = f[0] as i32; }", fb, {0, 1});
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "ftoi"));
    }
    SUBCASE("ftoi out of range") {
        std::vector<Buffer> fb = {floatBuffer({3.0e9f}), intBuffer({0})};
        CHECK_FALSE(runSingle("kernel k(f: *f32, p: *i32) { p[0] = f[0] as i32; }", fb, {0, 1}).ok);
    }
    SUBCASE("ftoi of the largest representable in-range values is fine") {
        std::vector<Buffer> fb = {floatBuffer({-2147483648.0f, 2147483520.0f}), intBuffer({0, 0})};
        CHECK(runSingle("kernel k(f: *f32, p: *i32) { p[0] = f[0] as i32; p[1] = f[1] as i32; }", fb, {0, 1}).ok);
        CHECK(static_cast<int32_t>(fb[1].data[0]) == std::numeric_limits<int32_t>::min());
        CHECK(static_cast<int32_t>(fb[1].data[1]) == 2147483520);
    }
    SUBCASE("step limit") {
        ir::Function fn = compileKernel("kernel k(p: *i32) { while (true) { p[0] = 1; } }");
        Result r = runOn(fn, bufs, {0}, 1, 1, 1000);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "step limit exceeded"));
    }
    SUBCASE("divergent barrier: some threads returned") {
        ir::Function fn = compileKernel("kernel k(p: *i32) { if (thread_idx.x == 0) { return; } sync(); }");
        Result r = runOn(fn, bufs, {0}, 1, 4);
        CHECK_FALSE(r.ok);
        CHECK(mentions(r.error, "divergent barrier"));
    }
    SUBCASE("a barrier every thread reaches is fine") {
        ir::Function fn = compileKernel("kernel k(p: *i32) { sync(); sync(); }");
        CHECK(runOn(fn, bufs, {0}, 2, 4).ok);
    }
}

TEST_CASE("interpreter: a failing thread is identified") {
    std::vector<Buffer> bufs = {intBuffer({0, 0, 0})};
    ir::Function fn = compileKernel("kernel k(p: *i32) { p[block_idx.x * block_dim.x + thread_idx.x] = 1; }");
    Result r = runOn(fn, bufs, {0}, 2, 2);  // 4 threads, 3 elements: block 1, thread 1 (linear thread 1) overflows
    CHECK_FALSE(r.ok);
    CHECK(mentions(r.error, "p[3] (size 3) in kernel k, block 1, thread 1"));
}

TEST_CASE("interpreter: phis are parallel assignments") {
    using namespace ir;
    // entry: jmp loop
    // loop:  a = phi [1, entry], [b, loop]; b = phi [2, entry], [a, loop]; i = phi [0, entry], [i2, loop]
    //        i2 = i + 1; c = i2 < 3; br c, loop, done
    // done:  out[0] = a; out[1] = b; ret
    Function fn;
    fn.name = "swap";
    int out = fn.newReg(Ty::PtrI32, "out");
    fn.numParams = 1;
    Builder bld(fn);
    BasicBlock* entry = fn.newBlock("entry");
    BasicBlock* loop = fn.newBlock("loop");
    BasicBlock* done = fn.newBlock("done");
    bld.setBlock(entry);
    bld.emitJmp(loop);

    int a = fn.newReg(Ty::I32, "a"), b = fn.newReg(Ty::I32, "b"), i = fn.newReg(Ty::I32, "i");
    int i2 = fn.newReg(Ty::I32, "i2"), c = fn.newReg(Ty::Bool, "c");
    auto phi = [&](int dst, Operand init, Operand back) {
        Inst p;
        p.op = Op::Phi;
        p.ty = Ty::I32;
        p.dst = dst;
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
    bld.emitRet();
    cfg::rebuild(fn);
    CHECK(verify(fn, Form::SSA).empty());  // (phi placement checks arrive with the SSA verifier)

    // The phis run on entry from `entry`, then twice from `loop` (the branch is taken for i2 = 1, 2):
    // (a,b) = (1,2) -> (2,1) -> (1,2). Evaluating the phis one after another would give (2,2) instead.
    std::vector<Buffer> bufs = {intBuffer({0, 0})};
    REQUIRE(runOn(fn, bufs, {0}).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == 1);
    CHECK(static_cast<int32_t>(bufs[0].data[1]) == 2);
}

TEST_CASE("interpreter: undef phi operands leave the register alone and a missing edge is an error") {
    using namespace ir;
    Function fn;
    fn.name = "k";
    int out = fn.newReg(Ty::PtrI32, "out");
    fn.numParams = 1;
    Builder bld(fn);
    BasicBlock* entry = fn.newBlock("entry");
    BasicBlock* next = fn.newBlock("next");
    BasicBlock* other = fn.newBlock("other");
    bld.setBlock(entry);
    bld.emitJmp(next);
    bld.setBlock(other);
    bld.emitJmp(next);
    bld.setBlock(next);
    int v = fn.newReg(Ty::I32, "v");
    Inst p;
    p.op = Op::Phi; p.ty = Ty::I32; p.dst = v;
    p.args = {Operand::undef()};
    p.blocks = {entry};
    bld.append(p);
    bld.emitStore(Ty::I32, Space::Global, Operand::r(out), Operand::i32(0), Operand::r(v));
    bld.emitRet();
    cfg::rebuild(fn);
    std::vector<Buffer> bufs = {intBuffer({7})};
    REQUIRE(runOn(fn, bufs, {0}).ok);
    CHECK(static_cast<int32_t>(bufs[0].data[0]) == 0);  // registers start at zero; undef writes nothing

    // Entering `next` from `other` (the phi only lists `entry`) is an error.
    fn.blocks[0]->insts.back().blocks = {other};
    cfg::rebuild(fn);
    Result r = runOn(fn, bufs, {0});
    CHECK_FALSE(r.ok);
    CHECK(mentions(r.error, "phi has no value for predecessor other"));
}

TEST_CASE("interpreter: argument count mismatch is an error") {
    ir::Function fn = compileKernel("kernel k(p: *i32) {}");
    std::vector<Buffer> bufs;
    CHECK_FALSE(runOn(fn, bufs, {}).ok);
}
