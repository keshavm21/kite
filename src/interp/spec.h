#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "interp/interp.h"
#include "ir/ir.h"

// The `// @kernel ... // @check` header of an executable .kite file (docs/TESTING.md section 2).
// The syntax and the RNG must match gpu/kite_spec.py exactly.
namespace kite::interp {

// Marsaglia xorshift32. The sequence is part of the test contract (known-answer test in tests/unit).
struct XorShift32 {
    uint32_t s;
    explicit XorShift32(uint32_t seed) : s(seed ? seed : 0x9E3779B9u) {}
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    float nextF32() { return ((next() >> 8) / 16777216.0f) * 2.0f - 1.0f; }  // [-1, 1), exact in f32
    int32_t nextI32() { return static_cast<int32_t>((next() >> 8) % 201) - 100; }  // [-100, 100]
};

struct SpecArg {
    std::string name;
    ir::Ty elem = ir::Ty::I32;  // I32 or F32
    bool isBuffer = false;
    long long count = 0;        // buffers only
    std::string init;           // scalar literal, or zeros|ones|iota|rand(S)|fill(V)
};

struct Spec {
    std::string kernel;
    int grid[3] = {1, 1, 1};
    int block[3] = {1, 1, 1};
    std::vector<SpecArg> args;
    std::vector<std::string> check;
};

// Parses every `// @directive ...` line of `source`. Returns false and sets `err` on bad syntax.
bool parseSpec(const std::string& source, Spec& out, std::string& err);

// Checks the @arg list against the kernel's parameters (count, order, names, types).
bool checkSignature(const Spec& spec, const ir::Function& fn, std::string& err);

struct Inputs {
    std::vector<Buffer> buffers;   // one per buffer @arg, in order
    std::vector<uint32_t> args;    // one per @arg: scalar bits, or the index into `buffers`
};

// Builds the argument values and initial buffer contents.
bool makeInputs(const Spec& spec, Inputs& out, std::string& err);

// One line per @check buffer, in @arg order: `name: v0 v1 ...` (%.9g for f32, %d for i32).
std::string formatOutputs(const Spec& spec, const Inputs& inputs);

}  // namespace kite::interp
