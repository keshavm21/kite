#pragma once
#include <cstring>
#include <sstream>
#include <string>

#include "doctest/doctest.h"
#include "frontend/parser.h"
#include "frontend/sema.h"
#include "interp/interp.h"
#include "ir/lower.h"
#include "ir/printer.h"
#include "ir/verifier.h"

namespace kite::test {

// Parses, checks and lowers `src`; fails the test if it has a front-end error or the IR does not verify.
inline ir::Module compile(const std::string& src) {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    auto program = parseSource(src, diag);
    REQUIRE_MESSAGE(program.has_value(), err.str());
    Sema(diag).check(*program);
    REQUIRE_MESSAGE(!diag.hasErrors(), err.str());
    ir::Module m = ir::lowerProgram(*program);
    for (const ir::Function& fn : m.kernels) {
        for (const std::string& problem : ir::verify(fn)) FAIL_CHECK(problem);
    }
    return m;
}

inline ir::Function compileKernel(const std::string& src) {
    ir::Module m = compile(src);
    REQUIRE(m.kernels.size() == 1);
    return std::move(m.kernels[0]);
}

inline uint32_t bitsOf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}
inline float floatOf(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

inline interp::Buffer intBuffer(std::vector<int32_t> v) {
    interp::Buffer b;
    b.elem = ir::Ty::I32;
    for (int32_t x : v) b.data.push_back(static_cast<uint32_t>(x));
    return b;
}
inline interp::Buffer floatBuffer(std::vector<float> v) {
    interp::Buffer b;
    b.elem = ir::Ty::F32;
    for (float x : v) b.data.push_back(bitsOf(x));
    return b;
}

}  // namespace kite::test
