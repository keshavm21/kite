#include <sstream>

#include "doctest/doctest.h"
#include "frontend/ast_printer.h"
#include "frontend/parser.h"

using namespace kite;

namespace {

// Parses `EXPR` as the initializer of a variable and returns its S-expression form.
std::string expr(const std::string& e) {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    auto p = parseSource("kernel k() { var r: i32 = " + e + "; }", diag);
    REQUIRE_MESSAGE(p.has_value(), err.str());
    auto& decl = static_cast<const VarDeclStmt&>(*p->kernels[0].body->stmts[0]);
    return exprToString(*decl.init);
}

// Parses a statement list inside a kernel body and returns the dumped body lines.
std::string stmts(const std::string& body) {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    auto p = parseSource("kernel k(p: *f32, n: i32) { " + body + " }", diag);
    REQUIRE_MESSAGE(p.has_value(), err.str());
    return dumpProgram(*p);
}

std::string parseError(const std::string& src) {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    CHECK_FALSE(parseSource(src, diag).has_value());
    return err.str();
}

}  // namespace

TEST_CASE("parser: precedence and associativity") {
    CHECK(expr("1 + 2 * 3") == "(+ 1 (* 2 3))");
    CHECK(expr("1 - 2 - 3") == "(- (- 1 2) 3)");
    CHECK(expr("8 / 4 / 2") == "(/ (/ 8 4) 2)");
    CHECK(expr("(1 + 2) * 3") == "(* (+ 1 2) 3)");
    CHECK(expr("a || b && c") == "(|| a (&& b c))");
    CHECK(expr("a && b || c") == "(|| (&& a b) c)");
    CHECK(expr("a < b == c < d") == "(== (< a b) (< c d))");
    CHECK(expr("a + b < c * d") == "(< (+ a b) (* c d))");
    CHECK(expr("a % b * c") == "(* (% a b) c)");
}

TEST_CASE("parser: unary and cast") {
    CHECK(expr("-x as f32") == "(as (neg x) f32)");
    CHECK(expr("t * 2 + flag as i32") == "(+ (* t 2) (as flag i32))");
    CHECK(expr("x as i32 as f32") == "(as (as x i32) f32)");
    CHECK(expr("!a && b") == "(&& (not a) b)");
    CHECK(expr("- - x") == "(neg (neg x))");
}

TEST_CASE("parser: primaries") {
    CHECK(expr("2.5") == "2.5");
    CHECK(expr("1.0") == "1.0");
    CHECK(expr("true || false") == "(|| true false)");
    CHECK(expr("x[i + 1]") == "x[(+ i 1)]");
    CHECK(expr("block_idx.y * block_dim.y + thread_idx.z") == "(+ (* block_idx.y block_dim.y) thread_idx.z)");
    CHECK(expr("max(a, min(b, 3))") == "(max a (min b 3))");
    CHECK(expr("sqrt(abs(v))") == "(sqrt (abs v))");
}

TEST_CASE("parser: compound assignment is desugared") {
    CHECK(stmts("n += 2;") ==
          "Program\n  Kernel k\n    Param p: *f32\n    Param n: i32\n    Block\n      Assign n = (+ n 2)\n");
    std::string d = stmts("p[n * 2] /= 4.0;");
    CHECK(d.find("Assign p[(* n 2)] = (/ p[(* n 2)] 4.0)") != std::string::npos);
}

TEST_CASE("parser: control flow shapes") {
    std::string d = stmts("if (n < 1) { return; } else if (n < 2) { break; } else { continue; }");
    CHECK(d.find("If (< n 1)\n") != std::string::npos);
    CHECK(d.find("Else\n          If (< n 2)") != std::string::npos);
    CHECK(d.find("Else\n              Block\n                Continue") != std::string::npos);

    d = stmts("for (var k: i32 = 0; k < 8; k += 1) { sync(); }");
    CHECK(d.find("Init\n          VarDecl k: i32 = 0") != std::string::npos);
    CHECK(d.find("Cond (< k 8)") != std::string::npos);
    CHECK(d.find("Step\n          Assign k = (+ k 1)") != std::string::npos);

    d = stmts("for (n = 0; n < 4; n += 1) { }");
    CHECK(d.find("Init\n          Assign n = 0") != std::string::npos);

    d = stmts("while (true) { } shared t: f32[16];");
    CHECK(d.find("Shared t: f32[16]") != std::string::npos);
}

TEST_CASE("parser: builtins can be parsed as assignment targets (sema rejects them)") {
    CHECK(stmts("thread_idx.x = 1;").find("Assign thread_idx.x = 1") != std::string::npos);
}

TEST_CASE("parser: error messages and locations") {
    CHECK(parseError("kernel k() {\n  var x: i32 = 1\n}") == "t.kite:3:1: error: expected ';' after statement\n");
    CHECK(parseError("kernel k() {\n  return;\n") == "t.kite:3:1: error: expected '}' before end of file\n");
    CHECK(parseError("kernel k() { var x: i32; }") == "t.kite:1:24: error: expected '=' (variables must be initialized)\n");
    CHECK(parseError("kernel k() { var x: f32 = 1.; }") == "t.kite:1:27: error: malformed float literal: digits required after '.'\n");
    CHECK(parseError("kernel k() { x = ; }") == "t.kite:1:18: error: expected an expression\n");
    CHECK(parseError("kernel k() { thread_idx.w; }") == "t.kite:1:25: error: expected axis 'x', 'y' or 'z'\n");
    CHECK(parseError("var x: i32 = 1;") == "t.kite:1:1: error: expected 'kernel'\n");
}

TEST_CASE("parser: empty program and multiple kernels") {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    auto p = parseSource("// nothing\n", diag);
    REQUIRE(p.has_value());
    CHECK(p->kernels.empty());
    p = parseSource("kernel a() {} kernel b(x: *i32, f: bool) {}", diag);
    REQUIRE(p.has_value());
    REQUIRE(p->kernels.size() == 2);
    CHECK(p->kernels[1].params[0].type == Type::PtrI32);
    CHECK(p->kernels[1].params[1].type == Type::Bool);
}
