#include <sstream>

#include "doctest/doctest.h"
#include "frontend/parser.h"
#include "frontend/sema.h"

using namespace kite;

namespace {

struct Checked {
    std::optional<Program> program;
    std::string errors;
};

Checked analyze(const std::string& src) {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    Checked c;
    c.program = parseSource(src, diag);
    REQUIRE_MESSAGE(c.program.has_value(), err.str());
    Sema(diag).check(*c.program);
    c.errors = err.str();
    return c;
}

// Errors for a kernel body with parameters (n: i32, f: f32, b: bool, p: *f32, q: *i32).
// The body starts on line 2, column 1, so a column is just an offset into `body`.
std::string errorsIn(const std::string& body) {
    return analyze("kernel k(n: i32, f: f32, b: bool, p: *f32, q: *i32) {\n" + body + "\n}").errors;
}

// The diagnostic expected for `msg` reported at the first occurrence of `needle` in `body`.
std::string errAt(const std::string& body, const std::string& needle, const std::string& msg) {
    size_t at = body.find(needle);
    REQUIRE_MESSAGE(at != std::string::npos, "needle not in body: " << needle);
    return "t.kite:2:" + std::to_string(at + 1) + ": error: " + msg + "\n";
}

// Checks that `body` produces exactly one error, `msg`, at `needle`.
void expectError(const std::string& body, const std::string& needle, const std::string& msg) {
    CHECK_MESSAGE(errorsIn(body) == errAt(body, needle, msg), body);
}

const Stmt& stmtAt(const Checked& c, size_t i) { return *c.program->kernels[0].body->stmts[i]; }

}  // namespace

TEST_CASE("sema: valid programs produce no errors") {
    CHECK(errorsIn("var x: i32 = n * 2 + 1; var y: f32 = x as f32 * f; p[x] = y; q[0] = x % 3;").empty());
    CHECK(errorsIn("var c: bool = n < 3 && f >= 1.0 || !b; var d: bool = c == b; var z: i32 = c as i32;").empty());
    CHECK(errorsIn("var s: f32 = sqrt(exp(abs(f))) + min(f, 1.0) + max(f, 2.0);").empty());
    CHECK(errorsIn("var m: i32 = abs(min(n, max(n, 3)));").empty());
    CHECK(errorsIn("while (b) { if (n == 0) { break; } else { continue; } }").empty());
    CHECK(errorsIn("var t: i32 = thread_idx.x + block_idx.y * block_dim.z - grid_dim.x;").empty());
}

TEST_CASE("sema: every error row (E1-E14) with line and column") {
    expectError("var x: i32 = foo;", "foo", "use of undeclared identifier 'foo'");                       // E1
    expectError("var x: i32 = 1; var x: i32 = 2;", "x: i32 = 2", "redefinition of 'x'");                // E2
    expectError("var x: i32 = n + f;", "+", "invalid operands to '+': i32 and f32");                     // E3
    expectError("var x: i32 = f;", "f;", "cannot assign f32 to i32");                                   // E4
    expectError("var x: f32 = 1.0; x = n;", "n;", "cannot assign i32 to f32");                          // E4
    expectError("if (n) { }", "n)", "condition must be bool, found i32");                               // E5
    expectError("while (f) { }", "f)", "condition must be bool, found f32");                            // E5
    expectError("var x: i32 = n[0];", "n[", "cannot index 'n' of type i32");                            // E6
    expectError("var x: f32 = p[f];", "f]", "index must be i32");                                       // E6
    expectError("n = 1;", "n =", "cannot assign to parameter 'n'");                                     // E7
    expectError("p = 1.0;", "p =", "cannot assign to parameter 'p'");                                   // E7 beats the type check
    expectError("block_dim.y = 1;", "block_dim", "cannot assign to builtin 'block_dim'");               // E7
    expectError("break;", "break", "'break' outside of a loop");                                        // E8
    expectError("continue;", "continue", "'continue' outside of a loop");                               // E8
    expectError("var x: bool = n as bool;", "as", "cannot cast i32 to bool");                           // E9
    expectError("var x: f32 = sqrt(n);", "sqrt", "'sqrt' expects 1 argument of type f32");              // E10
    expectError("var x: f32 = sqrt(f, f);", "sqrt", "'sqrt' expects 1 argument of type f32");           // E10
    expectError("var x: i32 = abs(b);", "abs", "'abs' expects 1 argument of type i32 or f32");          // E10
    expectError("var x: f32 = max(f);", "max", "'max' expects 2 arguments of the same numeric type");   // E10
    expectError("var x: f32 = min(f, n);", "min", "'min' expects 2 arguments of the same numeric type"); // E10
    expectError("var x: f32 = p + 1.0;", "p +", "'p' is an array; index it");                           // E11
    expectError("shared s: f32[4]; var x: f32 = s;", "s;", "'s' is an array; index it");                // E11
    expectError("shared s: f32[4]; s = 1.0;", "s =", "'s' is an array; index it");                      // E11
    expectError("if (b) { shared s: f32[4]; }", "shared", "shared arrays must be declared at kernel top level"); // E12
    expectError("shared s: i32[0];", "shared", "shared array size must be positive");                   // E12
    expectError("shared a: f32[8192]; shared c: f32[4097];", "shared c",
                "shared memory per kernel exceeds 48 KiB (49152 bytes)");                               // E12
    CHECK(errorsIn("shared a: f32[8192]; shared c: f32[4096];").empty());                               // exactly 48 KiB
    expectError("var x: i32 = 2147483648;", "2147483648", "integer literal too large for i32");         // E13
    expectError("var x: f32 = f % f;", "%", "'%' requires i32 operands");                               // E14
}

TEST_CASE("sema: operand rules") {
    CHECK(errorsIn("var x: bool = b + b;").find("invalid operands to '+': bool and bool") != std::string::npos);
    CHECK(errorsIn("var x: bool = n < b;").find("invalid operands to '<': i32 and bool") != std::string::npos);
    CHECK(errorsIn("var x: bool = n == f;").find("invalid operands to '==': i32 and f32") != std::string::npos);
    CHECK(errorsIn("var x: bool = n && b;").find("invalid operands to '&&': i32 and bool") != std::string::npos);
    CHECK(errorsIn("var x: i32 = !n;").find("invalid operand to unary '!': i32") != std::string::npos);
    CHECK(errorsIn("var x: bool = -b;").find("invalid operand to unary '-': bool") != std::string::npos);
    CHECK(errorsIn("var x: i32 = n % f;").find("invalid operands to '%': i32 and f32") != std::string::npos);
    CHECK(errorsIn("var x: bool = f as bool;").find("cannot cast f32 to bool") != std::string::npos);
    CHECK(errorsIn("var x: bool = b as f32 > 0.0;").find("cannot cast bool to f32") != std::string::npos);
    CHECK(errorsIn("var x: i32 = true as i32; var y: f32 = n as f32; var z: i32 = f as i32; var w: i32 = n as i32;").empty());
}

TEST_CASE("sema: all errors are reported, without cascades") {
    // Two independent mistakes: both reported, in source order.
    std::string two = "var a: i32 = foo; var c: i32 = f;";
    CHECK(errorsIn(two) == errAt(two, "foo", "use of undeclared identifier 'foo'") +
                               errAt(two, "f;", "cannot assign f32 to i32"));

    // An undeclared name poisons its expression silently: one error per use of `foo`.
    std::string poisoned = "var a: i32 = foo + 1; var c: f32 = foo * 2.0 + f;";
    CHECK(errorsIn(poisoned) == errAt(poisoned, "foo + 1", "use of undeclared identifier 'foo'") +
                                    errAt(poisoned, "foo * 2.0", "use of undeclared identifier 'foo'"));

    // A failed declaration still declares the name, so later uses do not add E1.
    expectError("var a: i32 = f; var c: i32 = a + 1;", "f;", "cannot assign f32 to i32");

    // Compound assignment desugars to a clone of the target; the shared error is reported once.
    expectError("p[foo] += 1.0;", "foo", "use of undeclared identifier 'foo'");
}

TEST_CASE("sema: scoping") {
    CHECK(errorsIn("var x: i32 = 1; if (b) { var x: f32 = 1.0; }").empty());                    // shadowing is fine
    CHECK(errorsIn("if (b) { var x: i32 = 1; } var y: i32 = x;").find("undeclared identifier 'x'") != std::string::npos);
    CHECK(errorsIn("for (var i: i32 = 0; i < 2; i += 1) { } for (var i: i32 = 0; i < 2; i += 1) { }").empty());
    CHECK(errorsIn("for (var i: i32 = 0; i < 2; i += 1) { } var y: i32 = i;").find("undeclared identifier 'i'") != std::string::npos);
    CHECK(errorsIn("var x: i32 = x;").find("undeclared identifier 'x'") != std::string::npos);  // initializer cannot see itself
    CHECK(errorsIn("var n: i32 = 1;").find("redefinition of 'n'") != std::string::npos);        // body shares the parameter scope
    CHECK(errorsIn("shared n: f32[4];").find("redefinition of 'n'") != std::string::npos);
    CHECK(errorsIn("break; while (b) { while (b) { break; } continue; }").find("'break' outside of a loop") != std::string::npos);
    CHECK(errorsIn("while (b) { { break; } }").empty());
}

TEST_CASE("sema: reserved names and duplicate kernels") {
    CHECK(errorsIn("var thread_idx: i32 = 1;").find("'thread_idx' is a reserved name") != std::string::npos);
    CHECK(analyze("kernel sqrt() {}").errors.find("'sqrt' is a reserved name") != std::string::npos);
    CHECK(analyze("kernel a(max: i32) {}").errors.find("'max' is a reserved name") != std::string::npos);
    CHECK(analyze("kernel a() {} kernel b() {} kernel a() {}").errors == "t.kite:1:36: error: redefinition of 'a'\n");
}

TEST_CASE("sema: annotates types and resolves symbols") {
    auto c = analyze("kernel k(n: i32, p: *f32) { var x: f32 = p[n] * 2.0; var n2: i32 = n; { var x: i32 = 1; n2 = x; } }");
    REQUIRE(c.errors.empty());
    Kernel& k = c.program->kernels[0];

    auto& x = static_cast<const VarDeclStmt&>(stmtAt(c, 0));
    CHECK(x.init->type == Type::F32);
    CHECK(x.sym->kind == Symbol::Kind::Local);
    CHECK(x.sym->index == 0);
    auto& mul = static_cast<const BinaryExpr&>(*x.init);
    auto& idx = static_cast<const IndexExpr&>(*mul.lhs);
    CHECK(idx.sym == k.params[1].sym);
    CHECK(idx.sym->kind == Symbol::Kind::Param);
    CHECK(static_cast<const NameExpr&>(*idx.index).sym == k.params[0].sym);
    CHECK(k.params[0].sym->index == 0);
    CHECK(k.params[1].sym->index == 1);

    // The inner `x` is a different Symbol from the outer one, and the use binds to the inner one.
    auto& inner = static_cast<const BlockStmt&>(stmtAt(c, 2));
    auto& innerDecl = static_cast<const VarDeclStmt&>(*inner.stmts[0]);
    auto& assign = static_cast<const AssignStmt&>(*inner.stmts[1]);
    CHECK(innerDecl.sym != x.sym);
    CHECK(static_cast<const NameExpr&>(*assign.value).sym == innerDecl.sym);
    CHECK(k.symbols.size() == 5);  // n, p, x, n2, inner x
}

TEST_CASE("sema: -2147483648 folds to one negative literal") {
    auto c = analyze("kernel k(q: *i32) { q[0] = -2147483648; q[1] = - -5; q[2] = -2147483647; }");
    REQUIRE(c.errors.empty());
    auto& a = static_cast<const AssignStmt&>(stmtAt(c, 0));
    REQUIRE(a.value->kind == ExprKind::IntLit);
    CHECK(static_cast<const IntLitExpr&>(*a.value).value == -2147483648LL);
    CHECK(a.value->type == Type::I32);
    auto& b = static_cast<const AssignStmt&>(stmtAt(c, 1));
    CHECK(b.value->kind == ExprKind::Unary);  // ordinary negations are untouched
    // Without the negation the same literal is an error.
    CHECK(analyze("kernel k(q: *i32) { q[0] = 2147483648; }").errors.find("integer literal too large") != std::string::npos);
}
