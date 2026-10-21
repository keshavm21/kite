#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend/ast.h"
#include "support/diagnostics.h"

namespace kite {

// A stack of lexical frames. declare() fails only on a clash within the innermost frame,
// so shadowing an outer name is allowed while redeclaring in the same scope is not (E2).
class Scope {
public:
    void push() { frames_.emplace_back(); }
    void pop() { frames_.pop_back(); }
    bool declare(Symbol* s) { return frames_.back().emplace(s->name, s).second; }
    Symbol* lookup(const std::string& name) const {
        for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        return nullptr;
    }

private:
    std::vector<std::unordered_map<std::string, Symbol*>> frames_;
};

// Name resolution and type checking (LANGUAGE.md section 8, errors E1-E14). Annotates the
// AST in place: Expr::type on every expression, Symbol* on names, declarations and
// parameters, and Kernel::symbols. Reports every error it finds; an operand of type
// Type::Error silences further errors on the expressions built from it.
class Sema {
public:
    explicit Sema(Diagnostics& diag) : diag_(diag) {}
    void check(Program& program);

private:
    void checkKernel(Kernel& k);
    void checkBlock(BlockStmt& b);
    void checkStmt(Stmt& s, bool topLevel);
    void checkAssign(AssignStmt& a);
    void checkCondition(ExprPtr& cond);
    Type checkExpr(ExprPtr& e);
    Type checkExprImpl(ExprPtr& e);
    Type checkIndex(IndexExpr& e);
    Type checkBinary(BinaryExpr& e, Type l, Type r);
    Type checkCall(CallExpr& e);

    Symbol* declare(Symbol::Kind kind, const std::string& name, Type type, int index, SourceLoc loc);

    Diagnostics& diag_;
    Scope scope_;
    Kernel* kernel_ = nullptr;
    int loopDepth_ = 0;
    int localCount_ = 0;
    int sharedCount_ = 0;
    long long sharedBytes_ = 0;
};

}  // namespace kite
