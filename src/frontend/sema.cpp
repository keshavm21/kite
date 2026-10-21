#include "frontend/sema.h"

#include <unordered_set>

namespace kite {

namespace {

constexpr long long kMaxSharedBytes = 48 * 1024;
constexpr long long kIntMax = 2147483647LL;

bool isNumeric(Type t) { return t == Type::I32 || t == Type::F32; }
bool isArray(Type t) {
    return t == Type::PtrI32 || t == Type::PtrF32 || t == Type::SharedI32 || t == Type::SharedF32;
}
Type elementType(Type t) {
    return (t == Type::PtrI32 || t == Type::SharedI32) ? Type::I32 : Type::F32;
}

// Diagnostics for a whole expression point at its first token, not at its operator.
SourceLoc startLoc(const Expr& e) {
    switch (e.kind) {
        case ExprKind::Binary: return startLoc(*static_cast<const BinaryExpr&>(e).lhs);
        case ExprKind::Cast: return startLoc(*static_cast<const CastExpr&>(e).operand);
        default: return e.loc;
    }
}

std::string quoted(const std::string& s) { return "'" + s + "'"; }

}  // namespace

void Sema::check(Program& program) {
    std::unordered_set<std::string> kernelNames;
    for (Kernel& k : program.kernels) {
        if (isReservedName(k.name)) {
            diag_.error(k.nameLoc, quoted(k.name) + " is a reserved name");
        } else if (!kernelNames.insert(k.name).second) {
            diag_.error(k.nameLoc, "redefinition of " + quoted(k.name));
        }
        checkKernel(k);
    }
}

Symbol* Sema::declare(Symbol::Kind kind, const std::string& name, Type type, int index, SourceLoc loc) {
    kernel_->symbols.push_back(std::make_unique<Symbol>(Symbol{kind, name, type, index, loc}));
    Symbol* sym = kernel_->symbols.back().get();
    if (isReservedName(name)) {
        diag_.error(loc, quoted(name) + " is a reserved name");
    } else if (!scope_.declare(sym)) {
        diag_.error(loc, "redefinition of " + quoted(name));
    }
    return sym;
}

void Sema::checkKernel(Kernel& k) {
    kernel_ = &k;
    loopDepth_ = localCount_ = sharedCount_ = 0;
    sharedBytes_ = 0;
    scope_ = Scope();
    scope_.push();  // parameters and the body's top-level declarations share this frame
    int index = 0;
    for (Param& p : k.params) p.sym = declare(Symbol::Kind::Param, p.name, p.type, index++, p.loc);
    for (auto& s : k.body->stmts) checkStmt(*s, /*topLevel=*/true);
    scope_.pop();
    kernel_ = nullptr;
}

void Sema::checkBlock(BlockStmt& b) {
    scope_.push();
    for (auto& s : b.stmts) checkStmt(*s, /*topLevel=*/false);
    scope_.pop();
}

void Sema::checkStmt(Stmt& s, bool topLevel) {
    switch (s.kind) {
        case StmtKind::VarDecl: {
            auto& d = static_cast<VarDeclStmt&>(s);
            Type t = checkExpr(d.init);  // before declaring: `var x: i32 = x;` does not see itself
            if (t != Type::Error && t != d.type) {
                diag_.error(startLoc(*d.init), std::string("cannot assign ") + typeName(t) + " to " + typeName(d.type));
            }
            d.sym = declare(Symbol::Kind::Local, d.name, d.type, localCount_++, d.nameLoc);
            break;
        }
        case StmtKind::SharedDecl: {
            auto& d = static_cast<SharedDeclStmt&>(s);
            if (!topLevel) {
                diag_.error(d.loc, "shared arrays must be declared at kernel top level");
            }
            if (d.size <= 0) {
                diag_.error(d.loc, "shared array size must be positive");
            } else if (d.size > kMaxSharedBytes / 4 || sharedBytes_ + d.size * 4 > kMaxSharedBytes) {
                diag_.error(d.loc, "shared memory per kernel exceeds 48 KiB (49152 bytes)");
            } else {
                sharedBytes_ += d.size * 4;
            }
            d.sym = declare(Symbol::Kind::Shared, d.name, d.type, sharedCount_++, d.nameLoc);
            break;
        }
        case StmtKind::Assign:
            checkAssign(static_cast<AssignStmt&>(s));
            break;
        case StmtKind::If: {
            auto& i = static_cast<IfStmt&>(s);
            checkCondition(i.cond);
            checkBlock(*i.thenBlock);
            if (i.elseStmt) checkStmt(*i.elseStmt, false);
            break;
        }
        case StmtKind::While: {
            auto& w = static_cast<WhileStmt&>(s);
            checkCondition(w.cond);
            ++loopDepth_;
            checkBlock(*w.body);
            --loopDepth_;
            break;
        }
        case StmtKind::For: {
            auto& f = static_cast<ForStmt&>(s);
            scope_.push();  // the init variable lives for the loop only
            checkStmt(*f.init, false);
            checkCondition(f.cond);
            checkStmt(*f.step, false);
            ++loopDepth_;
            checkBlock(*f.body);
            --loopDepth_;
            scope_.pop();
            break;
        }
        case StmtKind::Break:
            if (loopDepth_ == 0) diag_.error(s.loc, "'break' outside of a loop");
            break;
        case StmtKind::Continue:
            if (loopDepth_ == 0) diag_.error(s.loc, "'continue' outside of a loop");
            break;
        case StmtKind::Return:
        case StmtKind::Sync:
            break;
        case StmtKind::Block:
            checkBlock(static_cast<BlockStmt&>(s));
            break;
    }
}

void Sema::checkCondition(ExprPtr& cond) {
    Type t = checkExpr(cond);
    if (t != Type::Error && t != Type::Bool) {
        diag_.error(startLoc(*cond), std::string("condition must be bool, found ") + typeName(t));
    }
}

void Sema::checkAssign(AssignStmt& a) {
    Type targetType = Type::Error;
    Expr& target = *a.target;
    switch (target.kind) {
        case ExprKind::Name: {
            auto& n = static_cast<NameExpr&>(target);
            Symbol* sym = scope_.lookup(n.name);
            if (!sym) {
                diag_.error(n.loc, "use of undeclared identifier " + quoted(n.name));
            } else if (sym->kind == Symbol::Kind::Param) {
                n.sym = sym;
                diag_.error(n.loc, "cannot assign to parameter " + quoted(n.name));
            } else if (sym->kind == Symbol::Kind::Shared) {
                n.sym = sym;
                diag_.error(n.loc, quoted(n.name) + " is an array; index it");
            } else {
                n.sym = sym;
                targetType = sym->type;
            }
            n.type = targetType;
            break;
        }
        case ExprKind::Index:
            targetType = checkExpr(a.target);
            break;
        case ExprKind::Builtin:
            diag_.error(target.loc, "cannot assign to builtin " + quoted(builtinName(static_cast<BuiltinExpr&>(target).var)));
            break;
        default:
            break;  // the parser only produces the three kinds above
    }
    Type valueType = checkExpr(a.value);
    if (targetType != Type::Error && valueType != Type::Error && targetType != valueType) {
        diag_.error(startLoc(*a.value),
                    std::string("cannot assign ") + typeName(valueType) + " to " + typeName(targetType));
    }
}

Type Sema::checkExpr(ExprPtr& e) {
    // `-2147483648` is the one place a literal may exceed INT_MAX: fold it into a single
    // negative literal so later stages never see the out-of-range positive value.
    if (e->kind == ExprKind::Unary) {
        auto& u = static_cast<UnaryExpr&>(*e);
        if (u.op == UnOp::Neg && u.operand->kind == ExprKind::IntLit &&
            static_cast<IntLitExpr&>(*u.operand).value == kIntMax + 1) {
            e = std::make_unique<IntLitExpr>(e->loc, -(kIntMax + 1));
        }
    }
    Type t = checkExprImpl(e);
    e->type = t;
    return t;
}

Type Sema::checkExprImpl(ExprPtr& ep) {
    Expr& e = *ep;
    switch (e.kind) {
        case ExprKind::IntLit: {
            long long v = static_cast<IntLitExpr&>(e).value;
            if (v > kIntMax) {
                diag_.error(e.loc, "integer literal too large for i32");
                return Type::Error;
            }
            return Type::I32;
        }
        case ExprKind::FloatLit: return Type::F32;
        case ExprKind::BoolLit: return Type::Bool;
        case ExprKind::Name: {
            auto& n = static_cast<NameExpr&>(e);
            Symbol* sym = scope_.lookup(n.name);
            if (!sym) {
                diag_.error(n.loc, "use of undeclared identifier " + quoted(n.name));
                return Type::Error;
            }
            n.sym = sym;
            if (isArray(sym->type)) {
                diag_.error(n.loc, quoted(n.name) + " is an array; index it");
                return Type::Error;
            }
            return sym->type;
        }
        case ExprKind::Index: return checkIndex(static_cast<IndexExpr&>(e));
        case ExprKind::Unary: {
            auto& u = static_cast<UnaryExpr&>(e);
            Type t = checkExpr(u.operand);
            if (t == Type::Error) return Type::Error;
            if (u.op == UnOp::Neg ? isNumeric(t) : t == Type::Bool) return t;
            diag_.error(u.loc, std::string("invalid operand to unary ") + (u.op == UnOp::Neg ? "'-'" : "'!'") + ": " + typeName(t));
            return Type::Error;
        }
        case ExprKind::Binary: {
            auto& b = static_cast<BinaryExpr&>(e);
            Type l = checkExpr(b.lhs);
            Type r = checkExpr(b.rhs);
            if (l == Type::Error || r == Type::Error) return Type::Error;
            return checkBinary(b, l, r);
        }
        case ExprKind::Cast: {
            auto& c = static_cast<CastExpr&>(e);
            Type from = checkExpr(c.operand);
            if (from == Type::Error) return Type::Error;
            bool ok = from == c.target || (from == Type::I32 && c.target == Type::F32) ||
                      (from == Type::F32 && c.target == Type::I32) || (from == Type::Bool && c.target == Type::I32);
            if (ok) return c.target;
            diag_.error(c.loc, std::string("cannot cast ") + typeName(from) + " to " + typeName(c.target));
            return Type::Error;
        }
        case ExprKind::Call: return checkCall(static_cast<CallExpr&>(e));
        case ExprKind::Builtin: return Type::I32;
    }
    return Type::Error;
}

Type Sema::checkIndex(IndexExpr& e) {
    Type result = Type::Error;
    Symbol* sym = scope_.lookup(e.name);
    if (!sym) {
        diag_.error(e.loc, "use of undeclared identifier " + quoted(e.name));
    } else {
        e.sym = sym;
        if (!isArray(sym->type)) {
            diag_.error(e.loc, "cannot index " + quoted(e.name) + " of type " + typeName(sym->type));
        } else {
            result = elementType(sym->type);
        }
    }
    Type idx = checkExpr(e.index);  // always checked, so its own errors are reported too
    if (idx != Type::Error && idx != Type::I32) {
        diag_.error(startLoc(*e.index), "index must be i32");
        return Type::Error;
    }
    return result;
}

Type Sema::checkBinary(BinaryExpr& e, Type l, Type r) {
    switch (e.op) {
        case BinOp::Add:
        case BinOp::Sub:
        case BinOp::Mul:
        case BinOp::Div:
            if (l == r && isNumeric(l)) return l;
            break;
        case BinOp::Rem:
            if (l == Type::I32 && r == Type::I32) return Type::I32;
            if (l == Type::F32 && r == Type::F32) {
                diag_.error(e.loc, "'%' requires i32 operands");
                return Type::Error;
            }
            break;
        case BinOp::Lt:
        case BinOp::Le:
        case BinOp::Gt:
        case BinOp::Ge:
            if (l == r && isNumeric(l)) return Type::Bool;
            break;
        case BinOp::Eq:
        case BinOp::Ne:
            if (l == r) return Type::Bool;
            break;
        case BinOp::And:
        case BinOp::Or:
            if (l == Type::Bool && r == Type::Bool) return Type::Bool;
            break;
    }
    diag_.error(e.loc, std::string("invalid operands to '") + binOpSymbol(e.op) + "': " + typeName(l) + " and " + typeName(r));
    return Type::Error;
}

Type Sema::checkCall(CallExpr& e) {
    std::vector<Type> types;
    for (auto& a : e.args) types.push_back(checkExpr(a));

    const char* name = intrinsicName(e.fn);
    bool binary = e.fn == Intrinsic::Min || e.fn == Intrinsic::Max;
    size_t arity = binary ? 2 : 1;
    auto bad = [&](const std::string& expects) {
        diag_.error(e.loc, quoted(name) + " expects " + expects);
        return Type::Error;
    };

    if (e.args.size() != arity) {
        switch (e.fn) {
            case Intrinsic::Sqrt:
            case Intrinsic::Exp: return bad("1 argument of type f32");
            case Intrinsic::Abs: return bad("1 argument of type i32 or f32");
            default: return bad("2 arguments of the same numeric type");
        }
    }
    for (Type t : types) {
        if (t == Type::Error) return Type::Error;
    }
    switch (e.fn) {
        case Intrinsic::Sqrt:
        case Intrinsic::Exp:
            return types[0] == Type::F32 ? Type::F32 : bad("1 argument of type f32");
        case Intrinsic::Abs:
            return isNumeric(types[0]) ? types[0] : bad("1 argument of type i32 or f32");
        default:
            return (types[0] == types[1] && isNumeric(types[0])) ? types[0] : bad("2 arguments of the same numeric type");
    }
}

}  // namespace kite
