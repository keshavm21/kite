#include "frontend/ast.h"

namespace kite {

const char* typeName(Type t) {
    switch (t) {
        case Type::Error: return "<error>";
        case Type::Void: return "void";
        case Type::Bool: return "bool";
        case Type::I32: return "i32";
        case Type::F32: return "f32";
        case Type::PtrI32: return "*i32";
        case Type::PtrF32: return "*f32";
        case Type::SharedI32: return "i32[]";
        case Type::SharedF32: return "f32[]";
    }
    return "?";
}

const char* binOpSymbol(BinOp op) {
    switch (op) {
        case BinOp::Add: return "+";
        case BinOp::Sub: return "-";
        case BinOp::Mul: return "*";
        case BinOp::Div: return "/";
        case BinOp::Rem: return "%";
        case BinOp::Lt: return "<";
        case BinOp::Le: return "<=";
        case BinOp::Gt: return ">";
        case BinOp::Ge: return ">=";
        case BinOp::Eq: return "==";
        case BinOp::Ne: return "!=";
        case BinOp::And: return "&&";
        case BinOp::Or: return "||";
    }
    return "?";
}

const char* builtinName(BuiltinVar v) {
    switch (v) {
        case BuiltinVar::ThreadIdx: return "thread_idx";
        case BuiltinVar::BlockIdx: return "block_idx";
        case BuiltinVar::BlockDim: return "block_dim";
        case BuiltinVar::GridDim: return "grid_dim";
    }
    return "?";
}

const char* intrinsicName(Intrinsic f) {
    switch (f) {
        case Intrinsic::Sqrt: return "sqrt";
        case Intrinsic::Exp: return "exp";
        case Intrinsic::Abs: return "abs";
        case Intrinsic::Min: return "min";
        case Intrinsic::Max: return "max";
    }
    return "?";
}

ExprPtr cloneExpr(const Expr& e) {
    ExprPtr copy;
    switch (e.kind) {
        case ExprKind::IntLit:
            copy = std::make_unique<IntLitExpr>(e.loc, static_cast<const IntLitExpr&>(e).value);
            break;
        case ExprKind::FloatLit:
            copy = std::make_unique<FloatLitExpr>(e.loc, static_cast<const FloatLitExpr&>(e).value);
            break;
        case ExprKind::BoolLit:
            copy = std::make_unique<BoolLitExpr>(e.loc, static_cast<const BoolLitExpr&>(e).value);
            break;
        case ExprKind::Name:
            copy = std::make_unique<NameExpr>(e.loc, static_cast<const NameExpr&>(e).name);
            break;
        case ExprKind::Index: {
            auto& x = static_cast<const IndexExpr&>(e);
            copy = std::make_unique<IndexExpr>(e.loc, x.name, cloneExpr(*x.index));
            break;
        }
        case ExprKind::Unary: {
            auto& x = static_cast<const UnaryExpr&>(e);
            copy = std::make_unique<UnaryExpr>(e.loc, x.op, cloneExpr(*x.operand));
            break;
        }
        case ExprKind::Binary: {
            auto& x = static_cast<const BinaryExpr&>(e);
            copy = std::make_unique<BinaryExpr>(e.loc, x.op, cloneExpr(*x.lhs), cloneExpr(*x.rhs));
            break;
        }
        case ExprKind::Cast: {
            auto& x = static_cast<const CastExpr&>(e);
            copy = std::make_unique<CastExpr>(e.loc, cloneExpr(*x.operand), x.target);
            break;
        }
        case ExprKind::Call: {
            auto& x = static_cast<const CallExpr&>(e);
            std::vector<ExprPtr> args;
            for (auto& a : x.args) args.push_back(cloneExpr(*a));
            copy = std::make_unique<CallExpr>(e.loc, x.fn, std::move(args));
            break;
        }
        case ExprKind::Builtin: {
            auto& x = static_cast<const BuiltinExpr&>(e);
            copy = std::make_unique<BuiltinExpr>(e.loc, x.var, x.axis);
            break;
        }
    }
    copy->type = e.type;
    return copy;
}

}  // namespace kite
