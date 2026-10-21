#include "frontend/ast_printer.h"

#include <cstdio>
#include <cstring>

namespace kite {

std::string exprToString(const Expr& e) {
    switch (e.kind) {
        case ExprKind::IntLit:
            return std::to_string(static_cast<const IntLitExpr&>(e).value);
        case ExprKind::FloatLit: {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(static_cast<const FloatLitExpr&>(e).value));
            std::string s = buf;
            if (s.find_first_of(".eEni") == std::string::npos) s += ".0";  // keep floats visibly floats
            return s;
        }
        case ExprKind::BoolLit:
            return static_cast<const BoolLitExpr&>(e).value ? "true" : "false";
        case ExprKind::Name:
            return static_cast<const NameExpr&>(e).name;
        case ExprKind::Index: {
            auto& x = static_cast<const IndexExpr&>(e);
            return x.name + "[" + exprToString(*x.index) + "]";
        }
        case ExprKind::Unary: {
            auto& x = static_cast<const UnaryExpr&>(e);
            return std::string("(") + (x.op == UnOp::Neg ? "neg " : "not ") + exprToString(*x.operand) + ")";
        }
        case ExprKind::Binary: {
            auto& x = static_cast<const BinaryExpr&>(e);
            return std::string("(") + binOpSymbol(x.op) + " " + exprToString(*x.lhs) + " " + exprToString(*x.rhs) + ")";
        }
        case ExprKind::Cast: {
            auto& x = static_cast<const CastExpr&>(e);
            return "(as " + exprToString(*x.operand) + " " + typeName(x.target) + ")";
        }
        case ExprKind::Call: {
            auto& x = static_cast<const CallExpr&>(e);
            std::string s = std::string("(") + intrinsicName(x.fn);
            for (auto& a : x.args) s += " " + exprToString(*a);
            return s + ")";
        }
        case ExprKind::Builtin: {
            auto& x = static_cast<const BuiltinExpr&>(e);
            return std::string(builtinName(x.var)) + "." + "xyz"[x.axis];
        }
    }
    return "?";
}

namespace {

class Printer {
public:
    std::string out;

    void line(int depth, const std::string& text) {
        out.append(static_cast<size_t>(depth) * 2, ' ');
        out += text;
        out += '\n';
    }

    void stmt(const Stmt& s, int d) {
        switch (s.kind) {
            case StmtKind::VarDecl: {
                auto& x = static_cast<const VarDeclStmt&>(s);
                line(d, "VarDecl " + x.name + ": " + typeName(x.type) + " = " + exprToString(*x.init));
                break;
            }
            case StmtKind::SharedDecl: {
                auto& x = static_cast<const SharedDeclStmt&>(s);
                const char* elem = x.type == Type::SharedF32 ? "f32" : "i32";
                line(d, "Shared " + x.name + ": " + elem + "[" + std::to_string(x.size) + "]");
                break;
            }
            case StmtKind::Assign: {
                auto& x = static_cast<const AssignStmt&>(s);
                line(d, "Assign " + exprToString(*x.target) + " = " + exprToString(*x.value));
                break;
            }
            case StmtKind::If: {
                auto& x = static_cast<const IfStmt&>(s);
                line(d, "If " + exprToString(*x.cond));
                stmt(*x.thenBlock, d + 1);
                if (x.elseStmt) {
                    line(d + 1, "Else");
                    stmt(*x.elseStmt, d + 2);
                }
                break;
            }
            case StmtKind::While: {
                auto& x = static_cast<const WhileStmt&>(s);
                line(d, "While " + exprToString(*x.cond));
                stmt(*x.body, d + 1);
                break;
            }
            case StmtKind::For: {
                auto& x = static_cast<const ForStmt&>(s);
                line(d, "For");
                line(d + 1, "Init");
                stmt(*x.init, d + 2);
                line(d + 1, "Cond " + exprToString(*x.cond));
                line(d + 1, "Step");
                stmt(*x.step, d + 2);
                stmt(*x.body, d + 1);
                break;
            }
            case StmtKind::Break: line(d, "Break"); break;
            case StmtKind::Continue: line(d, "Continue"); break;
            case StmtKind::Return: line(d, "Return"); break;
            case StmtKind::Sync: line(d, "Sync"); break;
            case StmtKind::Block: {
                line(d, "Block");
                for (auto& child : static_cast<const BlockStmt&>(s).stmts) stmt(*child, d + 1);
                break;
            }
        }
    }
};

}  // namespace

std::string dumpProgram(const Program& program) {
    Printer p;
    p.line(0, "Program");
    for (const Kernel& k : program.kernels) {
        p.line(1, "Kernel " + k.name);
        for (const Param& param : k.params) {
            p.line(2, "Param " + param.name + ": " + typeName(param.type));
        }
        p.stmt(*k.body, 2);
    }
    return p.out;
}

}  // namespace kite
