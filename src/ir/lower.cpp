#include "ir/lower.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>

#include "analysis/cfg.h"
#include "ir/builder.h"

namespace kite::ir {

namespace {

Ty toTy(Type t) {
    switch (t) {
        case Type::Bool: return Ty::Bool;
        case Type::I32: case Type::SharedI32: return Ty::I32;
        case Type::F32: case Type::SharedF32: return Ty::F32;
        case Type::PtrI32: return Ty::PtrI32;
        case Type::PtrF32: return Ty::PtrF32;
        default: return Ty::Void;
    }
}

Op arithOp(BinOp op) {
    switch (op) {
        case BinOp::Add: return Op::Add;
        case BinOp::Sub: return Op::Sub;
        case BinOp::Mul: return Op::Mul;
        case BinOp::Div: return Op::Div;
        case BinOp::Rem: return Op::Rem;
        case BinOp::Lt: return Op::Lt;
        case BinOp::Le: return Op::Le;
        case BinOp::Gt: return Op::Gt;
        case BinOp::Ge: return Op::Ge;
        case BinOp::Eq: return Op::Eq;
        case BinOp::Ne: return Op::Ne;
        default: break;
    }
    assert(false && "&& and || are lowered to branches");
    return Op::Copy;
}

SReg specialReg(BuiltinVar v) {
    switch (v) {
        case BuiltinVar::ThreadIdx: return SReg::Tid;
        case BuiltinVar::BlockIdx: return SReg::Ctaid;
        case BuiltinVar::BlockDim: return SReg::Ntid;
        case BuiltinVar::GridDim: return SReg::Nctaid;
    }
    return SReg::Tid;
}

class Lowerer {
public:
    explicit Lowerer(const Kernel& k) : kernel_(k), b_(fn_) {}

    Function run() {
        fn_.name = kernel_.name;
        for (const Param& p : kernel_.params) vars_[p.sym] = Operand::r(fn_.newReg(toTy(p.type), p.name));
        fn_.numParams = static_cast<int>(kernel_.params.size());

        b_.setBlock(fn_.newBlock("entry"));
        lowerBlock(*kernel_.body);
        if (!b_.terminated()) b_.emitRet();

        // Blocks were stored in creation order; reorder them to program order so the IR reads top to bottom.
        std::vector<std::unique_ptr<BasicBlock>> ordered;
        for (BasicBlock* bb : b_.placed()) {
            auto it = std::find_if(fn_.blocks.begin(), fn_.blocks.end(), [&](auto& p) { return p.get() == bb; });
            ordered.push_back(std::move(*it));
        }
        assert(ordered.size() == fn_.blocks.size() && "every block must be placed");
        fn_.blocks = std::move(ordered);
        cfg::rebuild(fn_);
        return std::move(fn_);
    }

private:
    // ---- statements ----
    void lowerBlock(const BlockStmt& block) {
        for (const StmtPtr& s : block.stmts) lowerStmt(*s);
    }

    void lowerStmt(const Stmt& s) {
        switch (s.kind) {
            case StmtKind::VarDecl: {
                auto& d = static_cast<const VarDeclStmt&>(s);
                Operand init = lowerExpr(*d.init);  // before the register exists: `var x = x` cannot see itself
                Operand reg = Operand::r(fn_.newReg(toTy(d.type), d.name));
                b_.emitCopyTo(reg.reg, toTy(d.type), init);
                vars_[d.sym] = reg;
                break;
            }
            case StmtKind::SharedDecl: {
                auto& d = static_cast<const SharedDeclStmt&>(s);
                fn_.shared.push_back({d.name, toTy(d.type), static_cast<int>(d.size)});
                vars_[d.sym] = Operand::shared(static_cast<int>(fn_.shared.size()) - 1);
                break;
            }
            case StmtKind::Assign:
                lowerAssign(static_cast<const AssignStmt&>(s));
                break;
            case StmtKind::If:
                lowerIf(static_cast<const IfStmt&>(s));
                break;
            case StmtKind::While: {
                auto& w = static_cast<const WhileStmt&>(s);
                BasicBlock* cond = fn_.newBlock("while.cond");
                BasicBlock* body = fn_.newBlock("while.body");
                BasicBlock* end = fn_.newBlock("while.end");
                b_.emitJmp(cond);
                b_.setBlock(cond);
                lowerCond(*w.cond, body, end);
                b_.setBlock(body);
                loops_.push_back({end, cond});
                lowerBlock(*w.body);
                loops_.pop_back();
                if (!b_.terminated()) b_.emitJmp(cond);
                b_.setBlock(end);
                break;
            }
            case StmtKind::For: {
                auto& f = static_cast<const ForStmt&>(s);
                lowerStmt(*f.init);
                BasicBlock* cond = fn_.newBlock("for.cond");
                BasicBlock* body = fn_.newBlock("for.body");
                BasicBlock* step = fn_.newBlock("for.step");
                BasicBlock* end = fn_.newBlock("for.end");
                b_.emitJmp(cond);
                b_.setBlock(cond);
                lowerCond(*f.cond, body, end);
                b_.setBlock(body);
                loops_.push_back({end, step});  // `continue` runs the step
                lowerBlock(*f.body);
                loops_.pop_back();
                if (!b_.terminated()) b_.emitJmp(step);
                b_.setBlock(step);
                lowerStmt(*f.step);
                b_.emitJmp(cond);
                b_.setBlock(end);
                break;
            }
            case StmtKind::Break:
                b_.emitJmp(loops_.back().breakTo);
                break;
            case StmtKind::Continue:
                b_.emitJmp(loops_.back().continueTo);
                break;
            case StmtKind::Return:
                b_.emitRet();
                break;
            case StmtKind::Sync:
                b_.emitBarrier();
                break;
            case StmtKind::Block:
                lowerBlock(static_cast<const BlockStmt&>(s));
                break;
        }
    }

    void lowerAssign(const AssignStmt& a) {
        if (a.target->kind == ExprKind::Name) {
            auto& n = static_cast<const NameExpr&>(*a.target);
            Operand value = lowerExpr(*a.value);
            b_.emitCopyTo(vars_.at(n.sym).reg, toTy(n.type), value);
            return;
        }
        auto& ix = static_cast<const IndexExpr&>(*a.target);
        Operand index = lowerExpr(*ix.index);
        Operand value = lowerExpr(*a.value);
        b_.emitStore(toTy(ix.type), spaceOf(*ix.sym), vars_.at(ix.sym), index, value);
    }

    void lowerIf(const IfStmt& s) {
        BasicBlock* thenB = fn_.newBlock("if.then");
        BasicBlock* elseB = s.elseStmt ? fn_.newBlock("if.else") : nullptr;
        BasicBlock* end = fn_.newBlock("if.end");
        lowerCond(*s.cond, thenB, elseB ? elseB : end);
        b_.setBlock(thenB);
        lowerBlock(*s.thenBlock);
        if (!b_.terminated()) b_.emitJmp(end);
        if (elseB) {
            b_.setBlock(elseB);
            lowerStmt(*s.elseStmt);
            if (!b_.terminated()) b_.emitJmp(end);
        }
        b_.setBlock(end);
    }

    // ---- expressions ----
    static Space spaceOf(const Symbol& sym) { return sym.kind == Symbol::Kind::Shared ? Space::Shared : Space::Global; }

    // Branches to `t` if the bool expression is true and to `f` otherwise. && and || short-circuit
    // by splitting blocks; everything else evaluates to a bool and ends in one `br`.
    void lowerCond(const Expr& e, BasicBlock* t, BasicBlock* f) {
        if (e.kind == ExprKind::Binary) {
            auto& bin = static_cast<const BinaryExpr&>(e);
            if (bin.op == BinOp::And) {
                BasicBlock* rhs = fn_.newBlock("and.rhs");
                lowerCond(*bin.lhs, rhs, f);
                b_.setBlock(rhs);
                lowerCond(*bin.rhs, t, f);
                return;
            }
            if (bin.op == BinOp::Or) {
                BasicBlock* rhs = fn_.newBlock("or.rhs");
                lowerCond(*bin.lhs, t, rhs);
                b_.setBlock(rhs);
                lowerCond(*bin.rhs, t, f);
                return;
            }
        } else if (e.kind == ExprKind::Unary && static_cast<const UnaryExpr&>(e).op == UnOp::Not) {
            lowerCond(*static_cast<const UnaryExpr&>(e).operand, f, t);
            return;
        }
        b_.emitBr(lowerExpr(e), t, f);
    }

    // && or || used as a value (`var f: bool = a || b`): materialise through branches.
    Operand lowerShortCircuit(const Expr& e) {
        int r = fn_.newReg(Ty::Bool);
        BasicBlock* tb = fn_.newBlock("sc.true");
        BasicBlock* fb = fn_.newBlock("sc.false");
        BasicBlock* join = fn_.newBlock("sc.end");
        lowerCond(e, tb, fb);
        b_.setBlock(tb);
        b_.emitCopyTo(r, Ty::Bool, Operand::boolean(true));
        b_.emitJmp(join);
        b_.setBlock(fb);
        b_.emitCopyTo(r, Ty::Bool, Operand::boolean(false));
        b_.emitJmp(join);
        b_.setBlock(join);
        return Operand::r(r);
    }

    Operand lowerExpr(const Expr& e) {
        switch (e.kind) {
            case ExprKind::IntLit:
                return Operand::i32(static_cast<int32_t>(static_cast<const IntLitExpr&>(e).value));
            case ExprKind::FloatLit:
                return Operand::f32(static_cast<const FloatLitExpr&>(e).value);
            case ExprKind::BoolLit:
                return Operand::boolean(static_cast<const BoolLitExpr&>(e).value);
            case ExprKind::Name:
                return vars_.at(static_cast<const NameExpr&>(e).sym);
            case ExprKind::Index: {
                auto& ix = static_cast<const IndexExpr&>(e);
                Operand index = lowerExpr(*ix.index);
                return b_.emitLoad(toTy(ix.type), spaceOf(*ix.sym), vars_.at(ix.sym), index);
            }
            case ExprKind::Unary: {
                auto& u = static_cast<const UnaryExpr&>(e);
                Operand v = lowerExpr(*u.operand);
                return b_.emitUnary(u.op == UnOp::Neg ? Op::Neg : Op::Not, toTy(e.type), v);
            }
            case ExprKind::Binary: {
                auto& bin = static_cast<const BinaryExpr&>(e);
                if (bin.op == BinOp::And || bin.op == BinOp::Or) return lowerShortCircuit(e);
                Operand l = lowerExpr(*bin.lhs);
                Operand r = lowerExpr(*bin.rhs);
                return b_.emitBinary(arithOp(bin.op), toTy(bin.lhs->type), l, r);  // compares carry the operand type
            }
            case ExprKind::Cast: {
                auto& c = static_cast<const CastExpr&>(e);
                Operand v = lowerExpr(*c.operand);
                Type from = c.operand->type;
                if (from == c.target) return v;
                if (from == Type::I32 && c.target == Type::F32) return b_.emitCast(Op::IToF, v);
                if (from == Type::F32 && c.target == Type::I32) return b_.emitCast(Op::FToI, v);
                assert(from == Type::Bool && c.target == Type::I32);
                return b_.emitCast(Op::BToI, v);
            }
            case ExprKind::Call: {
                auto& c = static_cast<const CallExpr&>(e);
                Operand a = lowerExpr(*c.args[0]);
                switch (c.fn) {
                    case Intrinsic::Sqrt: return b_.emitUnary(Op::Sqrt, Ty::F32, a);
                    case Intrinsic::Exp: return b_.emitUnary(Op::Exp, Ty::F32, a);
                    case Intrinsic::Abs: return b_.emitUnary(Op::Abs, toTy(e.type), a);
                    case Intrinsic::Min: return b_.emitBinary(Op::Min, toTy(e.type), a, lowerExpr(*c.args[1]));
                    case Intrinsic::Max: return b_.emitBinary(Op::Max, toTy(e.type), a, lowerExpr(*c.args[1]));
                }
                break;
            }
            case ExprKind::Builtin: {
                auto& bi = static_cast<const BuiltinExpr&>(e);
                return b_.emitSpecial(specialReg(bi.var), bi.axis);
            }
        }
        assert(false && "unhandled expression");
        return Operand::undef();
    }

    struct Loop {
        BasicBlock* breakTo;
        BasicBlock* continueTo;
    };

    const Kernel& kernel_;
    Function fn_;
    Builder b_;
    std::unordered_map<const Symbol*, Operand> vars_;  // var/param -> register, shared array -> @array
    std::vector<Loop> loops_;
};

}  // namespace

Function lowerKernel(const Kernel& kernel) { return Lowerer(kernel).run(); }

Module lowerProgram(const Program& program) {
    Module m;
    for (const Kernel& k : program.kernels) m.kernels.push_back(lowerKernel(k));
    return m;
}

}  // namespace kite::ir
