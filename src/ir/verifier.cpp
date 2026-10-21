#include "ir/verifier.h"

#include <algorithm>
#include <unordered_set>

#include "ir/printer.h"

namespace kite::ir {

namespace {

bool isNumeric(Ty t) { return t == Ty::I32 || t == Ty::F32; }
bool isScalar(Ty t) { return t == Ty::Bool || isNumeric(t); }

class Verifier {
public:
    Verifier(const Function& fn, Form form) : fn_(fn), form_(form) {
        for (const auto& bb : fn.blocks) blocks_.insert(bb.get());
    }

    std::vector<std::string> run() {
        if (fn_.blocks.empty()) {
            problem("function has no blocks");
            return errors_;
        }
        if (fn_.regHint.size() != fn_.regTy.size()) problem("regTy and regHint have different sizes");
        if (fn_.numParams > fn_.numRegs()) problem("more parameters than registers");
        for (int p = 0; p < fn_.numParams && p < fn_.numRegs(); ++p) {
            Ty t = fn_.regTy[p];
            if (!isScalar(t) && !isPointer(t)) problem("parameter " + fn_.regName(p) + " has type " + tyName(t));
        }
        for (const SharedArray& s : fn_.shared) {
            if (!isNumeric(s.elemTy) || s.size <= 0) problem("shared array @" + s.name + " is malformed");
        }
        for (const auto& bb : fn_.blocks) checkBlock(*bb);
        checkEdges();
        return errors_;
    }

private:
    void problem(const std::string& msg) { errors_.push_back("verifier: kernel " + fn_.name + ": " + msg); }
    void problem(const BasicBlock& bb, int inst, const std::string& msg) {
        errors_.push_back("verifier: kernel " + fn_.name + ", block " + bb.name() + ", inst " + std::to_string(inst) + ": " + msg);
    }

    // Type of an operand, or Void when it has none (undef, shared arrays) or is invalid.
    Ty typeOf(const Operand& o) const {
        switch (o.kind) {
            case Operand::Reg: return o.reg >= 0 && o.reg < fn_.numRegs() ? fn_.regTy[o.reg] : Ty::Void;
            case Operand::ImmI32: return Ty::I32;
            case Operand::ImmF32: return Ty::F32;
            case Operand::ImmBool: return Ty::Bool;
            default: return Ty::Void;
        }
    }

    // Checks operand `k` of the current instruction has type `want`.
    void expectArg(const BasicBlock& bb, int n, const Inst& in, size_t k, Ty want) {
        const Operand& o = in.args[k];
        if (o.kind == Operand::Reg && (o.reg < 0 || o.reg >= fn_.numRegs())) {
            problem(bb, n, "operand " + std::to_string(k) + " uses nonexistent register " + std::to_string(o.reg));
        } else if (o.kind == Operand::Undef || o.kind == Operand::SharedArr) {
            problem(bb, n, "operand " + std::to_string(k) + " must be a value of type " + tyName(want));
        } else if (typeOf(o) != want) {
            problem(bb, n, "operand " + std::to_string(k) + " (" + toString(fn_, o) + ") has type " + tyName(typeOf(o)) +
                               ", expected " + tyName(want));
        }
    }

    void checkBlock(const BasicBlock& bb) {
        if (bb.insts.empty()) {
            problem("block " + bb.name() + " is empty");
            return;
        }
        for (size_t k = 0; k < bb.insts.size(); ++k) {
            const Inst& in = bb.insts[k];
            int n = static_cast<int>(k);
            bool last = k + 1 == bb.insts.size();
            if (in.isTerminator() && !last) problem(bb, n, "terminator before the end of the block");
            if (last && !in.isTerminator()) problem(bb, n, "block does not end with a terminator");
            checkInst(bb, n, in);
        }
    }

    void checkInst(const BasicBlock& bb, int n, const Inst& in) {
        // Shape: operand count, destination.
        size_t wantArgs = 0;
        switch (in.op) {
            case Op::Copy: case Op::Neg: case Op::Not: case Op::IToF: case Op::FToI: case Op::BToI:
            case Op::Sqrt: case Op::Exp: case Op::Abs: case Op::Br:
                wantArgs = 1;
                break;
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Rem: case Op::Shl:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            case Op::And: case Op::Or: case Op::Xor: case Op::Min: case Op::Max: case Op::Load:
                wantArgs = 2;
                break;
            case Op::Store: wantArgs = 3; break;
            case Op::Special: case Op::Barrier: case Op::Jmp: case Op::Ret: wantArgs = 0; break;
            case Op::Phi: wantArgs = in.args.size(); break;
        }
        if (in.args.size() != wantArgs) {
            problem(bb, n, std::string(opName(in.op)) + " takes " + std::to_string(wantArgs) + " operands, has " +
                               std::to_string(in.args.size()));
            return;
        }
        Ty result = in.resultType();
        if (result == Ty::Void) {
            if (in.dst != -1) problem(bb, n, std::string(opName(in.op)) + " must not have a destination");
        } else if (in.dst < 0 || in.dst >= fn_.numRegs()) {
            problem(bb, n, "missing or nonexistent destination register");
        } else if (fn_.regTy[in.dst] != result) {
            problem(bb, n, "destination " + fn_.regName(in.dst) + " has type " + tyName(fn_.regTy[in.dst]) + ", result is " +
                               tyName(result));
        }
        if (in.op != Op::Phi) {
            for (size_t k = 0; k < in.args.size(); ++k) {
                if (in.args[k].kind == Operand::Undef) problem(bb, n, "undef is only allowed as a phi operand");
            }
        }

        // Types.
        Ty t = in.ty;
        auto needTy = [&](bool ok) {
            if (!ok) problem(bb, n, std::string(opName(in.op)) + " does not support type " + tyName(t));
            return ok;
        };
        switch (in.op) {
            case Op::Copy:
                if (needTy(isScalar(t))) expectArg(bb, n, in, 0, t);
                break;
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Min: case Op::Max:
                if (needTy(isNumeric(t))) { expectArg(bb, n, in, 0, t); expectArg(bb, n, in, 1, t); }
                break;
            case Op::Rem: case Op::Shl:
                if (needTy(t == Ty::I32)) { expectArg(bb, n, in, 0, t); expectArg(bb, n, in, 1, t); }
                break;
            case Op::Neg: case Op::Abs:
                if (needTy(isNumeric(t))) expectArg(bb, n, in, 0, t);
                break;
            case Op::Sqrt: case Op::Exp:
                if (needTy(t == Ty::F32)) expectArg(bb, n, in, 0, t);
                break;
            case Op::Eq: case Op::Ne:
                if (needTy(isScalar(t))) { expectArg(bb, n, in, 0, t); expectArg(bb, n, in, 1, t); }
                break;
            case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
                if (needTy(isNumeric(t))) { expectArg(bb, n, in, 0, t); expectArg(bb, n, in, 1, t); }
                break;
            case Op::And: case Op::Or: case Op::Xor:
                if (needTy(t == Ty::Bool)) { expectArg(bb, n, in, 0, t); expectArg(bb, n, in, 1, t); }
                break;
            case Op::Not:
                if (needTy(t == Ty::Bool)) expectArg(bb, n, in, 0, t);
                break;
            case Op::IToF: expectArg(bb, n, in, 0, Ty::I32); break;
            case Op::FToI: expectArg(bb, n, in, 0, Ty::F32); break;
            case Op::BToI: expectArg(bb, n, in, 0, Ty::Bool); break;
            case Op::Special:
                if (in.dim < 0 || in.dim > 2) problem(bb, n, "special dimension must be 0, 1 or 2");
                break;
            case Op::Phi:
                if (form_ != Form::SSA) problem(bb, n, "phi outside SSA form");
                if (needTy(isScalar(t))) {
                    for (size_t k = 0; k < in.args.size(); ++k) {
                        if (in.args[k].kind != Operand::Undef) expectArg(bb, n, in, k, t);
                    }
                }
                if (in.blocks.size() != in.args.size()) problem(bb, n, "phi has different numbers of values and blocks");
                for (const BasicBlock* from : in.blocks) {
                    if (!blocks_.count(from)) problem(bb, n, "phi refers to a block outside this function");
                }
                break;
            case Op::Load: case Op::Store:
                checkMemory(bb, n, in);
                break;
            case Op::Br:
                expectArg(bb, n, in, 0, Ty::Bool);
                checkTargets(bb, n, in, 2);
                break;
            case Op::Jmp:
                checkTargets(bb, n, in, 1);
                break;
            case Op::Barrier: case Op::Ret:
                break;
        }
    }

    void checkMemory(const BasicBlock& bb, int n, const Inst& in) {
        bool isLoad = in.op == Op::Load;
        Ty t = in.ty;
        if (t != Ty::I32 && t != Ty::F32) {
            problem(bb, n, std::string(opName(in.op)) + " does not support type " + tyName(t));
            return;
        }
        const Operand& base = in.args[0];
        if (in.space == Space::Shared) {
            if (base.kind != Operand::SharedArr || base.i < 0 || base.i >= static_cast<int>(fn_.shared.size())) {
                problem(bb, n, "shared access needs the base of a declared shared array");
            } else if (fn_.shared[base.i].elemTy != t) {
                problem(bb, n, std::string("shared array @") + fn_.shared[base.i].name + " holds " +
                                   tyName(fn_.shared[base.i].elemTy) + ", accessed as " + tyName(t));
            }
        } else {
            Ty want = t == Ty::F32 ? Ty::PtrF32 : Ty::PtrI32;
            if (base.kind != Operand::Reg) problem(bb, n, "global access needs a pointer register as its base");
            else expectArg(bb, n, in, 0, want);
        }
        expectArg(bb, n, in, 1, Ty::I32);
        if (!isLoad) expectArg(bb, n, in, 2, t);
    }

    void checkTargets(const BasicBlock& bb, int n, const Inst& in, size_t want) {
        if (in.blocks.size() != want) {
            problem(bb, n, std::string(opName(in.op)) + " needs " + std::to_string(want) + " targets");
            return;
        }
        for (const BasicBlock* target : in.blocks) {
            if (!target || !blocks_.count(target)) problem(bb, n, "branch target is not a block of this function");
        }
    }

    // Invariants 2 and 3: preds/succs mirror the terminators, and the entry has no predecessors.
    void checkEdges() {
        std::vector<std::pair<const BasicBlock*, const BasicBlock*>> want, haveSucc, havePred;
        for (const auto& bb : fn_.blocks) {
            if (bb->hasTerminator()) {
                for (const BasicBlock* t : bb->terminator().blocks) want.push_back({bb.get(), t});
            }
            for (const BasicBlock* s : bb->succs) haveSucc.push_back({bb.get(), s});
            for (const BasicBlock* p : bb->preds) havePred.push_back({p, bb.get()});
        }
        std::sort(want.begin(), want.end());
        std::sort(haveSucc.begin(), haveSucc.end());
        std::sort(havePred.begin(), havePred.end());
        if (want != haveSucc) problem("succs do not match the terminators (run cfg::rebuild)");
        if (want != havePred) problem("preds do not match the terminators (run cfg::rebuild)");
        const BasicBlock* entry = fn_.blocks[0].get();
        for (const auto& bb : fn_.blocks) {
            if (!bb->hasTerminator()) continue;
            for (const BasicBlock* t : bb->terminator().blocks) {
                if (t == entry) problem("the entry block has a predecessor (" + bb->name() + ")");
            }
        }
    }

    const Function& fn_;
    Form form_;
    std::unordered_set<const BasicBlock*> blocks_;
    std::vector<std::string> errors_;
};

}  // namespace

std::vector<std::string> verify(const Function& fn, Form form) { return Verifier(fn, form).run(); }

}  // namespace kite::ir
