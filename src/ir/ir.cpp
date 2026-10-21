#include "ir/ir.h"

namespace kite::ir {

const char* tyName(Ty t) {
    switch (t) {
        case Ty::Void: return "void";
        case Ty::Bool: return "bool";
        case Ty::I32: return "i32";
        case Ty::F32: return "f32";
        case Ty::PtrF32: return "ptr.f32";
        case Ty::PtrI32: return "ptr.i32";
    }
    return "?";
}

const char* opName(Op op) {
    switch (op) {
        case Op::Copy: return "copy";
        case Op::Add: return "add";
        case Op::Sub: return "sub";
        case Op::Mul: return "mul";
        case Op::Div: return "div";
        case Op::Rem: return "rem";
        case Op::Neg: return "neg";
        case Op::Shl: return "shl";
        case Op::Eq: return "eq";
        case Op::Ne: return "ne";
        case Op::Lt: return "lt";
        case Op::Le: return "le";
        case Op::Gt: return "gt";
        case Op::Ge: return "ge";
        case Op::And: return "and";
        case Op::Or: return "or";
        case Op::Xor: return "xor";
        case Op::Not: return "not";
        case Op::IToF: return "itof";
        case Op::FToI: return "ftoi";
        case Op::BToI: return "btoi";
        case Op::Sqrt: return "sqrt";
        case Op::Exp: return "exp";
        case Op::Abs: return "abs";
        case Op::Min: return "min";
        case Op::Max: return "max";
        case Op::Special: return "special";
        case Op::Phi: return "phi";
        case Op::Load: return "load";
        case Op::Store: return "store";
        case Op::Barrier: return "barrier";
        case Op::Br: return "br";
        case Op::Jmp: return "jmp";
        case Op::Ret: return "ret";
    }
    return "?";
}

const char* sregName(SReg r) {
    switch (r) {
        case SReg::Tid: return "tid";
        case SReg::Ntid: return "ntid";
        case SReg::Ctaid: return "ctaid";
        case SReg::Nctaid: return "nctaid";
    }
    return "?";
}

bool isPointer(Ty t) { return t == Ty::PtrF32 || t == Ty::PtrI32; }

Ty pointee(Ty ptr) { return ptr == Ty::PtrF32 ? Ty::F32 : Ty::I32; }

Ty Inst::resultType() const {
    switch (op) {
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::And: case Op::Or: case Op::Xor: case Op::Not:
            return Ty::Bool;
        case Op::IToF: return Ty::F32;
        case Op::FToI: case Op::BToI: case Op::Special: return Ty::I32;
        case Op::Store: case Op::Barrier: case Op::Br: case Op::Jmp: case Op::Ret:
            return Ty::Void;
        default:
            return ty;
    }
}

std::string BasicBlock::name() const { return id == 0 ? "entry" : hint + "." + std::to_string(id); }

int Function::newReg(Ty t, std::string hint) {
    regTy.push_back(t);
    regHint.push_back(std::move(hint));
    return numRegs() - 1;
}

BasicBlock* Function::newBlock(const std::string& hint) {
    blocks.push_back(std::make_unique<BasicBlock>());
    BasicBlock* bb = blocks.back().get();
    bb->id = nextBlockId_++;
    bb->hint = hint;
    return bb;
}

std::string Function::regName(int reg) const {
    const std::string& hint = regHint[reg];
    return "%" + (hint.empty() ? "" : hint + ".") + std::to_string(reg);
}

}  // namespace kite::ir
