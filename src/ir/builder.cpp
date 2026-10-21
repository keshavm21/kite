#include "ir/builder.h"

#include <algorithm>

namespace kite::ir {

void Builder::setBlock(BasicBlock* bb) {
    bb_ = bb;
    if (std::find(placed_.begin(), placed_.end(), bb) == placed_.end()) placed_.push_back(bb);
}

Inst& Builder::append(Inst inst) {
    if (terminated()) setBlock(fn_.newBlock("dead"));
    bb_->insts.push_back(std::move(inst));
    return bb_->insts.back();
}

Operand Builder::define(Inst inst, const std::string& hint) {
    inst.dst = fn_.newReg(inst.resultType(), hint);
    int dst = inst.dst;
    append(std::move(inst));
    return Operand::r(dst);
}

Operand Builder::emitCopy(Ty ty, Operand a, const std::string& hint) {
    Inst i;
    i.op = Op::Copy;
    i.ty = ty;
    i.args = {a};
    return define(std::move(i), hint);
}

void Builder::emitCopyTo(int dst, Ty ty, Operand a) {
    Inst i;
    i.op = Op::Copy;
    i.ty = ty;
    i.dst = dst;
    i.args = {a};
    append(std::move(i));
}

Operand Builder::emitBinary(Op op, Ty ty, Operand a, Operand b) {
    Inst i;
    i.op = op;
    i.ty = ty;
    i.args = {a, b};
    return define(std::move(i));
}

Operand Builder::emitUnary(Op op, Ty ty, Operand a) {
    Inst i;
    i.op = op;
    i.ty = ty;
    i.args = {a};
    return define(std::move(i));
}

Operand Builder::emitCast(Op op, Operand a) {
    Inst i;
    i.op = op;
    i.args = {a};
    i.ty = i.resultType();
    return define(std::move(i));
}

Operand Builder::emitSpecial(SReg reg, int dim) {
    Inst i;
    i.op = Op::Special;
    i.ty = Ty::I32;
    i.sreg = reg;
    i.dim = dim;
    return define(std::move(i));
}

Operand Builder::emitPhi(Ty ty, std::vector<std::pair<Operand, BasicBlock*>> incoming, const std::string& hint) {
    Inst i;
    i.op = Op::Phi;
    i.ty = ty;
    for (auto& [value, from] : incoming) {
        i.args.push_back(value);
        i.blocks.push_back(from);
    }
    return define(std::move(i), hint);
}

Operand Builder::emitLoad(Ty ty, Space space, Operand base, Operand index) {
    Inst i;
    i.op = Op::Load;
    i.ty = ty;
    i.space = space;
    i.args = {base, index};
    return define(std::move(i));
}

void Builder::emitStore(Ty ty, Space space, Operand base, Operand index, Operand value) {
    Inst i;
    i.op = Op::Store;
    i.ty = ty;
    i.space = space;
    i.args = {base, index, value};
    append(std::move(i));
}

void Builder::emitBarrier() {
    Inst i;
    i.op = Op::Barrier;
    append(std::move(i));
}

void Builder::emitBr(Operand cond, BasicBlock* thenBlock, BasicBlock* elseBlock) {
    Inst i;
    i.op = Op::Br;
    i.args = {cond};
    i.blocks = {thenBlock, elseBlock};
    append(std::move(i));
}

void Builder::emitJmp(BasicBlock* target) {
    Inst i;
    i.op = Op::Jmp;
    i.blocks = {target};
    append(std::move(i));
}

void Builder::emitRet() {
    Inst i;
    i.op = Op::Ret;
    append(std::move(i));
}

}  // namespace kite::ir
