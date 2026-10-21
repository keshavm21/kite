#pragma once
#include <utility>
#include <vector>

#include "ir/ir.h"

namespace kite::ir {

// Appends instructions to a current block. Each emit* that produces a value creates its
// destination register and returns it as an operand.
//
// Emitting into a block that already ends in a terminator (code after `return`, `break`, ...)
// does not corrupt it: the builder first switches to a fresh `dead` block, which has no
// predecessors. simplifycfg deletes it at -O1; at -O0 it is harmless.
class Builder {
public:
    explicit Builder(Function& fn) : fn_(fn) {}

    void setBlock(BasicBlock* bb);
    BasicBlock* block() const { return bb_; }
    bool terminated() const { return bb_->hasTerminator(); }
    // Every block the builder has ever been pointed at, in first-use order. The lowerer sorts
    // Function::blocks by it so the printed IR reads in program order.
    const std::vector<BasicBlock*>& placed() const { return placed_; }

    Inst& append(Inst inst);

    Operand emitCopy(Ty ty, Operand a, const std::string& hint = "");
    void emitCopyTo(int dst, Ty ty, Operand a);  // assigns an existing register (lowered IR only)
    Operand emitBinary(Op op, Ty ty, Operand a, Operand b);  // arithmetic, compare (ty = operand type), min/max, bool ops
    Operand emitUnary(Op op, Ty ty, Operand a);               // neg, not, abs, sqrt, exp
    Operand emitCast(Op op, Operand a);                       // itof, ftoi, btoi
    Operand emitSpecial(SReg reg, int dim);
    Operand emitPhi(Ty ty, std::vector<std::pair<Operand, BasicBlock*>> incoming, const std::string& hint = "");
    Operand emitLoad(Ty ty, Space space, Operand base, Operand index);
    void emitStore(Ty ty, Space space, Operand base, Operand index, Operand value);
    void emitBarrier();
    void emitBr(Operand cond, BasicBlock* thenBlock, BasicBlock* elseBlock);
    void emitJmp(BasicBlock* target);
    void emitRet();

private:
    Operand define(Inst inst, const std::string& hint = "");

    Function& fn_;
    BasicBlock* bb_ = nullptr;
    std::vector<BasicBlock*> placed_;
};

}  // namespace kite::ir
