#pragma once
#include <sstream>
#include <string>

#include "ir/ir.h"

namespace kite::codegen {

// IR -> PTX text (docs/PTX_BACKEND.md). Accepts lowered or post-SSA IR; phis are an error.
class PtxEmitter {
public:
    explicit PtxEmitter(const ir::Module& module, std::string target = "sm_75")
        : module_(module), target_(std::move(target)) {}

    // Throws std::logic_error if the IR contains a phi (or another instruction PTX cannot express).
    std::string emit();

private:
    enum Cls { Pred, B32, F32, B64, NumClasses };

    void emitKernel(const ir::Function& fn);
    void emitBlock(const ir::BasicBlock& bb, const ir::BasicBlock* next);
    void emitInst(const ir::Inst& in, const ir::BasicBlock* next);
    void emitBranch(const ir::Inst& in, const ir::BasicBlock* next);
    void emitMemory(const ir::Inst& in);
    void emitCompare(const ir::Inst& in, const std::string& d);

    static Cls clsOf(ir::Ty t);
    std::string regName(ir::Ty t, int id);      // "%r7" / "%f7" / "%p7" / "%rd7"; records the id for the .reg declarations
    std::string reg(int id);                     // IR register -> name
    std::string newTemp(ir::Ty t);               // backend temporary; ids start past the IR registers
    std::string src(const ir::Operand& o);       // register, integer, 0fXXXXXXXX or 1/0
    std::string regOf(const ir::Operand& o, ir::Ty ty);  // like src(), but immediates are first moved into a temp
    std::string label(const ir::BasicBlock& bb) const;
    void line(const std::string& mnemonic, const std::string& operands = "");

    const ir::Module& module_;
    std::string target_;
    std::ostringstream out_;

    // Per-kernel state.
    const ir::Function* fn_ = nullptr;
    std::ostringstream body_;
    int maxId_[NumClasses];
    int nextTemp_ = 0;
    std::vector<std::string> sharedBase_;  // register holding each shared array's address
};

}  // namespace kite::codegen
