#include "transform/opt.h"

namespace kite::transform {

void applyReplacements(ir::Function& fn, const Replacements& repl) {
    if (repl.empty()) return;
    for (auto& bb : fn.blocks) {
        for (ir::Inst& in : bb->insts) {
            for (ir::Operand& a : in.args) {
                for (auto it = repl.find(a.reg); a.isReg() && it != repl.end(); it = repl.find(a.reg)) a = it->second;
            }
        }
    }
}

size_t instructionCount(const ir::Function& fn) {
    size_t n = 0;
    for (const auto& bb : fn.blocks) n += bb->insts.size();
    return n;
}

}  // namespace kite::transform
