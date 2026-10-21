#include "analysis/cfg.h"

namespace kite::cfg {

void rebuild(ir::Function& fn) {
    for (auto& bb : fn.blocks) {
        bb->preds.clear();
        bb->succs.clear();
    }
    for (auto& bb : fn.blocks) {
        if (!bb->hasTerminator()) continue;
        for (ir::BasicBlock* target : bb->terminator().blocks) {
            bb->succs.push_back(target);
            target->preds.push_back(bb.get());
        }
    }
}

}  // namespace kite::cfg
