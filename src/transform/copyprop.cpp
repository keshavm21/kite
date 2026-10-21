#include <algorithm>

#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

bool copyprop(Function& fn) {
    Replacements repl;
    for (auto& bb : fn.blocks) {
        for (const Inst& in : bb->insts) {
            if (in.op == Op::Copy && in.args[0].kind != Operand::Undef) repl[in.dst] = in.args[0];
        }
    }
    if (repl.empty()) return false;
    applyReplacements(fn, repl);
    for (auto& bb : fn.blocks) {
        auto& v = bb->insts;
        v.erase(std::remove_if(v.begin(), v.end(), [&](const Inst& in) { return in.op == Op::Copy && repl.count(in.dst); }), v.end());
    }
    return true;
}

}  // namespace kite::transform
