#include <algorithm>
#include <set>

#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

// Mark and sweep. An instruction is live if it is a root (store, barrier, terminator) or defines a
// register some live instruction reads. Working on registers rather than single definitions makes this
// correct for non-SSA IR as well, and it removes dead phi cycles that "delete if unused" would keep.
bool dce(Function& fn) {
    std::vector<bool> liveReg(fn.numRegs(), false);
    std::set<const Inst*> live;
    auto isRoot = [](const Inst& in) { return in.hasSideEffects() || in.isTerminator(); };
    bool grew = true;
    while (grew) {
        grew = false;
        for (auto& bb : fn.blocks) {
            for (const Inst& in : bb->insts) {
                if (live.count(&in)) continue;
                if (!isRoot(in) && !(in.dst >= 0 && liveReg[in.dst])) continue;
                live.insert(&in);
                grew = true;
                for (const Operand& a : in.args) {
                    if (a.isReg()) liveReg[a.reg] = true;
                }
            }
        }
    }
    bool changed = false;
    for (auto& bb : fn.blocks) {
        auto& v = bb->insts;
        size_t before = v.size();
        v.erase(std::remove_if(v.begin(), v.end(), [&](const Inst& in) { return !live.count(&in); }), v.end());
        changed |= v.size() != before;
    }
    return changed;
}

}  // namespace kite::transform
