#include <algorithm>
#include <unordered_map>

#include "analysis/cfg.h"
#include "analysis/dominators.h"
#include "analysis/loops.h"
#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

namespace {

LicmStats g_totals;

// Safe to execute on a path where the original program would not have: no side effects, no memory, and no
// way to trap. i32 div/rem can divide by zero, and ftoi is undefined for NaN and out-of-range values.
bool speculatable(const Inst& in) {
    if (in.dst < 0 || !in.isPure() || in.op == Op::Phi) return false;
    return in.op != Op::Div && in.op != Op::Rem && in.op != Op::FToI;
}

}  // namespace

LicmStats& licmTotals() { return g_totals; }

bool licm(Function& fn) {
    cfg::rebuild(fn);
    analysis::DomTree dt(fn);
    std::vector<analysis::Loop> loops = analysis::findLoops(fn, dt);
    bool changed = false;
    for (const analysis::Loop& loop : loops) {  // innermost first, so an inner hoist can move again in the outer loop
        ++g_totals.loops;
        if (!loop.preheader) {
            ++g_totals.skipped;
            continue;
        }
        std::vector<bool> inLoop(fn.numRegs(), false);  // registers defined inside the loop
        for (const BasicBlock* b : loop.blocks) {
            for (const Inst& in : b->insts) {
                if (in.dst >= 0) inLoop[in.dst] = true;
            }
        }
        BasicBlock* pre = loop.preheader;
        for (bool moved = true; moved;) {
            moved = false;
            for (BasicBlock* b : dt.rpo()) {
                if (!loop.contains(b)) continue;
                for (size_t i = 0; i < b->insts.size();) {
                    const Inst& in = b->insts[i];
                    bool invariant = speculatable(in);
                    for (const Operand& a : in.args) invariant = invariant && !(a.isReg() && inLoop[a.reg]);
                    if (!invariant) {
                        ++i;
                        continue;
                    }
                    inLoop[in.dst] = false;
                    pre->insts.insert(pre->insts.end() - 1, std::move(b->insts[i]));
                    b->insts.erase(b->insts.begin() + i);
                    ++g_totals.hoisted;
                    moved = changed = true;
                }
            }
        }
    }
    return changed;
}

}  // namespace kite::transform
