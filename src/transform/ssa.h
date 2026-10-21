#pragma once
#include <functional>
#include <vector>

#include "ir/ir.h"

namespace kite::transform {

// Lowered IR -> SSA (Cytron et al. 1991, docs/IMPLEMENTATION_PLAN.md 5.3). Removes unreachable blocks
// first. Registers with several definitions get one fresh register per definition; phis are placed at the
// iterated dominance frontier, but only for registers that are live across a block boundary
// ("semi-pruned" SSA), so block-local temporaries do not get phis. Copies are left in place: copyprop
// and dce (Milestone 6) clean them up.
void constructSSA(ir::Function& fn);

// SSA -> post-SSA (5.4): splits edges P->S where P has several successors and S has phis, then replaces
// every phi by a sequentialized parallel copy at the end of each predecessor.
void destructSSA(ir::Function& fn);

// `dst <- src` where dst is a register.
struct Copy {
    int dst;
    ir::Operand src;
};

// Turns a parallel copy (every source is read before any destination is written) into an ordered list of
// plain copies. Copies with dst == src are dropped; cycles such as {a<-b, b<-a} are broken with a temporary
// obtained from `newTempLike(reg)` (a fresh register of the same type as `reg`).
std::vector<Copy> sequentializeParallelCopy(std::vector<Copy> parallel, const std::function<int(int)>& newTempLike);

}  // namespace kite::transform
