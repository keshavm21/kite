#pragma once
#include <vector>

#include "ir/ir.h"

namespace kite::cfg {

// Recomputes every block's succs (the terminator's targets, in order, duplicates kept)
// and preds (one entry per incoming edge). Call after editing terminators.
void rebuild(ir::Function& fn);

// Blocks reachable from the entry, in reverse postorder (the entry first). Needs current succs.
std::vector<ir::BasicBlock*> reversePostOrder(const ir::Function& fn);

// Deletes every block not reachable from the entry, drops the matching entries from phis in the
// surviving blocks, and rebuilds preds/succs. Returns true if anything was removed.
bool removeUnreachable(ir::Function& fn);

}  // namespace kite::cfg
