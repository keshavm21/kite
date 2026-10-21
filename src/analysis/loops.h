#pragma once
#include <set>
#include <vector>

#include "analysis/dominators.h"
#include "ir/ir.h"

namespace kite::analysis {

// A natural loop: the header plus every block that reaches a back-edge source without leaving through the
// header. Back edges that share a header are merged into one loop.
struct Loop {
    ir::BasicBlock* header = nullptr;
    std::set<const ir::BasicBlock*> blocks;  // includes the header
    std::vector<ir::BasicBlock*> latches;    // sources of the back edges
    // The header's unique predecessor outside the loop, if that block's only successor is the header.
    // Null when there is no such block.
    ir::BasicBlock* preheader = nullptr;

    bool contains(const ir::BasicBlock* b) const { return blocks.count(b) != 0; }
};

// All natural loops of the reachable part of `fn`, innermost first (ascending body size, ties by header
// id). Needs current preds/succs (cfg::rebuild).
std::vector<Loop> findLoops(const ir::Function& fn, const DomTree& dt);

}  // namespace kite::analysis
