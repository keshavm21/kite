#pragma once
#include <map>
#include <unordered_map>
#include <vector>

#include "ir/ir.h"

namespace kite::analysis {

// Dominator tree of the blocks reachable from the entry (Cooper, Harvey, Kennedy 2001).
// Needs current preds/succs (cfg::rebuild). Unreachable blocks are not in the tree: idom() is null for
// them and for the entry, and they dominate nothing and are dominated by nothing.
class DomTree {
public:
    explicit DomTree(const ir::Function& fn);

    const std::vector<ir::BasicBlock*>& rpo() const { return rpo_; }  // reachable blocks, entry first
    bool reachable(const ir::BasicBlock* b) const { return index_.count(b) != 0; }
    ir::BasicBlock* idom(const ir::BasicBlock* b) const;  // null for the entry and unreachable blocks
    const std::vector<ir::BasicBlock*>& children(const ir::BasicBlock* b) const;  // in rpo order
    bool dominates(const ir::BasicBlock* a, const ir::BasicBlock* b) const;       // reflexive

private:
    std::vector<ir::BasicBlock*> rpo_;
    std::unordered_map<const ir::BasicBlock*, int> index_;  // position in rpo_
    std::vector<int> idom_;                                 // by rpo index; entry maps to itself
    std::vector<std::vector<ir::BasicBlock*>> children_;
    std::vector<ir::BasicBlock*> none_;
};

// DF[b] for every reachable block b: the blocks where b's dominance ends. Each set is ordered by
// block id so that anything built from it is deterministic.
using DomFrontier = std::map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>;
DomFrontier dominanceFrontiers(const ir::Function& fn, const DomTree& dt);

}  // namespace kite::analysis
