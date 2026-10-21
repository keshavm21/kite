#include "analysis/loops.h"

#include <algorithm>
#include <map>

namespace kite::analysis {

using ir::BasicBlock;

std::vector<Loop> findLoops(const ir::Function& fn, const DomTree& dt) {
    (void)fn;
    std::map<BasicBlock*, Loop> byHeader;  // ordered by pointer, only used for lookup; sorted below
    for (BasicBlock* t : dt.rpo()) {
        for (BasicBlock* h : t->succs) {
            if (!dt.dominates(h, t)) continue;  // not a back edge
            Loop& loop = byHeader[h];
            loop.header = h;
            loop.latches.push_back(t);
            // Reverse DFS from the latch, stopping at the header.
            loop.blocks.insert(h);
            std::vector<BasicBlock*> work;
            if (loop.blocks.insert(t).second) work.push_back(t);
            while (!work.empty()) {
                BasicBlock* b = work.back();
                work.pop_back();
                for (BasicBlock* p : b->preds) {
                    if (dt.reachable(p) && loop.blocks.insert(p).second) work.push_back(p);
                }
            }
        }
    }
    std::vector<Loop> loops;
    for (auto& [h, loop] : byHeader) {
        BasicBlock* outside = nullptr;
        int outsideCount = 0;
        for (BasicBlock* p : h->preds) {
            if (!dt.reachable(p) || loop.contains(p)) continue;
            if (p != outside) ++outsideCount;
            outside = p;
        }
        if (outsideCount == 1 && outside->succs.size() == 1) loop.preheader = outside;
        loops.push_back(std::move(loop));
    }
    std::sort(loops.begin(), loops.end(), [](const Loop& a, const Loop& b) {
        if (a.blocks.size() != b.blocks.size()) return a.blocks.size() < b.blocks.size();
        return a.header->id < b.header->id;
    });
    return loops;
}

}  // namespace kite::analysis
