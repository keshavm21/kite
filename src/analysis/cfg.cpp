#include "analysis/cfg.h"

#include <algorithm>
#include <unordered_set>

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

std::vector<ir::BasicBlock*> reversePostOrder(const ir::Function& fn) {
    std::vector<ir::BasicBlock*> post;
    std::unordered_set<const ir::BasicBlock*> seen;
    // Iterative DFS: (block, index of the next successor to visit).
    std::vector<std::pair<ir::BasicBlock*, size_t>> stack;
    ir::BasicBlock* entry = fn.blocks[0].get();
    seen.insert(entry);
    stack.push_back({entry, 0});
    while (!stack.empty()) {
        auto& [bb, next] = stack.back();
        if (next < bb->succs.size()) {
            ir::BasicBlock* s = bb->succs[next++];
            if (seen.insert(s).second) stack.push_back({s, 0});
        } else {
            post.push_back(bb);
            stack.pop_back();
        }
    }
    std::reverse(post.begin(), post.end());
    return post;
}

bool removeUnreachable(ir::Function& fn) {
    rebuild(fn);
    std::unordered_set<const ir::BasicBlock*> live;
    for (ir::BasicBlock* bb : reversePostOrder(fn)) live.insert(bb);
    if (live.size() == fn.blocks.size()) return false;
    for (auto& bb : fn.blocks) {
        if (!live.count(bb.get())) continue;
        for (ir::Inst& in : bb->insts) {
            if (in.op != ir::Op::Phi) break;  // phis come first
            for (size_t k = in.blocks.size(); k-- > 0;) {
                if (live.count(in.blocks[k])) continue;
                in.blocks.erase(in.blocks.begin() + k);
                in.args.erase(in.args.begin() + k);
            }
        }
    }
    fn.blocks.erase(std::remove_if(fn.blocks.begin(), fn.blocks.end(), [&](auto& bb) { return !live.count(bb.get()); }),
                    fn.blocks.end());
    rebuild(fn);
    return true;
}

}  // namespace kite::cfg
