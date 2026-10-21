#include "analysis/dominators.h"

#include <algorithm>

#include "analysis/cfg.h"

namespace kite::analysis {

using ir::BasicBlock;

DomTree::DomTree(const ir::Function& fn) {
    rpo_ = cfg::reversePostOrder(fn);
    for (size_t k = 0; k < rpo_.size(); ++k) index_[rpo_[k]] = static_cast<int>(k);
    const int n = static_cast<int>(rpo_.size());
    idom_.assign(n, -1);
    idom_[0] = 0;
    auto intersect = [&](int a, int b) {
        while (a != b) {
            while (a > b) a = idom_[a];
            while (b > a) b = idom_[b];
        }
        return a;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (int b = 1; b < n; ++b) {
            int fresh = -1;
            for (const BasicBlock* p : rpo_[b]->preds) {
                auto it = index_.find(p);
                if (it == index_.end() || idom_[it->second] < 0) continue;  // unreachable, or not processed yet
                fresh = fresh < 0 ? it->second : intersect(it->second, fresh);
            }
            if (fresh != idom_[b]) {
                idom_[b] = fresh;
                changed = true;
            }
        }
    }
    children_.assign(n, {});
    for (int b = 1; b < n; ++b) children_[idom_[b]].push_back(rpo_[b]);
}

BasicBlock* DomTree::idom(const BasicBlock* b) const {
    auto it = index_.find(b);
    if (it == index_.end() || it->second == 0) return nullptr;
    return rpo_[idom_[it->second]];
}

const std::vector<BasicBlock*>& DomTree::children(const BasicBlock* b) const {
    auto it = index_.find(b);
    return it == index_.end() ? none_ : children_[it->second];
}

bool DomTree::dominates(const BasicBlock* a, const BasicBlock* b) const {
    auto ia = index_.find(a), ib = index_.find(b);
    if (ia == index_.end() || ib == index_.end()) return false;
    int x = ib->second;
    while (x > ia->second) x = idom_[x];  // an ancestor has a smaller rpo index
    return x == ia->second;
}

DomFrontier dominanceFrontiers(const ir::Function& fn, const DomTree& dt) {
    (void)fn;
    DomFrontier df;
    for (BasicBlock* b : dt.rpo()) df[b];
    for (BasicBlock* b : dt.rpo()) {
        if (b->preds.size() < 2) continue;
        for (BasicBlock* p : b->preds) {
            if (!dt.reachable(p)) continue;
            for (BasicBlock* runner = p; runner != dt.idom(b); runner = dt.idom(runner)) {
                auto& set = df[runner];
                if (std::find(set.begin(), set.end(), b) == set.end()) set.push_back(b);
            }
        }
    }
    for (auto& [b, set] : df) {
        std::sort(set.begin(), set.end(), [](const BasicBlock* x, const BasicBlock* y) { return x->id < y->id; });
    }
    return df;
}

}  // namespace kite::analysis
