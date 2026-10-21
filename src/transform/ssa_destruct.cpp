#include <algorithm>

#include "analysis/cfg.h"
#include "transform/ssa.h"

namespace kite::transform {

using namespace ir;

std::vector<Copy> sequentializeParallelCopy(std::vector<Copy> pending, const std::function<int(int)>& newTempLike) {
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [](const Copy& c) { return c.src.isReg() && c.src.reg == c.dst; }),
                  pending.end());
    std::vector<Copy> out;
    auto isPendingSource = [&](int reg, size_t except) {
        for (size_t j = 0; j < pending.size(); ++j) {
            if (j != except && pending[j].src.isReg() && pending[j].src.reg == reg) return true;
        }
        return false;
    };
    while (!pending.empty()) {
        size_t ready = pending.size();
        for (size_t i = 0; i < pending.size(); ++i) {
            if (!isPendingSource(pending[i].dst, i)) {  // nobody still needs the old value of dst
                ready = i;
                break;
            }
        }
        if (ready < pending.size()) {
            out.push_back(pending[ready]);
            pending.erase(pending.begin() + ready);
            continue;
        }
        // Only cycles remain: save one destination's old value and redirect its readers.
        int d = pending[0].dst;
        int t = newTempLike(d);
        out.push_back({t, Operand::r(d)});
        for (Copy& c : pending) {
            if (c.src.isReg() && c.src.reg == d) c.src = Operand::r(t);
        }
    }
    return out;
}

namespace {

bool startsWithPhi(const BasicBlock& bb) { return !bb.insts.empty() && bb.insts[0].op == Op::Phi; }

// Gives every edge P->S, where P branches and S has phis, its own block so the phi copies have a home.
void splitCriticalEdges(Function& fn) {
    std::vector<BasicBlock*> original;
    for (auto& bb : fn.blocks) original.push_back(bb.get());
    for (BasicBlock* p : original) {
        if (!p->hasTerminator()) continue;
        Inst& term = p->terminator();
        if (term.op != Op::Br || term.blocks[0] == term.blocks[1]) continue;
        for (size_t k = 0; k < term.blocks.size(); ++k) {
            BasicBlock* s = term.blocks[k];
            if (!startsWithPhi(*s)) continue;
            BasicBlock* e = fn.newBlock("split");
            auto pos = std::find_if(fn.blocks.begin(), fn.blocks.end(), [&](auto& b) { return b.get() == p; });
            std::rotate(pos + 1, fn.blocks.end() - 1, fn.blocks.end());  // place the new block right after P
            Inst jmp;
            jmp.op = Op::Jmp;
            jmp.blocks = {s};
            e->insts.push_back(jmp);
            term.blocks[k] = e;
            for (Inst& phi : s->insts) {
                if (phi.op != Op::Phi) break;
                for (BasicBlock*& from : phi.blocks) {
                    if (from == p) from = e;
                }
            }
        }
    }
    cfg::rebuild(fn);
}

}  // namespace

void destructSSA(Function& fn) {
    cfg::rebuild(fn);
    splitCriticalEdges(fn);
    for (auto& s : fn.blocks) {
        if (!startsWithPhi(*s)) continue;
        std::vector<BasicBlock*> preds;
        for (BasicBlock* p : s->preds) {
            if (std::find(preds.begin(), preds.end(), p) == preds.end()) preds.push_back(p);
        }
        for (BasicBlock* p : preds) {
            std::vector<Copy> parallel;
            for (const Inst& phi : s->insts) {
                if (phi.op != Op::Phi) break;
                for (size_t j = 0; j < phi.blocks.size(); ++j) {
                    if (phi.blocks[j] != p || phi.args[j].kind == Operand::Undef) continue;  // undef: no copy
                    parallel.push_back({phi.dst, phi.args[j]});
                }
            }
            auto newTemp = [&](int like) { return fn.newReg(fn.regTy[like], "swap"); };
            std::vector<Copy> ordered = sequentializeParallelCopy(std::move(parallel), newTemp);
            std::vector<Inst> insts;
            for (const Copy& c : ordered) {
                Inst copy;
                copy.op = Op::Copy;
                copy.ty = fn.regTy[c.dst];
                copy.dst = c.dst;
                copy.args = {c.src};
                insts.push_back(std::move(copy));
            }
            p->insts.insert(p->insts.end() - 1, insts.begin(), insts.end());  // before the terminator
        }
        s->insts.erase(s->insts.begin(), std::find_if(s->insts.begin(), s->insts.end(), [](const Inst& i) { return i.op != Op::Phi; }));
    }
    cfg::rebuild(fn);
}

}  // namespace kite::transform
