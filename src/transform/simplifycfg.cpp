#include <algorithm>

#include "analysis/cfg.h"
#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

namespace {

void dropPhiEntries(BasicBlock* s, const BasicBlock* from) {
    for (Inst& phi : s->insts) {
        if (phi.op != Op::Phi) break;
        for (size_t k = phi.blocks.size(); k-- > 0;) {
            if (phi.blocks[k] != from) continue;
            phi.blocks.erase(phi.blocks.begin() + k);
            phi.args.erase(phi.args.begin() + k);
        }
    }
}

// br true/false -> jmp, and br c, X, X -> jmp X. Returns true if a terminator changed.
bool foldBranches(Function& fn) {
    bool changed = false;
    for (auto& bb : fn.blocks) {
        if (!bb->hasTerminator()) continue;
        Inst& t = bb->terminator();
        if (t.op != Op::Br) continue;
        BasicBlock *taken = nullptr, *dropped = nullptr;
        if (t.args[0].kind == Operand::ImmBool) {
            taken = t.args[0].i ? t.blocks[0] : t.blocks[1];
            dropped = t.args[0].i ? t.blocks[1] : t.blocks[0];
        } else if (t.blocks[0] == t.blocks[1]) {
            taken = t.blocks[0];
        }
        if (!taken) continue;
        if (dropped && dropped != taken) dropPhiEntries(dropped, bb.get());
        t.op = Op::Jmp;
        t.args.clear();
        t.blocks = {taken};
        changed = true;
    }
    return changed;
}

// Merges B into P when P ends in `jmp B` and P is B's only predecessor. Returns true if it merged one pair.
bool mergeOne(Function& fn) {
    for (auto& pp : fn.blocks) {
        BasicBlock* p = pp.get();
        if (!p->hasTerminator() || p->terminator().op != Op::Jmp) continue;
        BasicBlock* b = p->terminator().blocks[0];
        if (b == p || b == fn.blocks[0].get() || b->preds.size() != 1 || b->preds[0] != p) continue;
        Replacements repl;
        size_t nPhis = 0;
        bool ok = true;
        for (const Inst& phi : b->insts) {
            if (phi.op != Op::Phi) break;
            ++nPhis;
            if (phi.args.size() != 1 || phi.args[0].kind == Operand::Undef) ok = false;
            else repl[phi.dst] = phi.args[0];
        }
        if (!ok) continue;
        applyReplacements(fn, repl);
        p->insts.pop_back();  // the jmp
        p->insts.insert(p->insts.end(), std::make_move_iterator(b->insts.begin() + nPhis), std::make_move_iterator(b->insts.end()));
        for (BasicBlock* s : p->terminator().blocks) {  // B's successors now come from P
            for (Inst& phi : s->insts) {
                if (phi.op != Op::Phi) break;
                for (BasicBlock*& from : phi.blocks) {
                    if (from == b) from = p;
                }
            }
        }
        fn.blocks.erase(std::find_if(fn.blocks.begin(), fn.blocks.end(), [&](auto& x) { return x.get() == b; }));
        cfg::rebuild(fn);
        return true;
    }
    return false;
}

}  // namespace

bool simplifycfg(Function& fn) {
    bool any = false;
    cfg::rebuild(fn);
    for (bool changed = true; changed;) {
        changed = false;
        if (foldBranches(fn)) {
            cfg::rebuild(fn);
            changed = true;
        }
        if (cfg::removeUnreachable(fn)) changed = true;
        while (mergeOne(fn)) changed = true;
        any |= changed;
    }
    return any;
}

}  // namespace kite::transform
