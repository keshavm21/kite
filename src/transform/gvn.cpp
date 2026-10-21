#include <cstring>
#include <map>
#include <set>

#include <algorithm>

#include "analysis/cfg.h"
#include "analysis/dominators.h"
#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

namespace {

bool commutative(Op op) {
    switch (op) {
        case Op::Add: case Op::Mul: case Op::Eq: case Op::Ne: case Op::And: case Op::Or: case Op::Xor: case Op::Min: case Op::Max:
            return true;
        default:
            return false;
    }
}

using Key = std::vector<int64_t>;

Key operandKey(const Operand& o) {
    int64_t bits = o.i;
    if (o.kind == Operand::ImmF32) {
        uint32_t u;
        std::memcpy(&u, &o.f, sizeof u);
        bits = u;
    }
    return {o.kind, o.isReg() ? o.reg : -1, bits};
}

class Numbering {
public:
    Numbering(Function& fn) : fn_(fn), dt_(fn) {}

    bool run() {
        visit(fn_.blocks[0].get());
        if (repl_.empty()) return false;
        applyReplacements(fn_, repl_);  // phis may use values numbered after the phi's block was visited
        for (auto& bb : fn_.blocks) {
            auto& v = bb->insts;
            v.erase(std::remove_if(v.begin(), v.end(), [&](const Inst& in) { return dead_.count(&in) != 0; }), v.end());
        }
        return true;
    }

private:
    void visit(BasicBlock* b) {
        std::vector<Key> added;
        for (Inst& in : b->insts) {
            for (Operand& a : in.args) {
                for (auto it = repl_.find(a.reg); a.isReg() && it != repl_.end(); it = repl_.find(a.reg)) a = it->second;
            }
            // Never number loads (memory can change), phis (they depend on the edge taken) or effects.
            if (in.dst < 0 || !in.isPure() || in.op == Op::Phi) continue;
            std::vector<Key> ops;
            for (const Operand& a : in.args) ops.push_back(operandKey(a));
            if (commutative(in.op)) std::sort(ops.begin(), ops.end());
            Key key = {static_cast<int64_t>(in.op), static_cast<int64_t>(in.ty), static_cast<int64_t>(in.sreg), in.dim};
            for (const Key& k : ops) key.insert(key.end(), k.begin(), k.end());
            auto hit = table_.find(key);
            if (hit != table_.end()) {
                repl_[in.dst] = Operand::r(hit->second);
                dead_.insert(&in);
            } else {
                table_[key] = in.dst;
                added.push_back(key);
            }
        }
        for (BasicBlock* c : dt_.children(b)) visit(c);
        for (const Key& k : added) table_.erase(k);  // leave this block's scope
    }

    Function& fn_;
    analysis::DomTree dt_;
    std::map<Key, int> table_;
    Replacements repl_;
    std::set<const Inst*> dead_;
};

}  // namespace

bool gvn(Function& fn) {
    cfg::rebuild(fn);
    return Numbering(fn).run();
}

}  // namespace kite::transform
