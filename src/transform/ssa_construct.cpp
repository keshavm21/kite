#include <algorithm>
#include <set>

#include "analysis/cfg.h"
#include "analysis/dominators.h"
#include "transform/ssa.h"

namespace kite::transform {

using namespace ir;

namespace {

class SSABuilder {
public:
    explicit SSABuilder(Function& fn) : fn_(fn), dt_(fn) {}

    void run() {
        findVariables();
        placePhis();
        stacks_.assign(isVar_.size(), {});
        rename(fn_.blocks[0].get());
    }

private:
    // A variable is a register with more than one definition. Only variables that are used in a block
    // before being defined there ("globals") can need a phi.
    void findVariables() {
        const int n = fn_.numRegs();
        std::vector<int> defs(n, 0);
        for (int p = 0; p < fn_.numParams; ++p) defs[p] = 1;
        std::vector<bool> global(n, false);
        defBlocks_.assign(n, {});
        for (BasicBlock* bb : dt_.rpo()) {
            std::vector<bool> killed(n, false);
            for (const Inst& in : bb->insts) {
                for (const Operand& a : in.args) {
                    if (a.isReg() && !killed[a.reg]) global[a.reg] = true;
                }
                if (in.dst < 0) continue;
                killed[in.dst] = true;
                ++defs[in.dst];
                if (defBlocks_[in.dst].empty() || defBlocks_[in.dst].back() != bb) defBlocks_[in.dst].push_back(bb);
            }
        }
        isVar_.assign(n, false);
        needsPhi_.assign(n, false);
        for (int r = 0; r < n; ++r) {
            isVar_[r] = defs[r] > 1;
            needsPhi_[r] = isVar_[r] && global[r];
        }
    }

    void placePhis() {
        analysis::DomFrontier df = analysis::dominanceFrontiers(fn_, dt_);
        for (int v = 0; v < fn_.numRegs(); ++v) {
            if (!needsPhi_[v]) continue;
            std::vector<BasicBlock*> work = defBlocks_[v];
            std::set<const BasicBlock*> placed;
            while (!work.empty()) {
                BasicBlock* x = work.back();
                work.pop_back();
                for (BasicBlock* y : df[x]) {
                    if (!placed.insert(y).second) continue;
                    phiVars_[y].push_back(v);
                    bool definesV = std::find(defBlocks_[v].begin(), defBlocks_[v].end(), y) != defBlocks_[v].end();
                    if (!definesV) work.push_back(y);
                }
            }
        }
        for (auto& [bb, vars] : phiVars_) {
            std::vector<BasicBlock*> preds;  // each reachable predecessor once
            for (BasicBlock* p : bb->preds) {
                if (dt_.reachable(p) && std::find(preds.begin(), preds.end(), p) == preds.end()) preds.push_back(p);
            }
            std::vector<Inst> phis;
            for (int v : vars) {
                Inst phi;
                phi.op = Op::Phi;
                phi.ty = fn_.regTy[v];
                phi.blocks = preds;
                phi.args.assign(preds.size(), Operand::undef());
                phis.push_back(std::move(phi));
            }
            bb->insts.insert(bb->insts.begin(), phis.begin(), phis.end());
        }
    }

    int freshFor(int v) {
        std::string hint = fn_.regHint[v];  // copied: newReg may reallocate the vector
        return fn_.newReg(fn_.regTy[v], hint);
    }

    Operand top(int v) const { return stacks_[v].empty() ? Operand::undef() : Operand::r(stacks_[v].back()); }

    void rename(BasicBlock* b) {
        std::vector<int> pushed;
        size_t nPhis = 0;
        auto it = phiVars_.find(b);
        if (it != phiVars_.end()) {
            nPhis = it->second.size();
            for (size_t k = 0; k < nPhis; ++k) {
                int v = it->second[k];
                int d = freshFor(v);
                b->insts[k].dst = d;
                stacks_[v].push_back(d);
                pushed.push_back(v);
            }
        }
        for (size_t k = nPhis; k < b->insts.size(); ++k) {
            Inst& in = b->insts[k];
            for (Operand& a : in.args) {
                if (a.isReg() && isVar_[a.reg]) a = top(a.reg);
            }
            if (in.dst >= 0 && isVar_[in.dst]) {
                int v = in.dst;
                in.dst = freshFor(v);
                stacks_[v].push_back(in.dst);
                pushed.push_back(v);
            }
        }
        for (BasicBlock* s : b->succs) {
            auto sit = phiVars_.find(s);
            if (sit == phiVars_.end()) continue;
            for (size_t k = 0; k < sit->second.size(); ++k) {
                Inst& phi = s->insts[k];
                for (size_t j = 0; j < phi.blocks.size(); ++j) {
                    if (phi.blocks[j] == b) phi.args[j] = top(sit->second[k]);
                }
            }
        }
        for (BasicBlock* c : dt_.children(b)) rename(c);
        for (int v : pushed) stacks_[v].pop_back();
    }

    Function& fn_;
    analysis::DomTree dt_;
    std::vector<bool> isVar_, needsPhi_;
    std::vector<std::vector<BasicBlock*>> defBlocks_;
    std::map<BasicBlock*, std::vector<int>, bool (*)(const BasicBlock*, const BasicBlock*)> phiVars_{
        [](const BasicBlock* x, const BasicBlock* y) { return x->id < y->id; }};
    std::vector<std::vector<int>> stacks_;
};

}  // namespace

void constructSSA(Function& fn) {
    cfg::removeUnreachable(fn);  // also rebuilds preds/succs
    SSABuilder(fn).run();
}

}  // namespace kite::transform
