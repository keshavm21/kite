#include <optional>
#include <algorithm>
#include <unordered_map>

#include "analysis/cfg.h"
#include "analysis/dominators.h"
#include "analysis/loops.h"
#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

namespace {

constexpr int kMaxTrips = 64;
constexpr size_t kMaxInsts = 400;  // trip count * (header + body instructions)

bool compare(Op op, int32_t a, int32_t b) {
    switch (op) {
        case Op::Eq: return a == b;
        case Op::Ne: return a != b;
        case Op::Lt: return a < b;
        case Op::Le: return a <= b;
        case Op::Gt: return a > b;
        case Op::Ge: return a >= b;
        default: return false;
    }
}

// What a fully unrollable loop looks like (after simplifycfg merged the for.step block into the body):
//
//   P:  ...; jmp H                       P = preheader
//   H:  i = phi [c0, P], [i2, B] ...     H = header: phis, pure code, `cond = cmp i, bound`, br cond, B, E
//   B:  ...; i2 = add i, step; jmp H     B = the only other block of the loop
//
// c0, step and bound are immediates, so the trip count is known.
struct Candidate {
    BasicBlock *pre = nullptr, *header = nullptr, *body = nullptr, *exit = nullptr;
    int trips = 0;
};

std::optional<Candidate> match(const analysis::Loop& loop) {
    BasicBlock* pre = loop.preheader;
    BasicBlock* h = loop.header;
    if (!pre || loop.blocks.size() != 2 || loop.latches.size() != 1) return std::nullopt;
    BasicBlock* b = loop.latches[0];
    if (b == h || h->preds.size() != 2 || b->preds.size() != 1 || b->preds[0] != h) return std::nullopt;
    if (b->terminator().op != Op::Jmp || h->terminator().op != Op::Br) return std::nullopt;
    const Inst& br = h->terminator();
    bool continueOnTrue = br.blocks[0] == b;
    if (br.blocks[continueOnTrue ? 1 : 0] == b || br.blocks[continueOnTrue ? 0 : 1] == h) return std::nullopt;
    BasicBlock* exit = br.blocks[continueOnTrue ? 1 : 0];
    if (loop.contains(exit)) return std::nullopt;

    size_t nPhis = 0;
    while (nPhis < h->insts.size() && h->insts[nPhis].op == Op::Phi) ++nPhis;
    for (size_t i = nPhis; i < h->insts.size(); ++i) {
        if (h->insts[i].hasSideEffects()) return std::nullopt;  // the header runs trips + 1 times; keep it effect-free
    }
    // The branch condition: `cmp i, imm` (either order) defined in the header.
    if (!br.args[0].isReg()) return std::nullopt;
    const Inst* cond = nullptr;
    for (size_t i = nPhis; i < h->insts.size(); ++i) {
        if (h->insts[i].dst == br.args[0].reg) cond = &h->insts[i];
    }
    if (!cond || !cond->isCompare() || cond->ty != Ty::I32) return std::nullopt;
    bool regFirst = cond->args[0].isReg();
    const Operand& iv = cond->args[regFirst ? 0 : 1];
    const Operand& bound = cond->args[regFirst ? 1 : 0];
    if (!iv.isReg() || bound.kind != Operand::ImmI32) return std::nullopt;
    // iv must be one of the header's phis, starting at an immediate and stepping by an immediate.
    const Inst* phi = nullptr;
    for (size_t i = 0; i < nPhis; ++i) {
        if (h->insts[i].dst == iv.reg) phi = &h->insts[i];
    }
    if (!phi || phi->args.size() != 2) return std::nullopt;
    int fromPre = phi->blocks[0] == pre ? 0 : 1, fromBody = 1 - fromPre;
    if (phi->blocks[fromPre] != pre || phi->blocks[fromBody] != b) return std::nullopt;
    if (phi->args[fromPre].kind != Operand::ImmI32 || !phi->args[fromBody].isReg()) return std::nullopt;
    const Inst* step = nullptr;
    for (const Inst& in : b->insts) {
        if (in.dst == phi->args[fromBody].reg) step = &in;
    }
    if (!step || step->ty != Ty::I32 || (step->op != Op::Add && step->op != Op::Sub)) return std::nullopt;
    int32_t delta;
    if (step->args[0].isReg() && step->args[0].reg == iv.reg && step->args[1].kind == Operand::ImmI32) {
        delta = step->args[1].i;
    } else if (step->op == Op::Add && step->args[1].isReg() && step->args[1].reg == iv.reg && step->args[0].kind == Operand::ImmI32) {
        delta = step->args[0].i;
    } else {
        return std::nullopt;
    }
    if (step->op == Op::Sub) delta = static_cast<int32_t>(0u - static_cast<uint32_t>(delta));

    // Count the trips by running the induction variable (i32 wraps).
    int32_t i = phi->args[fromPre].i;
    int trips = 0;
    for (;; ++trips) {
        bool c = regFirst ? compare(cond->op, i, bound.i) : compare(cond->op, bound.i, i);
        if (c != continueOnTrue) break;
        if (trips == kMaxTrips) return std::nullopt;
        i = static_cast<int32_t>(static_cast<uint32_t>(i) + static_cast<uint32_t>(delta));
    }
    if (static_cast<size_t>(trips + 1) * (h->insts.size() + b->insts.size()) > kMaxInsts) return std::nullopt;
    return Candidate{pre, h, b, exit, trips};
}

// Replaces the loop by `trips` copies of its body, appended to the preheader. The header is cloned once more
// than the body, because it runs one extra time to find out that the loop is over; values it defines are used
// after the loop, so the final clone provides them (dce removes whatever is unused).
void unrollLoop(Function& fn, const Candidate& c) {
    BasicBlock *pre = c.pre, *h = c.header, *b = c.body;
    size_t nPhis = 0;
    while (h->insts[nPhis].op == Op::Phi) ++nPhis;

    std::unordered_map<int, Operand> cur;  // original register of the loop -> its value in the current iteration
    auto val = [&](Operand o) {
        if (o.isReg()) {
            auto it = cur.find(o.reg);
            if (it != cur.end()) return it->second;
        }
        return o;
    };
    pre->insts.pop_back();  // the `jmp H`
    auto clone = [&](const Inst& in) {
        Inst copy = in;
        for (Operand& a : copy.args) a = val(a);
        if (in.dst >= 0) copy.dst = fn.newReg(fn.regTy[in.dst], fn.regHint[in.dst]);
        pre->insts.push_back(copy);
        if (in.dst >= 0) cur[in.dst] = Operand::r(copy.dst);
    };
    std::vector<Operand> phiIn(nPhis);
    for (size_t p = 0; p < nPhis; ++p) {
        const Inst& phi = h->insts[p];
        phiIn[p] = phi.args[phi.blocks[0] == pre ? 0 : 1];
    }
    for (int k = 0; k <= c.trips; ++k) {
        for (size_t p = 0; p < nPhis; ++p) cur[h->insts[p].dst] = phiIn[p];
        for (size_t i = nPhis; i + 1 < h->insts.size(); ++i) clone(h->insts[i]);  // all but the terminator
        if (k == c.trips) break;
        for (size_t i = 0; i + 1 < b->insts.size(); ++i) clone(b->insts[i]);
        for (size_t p = 0; p < nPhis; ++p) {  // parallel assignment: read every new value before writing any
            const Inst& phi = h->insts[p];
            phiIn[p] = val(phi.args[phi.blocks[0] == pre ? 1 : 0]);
        }
    }
    Inst jmp;
    jmp.op = Op::Jmp;
    jmp.blocks = {c.exit};
    pre->insts.push_back(jmp);

    // Uses after the loop see the values of the final header evaluation. Only header definitions can be used
    // outside the loop: the body does not dominate the exit.
    Replacements repl;
    for (const Inst& in : h->insts) {
        if (in.dst >= 0) repl[in.dst] = cur.at(in.dst);
    }
    for (Inst& in : c.exit->insts) {
        if (in.op != Op::Phi) break;
        for (BasicBlock*& blk : in.blocks) {
            if (blk == h) blk = pre;
        }
    }
    applyReplacements(fn, repl);
    fn.blocks.erase(std::remove_if(fn.blocks.begin(), fn.blocks.end(), [&](const std::unique_ptr<BasicBlock>& x) { return x.get() == h || x.get() == b; }),
                    fn.blocks.end());
    cfg::rebuild(fn);
}

}  // namespace

// Full unrolling of loops whose trip count is a compile-time constant, innermost first. The unrolled code has
// immediate induction values, so constfold, gvn and strength can simplify the address arithmetic, and ptxas
// sees one long basic block to schedule.
bool unroll(Function& fn) {
    bool changed = false;
    for (bool again = true; again;) {
        again = false;
        cfg::rebuild(fn);
        analysis::DomTree dt(fn);
        for (const analysis::Loop& loop : analysis::findLoops(fn, dt)) {
            if (std::optional<Candidate> c = match(loop)) {
                unrollLoop(fn, *c);
                changed = again = true;
                break;  // the loop list is stale now
            }
        }
    }
    return changed;
}

}  // namespace kite::transform
