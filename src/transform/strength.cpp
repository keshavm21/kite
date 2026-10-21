#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

namespace {

// k if v == 2^k and k >= 1, else -1.
int log2Exact(int32_t v) {
    if (v <= 1 || (v & (v - 1)) != 0) return -1;
    int k = 0;
    while ((1 << k) != v) ++k;
    return k;
}

}  // namespace

// `mul i32 x, 2^k` -> `shl i32 x, k` (either operand order). Wraparound is identical: both compute the low
// 32 bits of x * 2^k. A signed `div` by 2^k is NOT rewritten: div rounds toward zero (-7 / 2 == -3) but an
// arithmetic shift rounds toward minus infinity (-7 >> 1 == -4), and Kite has no unsigned shift to fix it up.
bool strength(Function& fn) {
    bool changed = false;
    for (auto& bb : fn.blocks) {
        for (Inst& in : bb->insts) {
            if (in.op != Op::Mul || in.ty != Ty::I32) continue;
            for (int c = 0; c < 2; ++c) {
                if (in.args[c].kind != Operand::ImmI32 || in.args[1 - c].kind == Operand::ImmI32) continue;
                int k = log2Exact(in.args[c].i);
                if (k < 0) continue;
                Operand x = in.args[1 - c];
                in.op = Op::Shl;
                in.args = {x, Operand::i32(k)};
                changed = true;
                break;
            }
        }
    }
    return changed;
}

}  // namespace kite::transform
