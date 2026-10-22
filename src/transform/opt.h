#pragma once
#include <unordered_map>

#include "ir/ir.h"

// The optimization passes (docs/IMPLEMENTATION_PLAN.md Milestone 6). Each returns true if it changed
// the function. constfold, copyprop and gvn need SSA form; dce and simplifycfg work on any form.
namespace kite::transform {

bool constfold(ir::Function& fn);     // folds immediates (f32 semantics respected), algebraic identities, trivial phis
bool copyprop(ir::Function& fn);      // replaces uses of `d = copy a` with `a`
bool dce(ir::Function& fn);           // mark and sweep from stores, barriers and terminators
bool simplifycfg(ir::Function& fn);   // constant branches, br c,X,X, unreachable blocks, block merging
bool gvn(ir::Function& fn);           // dominator-scoped value numbering of pure instructions
bool licm(ir::Function& fn);          // hoists loop-invariant pure instructions to the preheader (SSA)
bool unroll(ir::Function& fn);        // fully unrolls small loops with a constant trip count (SSA)
bool strength(ir::Function& fn);      // mul i32 x, 2^k -> shl i32 x, k

// What licm has done so far in this process (summed over functions); shown by --stats.
struct LicmStats {
    int loops = 0, hoisted = 0, skipped = 0;  // skipped: loops without a usable preheader
};
LicmStats& licmTotals();

// Helpers shared by the passes.
using Replacements = std::unordered_map<int, ir::Operand>;  // register -> what its uses become
// Rewrites every use of a replaced register (following chains) in every instruction, phis included.
void applyReplacements(ir::Function& fn, const Replacements& repl);
size_t instructionCount(const ir::Function& fn);

}  // namespace kite::transform
