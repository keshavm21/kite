#pragma once
#include "ir/ir.h"

namespace kite::cfg {

// Recomputes every block's succs (the terminator's targets, in order, duplicates kept)
// and preds (one entry per incoming edge). Call after editing terminators.
void rebuild(ir::Function& fn);

}  // namespace kite::cfg
