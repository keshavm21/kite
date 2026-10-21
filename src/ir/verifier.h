#pragma once
#include <string>
#include <vector>

#include "ir/ir.h"

namespace kite::ir {

enum class Form { Lowered, SSA, PostSSA };

// Checks the invariants of docs/IR.md section 6 and returns one message per violation, formatted
// `verifier: kernel saxpy, block if.then.1, inst 2: <problem>`. Empty means the function is valid.
// Phis are rejected outside Form::SSA. Form::SSA also checks invariants 7-10 (single definition, dominance, phi shape).
std::vector<std::string> verify(const Function& fn, Form form = Form::Lowered);

}  // namespace kite::ir
