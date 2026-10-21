#pragma once
#include <string>
#include <vector>

#include "ir/ir.h"
#include "ir/verifier.h"

namespace kite::transform {

// The names accepted by `--passes=` (comma separated), in the order a -O1 pipeline runs them.
const std::vector<std::string>& o1Pipeline();

// Runs the named passes over every kernel, in order. Tracks the IR form (lowered -> SSA -> post-SSA) and
// refuses a pass whose input form is wrong. With `verifyEach` the verifier runs after every pass, otherwise
// once at the end. On failure returns false and fills `err` (verifier output or a message).
// `form` is the form of the IR on entry and is updated to the form on exit.
bool runPasses(ir::Module& module, const std::vector<std::string>& names, bool verifyEach, ir::Form& form,
               std::string& err);

}  // namespace kite::transform
