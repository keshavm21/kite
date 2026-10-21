#pragma once
#include <string>

#include "frontend/ast.h"

namespace kite {

// Deterministic text forms used by --emit=ast and by the golden and unit tests.
// Statements print as an indented tree; expressions print inline as S-expressions,
// e.g. (+ 1 (* 2 3)), so precedence and associativity are visible at a glance.
std::string dumpProgram(const Program& program);
std::string exprToString(const Expr& e);

}  // namespace kite
