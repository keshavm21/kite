#pragma once
#include "frontend/ast.h"
#include "ir/ir.h"

namespace kite::ir {

// AST -> lowered IR (docs/IR.md sections 7-8, IMPLEMENTATION_PLAN.md 3.2). The program must have
// passed Sema: lowering relies on Expr::type and the Symbol* annotations.
Function lowerKernel(const Kernel& kernel);
Module lowerProgram(const Program& program);

}  // namespace kite::ir
