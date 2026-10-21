#pragma once
#include <ostream>
#include <string>

#include "ir/ir.h"

namespace kite::ir {

// Text format of docs/IR.md section 5. This output is what the golden tests compare.
std::string toString(const Function& fn, const Operand& o);
std::string toString(const Function& fn, const Inst& inst);  // one instruction, no indentation
void print(std::ostream& os, const Function& fn);
void print(std::ostream& os, const Module& m);  // kernels separated by a blank line
std::string toString(const Function& fn);

}  // namespace kite::ir
