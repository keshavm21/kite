#pragma once
#include <optional>
#include <string>

#include "frontend/ast.h"
#include "support/diagnostics.h"

namespace kite {

// Parses a whole source file. The parser stops at the first error (no recovery): it
// reports through `diag` and returns nullopt. Lexical errors are reported by the lexer.
std::optional<Program> parseSource(const std::string& src, Diagnostics& diag);

}  // namespace kite
