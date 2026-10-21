#include "support/diagnostics.h"

namespace kite {

void Diagnostics::error(SourceLoc loc, const std::string& msg) {
    if (!seen_.insert({loc.line, loc.col, msg}).second) return;
    ++errors_;
    out_ << filename_ << ':' << loc.line << ':' << loc.col << ": error: " << msg << '\n';
}

}  // namespace kite
