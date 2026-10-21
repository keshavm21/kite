#include "support/diagnostics.h"

namespace kite {

void Diagnostics::error(SourceLoc loc, const std::string& msg) {
    ++errors_;
    out_ << filename_ << ':' << loc.line << ':' << loc.col << ": error: " << msg << '\n';
}

}  // namespace kite
