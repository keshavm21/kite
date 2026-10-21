#pragma once
#include <iostream>
#include <string>

#include "support/source.h"

namespace kite {

// Collects and prints "file:line:col: error: message". Output goes to `out` (stderr by
// default) so unit tests can capture it with an ostringstream.
class Diagnostics {
public:
    explicit Diagnostics(std::string filename, std::ostream& out = std::cerr)
        : filename_(std::move(filename)), out_(out) {}

    void error(SourceLoc loc, const std::string& msg);
    bool hasErrors() const { return errors_ > 0; }
    int errorCount() const { return errors_; }

private:
    std::string filename_;
    std::ostream& out_;
    int errors_ = 0;
};

}  // namespace kite
