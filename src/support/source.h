#pragma once
#include <string>

namespace kite {

// 1-based position in a source file; every token, AST node and diagnostic carries one.
struct SourceLoc {
    int line = 1;
    int col = 1;
};

// Reads a whole file into `out`. Returns false if it cannot be opened.
bool readFile(const std::string& path, std::string& out);

}  // namespace kite
