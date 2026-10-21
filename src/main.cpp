#include <fstream>
#include <iostream>
#include <string>

#include "frontend/ast_printer.h"
#include "frontend/lexer.h"
#include "frontend/parser.h"
#include "frontend/sema.h"
#include "support/diagnostics.h"
#include "support/source.h"

namespace {

void usage() {
    std::cerr << "usage: kitec FILE.kite [-o OUT] [-O0|-O1] [--emit=tokens|ast]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string inPath, outPath, emit;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) outPath = argv[++i];
        else if (a == "-O0" || a == "-O1") continue;  // accepted now, used once the optimizer exists
        else if (a.rfind("--emit=", 0) == 0) emit = a.substr(7);
        else if (!a.empty() && a[0] != '-' && inPath.empty()) inPath = a;
        else {
            usage();
            return 2;
        }
    }
    if (inPath.empty() || (!emit.empty() && emit != "tokens" && emit != "ast")) {
        usage();
        return 2;
    }

    std::string src;
    if (!kite::readFile(inPath, src)) {
        std::cerr << "kitec: cannot read '" << inPath << "'\n";
        return 1;
    }

    std::ofstream file;
    if (!outPath.empty()) {
        file.open(outPath);
        if (!file) {
            std::cerr << "kitec: cannot write '" << outPath << "'\n";
            return 1;
        }
    }
    std::ostream& out = outPath.empty() ? std::cout : file;

    kite::Diagnostics diag(inPath);

    if (emit == "tokens") {
        kite::Lexer lexer(src, diag);
        for (kite::Token t = lexer.next(); t.kind != kite::Tok::Eof; t = lexer.next()) {
            out << t.loc.line << ':' << t.loc.col << ' ' << kite::tokName(t.kind) << " '" << t.text << "'\n";
        }
        return diag.hasErrors() ? 1 : 0;
    }

    auto program = kite::parseSource(src, diag);
    if (!program) return 1;

    kite::Sema sema(diag);
    sema.check(*program);
    if (diag.hasErrors()) return 1;

    if (emit == "ast") {
        out << kite::dumpProgram(*program);
        return 0;
    }

    std::cerr << "kitec: code generation is not implemented yet (Milestone 4); use --emit=tokens or --emit=ast\n";
    return 1;
}
