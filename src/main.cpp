#include <fstream>
#include <iostream>
#include <string>

#include "frontend/ast_printer.h"
#include "frontend/lexer.h"
#include "frontend/parser.h"
#include "frontend/sema.h"
#include "interp/interp.h"
#include "interp/spec.h"
#include "ir/lower.h"
#include "ir/printer.h"
#include "ir/verifier.h"
#include "support/diagnostics.h"
#include "support/source.h"

namespace {

void usage() {
    std::cerr << "usage: kitec FILE.kite [-o OUT] [-O0|-O1] [--emit=tokens|ast|ir] [--verify-each]\n"
                 "       kitec run FILE.kite [-O0|-O1] [--verify-each]\n";
}

// Verifies every kernel; prints the problems and returns false if there are any.
bool verifyModule(const kite::ir::Module& m, kite::ir::Form form) {
    bool ok = true;
    for (const kite::ir::Function& fn : m.kernels) {
        for (const std::string& problem : kite::ir::verify(fn, form)) {
            std::cerr << problem << '\n';
            ok = false;
        }
    }
    return ok;
}

// `kitec run`: launches the kernel named by the @spec header on the interpreter and prints the @check buffers.
int runSpec(const std::string& src, const kite::ir::Module& module, std::ostream& out) {
    kite::interp::Spec spec;
    std::string err;
    if (!kite::interp::parseSpec(src, spec, err)) {
        std::cerr << "kitec: bad @spec: " << err << '\n';
        return 1;
    }
    const kite::ir::Function* fn = nullptr;
    for (const kite::ir::Function& k : module.kernels) {
        if (k.name == spec.kernel) fn = &k;
    }
    if (!fn) {
        std::cerr << "kitec: no kernel named '" << spec.kernel << "'\n";
        return 1;
    }
    kite::interp::Inputs inputs;
    if (!kite::interp::checkSignature(spec, *fn, err) || !kite::interp::makeInputs(spec, inputs, err)) {
        std::cerr << "kitec: " << err << '\n';
        return 1;
    }
    kite::interp::Launch launch;
    for (int d = 0; d < 3; ++d) {
        launch.grid[d] = spec.grid[d];
        launch.block[d] = spec.block[d];
    }
    kite::interp::Result r = kite::interp::run(*fn, launch, inputs.args, inputs.buffers);
    if (!r.ok) {
        std::cerr << r.error << '\n';
        return 1;
    }
    out << kite::interp::formatOutputs(spec, inputs);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string inPath, outPath, emit;
    bool runMode = false;
    int optLevel = 0;  // -O1 is accepted now and takes effect once the pass manager exists (Milestone 5)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (i == 1 && a == "run") runMode = true;
        else if (a == "-o" && i + 1 < argc) outPath = argv[++i];
        else if (a == "-O0") optLevel = 0;
        else if (a == "-O1") optLevel = 1;
        else if (a == "--verify-each") continue;  // the IR is verified after lowering always; passes arrive in Milestone 5
        else if (a.rfind("--emit=", 0) == 0) emit = a.substr(7);
        else if (!a.empty() && a[0] != '-' && inPath.empty()) inPath = a;
        else {
            usage();
            return 2;
        }
    }
    (void)optLevel;
    if (inPath.empty() || (!emit.empty() && emit != "tokens" && emit != "ast" && emit != "ir") ||
        (runMode && !emit.empty())) {
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

    kite::ir::Module module = kite::ir::lowerProgram(*program);
    if (!verifyModule(module, kite::ir::Form::Lowered)) return 3;  // a compiler bug, not a user error

    if (runMode) return runSpec(src, module, out);
    if (emit == "ir") {
        kite::ir::print(out, module);
        return 0;
    }

    std::cerr << "kitec: PTX generation is not implemented yet (Milestone 4); use --emit=tokens, ast or ir, or `kitec run`\n";
    return 1;
}
