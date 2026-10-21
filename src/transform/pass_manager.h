#pragma once
#include <ostream>
#include <string>
#include <vector>

#include "ir/ir.h"
#include "ir/verifier.h"

namespace kite::transform {

// One pipeline step: the named passes run in order. With maxRounds > 1 the whole group repeats until a
// round changes nothing (or maxRounds is reached).
struct Step {
    std::vector<std::string> passes;
    int maxRounds = 1;
};

// The -O1 pipeline (LICM and strength reduction join in Milestone 8):
//   remove-unreachable, ssa, { constfold copyprop gvn dce simplifycfg } until stable (max 4 rounds),
//   constfold, dce, simplifycfg, out-of-ssa
std::vector<Step> o1Pipeline();
std::vector<Step> flatPipeline(const std::vector<std::string>& names);  // each name is a single step
std::vector<std::string> passNames();

struct Options {
    bool verifyEach = false;
    bool printAfterAll = false;  // dump every kernel's IR to `log` after every pass
    bool collectStats = false;   // record the instruction count of every kernel after every pass
};

class PassManager {
public:
    explicit PassManager(Options opts, std::ostream* log = nullptr) : opts_(opts), log_(log) {}

    // Runs `steps` over every kernel. Tracks the IR form (lowered -> SSA -> post-SSA) and refuses a pass whose
    // input form is wrong. The verifier runs after every pass with verifyEach, else once at the end.
    // On failure returns false and fills `err` (starting with "verifier:" for verifier output).
    // `form` is the form on entry and is updated.
    bool run(ir::Module& module, const std::vector<Step>& steps, ir::Form& form, std::string& err);

    // A table of instruction counts: one row per pass invocation, one column per kernel.
    void printStats(std::ostream& os) const;

private:
    struct Row {
        std::string label;
        std::vector<size_t> counts;
        bool changed;
    };
    void record(const std::string& label, const ir::Module& m, bool changed);

    Options opts_;
    std::ostream* log_;
    std::vector<std::string> kernelNames_;
    std::vector<Row> rows_;
};

}  // namespace kite::transform
