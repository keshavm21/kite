#include "transform/pass_manager.h"

#include <iomanip>

#include "analysis/cfg.h"
#include "ir/printer.h"
#include "transform/opt.h"
#include "transform/ssa.h"

namespace kite::transform {

using ir::Form;

namespace {

struct PassInfo {
    const char* name;
    int requires_;  // a Form, or -1 for any
    int produces;   // a Form, or -1 for unchanged
    bool (*run)(ir::Function&);
};

constexpr int L = static_cast<int>(Form::Lowered), S = static_cast<int>(Form::SSA), P = static_cast<int>(Form::PostSSA);

const PassInfo kPasses[] = {
    {"remove-unreachable", -1, -1, [](ir::Function& fn) { return cfg::removeUnreachable(fn); }},
    {"ssa", L, S, [](ir::Function& fn) { constructSSA(fn); return true; }},
    {"out-of-ssa", S, P, [](ir::Function& fn) { destructSSA(fn); return true; }},
    {"constfold", S, -1, constfold},
    {"copyprop", S, -1, copyprop},
    {"gvn", S, -1, gvn},
    {"licm", S, -1, licm},
    {"strength", -1, -1, strength},
    {"dce", -1, -1, dce},
    {"simplifycfg", -1, -1, simplifycfg},
};

const char* formName(Form f) { return f == Form::Lowered ? "lowered" : f == Form::SSA ? "SSA" : "post-SSA"; }

}  // namespace

std::vector<std::string> passNames() {
    std::vector<std::string> names;
    for (const PassInfo& p : kPasses) names.push_back(p.name);
    return names;
}

std::vector<Step> o1Pipeline() {
    return {{{"remove-unreachable", "ssa"}, 1},
            {{"constfold", "copyprop", "gvn", "dce", "simplifycfg"}, 4},
            {{"licm", "gvn", "strength", "constfold", "dce", "simplifycfg", "out-of-ssa"}, 1}};
}

std::vector<Step> flatPipeline(const std::vector<std::string>& names) {
    std::vector<Step> steps;
    for (const std::string& n : names) steps.push_back({{n}, 1});
    return steps;
}

void PassManager::record(const std::string& label, const ir::Module& m, bool changed) {
    if (!opts_.collectStats) return;
    kernelNames_.clear();
    Row row{label, {}, changed};
    for (const ir::Function& fn : m.kernels) {
        kernelNames_.push_back(fn.name);
        row.counts.push_back(instructionCount(fn));
    }
    rows_.push_back(std::move(row));
}

bool PassManager::run(ir::Module& module, const std::vector<Step>& steps, Form& form, std::string& err) {
    auto verifyAll = [&](const std::string& after) {
        for (const ir::Function& fn : module.kernels) {
            std::vector<std::string> problems = ir::verify(fn, form);
            if (problems.empty()) continue;
            err = "after " + after + ":\n";
            for (const std::string& p : problems) err += p + "\n";
            err = "verifier: " + err;
            return false;
        }
        return true;
    };
    licmTotals() = {};
    record("lowered", module, false);
    bool verified = true;  // is the current IR known to verify?
    for (const Step& step : steps) {
        for (int round = 1; round <= step.maxRounds; ++round) {
            bool anyChange = false;
            for (const std::string& name : step.passes) {
                const PassInfo* pass = nullptr;
                for (const PassInfo& p : kPasses) {
                    if (name == p.name) pass = &p;
                }
                if (!pass) {
                    err = "unknown pass '" + name + "' (known: ";
                    for (const std::string& n : passNames()) err += n + (n == passNames().back() ? ")" : ", ");
                    return false;
                }
                if (pass->requires_ >= 0 && static_cast<Form>(pass->requires_) != form) {
                    err = "pass '" + name + "' needs " + formName(static_cast<Form>(pass->requires_)) + " IR but the IR is " + formName(form);
                    return false;
                }
                bool changed = false;
                for (ir::Function& fn : module.kernels) changed |= pass->run(fn);
                if (pass->produces >= 0) form = static_cast<Form>(pass->produces);
                anyChange |= changed && pass->produces < 0;  // form changes do not count as "still optimizing"
                verified = false;
                std::string label = name + (step.maxRounds > 1 ? " (round " + std::to_string(round) + ")" : "");
                record(label, module, changed);
                if (opts_.verifyEach && !verifyAll("pass '" + label + "'")) return false;
                verified = opts_.verifyEach;
                if (opts_.printAfterAll && log_) {
                    *log_ << "; --- after " << label << " ---\n";
                    ir::print(*log_, module);
                }
            }
            if (!anyChange) break;
        }
    }
    return verified || verifyAll("the last pass");
}

void PassManager::printStats(std::ostream& os) const {
    if (rows_.empty()) return;
    size_t w = 8;
    for (const Row& r : rows_) w = std::max(w, r.label.size());
    os << "instruction count after each pass\n" << std::left << std::setw(w + 2) << "pass";
    for (const std::string& k : kernelNames_) os << std::right << std::setw(std::max<size_t>(k.size(), 7) + 2) << k;
    os << "  changed\n";
    for (const Row& r : rows_) {
        os << std::left << std::setw(w + 2) << r.label;
        for (size_t k = 0; k < r.counts.size(); ++k) os << std::right << std::setw(std::max<size_t>(kernelNames_[k].size(), 7) + 2) << r.counts[k];
        os << "  " << (r.changed ? "yes" : "-") << '\n';
    }
    const LicmStats& l = licmTotals();
    if (l.loops > 0) os << "licm: " << l.loops << " loop(s), " << l.hoisted << " instruction(s) hoisted, " << l.skipped << " skipped (no preheader)\n";
}

}  // namespace kite::transform
