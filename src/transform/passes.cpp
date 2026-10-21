#include "transform/passes.h"

#include "analysis/cfg.h"
#include "transform/ssa.h"

namespace kite::transform {

using ir::Form;

namespace {

struct PassInfo {
    const char* name;
    int requires_;  // a Form, or -1 for any
    int produces;   // a Form, or -1 for unchanged
    void (*run)(ir::Function&);
};

const PassInfo kPasses[] = {
    {"remove-unreachable", -1, -1, [](ir::Function& fn) { cfg::removeUnreachable(fn); }},
    {"ssa", static_cast<int>(Form::Lowered), static_cast<int>(Form::SSA), constructSSA},
    {"out-of-ssa", static_cast<int>(Form::SSA), static_cast<int>(Form::PostSSA), destructSSA},
};

const char* formName(Form f) { return f == Form::Lowered ? "lowered" : f == Form::SSA ? "SSA" : "post-SSA"; }

}  // namespace

const std::vector<std::string>& o1Pipeline() {
    static const std::vector<std::string> p = {"remove-unreachable", "ssa", "out-of-ssa"};
    return p;
}

bool runPasses(ir::Module& module, const std::vector<std::string>& names, bool verifyEach, Form& form, std::string& err) {
    auto verifyAll = [&](const std::string& after) {
        for (const ir::Function& fn : module.kernels) {
            std::vector<std::string> problems = ir::verify(fn, form);
            if (problems.empty()) continue;
            err = "after " + after + ":\n";
            for (const std::string& p : problems) err += p + "\n";
            return false;
        }
        return true;
    };
    for (const std::string& name : names) {
        const PassInfo* pass = nullptr;
        for (const PassInfo& p : kPasses) {
            if (name == p.name) pass = &p;
        }
        if (!pass) {
            err = "unknown pass '" + name + "'";
            return false;
        }
        if (pass->requires_ >= 0 && static_cast<Form>(pass->requires_) != form) {
            err = "pass '" + name + "' needs " + formName(static_cast<Form>(pass->requires_)) + " IR but the IR is " + formName(form);
            return false;
        }
        for (ir::Function& fn : module.kernels) pass->run(fn);
        if (pass->produces >= 0) form = static_cast<Form>(pass->produces);
        if (verifyEach && !verifyAll("pass '" + name + "'")) return false;
    }
    return verifyEach || names.empty() ? true : verifyAll("the last pass");
}

}  // namespace kite::transform
