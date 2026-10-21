#include "interp/interp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "ir/printer.h"

namespace kite::interp {

using namespace ir;

namespace {

struct Trap {  // undefined behaviour (or a malformed program) found while executing
    std::string reason;
};

enum class Status { Running, AtBarrier, Done };

struct Thread {
    std::vector<uint32_t> regs;
    const BasicBlock* cur = nullptr;
    const BasicBlock* prev = nullptr;
    size_t ip = 0;
    uint64_t steps = 0;
    Status status = Status::Running;
};

float asF32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
uint32_t bitsOf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}
int32_t asI32(uint32_t bits) { return static_cast<int32_t>(bits); }

// f32 operations are kept in separate functions so no expression can be contracted into an FMA.
float fadd(float a, float b) { return a + b; }
float fsub(float a, float b) { return a - b; }
float fmul(float a, float b) { return a * b; }
float fdiv(float a, float b) { return a / b; }

class Machine {
public:
    Machine(const Function& fn, const Launch& launch, const std::vector<uint32_t>& args, std::vector<Buffer>& buffers,
            uint64_t stepLimit)
        : fn_(fn), launch_(launch), args_(args), buffers_(buffers), stepLimit_(stepLimit) {}

    Result runGrid() {
        const int nBlocks = launch_.grid[0] * launch_.grid[1] * launch_.grid[2];
        for (blockLin_ = 0; blockLin_ < nBlocks; ++blockLin_) {
            std::string err = runBlock();
            if (!err.empty()) return {false, err};
        }
        return {};
    }

private:
    std::string runBlock() {
        const int nThreads = launch_.block[0] * launch_.block[1] * launch_.block[2];
        shared_.clear();
        for (const SharedArray& s : fn_.shared) shared_.emplace_back(static_cast<size_t>(s.size), 0u);
        threads_.assign(nThreads, Thread{});
        for (Thread& t : threads_) {
            t.regs.assign(fn_.numRegs(), 0u);
            for (size_t p = 0; p < args_.size(); ++p) t.regs[p] = args_[p];
            t.cur = fn_.blocks[0].get();
        }
        while (true) {
            for (threadLin_ = 0; threadLin_ < nThreads; ++threadLin_) {
                Thread& t = threads_[threadLin_];
                if (t.status != Status::Running) continue;
                try {
                    runThread(t);
                } catch (const Trap& trap) {
                    return describe(trap.reason, t);
                }
            }
            int waiting = 0, done = 0;
            for (const Thread& t : threads_) {
                waiting += t.status == Status::AtBarrier;
                done += t.status == Status::Done;
            }
            if (waiting == 0) return "";
            if (done > 0) {
                for (threadLin_ = 0; threadLin_ < nThreads; ++threadLin_) {
                    if (threads_[threadLin_].status != Status::AtBarrier) continue;
                    return describe("divergent barrier: " + std::to_string(waiting) + " of " + std::to_string(nThreads) +
                                        " threads reached sync(), the other " + std::to_string(done) + " returned",
                                    threads_[threadLin_], /*atPrevious=*/true);
                }
            }
            for (Thread& t : threads_) t.status = Status::Running;  // everyone arrived: release
        }
    }

    // `atPrevious`: the thread is parked just past its barrier, so report the barrier itself.
    std::string describe(const std::string& reason, const Thread& t, bool atPrevious = false) const {
        size_t ip = atPrevious ? t.ip - 1 : t.ip;
        const Inst& in = t.cur->insts[ip];
        return "error: " + reason + " in kernel " + fn_.name + ", block " + std::to_string(blockLin_) + ", thread " +
               std::to_string(threadLin_) + ", at " + t.cur->name() + ":" + std::to_string(ip) + ": " + toString(fn_, in);
    }

    uint32_t get(const Thread& t, const Operand& o) const {
        switch (o.kind) {
            case Operand::Reg: return t.regs[o.reg];
            case Operand::ImmI32: case Operand::ImmBool: return static_cast<uint32_t>(o.i);
            case Operand::ImmF32: return bitsOf(o.f);
            default: return 0;
        }
    }

    // Moves to `target`. Phis are parallel assignments: every incoming value is read before any is written.
    void enter(Thread& t, const BasicBlock* target) {
        t.prev = t.cur;
        t.cur = target;
        t.ip = 0;
        size_t nPhis = 0;
        while (nPhis < target->insts.size() && target->insts[nPhis].op == Op::Phi) ++nPhis;
        if (nPhis == 0) return;
        std::vector<uint32_t> values(nPhis);
        std::vector<bool> defined(nPhis, true);
        for (size_t k = 0; k < nPhis; ++k) {
            const Inst& phi = target->insts[k];
            size_t j = 0;
            while (j < phi.blocks.size() && phi.blocks[j] != t.prev) ++j;
            if (j == phi.blocks.size()) throw Trap{"phi has no value for predecessor " + t.prev->name()};
            if (phi.args[j].kind == Operand::Undef) defined[k] = false;  // no definition reaches: leave the register
            else values[k] = get(t, phi.args[j]);
        }
        for (size_t k = 0; k < nPhis; ++k) {
            if (defined[k]) t.regs[target->insts[k].dst] = values[k];
        }
        t.ip = nPhis;
        t.steps += nPhis;
    }

    void runThread(Thread& t) {
        while (true) {
            if (t.ip >= t.cur->insts.size()) throw Trap{"fell off the end of block " + t.cur->name()};
            const Inst& in = t.cur->insts[t.ip];
            if (++t.steps > stepLimit_) throw Trap{"step limit exceeded"};
            switch (in.op) {
                case Op::Br:
                    enter(t, in.blocks[get(t, in.args[0]) ? 0 : 1]);
                    continue;
                case Op::Jmp:
                    enter(t, in.blocks[0]);
                    continue;
                case Op::Ret:
                    t.status = Status::Done;
                    return;
                case Op::Barrier:
                    ++t.ip;
                    t.status = Status::AtBarrier;
                    return;
                case Op::Phi:
                    throw Trap{"phi not at the start of its block"};
                default:
                    exec(t, in);
                    ++t.ip;
            }
        }
    }

    void exec(Thread& t, const Inst& in) {
        auto a = [&](size_t k) { return get(t, in.args[k]); };
        const bool isF = in.ty == Ty::F32;
        uint32_t r = 0;
        switch (in.op) {
            case Op::Copy: r = a(0); break;
            case Op::Add: r = isF ? bitsOf(fadd(asF32(a(0)), asF32(a(1)))) : a(0) + a(1); break;
            case Op::Sub: r = isF ? bitsOf(fsub(asF32(a(0)), asF32(a(1)))) : a(0) - a(1); break;
            case Op::Mul: r = isF ? bitsOf(fmul(asF32(a(0)), asF32(a(1)))) : a(0) * a(1); break;
            case Op::Div:
                if (isF) {
                    r = bitsOf(fdiv(asF32(a(0)), asF32(a(1))));
                } else {
                    checkDivisor(asI32(a(0)), asI32(a(1)));
                    r = static_cast<uint32_t>(asI32(a(0)) / asI32(a(1)));
                }
                break;
            case Op::Rem:
                checkDivisor(asI32(a(0)), asI32(a(1)));
                r = static_cast<uint32_t>(asI32(a(0)) % asI32(a(1)));
                break;
            case Op::Neg: r = isF ? bitsOf(-asF32(a(0))) : 0u - a(0); break;
            case Op::Shl: r = a(1) > 31 ? 0u : a(0) << a(1); break;  // PTX shl.b32 semantics
            case Op::Eq: r = compare(in, [](auto x, auto y) { return x == y; }, t); break;
            case Op::Ne: r = compare(in, [](auto x, auto y) { return x != y; }, t); break;
            case Op::Lt: r = compare(in, [](auto x, auto y) { return x < y; }, t); break;
            case Op::Le: r = compare(in, [](auto x, auto y) { return x <= y; }, t); break;
            case Op::Gt: r = compare(in, [](auto x, auto y) { return x > y; }, t); break;
            case Op::Ge: r = compare(in, [](auto x, auto y) { return x >= y; }, t); break;
            case Op::And: r = a(0) & a(1); break;
            case Op::Or: r = a(0) | a(1); break;
            case Op::Xor: r = a(0) ^ a(1); break;
            case Op::Not: r = a(0) ^ 1u; break;
            case Op::IToF: r = bitsOf(static_cast<float>(asI32(a(0)))); break;
            case Op::FToI: {
                float f = asF32(a(0));
                if (!(f >= -2147483648.0f && f < 2147483648.0f)) throw Trap{"ftoi of out-of-range or NaN value"};
                r = static_cast<uint32_t>(static_cast<int32_t>(f));
                break;
            }
            case Op::BToI: r = a(0); break;
            case Op::Sqrt: r = bitsOf(std::sqrt(asF32(a(0)))); break;
            case Op::Exp: r = bitsOf(std::exp(asF32(a(0)))); break;
            case Op::Abs:
                r = isF ? bitsOf(std::fabs(asF32(a(0)))) : (asI32(a(0)) < 0 ? 0u - a(0) : a(0));
                break;
            case Op::Min:
                r = isF ? bitsOf(std::fmin(asF32(a(0)), asF32(a(1)))) : static_cast<uint32_t>(std::min(asI32(a(0)), asI32(a(1))));
                break;
            case Op::Max:
                r = isF ? bitsOf(std::fmax(asF32(a(0)), asF32(a(1)))) : static_cast<uint32_t>(std::max(asI32(a(0)), asI32(a(1))));
                break;
            case Op::Special: r = static_cast<uint32_t>(special(in)); break;
            case Op::Load: r = load(t, in); break;
            case Op::Store:
                store(t, in);
                return;
            default:
                throw Trap{std::string("cannot execute ") + opName(in.op)};
        }
        t.regs[in.dst] = r;
    }

    static void checkDivisor(int32_t a, int32_t b) {
        if (b == 0) throw Trap{"integer division by zero"};
        if (a == std::numeric_limits<int32_t>::min() && b == -1) throw Trap{"integer division overflow (INT_MIN / -1)"};
    }

    template <class F>
    uint32_t compare(const Inst& in, F cmp, const Thread& t) const {
        uint32_t x = get(t, in.args[0]), y = get(t, in.args[1]);
        if (in.ty == Ty::F32) return cmp(asF32(x), asF32(y));
        if (in.ty == Ty::I32) return cmp(asI32(x), asI32(y));
        return cmp(x, y);  // bool
    }

    int special(const Inst& in) const {
        const int d = in.dim;
        switch (in.sreg) {
            case SReg::Tid: {
                const int* b = launch_.block;
                int l = threadLin_;
                return d == 0 ? l % b[0] : d == 1 ? (l / b[0]) % b[1] : l / (b[0] * b[1]);
            }
            case SReg::Ntid: return launch_.block[d];
            case SReg::Ctaid: {
                const int* g = launch_.grid;
                int l = blockLin_;
                return d == 0 ? l % g[0] : d == 1 ? (l / g[0]) % g[1] : l / (g[0] * g[1]);
            }
            case SReg::Nctaid: return launch_.grid[d];
        }
        return 0;
    }

    // Resolves the memory an access refers to, and checks the index (UB #1).
    uint32_t* element(const Thread& t, const Inst& in) {
        const char* what = in.op == Op::Load ? "load" : "store";
        int32_t idx = asI32(get(t, in.args[1]));
        const Operand& base = in.args[0];
        std::vector<uint32_t>* mem;
        std::string name;
        if (in.space == Space::Shared) {
            mem = &shared_[base.i];
            name = fn_.shared[base.i].name;
        } else {
            uint32_t id = get(t, base);
            if (id >= buffers_.size()) throw Trap{"pointer does not refer to a buffer"};
            mem = &buffers_[id].data;
            name = fn_.regHint[base.reg].empty() ? fn_.regName(base.reg) : fn_.regHint[base.reg];
        }
        if (idx < 0 || static_cast<size_t>(idx) >= mem->size()) {
            throw Trap{std::string("out-of-bounds ") + what + " " + name + "[" + std::to_string(idx) + "] (size " +
                       std::to_string(mem->size()) + ")"};
        }
        return &(*mem)[idx];
    }

    uint32_t load(const Thread& t, const Inst& in) { return *element(t, in); }
    void store(const Thread& t, const Inst& in) { *element(t, in) = get(t, in.args[2]); }

    const Function& fn_;
    const Launch& launch_;
    const std::vector<uint32_t>& args_;
    std::vector<Buffer>& buffers_;
    uint64_t stepLimit_;
    std::vector<std::vector<uint32_t>> shared_;
    std::vector<Thread> threads_;
    int blockLin_ = 0;
    int threadLin_ = 0;
};

}  // namespace

Result run(const Function& fn, const Launch& launch, const std::vector<uint32_t>& args, std::vector<Buffer>& buffers,
           uint64_t stepLimit) {
    if (args.size() != static_cast<size_t>(fn.numParams)) {
        return {false, "error: kernel " + fn.name + " takes " + std::to_string(fn.numParams) + " arguments, got " +
                           std::to_string(args.size())};
    }
    for (int d = 0; d < 3; ++d) {
        if (launch.grid[d] < 1 || launch.block[d] < 1) return {false, "error: grid and block dimensions must be positive"};
    }
    return Machine(fn, launch, args, buffers, stepLimit).runGrid();
}

}  // namespace kite::interp
