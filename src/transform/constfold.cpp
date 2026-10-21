#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>

#include "transform/opt.h"

namespace kite::transform {

using namespace ir;

namespace {

uint32_t bitsOf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

bool isImm(const Operand& o) { return o.kind == Operand::ImmI32 || o.kind == Operand::ImmF32 || o.kind == Operand::ImmBool; }
bool isI32(const Operand& o, int32_t v) { return o.kind == Operand::ImmI32 && o.i == v; }
bool isF32Bits(const Operand& o, uint32_t bits) { return o.kind == Operand::ImmF32 && bitsOf(o.f) == bits; }
bool isBool(const Operand& o, bool v) { return o.kind == Operand::ImmBool && (o.i != 0) == v; }

bool sameOperand(const Operand& a, const Operand& b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case Operand::Reg: return a.reg == b.reg;
        case Operand::ImmF32: return bitsOf(a.f) == bitsOf(b.f);
        case Operand::Undef: return false;
        default: return a.i == b.i;
    }
}

// A float result becomes an immediate only if it is a number: NaN payloads and signs differ between the
// host and the GPU, so a NaN is left to be computed at run time.
std::optional<Operand> floatResult(float f) {
    if (std::isnan(f)) return std::nullopt;
    return Operand::f32(f);
}

// Evaluates `in` when every operand is an immediate. Wraparound and the float rounding follow the
// interpreter exactly (i32 in uint32_t, f32 in float). Division by zero and INT_MIN / -1 are never folded.
std::optional<Operand> evaluate(const Inst& in, const std::vector<Operand>& a) {
    const bool isF = in.ty == Ty::F32;
    auto u = [&](size_t k) { return static_cast<uint32_t>(a[k].i); };
    auto i = [&](size_t k) { return a[k].i; };
    auto f = [&](size_t k) { return a[k].f; };
    auto fadd = [](float x, float y) { return x + y; };  // separate functions: no FMA contraction
    auto fsub = [](float x, float y) { return x - y; };
    auto fmul = [](float x, float y) { return x * y; };
    auto fdiv = [](float x, float y) { return x / y; };
    switch (in.op) {
        case Op::Copy: return a[0];
        case Op::Add: return isF ? floatResult(fadd(f(0), f(1))) : Operand::i32(static_cast<int32_t>(u(0) + u(1)));
        case Op::Sub: return isF ? floatResult(fsub(f(0), f(1))) : Operand::i32(static_cast<int32_t>(u(0) - u(1)));
        case Op::Mul: return isF ? floatResult(fmul(f(0), f(1))) : Operand::i32(static_cast<int32_t>(u(0) * u(1)));
        case Op::Div:
            if (isF) return floatResult(fdiv(f(0), f(1)));
            if (i(1) == 0 || (i(0) == std::numeric_limits<int32_t>::min() && i(1) == -1)) return std::nullopt;
            return Operand::i32(i(0) / i(1));
        case Op::Rem:
            if (i(1) == 0 || (i(0) == std::numeric_limits<int32_t>::min() && i(1) == -1)) return std::nullopt;
            return Operand::i32(i(0) % i(1));
        case Op::Neg: return isF ? floatResult(-f(0)) : Operand::i32(static_cast<int32_t>(0u - u(0)));
        case Op::Shl: return Operand::i32(static_cast<int32_t>(u(1) > 31 ? 0u : u(0) << u(1)));  // PTX shl.b32
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge: {
            auto cmp = [&](auto x, auto y) {
                switch (in.op) {
                    case Op::Eq: return x == y;
                    case Op::Ne: return x != y;
                    case Op::Lt: return x < y;
                    case Op::Le: return x <= y;
                    case Op::Gt: return x > y;
                    default: return x >= y;
                }
            };
            if (in.ty == Ty::F32) return Operand::boolean(cmp(f(0), f(1)));
            return Operand::boolean(cmp(i(0), i(1)));  // i32 and bool (0/1) compare the same way
        }
        case Op::And: return Operand::boolean(i(0) && i(1));
        case Op::Or: return Operand::boolean(i(0) || i(1));
        case Op::Xor: return Operand::boolean((i(0) != 0) != (i(1) != 0));
        case Op::Not: return Operand::boolean(!i(0));
        case Op::IToF: return Operand::f32(static_cast<float>(i(0)));
        case Op::FToI:
            if (!(f(0) >= -2147483648.0f && f(0) < 2147483648.0f)) return std::nullopt;  // out of range or NaN: run-time UB
            return Operand::i32(static_cast<int32_t>(f(0)));
        case Op::BToI: return Operand::i32(i(0));
        case Op::Sqrt: return floatResult(std::sqrt(f(0)));
        case Op::Abs: return isF ? Operand::f32(std::fabs(f(0))) : Operand::i32(i(0) < 0 ? static_cast<int32_t>(0u - u(0)) : i(0));
        case Op::Min: return isF ? floatResult(std::fmin(f(0), f(1))) : Operand::i32(std::min(i(0), i(1)));
        case Op::Max: return isF ? floatResult(std::fmax(f(0), f(1))) : Operand::i32(std::max(i(0), i(1)));
        default: return std::nullopt;  // exp (the GPU's is approximate), special, loads: never folded
    }
}

// Rewrites that hold for every input including NaN, -0.0 and wraparound. f32 `x + 0.0` and `x * 0.0` are
// deliberately absent: -0.0 + 0.0 is +0.0, and NaN * 0.0 is NaN.
std::optional<Operand> simplify(const Inst& in, const std::vector<Operand>& a) {
    switch (in.op) {
        case Op::Add:
            if (in.ty != Ty::I32) return std::nullopt;
            if (isI32(a[1], 0)) return a[0];
            if (isI32(a[0], 0)) return a[1];
            return std::nullopt;
        case Op::Sub:
            if (in.ty == Ty::I32) {
                if (isI32(a[1], 0)) return a[0];
                if (a[0].isReg() && a[1].isReg() && a[0].reg == a[1].reg) return Operand::i32(0);
            } else if (isF32Bits(a[1], 0)) {  // x - (+0.0) == x, including x == -0.0
                return a[0];
            }
            return std::nullopt;
        case Op::Mul:
            if (in.ty == Ty::I32) {
                if (isI32(a[1], 1)) return a[0];
                if (isI32(a[0], 1)) return a[1];
                if (isI32(a[0], 0) || isI32(a[1], 0)) return Operand::i32(0);
            } else {
                if (isF32Bits(a[1], bitsOf(1.0f))) return a[0];
                if (isF32Bits(a[0], bitsOf(1.0f))) return a[1];
            }
            return std::nullopt;
        case Op::Div:
            if (in.ty == Ty::I32 ? isI32(a[1], 1) : isF32Bits(a[1], bitsOf(1.0f))) return a[0];
            return std::nullopt;
        case Op::Shl:
            if (isI32(a[1], 0)) return a[0];
            return std::nullopt;
        case Op::And:
            if (isBool(a[1], true)) return a[0];
            if (isBool(a[0], true)) return a[1];
            if (isBool(a[0], false) || isBool(a[1], false)) return Operand::boolean(false);
            return std::nullopt;
        case Op::Or:
            if (isBool(a[1], false)) return a[0];
            if (isBool(a[0], false)) return a[1];
            if (isBool(a[0], true) || isBool(a[1], true)) return Operand::boolean(true);
            return std::nullopt;
        default:
            return std::nullopt;
    }
}

}  // namespace

bool constfold(Function& fn) {
    bool any = false;
    for (bool progress = true; progress;) {
        progress = false;
        Replacements repl;
        std::set<const Inst*> dead;
        auto resolve = [&](Operand o) {
            for (auto it = repl.find(o.reg); o.isReg() && it != repl.end(); it = repl.find(o.reg)) o = it->second;
            return o;
        };
        for (auto& bb : fn.blocks) {
            for (const Inst& in : bb->insts) {
                if (in.dst < 0 || !in.isPure()) continue;
                std::optional<Operand> r;
                if (in.op == Op::Phi) {
                    // All incoming values equal (ignoring the phi itself): the phi is that value.
                    bool ok = true;
                    std::optional<Operand> only;
                    for (const Operand& raw : in.args) {
                        Operand o = resolve(raw);
                        if (o.isReg() && o.reg == in.dst) continue;
                        if (o.kind == Operand::Undef || (only && !sameOperand(*only, o))) ok = false;
                        else only = o;
                    }
                    if (ok && only) r = only;
                } else {
                    std::vector<Operand> a;
                    bool allImm = true;
                    for (const Operand& raw : in.args) {
                        a.push_back(resolve(raw));
                        allImm &= isImm(a.back());
                    }
                    r = allImm ? evaluate(in, a) : simplify(in, a);
                }
                if (!r) continue;
                repl[in.dst] = *r;
                dead.insert(&in);
            }
        }
        if (repl.empty()) break;
        applyReplacements(fn, repl);
        for (auto& bb : fn.blocks) {
            auto& v = bb->insts;
            v.erase(std::remove_if(v.begin(), v.end(), [&](const Inst& in) { return dead.count(&in) != 0; }), v.end());
        }
        progress = any = true;
    }
    return any;
}

}  // namespace kite::transform
