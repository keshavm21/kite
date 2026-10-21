#include "ir/printer.h"

#include <cstdio>
#include <sstream>

namespace kite::ir {

namespace {

std::string floatLiteral(float v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
    std::string s = buf;
    if (s.find_first_of(".en") == std::string::npos) s += ".0";  // no '.', exponent, inf or nan
    return s;
}

}  // namespace

std::string toString(const Function& fn, const Operand& o) {
    switch (o.kind) {
        case Operand::Reg: return fn.regName(o.reg);
        case Operand::ImmI32: return std::to_string(o.i);
        case Operand::ImmF32: return floatLiteral(o.f);
        case Operand::ImmBool: return o.i ? "true" : "false";
        case Operand::Undef: return "undef";
        case Operand::SharedArr: return "@" + fn.shared[o.i].name;
    }
    return "?";
}

std::string toString(const Function& fn, const Inst& in) {
    std::ostringstream os;
    auto arg = [&](size_t k) { return toString(fn, in.args[k]); };
    if (in.dst >= 0) os << fn.regName(in.dst) << " = ";
    os << opName(in.op);
    switch (in.op) {
        case Op::IToF: case Op::FToI: case Op::BToI:
            os << ' ' << arg(0);
            break;
        case Op::Special:
            os << ' ' << tyName(in.ty) << ' ' << sregName(in.sreg) << '.' << "xyz"[in.dim];
            break;
        case Op::Phi:
            os << ' ' << tyName(in.ty);
            for (size_t k = 0; k < in.args.size(); ++k) {
                os << (k ? ", [" : " [") << arg(k) << ", " << in.blocks[k]->name() << ']';
            }
            break;
        case Op::Load:
            os << ' ' << tyName(in.ty) << (in.space == Space::Global ? " global " : " shared ") << arg(0) << '[' << arg(1) << ']';
            break;
        case Op::Store:
            os << ' ' << tyName(in.ty) << (in.space == Space::Global ? " global " : " shared ") << arg(0) << '[' << arg(1)
               << "], " << arg(2);
            break;
        case Op::Br:
            os << ' ' << arg(0) << ", " << in.blocks[0]->name() << ", " << in.blocks[1]->name();
            break;
        case Op::Jmp:
            os << ' ' << in.blocks[0]->name();
            break;
        case Op::Barrier: case Op::Ret:
            break;
        default:  // copy, arithmetic, compares, bool ops, intrinsics: `op T a[, b]`
            os << ' ' << tyName(in.ty);
            for (size_t k = 0; k < in.args.size(); ++k) os << (k ? ", " : " ") << arg(k);
            break;
    }
    return os.str();
}

void print(std::ostream& os, const Function& fn) {
    os << "kernel @" << fn.name << '(';
    for (int p = 0; p < fn.numParams; ++p) os << (p ? ", " : "") << tyName(fn.regTy[p]) << ' ' << fn.regName(p);
    os << ") {\n";
    for (const SharedArray& s : fn.shared) os << "  shared " << tyName(s.elemTy) << " @" << s.name << '[' << s.size << "]\n";
    for (const auto& bb : fn.blocks) {
        os << bb->name() << ":\n";
        for (const Inst& in : bb->insts) os << "  " << toString(fn, in) << '\n';
    }
    os << "}\n";
}

void print(std::ostream& os, const Module& m) {
    for (size_t k = 0; k < m.kernels.size(); ++k) {
        if (k) os << '\n';
        print(os, m.kernels[k]);
    }
}

std::string toString(const Function& fn) {
    std::ostringstream os;
    print(os, fn);
    return os.str();
}

}  // namespace kite::ir
