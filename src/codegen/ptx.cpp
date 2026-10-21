#include "codegen/ptx.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace kite::codegen {

using namespace ir;

namespace {

const char* kPrefix[] = {"p", "r", "f", "rd"};
const char* kDecl[] = {".pred", ".b32", ".f32", ".b64"};

std::string floatImm(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    char buf[16];
    std::snprintf(buf, sizeof buf, "0f%08X", bits);
    return buf;
}

const char* suffix(Ty t) { return t == Ty::F32 ? "f32" : "s32"; }

}  // namespace

PtxEmitter::Cls PtxEmitter::clsOf(Ty t) {
    switch (t) {
        case Ty::Bool: return Pred;
        case Ty::F32: return F32;
        case Ty::PtrF32: case Ty::PtrI32: return B64;
        default: return B32;
    }
}

std::string PtxEmitter::regName(Ty t, int id) {
    Cls c = clsOf(t);
    if (id > maxId_[c]) maxId_[c] = id;
    return std::string("%") + kPrefix[c] + std::to_string(id);
}

std::string PtxEmitter::reg(int id) { return regName(fn_->regTy[id], id); }

std::string PtxEmitter::newTemp(Ty t) { return regName(t, nextTemp_++); }

std::string PtxEmitter::src(const Operand& o) {
    switch (o.kind) {
        case Operand::Reg: return reg(o.reg);
        case Operand::ImmI32: return std::to_string(o.i);
        case Operand::ImmF32: return floatImm(o.f);
        case Operand::ImmBool: return o.i ? "1" : "0";
        default: throw std::logic_error("ptx: operand has no PTX form");
    }
}

std::string PtxEmitter::regOf(const Operand& o, Ty ty) {
    if (o.isReg()) return reg(o.reg);
    std::string t = newTemp(ty);
    line(ty == Ty::F32 ? "mov.f32" : ty == Ty::Bool ? "mov.pred" : "mov.u32", t + ", " + src(o));
    return t;
}

std::string PtxEmitter::label(const BasicBlock& bb) const {
    std::string s = bb.name();
    for (char& c : s) {
        if (c == '.') c = '_';
    }
    return "$L_" + s;
}

void PtxEmitter::line(const std::string& mnemonic, const std::string& operands) {
    body_ << "    " << mnemonic;
    if (!operands.empty()) {
        body_ << std::string(mnemonic.size() < 20 ? 20 - mnemonic.size() : 1, ' ') << operands;
    }
    body_ << ";\n";
}

std::string PtxEmitter::emit() {
    out_.str("");
    out_ << ".version 7.0\n.target " << target_ << "\n.address_size 64\n";
    for (const Function& fn : module_.kernels) emitKernel(fn);
    return out_.str();
}

void PtxEmitter::emitKernel(const Function& fn) {
    fn_ = &fn;
    body_.str("");
    for (int& m : maxId_) m = -1;
    nextTemp_ = fn.numRegs();
    sharedBase_.clear();

    // Prologue: parameters, then the address of each shared array.
    std::string sig;
    for (int p = 0; p < fn.numParams; ++p) {
        Ty t = fn.regTy[p];
        const char* ptype = t == Ty::F32 ? ".f32" : isPointer(t) ? ".u64" : ".u32";
        sig += std::string(p ? ",\n" : "") + "    .param " + ptype + " " + fn.name + "_param_" + std::to_string(p);
        std::string param = "[" + fn.name + "_param_" + std::to_string(p) + "]";
        if (t == Ty::F32) {
            line("ld.param.f32", reg(p) + ", " + param);
        } else if (isPointer(t)) {
            line("ld.param.u64", reg(p) + ", " + param);
            line("cvta.to.global.u64", reg(p) + ", " + reg(p));
        } else if (t == Ty::Bool) {  // the host passes 0/1 as a u32
            std::string tmp = newTemp(Ty::I32);
            line("ld.param.u32", tmp + ", " + param);
            line("setp.ne.s32", reg(p) + ", " + tmp + ", 0");
        } else {
            line("ld.param.u32", reg(p) + ", " + param);
        }
    }
    for (const SharedArray& s : fn.shared) {
        sharedBase_.push_back(newTemp(Ty::I32));
        line("mov.u32", sharedBase_.back() + ", " + fn.name + "_" + s.name);
    }

    for (size_t k = 0; k < fn.blocks.size(); ++k) {
        emitBlock(*fn.blocks[k], k + 1 < fn.blocks.size() ? fn.blocks[k + 1].get() : nullptr);
    }

    // Shared arrays live at module scope, named <kernel>_<array>.
    if (!fn.shared.empty()) out_ << '\n';
    for (const SharedArray& s : fn.shared) {
        out_ << ".shared .align 4 " << (s.elemTy == Ty::F32 ? ".f32 " : ".b32 ") << fn.name << '_' << s.name << '[' << s.size << "];\n";
    }
    out_ << "\n.visible .entry " << fn.name << "(\n" << sig << "\n)\n{\n";
    for (int c = 0; c < NumClasses; ++c) {
        if (maxId_[c] < 0) continue;
        char decl[64];
        std::snprintf(decl, sizeof decl, "    .reg %-6s %%%s<%d>;\n", kDecl[c], kPrefix[c], maxId_[c] + 1);
        out_ << decl;
    }
    out_ << '\n' << body_.str() << "}\n";
}

void PtxEmitter::emitBlock(const BasicBlock& bb, const BasicBlock* next) {
    body_ << '\n' << label(bb) << ":\n";
    for (const Inst& in : bb.insts) emitInst(in, next);
}

void PtxEmitter::emitBranch(const Inst& in, const BasicBlock* next) {
    if (in.op == Op::Jmp) {
        if (in.blocks[0] != next) line("bra.uni", label(*in.blocks[0]));
        return;
    }
    const BasicBlock *t = in.blocks[0], *f = in.blocks[1];
    const Operand& c = in.args[0];
    if (!c.isReg()) {  // constant condition
        const BasicBlock* taken = c.i ? t : f;
        if (taken != next) line("bra.uni", label(*taken));
        return;
    }
    std::string p = reg(c.reg);
    if (t == f) {
        if (t != next) line("bra.uni", label(*t));
    } else if (f == next) {
        line("@" + p + " bra", label(*t));
    } else if (t == next) {
        line("@!" + p + " bra", label(*f));
    } else {
        line("@" + p + " bra", label(*t));
        line("bra.uni", label(*f));
    }
}

void PtxEmitter::emitCompare(const Inst& in, const std::string& d) {
    if (in.ty == Ty::Bool) {  // eq / ne on predicates
        std::string a = regOf(in.args[0], Ty::Bool), b = regOf(in.args[1], Ty::Bool);
        if (in.op == Op::Ne) {
            line("xor.pred", d + ", " + a + ", " + b);
        } else {
            std::string t = newTemp(Ty::Bool);
            line("xor.pred", t + ", " + a + ", " + b);
            line("not.pred", d + ", " + t);
        }
        return;
    }
    const char* cmp = "";
    switch (in.op) {
        case Op::Eq: cmp = "eq"; break;
        case Op::Ne: cmp = in.ty == Ty::F32 ? "neu" : "ne"; break;  // C's != is true for NaN
        case Op::Lt: cmp = "lt"; break;
        case Op::Le: cmp = "le"; break;
        case Op::Gt: cmp = "gt"; break;
        default: cmp = "ge"; break;
    }
    // setp wants at least one register operand.
    std::string a = in.args[0].isReg() ? src(in.args[0]) : regOf(in.args[0], in.ty);
    line(std::string("setp.") + cmp + "." + suffix(in.ty), d + ", " + a + ", " + src(in.args[1]));
}

// Element address: `[base]` or `[base+imm]` for a constant index, else mul.wide / shl + add into a temp.
void PtxEmitter::emitMemory(const Inst& in) {
    const bool isLoad = in.op == Op::Load;
    const bool shared = in.space == Space::Shared;
    const std::string base = shared ? sharedBase_[in.args[0].i] : reg(in.args[0].reg);
    const Operand& idx = in.args[1];
    std::string addr;
    if (idx.kind == Operand::ImmI32 && idx.i >= 0 && idx.i < (1 << 28)) {
        addr = idx.i == 0 ? "[" + base + "]" : "[" + base + "+" + std::to_string(idx.i * 4) + "]";
    } else {
        std::string i = regOf(idx, Ty::I32);
        if (shared) {
            std::string off = newTemp(Ty::I32), a = newTemp(Ty::I32);
            line("shl.b32", off + ", " + i + ", 2");
            line("add.s32", a + ", " + base + ", " + off);
            addr = "[" + a + "]";
        } else {
            std::string off = newTemp(Ty::PtrF32), a = newTemp(Ty::PtrF32);
            line("mul.wide.s32", off + ", " + i + ", 4");
            line("add.s64", a + ", " + base + ", " + off);
            addr = "[" + a + "]";
        }
    }
    const char* space = shared ? "shared" : "global";
    const char* type = in.ty == Ty::F32 ? "f32" : "u32";
    if (isLoad) {
        line(std::string("ld.") + space + "." + type, reg(in.dst) + ", " + addr);
    } else {
        line(std::string("st.") + space + "." + type, addr + ", " + regOf(in.args[2], in.ty));
    }
}

void PtxEmitter::emitInst(const Inst& in, const BasicBlock* next) {
    const bool isF = in.ty == Ty::F32;
    const std::string d = in.dst >= 0 ? reg(in.dst) : "";
    // Operands of an instruction with two immediates would be unusual PTX: move the first into a register.
    auto a = [&] { return src(in.args[0]); };
    auto b = [&] { return src(in.args[1]); };
    auto binary = [&](const char* mnemonic) {
        std::string x = in.args[1].isReg() ? src(in.args[0]) : regOf(in.args[0], in.ty);
        line(mnemonic, d + ", " + x + ", " + b());
    };
    switch (in.op) {
        case Op::Phi: throw std::logic_error("ptx: phi reached the backend (run out-of-SSA first)");
        case Op::Copy:
            line(in.ty == Ty::Bool ? "mov.pred" : isF ? "mov.f32" : "mov.u32", d + ", " + a());
            break;
        case Op::Add: binary(isF ? "add.rn.f32" : "add.s32"); break;
        case Op::Sub: binary(isF ? "sub.rn.f32" : "sub.s32"); break;
        case Op::Mul: binary(isF ? "mul.rn.f32" : "mul.lo.s32"); break;
        case Op::Div: binary(isF ? "div.rn.f32" : "div.s32"); break;
        case Op::Rem: binary("rem.s32"); break;
        case Op::Shl: binary("shl.b32"); break;
        case Op::Min: binary(isF ? "min.f32" : "min.s32"); break;
        case Op::Max: binary(isF ? "max.f32" : "max.s32"); break;
        case Op::Neg: line(isF ? "neg.f32" : "neg.s32", d + ", " + regOf(in.args[0], in.ty)); break;
        case Op::Abs: line(isF ? "abs.f32" : "abs.s32", d + ", " + regOf(in.args[0], in.ty)); break;
        case Op::Sqrt: line("sqrt.rn.f32", d + ", " + regOf(in.args[0], Ty::F32)); break;
        case Op::Exp: {  // e^x = 2^(x * log2 e)
            std::string t = newTemp(Ty::F32);
            line("mul.rn.f32", t + ", " + regOf(in.args[0], Ty::F32) + ", 0f3FB8AA3B");
            line("ex2.approx.f32", d + ", " + t);
            break;
        }
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            emitCompare(in, d);
            break;
        case Op::And: line("and.pred", d + ", " + regOf(in.args[0], Ty::Bool) + ", " + regOf(in.args[1], Ty::Bool)); break;
        case Op::Or: line("or.pred", d + ", " + regOf(in.args[0], Ty::Bool) + ", " + regOf(in.args[1], Ty::Bool)); break;
        case Op::Xor: line("xor.pred", d + ", " + regOf(in.args[0], Ty::Bool) + ", " + regOf(in.args[1], Ty::Bool)); break;
        case Op::Not: line("not.pred", d + ", " + regOf(in.args[0], Ty::Bool)); break;
        case Op::IToF: line("cvt.rn.f32.s32", d + ", " + regOf(in.args[0], Ty::I32)); break;
        case Op::FToI: line("cvt.rzi.s32.f32", d + ", " + regOf(in.args[0], Ty::F32)); break;
        case Op::BToI: line("selp.s32", d + ", 1, 0, " + regOf(in.args[0], Ty::Bool)); break;
        case Op::Special:
            line("mov.u32", d + ", %" + sregName(in.sreg) + "." + "xyz"[in.dim]);
            break;
        case Op::Load: case Op::Store: emitMemory(in); break;
        case Op::Barrier: line("bar.sync", "0"); break;
        case Op::Br: case Op::Jmp: emitBranch(in, next); break;
        case Op::Ret: line("ret"); break;
    }
}

}  // namespace kite::codegen
