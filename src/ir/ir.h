#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Data structures for the Kite IR (docs/IR.md section 9).
namespace kite::ir {

enum class Ty : uint8_t { Void, Bool, I32, F32, PtrF32, PtrI32 };

enum class Op : uint8_t {
    Copy, Add, Sub, Mul, Div, Rem, Neg, Shl,
    Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, Not,
    IToF, FToI, BToI, Sqrt, Exp, Abs, Min, Max,
    Special, Phi,
    Load, Store, Barrier,
    Br, Jmp, Ret,
};

enum class Space : uint8_t { Global, Shared };
enum class SReg : uint8_t { Tid, Ntid, Ctaid, Nctaid };  // the dimension (x/y/z) is Inst::dim

const char* tyName(Ty t);
const char* opName(Op op);
const char* sregName(SReg r);
bool isPointer(Ty t);
Ty pointee(Ty ptr);  // PtrF32 -> F32, PtrI32 -> I32

struct Operand {
    enum Kind : uint8_t { Reg, ImmI32, ImmF32, ImmBool, Undef, SharedArr } kind = Undef;
    int reg = -1;     // Reg
    int32_t i = 0;    // ImmI32, ImmBool (0/1), SharedArr (index into Function::shared)
    float f = 0.0f;   // ImmF32

    static Operand r(int id) { return make(Reg, id, 0, 0.0f); }
    static Operand i32(int32_t v) { return make(ImmI32, -1, v, 0.0f); }
    static Operand f32(float v) { return make(ImmF32, -1, 0, v); }
    static Operand boolean(bool v) { return make(ImmBool, -1, v ? 1 : 0, 0.0f); }
    static Operand undef() { return make(Undef, -1, 0, 0.0f); }
    static Operand shared(int arrayIndex) { return make(SharedArr, -1, arrayIndex, 0.0f); }

    bool isReg() const { return kind == Reg; }

private:
    static Operand make(Kind k, int reg, int32_t i, float f) {
        Operand o;
        o.kind = k;
        o.reg = reg;
        o.i = i;
        o.f = f;
        return o;
    }
};

struct BasicBlock;

struct Inst {
    Op op = Op::Ret;
    Ty ty = Ty::Void;                 // operation type (the operand type for compares; the result type for casts)
    int dst = -1;                     // destination register, or -1
    std::vector<Operand> args;        // load: {base, idx}; store: {base, idx, val}; phi: incoming values; br: {cond}
    std::vector<BasicBlock*> blocks;  // br: {then, else}; jmp: {target}; phi: incoming blocks (parallel to args)
    Space space = Space::Global;      // load / store
    SReg sreg = SReg::Tid;            // special
    int dim = 0;                      // special: 0 = x, 1 = y, 2 = z

    bool isTerminator() const { return op == Op::Br || op == Op::Jmp || op == Op::Ret; }
    bool isCompare() const { return op >= Op::Eq && op <= Op::Ge; }
    // No side effects and independent of memory: safe to delete if unused, safe to CSE or hoist
    // (except i32 div/rem, which can trap). Loads are deliberately not pure (IR.md section 4.2).
    bool isPure() const { return !(isTerminator() || op == Op::Load || op == Op::Store || op == Op::Barrier); }
    bool hasSideEffects() const { return op == Op::Store || op == Op::Barrier; }
    // Type of the register this instruction defines; Void if it defines none.
    Ty resultType() const;
};

struct BasicBlock {
    int id = 0;
    std::string hint;
    std::vector<Inst> insts;
    std::vector<BasicBlock*> preds, succs;  // derived; rebuild with cfg::rebuild(fn)

    // "entry" for the entry block, else "<hint>.<id>".
    std::string name() const;
    bool hasTerminator() const { return !insts.empty() && insts.back().isTerminator(); }
    Inst& terminator() { return insts.back(); }
    const Inst& terminator() const { return insts.back(); }
};

struct SharedArray {
    std::string name;
    Ty elemTy;
    int size;
};

struct Function {
    std::string name;
    int numParams = 0;  // parameters are registers 0..numParams-1, in order
    std::vector<SharedArray> shared;
    std::vector<std::unique_ptr<BasicBlock>> blocks;  // blocks[0] is the entry block
    std::vector<Ty> regTy;                            // type of every register
    std::vector<std::string> regHint;

    int newReg(Ty t, std::string hint = "");
    // Creates a block and appends it to `blocks`. The first block created gets id 0 and is the entry.
    BasicBlock* newBlock(const std::string& hint);
    int numRegs() const { return static_cast<int>(regTy.size()); }
    std::string regName(int reg) const;  // "%hint.id" or "%id"

private:
    int nextBlockId_ = 0;
};

struct Module {
    std::vector<Function> kernels;
};

}  // namespace kite::ir
