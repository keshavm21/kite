#pragma once
#include <memory>
#include <string>
#include <vector>

#include "support/source.h"

namespace kite {

enum class Type { Error, Void, Bool, I32, F32, PtrI32, PtrF32, SharedI32, SharedF32 };

enum class BinOp { Add, Sub, Mul, Div, Rem, Lt, Le, Gt, Ge, Eq, Ne, And, Or };
enum class UnOp { Neg, Not };
enum class BuiltinVar { ThreadIdx, BlockIdx, BlockDim, GridDim };
enum class Intrinsic { Sqrt, Exp, Abs, Min, Max };

const char* typeName(Type t);
const char* binOpSymbol(BinOp op);
const char* builtinName(BuiltinVar v);
const char* intrinsicName(Intrinsic f);

// A resolved declaration. Sema creates one per parameter, local variable (each shadowing
// declaration gets its own) and shared array; the Kernel owns them and AST nodes point at them.
struct Symbol {
    enum class Kind { Local, Param, Shared };
    Kind kind;
    std::string name;
    Type type;
    int index;  // Param: position in the parameter list; Local/Shared: declaration order in the kernel
    SourceLoc loc;
};

bool isReservedName(const std::string& name);  // builtin variables and intrinsics

// ---- Expressions -------------------------------------------------------------------

enum class ExprKind { IntLit, FloatLit, BoolLit, Name, Index, Unary, Binary, Cast, Call, Builtin };

struct Expr {
    const ExprKind kind;
    SourceLoc loc;
    Type type = Type::Error;  // filled in by sema
    virtual ~Expr() = default;

protected:
    Expr(ExprKind k, SourceLoc l) : kind(k), loc(l) {}
};
using ExprPtr = std::unique_ptr<Expr>;

struct IntLitExpr : Expr {
    long long value;  // wider than i32 so sema can report out-of-range literals (E13)
    IntLitExpr(SourceLoc l, long long v) : Expr(ExprKind::IntLit, l), value(v) {}
};
struct FloatLitExpr : Expr {
    float value;
    FloatLitExpr(SourceLoc l, float v) : Expr(ExprKind::FloatLit, l), value(v) {}
};
struct BoolLitExpr : Expr {
    bool value;
    BoolLitExpr(SourceLoc l, bool v) : Expr(ExprKind::BoolLit, l), value(v) {}
};
struct NameExpr : Expr {
    std::string name;
    Symbol* sym = nullptr;  // set by sema
    NameExpr(SourceLoc l, std::string n) : Expr(ExprKind::Name, l), name(std::move(n)) {}
};
struct IndexExpr : Expr {
    std::string name;  // the indexed pointer parameter or shared array
    Symbol* sym = nullptr;  // set by sema
    ExprPtr index;
    IndexExpr(SourceLoc l, std::string n, ExprPtr i)
        : Expr(ExprKind::Index, l), name(std::move(n)), index(std::move(i)) {}
};
struct UnaryExpr : Expr {
    UnOp op;
    ExprPtr operand;
    UnaryExpr(SourceLoc l, UnOp o, ExprPtr e) : Expr(ExprKind::Unary, l), op(o), operand(std::move(e)) {}
};
struct BinaryExpr : Expr {
    BinOp op;
    ExprPtr lhs, rhs;
    BinaryExpr(SourceLoc l, BinOp o, ExprPtr a, ExprPtr b)
        : Expr(ExprKind::Binary, l), op(o), lhs(std::move(a)), rhs(std::move(b)) {}
};
struct CastExpr : Expr {
    ExprPtr operand;
    Type target;
    CastExpr(SourceLoc l, ExprPtr e, Type t) : Expr(ExprKind::Cast, l), operand(std::move(e)), target(t) {}
};
struct CallExpr : Expr {
    Intrinsic fn;
    std::vector<ExprPtr> args;  // arity is checked by sema (E10), not the parser
    CallExpr(SourceLoc l, Intrinsic f, std::vector<ExprPtr> a)
        : Expr(ExprKind::Call, l), fn(f), args(std::move(a)) {}
};
struct BuiltinExpr : Expr {
    BuiltinVar var;
    int axis;  // 0 = x, 1 = y, 2 = z
    BuiltinExpr(SourceLoc l, BuiltinVar v, int a) : Expr(ExprKind::Builtin, l), var(v), axis(a) {}
};

// Deep copy; the parser uses it to desugar `p[i] += e` into `p[i] = p[i] + e`.
ExprPtr cloneExpr(const Expr& e);

// ---- Statements --------------------------------------------------------------------

enum class StmtKind { VarDecl, SharedDecl, Assign, If, While, For, Break, Continue, Return, Sync, Block };

struct Stmt {
    const StmtKind kind;
    SourceLoc loc;
    virtual ~Stmt() = default;

protected:
    Stmt(StmtKind k, SourceLoc l) : kind(k), loc(l) {}
};
using StmtPtr = std::unique_ptr<Stmt>;

struct VarDeclStmt : Stmt {
    std::string name;
    SourceLoc nameLoc;
    Symbol* sym = nullptr;  // set by sema
    Type type;
    ExprPtr init;
    VarDeclStmt(SourceLoc l, std::string n, Type t, ExprPtr i)
        : Stmt(StmtKind::VarDecl, l), name(std::move(n)), type(t), init(std::move(i)) {}
};
struct SharedDeclStmt : Stmt {
    std::string name;
    SourceLoc nameLoc;
    Symbol* sym = nullptr;  // set by sema
    Type type;       // SharedI32 or SharedF32
    long long size;  // element count; sema checks > 0 and the 48 KiB total (E12)
    SharedDeclStmt(SourceLoc l, std::string n, Type t, long long s)
        : Stmt(StmtKind::SharedDecl, l), name(std::move(n)), type(t), size(s) {}
};
struct AssignStmt : Stmt {
    ExprPtr target;  // NameExpr, IndexExpr or BuiltinExpr (the last is rejected by sema, E7)
    ExprPtr value;   // compound assignments are already desugared to `target op value`
    AssignStmt(SourceLoc l, ExprPtr t, ExprPtr v)
        : Stmt(StmtKind::Assign, l), target(std::move(t)), value(std::move(v)) {}
};
struct BlockStmt : Stmt {
    std::vector<StmtPtr> stmts;
    explicit BlockStmt(SourceLoc l) : Stmt(StmtKind::Block, l) {}
};
struct IfStmt : Stmt {
    ExprPtr cond;
    std::unique_ptr<BlockStmt> thenBlock;
    StmtPtr elseStmt;  // null, a BlockStmt, or another IfStmt (else-if chain)
    IfStmt(SourceLoc l, ExprPtr c, std::unique_ptr<BlockStmt> t, StmtPtr e)
        : Stmt(StmtKind::If, l), cond(std::move(c)), thenBlock(std::move(t)), elseStmt(std::move(e)) {}
};
struct WhileStmt : Stmt {
    ExprPtr cond;
    std::unique_ptr<BlockStmt> body;
    WhileStmt(SourceLoc l, ExprPtr c, std::unique_ptr<BlockStmt> b)
        : Stmt(StmtKind::While, l), cond(std::move(c)), body(std::move(b)) {}
};
struct ForStmt : Stmt {
    StmtPtr init;  // VarDeclStmt or AssignStmt
    ExprPtr cond;
    StmtPtr step;  // AssignStmt
    std::unique_ptr<BlockStmt> body;
    ForStmt(SourceLoc l, StmtPtr i, ExprPtr c, StmtPtr s, std::unique_ptr<BlockStmt> b)
        : Stmt(StmtKind::For, l), init(std::move(i)), cond(std::move(c)), step(std::move(s)), body(std::move(b)) {}
};
// break / continue / return / sync carry no payload.
struct SimpleStmt : Stmt {
    SimpleStmt(StmtKind k, SourceLoc l) : Stmt(k, l) {}
};

// ---- Top level ---------------------------------------------------------------------

struct Param {
    std::string name;
    Type type;
    SourceLoc loc;
    Symbol* sym = nullptr;  // set by sema
};

struct Kernel {
    std::string name;
    SourceLoc nameLoc;
    std::vector<Param> params;
    std::unique_ptr<BlockStmt> body;
    SourceLoc loc;
    std::vector<std::unique_ptr<Symbol>> symbols;  // owns every Symbol in this kernel (filled by sema)
};

struct Program {
    std::vector<Kernel> kernels;
};

}  // namespace kite
