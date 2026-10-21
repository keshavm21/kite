#include "frontend/parser.h"

#include <cstdlib>

#include "frontend/lexer.h"

namespace kite {

namespace {

struct ParseError {};  // thrown after the error has been reported; caught in parseSource

class Parser {
public:
    Parser(const std::string& src, Diagnostics& diag) : lexer_(src, diag), diag_(diag) { advance(); }

    Program parseProgram() {
        Program program;
        while (!check(Tok::Eof)) program.kernels.push_back(parseKernel());
        return program;
    }

private:
    // ---- token plumbing ----
    void advance() { cur_ = lexer_.next(); }
    bool check(Tok k) const { return cur_.kind == k; }
    bool accept(Tok k) {
        if (!check(k)) return false;
        advance();
        return true;
    }
    [[noreturn]] void fail(const std::string& msg) {
        if (cur_.kind != Tok::Error) diag_.error(cur_.loc, msg);  // the lexer already reported Error tokens
        throw ParseError{};
    }
    Token expect(Tok k, const char* what) {
        if (!check(k)) fail(std::string("expected ") + what);
        Token t = cur_;
        advance();
        return t;
    }

    // ---- declarations ----
    Kernel parseKernel() {
        Kernel k;
        k.loc = cur_.loc;
        expect(Tok::KwKernel, "'kernel'");
        k.name = expect(Tok::Ident, "kernel name").text;
        expect(Tok::LParen, "'(' after kernel name");
        if (!check(Tok::RParen)) {
            do {
                k.params.push_back(parseParam());
            } while (accept(Tok::Comma));
        }
        expect(Tok::RParen, "')' after parameters");
        k.body = parseBlock();
        return k;
    }

    Param parseParam() {
        Token name = expect(Tok::Ident, "parameter name");
        expect(Tok::Colon, "':' after parameter name");
        Type t;
        if (accept(Tok::Star)) {
            if (accept(Tok::KwI32)) t = Type::PtrI32;
            else if (accept(Tok::KwF32)) t = Type::PtrF32;
            else fail("expected 'i32' or 'f32' after '*'");
        } else {
            t = parseScalarType();
        }
        return Param{name.text, t, name.loc};
    }

    Type parseScalarType() {
        if (accept(Tok::KwI32)) return Type::I32;
        if (accept(Tok::KwF32)) return Type::F32;
        if (accept(Tok::KwBool)) return Type::Bool;
        fail("expected a type (i32, f32 or bool)");
    }

    // ---- statements ----
    std::unique_ptr<BlockStmt> parseBlock() {
        auto block = std::make_unique<BlockStmt>(cur_.loc);
        expect(Tok::LBrace, "'{'");
        while (!check(Tok::RBrace)) {
            if (check(Tok::Eof)) fail("expected '}' before end of file");
            block->stmts.push_back(parseStmt());
        }
        advance();  // '}'
        return block;
    }

    StmtPtr parseStmt() {
        SourceLoc loc = cur_.loc;
        switch (cur_.kind) {
            case Tok::KwVar: {
                auto s = parseVarDecl();
                expect(Tok::Semi, "';' after statement");
                return s;
            }
            case Tok::KwShared: return parseSharedDecl();
            case Tok::KwIf: return parseIf();
            case Tok::KwWhile: {
                advance();
                expect(Tok::LParen, "'(' after 'while'");
                ExprPtr cond = parseExpr();
                expect(Tok::RParen, "')' after condition");
                return std::make_unique<WhileStmt>(loc, std::move(cond), parseBlock());
            }
            case Tok::KwFor: return parseFor();
            case Tok::KwBreak: return parseSimple(StmtKind::Break);
            case Tok::KwContinue: return parseSimple(StmtKind::Continue);
            case Tok::KwReturn: return parseSimple(StmtKind::Return);
            case Tok::KwSync: {
                advance();
                expect(Tok::LParen, "'(' after 'sync'");
                expect(Tok::RParen, "')' after 'sync('");
                expect(Tok::Semi, "';' after statement");
                return std::make_unique<SimpleStmt>(StmtKind::Sync, loc);
            }
            case Tok::LBrace: return parseBlock();
            case Tok::Ident: {
                auto s = parseAssign();
                expect(Tok::Semi, "';' after statement");
                return s;
            }
            default: fail("expected a statement");
        }
    }

    StmtPtr parseSimple(StmtKind kind) {
        SourceLoc loc = cur_.loc;
        advance();
        expect(Tok::Semi, "';' after statement");
        return std::make_unique<SimpleStmt>(kind, loc);
    }

    // var_decl without the trailing ';' (a for-init reuses it).
    std::unique_ptr<VarDeclStmt> parseVarDecl() {
        SourceLoc loc = cur_.loc;
        expect(Tok::KwVar, "'var'");
        std::string name = expect(Tok::Ident, "variable name").text;
        expect(Tok::Colon, "':' after variable name");
        Type t = parseScalarType();
        expect(Tok::Assign, "'=' (variables must be initialized)");
        return std::make_unique<VarDeclStmt>(loc, std::move(name), t, parseExpr());
    }

    StmtPtr parseSharedDecl() {
        SourceLoc loc = cur_.loc;
        advance();  // 'shared'
        std::string name = expect(Tok::Ident, "shared array name").text;
        expect(Tok::Colon, "':' after shared array name");
        Type t;
        if (accept(Tok::KwI32)) t = Type::SharedI32;
        else if (accept(Tok::KwF32)) t = Type::SharedF32;
        else fail("expected 'i32' or 'f32' as the shared array element type");
        expect(Tok::LBracket, "'[' and an array size");
        long long size = std::strtoll(expect(Tok::IntLit, "an integer array size").text.c_str(), nullptr, 10);
        expect(Tok::RBracket, "']' after array size");
        expect(Tok::Semi, "';' after statement");
        return std::make_unique<SharedDeclStmt>(loc, std::move(name), t, size);
    }

    // assign = lvalue ("=" | "+=" | ...) expr, without the trailing ';'.
    // Compound forms are desugared here: `p[i] += e` becomes `p[i] = p[i] + e`.
    std::unique_ptr<AssignStmt> parseAssign() {
        SourceLoc loc = cur_.loc;
        ExprPtr target = parseLvalue();
        Token opTok = cur_;
        bool compound = true;
        BinOp op = BinOp::Add;
        switch (cur_.kind) {
            case Tok::Assign: compound = false; break;
            case Tok::PlusAssign: op = BinOp::Add; break;
            case Tok::MinusAssign: op = BinOp::Sub; break;
            case Tok::StarAssign: op = BinOp::Mul; break;
            case Tok::SlashAssign: op = BinOp::Div; break;
            default: fail("expected an assignment operator ('=', '+=', '-=', '*=' or '/=')");
        }
        advance();
        ExprPtr value = parseExpr();
        if (compound) {
            value = std::make_unique<BinaryExpr>(opTok.loc, op, cloneExpr(*target), std::move(value));
        }
        return std::make_unique<AssignStmt>(loc, std::move(target), std::move(value));
    }

    ExprPtr parseLvalue() {
        Token name = expect(Tok::Ident, "an assignment target");
        if (isBuiltinVar(name.text)) return parseBuiltinAfterName(name);  // sema rejects it (E7)
        if (accept(Tok::LBracket)) {
            ExprPtr idx = parseExpr();
            expect(Tok::RBracket, "']' after index");
            return std::make_unique<IndexExpr>(name.loc, name.text, std::move(idx));
        }
        return std::make_unique<NameExpr>(name.loc, name.text);
    }

    StmtPtr parseIf() {
        SourceLoc loc = cur_.loc;
        advance();  // 'if'
        expect(Tok::LParen, "'(' after 'if'");
        ExprPtr cond = parseExpr();
        expect(Tok::RParen, "')' after condition");
        auto thenBlock = parseBlock();
        StmtPtr elseStmt;
        if (accept(Tok::KwElse)) {
            elseStmt = check(Tok::KwIf) ? parseIf() : StmtPtr(parseBlock());
        }
        return std::make_unique<IfStmt>(loc, std::move(cond), std::move(thenBlock), std::move(elseStmt));
    }

    StmtPtr parseFor() {
        SourceLoc loc = cur_.loc;
        advance();  // 'for'
        expect(Tok::LParen, "'(' after 'for'");
        StmtPtr init = check(Tok::KwVar) ? StmtPtr(parseVarDecl()) : StmtPtr(parseAssign());
        expect(Tok::Semi, "';' after for-loop initializer");
        ExprPtr cond = parseExpr();
        expect(Tok::Semi, "';' after for-loop condition");
        StmtPtr step = parseAssign();
        expect(Tok::RParen, "')' after for-loop step");
        return std::make_unique<ForStmt>(loc, std::move(init), std::move(cond), std::move(step), parseBlock());
    }

    // ---- expressions: one function per precedence level, lowest binding first ----
    ExprPtr parseExpr() { return parseOr(); }

    ExprPtr parseOr() {
        ExprPtr lhs = parseAnd();
        while (check(Tok::OrOr)) {
            SourceLoc loc = cur_.loc;
            advance();
            lhs = std::make_unique<BinaryExpr>(loc, BinOp::Or, std::move(lhs), parseAnd());
        }
        return lhs;
    }

    ExprPtr parseAnd() {
        ExprPtr lhs = parseEquality();
        while (check(Tok::AndAnd)) {
            SourceLoc loc = cur_.loc;
            advance();
            lhs = std::make_unique<BinaryExpr>(loc, BinOp::And, std::move(lhs), parseEquality());
        }
        return lhs;
    }

    ExprPtr parseEquality() {
        ExprPtr lhs = parseRelational();
        while (check(Tok::EqEq) || check(Tok::NotEq)) {
            SourceLoc loc = cur_.loc;
            BinOp op = check(Tok::EqEq) ? BinOp::Eq : BinOp::Ne;
            advance();
            lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), parseRelational());
        }
        return lhs;
    }

    ExprPtr parseRelational() {
        ExprPtr lhs = parseAdditive();
        while (check(Tok::Lt) || check(Tok::Le) || check(Tok::Gt) || check(Tok::Ge)) {
            SourceLoc loc = cur_.loc;
            BinOp op = check(Tok::Lt) ? BinOp::Lt : check(Tok::Le) ? BinOp::Le : check(Tok::Gt) ? BinOp::Gt : BinOp::Ge;
            advance();
            lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), parseAdditive());
        }
        return lhs;
    }

    ExprPtr parseAdditive() {
        ExprPtr lhs = parseMultiplicative();
        while (check(Tok::Plus) || check(Tok::Minus)) {
            SourceLoc loc = cur_.loc;
            BinOp op = check(Tok::Plus) ? BinOp::Add : BinOp::Sub;
            advance();
            lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), parseMultiplicative());
        }
        return lhs;
    }

    ExprPtr parseMultiplicative() {
        ExprPtr lhs = parseCast();
        while (check(Tok::Star) || check(Tok::Slash) || check(Tok::Percent)) {
            SourceLoc loc = cur_.loc;
            BinOp op = check(Tok::Star) ? BinOp::Mul : check(Tok::Slash) ? BinOp::Div : BinOp::Rem;
            advance();
            lhs = std::make_unique<BinaryExpr>(loc, op, std::move(lhs), parseCast());
        }
        return lhs;
    }

    ExprPtr parseCast() {
        ExprPtr e = parseUnary();
        while (check(Tok::KwAs)) {
            SourceLoc loc = cur_.loc;
            advance();
            e = std::make_unique<CastExpr>(loc, std::move(e), parseScalarType());
        }
        return e;
    }

    ExprPtr parseUnary() {
        if (check(Tok::Minus) || check(Tok::Bang)) {
            SourceLoc loc = cur_.loc;
            UnOp op = check(Tok::Minus) ? UnOp::Neg : UnOp::Not;
            advance();
            return std::make_unique<UnaryExpr>(loc, op, parseUnary());
        }
        return parsePrimary();
    }

    ExprPtr parsePrimary() {
        SourceLoc loc = cur_.loc;
        switch (cur_.kind) {
            case Tok::IntLit: {
                long long v = std::strtoll(cur_.text.c_str(), nullptr, 10);  // saturates; sema range-checks (E13)
                advance();
                return std::make_unique<IntLitExpr>(loc, v);
            }
            case Tok::FloatLit: {
                float v = std::strtof(cur_.text.c_str(), nullptr);
                advance();
                return std::make_unique<FloatLitExpr>(loc, v);
            }
            case Tok::KwTrue: advance(); return std::make_unique<BoolLitExpr>(loc, true);
            case Tok::KwFalse: advance(); return std::make_unique<BoolLitExpr>(loc, false);
            case Tok::LParen: {
                advance();
                ExprPtr e = parseExpr();
                expect(Tok::RParen, "')'");
                return e;
            }
            case Tok::Ident: {
                Token name = cur_;
                advance();
                if (isBuiltinVar(name.text)) return parseBuiltinAfterName(name);
                if (Intrinsic fn; isIntrinsic(name.text, fn)) return parseCall(name, fn);
                if (accept(Tok::LBracket)) {
                    ExprPtr idx = parseExpr();
                    expect(Tok::RBracket, "']' after index");
                    return std::make_unique<IndexExpr>(loc, name.text, std::move(idx));
                }
                return std::make_unique<NameExpr>(loc, name.text);
            }
            default: fail("expected an expression");
        }
    }

    // `thread_idx` etc. are lexed as Ident, Dot, Ident; `name` was already consumed.
    ExprPtr parseBuiltinAfterName(const Token& name) {
        expect(Tok::Dot, "'.' and an axis (x, y or z) after builtin name");
        Token axis = expect(Tok::Ident, "axis 'x', 'y' or 'z'");
        if (axis.text != "x" && axis.text != "y" && axis.text != "z") {
            diag_.error(axis.loc, "expected axis 'x', 'y' or 'z'");
            throw ParseError{};
        }
        return std::make_unique<BuiltinExpr>(name.loc, builtinFromName(name.text), axis.text[0] - 'x');
    }

    ExprPtr parseCall(const Token& name, Intrinsic fn) {
        expect(Tok::LParen, "'(' after intrinsic name");
        std::vector<ExprPtr> args;
        if (!check(Tok::RParen)) {
            do {
                args.push_back(parseExpr());
            } while (accept(Tok::Comma));
        }
        expect(Tok::RParen, "')' after arguments");
        return std::make_unique<CallExpr>(name.loc, fn, std::move(args));
    }

    static bool isBuiltinVar(const std::string& s) {
        return s == "thread_idx" || s == "block_idx" || s == "block_dim" || s == "grid_dim";
    }
    static BuiltinVar builtinFromName(const std::string& s) {
        if (s == "thread_idx") return BuiltinVar::ThreadIdx;
        if (s == "block_idx") return BuiltinVar::BlockIdx;
        if (s == "block_dim") return BuiltinVar::BlockDim;
        return BuiltinVar::GridDim;
    }
    static bool isIntrinsic(const std::string& s, Intrinsic& out) {
        if (s == "sqrt") out = Intrinsic::Sqrt;
        else if (s == "exp") out = Intrinsic::Exp;
        else if (s == "abs") out = Intrinsic::Abs;
        else if (s == "min") out = Intrinsic::Min;
        else if (s == "max") out = Intrinsic::Max;
        else return false;
        return true;
    }

    Lexer lexer_;
    Diagnostics& diag_;
    Token cur_;
};

}  // namespace

std::optional<Program> parseSource(const std::string& src, Diagnostics& diag) {
    try {
        Parser parser(src, diag);
        return parser.parseProgram();
    } catch (const ParseError&) {
        return std::nullopt;
    }
}

}  // namespace kite
