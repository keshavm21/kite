#include <sstream>
#include <vector>

#include "doctest/doctest.h"
#include "frontend/lexer.h"

using namespace kite;

namespace {

struct LexResult {
    std::vector<Token> tokens;  // excludes Eof
    std::string errors;
};

LexResult lex(const std::string& src) {
    std::ostringstream err;
    Diagnostics diag("t.kite", err);
    Lexer lexer(src, diag);
    LexResult r;
    for (Token t = lexer.next(); t.kind != Tok::Eof; t = lexer.next()) r.tokens.push_back(t);
    r.errors = err.str();
    return r;
}

std::vector<Tok> kinds(const std::string& src) {
    std::vector<Tok> k;
    for (auto& t : lex(src).tokens) k.push_back(t.kind);
    return k;
}

}  // namespace

TEST_CASE("lexer: every keyword") {
    CHECK(kinds("kernel var shared if else while for break continue return sync true false as i32 f32 bool") ==
          std::vector<Tok>{Tok::KwKernel, Tok::KwVar, Tok::KwShared, Tok::KwIf, Tok::KwElse, Tok::KwWhile,
                           Tok::KwFor, Tok::KwBreak, Tok::KwContinue, Tok::KwReturn, Tok::KwSync, Tok::KwTrue,
                           Tok::KwFalse, Tok::KwAs, Tok::KwI32, Tok::KwF32, Tok::KwBool});
}

TEST_CASE("lexer: every operator and punctuation, maximal munch") {
    CHECK(kinds("+ - * / % = += -= *= /= == != < <= > >= && || ! ( ) { } [ ] , ; : .") ==
          std::vector<Tok>{Tok::Plus, Tok::Minus, Tok::Star, Tok::Slash, Tok::Percent, Tok::Assign,
                           Tok::PlusAssign, Tok::MinusAssign, Tok::StarAssign, Tok::SlashAssign, Tok::EqEq,
                           Tok::NotEq, Tok::Lt, Tok::Le, Tok::Gt, Tok::Ge, Tok::AndAnd, Tok::OrOr, Tok::Bang,
                           Tok::LParen, Tok::RParen, Tok::LBrace, Tok::RBrace, Tok::LBracket, Tok::RBracket,
                           Tok::Comma, Tok::Semi, Tok::Colon, Tok::Dot});
    CHECK(kinds("<=") == std::vector<Tok>{Tok::Le});
    CHECK(kinds("< =") == std::vector<Tok>{Tok::Lt, Tok::Assign});
    CHECK(kinds("a<=b") == std::vector<Tok>{Tok::Ident, Tok::Le, Tok::Ident});
}

TEST_CASE("lexer: identifiers, ints and floats") {
    auto r = lex("_tmp1 row_sum 0 256 2.5 1.0e-3 3.0E+2 7.25e2");
    REQUIRE(r.tokens.size() == 8);
    CHECK(r.tokens[0].kind == Tok::Ident);
    CHECK(r.tokens[0].text == "_tmp1");
    CHECK(r.tokens[2].kind == Tok::IntLit);
    CHECK(r.tokens[3].text == "256");
    for (int i = 4; i < 8; ++i) CHECK(r.tokens[i].kind == Tok::FloatLit);
    CHECK(r.tokens[5].text == "1.0e-3");
    CHECK(r.errors.empty());
}

TEST_CASE("lexer: builtin member access is Ident Dot Ident") {
    CHECK(kinds("thread_idx.x") == std::vector<Tok>{Tok::Ident, Tok::Dot, Tok::Ident});
}

TEST_CASE("lexer: comments are skipped") {
    CHECK(kinds("a // line comment\n b /* block\n comment */ c") ==
          std::vector<Tok>{Tok::Ident, Tok::Ident, Tok::Ident});
}

TEST_CASE("lexer: line and column tracking") {
    auto r = lex("ab\n  cd /* x\ny */ ef\n");
    REQUIRE(r.tokens.size() == 3);
    CHECK(r.tokens[0].loc.line == 1);
    CHECK(r.tokens[0].loc.col == 1);
    CHECK(r.tokens[1].loc.line == 2);
    CHECK(r.tokens[1].loc.col == 3);
    CHECK(r.tokens[2].loc.line == 3);
    CHECK(r.tokens[2].loc.col == 6);
}

TEST_CASE("lexer: unterminated block comment") {
    auto r = lex("a\n  /* never closed");
    CHECK(r.errors == "t.kite:2:3: error: unterminated block comment\n");
    REQUIRE(r.tokens.size() == 2);
    CHECK(r.tokens[1].kind == Tok::Error);
}

TEST_CASE("lexer: malformed floats") {
    CHECK(lex("1.").errors == "t.kite:1:1: error: malformed float literal: digits required after '.'\n");
    CHECK(lex("x = 2.e3").errors == "t.kite:1:5: error: malformed float literal: digits required after '.'\n");
    CHECK(lex("1.0e+").errors == "t.kite:1:1: error: malformed float literal: digits required in exponent\n");
}

TEST_CASE("lexer: stray characters") {
    CHECK(lex("a & b").errors == "t.kite:1:3: error: unexpected character '&' (did you mean '&&'?)\n");
    CHECK(lex("a | b").errors == "t.kite:1:3: error: unexpected character '|' (did you mean '||'?)\n");
    CHECK(lex("a # b").errors == "t.kite:1:3: error: unexpected character '#'\n");
}
