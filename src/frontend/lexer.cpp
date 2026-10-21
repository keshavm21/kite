#include "frontend/lexer.h"

#include <cctype>
#include <unordered_map>

namespace kite {

const char* tokName(Tok kind) {
    switch (kind) {
#define X(name) \
    case Tok::name: return #name;
        KITE_TOKENS(X)
#undef X
    }
    return "?";
}

namespace {

bool isDigit(char c) { return c >= '0' && c <= '9'; }
bool isIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool isIdentChar(char c) { return isIdentStart(c) || isDigit(c); }

const std::unordered_map<std::string, Tok>& keywords() {
    static const std::unordered_map<std::string, Tok> table = {
        {"kernel", Tok::KwKernel}, {"var", Tok::KwVar},       {"shared", Tok::KwShared},
        {"if", Tok::KwIf},         {"else", Tok::KwElse},     {"while", Tok::KwWhile},
        {"for", Tok::KwFor},       {"break", Tok::KwBreak},   {"continue", Tok::KwContinue},
        {"return", Tok::KwReturn}, {"sync", Tok::KwSync},     {"true", Tok::KwTrue},
        {"false", Tok::KwFalse},   {"as", Tok::KwAs},         {"i32", Tok::KwI32},
        {"f32", Tok::KwF32},       {"bool", Tok::KwBool},
    };
    return table;
}

}  // namespace

char Lexer::advance() {
    char c = src_[pos_++];
    if (c == '\n') {
        ++loc_.line;
        loc_.col = 1;
    } else {
        ++loc_.col;
    }
    return c;
}

Token Lexer::next() {
    // Skip whitespace and comments.
    for (;;) {
        char c = peek();
        if (atEnd()) break;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
        } else if (c == '/' && peek(1) == '/') {
            while (!atEnd() && peek() != '\n') advance();
        } else if (c == '/' && peek(1) == '*') {
            SourceLoc start = loc_;
            size_t begin = pos_;
            advance();
            advance();
            while (!atEnd() && !(peek() == '*' && peek(1) == '/')) advance();
            if (atEnd()) {
                diag_.error(start, "unterminated block comment");
                return Token{Tok::Error, src_.substr(begin, 2), start};
            }
            advance();
            advance();
        } else {
            break;
        }
    }

    SourceLoc start = loc_;
    size_t begin = pos_;
    if (atEnd()) return Token{Tok::Eof, "", start};

    char c = peek();
    if (isDigit(c)) return lexNumber();
    if (isIdentStart(c)) return lexIdentOrKeyword();

    advance();
    // Maximal munch: take the second character if it forms a longer operator.
    auto two = [&](char second, Tok both, Tok single) {
        if (peek() == second) {
            advance();
            return make(both, begin, start);
        }
        return make(single, begin, start);
    };
    switch (c) {
        case '+': return two('=', Tok::PlusAssign, Tok::Plus);
        case '-': return two('=', Tok::MinusAssign, Tok::Minus);
        case '*': return two('=', Tok::StarAssign, Tok::Star);
        case '/': return two('=', Tok::SlashAssign, Tok::Slash);
        case '=': return two('=', Tok::EqEq, Tok::Assign);
        case '!': return two('=', Tok::NotEq, Tok::Bang);
        case '<': return two('=', Tok::Le, Tok::Lt);
        case '>': return two('=', Tok::Ge, Tok::Gt);
        case '%': return make(Tok::Percent, begin, start);
        case '(': return make(Tok::LParen, begin, start);
        case ')': return make(Tok::RParen, begin, start);
        case '{': return make(Tok::LBrace, begin, start);
        case '}': return make(Tok::RBrace, begin, start);
        case '[': return make(Tok::LBracket, begin, start);
        case ']': return make(Tok::RBracket, begin, start);
        case ',': return make(Tok::Comma, begin, start);
        case ';': return make(Tok::Semi, begin, start);
        case ':': return make(Tok::Colon, begin, start);
        case '.': return make(Tok::Dot, begin, start);
        case '&':
            if (peek() == '&') {
                advance();
                return make(Tok::AndAnd, begin, start);
            }
            diag_.error(start, "unexpected character '&' (did you mean '&&'?)");
            return make(Tok::Error, begin, start);
        case '|':
            if (peek() == '|') {
                advance();
                return make(Tok::OrOr, begin, start);
            }
            diag_.error(start, "unexpected character '|' (did you mean '||'?)");
            return make(Tok::Error, begin, start);
        default:
            diag_.error(start, std::string("unexpected character '") + c + "'");
            return make(Tok::Error, begin, start);
    }
}

// INT = digits; FLOAT = digits "." digits [eE [+-] digits]. Digits are required on both
// sides of the dot, so "1." and ".5" are errors rather than silently accepted.
Token Lexer::lexNumber() {
    SourceLoc start = loc_;
    size_t begin = pos_;
    while (isDigit(peek())) advance();
    if (peek() != '.') return make(Tok::IntLit, begin, start);

    advance();
    if (!isDigit(peek())) {
        diag_.error(start, "malformed float literal: digits required after '.'");
        return make(Tok::Error, begin, start);
    }
    while (isDigit(peek())) advance();

    if (peek() == 'e' || peek() == 'E') {
        advance();
        if (peek() == '+' || peek() == '-') advance();
        if (!isDigit(peek())) {
            diag_.error(start, "malformed float literal: digits required in exponent");
            return make(Tok::Error, begin, start);
        }
        while (isDigit(peek())) advance();
    }
    return make(Tok::FloatLit, begin, start);
}

Token Lexer::lexIdentOrKeyword() {
    SourceLoc start = loc_;
    size_t begin = pos_;
    while (isIdentChar(peek())) advance();
    Token tok = make(Tok::Ident, begin, start);
    auto it = keywords().find(tok.text);
    if (it != keywords().end()) tok.kind = it->second;
    return tok;
}

}  // namespace kite
