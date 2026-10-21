#pragma once
#include <string>

#include "frontend/token.h"
#include "support/diagnostics.h"

namespace kite {

// Hand-written scanner. next() returns Tok::Eof forever once the input is exhausted.
// On a lexical error it reports through Diagnostics and returns a Tok::Error token, so
// the parser can stop without printing a second message.
class Lexer {
public:
    Lexer(const std::string& src, Diagnostics& diag) : src_(src), diag_(diag) {}
    Token next();

private:
    bool atEnd() const { return pos_ >= src_.size(); }
    char peek(size_t ahead = 0) const {
        return pos_ + ahead < src_.size() ? src_[pos_ + ahead] : '\0';
    }
    char advance();
    Token lexNumber();
    Token lexIdentOrKeyword();
    Token make(Tok kind, size_t begin, SourceLoc loc) const {
        return Token{kind, src_.substr(begin, pos_ - begin), loc};
    }

    const std::string& src_;
    Diagnostics& diag_;
    size_t pos_ = 0;
    SourceLoc loc_;
};

}  // namespace kite
