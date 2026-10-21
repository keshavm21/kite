#pragma once
#include <string>

#include "support/source.h"

namespace kite {

// X-macro: one list drives both the enum and tokName(), so they cannot drift apart.
#define KITE_TOKENS(X)                                                                        \
    X(Ident) X(IntLit) X(FloatLit)                                                            \
    X(KwKernel) X(KwVar) X(KwShared) X(KwIf) X(KwElse) X(KwWhile) X(KwFor) X(KwBreak)         \
    X(KwContinue) X(KwReturn) X(KwSync) X(KwTrue) X(KwFalse) X(KwAs) X(KwI32) X(KwF32)        \
    X(KwBool)                                                                                 \
    X(Plus) X(Minus) X(Star) X(Slash) X(Percent) X(Assign) X(PlusAssign) X(MinusAssign)       \
    X(StarAssign) X(SlashAssign) X(EqEq) X(NotEq) X(Lt) X(Le) X(Gt) X(Ge) X(AndAnd) X(OrOr)   \
    X(Bang)                                                                                   \
    X(LParen) X(RParen) X(LBrace) X(RBrace) X(LBracket) X(RBracket) X(Comma) X(Semi)         \
    X(Colon) X(Dot)                                                                           \
    X(Eof) X(Error)

enum class Tok {
#define X(name) name,
    KITE_TOKENS(X)
#undef X
};

const char* tokName(Tok kind);

struct Token {
    Tok kind = Tok::Eof;
    std::string text;
    SourceLoc loc;
};

}  // namespace kite
