#include "interp/spec.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace kite::interp {

using ir::Ty;

namespace {

constexpr long long kMaxElements = 1LL << 26;  // sanity cap on a buffer's size

bool parseInt(const std::string& s, long long& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    out = std::strtoll(s.c_str(), &end, 10);
    return *end == '\0' && errno == 0;
}

// Same as Python's float(): parsed as a double, then narrowed by the caller.
bool parseDouble(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return *end == '\0';
}

bool parseDims(const std::vector<std::string>& words, const char* what, int line, int out[3], std::string& err) {
    if (words.empty() || words.size() > 3) {
        err = "line " + std::to_string(line) + ": @" + what + " takes 1-3 integers";
        return false;
    }
    for (int d = 0; d < 3; ++d) {
        long long v = 1;
        if (d < static_cast<int>(words.size()) && (!parseInt(words[d], v) || v < 1 || v > 1 << 20)) {
            err = "line " + std::to_string(line) + ": bad @" + what + " dimension '" + words[d] + "'";
            return false;
        }
        out[d] = static_cast<int>(v);
    }
    return true;
}

// "<prefix>(<inner>)" -> inner
bool callForm(const std::string& s, const char* prefix, std::string& inner) {
    size_t n = std::strlen(prefix);
    if (s.size() < n + 2 || s.compare(0, n, prefix) != 0 || s[n] != '(' || s.back() != ')') return false;
    inner = s.substr(n + 1, s.size() - n - 2);
    return !inner.empty();
}

bool scalarBits(const SpecArg& a, uint32_t& bits, std::string& err) {
    if (a.elem == Ty::F32) {
        double d;
        if (!parseDouble(a.init, d)) {
            err = "@arg " + a.name + ": bad f32 value '" + a.init + "'";
            return false;
        }
        float f = static_cast<float>(d);
        std::memcpy(&bits, &f, sizeof bits);
        return true;
    }
    long long v;
    if (!parseInt(a.init, v) || v < INT32_MIN || v > INT32_MAX) {
        err = "@arg " + a.name + ": bad i32 value '" + a.init + "'";
        return false;
    }
    bits = static_cast<uint32_t>(static_cast<int32_t>(v));
    return true;
}

}  // namespace

bool parseSpec(const std::string& source, Spec& out, std::string& err) {
    out = Spec{};
    bool haveKernel = false, haveGrid = false, haveBlock = false;
    std::istringstream lines(source);
    std::string line;
    int lineno = 0;
    while (std::getline(lines, line)) {
        ++lineno;
        size_t p = 0;
        while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) ++p;
        if (line.compare(p, 2, "//") != 0) continue;
        p += 2;
        while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) ++p;
        if (p >= line.size() || line[p] != '@') continue;
        ++p;
        size_t q = p;
        while (q < line.size() && (std::isalnum(static_cast<unsigned char>(line[q])) || line[q] == '_')) ++q;
        if (q == p) continue;
        std::string key = line.substr(p, q - p);
        std::vector<std::string> rest;
        std::istringstream words(line.substr(q));
        for (std::string w; words >> w;) rest.push_back(w);
        std::string at = "line " + std::to_string(lineno) + ": ";

        if (key == "kernel") {
            if (rest.empty()) { err = at + "@kernel needs a name"; return false; }
            out.kernel = rest[0];
            haveKernel = true;
        } else if (key == "grid") {
            if (!parseDims(rest, "grid", lineno, out.grid, err)) return false;
            haveGrid = true;
        } else if (key == "block") {
            if (!parseDims(rest, "block", lineno, out.block, err)) return false;
            haveBlock = true;
        } else if (key == "arg") {
            if (rest.size() != 3) { err = at + "@arg NAME TYPE INIT"; return false; }
            SpecArg a;
            a.name = rest[0];
            a.init = rest[2];
            const std::string& ty = rest[1];
            size_t bracket = ty.find('[');
            std::string elem = ty.substr(0, bracket);
            if (elem != "i32" && elem != "f32") { err = at + "bad type '" + ty + "'"; return false; }
            a.elem = elem == "i32" ? Ty::I32 : Ty::F32;
            if (bracket != std::string::npos) {
                std::string n = ty.substr(bracket + 1);
                if (n.empty() || n.back() != ']') { err = at + "bad type '" + ty + "'"; return false; }
                n.pop_back();
                if (n.empty() || n.find_first_not_of("0123456789") != std::string::npos || !parseInt(n, a.count)) {
                    err = at + "bad type '" + ty + "'";
                    return false;
                }
                a.isBuffer = true;
            }
            out.args.push_back(a);
        } else if (key == "check") {
            out.check.insert(out.check.end(), rest.begin(), rest.end());
        } else {
            err = at + "unknown directive @" + key;
            return false;
        }
    }
    if (!haveKernel || !haveGrid || !haveBlock) {
        err = "spec needs @kernel, @grid and @block";
        return false;
    }
    for (const std::string& c : out.check) {
        bool found = false;
        for (const SpecArg& a : out.args) found |= a.name == c && a.isBuffer;
        if (!found) { err = "@check " + c + ": no such buffer @arg"; return false; }
    }
    return true;
}

bool checkSignature(const Spec& spec, const ir::Function& fn, std::string& err) {
    if (spec.args.size() != static_cast<size_t>(fn.numParams)) {
        err = "kernel " + fn.name + " takes " + std::to_string(fn.numParams) + " parameters but the spec has " +
              std::to_string(spec.args.size()) + " @arg lines";
        return false;
    }
    for (int p = 0; p < fn.numParams; ++p) {
        const SpecArg& a = spec.args[p];
        Ty want = fn.regTy[p];
        bool typeOk = a.isBuffer ? want == (a.elem == Ty::F32 ? Ty::PtrF32 : Ty::PtrI32) : want == a.elem;
        if (a.name != fn.regHint[p] || !typeOk) {
            err = "@arg " + a.name + " does not match parameter " + std::to_string(p) + " of kernel " + fn.name + " (" +
                  fn.regHint[p] + ": " + ir::tyName(want) + ")";
            return false;
        }
    }
    return true;
}

bool makeInputs(const Spec& spec, Inputs& out, std::string& err) {
    out = Inputs{};
    for (const SpecArg& a : spec.args) {
        if (!a.isBuffer) {
            uint32_t bits;
            if (!scalarBits(a, bits, err)) return false;
            out.args.push_back(bits);
            continue;
        }
        if (a.count < 1 || a.count > kMaxElements) {
            err = "@arg " + a.name + ": buffer size out of range";
            return false;
        }
        Buffer buf;
        buf.elem = a.elem;
        buf.data.assign(static_cast<size_t>(a.count), 0u);
        const bool isF = a.elem == Ty::F32;
        auto fill = [&](uint32_t bits) { buf.data.assign(buf.data.size(), bits); };
        auto floatBits = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, sizeof u);
            return u;
        };
        std::string inner;
        if (a.init == "zeros") {
            // already zero
        } else if (a.init == "ones") {
            fill(isF ? floatBits(1.0f) : 1u);
        } else if (a.init == "iota") {
            for (size_t k = 0; k < buf.data.size(); ++k) {
                buf.data[k] = isF ? floatBits(static_cast<float>(k)) : static_cast<uint32_t>(k);
            }
        } else if (callForm(a.init, "fill", inner)) {
            SpecArg v = a;
            v.init = inner;
            uint32_t bits;
            if (!scalarBits(v, bits, err)) return false;
            fill(bits);
        } else if (callForm(a.init, "rand", inner)) {
            long long seed;
            if (inner.find_first_not_of("0123456789") != std::string::npos || !parseInt(inner, seed)) {
                err = "@arg " + a.name + ": bad initializer '" + a.init + "'";
                return false;
            }
            XorShift32 g(static_cast<uint32_t>(seed));
            for (uint32_t& w : buf.data) w = isF ? floatBits(g.nextF32()) : static_cast<uint32_t>(g.nextI32());
        } else {
            err = "@arg " + a.name + ": bad initializer '" + a.init + "'";
            return false;
        }
        out.args.push_back(static_cast<uint32_t>(out.buffers.size()));
        out.buffers.push_back(std::move(buf));
    }
    return true;
}

std::string formatOutputs(const Spec& spec, const Inputs& inputs) {
    std::string text;
    size_t bufferIndex = 0;
    for (const SpecArg& a : spec.args) {
        if (!a.isBuffer) continue;
        const Buffer& buf = inputs.buffers[bufferIndex++];
        bool checked = false;
        for (const std::string& c : spec.check) checked |= c == a.name;
        if (!checked) continue;
        text += a.name + ":";
        char tmp[64];
        for (uint32_t w : buf.data) {
            if (a.elem == Ty::F32) {
                float f;
                std::memcpy(&f, &w, sizeof f);
                if (std::isnan(f)) std::snprintf(tmp, sizeof tmp, "nan");
                else std::snprintf(tmp, sizeof tmp, "%.9g", static_cast<double>(f));
            } else {
                std::snprintf(tmp, sizeof tmp, "%d", static_cast<int32_t>(w));
            }
            text += ' ';
            text += tmp;
        }
        text += '\n';
    }
    return text;
}

}  // namespace kite::interp
