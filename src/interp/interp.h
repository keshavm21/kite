#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "ir/ir.h"

// The IR interpreter: the correctness oracle (docs/TESTING.md section 3). It runs IR in any form.
namespace kite::interp {

// A global-memory array. Elements are stored as raw 32-bit words (i32, or the bits of an f32).
struct Buffer {
    ir::Ty elem = ir::Ty::F32;  // I32 or F32
    std::vector<uint32_t> data;
};

struct Launch {
    int grid[3] = {1, 1, 1};
    int block[3] = {1, 1, 1};
};

struct Result {
    bool ok = true;
    std::string error;  // "error: <reason> in kernel K, block B, thread T, at bb:ip: <instruction>"
};

constexpr uint64_t kDefaultStepLimit = 10'000'000;  // instructions per thread

// Runs `fn` over the whole grid. `args[k]` is the value of parameter k: the raw bits of an i32/f32/bool
// scalar, or an index into `buffers` for a pointer. Stops at the first undefined behaviour.
Result run(const ir::Function& fn, const Launch& launch, const std::vector<uint32_t>& args,
           std::vector<Buffer>& buffers, uint64_t stepLimit = kDefaultStepLimit);

}  // namespace kite::interp
