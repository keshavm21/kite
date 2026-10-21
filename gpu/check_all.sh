#!/usr/bin/env bash
# Compile every example with kitec, assemble it with ptxas, run it on the GPU and compare with the
# interpreter (docs/TESTING.md section 8). Run from the repo root on a machine with an NVIDIA GPU:
#     gpu/check_all.sh            # -O0 (the only level until Milestone 5)
#     OPT=-O1 gpu/check_all.sh
# Floats are compared with rtol=1e-5, atol=1e-6 (only exp needs it); integers must match exactly.
set -u
KITEC=${KITEC:-build/kitec}
OPT=${OPT:--O0}
OUT=${OUT:-results/gpu}
mkdir -p "$OUT"
fail=0
for src in examples/*.kite; do
  name=$(basename "$src" .kite)
  {
    "$KITEC" "$OPT" "$src" -o "$OUT/$name.ptx" &&
    ptxas -arch=sm_75 -v "$OUT/$name.ptx" -o "$OUT/$name.cubin" 2> "$OUT/$name.ptxas.txt" &&
    python3 gpu/run_gpu.py "$src" "$OUT/$name.ptx" > "$OUT/$name.gpu.out" &&
    "$KITEC" run "$OPT" "$src" > "$OUT/$name.interp.out" &&
    python3 gpu/compare.py "$OUT/$name.interp.out" "$OUT/$name.gpu.out" > "$OUT/$name.compare.txt"
  }
  if [ $? -eq 0 ]; then echo "ok    $name"; else echo "FAIL  $name (see $OUT/$name.*)"; fail=1; fi
done
exit $fail
