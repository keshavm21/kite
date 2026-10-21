"""Milestone 0: run the hand-written gpu/hello.ptx (SAXPY) on an NVIDIA GPU.

    python3 gpu/run_hello.py            # on Colab: Runtime -> Change runtime type -> T4 GPU

Prints `OK` if the GPU result matches NumPy bit-for-bit (hello.ptx uses
mul.rn/add.rn, so there is no FMA contraction and results are exact).
"""
import os
import sys

import numpy as np

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy not found. On Colab it is preinstalled; elsewhere: pip install cupy-cuda12x")

here = os.path.dirname(os.path.abspath(__file__))
module = cp.RawModule(path=os.path.join(here, "hello.ptx"))   # driver JIT-compiles the PTX
saxpy = module.get_function("saxpy")

n, a = 1000, np.float32(2.5)
rng = np.random.default_rng(0)
x = rng.standard_normal(n, dtype=np.float32)
y = rng.standard_normal(n, dtype=np.float32)

dx, dy = cp.asarray(x), cp.asarray(y)
block = 256
grid = (n + block - 1) // block
saxpy((grid,), (block,), (np.int32(n), a, dx, dy))     # scalars must be NumPy-typed
cp.cuda.Device().synchronize()

got = cp.asnumpy(dy)
want = (a * x).astype(np.float32) + y                  # two roundings, like mul.rn + add.rn
print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("max abs diff:", float(np.abs(got - want).max()))
print("OK" if np.array_equal(got, want) else "MISMATCH")
