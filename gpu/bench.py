"""Benchmarks: Kite -O0 and -O1 against NVRTC CUDA C and the library version (docs/IMPLEMENTATION_PLAN.md 8.3).

    python3 gpu/bench.py [--kitec build/kitec] [--out results/bench_T4.md] [--arch sm_75] [--quick]

Needs an NVIDIA GPU and CuPy (Colab T4 is enough). `ptxas`, `nvcc` and `cuobjdump` are optional: without them
the register and SASS columns show "n/a" for the variants they would measure.

For every kernel the script
  * compiles the Kite source to PTX at -O0 and -O1 and loads it with the driver JIT (cp.RawModule),
  * compiles the same kernel written in CUDA C with NVRTC (cp.RawKernel) using --fmad=false, because Kite never
    emits fma (it uses mul.rn + add.rn), so this keeps the arithmetic comparable,
  * times everything with cuda events (3 warm-up runs, median of 20),
  * checks every variant against the library result before it reports a time,
  * reads registers per thread from the loaded kernel, and the SASS instruction count from cuobjdump.
It then writes the tables to --out. Nothing in the report is hand-edited: if -O1 does not change runtime, the
table says so.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kite_spec import XorShift32  # noqa: E402

# Tesla T4 datasheet numbers, printed only as a reference line for the tables.
T4_PEAK_GBS = 320.0
T4_PEAK_GFLOPS = 8100.0

WARMUP, REPS = 3, 20
NOISE = 0.03  # runtime differences below 3% are reported as "same"

CUDA_SAXPY = r"""
extern "C" __global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}"""

CUDA_RELU = r"""
extern "C" __global__ void relu(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.0f);
}"""

CUDA_SOFTMAX = r"""
extern "C" __global__ void softmax_rows(int rows, int cols, const float* x, float* y) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int base = r * cols;
    float m = x[base];
    for (int c = 1; c < cols; c += 1) m = fmaxf(m, x[base + c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; c += 1) {
        float e = expf(x[base + c] - m);
        y[base + c] = e;
        sum += e;
    }
    for (int c = 0; c < cols; c += 1) y[base + c] = y[base + c] / sum;
}"""

CUDA_MATMUL_NAIVE = r"""
extern "C" __global__ void matmul_naive(int m, int n, int k, const float* a, const float* b, float* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; i += 1) acc += a[row * k + i] * b[i * n + col];
        c[row * n + col] = acc;
    }
}"""

CUDA_MATMUL_TILED = r"""
extern "C" __global__ void matmul_tiled(int m, int n, int k, const float* a, const float* b, float* c) {
    __shared__ float ta[256];
    __shared__ float tb[256];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int t = 0; t < (k + 15) / 16; t += 1) {
        int ac = t * 16 + tx, br = t * 16 + ty;
        if (row < m && ac < k) ta[ty * 16 + tx] = a[row * k + ac]; else ta[ty * 16 + tx] = 0.0f;
        if (br < k && col < n) tb[ty * 16 + tx] = b[br * n + col]; else tb[ty * 16 + tx] = 0.0f;
        __syncthreads();
        for (int i = 0; i < 16; i += 1) acc += ta[ty * 16 + i] * tb[i * 16 + tx];
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = acc;
}"""


def rand_f32(cp, count: int, seed: int):
    """Same generator as the interpreter and the tests: uniform in [-1, 1)."""
    rng = XorShift32(seed)
    return cp.asarray(np.array([rng.next_f32() for _ in range(count)], dtype=np.float32))


class Bench:
    """One kernel: how to launch it and what to compare it with."""

    def __init__(self, name, kite, cuda, kind, rtol, atol):
        self.name, self.kite, self.cuda, self.kind = name, kite, cuda, kind  # kind: "mem" or "flop"
        self.rtol, self.atol = rtol, atol

    # Overridden per benchmark -------------------------------------------------------------------
    def setup(self, cp, quick):  # -> None; creates buffers on self
        raise NotImplementedError

    def args(self, cp, out):  # -> launch arguments with `out` as the output buffer
        raise NotImplementedError

    def launch_dims(self):  # -> (grid, block)
        raise NotImplementedError

    def library(self, cp):  # -> (callable, description); the callable returns the output array
        raise NotImplementedError

    work = 0.0  # bytes (kind "mem") or flops (kind "flop") per launch
    size = ""


class Saxpy(Bench):
    def setup(self, cp, quick):
        self.n = 1 << (20 if quick else 24)
        self.x, self.y0 = rand_f32(cp, 1 << 16, 1), rand_f32(cp, 1 << 16, 2)
        self.x = cp.tile(self.x, self.n >> 16)
        self.y0 = cp.tile(self.y0, self.n >> 16)
        self.work, self.size = 12.0 * self.n, f"n = 2^{self.n.bit_length() - 1}"

    def args(self, cp, out):
        return (np.int32(self.n), np.float32(2.5), self.x, out)

    def launch_dims(self):
        return ((self.n + 255) // 256,), (256,)

    def init_out(self, cp):
        return self.y0.copy()

    def library(self, cp):
        return lambda: 2.5 * self.x + self.y0, "CuPy `2.5*x+y` (2 kernels, 1 temporary)"


class Relu(Bench):
    def setup(self, cp, quick):
        self.n = 1 << (20 if quick else 24)
        self.x = cp.tile(rand_f32(cp, 1 << 16, 3), self.n >> 16)
        self.work, self.size = 8.0 * self.n, f"n = 2^{self.n.bit_length() - 1}"

    def args(self, cp, out):
        return (np.int32(self.n), self.x, out)

    def launch_dims(self):
        return ((self.n + 255) // 256,), (256,)

    def init_out(self, cp):
        return cp.zeros(self.n, dtype=cp.float32)

    def library(self, cp):
        return lambda: cp.maximum(self.x, 0), "CuPy `maximum(x, 0)`"


class Softmax(Bench):
    def setup(self, cp, quick):
        self.rows, self.cols = (1024, 256) if quick else (8192, 1024)
        self.x = rand_f32(cp, self.rows * self.cols, 5)
        self.work, self.size = 8.0 * self.rows * self.cols, f"{self.rows} rows x {self.cols}"

    def args(self, cp, out):
        return (np.int32(self.rows), np.int32(self.cols), self.x, out)

    def launch_dims(self):
        return ((self.rows + 31) // 32,), (32,)

    def init_out(self, cp):
        return cp.zeros(self.rows * self.cols, dtype=cp.float32)

    def library(self, cp):
        x2 = self.x.reshape(self.rows, self.cols)

        def run():
            e = cp.exp(x2 - x2.max(axis=1, keepdims=True))
            return (e / e.sum(axis=1, keepdims=True)).ravel()

        return run, "CuPy exp / sum (several kernels)"


class Matmul(Bench):
    def setup(self, cp, quick):
        self.m = self.n = self.k = 256 if quick else 1024
        self.a, self.b = rand_f32(cp, self.m * self.k, 8), rand_f32(cp, self.k * self.n, 9)
        self.work, self.size = 2.0 * self.m * self.n * self.k, f"{self.m}^3"

    def args(self, cp, out):
        return (np.int32(self.m), np.int32(self.n), np.int32(self.k), self.a, self.b, out)

    def launch_dims(self):
        return ((self.n + 15) // 16, (self.m + 15) // 16), (16, 16)

    def init_out(self, cp):
        return cp.zeros(self.m * self.n, dtype=cp.float32)

    def library(self, cp):
        a, b = self.a.reshape(self.m, self.k), self.b.reshape(self.k, self.n)
        return lambda: (a @ b).ravel(), "CuPy `a @ b` (cuBLAS sgemm)"


def median_ms(cp, fn) -> float:
    for _ in range(WARMUP):
        fn()
    times = []
    start, end = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(REPS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))
    return statistics.median(times)


def kitec_ptx(kitec: Path, src: Path, opt: str, arch: str, work: Path, extra=(), tag="") -> Path:
    out = work / f"{src.stem}{opt}{tag}.ptx"
    r = subprocess.run([str(kitec), opt, str(src), f"--target={arch}", *extra, "-o", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"kitec failed on {src}:\n{r.stderr}")
    return out


def sass_count(cubin: Path):
    if not shutil.which("cuobjdump"):
        return None
    r = subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True)
    return sum(1 for line in r.stdout.splitlines() if re.match(r"^\s*/\*[0-9a-f]+\*/", line)) if r.returncode == 0 else None


def cubin_from_ptx(ptx: Path, arch: str):
    if not shutil.which("ptxas"):
        return None
    cubin = ptx.with_suffix(".cubin")
    r = subprocess.run(["ptxas", f"-arch={arch}", str(ptx), "-o", str(cubin)], capture_output=True, text=True)
    return cubin if r.returncode == 0 else None


def cubin_from_cuda(source: str, name: str, arch: str, work: Path):
    if not shutil.which("nvcc"):
        return None
    cu, cubin = work / f"{name}_cuda.cu", work / f"{name}_cuda.cubin"
    cu.write_text(source)
    r = subprocess.run(["nvcc", "-cubin", f"-arch={arch}", "-fmad=false", "-o", str(cubin), str(cu)], capture_output=True, text=True)
    return cubin if r.returncode == 0 else None


def fmt(v, spec="{}"):
    return "n/a" if v is None else spec.format(v)


def run_bench(cp, b: Bench, kitec: Path, arch: str, quick: bool, work: Path):
    b.setup(cp, quick)
    grid, block = b.launch_dims()
    lib_fn, lib_desc = b.library(cp)
    reference = lib_fn()
    cp.cuda.Device().synchronize()

    variants = []  # (label, kernel, cubin)
    for label, opt, extra, tag in (("Kite -O0", "-O0", (), ""),
                                   ("Kite -O1 without unroll", "-O1", ("--disable-pass=unroll",), "_nounroll"),
                                   ("Kite -O1", "-O1", (), "")):
        ptx = kitec_ptx(kitec, ROOT / "examples" / f"{b.kite}.kite", opt, arch, work, extra, tag)
        kernel = cp.RawModule(path=str(ptx)).get_function(b.name)
        variants.append((label, kernel, cubin_from_ptx(ptx, arch), hash(ptx.read_text())))
    nvrtc = cp.RawKernel(b.cuda, b.name, options=("--fmad=false",))
    variants.append(("CUDA C (NVRTC)", nvrtc, cubin_from_cuda(b.cuda, b.name, arch, work), None))

    rows = []
    for label, kernel, cubin, code in variants:
        out = b.init_out(cp)
        kernel(grid, block, b.args(cp, out))
        cp.cuda.Device().synchronize()
        ok = bool(cp.allclose(out, reference, rtol=b.rtol, atol=b.atol))
        scratch = b.init_out(cp)
        args = b.args(cp, scratch)
        ms = median_ms(cp, lambda: kernel(grid, block, args))
        regs = kernel.attributes["num_regs"]
        rows.append((label, ms, regs, sass_count(cubin) if cubin else None, ok, code))
    lib_ms = median_ms(cp, lib_fn)
    rows.append((lib_desc, lib_ms, None, None, True, None))
    return rows


def render(b: Bench, rows) -> str:
    unit = "GB/s" if b.kind == "mem" else "GFLOP/s"
    scale = 1e6  # work / (ms * 1e6) = work per second / 1e9
    peak = T4_PEAK_GBS if b.kind == "mem" else T4_PEAK_GFLOPS
    lines = [f"### {b.name} ({b.size})", "", f"| variant | time (ms) | {unit} | % of T4 peak | regs | SASS insts | matches library |", "|---|---:|---:|---:|---:|---:|:-:|"]
    for label, ms, regs, sass, ok, _ in rows:
        rate = b.work / (ms * scale)
        lines.append(f"| {label} | {ms:.4f} | {rate:.1f} | {100 * rate / peak:.0f}% | {fmt(regs)} | {fmt(sass)} | {'yes' if ok else '**NO**'} |")
    def compare(base, new, what):
        change = (base[1] - new[1]) / base[1]
        if base[5] is not None and base[5] == new[5]:  # same PTX: whatever differs is noise
            return f"{what}: **identical PTX**, so the measured {100 * abs(change):.0f}% difference ({base[1]:.4f} vs {new[1]:.4f} ms) is run-to-run noise."
        verdict = "no measurable change" if abs(change) < NOISE else (f"{100 * change:.0f}% faster" if change > 0 else f"{-100 * change:.0f}% slower")
        sass = f"; SASS {base[3]} -> {new[3]} instructions" if base[3] is not None and new[3] is not None else ""
        return f"{what}: **{verdict}** ({base[1]:.4f} -> {new[1]:.4f} ms), registers {base[2]} -> {new[2]}{sass}."

    o0, nounroll, o1 = rows[0], rows[1], rows[2]
    lines += ["", compare(o0, o1, "-O1 vs -O0 runtime"), "", compare(nounroll, o1, "Loop unrolling alone (-O1 vs -O1 without unroll)"), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitec", default=str(ROOT / "build" / "kitec"))
    ap.add_argument("--out", default=str(ROOT / "results" / "bench_T4.md"))
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--quick", action="store_true", help="small sizes, to check the script itself")
    opts = ap.parse_args()

    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    device = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    benches = [
        Saxpy("saxpy", "saxpy", CUDA_SAXPY, "mem", 1e-6, 1e-6),
        Relu("relu", "relu", CUDA_RELU, "mem", 0, 0),
        Softmax("softmax_rows", "softmax_rows", CUDA_SOFTMAX, "mem", 1e-4, 1e-6),
        Matmul("matmul_naive", "matmul_naive", CUDA_MATMUL_NAIVE, "flop", 1e-3, 1e-3),
        Matmul("matmul_tiled", "matmul_tiled", CUDA_MATMUL_TILED, "flop", 1e-3, 1e-3),
    ]
    sections, failed = [], False
    with tempfile.TemporaryDirectory() as tmp:
        for b in benches:
            print(f"benchmarking {b.name} ...", flush=True)
            rows = run_bench(cp, b, Path(opts.kitec), opts.arch, opts.quick, Path(tmp))
            failed |= not all(r[4] for r in rows)
            sections.append(render(b, rows))

    header = [
        f"# Benchmarks on {device}",
        "",
        "Generated by `gpu/bench.py` (do not edit by hand). "
        f"Times are the median of {REPS} launches after {WARMUP} warm-ups, measured with CUDA events. "
        f"Sizes are {'the --quick sizes' if opts.quick else 'realistic: 2^24 elements for saxpy and relu, 1024^3 for matmul'}.",
        "",
        f"- GPU: {device}, target `{opts.arch}`, CuPy {cp.__version__}.",
        f"- Percent-of-peak uses the T4 datasheet numbers ({T4_PEAK_GBS:.0f} GB/s, {T4_PEAK_GFLOPS:.0f} GFLOP/s fp32); they are a ceiling, not a target.",
        "- GB/s counts the bytes the algorithm must move once (saxpy: read x and y, write y; relu and softmax: read x, write y). "
        "`softmax_rows` re-reads rows, so its figure is an *effective* rate.",
        "- GFLOP/s is 2*m*n*k / time.",
        "- Kite arithmetic is `mul.rn`/`add.rn` with no fma. The CUDA C variants use `--fmad=false` so the arithmetic is comparable. "
        "The library rows (CuPy, cuBLAS) use whatever they like, including fma and heavily tuned tiling.",
        "- Registers come from the loaded kernel; SASS counts come from `cuobjdump -sass` on the ptxas (Kite) or nvcc (CUDA C) cubin.",
        "- Every variant is compared with the library result first; the last column says whether it matched "
        "(`rtol`/`atol` per kernel: exact for relu, 1e-6 for saxpy, 1e-4 for softmax because of `exp`, 1e-3 for matmul because of summation order).",
        "- ptxas re-optimizes the PTX, so a smaller `-O1` PTX does not have to mean faster SASS (docs/PTX_BACKEND.md section 11).",
        "",
    ]
    text = "\n".join(header + sections)
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if failed:
        sys.exit("some variant did not match the library result")


if __name__ == "__main__":
    main()
