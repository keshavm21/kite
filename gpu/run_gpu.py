"""Run a PTX file produced by kitec on an NVIDIA GPU, driven by the .kite @spec header.

    kitec -O1 examples/saxpy.kite -o saxpy.ptx
    python3 gpu/run_gpu.py examples/saxpy.kite saxpy.ptx > gpu.out
    kitec run -O1 examples/saxpy.kite > interp.out
    python3 gpu/compare.py interp.out gpu.out

Output format is identical to `kitec run`, so the two can be diffed with tolerance.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kite_spec import format_outputs, make_input, parse_spec  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: run_gpu.py FILE.kite FILE.ptx")
    import cupy as cp

    spec = parse_spec(open(sys.argv[1]).read())
    kernel = cp.RawModule(path=sys.argv[2]).get_function(spec.kernel)

    host, launch_args = {}, []
    for a in spec.args:
        v = make_input(a)
        if a.is_buffer:
            host[a.name] = cp.asarray(v)
            launch_args.append(host[a.name])
        elif a.elem == "f32":
            launch_args.append(np.float32(v))
        else:
            launch_args.append(np.int32(v))

    kernel(spec.grid, spec.block, tuple(launch_args))
    cp.cuda.Device().synchronize()
    sys.stdout.write(format_outputs(spec, {k: cp.asnumpy(v) for k, v in host.items()}))


if __name__ == "__main__":
    main()
