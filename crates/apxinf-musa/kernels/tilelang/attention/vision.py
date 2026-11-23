"""MP22 BF16 attention with stable online softmax and native MMA fragments."""

import tilelang
import tilelang.language as T

from pathlib import Path

CPP = Path(__file__).with_name("vision.muh").read_text()

def make_kernels(batch, splits=1, chunk=32, backend="tvm_ffi"):
    @tilelang.jit(
        execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
    )
    def build():
        @T.prim_func
        def kernel(
            Q: T.Tensor((batch, 256, 16, 72), "bfloat16"),
            K: T.Tensor((batch, 256, 16, 72), "bfloat16"),
            V: T.Tensor((batch, 256, 16, 72), "bfloat16"),
            O: T.Tensor((batch, 256, 16, 72), "bfloat16"),
        ):
            with T.Kernel(2, 16, batch, threads=512) as (br, h, b):
                mem = T.alloc_shared((5632,), "bfloat16")
                T.import_source(CPP)
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "vision_register",
                        Q.access_ptr("r"),
                        K.access_ptr("r"),
                        V.access_ptr("r"),
                        O.access_ptr("w"),
                        br,
                        h,
                        b,
                        mem.access_ptr("rw"),
                    )
                )

        return kernel

    return (build(),)
