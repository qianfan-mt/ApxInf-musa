"""Three-stage BF16 MQA with CTA-local split PV and ordered FP32 merge."""

from pathlib import Path

import tilelang
import tilelang.language as T


def make_kernels(kv, splits=4, chunk=32, group=1, backend="tvm_ffi"):
    # QK and PV are expressed with TileLang; only the warp-level softmax helper
    # remains embedded because this compiler cannot infer its reduction layout.
    rows, dim, bm, bn = 80, 256, 32, 32
    tiles = (kv + bn - 1) // bn
    width = 1 << (kv - 1).bit_length()

    action_cpp = Path(__file__).with_name("action.muh").read_text()

    @tilelang.jit(
        execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
    )
    def compile_qk():
        @T.prim_func
        def qk(
            Q: T.Tensor((rows, dim), "bfloat16"),
            K: T.Tensor((kv, dim), "bfloat16"),
            S: T.Tensor((rows, kv), "float32"),
        ):
            with T.Kernel(T.ceildiv(rows, bm), tiles, threads=128) as (br, bk):
                qs = T.alloc_fragment((bm, dim), "bfloat16")
                ks = T.alloc_fragment((bn, dim), "bfloat16")
                acc = T.alloc_fragment((bm, bn), "float32")
                # TileLang owns the fragment loads; the GEMM layout is supplied by
                # the compiler instead of the former hand-written lane swizzle.
                T.copy(Q[br * bm, 0], qs)
                T.copy(K[bk * bn, 0], ks)
                T.clear(acc)
                T.gemm(qs, ks, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, bn):
                    if br * bm + i < rows and bk * bn + j < kv:
                        S[br * bm + i, bk * bn + j] = acc[i, j]

        return qk

    softmax_cpp = (
        "#define SPLITS " + str(splits) + "\n#define GROUP_SIZE " + str(128 // group)
        + "\n#define CACHE_SIZE "
        + str((kv + 128 // group - 1) // (128 // group))
        + "\n" + action_cpp
    )

    @tilelang.jit(
        execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
    )
    def compile_softmax():
        @T.prim_func
        def softmax(
            S: T.Tensor((rows, kv), "float32"),
            P: T.Tensor((rows, kv), "bfloat16"),
            D: T.Tensor((rows,), "float32"),
        ):
            with T.Kernel(rows // group, threads=128) as row:
                T.import_source(softmax_cpp)
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "row_softmax",
                        S.access_ptr("r", offset=row * group * kv),
                        P.access_ptr("w", offset=row * group * kv),
                        D.access_ptr("w", offset=row * group),
                        kv,
                    )
                )

        return softmax

    pv_cpp = (
        "#define SPLITS " + str(splits)
        + "\n#define GROUP_SIZE " + str(128 // group)
        + "\n#define CACHE_SIZE "
        + str((kv + 128 // group - 1) // (128 // group))
        + "\n" + action_cpp
    )

    @tilelang.jit(
        execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
    )
    def compile_pv():
        @T.prim_func
        def pv(
            P: T.Tensor((rows, kv), "bfloat16"),
            V: T.Tensor((kv, dim), "bfloat16"),
            D: T.Tensor((rows,), "float32"),
            O: T.Tensor((rows, dim), "bfloat16"),
        ):
            with T.Kernel(3, 8, threads=128 * splits) as (br, bc):
                partial = T.alloc_shared((splits, 32, 32), "float32")
                T.import_source(pv_cpp)
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "pv_compute_cta",
                        P.access_ptr("r"),
                        V.access_ptr("r"),
                        D.access_ptr("r"),
                        O.access_ptr("w"),
                        partial.access_ptr("rw"),
                        br,
                        bc,
                        kv,
                    )
                )
                T.sync_threads()
                for i, j in T.Parallel(32, 32):
                    if br * 32 + i < rows and bc * 32 + j < dim:
                        value = T.alloc_var("float32")
                        value = 0.0
                        for part in T.serial(splits):
                            value += partial[part, i, j]
                        O[br * 32 + i, bc * 32 + j] = value / D[br * 32 + i]

        return pv

    return compile_qk(), compile_softmax(), compile_pv()
