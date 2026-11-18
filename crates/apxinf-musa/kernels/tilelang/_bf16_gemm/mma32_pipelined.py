import tilelang
import tilelang.language as T


def build_mma32(op):
    m, n, k = op["shape"]
    tile_m, tile_n, tile_k, threads = 32, 32, 16, 128
    assert m == 10 and n == tile_n and k % tile_k == 0

    @tilelang.jit(
        out_idx=[], target="musa", execution_backend="cython",
        compile_flags=op["flags"],
    )
    def compile_kernel():
        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), "bfloat16"),
            B: T.Tensor((n, k), "bfloat16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(1, threads=threads):
                As = T.alloc_shared((tile_m, tile_k), "bfloat16")
                Bs = T.alloc_shared((tile_n, tile_k), "bfloat16")
                Acc = T.alloc_fragment((tile_m, tile_n), "float32")
                A_padded = T.alias_buffer(A, (tile_m, k))
                A_bounds = T.make_robust_desc(
                    T.address_of(A[0, 0]), m * k * 2
                )
                T.clear(Acc)
                for kk in T.Pipelined(k // tile_k, num_stages=2):
                    T.copy(
                        A_padded[0, kk * tile_k],
                        As,
                        src_robust_desc=A_bounds,
                        force_async_copy=True,
                    )
                    T.copy(
                        B[0:tile_n, kk * tile_k:(kk + 1) * tile_k],
                        Bs,
                        force_async_copy=True,
                    )
                    T.gemm(As, Bs, Acc, transpose_B=True)
                for row, column in T.Parallel(tile_m, tile_n):
                    if row < m:
                        C[row, column] = T.cast(
                            Acc[row, column], "bfloat16"
                        )

        return kernel

    return compile_kernel()
