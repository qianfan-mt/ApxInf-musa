"""BF16 prefix attention with shared source and shape-specific launch tuning.

Lengths remain JIT constants. Packing, range checks and compiler flags retain
both original specializations; all three stages must use the same stream.
"""
import tilelang
import tilelang.language as T

from pathlib import Path


def load_source(compact: bool) -> str:
    """Select one compile-time layout from the shared device source."""
    text = Path(__file__).with_name("prefix.muh").read_text()
    tag = "COMPACT" if compact else "WIDE"
    begin = f"// VARIANT_{tag}_BEGIN\n"
    end = f"// VARIANT_{tag}_END"
    start = text.index(begin) + len(begin)
    return text[start:text.index(end, start)]



def make_kernels(kv, splits=1, chunk=32, backend="tvm_ffi"):
    if not 256 <= kv <= 1024 or kv % 2:
        raise ValueError("prefix attention supports even lengths in [256, 1024]")
    # 522 and nearby lengths use the compact pack layout; longer lengths use
    # the wide layout.  The selected layout is a compile-time specialization.
    compact = kv <= 640
    group = 6 if compact else 5
    rows_per_block = group * 32
    threads = group * 128
    CPP = load_source(compact)
    fast_flags = ["-O3", "-Od3"]
    if compact:
        fast_flags += ["-mllvm", "-mtgpu-attr-alloc-first"]
    tiles = (kv + 31) // 32

    if compact:
        @tilelang.jit(
            execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
        )
        def build_pack():
            @T.prim_func
            def kernel(
                K: T.Tensor((kv, 256), "bfloat16"),
                V: T.Tensor((kv, 256), "bfloat16"),
                KP: T.Tensor((tiles, 8192), "bfloat16"),
                VP: T.Tensor((tiles, 8192), "bfloat16"),
            ):
                with T.Kernel(tiles, 4, threads=256) as (tile, chunk):
                    ks = T.alloc_shared((32, 64), "bfloat16")
                    vs = T.alloc_shared((32, 64), "bfloat16")
                    k_desc = T.make_robust_desc(T.address_of(K[0, 0]), kv * 256 * 2)
                    v_desc = T.make_robust_desc(T.address_of(V[0, 0]), kv * 256 * 2)
                    T.copy(K[tile * 32, chunk * 64], ks, src_robust_desc=k_desc)
                    T.copy(V[tile * 32, chunk * 64], vs, src_robust_desc=v_desc)
                    for t, word in T.Parallel(256, 4):
                        kr = (t % 128) % 64 // 4 + 16 * (word % 2)
                        kc = (t // 128) * 16 + (word // 2) * 8 + t % 4 + ((t % 128) // 64) * 4
                        for half in T.serial(2):
                            src_k = ks[kr, (kc % 32) * 2 + half]
                            kp_dst = chunk * 2048 + t * 8 + word * 2 + half
                            KP[tile, kp_dst] = src_k
                        vr = (t // 128) * 16 + (t % 128) // 8
                        vc = word * 8 + t % 8
                        for half in T.serial(2):
                            src_v = vs[vr, vc * 2 + half]
                            vp_dst = (t // 128) * 4096 + chunk * 1024 + (t % 128) * 8 + word * 2 + half
                            VP[tile, vp_dst] = src_v

            return kernel

    else:
        @tilelang.jit(
            execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
        )
        def build_pack():
            @T.prim_func
            def kernel(
                K: T.Tensor((kv, 256), "bfloat16"),
                V: T.Tensor((kv, 256), "bfloat16"),
                KP: T.Tensor((tiles, 8192), "bfloat16"),
                VP: T.Tensor((tiles, 8192), "bfloat16"),
            ):
                with T.Kernel(tiles, threads=1024) as tile:
                    ks = T.alloc_shared((32, 256), "bfloat16")
                    vs = T.alloc_shared((32, 256), "bfloat16")
                    k_desc = T.make_robust_desc(T.address_of(K[0, 0]), kv * 256 * 2)
                    v_desc = T.make_robust_desc(T.address_of(V[0, 0]), kv * 256 * 2)
                    T.copy(K[tile * 32, 0], ks, src_robust_desc=k_desc)
                    T.copy(V[tile * 32, 0], vs, src_robust_desc=v_desc)
                    for t, word in T.Parallel(1024, 4):
                        kt = t % 128
                        kp = t // 128
                        kr = kt % 64 // 4 + 16 * (word % 2)
                        pair = kp * 16 + (word // 2) * 8 + kt % 4 + (kt % 128 // 64) * 4
                        for half in T.serial(2):
                            KP[tile, t * 8 + word * 2 + half] = ks[kr, pair * 2 + half]
                        vr = t // 512 * 16 + (t % 128) // 8
                        vp = (t % 512) // 128 * 32 + word * 8 + t % 8
                        for half in T.serial(2):
                            VP[tile, t * 8 + word * 2 + half] = vs[vr, vp * 2 + half]

            return kernel

    @tilelang.jit(
        execution_backend=backend,
        out_idx=[],
        target="musa",
        compile_flags=fast_flags,
    )
    def build_attention():
        @T.prim_func
        def kernel(
            Q: T.Tensor((kv * 8, 256), "bfloat16"),
            KP: T.Tensor((tiles, 8192), "bfloat16"),
            VP: T.Tensor((tiles, 8192), "bfloat16"),
            O: T.Tensor((kv * 8, 256), "bfloat16"),
            F: T.Tensor((T.ceildiv(kv * 8, rows_per_block) * group,), "int32"),
        ):
            with T.Kernel(T.ceildiv(kv * 8, rows_per_block), threads=threads) as br:
                mem = T.alloc_shared((1,), "bfloat16")
                T.import_source(CPP)
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "prefix_fast",
                        Q.access_ptr("r"),
                        KP.access_ptr("r"),
                        VP.access_ptr("r"),
                        O.access_ptr("w"),
                        br,
                        kv,
                        group,
                        mem.access_ptr("rw"),
                        F.access_ptr("w"),
                    )
                )

        return kernel

    @tilelang.jit(
        execution_backend=backend, out_idx=[], target="musa", compile_flags=["-O3", "-Od3"]
    )
    def build_fallback():
        @T.prim_func
        def kernel(
            Q: T.Tensor((kv * 8, 256), "bfloat16"),
            KP: T.Tensor((tiles, 8192), "bfloat16"),
            VP: T.Tensor((tiles, 8192), "bfloat16"),
            O: T.Tensor((kv * 8, 256), "bfloat16"),
            F: T.Tensor((T.ceildiv(kv * 8, rows_per_block) * group,), "int32"),
        ):
            with T.Kernel(T.ceildiv(kv * 8, rows_per_block), threads=threads) as br:
                T.import_source(CPP)
                T.evaluate(
                    T.call_extern(
                        "handle",
                        "prefix_fallback",
                        Q.access_ptr("r"),
                        KP.access_ptr("r"),
                        VP.access_ptr("r"),
                        O.access_ptr("w"),
                        br,
                        kv,
                        group,
                        F.access_ptr("r"),
                    )
                )

        return kernel

    return build_pack(), build_attention(), build_fallback()
