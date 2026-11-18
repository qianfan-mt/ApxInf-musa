"""Small-M BF16 matrix multiplication with FP32 accumulation."""

from ._bf16_gemm.registry import OPS

__all__ = ["OPS", "build"]


def build(op, entries):
    """Dispatch a registry entry; each algorithm returns the shared BuildResult."""
    if op["algorithm"] == "mma-fma-tail":
        from ._bf16_gemm.mma_fma import build_hybrid

        return build_hybrid(op)
    if op["algorithm"] == "fma":
        from ._bf16_gemm.fma import build_fma

        return build_fma(op)
    if op["algorithm"] == "packed-register":
        from ._bf16_gemm.mma_packed import build_packed

        return build_packed(op)
    if op["algorithm"] == "mma-block-rows":
        from ._bf16_gemm.mma_block_rows import build_block_rows

        return build_block_rows(op)
    from ._bf16_gemm.mma32 import build as build_small

    return build_small(op)
