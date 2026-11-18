"""Dispatch the two narrow BF16 GEMM shapes."""

from .. import BuildResult
from .weight_pack import build_weight_packer


def build(op) -> BuildResult:
    if op["algorithm"] == "mma32-small-k":
        from .mma32_serial import build_mma32
    elif op["algorithm"] == "mma32-small-n":
        from .mma32_pipelined import build_mma32
    else:
        raise ValueError(f"Unsupported narrow GEMM algorithm: {op['algorithm']}")
    return BuildResult({"part": build_mma32(op), "pack": build_weight_packer(op)}, 0)
