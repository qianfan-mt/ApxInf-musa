"""Exact MP22 PI0.5 attention shapes, compiled into the native kernel library."""
from . import BuildResult

OPS = []
for kind, sizes, stages in (
    ("prefix", (522, 778), ("pack", "fast", "fallback")),
    ("action", (532, 788), ("qk", "softmax", "pv")),
    ("vision", (2, 3), ("run",)),
):
    for size in sizes:
        shape = ((size, size, 256) if kind == "prefix" else
                 (10, size, 256) if kind == "action" else (size, 256, 72))
        scratch = (2 * ((size + 31) // 32) * 8192 * 2 +
                   (132 if size == 522 else 195) * 4 if kind == "prefix" else
                   80 * size * 6 + 80 * 4 if kind == "action" else 0)
        op = dict(id=f"attention-{kind}-{size}", type=f"attention-{kind}-bf16",
                  kind=kind, size=size, shape=shape, stages=stages,
                  supported_arches=["mp_22"], status="enabled",
                  scratch_bytes=scratch, flags=["-O3", "-Od3"])
        if kind == "prefix" and size == 522:
            op["stage_flags"] = {"fast": ["-O3", "-Od3", "-mllvm", "-mtgpu-attr-alloc-first"]}
        OPS.append(op)


def build(op, entries):
    kind, size = op["kind"], op["size"]
    if kind == "prefix":
        from .attention.prefix import make_kernels
    elif kind == "action":
        from .attention.action import make_kernels
    else:
        from .attention.vision import make_kernels
    kernels = make_kernels(size, backend="cython")
    return BuildResult(dict(zip(op["stages"], kernels)), op["scratch_bytes"])
