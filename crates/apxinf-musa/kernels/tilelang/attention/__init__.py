"""MP22 BF16 attention builders; attention_bf16.py registers offline dispatch.

Inputs and outputs must be contiguous BF16 tensors on the same MUSA device.
Builders return JIT stages; callers own output and scratch storage, stream
ordering, initialization before capture, and the lifetime of captured resources.
"""


def build_prefix(tokens: int):
    """Noncausal MQA, Q/O [tokens*8,256], K/V [tokens,256], scale 1/16.

    Stages: pack(K,V,KP,VP), fast(Q,KP,VP,O,F), fallback(Q,KP,VP,O,F).
    Supported even lengths 256..1024; 522 uses the compact layout and 778
    the wide layout. KP/VP and F sizes are derived from the selected length.
    Always run all three
    stages on the same stream; fast initializes every flag each call.
    """
    from .prefix import make_kernels
    return make_kernels(tokens)


def build_vision(batch: int):
    """Independent views, Q/K/V/O [batch,256,16,72], scale 1/sqrt(72)."""
    if batch not in (2, 3):
        raise ValueError("vision attention supports exactly two or three views")
    from .vision import make_kernels

    return make_kernels(batch)


def build_action(keys: int):
    """Noncausal MQA, Q/O [80,256], K/V [keys,256], scale 1/16.

    Stages: qk(Q,K,S), softmax(S,P,D), pv_merge(P,V,D,O).
    Scratch: S [80,keys] FP32, P [80,keys] BF16, D [80] FP32.
    PV merges four ordered FP32 partials in CTA-local shared memory.
    Query rows flatten ten tokens and eight heads.
    """
    if keys not in (532, 788):
        raise ValueError("action attention supports exactly 532 or 788 keys")
    from .action import make_kernels

    return make_kernels(keys, splits=4, chunk=32, group=1)
