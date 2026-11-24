# Qwen3.8-27B on MUSA S5000: Migration and Validation Status

**Audience:** engineers continuing the Qwen3.8 MUSA/APXInf work.

**Last updated:** 2026-11-24

## 1. Goal and scope

The goal is to run the official Hugging Face `Qwen/Qwen3.8-27B` checkpoint directly on MUSA S5000 (`mp_31`) using BF16 weights.

Current scope:

- Official HF BF16 checkpoint; no NVFP4, FP8, or INT8 conversion.
- Text model first; vision support remains future work.
- Correctness before optimization.
- Default execution remains eager until an optimization passes the numerical gate.

## 2. Phase 0 status

Phase 0 is complete:

- Official checkpoint downloaded: 18 safetensor shards, approximately 52 GB.
- Tensor audit passed: 851 text tensors mapped, zero unmapped tensors and zero shape errors.
- Single-card memory gate passed: approximately 54.3 GiB used with approximately 25.7 GiB headroom.
- The official checkpoint layout is understood and validated.
- GDN geometry is fixed:
  - Q/K heads: 16
  - V heads: 48
  - head dimension: 128
  - recurrent state: F32 `[48, 128, 128]`
  - causal convolution width: 4

## 3. APXInf code status

Collaboration branch: `qwen38-native-gdn-ffi-20261119`. The fixed workspace and device argmax baseline is `84ed9a56`; the optional native GDN integration is:

```text
commit a2f17fac
feat(musa): add optional native qwen gdn launcher
```

The implementation includes:

- Optional native GDN dynamic loader using `dlopen`/`dlsym`.
- Explicit lookup of the `mate_native_launch` symbol (the shared library must implement the expected ABI).
- Dynamic-library lifetime ownership.
- Current MUSA stream forwarding, including graph-capture streams.
- Fixed Qwen3.8 geometry and dtype/device validation.
- F32 preparation of `A_log` and `dt_bias`.
- `LayerWorkspace` integration.
- Default-off behavior.
- Native loader contract tests.

Enable the experimental native path with:

```bash
export APXINF_MUSA_NATIVE_GDN_LIB=/path/to/libmate_native.so
```

If the variable is unset, APXInf uses the original eager GDN path.

The native `.so` is currently an experiment artifact, not a repository-built release artifact. Production packaging still needs to be defined.

## 4. Native GDN implementation

The native kernel is based on the Moore Threads MATE MUSA/TileLang GDN decode implementation. It performs the fused recurrent GDN operation, including the relevant gating and state update work.

The APXInf integration replaces the following old per-layer sequence when enabled:

```text
Q/K normalization
→ decay/beta preparation
→ recurrent state update
```

The rest of the layer remains APXInf-owned:

```text
projection GEMMs
→ causal convolution
→ native GDN
→ gated RMSNorm
→ output projection
→ residual
→ MLP
```

## 5. Native GDN correctness results

### Kernel-level validation

On an S5000, the native launcher was compared directly with the MATE 0.2.8.post2 API for 32 changing inputs and non-zero recurrent state.

Worst observed errors:

| Output | Worst max absolute error | Worst relative L2 |
|---|---:|---:|
| BF16 output | `7.63e-6` | `1.70e-5` |
| F32 recurrent state | `1.19e-7` | `1.80e-8` |

All values were finite and all 32 steps passed the selected tolerance.

### APXInf fixed workspace

The native path was compared with the original APXInf fixed workspace on an S5000:

```text
fixed_workspace_matches_eager_output_and_state: PASS
```

The test covers the GDN plus MLP layer path and recurrent state progression.

### Full-model decode

The same 25-token prompt was decoded for 32 tokens on a single S5000.

#### Original S5000 host

| Metric | Original eager | Native GDN |
|---|---:|---:|
| Load time | 287.14 s | 295.67 s |
| Prefill | 4.165 s | 4.163 s |
| Decode | 92.19 ms/token | 87.61 ms/token |

#### Second S5000 host

| Metric | Original eager | Native GDN |
|---|---:|---:|
| Load time | 286.45 s | 284.17 s |
| Prefill | 4.326 s | 4.239 s |
| Decode | 96.69 ms/token | 92.11 ms/token |

On both machines:

- 32/32 generated token IDs matched.
- Generated text matched.
- Both arms exited successfully.
- Native decode improvement was approximately 4.7–5.0% in this single-prompt measurement.

These are eager-runner measurements, not full Graph measurements.

## 6. Graph status

A batch-1 single-token Graph prototype exists separately; its source changes are not included in this branch.

The Graph path uses:

- Fixed-capacity KV storage.
- Device-side I32 position input.
- Device-side KV append at the current position.
- Fixed-capacity attention with a prefix mask.
- Replay on the same MUSA stream.

### What passed

- Graph capture and replay complete successfully.
- Graph replay and the same operators executed directly are identical for 32 steps.
- Position updates and KV writes were validated with focused tests.
- Top-1 generated tokens match eager in the full-model test.

### What failed

The full-model Graph output does not satisfy the current numerical gate against eager:

```text
required: logits relative L2 < 1%
observed: approximately 1.20% at step 18
```

An earlier version reached approximately 1.66% at step 3. A softmax BF16 rounding adjustment improved the drift but did not clear the 1% gate.

The difference is localized to the custom Graph attention path versus eager muDNN SDPA. It is most consistent with differences in:

- online-softmax accumulation order;
- BF16 weight rounding;
- FP32 rescaling and accumulation;
- output cast timing.

It has not been proven to be a KV index or Graph replay bug.

**Graph remains disabled by default and must not be merged as the default path until the 32-step logits gate passes.**

## 7. muDNN varlen investigation

MUSA SDK 5.2 exposes:

```cpp
ScaledDotProductAttention::RunFlashVarlen(...)
```

with device `cu_seqlens_q` and `cu_seqlens_k` tensors. In principle this could support fixed-capacity KV plus a device-updated prefix length during Graph replay.

However:

- The current APXInf muDNN adapter only exposes fixed-shape `RunFlash`.
- A direct Qwen3.8 varlen probe reported that the available muDNN Flash Varlen path does not support head dimension 256.
- No successful Qwen3.8 dynamic-prefix Graph run has been recorded.

Therefore varlen muDNN is a research direction, not an available solution.

## 8. Independent reproduction on a second S5000 host

The existing checkpoint was reused after SHA-256 verification of all 18 shards and four metadata files. The Rust release harness and native GDN library were rebuilt on the second host. The rebuilt GDN passed a separate 32-step CPU float64 comparison (worst relative L2: 0.174% for output, 1.18e-7 for recurrent state). The full-model token and latency results above are from an eager run, not a successful full-model Graph run.

## 9. Recommended collaboration plan

### Track A: Native GDN productionization

- Reproduce the native launcher from pinned MATE/TileLang sources.
- Define license and provenance handling.
- Add architecture and MUSA SDK checks.
- Replace the experimental external `.so` dependency with a reproducible build or packaged artifact.
- Add multi-prompt and repeated A/B benchmarks.

### Track B: Graph attention correctness

- Keep Graph disabled by default.
- Compare custom Graph attention, muDNN SDPA, and CPU FP64 on identical Q/K/V/KV inputs.
- Investigate softmax accumulation and BF16 cast points.
- Do not repeat the same `RunFlashVarlen` probe on muDNN 3.4.0 expecting d256 support; retest if the backend version or supported geometry changes.
- Do not relax the 1% logits gate.

### Track C: Performance

After correctness is stable:

1. Repeat native GDN A/B with multiple prompts and alternating arm order.
2. Fuse the four GDN input projections:
   ```text
   qkv + z + b + a → qkvz + ba
   ```
3. Fuse causal convolution with Q/K/V layout preparation where safe.
4. Reduce per-token host/device synchronization.
5. Preallocate logits and temporary buffers.
6. Revisit Graph only after the logits gate passes.

## 10. Non-negotiable validation gates

A change is not ready to become the default path unless it has:

```text
- output and recurrent-state finite checks;
- layer-level numerical comparison;
- 32-step full-model logits relative L2 < 1%;
- 32/32 greedy token parity;
- repeated runs with stable results;
- no unowned GPU/process interference;
- reproducible build/runtime provenance.
```

Current default recommendation:

```text
Native GDN: opt-in and validated on two S5000 hosts.
Graph: experimental and disabled by default.
Eager: current correctness baseline.
```
