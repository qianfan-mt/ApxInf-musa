//! Standalone native-HF BF16 eager execution. No registry or graph integration.
use super::{
    config::*,
    model::{
        attention::{AttentionBlock, KvState},
        mlp::Qwen38MlpBf16,
        Qwen38GdnBf16Block, Qwen38GdnBf16State, Qwen38GdnBf16Weights,
    },
    request::RequestState,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_musa::{kernels::qwen38, ops, transfers, MusaBuffer, MusaContext};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    time::Instant,
};

/// Intrusive synchronized wall timings, in milliseconds; not kernel event timings.
#[derive(Debug, Default)]
pub struct ForwardProfile {
    pub gdn_ms: f64,
    pub attention_ms: f64,
    pub mlp_ms: f64,
    pub final_norm_ms: f64,
    pub head_ms: f64,
    pub total_ms: f64,
}
fn validate_tuning(position: usize, poisoned: bool) -> Result<()> {
    if poisoned || position != 0 {
        return Err(error(
            "tuning requires a healthy request at position zero; reset first",
        ));
    }
    Ok(())
}
fn timed<T>(
    ctx: &MusaContext,
    elapsed: Option<&mut f64>,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match elapsed {
        None => f(),
        Some(elapsed) => {
            ctx.synchronize()?;
            let start = Instant::now();
            let result = f()?;
            ctx.synchronize()?;
            *elapsed += start.elapsed().as_secs_f64() * 1000.0;
            Ok(result)
        }
    }
}

const EMBED: &str = "model.language_model.embed_tokens.weight";
const NORM: &str = "model.language_model.norm.weight";
const HEAD: &str = "lm_head.weight";
const MLP: &[&str] = &[
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];
const ATTENTION: &[&str] = &[
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
];
const GDN: &[&str] = &[
    "input_layernorm.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
];
fn error(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}
fn skipped(name: &str) -> bool {
    ["model.visual.", "model.mtp.", "visual.", "mtp."]
        .iter()
        .any(|p| name.starts_with(p))
}
fn layer_names(layer: usize, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|s| format!("model.language_model.layers.{layer}.{s}"))
        .collect()
}
fn required_names() -> BTreeSet<String> {
    let mut names: BTreeSet<_> = [EMBED, NORM, HEAD].into_iter().map(str::to_owned).collect();
    for layer in 0..LAYERS {
        names.extend(layer_names(layer, MLP));
        names.extend(layer_names(
            layer,
            if is_full_attention(layer) {
                ATTENTION
            } else {
                GDN
            },
        ));
    }
    names
}

/// One native shard plus one block's raw tensors at a time. A boundary shard may
/// be reread; we deliberately trade I/O for bounded host memory rather than
/// retaining the entire 27B checkpoint. The native loader itself copies a shard.
struct Shards {
    locations: BTreeMap<String, PathBuf>,
    cached_path: Option<PathBuf>,
    cached: HashMap<String, Tensor>,
}
impl Shards {
    fn open(path: &Path) -> Result<Self> {
        let files = apxinf_loader::safetensors::safetensors_files(path).map_err(error)?;
        let index = if path.is_dir() {
            path.join("model.safetensors.index.json")
        } else {
            path.to_owned()
        };
        let (locations, cached_path, cached) =
            if index.extension().and_then(|e| e.to_str()) == Some("json") && index.is_file() {
                #[derive(serde::Deserialize)]
                struct Index {
                    weight_map: BTreeMap<String, String>,
                }
                let index_data: Index = serde_json::from_slice(
                    &std::fs::read(&index).map_err(|e| error(e.to_string()))?,
                )
                .map_err(|e| error(e.to_string()))?;
                let root = index.parent().unwrap_or(Path::new("."));
                let locations = index_data
                    .weight_map
                    .into_iter()
                    .map(|(n, p)| (n, root.join(p)))
                    .collect();
                (locations, None, HashMap::new())
            } else {
                if files.len() != 1 {
                    return Err(error("checkpoint requires a shard index"));
                }
                let (map, _) = apxinf_loader::safetensors::load_native(&files[0]).map_err(error)?;
                let locations = map.keys().map(|n| (n.clone(), files[0].clone())).collect();
                (locations, Some(files[0].clone()), map)
            };
        let this = Self {
            locations,
            cached_path,
            cached,
        };
        let actual: BTreeSet<_> = this
            .locations
            .keys()
            .filter(|n| !skipped(n))
            .cloned()
            .collect();
        let required = required_names();
        if actual != required {
            return Err(error(format!(
                "HF text inventory must contain exactly 851 keys; missing={:?}, unexpected={:?}",
                required.difference(&actual).collect::<Vec<_>>(),
                actual.difference(&required).collect::<Vec<_>>()
            )));
        }
        Ok(this)
    }
    fn take(&mut self, name: &str) -> Result<Tensor> {
        let path = self
            .locations
            .get(name)
            .ok_or_else(|| error(format!("missing {name}")))?
            .clone();
        if self.cached_path.as_ref() != Some(&path) {
            self.cached.clear();
            self.cached_path = None;
            let (map, _) = apxinf_loader::safetensors::load_native(&path).map_err(error)?;
            // Check actual text names against the index, not only index claims.
            for key in map.keys().filter(|n| !skipped(n)) {
                if self.locations.get(key) != Some(&path) {
                    return Err(error(format!("unindexed or misplaced tensor {key}")));
                }
            }
            for (key, assigned) in &self.locations {
                if assigned == &path && !skipped(key) && !map.contains_key(key) {
                    return Err(error(format!("missing {key} in {}", path.display())));
                }
            }
            self.cached = map;
            self.cached_path = Some(path);
        }
        self.cached
            .remove(name)
            .ok_or_else(|| error(format!("missing or duplicate consumption of {name}")))
    }
    fn layer(&mut self, layer: usize, suffixes: &[&str]) -> Result<HashMap<String, Tensor>> {
        layer_names(layer, suffixes)
            .into_iter()
            .map(|name| {
                let t = self.take(&name)?;
                Ok((name, t))
            })
            .collect()
    }
}

enum Mixer {
    Gdn {
        block: Qwen38GdnBf16Block,
        state: Qwen38GdnBf16State,
    },
    Attention {
        block: AttentionBlock,
        cache: KvState,
    },
}
struct Layer {
    mixer: Mixer,
    mlp: Qwen38MlpBf16,
}

/// Single-sequence eager runner. Outputs remain on the MUSA device in BF16.
/// An execution error poisons the whole request until a successful reset.
/// Every token is synchronized before its position is committed (correctness,
/// not a performance path). No quantization or Gemma embedding scaling is used.
pub struct EagerRunner {
    ctx: MusaContext,
    embedding: Tensor,
    final_norm: Tensor,
    lm_head: Tensor,
    layers: Vec<Layer>,
    request: RequestState,
    capacity: usize,
    poisoned: bool,
}
fn validate_capacity(capacity: usize, maximum: usize) -> Result<()> {
    if capacity == 0 || capacity > maximum || capacity > i32::MAX as usize {
        return Err(error(format!(
            "capacity must be in 1..={maximum} and fit i32"
        )));
    }
    Ok(())
}
fn validate_forward(
    tokens: &[u32],
    start: usize,
    position: usize,
    capacity: usize,
    poisoned: bool,
) -> Result<()> {
    if poisoned {
        return Err(error("request is poisoned; reset required"));
    }
    if tokens.is_empty() {
        return Err(error("tokens must not be empty"));
    }
    if start != position {
        return Err(error("start_pos must equal current request position"));
    }
    if start
        .checked_add(tokens.len())
        .filter(|&end| end <= capacity)
        .is_none()
    {
        return Err(error("token range exceeds capacity"));
    }
    if tokens.iter().any(|&t| t as usize >= VOCAB) {
        return Err(error("token id outside vocabulary"));
    }
    Ok(())
}
fn upload_root(
    shards: &mut Shards,
    ctx: &MusaContext,
    name: &str,
    shape: &[usize],
    transpose: bool,
) -> Result<Tensor> {
    let raw = shards.take(name)?;
    if raw.dtype() != DType::BF16 || raw.shape().dims() != shape {
        return Err(error(format!("{name}: expected native BF16 {shape:?}")));
    }
    let prepared = if transpose {
        let source = raw.as_bf16()?;
        let (rows, cols) = (shape[0], shape[1]);
        let mut output = vec![half::bf16::ZERO; source.len()];
        for r in 0..rows {
            for c in 0..cols {
                output[c * rows + r] = source[r * cols + c];
            }
        }
        Tensor::from_bf16(vec![cols, rows], &output)?
    } else {
        raw
    };
    transfers::to_device(&prepared, ctx.device_id())
}
impl EagerRunner {
    pub fn load(path: impl AsRef<Path>, device_id: usize, capacity: usize) -> Result<Self> {
        validate_capacity(capacity, i32::MAX as usize)?;
        let path = path.as_ref();
        let root = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(Path::new("."))
        };
        let config = Qwen38Config::from_json_file(&root.join("config.json"))?;
        validate_capacity(capacity, config.text.max_seq_len)?;
        let mut shards = Shards::open(path)?;
        let ctx = MusaContext::new(device_id)?;
        let embedding = upload_root(&mut shards, &ctx, EMBED, &[VOCAB, HIDDEN], false)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let mixer = if is_full_attention(layer) {
                Mixer::Attention {
                    block: AttentionBlock::from_hf_layer(
                        &ctx,
                        layer,
                        shards.layer(layer, ATTENTION)?,
                    )?,
                    cache: KvState::new(&ctx, capacity)?,
                }
            } else {
                Mixer::Gdn {
                    block: Qwen38GdnBf16Block::new(Qwen38GdnBf16Weights::from_hf_layer(
                        &ctx,
                        layer,
                        shards.layer(layer, GDN)?,
                    )?),
                    state: Qwen38GdnBf16State::zeroed(&ctx)?,
                }
            };
            let mlp = Qwen38MlpBf16::from_hf_layer(&ctx, layer, shards.layer(layer, MLP)?)?;
            layers.push(Layer { mixer, mlp });
        }
        let final_norm = upload_root(&mut shards, &ctx, NORM, &[HIDDEN], false)?;
        let lm_head = upload_root(&mut shards, &ctx, HEAD, &[VOCAB, HIDDEN], true)?;
        ctx.synchronize()?;
        Ok(Self {
            ctx,
            embedding,
            final_norm,
            lm_head,
            layers,
            request: RequestState::new(capacity)?,
            capacity,
            poisoned: false,
        })
    }
    pub fn position(&self) -> usize {
        self.request.position()
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Replace the existing GEMM tuning session only between healthy requests.
    pub fn configure_tuning(
        &mut self,
        mode: apxinf_musa::tuning::TuningMode,
        databases: &[apxinf_musa::tuning::TuningDb],
        paths: Option<apxinf_musa::tuning::TuningPaths>,
    ) -> Result<()> {
        validate_tuning(self.position(), self.poisoned)?;
        self.poisoned = true;
        apxinf_musa::kernels::gemm::configure_tuning(&self.ctx, mode, databases, paths)?;
        self.poisoned = false;
        Ok(())
    }
    /// Diagnostic traversal with synchronization around every measured category.
    /// Normal forward keeps only its existing per-token synchronization.
    pub fn forward_profiled(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Tensor, ForwardProfile)> {
        let mut profile = ForwardProfile::default();
        let start = Instant::now();
        let output = self.forward_impl(tokens, start_pos, Some(&mut profile))?;
        profile.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((output, profile))
    }
    /// Traverse tokens sequentially and return all logits, `[tokens.len(), 248320]`.
    /// All caller errors are checked before allocation, cache or ledger mutation.
    pub fn forward(&mut self, tokens: &[u32], start_pos: usize) -> Result<Tensor> {
        self.forward_impl(tokens, start_pos, None)
    }
    fn forward_impl(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        profile: Option<&mut ForwardProfile>,
    ) -> Result<Tensor> {
        validate_forward(
            tokens,
            start_pos,
            self.position(),
            self.capacity,
            self.poisoned,
        )?;
        self.poisoned = true;
        let result = self.forward_inner(tokens, start_pos, profile);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }
    fn forward_inner(
        &mut self,
        tokens: &[u32],
        start: usize,
        mut profile: Option<&mut ForwardProfile>,
    ) -> Result<Tensor> {
        let output = self
            .ctx
            .allocate_output(Shape::new(vec![tokens.len(), VOCAB]), DType::BF16)?;
        let destination = MusaBuffer::from_tensor(&output).map_err(error)?;
        for (index, &token) in tokens.iter().enumerate() {
            self.request.begin(start + index)?;
            // MusaBuffer views retain the allocation owner; no host embedding readback.
            let mut hidden = MusaBuffer::from_tensor(&self.embedding)
                .map_err(error)?
                .view(token as usize * HIDDEN * 2, HIDDEN * 2)
                .map_err(error)?
                .as_tensor(Shape::new(vec![1, HIDDEN]), DType::BF16)
                .map_err(error)?;
            for layer in &mut self.layers {
                hidden = match &mut layer.mixer {
                    Mixer::Gdn { block, state } => timed(
                        &self.ctx,
                        profile.as_deref_mut().map(|p| &mut p.gdn_ms),
                        || block.forward_decode(&self.ctx, &hidden, state),
                    )?,
                    Mixer::Attention { block, cache } => timed(
                        &self.ctx,
                        profile.as_deref_mut().map(|p| &mut p.attention_ms),
                        || block.forward_decode(&self.ctx, &hidden, cache),
                    )?,
                };
                hidden = timed(
                    &self.ctx,
                    profile.as_deref_mut().map(|p| &mut p.mlp_ms),
                    || layer.mlp.forward(&self.ctx, &hidden),
                )?;
            }
            let normalized = timed(
                &self.ctx,
                profile.as_deref_mut().map(|p| &mut p.final_norm_ms),
                || {
                    let mut normalized = self
                        .ctx
                        .allocate_output(Shape::new(vec![1, HIDDEN]), DType::BF16)?;
                    qwen38::offset_rms_bf16(
                        &self.ctx,
                        &hidden,
                        &self.final_norm,
                        &mut normalized,
                        RMS_EPS,
                    )?;
                    Ok(normalized)
                },
            )?;
            let logits = timed(
                &self.ctx,
                profile.as_deref_mut().map(|p| &mut p.head_ms),
                || {
                    let mut logits = self
                        .ctx
                        .allocate_output(Shape::new(vec![1, VOCAB]), DType::BF16)?;
                    ops::gemm(
                        &self.ctx,
                        ops::GemmArgs::new(&normalized, &self.lm_head, &mut logits),
                    )?;
                    Ok(logits)
                },
            )?;
            destination
                .view(index * VOCAB * 2, VOCAB * 2)
                .map_err(error)?
                .copy_from_device_async(
                    &MusaBuffer::from_tensor(&logits).map_err(error)?,
                    VOCAB * 2,
                    self.ctx.stream(),
                )
                .map_err(error)?;
            // Async faults must not turn a partially advanced request into a reusable session.
            self.ctx.synchronize()?;
            self.request.complete()?;
        }
        Ok(output)
    }
    /// Clear all GDN state and invalidate KV prefixes. Failed reset stays poisoned.
    pub fn reset(&mut self) -> Result<()> {
        self.poisoned = true;
        self.ctx.synchronize()?;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Gdn { state, .. } => {
                    for tensor in [&state.recurrent, &state.conv_window] {
                        let buffer = MusaBuffer::from_tensor(tensor).map_err(error)?;
                        buffer
                            .memset_async(0, buffer.len(), self.ctx.stream())
                            .map_err(error)?;
                    }
                }
                Mixer::Attention { cache, .. } => cache.reset(),
            }
        }
        self.ctx.synchronize()?;
        self.request.reset();
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tuning_requires_healthy_empty_request() {
        assert!(validate_tuning(0, false).is_ok());
        assert!(validate_tuning(1, false).is_err());
        assert!(validate_tuning(0, true).is_err());
        assert!(validate_tuning(1, true).is_err());
    }
    #[test]
    fn profile_defaults_to_zero() {
        let p = ForwardProfile::default();
        assert_eq!(
            p.gdn_ms + p.attention_ms + p.mlp_ms + p.final_norm_ms + p.head_ms + p.total_ms,
            0.0
        );
    }
    #[test]
    fn exact_text_inventory() {
        assert_eq!(required_names().len(), 851);
    }
    #[test]
    fn preflight_rejects_whole_batch_without_advancing_ledger() {
        let request = RequestState::new(3).unwrap();
        for (tokens, start) in [
            (vec![], 0),
            (vec![1, VOCAB as u32], 0),
            (vec![1], 1),
            (vec![1, 2, 3, 4], 0),
            (vec![1], usize::MAX),
        ] {
            assert!(validate_forward(&tokens, start, request.position(), 3, false).is_err());
            assert_eq!(request.position(), 0);
        }
        assert!(validate_forward(&[1], 0, 0, 3, true).is_err());
        assert!(validate_forward(&[1, 42], 0, 0, 3, false).is_ok());
        assert!(validate_forward(&[1], usize::MAX, usize::MAX, usize::MAX, false).is_err());
    }
}
