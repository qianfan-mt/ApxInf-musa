//! Fixed-buffer, one-token Qwen3.8 GDN + MLP experiment.
//!
//! This deliberately mirrors the eager path. It owns every transient tensor and
//! writes into those tensors on each call; it does not change kernels or claim
//! graph-capture support.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_musa::{kernels::qwen38, ops, MusaBuffer, MusaContext};
use super::{Qwen38GdnBf16Block, Qwen38GdnBf16State};
use super::mlp::Qwen38MlpBf16;

const HIDDEN: usize = 5120;
const QKV: usize = 10240;
const Z: usize = 6144;
const AB: usize = 48;
const V_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 17408;
const EPSILON: f32 = 1e-6;

fn buffer_tensor(ctx: &MusaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(shape), dtype)
}

fn gemm_into(ctx: &MusaContext, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    ops::gemm(ctx, ops::GemmArgs::new(a, b, out))
}

fn view(tensor: &Tensor, offset: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
    MusaBuffer::from_tensor(tensor)
        .map_err(Error::Other)?
        .view(offset * 2, len * 2)
        .map_err(Error::Other)?
        .as_tensor(Shape::new(shape), DType::BF16)
        .map_err(Error::Other)
}

/// All one-token GDN and MLP intermediates, allocated once at construction.
pub struct LayerWorkspace {
    normalized: Tensor,
    qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    conv: Tensor,
    decay: Tensor,
    beta: Tensor,
    readout: Tensor,
    gated: Tensor,
    gdn_output: Tensor,
    mlp_normalized: Tensor,
    gate: Tensor,
    up: Tensor,
    packed: Tensor,
    activation: Tensor,
    output: Tensor,
}

impl LayerWorkspace {
    pub fn new(ctx: &MusaContext) -> Result<Self> {
        Ok(Self {
            normalized: buffer_tensor(ctx, vec![1, HIDDEN], DType::BF16)?,
            qkv: buffer_tensor(ctx, vec![QKV], DType::BF16)?,
            z: buffer_tensor(ctx, vec![1, Z], DType::BF16)?,
            a: buffer_tensor(ctx, vec![AB], DType::BF16)?,
            b: buffer_tensor(ctx, vec![AB], DType::BF16)?,
            conv: buffer_tensor(ctx, vec![QKV], DType::BF16)?,
            decay: buffer_tensor(ctx, vec![V_HEADS], DType::F32)?,
            beta: buffer_tensor(ctx, vec![V_HEADS], DType::F32)?,
            readout: buffer_tensor(ctx, vec![V_HEADS, HEAD_DIM], DType::BF16)?,
            gated: buffer_tensor(ctx, vec![V_HEADS, HEAD_DIM], DType::BF16)?,
            gdn_output: buffer_tensor(ctx, vec![1, HIDDEN], DType::BF16)?,
            mlp_normalized: buffer_tensor(ctx, vec![1, HIDDEN], DType::BF16)?,
            gate: buffer_tensor(ctx, vec![1, INTERMEDIATE], DType::BF16)?,
            up: buffer_tensor(ctx, vec![1, INTERMEDIATE], DType::BF16)?,
            packed: buffer_tensor(ctx, vec![1, 2 * INTERMEDIATE], DType::BF16)?,
            activation: buffer_tensor(ctx, vec![1, INTERMEDIATE], DType::BF16)?,
            output: buffer_tensor(ctx, vec![1, HIDDEN], DType::BF16)?,
        })
    }

    /// Run GDN followed by MLP, advancing `state` exactly as the eager block.
    pub fn forward_decode(
        &mut self,
        ctx: &MusaContext,
        hidden: &Tensor,
        state: &mut Qwen38GdnBf16State,
        gdn: &Qwen38GdnBf16Block,
        mlp: &Qwen38MlpBf16,
    ) -> Result<Tensor> {
        if hidden.shape().dims() != [1, HIDDEN] || hidden.dtype() != DType::BF16 {
            return Err(Error::Other("fixed Qwen3.8 workspace expects BF16 [1, 5120]".into()));
        }
        qwen38::offset_rms_bf16(ctx, hidden, &gdn.weights.input_norm, &mut self.normalized, EPSILON)?;
        gemm_into(ctx, &self.normalized, &gdn.weights.qkv, &mut self.qkv.reshape(vec![1, QKV])?)?;
        gemm_into(ctx, &self.normalized, &gdn.weights.z, &mut self.z)?;
        gemm_into(ctx, &self.normalized, &gdn.weights.a, &mut self.a.reshape(vec![1, AB])?)?;
        gemm_into(ctx, &self.normalized, &gdn.weights.b, &mut self.b.reshape(vec![1, AB])?)?;
        qwen38::gdn_causal_conv_step(ctx, &mut state.conv_window, &self.qkv, &gdn.weights.conv, &mut self.conv)?;
        let mut q = view(&self.conv, 0, 2048, vec![16, HEAD_DIM])?;
        let mut k = view(&self.conv, 2048, 2048, vec![16, HEAD_DIM])?;
        let v = view(&self.conv, 4096, 6144, vec![V_HEADS, HEAD_DIM])?;
        qwen38::gdn_l2_normalize_heads(ctx, &mut q, EPSILON)?;
        qwen38::gdn_l2_normalize_heads(ctx, &mut k, EPSILON)?;
        qwen38::gdn_decay_beta_step(ctx, &self.a.reshape(vec![AB])?, &self.b.reshape(vec![AB])?, &gdn.weights.a_log, &gdn.weights.dt_bias, &mut self.decay, &mut self.beta)?;
        qwen38::gdn_recurrent_step(ctx, &mut state.recurrent, &q, &k, &v, &self.decay, &self.beta, &mut self.readout)?;
        qwen38::gated_rms_bf16(ctx, &self.readout, &gdn.weights.gdn_norm, &self.z.reshape(vec![V_HEADS, HEAD_DIM])?, &mut self.gated, EPSILON)?;
        gemm_into(ctx, &self.gated.reshape(vec![1, Z])?, &gdn.weights.out, &mut self.gdn_output)?;
        qwen38::add_into(ctx, &mut self.gdn_output, hidden)?;

        qwen38::offset_rms_bf16(ctx, &self.gdn_output, &mlp.norm, &mut self.mlp_normalized, EPSILON)?;
        gemm_into(ctx, &self.mlp_normalized, &mlp.gate, &mut self.gate)?;
        gemm_into(ctx, &self.mlp_normalized, &mlp.up, &mut self.up)?;
        // Avoid concat_rows' allocation by copying both rows into our packed buffer.
        let packed = MusaBuffer::from_tensor(&self.packed).map_err(Error::Other)?;
        let gate = MusaBuffer::from_tensor(&self.gate).map_err(Error::Other)?;
        let up = MusaBuffer::from_tensor(&self.up).map_err(Error::Other)?;
        packed.copy_from_device_async(&gate, gate.len(), ctx.stream()).map_err(Error::Other)?;
        packed.view(gate.len(), up.len()).map_err(Error::Other)?.copy_from_device_async(&up, up.len(), ctx.stream()).map_err(Error::Other)?;
        qwen38::swiglu_bf16(ctx, &self.packed, &mut self.activation)?;
        gemm_into(ctx, &self.activation, &mlp.down, &mut self.output)?;
        qwen38::add_into(ctx, &mut self.output, &self.gdn_output)?;
        Ok(self.output.clone())
    }
}

// Keep the device check close to the experiment API without changing model code.
#[allow(dead_code)]
fn _device(ctx: &MusaContext, tensor: &Tensor) -> bool {
    tensor.device() == Device::Musa(ctx.device_id())
}
