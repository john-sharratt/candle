//! The qwen4exp attention layer as the engine's [`BatchedAttentionLayer`].
//!
//! The projection half is the qwen35 lineage's exactly (§12.5 of the design
//! doc: interleaved `[q | gate]`, per-head Q/K norms, partial rotary 64/256,
//! sigmoid output gate) — shared through
//! [`project_qkv_gated`](crate::models::qwen35::quantized_attention::project_qkv_gated).
//!
//! What differs is the **block input**: it arrives pre-mixed by the Gated
//! Residual (`hc_mix` collapses the wide stream and IS this layer's norm —
//! there are no `attn_norm` tensors in the checkpoint), so `attention_norm`
//! here is a pass-through that only prepares the activation for the numeric
//! path (`to_dynamic`: the q8a128 quantize on int8 modes, a plain wrap on
//! `Off`). Applying any RMSNorm here would norm a second time.
//!
//! The FFN hooks are unreachable by contract: the qwen4exp sweep drives its
//! MoE through the Gated Residual itself and only ever calls
//! `forward_attn_batched` on this wrapper.

use candle::quantized::cuda::{to_dynamic, DynamicActs};
use candle::quantized::{Int8Mode, SumScale};
use candle::{DType, LiveTensor, Result, Tensor};
use candle_nn::kv_cache::WaveGeneration;

use crate::models::batched_layer::{BatchedAttentionLayer, QkvProjection, WaveRef};
use crate::models::lora::LayerLora;
use crate::models::quantized_matmul::QMatMul;
use crate::models::qwen35::quantized_attention::project_qkv_gated;
use crate::models::qwen35::quantized_weights::QuantAttentionWeights;
use crate::models::rotary_layout::RotaryLayout;

/// One qwen4exp attention layer, bound to its geometry. Borrows the engine's
/// weights, so it is built per wave without copying anything.
pub struct Qwen4ExpAttentionLayer<'a> {
    pub w: &'a QuantAttentionWeights,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rotary: &'a RotaryLayout,
}

impl BatchedAttentionLayer for Qwen4ExpAttentionLayer<'_> {
    fn n_head(&self) -> usize {
        self.n_head
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn int8mode(&self) -> Int8Mode {
        self.w.wo.int8mode()
    }

    /// Pass-through: the Gated Residual's `hc_mix` already produced the block
    /// input (it IS this layer's norm), so this only prepares the numeric
    /// operand — quantize-only on int8 modes, a wrap on `Off`.
    ///
    /// The phase is already seeded: `x` is the pre-mix's output, carved from
    /// the attention phase, so the projections inherit it and `wave` names
    /// nothing new.
    fn attention_norm<'w>(
        &self,
        x: &LiveTensor<'w>,
        mode: Int8Mode,
        _wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        let candle::Device::Cuda(dev) = x.device() else {
            candle::bail!("qwen4exp attention runs on CUDA");
        };
        // Raw Σx — a language model's block sums stay far below f16's ceiling.
        to_dynamic(x, mode, dev, SumScale::Raw)
    }

    fn ffn_norm<'w>(
        &self,
        _x: &Tensor,
        _mode: Int8Mode,
        _wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        candle::bail!(
            "qwen4exp: the FFN runs under the Gated Residual in the sweep, never \
             through the attention layer's hooks"
        )
    }

    fn ffn_forward<'w>(
        &self,
        _acts: DynamicActs<'w>,
        _work_dtype: DType,
        _out_dtype: DType,
        _decode_tokens: usize,
        _wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        candle::bail!(
            "qwen4exp: the FFN runs under the Gated Residual in the sweep, never \
             through the attention layer's hooks"
        )
    }

    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
    ) -> Result<QkvProjection<'w>> {
        // Unadapted: qwen4exp's sweep runs its layers under the Gated Residual,
        // which has no adapter plumbing of its own, so no LoRA reaches here.
        project_qkv_gated(
            self.w,
            self.rotary,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            acts,
            out_dtype,
            LayerLora::default(),
        )
    }

    fn o_proj(&self) -> &QMatMul {
        &self.w.wo
    }
}
