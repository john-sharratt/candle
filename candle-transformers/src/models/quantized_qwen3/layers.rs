//! Runs one Qwen3 decoder layer over the batched attention and MLP paths, with the
//! per-head query and key normalisation applied after the projections.
//!
//! The query, key and value weights stay separate. On the int8 path one segmented
//! launch projects all three over the fused input norm, which gives the same result
//! as three matmuls and keeps the small key and value projections from starving the
//! GPU, so nothing has to be concatenated at load. The per-head normalisation runs
//! over the last axis of the token-major rows, so no transpose is needed around it.
//! The attention path reads the cache's contiguous views directly, and flash
//! attention is used for multi-token sequences where it applies, since it supplies
//! its own causal mask. The dense MLP has no expert cache, so the wave-scoped
//! parameter that the trait carries for mixture-of-experts layers has no effect here.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct AttentionWeights {
    // Separate q/k/v weights. On the int8 path one segmented kernel launch covers all three (no
    // weight stitching); the FP path runs them as three matmuls.
    pub(super) q_proj: QMatMul,
    pub(super) k_proj: QMatMul,
    pub(super) v_proj: QMatMul,
    pub(super) o_proj: QMatMul,
    pub(super) q_norm: RmsNorm,
    pub(super) k_norm: RmsNorm,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) num_kv_groups: usize,
    pub(super) head_dim: usize,
    pub(super) rotary_emb: Arc<RotaryEmbedding>,
    pub(super) span_attn: tracing::Span,
}

impl AttentionWeights {
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_w = gg.tensor(&format!("{prefix}.attn_q.weight"))?;
        let k_w = gg.tensor(&format!("{prefix}.attn_k.weight"))?;
        let v_w = gg.tensor(&format!("{prefix}.attn_v.weight"))?;

        // q/k/v kept separate (no concat): int8 fuses them at launch via the segmented kernel.
        let q_proj = QMatMul::from_qtensor(q_w)?;
        let k_proj = QMatMul::from_qtensor(k_w)?;
        let v_proj = QMatMul::from_qtensor(v_w)?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            span_attn,
        })
    }

    #[inline]
    pub(super) fn project_qkv_with_compute_type(
        &self,
        x: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            self.q_proj.forward(x)?,
            self.k_proj.forward(x)?,
            self.v_proj.forward(x)?,
        ))
    }

    /// B1 consumer: q/k/v over a producer-prepared (fused ln1) activation.
    #[cfg(feature = "cuda")]
    pub(super) fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
    ) -> Result<(LiveTensor<'w>, LiveTensor<'w>, LiveTensor<'w>)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = &self.q_proj;
        let k = &self.k_proj;
        let v = &self.v_proj;
        match acts {
            // int8: ONE segmented launch over the three KO weights (no concat) — float-identical
            // to three separate matmuls, with full GPU occupancy so the tiny k/v stop starving.
            DynamicActs::Int8(op) => {
                let qkv = candle::quantized::QMatMul::qkv_segmented(
                    op,
                    &[q.inner(), k.inner(), v.inner()],
                    out_dtype,
                )?;
                let r = qkv.rank() - 1;
                Ok((
                    qkv.narrow(r, 0, q_dim)?,
                    qkv.narrow(r, q_dim, kv_dim)?,
                    qkv.narrow(r, q_dim + kv_dim, kv_dim)?,
                ))
            }
            DynamicActs::Float(_) => Ok((
                q.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                k.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                v.forward_dynamic(acts.as_dynamic(), out_dtype)?,
            )),
        }
    }

    pub(super) fn forward(&self, cache: &mut KvCache, x: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let (q, k, v) = self.project_qkv_with_compute_type(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Reset KV cache if we're at the first position
        if offset == 0 {
            cache.reset();
        }
        let (k, v) = cache.append(&k, &v)?;

        // KV cache already returns contiguous tensors, repeat_kv works with views
        // Removing redundant contiguous() saves 3 memory allocations per layer

        // Standard attention implementation - used as fallback or primary path
        let standard_attention = || -> Result<Tensor> {
            let k = repeat_kv(k.clone(), self.num_kv_groups)?;
            let v = repeat_kv(v.clone(), self.num_kv_groups)?;
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if l > 1 {
                // Generate causal mask on-demand for multi-token sequences
                let cache_len = offset + l;
                let mask: Vec<_> = (0..l)
                    .flat_map(|i| {
                        (0..cache_len).map(move |j| {
                            if j > offset + i {
                                f32::NEG_INFINITY
                            } else {
                                0.0f32
                            }
                        })
                    })
                    .collect();
                let mask_tensor = Tensor::from_vec(mask, (1, 1, l, cache_len), q.device())?
                    .to_dtype(scores.dtype())?;
                scores = scores.broadcast_add(&mask_tensor)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            probs.matmul(&v)
        };

        let ctx = if l > 1 {
            // Use Flash Attention for multi-token sequences
            // Flash Attention provides its own causal masking, so we ignore attn_mask
            #[cfg(feature = "flash-attn")]
            {
                // Match Llama: dtype conversion handled internally by flash-attn.
                let q_fa = q.transpose(1, 2)?;
                let k_fa = k.transpose(1, 2)?;
                let v_fa = v.transpose(1, 2)?;
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                match candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true) {
                    Ok(out) => {
                        let out = out.transpose(1, 2)?;
                        if out.dtype() != q.dtype() {
                            out.to_dtype(q.dtype())?
                        } else {
                            out
                        }
                    }
                    Err(_) => standard_attention()?,
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            standard_attention()?
        } else {
            standard_attention()?
        }; // (B, H, L, D)
        let reshaped_ctx =
            ctx.transpose(1, 2)?
                .contiguous()?
                .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub(super) self_attn: AttentionWeights,
    pub(super) mlp: QuantizedMlp,
    pub(super) ln1: RmsNorm,
    pub(super) ln2: RmsNorm,
}

/// Implement the `BatchedAttentionLayer` trait for Qwen3 layers.
///
/// This enables the use of generic batched layer processing from `batched_layer` module.
/// Qwen3 has Q/K normalization after QKV projection, which is applied inside `project_qkv()`.
#[cfg(feature = "cuda")]
impl BatchedAttentionLayer for LayerWeights {
    fn n_head(&self) -> usize {
        self.self_attn.num_heads
    }

    fn n_kv_head(&self) -> usize {
        self.self_attn.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.self_attn.head_dim
    }

    fn o_proj(&self) -> &QMatMul {
        &self.self_attn.o_proj
    }

    /// B1 producer: fuse ln1 -> q8a128 (int8) or FP rms_norm (Off) in one kernel.
    #[cfg(feature = "cuda")]
    fn attention_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        self.ln1.forward_dynamic(x, mode, wave_root(wave))
    }

    /// B1 consumer: q/k/v over the fused ln1 activation, then q/k/v RMSNorm + reshapes.
    #[cfg(feature = "cuda")]
    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
        // As llama: separate q/k/v weights, no stacked group to split.
        _wave: WaveRef<'w>,
    ) -> Result<QkvProjection<'w>> {
        let (b_sz, seq_len) = match acts {
            DynamicActs::Float(t) => {
                let (b, s, _) = t.dims3()?;
                (b, s)
            }
            DynamicActs::Int8(op) => match op.lead.as_slice() {
                &[b, s] => (b, s),
                other => {
                    candle::bail!("project_qkv: expected [b, seq] lead, got {other:?}")
                }
            },
        };
        let (q, k, v) = self.self_attn.project_qkv(acts, out_dtype)?;
        let n_head = self.self_attn.num_heads;
        let n_kv_head = self.self_attn.num_kv_heads;
        let head_dim = self.self_attn.head_dim;
        // The per-head norm is over the last axis, so it runs on the token-major
        // rows as they are — `[tokens·heads, head_dim]` is the same rows a
        // head-major transpose would visit, and the transpose (with its copies
        // in and out) buys nothing.
        let q = self
            .self_attn
            .q_norm
            .forward_live(&q.reshape((b_sz * seq_len * n_head, head_dim))?)?
            .reshape((b_sz, seq_len, n_head * head_dim))?;
        let k = self
            .self_attn
            .k_norm
            .forward_live(&k.reshape((b_sz * seq_len * n_kv_head, head_dim))?)?
            .reshape((b_sz, seq_len, n_kv_head * head_dim))?;
        Ok(QkvProjection {
            q,
            k,
            v,
            gate: None,
        })
    }

    /// B3 producer: fuse ln2 -> q8a128 (int8) or FP rms_norm (Off).
    #[cfg(feature = "cuda")]
    fn ffn_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        self.ln2.forward_dynamic(x, mode, wave_root(wave))
    }

    /// B3 consumer: dense MLP over the fused ln2 activation.
    #[cfg(feature = "cuda")]
    fn ffn_residual<'w>(
        &self,
        x: &mut Tensor,
        acts: DynamicActs<'w>,
        work_dtype: DType,
        // A dense MLP has no expert cache to score, so the decode/prefill row
        // split says nothing here.
        _decode: &DecodeRows,
        // A dense MLP allocates its own output, so nothing here is
        // wave-scoped; the parameter is the trait's, for the MoE case.
        _wave: Option<&'w WaveGeneration>,
    ) -> Result<()> {
        let h = self.mlp.forward_dynamic(&acts, work_dtype, x.dtype())?;
        add_ffn_residual(x, &h)
    }
}

impl LayerWeights {
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let mlp = QuantizedMlp::from_weights(
            gg.tensor(&format!("{prefix}.ffn_gate.weight"))?,
            gg.tensor(&format!("{prefix}.ffn_up.weight"))?,
            gg.tensor(&format!("{prefix}.ffn_down.weight"))?,
            Int8Mode::Off,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    pub(super) fn forward(&self, cache: &mut KvCache, x: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(cache, &h, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}
