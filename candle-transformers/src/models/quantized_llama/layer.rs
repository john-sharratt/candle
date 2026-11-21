//! The per-layer weights and forward paths of the quantized Llama model.
//!
//! `MlpOrMoe` is matched in the forward path rather than boxed, to avoid a
//! pointer chase on every FFN. The fused q/k/v weight is consumed only by the CUDA
//! batched projection, so the non-batched `forward()` uses the separate weights
//! and the field is unread off CUDA. The MoE path takes top-k from the router and
//! runs each expert over its tokens, scaling by the normalized routing weights.
//! The `BatchedAttentionLayer` implementation lets the batched driver run a layer
//! with int8 activations for the dense MLP and floating-point activations for MoE.

use super::*;

// One per layer for the model's lifetime, and matched on in the forward path —
// a `Box` would add a pointer chase to every FFN to save a few hundred bytes.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(super) enum MlpOrMoe {
    Mlp(QuantizedMlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<QuantizedMlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    // The expert runs on `[1, tokens, hidden]` (the MLP reads a
                    // batched 3-D activation); the result flattens back to the
                    // `[tokens, hidden]` rows the scatter below indexes.
                    let current_state = xs.index_select(&top_x, 0)?.reshape((1, (), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer
                        .forward(&current_state)?
                        .reshape(((), hidden_dim))?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub(super) attention_wq: QMatMul,
    pub(super) attention_wk: QMatMul,
    pub(super) attention_wv: QMatMul,
    // Fused q/k/v weight: consumed only by the CUDA batched `project_qkv`; the non-batched
    // `forward()` uses the separate wq/wk/wv, so it is intentionally unread off-CUDA.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) attention_wo: QMatMul,
    pub(super) attention_norm: RmsNorm,
    pub(super) mlp_or_moe: MlpOrMoe,
    pub(super) ffn_norm: RmsNorm,
    pub(super) n_head: usize,
    pub(super) n_kv_head: usize,
    pub(super) head_dim: usize,
    pub(super) cis: SharedCis,
    pub(super) neg_inf: Tensor,
    pub(super) span_attn: tracing::Span,
    pub(super) span_rot: tracing::Span,
    pub(super) span_mlp: tracing::Span,
}

pub(super) fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    // mask: 0.0 = visible, -inf = masked (can be any dtype: F32, F16, BF16)
    // Convert to U8 for where_cond: 0.0 -> 0u8 (false), -inf -> 1u8 (true)
    // Create zero in the same dtype as mask to avoid comparison dtype mismatch
    let zero = Tensor::new(0f32, mask.device())?
        .to_dtype(mask.dtype())?
        .broadcast_as(shape.dims())?;
    let mask_u8 = mask.ne(&zero)?; // not equal to 0.0 gives U8 tensor
    let m = mask_u8.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    pub(super) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let (cos, sin) = {
            let mut cis = self
                .cis
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cis.narrow_growable(0, index_pos, seq_len, x.dtype())?
        };
        // The call to contiguous below is only necessary when processing the prompt.
        // When the seq_len is 1 in the inference loop, this is a no-op.
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    /// Apply rotary position embeddings in batched mode.
    ///
    /// Note: This method is no longer used by the generic batched layer processing,
    /// which now receives precomputed RoPE (cos, sin) from the model level.
    /// Kept for potential direct use or debugging.
    #[allow(dead_code)]
    pub(super) fn apply_rotary_emb_batched(
        &self,
        x: &Tensor,
        offsets: &[usize],
        offsets_t: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;

        if offsets.len() != b_sz {
            candle::bail!(
                "Offset count mismatch: got {} offsets for batch size {}",
                offsets.len(),
                b_sz
            );
        }

        // Fast path for the decode loop (seq_len == 1): do a single rope_i call by
        // building per-batch (cos, sin) via index_select.
        if seq_len == 1 {
            let offsets_t_owned = if offsets_t.is_none() {
                Some(decode_utils::offsets_to_u32_tensor(offsets, x.device())?)
            } else {
                None
            };

            let offsets_t = match offsets_t {
                Some(t) => t,
                None => offsets_t_owned
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("missing owned offsets_t".into()))?,
            };

            let offsets_len = offsets_t.dim(0)?;
            if offsets_len != b_sz {
                candle::bail!(
                    "offsets_t length mismatch: got {}, expected {}",
                    offsets_len,
                    b_sz
                );
            }

            // Ensure RoPE tables cover the requested offsets, then gather from the
            // contiguous precomputed tables.
            let required_len = offsets.iter().copied().max().unwrap_or(0) + 1;
            let needs_extend = {
                let cis = self
                    .cis
                    .read()
                    .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
                required_len > cis.max_seq_len()
            };
            if needs_extend {
                let mut cis = self
                    .cis
                    .write()
                    .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
                cis.ensure_len(required_len)?;
            }
            let (cos_all, sin_all) = {
                let cis = self
                    .cis
                    .read()
                    .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
                let cis = cis.get_for_dtype(x.dtype())?;
                (cis.cos.clone(), cis.sin.clone())
            };
            let (cos, sin) = decode_utils::gather_rope_cos_sin(&cos_all, &sin_all, offsets_t)?;
            let x = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous()?
            };
            return candle_nn::rotary_emb::rope_i(&x, &cos, &sin);
        }

        // Multi-token prompt processing: extend once (if needed), then slice from the
        // contiguous tables for each sequence.
        let required_len = offsets
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(seq_len)
            .ok_or_else(|| candle::Error::Msg("rope range overflow".into()))?;

        let needs_extend = {
            let cis = self
                .cis
                .read()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            required_len > cis.max_seq_len()
        };
        if needs_extend {
            let mut cis = self
                .cis
                .write()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            cis.ensure_len(required_len)?;
        }
        let (cos_all, sin_all) = {
            let cis = self
                .cis
                .read()
                .map_err(|_| candle::Error::Msg("poisoned RoPE lock".into()))?;
            let cis = cis.get_for_dtype(x.dtype())?;
            (cis.cos.clone(), cis.sin.clone())
        };

        let mut results = Vec::with_capacity(b_sz);
        for (batch_idx, &offset) in offsets.iter().enumerate() {
            let x_slice = x.narrow(0, batch_idx, 1)?;
            let cos = cos_all.narrow(0, offset, seq_len)?;
            let sin = sin_all.narrow(0, offset, seq_len)?;
            let rotated = candle_nn::rotary_emb::rope_i(&x_slice.contiguous()?, &cos, &sin)?;
            results.push(rotated);
        }

        let results_refs: Vec<&Tensor> = results.iter().collect();
        Tensor::cat(&results_refs, 0)
    }

    /// Forward attention for quantized operations.
    pub(super) fn forward_attn(
        &self,
        cache: &mut KvCache,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        // The phase is known before the span opens (`seq_len == 1` is decode), so
        // the name is chosen up front rather than at the record, as the
        // host-timed version could.
        let dec = seq_len == 1;
        let g_qkv = gpu_span_phase(dec, "decode:qkv_proj", "prefill:qkv_proj", x.device());
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        // Reshape and transpose in one go for better performance
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        // V contiguous only needed for prompt processing, no-op for single token
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;
        g_qkv.end();

        // Reset KV cache if we're at the first position
        let g_alloc = gpu_span_phase(dec, "decode:alloc", "prefill:alloc", x.device());
        if index_pos == 0 {
            cache.reset();
        }
        let (k, v) = cache.append(&k, &v)?;
        g_alloc.end();

        // Use optimized attention kernels when available
        let cache_dtype = cache.dtype();
        let standard_attention = || -> Result<Tensor> {
            // Convert q to cache dtype if needed (k and v are already in cache dtype)
            let q = q.to_dtype(cache_dtype)?;
            let k = crate::utils::repeat_kv(k.clone(), self.n_head / self.n_kv_head)?;
            let v = crate::utils::repeat_kv(v.clone(), self.n_head / self.n_kv_head)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len > 1 {
                let cache_len = index_pos + seq_len;
                let mask: Vec<_> = (0..seq_len)
                    .flat_map(|i| {
                        (0..cache_len).map(move |j| {
                            if j > index_pos + i {
                                f32::NEG_INFINITY
                            } else {
                                0.0f32
                            }
                        })
                    })
                    .collect();
                let mask = Tensor::from_vec(mask, (1, 1, seq_len, cache_len), q.device())?
                    .to_dtype(cache_dtype)?;
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf.to_dtype(cache_dtype)?)?
            } else {
                att
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v)
        };

        let g_kernel = gpu_span_phase(dec, "decode:kernel", "prefill:kernel", x.device());
        let y = if seq_len == 1 && q.device().is_metal() {
            candle_nn::ops::sdpa(&q, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else if seq_len > 1 && matches!(q.device(), Device::Cuda(_)) {
            #[cfg(feature = "flash-attn")]
            {
                let q_fa = q.transpose(1, 2)?;
                let k_fa = k.transpose(1, 2)?;
                let v_fa = v.transpose(1, 2)?;
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                match candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true) {
                    Ok(out) => out.transpose(1, 2)?,
                    Err(_) => standard_attention()?,
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            standard_attention()?
        } else {
            standard_attention()?
        };
        g_kernel.end();

        let g_out_proj = gpu_span_phase(dec, "decode:out_proj", "prefill:out_proj", x.device());
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        g_out_proj.end();
        Ok(y)
    }
}

/// Implement the `BatchedAttentionLayer` trait for quantized llama layers.
///
/// This enables the use of generic batched layer processing from `batched_layer` module.
#[cfg(feature = "cuda")]
impl BatchedAttentionLayer for LayerWeights {
    fn n_head(&self) -> usize {
        self.n_head
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn o_proj(&self) -> &QMatMul {
        &self.attention_wo
    }

    /// B3 producer: fuse ffn_norm -> q8a128 only for the dense MLP path; MoE stays FP.
    #[cfg(feature = "cuda")]
    fn ffn_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        match &self.mlp_or_moe {
            MlpOrMoe::Mlp(_) => self.ffn_norm.forward_dynamic(x, mode, wave_root(wave)),
            _ => Ok(DynamicActs::Float(self.ffn_norm.forward(x)?)),
        }
    }

    /// B3 consumer: dense MLP over the fused activation; MoE falls back to FP.
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
        let out_dtype = x.dtype();
        let h = match &self.mlp_or_moe {
            MlpOrMoe::Mlp(m) => m.forward_dynamic(&acts, work_dtype, out_dtype)?,
            _ => match acts {
                // The `Module`-shaped MoE has no store-width parameter, so this
                // arm narrows the result instead of storing it narrow.
                DynamicActs::Float(t) => {
                    let mut out = self
                        .mlp_or_moe
                        .forward(&t.to_owned_tensor()?.to_dtype(work_dtype)?)?;
                    out.to_dtype_mut(out_dtype)?;
                    out
                }
                DynamicActs::Int8(_) => {
                    candle::bail!("llama MoE ffn_residual received int8 acts")
                }
            },
        };
        add_ffn_residual(x, &h)
    }

    /// B1 producer: fuse attention_norm -> q8a128 (int8) or FP rms_norm (Off).
    #[cfg(feature = "cuda")]
    fn attention_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        self.attention_norm
            .forward_dynamic(x, mode, wave_root(wave))
    }

    /// B1 consumer: q/k/v over the fused activation (fused-qkv or separate).
    #[cfg(feature = "cuda")]
    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
        // This family projects each of q/k/v from its own weight, so there is
        // no stacked group to split and nothing here needs a root.
        _wave: WaveRef<'w>,
    ) -> Result<QkvProjection<'w>> {
        let q_dim = self.n_head * self.head_dim;
        let kv_dim = self.n_kv_head * self.head_dim;
        let (q, k, v) = match acts {
            // int8: ONE segmented launch over the three KO weights (no concat) — float-identical
            // to three separate matmuls, full GPU occupancy.
            DynamicActs::Int8(op) => {
                let qkv = candle::quantized::QMatMul::qkv_segmented(
                    op,
                    &[
                        self.attention_wq.inner(),
                        self.attention_wk.inner(),
                        self.attention_wv.inner(),
                    ],
                    out_dtype,
                )?;
                let r = qkv.rank() - 1;
                (
                    qkv.narrow(r, 0, q_dim)?,
                    qkv.narrow(r, q_dim, kv_dim)?,
                    qkv.narrow(r, q_dim + kv_dim, kv_dim)?,
                )
            }
            DynamicActs::Float(_) => (
                self.attention_wq
                    .forward_dynamic(acts.as_dynamic(), out_dtype)?,
                self.attention_wk
                    .forward_dynamic(acts.as_dynamic(), out_dtype)?,
                self.attention_wv
                    .forward_dynamic(acts.as_dynamic(), out_dtype)?,
            ),
        };
        Ok(QkvProjection {
            q,
            k,
            v,
            gate: None,
        })
    }
}
