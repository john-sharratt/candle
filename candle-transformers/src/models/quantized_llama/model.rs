//! The quantized Llama model: loading its GGUF weights and running batched and
//! sequential forward passes over them.
//!
//! `from_gguf` memory-maps the file so each tensor goes to the GPU in one copy,
//! and the resident weight size is the driver-used delta across loading, zero off
//! CUDA. The expert and transient spans are sized from the governor's measured
//! capacity and set up before that baseline, so the span's own mapping is not
//! counted as weights. The `_v2` and `_v3` constructors pin a generation's
//! compression factor, which scales all four calibration rows together.
//! `forward_all_logits` returns logits at every position for perplexity.

use super::*;

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub(super) embeddings: Embedding,
    pub(super) layers: Vec<LayerWeights>,
    pub(super) norm: RmsNorm,
    pub(super) output: QMatMul,
    pub(super) device: Device,
    pub(super) span: tracing::Span,
    pub(super) span_output: tracing::Span,
    /// Fixed resident-weight VRAM (bytes), measured at load as the driver-used
    /// delta across weight loading. Dense model: all weights stay VRAM-resident.
    /// `0` on non-CUDA.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) base_weight_bytes: usize,
    pub(super) compression_error_factor: f32,
}

/// Implementation of BatchedModelCore for use with BatchedInference wrapper.
///
/// This is the new recommended way to use batched inference. The BatchedInference
/// wrapper handles RoPE caching at the model level, so this implementation is simpler.
#[cfg(feature = "cuda")]
impl BatchedModelCore for ModelWeights {
    type Layer = LayerWeights;

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        for layer in &self.layers {
            layer.attention_norm.maybe_change_dtype(dtype)?;
            layer.ffn_norm.maybe_change_dtype(dtype)?;
        }
        self.norm.maybe_change_dtype(dtype)
    }

    /// Recovered from the down-projection's own weight, whose shape is
    /// `[hidden, intermediate]`. Llama layers may be dense or MoE, so the
    /// routed case reads one expert's projection and its own `n_expert_used`
    /// rather than assuming a dense FFN.
    fn wave_shapes(&self) -> WaveShapes {
        let (mlp, experts_per_tok, n_experts) = match &self.layers[0].mlp_or_moe {
            MlpOrMoe::Mlp(mlp) => (mlp, 1, 1),
            MlpOrMoe::MoE {
                n_expert_used,
                experts,
                ..
            } => (&experts[0], *n_expert_used, experts.len()),
        };
        // Panics on a non-2-D `ffn_down`: a zeroed shape would silently
        // mis-size the transient plan, which is worse than failing where the
        // weight is wrong.
        let (hidden, intermediate) = mlp
            .hidden_and_intermediate()
            .expect("ffn_down has a 2-D shape");
        WaveShapes {
            hidden,
            intermediate,
            experts_per_tok,
            n_experts,
            vocab: self.output.weight_dims().first().copied().unwrap_or(0),
            head_qk_norm: false,
            qkv_bias: false,
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_head(&self) -> usize {
        self.layers.first().map(|l| l.n_kv_head).unwrap_or(0)
    }

    fn head_dim(&self) -> usize {
        self.layers.first().map(|l| l.head_dim).unwrap_or(0)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn embeddings(&self) -> Option<&Embedding> {
        Some(&self.embeddings)
    }

    fn layer(&self, idx: usize) -> &Self::Layer {
        &self.layers[idx]
    }

    fn final_norm(&self) -> &RmsNorm {
        &self.norm
    }

    fn output_proj(&self) -> &QMatMul {
        &self.output
    }

    fn resident_weight_bytes(&self) -> Option<usize> {
        // Dense model: all weights are fixed and fully VRAM-resident.
        Some(self.base_weight_bytes)
    }

    fn rope_interleaved(&self) -> bool {
        // LLaMA uses interleaved RoPE format
        true
    }

    fn take_device_fault(&self) -> Result<()> {
        Ok(())
    }

    fn prune(&self) -> Result<()> {
        // Compact the embedding cache to free GPU copies
        self.embeddings.compact();

        // Compact the RoPE cache (shared across all layers via Arc)
        if let Some(layer) = self.layers.first() {
            if let Ok(mut cis) = layer.cis.write() {
                cis.compact();
            }
        }

        Ok(())
    }

    // The family row scaled by this checkpoint's per-generation scalar
    // (`compression_error_factor`, pinned by the `_v2`/`_v3` constructors) —
    // one multiplier across all four rows, so a generation dials the whole
    // calibration tighter or looser without forking the row.
    fn k_hi_error_threshold_factor(&self) -> f32 {
        LLAMA_KV_FACTORS.k_hi * self.compression_error_factor
    }

    fn k_low_error_threshold_factor(&self) -> f32 {
        LLAMA_KV_FACTORS.k_low * self.compression_error_factor
    }

    fn v_hi_error_threshold_factor(&self) -> f32 {
        LLAMA_KV_FACTORS.v_hi * self.compression_error_factor
    }

    fn v_low_error_threshold_factor(&self) -> f32 {
        LLAMA_KV_FACTORS.v_low * self.compression_error_factor
    }
}

impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let cis: SharedCis = Arc::new(RwLock::new(CisPrecomputations::new_growable(
            head_dim,
            10000.,
            MAX_ROPE_SEQ_LEN,
            ROPE_EXTEND_CHUNK,
            &ct.device,
        )?));
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;

            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                // w1 is the gate projection, w3 the up, w2 the down.
                MlpOrMoe::Mlp(QuantizedMlp::from_weights(
                    feed_forward_w1,
                    feed_forward_w3,
                    feed_forward_w2,
                    Int8Mode::Off,
                )?)
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cis: cis.clone(),
                neg_inf: neg_inf.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize)?,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: ct.device.clone(),
            span,
            span_output,
            // Legacy GGML path (not the GGUF load the daemon uses) — left unmeasured.
            base_weight_bytes: 0,
            compression_error_factor: 0.9,
        })
    }

    pub fn from_gguf_v2<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf(ct, reader, device, LLAMA2_KV_FACTOR)
    }

    pub fn from_gguf_v3<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf(ct, reader, device, LLAMA3_KV_FACTOR)
    }

    pub(super) fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        compression_error_factor: f32,
    ) -> Result<Self> {
        // The span is sized from the governor's measured capacity — without
        // one, the governor-less test constant cannot hold a 32-layer model's
        // arena floor plus the wave transient tier.
        #[cfg(feature = "cuda")]
        super::super::batched_model::ensure_vram_governor(device);

        // After the governor, so the span is sized from a measured card, and
        // before the baseline below, so the span's own mapping is not read as
        // weight bytes (`dense_span`).
        let _load = dense_span::open_for_load(device, &ct)?;

        // Driver-used VRAM baseline before any weights load (delta = weight footprint).
        #[cfg(feature = "cuda")]
        let used_before = super::super::batched_model::driver_used_bytes(device);
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        // Try to parse Llama-3-style RoPE scaling parameters if present.
        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok());
        let md_opt_str = |k: &str| ct.metadata.get(k).and_then(|v| v.to_string().ok()).cloned();

        let rope_scaling = {
            let factor = md_opt_f32("llama.rope.scaling.factor")
                .or_else(|| md_opt_f32("llama.rope.scale_factor"));
            let low_freq_factor = md_opt_f32("llama.rope.scaling.low_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.low_freq"));
            let high_freq_factor = md_opt_f32("llama.rope.scaling.high_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.high_freq"));
            let original_max_position_embeddings =
                md_opt_u32("llama.rope.scaling.original_max_position_embeddings")
                    .or_else(|| md_opt_u32("llama.rope.scaling.original_context_length"))
                    .map(|v| v as usize);

            let rope_type = md_opt_str("llama.rope.scaling.type")
                .or_else(|| md_opt_str("llama.rope.scaling.rope_type"))
                .map(|s| match s.as_str() {
                    "llama3" => Llama3RopeType::Llama3,
                    _ => Llama3RopeType::Default,
                })
                .unwrap_or(Llama3RopeType::Default);

            match (
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            ) {
                (Some(factor), Some(low), Some(high), Some(orig)) => Some(Llama3RopeConfig {
                    factor,
                    low_freq_factor: low,
                    high_freq_factor: high,
                    original_max_position_embeddings: orig,
                    rope_type,
                }),
                _ => None,
            }
        };

        let rope_freqs = match ct.tensor_infos.contains_key(ROPE_FREQS) {
            true => Some(ct.tensor(reader, ROPE_FREQS, &Device::Cpu)?),
            false => None,
        };
        let inv_freq = stated_inv_freq(rope_dim, rope_freq_base, rope_scaling, rope_freqs)?;
        let cis: SharedCis = Arc::new(RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
            inv_freq,
            MAX_ROPE_SEQ_LEN,
            ROPE_EXTEND_CHUNK,
            device,
        )?));
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp_or_moe = if n_expert <= 1 {
                MlpOrMoe::Mlp(QuantizedMlp::from_weights(
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?,
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?,
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?,
                    Int8Mode::Off,
                )?)
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    experts.push(QuantizedMlp::from_weights(
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?,
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?,
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?,
                        Int8Mode::Off,
                    )?)
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cis: cis.clone(),
                neg_inf: neg_inf.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        // Both halves: the driver delta sees only what went to the CUDA pool,
        // and whatever was carved from the span sits inside the mapping the
        // baseline already counted, so it has to be added rather than inferred.
        let dense_in_span = dense_span::close_load(device)?;
        #[cfg(feature = "cuda")]
        let base_weight_bytes = super::super::batched_model::driver_used_bytes(device)
            .saturating_sub(used_before)
            + dense_in_span;
        #[cfg(not(feature = "cuda"))]
        let base_weight_bytes = dense_in_span;
        Ok(Self {
            embeddings: Embedding::new(tok_embeddings, embedding_length)?,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: device.clone(),
            span,
            span_output,
            base_weight_bytes,
            compression_error_factor,
        })
    }

    /// Llama-2-family checkpoint by path, int8 mode picked from the device —
    /// the by-path twin of [`Self::from_gguf_v2`].
    pub fn from_gguf_by_path_v2(file_path: &std::path::Path, device: &Device) -> Result<Self> {
        Self::from_gguf_by_path_with_int8_v2(file_path, device, Int8Mode::auto(device))
    }

    /// Llama-3-family checkpoint by path, int8 mode picked from the device —
    /// the by-path twin of [`Self::from_gguf_v3`].
    pub fn from_gguf_by_path_v3(file_path: &std::path::Path, device: &Device) -> Result<Self> {
        Self::from_gguf_by_path_with_int8_v3(file_path, device, Int8Mode::auto(device))
    }

    pub fn from_gguf_by_path_with_int8_v2(
        file_path: &std::path::Path,
        device: &Device,
        int8mode: Int8Mode,
    ) -> Result<Self> {
        Self::from_gguf_by_path_with_int8(file_path, device, int8mode, LLAMA2_KV_FACTOR)
    }

    pub fn from_gguf_by_path_with_int8_v3(
        file_path: &std::path::Path,
        device: &Device,
        int8mode: Int8Mode,
    ) -> Result<Self> {
        Self::from_gguf_by_path_with_int8(file_path, device, int8mode, LLAMA3_KV_FACTOR)
    }

    /// Load from a GGUF file using memory-mapped I/O for zero-copy tensor
    /// loading: File (mmap) → GPU, one copy, no intermediate `Vec<u8>` and no
    /// doubled peak RAM, with the OS page cache doing the read scheduling.
    ///
    /// Callers go through the `_v2`/`_v3` wrappers, which pin the generation's
    /// KV factor; the int8 mode is explicit here and device-derived in
    /// [`Self::from_gguf_by_path_v2`] / [`Self::from_gguf_by_path_v3`].
    ///
    /// # Example
    /// ```no_run
    /// use candle::Device;
    /// use candle_transformers::models::quantized_llama::ModelWeights;
    /// use std::path::Path;
    ///
    /// let path = Path::new("model.gguf");
    /// let device = Device::cuda_if_available(0)?;
    /// let model = ModelWeights::from_gguf_by_path_v3(path, &device)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub(super) fn from_gguf_by_path_with_int8(
        file_path: &std::path::Path,
        device: &Device,
        int8mode: Int8Mode,
        compression_error_factor: f32,
    ) -> Result<Self> {
        use memmap2::MmapOptions;

        // Open file and create memory map for zero-copy access
        let file = std::fs::File::open(file_path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| candle::Error::Msg(format!("Failed to mmap file: {}", e)))?
        };

        // Register mmap with CUDA for DMA-accelerated host-to-device transfers
        #[cfg(feature = "cuda")]
        let _mmap_guard = if matches!(device, Device::Cuda(_)) {
            register_mmap_cuda(&mmap)
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let _mmap_guard: Option<()> = None;

        // Parse GGUF metadata from mmap (23x faster than reading from File!)
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let ct = gguf_file::Content::read(&mut cursor)?;

        // The span is sized from the governor's measured capacity — without
        // one, the governor-less test constant cannot hold a 32-layer model's
        // arena floor plus the wave transient tier.
        #[cfg(feature = "cuda")]
        super::super::batched_model::ensure_vram_governor(device);

        // After the governor, so the span is sized from a measured card, and
        // before the baseline below, so the span's own mapping is not read as
        // weight bytes (`dense_span`).
        let _load = dense_span::open_for_load(device, &ct)?;

        // Driver-used VRAM baseline before any weights load (delta = weight footprint).
        #[cfg(feature = "cuda")]
        let used_before = super::super::batched_model::driver_used_bytes(device);
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok());
        let md_opt_str = |k: &str| ct.metadata.get(k).and_then(|v| v.to_string().ok()).cloned();

        let rope_scaling = {
            let factor = md_opt_f32("llama.rope.scaling.factor")
                .or_else(|| md_opt_f32("llama.rope.scale_factor"));
            let low_freq_factor = md_opt_f32("llama.rope.scaling.low_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.low_freq"));
            let high_freq_factor = md_opt_f32("llama.rope.scaling.high_freq_factor")
                .or_else(|| md_opt_f32("llama.rope.scaling.high_freq"));
            let original_max_position_embeddings =
                md_opt_u32("llama.rope.scaling.original_max_position_embeddings")
                    .or_else(|| md_opt_u32("llama.rope.scaling.original_context_length"))
                    .map(|v| v as usize);

            let rope_type = md_opt_str("llama.rope.scaling.type")
                .or_else(|| md_opt_str("llama.rope.scaling.rope_type"))
                .map(|s| match s.as_str() {
                    "llama3" => Llama3RopeType::Llama3,
                    _ => Llama3RopeType::Default,
                })
                .unwrap_or(Llama3RopeType::Default);

            match (
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            ) {
                (Some(factor), Some(low), Some(high), Some(orig)) => Some(Llama3RopeConfig {
                    factor,
                    low_freq_factor: low,
                    high_freq_factor: high,
                    original_max_position_embeddings: orig,
                    rope_type,
                }),
                _ => None,
            }
        };

        let rope_freqs = match ct.tensor_infos.get(ROPE_FREQS) {
            Some(info) => Some(info.read_from_mmap(&mmap, ct.tensor_data_offset, &Device::Cpu)?),
            None => None,
        };
        let inv_freq = stated_inv_freq(rope_dim, rope_freq_base, rope_scaling, rope_freqs)?;
        let cis: SharedCis = Arc::new(RwLock::new(CisPrecomputations::new_growable_with_inv_freq(
            inv_freq,
            MAX_ROPE_SEQ_LEN,
            ROPE_EXTEND_CHUNK,
            device,
        )?));
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Helper to load tensor from mmap
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)
        };

        let tok_embeddings_q = load_tensor("token_embd.weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(load_tensor("output_norm.weight")?, rms_norm_eps)?;
        let output = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let attention_wk = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let attention_wv = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let attention_wo = load_tensor(&format!("{prefix}.attn_output.weight"))?;

            let mlp_or_moe = if n_expert <= 1 {
                // The dense MLP's `ffn_norm` emits int8 (q8a128) activations in int8 mode, so
                // its weights must be the matching KO twins. MoE experts receive FP activations
                // and pass `Int8Mode::Off`.
                MlpOrMoe::Mlp(QuantizedMlp::from_weights(
                    load_tensor(&format!("{prefix}.ffn_gate.weight"))?,
                    load_tensor(&format!("{prefix}.ffn_up.weight"))?,
                    load_tensor(&format!("{prefix}.ffn_down.weight"))?,
                    int8mode,
                )?)
            } else {
                let feed_forward_gate_inp = load_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    experts.push(QuantizedMlp::from_weights(
                        load_tensor(&format!("{prefix}.ffn_gate.{i}.weight"))?,
                        load_tensor(&format!("{prefix}.ffn_up.{i}.weight"))?,
                        load_tensor(&format!("{prefix}.ffn_down.{i}.weight"))?,
                        Int8Mode::Off,
                    )?)
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor_with_mode(
                        feed_forward_gate_inp,
                        int8mode,
                    )?,
                    experts,
                }
            };

            let attention_norm = load_tensor(&format!("{prefix}.attn_norm.weight"))?;
            let ffn_norm = load_tensor(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor_with_mode(attention_wq, int8mode)?,
                attention_wk: QMatMul::from_qtensor_with_mode(attention_wk, int8mode)?,
                attention_wv: QMatMul::from_qtensor_with_mode(attention_wv, int8mode)?,
                attention_wo: QMatMul::from_qtensor_with_mode(attention_wo, int8mode)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cis: cis.clone(),
                neg_inf: neg_inf.clone(),
                span_attn,
                span_rot,
                span_mlp,
            })
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        // Both halves: the driver delta sees only what went to the CUDA pool,
        // and whatever was carved from the span sits inside the mapping the
        // baseline already counted, so it has to be added rather than inferred.
        let dense_in_span = dense_span::close_load(device)?;
        #[cfg(feature = "cuda")]
        let base_weight_bytes = super::super::batched_model::driver_used_bytes(device)
            .saturating_sub(used_before)
            + dense_in_span;
        #[cfg(not(feature = "cuda"))]
        let base_weight_bytes = dense_in_span;
        Ok(Self {
            embeddings: Embedding::new(tok_embeddings, embedding_length)?,
            layers,
            norm,
            output: QMatMul::from_qtensor_with_mode(output, int8mode)?,
            device: device.clone(),
            span,
            span_output,
            base_weight_bytes,
            compression_error_factor,
        })
    }

    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = self
            .layers
            .iter()
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
    }

    /// Get the RoPE inv_freq values for use with BatchedInference wrapper.
    ///
    /// This returns the custom inv_freq computed during model loading (which may include
    /// rope scaling) — the file-stated frequencies a `RopeSchedule` is built over.
    pub fn rope_inv_freq(&self) -> Option<Vec<f32>> {
        self.layers
            .first()
            .and_then(|l| l.cis.read().ok().and_then(|cis| cis.inv_freq_vec()))
    }

    /// Forward pass (backwards compatible API).
    pub fn forward(&self, caches: &mut KvCaches, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward_with_context(&mut SequenceContext {
            kv_caches: caches,
            offset: index_pos,
            input_ids: x,
            input_len: x.dims2()?.1,
        })
    }

    /// Forward pass with strongly-typed sequence context.
    ///
    /// This is the preferred API for continuous batching scenarios where you manage
    /// multiple independent sequences. Each sequence has its own `KvCaches` instance.
    pub fn forward_with_context(&self, ctx: &mut SequenceContext<'_>) -> Result<Tensor> {
        if ctx.kv_caches.layer_count() != self.layers.len() {
            candle::bail!(
                "Cache count mismatch: expected {} caches, got {}",
                self.layers.len(),
                ctx.kv_caches.layer_count()
            );
        }
        let _enter = self.span.enter();
        let (_b, seq_len) = ctx.input_ids.dims2()?;
        let stage_prefix = if seq_len == 1 { "decode" } else { "prefill" };

        // Derive dtype from KV cache to ensure consistency throughout forward pass
        // For FP8 KV cache, use F16 activations (mixed precision)
        let embed_dtype = ctx.kv_caches.dtype();
        let mut layer_in = self
            .embeddings
            .forward(ctx.input_ids)?
            .to_dtype(embed_dtype)?;

        // Use forward_attn for consistent attention computation
        for (layer, cache) in self.layers.iter().zip(ctx.kv_caches.caches.iter_mut()) {
            let x = layer_in;
            let residual = &x;

            let dec = stage_prefix == "decode";
            let dv = x.device().clone();
            let g_attn_total = gpu_span_phase(
                dec,
                "decode:model:attn:total",
                "prefill:model:attn:total",
                &dv,
            );
            let g_attn_norm = gpu_span_phase(
                dec,
                "decode:model:attn:norm",
                "prefill:model:attn:norm",
                &dv,
            );
            let x = layer.attention_norm.forward(&x)?;
            g_attn_norm.end();

            let g_attn_core = gpu_span_phase(
                dec,
                "decode:model:attn:core",
                "prefill:model:attn:core",
                &dv,
            );
            let attn = layer
                .forward_attn(cache, &x, ctx.offset)?
                .to_dtype(embed_dtype)?;
            g_attn_core.end();

            let g_attn_resid = gpu_span_phase(
                dec,
                "decode:model:attn:resid",
                "prefill:model:attn:resid",
                &dv,
            );
            let x = (attn + residual)?;
            g_attn_resid.end();
            g_attn_total.end();

            // MLP
            let g_mlp_total = gpu_span_phase(
                dec,
                "decode:model:mlp:total",
                "prefill:model:mlp:total",
                &dv,
            );
            let _enter = layer.span_mlp.enter();
            let residual = &x;

            let g_mlp_norm =
                gpu_span_phase(dec, "decode:model:mlp:norm", "prefill:model:mlp:norm", &dv);
            let x = layer.ffn_norm.forward(&x)?;
            g_mlp_norm.end();

            let g_ffn = gpu_span_phase(
                dec,
                "decode:model:ffn:total",
                "prefill:model:ffn:total",
                &dv,
            );
            let x = layer.mlp_or_moe.forward(&x)?;
            g_ffn.end();

            let g_mlp_resid = gpu_span_phase(
                dec,
                "decode:model:mlp:resid",
                "prefill:model:mlp:resid",
                &dv,
            );
            let x = (x + residual)?;
            g_mlp_resid.end();
            g_mlp_total.end();

            layer_in = x
        }
        let g_norm = gpu_span_phase(
            stage_prefix == "decode",
            "decode:model:norm+proj",
            "prefill:model:norm+proj",
            layer_in.device(),
        );
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let _enter = self.span_output.enter();
        let out = self.output.forward(&x)?;
        g_norm.end();
        Ok(out)
    }

    /// Forward pass returning logits for ALL positions (for perplexity evaluation).
    ///
    /// Returns `[batch, seq_len, vocab]` instead of `[batch, vocab]`.
    pub fn forward_all_logits(
        &self,
        caches: &mut KvCaches,
        input: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        if caches.layer_count() != self.layers.len() {
            candle::bail!(
                "Cache count mismatch: expected {} caches, got {}",
                self.layers.len(),
                caches.layer_count()
            );
        }
        let _enter = self.span.enter();
        let embed_dtype = caches.dtype();
        let mut layer_in = self.embeddings.forward(input)?.to_dtype(embed_dtype)?;
        for (layer, cache) in self.layers.iter().zip(caches.caches.iter_mut()) {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer
                .forward_attn(cache, &x, offset)?
                .to_dtype(embed_dtype)?;
            let x = (attn + residual)?;
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}
