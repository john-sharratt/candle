//! Loads a Qwen3 GGUF checkpoint onto the device and runs its forward passes, with the
//! resident weight footprint measured at load.
//!
//! The footprint is the driver's allocation delta across loading, plus whatever the
//! weights carved from the reserved span, since span memory is already inside the
//! baseline and does not show in the delta. Reporting the delta alone would make a
//! model that moved into the reservation look almost empty. Loading reads the
//! tensors through a memory map, so each one is copied to the device once. The output
//! projection falls back to the tied embedding when the checkpoint has none. The
//! model's dtype comes from its metadata, and its key names accept a few converter
//! variants. A forward pass takes a per-sequence cache set, so independent sequences
//! share weights while keeping their own caches. Sequential calls reuse the weights
//! but do not batch the kernels across sequences, since each sequence's rotary
//! positions differ.

use super::*;

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub(super) embeddings: Embedding,
    pub(super) layers: Vec<LayerWeights>,
    pub(super) norm: RmsNorm,
    pub(super) lm_head: QMatMul,
    pub(super) device: Device,
    pub(super) span: tracing::Span,
    pub(super) span_output: tracing::Span,
    /// Fixed resident-weight VRAM (bytes), measured at load as the driver-used
    /// delta across weight loading. Dense model: all weights stay VRAM-resident,
    /// so this is the whole weight footprint. `0` on non-CUDA.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) base_weight_bytes: usize,
}

/// Implementation of `BatchedModelCore` for use with `BatchedInference` wrapper.
///
/// This is the new recommended way to use batched inference. The `BatchedInference`
/// wrapper handles RoPE caching at the model level, so this implementation is simpler.
#[cfg(feature = "cuda")]
impl BatchedModelCore for ModelWeights {
    type Layer = LayerWeights;

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        for layer in &self.layers {
            layer.ln1.maybe_change_dtype(dtype)?;
            layer.ln2.maybe_change_dtype(dtype)?;
            layer.self_attn.q_norm.maybe_change_dtype(dtype)?;
            layer.self_attn.k_norm.maybe_change_dtype(dtype)?;
        }
        self.norm.maybe_change_dtype(dtype)
    }

    /// Recovered from the down-projection's own weight — see
    /// [`QuantizedMlp::hidden_and_intermediate`].
    fn wave_shapes(&self) -> WaveShapes {
        // Panics on a non-2-D `ffn_down`, exactly as the direct indexing it
        // replaced did: a zeroed shape would silently mis-size the transient
        // plan, which is worse than failing where the weight is wrong.
        let (hidden, intermediate) = self.layers[0]
            .mlp
            .hidden_and_intermediate()
            .expect("ffn_down has a 2-D shape");
        WaveShapes {
            hidden,
            intermediate,
            experts_per_tok: 1,
            n_experts: 1,
            vocab: self.lm_head.weight_dims().first().copied().unwrap_or(0),
            head_qk_norm: true,
            qkv_bias: false,
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_kv_head(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.num_kv_heads)
            .unwrap_or(0)
    }

    fn head_dim(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.head_dim)
            .unwrap_or(0)
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
        &self.lm_head
    }

    fn resident_weight_bytes(&self) -> Option<usize> {
        // Dense model: all weights are fixed and fully VRAM-resident.
        Some(self.base_weight_bytes)
    }

    fn rope_interleaved(&self) -> bool {
        // Qwen3 uses standard (non-interleaved) RoPE format
        false
    }

    fn take_device_fault(&self) -> Result<()> {
        Ok(())
    }

    fn prune(&self) -> Result<()> {
        self.embeddings.compact();
        if let Some(layer) = self.layers.first() {
            if let Ok(mut cis) = layer.self_attn.rotary_emb.cis.write() {
                cis.compact();
            }
        }
        Ok(())
    }

    fn k_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.k_hi
    }

    fn k_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.k_low
    }

    fn v_hi_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.v_hi
    }

    fn v_low_error_threshold_factor(&self) -> f32 {
        candle_nn::kv_cache::QWEN3_8B_KV_FACTORS.v_low
    }
}

impl ModelWeights {
    /// Load model from GGUF file with
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Claim the reservation before the baseline below is taken, so the span's
        // own mapping is not read as weight bytes — and before any tensor, which
        // is what lets the weights be carved out of it (`dense_span`).
        let _load = dense_span::open_for_load(device, &ct)?;
        let mut gg = Gguf::new(ct, reader, device.clone());
        // Driver-used VRAM baseline before any weights load — the delta to the
        // fully-built model is the fixed resident-weight footprint.
        #[cfg(feature = "cuda")]
        let used_before = super::super::batched_model::driver_used_bytes(device);
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let md_opt_f32 = |k: &str| gg.metadata().get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| gg.metadata().get(k).and_then(|v| v.to_u32().ok());

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;

        // Some converters use different key names; accept a few common ones.
        let head_dim = md_opt_u32("qwen3.attention.key_length")
            .or_else(|| md_opt_u32("qwen3.attention.head_dim"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);

        let max_position_embeddings = md_opt_u32("qwen3.context_length")
            .or_else(|| md_opt_u32("qwen3.max_position_embeddings"))
            .or_else(|| md_opt_u32("max_position_embeddings"))
            // 0 means "no key found"; the check below turns that into the
            // original hard failure rather than silently accepting it.
            .unwrap_or(0) as usize;
        if max_position_embeddings == 0 {
            let _ = md_get("qwen3.context_length")?;
        }

        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_opt_f32("qwen3.rope.freq_base")
            .or_else(|| md_opt_f32("qwen3.rope.theta"))
            .or_else(|| md_opt_f32("rope.freq_base"))
            .or_else(|| md_opt_f32("rope.theta"))
            .unwrap_or(1_000_000f32) as f64;

        let declared = DeclaredScaling::from_gguf(gg.metadata(), "qwen3")?;

        // Extract model's native dtype from metadata
        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;

        let tok_embed = embed_tensor.dequantize(device)?;
        let embeddings = Embedding::new(tok_embed, hidden_size)?;

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            declared,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        // **Both halves, because the weights now land in two places.** The driver
        // delta sees only what went to the CUDA pool; whatever was carved from
        // the span is already inside the mapping the baseline counted, so it
        // contributes nothing to the delta and has to be added explicitly.
        // Reporting the delta alone would show a model that shrank to almost
        // nothing the moment it moved into the reservation.
        let dense_in_span = dense_span::close_load(device)?;
        #[cfg(feature = "cuda")]
        let base_weight_bytes = super::super::batched_model::driver_used_bytes(device)
            .saturating_sub(used_before)
            + dense_in_span;
        #[cfg(not(feature = "cuda"))]
        let base_weight_bytes = dense_in_span;
        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            span,
            span_output,
            base_weight_bytes,
        })
    }

    /// Load model from GGUF file using memory-mapped I/O for zero-copy tensor loading.
    ///
    /// This method eliminates intermediate RAM allocations and copies by using mmap:
    /// - Traditional: File → Vec<u8> → GPU (2 copies, 2x peak RAM)
    /// - This method: File (mmap) → GPU (1 copy, 1x peak RAM)
    ///
    /// Benefits:
    /// - **Eliminates RAM allocation** for tensor data
    /// - **Eliminates file→RAM copy** - only mmap→GPU remains
    /// - **Lower peak memory usage** - no temporary buffers
    /// - **OS page cache efficiency** - kernel optimizes page access
    ///
    /// # Arguments
    /// * `file_path` - Path to the GGUF file
    /// * `device` - Device to load tensors onto
    ///
    /// # Example
    /// ```no_run
    /// use candle::Device;
    /// use candle_transformers::models::quantized_qwen3::ModelWeights;
    /// use std::path::Path;
    ///
    /// let path = Path::new("model.gguf");
    /// let device = Device::cuda_if_available(0)?;
    /// let model = ModelWeights::from_gguf_by_path(path, &device)?;
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn from_gguf_by_path(file_path: &std::path::Path, device: &Device) -> Result<Self> {
        // VRAM-aware auto: Precision when the weights leave headroom, else
        // Performance (smaller); Off on CPU. Sized by the GGUF length.
        let model_bytes = std::fs::metadata(file_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        Self::from_gguf_by_path_with_int8(
            file_path,
            device,
            Int8Mode::auto_sized(device, model_bytes),
        )
    }

    /// Like from_gguf_by_path but with an explicit numeric int8mode; from_gguf_by_path
    /// defaults it to Int8Mode::auto.
    pub fn from_gguf_by_path_with_int8(
        file_path: &std::path::Path,
        device: &Device,
        int8mode: Int8Mode,
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

        // Before the baseline, so the span's own mapping is not read as weight
        // bytes, and before any tensor, so the weights can be carved from it.
        let _load = dense_span::open_for_load(device, &ct)?;

        // Driver-used VRAM baseline before any weights load — the delta to the
        // fully-built model is the fixed resident-weight footprint.
        #[cfg(feature = "cuda")]
        let used_before = super::super::batched_model::driver_used_bytes(device);

        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Extract model hyperparameters
        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());

        let rope_freq_base = md_opt_f32("qwen3.rope.freq_base")
            .or_else(|| md_opt_f32("qwen3.rope.theta"))
            .or_else(|| md_opt_f32("rope.freq_base"))
            .or_else(|| md_opt_f32("rope.theta"))
            .unwrap_or(1_000_000f32) as f64;

        let declared = DeclaredScaling::from_gguf(&ct.metadata, "qwen3")?;

        // Extract model's native dtype from metadata
        let dtype = match ct.metadata.get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        // Helper to load tensor from mmap
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)
        };

        let load_qmatmul = |name: &str| -> Result<QMatMul> {
            QMatMul::from_weights_with_mode(load_tensor(name)?.into(), int8mode)
        };

        let load_rms_norm = |name: &str, eps: f64| -> Result<RmsNorm> {
            RmsNorm::from_qtensor(load_tensor(name)?, eps)
        };

        // Load embeddings
        let embed_tensor = load_tensor("token_embd.weight")?;
        let tok_embed = embed_tensor.dequantize(device)?;
        let embeddings = Embedding::new(tok_embed, hidden_size)?;

        // Create rotary embeddings
        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            declared,
            device,
        )?);

        // Load all layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            let ln1 = load_rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
            let ln2 = load_rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

            // Load attention weights
            let q_w = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let k_w = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let v_w = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let o_proj = load_qmatmul(&format!("{prefix}.attn_output.weight"))?;
            let q_norm = load_rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
            let k_norm = load_rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

            // q/k/v kept separate KO twins (no concat): the segmented kernel fuses them at launch.
            let q_proj = QMatMul::from_weights_with_mode(q_w.into(), int8mode)?;
            let k_proj = QMatMul::from_weights_with_mode(k_w.into(), int8mode)?;
            let v_proj = QMatMul::from_weights_with_mode(v_w.into(), int8mode)?;

            let self_attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads,
                num_kv_groups: num_attention_heads / num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
            };

            let mlp = QuantizedMlp::from_weights(
                load_tensor(&format!("{prefix}.ffn_gate.weight"))?,
                load_tensor(&format!("{prefix}.ffn_up.weight"))?,
                load_tensor(&format!("{prefix}.ffn_down.weight"))?,
                int8mode,
            )?;

            layers.push(LayerWeights {
                self_attn,
                mlp,
                ln1,
                ln2,
            });
        }

        // Load final norm and output projection
        let norm = load_rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => load_tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights_with_mode(lm_head_tensor.into(), int8mode)?;

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        // Both halves — see the sibling loader above.
        let dense_in_span = dense_span::close_load(device)?;
        #[cfg(feature = "cuda")]
        let base_weight_bytes = super::super::batched_model::driver_used_bytes(device)
            .saturating_sub(used_before)
            + dense_in_span;
        #[cfg(not(feature = "cuda"))]
        let base_weight_bytes = dense_in_span;
        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            span,
            span_output,
            base_weight_bytes,
        })
    }

    /// Create KV caches for all layers (regular)
    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = (0..self.layers.len())
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
    }

    /// Forward pass with strongly-typed sequence context.
    ///
    /// This is the preferred API for continuous batching scenarios where you manage
    /// multiple independent sequences. Each sequence has its own `KvCaches` instance.
    ///
    /// # Continuous Batching Semantics
    ///
    /// **Important**: Sequential calls to this method provide *continuous batching*
    /// (concurrent sequence management with weight reuse) but NOT *true GPU batch
    /// parallelism* (processing multiple sequences simultaneously in one kernel launch).
    ///
    /// Benefits achieved:
    /// - **Weight caching**: Model weights stay in L2 cache across sequences
    /// - **Concurrent management**: Handle multiple sequences in flight
    /// - **Memory efficiency**: Share model weights, separate KV caches
    ///
    /// Benefits NOT achieved:
    /// - **Batch parallelism**: Each forward() processes one sequence independently
    /// - **Tensor core batching**: No amortized kernel launch overhead
    ///
    /// This is the correct design because RoPE position embeddings are per-token
    /// and prevent batching sequences at different offsets without custom kernels.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use candle::{Device, Tensor};
    /// # use candle_transformers::models::quantized_qwen3::ModelWeights;
    /// # use candle_transformers::models::kv_cache_utils::SequenceContext;
    /// # fn example() -> candle::Result<()> {
    /// # let device = Device::cuda_if_available(0)?;
    /// # let model_path = std::path::Path::new("model.gguf");
    /// # let model = ModelWeights::from_gguf_by_path(model_path, &device)?;
    /// // Manage multiple independent sequences
    /// let mut seq1_caches = model.create_kv_caches(512);
    /// let mut seq2_caches = model.create_kv_caches(512);
    ///
    /// // Process sequence 1
    /// let seq1_tokens = Tensor::new(&[1u32, 2, 3], &device)?.unsqueeze(0)?;
    /// let ctx1 = SequenceContext {
    ///     kv_caches: &mut seq1_caches,
    ///     offset: 0,
    ///     input_ids: &seq1_tokens,
    ///     input_len: 3,
    /// };
    /// let output1 = model.forward_with_context(ctx1)?;
    ///
    /// // Process sequence 2 (independent, weights cached in L2)
    /// let seq2_tokens = Tensor::new(&[4u32, 5, 6], &device)?.unsqueeze(0)?;
    /// let ctx2 = SequenceContext {
    ///     kv_caches: &mut seq2_caches,
    ///     offset: 0,
    ///     input_ids: &seq2_tokens,
    ///     input_len: 3,
    /// };
    /// let output2 = model.forward_with_context(ctx2)?;
    /// # Ok(())
    /// # }
    /// ```
    /// Forward pass with typed context struct (preferred API).
    pub fn forward_with_context(&self, ctx: SequenceContext<'_>) -> Result<Tensor> {
        if ctx.kv_caches.layer_count() != self.layers.len() {
            candle::bail!(
                "Cache count mismatch: expected {} caches, got {}",
                self.layers.len(),
                ctx.kv_caches.layer_count()
            );
        }
        let _enter = self.span.enter();
        let (_b, l) = ctx.input_ids.dims2()?;

        // Derive dtype from KV cache to ensure consistency throughout forward pass
        let embed_dtype = ctx.kv_caches.dtype();
        let mut h = self
            .embeddings
            .forward_as_dtype(ctx.input_ids, embed_dtype)?
            .contiguous()?;

        for (layer, cache) in self.layers.iter().zip(ctx.kv_caches.caches.iter_mut()) {
            h = layer.forward(cache, &h, ctx.offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?.contiguous()?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    /// Forward pass (backwards compatible API).
    pub fn forward(&self, caches: &mut KvCaches, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_with_context(SequenceContext {
            kv_caches: caches,
            offset,
            input_ids: input,
            input_len: input.dims2()?.1,
        })
    }

    /// Forward pass returning logits for ALL positions (for perplexity evaluation).
    ///
    /// Unlike `forward` which returns only the last token's logits `[batch, vocab]`,
    /// this returns `[batch, seq_len, vocab]` so cross-entropy loss can be computed
    /// against shifted targets.
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
        let mut h = self
            .embeddings
            .forward_as_dtype(input, embed_dtype)?
            .contiguous()?;
        for (layer, cache) in self.layers.iter().zip(caches.caches.iter_mut()) {
            h = layer.forward(cache, &h, offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        self.lm_head.forward(&h)
    }

    /// Returns the RoPE inverse frequency vector used by this model.
    ///
    /// This includes any RoPE scaling (e.g., for extended context) that was
    /// configured when the model was loaded — the file-stated frequencies a
    /// `RopeSchedule` is built over.
    pub fn rope_inv_freq(&self) -> Option<Vec<f32>> {
        self.layers.first().and_then(|l| {
            l.self_attn
                .rotary_emb
                .cis
                .read()
                .ok()
                .and_then(|cis| cis.inv_freq_vec())
        })
    }
}
