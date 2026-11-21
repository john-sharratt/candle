//! Loads a Qwen3-MoE model from its model pack: dense weights to the device, and
//! experts into the expert cache.
//!
//! Loading runs in a fixed order. The dense tensors are resident first, so the
//! weight span is sized from the free ground that remains. The span is then
//! reserved, the expert cache is filled into it, and the layers are grafted on.
//! The memory-mapped pack is copied out and not held, because an open mapping
//! slows the expert reads on Windows. The VRAM governor records the dense weights
//! by summing their tensors, since a driver delta cannot see through the caching
//! allocator. An embedding table that is a large share of the card is kept in
//! pinned host memory. `detect_arch_prefix` finds the GGUF metadata prefix, and
//! `warm_mmap` pre-loads the mapping's pages before the load reads them.

use super::*;

// VRAM budget helpers

/// Detect the GGUF metadata prefix for Qwen3-MoE models.
///
/// Tries `qwen3moe`, `qwen2moe`, then falls back to whatever
/// `general.architecture` says. Returns the prefix string (e.g. "qwen2moe").
pub(super) fn detect_arch_prefix(metadata: &HashMap<String, gguf_file::Value>) -> String {
    // Check general.architecture first
    if let Some(v) = metadata.get("general.architecture") {
        if let Ok(arch) = v.to_string() {
            let arch = arch.to_string();
            // The architecture value IS the prefix
            if metadata.contains_key(&format!("{arch}.block_count")) {
                tracing::debug!("GGUF arch prefix: '{arch}' (from general.architecture)");
                return arch;
            }
        }
    }
    // Fallback: probe known prefixes
    for prefix in &["qwen3moe", "qwen2moe", "qwen3", "llama"] {
        if metadata.contains_key(&format!("{prefix}.block_count")) {
            tracing::debug!("GGUF arch prefix: '{prefix}' (probed)");
            return prefix.to_string();
        }
    }
    tracing::debug!("GGUF arch prefix: 'qwen2moe' (default fallback)");
    "qwen2moe".to_string()
}

/// Pre-populate the OS page cache for a memory-mapped file.
///
/// 1. On Unix, calls `madvise(MADV_SEQUENTIAL)` then `madvise(MADV_WILLNEED)` so
///    the kernel begins readahead immediately.
/// 2. On all platforms, walks the mapping **backwards** in 8 MiB chunks, touching
///    one byte per 4 KiB page. Reading backwards ensures that if physical RAM is
///    insufficient to hold the entire file, the *beginning* of the file (read last)
///    remains hottest in the page cache — which is exactly what the model needs
///    first when loading layers sequentially.
pub(super) fn warm_mmap(mmap: &memmap2::Mmap) {
    let len = mmap.len();
    let size_mb = len as f64 / (1024.0 * 1024.0);
    tracing::debug!("warm_mmap: warming {size_mb:.1} MiB …");
    let t0 = std::time::Instant::now();

    // ── 1. madvise (Unix only) ───────────────────────────────────────────
    #[cfg(unix)]
    {
        use memmap2::Advice;
        // Sequential tells the kernel we will walk linearly (enables readahead).
        let _ = mmap.advise(Advice::Sequential);
        // WillNeed asks the kernel to start paging-in the entire region now.
        let _ = mmap.advise(Advice::WillNeed);
    }

    // ── 2. Backwards page-touch ──────────────────────────────────────────
    const CHUNK: usize = 8 * 1024 * 1024; // 8 MiB
    const PAGE: usize = 4096; // OS page size

    let ptr = mmap.as_ptr();
    let mut offset = len;
    let mut _acc: u8 = 0; // accumulator to prevent dead-code elimination

    while offset > 0 {
        let chunk_start = offset.saturating_sub(CHUNK);
        // Touch one byte per page inside this chunk (forward within the chunk).
        let mut pos = chunk_start;
        while pos < offset {
            // SAFETY: pos is always < mmap.len(), ptr valid for the mapping lifetime.
            _acc = _acc.wrapping_add(unsafe { *ptr.add(pos) });
            pos += PAGE;
        }
        offset = chunk_start;
    }

    // Ensure the accumulator is not optimised away.
    std::hint::black_box(_acc);

    let elapsed = t0.elapsed();
    tracing::debug!(
        "warm_mmap: done in {:.2}s ({:.0} MiB/s)",
        elapsed.as_secs_f64(),
        size_mb / elapsed.as_secs_f64()
    );
}

impl ModelWeights {
    /// Load the model from its model pack: the dense weights from the pack's
    /// GGUF part, straight to VRAM, and the experts into the expert cache from
    /// its expert section.
    ///
    /// The numeric mode is the one the pack was built for. An int8 mode repacks
    /// every dense weight (attention q/k/v/o, MoE router gate, dense-MLP
    /// gate/up/down, lm_head) to its KO twin so forward runs the q8a128 int8
    /// tensor-core matmul; the experts are already KO twins in the pack.
    ///
    /// `progress`, when supplied, is told each phase of the load and how far
    /// through it is — the dense layers mounted, then the expert cache's fill.
    pub fn from_pack(
        path: &std::path::Path,
        device: &Device,
        progress: Option<LoadProgress<'_>>,
    ) -> Result<Self> {
        let pack = ModelPack::open(path)?;
        let int8mode = pack.int8_mode;

        tracing::info!(
            "Inference int8 mode: {int8mode:?} (dense projections + MoE experts; \
             GPU int8 m16n8k32 MMA capable: {})",
            Int8Mode::auto(device).is_int8(),
        );

        let mmap = pack.mmap.clone();
        // Feed the host-RAM budget: the budget reserves the mapping so warm-KV
        // growth can never push weight pages out of RAM. The mapping is the
        // pack's GGUF part — the sections are read with direct I/O.
        candle::vram::set_weights_mmap(mmap.len() as u64);

        // **The mapping is deliberately not host-registered, and it does not
        // outlive the load.** Every tensor is copied out of it — the dense ones to
        // the device, the embedding into its own pinned table — so nothing reads
        // it after this function returns. Holding it would cost the expert reads
        // their overlap: on Windows a file with any cached view open has its
        // unbuffered reads served one at a time (6.5 → 4.1 GB/s on the 16 GB
        // laptop, `direct_io::tests::read_method_sweep`), and the expert section
        // is in the same file.
        let ct = &pack.content;

        // Before any tensor, so the weights are carved from the reservation
        // rather than from the CUDA pool (`dense_span`).
        let _load = dense_span::open_for_load(device, ct)?;

        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let md_opt_f32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_f32().ok());
        let md_opt_u32 = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok());

        let p = detect_arch_prefix(&ct.metadata);

        let num_attention_heads = md_get(&format!("{p}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads = md_get(&format!("{p}.attention.head_count_kv"))?.to_u32()? as usize;
        let num_layers = md_get(&format!("{p}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{p}.embedding_length"))?.to_u32()? as usize;

        let head_dim = md_opt_u32(&format!("{p}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| hidden_size / num_attention_heads);

        let max_position_embeddings = md_opt_u32(&format!("{p}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(32768);

        let rms_norm_eps =
            md_get(&format!("{p}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;

        let rope_freq_base = md_opt_f32(&format!("{p}.rope.freq_base"))
            .map(|v| v as f64)
            .unwrap_or(1_000_000.0);

        let declared = DeclaredScaling::from_gguf(&ct.metadata, &p)?;

        let n_expert = md_opt_u32(&format!("{p}.expert_count")).unwrap_or(1) as usize;
        let n_expert_used = md_opt_u32(&format!("{p}.expert_used_count")).unwrap_or(1) as usize;
        // Qwen3-MoE always uses norm_topk_prob=true. The GGUF key may be absent,
        // and the pack is the model's only file, so absent means the lineage's own
        // `true`.
        let norm_topk_prob = md_opt_u32(&format!("{p}.expert_weights_norm"))
            .map(|v| v == 1)
            .unwrap_or(true);

        tracing::debug!("GGUF arch: {p}  layers={num_layers} hidden={hidden_size} eps={rms_norm_eps:.2e} heads={num_attention_heads}Q/{num_kv_heads}KV head_dim={head_dim} ctx={max_position_embeddings} rope_base={rope_freq_base} experts={n_expert}/{n_expert_used} norm={norm_topk_prob}");

        // Expert FFN size (moe_intermediate_size)
        let expert_ffn_size =
            md_opt_u32(&format!("{p}.expert_feed_forward_length")).unwrap_or(2048) as usize;

        let dtype_model = DType::F16;

        // ── VRAM info (budget computed later once expert sizes are known) ──
        #[cfg(feature = "cuda")]
        let (free_vram, total_vram) = if matches!(device, Device::Cuda(_)) {
            get_vram_info()?
        } else {
            (0, 0)
        };
        #[cfg(not(feature = "cuda"))]
        let (free_vram, total_vram): (usize, usize) = (0, 0);

        tracing::debug!(
            "VRAM: total={:.1} GB, free={:.1} GB",
            total_vram as f64 / 1e9,
            free_vram as f64 / 1e9,
        );

        // ── VRAM Governor ──
        // Balloon (fast-path skip on a free card) to measure the real resident
        // capacity C, then install it so the expert budget, KV budget, and
        // scheduler all coordinate through one authority (see
        // `docs/vram_governor_design.md`).
        #[cfg(feature = "cuda")]
        let gpu_id = match device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
            _ => 0,
        };
        #[cfg(feature = "cuda")]
        super::super::batched_model::ensure_vram_governor(device);

        // Running total of the DENSE (non-expert) weight bytes this load puts on
        // the device — every tensor that goes through `load_tensor`, which is all
        // of them except the experts (those are read as raw `MmapExpertRef`
        // offsets straight into the cache's own slots).
        //
        // Summed rather than measured as a driver delta, because a driver delta
        // cannot see through the caching allocator: the governor's balloon
        // releases ~13 GiB back into the CUDA pool rather than to the driver, the
        // weights then allocate out of those cached blocks, and driver-used
        // barely moves. Measured, that read a 1313 MiB pre-load baseline against
        // a ~230 MiB context and under-reported the weights by a third.
        let dense_bytes = std::cell::Cell::new(0usize);

        // Helper: load tensor directly to VRAM
        let load_tensor = |name: &str| -> Result<QTensor> {
            let tensor_info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| candle::Error::Msg(format!("tensor {} not found", name)))?;
            let t = tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, device)?;
            dense_bytes.set(dense_bytes.get() + t.storage_size_in_bytes());
            Ok(t)
        };

        // Embeddings. The table is the worst VRAM-per-access ratio in the model —
        // one row read per token, but the whole thing resident and dequantized —
        // so on a card where it is a meaningful fraction of capacity it is copied
        // into pinned host memory and gathered per forward instead. See
        // `crate::models::host_embedding`.
        #[cfg(feature = "cuda")]
        let (embeddings, host_embedding, embed_dense_bytes) = {
            let info = ct
                .tensor_infos
                .get("token_embd.weight")
                .ok_or_else(|| candle::Error::Msg("tensor token_embd.weight not found".into()))?;
            // `[vocab, hidden]`: a row is one token's embedding, so `rows` is the
            // vocabulary and `cols` the hidden size. Not swapped — the shape is
            // already row-major here, and inverting it makes `cols` the vocab,
            // which fails the whole-blocks check rather than gathering garbage.
            let (rows, cols) = match info.shape.dims() {
                [r, c] => (*r, *c),
                dims => candle::bail!("token_embd.weight is not 2-D: {dims:?}"),
            };
            // What the table costs the CARD if it stays resident: `Embedding`
            // keeps its source on CPU and materialises a device copy at the
            // compute dtype on first forward, so the F32 dequantize below is a
            // transient, not the footprint.
            let resident_bytes = (rows * cols * dtype_model.size_in_bytes()) as u64;
            let capacity = candle::vram::get(gpu_id).map_or(0, |g| g.capacity());
            let serve_from_host =
                crate::models::host_embedding::should_serve_from_host(resident_bytes, capacity);

            let host = if serve_from_host {
                let byte_offset = ct.tensor_data_offset as usize + info.offset as usize;
                match crate::models::host_embedding::HostEmbedding::new(
                    &mmap,
                    byte_offset,
                    info.ggml_dtype,
                    rows,
                    cols,
                ) {
                    Ok(h) => Some(h),
                    // Not GPU-reachable (mmap registration failed) — fall back to
                    // the resident table rather than failing the load.
                    Err(e) => {
                        tracing::warn!(
                            target: "candle_transformers::quantized_qwen3_moe",
                            "host embedding unavailable ({e}); keeping it resident"
                        );
                        None
                    }
                }
            } else {
                None
            };

            match host {
                Some(h) => {
                    tracing::info!(
                        target: "candle_transformers::quantized_qwen3_moe",
                        resident_mib = resident_bytes >> 20,
                        capacity_mib = capacity >> 20,
                        "embedding served from pinned host memory; VRAM reclaimed"
                    );
                    (None, Some(h), 0usize)
                }
                None => {
                    let q = load_tensor("token_embd.weight")?;
                    let q_bytes = q.storage_size_in_bytes();
                    let t = q.dequantize(device)?;
                    dense_bytes.set(dense_bytes.get() - q_bytes);
                    // Charge the device cache, not the dequantize: `Embedding::new`
                    // moves `t` to CPU, so the F32 copy is gone before the first
                    // forward and only the compute-dtype variant is ever resident.
                    (
                        Some(Embedding::new(t, hidden_size)?),
                        None,
                        resident_bytes as usize,
                    )
                }
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (embeddings, embed_dense_bytes) = {
            let q = load_tensor("token_embd.weight")?;
            let q_bytes = q.storage_size_in_bytes();
            let t = q.dequantize(device)?;
            // As above: the resident copy is the compute-dtype device cache.
            let bytes = t.elem_count() * dtype_model.size_in_bytes();
            dense_bytes.set(dense_bytes.get() - q_bytes);
            (Some(Embedding::new(t, hidden_size)?), bytes)
        };
        dense_bytes.set(dense_bytes.get() + embed_dense_bytes);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype_model,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            declared,
            device,
        )?);

        // Count MoE layers for cache indexing
        let mut moe_count: usize = 0;

        // ── Compute expert byte offsets (3D merged primary, 2D fallback) ──
        // We collect these before loading layers so the ExpertCache can be built first.
        // Detect MoE layers by checking for expert tensors (not metadata-based).
        let moe_layer_indices: Vec<usize> = (0..num_layers)
            .filter(|i| {
                let prefix = format!("blk.{i}");
                ct.tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
                    || ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.ffn_gate_exps.weight"))
            })
            .collect();
        let num_moe_layers = moe_layer_indices.len();

        // ── The expert section: every MoE layer's records, in block order ──
        //
        // The pack holds the experts repacked; the GGUF part does not hold them
        // at all. The section's layers must be exactly the blocks that route,
        // in order, or a router's `moe_layer_idx` names another layer's experts.
        #[cfg(feature = "cuda")]
        let expert_section = match (num_moe_layers, pack.experts, device) {
            (0, _, _) => None,
            (_, None, _) => candle::bail!(
                "{} routes but its pack has no expert section",
                pack.path.display()
            ),
            (_, Some(at), Device::Cuda(cuda)) => {
                let section = open_section(&pack.path, at.offset, cuda)?;
                let header = section.header();
                let blocks: Vec<usize> = header.layers.iter().map(|l| l.block as usize).collect();
                if blocks != moe_layer_indices {
                    candle::bail!(
                        "the expert section holds blocks {blocks:?}, the model routes {:?}",
                        moe_layer_indices
                    );
                }
                if header.experts_per_layer as usize != n_expert {
                    candle::bail!(
                        "the expert section holds {} experts per layer, the metadata declares \
                         {n_expert}",
                        header.experts_per_layer
                    );
                }
                Some(section)
            }
            (_, Some(_), other) => {
                candle::bail!("the expert cache is a CUDA-only path, the model was given {other:?}")
            }
        };
        #[cfg(not(feature = "cuda"))]
        if num_moe_layers > 0 {
            candle::bail!("the expert cache is a CUDA-only path — build with the `cuda` feature");
        }

        // The dense weights load first and the expert cache after them, each
        // reporting its own phase. The order is fixed: the dense weights must
        // be **resident** before the span is reserved, because the span is
        // sized from `usable()` — a live measurement — and that is what
        // subtracts them exactly once.

        // The expert cache is built **after** the dense weights, further down.
        // Only the "is there one at all" question is answered here, because a
        // model without MoE layers warms its whole mmap now and never comes
        // back to this.
        let has_experts = num_moe_layers > 0 && n_expert > 0;
        if !has_experts {
            warm_mmap(&mmap);
        }

        // ── Load layers (dense) ──
        //
        // Everything except the experts. The FFN half of a MoE layer needs the
        // expert cache, which needs the span, which is sized from a measurement
        // taken once these are resident — so the loop yields a [`PendingLayer`]
        // and the cache is grafted on below.
        let mut pending: Vec<PendingLayer> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            let ln1 = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_norm.weight"))?,
                rms_norm_eps,
            )?;
            let ln2 = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.ffn_norm.weight"))?,
                rms_norm_eps,
            )?;

            // Attention
            let q_w = load_tensor(&format!("{prefix}.attn_q.weight"))?;
            let k_w = load_tensor(&format!("{prefix}.attn_k.weight"))?;
            let v_w = load_tensor(&format!("{prefix}.attn_v.weight"))?;
            let o_proj = QMatMul::from_weights_with_mode(
                load_tensor(&format!("{prefix}.attn_output.weight"))?.into(),
                int8mode,
            )?;
            let q_norm = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_q_norm.weight"))?,
                rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                load_tensor(&format!("{prefix}.attn_k_norm.weight"))?,
                rms_norm_eps,
            )?;

            // q/k/v kept separate KO twins (no concat): the segmented kernel fuses the three at
            // launch — any KO formats, one occupied launch.
            let q_proj = Some(QMatMul::from_weights_with_mode(q_w.into(), int8mode)?);
            let k_proj = Some(QMatMul::from_weights_with_mode(k_w.into(), int8mode)?);
            let v_proj = Some(QMatMul::from_weights_with_mode(v_w.into(), int8mode)?);

            let self_attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: num_attention_heads,
                num_kv_heads,
                head_dim,
                rotary_emb: rotary.clone(),
            };

            // FFN: detect MoE vs dense by checking for expert tensors
            let has_moe_tensors = ct
                .tensor_infos
                .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
                || ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

            let ffn = if has_moe_tensors && n_expert > 1 {
                let gate = QMatMul::from_weights_with_mode(
                    load_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?.into(),
                    int8mode,
                )?;
                let moe_layer_idx = moe_count;
                moe_count += 1;
                PendingFfn::MoE {
                    gate,
                    moe_layer_idx,
                }
            } else {
                // Dense MLP
                PendingFfn::Mlp(QuantizedMlp::from_weights(
                    load_tensor(&format!("{prefix}.ffn_gate.weight"))?,
                    load_tensor(&format!("{prefix}.ffn_up.weight"))?,
                    load_tensor(&format!("{prefix}.ffn_down.weight"))?,
                    int8mode,
                )?)
            };

            pending.push(PendingLayer {
                self_attn,
                ffn,
                ln1,
                ln2,
            });

            if (i + 1) % 8 == 0 || i == num_layers - 1 {
                tracing::debug!("Layer {}/{} loaded", i + 1, num_layers);
            }
            if let Some(report) = progress {
                report(LoadPhase::Weights, i + 1, num_layers);
            }
        }

        // Load final norm and output projection — the last of the dense weights.
        let norm = RmsNorm::from_qtensor(load_tensor("output_norm.weight")?, rms_norm_eps)?;
        let lm_head_tensor = match load_tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => load_tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights_with_mode(lm_head_tensor.into(), int8mode)?;
        // Read before the struct takes ownership of `lm_head`.
        let vocab = lm_head.weight_dims().first().copied().unwrap_or(0);

        // ── Reserve the span, then build the expert cache into it ──
        //
        // This is the moment §4 describes: every dense tensor is resident, so
        // `usable()` reports what is genuinely left, and the reservation takes
        // it. Nothing after this point allocates weights outside the span.
        #[cfg(feature = "cuda")]
        if let Some(g) = candle::vram::get(gpu_id) {
            g.set_class(candle::vram::AllocClass::Weights, dense_bytes.get() as u64);
        }

        // The load phase closes here, before the expert cache sizes its zone from
        // the span's free ground — that is not knowable while the dense block can
        // still grow. `base_weight_bytes` below needs no adjustment: it is summed
        // from the tensors themselves (`dense_bytes`), which is immune to where
        // those tensors were placed, unlike the driver deltas the dense loaders use.
        dense_span::close_load(device)?;

        #[cfg(feature = "cuda")]
        let expert_cache = if let Some(section) = expert_section {
            let total_experts = num_moe_layers * n_expert;
            //
            // There used to be one — `pending_dense_bytes`, summed from the GGUF
            // tensor table — because the expert budget was computed while ~1 GiB
            // of dense tensors had not loaded yet, so the KV side paid for the
            // model's own weights (measured at 2,912 MiB where the partition
            // intended ~4,439). The forecast existed only because the ordering
            // was wrong.
            //
            // Under the current order every dense tensor is resident before this
            // point, so `usable()` — the drop in headroom since `C` was measured
            // — has already netted them out, exactly once. Re-declaring them
            // would book the same bytes twice.
            //
            // ── The span decides the slot count ──
            //
            // `VramGovernor::expert_budget()` used to compute a byte budget here
            // and divide it by `max_expert_size`. Both are gone. The weight zone
            // is a region of the reservation, its capacity is
            // `(span − MIN_ELASTIC_RESERVE) / slot_bytes`, and that quotient
            // **is** the resident-expert count — there is no second arithmetic
            // deriving one from the other, and no forecast of anything.
            //
            // `slot_bytes` comes from the *repacked* geometry rather than the
            // raw GGML lengths the old budget used. A slot holds one expert's
            // three projections at aligned offsets, and it is what the zone is
            // carved into, so it has to be the exact figure the upload writes.
            #[cfg(feature = "cuda")]
            let zone = if let Device::Cuda(cuda_dev) = device {
                let stream = cuda_dev.cuda_stream();
                let geoms = geometries_of(section.header())?;
                let slot_bytes = slot_bytes_for(&geoms);
                // Two different numbers: where the boundary starts, and how far
                // it may ever go. The zone opens at `initial` — sized to leave
                // the KV side its measured cold-boot peak — and its `limit` is
                // the hard floor it may grow to once the KV side has shown what
                // it actually uses.
                let limit_bytes = weight_capacity_bytes(&stream)?;
                let initial_bytes = initial_weight_bytes(&stream)?;
                let slots_in = |bytes: usize| {
                    bytes
                        .checked_div(slot_bytes)
                        .map_or(0, |n| n.min(total_experts))
                };
                // The zone's floor, and the only bound on how much expert
                // residency the KV side can buy — so it is the pinning rule's
                // own arithmetic, not a fraction of wherever the boundary opened.
                let floor = minimum_resident_slots(n_expert);
                // **Opened at the floor at least**, as the qwen35 loader does.
                // `PINNED_LAYERS` is a constant now, so a cache below the floor
                // cannot shed pinned layers to cope — it fills with experts the
                // eviction scan may not touch and then fails every load. Ground
                // taken here is given back by the elastic boundary once the KV
                // side shows what it really uses; opening below the floor is not
                // a slower engine but a stopped one.
                let capacity = slots_in(initial_bytes).max(floor).min(total_experts);
                let limit = slots_in(limit_bytes).max(capacity);
                let zone = WeightZone::new(span_end(&stream)?, slot_bytes, capacity, limit, floor);
                // Place the boundary. Everything left of it belongs to the KV
                // side, and the region count is re-derived from it here rather
                // than assumed anywhere.
                let regions = set_weight_floor(&stream, zone.frontier_for_capacity())?;
                tracing::info!(
                    target: "candle_transformers::quantized_qwen3_moe",
                    slots = capacity,
                    max_slots = limit,
                    floor_slots = floor,
                    total_experts,
                    slot_bytes,
                    weight_gib = (capacity * slot_bytes) as f64 / 1e9,
                    model_gib = (total_experts * slot_bytes) as f64 / 1e9,
                    kv_regions = regions,
                    "expert cache opened against the span"
                );
                zone
            } else {
                // A cuda build on a CPU device has no expert path at all, and
                // `ExpertCache::new` says so. The zone it would be handed has no
                // device reservation behind it, so a plausible-looking one built
                // here would only make the failure land further from its cause.
                WeightZone::new(0, 0, 0, 0, 0)
            };
            let capacity = zone.capacity();
            let slot_bytes = zone.slot_bytes();

            let cache = ExpertCache::new(ExpertCacheSetup {
                pack: section,
                zone,
                device,
                experts_per_layer: n_expert,
                experts_used: n_expert_used,
                progress,
                int8mode,
                mapped_bytes: pack.gguf_len(),
                // Nothing in the mapping is read from the host after load: the
                // dense tensors above are resident on the card, and the
                // embedding is either copied into its own pinned allocation or
                // dequantized into a host tensor of its own. Counted as live
                // weight, the whole GGUF part would be reserved as page cache
                // out of the warm tier's RAM.
                offloaded_bytes: pack.gguf_len(),
            })?;
            // Record the resident expert footprint with the governor. Reporting
            // only — nothing sizes itself from this any more.
            #[cfg(feature = "cuda")]
            if let Some(g) = candle::vram::get(gpu_id) {
                g.set_class(
                    candle::vram::AllocClass::Expert,
                    (capacity * slot_bytes) as u64,
                );
            }
            let cache = Arc::new(cache);
            // **Open the shop.** From here a KV arena claim that runs out of
            // ground can buy more, at the price of expert residency, instead of
            // refusing and leaving the demand for something else to interpret.
            //
            // Registered against this device's ordinal, because the weight zone
            // it sells from is this device's. A `Weak` so the registry — which
            // outlives every model, being static — does not keep the cache alive
            // past the model that owns it; a dead reference answers zero, the
            // same answer as no seller, and the next model on this ordinal
            // replaces the registration outright.
            #[cfg(feature = "cuda")]
            {
                let seller = Arc::downgrade(&cache);
                candle_nn::kv_cache::set_ground_broker(gpu_id, move |regions| {
                    seller
                        .upgrade()
                        .map_or(0, |cache| cache.request_kv_ground(regions))
                });
            }
            Some(cache)
        } else {
            None
        };
        // A build without CUDA refused a routed model above, so there is no
        // cache to build here.
        #[cfg(not(feature = "cuda"))]
        let expert_cache: Option<Arc<ExpertCache>> = None;

        // ── Graft the cache onto the layers ──
        let layers = pending
            .into_iter()
            .map(|p| p.resolve(expert_cache.as_ref(), n_expert_used, norm_topk_prob))
            .collect::<Result<Vec<_>>>()?;

        tracing::debug!(
            "Model loaded: {} layers ({} MoE), int8mode={:?}",
            num_layers,
            moe_count,
            int8mode
        );

        // Dense (non-expert) weight VRAM: the exact sum `load_tensor` accumulated.
        // Fixed for the session — dense weights never move — while the experts
        // are the time-varying part `resident_weight_bytes` adds back on top, so
        // nothing double-counts.
        //
        // This was a driver delta bracketed around the expert load, and it was
        // wrong twice over: it read 3936 MiB, then 733 MiB after the bracket was
        // anchored on the expert gauge, against a true ~990 MiB (the embedding
        // tensor alone is 594 MiB dequantized, and `lm_head` another ~167 MiB).
        // A driver delta cannot measure this at all — the governor's balloon
        // releases ~13 GiB into the CUDA pool rather than back to the driver, the
        // weights then allocate out of those cached blocks, and driver-used
        // barely moves. Summing the tensors is immune to both the caching
        // allocator and to load-time transients.
        let base_weight_bytes: usize = dense_bytes.get();
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            let expert_gauge = expert_cache.as_ref().map_or(0, |c| c.resident_vram_bytes());
            tracing::info!(
                target: "candle_transformers::quantized_qwen3_moe",
                base_mib = base_weight_bytes >> 20,
                expert_gauge_mib = expert_gauge >> 20,
                "weight VRAM breakdown (base = summed dense tensors)"
            );
            // Record the dense weights with the governor. `kv_floor` is
            // `abs + pct x (C - weights)`, so leaving this unrecorded computes
            // the floor against the whole card as though the model were free —
            // inflating the floor and taking the difference straight out of the
            // expert budget, which is the scarcest thing on a tight card.
            // Set, not add: this is the session's whole dense footprint, and
            // adding it a second time would drive `C - weights` to zero.
            if let Some(g) = candle::vram::get(gpu_id) {
                g.set_class(candle::vram::AllocClass::Weights, base_weight_bytes as u64);
            }
        }

        Ok(Self {
            embeddings,
            #[cfg(feature = "cuda")]
            host_embedding,
            layers,
            norm,
            lm_head,
            expert_cache,
            wave_shapes: WaveShapes {
                hidden: hidden_size,
                intermediate: expert_ffn_size,
                experts_per_tok: n_expert_used,
                n_experts: n_expert,
                vocab,
                head_qk_norm: true,
                qkv_bias: false,
            },
            device: device.clone(),
            int8mode,
            base_weight_bytes,
        })
    }

    /// Create KV caches for all layers.
    pub fn create_kv_caches(&self, initial_capacity: usize) -> KvCaches {
        let caches = (0..self.layers.len())
            .map(|_| KvCache::new(2, initial_capacity))
            .collect();
        new_kv_caches(caches, self.device.clone())
    }

    /// Returns the RoPE inverse frequency vector used by this model.
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
