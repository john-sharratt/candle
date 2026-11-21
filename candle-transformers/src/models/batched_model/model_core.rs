//! The contract a model implements to run batched waves, and the shapes and
//! dtypes that a wave's memory is sized from.
//!
//! A model supplies its layers and accessors through `BatchedModelCore`, and
//! reports the shapes it was loaded with. From those the transient tier a wave
//! needs is priced, and the same pricing is the one the tier is placed from.
//! Activations take the dtype of the KV cache, except F8E4M3, which has no
//! compute kernels and so computes in BF16. Norm weights are converted to the
//! activation dtype once, when a session is created, and never inside a wave,
//! because a conversion per norm per token is invisible in a profile and costs
//! a launch each time.

use super::*;

/// Driver-used VRAM right now (`total − free` from `cuMemGetInfo`), in bytes;
/// `0` on non-CUDA / query failure. Dense models capture this at the start and
/// end of weight loading — the delta is their fixed resident-weight footprint,
/// reported through [`BatchedModelCore::resident_weight_bytes`] for the
/// whole-card VRAM decomposition.
#[cfg(feature = "cuda")]
pub(crate) fn driver_used_bytes(device: &Device) -> usize {
    if matches!(device, Device::Cuda(_)) {
        candle::quantized::get_vram_info()
            .map(|(free, total)| total.saturating_sub(free))
            .unwrap_or(0)
    } else {
        0
    }
}

/// Install the VRAM governor for `device` if none is registered yet.
///
/// **Every model loader must call this before anything allocates**: the KV
/// region span is sized from the governor's balloon-measured capacity, and
/// without one it falls back to the small governor-less test constant
/// (`TEST_SPAN_BYTES`), which cannot hold a many-layer model's
/// per-(layer × size-class) arena floor plus the wave transient tier —
/// measured on Llama-2 7B F32: 146 live regions (~2.3 GiB of arenas) against
/// a ~3 GiB span left the ~1 GiB tier 22 regions short of placeable.
/// Idempotent (keyed on the device ordinal), so co-loaded models share one
/// governor; failures degrade to a warning because a missing governor is
/// survivable on small workloads and the span sizing reports its own refusal
/// on large ones.
#[cfg(feature = "cuda")]
pub(crate) fn ensure_vram_governor(device: &Device) {
    let gpu_id = match device.location() {
        candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
        _ => return,
    };
    if candle::vram::get(gpu_id).is_some() {
        return;
    }
    match candle::vram::VramGovernor::from_device(device, gpu_id) {
        Ok(gov) => {
            let mut balloon = candle::vram::balloon::DeviceBalloonAllocator::new(device.clone());
            match gov.run_balloon(&mut balloon) {
                Ok(c) => tracing::info!(
                    target: "candle_core::vram",
                    "VRAM governor installed: capacity C={:.1}GB",
                    c as f64 / 1e9
                ),
                Err(e) => tracing::warn!("VRAM governor balloon failed: {e}"),
            }
            candle::vram::install(gov);
        }
        Err(e) => tracing::warn!("VRAM governor init failed: {e}"),
    }
}

/// The load-time shapes a wave's transient buffers are sized from.
///
/// Deliberately **width-free**: a wave's row count is an argument to the sizing
/// functions, not a field here. Width is what admission is deciding, so baking
/// an assumed width in would make the plan answer a question it is supposed to
/// be asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveShapes {
    /// Model hidden size.
    pub hidden: usize,
    /// FFN intermediate size. On a MoE model this is the *per-expert*
    /// intermediate, not the dense equivalent.
    pub intermediate: usize,
    /// Experts each token routes to. `1` for a dense model, which collapses the
    /// MoE terms to the dense FFN shapes rather than needing a second branch.
    pub experts_per_tok: usize,
    /// Tokens the LM head scores over. Sizes the forward phase, which holds the
    /// head's logits — one row per scored row, `vocab` wide.
    pub vocab: usize,
    /// Whether this stack applies a per-head RMSNorm to Q and K. Qwen3 and
    /// later do; Llama and Qwen2 have no such weight, and pricing one charged
    /// them for six buffers they never allocate.
    pub head_qk_norm: bool,
    /// Whether Q/K/V carry a bias. Qwen2 does; Llama and Qwen3 do not.
    pub qkv_bias: bool,
    /// Experts the router scores over — the width of the per-token logits the
    /// FFN phase carries. `1` on a dense model, which has no router.
    pub n_experts: usize,
}

/// The dtype activations are carried in, for a KV cache stored as `cache_dtype`.
///
/// Activations follow the cache so the two never need a conversion between them
/// — except for F8E4M3, which is a storage format with no compute kernels, so a
/// cache in it computes in BF16.
///
/// One definition, because two would drift: the forward derives the activation
/// dtype from the session's cache on every wave, and whoever configures the
/// session has to materialise the norm weights in the *same* dtype or the
/// forward refuses.
pub fn activation_dtype(cache_dtype: DType) -> DType {
    if cache_dtype == DType::F8E4M3 {
        DType::BF16
    } else {
        cache_dtype
    }
}

// Core Model Trait (Simple Accessors Only)

/// Minimal trait for models that support batched forward passes.
///
/// This trait only requires simple accessor methods. All complex logic
/// (RoPE caching, forward_batch implementation) lives in [`BatchedInference`].
pub trait BatchedModelCore {
    /// Type of the layer that implements [`BatchedAttentionLayer`].
    type Layer: BatchedAttentionLayer;

    /// Shapes the transient tier is sized from, captured at load.
    ///
    /// Read from the checkpoint's own config rather than derived from weight
    /// dimensions: `hidden` is not `n_head * head_dim` on every model (Qwen3
    /// carries 2048 against 32x128), so deriving it would be right for some
    /// architectures and quietly wrong for others.
    fn wave_shapes(&self) -> WaveShapes;

    /// The geometry [`candle_nn::kv_cache::WavePlan`] prices a wave from.
    ///
    /// Pairs the load-time shapes with the session's activation dtype, which is
    /// an inference-mode choice rather than a property of the weights. Head
    /// geometry comes from layer 0 — every layer of a model shares it, and the
    /// plan needs one layer's working set, not the whole stack's.
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        let shapes = self.wave_shapes();
        ModelGeometry {
            hidden: shapes.hidden,
            intermediate: shapes.intermediate,
            n_head: self.layer(0).n_head(),
            n_kv_head: self.n_kv_head(),
            head_dim: self.head_dim(),
            experts_per_tok: shapes.experts_per_tok,
            n_experts: shapes.n_experts,
            act_dtype,
            // The int8 tensor-core kernels emit **F32** and the cast back to
            // `act_dtype` is a second buffer, so the plan charges for both
            // rather than for the wider of the two.
            //
            // This said BF16 until the census measured it: the expert GEMM
            // outputs came back at four bytes an element, not two, so every
            // accumulate-dtype term in the FFN phase — gate, up and down, the
            // three largest buffers a MoE layer allocates — was priced at half
            // its size.
            accum_dtype: DType::F32,
            vocab: shapes.vocab,
            // Every layer of these stacks attends, so the attention chain sizes
            // the phase on its own — there is no second mixer to compare it to.
            delta_net: None,
            // No always-active shared expert on this path's checkpoints.
            shared_expert: None,
            // An int8 session's RMSNorm fuses its quantize into the epilogue, so
            // the norms emit q8a128 and the projections consume it directly.
            // Read from layer 0 because the mode is fixed at load and every
            // layer shares it.
            packed_norm: self.layer(0).int8mode().is_int8(),
            // **The head's own weight decides the head's encoding.** A head that
            // could not be KO-repacked stays on the dequant path while every
            // layer around it runs int8 — Qwen2's case, measured.
            packed_head: self.output_proj().int8mode().is_int8(),
            gated_qkv: false,
            // `project_qkv` forks on the operand it is handed: an `Int8` one
            // takes `qkv_segmented` — one launch over the three KO weights, then
            // three narrows that copy — and a `Float` one takes three separate
            // calls whose outputs are already contiguous. So the splits exist
            // exactly when the norm is packed.
            fused_qkv: self.layer(0).int8mode().is_int8(),
            qkv_bias: shapes.qkv_bias,
            head_qk_norm: shapes.head_qk_norm,
            // The wave is carried as `[batch, seq, heads · dim]`, so where there
            // IS a per-head norm it copies in and transposes back.
            head_norm_reshapes: true,
            // `want_q8` in the decode path, asked of the same predicate: an int8
            // layer at a head dim the fused q8 combine serves. Qwen2's 64 is not
            // one, and its context was priced on a span it never lands on.
            decode_q8_context: self.layer(0).int8mode().is_int8()
                && paged_decode_q8_head_dim(self.head_dim()),
            partial_rotary: false,
            // One residual stream: hyper-connections are the Flash-Next
            // lineage's, and no model on this path carries them.
            hyper: None,
            // Nor a NextN draft head.
            mtp_head: false,
        }
    }

    /// Re-materialise every norm weight in the dtype activations will arrive in,
    /// if it is not already.
    ///
    /// Called where a session is created — never inside a wave. A quantized
    /// checkpoint dequantizes its norm weights to F32 while inference runs F16 or
    /// BF16, so the norm kernels need them in the activation dtype. Converting on
    /// demand inside the forward costs one device allocation and one launch per
    /// norm, per layer, per token, and is invisible in a profile: it surfaces
    /// only as a slightly slower forward. So the forward path *refuses* a dtype
    /// it was not prepared for (`RmsNorm::weight_for`) and this is the only place
    /// the conversion happens.
    ///
    /// The weight is reloaded from the retained quantized source rather than cast
    /// from a resident copy, so switching dtypes costs a reload and never leaves
    /// a second copy behind.
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()>;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// Number of KV heads per layer.
    fn n_kv_head(&self) -> usize;

    /// Dimension of each attention head.
    fn head_dim(&self) -> usize;

    /// Device the model is on.
    fn device(&self) -> &Device;

    /// Access the resident embedding layer.
    ///
    /// `None` when the table is served from host memory instead — see
    /// [`Self::host_embedding`]. Exactly one of the two is populated.
    fn embeddings(&self) -> Option<&Embedding>;

    /// The token embedding served from the GGUF mmap rather than VRAM.
    ///
    /// `Some` only when the table is large enough relative to the card that
    /// keeping it resident is not worth the VRAM (see
    /// [`crate::models::host_embedding`]); the rows are then gathered per
    /// forward instead. `None` keeps the resident path.
    #[cfg(feature = "cuda")]
    fn host_embedding(&self) -> Option<&crate::models::host_embedding::HostEmbedding> {
        None
    }

    /// Access a layer by index.
    fn layer(&self, idx: usize) -> &Self::Layer;

    /// Access the final RMS normalization layer.
    fn final_norm(&self) -> &RmsNorm;

    /// Access the output projection (LM head).
    fn output_proj(&self) -> &QMatMul;

    /// Whether RoPE uses interleaved format.
    ///
    /// - `false`: Standard format, uses `rope()`
    /// - `true`: Interleaved format, uses `rope_i()`
    fn rope_interleaved(&self) -> bool;

    /// Prune excess memory usage (e.g., compact embeddings).
    fn prune(&self) -> Result<()>;

    /// Snapshot expert pipeline telemetry counters (if this model has an expert cache).
    fn expert_stats(&self) -> Option<PipelineStats> {
        None
    }

    /// Whether the expert cache's pipeline thread or stager has exited — see
    /// `ManagedBatchedModel::expert_pipeline_dead`.
    fn expert_pipeline_dead(&self) -> bool {
        false
    }

    /// The expert pipeline counters' shared home, for a reader on another
    /// thread that reads gauges while forwards run (`ExpertCache::stats_shared`).
    fn expert_stats_shared(&self) -> Option<Arc<Mutex<PipelineStats>>> {
        None
    }

    /// The expert cache's `(hits, misses)` tallies so far, unsettled
    /// (`ExpertCache::hit_counts`) — the per-step read, where
    /// [`Self::expert_stats`] waits for the pipeline thread.
    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        None
    }

    /// The expert cache's residency references over the interval the counters
    /// cover, `(lru, min)` (`ExpertCache::hit_references`), if this model has
    /// one.
    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        None
    }

    /// Fail the segment just swept if the expert cache's workers gave a cold
    /// expert up (`WaveSweep::take_device_fault`, through
    /// `ExpertCache::take_segment_fault`, which waits for the device only when
    /// the cache can fault); `Ok` for a model without one, which has nothing the
    /// device can give up. Required, so a model gaining an expert cache states
    /// it.
    fn take_device_fault(&self) -> Result<()>;

    /// The wave transient tier a prefill of `rows` rows across `sequences`
    /// would need, in bytes, with its attention launch carving the pre-stage,
    /// carry and resume table of chunks of `stage_q_lens` rows at
    /// `stage_offsets` (`candle_nn::kv_cache::PrefillKvStageLayout` — the one
    /// quantity in the tier that follows context depth rather than width, up
    /// to its key-chunk bound).
    ///
    /// `stage_seqs` names each stage's sequence where it has one (empty, or a
    /// sequence that holds no index yet, prices as a fresh one): a sequence's
    /// selection carve is taken over its **live** index, as the step takes it —
    /// on a deep conversation that is most of the final window's forward phase,
    /// which a fresh index prices at a fraction (WhiteWorld 2026-10-19: a
    /// 12-row cohort fitted at 32 MiB, its window placing 80 MiB, 35.6 MB of
    /// it the forward phase).
    ///
    /// `decode_stage` is the decodes co-batched in the same step, as each
    /// one's `(sequence, position, block)`, the block being the rows it runs —
    /// one for a plain decode, its committed token and proposals for one
    /// verifying a draft, every one of which the head scores: each is a row in
    /// the trunk's phases and a row the head scores. A creep step carries the running decodes beside its
    /// window, and the head scores them with the members the final window
    /// finishes; priced as prefill alone, a cohort's tier came out a region
    /// short at exactly that window (3090, 2026-10-18 and 19 — cursor 47 of 48,
    /// five runs). And each carves its selection over its own live index at its
    /// own depth, as a stage does: priced as a bare row count, a cohort beside
    /// seven deep decodes fitted a nine-region tier its window placed at
    /// thirteen (WhiteWorld, 98860c583, 2026-10-19).
    ///
    /// **The same function the tier is actually placed from**, not an estimate
    /// of it: admission judges an offer on the residency it dislodges, and the
    /// tier dislodges weights exactly as a region claim does. Pricing it any
    /// other way lets the two figures drift, and the one that drifts is the one
    /// the placement then refuses.
    ///
    /// `None` for a model that cannot price a wave — the caller then charges no
    /// tier, which is what it did before this existed.
    // One wave's description: its rows and sequences, then each stage's length,
    // depth and sequence, and the decodes beside it.
    #[allow(clippy::too_many_arguments)]
    fn wave_tier_bytes(
        &self,
        _rows: usize,
        _sequences: usize,
        _stage_q_lens: &[usize],
        _stage_offsets: &[usize],
        _stage_seqs: &[usize],
        _decode_stage: &[(usize, usize, usize)],
        _act_dtype: DType,
    ) -> Option<u64> {
        None
    }

    /// The weight side as the wave rate planner prices it — MoE geometry, and
    /// the range the expert zone may move in.
    ///
    /// Derived from [`Self::expert_stats`] rather than plumbed separately, so a
    /// model that reports its cache reports this too and the two can never
    /// describe different moments. `None` for a dense model, for a cache that
    /// has not yet run a classify, and off CUDA — see
    /// [`WeightPlan::from_stats`], which refuses a partial gauge set outright
    /// because every missing field makes a routed expert look free.
    fn weight_plan(&self) -> WeightPlanning {
        // No expert cache at all is `Dense`, not a defect — the `?` would have made it
        // indistinguishable from a cache whose gauges are zero.
        match self.expert_stats() {
            Some(s) => WeightPlan::from_stats(&s),
            None => WeightPlanning::Dense,
        }
    }

    /// Snapshot the layer-streaming counters, if this model's weights are slot
    /// tenants.
    ///
    /// The dense counterpart of [`Self::expert_stats`] and needed for the same
    /// reason: a run whose layers never streamed says nothing about the path
    /// that streams them, and the two regimes are indistinguishable from
    /// throughput alone. `(hits, warm, cold, joins, evictions, homed, streaming)`.
    ///
    /// The last two are **current state, not totals**, and they are what makes
    /// the rest readable: the weight zone grows into spare KV ground over the
    /// first forwards, so the residency a load reports at its carve is not the
    /// residency the run was measured at. Without them a low transfer count has
    /// two explanations — the zone went whole, or nothing was ever asked of it —
    /// and only arithmetic on the counters tells them apart.
    fn layer_stream_stats(&self) -> Option<[usize; 7]> {
        None
    }

    /// Ask the weight side to hand ground to the KV side now, answering with the
    /// bytes conceded.
    ///
    /// For a caller whose KV allocation just failed, or which can see it is
    /// about to. The other direction runs only between forwards
    /// ([`Self::reclaim_spare_ground`]), which a wave that cannot allocate never
    /// reaches. `regions` is the quantity the caller measured — never a counter
    /// it drained, which is what this replaced and what took the expert zone
    /// below its working minimum. Zero is an ordinary answer: the zone is at its
    /// floor, or a wave is still in flight.
    ///
    /// Dense models have no movable boundary and return zero.
    fn request_kv_ground(&self, regions: usize) -> u64 {
        let _ = regions;
        0
    }

    /// The opposite direction: let the weight side take back KV regions that are
    /// standing free.
    ///
    /// **Only legal between forwards.** A boundary move evicts and relocates
    /// expert slots, so it may not run under a live wave generation — see
    /// `ExpertCache::reclaim_spare_ground` for what happened when it was driven
    /// from the expert pipeline's end-of-pass instead.
    ///
    /// Dense models have no movable boundary and do nothing.
    fn reclaim_spare_ground(&self) {}

    /// [`Self::request_kv_ground`] for a guest drain, with the engine stopped
    /// between waves: the weight side may go below the floor it needs to serve a
    /// token, down to what has no copy anywhere else, because nothing forwards
    /// until [`Self::restore_serving_floor`] has answered `true`.
    ///
    /// Dense models have no movable boundary and return zero.
    fn concede_for_guest(&self, regions: usize) -> u64 {
        let _ = regions;
        0
    }

    /// After [`Self::concede_for_guest`], for a drain still short: copy what has
    /// no copy anywhere else — the pinned experts — to host memory and concede
    /// its ground as well. Restored by [`Self::restore_serving_floor`], which
    /// answers `false` until it is back. Answers the bytes conceded.
    ///
    /// Dense models have nothing pinned to move and return zero.
    fn stash_pinned_for_guest(&self) -> u64 {
        0
    }

    /// Grow the weight side back to its serving floor after
    /// [`Self::concede_for_guest`], answering whether it stands there. Only legal
    /// between forwards, like [`Self::reclaim_spare_ground`].
    ///
    /// Dense models never leave their floor and answer `true`.
    fn restore_serving_floor(&self) -> bool {
        true
    }

    /// Live VRAM bytes held by the model's weights — fixed base weights plus the
    /// **time-varying** resident-expert footprint (MoE experts page VRAM↔RAM
    /// under pressure). `None` if the model can't report it (dense models, no
    /// expert cache, non-CUDA). Feeds the whole-card VRAM decomposition.
    fn resident_weight_bytes(&self) -> Option<usize> {
        None
    }

    /// Reservation bytes held by per-sequence recurrent state, for the same
    /// decomposition.
    ///
    /// Zero rather than `None`: a stack with no recurrent layers holds none, it
    /// does not fail to know. On a hybrid stack this is the largest in-span
    /// tenant that is neither KV nor weights (~126 MiB per sequence), and it
    /// moves with wave width — so a total that omits it makes the span read
    /// emptier than it is.
    fn recurrent_reserved_bytes(&self) -> usize {
        0
    }

    /// What **one** sequence's recurrent state costs, priced from the model's
    /// geometry rather than from what is resident.
    ///
    /// This is the admission figure. [`Self::recurrent_reserved_bytes`] is a
    /// total for the whole-card decomposition and cannot stand in for it: it
    /// answers zero before the first store exists — pricing a claim that is
    /// about to arrive as free — and dividing it by the sequences in flight
    /// gives a number that grows with the idle ones, because the sum and the
    /// divisor range over different populations.
    ///
    /// Zero on a stack with no recurrent layers, which is the true cost there.
    fn recurrent_store_bytes(&self) -> usize {
        0
    }

    /// Compact the arenas per-sequence recurrent state lives in, moving at most
    /// `max_moves` layer states (zero for no bound). Between forwards only.
    ///
    /// Nothing to do on a stack with no recurrent layers, which is the default.
    fn compact_recurrent(&self, _max_moves: usize) -> Result<RecurrentCompaction> {
        Ok(RecurrentCompaction::default())
    }

    /// Reset expert pipeline telemetry counters to zero.
    fn reset_expert_stats(&self) {}

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        ProfileSnapshot::default()
    }

    /// Per-model multiplier for the K high adaptive threshold.
    fn k_hi_error_threshold_factor(&self) -> f32 {
        1.0
    }

    /// Per-model multiplier for the K low adaptive threshold.
    fn k_low_error_threshold_factor(&self) -> f32 {
        1.0
    }

    /// Per-model multiplier for the V high (strict) adaptive threshold.
    fn v_hi_error_threshold_factor(&self) -> f32 {
        1.0
    }

    /// Per-model multiplier for the V low (lenient) adaptive threshold.
    fn v_low_error_threshold_factor(&self) -> f32 {
        1.0
    }
}
