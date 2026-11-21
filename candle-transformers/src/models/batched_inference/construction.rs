//! Creates a batched session and holds the invariants its later waves rely on:
//! shared backings, the RoPE rung schedule, glue staging and per-layer seal caps.
//!
//! Every layer shares one backing, since they have the same geometry, so they
//! share arenas, the GID pool and the arena table. A sibling session built over
//! existing backings borrows chunks through that same pool. `expect_rope_select`
//! refuses a forward whose rung choice differs from the session's, because a
//! disagreement would name rungs the schedule does not hold. A layer whose error
//! budget differs from the stack's can be capped to a shallower seal level. The
//! cap only lowers, and it applies on the seal path where compression happens.

use super::*;

impl BatchedInferenceSession {
    /// Create a new batched inference session.
    pub fn new(
        layers: KvLayers,
        n_kv_head: usize,
        head_dim: usize,
        device: &Device,
        config: BatchedConfig,
    ) -> Result<Self> {
        let num_layers = layers.total();
        // Create one backing and share it across all layers.
        // All layers have the same (n_kv_head, head_dim, format) so they
        // share arenas, the GID pool, and the arena table. Each layer gets
        // its own per-layer state (sequences, max_blocks).
        let first_backing = ChunkedKvBacking::new_with_format_adaptive(
            1, // Initial batch size of 1, will grow dynamically
            n_kv_head,
            head_dim,
            config.k_format,
            config.v_format,
            device,
            config.initial_seq_len,
            config.compression_policy(),
        )?;

        let mut backings = Vec::with_capacity(num_layers);
        backings.push(first_backing.clone());
        for layer_idx in 1..num_layers {
            backings.push(first_backing.new_layer(layer_idx, 1, config.initial_seq_len));
        }

        let mut session = Self {
            backings,
            sequences: Vec::new(),
            reserved: HashSet::new(),
            config,
            num_layers,
            stream_layers: layers.stream,
            layer_seal_cap: vec![None; num_layers],
            device: device.clone(),
            pending_glue: None,
            rope: RungSelect::unbounded(),
            accept_in_place: false,
            stencil_rows: 0,
            walk_tokens: None,
            #[cfg(feature = "cuda")]
            gap: None,
        };
        // Before any forward: the gap's scratch is allocated with nothing in
        // flight to wait behind.
        #[cfg(feature = "cuda")]
        session.ensure_gap()?;
        Ok(session)
    }

    /// Set how the headers pick each sequence's rung — the model's own, from
    /// its `RopeRungs::select`.
    pub fn set_rope_select(&mut self, select: RungSelect) {
        self.rope = select;
    }

    /// The RoPE rung a sequence reaching `reach` positions rotates by; a reach
    /// past the schedule's supported maximum is refused.
    pub fn rope_rung_for(&self, reach: usize) -> Result<u32> {
        self.rope.rung_for(reach)
    }

    /// The deepest reach, in positions, any sequence may have: the last
    /// ceiling. A sequence that would write past it is refused.
    pub fn rope_reach(&self) -> usize {
        self.rope.reach()
    }

    /// Refuse a forward whose model picks rungs other than the way this
    /// session's headers do.
    ///
    /// Decode headers take a sequence's rung from the session while prefill,
    /// glue and the QSA indexer take it from the model's rung set, so the two
    /// must be one schedule with one floor: a session opened before the
    /// model's schedule was replaced, or one never given the model's, would
    /// name rungs the set does not hold — a kernel trap — or disagree with
    /// prefill about a sequence's rung.
    pub fn expect_rope_select(&self, model: &RungSelect) -> Result<()> {
        if &self.rope != model {
            candle::bail!(
                "rope: this session picks rungs by {:?} but the model by {model:?} — the \
                 session was opened under another schedule",
                self.rope
            );
        }
        Ok(())
    }

    /// Hold one layer's seals at or below `max_level`, whatever the rest of the
    /// ladder runs at.
    ///
    /// **For a layer whose error budget is not the stack's.** A trunk spreads
    /// every read over many KV layers, so a level's quantisation error is
    /// partly averaged over the depth; a one-block draft head has a single
    /// layer and absorbs none of it.
    ///
    /// A **ceiling**, not a setting: the cap only ever lowers, so a session
    /// already running below it is untouched and the caller does not have to
    /// know what the session was configured with. Capping to a level *deeper*
    /// than the session's would be compressing one layer harder than the stack,
    /// which is never what this is for.
    ///
    /// A *level*, not a format, so the layer takes the same code path and the
    /// same candidate machinery as every other — only less aggressively. The
    /// alternative of excluding the layer from the seal entirely walks into a
    /// known corruption: the round-trip truncates every layer to zero blocks
    /// and re-injects, and "re-injecting Arc-shared chunks under a lone
    /// sequence corrupted the decode read".
    ///
    /// **Note this is the seal path, which is where compression actually
    /// happens.** A per-`KvCache` policy does not reach it — the seal reads the
    /// session's — so setting one of those is a silent no-op that looks exactly
    /// like a null result, and cost this codebase a wrongly-refuted hypothesis.
    pub fn cap_layer_seal_level(&mut self, layer: usize, max_level: u8) -> Result<()> {
        let slot = self.layer_seal_cap.get_mut(layer).ok_or_else(|| {
            candle::Error::Msg(format!(
                "cap_layer_seal_level: layer {layer} is past the session's {} KV layers",
                self.num_layers
            ))
        })?;
        *slot = Some(max_level);
        Ok(())
    }

    /// Create a session that shares the KV arena pool with an existing session.
    ///
    /// Each `ChunkedKvBacking` is an `Arc<BackingInner>`, so cloning shares the
    /// underlying storage, slot table, and arena table between this new session
    /// and `source`.  This is the correct way to create a sibling session (e.g. a
    /// float prototype session for boundary injection) that needs to borrow chunks
    /// into the main session via `append_borrowed_chunks_cow` — because that
    /// operation scans the same slot table to find chunk handles.
    ///
    /// Slot collisions are avoided by consulting the shared backing when picking
    /// a new slot index in `create_sequence` (see [`BatchedInferenceSession::create_sequence`]).
    pub fn new_with_backings(
        backings: Vec<ChunkedKvBacking>,
        config: BatchedConfig,
        device: &Device,
    ) -> Self {
        let num_layers = backings.len();
        Self {
            backings,
            sequences: Vec::new(),
            reserved: HashSet::new(),
            config,
            num_layers,
            // Every backing handed in is a stream layer: a sibling session
            // exists to borrow chunks through the shared arena pool, and no
            // draft head runs against it.
            stream_layers: num_layers,
            layer_seal_cap: vec![None; num_layers],
            device: device.clone(),
            pending_glue: None,
            rope: RungSelect::unbounded(),
            accept_in_place: false,
            stencil_rows: 0,
            walk_tokens: None,
            #[cfg(feature = "cuda")]
            gap: None,
        }
    }

    /// Stage per-slot reprojection-glue descriptors for the next
    /// `forward_batched`. `pending[i]` aligns with the `seq_indices` of the
    /// imminent gap-fill forward. Consumed (and cleared) by that single forward.
    pub fn set_pending_glue(&mut self, pending: Vec<PendingGlue>) {
        self.pending_glue = Some(pending);
    }

    /// Take + clear the staged glue descriptors (one forward's worth).
    pub fn take_pending_glue(&mut self) -> Option<Vec<PendingGlue>> {
        self.pending_glue.take()
    }

    /// Mark whether the waves issued from here hand their logits to an
    /// in-place reader — see the field.
    pub fn set_accept_in_place(&mut self, on: bool) {
        self.accept_in_place = on;
    }

    /// Whether the wave being issued hands its logits to an in-place reader,
    /// which decides whether the forward reserves the accept walk's rows.
    pub fn accept_in_place(&self) -> bool {
        self.accept_in_place
    }

    /// Say how many rows of the waves issued from here the sampler holds to a
    /// grammar's bitmap — see the field.
    pub fn set_stencil_rows(&mut self, rows: usize) {
        self.stencil_rows = rows;
    }

    /// Rows of the wave being issued held to a grammar's bitmap.
    pub fn stencil_rows(&self) -> usize {
        self.stencil_rows
    }

    /// A contiguous `[rows, n]` `u32` buffer for one draft walk: its committed
    /// tokens and every step's pick, one row each.
    ///
    /// Held by the session and grown only when a walk asks for more than any
    /// walk before it, so a steady speculative decode reuses it and allocates
    /// nothing per step. The walk reads its picks back once, at the end; the
    /// next walk on this session's stream is ordered after that readback, so
    /// it never overwrites a pick still in flight.
    pub fn walk_tokens(&mut self, rows: usize, n: usize) -> Result<Tensor> {
        let want = rows * n;
        let held = self.walk_tokens.as_ref().map_or(0, Tensor::elem_count);
        if held < want {
            // By doubling, so a cohort ramping up settles in a few steps; on
            // held ground, since it lives as long as the session.
            let grown = (held * 2).max(want);
            self.walk_tokens = Some(held_empty(grown, DType::U32, &self.device)?);
        }
        let buf = self.walk_tokens.as_ref().expect("sized above");
        buf.narrow(0, 0, want)?.reshape((rows, n))
    }

    /// Get the configuration used for this session.
    pub fn config(&self) -> &BatchedConfig {
        &self.config
    }

    /// Build the session's compression policy from its config.
    /// Returns `None` for uniform / non-compression modes.
    pub fn compression_policy(&self) -> Option<CompressionPolicy> {
        self.config.compression_policy()
    }

    /// The policy the session's seals run under, or `None` when it does not
    /// quantize (F16/BF16): its compression level's adaptive policy, or for a
    /// fixed-format mode (Q8_0, Q4_0, …) a uniform override of that format on
    /// both sides. The override's level only feeds candidate scaffolding the
    /// override bypasses.
    pub fn seal_policy(&self) -> Option<CompressionPolicy> {
        if let Some(p) = self.compression_policy() {
            return Some(p);
        }
        let k_fmt = self.config.k_format.as_quant()?;
        let v_fmt = self.config.v_format.as_quant().unwrap_or(k_fmt);
        Some(
            CompressionPolicy::new_with_error_threshold_factors(5, 1.0, 1.0, 1.0, 1.0)
                .with_override_k_quant(Some(k_fmt))
                .with_override_v_quant(Some(v_fmt)),
        )
    }

    /// Each layer's seal policy under `policy`: `policy`, or for a layer capped
    /// below its level ([`Self::cap_layer_seal_level`]) its cap's.
    pub fn layer_seal_policies(&self, policy: &CompressionPolicy) -> Vec<CompressionPolicy> {
        (0..self.backings.len())
            .map(|layer_idx| {
                self.layer_seal_cap[layer_idx]
                    .filter(|&cap| cap < policy.compression_level)
                    .and_then(|cap| self.config.compression_policy_at(cap))
                    .unwrap_or(*policy)
            })
            .collect()
    }

    /// Get the device the session is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the number of currently active sequences.
    pub fn active_sequence_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.as_ref().is_some_and(|s| s.active))
            .count()
    }

    /// Get the list of active sequence indices.
    pub fn active_sequences(&self) -> Vec<usize> {
        self.sequences
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| {
                if s.as_ref().is_some_and(|s| s.active) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
}
