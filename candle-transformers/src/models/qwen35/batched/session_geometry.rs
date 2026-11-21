//! Answers the engine's geometry and session questions for a hybrid model, where
//! only some layers attend.
//!
//! Only attention layers hold KV, so the KV layer count, the width cap and the
//! threshold row all follow the attention layers rather than the transformer
//! depth. Pricing by depth would refuse far more prefill than the cache can hold.
//! Norm weights and LoRA adapters are converted once, at session creation and
//! never inside a wave, because a matmul refuses mismatched dtypes. A session
//! with a different activation dtype is a new session, since draft state
//! captured under the old dtype belongs to nothing that exists now.

use super::*;

impl HybridBatched {
    // ── The engine's questions, answered for a hybrid ──────────────────────

    /// Transformer depth — what bounds a wave's layer range.
    pub fn num_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    /// KV heads per *attention* layer.
    pub fn n_kv_head(&self) -> usize {
        self.model.cfg.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.model.cfg.attn_head_dim
    }

    pub fn device(&self) -> &Device {
        &self.model.device
    }

    pub fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        // The session's int8 mode, not the config's: the KO twins are chosen at
        // load and decide what each norm's fused epilogue emits, which is a real
        // difference in what the span holds.
        wave_geometry(
            &self.model.cfg,
            act_dtype,
            self.int8mode(),
            self.model.mtp.is_some(),
        )
    }

    /// Re-materialise every norm weight in the session's activation dtype.
    ///
    /// The forward *refuses* a dtype it was not prepared for, so this is the
    /// single place the conversion happens — at session creation, never
    /// inside a wave.
    pub fn maybe_change_dtype(&self, dtype: DType, kv_dtype: DType) -> Result<()> {
        // A dtype change means a NEW session — this is the one place that
        // happens, and only a new session can change it. Every piece of draft
        // state is an activation captured under the old one, belonging to a
        // sequence numbering the new session will reuse for something else, so
        // all of it dies here. Nothing is dropped when the dtype is unchanged,
        // which is what a sibling session sharing the backings gets.
        //
        // Not hypothetical: a gate that runs F16 and then BF16 against one
        // loaded model would otherwise hand the draft head an F16 hidden to
        // concatenate with a BF16 embedding.
        let stale = self
            .verify_hidden
            .lock()
            .map(|m| m.values().any(|t| t.dtype() != dtype))
            .unwrap_or(false)
            || self
                .seed
                .lock()
                .map(|m| m.values().any(|t| t.dtype() != dtype))
                .unwrap_or(false);
        if stale {
            if let Ok(mut m) = self.verify_hidden.lock() {
                m.clear();
            }
            if let Ok(mut m) = self.seed.lock() {
                m.clear();
            }
            if let Ok(mut m) = self.capture_rows.lock() {
                m.clear();
            }
        }
        // Through the residue, which is where every norm of every layer lives —
        // resident in both stores, so this reaches a streamed checkpoint's
        // layers without pulling one of them over PCIe. It carries the two
        // widths main introduced here: the residual-stream norms take the
        // activation dtype, Q/K's take the KV arena's.
        for li in 0..self.model.layers.len() {
            self.model
                .layers
                .residue(li)?
                .set_activation_dtype(dtype, kv_dtype)?;
        }
        // The draft head's block runs in the wave's dtype like any other, so
        // its norms are materialised with the trunk's — the head is a layer of
        // this model, not a sidecar that got to keep the loader's dtype.
        if let Some(head) = &self.model.mtp {
            head.block.residue().set_activation_dtype(dtype, kv_dtype)?;
            head.input.enorm.maybe_change_dtype(dtype)?;
            head.input.hnorm.maybe_change_dtype(dtype)?;
            head.head_norm.maybe_change_dtype(dtype)?;
        }
        // LoRA adapters, for exactly the reason the norms are here: their
        // weights are operands of matmuls whose other operand is an activation,
        // and a matmul refuses mismatched dtypes outright.
        //
        // **Three widths, not one**, because a layer does not use a single one:
        //
        // * `dtype` — the residual stream, which is what `gate`/`up` read and
        //   what `down` and the output projection write back into it.
        // * `kv_dtype` — what Q/K/V are projected in, because they become the
        //   arena's contents. Equal to `dtype` unless the model computes wider
        //   than it stores.
        // * the **SwiGLU promotion** — an F16 session runs its MLP intermediates
        //   in BF16, since they can exceed F16's range. That makes `gate`'s
        //   adapter write BF16 from an F16 input, and `down`'s read BF16 and
        //   write F16, inside the same layer.
        //
        // Every one of them is materialised here, at session creation, from the
        // host's F32 master — so nothing converts a weight inside a wave, and a
        // width the forward asks for is always already there. Missing one is not
        // a silent wrong answer: the matmul refuses it by name.
        //
        // Idempotent. A session whose widths match the last moves no bytes; in
        // production, where the stream and the arena agree and nothing is
        // promoted, this is a single resident copy.
        let mut widths = vec![dtype, kv_dtype];
        if dtype == DType::F16 {
            widths.push(DType::BF16);
        }
        widths.dedup();
        for adapter in self.adapters.values() {
            adapter.maybe_change_dtype(&widths, &self.model.device)?;
        }
        self.model.final_norm.maybe_change_dtype(dtype)
    }

    /// A session whose KV is allocated per attention layer, with the norms
    /// materialised for its activation dtype.
    ///
    /// The model's KV threshold factor row is folded into the session config
    /// here, exactly as the `ManagedBatchedModel` default does — this
    /// override replaces that default (the KV layer count differs from the
    /// transformer depth on a hybrid), so it must also replace the factor
    /// fold, or the per-model calibration silently never reaches the
    /// compression policy.
    pub fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let mut config = config;
        config.k_hi_error_threshold_factor *= self.kv_factors.k_hi;
        config.k_low_error_threshold_factor *= self.kv_factors.k_low;
        config.v_hi_error_threshold_factor *= self.kv_factors.v_hi;
        config.v_low_error_threshold_factor *= self.kv_factors.v_low;
        // **This duplicates the generic `create_batched_session`.** It reads
        // `kv_factors` directly rather than `model_core_properties()`, so any
        // per-model property added to that struct lands there and is silently
        // dropped here — a new field looks wired, builds clean, and simply never
        // reaches this model. Extend both when adding one.
        let mut session = create_session(&self.model.cfg, &self.model.device, config)?;
        session.set_rope_select(self.rope.select().clone());
        self.maybe_change_dtype(session.activation_dtype(), session.kv_live_dtype())?;
        Ok(session)
    }

    /// Static properties, with the provenance depths snapped onto layers that
    /// actually attend.
    ///
    /// Infallible: [`Self::new`] refuses a stack with no attention layers, so
    /// the snap always has somewhere to land.
    pub fn model_core_properties(&self) -> ModelCoreProperties {
        let provenance_layer_indices = self.provenance;
        ModelCoreProperties {
            num_layers: self.model.cfg.num_layers,
            n_kv_heads: self.model.cfg.num_kv_heads,
            head_dim: self.model.cfg.attn_head_dim,
            provenance_layer_indices,
            // The concrete model's named factor row, supplied at construction —
            // one source of truth shared with the offline report, never inline
            // literals.
            k_hi_error_threshold_factor: self.kv_factors.k_hi,
            k_low_error_threshold_factor: self.kv_factors.k_low,
            v_hi_error_threshold_factor: self.kv_factors.v_hi,
            v_low_error_threshold_factor: self.kv_factors.v_low,
            // **No.** Three quarters of this stack mixes tokens through a
            // recurrence, and a recurrence cannot compute the output of a token
            // inserted mid-sequence: its state has already accumulated past the
            // hole and there is no way to re-enter it there. The planner has to
            // know before it plans, not discover it from a bail.
            // **Attention layers only.** Three quarters of this stack is
            // recurrent and has no Q in a KV cache to capture, so the fold's
            // `[n − 2, 1, 1]` groups over 10 layers here, not 40.
            provenance_capture_layers: self.kv_map.num_kv_layers(),
            can_gap_fill: false,
            carries_recurrent_state: true,
        }
    }

    /// Rows the KV side can admit, priced against the layers a row actually
    /// occupies.
    ///
    /// The trait's default multiplies the per-row cost by transformer depth,
    /// which on a 3:1 hybrid over-charges by 4× and refuses four times more
    /// prefill than the cache can hold. The count that is right is the
    /// session's, not the layer map's: a checkpoint with an MTP head pages one
    /// layer MORE than the map describes ([`session_kv_layers`]), because the
    /// head's own KV is a layer past every trunk one. Pricing on the map's
    /// eight while a row costs nine admitted 12.5% more prefill than the cache
    /// could hold, and the shortfall did not surface as a refusal — it surfaced
    /// as a chunk that could not be claimed from inside the forward that needed
    /// it, three layers into the sweep, on the widest configuration only.
    pub fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        let stats = candle_nn::kv_cache::region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_sub(1);
        let per_row = 2 * self.n_kv_head() * self.head_dim() * act_dtype.size_in_bytes();
        let per_row_all_kv_layers =
            per_row.checked_mul(session_kv_layers(&self.model.cfg).ok()?)?;
        if per_row_all_kv_layers == 0 {
            return None;
        }
        let kv_bytes = free.saturating_mul(candle_nn::kv_cache::REGION_BYTES);
        // Never zero: a cap of nought is not a narrow wave but no wave, and
        // once the KV side is full it would be permanent.
        Some((kv_bytes / per_row_all_kv_layers).max(1))
    }
}
