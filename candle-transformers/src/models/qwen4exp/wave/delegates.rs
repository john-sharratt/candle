//! The scheduler's view of Qwen4ExpBatched: how a wave is priced and placed, what its carried state costs, and how speculative blocks roll back.
//!
//! The generic pricing does not account for this stack's hybrid of gated DeltaNet and full attention, its shared experts or its draft head, so each method here prices or rewinds one part the generic one misses. A wave's tier is priced by the same `select_layer_bytes` the wave is placed by, with decodes counted as they lay out, so admission and placement agree on one figure. The sparse selection's carve follows context depth past the pre-stage's key-chunk bound, and the widest prefill is bounded by the ground this card can place together with the KV its rows write. Rewinds restore all three recurrences exactly for the targets they accept and refuse any they cannot roll back.

use super::*;

impl ManagedBatchedModel for Qwen4ExpBatched {
    fn wave_geometry(&self, wave_dtype: DType) -> ModelGeometry {
        // The Gated Residual carries F32 throughout, so the wave's own dtype is
        // not what the spans are priced in — except the attention mixer's own
        // carves, which run in it (`HyperWidths::mixer_dtype`).
        let act_dtype = DType::F32;
        let cfg = &self.model.cfg;
        let int8 = self.model.lm_head.int8mode().is_int8();
        ModelGeometry {
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            intermediate: cfg.moe.expert_ffn_size,
            n_head: cfg.num_attention_heads,
            n_kv_head: cfg.num_kv_heads,
            head_dim: cfg.attn_head_dim,
            experts_per_tok: cfg.moe.n_experts_used.max(1),
            n_experts: cfg.moe.n_experts.max(1),
            // The 3:1 hybrid's DeltaNet mixer carves from the attention arena,
            // so its widths are priced beside the attention chain.
            delta_net: cfg
                .layer_kinds
                .iter()
                .any(|k| matches!(k, LayerKind::DeltaNet))
                .then_some(DeltaNetWidths {
                    conv_dim: cfg.delta_net.conv_dim(),
                    value_dim: cfg.delta_net.value_dim(),
                    n_v_heads: cfg.delta_net.n_v_heads,
                    layers: cfg
                        .layer_kinds
                        .iter()
                        .filter(|k| matches!(k, LayerKind::DeltaNet))
                        .count(),
                }),
            // Every MoE layer adds an always-active shared expert to the routed
            // block, its gate projection stored padded to one KO tile.
            shared_expert: Some(SharedExpertWidths {
                intermediate: cfg.moe.shared_expert_ffn_size,
                gate_cols: SHARED_GATE_TILE,
            }),
            act_dtype,
            accum_dtype: if int8 {
                DType::F32
            } else {
                ffn_work_dtype(act_dtype)
            },
            packed_norm: int8,
            packed_head: int8,
            // The Qwen3.5 lineage's attention: an interleaved `[q | gate]` Q
            // weight that does not pack with K and V, per-head Q/K norms, and a
            // wave flattened to `[rows, hidden]` before the sweep.
            gated_qkv: true,
            fused_qkv: false,
            // No Q/K/V biases in this lineage.
            qkv_bias: false,
            // The fused q8 decode combine serves this head dim on an int8
            // session, through the same predicate the dispatch asks.
            decode_q8_context: int8 && paged_decode_q8_head_dim(cfg.attn_head_dim),
            head_qk_norm: true,
            head_norm_reshapes: false,
            partial_rotary: cfg.rope_dim < cfg.attn_head_dim,
            // The hyper-connection streams — what made the Gated Residual's
            // transients unpriceable before this geometry carried them.
            hyper: Some(HyperWidths {
                streams: cfg.hc.count,
                low_rank: cfg.hc.low_rank,
                draft_head: self.model.mtp.is_some(),
                ple: true,
                mixer_dtype: wave_dtype,
            }),
            // The multi-stream head is the one `HyperWidths` states.
            mtp_head: false,
        }
    }

    /// The generic pricing plus **the sparse selection's carve**, the one term
    /// of this model's tier that follows context depth past the pre-stage's
    /// key-chunk bound: the score buffer and the scoring products are as wide
    /// as the candidate blocks the deepest row can see. Left to the generic
    /// pricing it is zero, and admission priced a 128K turn's deepest chunk at
    /// the tier of its first.
    ///
    /// Priced through the same `select_layer_bytes` the wave places its tier
    /// by, over every sequence the step carries: each running decode at its own
    /// depth over its live index (`decode_stage`, the `(sequence, position)` of
    /// each), then each stage over its own when it is named and over a fresh
    /// contiguous index — the shape a sequence prefilling its whole prompt has —
    /// when it is not. With no chunk list the rows are priced as one chunk from
    /// position zero.
    // One wave's description: its rows and sequences, then each stage's length,
    // depth and sequence, and the decodes beside it.
    #[allow(clippy::too_many_arguments)]
    fn wave_tier_bytes(
        &self,
        rows: usize,
        sequences: usize,
        stage_q_lens: &[usize],
        stage_offsets: &[usize],
        stage_seqs: &[usize],
        decode_stage: &[(usize, usize, usize)],
        act_dtype: DType,
    ) -> Option<u64> {
        let geometry = self.wave_geometry(act_dtype);
        let (mut q_lens, mut offsets) = if stage_q_lens.is_empty() {
            (vec![rows], vec![0])
        } else {
            (stage_q_lens.to_vec(), stage_offsets.to_vec())
        };
        let named = !stage_seqs.is_empty() && stage_seqs.len() == q_lens.len();
        let mut seqs: Vec<usize> = (0..q_lens.len())
            .map(|i| {
                if named {
                    stage_seqs[i]
                } else {
                    // Ids no live sequence holds: slot ids are small indices.
                    usize::MAX - i
                }
            })
            .collect();
        // **The decodes as the step lays them out.** A plain one is a decode row;
        // one verifying a draft is a span of its block — the committed token and
        // its proposals — staged and carved like a prefill's, with every row a
        // prediction the head scores. Priced as one row each, sixteen decodes
        // verifying drafts placed a 105 MB forward phase, their head's logits,
        // beside a cohort fitted at 96 MiB whole, and the cohort was lost two
        // regions short (WhiteWorld, c8221fe39, 2026-10-19).
        let plain: Vec<(usize, usize)> = decode_stage
            .iter()
            .filter(|&&(_, _, block)| block <= 1)
            .map(|&(seq, at, _)| (seq, at))
            .collect();
        let mut verify_rows = 0usize;
        let mut verify_spans = 0usize;
        for &(seq, at, block) in decode_stage.iter().filter(|&&(_, _, b)| b > 1) {
            q_lens.push(block);
            offsets.push(at);
            seqs.push(seq);
            verify_rows += block;
            verify_spans += 1;
        }
        let decodes = plain.len();
        // Over every sequence the step carries, as it carves them: the plain
        // decodes first, a row each at its own depth and over its own live
        // index, then the spans. A span not named is priced fresh inside the
        // same walk. The decodes are not optional: a decode deep in its context
        // carves its selection over every candidate block it holds, and priced
        // as a bare row count a nine-member cohort beside seven decodes fitted a
        // nine-region tier its window placed at thirteen (WhiteWorld,
        // 98860c583, 2026-10-19).
        #[cfg(feature = "cuda")]
        let carve = {
            let all_seqs: Vec<usize> = plain
                .iter()
                .map(|&(seq, _)| seq)
                .chain(seqs.iter().copied())
                .collect();
            let lens: Vec<usize> = std::iter::repeat_n(1, decodes)
                .chain(q_lens.iter().copied())
                .collect();
            let depths: Vec<usize> = plain
                .iter()
                .map(|&(_, at)| at)
                .chain(offsets.iter().copied())
                .collect();
            self.wave_qsa_bytes(&all_seqs, &lens, &depths, 0, self.model.cfg.num_layers)
        };
        #[cfg(not(feature = "cuda"))]
        let carve = {
            let _ = &seqs;
            self.fresh_qsa_bytes(&q_lens, &offsets)
        };
        let qsa_bytes = match carve {
            Ok(bytes) => bytes,
            Err(e) => {
                // The wave this would price refuses by the same error at
                // placement; admission says why it charged no tier.
                tracing::warn!(error = %e, rows, "qwen4exp: selection carve unpriceable");
                return None;
            }
        };
        let span_rows = rows + verify_rows;
        let spans = sequences.max(1) + verify_spans;
        // The head scores each prefill span's last row, every verify row and
        // every plain decode.
        let scored = sequences.max(1) + verify_rows + decodes;
        let width = WaveWidth {
            kv_stage: PrefillKvStageLayout::new(
                &q_lens,
                &offsets,
                geometry.n_head,
                geometry.n_kv_head,
                geometry.head_dim,
                PrefillLaunchBounds::PRODUCTION,
            ),
            qsa_bytes,
            // As the step sets them: the look-ahead over every row, decodes
            // included, and the accept walk's selection on the head's span —
            // priced as if it reads the logits in place, the wider case, and
            // every scored row as if a grammar's bitmap held it.
            look_ahead_hops: look_ahead_hops(span_rows + decodes),
            decode_rows: decodes,
            scored_rows: scored,
            accept_rows: scored,
            stencil_rows: scored,
            ..WaveWidth::prefill(span_rows, spans)
        };
        let scratch = launch_scratch_reserve(&self.model.device);
        Some(WavePlan::new(geometry).tier_bytes(width, scratch) as u64)
    }

    /// The widest prefill **this card** can place: the rows whose whole tier —
    /// every phase, priced in [`Self::wave_geometry`] — fits the ground a
    /// forward can reach ([`Self::placeable_tier_bytes`]) **together with the
    /// KV those rows write**, bounded by where compute saturates and by what the
    /// KV side can hold.
    ///
    /// Together, because admit claims the wave's KV out of that same ground
    /// before the tier is placed in what is left. Priced apart, a wave fitted
    /// each half alone and its tier came up short by its own KV — on a 16 GB
    /// card at BF16 ×8, by exactly the eight regions its prompts wrote.
    ///
    /// Not the generic single-phase bound (rows that fit a fixed 512 MiB FFN
    /// span), which cuts a 72 GB card's prefill to a fraction of the width its
    /// tier can carry; and not unbounded either, which hands a 16 GB card slabs
    /// whose tier it cannot place. On a card with tens of GB the weight side
    /// could concede, this opens to the compute ceiling and the admission rate
    /// model picks the width within it; on a small card it sits near the tier
    /// the partition always has room for.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        self.prefill_width_cap_withholding(act_dtype, 0)
    }

    /// The width cap against the ground less `withheld`: the tier and the KV
    /// side alike are paid from it, so both terms shrink by what others' KV will
    /// have claimed by the sweep's last window. Never under the tier the
    /// partition always leaves room for ([`WAVE_SPAN_BYTES`]), nor zero rows.
    fn prefill_width_cap_withholding(&self, act_dtype: DType, withheld: usize) -> usize {
        let mut cap = MAX_PREFILL_TOKENS;
        let kv_row = self.kv_bytes_per_row(act_dtype).unwrap_or(0);
        if let Some(ground) = self.placeable_tier_bytes() {
            let ground = ground.saturating_sub(withheld).max(WAVE_SPAN_BYTES);
            let fits = WavePlan::new(self.wave_geometry(act_dtype))
                .max_prefill_rows_for_tier_and_kv(
                    ground,
                    WaveWidth::default(),
                    kv_row,
                    launch_scratch_reserve(&self.model.device),
                );
            if fits > 0 {
                cap = cap.min(fits);
            }
        }
        if let Some(kv_fits) = self.kv_width_cap(act_dtype) {
            let held_rows = withheld.checked_div(kv_row).unwrap_or(0);
            cap = cap.min(kv_fits.saturating_sub(held_rows).max(1));
        }
        cap
    }

    /// The same ground the width cap sizes against — [`Self::placeable_tier_bytes`].
    fn prefill_ground_bytes(&self) -> Option<usize> {
        self.placeable_tier_bytes()
    }

    /// The expert dispatch's look-ahead counts each expert's votes in 16 bits
    /// (`moe_predict_votes`), over every row of the forward.
    fn forward_row_ceiling(&self) -> Option<usize> {
        Some(MOE_PREDICT_VOTES_MAX_TOKENS)
    }

    /// K and V per KV head on the layers that page KV: the attention layers and
    /// the draft head's — a quarter of the trunk — not every layer, which the
    /// default charges.
    fn kv_bytes_per_row(&self, act_dtype: DType) -> Option<usize> {
        let cfg = &self.model.cfg;
        let per_layer = 2 * cfg.num_kv_heads * cfg.attn_head_dim * act_dtype.size_in_bytes();
        per_layer
            .checked_mul(cfg.kv_layers().total())
            .filter(|&bytes| bytes > 0)
    }

    fn reclaimable_kv_bytes(&self) -> usize {
        self.model.experts.cedeable_span_bytes()
    }

    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        Qwen4ExpBatched::maybe_change_dtype(self, dtype, dtype)
    }

    fn num_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    fn n_kv_head(&self) -> usize {
        self.model.cfg.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.model.cfg.attn_head_dim
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn carries_recurrent_state(&self) -> bool {
        // The GDN state is an accumulated sum with no per-token decomposition,
        // and the PLE conv history is likewise irrecoverable from the KV.
        true
    }

    fn short_span_rows(&self) -> Option<usize> {
        Some(TIMELINE_ROWS)
    }

    fn has_recurrent_state(&self, seq: usize) -> bool {
        self.has_recurrent(seq).unwrap_or(false)
    }

    fn recurrent_census(&self) -> (usize, usize) {
        (
            self.recurrent_len().unwrap_or(0),
            self.parked.read().map_or(0, |p| p.len()),
        )
    }

    fn prefill_scratch_bytes(&self, seq: usize) -> usize {
        let held = self
            .recurrent
            .read()
            .ok()
            .and_then(|map| map.get(&seq).map(|store| store.holds_scratch()));
        match held {
            Some(false) => {
                let cfg = &self.model.cfg;
                RecurrentStateStore::scratch_bytes_for(&cfg.layer_kinds, &cfg.delta_net)
            }
            _ => 0,
        }
    }

    /// Per selecting layer, the live tail's Q8_0 key pages a prefill of `rows`
    /// rows from `offset` opens ([`key_page_growth_bytes`]). Unpriced, the index
    /// grew two regions under a cohort fitted with 84 MiB to spare beside other
    /// unpriced claims, and its window was refused eight regions short
    /// (WhiteWorld, f3bf5e313, 2026-10-19).
    fn index_growth_bytes(&self, offset: usize, rows: usize) -> usize {
        let d = self.model.cfg.indexer.head_dim;
        self.attention_ratios()
            .into_iter()
            .map(|ratio| key_page_growth_bytes(d, ratio, offset, rows))
            .sum()
    }

    fn park_recurrent(&self, seq: usize) -> Result<bool> {
        Qwen4ExpBatched::park_recurrent(self, seq)
    }

    fn recurrent_parked(&self, seq: usize) -> bool {
        Qwen4ExpBatched::recurrent_parked(self, seq)
    }

    fn claim_carried_state(&self, seq: usize) -> Result<()> {
        Qwen4ExpBatched::claim_carried_state(self, seq)
    }

    fn recurrent_memory_count(&self) -> usize {
        Qwen4ExpBatched::recurrent_len(self).unwrap_or(0)
    }

    fn recurrent_reserved_bytes(&self) -> usize {
        Qwen4ExpBatched::recurrent_reserved_bytes(self)
    }

    fn recurrent_store_bytes(&self) -> usize {
        Qwen4ExpBatched::recurrent_store_bytes(self)
    }

    fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        Qwen4ExpBatched::compact_recurrent(self, max_moves)
    }

    /// A view carve. The child borrows the parent's K/V; its carried state has
    /// to be copied, because it is about to advance it.
    fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        Qwen4ExpBatched::fork_recurrent(self, parent, child)
    }

    /// A view finalizes: its decoded blocks transfer to the parent, and its
    /// carried state goes with them.
    fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        Qwen4ExpBatched::move_recurrent(self, child, parent)
    }

    fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        Qwen4ExpBatched::export_recurrent(self, seq)
    }

    fn arm_recurrent_span(&self, seq: usize, at: usize, len: usize, kind: ArmKind) -> Result<bool> {
        let cfg = &self.model.cfg;
        let layers: Vec<u32> = cfg
            .layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| matches!(k, LayerKind::DeltaNet))
            .map(|(i, _)| i as u32)
            .collect();
        self.arms
            .write()
            .map_err(|_| candle::Error::Msg("arms lock poisoned".into()))?
            .arm(
                seq,
                (at, len),
                kind,
                &layers,
                &cfg.delta_net,
                &self.model.device,
            )?;
        Ok(true)
    }

    fn take_recurrent_capture(&self, seq: usize) -> Result<Option<ReplyRows>> {
        self.arms
            .write()
            .map_err(|_| candle::Error::Msg("arms lock poisoned".into()))?
            .take_capture(seq)
    }

    fn export_aux_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        Qwen4ExpBatched::export_aux_state(self, seq)
    }

    /// **Attention layers only, and not the draft head's.**
    ///
    /// The default assumes a uniform transformer where every layer attends and
    /// so every layer has a Q in the KV cache to capture. Three quarters of
    /// this stack is gated DeltaNet and has no Q at all, and the layer past the
    /// trunk is the speculative head's — its Q is about a continuation the head
    /// proposed, not about the conversation.
    ///
    /// Reporting the trunk's 48 here is not a slow path, it is silence: the
    /// fold is derived from this number, so it grouped `[46, 1, 1]` over a
    /// stack that offers 12 KV layers, could not fill its three layer-groups,
    /// and refused every capture — turns sealed, K/V was written, and only the
    /// signature was quietly missing, leaving retrieval on recency.
    fn model_core_properties(&self) -> ModelCoreProperties {
        let mut props = ManagedBatchedModel::default_core_properties(self);
        props.provenance_capture_layers = self.model.cfg.n_attention_layers();
        props
    }

    fn carries_positional_state(&self) -> bool {
        // The QSA index: one pooled key per `ratio` tokens per attention layer,
        // derived from hidden states rather than from stored K, so borrowing a
        // prefix's K/V does not bring it along.
        true
    }

    fn reset_positional_state(&self, seq: usize) -> Result<()> {
        Qwen4ExpBatched::reset_positional_state(self, seq)
    }

    fn positional_cut_floor(&self, seq: usize, tokens: usize) -> Result<usize> {
        Qwen4ExpBatched::positional_cut_floor(self, seq, tokens)
    }

    fn truncate_positional_state(&self, seq: usize, tokens: usize) -> Result<()> {
        Qwen4ExpBatched::truncate_positional_state(self, seq, tokens)
    }

    fn splice_positional_state(
        &self,
        seq: usize,
        from: usize,
        to: usize,
        pieces: &[PositionalPiece],
    ) -> Result<()> {
        Qwen4ExpBatched::splice_positional_state(self, seq, from, to, pieces)
    }

    fn seal_positional_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        Qwen4ExpBatched::seal_positional_state(self, seq)
    }

    fn seal_positional_range(&self, seq: usize, start_pos: usize) -> Result<Option<Vec<u8>>> {
        Qwen4ExpBatched::seal_positional_range(self, seq, start_pos)
    }

    fn seal_positional_tail_span(
        &self,
        seq: usize,
        tokens: usize,
    ) -> Result<Vec<(usize, Vec<u8>)>> {
        Qwen4ExpBatched::seal_positional_tail_span(self, seq, tokens)
    }

    fn close_positional_page(&self, seq: usize) -> Result<usize> {
        Qwen4ExpBatched::close_positional_page(self, seq)
    }

    fn positional_coverage(&self, seq: usize) -> Option<usize> {
        let map = self.index.read().ok()?;
        // **A sequence with no entry covers nothing, and says so.** Returning
        // `None` here reads as "this model keeps no per-position state" to every
        // caller, which is how a slot holding borrowed K/V and no index at all
        // slipped past guards written as `if let Some(cov)` — the one case they
        // most needed to catch.
        let Some(caches) = map.get(&seq) else {
            return Some(0);
        };
        // The narrowest live layer: they index one stream, so the smallest is
        // what the sequence can actually select against.
        caches
            .iter()
            .zip(self.attention_ratios())
            .filter(|(_, ratio)| *ratio > 0)
            .map(|(c, ratio)| c.indexed_tokens(ratio))
            .min()
            .or(Some(0))
    }

    fn push_positional_state(&self, seq: usize, blob: &[u8], key: &PieceKey) -> Result<bool> {
        Qwen4ExpBatched::push_positional_state(self, seq, blob, key)?;
        Ok(true)
    }

    fn pool_positional_state(&self, seq: usize, blob: &[u8], key: &PieceKey) -> Result<usize> {
        Qwen4ExpBatched::pool_positional_state(self, seq, blob, key)
    }

    fn retain_pooled(&self, seq: usize, keep: &[PieceKey]) -> Result<()> {
        Qwen4ExpBatched::retain_pooled(self, seq, keep)
    }

    fn pooled_keys(&self, seq: usize) -> Vec<PieceKey> {
        Qwen4ExpBatched::pooled_keys(self, seq)
    }

    fn positional_lift_bytes(&self, page: &[u8], key: &PieceKey) -> Result<usize> {
        Qwen4ExpBatched::positional_lift_bytes(self, page, key)
    }

    fn creep_residual_bytes(&self, session: &BatchedInferenceSession, rows: usize) -> usize {
        // Two halves of the rows' hidden state — `CreepResidual::claim` — each
        // the region-sized blocks its arena claims.
        WaveSweep::residual_row(self, session).map_or(0, |(width, dtype)| {
            2 * ResidualArena::claim_bytes(&self.model.device, rows, width, dtype)
        })
    }
    fn push_positional_gap(&self, seq: usize, tokens: usize) -> Result<()> {
        Qwen4ExpBatched::push_positional_gap(self, seq, tokens)
    }

    fn restore_aux_state(&self, seq: usize, blob: &[u8]) -> Result<bool> {
        Qwen4ExpBatched::restore_aux_state(self, seq, blob)?;
        Ok(true)
    }

    fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<bool> {
        Qwen4ExpBatched::restore_recurrent(self, seq, schedule_hash, layers)?;
        Ok(true)
    }

    fn stage_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: Arc<[ExportedLayerState]>,
    ) -> Result<bool> {
        Qwen4ExpBatched::stage_recurrent(self, seq, schedule_hash, layers)?;
        Ok(true)
    }

    /// This card's swept ceiling, gated on the head actually being loaded — a
    /// conversion without the NextN tensors would otherwise pay a drafting call
    /// on every wave that can only return nothing.
    fn draft_budget(&self, width: usize) -> usize {
        if self.model.mtp.is_none() {
            return 0;
        }
        // Bounded by the timeline: a block of `k` drafts verifies `k + 1` rows,
        // and a block deeper than each DeltaNet layer's timeline is refused
        // (`begin_verify`). The rewind costs no memory a deeper block adds to,
        // so depth is otherwise the sweep's alone.
        self.draft.budget(width).min(TIMELINE_ROWS - 1)
    }

    /// The checkpoint's measured drafted-token cost.
    fn draft_token_cost(&self) -> f32 {
        flash_next_sweep::TOKEN_COST
    }

    /// The NextN head, when loaded: it runs over every row of a sweep that
    /// reaches the last trunk layer (`head_wave_pass`).
    fn sweep_head_layers(&self) -> usize {
        usize::from(self.model.mtp.is_some())
    }

    /// Draft with the checkpoint's own NextN head, for the whole cohort in one
    /// batched walk ([`super::draft`]).
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.mtp_draft(session, seqs, committed, max_len)
    }

    /// A rewind is expressed, so the entry point admits this model.
    ///
    /// The claim is about [`Self::truncate_sequences`]: that it restores all
    /// three recurrences exactly for the targets it accepts. It is not a claim
    /// that every offset is rewindable — a fold covers the block whose timeline
    /// the verify wave wrote and nothing else, and `truncate_sequences` refuses
    /// an uncovered target itself, at the one place that knows.
    fn can_rewind_speculative_block(&self) -> bool {
        true
    }

    /// Plan a wave that verifies every drafted block alongside the plain
    /// cohort, and arm what that wave has to capture.
    ///
    /// Each block is a **prefill span**, not a run of decode rows: all three of
    /// this stack's recurrences are sequential within a sequence, so two rows
    /// of one sequence cannot decode in parallel against a single carried
    /// state. The prefill scan is the form that walks them in order. Plain rows
    /// lead as ordinary decode rows in the same wave, so the step pays one
    /// launch floor rather than two and the MoE's expert traffic amortises
    /// across both cohorts.
    fn begin_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        budget: usize,
    ) -> Result<Option<VerifyPlan>> {
        let _ = (session, budget);
        if plain.is_empty() && seqs.is_empty() {
            return Ok(Some(VerifyPlan {
                decode_seqs: Vec::new(),
                decode_inputs: Vec::new(),
                verify_seqs: Vec::new(),
                verify_inputs: Vec::new(),
                rows: 0,
            }));
        }
        if seqs.len() != blocks.len() {
            candle::bail!(
                "qwen4exp verify: {} sequences against {} blocks",
                seqs.len(),
                blocks.len()
            );
        }
        // A one-token block is a decode step; the driver routes those to
        // `plain` and never here. Refused rather than folded, because the
        // capture's `len` would then disagree with the block.
        if let Some(b) = blocks.iter().find(|b| b.len() < 2) {
            candle::bail!(
                "qwen4exp verify: a {}-token block is a plain decode step, not a verify",
                b.len()
            );
        }
        // A block deeper than the DeltaNet timeline would run as a prefill span,
        // which writes a whole advanced state and cannot keep only a prefix of
        // it — so it could never be partially accepted. Refused where it is
        // formed, never run.
        if let Some(b) = blocks.iter().find(|b| b.len() > TIMELINE_ROWS) {
            candle::bail!(
                "qwen4exp verify: a {}-row block is deeper than the {TIMELINE_ROWS}-row \
                 timeline — at most {} drafts a block",
                b.len(),
                TIMELINE_ROWS - 1
            );
        }
        // The model's device, not the host: these rows are cat'd with whatever
        // else the caller has on the wave. One upload per group.
        let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
        let tokens: Vec<u32> = plain.iter().map(|&(_, t)| t).collect();
        let (dinputs, pinputs) = upload_plan_rows(&tokens, blocks)?;

        // **Size the capture's rows before the forward opens.** A wave's storage
        // is claimed by `admit_wave_kv` and the transient tier placed against
        // that claim, so the arena refuses a device allocation from inside the
        // forward — which is what rows allocated as the sweep reached each layer
        // would be.
        let cohort: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, blocks[i].len()))
            .collect();
        // A capture still armed from a step that never reached its rewind is
        // released first — its timelines discarded, its rows the ones this step
        // reuses.
        let stale = self
            .verify
            .write()
            .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?
            .take();
        if let Some(old) = stale {
            self.release_capture(old)?;
        }
        let rows: usize = cohort.iter().map(|&(_, n)| n).sum();
        let cap = SpecCapture::new(&cohort, self.capture_rows_for(rows)?)?;
        let mut tokens: Vec<(usize, Vec<u32>)> = Vec::with_capacity(seqs.len());
        for (i, &s) in seqs.iter().enumerate() {
            tokens.push((s, blocks[i].clone()));
        }
        {
            let mut g = self
                .verify
                .write()
                .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
            let mut cap = cap;
            for (s, t) in tokens {
                if let Some(slot) = cap.seqs.get_mut(&s) {
                    slot.tokens = t;
                }
            }
            *g = Some(cap);
        }
        Ok(Some(VerifyPlan {
            decode_seqs: dseqs,
            decode_inputs: dinputs,
            verify_seqs: seqs.to_vec(),
            verify_inputs: pinputs,
            rows: plain.len() + blocks.iter().map(|b| b.len()).sum::<usize>(),
        }))
    }

    /// Read the verify wave's rows back and advance what it wrote.
    ///
    /// The capture deliberately OUTLIVES this: the driver truncates back to the
    /// accepted prefix afterwards, and that rewind is what the capture exists
    /// for. [`Self::truncate_sequences`] is what releases it.
    fn end_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        logits: Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        for (i, &seq) in seqs.iter().enumerate() {
            session.advance_sequence(seq, blocks[i].len())?;
        }
        let lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
        split_block_rows(&logits, plain.len(), &lens)
    }

    /// The step keeps none of its blocks: the capture goes, and with it every
    /// armed sequence's pending timelines — each sequence's GDN state stands
    /// exactly as the step entered it. Left in place, a later truncate would
    /// fold rows nothing kept.
    fn abort_verify(&self, seqs: &[usize]) {
        let _ = seqs;
        let released = self.verify.write().ok().and_then(|mut g| g.take());
        if let Some(cap) = released {
            if let Err(e) = self.release_capture(cap) {
                tracing::error!("qwen4exp abort_verify: {e}");
            }
        }
    }

    /// Snapshot what every sequence of `seqs` carries outside the K/V, and open
    /// a step on its recurrent store. A sequence with no carried state yet —
    /// one this step creates it for — is left out: nothing it entered with
    /// could be restored, and it is no worse off than before the bracket.
    fn begin_step(&self, seqs: &[usize]) -> Result<()> {
        let poisoned = |what: &str| candle::Error::Msg(format!("qwen4exp: {what} lock poisoned"));
        let mut step = self.step.lock().map_err(|_| poisoned("step"))?;
        if step.is_some() {
            candle::bail!("qwen4exp: begin_step with a step already open");
        }
        let index = self.index.read().map_err(|_| poisoned("index"))?;
        let ple = self.ple.read().map_err(|_| poisoned("ple"))?;
        let seeds = self.seeds.read().map_err(|_| poisoned("seed"))?;
        let mut rec = self.recurrent.write().map_err(|_| poisoned("recurrent"))?;
        let mut snap = StepSnapshot {
            seqs: Vec::new(),
            index: Vec::new(),
            ple: Vec::new(),
            seeds: Vec::new(),
        };
        let opened = seqs.iter().try_for_each(|&s| {
            let (Some(caches), Some(p), true) = (index.get(&s), ple.get(&s), rec.contains_key(&s))
            else {
                return Ok(());
            };
            let idx = caches
                .iter()
                .map(IndexCache::snapshot)
                .collect::<Result<Vec<_>>>()?;
            rec.get_mut(&s).expect("checked above").begin_step()?;
            snap.seqs.push(s);
            snap.index.push((s, idx));
            snap.ple.push((s, p.clone()));
            snap.seeds.push((s, seeds.get(&s).cloned()));
            Ok(())
        });
        if let Err(e) = opened {
            // No step is open, so none of the stores opened so far may keep one.
            // No wave of the step ran, so none is owed a fold.
            let mut opened: Vec<&mut RecurrentStateStore> = rec
                .iter_mut()
                .filter(|(s, _)| snap.seqs.contains(s))
                .map(|(_, store)| store)
                .collect();
            end_steps(&mut opened)?;
            return Err(e);
        }
        *step = Some(snap);
        Ok(())
    }

    /// Keep the step — folding, in one launch, the rows its commits owed the
    /// recurrent stores — or put every sequence it covered back where the step
    /// found it: K/V on the KV layers of trunk layers `[0, committed)`, the
    /// recurrent stores by dropping the rows the step's waves left pending, and
    /// the index, PLE history and seed from the snapshot. Every part is
    /// attempted; the first failure is returned.
    fn end_step(
        &self,
        session: &mut BatchedInferenceSession,
        keep: bool,
        committed: usize,
    ) -> Result<()> {
        let poisoned = |what: &str| candle::Error::Msg(format!("qwen4exp: {what} lock poisoned"));
        let Some(snap) = self.step.lock().map_err(|_| poisoned("step"))?.take() else {
            return Ok(());
        };
        let mut first_err: Option<candle::Error> = None;
        let mut note = |r: Result<()>| {
            if let Err(e) = r {
                first_err.get_or_insert(e);
            }
        };
        {
            let mut rec = self.recurrent.write().map_err(|_| poisoned("recurrent"))?;
            let mut stores: Vec<&mut RecurrentStateStore> = rec
                .iter_mut()
                .filter(|(s, _)| snap.seqs.contains(s))
                .map(|(_, store)| store)
                .collect();
            if keep {
                return end_steps(&mut stores);
            }
            for store in stores {
                note(store.undo_step());
            }
        }
        let (kv_start, kv_end) = self.kv_layer_range(0, committed.min(self.model.cfg.num_layers));
        note(crate::models::wave_admit::rollback_step_kv(
            session, &snap.seqs, kv_start, kv_end,
        ));
        {
            let mut idx = self.index.write().map_err(|_| poisoned("index"))?;
            for (s, snaps) in &snap.index {
                if let Some(caches) = idx.get_mut(s) {
                    for (cache, sn) in caches.iter_mut().zip(snaps) {
                        note(cache.restore(sn));
                    }
                }
            }
        }
        {
            let mut ple = self.ple.write().map_err(|_| poisoned("ple"))?;
            for (s, st) in snap.ple {
                ple.insert(s, st);
            }
        }
        {
            let mut seeds = self.seeds.write().map_err(|_| poisoned("seed"))?;
            let mut spares = self.seed_spares.write().map_err(|_| poisoned("seed"))?;
            for (s, seed) in snap.seeds {
                // As the wave bracket's restore: the seed put back may already
                // be the spare, which the next carry would then write.
                spares.remove(&s);
                match seed {
                    Some(prev) => {
                        seeds.insert(s, prev);
                    }
                    None => {
                        seeds.remove(&s);
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Roll every target back to its accepted prefix — K/V and all three
    /// recurrences — in one pass over the cohort.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        self.rewind_cohort(session, targets)
    }

    fn rope_select(&self) -> RungSelect {
        self.rope.select().clone()
    }

    fn set_selection_prompt(&self, seq: usize, tokens: usize) -> Result<()> {
        Qwen4ExpBatched::set_selection_prompt(self, seq, tokens)
    }

    fn set_selection_pins(&self, seq: usize, ranges: &[Range<usize>]) -> Result<()> {
        Qwen4ExpBatched::set_selection_pins(self, seq, ranges)
    }

    fn set_hit_layers(&mut self, layers: &HitPercentiles) -> Result<()> {
        Qwen4ExpBatched::set_hit_layers(self, layers)
    }

    fn set_hit_rounds(
        &self,
        seq: usize,
        rounds: &[Range<usize>],
        pooled: &[PooledRound],
    ) -> Result<()> {
        Qwen4ExpBatched::set_hit_rounds(self, seq, rounds, pooled)
    }

    fn take_hits(&self, seq: usize) -> Result<Option<PieceHits>> {
        Qwen4ExpBatched::take_hits(self, seq)
    }

    fn set_selection_strata(&mut self, strata: StrataTokens) -> Result<()> {
        Qwen4ExpBatched::set_selection_strata(self, strata)
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let cfg = &self.model.cfg;
        // This model's KV threshold calibration, folded in here because this
        // override replaces the `ManagedBatchedModel` default that would
        // otherwise do it — the KV layer count differs from the transformer
        // depth on a hybrid, and dropping the fold would leave the per-model
        // row silently never reaching the compression policy. The row is the
        // one calibrated for this artifact's expert format (`kv_factors_for`).
        let mut config = config;
        let row = kv_factors_for(self.model.expert_format);
        config.k_hi_error_threshold_factor *= row.k_hi;
        config.k_low_error_threshold_factor *= row.k_low;
        config.v_hi_error_threshold_factor *= row.v_hi;
        config.v_low_error_threshold_factor *= row.v_low;
        // The trunk's KV layers plus the draft head's, so the head holds its
        // keys in this same paged cache and prefills, decodes and seals
        // alongside the trunk without any session-wide operation knowing it
        // exists. The head's layer is the last one and is stepped only by the
        // head's own pass — see `Qwen4ExpConfig::mtp_kv_layer`.
        let mut session = BatchedInferenceSession::new(
            cfg.kv_layers(),
            cfg.num_kv_heads,
            cfg.attn_head_dim,
            &self.model.device,
            config,
        )?;
        session.set_rope_select(self.rope.select().clone());
        // **The draft head's layer is capped; the trunk's twelve take the
        // session's level unchanged.**
        //
        // The trunk spreads every read over twelve KV layers, so a level's key
        // error is partly averaged over the depth. The head is one block with
        // one KV layer and absorbs none of it — and a proposal is worth
        // something only if it reproduces the trunk's argmax, which is a far
        // sharper test than "the text still reads correctly". Measured: capping
        // this one layer takes C6–C9 acceptance from 1.5–4.9% back to ~90%, and
        // throughput from ~25 to ~117 tok/s, with the trunk untouched.
        //
        // A ceiling rather than a setting, so a session running at C0–C3 keeps
        // its own level and only the deeper rungs are held back. One layer of
        // thirteen, so the ladder's calibrated KV footprint moves by well under
        // a tenth.
        if let Some(head_kv) = cfg.mtp_kv_layer() {
            session.cap_layer_seal_level(head_kv, DRAFT_HEAD_MAX_COMPRESSION)?;
        }
        // The Q/K norm weights meet activations at the KV arena's width, which
        // is NOT the residual stream's — see `maybe_change_dtype`. Both are
        // passed so the call site says which is which.
        Qwen4ExpBatched::maybe_change_dtype(
            self,
            session.activation_dtype(),
            session.kv_live_dtype(),
        )?;
        Ok(session)
    }

    fn prune(&self) -> Result<()> {
        if let Ok(mut m) = self.parked.write() {
            m.clear();
        }
        if let Ok(mut m) = self.recurrent.write() {
            m.clear();
        }
        if let Ok(mut m) = self.pristine.write() {
            m.clear();
        }
        if let Ok(mut m) = self.staged.write() {
            m.clear();
        }
        if let Ok(mut m) = self.ple.write() {
            m.clear();
        }
        if let Ok(mut m) = self.index.write() {
            m.clear();
        }
        if let Ok(mut m) = self.retired.write() {
            m.clear();
        }
        if let Ok(mut m) = self.seeds.write() {
            m.clear();
        }
        if let Ok(mut m) = self.seed_spares.write() {
            m.clear();
        }
        // A recycled id must not count into the previous conversation's
        // pieces.
        if let Ok(mut m) = self.hits.write() {
            m.clear();
        }
        if let Ok(mut t) = self.hit_table.lock() {
            *t = None;
        }
        Ok(())
    }

    /// The model's half of freeing a sequence: the session owns the KV slots,
    /// these maps own the GDN stores (span-reservation tenants), the PLE
    /// state and the QSA index caches — leaving them keyed holds their
    /// regions for the life of the process while every KV-side diagnostic
    /// reads healthy.
    fn release_sequence(&self, seq: usize) -> Result<()> {
        self.clear_pristine(seq);
        self.unpark(seq);
        if let Ok(mut m) = self.staged.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.recurrent.write() {
            m.remove(&seq);
            // **The parked capture rows go with the last sequence.** They are
            // kept across steps for the next verify — so with nobody left to
            // verify they would be held for the life of the process.
            if m.is_empty() {
                if let Ok(mut parked) = self.parked_rows.lock() {
                    *parked = None;
                }
            }
        }
        if let Ok(mut m) = self.ple.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.arms.write() {
            m.forget(seq);
        }
        if let Ok(mut m) = self.index.write() {
            m.remove(&seq);
        }
        self.pooled.release(seq);
        // A recycled id must not inherit the previous conversation's pieces.
        if let Ok(mut m) = self.hits.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.retired.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.seeds.write() {
            m.remove(&seq);
        }
        if let Ok(mut m) = self.seed_spares.write() {
            m.remove(&seq);
        }
        // Or the next sequence to be handed this slot id inherits a suppression
        // it never asked for, and its first wave keeps whatever the released
        // one left behind.
        if let Ok(mut m) = self.seeded.write() {
            m.remove(&seq);
        }
        Ok(())
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        Some(self.model.experts.expert_stats())
    }

    fn expert_pipeline_dead(&self) -> bool {
        self.model.experts.pipeline_dead()
    }

    fn expert_stats_shared(&self) -> Option<Arc<Mutex<PipelineStats>>> {
        Some(self.model.experts.stats_shared())
    }

    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        Some(self.model.experts.hit_counts())
    }

    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        self.model.experts.hit_references()
    }

    fn write_expert_routing_trace(&self, path: &Path) -> Result<()> {
        self.model.experts.write_routing_trace(path)
    }

    fn weight_plan(&self) -> WeightPlanning {
        WeightPlan::from_stats(&self.model.experts.expert_stats())
    }

    /// The PLE row cache's hit/miss/eviction counters (§0.1). The table is
    /// disk-resident and never enters VRAM, so this is the number that says
    /// whether the 2 GB cache is sized for the traffic in front of it.
    fn row_cache_stats(&self) -> Option<(u64, u64, u64)> {
        self.model
            .ple_table
            .cache_stats()
            .map(|s| (s.hits, s.misses, s.evictions))
    }

    fn reset_expert_stats(&self) {
        self.model.experts.reset_expert_stats();
    }

    /// Buy `regions` of expert-zone ground for the KV side, between forwards — the
    /// scheduler's half of the boundary. See `ExpertCache::request_kv_ground`.
    fn request_kv_ground(&self, regions: usize) -> u64 {
        self.model.experts.request_kv_ground(regions)
    }

    /// Let the expert zone take back KV ground standing free, between forwards.
    ///
    /// The wave asks this at the head of every forward, but a scheduler that has
    /// just packed the pools or watched a turn leave the device has to be able to
    /// ask too: left to the next forward, the question lands in a prefill that is
    /// buying ground and is refused, so the zone stayed down at its 128K low from
    /// one Strata run to the next — 24,914 → 23,430 → 21,953 slots.
    fn reclaim_spare_ground(&self) {
        self.model.experts.reclaim_spare_ground();
    }

    /// The experts' working room, down to the pinned layers, for a guest that
    /// runs while the waves are paused. Left to the trait's default this model
    /// conceded nothing, and every image draw on a busy WhiteWorld waited out
    /// its patience 7 GiB short (`conceded_mib=0`, 2026-10-27).
    fn concede_for_guest(&self, regions: usize) -> u64 {
        self.model.experts.concede_for_guest(regions)
    }

    fn stash_pinned_for_guest(&self) -> u64 {
        self.model.experts.stash_pinned_for_guest()
    }

    fn restore_serving_floor(&self) -> bool {
        self.model.experts.restore_serving_floor()
    }

    /// The expert pipeline thread's spans — where its time goes serving each
    /// routed layer, which decides how far behind the GPU it runs.
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        self.model.experts.snapshot_profiles()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        session: &mut BatchedInferenceSession,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        glue_seqs: &[usize],
        glue_inputs: &[Tensor],
        layer_start: usize,
        layer_end: usize,
        residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        session.expect_rope_select(self.rope.select())?;
        drive_wave(
            self,
            session,
            decode_seqs,
            decode_inputs,
            prefill_seqs,
            prefill_inputs,
            glue_seqs,
            glue_inputs,
            layer_start,
            layer_end,
            residual,
        )
    }

    fn forward_step(
        &self,
        session: &mut BatchedInferenceSession,
        spec: StepSpec<'_>,
    ) -> Result<StepOutcome> {
        session.expect_rope_select(self.rope.select())?;
        drive_step(self, session, spec)
    }

    fn claim_residual(
        &self,
        session: &BatchedInferenceSession,
        rows: usize,
    ) -> Result<CreepResidual> {
        let (width, dtype) = WaveSweep::residual_row(self, session)?;
        CreepResidual::claim(&self.model.device, rows, width, dtype)
    }
}
