//! The hybrid as the scheduler drives it: pricing, drafting and rewinding.
//!
//! Geometry and KV admission are priced against the attention layers only, since
//! counting every trunk layer would over-charge a 3:1 hybrid by four times. A
//! verified draft block runs as a prefill span, because its recurrence is
//! sequential within one sequence, and its DeltaNet timeline stays pending past
//! the commit so the accepted prefix can be folded in. A rewind is exact only
//! inside a block this model armed, or to a full reset. Any other target is
//! refused, since rewinding KV under a recurrent state that still holds the
//! discarded tokens would answer as though those tokens were remembered.

use super::*;

/// The hybrid as the scheduler drives it.
///
/// Every method here either answers from the hybrid's own geometry (depth is
/// the trunk, KV heads are the attention layers') or hands the question to the
/// shared machinery — the wave driver, the expert cache. `forward_wave` is the
/// second kind: the sweep above is the model-specific half, and everything
/// around it is [`drive_wave`].
impl ManagedBatchedModel for HybridBatched {
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        HybridBatched::wave_geometry(self, act_dtype)
    }

    /// Rows the KV side can admit, priced against **attention** layers.
    ///
    /// The trait's default multiplies the per-row cost by trunk depth, which on
    /// a 3:1 hybrid over-charges by 4× and refuses four times more prefill than
    /// the cache can hold.
    fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        HybridBatched::kv_width_cap(self, act_dtype)
    }

    /// Weight ground the elastic boundary would cede to a stuck KV claim.
    ///
    /// Counted by `kv_width_cap` when sizing a prefill wave, so the wave is
    /// sliced against what the partition *can* admit rather than against what
    /// happens to stand free — on a streamed-expert model those differ by tens
    /// of gigabytes, and the difference is the whole ingest sweep.
    fn reclaimable_kv_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map_or(0, |cache| cache.cedeable_span_bytes())
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// The trait form carries one width, so it names the same one twice.
    ///
    /// The two-width call is [`HybridBatched::create_batched_session`], which
    /// holds the session and can ask it for both. This arm is reached only by
    /// the generic session helpers, where the caller has already decided that
    /// activations and KV share a width.
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        HybridBatched::maybe_change_dtype(self, dtype, dtype)
    }

    fn num_layers(&self) -> usize {
        HybridBatched::num_layers(self)
    }

    fn n_kv_head(&self) -> usize {
        HybridBatched::n_kv_head(self)
    }

    fn head_dim(&self) -> usize {
        HybridBatched::head_dim(self)
    }

    fn device(&self) -> &Device {
        HybridBatched::device(self)
    }

    fn model_core_properties(&self) -> ModelCoreProperties {
        HybridBatched::model_core_properties(self)
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        HybridBatched::create_batched_session(self, config)
    }

    fn rope_select(&self) -> RungSelect {
        self.rope().select().clone()
    }

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
        session.expect_rope_select(self.rope().select())?;
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
        session.expect_rope_select(self.rope().select())?;
        drive_step(self, session, spec)
    }

    fn claim_residual(
        &self,
        session: &BatchedInferenceSession,
        rows: usize,
    ) -> Result<CreepResidual> {
        let (width, dtype) = WaveSweep::residual_row(self, session)?;
        CreepResidual::claim(WaveSweep::device(self), rows, width, dtype)
    }

    /// This checkpoint's own ladder, supplied at construction — see
    /// [`crate::models::draft_ladder`].
    fn draft_budget(&self, width: usize) -> usize {
        self.draft_budget_for(width)
    }

    /// This checkpoint's own ladder's drafted-token cost.
    fn draft_token_cost(&self) -> f32 {
        self.draft_token_cost_for()
    }

    /// The NextN/MTP head, when loaded: it runs over every row of a sweep that
    /// reaches the last trunk layer (`head_wave_pass`).
    fn sweep_head_layers(&self) -> usize {
        usize::from(self.has_drafter())
    }

    /// Draft with the checkpoint's own NextN/MTP head ([`super::mtp`]), for the
    /// whole cohort in one batched pass.
    ///
    /// Empty — a plain decode step — when the conversion carried no head, or
    /// before a sequence has a seed. Lossless either way: the verify pass keeps
    /// only this model's own argmaxes.
    ///
    /// The budget is capped at [`MTP_MAX_DRAFT`] whatever the caller asks for.
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.mtp_draft(session, seqs, committed, max_len.min(MTP_MAX_DRAFT))
    }

    /// Plan the rows for a wave that verifies every drafted block alongside the
    /// plain cohort, and arm what that wave has to write.
    ///
    /// Each block is a **prefill span**, not a run of decode rows: the
    /// recurrence is sequential within a sequence, so two rows of one sequence
    /// cannot decode in parallel against a single carried state — the prefill
    /// scan is the form that walks them in order. Plain rows lead as ordinary
    /// decode rows in the same wave, so the whole step pays one launch floor
    /// rather than two, and the MoE's per-layer expert traffic amortizes across
    /// both cohorts.
    ///
    /// Naming the verifying sequences does two things: the head scores every
    /// one of their rows (a block position's prediction is what a proposal is
    /// checked against), and the sequences are **armed** — their DeltaNet layers
    /// keep the block's timeline pending past the forward's commit, so
    /// [`Self::truncate_sequences`] can fold the accepted prefix. The row set is
    /// cleared in `end_verify` / `abort_verify`, error or not; the arming ends
    /// at the truncate or the abort.
    fn begin_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        budget: usize,
    ) -> Result<Option<VerifyPlan>> {
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
                "qwen35 verify: {} sequences against {} blocks",
                seqs.len(),
                blocks.len()
            );
        }
        // A one-token block is a decode step; the driver routes those to
        // `plain` and never here, and a 1-row prefill would be folded into the
        // decode group by the driver — where the head scores it as one row
        // anyway, so the row split below would still hold. Refuse it rather
        // than rely on that: the armed block's `len` would disagree with it.
        if let Some(b) = blocks.iter().find(|b| b.len() < 2) {
            candle::bail!(
                "qwen35 verify: a {}-token block is a plain decode step, not a verify",
                b.len()
            );
        }
        // The model's device, not the host: these rows are cat'd with whatever
        // else the caller has on the wave, and the scheduler's creep group and
        // glue are device-side. One upload per group.
        let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
        let tokens: Vec<u32> = plain.iter().map(|&(_, t)| t).collect();
        let (dinputs, pinputs) = upload_plan_rows(&tokens, blocks)?;

        // Arm the cohort: each block's DeltaNet timeline stays pending past the
        // forward's commit until the accept says how many rows it keeps. A
        // block deeper than the timeline is refused here, before any forward.
        let cohort: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(i, &seq)| (seq, blocks[i].len()))
            .collect();
        self.arm_verify(&cohort)?;
        // Arm the MTP seed capture over BOTH cohorts: a plain row's hidden is
        // what lets a sequence that drafted nothing this step draft on the
        // next, and it is the only way the very first step after prefill ever
        // acquires a seed.
        //
        // **Skipped when the caller has switched drafting off** — above the
        // width its ladder speculates at, every seed captured is a seed nothing
        // will ever draft from, and the sweep still pays to write one hidden row
        // per sequence per layer sweep for it. Keyed on the budget rather than on
        // `seqs` being empty, because an empty verify cohort also means "the
        // drafter had no seed yet", which is exactly when this must run.
        //
        // The cost of skipping is one step: when the wave narrows back to a
        // width that drafts, the sequences have no seed and take plain rows,
        // that wave arms capture again, and the step after it drafts. Self-
        // healing, and paid only on the transition.
        if self.has_drafter() {
            if budget > 0 {
                let mut want: Vec<(usize, usize)> = plain.iter().map(|&(s, _)| (s, 1)).collect();
                want.extend(seqs.iter().enumerate().map(|(i, &s)| (s, blocks[i].len())));
                self.arm_hidden_capture(&want, session.activation_dtype())?;
            } else {
                // **Skipping the capture is not enough — the old one has to go.**
                // `capture_rows` and `verify_hidden` deliberately outlive
                // `disarm_hidden_capture`, so this step's `mtp_take_seeds` would
                // otherwise find the buffer some earlier wave filled and install
                // a row of it as the seed. The sequence would then look seeded
                // while holding a hidden from an arbitrary number of steps ago,
                // and the first wave that narrowed back under the ladder's
                // bracket would draft from it — losslessly, so nothing fails; it
                // just proposes badly and acceptance sits near 1.00.
                //
                // Dropping the state instead leaves the sequence honestly
                // seedless, which is the one-step cost the ladder's doc
                // describes: it takes plain rows on the wave that re-arms
                // capture, and drafts on the one after.
                for &(seq, _) in plain {
                    self.release_draft(seq);
                }
                for &seq in seqs {
                    self.release_draft(seq);
                }
            }
        }
        // Naming the verifying sequences makes the head score every one of
        // their rows. Cleared in `end_verify` / `abort_verify`, error or not.
        self.set_verify_row_seqs(seqs)?;
        Ok(Some(VerifyPlan {
            decode_seqs: dseqs,
            decode_inputs: dinputs,
            verify_seqs: seqs.to_vec(),
            verify_inputs: pinputs,
            rows: plain.len() + blocks.iter().map(|b| b.len()).sum::<usize>(),
        }))
    }

    fn end_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        logits: Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        self.set_verify_row_seqs(&[])?;
        self.disarm_hidden_capture();
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        for (i, &seq) in seqs.iter().enumerate() {
            session.advance_sequence(seq, blocks[i].len())?;
        }
        // Owned rows from the scheduler, or views on the verify wave's head span
        // from the step driver, which holds that wave until its accept walk has
        // read them and drops both before the rewind.
        let lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
        split_block_rows(&logits, plain.len(), &lens)
    }

    fn abort_verify(&self, seqs: &[usize]) {
        let _ = self.set_verify_row_seqs(&[]);
        self.disarm_hidden_capture();
        // The step keeps none of its blocks: their timelines go, and each
        // sequence's state stands exactly as the step entered it.
        if let Err(e) = self.disarm_verify(seqs) {
            tracing::error!("qwen35 abort_verify: {e}");
        }
    }

    /// Roll `seq` back to `tokens`, recurrent state included.
    ///
    /// Three cases, and only the middle one is new:
    ///
    /// * `tokens == 0` — a full reset. The state returns to its sequence-start
    ///   value, which is what a fresh store holds, so the store is dropped.
    /// * `tokens` inside a block this model just verified — the speculative
    ///   rewind. `S` is a running sum with no suffix to remove, but the block
    ///   never wrote it: each DeltaNet layer holds the block's timeline, and
    ///   the fold applies the accepted prefix of it ([`super::spec`]). Exact,
    ///   not approximate: the fold is the scan's own state carry over those
    ///   rows.
    /// * `tokens` equal to the sequence's current length — a no-op, which is
    ///   what the driver's uniform truncate is for a plain decode row.
    ///
    /// Anything else has no answer and is refused. KV rewound under a state
    /// that still holds the un-truncated history is silent corruption: the
    /// model answers as though it remembers tokens the cache no longer has
    /// (measured: re-prefilling a truncated prompt diverges by ~9.5 in the
    /// logits).
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        self.truncate_sequences(session, &[(seq, tokens)])
    }

    /// The whole step's truncates in one call, so the cohort's folds batch:
    /// every armed sequence's kept rows fold in ONE launch over every sequence
    /// and DeltaNet layer, instead of one per layer per sequence.
    ///
    /// Validation runs for every target before anything mutates — a target with
    /// no rewind point aborts the call with nothing truncated and nothing
    /// folded, and the armed cohort stands for the caller's abort to discard.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }

        // What each target needs, decided before anything is touched.
        enum Plan {
            Reset,
            NoOp,
            Fold { kept: usize },
        }
        let mut plans: Vec<Plan> = Vec::with_capacity(targets.len());
        for &(seq, tokens) in targets {
            if tokens == 0 {
                plans.push(Plan::Reset);
                continue;
            }
            let current = session.sequence_offset(seq).unwrap_or(0);
            // The verify wave advanced the sequence by the whole block, so the
            // block starts `len` before where it stands.
            let block = self
                .armed_block(seq)?
                .and_then(|len| current.checked_sub(len));
            match block {
                Some(start) if tokens >= start && tokens <= current => {
                    plans.push(Plan::Fold {
                        kept: tokens - start,
                    });
                }
                _ if tokens == current => plans.push(Plan::NoOp),
                _ => candle::bail!(
                    "qwen35: cannot truncate sequence {seq} to {tokens} tokens — it \
                     stands at {current} and no verified block covers that offset, so \
                     the DeltaNet recurrent state has no rewind point. Truncate to 0 \
                     (full reset), or rewind inside a block this model just verified."
                ),
            }
        }

        let mut jobs: Vec<(usize, usize)> = Vec::new();
        (|| -> Result<()> {
            // Take each sequence's next draft seed from the hiddens this wave
            // captured — the row at the last position the accept kept. This
            // runs for no-ops exactly as for rewinds, because a PLAIN row's
            // one-token block is how a sequence that drafted nothing acquires
            // the seed to draft next step; only a reset skips it, and its
            // sequence has no history left to seed from.
            //
            // A rewind carries its own row count; a **no-op does not**, and
            // must not be given one. It means the accept kept everything the
            // wave ran, which is a count only the capture knows — `None` asks
            // it. Hardcoding 1 here would be right for every plain decode row
            // and silently wrong for anything else that ever classifies as a
            // no-op, seeding from the block's first hidden instead of its last.
            //
            // The head's KV needs nothing here. It took every one of these
            // positions inside the wave, as a layer, and the truncate below
            // rolls its rejected tail back with every other layer's.
            let seeds: Vec<(usize, Option<usize>)> = targets
                .iter()
                .zip(&plans)
                .filter_map(|(&(seq, _), plan)| match plan {
                    Plan::Reset => None,
                    Plan::NoOp => Some((seq, None)),
                    Plan::Fold { kept } => Some((seq, Some(*kept))),
                })
                .collect();
            self.mtp_take_seeds(&seeds)?;

            for (&(seq, tokens), plan) in targets.iter().zip(&plans) {
                match plan {
                    Plan::Reset => {
                        session.truncate_sequence_to_tokens(seq, tokens)?;
                        self.release_recurrent(seq)?;
                    }
                    Plan::NoOp => {}
                    Plan::Fold { kept } => {
                        jobs.push((seq, *kept));
                        session.truncate_sequence_to_tokens(seq, tokens)?;
                    }
                }
            }
            // The KV truncate above walks every layer's chunk list; the fold
            // below applies the kept rows alone. They scale with different
            // things — the cache's depth against the cohort — so they are timed
            // apart.
            let t_fold = std::time::Instant::now();
            let r = self.fold_armed(&jobs);
            pipeline_record_duration("rewind:fold", t_fold.elapsed(), 1);
            r
        })()
    }

    /// **This lineage cannot be rewound.** The DeltaNet state is a running sum
    /// with no per-token decomposition, so there is no suffix to remove and no
    /// inverse to apply, and K/V rewound under a state that still holds the
    /// un-truncated history is silent corruption — the model answers as though
    /// it remembers tokens the cache no longer has (measured: re-prefilling a
    /// truncated prompt diverges by ~9.5 in the logits).
    ///
    /// What this declares is that the state exists outside the session's KV and
    /// must be snapshotted, forked and released with the sequence. It is no
    /// longer also a statement that the state cannot be rewound — see
    /// [`Self::can_rewind_speculative_block`] below.
    fn carries_recurrent_state(&self) -> bool {
        true
    }

    fn short_span_rows(&self) -> Option<usize> {
        Some(TIMELINE_ROWS)
    }

    fn has_recurrent_state(&self, seq: usize) -> bool {
        self.has_recurrent(seq).unwrap_or(false)
    }

    /// **The hybrid can rewind a verified block**, which is why it does not take
    /// the default `!carries_recurrent_state()`.
    ///
    /// `S` has no per-token decomposition, so there is no suffix to subtract —
    /// but a rewind does not need one. The verify block never wrote `S`: each
    /// DeltaNet layer holds the block's timeline, every row of which depends on
    /// no row after it, and [`Self::truncate_sequences`] folds the accepted
    /// prefix. The result is the state those `m` tokens alone would have
    /// produced (`super::spec`).
    ///
    /// The claim is bounded: it covers offsets inside a block this model armed.
    /// `truncate_sequences` refuses anything else by name rather than silently
    /// resetting, so a caller that rewinds somewhere no fold can reach gets an
    /// error instead of a zeroed recurrence.
    fn can_rewind_speculative_block(&self) -> bool {
        true
    }

    /// The recurrent map is keyed by sequence id and owned by the *model*, so
    /// the scheduler freeing the KV slot does not touch it. This is the hook
    /// that ties the two lifetimes together.
    fn release_sequence(&self, seq: usize) -> Result<()> {
        self.release_recurrent(seq)
    }

    fn recurrent_memory_count(&self) -> usize {
        HybridBatched::recurrent_len(self).unwrap_or(0)
    }

    /// A view carve. The child borrows the parent's KV; its recurrent state has
    /// to be copied, because it is about to advance it.
    ///
    /// Tolerant of a parent with no state yet: a view can be carved before the
    /// parent has ever run a wave (a brand-new conversation's first turn), and
    /// there the child correctly starts from the sequence-start value — a fresh
    /// store holds exactly that, so the child's own `ensure_recurrent` does the
    /// right thing and there is nothing to copy. A parent holding a snapshot
    /// it has not built a store from yet has state, and its child is built
    /// from that.
    fn fork_recurrent(&self, parent: usize, child: usize) -> Result<()> {
        if !self.holds_recurrent(parent)? {
            return Ok(());
        }
        HybridBatched::fork_recurrent(self, parent, child)
    }

    /// A view finalizes: its decoded blocks transfer to the parent, and its
    /// state goes with them.
    ///
    /// Tolerant in the same direction and for the same reason — a view that
    /// never ran a wave has nothing to move, and the parent keeps what it had.
    fn move_recurrent(&self, child: usize, parent: usize) -> Result<()> {
        if !self.has_recurrent(child)? {
            return Ok(());
        }
        HybridBatched::move_recurrent(self, child, parent)
    }

    fn export_recurrent(&self, seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        HybridBatched::export_recurrent(self, seq)
    }

    fn restore_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<bool> {
        HybridBatched::restore_recurrent(self, seq, schedule_hash, layers)?;
        Ok(true)
    }

    fn stage_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: Arc<[ExportedLayerState]>,
    ) -> Result<bool> {
        HybridBatched::stage_recurrent(self, seq, schedule_hash, layers)?;
        Ok(true)
    }

    fn prune(&self) -> Result<()> {
        Ok(())
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        #[cfg(feature = "cuda")]
        {
            self.model().experts.as_ref().map(|c| c.expert_stats())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// With an expert cache, its dispatch's look-ahead counts each expert's
    /// votes in 16 bits (`moe_predict_votes`), over every row of the forward.
    fn forward_row_ceiling(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map(|_| MOE_PREDICT_VOTES_MAX_TOKENS)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn expert_pipeline_dead(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .is_some_and(|c| c.pipeline_dead())
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn expert_stats_shared(&self) -> Option<Arc<Mutex<PipelineStats>>> {
        #[cfg(feature = "cuda")]
        {
            self.model().experts.as_ref().map(|c| c.stats_shared())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            self.model().experts.as_ref().map(|c| c.hit_counts())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .and_then(|c| c.hit_references())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn weight_plan(&self) -> WeightPlanning {
        // A dense checkpoint of this lineage has no expert cache, which is `Dense` and
        // not a broken gauge set — see `WeightPlanning`.
        match self.expert_stats() {
            Some(s) => WeightPlan::from_stats(&s),
            None => WeightPlanning::Dense,
        }
    }

    fn layer_stream_stats(&self) -> Option<[usize; 7]> {
        #[cfg(feature = "cuda")]
        {
            self.model().layers.stats().ok().flatten().map(|s| {
                [
                    s.hits,
                    s.warm_hits,
                    s.cold_reads,
                    s.fence_joins,
                    s.evictions,
                    s.homed,
                    s.streaming,
                ]
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn request_kv_ground(&self, regions: usize) -> u64 {
        #[cfg(feature = "cuda")]
        {
            // Whichever side of this model holds tradeable ground. A checkpoint
            // is routed or dense: a routed one's slack is in its expert slots, a
            // dense one's in its layer slots, and neither has both. Asking both
            // is how the caller stays ignorant of which it is holding.
            self.model()
                .experts
                .as_ref()
                .map_or(0, |c| c.request_kv_ground(regions))
                + self.model().layers.request_kv_ground(regions)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = regions;
            0
        }
    }

    /// Let the weight side — experts or streamed layers — take back KV ground
    /// standing free, between forwards. The wave asks at its own head; this is the
    /// scheduler asking after a compaction or a settle, which would otherwise wait
    /// for the next forward and find it buying ground instead.
    fn reclaim_spare_ground(&self) {
        HybridBatched::reclaim_spare_ground(self);
    }

    /// The experts only: a streamed layer's floor is the pinned head plus the
    /// one cell a streamed layer lands in, which has no copy in any tier.
    fn concede_for_guest(&self, regions: usize) -> u64 {
        #[cfg(feature = "cuda")]
        {
            let Some(cache) = self.model().experts.as_ref() else {
                tracing::info!(
                    target: "candle_transformers::expert_lre",
                    regions,
                    "a guest asked for the experts' working room, and this model has no expert cache"
                );
                return 0;
            };
            cache.concede_for_guest(regions)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = regions;
            0
        }
    }

    fn restore_serving_floor(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .is_none_or(|c| c.restore_serving_floor())
        }
        #[cfg(not(feature = "cuda"))]
        {
            true
        }
    }

    /// Dense tensors **plus** whatever experts are resident, which is what the
    /// trait asks for and what the whole-card decomposition subtracts the expert
    /// half back out of. Reporting only the experts made the dense half cancel
    /// to zero — several GiB that no report, and no budget, could see.
    fn resident_weight_bytes(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            let m = self.model();
            let experts = m.experts.as_ref().map_or(0, |c| c.resident_vram_bytes());
            Some(m.dense_bytes + experts)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn recurrent_reserved_bytes(&self) -> usize {
        HybridBatched::recurrent_reserved_bytes(self)
    }

    fn recurrent_store_bytes(&self) -> usize {
        HybridBatched::recurrent_store_bytes(self)
    }

    fn compact_recurrent(&self, max_moves: usize) -> Result<RecurrentCompaction> {
        HybridBatched::compact_recurrent(self, max_moves)
    }

    fn reset_expert_stats(&self) {
        #[cfg(feature = "cuda")]
        if let Some(c) = self.model().experts.as_ref() {
            c.reset_expert_stats();
        }
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        #[cfg(feature = "cuda")]
        {
            self.model()
                .experts
                .as_ref()
                .map_or_else(ProfileSnapshot::default, |c| c.snapshot_profiles())
        }
        #[cfg(not(feature = "cuda"))]
        {
            ProfileSnapshot::default()
        }
    }
}
