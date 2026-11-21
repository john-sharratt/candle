//! The scheduler-facing hooks of the DeepSeek batched engine.
//!
//! The hooks price a row's transient memory and report the weight zone's
//! capacity, so the scheduler can size prefill waves against the real ceilings.
//! The engine's KV ground is adjusted between forwards, and speculative draft and
//! verify are reached through the same trait. DeepSeek's forward takes its
//! transients from the CUDA pool rather than from the wave span, and its dtypes
//! are fixed at load, so the forward accepts the activation dtype the engine was
//! built for. A rejected draft is rolled back through the verify snapshots, so
//! rejected tokens never stay absorbed in the streaming compressors.

use super::*;

impl ManagedBatchedModel for BatchedEngine {
    fn num_layers(&self) -> usize {
        self.engine.layer_count()
    }

    fn n_kv_head(&self) -> usize {
        1
    }

    fn head_dim(&self) -> usize {
        HEAD_DIM
    }

    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        let cfg = self.engine.cfg();
        ModelGeometry {
            hidden: cfg.dim,
            // Per-expert intermediate — the MoE FFN phase is priced per routed
            // expert row, exactly like the qwen3 geometry.
            intermediate: cfg.moe_inter_dim,
            n_head: cfg.n_heads,
            // Single-latent MLA: one 576-wide latent per token stands for K≡V.
            n_kv_head: 1,
            head_dim: HEAD_DIM,
            experts_per_tok: cfg.n_activated_experts.max(1),
            n_experts: cfg.n_routed_experts.max(1),
            act_dtype,
            // The int8 tensor-core kernels emit F32 before the cast back to
            // `act_dtype`; both buffers are live at once, so both are planned.
            accum_dtype: DType::F32,
            vocab: cfg.vocab_size,
            // Every layer of this stack attends through MLA, so there is no
            // second mixer chain for the phase to be the max of.
            delta_net: None,
            // The shared expert is folded into this family's routed block
            // rather than priced beside it.
            shared_expert: None,
            // This family's forward takes its transients from the CUDA pool
            // rather than the span (see `prefill_width_cap`), so nothing here is
            // carved from a wave arena and the encoding flags only have to be
            // self-consistent. Stated as the FP path, which is what it runs.
            packed_norm: false,
            packed_head: false,
            gated_qkv: false,
            fused_qkv: false,
            qkv_bias: false,
            head_qk_norm: false,
            head_norm_reshapes: false,
            partial_rotary: false,
            decode_q8_context: false,
            // No hyper-connections: one residual stream, as everything but the
            // Flash-Next lineage.
            hyper: None,
            // Its drafter is not a NextN head on the wave.
            mtp_head: false,
        }
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        // DeepSeek's forward takes its transients from the CUDA pool (it has
        // not adopted the span's wave arenas), so the default cap's FFN-span
        // pricing bounds a tier this model never allocates from — and at the
        // 8-way expert fan-out it sliced an 8-prompt fleet into three waves,
        // tripling the per-wave fixed costs (every layer's launches, and the
        // routed experts each wave fetches again) that dominate prefill. The
        // engine's pool
        // cushion is reserved at load for exactly this activation peak, so the
        // real ceilings are compute saturation and what the KV side can admit
        // — which includes the expert ground the elastic boundary would cede
        // (`reclaimable_kv_bytes` below).
        let mut cap = MAX_PREFILL_TOKENS;
        if let Some(kv_fits) = self.kv_width_cap(act_dtype) {
            cap = cap.min(kv_fits);
        }
        cap
    }

    fn reclaimable_kv_bytes(&self) -> usize {
        // The weight zone's capacity above its floor: `request_kv_ground` cedes
        // it to stuck KV claims, so a prefill wave sized against it is
        // admissible even when few KV regions stand free. Without this the
        // pure-prefill slicer sized ingest sweeps against the ~332 load-time
        // regions (~6.5k tokens) while ~39 GB of cedeable expert span sat
        // uncounted; `MAX_PREFILL_TOKENS` remains the ceiling above.
        self.engine.experts().cedeable_span_bytes()
    }

    fn maybe_change_dtype(&self, _dtype: DType) -> Result<()> {
        // DeepSeek's dtypes are baked at load: norm/compressor constants are
        // widened to F32 once (`load_compressor`), the attention kernels take
        // bf16 in and emit F32, and the int8 projections quantize from F32.
        // There is nothing to re-materialise per session dtype — the forward
        // accepts the activation dtype the engine was built for.
        Ok(())
    }

    fn device(&self) -> &Device {
        self.engine.engine_device()
    }

    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};
        // DeepSeek pins the WRITER format to the reference two-region latent
        // window: the non-RoPE dims `[0,448)` are FP8 E4M3, the 64 RoPE dims
        // `[448,512)` stay BF16 (the rope tail must not lose precision). The
        // single-latent backing splits the two regions per head automatically
        // (`alloc_block_chunks`): the nope bands take this writer format tag,
        // the rope bands are pinned BF16. The fused scatter + fresh-diagonal
        // quant dispatch on the per-band format tag. The window stores FP8 at
        // unit scale (a per-64 ue8m0 scale would need per-token record scales —
        // the window ring is written a token at a time into a per-chunk scalar
        // record, so a non-unit scale is not representable here).
        let cfg = BatchedConfig {
            k_format: KvFormat::Float(DType::F8E4M3),
            v_format: KvFormat::Float(DType::F8E4M3),
            ..config
        };
        let first = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            cfg.k_format,
            cfg.v_format,
            self.engine.engine_device(),
            cfg.initial_seq_len,
            None,
        )?;
        first.set_single_latent(true);
        let mut backings = Vec::with_capacity(self.engine.layer_count());
        backings.push(first.clone());
        for layer_idx in 1..self.engine.layer_count() {
            backings.push(first.new_layer(layer_idx, 1, cfg.initial_seq_len));
        }
        Ok(BatchedInferenceSession::new_with_backings(
            backings,
            cfg,
            self.engine.engine_device(),
        ))
    }

    fn prune(&self) -> Result<()> {
        if let Ok(mut map) = self.seq_state.write() {
            map.clear();
        }
        if let Ok(mut map) = self.verify_snap.write() {
            map.clear();
        }
        if let Ok(mut map) = self.target_feat.write() {
            map.clear();
        }
        if let Ok(mut map) = self.spec_stats.write() {
            map.clear();
        }
        Ok(())
    }

    /// Session KV truncation + streaming-corpus rollback: consume the pre-verify
    /// [`VerifySnapshot`], restore the compressor/gallery state, and replay
    /// exactly the accepted prefix — so rejected draft tokens never stay
    /// absorbed (the KV truncation alone cannot see that state).
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        session.truncate_sequence_to_tokens(seq, tokens)?;
        self.rollback_verify_state(seq, tokens)
    }

    fn expert_stats(&self) -> Option<PipelineStats> {
        Some(self.engine.experts().expert_stats())
    }

    fn expert_pipeline_dead(&self) -> bool {
        self.engine.experts().pipeline_dead()
    }

    fn expert_stats_shared(&self) -> Option<Arc<Mutex<PipelineStats>>> {
        Some(self.engine.experts().stats_shared())
    }

    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        Some(self.engine.experts().hit_counts())
    }

    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        self.engine.experts().hit_references()
    }

    fn reset_expert_stats(&self) {
        self.engine.experts().reset_expert_stats();
    }

    /// Buy `regions` of expert-zone ground for the KV side, between forwards — the
    /// scheduler's half of the boundary. See `ExpertCache::request_kv_ground`.
    fn request_kv_ground(&self, regions: usize) -> u64 {
        self.engine.experts().request_kv_ground(regions)
    }

    /// Let the expert zone take back KV ground standing free, between forwards.
    /// The wave asks at its own head; this is the scheduler asking after a
    /// compaction or a settle, which would otherwise wait for the next forward and
    /// find it buying ground instead.
    fn reclaim_spare_ground(&self) {
        self.engine.experts().reclaim_spare_ground();
    }

    /// The experts' working room, down to the pinned layers, for a paused guest.
    fn concede_for_guest(&self, regions: usize) -> u64 {
        self.engine.experts().concede_for_guest(regions)
    }

    fn stash_pinned_for_guest(&self) -> u64 {
        self.engine.experts().stash_pinned_for_guest()
    }

    fn restore_serving_floor(&self) -> bool {
        self.engine.experts().restore_serving_floor()
    }

    fn snapshot_profiles(&self) -> ProfileSnapshot {
        // Drain the expert pipeline thread's `pipe_*` spans and, in a profile
        // build, the expert GEMMs' worker counters (copy and cold-wait time) so
        // the per-phase Bulk/Single profile tables surface the MoE-internal
        // breakdown the forward thread's own spans cannot see.
        self.engine.experts().snapshot_profiles()
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

    /// Batched speculative verify: every sequence's `[committed, drafts…]`
    /// block rides ONE wave — each block as one multi-token member of the
    /// PREFILL slot (the encoding `verify_wave` documents for the qwen35
    /// lineage), while the plain cohort's committed tokens lead as ordinary
    /// decode rows. Inside the sweep the blocks are expanded back to one
    /// virtual decode slot per position — decode numerics by construction —
    /// so the two encodings meet at the same kernels; what the prefill-slot
    /// form buys is the wave driver's contract (one context per sequence, no
    /// duplicated ids) and with it the fair-wave co-batch. The wave is
    /// launch-bound, so its fixed costs (every layer's launches, expert
    /// fetches) amortize across every session instead of being paid once
    /// per session. Lossless is unchanged: the driver still accepts only the
    /// model's own argmaxes. `verify_all_rows` marks the verifying sequences
    /// for the sweep (row expansion + full-row head scoring); it is cleared by
    /// `end_verify`/`abort_verify`. Advances each sequence by its block
    /// length; the driver truncates back to the accepted lengths.
    fn begin_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        budget: usize,
    ) -> Result<Option<VerifyPlan>> {
        // Nothing here is armed speculatively: the corpus-state snapshots below
        // are taken per verifying sequence, and a zero budget leaves that cohort
        // empty, so this model already does no work it will not use.
        let _ = budget;
        if plain.is_empty() && seqs.is_empty() {
            return Ok(Some(VerifyPlan {
                decode_seqs: Vec::new(),
                decode_inputs: Vec::new(),
                verify_seqs: Vec::new(),
                verify_inputs: Vec::new(),
                rows: 0,
            }));
        }
        // Pre-verify corpus-state snapshots: the forward below absorbs the WHOLE
        // blocks (including drafts the driver may reject) into the streaming
        // compressors/galleries; the driver's `truncate_sequence` consumes each
        // snapshot to roll that state back to the accepted prefix — the KV
        // truncation alone cannot see it.
        let s_snap = span("verify:snapshot");
        let n_rows: usize = plain.len() + blocks.iter().map(|b| b.len()).sum::<usize>();
        let decode_seqs: Vec<usize> = plain.iter().map(|&(seq, _)| seq).collect();
        let mut verify_seqs: Vec<usize> = Vec::with_capacity(seqs.len());
        for (i, &seq) in seqs.iter().enumerate() {
            if blocks[i].is_empty() {
                candle::bail!("verify_blocks: empty block for seq {seq}");
            }
            let q_start = session.sequence_offset(seq).unwrap_or(0);
            self.snapshot_verify_state(seq, q_start, blocks[i].len())?;
            verify_seqs.push(seq);
        }
        // One upload per group, each row a view of it.
        let tokens: Vec<u32> = plain.iter().map(|&(_, tok)| tok).collect();
        let (decode_inputs, verify_inputs) = upload_plan_rows(&tokens, blocks)?;
        s_snap.end();
        *self
            .verify_all_rows
            .write()
            .map_err(|_| candle::Error::Msg("verify_all_rows lock poisoned".into()))? =
            seqs.to_vec();
        Ok(Some(VerifyPlan {
            decode_seqs,
            decode_inputs,
            verify_seqs,
            verify_inputs,
            rows: n_rows,
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
        self.verify_all_rows
            .write()
            .map_err(|_| candle::Error::Msg("verify_all_rows lock poisoned".into()))?
            .clear();
        let n_rows: usize = plain.len() + blocks.iter().map(|b| b.len()).sum::<usize>();
        let s_post = span("verify:post");
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        for (i, &seq) in seqs.iter().enumerate() {
            session.advance_sequence(seq, blocks[i].len())?;
        }
        // Either owned rows (the scheduler owns them before handing them over)
        // or views on the verify wave's head span, valid while the step driver
        // holds that wave — which drops every row and the wave itself before
        // the rollback, since a rollback may run a NEXT forward on partial
        // accept. Read here and split, nothing more.
        if logits.len() != n_rows {
            candle::bail!(
                "end_verify: expected {} scored rows, got {}",
                n_rows,
                logits.len()
            );
        }
        // Split the scored rows back per cohort/sequence (row order: plain
        // prefix, then each block's rows).
        let plain_out = logits[..plain.len()].to_vec();
        let mut out = Vec::with_capacity(seqs.len());
        let mut off = plain.len();
        for b in blocks {
            out.push(logits[off..off + b.len()].to_vec());
            off += b.len();
        }
        s_post.end();
        Ok((plain_out, out))
    }

    fn abort_verify(&self, seqs: &[usize]) {
        if let Ok(mut v) = self.verify_all_rows.write() {
            v.clear();
        }
        // Failed verify: drop the snapshots so a later truncate cannot replay
        // from a half-populated one.
        if let Ok(mut m) = self.verify_snap.write() {
            for &s in seqs {
                m.remove(&s);
            }
        }
    }

    /// DSpark speculative draft: propose a block of up to `max_len` tokens after `committed`,
    /// conditioned on `seq`'s stashed target feature (the concatenated `dflash.target_layers`
    /// hidden states — paper Eq. 2). Lossless — the caller verifies every proposal against the
    /// target — so the conditioning only affects acceptance, never output. Returns empty (⇒ plain
    /// decode) when no drafter is attached or the sequence has no stashed feature yet.
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if committed.len() != seqs.len() {
            candle::bail!(
                "dspark draft: {} committed tokens for {} sequences",
                committed.len(),
                seqs.len()
            );
        }
        // Per sequence, because DSpark conditions each proposal on that
        // sequence's own stashed target feature and gates it on that sequence's
        // own acceptance EMA. The qwen35 NextN head batches its cohort instead
        // (`qwen35::mtp::MtpHead::draft_cohort`), which is what removes the
        // per-session weight read the width-aware break-even below is priced
        // against; doing the same here is a separate change to a drafter with
        // its own measured schedule, not a rename.
        let cohort = seqs.len();
        seqs.iter()
            .zip(committed)
            .map(|(&seq, &tok)| self.draft_one(session, seq, tok, max_len, cohort))
            .collect()
    }
}

impl WaveSweep for BatchedEngine {
    fn device(&self) -> &Device {
        self.engine.engine_device()
    }

    fn take_device_fault(&self) -> Result<()> {
        self.engine
            .experts()
            .take_segment_fault(self.engine.engine_device())
    }

    fn num_layers(&self) -> usize {
        self.engine.layer_count()
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    /// The mHC stream: `hc_mult` copies of `dim`, in F32 — what the embedding
    /// expands into and every layer's post-mix writes.
    fn residual_row(&self, _session: &BatchedInferenceSession) -> Result<(usize, DType)> {
        let cfg = self.engine.cfg();
        Ok((cfg.hc_mult * cfg.dim, DType::F32))
    }

    /// No reconciliation: this model's session offsets are ABSOLUTE while
    /// `sequence_backing_tokens` counts the RESIDENT chunks of the sliding
    /// window ring — once the ring has slid, the default's `offset == backing`
    /// clamp would silently re-base every slid sequence. The sweep re-derives
    /// its arena lengths from the session at entry (`set_len` in resident
    /// terms after `evict_window_front`), which is this model's form of the
    /// same invariant.
    fn reconcile_entry_offsets(
        &self,
        _session: &mut BatchedInferenceSession,
        _seq_groups: [&[usize]; 3],
    ) -> Result<()> {
        Ok(())
    }

    /// The decode write-length is committed ON-DEVICE by the paged decode
    /// kernel (`commit_write_len`), and every wave entry re-commits the
    /// arena's writer prefix from the session offsets in RESIDENT terms. The
    /// default's `set_current_seq_len(offset + 1)` would stamp the ABSOLUTE
    /// offset into a ring the arena addresses relatively — a deliberate no-op,
    /// not an omission.
    fn advance_decode_rows(
        &self,
        _contexts: &mut [SequenceContext],
        _n_decode: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// KV lengths are re-committed from the session at every wave entry, so a
    /// failed wave's partial per-layer advance is corrected by the retry's own
    /// entry rather than by a truncate here — and the default's
    /// absolute-offset truncate would mis-address a slid ring. Corpus state
    /// (galleries/compressors) is rolled back through the verify snapshots
    /// where one is armed.
    fn rollback_wave(
        &self,
        _contexts: &mut [SequenceContext],
        _kv_start: usize,
        _kv_end: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// The failure-atomicity bracket around the sweep body: **a failed wave
    /// leaves no corpus trace.** The layer loop advances the galleries and
    /// streaming compressors in place as each layer runs, so an error
    /// mid-sweep — routine under the relief design — would leave the early
    /// layers' corpus one token ahead of the rest, and the retry would absorb
    /// the same positions AGAIN as duplicate, shifted groups (the re-attended
    /// duplicated-context failure [`VerifySnapshot`] documents for verify
    /// blocks, here for ANY failed wave). This also covers a FAILED verify
    /// wave, whose `abort_verify` drops the per-seq snapshots without
    /// rewinding the absorbed block. KV needs no twin: wave entry re-commits
    /// every arena length from the session.
    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        // A segment being prepared for a step: the boundary move, and nothing
        // priced — this engine keeps no buffer on the transient tier, so the
        // step places none for it. Its claims happen inside its own sweep, as
        // they do for a forward on its own.
        if matches!(wave.mode, SweepMode::Prepare) {
            self.engine.experts().reclaim_spare_ground();
            return Ok((WavePhase::Priced([0; 3]), None));
        }
        let snaps = self.capture_corpus_state(wave.seq_ids)?;
        match self.sweep_wave_inner(session, wave) {
            Ok(v) => Ok(v),
            Err(e) => {
                if let Err(restore) = self.restore_corpus_state(snaps) {
                    candle::bail!(
                        "wave failed ({e}) and the corpus-state restore that keeps that \
                         failure recoverable also failed ({restore})"
                    );
                }
                Err(e)
            }
        }
    }
}
