//! Defines the trait a scheduler drives a batched model through: wave forwards,
//! width caps, speculative decoding hooks, and the per-sequence state a model
//! keeps beside its KV.
//!
//! The wave width is the smallest of three ceilings: the compute-saturation
//! point, the rows the FFN span holds, and the rows the KV side can still admit.
//! Taking the minimum means a wave narrows under pressure instead of failing
//! partway through its claims. Each term falls back to permissive when it cannot
//! answer, since a zero-width wave makes no progress. A model with no drafter has
//! a zero draft budget and takes plain decode rows, and a model that cannot
//! rewind is refused at the speculative entry point before anything is drafted.

use super::*;

/// **You don't need to implement this trait manually.** Any type that implements
/// [`BatchedModel`] automatically gets a `ManagedBatchedModel` implementation
/// via the blanket impl.
pub trait ManagedBatchedModel {
    /// The geometry [`candle_nn::kv_cache::WavePlan`] prices a wave from.
    ///
    /// Exposed here as well as on [`BatchedModelCore`] because the scheduler
    /// holds the model behind this trait object, and admission has to price a
    /// wave's transient buffers before it decides what to admit into it.
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry;

    /// Widest prefill this model will run in one forward, in tokens.
    ///
    /// The smallest of three unrelated ceilings
    /// (`R = min(8192, transient-fits, KV-fits)`):
    ///
    /// * `MAX_PREFILL_TOKENS` — where GPU compute saturates. Above it a wider
    ///   forward costs the same per token, so slicing is free.
    /// * [`candle_nn::kv_cache::WavePlan::max_rows_within`] — how many rows the
    ///   FFN span actually holds, computed from this model's geometry. On a MoE
    ///   model the expert chain sees `rows × experts_per_tok`, so the same token
    ///   count needs several times the span a dense model would.
    /// * **What the KV side can still hold.** The admit phase claims every chunk
    ///   the wave will write before it computes anything, so a wave admitted
    ///   wider than the free regions can back is one that fails partway through
    ///   claiming — with some layers extended and some not. Capping here turns
    ///   that into a narrower wave, which is the outcome the whole partition is
    ///   for: under pressure the engine slows down instead of failing.
    ///
    /// Taking the min is what stops a wave being sized by a constant that has
    /// never seen the model. Each term falls back to permissive when it cannot
    /// answer — no reservation yet, or a plan that cannot price a single row —
    /// because a zero-width wave makes no progress, and refusing here would abort
    /// a forward that can still run.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        let mut cap = MAX_PREFILL_TOKENS;
        // Priced from an empty wave: this cap is the model's own bound, asked
        // before any wave is composed, so there is no head to widen from.
        let fits = WavePlan::new(self.wave_geometry(act_dtype))
            .max_rows_within(WAVE_FFN_BYTES, WaveWidth::default());
        if fits > 0 {
            cap = cap.min(fits);
        }
        if let Some(kv_fits) = self.kv_width_cap(act_dtype) {
            cap = cap.min(kv_fits);
        }
        cap
    }

    /// [`Self::prefill_width_cap`] for a cohort whose sweep outlasts the
    /// ground it is priced against: `withheld` bytes of that ground will be
    /// claimed by others' KV before the sweep's last window places its tier,
    /// so the width is sized against what is left.
    ///
    /// A creep cohort's width is fixed when it forms and its tier is placed
    /// again at every window, minutes later, on ground nothing holds for it in
    /// between. Measured on the RTX 3090 (Flash-Next, 2026-10-18): an 8,192-row
    /// cohort formed against ~10 GB of ground; decode KV grew 3.6 → 7.2 GB over
    /// its two-minute sweep and drove the weight side to its floor, and window
    /// 23 found 453 regions for a 454-region tier.
    ///
    /// The default ignores `withheld`: a model whose cap is not sized from the
    /// elastic ground has nothing to subtract it from.
    fn prefill_width_cap_withholding(&self, act_dtype: DType, withheld: usize) -> usize {
        let _ = withheld;
        self.prefill_width_cap(act_dtype)
    }

    /// The wave transient tier a prefill of `rows` rows across `sequences`
    /// would need, in bytes, with its attention launch carving for chunks of
    /// `stage_q_lens` rows at `stage_offsets` ([`WaveWidth::kv_stage`]).
    ///
    /// **The same function the tier is actually placed from**, not an estimate
    /// of it: admission judges an offer on the residency it dislodges, and the
    /// tier dislodges weights exactly as a region claim does. Pricing it any
    /// other way lets the two figures drift, and the one that drifts is the one
    /// the placement then refuses. The launch's carve is the one buffer priced
    /// by depth rather than width — until a sequence passes one key chunk, past
    /// which it is the same ~153 MB a launch on Flash-Next at any depth — so
    /// the chunks and their depths come from the caller, who knows them.
    ///
    /// The tier is superlinear in *spans*, not only in rows — the mixer's span
    /// tables hold an entry per span and the prefill scan's transients turn on
    /// with the first — so `sequences` is not decoration, and a caller that
    /// prices N admissions as N separate one-sequence waves understates the wave
    /// they compose.
    // One wave's description: its rows and sequences, then each stage's length,
    // depth and sequence, and the decodes beside it.
    #[allow(clippy::too_many_arguments)]
    fn wave_tier_bytes(
        &self,
        rows: usize,
        sequences: usize,
        stage_q_lens: &[usize],
        stage_offsets: &[usize],
        _stage_seqs: &[usize],
        decode_stage: &[(usize, usize, usize)],
        act_dtype: DType,
    ) -> Option<u64> {
        let geometry = self.wave_geometry(act_dtype);
        let prefill = WaveWidth::prefill(rows, sequences.max(1));
        let width = WaveWidth {
            kv_stage: PrefillKvStageLayout::new(
                stage_q_lens,
                stage_offsets,
                geometry.n_head,
                geometry.n_kv_head,
                geometry.head_dim,
                PrefillLaunchBounds::PRODUCTION,
            ),
            decode_rows: decode_stage.len(),
            scored_rows: prefill.scored_rows
                + decode_stage
                    .iter()
                    .map(|&(_, _, block)| block.max(1))
                    .sum::<usize>(),
            ..prefill
        };
        let scratch = launch_scratch_reserve(self.device());
        Some(WavePlan::new(geometry).tier_bytes(width, scratch) as u64)
    }

    /// Rows the KV side has room to admit, or `None` when it cannot say.
    ///
    /// Every token writes one K and one V element per KV head per layer, so the
    /// per-row cost is fixed by geometry and the only variable is how many free
    /// regions remain. Deliberately measured against `free + blocked` rather
    /// than against the elastic floor: the boundary moves at the expert
    /// pipeline's end of pass, so ground the *weight side* is holding is not
    /// available to this wave however willing it might be to give it up later —
    /// but ground the previous forward's transient tier has blocked is. This
    /// cap is read at the top of `forward_wave`, when that tier is still
    /// standing; the forward it sizes releases the tier in its own phase 0,
    /// before any claim priced here runs. Counting only `free` made every wide
    /// prefill's own tier throttle the next one: the tier's ground read as
    /// gone, the cap collapsed to a few hundred tokens, and a ten-context
    /// prefill fragmented into a dozen narrow waves that each re-loaded the
    /// expert working set — bulk prefill fell from ~2.1K t/s to ~700.
    ///
    /// One region of margin, because a sequence's chunks do not pack a region
    /// exactly and the last one is partly wasted.
    fn kv_width_cap(&self, act_dtype: DType) -> Option<usize> {
        let stats = candle_nn::kv_cache::region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_sub(1);
        let per_row_all_layers = self.kv_bytes_per_row(act_dtype)?;
        // Admissible KV = what stands free PLUS what the model's elastic
        // boundary would cede to a stuck claim ([`Self::reclaimable_kv_bytes`]).
        // Counting only the free regions under-reports capacity by whatever the
        // weight side happens to be holding above its floor (tens of GB on the
        // streamed-expert engine) and pre-slices pure-prefill sweeps against a
        // number the first stuck claim would have doubled. The result is still
        // bounded above by `MAX_PREFILL_TOKENS` in the callers.
        let kv_bytes = free
            .saturating_mul(candle_nn::kv_cache::REGION_BYTES)
            .saturating_add(self.reclaimable_kv_bytes());
        let rows = kv_bytes / per_row_all_layers;
        // **Never zero.** A width cap of nought is not a narrow wave, it is no
        // wave — and once the KV side is full it would be permanent: no forward
        // runs, so nothing completes, so nothing is freed, so the cap stays at
        // nought. The partition's answer to genuine exhaustion is a refused
        // claim and the relief pass behind it, both of which need a forward to
        // have been attempted.
        Some(rows.max(1))
    }

    /// Bytes a forward's tier and the KV it writes can be paid from: the KV
    /// side's free regions — counting those the standing tier blocks, which the
    /// next forward releases first — plus what the weight side would concede
    /// ([`Self::reclaimable_kv_bytes`]). `None` where there is no reservation
    /// to measure.
    ///
    /// What a caller composing a prefill checks [`Self::wave_tier_bytes`] of
    /// the *actual* wave against, together with its KV. The width cap cannot
    /// answer that: it prices a one-sequence wave, and the tier grows with
    /// sequences as well as rows.
    fn prefill_ground_bytes(&self) -> Option<usize> {
        let stats = region_stats(0)?;
        Some(
            (stats.free + stats.blocked)
                .saturating_mul(REGION_BYTES)
                .saturating_add(self.reclaimable_kv_bytes()),
        )
    }

    /// The most rows one forward can carry whatever the ground — a kernel's own
    /// limit, not a price — or `None` where nothing bounds it. A caller that
    /// sizes a forward against the ground alone (a sliced prefill) stops here
    /// too.
    fn forward_row_ceiling(&self) -> Option<usize> {
        None
    }

    /// KV bytes one prefill row writes: one K and one V element per KV head on
    /// every layer that pages KV — every layer, unless a model says otherwise.
    /// `None` when the geometry prices a row at nothing.
    fn kv_bytes_per_row(&self, act_dtype: DType) -> Option<usize> {
        let per_row = 2 * self.n_kv_head() * self.head_dim() * act_dtype.size_in_bytes();
        per_row
            .checked_mul(self.num_layers())
            .filter(|&bytes| bytes > 0)
    }

    /// KV-side bytes this model's memory layout could free ON DEMAND beyond
    /// what stands free — e.g. an elastic weight/KV boundary that cedes expert
    /// ground to a stuck KV claim. Counted by [`Self::kv_width_cap`] when
    /// sizing a prefill wave, so the wave is sliced against what the partition
    /// CAN admit, not what it happens to have standing free. `0` for models
    /// with a static layout (nothing to cede).
    fn reclaimable_kv_bytes(&self) -> usize {
        0
    }

    /// Re-materialise every norm weight in the activation dtype — see
    /// [`crate::models::batched_model::BatchedModelCore::maybe_change_dtype`].
    ///
    /// On this trait too because the scheduler holds the model behind it, and
    /// the activation dtype is chosen where the KV cache is configured rather
    /// than at load.
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()>;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;
    /// Number of KV heads per layer.
    fn n_kv_head(&self) -> usize;
    /// Dimension of each attention head.
    fn head_dim(&self) -> usize;
    /// Device the model is on.
    fn device(&self) -> &Device;

    /// Snapshot of all static model properties.
    ///
    /// Cheap to copy; use this to capture model metadata before the model is moved to
    /// another thread.  The default derives `provenance_layer_indices` from `num_layers()`
    /// and sets threshold factors to `1.0`.  The blanket impl for `BatchedInference<M>`
    /// overrides this to read per-model threshold factors from the inner `BatchedModelCore`.
    fn model_core_properties(&self) -> ModelCoreProperties {
        self.default_core_properties()
    }

    /// The uniform-transformer answer, so an override can adjust ONE field
    /// instead of restating all eleven.
    ///
    /// A model that differs in one respect — a hybrid whose provenance lives in
    /// its attention layers only, say — should not have to copy the threshold
    /// factors and layer indices to say so, because a copy is a second place
    /// for them to drift.
    fn default_core_properties(&self) -> ModelCoreProperties {
        let n = self.num_layers();
        let provenance_layer_indices = if n == 0 {
            ProvenanceLayerIndices {
                syn_l0: 0,
                syn_l4: 0,
                sem_l0: 0,
                sem_l4: 0,
                prag_l0: 0,
                prag_l4: 0,
            }
        } else {
            let syn = (n * 15 / 100).max(1);
            let sem = n / 2;
            let prag = (n * 85 / 100).min(n - 1);
            ProvenanceLayerIndices {
                syn_l0: syn.saturating_sub(4),
                syn_l4: syn,
                sem_l0: sem.saturating_sub(4),
                sem_l4: sem,
                prag_l0: prag.saturating_sub(4),
                prag_l4: prag,
            }
        };
        ModelCoreProperties {
            num_layers: n,
            n_kv_heads: self.n_kv_head(),
            head_dim: self.head_dim(),
            provenance_layer_indices,
            k_hi_error_threshold_factor: 1.0,
            k_low_error_threshold_factor: 1.0,
            v_hi_error_threshold_factor: 1.0,
            v_low_error_threshold_factor: 1.0,
            // Attention computes each token's output from what precedes it
            // positionally, so a hole in the middle of a sequence is fillable.
            // The default is the uniform-transformer answer; a model whose
            // per-sequence memory is a recurrence overrides it.
            // Uniform transformer: every layer attends, so every layer has a Q.
            provenance_capture_layers: self.num_layers(),
            can_gap_fill: !self.carries_recurrent_state(),
            carries_recurrent_state: self.carries_recurrent_state(),
        }
    }

    /// Co-batched continuous-fair-wave forward: run decode (q=1) + prefill (q=N) +
    /// glue (q=G) rows through ONE forward (layer range `[layer_start, layer_end)`)
    /// with the mixed attention dispatch and the single shared FFN/MoE
    /// (`docs/continuous_fair_waves.md`). Glue rows are staged on the session via
    /// `set_pending_glue` before the call, as for a plain glue forward.
    ///
    /// Returns a [`WaveStep`] carrying `logits` for the **decode + prefill** rows
    /// only (in `decode_seqs ++ prefill_seqs` order — glue rows carry no
    /// logits) when the range reaches the head, and none when it stops short.
    ///
    /// A range starting past layer 0 resumes from `residual`'s committed rows; a
    /// range stopping short of the head writes its residual there and commits it
    /// (`docs/wave_step.md` §3.2). [`Self::claim_residual`] sizes one.
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
    ) -> Result<WaveResult>;

    /// One co-batched step: rows that cross every layer, and a creep that
    /// crosses its window, in one forward span with one tier
    /// (`docs/wave_step.md` §2).
    fn forward_step(
        &self,
        session: &mut BatchedInferenceSession,
        spec: StepSpec<'_>,
    ) -> Result<StepOutcome>;

    /// Room for `rows` rows of this model's residual stream, held between
    /// forwards — for a sweep paused and resumed by layer range. Between
    /// forwards, as every span claim is.
    fn claim_residual(
        &self,
        session: &BatchedInferenceSession,
        rows: usize,
    ) -> Result<CreepResidual>;

    // ── Speculative decoding (model-agnostic hook) ──────────────────────────────
    //
    // Three composable methods let ANY model do lossless speculative decoding through the
    // generic `speculative_decode_step` driver. A model with no drafter inherits the defaults
    // and `speculative_decode_step` degrades to a single plain decode — so the hook is always
    // safe to call. A model with a drafter overrides `speculative_draft` (and, for the actual
    // speedup, `verify_block`). The accepted tokens are always drawn by the caller's own
    // `TokenChooser` from this model's own logits, so the driver adds no departure of its own:
    // with an exact chooser the output is distributed exactly as plain decoding through it would
    // be, regardless of draft quality — bit-identical under `GreedyChooser`. A chooser that
    // applies typical acceptance (the scheduler's) departs only where that rule accepts a draft.

    /// Tokens this model wants each sequence to draft on a wave of `width` sequences.
    ///
    /// **Zero means take a plain decode row**, and that is the default: a model with no drafter
    /// never speculates, and one whose ladder has not been measured does not guess.
    ///
    /// The budget belongs to the model because the trade does. A verify block scores `k + 1` rows
    /// where a plain decode scores one, and writes `k + 1` KV entries where a plain decode writes
    /// one. Both are nearly free while the wave is memory-bandwidth-bound — a narrow decode reads
    /// the entire weight set to score a handful of rows — and both cost what they weigh once it is
    /// compute-bound. Where that turns over depends on the checkpoint's shape: how much weight a
    /// step reads, whether experts stream, how wide the KV rows are. It is measured per model, by
    /// the `speculative_decode_*` gates' width sweep, and recorded on the model.
    ///
    /// Called once per wave, so it must be cheap and must not touch the device.
    fn draft_budget(&self, width: usize) -> usize {
        let _ = width;
        0
    }

    /// What one drafted token costs relative to a step's fixed cost — the price
    /// [`super::draft_depth::DraftDepth::budget`] weighs a depth's expected
    /// tokens against. Carried on the model's ladder beside its ceilings
    /// ([`super::draft_ladder::DraftLadder::token_cost`]).
    ///
    /// Read only when [`Self::draft_budget`] is non-zero. The default prices a
    /// drafted token at nothing, so a drafting model that names no cost drafts
    /// its ceiling — the ladder's own figure.
    fn draft_token_cost(&self) -> f32 {
        0.0
    }

    /// Blocks past the trunk that a sweep reaching its last layer also runs, over
    /// every row it carries: a draft head with a KV layer of its own runs in the
    /// same wave over the same rows, so its history stays as long as the sequence
    /// it drafts for. Zero for a model without one.
    ///
    /// The creep slicer counts them as layers of the sweep (`creep_slice`): a
    /// window that reaches the last trunk layer pays for them too, so the slice
    /// that does must have budgeted them.
    ///
    /// Called once per wave, so it must be cheap and must not touch the device.
    fn sweep_head_layers(&self) -> usize {
        0
    }

    /// Draft up to `max_len` speculative next-tokens for **every** sequence in the step's
    /// cohort, each following its own `committed` token, using the model's own drafter (e.g.
    /// an MTP / DSpark head). Proposals only — the caller verifies them and keeps the
    /// converging prefix. Returns one proposal list per entry of `seqs`, in order; an empty
    /// list means that sequence takes a plain decode row instead of a verify block.
    ///
    /// **Takes the whole cohort, not one sequence, because a drafter is weight-bound.** The
    /// recurrence is serial *within* a sequence — step `j` needs `embed(argmax)` of step
    /// `j-1` — but step `j` of one sequence is independent of step `j` of every other, so the
    /// cohort's `j`-th steps are one batched pass. That matters because the dominant cost is
    /// reading weights, not arithmetic: on the 9B a drafted token spends 1.5 ms of 2.7 ms in
    /// one full read of the ~795 MiB output projection to score a single row, and four rows
    /// measured 1.69 ms against one row's 1.66. Drafting per sequence therefore reads the same
    /// weight once per session per step; drafting per cohort reads it once. That is also why
    /// a drafter's break-even used to rise with width — the answer is to batch the drafter,
    /// not to stop drafting at width.
    ///
    /// **Default: no drafter → no proposals**, and the step below degrades to one plain decode.
    /// A model drafts by overriding this with a head its checkpoint actually carries — the
    /// `qwen35` lineage's NextN/MTP block (`qwen35::mtp`), DeepSeek's DSpark. There is
    /// deliberately no generic fallback proposer: one that guesses from the sequence's own text
    /// can only re-propose what the sequence already said, which is worth nothing on the
    /// reasoning and first-draft tokens a decode loop actually spends its time on, and it costs
    /// a per-sequence token history that grows with context to buy it.
    fn speculative_draft(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let _ = (session, committed, max_len);
        Ok(vec![Vec::new(); seqs.len()])
    }

    /// Verify a block of `tokens` for `seq`: append them and return one next-token logits row
    /// `[1, vocab]` per input token (the model's prediction after each prefix). Default: run the
    /// tokens as sequential `forward_wave` decode steps — correct for ANY model, but no speedup
    /// (this is what makes speculative decode *lossless* by default). Models override with a
    /// single batched forward over the whole block for the throughput win. Advances the sequence
    /// by `tokens.len()`; the driver truncates back to the accepted length.
    ///
    /// **A model that overrides [`Self::speculative_draft`] but not this should not be driven
    /// speculatively.** The default spends one forward per block position to buy at most that
    /// many tokens, so a drafted step is break-even at best and a loss whenever a proposal is
    /// rejected. It stays correct so a new drafter can be brought up against it and then made
    /// fast; it is not a throughput path.
    fn verify_block(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: &[u32],
        layer_end: usize,
    ) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            let t = Tensor::from_vec(vec![tok], (1, 1), self.device())?;
            let step = self.forward_wave(
                session,
                &[seq],
                std::slice::from_ref(&t),
                &[],
                &[],
                &[],
                &[],
                0,
                layer_end,
                None,
            )?;
            // Copied off the span (`logits_owned`): the rows are accumulated
            // across this loop's forwards, so each iteration's span-backed
            // views would be invalidated by the next forward.
            let logits = step.logits_owned()?.pop().ok_or_else(|| {
                candle::Error::msg("verify_block: forward_wave produced no logits")
            })?;
            session.advance_sequence(seq, 1)?;
            out.push(logits);
        }
        Ok(out)
    }

    /// Whether this model carries per-sequence state that **cannot be rewound**.
    ///
    /// A recurrent state is an accumulated sum with no per-token decomposition:
    /// there is no suffix to remove and no inverse to apply. Rewinding the K/V
    /// under one is silent corruption — the model answers from tokens the cache
    /// no longer holds, fluently and without an error.
    ///
    /// This is a real model property, like `head_dim`, not a feature flag. It
    /// gates the one operation in the engine that means "put this sequence back
    /// the way it was `n` tokens ago" ([`Self::truncate_sequence`], reached
    /// only from speculative decode), refusing it up front rather than letting
    /// it corrupt state a layer down. `false` for every model whose entire
    /// per-sequence memory is the paged K/V.
    ///
    /// It also gates the snapshot / fork / branch-checkpoint machinery, so what
    /// qualifies matters. **Per-sequence state outside the K/V is not enough —
    /// the state must be irrecoverable.** `latent_moe`'s engine (DeepSeek-V4)
    /// carries a per-sequence compressor and provenance gallery, and answers
    /// `false` deliberately: both are *derived*, rebuildable from the corpus
    /// that is already durable, so there is nothing to snapshot that a replay
    /// could not reproduce. A delta-rule matrix has no such source — it is the
    /// only record of the tokens that built it — which is why the hybrid is the
    /// one lineage that answers `true`.
    ///
    /// The consequence of getting this wrong is asymmetric. A model that
    /// answers `true` without needing to pays ~63 MiB of export per seal for
    /// bytes nothing reads. One that answers `false` when it should not gets
    /// silently zeroed state and reads fluently, which is the failure this whole
    /// path exists to remove.
    fn carries_recurrent_state(&self) -> bool {
        false
    }

    /// Whether `seq` holds its recurrent store right now — forked from a
    /// parent, restored, or built by a wave it already rode.
    ///
    /// What admission reads to charge a turn's store once: a view carved before
    /// its prefill was admitted already holds one, and its regions are in the
    /// residency admission measures, so charging the store again prices the same
    /// ground twice. `false` on a stack with no recurrent layers, which holds
    /// none.
    fn has_recurrent_state(&self, _seq: usize) -> bool {
        false
    }

    /// The prefill scratch a prefill of `seq` would claim on top of the store it
    /// already holds — one more slot per recurrent layer, taken at the prefill's
    /// first window and held until it decodes. Zero when it holds the scratch
    /// already, or holds no store (whose cost [`Self::recurrent_store_bytes`]
    /// prices whole, scratch included), or the stack carries no recurrent state.
    ///
    /// Priced because a sequence whose store survived its last turn still claims
    /// this when it prefills again, and neither admission nor the cohort fit
    /// charged it: on WhiteWorld (2026-10-19) a cohort with no store to build
    /// grew the recurrent tenant by 18 regions over its sweep, and its final
    /// window was refused.
    fn prefill_scratch_bytes(&self, _seq: usize) -> usize {
        0
    }

    /// The prefill scratch any one sequence claims for a span longer than
    /// [`Self::short_span_rows`], priced from the geometry: part of
    /// [`Self::recurrent_store_bytes`], and the part a turn that rides only in
    /// short spans never claims. Zero on a stack with no recurrent state.
    fn sequence_scratch_bytes(&self) -> usize {
        0
    }

    /// The longest span this stack writes without claiming its sequence's
    /// prefill scratch: the recurrent timeline's rows, on a stack that carries
    /// recurrent state. A longer span writes a second copy of the whole state
    /// ([`Self::prefill_scratch_bytes`]). `None` on a stack with none, where a
    /// span of any length claims nothing beyond its K/V.
    ///
    /// What a stencil's static runs are cut to: a decode plays them as rows of
    /// its step's wave, and a row within this claims nothing a cohort or
    /// admission has to set aside (`docs/stencil_tree.md`).
    fn short_span_rows(&self) -> Option<usize> {
        None
    }

    /// The selection index pages a prefill of `rows` rows from `offset` opens:
    /// zero for a model with no sparse selection. Its live tail grows a page at
    /// a time, so what a prefill claims is the pages past those its offset
    /// already fills, on every layer that selects.
    fn index_growth_bytes(&self, _offset: usize, _rows: usize) -> usize {
        0
    }

    /// Recurrent stores held on the device, and sequences whose state is parked
    /// in their record instead — `(held, parked)`. What says whether idle
    /// conversations give their stores back. `(0, 0)` on a stack with none.
    fn recurrent_census(&self) -> (usize, usize) {
        (0, 0)
    }

    /// The sequences holding a recurrent store on the device, sorted — the
    /// `held` of [`Self::recurrent_census`], by sequence: what names a store
    /// whose sequence the scheduler no longer accounts for. Empty on a stack
    /// with none.
    fn recurrent_holders(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Claim `seq`'s carried per-sequence state now — whatever of it the
    /// sequence does not already hold — rather than at its first wave.
    ///
    /// Admission calls this the moment it takes a turn, so ground the turn was
    /// priced for is claimed between forwards, where the weight side can sell
    /// for it, and a refusal leaves the turn queued instead of failing a wave's
    /// whole cohort at entry. Nothing to claim on a stack with no carried state.
    fn claim_carried_state(&self, _seq: usize) -> Result<()> {
        Ok(())
    }

    /// Whether a speculative block's rejected tail can be rolled back on this model.
    ///
    /// **Distinct from [`Self::carries_recurrent_state`], which used to answer both.**
    /// Carrying recurrent state means there is something outside the session's KV to
    /// snapshot; for as long as no model could rewind such a state, one declaration
    /// served for both questions and the speculative entry point refused on it. The
    /// hybrid's timeline separates them. It still carries state — nothing else
    /// records the tokens that built `S` — and it can rewind *inside a block it
    /// just verified*: the block never wrote `S`, only its timeline, and the fold
    /// applies the accepted prefix of it (`qwen35::spec`).
    ///
    /// Default: a model carrying recurrent state cannot rewind, which keeps every
    /// model that has not implemented a rewind refused.
    ///
    /// Overriding to `true` is a claim about [`Self::truncate_sequences`], not about
    /// the state: that it restores the recurrence exactly for the targets it accepts.
    /// It is not a claim that every offset is rewindable — a fold covers the block
    /// the verify wave wrote and nothing else, so `truncate_sequences` is what
    /// refuses a target with no rewind point, at the one place that knows.
    fn can_rewind_speculative_block(&self) -> bool {
        !self.carries_recurrent_state()
    }

    /// The thresholds a sampling chooser accepts a draft on when its sample
    /// did not land on it — see [`TypicalAcceptance`].
    ///
    /// Default: the Medusa reference values. A model overrides this only when
    /// its own drafter's acceptance has been measured against other values;
    /// greedy verification never reads it, so the forward gates do not see it.
    fn typical_acceptance(&self) -> TypicalAcceptance {
        TypicalAcceptance::MEDUSA
    }

    /// Roll `seq` back to exactly `tokens` tokens after a speculative verify — called by the
    /// driver with `pos + kept` once the accepted prefix is known. Default: the session-level KV
    /// truncation. A model whose forward maintains per-sequence streaming state OUTSIDE the
    /// session's KV (compressors, corpus galleries) overrides this to roll that state back in the
    /// same call — otherwise a partial accept leaves rejected draft tokens absorbed in state the
    /// truncation can't see, and later attention reads corrupted context.
    ///
    /// **Reached only by the speculative verify path**, despite the broad name.
    /// A model that cannot express a rewind at all leaves
    /// [`Self::can_rewind_speculative_block`] at its default `false` and is
    /// refused at the entry point, so it never arrives here — the guard is the
    /// declaration, not a bail inside an override raised after the driver has
    /// committed. A model that *can* rewind, but only within a block it just
    /// verified, accepts here and refuses the uncovered offsets itself.
    fn truncate_sequence(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        tokens: usize,
    ) -> Result<()> {
        // Already there: nothing to roll back. This is the common case, not a
        // corner one — every decode step ends by reconciling each sequence to
        // what it kept, and a step that kept everything it wrote (any plain
        // decode row, and a fully accepted block) reconciles to the offset the
        // sequence already stands at. Without this the KV backing would seal,
        // release and re-claim chunks around the live tail once per sequence per
        // step, against the compressor and the sealing thread.
        if session.sequence_offset(seq) == Some(tokens) {
            return Ok(());
        }
        session.truncate_sequence_to_tokens(seq, tokens)
    }

    /// The whole step's rollbacks in one call — `(seq, tokens)` per sequence.
    /// Default: [`Self::truncate_sequence`] per target, which is correct for
    /// any model. A model whose rollback does real per-sequence work overrides
    /// this to batch it — the qwen35 lineage's recurrent fold runs one launch
    /// for the whole cohort and every DeltaNet layer here, against one per
    /// sequence through the per-target default.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        for &(seq, tokens) in targets {
            self.truncate_sequence(session, seq, tokens)?;
        }
        Ok(())
    }

    /// A sequence id is going away: release whatever per-sequence state this
    /// model keeps OUTSIDE the session's KV.
    ///
    /// The session owns the KV and frees it with the slot. A model whose
    /// forward carries per-sequence state of its own — a recurrent store, a
    /// streaming compressor, a corpus gallery — owns that separately and
    /// outlives every session, so nothing else can free it.
    ///
    /// Slot ids are pool indices and **are recycled**, which is what makes this
    /// a correctness hook rather than a leak fix: a recycled id whose first
    /// wave carries a non-zero offset finds the previous conversation's entry
    /// still sitting in the map and inherits it. The model then answers from a
    /// history the cache never had, fluently and without an error.
    fn release_sequence(&self, _seq: usize) -> Result<()> {
        Ok(())
    }

    /// `seq`'s first `tokens` positions hold the conversation's system prompt.
    ///
    /// For a model whose attention selects a subset of the context, the prompt
    /// is a span its selection can treat apart from the rest — a stratified QSA
    /// selection ranks it in every window (`docs/qsa_stratified_selection.md`).
    /// A model that attends every position has nothing to do with it.
    fn set_selection_prompt(&self, _seq: usize, _tokens: usize) -> Result<()> {
        Ok(())
    }

    /// `seq`'s pinned position ranges: spans of its system prompt every
    /// selecting query attends whole, outside the selection's budget
    /// (`docs/hit_driven_projection.md` §5). Replaces any declared before.
    /// A model that attends every position has nothing to pin.
    fn set_selection_pins(&self, _seq: usize, _ranges: &[Range<usize>]) -> Result<()> {
        Ok(())
    }

    /// Which selecting layers keep index rows for unloaded pieces and count
    /// hits ([`HitPercentiles`]). A model that attends every position has no
    /// selection to count.
    fn set_hit_layers(&mut self, _layers: &HitPercentiles) -> Result<()> {
        Ok(())
    }

    /// Declare the rounds `seq`'s projection placed, as position ranges in
    /// declaration order, so its sampled layers count what their selections
    /// choose of each (`docs/hit_driven_projection.md` §6.4), and after them
    /// the `pooled` rounds it did not place, which the shadow scan counts
    /// (§6.3). Replaces any declared before, dropping their counts.
    fn set_hit_rounds(
        &self,
        _seq: usize,
        _rounds: &[Range<usize>],
        _pooled: &[PooledRound],
    ) -> Result<()> {
        Ok(())
    }

    /// What `seq`'s sampled layers chose of each declared piece since the
    /// last read, as far as the device has published it — read without
    /// waiting on the device; `None` for a model with no selection to
    /// count, or a sequence with nothing declared.
    fn take_hits(&self, _seq: usize) -> Result<Option<PieceHits>> {
        Ok(None)
    }

    /// How the model's selection divides a query's candidates before ranking
    /// them, stated in positions ([`StrataTokens`]).
    ///
    /// A model whose attention selects turns it into its own blocks and checks
    /// it against its selection kernel, refusing one the kernel could not run
    /// at the deepest position the model reaches. A model that attends every
    /// position has no candidates to divide.
    fn set_selection_strata(&mut self, _strata: StrataTokens) -> Result<()> {
        Ok(())
    }

    /// How many sequences this model currently holds recurrent memory for.
    ///
    /// The leak gauge for [`Self::release_sequence`]. Slot ids are recycled pool
    /// indices, so memory that outlives its sequence is not merely wasted VRAM:
    /// the next conversation to land on that id inherits it. `0` for a model
    /// that carries none.
    fn recurrent_memory_count(&self) -> usize {
        0
    }

    /// `child` begins as a copy of `parent`'s per-sequence model state.
    ///
    /// The KV analogue is `create_view_sequence`, which lets a child slot
    /// borrow the parent's blocks zero-copy. State cannot be borrowed — the
    /// child is about to advance it — so it is copied, device-to-device.
    ///
    /// Called at every turn's view carve, not only on an explicit user fork:
    /// the turn loop decodes on a child slot, and a child that starts from
    /// nothing is a model running on whatever fraction of its stack is not
    /// recurrent.
    fn fork_recurrent(&self, _parent: usize, _child: usize) -> Result<()> {
        Ok(())
    }

    /// Read `seq`'s recurrent state back for the turn-seal snapshot record.
    ///
    /// Returns `(schedule_hash, layers)`, or `None` for a model that carries no
    /// such state — so the seal site is model-agnostic and a non-recurrent model
    /// pays nothing.
    ///
    /// **The layer rows, not the record.** `candle-conversation` depends on this
    /// crate and not the reverse, so `SnapshotPayload` cannot be named here; the
    /// scheduler assembles it from these rows, which are field-for-field its
    /// `SnapshotLayer`. Moving the record type down into the model crate would
    /// invert the dependency to save one struct literal.
    ///
    /// Refuses mid-wave — a snapshot must capture a sealed boundary, never a
    /// wave in flight. The seal runs outside the wave, so this is an assertion
    /// on that ordering rather than a case to handle.
    fn export_recurrent(&self, _seq: usize) -> Result<Option<(u64, Vec<ExportedLayerState>)>> {
        Ok(None)
    }

    /// Arm `seq`'s span of `len` tokens at absolute position `at` for a capture
    /// of its recurrent scan inputs, or for a replay that takes the span's
    /// advanced recurrent state from a record's rows
    /// (`docs/deltanet_timeline.md` §14). The wave that runs exactly that
    /// span honours it.
    ///
    /// `false` for a model with no recurrent layers, where a span has no such
    /// state to capture or replace.
    fn arm_recurrent_span(
        &self,
        _seq: usize,
        _at: usize,
        _len: usize,
        _kind: ArmKind,
    ) -> Result<bool> {
        Ok(false)
    }

    /// The rows a capture armed on `seq` took, once every recurrent layer has
    /// run its span — `None` before then, or with nothing armed.
    fn take_recurrent_capture(&self, _seq: usize) -> Result<Option<ReplyRows>> {
        Ok(None)
    }

    /// Scatter a snapshot back into `seq`'s recurrent state — the resume path.
    ///
    /// Returns `true` when the state was installed, `false` when this model
    /// carries none. **Errors** on a hash or geometry mismatch rather than
    /// silently starting from zero: falling back to zeros is the same fluent
    /// amnesia this whole path exists to remove, so the caller has to see it
    /// and log it distinguishably.
    fn restore_recurrent(
        &self,
        _seq: usize,
        _schedule_hash: u64,
        _layers: &[ExportedLayerState],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Hold a snapshot for `seq` and install it when the sequence's state is
    /// first wanted — a view forked from it, its admission's claim, or a wave
    /// run on it — rather than now.
    ///
    /// The branch-checkpoint install lands on a slot that has not run a turn,
    /// and may wait a long time before it does. Restored there, every new
    /// conversation would hold a whole recurrent store outside admission, and a
    /// window of them would take the span from under the turns admission had
    /// already priced. Held back, the store is claimed where admission prices it.
    ///
    /// Checked now, as [`Self::restore_recurrent`] checks: a hash or geometry
    /// mismatch errors here, not at first use. The default installs at once,
    /// which is what a model that keeps no store of its own does anyway.
    fn stage_recurrent(
        &self,
        seq: usize,
        schedule_hash: u64,
        layers: Arc<[ExportedLayerState]>,
    ) -> Result<bool> {
        self.restore_recurrent(seq, schedule_hash, &layers)
    }

    /// Drop `seq`'s recurrent store: its last seal's snapshot is durable and
    /// the sequence is out of every forward, so its state is the record's and
    /// the store's device ground is idle. Until it is restored — staged from
    /// the record ([`Self::stage_recurrent`]) — any use of its state is refused
    /// with [`RECURRENT_PARKED_MARKER`], never met with a sequence-start store.
    ///
    /// `false` when nothing was dropped: no store, or one a wave still has
    /// open. The default keeps everything, as a model without stores must.
    fn park_recurrent(&self, _seq: usize) -> Result<bool> {
        Ok(false)
    }

    /// Whether `seq`'s state was parked ([`Self::park_recurrent`]) and must be
    /// restored from its record before anything reads it.
    fn recurrent_parked(&self, _seq: usize) -> bool {
        false
    }

    /// Carried per-sequence state that is **not** a DeltaNet layer stack, as an
    /// opaque byte string the persistence layer stores verbatim.
    ///
    /// [`Self::export_recurrent`]'s rows are shaped for one thing — a per-layer
    /// `(S, conv_tail)` pair — because that is what the hybrid lineage carries.
    /// An architecture with a differently-shaped state has two bad options and
    /// one good one: widen `ExportedLayerState` with fields most models leave
    /// empty, invent a second record type the scheduler has to learn, or hand
    /// the scheduler bytes it never interprets. This is the third.
    ///
    /// The model owns the encoding **and its versioning**. Nothing above this
    /// line may parse the blob, so a model may change its layout freely as long
    /// as [`Self::restore_aux_state`] still reads what it once wrote — the same
    /// contract the record format keeps for its own wire versions.
    ///
    /// `None` for a model with no such state, which is every model but
    /// Qwen3.8-Flash-Next today (its PLE convolution window and hash history).
    /// Called at the same seal point as `export_recurrent`, so a model that
    /// returns rows there and `None` here is saying its state is entirely
    /// DeltaNet-shaped.
    fn export_aux_state(&self, _seq: usize) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Install what [`Self::export_aux_state`] wrote. Returns `true` when the
    /// blob was consumed.
    ///
    /// **Errors on a blob it cannot parse** rather than starting from zeros.
    /// The whole point of persisting this is that it cannot be recomputed from
    /// the K/V; silently discarding it produces a sequence that reads fluently
    /// against a history it never processed, which is the failure this path
    /// exists to make impossible.
    fn restore_aux_state(&self, _seq: usize, _blob: &[u8]) -> Result<bool> {
        Ok(false)
    }

    /// Whether this model keeps state indexed by token position that borrowing
    /// K/V does not bring along.
    ///
    /// Distinct from [`Self::carries_recurrent_state`]: a recurrence is one
    /// value carried forward and is rebuilt by replaying tokens in order, while
    /// this is per-position and has to travel with the positions it describes.
    /// Answering `true` is what makes a missing page worth a warning rather
    /// than silence.
    fn carries_positional_state(&self) -> bool {
        false
    }

    /// Discard per-position derived state for `seq` — paired with truncating
    /// the sequence's K/V to nothing.
    ///
    /// A slot being reused starts at position 0, and state describing the
    /// previous occupant's positions would put the next sequence's first token
    /// at the old prefix's end. No-op for a model with no such state.
    fn reset_positional_state(&self, _seq: usize) -> Result<()> {
        Ok(())
    }

    /// The largest position at or before `tokens` that `seq`'s per-position
    /// state can be cut back to with [`Self::truncate_positional_state`].
    ///
    /// Per-position state may be grouped more coarsely than the pieces a
    /// rebuild keeps — a page covering a slot's whole injected prefix — so a
    /// rebuild asks this before choosing where to cut, and keeps less when its
    /// piece boundary falls inside a group. `tokens` itself for a model with no
    /// such state.
    fn positional_cut_floor(&self, _seq: usize, tokens: usize) -> Result<usize> {
        Ok(tokens)
    }

    /// Cut `seq`'s per-position state back to its first `tokens` positions —
    /// paired with truncating the sequence's K/V to the piece boundary there, so
    /// a rebuild keeps the prefix it shares with the last one and re-injects only
    /// what follows. `tokens` is always a boundary between injected pieces that
    /// [`Self::positional_cut_floor`] accepts. No-op for a model with no such
    /// state.
    fn truncate_positional_state(&self, _seq: usize, _tokens: usize) -> Result<()> {
        Ok(())
    }

    /// Replace `seq`'s per-position state for positions `[from, to)` with
    /// `pieces`, placed from `from` on, and move everything after `to` by the
    /// change in width — paired with
    /// [`BatchedInferenceSession::splice_sealed`] putting those pieces' K/V in
    /// place, so a rebuild changes a piece in the middle of a slot without
    /// re-placing the pieces after it.
    ///
    /// Only a slot holding injected state alone splices: the rebuild cuts the
    /// live tail first ([`Self::truncate_positional_state`]). `from` and `to`
    /// are each a boundary [`Self::positional_cut_floor`] returns unchanged.
    /// No-op for a model with no such state.
    fn splice_positional_state(
        &self,
        _seq: usize,
        _from: usize,
        _to: usize,
        _pieces: &[PositionalPiece],
    ) -> Result<()> {
        Ok(())
    }

    /// The per-position state produced for whatever `seq` has forwarded since
    /// its last reset — a sealed section's page.
    ///
    /// **Sealed on a block boundary**, so the piece is self-contained: a
    /// section's tokens do not end on one, and rows left carried would belong
    /// to a block the *next* section completes. Returns `None` for a model with
    /// nothing of the kind.
    fn seal_positional_state(&self, _seq: usize) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    /// [`Self::seal_positional_state`] for a piece that is not the whole of
    /// what `seq` has forwarded — a conversation turn, whose slot carries the
    /// projection's injected prefix ahead of it and keeps decoding after.
    ///
    /// `start_pos` is the first token of the piece. Taken without disturbing the
    /// live state, because the sequence this is read from is still in use.
    fn seal_positional_range(&self, _seq: usize, _start_pos: usize) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    /// How many of `seq`'s tokens the model's per-position state actually
    /// covers, if it keeps any.
    ///
    /// The number that has to track the slot's K/V length. Exposed so the paths
    /// that move K/V around — carve a view, fork, rebuild a prefix — can say
    /// what they left behind, instead of the mismatch surfacing many layers
    /// later as a refused selection.
    fn positional_coverage(&self, _seq: usize) -> Option<usize> {
        None
    }

    /// Seal the per-position state for the last `tokens` tokens of `seq`,
    /// wherever it is held, as blobs to be re-installed in the order returned.
    ///
    /// Unlike [`Self::seal_positional_range`] this spans state that has already
    /// been folded into an injected piece, which is where part of a turn ends
    /// up once a reprojection has re-injected its opening half.
    /// Each blob is paired with the tokens it covers, so a caller can place a
    /// page in the span without parsing bytes it cannot read.
    fn seal_positional_tail_span(
        &self,
        _seq: usize,
        _tokens: usize,
    ) -> Result<Vec<(usize, Vec<u8>)>> {
        Ok(Vec::new())
    }

    /// Close `seq`'s per-position state on a page boundary, so what it forwards
    /// next begins a piece the seal can hand over — or drop — on its own.
    ///
    /// Called at the edges of a turn's `<think>…</think>` span during decode,
    /// which is the only moment the cut can be made: the rows for a span are
    /// pooled by the forward that carries it, and a block pooled across the
    /// boundary cannot be un-pooled afterwards.
    ///
    /// Returns the tokens the closed page covers, `0` when there was nothing to
    /// close — which is both the no-state case and the already-empty tail. A
    /// caller that needs to know the boundary actually landed must check it:
    /// a cut placed after the tokens it was meant to separate closes `0` and
    /// looks identical to one that was simply not needed.
    ///
    /// A no-op for a model that keeps no per-position state, so the caller can
    /// close unconditionally.
    fn close_positional_page(&self, _seq: usize) -> Result<usize> {
        Ok(0)
    }

    /// Install a sealed piece ahead of whatever `seq` forwards next — the
    /// counterpart of Arc-injecting that piece's K/V.
    ///
    /// **This is the one thing injection cannot copy.** K/V can be borrowed
    /// because it already exists; a recurrence cannot, which is what the branch
    /// checkpoint pass is for. Per-position state is a third case: it is not a
    /// recurrence, so no whole-prefix pass rebuilds it, and it is not K/V, so
    /// borrowing the chunks does not bring it along. Without it a query lands
    /// at a position whose blocks the model does not hold, and the select
    /// refuses rather than scoring against a prefix it never indexed.
    ///
    /// `key` is `page`'s identity ([`PieceKey::of`] its bytes), taken by the
    /// caller once when the page entered memory: the same pages are handed over
    /// on every projection that selects their piece, and a model that shares
    /// placed pages across slots finds them by it.
    ///
    /// Returns `false` when the model carries none, so the caller can offer it
    /// unconditionally.
    fn push_positional_state(&self, _seq: usize, _page: &[u8], _key: &PieceKey) -> Result<bool> {
        Ok(false)
    }

    /// Hold each `(page, key)`'s rows at the sampled layers in `seq`'s pool:
    /// pieces its projection pools with no K/V in the slot, whose rows the
    /// shadow scan reads (`docs/hit_driven_projection.md` §7.2). All of a
    /// refresh's pages at once, so a model places them in one launch and waits
    /// for its copies once; the pages are borrowed until it returns. A model
    /// that selects nothing holds none.
    fn pool_positional_states(&self, _seq: usize, _pages: &[(&[u8], PieceKey)]) -> Result<()> {
        Ok(())
    }

    /// The pieces `seq`'s pool holds now — what a refresh may keep without
    /// reading their pages again. Empty on a model that pools nothing.
    fn pooled_keys(&self, _seq: usize) -> Vec<PieceKey> {
        Vec::new()
    }

    /// Let go of `seq`'s pooled pieces not in `keep`.
    fn retain_pooled(&self, _seq: usize, _keep: &[PieceKey]) -> Result<()> {
        Ok(())
    }

    /// Device bytes [`Self::push_positional_state`] would newly place for
    /// `page`: zero when a piece with `key` is already resident (a placed page
    /// is shared across slots), else what placing it claims. What admission
    /// prices a projection's index pages at, before anything is pushed.
    fn positional_lift_bytes(&self, _page: &[u8], _key: &PieceKey) -> Result<usize> {
        Ok(0)
    }

    /// Device bytes a creep residual for `rows` holds while its sweep runs —
    /// what [`Self::claim_residual`] claims for them. Admission prices a
    /// prefill's rows at this as well as at their tier, because the residual is
    /// held from the cohort's formation to its end. Zero for a model whose
    /// sweeps keep no residual.
    fn creep_residual_bytes(&self, _session: &BatchedInferenceSession, _rows: usize) -> usize {
        0
    }

    /// Account for `tokens` of injected K/V that carry **no** per-position state.
    ///
    /// Advances where the next piece will be placed, and nothing else: the span
    /// is simply not indexed, which is the truth about it. Queries landing in it
    /// find no candidates, and every piece after it sits exactly where its K/V
    /// does.
    ///
    /// This used to have to do more. Positions were implied by the widths that
    /// came before, so a piece injected without its rows slid every later
    /// piece's start earlier by its own width and queries past it resolved
    /// through the wrong block — a wrong answer, not a missing one. A zero-row
    /// page had to stand in to keep the sum honest. Positions are recorded now
    /// (`IndexPage::roped_base` and the placement it is put at), so there is no
    /// sum left to keep honest.
    ///
    /// Call it wherever [`Self::push_positional_state`] would have been called
    /// but the piece has nothing to push. A no-op for a model that keeps no
    /// per-position state, so the caller declares it unconditionally.
    fn push_positional_gap(&self, _seq: usize, _tokens: usize) -> Result<()> {
        Ok(())
    }

    /// Rebuild any per-sequence index derived from the K/V, after the K/V has
    /// been restored but before the sequence decodes.
    ///
    /// The counterpart to the aux blob: state that IS derivable from the K/V,
    /// and large enough that deriving it beats storing it. Qwen3.8-Flash-Next's
    /// QSA index is ~16 MiB per attention layer at 128K tokens — two orders of
    /// magnitude past its recurrent snapshot — so it is rebuilt here from the
    /// keys the restore just brought back.
    ///
    /// A no-op for every model whose sequence state is fully described by its
    /// K/V plus its recurrent snapshot.
    fn reindex_restored_sequence(&self, _seq: usize, _tokens: usize) -> Result<()> {
        Ok(())
    }

    /// `child`'s state becomes `parent`'s — the linear join.
    ///
    /// A **move**, not a merge. A view is a linear continuation of its parent:
    /// it saw exactly the parent's tokens plus its own, so its state simply
    /// *is* the parent's new state. (Merging two divergent children has no
    /// defined arithmetic and is out of scope — `S` is an accumulated sum over
    /// one token order, and there is no operation meaning "and also these
    /// other tokens".)
    ///
    /// The rule for when to call it: **the state follows the K/V.** Wherever
    /// the view's decoded blocks transfer to the parent, this transfers with
    /// them; wherever the blocks are abandoned, this is not called and the
    /// state is dropped with them. Getting that backwards is silent in both
    /// directions — a missing move loses a turn's decode, and a spurious one
    /// re-introduces the `<think>` skew.
    fn move_recurrent(&self, _child: usize, _parent: usize) -> Result<()> {
        Ok(())
    }

    /// One speculative step's whole forward: decode every PLAIN sequence's committed token
    /// (`plain` = `(seq, committed)` pairs — sequences whose drafter proposed nothing) and verify
    /// one block per drafted sequence (append `blocks[i]` to `seqs[i]`), returning each plain
    /// row's logits and each block's per-position next-token logits rows. Default: one batched
    /// plain decode wave + sequential [`Self::verify_block`] calls — correct for any model. A
    /// model overrides with ONE wave carrying BOTH cohorts (plain rows leading, verify rows
    /// trailing): the per-wave fixed costs (every layer's launches, expert fetches)
    /// then amortize across every session AND the two cohorts stop paying two launch floors per
    /// step. Advances plain sequences by 1 and drafted sequences by their block length; the
    /// driver truncates drafted sequences back to the accepted lengths.
    /// Plan the rows this verify step's wave carries, and do whatever must
    /// happen *before* that wave opens.
    ///
    /// Returning `None` means this model has no one-wave verify — it has not
    /// promised the head will score every row of a multi-token member, which is
    /// the one thing a verify block needs — and [`Self::verify_blocks`] falls
    /// back to sequential single-token forwards. A model with a drafter
    /// overrides this and [`Self::end_verify`] together.
    ///
    /// **Split from the wave on purpose.** The setup here is real work with real
    /// ordering constraints — the `qwen35` lineage arms its verify cohort, whose
    /// DeltaNet rows the forward's commit then leaves pending for the rewind,
    /// and the MTP seed capture over both cohorts, sized before the forward
    /// opens because the arena refuses a device allocation from inside a wave;
    /// DeepSeek snapshots the streaming
    /// compressor/gallery state its blocks are about to absorb. None of it wants
    /// to know what else is riding the wave, and the caller that *does* know —
    /// the scheduler, folding these rows into the continuous-fair wave — cannot
    /// hand a `&mut self` borrow down into a model method. So the model plans,
    /// the caller runs the wave, and the model reads the rows back.
    ///
    /// `budget` is what the caller asked each sequence to draft this step, and
    /// it is **not** recoverable from `seqs`/`blocks`: an empty `seqs` means
    /// either that drafting was switched off for this wave *or* that the
    /// drafter had nothing to propose from yet — the first step after a prefill,
    /// which is precisely when the setup here has to run so a seed exists to
    /// draft from next time. A model that skips work when nothing will be
    /// drafted must read this, not the block shapes.
    ///
    /// On a wave that fails, the caller calls [`Self::abort_verify`] instead of
    /// [`Self::end_verify`].
    fn begin_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        budget: usize,
    ) -> Result<Option<VerifyPlan>> {
        let _ = (session, blocks, budget);
        // A model that has not overridden this has not promised its head scores
        // every row of a multi-token member, so it cannot verify a block. It can
        // still plan the *undrafted* case, and that is not a courtesy: it is what
        // lets the scheduler run one decode path for every model instead of a
        // speculative one and a plain one. A model with no drafter proposes
        // nothing, every sequence lands in `plain`, and this plan is an ordinary
        // one-token-per-sequence decode wave.
        if !seqs.is_empty() {
            return Ok(None);
        }
        // **The model's own device, not the host.** A plan's rows are cat'd with
        // whatever else the caller has on the wave — the scheduler's creep group
        // and glue live on the device — so a host-side row makes the wave's own
        // concatenation fail on a device mismatch. One upload for the cohort.
        let tokens: Vec<u32> = plain.iter().map(|&(_, t)| t).collect();
        let (decode_inputs, _) = upload_plan_rows(&tokens, &[])?;
        Ok(Some(VerifyPlan {
            decode_seqs: plain.iter().map(|&(s, _)| s).collect(),
            decode_inputs,
            verify_seqs: Vec::new(),
            verify_inputs: Vec::new(),
            rows: plain.len(),
        }))
    }

    /// Read a verify wave's scored rows back, and undo whatever
    /// [`Self::begin_verify`] armed.
    ///
    /// `logits` is the wave's `[decode | verify]` prefix in plan order. Returns
    /// the plain cohort's rows and one row per position of each block, which is
    /// what the accept walk reads. Advancing the sequences by what the wave
    /// wrote belongs here too — the driver truncates back to the accepted
    /// prefix afterwards.
    fn end_verify(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        logits: Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        let _ = blocks;
        if !seqs.is_empty() {
            candle::bail!(
                "end_verify: {} verify blocks came back to a model that planned none",
                seqs.len()
            );
        }
        if logits.len() != plain.len() {
            candle::bail!(
                "end_verify: wave scored {} rows for {} undrafted sequences",
                logits.len(),
                plain.len()
            );
        }
        for &(seq, _) in plain {
            session.advance_sequence(seq, 1)?;
        }
        Ok((logits, Vec::new()))
    }

    /// Release whatever [`Self::begin_verify`] armed for `seqs`, on any exit
    /// that keeps none of their blocks — a wave that failed, a readback or an
    /// accept walk that failed, a rollback that failed, or a row that committed
    /// nothing.
    ///
    /// Left in place, a snapshot taken before the wave names a rewind point
    /// that no longer exists, and a recurrent model's verify rows stay pending
    /// into the next step — a later truncate would rewind or fold from them.
    fn abort_verify(&self, seqs: &[usize]) {
        let _ = seqs;
    }

    /// Open a step over `seqs`: the [`Self::forward_wave`] calls that follow,
    /// until [`Self::end_step`], stand or fall together for these sequences.
    ///
    /// **A wave is atomic; a step swept in segments is not.** Each segment is
    /// its own wave and keeps what it did when it ends, so a step whose second
    /// segment fails leaves its first segment's layers advanced while the
    /// sequences' offsets stay where they were — and the retry runs those
    /// layers' tokens in again. The step bracket is what takes the committed
    /// segments back. No-op for a model whose waves carry nothing a retry
    /// could run in twice.
    fn begin_step(&self, seqs: &[usize]) -> Result<()> {
        let _ = seqs;
        Ok(())
    }

    /// Close the step [`Self::begin_step`] opened. With `keep`, its waves
    /// stand. Without, everything its waves committed for the step's
    /// sequences goes back to where the step found it: the K/V of trunk
    /// layers `[0, committed)` truncated to each sequence's offset, and every
    /// carried state restored. An error means the sequences' state can no
    /// longer be vouched for.
    fn end_step(
        &self,
        session: &mut BatchedInferenceSession,
        keep: bool,
        committed: usize,
    ) -> Result<()> {
        let _ = (session, keep, committed);
        Ok(())
    }

    fn verify_blocks(
        &self,
        session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        layer_end: usize,
        budget: usize,
    ) -> Result<VerifiedRows> {
        // The standalone shape: one wave carrying nothing but this step's own
        // rows. The scheduler does not come through here — it runs the same
        // three phases around its own co-batched wave.
        // Every exit below that keeps no block ends what `begin_verify` armed:
        // a recurrent model's verify rows stay pending until the rewind folds
        // them or this discards them.
        let planned = match self.begin_verify(session, plain, seqs, blocks, budget) {
            Ok(p) => p,
            Err(e) => {
                self.abort_verify(seqs);
                return Err(e);
            }
        };
        if let Some(plan) = planned {
            let issued = issue_verify_wave(
                self,
                session,
                &plan.decode_seqs,
                &plan.decode_inputs,
                &plan.verify_seqs,
                &plan.verify_inputs,
                layer_end,
            );
            return match issued {
                Ok(out) => {
                    if out.logits.len() != plan.rows {
                        self.abort_verify(seqs);
                        candle::bail!(
                            "verify_blocks: wave scored {} rows, plan wanted {}",
                            out.logits.len(),
                            plan.rows
                        );
                    }
                    let (plain_rows, block_rows) =
                        match self.end_verify(session, plain, seqs, blocks, out.logits) {
                            Ok(r) => r,
                            Err(e) => {
                                self.abort_verify(seqs);
                                return Err(e);
                            }
                        };
                    Ok(VerifiedRows {
                        plain: plain_rows,
                        blocks: block_rows,
                        head: Some(out.head),
                    })
                }
                Err(e) => {
                    self.abort_verify(seqs);
                    Err(e)
                }
            };
        }
        let mut plain_out = Vec::with_capacity(plain.len());
        if !plain.is_empty() {
            let dseqs: Vec<usize> = plain.iter().map(|&(s, _)| s).collect();
            let dinputs: Vec<Tensor> = plain
                .iter()
                .map(|&(_, t)| Tensor::from_vec(vec![t], (1, 1), &Device::Cpu))
                .collect::<Result<_>>()?;
            let step = self.forward_wave(
                session,
                &dseqs,
                &dinputs,
                &[],
                &[],
                &[],
                &[],
                0,
                layer_end,
                None,
            )?;
            plain_out = step.logits_owned()?;
            if plain_out.len() != plain.len() {
                candle::bail!(
                    "verify_blocks: plain wave scored {} rows for {} seqs",
                    plain_out.len(),
                    plain.len()
                );
            }
            for &(seq, _) in plain {
                session.advance_sequence(seq, 1)?;
            }
        }
        let mut out = Vec::with_capacity(seqs.len());
        for (i, &seq) in seqs.iter().enumerate() {
            out.push(self.verify_block(session, seq, &blocks[i], layer_end)?);
        }
        // Several forwards, so these rows were owned as each one returned.
        Ok(VerifiedRows {
            plain: plain_out,
            blocks: out,
            head: None,
        })
    }

    /// One speculative-decode step for `seq` (model-agnostic), as lossless as `chooser` — see
    /// [`Self::speculative_decode_step_batch`]. `committed` is the last
    /// accepted token, held OUT of the KV; it is placed at the current sequence offset. Drafts a
    /// block, verifies `[committed, drafts…]`, accepts the longest prefix whose proposals agree
    /// with what `chooser` draws for this model, and **emits each accepted token to `emit`, one at
    /// a time** — the model's own continuation, in order. This is what keeps speculative decode a
    /// *transparent accelerator*: the caller's main loop runs its normal per-token handling (stop
    /// sequences, EOS, sampling/steering decisions) on each token exactly as for plain decode,
    /// instead of the driver re-implementing any of it. `emit` returns `false` to stop generating
    /// after that token; the driver rolls the KV back to the emitted prefix. Returns
    /// `Some(next_committed)` — the seed to pass as `committed` on the next call (already emitted,
    /// held out of the KV) — or `None` when `emit` asked to stop. With no drafter this is one plain
    /// decode (emits exactly one token). At least one token is always emitted.
    // Session, sequence, seed, budget, layer bound, chooser and sink: seven
    // independent inputs plus `self`. A struct here would name a bag that no
    // caller holds as a unit.
    #[allow(clippy::too_many_arguments)]
    fn speculative_decode_step(
        &self,
        session: &mut BatchedInferenceSession,
        seq: usize,
        committed: u32,
        max_draft: usize,
        layer_end: usize,
        chooser: &mut dyn TokenChooser,
        emit: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Option<u32>> {
        // The batch-of-1 case of the batched driver — one implementation.
        let mut emits: Vec<Box<dyn FnMut(u32) -> bool + '_>> = vec![Box::new(emit)];
        let step = self.speculative_decode_step_batch(
            session,
            &[seq],
            &[committed],
            &[max_draft],
            layer_end,
            chooser,
            &mut emits,
        )?;
        Ok(step.next[0])
    }

    /// One speculative-decode step for MANY sequences — semantics identical to running
    /// [`Self::speculative_decode_step`] once per sequence, with the expensive parts batched:
    /// every block is verified in ONE `verify_blocks` call (a single wave when the model overrides
    /// it), and every scored row of the step is stacked once so `chooser` pays one dispatch per
    /// block position rather than one per row (per-row `to_scalar` round-trips are a
    /// launch-overhead wall at batch width). Each sequence keeps its own emit sink and truncates
    /// to its own accepted prefix. Returns each sequence's next `committed` seed (`None` where its
    /// emit stopped).
    ///
    /// `chooser` decides what each scored row commits, so the step is exactly as lossless as
    /// the chooser: an exact sampler makes it lossless under sampling as well as under greedy
    /// decode — see [`speculative_choice`] for why a greedy drafter reduces the textbook
    /// accept/reject rule to "sample the row, accept the proposal iff the sample agrees" — and
    /// one applying [`TypicalAcceptance`] departs where that rule takes a draft. Pass
    /// [`GreedyChooser`] for bit-identical greedy output.
    ///
    /// `max_drafts` is each sequence's own draft depth (see [`super::draft_depth`]): the drafter
    /// walks the cohort to the deepest of them, once, and each sequence keeps only its own
    /// prefix of what was drafted. The step reports what each sequence actually drafted
    /// ([`SpeculativeStep::drafted`]) — fewer than its depth where the drafter proposed less —
    /// which is what an acceptance estimate must divide by.
    // As [`Self::speculative_decode_step`], over a cohort.
    #[allow(clippy::too_many_arguments)]
    fn speculative_decode_step_batch(
        &self,
        session: &mut BatchedInferenceSession,
        seqs: &[usize],
        committed: &[u32],
        max_drafts: &[usize],
        layer_end: usize,
        chooser: &mut dyn TokenChooser,
        emits: &mut [Box<dyn FnMut(u32) -> bool + '_>],
    ) -> Result<SpeculativeStep> {
        // Everything before the forward: drafting, the plain/spec partition and
        // the per-sequence block clones. Spanned so the step's CPU time is
        // covered end to end — `spec:setup` + `spec:verify` + `spec:gather` +
        // `spec:accept` should now sum to the step.
        // The whole step — draft, verify, accept, rollback and every gap between
        // them — as one range, so a trace can take exactly one step's kernels.
        let _step = span("spec:step");
        let t_setup = std::time::Instant::now();
        if seqs.len() != committed.len()
            || seqs.len() != emits.len()
            || seqs.len() != max_drafts.len()
        {
            candle::bail!(
                "speculative_decode_step_batch: {} seqs, {} committed, {} emits, {} draft depths",
                seqs.len(),
                committed.len(),
                emits.len(),
                max_drafts.len()
            );
        }
        if seqs.is_empty() {
            return Ok(SpeculativeStep {
                next: Vec::new(),
                drafted: Vec::new(),
            });
        }
        let max_draft = max_drafts.iter().copied().max().unwrap_or(0);
        // **Refused up front for a model that cannot rewind.**
        //
        // Speculative decode is built on "decode the whole draft block, then put
        // the sequence back to the accepted prefix". The check belongs here, at
        // the one entry point, and not inside the rewind: by the time the driver
        // has drafted and verified, refusing is an error raised against work
        // already done, and *not* refusing is a state that has absorbed rejected
        // tokens the K/V no longer holds.
        //
        // The question is whether the model can REWIND, not whether it carries
        // recurrent state — the hybrid does both. Its verify block never writes
        // `S`, only each DeltaNet layer's timeline, and the fold applies the
        // accepted prefix of it — exactly the state those tokens alone would
        // have produced (`qwen35::spec`, `docs/deltanet_timeline.md`). Which
        // offsets are rewindable is a property of the verified block, so
        // `truncate_sequences` is what refuses one it has no rewind point for.
        //
        // And only a step that would actually DRAFT needs the rewind: at a zero
        // draft budget every sequence takes a plain decode wave below and
        // nothing is ever put back, so a recurrent model with no drafter (a
        // bring-up, a checkpoint whose conversion carries no MTP head) runs
        // through this driver legally.
        //
        // The test is `max_draft`, the width this step will actually hand
        // `speculative_draft` below — NOT `draft_budget`, which is the model's
        // own advice and defaults to 0. Testing the advice lets a caller pass a
        // real width to a model that never overrode it and draft without a
        // rewind point, which is precisely the case the bail exists to refuse.
        if !self.can_rewind_speculative_block() && max_draft > 0 {
            candle::bail!(
                "speculative decode is unavailable on this model: accepting k of n \
                 drafted tokens requires rewinding the sequence, and this model \
                 carries recurrent state with no per-token decomposition to rewind \
                 to. A model whose `truncate_sequences` can restore the recurrence \
                 — as the qwen35 lineage's replay does — declares \
                 `can_rewind_speculative_block`; without that, run it without a \
                 drafter."
            );
        }
        // **A recurrent model's block must fit its timeline.** Each DeltaNet
        // layer keeps a verify block's rows in a fixed `TIMELINE_ROWS`-row
        // timeline; a deeper block would run as a prefill span, which writes a
        // whole advanced state and can never be partially accepted. Refused
        // here, before anything is drafted.
        if self.carries_recurrent_state() && max_draft >= TIMELINE_ROWS {
            candle::bail!(
                "speculative_decode_step_batch: {max_draft} drafts make a {}-row block, deeper \
                 than the {TIMELINE_ROWS}-row DeltaNet timeline — at most {} drafts a block",
                max_draft + 1,
                TIMELINE_ROWS - 1
            );
        }
        // Draft per sequence. A sequence whose drafter proposes NOTHING (no
        // drafter, or the acceptance fallback holding it back) takes a PLAIN
        // decode wave instead of a 1-token verify: the verify path's
        // snapshot/rollback + slot-rebuild machinery costs measurably more
        // than a decode wave, which is exactly the loss the fallback exists
        // to avoid. Drafted sequences still verify together in one wave.
        let t_draft = std::time::Instant::now();
        let mut poss = Vec::with_capacity(seqs.len());
        for &seq in seqs {
            poss.push(session.sequence_offset(seq).ok_or_else(|| {
                candle::Error::msg("speculative_decode_step_batch: unknown sequence")
            })?);
        }
        // ONE call for the whole cohort: the drafter batches its own passes so
        // the weights it reads are read once for the step, not once per session.
        // It walks to the deepest sequence's depth; each sequence then keeps its
        // own prefix, so a shallow one verifies — and pays — only that.
        let mut drafts = self.speculative_draft(session, seqs, committed, max_draft)?;
        if drafts.len() != seqs.len() {
            candle::bail!(
                "speculative_decode_step_batch: drafter returned {} proposal lists for {} sequences",
                drafts.len(),
                seqs.len()
            );
        }
        for (d, &depth) in drafts.iter_mut().zip(max_drafts) {
            d.truncate(depth);
        }
        let blocks: Vec<Vec<u32>> = drafts
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut block = Vec::with_capacity(d.len() + 1);
                block.push(committed[i]);
                block.extend_from_slice(d);
                block
            })
            .collect();
        pipeline_record_duration("spec:draft", t_draft.elapsed(), 1);
        let plain: Vec<usize> = (0..seqs.len()).filter(|&i| blocks[i].len() == 1).collect();
        let spec: Vec<usize> = (0..seqs.len()).filter(|&i| blocks[i].len() > 1).collect();

        // ONE forward for both cohorts: every undrafted sequence's committed
        // token decodes as an ordinary plain row (live slot, on-device
        // write-len commit — the sequence advances inside, and the later
        // uniform truncate is a no-op for it since kept is always 1) and every
        // drafted block verifies as virtual rows, in the same wave when the
        // model overrides `verify_blocks` (one launch floor per step, not
        // two).
        pipeline_record_duration("spec:setup", t_setup.elapsed(), 1);
        // GPU-side elapsed for the whole forward, beside the CPU-side
        // `spec:verify` that launches it. Two stream-ordered events, so this is
        // execution plus any starvation gap *inside* the forward.
        //
        // **Read it against `wv:sweep`**, which is the layer sweep plus the
        // head. The difference is everything this step does around the model.
        // Whenever the accept walk waits far longer than the `decode:*` /
        // `dn:*` spans account for, this pair says whether the missing time is
        // inside the model or around it.
        let g_fwd = gpu_span("verify:fwd", session.device());
        let t_verify = std::time::Instant::now();
        let plain_pairs: Vec<(usize, u32)> =
            plain.iter().map(|&i| (seqs[i], committed[i])).collect();
        let spec_seqs: Vec<usize> = spec.iter().map(|&i| seqs[i]).collect();
        let spec_blocks: Vec<Vec<u32>> = spec.iter().map(|&i| blocks[i].clone()).collect();
        // The scored rows, and — when one wave produced them — that wave, held
        // so the rows (views on its head's span) stay valid through the walk.
        // The walk below reads the verify wave's logits on its head span, so
        // that wave reserves the walk's selections beside them.
        session.set_accept_in_place(true);
        let verified = self.verify_blocks(
            session,
            &plain_pairs,
            &spec_seqs,
            &spec_blocks,
            layer_end,
            max_draft,
        );
        session.set_accept_in_place(false);
        let VerifiedRows {
            plain: mut plain_rows,
            blocks: spec_logits,
            head,
        } = verified?;
        pipeline_record_duration("spec:verify", t_verify.elapsed(), 1);
        g_fwd.end();
        // Every exit between here and the rollback keeps none of the blocks,
        // so it ends what the verify armed: a recurrent model's verify rows
        // would otherwise stay pending into the next step.
        let abort = |e: candle::Error| {
            self.abort_verify(&spec_seqs);
            e
        };
        // The row gather between the forward and the accept walk, and the
        // `row_of` bookkeeping. Spanned because it is the only part of the step
        // that is neither `spec:verify` nor `spec:walk`.
        let t_gather = std::time::Instant::now();
        let mut plain_logits: Vec<Option<Tensor>> = vec![None; seqs.len()];
        for &i in plain.iter().rev() {
            plain_logits[i] = plain_rows.pop();
        }

        // Every scored row of BOTH waves, each a `[1, vocab]` view where its head
        // wrote it: plain rows lead (in `plain` order), then each verify block's
        // rows. `row_of[i]` locates sequence `i`'s run among them, so the accept
        // walk lifts one block position across the cohort by naming its rows.
        let rows: Vec<Tensor> = plain
            .iter()
            .map(|&i| plain_logits[i].as_ref().expect("filled above"))
            .chain(spec_logits.iter().flatten())
            .map(|t| t.flatten_all()?.unsqueeze(0))
            .collect::<Result<_>>()
            .map_err(abort)?;
        let mut row_of: Vec<(usize, usize)> = vec![(0, 0); seqs.len()];
        for (k, &i) in plain.iter().enumerate() {
            row_of[i] = (k, 1);
        }
        let mut cur = plain.len();
        for (k, &i) in spec.iter().enumerate() {
            let n = spec_logits[k].len();
            row_of[i] = (cur, n);
            cur += n;
        }
        // A block scores exactly one row per token and the walk indexes rows by
        // block position, so a verify that returned a different count would read
        // some other sequence's row instead of failing.
        for (i, &(_, n_rows)) in row_of.iter().enumerate() {
            if n_rows != blocks[i].len() {
                return Err(abort(candle::Error::Msg(format!(
                    "speculative_decode_step_batch: sequence {} verified {n_rows} rows for a \
                     {}-token block",
                    seqs[i],
                    blocks[i].len()
                ))));
            }
        }

        // Position-major accept, then one batched rollback over every target.
        //
        // Positions run in order and a sequence leaves `alive` as soon as it
        // commits a token that ends its block — either the model's own token
        // diverged from the proposal (that token IS the correction) or the block
        // ran out (its last row is a free bonus token). The cohort therefore
        // narrows as the walk goes, and the step costs at most `max_draft + 1`
        // chooser dispatches over the sequences still standing.
        //
        // Walking positions rather than deciding one stacked argmax over
        // everything is what lets a sampling chooser be exact: row `j` is scored
        // under the draft prefix that reaches it, so a chooser carrying
        // repetition penalties or a grammar stencil advances its per-sequence
        // state along exactly the path this loop commits.
        pipeline_record_duration("spec:gather", t_gather.elapsed(), 1);
        let t_accept = std::time::Instant::now();
        // A stateless chooser picks every scored row in one call — one launch
        // and one readback for the step, where walking costs one of each per
        // position, serialised: ~0.75 ms of Flash-Next's 29 ms single-session
        // step went to nine of those round trips. The walk below commits from
        // the picks by the same rule either way.
        let ahead = if chooser.stateless() {
            let all = match Tensor::cat_view(&rows, 0) {
                Some(all) => all,
                None => Tensor::cat(&rows, 0).map_err(abort)?,
            };
            Some(
                chooser
                    .choose(&all, &all_rows(&blocks, &row_of))
                    .map_err(abort)?,
            )
        } else {
            None
        };
        let mut walk = AcceptWalk::new(&blocks);
        while !walk.finished() {
            let tokens = match &ahead {
                Some(picks) => walk
                    .alive()
                    .iter()
                    .map(|&i| picks[row_of[i].0 + walk.position()])
                    .collect(),
                None => {
                    let walked = walk.rows();
                    // This position's rows across the cohort. Rows that sit
                    // together in their head's block are named as one view;
                    // others are gathered beside the logits on the span
                    // (`WaveBuffer::AcceptRows` — each scored row is selected
                    // at most once across the walk). No index is uploaded: the
                    // rows are named on the host, where they are known.
                    let picked: Vec<Tensor> = walk
                        .alive()
                        .iter()
                        .map(|&i| rows[row_of[i].0 + walk.position()].clone())
                        .collect();
                    let block = match Tensor::cat_view(&picked, 0) {
                        Some(block) => block,
                        None => Tensor::cat(&picked, 0).map_err(abort)?,
                    };
                    chooser.choose(&block, &walked).map_err(abort)?
                }
            };
            walk.commit(&tokens, |i, token| (emits[i])(token))
                .map_err(abort)?;
        }
        let (next, kept) = walk.finish();
        // The walk is bounded by the draft budget and the rollback is not, so
        // they are timed apart: one is `max_draft + 1` chooser dispatches over a
        // narrowing cohort, the other visits every layer's chunk list. A single
        // span over both reads as "accept got slower with depth" and hides which.
        pipeline_record_duration("spec:walk", t_accept.elapsed(), 1);
        // Every view on the head's span, then the wave that holds it: the
        // rollback below may open waves of its own (the recurrent fold), and the
        // span is reclaimed only once nothing names it.
        drop(rows);
        drop(plain_logits);
        drop(spec_logits);
        drop(head);
        let targets: Vec<(usize, usize)> = seqs
            .iter()
            .enumerate()
            .map(|(i, &seq)| (seq, poss[i] + kept[i]))
            .collect();
        // ONE rollback call for the step: every sequence's target at once, so a
        // model whose rollback does real work (the recurrent fold) batches it
        // across the cohort instead of paying it per sequence. A rollback that
        // fails keeps none of the blocks.
        let t_rollback = std::time::Instant::now();
        self.truncate_sequences(session, &targets).map_err(abort)?;
        pipeline_record_duration("spec:rollback", t_rollback.elapsed(), 1);
        pipeline_record_duration("spec:accept", t_accept.elapsed(), 1);
        Ok(SpeculativeStep {
            next,
            drafted: blocks.iter().map(|b| b.len() - 1).collect(),
        })
    }

    /// How a session's headers pick each sequence's RoPE rung — the model's
    /// schedule's ceilings (`docs/progressive_yarn.md` §8). One unbounded rung
    /// for a model whose kernels read none.
    fn rope_select(&self) -> RungSelect {
        RungSelect::unbounded()
    }

    /// Create a batched inference session configured for this model.
    fn create_batched_session(&self, config: BatchedConfig) -> Result<BatchedInferenceSession> {
        let mut config = config;
        let props = self.model_core_properties();
        config.k_hi_error_threshold_factor *= props.k_hi_error_threshold_factor;
        config.k_low_error_threshold_factor *= props.k_low_error_threshold_factor;
        config.v_hi_error_threshold_factor *= props.v_hi_error_threshold_factor;
        config.v_low_error_threshold_factor *= props.v_low_error_threshold_factor;
        let mut session = BatchedInferenceSession::new(
            // The default stack has no draft head; a model that carries one
            // overrides this and declares its head's layer.
            KvLayers::stream_only(props.num_layers),
            props.n_kv_heads,
            props.head_dim,
            self.device(),
            config,
        )?;
        session.set_rope_select(self.rope_select());
        // Materialise the norm weights for this session's activation dtype, here
        // rather than at each call site. A session is where the dtype is decided,
        // it is created outside any wave, and the forward *refuses* a mismatch —
        // so leaving the call to callers would mean every one of them has to
        // remember, and the one that forgets fails at its first forward instead
        // of at the line that was wrong.
        self.maybe_change_dtype(session.activation_dtype())?;
        Ok(session)
    }

    /// Create a sibling session that shares the KV arena pool with `source`.
    ///
    /// Each `ChunkedKvBacking` in `source` is an `Arc<BackingInner>`.  Cloning
    /// those arcs into the new session means both sessions operate on the same
    /// underlying tensors, slot table, and arena table.  This is required for
    /// `append_borrowed_chunks_cow` to find chunk handles produced in the sibling
    /// session (e.g. fixed boundary-injection chunks generated in `proto_session`).
    ///
    /// Slot-index collisions between the two sessions are avoided automatically:
    /// `create_sequence` consults the shared backing for the first free slot, so
    /// the sibling gets a slot index that does not overlap with the source session.
    fn create_session_sharing_backings(
        &self,
        source: &BatchedInferenceSession,
        config: BatchedConfig,
    ) -> Result<BatchedInferenceSession> {
        let mut config = config;
        let props = self.model_core_properties();
        config.k_hi_error_threshold_factor *= props.k_hi_error_threshold_factor;
        config.k_low_error_threshold_factor *= props.k_low_error_threshold_factor;
        config.v_hi_error_threshold_factor *= props.v_hi_error_threshold_factor;
        config.v_low_error_threshold_factor *= props.v_low_error_threshold_factor;
        let backings = source.backings().to_vec();
        let mut session =
            BatchedInferenceSession::new_with_backings(backings, config, source.device());
        session.set_rope_select(self.rope_select());
        // The other way a session comes into being, and it decides an activation
        // dtype just as `create_batched_session` does — so it materialises the
        // norm weights the same way.
        self.maybe_change_dtype(session.activation_dtype())?;
        Ok(session)
    }

    /// Prunes excess memory usage.
    fn prune(&self) -> Result<()>;

    /// Snapshot expert pipeline telemetry counters (if the model has an expert cache).
    fn expert_stats(&self) -> Option<PipelineStats> {
        None
    }

    /// Whether the expert cache's pipeline thread or stager has exited. Neither
    /// is restarted, and every forward after it fails, so a model whose cache
    /// reports this can serve nothing again.
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
    /// [`Self::expert_stats`] waits for the pipeline thread to serve every
    /// queued layer.
    fn expert_hit_counts(&self) -> Option<(usize, usize)> {
        None
    }

    /// The expert cache's residency references over the interval the counters
    /// cover, `(lru, min)` (`ExpertCache::hit_references`), if the model has
    /// one. A replay of the interval — for reports, not for a step.
    fn expert_hit_references(&self) -> Option<(f64, f64)> {
        None
    }

    /// Write the expert cache's routing trace to `path`
    /// (`ExpertCache::write_routing_trace`); nothing for a model without one.
    fn write_expert_routing_trace(&self, _path: &Path) -> Result<()> {
        Ok(())
    }

    /// Snapshot the row-cache counters of a disk-resident embedding tier, if
    /// this model has one.
    ///
    /// `(hits, misses, evictions)`. Only Qwen3.8-Flash-Next's PLE table is
    /// served this way today: 51 B parameters that never enter VRAM, fronted by
    /// a bounded RAM cache whose hit rate is an empirical property of the
    /// traffic rather than of the table. The counters existed from the start
    /// and nothing read them — which is why the tier's sizing has been a design
    /// estimate rather than a measurement.
    fn row_cache_stats(&self) -> Option<(u64, u64, u64)> {
        None
    }

    /// Snapshot the layer-streaming counters, if this model's weights are slot
    /// tenants. See `BatchedModelCore::layer_stream_stats`.
    fn layer_stream_stats(&self) -> Option<[usize; 7]> {
        None
    }

    /// Buy `regions` of weight-side ground for the KV side, answering with the
    /// bytes conceded. See `BatchedModel::request_kv_ground` — this is the path a
    /// stalled scheduler uses to break a wave that cannot allocate.
    fn request_kv_ground(&self, regions: usize) -> u64 {
        let _ = regions;
        0
    }

    /// The other direction: let the weight side take back KV regions standing
    /// free. See `BatchedModel::reclaim_spare_ground`.
    ///
    /// **Only legal between forwards** — a boundary move evicts and relocates
    /// expert slots. The scheduler's guest drain is the caller: after a
    /// co-resident model has been unloaded and its span ground returned, the
    /// weight side should take back whatever the drain's eviction made it
    /// concede, rather than waiting for the pressure signal to swing the other
    /// way. On a quiet world that may be a long time, and until it does the
    /// model runs with a smaller expert working set for no reason.
    fn reclaim_spare_ground(&self) {}

    /// Concede weight-side ground below the serving floor for a guest drain. See
    /// `BatchedModel::concede_for_guest`.
    fn concede_for_guest(&self, regions: usize) -> u64 {
        let _ = regions;
        0
    }

    /// Copy the pinned experts to host memory and concede their ground for a
    /// guest drain still short. See `BatchedModel::stash_pinned_for_guest`.
    fn stash_pinned_for_guest(&self) -> u64 {
        0
    }

    /// Grow the weight side back to its serving floor after a guest drain. See
    /// `BatchedModel::restore_serving_floor`.
    fn restore_serving_floor(&self) -> bool {
        true
    }

    /// Live VRAM held by the model's weights (fixed base + time-varying resident
    /// experts), for the whole-card VRAM decomposition. `None` if unavailable.
    fn resident_weight_bytes(&self) -> Option<usize> {
        None
    }

    /// The weight side as the wave rate planner prices it — MoE geometry, and
    /// the range the expert zone may move in.
    ///
    /// [`WeightPlanning::Dense`] by default: a stack with no expert cache has no
    /// residency to trade a wave's rows against, so not planning is the correct
    /// answer and the width backstop is the only bound.
    ///
    /// A routed model overrides this, and the distinction between "nothing to plan"
    /// and "the gauges are broken" lives in [`WeightPlanning`] rather than in an
    /// `Option` — see that type for what the conflation cost.
    fn weight_plan(&self) -> WeightPlanning {
        WeightPlanning::Dense
    }

    /// Reservation bytes held by per-sequence recurrent state, for the same
    /// decomposition.
    ///
    /// Zero is the honest answer for a model with no recurrent layers, which is
    /// why this is not an `Option`: a pure-attention stack holds none, it does
    /// not fail to know.
    ///
    /// Reported because it is large and moves: a hybrid stack's recurrent memory
    /// is ~126 MiB per sequence, so a wide wave puts several GiB of the
    /// reservation here, and a total that omits it makes the span look emptier
    /// than it is — the same class of blindness that let the dense weights hide.
    fn recurrent_reserved_bytes(&self) -> usize {
        0
    }

    /// What **one** sequence's carried state costs, from the model's geometry.
    ///
    /// The admission figure, and not derivable from
    /// [`Self::recurrent_reserved_bytes`]: that total is zero before the first
    /// store is built, and dividing it by the sequences in flight answers with
    /// a number that climbs as conversations go idle, since the sum counts
    /// every store held and the divisor only those in flight.
    fn recurrent_store_bytes(&self) -> usize {
        0
    }

    /// Compact the arenas per-sequence recurrent state lives in, moving at most
    /// `max_moves` layer states (zero for no bound). Between forwards only — the
    /// scheduler runs it in the same gap as the KV compaction, for the same holes.
    fn compact_recurrent(&self, _max_moves: usize) -> Result<RecurrentCompaction> {
        Ok(RecurrentCompaction::default())
    }

    /// Reset expert pipeline telemetry counters to zero.
    fn reset_expert_stats(&self) {}

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    fn snapshot_profiles(&self) -> ProfileSnapshot {
        ProfileSnapshot::default()
    }
}
