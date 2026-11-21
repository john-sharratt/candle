//! Builds the scheduler: derives its provenance fold, binds its thread to the
//! device context, and rebuilds the substrate from the redo log on startup.
//!
//! The provenance fold is derived once from the model's geometry, because every
//! stored signature is comparable only to one folded the same way, and the
//! substrate's read paths refuse a record stamped with a different fold. The
//! thread binds to the device context before it allocates pinned host memory,
//! since a failed allocation silently falls back to pageable memory and turns
//! every cold load into a synchronous copy. Startup replays each persisted turn
//! stream in turn order and leaves every turn cold until a projection needs it,
//! and the daemon's loading waiter is released whether or not the replay worked.

use super::*;

impl Scheduler {
    /// Create a new scheduler. Called once by `ConversationEngine`.
    pub fn new(
        rx: Receiver<SchedulerRequest>,
        model: Box<dyn ManagedBatchedModel + Send>,
        session: BatchedInferenceSession,
        tokenizer: tokenizers::Tokenizer,
        eos_tokens: TokenBuffer,
        vocab_size: usize,
        max_recent_len: usize,
        show_special_tokens: bool,
        penalty_log_path: Option<PathBuf>,
        health_config: DecodeHealthConfig,
        max_prefill_pass_tokens: usize,
        persist_trigger: PersistenceTrigger,
        summariser_trigger: SummariserTrigger,
        boundary_markers: projection_assembler::BoundaryMarkers,
        guests: Arc<Guests>,
    ) -> Self {
        let device = model.device().clone();

        // Derive this process's provenance fold from the model's own geometry,
        // once. Every stored signature is comparable only to one folded the same
        // way, so this single value both folds the captures and stamps the
        // records, and the substrate read paths check a record's stamp against
        // it and refuse a mismatch rather than scoring incomparable bits into a
        // confident number.
        let (prov_fold, prov_capture_layers) = {
            let props = model.model_core_properties();
            let f = crate::provenance::FoldParams::derive(
                props.n_kv_heads,
                props.provenance_capture_layers,
                props.head_dim,
            );
            crate::provenance::set_active_fold(f);
            (f, props.provenance_capture_layers)
        };

        // Force this thread to bind to the device's CUDA context
        // BEFORE we allocate pinned host memory below. The model and
        // session were created on another thread; the CUDA context is
        // per-thread on Windows, and `cuMemHostAlloc` returns
        // `CUDA_ERROR_NOT_INITIALIZED` if called from a thread that
        // hasn't bound to a context yet. A failed alloc silently falls
        // back to non-pinned heap memory, which turns every cold-load
        // HtoD into a synchronous copy at ~1 GB/s instead of an
        // async-pinned DMA at PCIe rate.
        if let Device::Cuda(d) = &device {
            let _ = d.cuda_context().bind_to_thread();
        }

        // Resident gallery arena for the paged belief scan, built for this
        // model's fold (`prov_fold`): `words_per_token` u64 per token across its
        // layer-groups. It must match the signatures the captures produce — a
        // fixed Qwen3-30B width (24) here dropped every token of any other
        // geometry (Qwen2-0.5B folds to 6) and zeroed the scan without a word.
        // `new` errors (→ None) on a non-CUDA device; it allocates no VRAM until
        // the first turn is made resident.
        //
        // The arena used to register an eviction closure with the VRAM
        // governor, at a cheap relief rung, so the governor would shed resident
        // galleries before it ever evicted model KV. The rungs are gone;
        // `relieve_vram_pressure` calls `evict_past_keep` directly and does it
        // before touching KV, which is the same priority expressed as call order.
        // **What this checkpoint actually brings to a decode.** Every one of
        // these is a capability the engine silently degrades around rather than
        // failing on: a model with no drafter reports `draft_budget == 0`, every
        // sequence takes a one-token row, and the speculative path becomes an
        // expensive no-op that logs nothing. That is a ~5x throughput difference
        // — measured 4.85 accepted/step in the batched-forward harness against
        // one token per forward in the daemon — and until now the only way to
        // tell the two apart was arithmetic on wave timings.
        //
        // Logged once, at load, so "what is running" is a fact in the log rather
        // than an inference from throughput.
        // The widths are the ladder's answer AT LOAD. A model may also clamp the
        // budget per wave against free VRAM (`affordable_draft_budget`), and that
        // clamp is inert here — nothing is resident yet — so a wide wave can run
        // shallower than the wide figure suggests. `w1` is the one to read for
        // "does this checkpoint speculate at all": 0 there means no drafter.
        let typical = model.typical_acceptance();
        typical.assert_valid();
        tracing::info!(
            target: "candle_conversation::engine_caps",
            typical_epsilon = typical.epsilon,
            typical_delta = typical.delta,
            draft_budget_w1 = model.draft_budget(1),
            draft_budget_w8 = model.draft_budget(8),
            draft_ladder_w64_unclamped = model.draft_budget(64),
            speculative_rewind = model.can_rewind_speculative_block(),
            recurrent_state = model.carries_recurrent_state(),
            positional_state = model.carries_positional_state(),
            recurrent_memories = model.recurrent_memory_count(),
            layers = model.num_layers(),
            vocab_size,
            "decode capabilities at load (draft_budget 0 ⇒ NO speculation: one token per forward)"
        );

        let gallery_arena = GalleryArena::new(
            &device,
            prov_fold.words_per_token(),
            prov_fold.group_sizes.len(),
        )
        .map(Arc::new)
        .ok();
        // The tokens the tokenizer can name; a checkpoint pads its logits row
        // past them, and the padded tail must carry no probability.
        let live_vocab = tokenizer
            .get_vocab(true)
            .values()
            .max()
            .map_or(vocab_size, |&id| id as usize + 1);
        let sampler = BatchedSampler::new(
            device.clone(),
            vocab_size,
            live_vocab,
            max_recent_len,
            eos_tokens.clone(),
            penalty_log_path,
        )
        .with_reference_flags(reference_flags(&tokenizer, vocab_size));

        let chunk_size = CHUNK_SIZE;
        // Before the tokenizer moves into `Self`. A multi-token spelling yields
        // `None` and the reasoning span is simply never located — see the field.
        // **Tokens that end an index page, whether decoded or prefilled.**
        //
        // Resolved by name, because the ids are not in any config — the turn
        // closer in particular is a dialect marker the checkpoint never
        // declares (`<|im_end|>` is 248046 here and appears nowhere in the
        // metadata). A spelling this tokenizer does not have as a single token
        // simply contributes no boundary.
        //
        // Three is the whole list, and a `Vec` rather than a set because a
        // linear scan over three `u32`s beats hashing one.
        //
        // `</think>` is taken from this same lookup rather than resolved again:
        // it is one fact — which id closes the reasoning — read for two
        // purposes, and resolving it twice is two places for it to disagree.
        let [think_open, think_close, turn_close] =
            ["<think>", "</think>", "<|im_end|>"].map(|name| {
                let id = tokenizer.token_to_id(name);
                if id.is_none() {
                    tracing::warn!(
                        "tokenizer has no single `{name}` token — it will not break an index \
                         page, so a region it bounds cannot be windowed out of a projection"
                    );
                }
                id
            });
        if think_close.is_none() {
            tracing::warn!(
                "tokenizer has no single `</think>` token — sealed turns will carry no \
                 reasoning span, so a projected turn is injected whole"
            );
        }
        let page_break_tokens: Vec<u32> = [think_open, think_close, turn_close]
            .into_iter()
            .flatten()
            .collect();
        tracing::info!(
            ?page_break_tokens,
            "index page breaks resolved by name (<think>, </think>, turn closer)"
        );
        let pool_size = PoolSize::for_card(
            model
                .device()
                .mem_get_info()
                .map_or(0, |(_, total)| total as u64),
        );
        Self {
            rx,
            model,
            session,
            gallery_arena,
            state_census: None,
            gallery_sweep: gallery_sweep::SweepState::default(),
            tag_ids: TagIds::of(&tokenizer),
            tokenizer,
            think_close,
            page_break_tokens,
            pool_size,
            eos_tokens,
            device,
            active_decodes: HashMap::new(),
            sampling_states: HashMap::new(),
            prefill_queue: VecDeque::new(),
            pending_submits: VecDeque::new(),
            waiting_told: None,
            // See the field: the first pass has nothing to wait for.
            settled_since_admit: true,
            settle_owed: false,
            settled_ground: None,
            settle_waiters: Vec::new(),
            norm_warm: NormWarm::default(),
            last_kv_compaction: None,
            kv_compaction_stall: compaction_stall::CompactionStall::default(),
            kv_gap: kv_gap::KvGapLine::default(),
            wave_forwards: 0,
            wave_overhead_us: 0,
            wave_rate: None,
            expert_hits_seen: 0,
            expert_misses_seen: 0,
            active_prefills: Vec::new(),
            section_queue: VecDeque::new(),
            active_section_ingests: Vec::new(),
            section_positional: SectionPages::default(),
            sampler,
            show_special_tokens,
            health_config,
            chunk_size,
            max_prefill_pass_tokens,
            rate_width: None,
            refused_width: None,
            refused_depth: None,
            slot_conversations: HashMap::new(),
            detached: Detached::default(),
            slot_targets: HashMap::new(),
            ephemeral_slots: std::collections::HashSet::new(),
            slot_bands: HashMap::new(),
            ephemeral_sigs: HashMap::new(),
            turn_views: HashMap::new(),
            turn_groups: HashMap::new(),
            next_turn_group: 0,
            carried_beliefs: HashMap::new(),
            guests,
            pending_reprojections: Vec::new(),
            refeeds: HashMap::new(),
            first_readings: HashMap::new(),
            slot_tokens: HashMap::new(),
            slot_projection_state: HashMap::new(),
            cold_load_stager: ColdLoadStager::with_preallocation(PINNED_PREALLOC_BYTES),
            persist_triggers: PersistenceTriggers::new(FIRST_ATTACHMENT, persist_trigger),
            attached_substrates: Vec::new(),
            summariser_trigger,
            pending_section_quantize: Vec::new(),
            quantize_pinned_scratch: preallocate_pinned_scratch(
                PINNED_PREALLOC_BYTES,
                "scheduler::quantize_pinned_scratch",
            ),
            warm_lift_ring: None,
            boundary_markers,
            prov_fold,
            prov_capture_layers,
            wave_stats: WaveStats::new(),
            compression_jobs: HashMap::new(),
            pending_compression_seals: HashMap::new(),
            next_compression_job_id: 0,
            admit_budget: MAX_PREFILL_WIDTH as u64 * admission::admit_quantum(),
            host_ram_probe: None,
            admit_grow_streak: 0,
            admit_ok_tokens_seen: 0,
            last_level_cut: None,
            promote_ok_tokens_seen: 0,
            promote_last_progress: None,
            ingest_timelines: HashSet::new(),
            batch_drain_gap_fills: false,
            deferred_glue_fires: Vec::new(),
            gallery_misses_seen: None,
            gallery_miss_rate: 0.0,
            gallery_logged_at: None,
            gallery_evicted_at: None,
            held_fit_claims: None,
            rates_logged_at: None,
            rates_priced_at: None,
            rates_width_hints: [None; 4],
            idle_since: HashMap::new(),
            high_drain: high_drain::HighDrain::default(),
            turn_content_seen: 0,
            stencil_rows: Vec::new(),
            urgent_hold: urgent_hold::UrgentHold::default(),
            step_rows: 0,
            parked_forks: Vec::new(),
            sealed_snapshots: HashMap::new(),
            parked_at: HashMap::new(),
            wave_prefill_residual: None,
            wave_prefill_cursor: 0,
            wave_prefill_members: Vec::new(),
            wave_prefill_rows: Vec::new(),
            paused_creep: None,
            prefill_gang: Vec::new(),
            request_laps: RequestLaps::default(),
            wave_sweep_failures: 0,
            sweep_forecast: sweep_forecast::SweepForecast::default(),
            prefill_waits: prefill_wait::PrefillWaits::default(),
            wave_step_open: None,
            wave_cohort_advanced: false,
            creep_owed: Duration::ZERO,
            creep_owed_band: DecodePriority::Low,
            wave_step_band: DecodePriority::Low,
            decode_only_step: None,
            width_steps: step_census::WidthSteps::default(),
            creep_layer_row_cost: None,
            creep_layer_cost: None,
            wave_step_creep_layer_rows: 0,
            wave_step_creep_layers: 0,
            section_relief_pacing: ReliefPacing::default(),
            reproject_census: ReprojectCensus::new(Instant::now()),
            pool_read_ring: PoolReadRing::for_pool(pool_size),
            wave_step_creep_layer_pairs: 0.0,
            wave_retry: wave_retry::WaveRetry::default(),
            decode_retry: wave_retry::WaveRetry::default(),
            wave_cobatched: false,
            wave_step_ratio: 1.0,
            wave_step_slice: 1.0,
            high_airtime: None,
            creep_airtime: turn_airtime::AirtimeRate::default(),
            segment_mean: turn_airtime::SegmentMean::default(),
            prompt_mean: turn_airtime::SegmentMean::default(),
            tokens_per_step: turn_airtime::SegmentMean::default(),
            step_census: step_census::StepCensus::default(),
            high_backlog_rows: 0,
            wave_section_advanced: false,
            shutdown_requested: false,
            section_name_cache: HashMap::new(),
            compression_event_sinks: HashMap::new(),
            timeline_projections: HashMap::new(),
            workspace_projection: None,
        }
    }

    /// Record `conversation`'s substrate as attached to this core, once.
    pub fn register_substrate(&mut self, conversation: &Conversation) {
        // A dialogue's slot is the engine's trained window when its attention
        // selects (`docs/hit_driven_projection.md` §4): the projection fills
        // to it.
        let trained = self
            .model
            .carries_positional_state()
            .then(|| self.model.rope_select().ceilings().first().copied())
            .flatten();
        conversation.update(|s| s.set_trained_window(trained));
        if !self
            .attached_substrates
            .iter()
            .any(|c| c.same_substrate(conversation))
        {
            self.attached_substrates.push(conversation.clone());
        }
    }

    /// Rebuild the workspace substrate from the persistence redo log on
    /// daemon startup (§16.12 substrate reload).
    ///
    /// **Cold-only restart.** Every persisted turn stream is recovered in
    /// `(timeline, turn_index)` order; for each, tokens + wide-Q signatures
    /// are replayed into the in-RAM substrate and the turn is registered
    /// cold-marker (`hot/warm = None`, `cold = Some(...)` from the
    /// manifest). The KV bytes stay on disk until the runtime inject
    /// path demand-materialises them via [`elevate_to_hot`].
    pub fn reconstruct_substrate(
        &self,
        conversation: &Conversation,
        status: &SubstrateReloadStatus,
    ) {
        let n_layers = self.session.backings().len();
        if n_layers == 0 {
            // Nothing to replay (CPU/test backings) — unblock the waiter.
            status.finish();
            return;
        }
        let progress = |done: usize, total: usize| status.record(done, total);
        let result = conversation.reconstruct_from_log(n_layers, Some(&progress));
        // Always unblock the daemon's loading-screen waiter, success or not.
        status.finish();
        match result {
            Ok(0) => {}
            Ok(n) => tracing::info!("substrate reload: {n} turns restored from redo log"),
            Err(e) => tracing::error!("substrate reload failed: {e}"),
        }
    }
}
