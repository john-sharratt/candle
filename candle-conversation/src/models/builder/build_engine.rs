//! Turning a builder's settings into an engine.
//!
//! The host-to-device link is measured before the model loads, while the card is
//! empty and the bus idle, because a reading taken during the load would compete
//! with the load's own reservation. The engine configuration is sized to the card
//! where it is made. The model loads before the substrate opens, so the expert
//! cache pins its warm tier against a machine that holds only the model, not the
//! substrate's in-memory index; a substrate another model wrote is refused when
//! it opens, after the load. An engine attached from another tenant takes that
//! engine's RoPE schedule and its merged settings, not the builder's own.

use super::*;

/// `device`'s pinned host→device copy rate in bytes a second, or 0 where there
/// is no link to measure (a CPU device, a probe that failed).
///
/// Measured the first time it is asked for and cached (`link_probe`), so the
/// first ask decides when it is read. **That ask is before the model loads** —
/// `load_model`'s first step, and `build`'s engine config ahead of its load —
/// with the card empty and the bus idle: later the reading is contended by the
/// expert cache's staging, and its landing window competes with a full
/// reservation for ground it would then take from the driver.
pub(super) fn link_rate(device: &Device) -> f64 {
    let Device::Cuda(d) = device else {
        return 0.0;
    };
    // The probe is the load's first step, and its window is the load's ground
    // (`docs/driver_model.md` §4): a driver allocation freed before the span is
    // claimed. Once the figure is cached this measures nothing.
    let _load = LoadWindow::open(d.cuda_stream().context().ordinal());
    match link_probe::host_to_device_bytes_per_s(d) {
        Ok(bps) => {
            tracing::info!(link_gbps = bps / 1e9, "host→device link measured");
            bps
        }
        Err(e) => {
            tracing::warn!("host→device link probe failed: {e}");
            0.0
        }
    }
}

// Read-back & terminal methods

impl ModelBuilder {
    /// Build an [`EngineConfig`] from the builder's settings, a tokenizer, the
    /// device the engine will run on — which sizes the prefill forward to the card
    /// unless [`Self::prefill_pass_tokens`] chose it — and the substrate
    /// [`Self::bind_substrate`] bound to this model.
    pub fn engine_config(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        device: &Device,
        substrate: BoundSubstrate,
    ) -> EngineConfig {
        let mut eos_tokens = Vec::new();
        // The spec's dialect is the source of truth for the model family's
        // end-of-turn/document markers (e.g. DeepSeek's `<｜end▁of▁sentence｜>`);
        // the fixed list covers the families whose dialects predate it.
        let dialect_ends = [
            self.spec.dialect.document_end,
            self.spec.dialect.turn_end,
            self.spec.dialect.assistant_end,
        ];
        for token in dialect_ends
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .chain([
                "<|im_end|>",
                "<|endoftext|>",
                "<|end_of_text|>",
                "<|eom_id|>",
                "<|eot_id|>",
                "<|reserved_special_token_31|>",
            ])
        {
            if let Some(id) = tokenizer.token_to_id(token) {
                if !eos_tokens.contains(&id) {
                    tracing::trace!("✅ EOS token registered: '{}' = ID {}", token, id);
                    eos_tokens.push(id);
                }
            } else {
                tracing::debug!("⚠️  EOS token NOT FOUND in tokenizer: '{}'", token);
            }
        }
        if eos_tokens.is_empty() {
            tracing::error!("No EOS tokens found!");
        }

        // Auto-detect vocab size from tokenizer (includes special tokens).
        // This MUST match the logits dimension from the model, otherwise
        // penalty buffers will be undersized and the CUDA kernel will
        // read out-of-bounds GPU memory.
        let tok_vocab = tokenizer.get_vocab(true).len();
        // Qwen3 (and several other models) ship a tokenizer vocab smaller
        // than the model's output projection (e.g. 151_669 vs 151_936).
        // The model's logits dimension is authoritative — read it from the
        // model file's header when this builder names one; fall back to the
        // tokenizer length otherwise.
        let vocab_size = match self
            .model_path
            .as_deref()
            .map(Self::detect_sampling_from_gguf)
        {
            Some(Ok(info)) => info.vocab_size.unwrap_or(tok_vocab).max(tok_vocab),
            _ => tok_vocab,
        };
        tracing::info!(
            "Vocab size: tokenizer={} -> using={}",
            tok_vocab,
            vocab_size
        );

        let mut ret = EngineConfig::new(eos_tokens.into(), substrate);
        ret.cancel = self.cancel.clone();
        ret.tenant_rank = self.tenant_rank;
        ret.layer_corrupt_turn = self.layer_corrupt_turn.clone();
        // **Sized to the card here, where every engine config is made**, so an
        // engine assembled from this config directly — the fragmentation probe,
        // `ruler_stream` — gets the same pass as one from `build`: by the card's
        // VRAM and its host→device link (`prefill_width`).
        //
        // The link is measured before the model loads (`link_rate`, which
        // `load_model` asks first and `build` asks before it loads), and this
        // reads the cached figure.
        let link = link_rate(device);
        // **The card's width is a ceiling, and an ask is clamped to it.** Tenants
        // ask for what their workload wants — npcd the model's ceiling, so a slab
        // carries several documents — and the merge takes the largest ask
        // (`CoreRequests::merge`). What the card can carry is a different
        // question: past it each window's tier is bought from the expert zone,
        // and decode pays for it after the window. On BlackWorld npcd's 8,192
        // overrode the card's 3,072 and every window conceded ~8 GB of experts.
        let total_vram = match device {
            Device::Cuda(d) => d
                .mem_get_info()
                .map(|(_free, total)| total as u64)
                .unwrap_or(0),
            _ => 0,
        };
        let card = prefill_pass_tokens(total_vram, link);
        let n = self
            .prefill_pass_tokens
            .map_or(card, |asked| asked.min(card));
        tracing::info!(
            total_vram_gib = total_vram / (1 << 30),
            link_gbps = link / 1e9,
            card_prefill_pass_tokens = card,
            asked_prefill_pass_tokens = ?self.prefill_pass_tokens,
            prefill_pass_tokens = n,
            "prefill forward sized to the card and its link"
        );
        ret.scheduler.large_prefill_max_tokens = n;
        ret.batched_config.compression_level = Some(self.kv_compression_level);
        // Stress test: uniform-K pin REMOVED — both K and V now use fully
        // adaptive per-(head,palette) selection with non-identity pal_maps,
        // exercising the palette-straddle decode path on BOTH sides.
        // (Production pins K to a uniform Q8_KS/identity layout because the
        // provenance recall K-signature reader assumes it; adaptive K measurably
        // degrades recall. Restore `Some(QuantFormat::Q8_KS)` if recall
        // regresses — the duplication control asserts on verbatim reproduction,
        // not recall quality, so this isolates the straddle path.)
        ret.batched_config.override_k_quant = None;
        ret.batched_config.override_v_quant = None;
        // The width this architecture's own reference implementation computes
        // in, where it is known. Without it the session derives the activation
        // width from the KV storage format — which reports F16 for the
        // quantized backing this engine always uses, regardless of what the
        // model was trained and published in. See
        // `ModelArch::native_activation_dtype`.
        ret.batched_config.activation_dtype = self.spec.arch.native_activation_dtype();
        ret.vocab_size = vocab_size;
        ret.max_concurrent_conversations = self.max_concurrent;
        ret.show_special_tokens = self.show_special_tokens;
        ret.penalty_log_path = self.penalty_log_path.clone();
        ret.health = self.health_config.clone();
        // The engine uses the model's dialect to pre-tokenise the
        // inter-turn boundary markers once at scheduler construction.
        ret.dialect = self.spec.dialect.clone();
        ret
    }

    /// Open this builder's substrate — the one [`Self::substrate`] handed over,
    /// or the one under [`Self::workspace_path`] (the process working directory
    /// when unset), read-only when [`Self::read_only_substrate`] says so — and
    /// bind it to `identity`. A substrate another model wrote is refused here.
    pub fn bind_substrate(&self, identity: &ModelIdentity) -> crate::Result<BoundSubstrate> {
        let shared = match &self.substrate {
            Some(shared) => shared.clone(),
            None => {
                let dir = match &self.workspace_path {
                    Some(p) => p.clone(),
                    None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                };
                // A read-only open writes nothing under `substrate/` and
                // requires the store to exist. A writable one refuses a store
                // this model cannot open before it rewrites anything.
                let opened = if self.read_only_substrate {
                    SharedSubstrate::open_in_read_only(&dir)
                } else {
                    let report = |stage: OpenStage, done: usize, total: usize| {
                        if let Some(hook) = &self.open_progress {
                            hook.report(stage, done, total);
                        }
                    };
                    SharedSubstrate::open_in_reporting(&dir, Admit::Key(identity.key()), &report)
                };
                opened
                    .map_err(|e| ConversationError::Other(format!("substrate persistence: {e}")))?
            }
        };
        if let Some(retention) = self.tombstone_retention {
            shared
                .persistence
                .call(|p| p.set_tombstone_retention(retention));
        }
        BoundSubstrate::bind(shared, identity).map_err(|e| ConversationError::Other(e.to_string()))
    }

    /// Resolve the model pack, load the model and its tokenizer, and build the
    /// engine.
    ///
    /// If [`model_path`](Self::model_path) / [`tokenizer_path`](Self::tokenizer_path)
    /// (or [`model_dir`](Self::model_dir)) have been set, those local files
    /// are used. Otherwise the spec's pack is taken from the model cache, or
    /// built from its sources (requires the `hub` crate feature).
    pub fn engine(&mut self, device: &Device) -> crate::Result<ConversationEngine> {
        self.engine_with_progress(device, None)
    }

    /// Same as [`Self::engine`] but told how the model load goes: each
    /// [`LoadPhase`] as it begins — the first before the weights load — and how
    /// far through it is. A loader that counts nothing within a phase
    /// still says when it begins.
    pub fn engine_with_progress(
        &mut self,
        device: &Device,
        progress: Option<LoadProgress<'_>>,
    ) -> crate::Result<ConversationEngine> {
        let prepared = self.prepare(device)?;
        self.engine_from_prepared(prepared, device, progress)
    }

    /// Attach this builder's substrate to an engine another tenant built
    /// (`crate::engine_slot`): resolve and read the model pack exactly as a
    /// build would — so this builder's sampling, thinking and context settings
    /// are the ones the model states — then bind the substrate to the model and
    /// attach it. No weight is loaded. `device` is the engine's: a pack is
    /// resolved for the card it runs on.
    ///
    /// The substrate is bound under the engine's RoPE schedule, not this
    /// builder's: the engine rotates every key this substrate will hold, and a
    /// shared engine's schedule is every tenant's merged. The model itself is
    /// checked against the engine's before the substrate is touched, so a
    /// different model leaves a fresh substrate unbound.
    pub fn attach_to(
        &mut self,
        attacher: &EngineAttacher,
        device: &Device,
    ) -> crate::Result<ConversationEngine> {
        let prepared = self.prepare(device)?;
        let weights = self.weight_files(&prepared.pack)?;
        let engine_key = attacher.model_key();
        let identity = self
            .model_identity(&weights)?
            .under_rope(engine_key.rope.clone());
        if !identity.key().same_model(engine_key) {
            return Err(ConversationError::Other(format!(
                "this tenant resolves {}, and the engine it would attach to runs {}",
                identity.key().describe(),
                engine_key.describe()
            )));
        }
        let substrate = self.bind_substrate(&identity)?;
        attacher.attach(
            substrate,
            self.cancel.clone(),
            self.tenant_rank,
            self.layer_corrupt_turn.clone(),
        )
    }

    /// Give up the engine-wide settings this builder asks for, for the slot to
    /// merge with other tenants' (`crate::engine_slot::CoreRequests`). The
    /// guests move out; [`Self::apply_core_requests`] gives the merge back.
    pub fn take_core_requests(&mut self) -> CoreRequests {
        CoreRequests {
            prefill_pass_tokens: self.prefill_pass_tokens,
            qsa_selection_budget: self.qsa_selection_budget,
            qsa_strata: self.qsa_strata,
            hit_layers: self.hit_layers.clone(),
            max_response_tokens: self.max_response_tokens,
            max_concurrent: self.max_concurrent,
            thinking: self.thinking,
            guests: std::mem::take(&mut self.guests),
        }
    }

    /// Build with `merged` — every tenant's engine-wide settings — in place of
    /// this builder's own.
    pub fn apply_core_requests(&mut self, merged: CoreRequests) {
        self.prefill_pass_tokens = merged.prefill_pass_tokens;
        self.qsa_selection_budget = merged.qsa_selection_budget;
        self.qsa_strata = merged.qsa_strata;
        self.hit_layers = merged.hit_layers;
        self.max_response_tokens = merged.max_response_tokens;
        self.max_concurrent = merged.max_concurrent;
        self.thinking = merged.thinking;
        self.guests = merged.guests;
    }

    /// Rank this builder's substrate as tenant `rank` — [`Self::tenant_rank`]
    /// through a reference, for the engine slot.
    pub(crate) fn set_tenant_rank(&mut self, rank: TenantRank) {
        self.tenant_rank = rank;
    }

    /// Build the engine from a [`Self::prepare`]d model pack.
    pub(super) fn engine_from_prepared(
        &mut self,
        prepared: Prepared,
        device: &Device,
        progress: Option<LoadProgress<'_>>,
    ) -> crate::Result<ConversationEngine> {
        let Prepared {
            pack,
            tokenizer,
            gguf_vocab_size,
        } = prepared;

        // The model's identity comes from the pack's header — its sources'
        // digests and its tokenizer — in milliseconds, and nothing from here
        // to the load needs more than that.
        let weights = self.weight_files(&pack)?;
        let identity = self.model_identity(&weights)?;
        // Everything up to the substrate open is the model: say so now, before
        // the load, rather than at the loader's first count.
        if let Some(report) = progress {
            report(LoadPhase::Weights, 0, 0);
        }

        // ── Load the model before the substrate opens ──────────────────
        // The expert cache sizes its pinned warm tier from the host RAM free
        // when it loads (`expert_lre::handle::sizing`), and pinned pages are
        // never given back. A substrate opened first takes its in-memory index
        // — tens of GB on a long-lived store, nearly all of it cold — out of
        // that figure for good: on WhiteWorld (2026-11-22) zend's open left
        // 11.5 GB free, the tier pinned 4,161 of 24,064 experts, and decode
        // missed to the pack. Loaded first, the tier takes its pin from a
        // machine that holds only the model, and the substrate's cold pages are
        // the ones the OS moves to the pagefile instead.
        //
        // Every driver allocation the load makes, by site: most are scratch the
        // load frees again, and the ones it keeps are what the CUDA pool still
        // holds outside the reservation once serving starts.
        #[cfg(feature = "forbidden_allocations")]
        let load_watch = candle::forbidden_alloc::armed();
        let model = self.load_model(&weights, device, progress)?.model;
        #[cfg(feature = "forbidden_allocations")]
        {
            drop(load_watch);
            let report = candle::forbidden_alloc::take_report();
            tracing::warn!(
                target: "candle_conversation::scheduler::forbidden_alloc",
                calls = report.total_calls,
                mib = report.total_bytes >> 20,
                "the load allocated outside the reservation\n{report}"
            );
        }

        // Hand the load's pool high-water back before serving starts — it is
        // several GiB held outside the KV reservation and never used again.
        // See `candle::vram::trim_pool_after_load` for why here and nowhere
        // later.
        if let Some((before, after)) = candle::vram::trim_pool_after_load(device) {
            tracing::info!(
                reclaimed_mib = before.saturating_sub(after) >> 20,
                pool_reserved_mib = after >> 20,
                "post-load: returned the load's pool high-water to the OS"
            );
        }

        // ── Bind the substrate ─────────────────────────────────────────
        // A substrate another model wrote is refused here, when its walk ends
        // (`SubstratePersistence::admit`), and the loaded model is dropped
        // with the error.
        let substrate = self.bind_substrate(&identity)?;
        let mut config = self.engine_config(&tokenizer, device, substrate);

        // Override vocab_size with the authoritative value from GGUF metadata
        // if available.  Models often pad vocab to a power-of-2 / multiple of
        // 128 for GPU efficiency, so tokenizer.get_vocab().len() can be smaller
        // than the model's actual logits dimension.
        if let Some(model_vocab) = gguf_vocab_size {
            if model_vocab != config.vocab_size {
                tracing::info!(
                    "Overriding vocab_size: tokenizer={} → GGUF metadata={}",
                    config.vocab_size,
                    model_vocab
                );
            }
            config.vocab_size = model_vocab;
        }

        // ── Log the effective configuration ────────────────────────────
        {
            let model_name = self
                .spec
                .model_filename
                .strip_suffix(".gguf")
                .unwrap_or(&self.spec.model_filename);
            tracing::info!("Engine config · {}", model_name);
            tracing::info!(
                "  Arch: {:?}   vocab: {}   max_seq: {}",
                self.spec.arch,
                config.vocab_size,
                self.max_seq_len
            );
            tracing::info!(
                "  Sampling: temp={:.3}  top_k={}  top_p={:.3}  repeat_penalty={:.3}",
                self.sampling.temperature,
                self.sampling.top_k,
                self.sampling.top_p,
                self.sampling.repeat_penalty,
            );
            tracing::info!("  Max response tokens: {}", self.max_response_tokens);
        }

        // Auto-derive max_hot_turns from arena geometry unless the caller
        // overrode it. Must happen before conversation_config() is called below.
        if self.max_hot_turns == 0 {
            // A hot turn's active K/V is F16, so the F16 size class is what
            // bounds how many turns a region holds. The class is a function of
            // the *model's* palette sub-band width — `head_dim / N_PALETTE` —
            // not of `CHUNK_SIZE`; those coincide only at `head_dim == 128`.
            let head_dim = model.model_core_properties().head_dim;
            let elems = elems_per_chunk((head_dim / N_PALETTE).max(1));
            let class = class_for_format(KvFormat::Float(DType::F16), elems).ok_or_else(|| {
                ConversationError::Other(format!(
                    "no size class holds an F16 chunk of {elems} elements \
                     (head_dim {head_dim}); the ladder's top rung is \
                     {} bytes",
                    SizeClass::at(SizeClass::COUNT - 1).bytes(),
                ))
            })?;
            let arena_chunks = class.chunks_per_region();
            self.max_hot_turns =
                pick_max_hot_turns(arena_chunks, CHUNK_SIZE, self.max_response_tokens);
            tracing::debug!(
                "auto max_hot_turns: {} (arena {} × chunk {} / {} max_response_tokens)",
                self.max_hot_turns,
                arena_chunks,
                CHUNK_SIZE,
                self.max_response_tokens,
            );
        }

        // A hub build fetches the model's own guests' weights the way it
        // fetches the model's; one without the hub has no way to.
        #[cfg(feature = "hub")]
        model_guests::register_model_guests(&self.spec.arch, &mut self.guests);
        let engine = ConversationEngine::new(model, tokenizer, config, self.guests.clone())?;
        Ok(engine)
    }
}
