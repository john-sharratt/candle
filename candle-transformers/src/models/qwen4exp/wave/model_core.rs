//! The per-sequence state Qwen4ExpBatched carries beside its paged K/V, and the setup run once when the model is built.
//!
//! The four carried classes divide on whether storing or rebuilding is cheaper. The GDN recurrence must be stored, the PLE history is kilobytes kept in the snapshot's auxiliary blob, and the QSA index is stored with the live tail because its keys cannot be rebuilt from K/V. Draft seeds are neither stored nor rebuilt, so a resumed sequence drafts nothing until its first wave seeds it. Each sequence's index is checked against its K/V on every wave entry: a short index makes selection refuse at depth, and a long one claims positions the sequence does not hold. The Q/K norm weights are converted once at session creation, never inside a wave.

use super::*;

/// The carried per-sequence state, and what persisting it costs.
///
/// This model carries four classes outside the paged K/V, and they divide on
/// one question — **is it cheaper to store or to rebuild?**
///
/// * **GDN recurrence** — an accumulated sum with no per-token decomposition.
///   Not derivable from anything; must be stored. [`RecurrentStateStore`]
///   already exports and imports it, geometry-checked against a schedule hash,
///   and is shared with the hybrid lineage.
/// * **PLE** — a `(conv_kernel − 1) × ngram_size` row convolution history plus
///   the last `ngram_size − 1` token ids. Kilobytes: stored, in the snapshot's
///   opaque auxiliary blob.
/// * **QSA index** — `n_blocks × head_dim` keys per attention layer, which at
///   128K tokens and ratio 4 is ~16 MiB a layer and ~192 MiB a sequence. That
///   is not a per-turn snapshot. The keys are the indexer's own projection of
///   each layer's hidden states, not a function of K, so they cannot be rebuilt
///   from resident K/V: the live tail is stored in the seal's auxiliary blob
///   (`seal_live`, restored by [`Self::restore_aux_state`]) and the injected
///   pages ahead of it come back with the projection that injects their K/V. An
///   index dropped any other way costs a re-prefill of its context.
/// * **Draft seeds** — one residual row per sequence, and only meaningful
///   inside the wave that produced it. Neither stored nor rebuilt: a resumed
///   sequence simply drafts nothing until its first wave seeds it, which costs
///   one step of speculation and no correctness.
impl Qwen4ExpBatched {
    /// Materialise the norm weights this model meets activations with.
    ///
    /// **Two widths, because this model computes wider than it stores.** The
    /// residual stream is BF16 (the lineage publishes `"dtype": "bfloat16"`),
    /// while a live sequence's K/V sits in the arena at F16 — `R16` is raw F16
    /// with Q-capture space. `kv` is the second, and passing the first in its
    /// place is not a rounding difference: the per-head Q/K norms run INSIDE
    /// the projection, on operands that *become* the arena's bytes, so a
    /// BF16 weight there meets an F16 activation and `weight_for` refuses it —
    /// every wave, from the first ingest, with the model fully loaded and
    /// nothing else wrong.
    ///
    /// The Gated Residual runs F32 end to end and the projections quantize
    /// their own activations, so the Q/K norms are the only weights that need
    /// materialising; both are done here, at session creation, never inside the
    /// wave.
    ///
    /// The draft head's block is included deliberately. It is a full-attention
    /// layer with Q/K norms of its own and it was POPPED out of `layers` at
    /// load, so a loop over `layers` alone leaves exactly one attention layer
    /// holding unconverted norms, and the head's first pass fails inside the
    /// wave rather than here.
    pub fn maybe_change_dtype(&self, _act: DType, kv: DType) -> Result<()> {
        let head_block = self.model.mtp.as_ref().map(|h| &h.block);
        for layer in self.model.layers.iter().chain(head_block) {
            if let GpuLayerMix::Attention { w, .. } = &layer.mix {
                w.q_norm.maybe_change_dtype(kv)?;
                w.k_norm.maybe_change_dtype(kv)?;
            }
        }
        Ok(())
    }

    pub fn new(model: Qwen4ExpGpu) -> Result<Self> {
        // Built once over the loaded model: part of its load.
        let _load = dense_span::open_window(&model.device);
        let schedule = flash_next_schedule(&model.cfg)?;
        let rope = RopeRungs::new(&schedule, &model.device)?;
        // The indexer rotates with the attention's own rungs at the same rotary
        // width, as the reference does.
        let index_rope = FactoredRope::over(&rope, &model.device)?;
        // The card's total VRAM picks its place between the swept cards, read
        // once: a context bind per step would cost the decode loop.
        let (_, total) = model.device.mem_get_info()?;
        let draft = DraftTable::new(blend(
            &flash_next_sweep::CARD_24GB,
            &flash_next_sweep::CARD_72GB,
            (total / (1 << 20)) as u64,
        ));
        let mut built = Self {
            model,
            draft,
            recurrent: RwLock::new(HashMap::new()),
            arms: RwLock::new(SpanArms::default()),
            seeds: RwLock::new(SeedStore::new()),
            seed_spares: RwLock::new(SeedStore::new()),
            draft_carry: Mutex::new(None),
            parked_rows: Mutex::new(None),
            seeded: RwLock::new(HashSet::new()),
            pristine: RwLock::new(HashSet::new()),
            staged: RwLock::new(HashMap::new()),
            parked: RwLock::new(HashSet::new()),
            verify: RwLock::new(None),
            ple: RwLock::new(HashMap::new()),
            index: RwLock::new(HashMap::new()),
            pages: PageRegistry::default(),
            seal_memo: SealMemo::default(),
            retired: RwLock::new(HashMap::new()),
            pooled: SequencePools::default(),
            index_rope,
            qsa_rows: AtomicU64::new(0),
            rope,
            step: Mutex::new(None),
            hit_kv: Vec::new(),
            hits: RwLock::new(HashMap::new()),
            hit_table: Mutex::new(None),
        };
        built.hit_kv = built.sampled_kv(&HitPercentiles::standard());
        Ok(built)
    }

    pub fn engine(&self) -> &Qwen4ExpGpu {
        &self.model
    }

    /// Query rows QSA has narrowed since load — see [`Self::qsa_rows`].
    pub fn qsa_rows_selected(&self) -> u64 {
        self.qsa_rows.load(Ordering::Relaxed)
    }

    /// Fresh-or-reset per-sequence state, keyed exactly as the oracle keys its
    /// sessions: a member arriving at offset 0 starts over.
    ///
    /// `tokens` is what the sequence will hold once this wave lands; the index
    /// caches are grown against the position that puts its next token at —
    /// `position` plus the wave's rows, past the slot's holes as the caches'
    /// tails are. Growth claims arena slots, and a claim inside the layer loop
    /// would move ground a placed tier is standing on (hot-path invariant 7).
    /// Admission is the place for it.
    ///
    /// **The reset needs `layer_start == 0` as well as `offset == 0`.** A sweep
    /// runs once per `forward_wave` layer window, and a creeping prefill does
    /// not advance the session's offset until it reaches the head — so every
    /// window of a fresh sequence's prompt arrives here at offset 0. Keyed on
    /// the offset alone, window 2 reset what window 1 had just built: the PLE
    /// layer's `conv_hist`/`prev`, the DeltaNet `S` those layers had
    /// accumulated over the prompt, and their filled QSA index caches. Only the
    /// last window's layers would then hold real state, the earlier ones
    /// standing at sequence-start as though the prompt had never run — with
    /// every shape still matching and nothing raised.
    ///
    /// `position` is where the sequence's next token is read — `offset` plus
    /// the holes its slot leaves (`docs/placed_section_layout.md` §4.1) — and is
    /// what the index's coverage is checked against, as its pages are placed.
    pub(super) fn ensure_seq_state(
        &self,
        seq: usize,
        offset: usize,
        position: usize,
        tokens: usize,
        layer_start: usize,
    ) -> Result<()> {
        // A wave run on a sequence whose snapshot is still held — one no claim
        // reached, such as a projection's own prefill on its slot — builds the
        // store from the snapshot before anything reads it.
        self.refuse_parked(seq)?;
        self.install_staged(seq)?;
        let cfg = &self.model.cfg;
        let hc_dim = cfg.hc.count * cfg.hidden_size;
        // Consumed here, once, whether or not it fires — see `seeded`. Taken
        // before the four class blocks so all of them see the same answer: a
        // flag read per class would be true for the first and false for the
        // rest, resetting three quarters of a restored sequence.
        let seeded = layer_start == 0
            && self
                .seeded
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: seeded lock poisoned".into()))?
                .remove(&seq);
        let starting_over = offset == 0 && layer_start == 0 && !seeded;
        {
            let mut idx = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            let caches = match idx.entry(seq) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                // One per KV layer, the draft head's included: the head selects
                // over its own cache exactly as a trunk attention layer does,
                // and its layer index — past every trunk layer — is also its
                // index here.
                std::collections::hash_map::Entry::Vacant(e) => e.insert(
                    (0..cfg.kv_layers().total())
                        .map(|_| IndexCache::new(cfg.indexer.head_dim, &self.model.device))
                        .collect::<Result<Vec<_>>>()?,
                ),
            };
            // **A sequence-start reset must not throw away installed pages.**
            //
            // `starting_over` means "this slot is at position 0 and nobody
            // declared its state deliberate", which for the index is only true
            // before anything was injected. A projection installs its pages and
            // THEN fills the K/V, so a wave that lands between those two steps
            // sees offset 0, resets, and silently discards the rows for a prefix
            // the slot is about to hold — leaving pages 0 against a K/V of
            // hundreds of tokens, invisible until the select refuses at depth.
            let injected_before_reset = starting_over
                && caches
                    .iter()
                    .zip(self.attention_ratios())
                    .any(|(c, ratio)| ratio > 0 && c.page_row_span() > 0);
            // Where the next token is read once this wave lands: the caches'
            // tails open at a position, past the slot's holes, and are grown
            // against one.
            let landed = position + tokens.saturating_sub(offset);
            for (cache, ratio) in caches.iter_mut().zip(self.attention_ratios()) {
                if starting_over {
                    // The prompt end is a declaration about what this slot is
                    // about to hold (`set_selection_prompt`), made before its
                    // first wave, so the sequence-start reset keeps it.
                    let prompt_end = cache.prompt_end();
                    cache.reset();
                    cache.set_prompt_end(prompt_end);
                }
                if ratio > 0 {
                    cache.ensure_capacity(landed, ratio)?;
                }
            }
            if injected_before_reset {
                tracing::warn!(
                    seq,
                    offset,
                    tokens,
                    "qwen4exp sequence-start reset discarded injected index pages — the \
                     slot keeps the K/V they described and loses every row, so the select \
                     refuses once it passes the identity threshold"
                );
            }
            // **The index covers exactly the K/V that is already there.**
            //
            // At wave entry the sequence holds `offset` tokens and every live
            // cache must already account for all of them — as injected pages,
            // as appended rows, or as a mix. Anything less and this slot carries
            // K/V nothing indexed, which is invisible while the sequence is
            // under the identity threshold and refuses the select the moment it
            // crosses. Checked here, on host counters already in hand (no
            // launch, no readback), because `score_rows` finds it at depth —
            // long after whichever step dropped the tokens, and only for the
            // sequences that get deep enough to look.
            //
            // Comparing the caches only against EACH OTHER is not this check:
            // a fork that rebuilds a child's caches from pages leaves all
            // thirteen short by the parent's un-sealed tail, in perfect
            // agreement.
            // **Both directions.** Short is the loud failure — the select
            // refuses. Long is the silent one: the rows are there, so nothing
            // errors, and the extra ones claim positions this slot does not
            // hold, which reads as a plausible answer about the wrong context.
            // A tail restored twice lands exactly here.
            // **Only on the window that enters the wave.** A sweep is split
            // into layer windows and `ensure_seq_state` runs per window, so by
            // the second one the earlier layers have already appended this
            // wave's rows — their coverage legitimately reads `tokens`, not
            // `offset`, and comparing there reports every wide wave as
            // over-covered. The first window is the only moment at which the
            // whole stack is still standing at `offset`.
            let off: Vec<(usize, usize)> = if layer_start != 0 {
                Vec::new()
            } else {
                // **Exact, to the position.** `indexed_tokens` counts the open
                // block's carried rows as well as the completed blocks and the
                // injected pages, placed past the slot's holes as its K/V is, so
                // an index that is where its K/V is agrees with `position`
                // exactly — and one that is off by less than a block is
                // misplaced just the same (see `coverage`).
                coverage_disagreements(
                    caches
                        .iter()
                        .zip(self.attention_ratios())
                        .enumerate()
                        .filter(|(_, (_, ratio))| *ratio > 0)
                        .map(|(layer, (c, ratio))| (layer, c.indexed_tokens(ratio))),
                    position,
                )
            };
            if !off.is_empty() {
                tracing::warn!(
                    seq,
                    offset,
                    position,
                    tokens,
                    starting_over,
                    layers = ?off,
                    "qwen4exp index disagrees with this sequence's K/V at wave entry — \
                     (kv layer, positions indexed) against {position} held: short refuses \
                     the select past the identity threshold, long selects over positions \
                     the slot does not hold"
                );
            }
        }
        // State claimed at admission is already sequence-start, so a sequence
        // starting over on it keeps it; any other start-over rebuilds. Consumed
        // on the window that enters the wave, as `seeded` is.
        let pristine = layer_start == 0 && self.clear_pristine(seq);
        let rebuild = starting_over && !pristine;
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            // **A restart zeroes the store it has; it never builds another.** A
            // dropped store's slots return only once nothing views them, so a
            // rebuild asked the span for a second 224 MiB while the first was
            // still held — and on a full span that claim failed part-way and
            // failed a turn that was only starting over.
            match rec.entry(seq) {
                Entry::Occupied(mut e) => {
                    if rebuild {
                        e.get_mut().reset()?;
                    }
                }
                Entry::Vacant(e) => {
                    e.insert(RecurrentStateStore::new(
                        &cfg.layer_kinds,
                        &cfg.delta_net,
                        &self.model.device,
                    )?);
                }
            }
            // A prefill span — longer than the timeline — writes its advanced
            // state into the prefill scratch, claimed here with the other claims,
            // before the tier stands; the commit exchanges it in.
            if tokens - offset > TIMELINE_ROWS {
                rec.get_mut(&seq).expect("ensured above").ensure_scratch()?;
            }
        }
        {
            let mut ple = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            if rebuild {
                ple.remove(&seq);
            }
            if let Entry::Vacant(e) = ple.entry(seq) {
                e.insert(PleState::zeros(&cfg.ple, hc_dim, &self.model.device)?);
            }
            // The fused block's second history buffer, made here rather than in
            // the forward: a new state has none, and a rewind drops it.
            ple.get_mut(&seq).expect("inserted above").ensure_spare()?;
        }
        // A sequence starting over has no previous row for the head to read,
        // and a stale seed here is the previous conversation's state feeding
        // this one's first proposal. Dropped rather than zeroed: the head pass
        // reads a missing seed as position 0 and takes zeros itself. A pristine
        // seed was claimed zero for this sequence and is kept.
        if rebuild {
            self.seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?
                .remove(&seq);
            self.seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?
                .remove(&seq);
        }
        // The head's two seed buffers, made here rather than in the forward. A
        // sequence with no seed is one the head has never run behind, whose
        // first row reaches back to zeros (see `SeedStore`); its spare is the
        // buffer the first carry writes. A failed wave drops the spare, and
        // this replaces it before the next.
        if self.model.mtp.is_some() {
            let dims = (1, cfg.hc.count, cfg.hidden_size);
            let dev = &self.model.device;
            let mut seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            if let Entry::Vacant(e) = seeds.entry(seq) {
                e.insert(state_buffer(dev, dims, true)?);
            }
            let mut spares = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            if let Entry::Vacant(e) = spares.entry(seq) {
                // Written by the carry before it is read (invariant 6).
                e.insert(state_buffer(dev, dims, false)?);
            }
        }
        Ok(())
    }

    /// Each KV layer's QSA compression ratio, in KV-layer order — the trunk's
    /// full-attention layers, then the draft head's if the artifact carries
    /// one. The head's cache is grown and reset with the rest, so the ratios
    /// have to reach it: zipped against the caches, a short list would silently
    /// leave the head's cache never sized and its first selection reading a
    /// zero-capacity buffer.
    pub(in super::super) fn attention_ratios(&self) -> Vec<usize> {
        let mut ratios: Vec<usize> = self
            .model
            .layers
            .iter()
            .filter_map(|l| match &l.mix {
                GpuLayerMix::Attention { compress_ratio, .. } => Some(*compress_ratio),
                GpuLayerMix::DeltaNet(_) => None,
            })
            .collect();
        if let Some(head) = &self.model.mtp {
            if let GpuLayerMix::Attention { compress_ratio, .. } = &head.block.mix {
                ratios.push(*compress_ratio);
            }
        }
        ratios
    }

    /// The most any one KV layer a sweep over `[layer_start, layer_end)` runs
    /// carves for its sparse selection — the trunk's attention layers in the
    /// window, and the draft head's when the sweep reaches it. What the wave
    /// plan prices as `WaveBuffer::QsaSelection`; see
    /// [`super::select_bytes`] for why the model states it.
    ///
    /// The rows are the wave's, decode then prefill, `offsets[i]` the first
    /// position of `seq_ids[i]`'s `q_lens[i]` rows.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn wave_qsa_bytes(
        &self,
        seq_ids: &[usize],
        q_lens: &[usize],
        offsets: &[usize],
        layer_start: usize,
        layer_end: usize,
    ) -> Result<usize> {
        let idx = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
        self.wave_qsa_bytes_in(&idx, seq_ids, q_lens, offsets, layer_start, layer_end)
    }

    /// [`Self::wave_qsa_bytes`] against index caches the caller already holds
    /// locked — a draft walk prices its tier under the write lock it grew the
    /// head's cache with.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn wave_qsa_bytes_in(
        &self,
        idx: &HashMap<usize, Vec<IndexCache>>,
        seq_ids: &[usize],
        q_lens: &[usize],
        offsets: &[usize],
        layer_start: usize,
        layer_end: usize,
    ) -> Result<usize> {
        let spans = seq_spans(seq_ids, q_lens)?;
        self.qsa_bytes_over(&spans, offsets, idx, layer_start, layer_end)
    }

    /// [`Self::wave_qsa_bytes`] for sequences that have not run yet — prefills of
    /// `q_lens[i]` rows from `offsets[i]` against fresh contiguous indexes, over
    /// the whole stack. What a turn's admission prices its chunks' tiers with,
    /// through the same `select_layer_bytes` the wave itself carves by (a sequence
    /// with no cache is priced there as one live tail from position zero).
    #[cfg(not(feature = "cuda"))]
    pub(super) fn fresh_qsa_bytes(&self, q_lens: &[usize], offsets: &[usize]) -> Result<usize> {
        // Ids no live sequence holds: slot ids are small indices.
        let seqs: Vec<usize> = (0..q_lens.len()).map(|i| usize::MAX - i).collect();
        let spans = seq_spans(&seqs, q_lens)?;
        self.qsa_bytes_over(
            &spans,
            offsets,
            &HashMap::new(),
            0,
            self.model.cfg.num_layers,
        )
    }

    /// The widest selection carve any KV layer of `[layer_start, layer_end)` makes
    /// over `spans`, given the sequences' caches in `idx` — the draft head's layer
    /// included when the range reaches the head.
    pub(super) fn qsa_bytes_over(
        &self,
        spans: &[SeqSpan],
        offsets: &[usize],
        idx: &HashMap<usize, Vec<IndexCache>>,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<usize> {
        let cfg = &self.model.cfg;
        let mut worst = 0usize;
        let mut kv = 0usize;
        for (li, layer) in self.model.layers.iter().enumerate() {
            if let GpuLayerMix::Attention { compress_ratio, .. } = &layer.mix {
                if (layer_start..layer_end).contains(&li) {
                    worst = worst.max(select_layer_bytes(
                        kv,
                        *compress_ratio,
                        spans,
                        offsets,
                        idx,
                        &cfg.indexer,
                    )?);
                }
                kv += 1;
            }
        }
        if layer_end == cfg.num_layers {
            if let Some(head) = &self.model.mtp {
                if let GpuLayerMix::Attention { compress_ratio, .. } = &head.block.mix {
                    worst = worst.max(select_layer_bytes(
                        kv,
                        *compress_ratio,
                        spans,
                        offsets,
                        idx,
                        &cfg.indexer,
                    )?);
                }
            }
        }
        Ok(worst)
    }

    /// Each KV layer's indexer weights, in the same KV-layer order as
    /// [`Self::attention_ratios`] — the two are zipped against the caches, so
    /// they must walk the layers identically.
    pub(in super::super) fn attention_indexers(&self) -> Vec<&IndexerWeights> {
        let mut out: Vec<&IndexerWeights> = self
            .model
            .layers
            .iter()
            .filter_map(|l| match &l.mix {
                GpuLayerMix::Attention { indexer, .. } => Some(indexer),
                GpuLayerMix::DeltaNet(_) => None,
            })
            .collect();
        if let Some(head) = &self.model.mtp {
            if let GpuLayerMix::Attention { indexer, .. } = &head.block.mix {
                out.push(indexer);
            }
        }
        out
    }

    /// The indexer's factored RoPE table, at the model's rotary width
    /// (`cfg.rope_dim`) and frequencies.
    pub(in super::super) fn index_rope(&self) -> &FactoredRope {
        &self.index_rope
    }

    /// The schedule's rungs, which every paged attention kernel rotates from.
    pub fn rope(&self) -> &RopeRungs {
        &self.rope
    }

    /// Run `schedule` in place of Flash-Next's own, for the attention and the
    /// indexer alike — the control a long-context gate measures the model's
    /// rungs against (`docs/progressive_yarn.md` §10). Nothing stored depends
    /// on a rung (I1), so a session opened after this simply reads the new
    /// tables. The rotary width must be the model's.
    pub fn set_rope_schedule(&mut self, schedule: &RopeSchedule) -> Result<()> {
        if schedule.pairs() != self.rope.pairs() {
            candle::bail!(
                "qwen4exp: a {}-pair schedule on a {}-pair rotary width",
                schedule.pairs(),
                self.rope.pairs()
            );
        }
        let rope = RopeRungs::new(schedule, &self.model.device)?;
        self.index_rope = FactoredRope::over(&rope, &self.model.device)?;
        self.rope = rope;
        Ok(())
    }
}
