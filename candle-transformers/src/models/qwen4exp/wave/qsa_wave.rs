//! The QSA selection's configuration on Qwen4ExpBatched: its budget, strata, the prompt and pinned spans it ranks, and the hit counters that feed projection.
//!
//! The budget is the number the selection engages against, and a caller can widen it past the prompt to read densely, which is the control a retrieval claim is measured against. A strata is refused at load when the selection kernel could not run it at every depth the RoPE schedule reaches. Declared prompt spans and pins are recorded on every index cache, so they travel with the index through a fork, a move or a truncate. The hit counters record what the sampled layers chose, and the projection reads them back without waiting on the device.

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
    /// The QSA selection budget, in positions.
    ///
    /// The checkpoint's own value is what production runs; this exists so a
    /// caller can widen it past the prompt and get the **same engine reading
    /// densely**. `selection_engages` is a comparison against this number, so a
    /// budget above the depth makes the selection the identity — the same code
    /// path over every cell, rather than a second build with the feature
    /// removed.
    ///
    /// That control is what makes a retrieval claim falsifiable. A needle the
    /// selected read loses says nothing on its own: it could be lost to the
    /// selection, or the checkpoint could simply not answer that prompt. Run
    /// both and the difference names which.
    ///
    /// Refused when the selection kernel could not run it — see
    /// [`budget_fits_kernel`].
    pub fn set_selection_budget(&mut self, top_k: usize) -> Result<()> {
        budget_fits_kernel(
            top_k,
            &self.attention_ratios(),
            self.model.cfg.max_position_embeddings,
            MAX_KEEP,
        )
        .map_err(candle::Error::Msg)?;
        // A strata set before the budget was checked against the old one.
        let strata = self.model.cfg.indexer.strata;
        if strata != Strata::WHOLE {
            self.check_strata(top_k, self.selecting_ratio()?, &strata)?;
        }
        self.model.cfg.indexer.top_k = top_k;
        Ok(())
    }

    /// The one compression ratio every selecting layer shares — what a strata
    /// stated in positions is converted at.
    pub(super) fn selecting_ratio(&self) -> Result<usize> {
        let mut ratios: Vec<usize> = self
            .attention_ratios()
            .into_iter()
            .filter(|&r| r > 0)
            .collect();
        ratios.sort_unstable();
        ratios.dedup();
        match ratios.as_slice() {
            &[ratio] => Ok(ratio),
            _ => candle::bail!(
                "qwen4exp: a stratified selection needs one compression ratio across the \
                 selecting layers, found {ratios:?}"
            ),
        }
    }

    /// Whether the selection kernel can run `strata` under a budget of `top_k`
    /// at every depth the RoPE schedule reaches.
    ///
    /// A budget at or past that reach never selects, so any strata runs. Below
    /// it, the deepest row is checked with one window more than `reach / ratio`
    /// candidates cut into: a sequence assembled from sealed pages has a short
    /// block at every page boundary, so its candidates can outnumber
    /// `reach / ratio`, and the headroom covers every count of boundaries
    /// narrower than a window.
    pub(super) fn check_strata(&self, top_k: usize, ratio: usize, strata: &Strata) -> Result<()> {
        let reach = self.rope.select().reach();
        if top_k >= reach {
            return Ok(());
        }
        selection_stride(ratio, top_k, reach / ratio + strata.window_blocks, strata)?;
        Ok(())
    }

    /// The budget [`Self::set_selection_budget`] is currently at.
    pub fn selection_budget(&self) -> usize {
        self.model.cfg.indexer.top_k
    }

    /// How the QSA selection divides a query's candidates before ranking them
    /// (`docs/qsa_stratified_selection.md`), stated in positions. The model
    /// loads at [`StrataTokens::WHOLE`], the checkpoint's own selection; an
    /// engine sets its own on top ([`StrataTokens::DEFAULT`] unless told
    /// otherwise).
    ///
    /// The positions become blocks at the selecting layers' compression ratio,
    /// which must be one ratio: a window of positions is a different number of
    /// blocks at each, and the indexer carries one strata. Checked against the
    /// selection kernel at the deepest position the RoPE schedule reaches,
    /// under the current budget ([`Self::check_strata`]): a strata whose windows
    /// would gather more entries than the kernel holds is refused here, at load,
    /// not at the depth that first reaches it.
    pub fn set_selection_strata(&mut self, tokens: StrataTokens) -> Result<()> {
        let ratio = self.selecting_ratio()?;
        let strata = Strata::from_tokens(tokens, ratio);
        self.check_strata(self.model.cfg.indexer.top_k, ratio, &strata)?;
        if strata.recent == Recent::Forced && strata.recent_blocks > 0 {
            tracing::warn!(
                recent = tokens.recent,
                "a forced recent span names its blocks in every row whatever they score, so \
                 no hits are counted: the hit-driven lift is off under these strata"
            );
        }
        self.model.cfg.indexer.strata = strata;
        Ok(())
    }

    /// The strata [`Self::set_selection_strata`] is currently at.
    pub fn selection_strata(&self) -> Strata {
        self.model.cfg.indexer.strata
    }

    /// Declare that `seq`'s first `tokens` positions hold the conversation's
    /// system prompt — the span a stratified selection ranks in every window.
    ///
    /// Recorded on every one of the sequence's index caches, creating them if
    /// the sequence has not run a wave yet, so the declaration travels with the
    /// index through a fork, a move or a truncate.
    pub fn set_selection_prompt(&self, seq: usize, tokens: usize) -> Result<()> {
        self.each_index_cache(seq, |cache| cache.set_prompt_end(tokens))
    }

    /// Run `f` on each of `seq`'s index caches, creating them if the sequence
    /// has not run a wave yet — the one place a sequence's caches are made
    /// for a declaration, so they travel with the index through a fork, a
    /// move or a truncate.
    pub(super) fn each_index_cache(
        &self,
        seq: usize,
        mut f: impl FnMut(&mut IndexCache),
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        let mut idx = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
        let caches = match idx.entry(seq) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(
                (0..cfg.kv_layers().total())
                    .map(|_| IndexCache::new(cfg.indexer.head_dim, &self.model.device))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        caches.iter_mut().for_each(&mut f);
        Ok(())
    }

    /// Count hits at the selecting layers `layers` lands on.
    pub fn set_hit_layers(&mut self, layers: &HitPercentiles) -> Result<()> {
        self.hit_kv = self.sampled_kv(layers);
        Ok(())
    }

    /// The KV layers `layers`' percentiles land on, over the trunk's
    /// SELECTING attention layers — those with an indexer — in layer order.
    pub(super) fn sampled_kv(&self, layers: &HitPercentiles) -> Vec<usize> {
        let ratios = self.attention_ratios();
        let selecting: Vec<usize> = (0..self.model.cfg.n_attention_layers())
            .filter(|&kv| ratios.get(kv).is_some_and(|&r| r > 0))
            .collect();
        layers
            .indices(selecting.len())
            .into_iter()
            .map(|i| selecting[i])
            .collect()
    }

    /// The shadow scan's geometry at a layer selecting at `ratio`.
    pub(super) fn shadow_geometry(&self, ratio: usize) -> ShadowGeometry {
        let ix = &self.model.cfg.indexer;
        ShadowGeometry {
            heads: ix.n_heads,
            head_dim: ix.head_dim,
            pairs: self.index_rope.pairs(),
            ratio,
            top_k: ix.top_k,
        }
    }

    /// The kept-query buffer every sequence's counters carry: a set of slots
    /// per sampled layer, each one indexer query.
    pub(super) fn keep_shape(&self) -> KeepShape {
        let ix = &self.model.cfg.indexer;
        KeepShape {
            layers: self.hit_kv.len(),
            q_words: ix.n_heads * ix.head_dim,
        }
    }

    /// The unloaded rows of `pooled` — rounds of pieces whose index pages are
    /// resident — at each sampled layer, counted from `first`; and each
    /// round's blocks. A round whose page is not resident at a layer has no
    /// rows there.
    pub(super) fn pooled_rows(&self, pooled: &[PooledRound], first: usize) -> Result<PooledRounds> {
        let ratio = self.selecting_ratio()?;
        let mut blocks: Vec<u32> = Vec::with_capacity(pooled.len());
        let mut layers: Vec<Vec<ShadowSegment>> = vec![Vec::new(); self.hit_kv.len()];
        for (i, round) in pooled.iter().enumerate() {
            let rows = block_range(&round.positions, |pos| (pos + 1) / ratio);
            let mut held = 0usize;
            for (li, &kv) in self.hit_kv.iter().enumerate() {
                let Some(page) = self.pages.layer(&round.piece, kv) else {
                    continue;
                };
                let r = rows.start.min(page.rows())..rows.end.min(page.rows());
                held = held.max(r.len());
                if !r.is_empty() {
                    layers[li].push(ShadowSegment {
                        page,
                        rows: r,
                        counter: (first + i) as u32,
                    });
                }
            }
            blocks.push(held as u32);
        }
        Ok(PooledRounds {
            blocks,
            layers: layers
                .into_iter()
                .map(|segments| ShadowRows::new(segments, &self.model.device).map(Arc::new))
                .collect::<Result<_>>()?,
        })
    }

    /// Declare the rounds `seq`'s projection placed, as position ranges in
    /// declaration order, and after them the `pooled` rounds it did not —
    /// what the shadow scan reads (`docs/hit_driven_projection.md` §6.3) —
    /// replacing any declared before, whose counts are dropped, so a caller
    /// reads them first ([`Self::take_hits`]).
    ///
    /// The placed ranges become blocks through the sequence's index layout at
    /// the first sampled layer; every layer of a sequence shares one layout.
    pub fn set_hit_rounds(
        &self,
        seq: usize,
        pieces: &[Range<usize>],
        pooled: &[PooledRound],
    ) -> Result<()> {
        let ratio = self.selecting_ratio()?;
        let pooled = self.pooled_rows(pooled, pieces.len())?;
        let map = {
            let idx = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            let kv = self.hit_kv.first().copied().unwrap_or(0);
            match idx.get(&seq).and_then(|c| c.get(kv)) {
                Some(cache) => block_map(pieces, |r| cache.block_range(r, ratio)),
                None => block_map(pieces, |r| block_range(r, |pos| (pos + 1) / ratio)),
            }
        };
        let mut hits = self
            .hits
            .write()
            .map_err(|_| candle::Error::Msg("hits lock poisoned".into()))?;
        // In place when the sequence owns its counters, so every view carved
        // from it counts against the new pieces too. A sequence holding a
        // parent's — a fork projecting for itself — gets counters of its own
        // and leaves the parent's as they were.
        match hits.get_mut(&seq).filter(|h| h.owned) {
            Some(h) => {
                h.generation = h
                    .hits
                    .redeclare(map, pieces.len(), pooled, &self.model.device)?;
            }
            None => {
                let h = SeqHits::new(
                    map,
                    pieces.len(),
                    pooled,
                    self.keep_shape(),
                    &self.model.device,
                )?;
                hits.insert(
                    seq,
                    HitHolder {
                        hits: h,
                        owned: true,
                        generation: 0,
                    },
                );
            }
        }
        Ok(())
    }

    /// What `seq`'s sampled layers chose since the last read, per declared
    /// piece — its own rows and every view's carved from it; `None` when it
    /// declared none of its own. Read from the summary the device last
    /// published, without waiting on it; what is still in flight is in the
    /// next read. A view holding its parent's counters reads nothing: they
    /// are the parent's to read.
    pub fn take_hits(&self, seq: usize) -> Result<Option<PieceHits>> {
        let hits = self
            .hits
            .read()
            .map_err(|_| candle::Error::Msg("hits lock poisoned".into()))?;
        hits.get(&seq)
            .filter(|h| h.owned)
            .map(|h| h.hits.take())
            .transpose()
    }

    /// Declare `seq`'s pinned position ranges — spans of its system prompt
    /// every selecting query attends whole (`docs/hit_driven_projection.md`
    /// §5) — replacing any declared before.
    ///
    /// Coalesced to the selection's table of [`MAX_PINS`] ranges across the
    /// smallest gaps, so the glue between a collection's members never splits
    /// a pinned span past it; each row is then held to the strata's
    /// [`Strata::pin_blocks`] as it is built ([`IndexCache::pin_row`]).
    /// Recorded on every one of the sequence's index caches, as the prompt
    /// span is, so it travels with the index.
    pub fn set_selection_pins(&self, seq: usize, ranges: &[Range<usize>]) -> Result<()> {
        let cfg = &self.model.cfg;
        let merged = coalesce_ranges(ranges, MAX_PINS);
        // Rows are trimmed to the bound inside the forward, silently; a bound
        // of zero would leave every declared pin attending nothing.
        if !merged.is_empty() && cfg.indexer.strata.pin_blocks == 0 {
            tracing::warn!(
                seq,
                ranges = merged.len(),
                "pinned ranges declared, but the selection strata pin nothing \
                 (StrataTokens::pinned is 0): no range will be attended whole"
            );
        }
        self.each_index_cache(seq, |cache| cache.set_pins(&merged))
    }
}
