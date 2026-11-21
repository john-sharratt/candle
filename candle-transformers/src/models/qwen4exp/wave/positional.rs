//! The QSA index of each attention layer, kept per sequence and reset, cut, sealed, spliced and injected as pages.
//!
//! The index is the indexer's own projection of hidden states, so it is carried with the sequence rather than rebuilt from K/V. Every attention layer indexes one stream, so a cut is made on all of them together or not at all. A sealed piece is self-contained and keeps its own block widths, which lets it be placed at any offset in any conversation. Each operation says what it refuses, and a turn's rows are sealed from wherever they live, since a reprojection can move the user's half into a page of its own.

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
    /// Drop `seq`'s index — the slot's K/V was truncated to nothing.
    ///
    /// The pages it held stay resident until `seq` resets again or is released
    /// (see the `retired` field), so the rebuild that follows a reset finds the
    /// pieces it re-injects already placed.
    pub fn reset_positional_state(&self, seq: usize) -> Result<()> {
        let mut dropped: Vec<Arc<ResidentPage>> = Vec::new();
        if let Ok(mut map) = self.index.write() {
            if let Some(caches) = map.get_mut(&seq) {
                for c in caches.iter_mut() {
                    dropped.extend(c.reset());
                }
            }
        }
        if let Ok(mut retired) = self.retired.write() {
            retired.insert(seq, dropped);
        }
        Ok(())
    }

    /// The largest position at or before `tokens` that every one of `seq`'s
    /// index caches can be cut back to (see [`IndexCache::cut_floor`]).
    pub fn positional_cut_floor(&self, seq: usize, tokens: usize) -> Result<usize> {
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        Ok(map.get(&seq).map_or(tokens, |caches| {
            caches
                .iter()
                .map(|c| c.cut_floor(tokens))
                .min()
                .unwrap_or(tokens)
        }))
    }

    /// Cut `seq`'s index back to `tokens` — the slot's K/V was truncated to a
    /// piece boundary there, and what follows is about to be re-injected. The
    /// pages above the cut are retired like a reset's (see the `retired` field).
    pub fn truncate_positional_state(&self, seq: usize, tokens: usize) -> Result<()> {
        let mut dropped: Vec<Arc<ResidentPage>> = Vec::new();
        {
            let mut map = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            if let Some(caches) = map.get_mut(&seq) {
                for c in caches.iter_mut() {
                    dropped.extend(c.truncate_to(tokens)?);
                }
            }
        }
        if let Ok(mut retired) = self.retired.write() {
            retired.insert(seq, dropped);
        }
        Ok(())
    }

    /// Splice `seq`'s index: on every layer, replace `[from, to)` with `pieces`
    /// and move the pages after it ([`IndexCache::splice_pages`]). Every layer
    /// is asked first, so a refusal changes none of them; the pages taken out
    /// are retired with the cut's (see the `retired` field).
    ///
    /// A layer takes a piece's page where the piece has one for it and skips it
    /// where it has none, as [`Self::push_positional_state`] does; a gap moves
    /// every layer, as [`Self::push_positional_gap`] does.
    #[cfg(feature = "cuda")]
    pub fn splice_positional_state(
        &self,
        seq: usize,
        from: usize,
        to: usize,
        pieces: &[PositionalPiece],
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        let ratios = self.attention_ratios();
        // Resolved before the lock is taken: a placement is a launch.
        let resolved: Vec<Option<Vec<Option<Arc<ResidentPage>>>>> = pieces
            .iter()
            .map(|p| match p {
                PositionalPiece::Page { bytes, key } => {
                    self.resident_pages(bytes, key, |_| true).map(Some)
                }
                PositionalPiece::Gap(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
        let mut taken: Vec<Arc<ResidentPage>> = Vec::new();
        {
            let mut map = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
            let caches = map.entry(seq).or_default();
            if caches.is_empty() {
                for _ in 0..want {
                    caches.push(IndexCache::new(cfg.indexer.head_dim, &self.model.device)?);
                }
            }
            for c in caches.iter() {
                c.check_splice(from, to)?;
            }
            for (li, (c, &ratio)) in caches.iter_mut().zip(ratios.iter()).enumerate() {
                let layer: Vec<SplicePiece> = pieces
                    .iter()
                    .zip(&resolved)
                    .filter_map(|(piece, pages)| match (piece, pages) {
                        (PositionalPiece::Gap(tokens), _) => Some(SplicePiece::Gap(*tokens)),
                        (PositionalPiece::Page { .. }, Some(layers)) => {
                            layers.get(li).cloned().flatten().map(SplicePiece::Page)
                        }
                        (PositionalPiece::Page { .. }, None) => None,
                    })
                    .collect();
                taken.extend(c.splice_pages(from, to, layer, ratio)?);
            }
        }
        if let Ok(mut retired) = self.retired.write() {
            retired.entry(seq).or_default().extend(taken);
        }
        Ok(())
    }

    /// Close `seq`'s index on a block boundary and hand back the page.
    ///
    /// The flush is what makes the piece self-contained: a section's tokens do
    /// not end on a block boundary, so `T mod ratio` rows sit carried, and a
    /// page without them would leave those tokens indexed by a block the *next*
    /// section completes. The flushed block summarises fewer than `ratio`
    /// tokens and is deliberately not what a continuous run would have produced
    /// for that span — the scorer carries each page's width and so can express
    /// a short block, where a block pooled across two sections is not
    /// correctable at all.
    #[cfg(feature = "cuda")]
    pub fn seal_positional_state(&self, seq: usize) -> Result<Option<Vec<u8>>> {
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get_mut(&seq) else {
            // Named, not silent. A slot with no index caches at seal time seals
            // a piece with no page, and the caller has no way to tell that from
            // "this model keeps no per-position state" — which is the confusion
            // that let an unindexed system prompt through.
            tracing::warn!(
                seq,
                live_slots = map.len(),
                "qsa seal: slot has no index caches, so this piece seals with no page — \
                 nothing ever appended to it, or its state was reset since",
            );
            return Ok(None);
        };
        let mut pages = Vec::with_capacity(caches.len());
        for ((c, &ratio), w) in caches.iter_mut().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                pages.push(seal_live(c, 1)?);
                continue;
            }
            // Sealing is not a forward: no phase is open to carve from. The
            // flush consumes the carried rows, so the page IS the whole piece
            // and there is no open block to carry with it.
            let cells = c.flush_open_block(w, cfg.rms_norm_eps)?;
            pages.push(seal_live(c, cells.unwrap_or(ratio))?);
        }
        Ok(Some(paged_index::encode_aux(&[], &pages)?))
    }

    /// Close `seq`'s index on a page boundary, on every attention layer.
    ///
    /// Called at a reasoning boundary during decode, so a turn's
    /// `<think>…</think>` occupies whole pages and can be dropped exactly when
    /// the turn is projected as history. Every layer closes together — they
    /// index one stream, so a cut on some of them would leave the rest
    /// addressing different blocks for the same position.
    ///
    /// Returns the tokens the closed page covers — `0` when the tail was already
    /// empty, which is the common case and a legitimate no-op.
    #[cfg(feature = "cuda")]
    pub fn close_positional_page(&self, seq: usize) -> Result<usize> {
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get_mut(&seq) else {
            return Ok(0);
        };
        // Every layer indexes the same stream, so they close the same width;
        // taken from whichever is walked last.
        let mut closed = 0usize;
        for ((c, &ratio), w) in caches.iter_mut().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                continue;
            }
            closed = c.close_tail_into_page(w, ratio, cfg.rms_norm_eps)?;
        }
        Ok(closed)
    }

    /// How many of `seq`'s last tokens a re-feed forwards again, at most `max`
    /// (`docs/reprojection_refeed.md` §2): the live index tail's re-feed cut
    /// ([`IndexCache::refeed_cut`]) — back to the newest block boundary within
    /// `max` tokens, never past the tail's start, which is the newest closed
    /// page. Every layer indexes one stream, so they agree; a disagreement is
    /// refused. `max` for a sequence with no index.
    pub fn refeed_span(&self, seq: usize, max: usize) -> Result<usize> {
        let ratios = self.attention_ratios();
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get(&seq) else {
            return Ok(max);
        };
        let mut span: Option<usize> = None;
        for (c, &ratio) in caches.iter().zip(ratios.iter()) {
            if ratio == 0 {
                continue;
            }
            let tokens = c.refeed_cut(ratio, max).tokens;
            match span {
                None => span = Some(tokens),
                Some(first) if first != tokens => candle::bail!(
                    "qsa re-feed: one layer's live tail re-feeds {first} token(s) and another's \
                     {tokens} — the layers have diverged"
                ),
                Some(_) => {}
            }
        }
        Ok(span.unwrap_or(max))
    }

    /// Drop `seq`'s last `tokens` index rows on every attention layer — the
    /// span [`Self::refeed_span`] chose, which the re-feed writes again. A span
    /// that is not a re-feed cut of every layer's tail is refused.
    pub fn rewind_positional_tail(&self, seq: usize, tokens: usize) -> Result<()> {
        let ratios = self.attention_ratios();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get_mut(&seq) else {
            return Ok(());
        };
        for (c, &ratio) in caches.iter_mut().zip(ratios.iter()) {
            // A layer with no indexer keeps one row a token.
            let cut = c.refeed_cut(ratio.max(1), tokens);
            if cut.tokens != tokens {
                candle::bail!(
                    "qsa re-feed: a rewind of {tokens} token(s) is not a cut of this layer's live \
                     tail, which re-feeds {}",
                    cut.tokens
                );
            }
            c.rewind_live(cut.keep_blocks)?;
        }
        Ok(())
    }

    /// The page covering `seq`'s tokens from `start_pos` onward.
    ///
    /// **Taken on a fork, because this slot keeps decoding.** A section is
    /// ingested on a throwaway slot, so closing its cache in place costs
    /// nothing; a conversation turn is sealed on the slot that produced it and
    /// the flush would leave the live cache with a short block mid-sequence,
    /// which every later position would then be addressed through. The fork is
    /// `n_blocks × head_dim` floats per attention layer — the same shape of cost
    /// a view carve already pays.
    ///
    /// `start_pos` is where the turn's own K/V begins. Rows below it belong to
    /// what the projection injected ahead of this turn and are already pages of
    /// their own; carrying them again would place the same block at two
    /// positions.
    #[cfg(feature = "cuda")]
    pub fn seal_positional_range(&self, seq: usize, start_pos: usize) -> Result<Option<Vec<u8>>> {
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get(&seq) else {
            return Ok(None);
        };
        let mut pages = Vec::with_capacity(caches.len());
        for ((c, &ratio), w) in caches.iter().zip(ratios.iter()).zip(indexers.iter()) {
            if ratio == 0 {
                pages.push(seal_live(c, 1)?);
                continue;
            }
            // **Two row spaces, and they are not the same one.**
            // `candidates_at` counts blocks over the WHOLE sequence — the
            // injected pages ahead of this turn and the live tail together —
            // while `live_rows` is the tail alone. Subtracting the page span
            // converts the global block index into an offset into the tail,
            // which is where a turn's own rows live. Without it the offset
            // overshoots by exactly the number of rows the projection injected,
            // and the narrow is refused.
            //
            // Saturating below the pages; refused above the tail. A `start_pos`
            // that lands inside the injected pages has no offset in tail row
            // space, and silently clamping it would hand back a page describing
            // a different span than the caller asked for — rows for the wrong
            // positions, which selects fluently against the wrong context. Said
            // plainly here instead: `narrow` reports this as
            // `start > dim_len` naming neither the sequence nor the span, from
            // a call site several layers removed.
            let first = if start_pos == 0 {
                0
            } else {
                c.candidates_at(start_pos - 1, ratio)
                    .saturating_sub(c.page_row_span())
            };
            let mut fork = c.fork()?;
            let cells = fork.flush_open_block(w, cfg.rms_norm_eps)?;
            let rows = fork.live_rows_host()?;
            let n = rows.n_rows();
            if first > n {
                candle::bail!(
                    "qsa seal: seq {seq} asked for the rows from position {start_pos}, which \
                     is row {first} of a {n}-row live tail — the span starts inside the \
                     injected pages, whose rows this cannot return. Seal the span from the \
                     sequence that forwarded it, where it is the tail."
                );
            }
            pages.push(SealedIndex {
                keys: rows.from_row(first),
                last_cells: cells.unwrap_or(ratio),
                open: fork.open_rows_host()?,
            });
        }
        Ok(Some(paged_index::encode_aux(&[], &pages)?))
    }

    /// Seal the rows for the last `tokens` tokens of `seq`, wherever they live.
    ///
    /// **Why this is not [`Self::seal_positional_range`].** That one returns
    /// live-tail rows only, which is right for a span the sequence forwarded
    /// and still holds. A turn's rows do not stay there: a reprojection
    /// re-injects the turn's user half from cache and pushes it as a *page*, so
    /// after one reprojection the turn is split — its opening tokens sit in a
    /// page and only its decoded tail sits in the live rows. Sealing the tail
    /// alone then drops exactly the user's message, and the model decodes a
    /// turn whose question it cannot attend to; measured, it invents a
    /// different question and answers that instead.
    ///
    /// Returns one blob per source page in ascending order, with the live tail
    /// last. They are pushed in that order, which preserves each piece's own
    /// ragged width — merging them into one page would re-pool rows across
    /// boundaries the pieces ended at.
    pub fn seal_positional_tail_span(
        &self,
        seq: usize,
        tokens: usize,
    ) -> Result<Vec<(usize, Vec<u8>)>> {
        if tokens == 0 {
            return Ok(Vec::new());
        }
        let cfg = &self.model.cfg;
        let ratios = self.attention_ratios();
        let indexers = self.attention_indexers();
        let map = self
            .index
            .read()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let Some(caches) = map.get(&seq) else {
            return Ok(Vec::new());
        };
        // The page structure is the same on every layer — they index one stream
        // — so the walk is decided once, on the first live cache.
        let Some((probe, &probe_ratio)) = caches.iter().zip(ratios.iter()).find(|(_, r)| **r > 0)
        else {
            return Ok(Vec::new());
        };
        let (blocks, open) = probe.seal_shape();
        let tail_tokens = blocks * probe_ratio + open;
        // Whole trailing pages, until the span is covered — but never a page
        // that would reach back past the span's own start. The decision is
        // [`paged_index::tail_span_pages`], a free function so the boundary
        // arithmetic is pinned by unit tests rather than by a live daemon; see
        // its notes for the regression that shaped it.
        let widths: Vec<usize> = (0..probe.page_count())
            .map(|i| probe.page_at(i).map_or(0, |(_, w)| w))
            .collect();
        let (first_page, covered) = paged_index::tail_span_pages(&widths, tail_tokens, tokens);
        if covered != tokens {
            // **This is an alarm, not a mode.** The walk's refusal of an
            // over-wide page is a backstop: a unit's rows begin on a page
            // boundary because the scheduler closes one when the unit's K/V
            // anchor is taken (`Scheduler::begin_unit`), so reaching here means
            // some path opened a unit without passing that boundary. The
            // shortfall is the width of what this turn shares a page with.
            //
            // The AVAILABLE page widths, not just the taken ones, and the width
            // of the page the walk stopped at. Without those two the caller can
            // see only that the span does not line up, which is the question
            // rather than the answer: the refused page's width is what names the
            // piece this turn's rows share a page with.
            let refused = first_page.checked_sub(1).map(|i| widths[i]);
            tracing::warn!(
                seq,
                tokens,
                covered,
                tail_tokens,
                first_page,
                refused_page_width = ?refused,
                all_page_widths = ?widths,
                "qsa seal: a turn's pages cover {} token(s) {} the turn — its rows do not \
                 begin on a page boundary, so a unit began without one being closed",
                covered.abs_diff(tokens),
                if covered > tokens {
                    "more than"
                } else {
                    "less than"
                },
            );
        }

        // Each blob carries the tokens it covers. The caller cannot read a page
        // — the bytes are this model's — so the width is the only handle it has
        // on where a page sits in the span, and it needs one: the projection
        // drops the pages covering a turn's reasoning, and the page COUNT is not
        // stable (a mid-decode reprojection closes an extra one). Position is.
        let mut blobs: Vec<(usize, Vec<u8>)> = Vec::new();
        for pi in first_page..probe.page_count() {
            // Two descriptions of the same page travel together from here: the
            // width, which the caller uses to place the page in the span, and
            // the rows the blob carries, which only this model can read. They
            // are the one thing nothing downstream can cross-check — so they are
            // checked here, where both are in hand.
            let mut width: Option<usize> = None;
            let mut pages = Vec::with_capacity(caches.len());
            for c in caches.iter() {
                let (p, w) = c
                    .page_at(pi)
                    .ok_or_else(|| candle::Error::Msg(format!("qsa seal: page {pi} vanished")))?;
                // Every layer indexes one stream, so a page spans the same
                // tokens on all of them. A layer that disagrees would ship rows
                // for a different span than the width the caller places them by,
                // and the projection would window out the wrong tokens on that
                // layer alone — visible only as degraded retrieval.
                match width {
                    None => width = Some(w),
                    Some(first) if first != w => candle::bail!(
                        "qsa seal: page {pi} is {first} token(s) wide on the first layer and {w} \
                         on another — the layers have diverged and the seal cannot say which \
                         span this page covers"
                    ),
                    Some(_) => {}
                }
                pages.push(p);
            }
            let width = width.unwrap_or(0);
            // A page every layer of which was placed from one record this loop
            // made is that record: its rows are the record's and never change,
            // and this loop reads them back to exactly those bytes.
            let source = pages.first().and_then(|p| p.source());
            let kept = source
                .filter(|k| pages.iter().all(|p| p.source() == Some(*k)))
                .and_then(|k| self.seal_memo.get(k));
            if let Some(record) = kept {
                blobs.push((width, record.to_vec()));
                continue;
            }
            // A closed page's rows are un-rotated and immutable, so the
            // record is the page's rows as they stand. A page is already
            // closed; only the live tail carries an open block.
            let layer_pages = pages
                .iter()
                .map(|p| {
                    Ok(SealedIndex {
                        keys: p.host_rows()?,
                        last_cells: p.last_cells(),
                        open: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let record = paged_index::encode_aux(&[], &layer_pages)?;
            self.seal_memo.put(&record);
            blobs.push((width, record));
        }

        if tail_tokens > 0 {
            let mut layer_pages = Vec::with_capacity(caches.len());
            for ((c, &ratio), w) in caches.iter().zip(ratios.iter()).zip(indexers.iter()) {
                if ratio == 0 {
                    layer_pages.push(seal_live(c, 1)?);
                    continue;
                }
                let mut fork = c.fork()?;
                let cells = fork.flush_open_block(w, cfg.rms_norm_eps)?;
                layer_pages.push(seal_live(&fork, cells.unwrap_or(ratio))?);
            }
            // Not kept in the memo: a live tail's record is made another way
            // than a closed page's, so the page it is placed as is read back
            // once, by the next seal, whose record is kept.
            blobs.push((tail_tokens, paged_index::encode_aux(&[], &layer_pages)?));
        }
        Ok(blobs)
    }

    /// The resident pages of a piece at the KV layers `wanted` names, one per
    /// KV layer (`None` where the layer indexes nothing or is not wanted): the
    /// ones already standing when any slot or pool still holds them, the rest
    /// decoded and placed now — every missing layer in one launch, into slots
    /// of the index tenant. A piece pooled at its sampled layers and then
    /// placed in a slot shares the sampled layers' rows and places the others.
    #[cfg(feature = "cuda")]
    pub(super) fn resident_pages(
        &self,
        blob: &[u8],
        key: &PieceKey,
        wanted: impl Fn(usize) -> bool,
    ) -> Result<Vec<Option<Arc<ResidentPage>>>> {
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        let ratios = self.attention_ratios();
        let mut layers: Vec<Option<Arc<ResidentPage>>> = vec![None; want];
        let mut missing: Vec<usize> = Vec::new();
        for (kv, &r) in ratios.iter().enumerate() {
            if r == 0 || !wanted(kv) {
                continue;
            }
            match self.pages.layer(key, kv) {
                Some(p) => layers[kv] = Some(p),
                None => missing.push(kv),
            }
        }
        if missing.is_empty() {
            return Ok(layers);
        }
        let _place = span("qsa:page:decode_place");
        // Only the layers to place are decoded: a container holds every layer,
        // and the pool places the sampled few.
        let (_, sections) = paged_index::aux_parts(blob)?;
        if sections.len() != want {
            candle::bail!(
                "qwen4exp: an injected piece carries {} index pages but this checkpoint has \
                 {want} KV layers — the layers it does not cover would select against an \
                 empty candidate set",
                sections.len()
            );
        }
        let sealed: Vec<SealedIndex> = missing
            .iter()
            .map(|&kv| paged_index::decode_page(sections[kv]))
            .collect::<Result<_>>()?;
        let host: Vec<(&Q8Keys, usize)> = sealed.iter().map(|s| (&s.keys, s.last_cells)).collect();
        let placed: Vec<(usize, Arc<ResidentPage>)> = missing
            .iter()
            .copied()
            .zip(ResidentPage::place_host(
                &host,
                cfg.indexer.head_dim,
                &self.model.device,
            )?)
            .collect();
        for (kv, page) in &placed {
            page.placed_from(*key);
            layers[*kv] = Some(Arc::clone(page));
        }
        self.pages.insert(*key, want, &placed);
        Ok(layers)
    }

    /// Device bytes pushing `blob` would newly place: each attention layer's
    /// rows not already resident (the registry shares a placed page across
    /// slots and pools), in whole `PAGE_BLOCKS` slots — the chunks
    /// [`ResidentPage::place_host`] claims.
    pub fn positional_lift_bytes(&self, blob: &[u8], key: &PieceKey) -> Result<usize> {
        self.positional_lift_bytes_at(blob, key, |_| true)
    }

    /// [`Self::positional_lift_bytes`] for the KV layers `wanted` names alone.
    pub(super) fn positional_lift_bytes_at(
        &self,
        blob: &[u8],
        key: &PieceKey,
        wanted: impl Fn(usize) -> bool,
    ) -> Result<usize> {
        let ratios = self.attention_ratios();
        let missing: Vec<usize> = (0..ratios.len())
            .filter(|&kv| ratios[kv] > 0 && wanted(kv) && self.pages.layer(key, kv).is_none())
            .collect();
        if missing.is_empty() {
            return Ok(0);
        }
        let d = self.model.cfg.indexer.head_dim;
        // Sized from the sections' headers: nothing is decoded to count it.
        let (_, sections) = paged_index::aux_parts(blob)?;
        let slot = candle_nn::kv_cache::slot_stride(super::super::index_keys::key_page_bytes(d));
        let mut bytes = 0usize;
        for section in missing.iter().filter_map(|&kv| sections.get(kv)) {
            let rows = paged_index::section_rows(section)?;
            bytes += rows.div_ceil(super::super::index_keys::PAGE_BLOCKS) * slot;
        }
        Ok(bytes)
    }

    /// Hold `blob`'s pages at the sampled layers in `seq`'s pool — a piece the
    /// pool admits with no K/V in the slot (`docs/hit_driven_projection.md`
    /// §7.2). Placed once per card and shared: a piece another sequence pools
    /// or places is not placed again. Returns the bytes newly placed.
    #[cfg(feature = "cuda")]
    pub fn pool_positional_state(&self, seq: usize, blob: &[u8], key: &PieceKey) -> Result<usize> {
        let sampled = self.hit_kv.clone();
        let before = self.positional_lift_bytes_at(blob, key, |kv| sampled.contains(&kv))?;
        let layers: Vec<Arc<ResidentPage>> = self
            .resident_pages(blob, key, |kv| sampled.contains(&kv))?
            .into_iter()
            .flatten()
            .collect();
        self.pooled.hold(seq, *key, layers)?;
        Ok(before)
    }

    /// Let go of `seq`'s pooled pieces not in `keep`: the pool dropped them,
    /// and their rows go back to the tenant once nothing else holds them.
    pub fn retain_pooled(&self, seq: usize, keep: &[PieceKey]) -> Result<()> {
        self.pooled.retain(seq, keep)
    }

    /// The pieces `seq`'s pool holds.
    pub fn pooled_keys(&self, seq: usize) -> Vec<PieceKey> {
        self.pooled.keys(seq)
    }

    /// Install a sealed page per attention layer ahead of `seq`'s live tail.
    /// `key` is the page's [`PieceKey`], taken by the caller.
    #[cfg(feature = "cuda")]
    pub fn push_positional_state(&self, seq: usize, blob: &[u8], key: &PieceKey) -> Result<()> {
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        let ratios = self.attention_ratios();
        // Resolved before the lock is taken: a placement is a launch.
        let layers = self.resident_pages(blob, key, |_| true)?;
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let caches = map.entry(seq).or_default();
        if caches.is_empty() {
            for _ in 0..want {
                caches.push(IndexCache::new(cfg.indexer.head_dim, &self.model.device)?);
            }
        }
        let mut pushed = 0usize;
        let mut had_open_tail = None;
        for ((c, &ratio), page) in caches.iter_mut().zip(ratios.iter()).zip(layers) {
            let Some(page) = page else { continue };
            if had_open_tail.is_none() {
                let (blocks, open) = c.seal_shape();
                had_open_tail = Some(blocks * ratio + open);
            }
            pushed = page.tokens(ratio);
            // Abutting the last placement. The base is where the scorer rotates
            // the page's rows to, so this is the one line that decides where the
            // piece's rows actually sit — and, since it is recorded rather than
            // accumulated, a preceding piece that carried no rows moves it and
            // nothing else.
            let base = c.next_base();
            c.push_page(page, base, ratio)?;
        }
        // **Only the pathological case is reported.** A page installed onto an
        // empty tail is the ordinary path and happens once per injected piece —
        // 1902 times in one startup sweep, which is noise, not evidence. A page
        // installed while the tail still holds live rows is the opposite: the
        // page lands *after* rows that chronologically precede it, so every
        // position past it resolves through the wrong block.
        if had_open_tail.is_some_and(|t| t > 0) {
            tracing::warn!(
                seq_id = seq,
                pushed,
                open_tail = had_open_tail.unwrap_or(0),
                "index: an injected page landed on a live tail — its rows sit before \
                 rows that precede them, so positions past it resolve through the wrong \
                 block"
            );
        }
        Ok(())
    }

    /// Account for injected K/V that carries no index rows, on every layer.
    ///
    /// Every layer indexes one stream, so they all move the same distance —
    /// moving some of them would leave the rest addressing different blocks for
    /// the same position, which is the divergence this exists to prevent.
    ///
    /// All this does is advance where the next page opens. It used to have to do
    /// more: positions were the running sum of the page widths, so a span that
    /// contributed none slid every later page's implied start earlier by its
    /// width, and a zero-row page had to stand in to keep the sum honest.
    /// Positions are recorded now, so an unindexed span is simply not indexed.
    pub fn push_positional_gap(&self, seq: usize, tokens: usize) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let cfg = &self.model.cfg;
        let want = cfg.kv_layers().total();
        let mut map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("qwen4exp: index lock poisoned".into()))?;
        let caches = map.entry(seq).or_default();
        if caches.is_empty() {
            for _ in 0..want {
                caches.push(IndexCache::new(cfg.indexer.head_dim, &self.model.device)?);
            }
        }
        for c in caches.iter_mut() {
            let to = c.next_base() + tokens;
            c.skip_to(to)?;
        }
        Ok(())
    }
}
