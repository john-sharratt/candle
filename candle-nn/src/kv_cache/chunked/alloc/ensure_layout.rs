//! Keeping a sequence's chunk layout the same on every layer, and repairing it
//! when a failed wave or a windowed prefill has left the layers apart.
//!
//! The decode builds one position map from the first layer, so every layer must
//! describe the same token windows. Layers that differ only in trailing empty
//! chunks are padded to a common writer, and a surplus token left by a failed
//! wave is truncated back to the delivered count. A difference in the windows
//! themselves is reported with the first chunk that disagrees, since that is a
//! corrupted block table.

use super::*;

impl ChunkedKvBacking {
    /// Force-push a fresh empty writer chunk onto a slot's chunk list.
    ///
    /// Unlike [`ensure_for_offset`] / [`ensure_for_batch_entries`], this
    /// always pushes a new chunk regardless of whether the slot's tail
    /// is technically writable.  Used by cumulative section ingest:
    /// after `inject_sealed_at_tail` Arc-clones the prefix sections'
    /// substrate chunks onto a fresh scratch slot, the slot's tail is
    /// the last prefix section's partial chunk (shared with substrate).
    /// Writing into it would mutate bytes other holders see as
    /// immutable section content.  Pushing a fresh empty chunk here
    /// makes the slot's *write target* a writer-owned chunk; the
    /// shared partial sits read-only just before it, and the prefill
    /// kernel starts writing at chunk-internal position 0 of the new
    /// chunk (= logical position `prefix_token_count`).
    pub fn push_empty_writer_chunk(&self, batch_idx: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let current_block_count = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.block_count())
                .unwrap_or(0)
        };
        self.ensure_max_blocks(current_block_count + 1)?;
        let cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
            slot.push_chunk(cw);
            // The freshly-appended empty chunk is now the writer. Advance the
            // writer boundary to it so any partial sealed chunk before it sits
            // below the boundary — its empty tail (gap) is then excluded from
            // the writer region and never written to or attended.
            let new_idx = slot.block_count().saturating_sub(1);
            slot.set_writer_start_idx(new_idx);
            slot.invalidate_gpu_chunks();
        } else {
            candle::bail!("push_empty_writer_chunk: slot {} not allocated", batch_idx)
        }
        Ok(())
    }

    /// Reserve an in-place **glue gap**: a fresh chunk of `n_tokens` valid slots
    /// appended at the slot tail, returning its block index. The gap's K/V is
    /// left uninitialised — the glue forward fills it via explicit `(slice,
    /// in_blk)` write targets, scattering before it streams, so nothing reads it
    /// unfilled.
    ///
    /// **The gap is full by construction.** It is allocated `offset =
    /// CHUNK_SIZE - n_tokens`, `usage = n_tokens`, so its valid window is the
    /// tail `[offset, CHUNK_SIZE)` and `offset + usage == CHUNK_SIZE`. This is
    /// the load-bearing invariant: a *partial* writer-owned chunk is, by the
    /// cache's own rules, an extendable writable tail — `extend_for_write_region`
    /// walks into it, `set_len` advances its usage, the writable-tail pass
    /// CoW-extends it. A *full* chunk is immutable to all of them: `write_slice`
    /// and `decode_write_chunk_idx` skip it, `set_len`'s cap is 0, `ensure`'s
    /// available-space sum counts it as 0, and the writable-tail pass pushes a
    /// fresh writer chunk instead of extending into it. The gap can therefore
    /// never be mistaken for the live writer region — which is exactly what makes
    /// the next prefill incapable of overflowing into it.
    ///
    /// `usage` is still exactly `n_tokens`, so the cumulative-usage `rope_base`
    /// of every later chunk equals its logical position by construction — the
    /// single positional convention the decode and glue kernels both read via
    /// `slice_rope` (a column's position is `slice_rope(c) + (in_blk - offset)`,
    /// so the tail window maps to `[rope_base, rope_base + n_tokens)`). The GIDs
    /// are unique (rc=1), so the glue's explicit write is safe and the next
    /// reproject's truncate frees them by refcount. `writer_start` is advanced
    /// PAST the gap so a subsequent sealed inject lands after it.
    ///
    /// Returns `(gap_block_index, in_blk_base)`, where `in_blk_base == offset` is
    /// the first valid slot of the tail window — the caller scatters the glue's
    /// K/V into `[in_blk_base, in_blk_base + n_tokens)` so the write lands exactly
    /// where this chunk's `slice_offset` expects it (no second, independent
    /// computation of the window can drift from the reservation).
    pub fn reserve_glue_gap_chunk(&self, batch_idx: usize, n_tokens: u32) -> Result<(usize, u32)> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        if n_tokens == 0 || n_tokens as usize > CHUNK_SIZE {
            candle::bail!(
                "reserve_glue_gap_chunk: n_tokens {} must be in 1..={CHUNK_SIZE}",
                n_tokens
            )
        }
        let current_block_count = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.block_count())
                .unwrap_or(0)
        };
        // +2: the immutable gap chunk PLUS an empty writable chunk placed after it
        // (see below).
        self.ensure_max_blocks(current_block_count + 2)?;
        // Full-by-construction: valid window is the chunk tail `[offset, 32)` with
        // `offset + usage == CHUNK_SIZE`, so the gap is immutable to every
        // writer-region scan (see the doc above).
        let offset = (CHUNK_SIZE as u32 - n_tokens) as u16;
        let cw = self.alloc_block_chunks(n_tokens, offset)?;
        // A fresh empty writer chunk to sit AFTER the gap. Without it the gap is
        // `last_chunk()`, and a co-batched decode/prefill on this same slot in the
        // unified wave validates its `last_chunk()` as the writable tail — the gap
        // is full-by-construction (`offset+usage == CHUNK_SIZE`), so the write-slice
        // check fails ("writable tail is already full/stale"), the wave forward
        // aborts, and the paged kernel is left to read a stale slot → illegal
        // address. Leaving an empty writable chunk past the gap keeps the decode's
        // `last_chunk()` a real writer (matching the write path, which already
        // targets `writer_start_idx`, not the gap). The crash root at 42553ca3.
        let writer_cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
            slot.push_chunk(cw);
            let gap_idx = slot.block_count().saturating_sub(1);
            // Advance the writer boundary PAST the gap so it is never the active
            // writer; the glue forward fills it by explicit target, and the next
            // sealed inject appends after it. The empty writer chunk we push next
            // is exactly at `gap_idx + 1`, so the boundary lands on a real tail.
            slot.set_writer_start_idx(gap_idx + 1);
            slot.push_chunk(writer_cw);
            slot.invalidate_gpu_chunks();
            Ok((gap_idx, offset as u32))
        } else {
            candle::bail!("reserve_glue_gap_chunk: slot {} not allocated", batch_idx)
        }
    }

    /// What [`Self::ensure_for_batch_entries_all`] needs to know about this
    /// layer, read under ONE guard: whether
    /// [`Self::ensure_for_batch_entries`] would allocate anything, and each
    /// entry's [`DecodeLayout`].
    ///
    /// Both answers come from the same block-table walk, and the steady-state
    /// decode step asks 48 layers this question per token, so they are read
    /// together rather than under a guard each. A layout is `None` for a slot
    /// that is not allocated — such an entry always reports work, and an
    /// unallocated slot has no layout to compare against the other layers.
    pub(super) fn probe_decode_entries(
        &self,
        entries: &[(usize, usize)],
        add: usize,
    ) -> Result<(bool, Vec<Option<DecodeLayout>>)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let mut needs_work = false;
        let mut layouts = Vec::with_capacity(entries.len());
        for &(batch_idx, _off) in entries.iter() {
            let slot = state.sequences.get(batch_idx).and_then(|s| s.as_ref());
            let Some(slot) = slot else {
                // Out of range or unallocated: let the real call allocate the
                // slot, or report the index.
                needs_work = true;
                layouts.push(None);
                continue;
            };
            layouts.push(Some(slot.decode_layout()));
            if add == 0 || needs_work {
                continue;
            }
            let available = slot
                .last_chunk()
                .map(|cw| CHUNK_SIZE - (cw.offset as usize + cw.usage as usize).min(CHUNK_SIZE))
                .unwrap_or(0);
            if available < add || self.tail_needs_new_block(&state, batch_idx) {
                needs_work = true;
            }
        }
        Ok((needs_work && add != 0, layouts))
    }

    /// Ensure EVERY layer's backing has capacity for the upcoming write.
    ///
    /// Hoisted out of the per-layer decode loop deliberately, but only the
    /// EXPENSIVE half was hoisted. `ensure_for_batch_entries` takes the state
    /// write guard, walks `ensure_max_blocks`, and allocates GIDs it usually
    /// drops again; running that 48 times per decoded token, in a steady state
    /// where the answer is "nothing to allocate", was almost entirely wasted
    /// work. What each layer still answers for itself is the read-only probe —
    /// one read guard over its own block table — and only the layers that
    /// report work enter the allocation path.
    ///
    /// **Each layer is asked, because layer 0 cannot answer for the rest.**
    /// Block structure is *meant* to be layer-invariant, and in steady decode it
    /// is, but it is not unconditionally so: a windowed creep prefill leaves the
    /// resumed layers holding an empty writer chunk for the next window while
    /// the layers still pending resume hold a full tail — the same skew
    /// `BatchedInferenceSession::reserve_glue_gap` has to pad out before it can
    /// reserve a gap index every layer agrees on. While layer 0 answered for all
    /// of them, "layer 0 has a writable tail" suppressed the allocation those
    /// other layers needed, and the first one reached refused the step:
    /// `computed write len 32 is invalid for chunk_size 32` out of
    /// `validate_decode_state`, mid-conversation, on a tail that was exactly
    /// full. A `debug_assert!` asserting the invariance is not a check in the
    /// release build the daemon runs.
    ///
    /// This is also the last point before the decode metadata builder collapses
    /// all 48 layers onto ONE position map, so the same probe carries the
    /// layout each layer would produce and [`Self::unify_decode_layout`]
    /// establishes the invariance that map depends on.
    ///
    /// **Called from inside recorded waves** — every draft-walk step builds its
    /// metadata in the recording — so the probe is all the steady state does,
    /// and anything past it runs in an eager section. Allocating a chunk can
    /// create an arena, compact, or reconcile a diverged writer chunk, and none
    /// of that is legal on a thread recording a wave: measured on Flash-Next's
    /// four-sequence draft walk, a step whose tail needed a fresh block despite
    /// the walk's up-front ensure invalidated the capture from here.
    pub fn ensure_for_batch_entries_all(
        backings: &[ChunkedKvBacking],
        entries: &[(usize, usize)],
        add: usize,
    ) -> Result<()> {
        let probes = backings
            .iter()
            .map(|b| b.probe_decode_entries(entries, add))
            .collect::<Result<Vec<_>>>()?;
        let diverged = probes.windows(2).any(|w| w[0].1 != w[1].1);
        let work = diverged || probes.iter().any(|(needs_work, _)| *needs_work);
        let _eager = match (work, backings.first()) {
            (true, Some(b)) => Some(b.device().eager()?),
            _ => None,
        };
        let mut layouts: Vec<Vec<Option<DecodeLayout>>> = Vec::with_capacity(backings.len());
        for (b, (needs_work, layout)) in backings.iter().zip(probes) {
            if needs_work {
                b.ensure_for_batch_entries(entries, add)?;
                // The allocation changed this layer's structure, so the layout
                // read alongside the predicate no longer describes it.
                layouts.push(b.probe_decode_entries(entries, 0)?.1);
            } else {
                layouts.push(layout);
            }
        }
        Self::unify_decode_layout(backings, entries, &layouts)
    }

    /// Bring every layer's block structure for `entries` back into agreement,
    /// so the one position map built from layer 0 describes all of them.
    ///
    /// The map encodes a `(slice_idx, in_blk)` pair per logical token and a
    /// final entry naming the write slot, and every layer's slot header points
    /// at it. The per-layer `write_slice` the kernel SCATTERS through, however,
    /// comes from that layer's own block table. Let the two disagree and the
    /// token is written into one chunk while attention is told to read it from
    /// another — no fault, no error, just a wrong answer.
    ///
    /// Divergence is repaired rather than reported because it is a state the
    /// engine legitimately produces: a windowed creep prefill leaves the
    /// resumed layers holding an empty writer chunk for the next window while
    /// the layers still pending resume do not have one, the same skew
    /// `BatchedInferenceSession::reserve_glue_gap` pads out before it can pick
    /// one gap index. The repair gives every layer a fresh empty writer chunk
    /// at a common index: pad the short layers up to the longest, then push one
    /// more onto ALL of them, so afterwards every layer has the same block
    /// count and its writer is that last, empty chunk. An empty chunk carries
    /// no tokens, so it shifts no position — the cumulative-usage rope base of
    /// every earlier chunk is untouched.
    ///
    /// What cannot be repaired is a difference in the token windows themselves:
    /// the shared prefix of the map is then wrong for some layer and no
    /// appending fixes it. That is a corrupted block table, and it errors.
    /// The first chunk index whose `(offset, usage)` differs between any two
    /// layers, with both readings and the layer numbers — the sentence that
    /// turns a digest mismatch into a place to look.
    ///
    /// Compares every layer against layer 0 and reports the earliest index that
    /// disagrees. Earliest rather than all of them because a partially-applied
    /// per-layer operation diverges from the point it stopped, so the first
    /// index *is* the boundary, and the layer number says how far it got.
    pub(super) fn first_window_divergence(
        backings: &[ChunkedKvBacking],
        batch_idx: usize,
    ) -> Option<String> {
        let windows = |b: &ChunkedKvBacking| -> Option<Vec<(u16, u32)>> {
            let state = b.state.read().ok()?;
            let slot = state.sequences.get(batch_idx)?.as_ref()?;
            Some(
                slot.chunks_slice()
                    .iter()
                    .map(|c| (c.offset, c.usage))
                    .collect(),
            )
        };
        let all: Vec<Option<Vec<(u16, u32)>>> = backings.iter().map(windows).collect();
        let base = all.first()?.as_ref()?;
        // The earliest chunk any layer disagrees with layer 0 about.
        let mut first_bad: Option<usize> = None;
        for other in all.iter().flatten().skip(1) {
            for i in 0..base.len().min(other.len()) {
                if base[i] != other[i] {
                    first_bad = Some(first_bad.map_or(i, |f: usize| f.min(i)));
                    break;
                }
            }
        }
        let Some(i) = first_bad else {
            let lens: Vec<usize> = all.iter().flatten().map(|w| w.len()).collect();
            let (min, max) = (lens.iter().min()?, lens.iter().max()?);
            return Some(format!(
                "Windows agree chunk-for-chunk; the layers hold between {min} and {max} \
                 chunks, so the difference is in trailing structure alone."
            ));
        };
        // **Group the layers by what they hold there.** Which layers differ is
        // the whole diagnosis: one layer out of forty-eight means something
        // special-cases that index, a contiguous prefix means a per-layer loop
        // stopped early, and a scatter means an operation applied per layer with
        // its own predicate. "Layer 0 differs from layer 1" distinguishes none
        // of those, which is why the first report of this could not be placed.
        let mut groups: Vec<((u16, u32), Vec<usize>)> = Vec::new();
        for (li, w) in all.iter().enumerate() {
            let Some(w) = w else { continue };
            let Some(&v) = w.get(i) else { continue };
            match groups.iter_mut().find(|(k, _)| *k == v) {
                Some((_, ls)) => ls.push(li),
                None => groups.push((v, vec![li])),
            }
        }
        let render = |ls: &[usize]| -> String {
            if ls.len() > 6 {
                format!("{} layers ({}…{})", ls.len(), ls[0], ls[ls.len() - 1])
            } else {
                format!(
                    "layer{} {}",
                    if ls.len() == 1 { "" } else { "s" },
                    ls.iter()
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        };
        let split = groups
            .iter()
            .map(|(v, ls)| format!("(offset {}, usage {}) on {}", v.0, v.1, render(ls)))
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "First difference at chunk {i} of {}: {split}.",
            base.len(),
        ))
    }

    /// Whether any two layers hold `batch_idx`'s **tokens** differently —
    /// per-chunk `(offset, usage)` with trailing empty chunks ignored, so
    /// trailing structure (which the writer-pad repair owns) does not count.
    pub(super) fn data_windows_disagree(backings: &[ChunkedKvBacking], batch_idx: usize) -> bool {
        let data_windows = |b: &ChunkedKvBacking| -> Option<Vec<(u16, u32)>> {
            let state = b.state.read().ok()?;
            let slot = state.sequences.get(batch_idx)?.as_ref()?;
            let mut w: Vec<(u16, u32)> = slot
                .chunks_slice()
                .iter()
                .map(|c| (c.offset, c.usage))
                .collect();
            while w.last().is_some_and(|&(_, u)| u == 0) {
                w.pop();
            }
            Some(w)
        };
        let mut layers = backings.iter().filter_map(data_windows);
        match layers.next() {
            Some(base) => !layers.all(|w| w == base),
            None => false,
        }
    }

    /// Repair a divergence confined to the tail: truncate every layer to the
    /// shortest layer's token count. Answers whether the layers agree afterwards.
    ///
    /// **A failed wave's signature, healed from its own properties.** The layer
    /// sweep advances usage per layer, so a wave that dies mid-sweep leaves the
    /// early layers up to one attention window ahead of the rest — tokens that
    /// were never delivered, since the wave that wrote them never retired. The
    /// rollback in `forward_wave` undoes that at the failure; this covers the
    /// state that predates the rollback or arrived through the substrate, where
    /// a sealed turn can persist the skew and reload it into a fresh sequence.
    ///
    /// Cutting to the shortest is sound precisely because the surplus is
    /// undelivered: no sampled token, no turn, no summary refers to it. Two
    /// guards keep this from ever being a destructive repair:
    ///
    /// - **Spread cap.** The layers may differ by at most `CHUNK_SIZE` tokens —
    ///   the most one failed operation leaves behind. A wider spread is not a
    ///   failed wave and gets no repair.
    /// - **The sealed prefix is untouchable.** `truncate_sequence_to_tokens`
    ///   clamps at the sealed boundary rather than cut Arc-shared ground, so a
    ///   "tail" divergence that is actually deep history leaves those layers
    ///   unchanged — and the verification below then reads the layers as still
    ///   disagreeing and reports the heal as failed, which it is.
    pub(super) fn heal_tail_divergence(
        backings: &[ChunkedKvBacking],
        batch_idx: usize,
        target_tokens: usize,
    ) -> Result<bool> {
        let totals: Vec<usize> = backings
            .iter()
            .filter_map(|b| {
                let state = b.state.read().ok()?;
                let slot = state.sequences.get(batch_idx)?.as_ref()?;
                Some(slot.chunks_slice().iter().map(|c| c.usage as usize).sum())
            })
            .collect();
        let (Some(&min), Some(&max)) = (totals.iter().min(), totals.iter().max()) else {
            return Ok(false);
        };
        // The session's own offset — how many tokens have actually been
        // delivered — is the anchor, not the shortest layer. Truncating to the
        // min looked equivalent until a failed wave left *every* layer with the
        // surplus token but one of them one further along: min was then
        // offset+1, and "heal to the shortest" preserved a token the caller
        // never received. Three honesty checks before touching anything:
        if min < target_tokens {
            // A layer holds fewer tokens than the session has delivered.
            // Truncation cannot restore missing history.
            return Ok(false);
        }
        if max == target_tokens {
            // Every layer already sits at the delivered count, yet the windows
            // disagree — that is mid-history corruption, not a tail surplus.
            return Ok(false);
        }
        if max - target_tokens > CHUNK_SIZE {
            // A surplus past one chunk is more than any single failed wave
            // leaves behind.
            return Ok(false);
        }
        for b in backings {
            b.truncate_sequence_to_tokens(batch_idx, target_tokens)?;
        }
        // Verified, not assumed — and on the *data* windows, with trailing
        // empty chunks ignored. The truncation can leave layers with different
        // counts of empty tail chunks, and that is trailing structure the
        // caller's writer-pad repair equalises; what the heal has to establish
        // is that every token the layers still hold is held identically.
        let healed = !Self::data_windows_disagree(backings, batch_idx);
        if healed {
            tracing::warn!(
                batch_idx,
                dropped = max - target_tokens,
                total = target_tokens,
                "chunked decode: healed per-layer tail divergence by truncating every \
                 layer to the session's delivered offset — the dropped tokens were \
                 written by a wave that failed before delivering them"
            );
        }
        Ok(healed)
    }

    pub(super) fn unify_decode_layout(
        backings: &[ChunkedKvBacking],
        entries: &[(usize, usize)],
        layouts: &[Vec<Option<DecodeLayout>>],
    ) -> Result<()> {
        for (ei, &(batch_idx, offset)) in entries.iter().enumerate() {
            let mut seen = layouts.iter().filter_map(|l| l[ei]);
            let Some(first) = seen.next() else {
                continue;
            };
            if seen.all(|l| l == first) {
                continue;
            }

            // Two repairs compose here, and **the heal runs first**. The
            // structure repair's `push_empty_writer_chunk` advances
            // `writer_start_idx` past every existing chunk — sealing them — and
            // a failed wave's surplus token lives in exactly the chunk that
            // would seal. Repairing structure first therefore puts the surplus
            // beyond the heal's reach (truncation clamps at the sealed
            // boundary), which is how the first ordering of this loop turned a
            // healable one-token skew into a refusal. So: while the layers hold
            // *tokens* differently, heal; once only trailing structure differs,
            // pad and push a common writer.
            let mut tail_healed = false;
            loop {
                if Self::data_windows_disagree(backings, batch_idx) {
                    if !tail_healed && Self::heal_tail_divergence(backings, batch_idx, offset)? {
                        tail_healed = true;
                        continue;
                    }
                } else {
                    // Tokens agree; equalise trailing structure. Re-read rather
                    // than trust: this round starts from the layers as they
                    // are, not from the caller's pre-repair probe.
                    let fresh: Vec<Option<DecodeLayout>> = backings
                        .iter()
                        .map(|b| b.probe_decode_entries(entries, 0).map(|(_, l)| l[ei]))
                        .collect::<Result<_>>()?;
                    let max_blocks = fresh.iter().flatten().map(|l| l.blocks).max().unwrap_or(0);
                    for (li, backing) in backings.iter().enumerate() {
                        let Some(layout) = fresh[li] else {
                            continue;
                        };
                        for _ in layout.blocks..=max_blocks {
                            backing.push_empty_writer_chunk(batch_idx)?;
                        }
                    }

                    let after: Vec<Option<DecodeLayout>> = backings
                        .iter()
                        .map(|b| b.probe_decode_entries(entries, 0).map(|(_, l)| l[ei]))
                        .collect::<Result<_>>()?;
                    let mut it = after.iter().flatten();
                    let Some(&expected) = it.next() else { break };
                    if it.all(|l| *l == expected) {
                        break;
                    }
                }

                // **Name the chunk, not just the digest.** The digest proves
                // a difference and says nothing about where, and "somewhere
                // in fifty-eight chunks" is not a lead — the first time this
                // fired the cause (a per-layer fork refused part-way) had to
                // be inferred from what the user happened to be doing. The
                // windows are already in hand; reporting the first index
                // that differs, with both sides' values, costs a walk of a
                // list this code has just read.
                let detail = Self::first_window_divergence(backings, batch_idx)
                    .unwrap_or_else(|| "no per-chunk difference found on re-read".to_string());
                candle::bail!(
                    "chunked decode layout diverged across layers for batch_idx {batch_idx} \
                     and could not be reconciled or healed: {detail} Appending a common \
                     writer chunk equalises trailing structure, and a tail difference of up \
                     to one chunk is healed by truncating every layer to the shortest — a \
                     difference that survives both is either deep in the sealed history or \
                     wider than any single failed operation leaves, which no repair from \
                     here can be trusted with."
                )
            }

            // `warn`, not `debug`: layers disagreeing on block structure is an
            // anomaly the engine repaired, not routine bookkeeping, and it is
            // rare by construction — the repair leaves every layer identical,
            // so a second report means something re-created the skew. If this
            // ever becomes frequent enough to be noise, the frequency is the
            // finding.
            tracing::warn!(
                batch_idx,
                tail_healed,
                "chunked decode: layer block structure diverged; reconciled onto a common \
                 writer chunk"
            );
        }
        Ok(())
    }
}
