//! Creates view sequences that borrow a parent's KV blocks without copying them,
//! and transfers the view's new blocks back when it finishes.
//!
//! The borrowed blocks are shared references, and the view's writable tail is a
//! fresh chunk past them, so the parent's sealed history is never written through
//! the view. Creation is all or nothing: a failure frees the view's slot before
//! it is reported, since a half-built view would pin the parent's prefix after
//! the parent let go of it. An empty borrow range is refused, because a view over
//! no blocks would decode at the parent's position with no K/V behind it.

use super::*;

impl BatchedInferenceSession {
    /// Create a view sequence that borrows KV blocks from a parent.
    ///
    /// Allocates a new sequence slot and populates it with Arc-shared refs to
    /// the specified parent blocks.  Callers write new tokens into the view, then
    /// call [`finalize_view`] to transfer those blocks back to the parent.
    ///
    /// **`visible_block_ranges` must be non-empty.** It names the blocks to
    /// borrow, so an empty slice borrows nothing and yields a view whose K/V is
    /// empty while it decodes at its parent's position — fluent, plausible text
    /// with no relation to the prompt, and no error anywhere. "Empty means every
    /// block" is the *scheduler wrapper's* convention; it expands to
    /// `[(0, total_blocks)]` before calling here. A caller that genuinely wants
    /// to borrow nothing wants [`Self::create_sequence`]. A zero-length range
    /// (`[(0, 0)]`) is still accepted: that is a parent with no blocks yet.
    pub fn create_view_sequence(
        &mut self,
        parent_idx: usize,
        visible_block_ranges: &[(usize, usize)],
    ) -> Result<ViewSequence> {
        if visible_block_ranges.is_empty() {
            candle::bail!(
                "create_view_sequence: no visible block ranges for a view of sequence \
                 {parent_idx}. Pass the ranges to borrow — `[(0, block_count)]` for the \
                 whole parent — or use `create_sequence` for a child that starts empty. \
                 A view that borrows nothing decodes from an empty context and reads \
                 perfectly."
            );
        }
        let view_idx = self.create_sequence()?;
        // The view's turn is its parent's conversation's: it seals alike.
        if let Some(seal) = self.sequence_seal(parent_idx) {
            self.set_sequence_seal(view_idx, seal)?;
        }
        // **All or nothing.** A view that fails part way holds the parent's
        // borrowed blocks on the layers it reached, with no writer opened on the
        // rest — a slot nobody will forward or free, pinning the parent's prefix
        // after the parent itself lets go of it, and claiming that sealed history
        // as its own writable region (compaction refuses to move it and says so
        // every pass). So a failure frees the slot before it is reported.
        match self.fill_view(view_idx, parent_idx, visible_block_ranges) {
            Ok(view) => Ok(view),
            Err(e) => match self.free_sequence(view_idx) {
                Ok(()) => Err(e),
                Err(free) => candle::bail!(
                    "{e} — and the half-built view {view_idx} could not be freed: {free}"
                ),
            },
        }
    }

    /// Borrow `parent_idx`'s `visible_block_ranges` into the fresh slot
    /// `view_idx` on every layer and open its writer — the fallible half of
    /// [`Self::create_view_sequence`], which frees the slot when this fails.
    pub(super) fn fill_view(
        &mut self,
        view_idx: usize,
        parent_idx: usize,
        visible_block_ranges: &[(usize, usize)],
    ) -> Result<ViewSequence> {
        // Every layer borrows its prefix at once — host bookkeeping only — and
        // then each opens its writer chunk here, where the allocation runs.
        let borrowed = layer_parallel::map(&self.backings, |backing| {
            backing.borrow_into_view(view_idx, parent_idx, visible_block_ranges)
        });
        let mut borrowed_block_count = 0;
        let mut borrowed_token_count = 0;
        for (i, (backing, result)) in self.backings.iter().zip(borrowed).enumerate() {
            let (n_blks, n_toks) = result.map_err(|e| {
                candle::Error::Msg(format!(
                    "create_view_sequence: layer {i} of {} refused a view of sequence \
                     {parent_idx} into {view_idx} — the view's layers now disagree on its \
                     length: {e}",
                    self.backings.len(),
                ))
            })?;
            if n_blks > 0 {
                backing.open_view_writer(view_idx)?;
            }
            if i == 0 {
                borrowed_block_count = n_blks;
                borrowed_token_count = n_toks;
            }
        }
        // The view's write cursor starts right after the borrowed tokens.
        let view_offset = borrowed_token_count;
        if let Some(ref mut state) = self.sequences[view_idx] {
            state.offset = view_offset;
        }
        Ok(ViewSequence {
            view_idx,
            borrowed_block_count,
        })
    }

    /// Transfer newly-written view blocks to the parent and free the view slot.
    ///
    /// `original_view_block_count` is the value returned by [`create_view_sequence`].
    pub fn finalize_view(
        &mut self,
        view_idx: usize,
        parent_idx: usize,
        original_view_block_count: usize,
    ) -> Result<()> {
        // Copy view's final offset to parent before any mutation
        let view_final_offset = self
            .sequences
            .get(view_idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.offset)
            .unwrap_or(0);

        for backing in &self.backings {
            backing.finalize_view(view_idx, parent_idx, original_view_block_count)?;
        }

        // Update parent offset to the view's final write position
        if let Some(Some(parent_state)) = self.sequences.get_mut(parent_idx) {
            parent_state.offset = view_final_offset;
        }
        // Remove view slot from session (backing already freed it)
        if view_idx < self.sequences.len() {
            self.sequences[view_idx] = None;
        }
        Ok(())
    }

    /// Get the current offset (position) of a sequence.
    pub fn sequence_offset(&self, idx: usize) -> Option<usize> {
        self.sequences.get(idx)?.as_ref().map(|s| s.offset)
    }

    /// The layer-0 writer boundary for a sequence — the first chunk index the
    /// slot may write into (chunks below it are Arc-shared / sealed).
    pub fn writer_start_idx(&self, idx: usize) -> Option<usize> {
        self.backings.first()?.writer_start_idx_for_seq(idx)
    }

    /// Mark every block `idx` currently holds (all layers) as immutable
    /// prefix — later writes append after them in a fresh writer chunk. The
    /// turn-seal re-prefill calls this after its truncate so the clean grid
    /// lands exactly at the seal anchor block.
    pub fn seal_writer_boundary(&self, idx: usize) -> Result<()> {
        for backing in &self.backings {
            backing.seal_writer_boundary(idx)?;
        }
        Ok(())
    }

    /// Get the chunk count of a sequence slot — the authoritative
    /// block total used to set borrow ranges, seal ranges, and any
    /// other "how many blocks does this slot have" question.
    ///
    /// **Why not `sequence_offset(idx) / CHUNK_SIZE`?** That formula
    /// matches only when every chunk in the slot is full.  After
    /// `inject_sealed_at_tail` materialises multiple sealed sections
    /// back-to-back, each section's trailing partial chunk stays a
    /// separate `ChunkWindow` — token-count / `CHUNK_SIZE`
    /// under-counts the real chunk total.  Using the divided value
    /// as a borrow-range bound leaves the slot's tail (one chunk
    /// per concatenated section) invisible to the view, which
    /// silently drops up to `(sections - 1) * (CHUNK_SIZE - 1)`
    /// tokens of context.
    pub fn sequence_block_count(&self, idx: usize) -> Option<usize> {
        // The backings are NOT always uniform: a windowed creep prefill can leave
        // layer 0 one empty (0-token) writer chunk AHEAD of the layers still
        // pending resume, so a section unit sealed mid-creep persists that skew.
        // Reading layer 0 would over-report by that phantom block and make a borrow
        // range (`create_view_sequence`) or seal range ask a lagging layer for a
        // block it doesn't have ("parent slot N has M blocks but range requests M").
        // The MIN across layers is the block count every layer actually holds — the
        // safe common prefix, and since the extra block is always the 0-token writer
        // chunk it carries no context to lose. This mirrors dcd075e0, which moved
        // the TOKEN reader (`sequence_backing_tokens`) to the min for the same skew;
        // the block-count reader was left on layer 0 and is the residual half of it.
        // Returns None only when the slot is unallocated on every layer.
        let counts: Vec<Option<usize>> = self
            .backings
            .iter()
            .map(|b| b.sequence_block_count(idx))
            .collect();
        let min = counts.iter().filter_map(|c| *c).min();
        if let Some(min) = min {
            // The comment above only holds if the extra block past `min` on an
            // ahead layer is genuinely empty. When it is not, this is not the
            // routine mid-creep skew — it is the same shape as the divergence
            // `heal_tail_divergence` refuses to trust, caught here instead of
            // silently truncated away by the `min()` this function returns.
            for (i, (backing, count)) in self.backings.iter().zip(&counts).enumerate() {
                let Some(count) = *count else { continue };
                if count <= min {
                    continue;
                }
                let usage = backing.block_usage(idx);
                let extra_usage: Vec<u32> = usage
                    .get(min..count)
                    .map(<[u32]>::to_vec)
                    .unwrap_or_default();
                if extra_usage.iter().any(|&u| u != 0) {
                    tracing::error!(
                        seq = idx,
                        layer = i,
                        layer_count = count,
                        min_count = min,
                        extra_block_usage = ?extra_usage,
                        all_counts = ?counts,
                        "sequence_block_count: layer {i} holds {count} blocks against a {min}-block \
                         minimum, and the block(s) past {min} are NOT empty (usage {extra_usage:?}) — \
                         real content is being dropped from this layer's view by the min-clamp, not \
                         just a phantom writer chunk",
                    );
                }
            }
        }
        min
    }

    /// Get mutable access to a sequence's KvCaches.
    pub fn sequence_caches_mut(&mut self, idx: usize) -> Option<&mut KvCaches> {
        self.sequences.get_mut(idx)?.as_mut().map(|s| &mut s.caches)
    }

    /// Get immutable access to a sequence's KvCaches.
    pub fn sequence_caches(&self, idx: usize) -> Option<&KvCaches> {
        self.sequences.get(idx)?.as_ref().map(|s| &s.caches)
    }

    /// Get mutable references to all active sequence caches and their offsets.
    ///
    /// Returns a Vec of (index, offset, &mut KvCaches) for all active sequences.
    /// This allows safe simultaneous mutable access to all caches.
    pub fn active_caches_mut(&mut self) -> Vec<(usize, usize, &mut KvCaches)> {
        self.sequences
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, slot)| {
                slot.as_mut().and_then(|state| {
                    if state.active {
                        Some((idx, state.offset, &mut state.caches))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Get mutable references to specific sequence caches and their offsets.
    ///
    /// Returns a Vec of (index, offset, &mut KvCaches) for the requested sequences.
    /// The order matches the order of seq_indices.
    pub fn caches_for_sequences(&self, seq_indices: &[usize]) -> Vec<(usize, usize, &KvCaches)> {
        // Create a set of requested indices for O(1) lookup
        let requested: HashSet<usize> = seq_indices.iter().copied().collect();

        // Collect all matching sequences
        let mut result: Vec<(usize, usize, &KvCaches)> = self
            .sequences
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                if requested.contains(&idx) {
                    slot.as_ref()
                        .map(|state| (idx, state.offset, &state.caches))
                } else {
                    None
                }
            })
            .collect();

        // Sort by the order in seq_indices
        let index_order: HashMap<usize, usize> = seq_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        result.sort_by_key(|(idx, _, _)| index_order.get(idx).copied().unwrap_or(usize::MAX));

        result
    }

    /// Get mutable references to specific sequence caches and their offsets.
    ///
    /// Returns a Vec of (index, offset, &mut KvCaches) for the requested sequences.
    /// The order matches the order of seq_indices.
    pub fn caches_for_sequences_mut(
        &mut self,
        seq_indices: &[usize],
    ) -> Vec<(usize, usize, &mut KvCaches)> {
        // Create a set of requested indices for O(1) lookup
        let requested: HashSet<usize> = seq_indices.iter().copied().collect();

        // Collect all matching sequences
        let mut result: Vec<(usize, usize, &mut KvCaches)> = self
            .sequences
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, slot)| {
                if requested.contains(&idx) {
                    slot.as_mut()
                        .map(|state| (idx, state.offset, &mut state.caches))
                } else {
                    None
                }
            })
            .collect();

        // Sort by the order in seq_indices
        let index_order: HashMap<usize, usize> = seq_indices
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();
        result.sort_by_key(|(idx, _, _)| index_order.get(idx).copied().unwrap_or(usize::MAX));

        result
    }
}

/// Result of a successful [`BatchedInferenceSession::create_view_sequence`] call.
///
/// `view_idx` is the freshly allocated view slot; `borrowed_block_count` is how
/// many blocks were shared from the parent into the view.  Pass
/// `borrowed_block_count` back to [`BatchedInferenceSession::finalize_view`] so
/// it can identify where parent-owned blocks end and newly-generated blocks begin.
#[derive(Debug, Copy, Clone)]
pub struct ViewSequence {
    /// Sequence index of the newly created view.
    pub view_idx: usize,
    /// Number of KV blocks borrowed from the parent into this view.
    /// All borrowed blocks are read-only Arc clones; the view's writable
    /// tail is a fresh chunk pushed past index `borrowed_block_count`.
    pub borrowed_block_count: usize,
}
