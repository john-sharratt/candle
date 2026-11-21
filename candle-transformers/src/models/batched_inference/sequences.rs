//! Owns the per-sequence lifecycle in a batched session: allocating slots,
//! forking and resetting them, and recording the adapter each one decodes with.
//!
//! When a sibling session shares the backing, the backing is the allocator for
//! slot indices, so two sessions cannot pick the same one. A sequence's adapter
//! is set before its first wave and left alone after, since KV written under one
//! adapter cannot be continued under another. A wave is adapter-homogeneous, and
//! a mixed wave is refused rather than given one group's adapter. A reserved slot
//! holds its index with no state, so a late handle cannot land on another
//! conversation's sequence.

use super::*;

/// Per-sequence state managed by the batch session.
#[derive(Debug)]
pub(super) struct SequenceState {
    /// Current position in the sequence (number of tokens processed).
    pub(super) offset: usize,
    /// Whether this slot is currently in use.
    pub(super) active: bool,
    /// KvCaches for this sequence (one cache per layer).
    pub(super) caches: KvCaches,
    /// The LoRA adapter this sequence decodes through, by name, or `None` for
    /// the base model.
    ///
    /// Per sequence rather than per session because one resident model serves
    /// both: the adapter is never merged into the weights, so an adapted and an
    /// unadapted conversation differ only in this field. A wave admits one
    /// value of it — see [`BatchedInferenceSession::wave_adapter`].
    pub(super) adapter: Option<Arc<str>>,
    /// What the gap quantizes this sequence's complete blocks under.
    pub(super) seal: GapSeal,
}

impl BatchedInferenceSession {
    /// Create a new sequence and return its index.
    ///
    /// The sequence starts at offset 0 (empty KV cache).
    ///
    /// When multiple sessions share the same `BackingInner` pool (created via
    /// [`new_with_backings`](Self::new_with_backings)), the backing is the
    /// canonical slot allocator.  Reading the first free slot from the backing
    /// guarantees that sibling sessions — e.g. a proto session for boundary
    /// injection — get non-overlapping slot indices even though each session
    /// maintains its own `sequences` metadata vector.
    pub fn create_sequence(&mut self) -> Result<usize> {
        // Ask the first backing for the first unallocated slot.  This is correct
        // for both the single-session case (backing only has our own slots) and
        // the shared-pool case (backing may also have slots owned by a sibling
        // session, so we must skip those to avoid collisions).
        let idx = if let Some(backing) = self.backings.first() {
            backing.first_free_slot()?
        } else {
            // No backing — fall back to scanning our own sequences vector,
            // passing over a reserved id.
            (0..)
                .find(|i| {
                    self.sequences.get(*i).is_none_or(|s| s.is_none()) && !self.reserved.contains(i)
                })
                .expect("an unbounded range has a free index")
        };

        // Ensure slot is allocated in all backings
        for backing in &self.backings {
            backing.ensure_sequence_allocated(idx)?;
        }

        // Create KvCaches for this sequence with chunked backing
        let caches = self.create_kv_caches_for_sequence(idx)?;

        // Grow the sequences vector to accommodate idx, filling any gap with None.
        // A gap arises when a sibling session owns the lower-numbered backing slots.
        if idx >= self.sequences.len() {
            self.sequences.resize_with(idx + 1, || None);
        }
        self.sequences[idx] = Some(SequenceState {
            offset: 0,
            active: true,
            caches,
            // A new slot decodes through the base model until something says
            // otherwise. The caller opts in with `set_sequence_adapter` before
            // the sequence's first wave.
            adapter: None,
            // The session's policy until the caller names the sequence's own
            // ([`Self::set_sequence_seal`]).
            seal: GapSeal::Session,
        });

        Ok(idx)
    }

    /// Choose what the gap quantizes `seq_idx`'s complete blocks under: its
    /// conversation's policy, or none — a lossless conversation's, or a
    /// section's, whose own seal quantizes it once its build has attended.
    pub fn set_sequence_seal(&mut self, seq_idx: usize, seal: GapSeal) -> Result<()> {
        match self.sequences.get_mut(seq_idx).and_then(|s| s.as_mut()) {
            Some(s) => {
                s.seal = seal;
                Ok(())
            }
            None => candle::bail!("set_sequence_seal: sequence {seq_idx} is not allocated"),
        }
    }

    /// What the gap quantizes `seq_idx` under, `None` for a free slot.
    pub fn sequence_seal(&self, seq_idx: usize) -> Option<GapSeal> {
        self.sequences
            .get(seq_idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.seal)
    }

    /// Choose the LoRA adapter this sequence decodes through, by name, or
    /// `None` for the base model.
    ///
    /// Set before the sequence's first wave and left alone after: the KV already
    /// written by an adapted prefill was produced by adapted projections, and
    /// continuing it unadapted would attend adapted keys with unadapted queries.
    pub fn set_sequence_adapter(
        &mut self,
        seq_idx: usize,
        adapter: Option<Arc<str>>,
    ) -> Result<()> {
        match self.sequences.get_mut(seq_idx).and_then(|s| s.as_mut()) {
            Some(s) => {
                s.adapter = adapter;
                Ok(())
            }
            None => candle::bail!("set_sequence_adapter: sequence {seq_idx} is not allocated"),
        }
    }

    /// The adapter `seq_idx` decodes through.
    pub fn sequence_adapter(&self, seq_idx: usize) -> Option<&str> {
        self.sequences
            .get(seq_idx)
            .and_then(|s| s.as_ref())
            .and_then(|s| s.adapter.as_deref())
    }

    /// The one adapter this wave's sequences share.
    ///
    /// **A wave is adapter-homogeneous by construction**, because the adapter
    /// changes the projections every sequence in the wave flows through
    /// together: one set of matmuls runs over the whole batch, so two sequences
    /// wanting different adapters cannot be in it. The scheduler groups by
    /// adapter and loops the waves; this is where that contract is enforced,
    /// and a mixed wave is refused rather than silently given one group's
    /// adapter — which would apply the wrong fine-tune to somebody's
    /// conversation and read as the model behaving oddly.
    /// Returns the name by `Arc` rather than by reference so the caller can
    /// hold it across the mutable borrow of the session's caches that
    /// immediately follows — an `Arc<str>` clone is a refcount bump.
    pub fn wave_adapter(&self, seqs: &[usize]) -> Result<Option<Arc<str>>> {
        let mut chosen: Option<Option<Arc<str>>> = None;
        for &s in seqs {
            let a = self
                .sequences
                .get(s)
                .and_then(|q| q.as_ref())
                .and_then(|q| q.adapter.clone());
            match &chosen {
                None => chosen = Some(a),
                Some(prev) if *prev == a => {}
                Some(prev) => candle::bail!(
                    "wave mixes LoRA adapters: sequence {s} wants {:?} but an earlier \
                     sequence in the same wave wants {:?} — waves must be grouped by adapter",
                    a,
                    prev
                ),
            }
        }
        Ok(chosen.flatten())
    }

    /// Mark the persistent decode GPU slot-state of every sequence in
    /// `seq_indices`, across every layer, after a multi-token prefill wrote
    /// tokens without the decode kernel's self-increment (a stencil static-run
    /// injection, a think-steer continuation prefill, a batched prefill's
    /// writeback). One state lock per backing for the whole set.
    ///
    /// The decode hot path trusts the cached GPU slot buffer, whose writer
    /// length self-increments only on decode steps — without this mark the
    /// next decode reuses a buffer that still ends where the injection began,
    /// writes its token over the first injected one, and leaves a slot the
    /// host counts unwritten. The buffer is marked here and its writer region
    /// re-serialised by the next sync that reads it — rebuilt in full instead
    /// when the injection crossed into a chunk the buffer was not built for,
    /// or the sequence has not decoded yet. No-op on contiguous backings.
    pub fn mark_decode_slot_states(&self, seq_indices: &[usize]) -> Result<()> {
        let entries: Vec<(usize, usize)> = seq_indices.iter().map(|&seq| (seq, 0)).collect();
        for backing in &self.backings {
            backing.mark_decode_writer_stale(&entries)?;
        }
        Ok(())
    }

    /// Create KvCaches for a sequence, wiring up chunked backing.
    pub(super) fn create_kv_caches_for_sequence(&self, seq_idx: usize) -> Result<KvCaches> {
        let compression = self.compression_policy();
        let mut caches: Vec<KvCache> = Vec::with_capacity(self.num_layers);
        for layer_idx in 0..self.num_layers {
            let mut cache = KvCache::new(2, 64); // Initial capacity, will use backing
                                                 // For float formats, set the dtype on the cache
            if let Some(dtype) = self.config.k_format.dtype() {
                cache.force_dtype(dtype);
            }
            if let Some(backing) = self.backings.get(layer_idx) {
                cache.set_chunked_backing(backing, seq_idx, compression)?;
            }
            caches.push(cache);
        }
        Ok(new_kv_caches(caches, self.device.clone()))
    }

    /// Begin a pinned-stager generation guard.
    ///
    /// All backings share the same `PinnedStager`, so we only need to call this
    /// on the first one.  The returned RAII guard keeps the pinned arena alive
    /// until dropped, ensuring device-mapped pointers remain valid for the
    /// duration of a forward pass.
    pub fn begin_stager_generation(&self) -> Generation {
        self.backings
            .first()
            .expect("begin_stager_generation: no backings")
            .begin_stager_generation_required()
    }

    /// Claim the write chunk each decode row's next token lands in, over the
    /// stream's layers — the allocation [`Self::build_decode_metadata`] makes
    /// before it describes the rows, which can create an arena.
    ///
    /// A step's prepare pass runs it, so the claim lands between forwards,
    /// before the step's tier and forward span stand; the build inside the step
    /// then finds nothing left to allocate (`docs/wave_step.md` §2.1).
    #[cfg(feature = "cuda")]
    pub fn ensure_decode_writes(&self, seq_indices: &[usize]) -> Result<()> {
        if seq_indices.is_empty() {
            return Ok(());
        }
        let entries: Vec<(usize, usize)> = seq_indices
            .iter()
            .map(|&seq_idx| {
                let offset = self
                    .sequences
                    .get(seq_idx)
                    .and_then(|s| s.as_ref())
                    .map_or(0, |s| s.offset);
                (seq_idx, offset)
            })
            .collect();
        ChunkedKvBacking::ensure_for_batch_entries_all(
            &self.backings[0..self.stream_layers],
            &entries,
            1,
        )
    }

    /// Hold slot `idx` out of the free pool, holding no KV — a slot whose state
    /// has been freed while a handle that names it may still be alive. Until
    /// that handle's own [`Self::free_sequence`] releases it, no new sequence
    /// takes the id, so a request the handle sends can never land on someone
    /// else's sequence. The backings mark it allocated too, so a sibling
    /// session sharing their pool passes over it as well.
    pub fn reserve_sequence(&mut self, idx: usize) -> Result<()> {
        if self.sequences.get(idx).is_some_and(|s| s.is_some()) {
            candle::bail!("reserve_sequence: sequence {idx} still holds state");
        }
        for backing in &self.backings {
            backing.ensure_sequence_allocated(idx)?;
        }
        self.reserved.insert(idx);
        Ok(())
    }

    /// Whether `idx` is held by [`Self::reserve_sequence`].
    pub fn is_reserved(&self, idx: usize) -> bool {
        self.reserved.contains(&idx)
    }

    /// Free a sequence, returning its resources to the pool — and releasing a
    /// [`Self::reserve_sequence`] hold on it.
    ///
    /// The sequence index should not be used after this call.
    pub fn free_sequence(&mut self, idx: usize) -> Result<()> {
        self.reserved.remove(&idx);
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }

        // Free in all backings. Each slot and its device buffer go here; the
        // block tables come back detached and drop across threads — a view's
        // is its parent's whole prefix in every layer.
        let tables = self
            .backings
            .iter()
            .map(|backing| backing.free_sequence_detached(idx))
            .collect::<Result<Vec<_>>>()?;
        layer_parallel::drop_all(tables);

        // Mark as inactive
        self.sequences[idx] = None;

        Ok(())
    }

    /// Fork a sequence, creating a new sequence that shares the KV cache prefix.
    ///
    /// This uses copy-on-write semantics - the forked sequence shares memory
    /// with the source until either is modified.
    pub fn fork_sequence(&mut self, source_idx: usize) -> Result<usize> {
        if source_idx >= self.sequences.len() {
            candle::bail!("invalid source sequence index {}", source_idx);
        }

        let source_state = self.sequences[source_idx]
            .as_ref()
            .ok_or_else(|| candle::Error::Msg(format!("sequence {} not allocated", source_idx)))?;
        if !source_state.active {
            candle::bail!("cannot fork inactive sequence {}", source_idx);
        }

        let seq_len = source_state.offset;

        // Allocate a new sequence
        let new_idx = self.create_sequence()?;

        // Fork in all backings
        for (i, backing) in self.backings.iter().enumerate() {
            backing
                .fork_sequence(source_idx, new_idx, seq_len)
                .map_err(|e| {
                    candle::Error::Msg(format!(
                        "fork_sequence: layer {i} of {} refused forking sequence {source_idx} \
                     (seq_len {seq_len}) into {new_idx} after {i} earlier layer(s) already \
                     forked theirs — the new sequence's layers now disagree on its length: {e}",
                        self.backings.len(),
                    ))
                })?;
        }

        // Set the offset to match source
        if let Some(ref mut state) = self.sequences[new_idx] {
            state.offset = seq_len;
        }

        Ok(new_idx)
    }

    /// Reset a sequence to the beginning (offset 0).
    ///
    /// This is more efficient than free+create for reusing a sequence slot.
    pub fn reset_sequence(&mut self, idx: usize) -> Result<()> {
        if idx >= self.sequences.len() {
            candle::bail!("invalid sequence index {}", idx);
        }

        // Free and re-allocate in all backings to clear KV data
        for backing in &self.backings {
            backing.free_sequence(idx)?;
            backing.ensure_sequence_allocated(idx)?;
        }

        // Create fresh caches (without borrowing sequences mutably)
        let caches = self.create_kv_caches_for_sequence(idx)?;

        // Now update the sequence state
        if let Some(ref mut state) = self.sequences[idx] {
            state.offset = 0;
            state.caches = caches;
        }

        Ok(())
    }
}
