//! The paired key and value cache an attention layer reads and writes, over either
//! contiguous or chunked storage.
//!
//! K and V share one backing, so every write and commit must reach both sides at the
//! same length. A read that pairs a key with the wrong value would be silent. A write
//! made by a prefill or a verify block must go through the commit that marks the decode
//! slot buffer stale, since the decode kernel keeps that buffer current only for its own
//! writes. Truncation runs on both sides even when the first fails, and reports the
//! first error. The NaN integrity check covers contiguous storage only, because chunked
//! storage is validated by the kernel tests.

use super::*;

/// Result of checking KV cache integrity for NaN values.
#[derive(Debug, Clone, PartialEq)]
pub enum CacheIntegrityResult {
    /// The cache is empty.
    Empty,
    /// The cache is valid (no NaN values found).
    Valid,
    /// The cache contains NaN values.
    Invalid {
        /// Number of NaN elements found.
        nan_count: usize,
        /// Total number of elements checked.
        total_elements: usize,
        /// Percentage of elements that are NaN.
        percentage: f64,
    },
}

/// A paired key-value cache for attention.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub(super) k: Cache,
    pub(super) v: Cache,
}

impl KvCache {
    /// Create a new KV cache with contiguous backing.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = Cache::new(dim, max_seq_len);
        let v = Cache::new(dim, max_seq_len);
        Self { k, v }
    }

    /// Get a reference to the K cache.
    pub fn k_cache(&self) -> &Cache {
        &self.k
    }

    /// Get a reference to the V cache.
    pub fn v_cache(&self) -> &Cache {
        &self.v
    }

    /// Get a mutable reference to the K cache.
    pub fn k_cache_mut(&mut self) -> &mut Cache {
        &mut self.k
    }

    /// Get a mutable reference to the V cache.
    pub fn v_cache_mut(&mut self) -> &mut Cache {
        &mut self.v
    }

    /// Get per-block valid token counts for this KV cache's sequence slot.
    /// Delegates to the K cache's chunked backing. Returns empty vec for contiguous caches.
    pub fn block_usage(&self) -> Vec<u32> {
        self.k.block_usage()
    }

    /// Get per-block absolute RoPE base positions for this KV cache's sequence slot.
    /// Delegates to the K cache's chunked backing. Returns empty vec for contiguous caches.
    pub fn chunk_rope_positions(&self) -> Vec<i32> {
        self.k.chunk_rope_positions()
    }

    /// Ensure the chunked backing has capacity for `offset + add` tokens.
    /// No-op for contiguous caches. Takes `&self` (uses interior mutability).
    pub fn ensure_chunked_for_offset(&self, offset: usize, add: usize) -> Result<()> {
        self.k.ensure_chunked_for_offset(offset, add)
    }

    /// Get byte-level compression stats for a sequence (delegates to K cache).
    /// Returns the BPE (bytes per element) for the sequence, weighted by the number of tokens in K and V.
    pub fn compression_bpe(&self, batch_idx: usize) -> Option<(f64, usize)> {
        let r1 = self.k.compression_bpe(batch_idx)?;
        let r2 = self.v.compression_bpe(batch_idx)?;
        let weighted_bpe = r1.0 + r2.0;
        let total_elements = r1.1 + r2.1;
        Some((weighted_bpe, total_elements))
    }

    pub fn compression_dist(&self, batch_idx: usize, ret: &mut HashMap<GgmlDType, usize>) {
        self.k.compression_dist(batch_idx, false, ret);
        self.v.compression_dist(batch_idx, true, ret);
    }

    /// One side's compression stats — K when `is_value` is false. Read from the
    /// K cache's backing, which holds both sides' bands.
    pub fn compression_bpe_side(&self, batch_idx: usize, is_value: bool) -> Option<(f64, usize)> {
        self.k.compression_bpe_side(batch_idx, is_value)
    }

    /// One side's format histogram — K when `is_value` is false.
    pub fn compression_dist_side(
        &self,
        batch_idx: usize,
        is_value: bool,
        ret: &mut HashMap<GgmlDType, usize>,
    ) {
        self.k.compression_dist(batch_idx, is_value, ret);
    }

    /// Ensure chunked backing has the chunks needed for a batched decode step.
    ///
    /// This allocates chunks (and new arenas as needed) and updates the shared block table.
    /// Supports partial batches: each cache's batch_idx is used independently.
    pub fn ensure_chunked_capacity_batch(
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        add: usize,
    ) -> Result<()> {
        if caches.is_empty() {
            return Ok(());
        }
        if caches.len() != offsets.len() {
            candle::bail!(
                "offset count mismatch: got {} offsets for {} caches",
                offsets.len(),
                caches.len()
            )
        }
        let backing = match &caches[0].k.storage {
            CacheStorage::Chunked(c) => c.backing.clone(),
            CacheStorage::Contiguous { .. } => candle::bail!("expected chunked backing"),
        };
        // Use sparse batched ensure to keep partial-batch support while taking
        // a single chunked-state lock for the full batch operation.
        let entries: Vec<(usize, usize)> = caches
            .iter()
            .zip(offsets.iter())
            .filter_map(|(cache, &offset)| cache.k.chunked_batch_idx().map(|b| (b, offset)))
            .collect();
        backing.ensure_for_batch_entries(&entries, add)?;
        if add == 1 {
            backing.validate_decode_batch_state(&entries)?;
        }
        Ok(())
    }

    /// Validate that the selected chunked caches are ready for a decode write.
    ///
    /// This is a lightweight host-side validation that checks the current write
    /// slice invariants without mutating chunk layout or uploading metadata.
    pub fn validate_chunked_decode_batch(caches: &[&mut KvCache], offsets: &[usize]) -> Result<()> {
        if caches.is_empty() {
            return Ok(());
        }
        if caches.len() != offsets.len() {
            candle::bail!(
                "offset count mismatch: got {} offsets for {} caches",
                offsets.len(),
                caches.len()
            )
        }
        let backing = match &caches[0].k.storage {
            CacheStorage::Chunked(c) => c.backing.clone(),
            CacheStorage::Contiguous { .. } => candle::bail!("expected chunked backing"),
        };
        let entries: Vec<(usize, usize)> = caches
            .iter()
            .zip(offsets.iter())
            .filter_map(|(cache, &offset)| cache.k.chunked_batch_idx().map(|b| (b, offset)))
            .collect();
        backing.validate_decode_batch_state(&entries)?;
        Ok(())
    }

    /// Bring every selected slot's persistent slot-state buffer up to date and
    /// return each slot's `(slices_ptr, n_slices, write_slice)` — the three
    /// fields a `SlotHeader` carries besides its position map.
    ///
    /// This is the SAME buffer the decode metadata builder reads, and the same
    /// serialized `TokenSlice` array both attention kernels walk. A chunk's
    /// slice entry changes only when the slot's chunk table changes, and every
    /// mutator of that table clears the buffer (`push_chunk`, `replace_chunks`,
    /// … — see `SequenceState::sync_decode_gpu_chunks`), so a slot whose table
    /// stood still since the last sync hands back its live pointer with no
    /// re-serialisation at all. That is what makes a per-layer header build
    /// O(1) instead of O(chunks): at depth the prefix is thousands of chunks
    /// that are byte-identical from one forward to the next.
    ///
    /// `offsets` are the slots' pre-write token counts, in `caches` order, and
    /// `arena_info` the caller's already-resolved arena table — resolving it
    /// walks every arena for a device pointer, so a caller that holds one hands
    /// it over rather than paying for a second.
    ///
    /// Also returns each slot's serialisation pins — hold them until the launch
    /// against these slices has retired — and, as the third value, whether at
    /// least one slot had to be re-serialised, which is the caller's cue to
    /// re-run any check that only needs to hold over a table that changed.
    pub fn sync_chunked_slot_states(
        caches: &[&mut KvCache],
        offsets: &[usize],
        arena_info: &[super::super::ResolvedArenaInfo],
        generation: &Generation,
    ) -> Result<SlotStateSync> {
        if caches.len() != offsets.len() {
            candle::bail!(
                "offset count mismatch: got {} offsets for {} caches",
                offsets.len(),
                caches.len()
            )
        }
        let Some(first) = caches.first() else {
            return Ok((Vec::new(), Vec::new(), DecodeGpuChunkSyncStats::default()));
        };
        let backing = match &first.k.storage {
            CacheStorage::Chunked(c) => c.backing.clone(),
            CacheStorage::Contiguous { .. } => candle::bail!("expected chunked backing"),
        };
        let mut entries: Vec<(usize, usize)> = Vec::with_capacity(caches.len());
        for (i, (cache, &offset)) in caches.iter().zip(offsets.iter()).enumerate() {
            let batch_idx = cache.k.chunked_batch_idx().ok_or_else(|| {
                candle::Error::Msg(format!("slot {i} has no chunked batch index"))
            })?;
            entries.push((batch_idx, offset));
        }
        let (slots, pins, stats) =
            backing.sync_decode_gpu_chunks(&entries, arena_info, generation)?;
        Ok((slots, pins, stats))
    }

    /// Finalize sequences after generation completes.
    /// Get the current K cache data (narrowed to current_seq_len).
    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    /// Get the current V cache data (narrowed to current_seq_len).
    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    /// Get the maximum sequence length (capacity) of the cache.
    pub fn max_seq_len(&self) -> usize {
        self.k.max_seq_len()
    }

    /// Force all data written to this cache to use the specified dtype.
    pub fn force_dtype(&mut self, dtype: candle::DType) {
        self.k.force_dtype(dtype);
        self.v.force_dtype(dtype);
    }

    /// Configure this cache to use chunked (paged) backing storage.
    ///
    /// `compression_policy=None` selects uniform storage; `Some(p)` enables
    /// adaptive per-block selection at level `p.compression_level`.
    pub fn set_chunked_backing(
        &mut self,
        backing: &ChunkedKvBacking,
        batch_idx: usize,
        compression_policy: Option<CompressionPolicy>,
    ) -> Result<()> {
        self.k
            .set_chunked_backing(backing.clone(), batch_idx, compression_policy)?;
        self.v
            .set_chunked_backing(backing.clone(), batch_idx, compression_policy)?;
        Ok(())
    }

    /// Write K/V data to chunked backing storage.
    ///
    /// Expects tensors shaped (1, n_kv_head, len, head_dim).
    /// For quantized storage, data is quantized on write.
    /// `k`/`v` may live on an inference wave: this copies their bytes into the
    /// arena rather than retaining a view, so nothing here outlives the
    /// generation they came from.
    pub fn chunked_write_kv(
        &self,
        offset: usize,
        k: &candle::LiveTensor<'_>,
        v: &candle::LiveTensor<'_>,
    ) -> Result<()> {
        match (&self.k.storage, &self.v.storage) {
            (CacheStorage::Chunked(k_c), CacheStorage::Chunked(_v_c)) => {
                // Both K and V share the same backing
                k_c.write_contiguous(offset, k, v)
            }
            _ => candle::bail!("chunked_write_kv requires chunked backing"),
        }
    }

    /// Read K/V data from chunked backing storage (dequantizes if needed).
    ///
    /// Returns tensors shaped (1, n_kv_head, len, head_dim).
    pub fn chunked_read_kv(&self, offset: usize, len: usize) -> Result<(Tensor, Tensor)> {
        match &self.k.storage {
            CacheStorage::Chunked(c) => c.read_contiguous(offset, len),
            _ => candle::bail!("chunked_read_kv requires chunked backing"),
        }
    }

    /// Calculate the percentage of a sequence's tokens stored in quantized arenas.
    ///
    /// Returns (quantized_tokens, total_tokens) based on which ChunkRefs point to quantized arenas.
    /// This validates that the actual data for a sequence is quantized, not just that quantized arenas exist.
    pub fn quantized_token_stats(
        &self,
        batch_idx: usize,
    ) -> Option<candle::Result<(usize, usize)>> {
        self.k.quantized_token_stats(batch_idx)
    }

    /// Get the current dtype of the cache.
    pub fn dtype(&self) -> DType {
        self.k.dtype()
    }

    /// Append K and V tensors to the cache, returning the full accumulated tensors.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    /// Get the current sequence length stored in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    /// Set the current sequence length.
    pub fn set_current_seq_len(&mut self, seq_len: usize) -> Result<()> {
        self.k.set_current_seq_len(seq_len)?;
        self.v.set_current_seq_len(seq_len)?;
        Ok(())
    }

    /// Commit `add` tokens written at `offset` by a path that is NOT the decode
    /// kernel — a prefill, a speculative verify block — and mark the cached
    /// decode slot buffer for the next sync to bring up to date.
    ///
    /// **Every such commit must come through here.** The decode kernel keeps
    /// its slot buffer current itself: it commits each token's length on the
    /// device, and the next step reuses the buffer as it stands. A commit made
    /// anywhere else advances only the host (`set_len` deliberately never
    /// writes the serialised buffer), so the buffer's writer slice is left
    /// short by `add`, and the next decode step that reuses it writes its new
    /// token at that stale slot — over a committed one — while the host counts
    /// the slot the write should have filled. Nothing ever writes that slot:
    /// it reads as whatever it last held, which since slots are recycled is
    /// another sequence's KV, and under `tensor-assert` the claim poison.
    ///
    /// The buffer is marked, not rewritten: the next slot-state sync — which
    /// every reader of the buffer goes through — re-serialises its writer
    /// region from the host state before handing it out, so the commit itself
    /// costs the stream nothing. See `SequenceState::mark_decode_writer_stale`.
    /// A sequence with no cached buffer has nothing to mark; its next sync
    /// rebuilds it.
    pub fn commit_written_tokens(&mut self, offset: usize, add: usize) -> Result<()> {
        self.set_current_seq_len(offset + add)?;
        if let CacheStorage::Chunked(c) = &self.k.storage {
            c.backing.mark_decode_writer_stale(&[(c.batch_idx, 0)])?;
        }
        Ok(())
    }

    /// [`Self::commit_written_tokens`] for every cache of one batched write:
    /// the caches of one layer, over one shared backing, as the batched prefill
    /// holds them. Each cache's `adds[i]` tokens at `offsets[i]` are committed,
    /// then the backing's decode slot buffers are marked in a single call —
    /// one state lock for the layer rather than one per sequence.
    pub fn commit_written_tokens_batch(
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        adds: &[usize],
    ) -> Result<()> {
        if caches.len() != offsets.len() || caches.len() != adds.len() {
            candle::bail!(
                "commit count mismatch: {} caches, {} offsets, {} lengths",
                caches.len(),
                offsets.len(),
                adds.len()
            )
        }
        let Some(first) = caches.first() else {
            return Ok(());
        };
        let backing = match &first.k.storage {
            CacheStorage::Chunked(c) => Some(c.backing.clone()),
            CacheStorage::Contiguous { .. } => None,
        };
        let mut entries: Vec<(usize, usize)> = Vec::with_capacity(caches.len());
        for (i, ((cache, &offset), &add)) in caches
            .iter_mut()
            .zip(offsets.iter())
            .zip(adds.iter())
            .enumerate()
        {
            cache.set_current_seq_len(offset + add)?;
            if let (CacheStorage::Chunked(c), Some(backing)) = (&cache.k.storage, &backing) {
                // One mark names sequences by batch index in ONE backing's
                // table; a cache on another backing would be looked up in the
                // wrong table and silently left stale.
                if !c.backing.shares_state_with(backing) {
                    candle::bail!(
                        "commit batch: cache {i} is on a different chunked backing from cache 0"
                    );
                }
                entries.push((c.batch_idx, 0));
            }
        }
        if let Some(backing) = backing {
            backing.mark_decode_writer_stale(&entries)?;
        }
        Ok(())
    }

    /// Truncate the cache to the specified sequence length.
    pub fn truncate(&mut self, seq_len: usize) -> Result<()> {
        self.k.truncate(seq_len)?;
        self.v.truncate(seq_len)?;
        Ok(())
    }

    /// Try to truncate, but if OOM, do a full reset instead.
    /// Returns true if truncate succeeded, false if reset was performed.
    pub fn try_truncate_or_reset(&mut self, seq_len: usize) -> Result<bool> {
        // Try k first
        let k_success = self.k.try_truncate_or_reset(seq_len)?;

        if !k_success {
            // K failed and was reset, reset v too for consistency
            self.v.reset();
            return Ok(false);
        }

        // Try v
        let v_success = self.v.try_truncate_or_reset(seq_len)?;

        if !v_success {
            // V failed, reset k for consistency
            self.k.reset();
            return Ok(false);
        }

        Ok(true)
    }

    /// Try to reserve space for additional tokens by expanding capacity.
    /// Maintains existing tokens. Returns true if successful, false if OOM.
    pub fn try_reserve(&mut self, additional_tokens: usize) -> bool {
        if self.k.is_chunked() || self.v.is_chunked() {
            // Chunked caches don't use the contiguous capacity growth logic.
            return true;
        }
        let current_len = self.current_seq_len();
        let required_capacity = current_len + additional_tokens;

        // Already have enough space
        if required_capacity <= self.k.max_seq_len {
            return true;
        }

        // Round up to nearest multiple of grow_by (e.g., pooled chunk size)
        let new_capacity = required_capacity.div_ceil(self.k.grow_by) * self.k.grow_by;

        // Expand k cache
        if let CacheStorage::Contiguous { all_data } = &mut self.k.storage {
            if let Some(old_k) = all_data.take() {
                let mut new_shape = old_k.dims().to_vec();
                new_shape[self.k.dim] = new_capacity;

                let new_k = match Tensor::zeros(new_shape, old_k.dtype(), old_k.device()) {
                    Ok(t) => t,
                    Err(_) => {
                        *all_data = Some(old_k);
                        return false;
                    }
                };

                if current_len > 0
                    && new_k
                        .slice_set(
                            &old_k.narrow(self.k.dim, 0, current_len).unwrap(),
                            self.k.dim,
                            0,
                        )
                        .is_err()
                {
                    *all_data = Some(old_k);
                    return false;
                }

                *all_data = Some(new_k);
                self.k.max_seq_len = new_capacity;
            } else {
                self.k.max_seq_len = new_capacity;
            }
        } else {
            return true;
        }

        // Expand v cache
        if let CacheStorage::Contiguous { all_data } = &mut self.v.storage {
            if let Some(old_v) = all_data.take() {
                let mut new_shape = old_v.dims().to_vec();
                new_shape[self.v.dim] = new_capacity;

                let new_v = match Tensor::zeros(new_shape, old_v.dtype(), old_v.device()) {
                    Ok(t) => t,
                    Err(_) => {
                        *all_data = Some(old_v);
                        return false;
                    }
                };

                if current_len > 0
                    && new_v
                        .slice_set(
                            &old_v.narrow(self.v.dim, 0, current_len).unwrap(),
                            self.v.dim,
                            0,
                        )
                        .is_err()
                {
                    *all_data = Some(old_v);
                    return false;
                }

                *all_data = Some(new_v);
                self.v.max_seq_len = new_capacity;
            } else {
                self.v.max_seq_len = new_capacity;
            }
        } else {
            return true;
        }

        true
    }

    /// Reset the cache, clearing all data.
    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }

    /// Truncate both K and V to exactly `offset` cum-tokens, freeing any chunks
    /// beyond it. Makes an offset-`N` (re)prefill idempotent (see
    /// [`Cache::truncate_chunked_to_tokens`]).
    ///
    /// Both sides run even when the first fails: K and V must land at the same
    /// length or every later read pairs a key with the wrong value, so a K
    /// failure must not leave V untried. The first error is still reported —
    /// after both truncations have executed.
    pub fn truncate_to_offset(&mut self, offset: usize) -> Result<()> {
        let k = self.k.truncate_chunked_to_tokens(offset);
        let v = self.v.truncate_chunked_to_tokens(offset);
        k.and(v)
    }

    /// Fork this KV cache, creating a new cache that shares data via copy-on-write.
    ///
    /// For chunked (paged) caches: complete blocks are shared via COW, partial
    /// blocks are copied. The new cache can be written to independently without
    /// affecting the original.
    ///
    /// For contiguous caches: the data is cloned.
    ///
    /// This is useful for beam search, speculative decoding, or any scenario
    /// where you need to branch from an existing sequence state.
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            k: self.k.fork()?,
            v: self.v.fork()?,
        })
    }

    /// Check the integrity of the last position in the KV cache for NaN values.
    /// Returns Empty if cache is empty, Valid if no NaNs found, or Invalid with count and percentage.
    pub fn check_integrity(&self) -> Result<CacheIntegrityResult> {
        let seq_len = self.current_seq_len();

        // Empty cache
        if seq_len == 0 {
            return Ok(CacheIntegrityResult::Empty);
        }

        let last_pos = seq_len - 1;

        if self.k.is_chunked() || self.v.is_chunked() {
            // Integrity checks currently only support contiguous caches.
            // Chunked KV is validated by kernel-level tests.
            return Ok(CacheIntegrityResult::Valid);
        }

        // Get the last position from both k and v caches
        let k_data = match &self.k.storage {
            CacheStorage::Contiguous { all_data } => all_data.as_ref(),
            CacheStorage::Chunked(_) => None,
        };
        let v_data = match &self.v.storage {
            CacheStorage::Contiguous { all_data } => all_data.as_ref(),
            CacheStorage::Chunked(_) => None,
        };

        if k_data.is_none() || v_data.is_none() {
            return Ok(CacheIntegrityResult::Empty);
        }

        let k_data = k_data.unwrap();
        let v_data = v_data.unwrap();

        // Extract last position slice
        let k_last = k_data.narrow(self.k.dim, last_pos, 1)?;
        let v_last = v_data.narrow(self.v.dim, last_pos, 1)?;

        // Flatten and check for NaNs (convert to F32 to handle any dtype)
        let k_flat = k_last
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let v_flat = v_last
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        let mut nan_count = 0;
        let mut total_elements = 0;

        for &val in k_flat.iter() {
            total_elements += 1;
            if val.is_nan() {
                nan_count += 1;
            }
        }

        for &val in v_flat.iter() {
            total_elements += 1;
            if val.is_nan() {
                nan_count += 1;
            }
        }

        let nan_count = nan_count;
        let total_elements = total_elements;

        if nan_count > 0 {
            let percentage = (nan_count as f64 / total_elements as f64) * 100.0;
            Ok(CacheIntegrityResult::Invalid {
                nan_count,
                total_elements,
                percentage,
            })
        } else {
            Ok(CacheIntegrityResult::Valid)
        }
    }

    /// Convert the cache to the specified dtype.
    pub fn convert_dtype(&mut self, dtype: DType) -> Result<()> {
        if self.k.is_chunked() || self.v.is_chunked() {
            candle::bail!("convert_dtype is not supported for chunked KV caches")
        }
        // Convert K cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.k.storage {
            if let Some(k_data) = all_data.take() {
                if k_data.dtype() != dtype {
                    let converted = k_data.to_dtype(dtype)?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(k_data);
                }
            }
        }
        // Convert V cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.v.storage {
            if let Some(v_data) = all_data.take() {
                if v_data.dtype() != dtype {
                    let converted = v_data.to_dtype(dtype)?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(v_data);
                }
            }
        }
        Ok(())
    }

    /// Ensure the KV cache is in one of the specified dtypes.
    /// If already in one of the specified dtypes, returns without conversion.
    /// Otherwise, converts to the first dtype in the list.
    pub fn ensure_dtype(&mut self, dtypes: &[DType]) -> Result<()> {
        if dtypes.is_empty() {
            candle::bail!("ensure_dtype requires at least one dtype")
        }

        if self.k.is_chunked() || self.v.is_chunked() {
            candle::bail!("ensure_dtype is not supported for chunked KV caches")
        }

        // Convert K cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.k.storage {
            if let Some(k_data) = all_data.take() {
                if !dtypes.contains(&k_data.dtype()) {
                    let converted = k_data.to_dtype(dtypes[0])?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(k_data);
                }
            }
        }

        // Convert V cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.v.storage {
            if let Some(v_data) = all_data.take() {
                if !dtypes.contains(&v_data.dtype()) {
                    let converted = v_data.to_dtype(dtypes[0])?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(v_data);
                }
            }
        }

        Ok(())
    }
}
