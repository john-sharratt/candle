//! Manages the sliding-window ring of a slot, which drops its front chunks once
//! they leave the window.
//!
//! `evict_window_front` frees the chunks that lie wholly outside the window and
//! folds their token count into the slot's evicted-front position. Positions
//! stay absolute, so the remaining chunks keep the same RoPE bases and attention
//! is unchanged, since only fully masked keys are dropped. Other functions read
//! or seed that position and set a block's geometry directly, which the resume
//! path uses to restore a persisted window. The test-only setters build the
//! partial-chunk seams that the decode tests need.

use super::*;

impl ChunkedKvBacking {
    /// Slide the sliding-window ring for `batch_idx`: free every front chunk
    /// that has fully exited the `window_size`-token window ending at absolute
    /// query position `abs_pos`, returning the slot's new evicted-token count
    /// (`base_pos` — the absolute position of the first resident token).
    ///
    /// This is what makes the raw window a bounded RING rather than an
    /// unbounded arena: without it the slot accumulates one chunk per 32 tokens
    /// forever and the decode kernel walks every one (masking all but the last
    /// `window_size`). Evicting fully-out-of-window front chunks bounds both the
    /// resident FP8 footprint and the per-step tile walk to `O(window_size)`.
    /// Positions stay ABSOLUTE across the slide (the freed tokens' count folds
    /// into `base_pos`, seeding every remaining chunk's serialised `rope_base`),
    /// so the attention numerics are identical to the never-evicted arena — only
    /// fully-masked keys are dropped. Safe because every evicted token has
    /// already been folded into the compressed corpus (`window_size ≥
    /// compress_ratio`). A no-op for `window_size == 0` or an empty slot.
    pub fn evict_window_front(
        &self,
        batch_idx: usize,
        window_size: usize,
        abs_pos: usize,
    ) -> Result<u32> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let Some(seq) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) else {
            return Ok(0);
        };
        Ok(seq.evict_front_window(window_size, abs_pos))
    }

    /// The sliding-window ring's evicted-front count for `batch_idx` — the
    /// absolute position of the first resident token (`base_pos`). Zero until
    /// the ring first slides; `Ok(0)` for an unallocated slot.
    pub fn window_base_pos(&self, batch_idx: usize) -> Result<u32> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        Ok(state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.base_pos())
            .unwrap_or(0))
    }

    /// The resident token count for `batch_idx` — the number of tokens currently
    /// held in the arena (the sliding-window tail after eviction). `Ok(0)` for
    /// an unallocated slot.
    pub fn resident_len(&self, batch_idx: usize) -> Result<usize> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        Ok(state
            .sequences
            .get(batch_idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.seq_len())
            .unwrap_or(0))
    }

    /// Seed the sliding-window ring's evicted-front count directly (turn-seal
    /// ring restore, Artifact A): after the resident window KV is written at
    /// resident offset 0, this stamps `base_pos` so the remaining chunks
    /// serialise ABSOLUTE `rope_base` positions and the resumed window continues
    /// the original absolute frame. Errors if the slot is not allocated.
    pub fn set_window_base_pos(&self, batch_idx: usize, base_pos: u32) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let seq = state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "set_window_base_pos: slot {batch_idx} not allocated"
                ))
            })?;
        seq.set_base_pos(base_pos);
        Ok(())
    }

    /// Set one block's window geometry — its skip `offset` and valid token
    /// `usage` — directly.
    ///
    /// Used by the resume / cold-load path: after [`Self::alloc_sealed_block`]
    /// allocates a chunk's GIDs, the persisted `offset` / `token_count` must
    /// be stamped onto the `ChunkWindow` so [`Self::record_turn`] snapshots
    /// the correct window. Errors if the slot or block is not allocated.
    pub fn set_block_window(
        &self,
        batch_idx: usize,
        block_idx: usize,
        offset: u16,
        usage: u32,
    ) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let seq = state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!("set_block_window: slot {batch_idx} not allocated"))
            })?;
        let cw = seq.chunk_at_mut(block_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "set_block_window: block {block_idx} not allocated in slot {batch_idx}"
            ))
        })?;
        cw.offset = offset;
        cw.usage = usage;
        Ok(())
    }

    /// Test-only: set the decode writer-start block index for `batch_idx`.
    ///
    /// Blocks before `writer_start` are treated as sealed (never selected as the
    /// decode writer). Combined with [`Self::set_block_window`], this builds the
    /// substrate-seal gap — a partial sealed chunk (`usage < CHUNK_SIZE`) followed
    /// by a fresh writer chunk — that the decode-kernel gap-handling tests need.
    /// Invalidates the cached GPU slot buffer so the next decode rebuilds it.
    pub fn test_set_writer_start(&self, batch_idx: usize, writer_start: usize) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let seq = state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "test_set_writer_start: slot {batch_idx} not allocated"
                ))
            })?;
        seq.set_writer_start_idx(writer_start);
        seq.invalidate_gpu_chunks();
        Ok(())
    }

    /// Test-only: set one block's palette maps and outer scales directly.
    ///
    /// `k_pal` / `v_pal` are packed 2-bit maps, `n_kv_head × (head_dim/4)` bytes
    /// each (empty ⇒ identity routing); `k_scale` / `v_scale` are
    /// `n_kv_head × N_PALETTE` f32s (empty ⇒ unity). Drops the block's resident
    /// meta record (the pal/scale it captured are now stale) and invalidates the
    /// cached GPU slot buffer so the next forward serializes the new routing.
    /// Combined with [`Self::set_block_window`] + [`Self::test_set_writer_start`],
    /// this builds a prefix whose adjacent chunks carry *different* palette maps —
    /// what the prefill straddle-routing regression test needs.
    pub fn test_set_block_palette(
        &self,
        batch_idx: usize,
        block_idx: usize,
        k_pal: Vec<u8>,
        v_pal: Vec<u8>,
        k_scale: Vec<f32>,
        v_scale: Vec<f32>,
    ) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let seq = state
            .sequences
            .get_mut(batch_idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "test_set_block_palette: slot {batch_idx} not allocated"
                ))
            })?;
        let cw = seq.chunk_at_mut(block_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "test_set_block_palette: block {block_idx} not allocated in slot {batch_idx}"
            ))
        })?;
        cw.k_pal = Arc::new(k_pal);
        cw.v_pal = Arc::new(v_pal);
        cw.k_scale = Arc::new(k_scale);
        cw.v_scale = Arc::new(v_scale);
        cw.meta = None;
        seq.invalidate_gpu_chunks();
        Ok(())
    }
}
