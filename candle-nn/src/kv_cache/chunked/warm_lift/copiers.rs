//! The host copy of a warm lift: pageable warm arenas into a pinned ring slot
//! (`docs/warm_lift.md` §3.3).
//!
//! A DMA engine reads only page-locked memory, so a warm band reaches the GPU
//! by a CPU copy first. One thread copying from pageable memory falls short of
//! a PCIe 5.0 link; [`HOST_COPIERS`] threads, each taking a contiguous share of
//! the slot's bands, outrun a ×16 link. The threads are scoped to the slot and
//! joined before it is handed to the scatter, and take each backing's storage
//! read lock for their own run of its bands — never the global rayon pool,
//! whose queue the belief scan's warm-ups share.

use std::thread;

use candle::Result;

use super::plan::{LiftBand, SlotBatch};
use crate::kv_cache::chunked::ChunkedKvBacking;

/// Threads copying one slot.
pub(crate) const HOST_COPIERS: usize = 4;

/// Copy `batch`'s bands of `bands` from their warm arenas in `backings` into
/// `payload`, each at its offset.
pub(crate) fn fill_slot(
    backings: &[ChunkedKvBacking],
    bands: &[LiftBand],
    batch: &SlotBatch,
    payload: &mut [u8],
) -> Result<()> {
    let mine = &bands[batch.bands.clone()];
    if mine.is_empty() {
        return Ok(());
    }
    // Shares of roughly equal bytes, each a run of whole bands.
    let per = batch.payload.div_ceil(HOST_COPIERS).max(1);
    let mut cuts: Vec<usize> = vec![0];
    for (i, &off) in batch.offsets.iter().enumerate().skip(1) {
        if off >= cuts.len() * per && cuts.len() < HOST_COPIERS {
            cuts.push(i);
        }
    }
    cuts.push(mine.len());
    // Disjoint byte ranges of the slot, one a share, so each thread owns its
    // part outright.
    let mut rest: &mut [u8] = &mut payload[..batch.payload];
    let mut base = 0usize;
    let mut parts: Vec<(usize, &mut [u8], &[LiftBand], &[usize])> = Vec::new();
    for w in cuts.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let start = batch.offsets[lo];
        let end = if hi < mine.len() {
            batch.offsets[hi]
        } else {
            batch.payload
        };
        let (_, tail) = rest.split_at_mut(start - base);
        let (part, after) = tail.split_at_mut(end - start);
        rest = after;
        base = end;
        parts.push((start, part, &mine[lo..hi], &batch.offsets[lo..hi]));
    }
    thread::scope(|s| -> Result<()> {
        let handles: Vec<_> = parts
            .into_iter()
            .map(|(start, part, share, offsets)| {
                s.spawn(move || copy_share(backings, share, offsets, start, part))
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(r) => r?,
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        Ok(())
    })
}

/// One thread's share: `share`'s bands into `part`, which begins at slot
/// offset `start`. Consecutive bands of one layer are copied under one read of
/// that layer's storage.
fn copy_share(
    backings: &[ChunkedKvBacking],
    share: &[LiftBand],
    offsets: &[usize],
    start: usize,
    part: &mut [u8],
) -> Result<()> {
    let mut i = 0usize;
    while i < share.len() {
        let layer = share[i].layer;
        let mut j = i;
        while j < share.len() && share[j].layer == layer {
            j += 1;
        }
        let backing = backings.get(layer).ok_or_else(|| {
            candle::Error::Msg(format!(
                "warm lift: band names layer {layer} past the backings"
            ))
        })?;
        backing.inner.storage.read(|s| -> Result<()> {
            for k in i..j {
                let band = &share[k];
                let at = offsets[k] - start;
                let arena = s.arenas().get(&band.src_arena).ok_or_else(|| {
                    candle::Error::Msg(format!("warm lift: arena {} missing", band.src_arena))
                })?;
                arena.copy_slot_bytes(band.src_chunk, &mut part[at..at + band.bytes])?;
            }
            Ok(())
        })??;
        i = j;
    }
    Ok(())
}
