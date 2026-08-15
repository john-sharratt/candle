//! Phase 1 of the phase-locked forward: claim every KV slot the wave will
//! write, for every layer, before anything else in the forward runs.
//!
//! # Why the claims have to happen here rather than where they are needed
//!
//! The transient tier is placed between the KV arenas and the expert weights,
//! and its position is decided once per forward. For that position to be the
//! *tight* one — hard against the arena frontier, leaving the whole remainder in
//! one run adjacent to the weight side — the frontier has to be final when the
//! tier is placed (`docs/elastic_vram_partition.md` §7). A layer that claims a
//! new arena halfway through the sweep moves the frontier under a tier that is
//! already sitting on it.
//!
//! The tier does not sit there yet — anchoring it at the frontier turned up a
//! separate ordering hazard and was reverted (§13b) — so today this phase buys
//! correctness of a different kind: the claims are made once per wave instead of
//! once per layer, and `region_stats` can report that **zero** claims reach the
//! pool after a wave begins, which is the precondition the anchor needs. That it
//! reads zero on the gate is what says this module is doing its job.
//!
//! The wave's three groups are in different states, and the difference is not
//! arbitrary:
//!
//! - **Decode is already exhaustive.** `ensure_for_batch_entries_all` runs once
//!   per decode step, over every layer's backing, before the forward is entered
//!   at all. It was hoisted for an unrelated reason — 48 lock acquisitions per
//!   token in a steady state where the answer is "nothing to allocate" — but it
//!   is exactly this phase, and nothing here needs to repeat it.
//! - **Glue claims nothing.** A glue row reprojects tokens that are already
//!   resident; `glue_write_slice` and `glue_write_in_blk` name slots that exist.
//!   It still needs the truncation below, because it shares the multi-token
//!   attention entry with prefill.
//! - **Prefill claimed per layer**, and this module is where that moves to.
//!
//! # The partner that had to move with it
//!
//! Lifting the allocation alone would have been wrong, and silently so. The
//! multi-token attention entry calls `truncate_caches_to_offset` immediately
//! before the prefill claims its chunks: a re-prefill at the same offset — the
//! benchmark harness's repeat loop, or any resumed sequence — discards the stale
//! tail chunks the previous run left, and doing that *after* the claim would
//! free the chunks the claim had just made. So the order is preserved exactly as
//! the per-layer path ran it: truncate, then reset a sequence starting from
//! zero, then claim.
//!
//! What is not preserved is *when*: all of it now happens for every layer in the
//! range before layer `layer_start` computes anything, which is what makes the
//! arena frontier final for the rest of the forward.

use candle::Result;
use candle_nn::kv_cache::KvCache;

use super::kv_cache_utils::SequenceContext;

/// Claim the KV every prefill row in this wave will write, across
/// `layer_start..layer_end`.
///
/// `contexts` is in `[decode | prefill | glue]` order, so the prefill rows are
/// `n_decode..n_decode + n_prefill` and the glue rows follow them.
///
/// A wave with no multi-token rows returns without taking a lock.
pub(crate) fn admit_wave_kv(
    contexts: &mut [SequenceContext],
    n_decode: usize,
    n_prefill: usize,
    layer_start: usize,
    layer_end: usize,
) -> Result<()> {
    let multi = contexts.len().saturating_sub(n_decode);
    if multi == 0 || layer_start >= layer_end {
        return Ok(());
    }
    // Read the shape of the wave once. The loop below borrows every context's
    // caches mutably, so the offsets and lengths cannot be read from inside it.
    let offsets: Vec<usize> = contexts[n_decode..].iter().map(|c| c.offset).collect();
    let q_lens: Vec<usize> = contexts[n_decode..].iter().map(|c| c.input_len).collect();

    for layer_idx in layer_start..layer_end {
        let mut caches: Vec<&mut KvCache> = contexts[n_decode..]
            .iter_mut()
            .map(|c| &mut c.kv_caches.caches[layer_idx])
            .collect();

        // Prefill idempotency, for prefill *and* glue: truncate each sequence's
        // KV to exactly its `offset` cum-tokens, so a re-prefill at the same
        // offset does not stack duplicate chunks behind the decode writer.
        // Propagated, not discarded: this runs once per layer with a common
        // offset, so a failure that is swallowed truncates some layers and not
        // others and leaves the sequence with per-layer token windows. Failing
        // the wave is the recoverable outcome; the divergence is not.
        for (cache, &offset) in caches.iter_mut().zip(offsets.iter()) {
            cache.truncate_to_offset(offset)?;
        }

        // Prefill only, from here down. A sequence starting at zero is cleared
        // outright rather than truncated — the truncation above leaves an empty
        // chunk list, this drops the sequence's slot state with it.
        for (cache, &offset) in caches[..n_prefill].iter_mut().zip(offsets.iter()) {
            if offset == 0 {
                cache.reset();
            }
        }

        // Each sequence's own token count, never the batch maximum: an
        // over-allocated tail chunk on a shorter sequence desyncs its decode
        // writer slice.
        for (i, &add) in q_lens[..n_prefill].iter().enumerate() {
            KvCache::ensure_chunked_capacity_batch(&mut caches[i..i + 1], &offsets[i..i + 1], add)?;
        }
    }
    Ok(())
}

/// Undo a failed wave's per-layer bookkeeping: truncate every row, on every
/// layer of the range, back to the length it entered the wave with.
///
/// **This is what makes a wave failure recoverable, and nothing else does.**
/// The engine's relief design *fails waves on purpose* — an expert-cache miss
/// under pressure, a refused claim, an admission shortfall all surface as a
/// failed wave that the scheduler relieves and retries. But the layer sweep
/// advances each layer's usage as that layer completes (`set_current_seq_len`
/// in the decode and prefill attention paths), so a wave that dies between
/// layer 0's advance and layer 47's leaves the sequence with per-layer token
/// windows — layer 0 one token ahead of the other forty-seven, in the
/// measured case. No later decode can describe that with the single position
/// map every layer shares, so the sequence is bricked by exactly the failure
/// the design calls routine.
///
/// The rows' entry lengths are still in hand — `SequenceContext::offset` is
/// the pre-wave length for all three groups (decode's headers assert it
/// against the backing) — and truncating to them is the same idempotency
/// operation admit performs on the way in. Chunks the advance touched are at
/// or past `writer_start_idx` by construction (`set_len` never writes below
/// it), and `truncate_sequence_to_tokens` clamps at the sealed boundary
/// besides, so the rollback can never reach Arc-shared ground — including the
/// reserved glue gaps of a slot whose deferred fire this failed wave was
/// carrying, which must survive for the retry to scatter into.
///
/// Covers every row including decode — admit skips decode because decode's
/// claims were made by the caller, but the *advance* happens for decode rows
/// too, so the rollback cannot.
pub(crate) fn rollback_wave_kv(
    contexts: &mut [SequenceContext],
    layer_start: usize,
    layer_end: usize,
) -> Result<()> {
    for c in contexts.iter_mut() {
        let offset = c.offset;
        for layer_idx in layer_start..layer_end {
            c.kv_caches.caches[layer_idx].truncate_to_offset(offset)?;
        }
    }
    Ok(())
}
