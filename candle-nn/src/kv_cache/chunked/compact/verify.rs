//! Checks after the copy that each destination holds the bytes its source held,
//! and rewrites this backing's own holders.
//!
//! Address checks cannot see a copy that landed with wrong bytes, and such bytes
//! decode to plausible wrong values. Every destination is compared with its
//! source after the fence, while the sources are still live, within a byte
//! budget. A mismatch is read twice, which separates a copy that never landed
//! from a source written after it did. `rewrite_own_holders` updates this
//! backing's block tables and reports which slots it touched.

use super::*;

impl super::super::backing::ChunkedKvBacking {
    /// Compare every one of the pass's destinations against its source.
    ///
    /// Called after the copy's fence, while both ends are still live — the sources are
    /// not released until the end of the pass, so this is the one window in which the
    /// two can be compared at all.
    ///
    /// Reads raw device addresses rather than tensors because that is what the records
    /// are: `srcs`/`dsts` are the addresses the migrate kernel was given, and checking
    /// anything else would be checking a different claim.
    #[cfg(feature = "tensor-assert")]
    pub(super) fn verify_copied_bytes(&self, records: &CopyRecords) -> Result<()> {
        /// Bytes this check may pull back per pass, over both ends.
        ///
        /// **Sized to cover a whole pass, not to sample one.** It began at sixteen
        /// slots, which against ten thousand moves is 0.16% coverage — and a check
        /// that inspects one slot in six hundred cannot distinguish "the copy is
        /// correct" from "the corruption missed my sample", which is the only
        /// question it exists to answer. A pass moves a few MB, so reading both ends
        /// of all of it is well under a millisecond on any of the three machines'
        /// links, against a pass that already synchronises the device twice.
        const BUDGET_BYTES: usize = 32 << 20;

        let candle::Device::Cuda(cuda) = self.device() else {
            return Ok(());
        };
        let n = records.srcs.len();
        if n == 0 {
            return Ok(());
        }
        cuda.bind_to_thread().map_err(candle::Error::wrap)?;
        let mut src_buf: Vec<u8> = Vec::new();
        let mut dst_buf: Vec<u8> = Vec::new();
        let mut spent = 0usize;
        let mut checked = 0usize;
        for i in 0..n {
            let len = records.lens[i].max(0) as usize;
            if len == 0 {
                continue;
            }
            // Stop at the budget rather than thinning across the plan: the records
            // are claimed in pool order, so a prefix is one size class — but a
            // complete prefix still proves something about the slots it covered,
            // where a 0.16% stride proves nothing about any of them.
            spent += len * 2;
            if spent > BUDGET_BYTES {
                tracing::debug!(
                    target: "candle_nn::kv_cache::compact",
                    checked, moves = n,
                    "copy verification stopped at its byte budget",
                );
                break;
            }
            checked += 1;
            src_buf.resize(len, 0);
            dst_buf.resize(len, 0);
            // SAFETY: both addresses name `len` bytes of one chunk slot inside the
            // reservation — the same pair the migrate kernel was handed — and the
            // caller has fenced the copy that filled the destination.
            unsafe {
                cuda.read_raw(records.srcs[i] as u64, &mut src_buf)?;
                cuda.read_raw(records.dsts[i] as u64, &mut dst_buf)?;
            }
            if src_buf != dst_buf {
                let at = src_buf
                    .iter()
                    .zip(&dst_buf)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                // **Which end moved?** A mismatch after the fence has two readings
                // and they call for opposite fixes: the copy did not land, or the
                // SOURCE was mutated after it did — in which case the destination is
                // correct and the bug is a concurrent writer. Reading both a second
                // time separates them, because a value that changes between two host
                // reads is being written now, and one that does not was already
                // wrong when the copy retired.
                let mut src2 = vec![0u8; len];
                let mut dst2 = vec![0u8; len];
                // SAFETY: as the reads above — same two addresses, same length.
                unsafe {
                    cuda.read_raw(records.srcs[i] as u64, &mut src2)?;
                    cuda.read_raw(records.dsts[i] as u64, &mut dst2)?;
                }
                let src_moved = src2 != src_buf;
                let dst_moved = dst2 != dst_buf;
                let verdict = match (src_moved, dst_moved) {
                    (false, false) => {
                        "both ends stable across two reads — the copy \
                                       did not land"
                    }
                    (true, false) => {
                        "the SOURCE changed between two reads — a \
                                      concurrent writer, and the destination may be \
                                      correct"
                    }
                    (false, true) => {
                        "the DESTINATION changed between two reads — \
                                      something is writing the slot this pass just \
                                      filled"
                    }
                    (true, true) => {
                        "both ends changed between two reads — the pass \
                                     is running against live writers"
                    }
                };
                tracing::error!(
                    target: "candle_nn::kv_cache::compact",
                    record = i, of = n, len, %verdict,
                    src = format_args!("{:#x}", records.srcs[i]),
                    dst = format_args!("{:#x}", records.dsts[i]),
                    "compaction copy verification: {verdict}",
                );
                candle::bail!(
                    "record {i} of {n}: {len} B from {:#x} to {:#x} differ at byte \
                     {at} (source {:#04x}, destination {:#04x}). The destination does \
                     not hold what the source holds, so every holder this pass is \
                     about to point at it will read bytes that decode to plausible, \
                     wrong values.",
                    records.srcs[i],
                    records.dsts[i],
                    src_buf[at],
                    dst_buf[at],
                )
            }
        }
        Ok(())
    }

    /// Rewrite this backing's own block tables, returning the slots touched.
    ///
    /// The live side of the sweep: one `ChunkWindow` per block per slot, whose
    /// `gids` are frequently the very same allocation a sealed chunk holds — an
    /// injected sealed chunk, a view borrowing its parent's blocks, a fork. They
    /// share one `Sweep`, so such an allocation is rewritten once and both holders
    /// receive the same replacement; rewriting them independently would leave two
    /// holders with equal-but-distinct gids and refcounts that disagree with the
    /// sharing the cache believes exists.
    ///
    /// The returned slots are what must have their cached decode buffers thrown
    /// away — see the note on `invalidate_decode_slot` in [`Self::compact`].
    pub(super) fn rewrite_own_holders(&self, sweep: &mut Sweep<'_>) -> Result<usize> {
        let mut touched = 0usize;
        let mut stranded: Vec<(usize, StrandedRecords)> = Vec::new();
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (slot, entry) in state.sequences.iter_mut().enumerate() {
                let Some(seq) = entry.as_mut() else { continue };
                // `rewrite_for_compaction` invalidates the slot's cached decode
                // buffer itself when it moves anything, so there is no separate
                // invalidation step for this pass to forget.
                let rewrite = seq.rewrite_for_compaction(sweep)?;
                if rewrite.moved {
                    touched += 1;
                }
                if let Some(s) = rewrite.stranded {
                    stranded.push((slot, s));
                }
            }
        }
        // **One line per slot, named, after the lock is released.** Logged per
        // chunk inside the pass it was hundreds of lines a pass — 66,471 in a
        // minute — that named no slot, the one fact a reader needs, and that
        // lengthened the quiesced pass it was logged from.
        for (slot, s) in stranded {
            tracing::error!(
                target: "candle_nn::kv_cache::compact",
                slot,
                first_block = s.first,
                last_block = s.last,
                chunks = s.count,
                writer_start = s.writer_start,
                "writer-owned chunks carry KvHead records: they are written through their \
                 gids and read through those records, so relocating them would let the two \
                 diverge on the next token. Left where they are — some path handed this \
                 slot records at or above its write boundary",
            );
        }
        Ok(touched)
    }
}
