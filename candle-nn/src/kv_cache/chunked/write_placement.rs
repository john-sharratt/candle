//! Where `write_contiguous` puts each token of a logical range.
//!
//! A sequence is not a flat `pos / CHUNK_SIZE` grid: chunks carry a window
//! (`offset`, `usage`), so a borrowed partial tail, an injected section or a
//! hole makes logical positions and chunk slots diverge. Tokens are placed by
//! the rule the rest of the backing uses:
//!
//! - a token the sequence already holds (logical position below its total
//!   usage) is at its chunk's `offset + (pos − chunk_start)`, chunks walked in
//!   cumulative-usage order — what `read_contiguous` reads; and
//! - a new token is appended forward from the writer boundary: the first
//!   writer chunk's free slots from `offset + usage`, then the next chunk's —
//!   exactly the slots `set_len` then commits and `ensure_for_batch_entries`
//!   allocates.
//!
//! Chunks before the writer boundary are shared read-only (a parent's
//! sections, a borrowed tail); a write that would land in one is refused.

use candle::Result;

use super::{SequenceState, CHUNK_SIZE};

/// One run of consecutive tokens in one chunk: tokens `src..src + count` of
/// the caller's range go to chunk `blk`, slots `in_blk..in_blk + count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WriteRun {
    pub blk: usize,
    pub in_blk: usize,
    pub src: usize,
    pub count: usize,
}

/// The runs that place logical tokens `offset..offset + len` of `seq`.
pub(super) fn write_runs(seq: &SequenceState, offset: usize, len: usize) -> Result<Vec<WriteRun>> {
    let chunks = seq.chunks_slice();
    let writer_start = seq.writer_start_idx();
    let end = offset + len;
    let mut runs = Vec::new();

    // Tokens the sequence already holds.
    let mut cum = 0usize;
    for (blk, c) in chunks.iter().enumerate() {
        let start = cum;
        cum += c.usage as usize;
        let (lo, hi) = (start.max(offset), cum.min(end));
        if lo >= hi {
            continue;
        }
        if blk < writer_start {
            candle::bail!(
                "write_contiguous: tokens {lo}..{hi} lie in chunk {blk}, which is shared \
                 read-only (writer chunks start at {writer_start})"
            );
        }
        runs.push(WriteRun {
            blk,
            in_blk: c.offset as usize + (lo - start),
            src: lo - offset,
            count: hi - lo,
        });
    }

    // New tokens: forward from the writer boundary. `skip` is the appended
    // tokens between the sequence's end and `offset` — a gap the write leaves.
    let total = cum;
    if end > total {
        let mut pos = offset.max(total);
        let mut skip = pos - total;
        let mut idx = writer_start;
        while pos < end {
            let Some(c) = chunks.get(idx) else {
                candle::bail!(
                    "write_contiguous: tokens {pos}..{end} have no chunk to go to — allocate \
                     them first (ensure_for_offset)"
                );
            };
            let base = c.offset as usize + c.usage as usize;
            let cap = CHUNK_SIZE - base;
            if skip >= cap {
                skip -= cap;
                idx += 1;
                continue;
            }
            let in_blk = base + skip;
            let count = (CHUNK_SIZE - in_blk).min(end - pos);
            runs.push(WriteRun {
                blk: idx,
                in_blk,
                src: pos - offset,
                count,
            });
            pos += count;
            skip = 0;
            idx += 1;
        }
    }
    Ok(runs)
}

/// How many tokens a write of `offset..offset + len` appends past the
/// sequence's current end — what the backing must allocate writer capacity
/// for.
pub(super) fn appended_tokens(seq: Option<&SequenceState>, offset: usize, len: usize) -> usize {
    let total: usize = seq
        .map(|s| s.chunks_slice().iter().map(|c| c.usage as usize).sum())
        .unwrap_or(0);
    (offset + len).saturating_sub(total)
}
