//! The per-layer summary of a sequence's block structure that decides whether
//! every layer can share one decode position map.
//!
//! The shared map is built from layer 0 and handed to all layers, which is sound
//! only while every layer's block table agrees. Comparing a fold over each
//! chunk's offset and usage is how the decode entry point checks that instead of
//! assuming it, because a windowed prefill can leave resumed layers holding an
//! empty writer chunk the pending layers lack. The same selection rule picks the
//! writer everywhere, so the K/V write, the rope base and attention agree on it.
//! A provenance gather reads the same windows to skip partial-chunk padding.

use super::*;

/// One layer's view of a sequence's block structure, reduced to the parts the
/// shared decode position map is built from.
///
/// Two layers whose `DecodeLayout` agree produce identical maps and can share
/// one; two that disagree cannot. Produced by
/// [`SequenceState::decode_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) struct DecodeLayout {
    /// Fold over every chunk's `(offset, usage)` window, in chunk order.
    pub(in super::super) digest: u64,
    /// Number of allocated chunks — the slice count the map indexes into.
    pub(in super::super) blocks: usize,
    /// Index of the chunk the write slot resolves to. This is the one field
    /// the map records for a token that does not exist yet, and the one the
    /// per-layer `write_slice` in the slot header must match.
    pub(in super::super) writer: usize,
}

impl SequenceState {
    /// Index of the chunk the decode kernel writes into: the first non-full
    /// chunk at or after `writer_start_idx`. Chunks after it are trailing
    /// empties — e.g. a freshly-appended empty writer sitting past a partial
    /// sealed chunk — and must be skipped. This is the same selection rule
    /// `set_len` and the position_map use, so the K/V write, the rope base,
    /// and attention all agree on which chunk is the writer.
    pub(in super::super) fn decode_write_chunk_idx(&self) -> usize {
        let n = self.chunks.len();
        if n == 0 {
            return 0;
        }
        let start = self.writer_start_idx().min(n - 1);
        self.chunks[start..]
            .iter()
            .position(|c| (c.offset as usize + c.usage as usize) < CHUNK_SIZE)
            .map_or(n - 1, |i| start + i)
    }

    /// Everything the shared decode position map is derived from, reduced to a
    /// comparable value.
    ///
    /// The decode metadata builder constructs ONE position map per sequence — a
    /// `(slice_idx, in_blk)` entry per logical token, plus a final entry naming
    /// the write slot — from layer 0's block table, and hands it to all 48
    /// layers. That is only sound while every layer's block table agrees, and
    /// block structure is not unconditionally layer-uniform: a windowed creep
    /// prefill leaves resumed layers holding an empty writer chunk the layers
    /// still pending resume do not have. Comparing this value across layers is
    /// how the decode entry point establishes the invariant instead of assuming
    /// it. See [`super::ChunkedKvBacking::ensure_for_batch_entries_all`].
    pub(in super::super) fn decode_layout(&self) -> DecodeLayout {
        // FNV-1a over each chunk's window. Order matters and the windows are the
        // whole of what the map encodes, so a fold over `(offset, usage)` in
        // chunk order captures exactly as much as the map does and no more.
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for c in &self.chunks {
            for field in [c.offset as u64, c.usage as u64] {
                digest ^= field;
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        DecodeLayout {
            digest,
            blocks: self.chunks.len(),
            writer: self.decode_write_chunk_idx(),
        }
    }

    /// Per-chunk `(offset, len, cum_before)` window using the SAME derivation the
    /// decode GPU buffer does ([`rebuild_decode`]): the writer chunk gets the
    /// `seq_offset`-derived length, every other chunk keeps its stored `usage`, and
    /// `offset` is the physical skip-count where valid data begins. This is exactly
    /// the real-token window attention reads, so a provenance / diagnostic gather can
    /// consult it to check only real slots and skip partial-chunk padding. Returned
    /// in chunk order; `cum_before` is the running real-token count preceding a chunk.
    pub(crate) fn provenance_chunk_layout(&self, seq_offset: usize) -> Vec<(u16, u16, usize)> {
        let n = self.chunks.len();
        if n == 0 {
            return Vec::new();
        }
        let wi = self.decode_write_chunk_idx();
        let before_wi: usize = self.chunks[..wi].iter().map(|c| c.usage as usize).sum();
        let write_len = seq_offset.saturating_sub(before_wi).min(CHUNK_SIZE);
        let mut out = Vec::with_capacity(n);
        let mut cum = 0usize;
        for (i, c) in self.chunks.iter().enumerate() {
            let len = if i == wi {
                write_len as u16
            } else {
                (c.usage as usize).min(CHUNK_SIZE) as u16
            };
            out.push((c.offset, len, cum));
            cum += len as usize;
        }
        out
    }
}
