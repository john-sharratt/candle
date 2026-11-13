//! Which of a sequence's blocks a gap quantizes (`docs/kv_quantize_in_the_gap.md`
//! §3.1).
//!
//! A gap takes, from every sequence a forward touched, the blocks completed
//! since the last gap: the native, full, writer-owned chunks from the
//! sequence's **cursor** — the first block not yet handed to a gap — up to the
//! first chunk a forward will still append to. The builder reads each layer's
//! shape of the sequence and nothing before the cursor, so its work is
//! proportional to what changed, never to the sequence's history.
//!
//! A block is taken only when every layer has it: a wave that died mid-sweep
//! leaves one layer a filled chunk ahead of the others, and that chunk waits
//! until the rest catch up, the rule `assert_sealed_layers_aligned` enforces at
//! a seal. The shape is read after the step's speculative rollback has settled,
//! so every full chunk holds committed tokens only.

use std::ops::Range;

use super::super::backing::ChunkedKvBacking;
use super::super::types::CHUNK_SIZE;

/// One layer's view of one sequence, as the builder reads it: the chunks from
/// `first` on, so a deep sequence is read only past its cursor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LayerShape {
    /// The layer's first writer-owned chunk. Chunks before it are shared with
    /// the substrate or a parent and are never the gap's.
    pub writer_start: usize,
    /// The index of `usage[0]` and `native[0]`.
    pub first: usize,
    /// Each chunk's valid token count, from `first`, in chunk order.
    pub usage: Vec<u32>,
    /// Whether each chunk is still native — every band takes active writes.
    /// A chunk already quantized is not taken again.
    pub native: Vec<bool>,
}

impl LayerShape {
    /// The end of the run of full native chunks that starts at `start`: the
    /// first chunk at or past `start` that is partial, empty or quantized, or
    /// the chunk count.
    fn complete_end(&self, start: usize) -> usize {
        let end = self.first + self.usage.len().min(self.native.len());
        let start = start.max(self.first);
        (start..end)
            .find(|&i| {
                let at = i - self.first;
                self.usage[at] as usize != CHUNK_SIZE || !self.native[at]
            })
            .unwrap_or(end.max(start))
    }
}

/// The chunks a gap takes from one sequence, given its `cursor` and every
/// layer's shape: empty when there is nothing complete past the cursor.
///
/// The range starts at the cursor or at the latest writer boundary, whichever
/// is later — a reprojection that injected sealed chunks moved the boundary
/// past them — and ends where the first layer stops having full native
/// chunks.
pub(crate) fn take_range(cursor: usize, layers: &[LayerShape]) -> Range<usize> {
    let Some(start) = layers.iter().map(|l| l.writer_start).max() else {
        return 0..0;
    };
    let start = start.max(cursor);
    let end = layers
        .iter()
        .map(|l| l.complete_end(start))
        .min()
        .unwrap_or(start);
    start..end.max(start)
}

/// Hand sequence `batch_idx`'s complete blocks to a gap: the chunk range every
/// layer of `backings` has complete past the sequence's cursor, with the
/// cursor moved past it on every layer. Empty when nothing is complete, or
/// when a layer does not hold the sequence.
pub fn gap_take(backings: &[ChunkedKvBacking], batch_idx: usize) -> Range<usize> {
    let Some(cursor) = backings
        .iter()
        .map(|b| b.gap_cursor_for_seq(batch_idx))
        .collect::<Option<Vec<usize>>>()
        .and_then(|c| c.into_iter().max())
    else {
        return 0..0;
    };
    let Some(shapes) = backings
        .iter()
        .map(|b| b.gap_layer_shape(batch_idx, cursor))
        .collect::<Option<Vec<LayerShape>>>()
    else {
        return 0..0;
    };
    let range = take_range(cursor, &shapes);
    if !range.is_empty() {
        for b in backings {
            b.set_gap_cursor_for_seq(batch_idx, range.end);
        }
    }
    range
}

/// The cursor after a truncate leaves `chunks` chunks: a handed block the
/// truncate cut off is no longer the gap's to take or install.
pub(crate) fn cursor_after_truncate(cursor: usize, chunks: usize) -> usize {
    cursor.min(chunks)
}

/// The cursor after the sliding ring evicted `evicted` chunks off the front:
/// every resident index moved down by that many.
pub(crate) fn cursor_after_front_eviction(cursor: usize, evicted: usize) -> usize {
    cursor.saturating_sub(evicted)
}

/// The cursor after `[start, end)` was replaced by `inserted` chunks: one past
/// the splice keeps pointing at the same chunk, and one inside it falls back
/// to the splice's start.
pub(crate) fn cursor_after_splice(
    cursor: usize,
    start: usize,
    end: usize,
    inserted: usize,
) -> usize {
    if cursor >= end {
        cursor - (end - start) + inserted
    } else {
        cursor.min(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: u32 = CHUNK_SIZE as u32;

    fn layer(writer_start: usize, usage: &[u32], native: &[bool]) -> LayerShape {
        LayerShape {
            writer_start,
            first: 0,
            usage: usage.to_vec(),
            native: native.to_vec(),
        }
    }

    #[test]
    fn a_shape_read_from_past_the_cursor_takes_the_same_blocks() {
        let whole = layer(0, &[FULL, FULL, FULL, FULL, 9], &[true; 5]);
        let tail = LayerShape {
            writer_start: 0,
            first: 2,
            usage: vec![FULL, FULL, 9],
            native: vec![true; 3],
        };
        assert_eq!(take_range(2, &[whole]), 2..4);
        assert_eq!(take_range(2, &[tail]), 2..4);
    }

    #[test]
    fn complete_blocks_are_taken_and_the_writer_chunk_is_not() {
        let usage = [FULL, FULL, FULL, 7];
        let native = [true; 4];
        assert_eq!(take_range(0, &[layer(0, &usage, &native)]), 0..3);
    }

    #[test]
    fn nothing_before_the_cursor_is_taken_again() {
        let usage = [FULL, FULL, FULL, 7];
        let native = [true; 4];
        assert_eq!(take_range(2, &[layer(0, &usage, &native)]), 2..3);
        assert_eq!(take_range(3, &[layer(0, &usage, &native)]), 3..3);
    }

    #[test]
    fn shared_chunks_below_the_writer_boundary_are_never_the_gaps() {
        let usage = [FULL, FULL, FULL, FULL, 3];
        let native = [false, false, true, true, true];
        assert_eq!(take_range(0, &[layer(2, &usage, &native)]), 2..4);
    }

    #[test]
    fn a_full_last_chunk_is_complete() {
        let usage = [FULL, FULL];
        let native = [true; 2];
        assert_eq!(take_range(0, &[layer(0, &usage, &native)]), 0..2);
    }

    #[test]
    fn the_empty_writer_and_a_pad_after_it_are_not_blocks() {
        let usage = [FULL, 0, 0];
        let native = [true; 3];
        assert_eq!(take_range(0, &[layer(0, &usage, &native)]), 0..1);
    }

    #[test]
    fn a_quantized_chunk_stops_the_run() {
        let usage = [FULL, FULL, FULL];
        let native = [true, false, true];
        assert_eq!(take_range(0, &[layer(0, &usage, &native)]), 0..1);
    }

    #[test]
    fn a_block_one_layer_lacks_waits_for_it() {
        let ahead = [FULL, FULL, FULL, 1];
        let behind = [FULL, FULL, 30];
        let native = [true; 4];
        assert_eq!(
            take_range(
                0,
                &[layer(0, &ahead, &native), layer(0, &behind, &native[..3])]
            ),
            0..2
        );
    }

    #[test]
    fn the_latest_writer_boundary_wins_over_the_cursor() {
        let usage = [FULL, FULL, FULL, FULL, 2];
        let native = [true; 5];
        assert_eq!(take_range(1, &[layer(3, &usage, &native)]), 3..4);
    }

    #[test]
    fn a_cursor_past_the_chunks_takes_nothing() {
        let usage = [FULL, 4];
        let native = [true; 2];
        assert_eq!(take_range(5, &[layer(0, &usage, &native)]), 5..5);
        assert!(take_range(5, &[layer(0, &usage, &native)]).is_empty());
    }

    #[test]
    fn no_layers_take_nothing() {
        assert_eq!(take_range(0, &[]), 0..0);
    }

    #[test]
    fn a_truncate_pulls_the_cursor_back_to_what_is_left() {
        assert_eq!(cursor_after_truncate(9, 4), 4);
        assert_eq!(cursor_after_truncate(3, 4), 3);
    }

    #[test]
    fn a_front_eviction_moves_the_cursor_down_with_the_chunks() {
        assert_eq!(cursor_after_front_eviction(9, 4), 5);
        assert_eq!(cursor_after_front_eviction(2, 4), 0);
    }

    #[test]
    fn a_splice_before_the_cursor_shifts_it_by_the_change_in_count() {
        // [2, 5) replaced by one chunk: the chunk at 7 is now at 5.
        assert_eq!(cursor_after_splice(7, 2, 5, 1), 5);
        // Exactly at the end of the splice.
        assert_eq!(cursor_after_splice(5, 2, 5, 4), 6);
        // Inside the splice: back to its start.
        assert_eq!(cursor_after_splice(3, 2, 5, 1), 2);
        // Before it: untouched.
        assert_eq!(cursor_after_splice(1, 2, 5, 1), 1);
    }
}
