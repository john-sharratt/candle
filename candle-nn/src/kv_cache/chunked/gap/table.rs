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
    /// the chunk count. A shape read from past `start` says nothing about the
    /// chunks before its `first`, so none of the run is known complete.
    fn complete_end(&self, start: usize) -> usize {
        if start < self.first {
            return start;
        }
        let end = self.first + self.usage.len().min(self.native.len());
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

/// Hand at most `max_blocks` of sequence `batch_idx`'s complete blocks to a
/// gap: the chunk range every layer of `backings` has complete past the
/// sequence's cursor, cut to `max_blocks`, with the cursor moved past what was
/// taken on every layer — never past a block the gap's table had no room for,
/// which the next gap takes. Empty when nothing is complete, when `max_blocks`
/// is zero, or when a layer does not hold the sequence.
pub fn gap_take(
    backings: &[ChunkedKvBacking],
    batch_idx: usize,
    max_blocks: usize,
) -> Range<usize> {
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
    let range = cut(take_range(cursor, &shapes), max_blocks);
    if !range.is_empty() {
        for b in backings {
            b.set_gap_cursor_for_seq(batch_idx, range.end);
        }
    }
    range
}

/// The blocks of `range` a seal quantizes on one layer, read from `shape`:
/// every native chunk holding tokens, from the writer boundary on. A sealed
/// sequence takes no more writes, so its writer chunk and a partial chunk are
/// blocks like any other — quantized over their valid tokens — and no
/// all-layer rule applies: each layer's job stands alone, and the seal's
/// snapshot checks the layers agree.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn seal_blocks(range: Range<usize>, shape: &LayerShape) -> Vec<usize> {
    let start = range.start.max(shape.writer_start).max(shape.first);
    let end = range
        .end
        .min(shape.first + shape.usage.len().min(shape.native.len()));
    (start..end.max(start))
        .filter(|&i| {
            let at = i - shape.first;
            shape.native[at] && shape.usage[at] > 0
        })
        .collect()
}

/// Every layer's blocks of `range` for a seal of sequence `batch_idx`, and the
/// cursor moved past `range` on every layer: what the seal takes no later gap
/// takes again. `None` when a layer does not hold the sequence.
#[cfg(feature = "cuda")]
pub(crate) fn seal_take(
    backings: &[ChunkedKvBacking],
    batch_idx: usize,
    range: Range<usize>,
) -> Option<Vec<Vec<usize>>> {
    let blocks = backings
        .iter()
        .map(|b| {
            b.gap_layer_shape(batch_idx, range.start)
                .map(|shape| seal_blocks(range.clone(), &shape))
        })
        .collect::<Option<Vec<Vec<usize>>>>()?;
    for b in backings {
        let cursor = b.gap_cursor_for_seq(batch_idx).unwrap_or(0);
        b.set_gap_cursor_for_seq(batch_idx, cursor.max(range.end));
    }
    Some(blocks)
}

/// `range` cut to its first `max` chunks.
fn cut(range: Range<usize>, max: usize) -> Range<usize> {
    range.start..range.end.min(range.start.saturating_add(max))
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
    fn a_shape_read_from_past_the_start_takes_nothing_it_did_not_read() {
        // The shape begins at chunk 3, the take at the cursor, 1: chunks 1 and
        // 2 were never read, so none of the range may be claimed.
        let shape = LayerShape {
            writer_start: 0,
            first: 3,
            usage: vec![FULL, FULL, 5],
            native: vec![true; 3],
        };
        assert!(take_range(1, &[shape]).is_empty());
    }

    #[test]
    fn usage_and_native_of_different_lengths_stop_at_the_shorter() {
        let longer_usage = layer(0, &[FULL, FULL, FULL], &[true, true]);
        assert_eq!(take_range(0, &[longer_usage]), 0..2);
        let longer_native = layer(0, &[FULL], &[true, true, true]);
        assert_eq!(take_range(0, &[longer_native]), 0..1);
    }

    #[test]
    fn an_overfull_chunk_is_not_complete() {
        let usage = [FULL, FULL + 1, FULL];
        assert_eq!(take_range(0, &[layer(0, &usage, &[true; 3])]), 0..1);
    }

    #[test]
    fn a_quantized_chunk_at_the_cursor_takes_nothing() {
        let usage = [FULL, FULL, FULL];
        let native = [true, false, true];
        assert!(take_range(1, &[layer(0, &usage, &native)]).is_empty());
    }

    #[test]
    fn layers_with_different_writer_boundaries_start_at_the_latest() {
        let usage = [FULL; 6];
        let native = [true; 6];
        assert_eq!(
            take_range(
                0,
                &[
                    layer(1, &usage, &native),
                    layer(4, &usage, &native),
                    layer(2, &usage, &native)
                ]
            ),
            4..6
        );
    }

    #[test]
    fn a_layer_with_no_chunks_takes_nothing_for_every_layer() {
        let usage = [FULL, FULL];
        assert!(take_range(0, &[layer(0, &usage, &[true; 2]), layer(0, &[], &[])]).is_empty());
    }

    #[test]
    fn a_truncate_to_nothing_and_an_eviction_of_everything_zero_the_cursor() {
        assert_eq!(cursor_after_truncate(7, 0), 0);
        assert_eq!(cursor_after_front_eviction(7, 7), 0);
        assert_eq!(cursor_after_front_eviction(0, 3), 0);
    }

    #[test]
    fn a_pure_insertion_at_the_cursor_moves_it_past_the_inserted_chunks() {
        // Nothing replaced, two chunks inserted at 4: the chunk the cursor
        // named is now at 6.
        assert_eq!(cursor_after_splice(4, 4, 4, 2), 6);
        // Inserted after the cursor: untouched.
        assert_eq!(cursor_after_splice(3, 4, 4, 2), 3);
    }

    #[test]
    fn a_pure_deletion_pulls_a_cursor_inside_it_to_its_start() {
        assert_eq!(cursor_after_splice(4, 2, 6, 0), 2);
        assert_eq!(cursor_after_splice(6, 2, 6, 0), 2);
        assert_eq!(cursor_after_splice(9, 2, 6, 0), 5);
        // The cursor exactly at the deletion's start stays there.
        assert_eq!(cursor_after_splice(2, 2, 6, 0), 2);
    }

    #[test]
    fn a_seal_takes_the_writer_and_partial_chunks_too() {
        let shape = layer(1, &[FULL, FULL, 9, FULL, 7], &[true; 5]);
        assert_eq!(seal_blocks(0..5, &shape), vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_seal_skips_quantized_and_empty_chunks_and_keeps_to_its_range() {
        let shape = layer(
            0,
            &[FULL, FULL, FULL, 0, 5],
            &[true, false, true, true, true],
        );
        assert_eq!(seal_blocks(0..5, &shape), vec![0, 2, 4]);
        assert_eq!(seal_blocks(2..4, &shape), vec![2]);
        assert!(seal_blocks(7..9, &shape).is_empty());
    }

    #[test]
    fn a_seal_reads_only_what_its_shape_holds() {
        let shape = LayerShape {
            writer_start: 0,
            first: 3,
            usage: vec![FULL, 4],
            native: vec![true, true],
        };
        assert_eq!(seal_blocks(1..9, &shape), vec![3, 4]);
    }

    #[test]
    fn a_cut_keeps_the_first_blocks_and_never_runs_past_the_range() {
        assert_eq!(cut(2..9, 3), 2..5);
        assert_eq!(cut(2..9, 100), 2..9);
        assert_eq!(cut(2..9, 0), 2..2);
        assert!(cut(2..9, 0).is_empty());
        assert_eq!(cut(4..4, 3), 4..4);
        assert_eq!(
            cut(usize::MAX - 1..usize::MAX, usize::MAX),
            usize::MAX - 1..usize::MAX
        );
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
