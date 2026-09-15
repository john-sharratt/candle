//! The agreement between a sequence's QSA index and the K/V it holds.
//!
//! Every attention layer's index cache has to account for exactly the tokens
//! the sequence holds — as injected pages, skipped (unindexed) spans, completed
//! blocks, and the open block's carried rows. `IndexCache::indexed_tokens`
//! counts all four, so at the moment a wave enters, before any layer has
//! appended its rows, the two numbers are equal or the index is misplaced.
//!
//! Exact, not within a block. A cache off by fewer tokens than the compression
//! ratio still resolves every query through a block that starts somewhere else:
//! the selection names the wrong cells and the rotation it scores with is taken
//! at the wrong distance, and nothing errors. A tolerance of one block is the
//! blind spot where that failure lives.

/// The layers whose index does not cover exactly `held` tokens, as
/// `(kv layer, tokens indexed)`, in the order given.
///
/// `indexed` is one `(kv layer, tokens indexed)` pair per attention layer.
/// Short means K/V arrived without its rows and the select refuses past the
/// identity threshold; long means rows claim positions the sequence does not
/// hold, which selects against them without erroring. Either is reported.
pub fn coverage_disagreements(
    indexed: impl IntoIterator<Item = (usize, usize)>,
    held: usize,
) -> Vec<(usize, usize)> {
    indexed
        .into_iter()
        .filter(|&(_, have)| have != held)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::coverage_disagreements;

    #[test]
    fn an_index_covering_exactly_the_held_tokens_reports_nothing() {
        let indexed = [(3, 7_656), (7, 7_656), (11, 7_656)];
        assert_eq!(coverage_disagreements(indexed, 7_656), Vec::new());
    }

    /// Every offset of one to three tokens, in both directions — the band a
    /// one-block tolerance at ratio 4 let through without a word.
    #[test]
    fn an_index_off_by_less_than_a_block_is_reported() {
        let indexed = [
            (0, 100),
            (1, 99),
            (2, 98),
            (3, 97),
            (4, 101),
            (5, 102),
            (6, 103),
        ];
        assert_eq!(
            coverage_disagreements(indexed, 100),
            vec![(1, 99), (2, 98), (3, 97), (4, 101), (5, 102), (6, 103)]
        );
    }

    #[test]
    fn an_index_off_by_whole_blocks_is_reported_with_its_count() {
        let indexed = [(0, 96), (1, 100), (2, 104), (3, 0)];
        assert_eq!(
            coverage_disagreements(indexed, 100),
            vec![(0, 96), (2, 104), (3, 0)]
        );
    }

    #[test]
    fn a_sequence_holding_nothing_agrees_with_an_empty_index() {
        assert_eq!(coverage_disagreements([(0, 0), (1, 0)], 0), Vec::new());
    }
}
