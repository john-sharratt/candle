//! A page table's starts as the tokens a prefill launch reads them at.
//!
//! The index places its pages at positions, holes and all
//! (`docs/placed_section_layout.md` §4.1), and the decode kernels read the page
//! table that way: they find a block's keys by their slices' RoPE bases. The
//! prefill kernel walks the slot's tokens instead — its causal bound, its key
//! count and its position map are all in tokens — so the table it is handed
//! names each page's first TOKEN. A page sits wholly inside one run of the
//! slot's [position runs](candle_nn::kv_cache::KvCache::chunked_position_runs),
//! since a hole is never inside one, so each start converts on its own.

/// The token `position` holds under `runs` (`[position, skipped]`, ascending,
/// the first at 0): the position less what the last run at or below it skips.
pub fn token_at(position: u32, runs: &[[u32; 2]]) -> u32 {
    let skipped = runs
        .iter()
        .take_while(|r| r[0] <= position)
        .last()
        .map_or(0, |r| r[1]);
    position - skipped
}

/// Convert a page table's `{start, blocks_before}` pairs — what
/// `IndexCache::page_prefixes` hands the kernels — from positions to tokens in
/// place. The block counts are the index's own and do not move.
pub fn pages_to_tokens(prefixes: &mut [u32], runs: &[[u32; 2]]) {
    for pair in prefixes.chunks_exact_mut(2) {
        pair[0] = token_at(pair[0], runs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holes of 100 after the first 7 tokens and of 20 after the next 2.
    const RUNS: [[u32; 2]; 3] = [[0, 0], [107, 100], [129, 120]];

    /// **A position past a hole is its token**, and one before every hole is
    /// itself.
    #[test]
    fn a_position_past_a_hole_is_its_token() {
        assert_eq!(token_at(0, &RUNS), 0);
        assert_eq!(token_at(6, &RUNS), 6);
        assert_eq!(token_at(107, &RUNS), 7);
        assert_eq!(token_at(108, &RUNS), 8);
        assert_eq!(token_at(129, &RUNS), 9);
        assert_eq!(token_at(140, &RUNS), 20);
    }

    /// **Without holes the table is unchanged.**
    #[test]
    fn without_holes_the_table_is_unchanged() {
        let mut t = vec![0, 0, 10, 3, 17, 5, 23, 7];
        pages_to_tokens(&mut t, &[[0, 0]]);
        assert_eq!(t, vec![0, 0, 10, 3, 17, 5, 23, 7]);
    }

    /// **Every page's start moves to its token; the block counts stay.** A
    /// page at 0 (3 blocks), one past the first hole at 107, one past the
    /// second at 129, and the live tail at 140.
    #[test]
    fn each_start_moves_to_its_token() {
        let mut t = vec![0, 0, 107, 3, 129, 4, 140, 7];
        pages_to_tokens(&mut t, &RUNS);
        assert_eq!(t, vec![0, 0, 7, 3, 9, 4, 20, 7]);
    }
}
