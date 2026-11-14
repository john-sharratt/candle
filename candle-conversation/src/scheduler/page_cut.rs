//! Where a wave prefill's chunk stops so each page break lands on a page edge.
//!
//! A forward pools the index rows of every token it carries into the open page,
//! so a page can only close between two forwards. The single-sequence prefill
//! splits its span at every break token for that reason (`break_pieces`); a
//! creeping wave group carries each member's prompt in chunks, so the same cut is
//! made by ending a member's chunk just past the first break it would cross. The
//! next chunk then opens the next page, and the user record, its `<|im_end|>`
//! included, occupies whole pages that it can be placed with alone
//! (`docs/hit_driven_projection.md` §8.1).

/// How far a chunk of `tokens` starting at `offset` advances when it may carry
/// at most `advance` rows: up to and including the first break token among
/// them, or all `advance` when none falls before its last row. A break that is
/// the chunk's own last row cuts nothing, exactly as `break_pieces` leaves a
/// trailing break in its piece.
pub(super) fn break_advance(
    tokens: &[u32],
    offset: usize,
    advance: usize,
    breaks: &[u32],
) -> usize {
    let end = offset.saturating_add(advance).min(tokens.len());
    tokens
        .get(offset..end)
        .and_then(|chunk| chunk.iter().position(|t| breaks.contains(t)))
        .map(|at| at + 1)
        .filter(|&at| at < end - offset)
        .unwrap_or(advance)
}

/// Whether a committed `chunk` closes its member's page: it ended on a break and
/// the prompt goes on past it. The prompt's last chunk closes none — the turn's
/// first generated token opens the next page.
pub(super) fn chunk_closes_page(chunk: &[u32], more: bool, breaks: &[u32]) -> bool {
    more && chunk.last().is_some_and(|t| breaks.contains(t))
}

#[cfg(test)]
mod tests {
    use super::{break_advance, chunk_closes_page};

    const IM_END: u32 = 900;
    const THINK: u32 = 901;
    const THINK_END: u32 = 902;
    const BREAKS: [u32; 3] = [THINK, THINK_END, IM_END];

    /// A user record of 91 tokens ending on its `<|im_end|>`, then an 18-token
    /// assistant opener carrying a closed think block — the npcd turn whose
    /// first page measured 109 against an edge of 91.
    fn npcd_turn() -> Vec<u32> {
        let mut t = vec![1u32; 90];
        t.push(IM_END);
        t.extend([2, 3, 4, THINK, 5]);
        t.push(THINK_END);
        t.extend([6u32; 12]);
        assert_eq!(t.len(), 109);
        t
    }

    /// Chunk widths a creep takes through `tokens` at `pass` rows a chunk.
    fn chunks(tokens: &[u32], pass: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut off = 0;
        while off < tokens.len() {
            let a = break_advance(tokens, off, pass.min(tokens.len() - off), &BREAKS);
            out.push(a);
            off += a;
        }
        out
    }

    /// **The first chunk ends at the edge between the records: 91, not 109.**
    /// Then one chunk per break, the same pieces `break_pieces` cuts.
    #[test]
    fn the_first_chunk_ends_at_the_user_records_edge() {
        let t = npcd_turn();
        assert_eq!(chunks(&t, 512), vec![91, 4, 2, 12]);
    }

    /// A pass narrower than the user record cuts on the pass, and the edge still
    /// lands on a chunk boundary.
    #[test]
    fn a_narrow_pass_still_lands_a_chunk_on_the_edge() {
        let t = npcd_turn();
        assert_eq!(chunks(&t, 40), vec![40, 40, 11, 4, 2, 12]);
    }

    /// A break that is the chunk's last row is no cut; with no break tokens — a
    /// model keeping no positional index — nothing is cut at all.
    #[test]
    fn a_trailing_break_and_no_breaks_cut_nothing() {
        let t = [1, 2, IM_END, 3];
        assert_eq!(break_advance(&t, 0, 3, &BREAKS), 3);
        assert_eq!(break_advance(&t, 0, 4, &[]), 4);
        assert_eq!(break_advance(&t, 0, 4, &BREAKS), 3);
    }

    /// The cut is found from the chunk's own offset, never from an earlier break.
    #[test]
    fn the_cut_is_relative_to_the_chunk() {
        let t = [IM_END, 1, 2, IM_END, 3];
        assert_eq!(break_advance(&t, 1, 4, &BREAKS), 3);
    }

    /// A chunk ending on a break closes the page while the prompt goes on, and
    /// the prompt's last chunk leaves the close to its first generated token.
    #[test]
    fn a_chunk_closes_its_page_only_with_more_to_come() {
        assert!(chunk_closes_page(&[1, IM_END], true, &BREAKS));
        assert!(!chunk_closes_page(&[1, IM_END], false, &BREAKS));
        assert!(!chunk_closes_page(&[1, 2], true, &BREAKS));
        assert!(!chunk_closes_page(&[1, IM_END], true, &[]));
    }
}
