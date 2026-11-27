//! The anchors of the placed sections: the chunks holding each section's first
//! `head` and last `tail` tokens, which a transplant keeps from the fresh
//! prefill while every other chunk takes the sealed K/V
//! ([`super::sealed_fresh::SealedFreshReq::anchor_head`]). Whether a section
//! read through sealed K/V needs its context everywhere, or only at its ends.
//!
//! Chunks are whole: an anchor is every chunk that holds any of its tokens, so
//! a `head` of 1 keeps a section's first chunk and a section shorter than
//! `head + tail` is kept whole.

/// The chunks, by index, holding the first `head` or last `tail` tokens of
/// each section. `widths` is every chunk's token count; `cuts` is each piece's
/// `(first, end)` chunks ([`super::sealed_fresh_layout::piece_chunks`]) and
/// `sections` whether it is a section — glue and turns keep no anchor.
pub fn anchor_chunks(
    widths: &[usize],
    cuts: &[(usize, usize)],
    sections: &[bool],
    head: usize,
    tail: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    for (&(first, end), _) in cuts.iter().zip(sections).filter(|(_, &s)| s) {
        let len: usize = widths[first..end].iter().sum();
        let tail_from = len.saturating_sub(tail);
        let mut at = 0;
        for (j, &w) in widths.iter().enumerate().take(end).skip(first) {
            if w > 0 && (at < head || at + w > tail_from) {
                out.push(j);
            }
            at += w;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sections of 100 and 40 tokens around a 3-token piece of glue,
    /// chunked as a projection lays them out, each ending in its own partial
    /// chunk: `[32, 32, 32, 4] [3] [32, 8]`.
    const WIDTHS: [usize; 7] = [32, 32, 32, 4, 3, 32, 8];
    const CUTS: [(usize, usize); 3] = [(0, 4), (4, 5), (5, 7)];
    const SECTIONS: [bool; 3] = [true, false, true];

    /// **A section's first and last chunks are its anchors; glue keeps
    /// none.**
    #[test]
    fn the_ends_of_each_section_are_its_anchors() {
        assert_eq!(
            anchor_chunks(&WIDTHS, &CUTS, &SECTIONS, 32, 4),
            vec![0, 3, 5, 6]
        );
    }

    /// **An anchor is every chunk holding any of its tokens**: a head of 33
    /// reaches the second chunk, a tail of 5 the last full one.
    #[test]
    fn an_anchor_takes_every_chunk_it_touches() {
        assert_eq!(
            anchor_chunks(&WIDTHS, &CUTS, &SECTIONS, 33, 5),
            vec![0, 1, 2, 3, 5, 6]
        );
        assert_eq!(anchor_chunks(&WIDTHS, &CUTS, &SECTIONS, 1, 0), vec![0, 5]);
        assert_eq!(anchor_chunks(&WIDTHS, &CUTS, &SECTIONS, 0, 1), vec![3, 6]);
    }

    /// **No anchor keeps nothing; a section shorter than both is kept
    /// whole.**
    #[test]
    fn none_keeps_nothing_and_a_short_section_is_whole() {
        assert!(anchor_chunks(&WIDTHS, &CUTS, &SECTIONS, 0, 0).is_empty());
        assert_eq!(
            anchor_chunks(&WIDTHS, &CUTS, &SECTIONS, 64, 64),
            vec![0, 1, 2, 3, 5, 6]
        );
    }
}
