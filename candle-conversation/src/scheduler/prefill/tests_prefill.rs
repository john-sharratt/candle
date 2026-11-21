//! Tests the reasoning-boundary split of a prefill span and the think-block state
//! a prefill leaves behind.
//!
//! A closer followed by text is cut into two pieces so the page closes between
//! them, every break with something after it is cut, and a trailing break cuts
//! nothing. A prefill's think block must read as closed when it was closed, a
//! bare opener must read as open, and only the most recent marker counts. A
//! quoted `<think>` in the user half of a prefill must not arm the flag, so only
//! the assistant lead is scanned, and a deep opener in that lead still arms it.

use super::*;

/// The slice boundary the helper above relies on, which main's own module does
/// not exercise: a prefill whose USER half quotes `<think>` while its assistant
/// lead closes its block.
#[cfg(test)]
pub(super) mod break_pieces_tests {
    use super::break_pieces;

    const CLOSE: u32 = 900;
    const THINK_END: u32 = 901;

    /// The turn closer the seal forwards — `<|im_end|>\n` — is two pieces: the
    /// marker, then what follows it, so the page closes between them.
    #[test]
    fn a_closer_followed_by_text_is_two_pieces() {
        assert_eq!(
            break_pieces(&[CLOSE, 7], &[CLOSE]),
            vec![&[CLOSE][..], &[7][..]]
        );
    }

    /// Every boundary is cut, not just the first.
    #[test]
    fn every_break_with_something_after_it_is_cut() {
        let t = [1, THINK_END, 2, 3, CLOSE, 4];
        assert_eq!(
            break_pieces(&t, &[THINK_END, CLOSE]),
            vec![&[1, THINK_END][..], &[2, 3, CLOSE][..], &[4][..]]
        );
    }

    /// A break at the very end cuts nothing — the next forward is the edge.
    #[test]
    fn a_trailing_break_is_not_a_cut() {
        assert_eq!(
            break_pieces(&[1, 2, CLOSE], &[CLOSE]),
            vec![&[1, 2, CLOSE][..]]
        );
    }

    /// With no break tokens — a model keeping no positional index — the closer is
    /// not a cut and the tail forwards whole.
    #[test]
    fn with_no_break_tokens_a_closer_is_not_a_cut() {
        assert_eq!(break_pieces(&[CLOSE, 7], &[]), vec![&[CLOSE, 7][..]]);
    }

    /// No break is one piece, and nothing is one empty piece.
    #[test]
    fn an_unbroken_span_is_one_piece() {
        assert_eq!(break_pieces(&[1, 2, 3], &[CLOSE]), vec![&[1, 2, 3][..]]);
        let empty: &[u32] = &[];
        assert_eq!(break_pieces(empty, &[CLOSE]), vec![empty]);
    }
}

#[cfg(test)]
pub(super) mod think_prefill_slice_tests {
    use super::prefill_leaves_think_open;

    const OPEN: u32 = 248068;
    const CLOSE: u32 = 248069;

    /// **User content that merely QUOTES `<think>` must not arm the flag.**
    #[test]
    fn only_the_assistant_lead_is_scanned() {
        // `[user … <think> … ] [assistant lead: closed block]`
        let grid = [9u32, OPEN, 9, 9, OPEN, 3, CLOSE, 4];
        let assistant_content_start = 4;
        assert!(
            !prefill_leaves_think_open(grid[assistant_content_start..].iter(), OPEN, Some(CLOSE)),
            "the assistant lead closed its block; quoted user text must not override that"
        );
        // What the slice prevents: handed the user half, the very same scan does
        // arm — so the boundary is doing the work, not the scan.
        assert!(prefill_leaves_think_open(
            grid[..2].iter(),
            OPEN,
            Some(CLOSE)
        ));
    }

    /// A deep opener in the assistant lead still arms. The fixed 5-token tail
    /// window this replaced would have missed it entirely.
    #[test]
    fn a_deep_opener_in_the_lead_still_arms() {
        let mut deep = vec![OPEN];
        deep.extend(std::iter::repeat_n(7u32, 40));
        assert!(prefill_leaves_think_open(deep.iter(), OPEN, Some(CLOSE)));
    }
}

#[cfg(test)]
pub(super) mod prefill_think_tests {
    use super::prefill_leaves_think_open;

    const OPEN: u32 = 89;
    const CLOSE: u32 = 90;
    const NL: u32 = 10;
    const TOOL_CALL: u32 = 91;
    const WORD: u32 = 5;

    fn open(tokens: &[u32]) -> bool {
        prefill_leaves_think_open(tokens.iter(), OPEN, Some(CLOSE))
    }

    /// **The suppression prefill is a closed block, and it must read as one.**
    ///
    /// The regression this exists for: Qwen3.5's `<think>\n\n</think>\n\n`
    /// contains the open marker, and the check that asked "does it contain one"
    /// put every think-suppressed turn's sampler inside a block it would never
    /// see close.
    #[test]
    fn a_prefilled_closed_block_leaves_nothing_open() {
        assert!(!open(&[WORD, OPEN, NL, CLOSE, NL]));
    }

    /// The acting turn's prefill: the closed block, then straight into the call.
    #[test]
    fn a_closed_block_followed_by_a_call_opener_leaves_nothing_open() {
        assert!(!open(&[WORD, OPEN, NL, CLOSE, NL, TOOL_CALL]));
    }

    /// A turn that is meant to reason prefills a bare `<think>`, and it must.
    #[test]
    fn a_bare_open_marker_leaves_the_block_open() {
        assert!(open(&[WORD, NL, OPEN]));
    }

    #[test]
    fn a_prefill_with_no_markers_leaves_nothing_open() {
        assert!(!open(&[WORD, NL, WORD]));
        assert!(!open(&[]));
    }

    /// The most recent marker decides, so an earlier turn's closed block
    /// further back in the buffer cannot mask a fresh open, and cannot fake one.
    #[test]
    fn only_the_most_recent_marker_counts() {
        assert!(open(&[OPEN, WORD, CLOSE, WORD, NL, OPEN]));
        assert!(!open(&[OPEN, WORD, CLOSE, WORD, OPEN, NL, CLOSE, NL]));
    }

    /// A vocabulary with no close token cannot close a block, so an open marker
    /// anywhere behind the boundary leaves it open.
    #[test]
    fn without_a_close_token_an_open_marker_stays_open() {
        assert!(prefill_leaves_think_open(
            [WORD, OPEN, NL, CLOSE].iter(),
            OPEN,
            None
        ));
    }
}
