//! The boundaries a reply takes early in its reasoning
//! (`docs/hit_driven_projection.md` §3, *Boundary*).
//!
//! A turn's opening ends on no boundary: the reply decodes on the slot its
//! question was prefilled against, and the question's hits wait for the first
//! of these. Without them the next one the rule fires is usually the cadence's,
//! 64 tokens in — by then the model has written down what it believes it has
//! read, and that sentence stays in its context whatever the slot holds
//! afterwards. The early boundaries move the slot while the reasoning is still
//! deciding what it knows: at [`EARLY_BOUNDARIES`] decoded tokens, before
//! `</think>`, outside a tool call.
//!
//! The first is at 9 so that its re-feed (`refeed::REFEED`, 8 tokens) is the
//! reasoning's first 8 tokens, the most it can be: the reply's first token
//! opens an index page, and the 9th is the token still pending when the
//! boundary fires.
//!
//! They are the decode loop's, not the boundary rule's: the gallery cuts every
//! sealed turn by the rule (`provenance::boundaries`), and these cut nothing.

/// The decoded counts at which a reply still reasoning takes a boundary.
pub(super) const EARLY_BOUNDARIES: [usize; 3] = [9, 16, 32];

/// Whether a reply takes an early boundary after its `count`-th decoded token
/// (counted from one): `reasoning` while its `</think>` has not been decoded,
/// and never inside a tool call's body.
pub(super) fn early_boundary(count: usize, reasoning: bool, in_tool_call: bool) -> bool {
    reasoning && !in_tool_call && EARLY_BOUNDARIES.contains(&count)
}

#[cfg(test)]
mod tests {
    use super::super::refeed::REFEED;
    use super::*;

    /// **A reasoning reply takes a boundary at 9, 16 and 32 tokens, and at no
    /// other count.**
    #[test]
    fn a_reasoning_reply_takes_its_early_boundaries() {
        let fired: Vec<usize> = (1..=128)
            .filter(|&n| early_boundary(n, true, false))
            .collect();
        assert_eq!(fired, vec![9, 16, 32]);
    }

    /// **The first early boundary re-feeds the whole of the reasoning so far**:
    /// at count 9 the turn has forwarded its first 8 decoded tokens, a full
    /// re-feed, with the 9th pending.
    #[test]
    fn the_first_boundary_refeeds_the_reasoning_so_far() {
        assert_eq!(EARLY_BOUNDARIES[0] - 1, REFEED);
    }

    /// **Past its `</think>`, or inside a tool call, a reply takes none**: the
    /// answer is being written, or the call's body must not move the tool.
    #[test]
    fn an_answer_or_a_tool_call_takes_none() {
        for n in EARLY_BOUNDARIES {
            assert!(!early_boundary(n, false, false), "answering at {n}");
            assert!(!early_boundary(n, true, true), "inside a call at {n}");
        }
    }
}
