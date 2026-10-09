//! Turns submitted as one group: every case prefilled as if it were alone,
//! all of them in the same forwards, each sealed as a turn of its own.
//!
//! # Why a group
//!
//! Calibration builds a provenance exemplar per case by prefilling it and
//! sealing the result, which captures the turn's per-token `sign(Q)` window; a
//! dream writes one turn per line. One case per submission puts the model deep
//! in the launch-overhead regime — measured 353 tokens across 2.4 sequences in
//! a 1,095 ms forward, 263 t/s against a >1000 t/s target — and the forward
//! costs about the same whether it carries 350 tokens or 3,500. So the cases go
//! in together.
//!
//! # Every case is masked to itself
//!
//! Each case is prefilled on its **own view** of the conversation's slot — the
//! projected prefix borrowed zero-copy, the per-sequence recurrent state forked
//! from it — and the views ride the scheduler's batched multi-sequence prefill
//! together (`Scheduler::start_turn_group`). So a case attends to the prefix and
//! to itself, never to a sibling, and a hybrid's recurrent layers start every
//! case from the prefix's state, not from the end of the case before it. Each
//! sealed turn — its K/V, its `sign(Q)` window, its recurrent snapshot — is what
//! a lone prefill of that case would have produced.
//!
//! The cases were once laid end to end in one prefill grid and carved apart
//! after, which ran causal attention and the recurrent state straight through
//! every case while documenting the opposite: a case's K/V and window were
//! computed over all the cases before it. No padding is needed now either — a
//! view writes from a fresh block, so no two cases can share one.

use crate::turn_layout::TurnLayout;

/// One case of a group: the tokens a lone prefill of it would lay down, and its
/// content bounds within them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseGrid {
    /// The case's tokens, in order.
    pub tokens: Vec<u32>,
    /// End of the user content within `tokens`, exclusive.
    pub user_content_end: u32,
    /// Start of the assistant content within `tokens`.
    pub assistant_content_start: u32,
}

/// A case as it will be sealed: its length and its content bounds, clamped
/// into the case and ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedCase {
    /// The case's tokens — exactly what its turn pins beside its K/V.
    pub token_len: usize,
    /// End of the user content.
    pub user_content_end: u32,
    /// Start of the assistant content.
    pub assistant_content_start: u32,
}

impl PlannedCase {
    /// Plan one case. `None` for a case with no tokens — a turn with no K/V and
    /// an empty gallery window is nothing worth sealing.
    ///
    /// The content bounds are clamped into the case and kept ordered: a
    /// tokenizer that merges across the user/assistant join can report an
    /// assistant start below the user end, which would invert the span and make
    /// the phase lens read a backwards range.
    pub fn of(case: &CaseGrid) -> Option<PlannedCase> {
        let token_len = case.tokens.len();
        if token_len == 0 {
            return None;
        }
        let user_content_end = (case.user_content_end as usize).min(token_len) as u32;
        let assistant_content_start = (case.assistant_content_start as usize)
            .min(token_len)
            .max(user_content_end as usize) as u32;
        Some(PlannedCase {
            token_len,
            user_content_end,
            assistant_content_start,
        })
    }

    /// The layout of this case's turn, tiling its tokens exactly.
    ///
    /// `head_len` is the baked opener's standalone length, clamped here against
    /// the case's user end: a tokenizer that merges across the head↔question
    /// join makes the standalone count an approximation of where the question
    /// starts, and an unclamped one could exceed the end and invert the span.
    ///
    /// `answer` is the assistant half's text, for the turn's stored transcript.
    /// A question exemplar passes `String::new()` — its assistant turn is empty
    /// by design, the routing happens on the question — while a prefilled
    /// assistant turn (a dream line) passes its body. Either way the token
    /// spans come from the numeric bounds, so the string is transcript only.
    pub fn layout(
        &self,
        head_len: u32,
        im_end_len: u32,
        assistant_start_len: u32,
        trailing_marker_len: u32,
        question: String,
        answer: String,
    ) -> TurnLayout {
        TurnLayout::from_flat_grid_with_tail(
            head_len.min(self.user_content_end),
            self.user_content_end,
            self.assistant_content_start,
            self.token_len as u32,
            im_end_len,
            assistant_start_len,
            trailing_marker_len,
            question,
            Some(answer),
            false,
        )
    }
}

/// Plan every case of a group, with the index of the case each came from.
///
/// Empty cases are dropped, so the plans do not line up with `cases` by
/// position — the indices do. Losing that correspondence would tag an exemplar
/// with the wrong tool, the worst failure this path can produce, since it
/// corrupts the corpus in a way that looks like a routing bug forever after.
pub fn plan_cases(cases: &[CaseGrid]) -> Vec<(usize, PlannedCase)> {
    cases
        .iter()
        .enumerate()
        .filter_map(|(i, case)| Some((i, PlannedCase::of(case)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(len: usize, user: u32, assistant: u32) -> CaseGrid {
        CaseGrid {
            tokens: (0..len as u32).collect(),
            user_content_end: user,
            assistant_content_start: assistant,
        }
    }

    /// An empty case claims no turn, and the others keep the index of the case
    /// they came from.
    #[test]
    fn empty_cases_are_dropped_and_the_rest_keep_their_indices() {
        let cases = vec![case(10, 5, 5), case(0, 0, 0), case(20, 5, 5)];
        let planned = plan_cases(&cases);
        assert_eq!(
            planned.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(planned[0].1.token_len, 10);
        assert_eq!(planned[1].1.token_len, 20);
        assert!(plan_cases(&[]).is_empty());
    }

    /// Bounds past the end of a case are clamped, and an assistant start below
    /// the user end is lifted to meet it — an inverted span would make the
    /// phase lens read a backwards range.
    #[test]
    fn out_of_range_and_inverted_content_bounds_are_repaired() {
        let over = PlannedCase::of(&case(10, 99, 99)).unwrap();
        assert_eq!(over.user_content_end, 10, "clamped to the case");
        assert_eq!(over.assistant_content_start, 10);
        let inverted = PlannedCase::of(&case(10, 8, 3)).unwrap();
        assert_eq!(inverted.user_content_end, 8);
        assert_eq!(
            inverted.assistant_content_start, 8,
            "an assistant start below the user end is lifted, never inverted",
        );
    }

    /// **A case's turn pins exactly its own tokens**, and its layout tiles them
    /// — no padding, at any length, because nothing has to align to a block.
    #[test]
    fn a_cases_layout_tiles_exactly_its_own_tokens() {
        for len in [1usize, 25, 31, 32, 33, 45, 77] {
            let user = (len as u32 / 2).max(1);
            let c = PlannedCase::of(&case(len, user, user)).unwrap();
            let layout = c.layout(1, 1, 1, 1, "q".to_string(), String::new());
            assert_eq!(
                layout.validate_tiling(len as u32),
                Ok(()),
                "a {len}-token case's layout must tile exactly its tokens",
            );
        }
    }

    /// The user span is the question's own tokens, past the baked opener, and
    /// no phase span runs past the case's tokens.
    #[test]
    fn phase_spans_land_on_the_cases_own_tokens() {
        use crate::normalization::Phase;
        use crate::turn_layout::phase_span_of;

        let head_len = 3u32;
        let c = PlannedCase::of(&case(45, 30, 34)).unwrap();
        let layout = c.layout(head_len, 2, 2, 2, "q".to_string(), String::new());
        let user = phase_span_of(&layout.segments, Phase::User)
            .expect("a question exemplar has a user span");
        assert_eq!(user, head_len as usize..30);
        for phase in [Phase::User, Phase::Thinking, Phase::Response] {
            if let Some(span) = phase_span_of(&layout.segments, phase) {
                assert!(span.end <= 45, "{phase:?} span {span:?} runs off the case");
            }
        }
    }

    /// **A prefilled assistant turn spans its own body.** A dream line's
    /// assistant half is real content: the turn gets a Response span over
    /// exactly those tokens, and its transcript carries the text.
    #[test]
    fn a_prefilled_assistant_turn_spans_its_own_body() {
        use crate::normalization::Phase;
        use crate::turn_layout::phase_span_of;

        // head(3) · user(10) · im_end(2) · assistant_start(2) · body(26) ·
        // assistant_end(2) = 45 tokens.
        let assistant_start = 17u32;
        let body_len = 26u32;
        let c = PlannedCase::of(&case(45, 13, assistant_start)).unwrap();
        let layout = c.layout(3, 2, 2, 2, "q".to_string(), "a dream".to_string());
        let response = phase_span_of(&layout.segments, Phase::Response)
            .expect("a prefilled assistant turn has a response span");
        assert_eq!(
            response,
            assistant_start as usize..(assistant_start + body_len) as usize
        );
        let assistant_text = layout.segments.iter().find_map(|s| match s {
            crate::turn_layout::TurnSegment::Assistant { text, .. } => text.clone(),
            _ => None,
        });
        assert_eq!(assistant_text.as_deref(), Some("a dream"));
    }
}
