//! Tests of the reasoning split: its spans, its text, and the cases where it is
//! declined.
//!
//! A reasoning split keeps the grid tiled and gives the reasoning only to the
//! thinking segment, so a consumer never renders it twice. The span is measured
//! from the decode's own `</think>` boundary against the grid, not against text
//! re-tokenised from the prose. A reasoning span that does not fit the assistant
//! region declines to split, because a clamped end would never be a page boundary.
//! A phase split across a tool response spans from its first segment to its last.

use super::*;

#[cfg(test)]
pub(super) mod thinking_length_tests {
    use super::*;

    fn thinking(text: &str, kv: Option<KvSpan>) -> TurnSegment {
        TurnSegment::Thinking {
            text: text.into(),
            kv,
        }
    }

    fn answer(offset: u32, len: u32) -> TurnSegment {
        TurnSegment::Assistant {
            text: None,
            kv: KvSpan::new(offset, len),
        }
    }

    /// **`has_assistant_text` answers what `assistant_text` would**, case by
    /// case: no assistant segment is no text whatever the reasoning holds, and
    /// either reasoning or an answer makes it some.
    #[test]
    fn has_assistant_text_agrees_with_the_built_text() {
        let said = |text: Option<&str>| TurnSegment::Assistant {
            text: text.map(Into::into),
            kv: KvSpan::new(0, 1),
        };
        let cases = [
            vec![],
            vec![thinking("<think>r</think>", None)],
            vec![answer(0, 2)],
            vec![said(Some(""))],
            vec![said(Some("ok"))],
            vec![thinking("", None), answer(0, 2)],
            vec![thinking("<think>r</think>", None), answer(0, 2)],
        ];
        let expected = [false, false, false, false, true, false, true];
        for (segments, want) in cases.into_iter().zip(expected) {
            let layout = TurnLayout { segments };
            let built = !layout.assistant_text().unwrap_or_default().is_empty();
            assert_eq!(built, want, "{:?}", layout.segments);
            assert_eq!(layout.has_assistant_text(), want, "{:?}", layout.segments);
        }
    }

    /// An estimator that must not be consulted: a real span is already exact.
    fn never(_: &str) -> u32 {
        panic!("a real reasoning span was re-estimated from its prose")
    }

    #[test]
    fn a_real_reasoning_span_is_its_exact_length() {
        let layout = TurnLayout {
            segments: vec![
                thinking("<think>why</think>", Some(KvSpan::new(4, 212))),
                answer(216, 9),
            ],
        };
        assert_eq!(
            layout.thinking_length(never),
            Some(ThinkingLength {
                tokens: 212,
                exact: true
            })
        );
    }

    #[test]
    fn a_dropped_reasoning_span_is_estimated_from_its_prose() {
        let layout = TurnLayout {
            segments: vec![thinking("<think>one two three</think>", None), answer(4, 9)],
        };
        let words = |t: &str| t.split_whitespace().count() as u32;
        assert_eq!(
            layout.thinking_length(words),
            Some(ThinkingLength {
                tokens: 3,
                exact: false
            })
        );
    }

    #[test]
    fn real_and_dropped_segments_sum_and_the_total_is_an_estimate() {
        let layout = TurnLayout {
            segments: vec![
                thinking("a", Some(KvSpan::new(0, 40))),
                thinking("b c", None),
                answer(40, 5),
            ],
        };
        assert_eq!(
            layout.thinking_length(|_| 7),
            Some(ThinkingLength {
                tokens: 47,
                exact: false
            })
        );
    }

    #[test]
    fn a_turn_that_did_not_reason_has_no_length() {
        let layout = TurnLayout {
            segments: vec![answer(0, 12)],
        };
        assert_eq!(layout.thinking_length(never), None);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Splitting a real thinking block out of the assistant body keeps the grid
    /// tiled (think + answer == the old assistant span).
    #[test]
    fn thinking_split_preserves_tiling() {
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            20, // assistant region [8,20) = 12 tokens
            2,
            3,
            "u".into(),
            Some("<think>r</think>a".into()),
            false,
        )
        .with_thinking_split("<think>r</think>".into(), 5, false);
        assert_eq!(layout.validate_tiling(20), Ok(()));
        // think [8,13), answer [13,20)
        let think = layout
            .segments
            .iter()
            .find_map(|s| match s {
                TurnSegment::Thinking { kv, .. } => *kv,
                _ => None,
            })
            .unwrap();
        assert_eq!(think, KvSpan::new(8, 5));
        // The reasoning prose lives ONLY on the Thinking segment; the Assistant
        // segment's text is the answer with the `<think>…</think>` block stripped
        // (so a consumer never renders the reasoning twice).
        let (think_text, asst_text) =
            layout
                .segments
                .iter()
                .fold((None, None), |(tt, at), s| match s {
                    TurnSegment::Thinking { text, .. } => (Some(text.clone()), at),
                    TurnSegment::Assistant { text, .. } => (tt, text.clone()),
                    _ => (tt, at),
                });
        assert_eq!(think_text.as_deref(), Some("<think>r</think>"));
        assert_eq!(asst_text.as_deref(), Some("a"));
        // The Assistant SEGMENT is answer-only (per-segment rendering), but the
        // `assistant_text()` ACCESSOR rejoins reasoning + answer into the full
        // verbatim reply the chat-history view needs.
        assert_eq!(
            layout.assistant_text().as_deref(),
            Some("<think>r</think>a")
        );
    }

    /// **A reasoning span too long for the assistant region seals no split.**
    ///
    /// Clamping it to fit produces an end that is provably not an index page
    /// boundary — the cuts were made against the real length — and every later
    /// projection selecting the turn then refuses to window it, permanently.
    /// Whole (reasoning visible) is recoverable; a poisoned conversation is not.
    ///
    /// This applies only where a span is actually recorded. The ethereal path
    /// records none, so an over-long `think_len` is expected there and is
    /// covered by `clean_grid_seals_thinking_text_without_its_kv`.
    #[test]
    fn an_over_long_reasoning_span_declines_to_split() {
        // [user(3)][im_end(2)][a_start(3)][assistant(2)] = 10.
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            10,
            2,
            3,
            "u".into(),
            Some("<think>rrrr</think>a".into()),
            false,
        )
        // 5 reasoning tokens will not fit the 2-token assistant region.
        .with_thinking_split("<think>rrrr</think>".into(), 5, false);

        assert!(
            !layout
                .segments
                .iter()
                .any(|s| matches!(s, TurnSegment::Thinking { .. })),
            "no Thinking segment may be sealed from a span that would need clamping"
        );
        assert_eq!(layout.validate_tiling(10), Ok(()), "the turn still tiles");
    }

    /// The clean-reprefill seal contract: the sealed grid physically OMITS the
    /// `<think>…</think>` tokens (they were re-prefilled away), so `realize()`
    /// reproduces a think-free grid and the K/V carries no reasoning — while the
    /// ethereal `Thinking` segment KEEPS the reasoning text, so `assistant_text()`
    /// still yields the full verbatim reply for display/history. This is what
    /// makes a projected past turn stop attending its own reasoning.
    #[test]
    fn clean_grid_seals_thinking_text_without_its_kv() {
        // CLEAN grid — no `<think>` tokens at all:
        // [user(3)][im_end(2)][a_start(3)][answer(2)] = 10 tokens.
        let clean_grid: Vec<u32> = vec![
            10, 11, 12, /*im_end*/ 90, 91, /*a_start*/ 80, 81, 82, /*answer*/ 1, 2,
        ];
        let layout = TurnLayout::from_flat_grid(
            0,  // user_content_start
            3,  // user_content_end
            8,  // assistant_start (3 + 2 + 3)
            10, // total — the CLEAN length (reasoning tokens absent)
            2,  // im_end_len
            3,  // assistant_start_len
            "u".into(),
            // The decoded text still carries the reasoning; the split moves it
            // onto an ethereal Thinking segment.
            Some("<think>r</think>a".into()),
            false,
        )
        .with_thinking_split("<think>r</think>".into(), 5, true);

        // Tiles the CLEAN grid — the reasoning contributes ZERO K/V positions.
        assert_eq!(layout.validate_tiling(10), Ok(()));
        assert_eq!(layout.kv_len(), 10);

        // The reasoning is recorded but ethereal (no span).
        assert!(layout.segments.iter().any(
            |s| matches!(s, TurnSegment::Thinking { kv: None, text } if text == "<think>r</think>")
        ));

        // `realize()` reproduces the clean grid EXACTLY — no reasoning tokens
        // appear in any real (K/V-bearing) segment.
        let rebuilt: Vec<u32> = layout
            .realize(&clean_grid)
            .iter()
            .flat_map(|(_, t)| t.iter().copied())
            .collect();
        assert_eq!(rebuilt, clean_grid);

        // …yet the reasoning text survives for display: `assistant_text()`
        // rejoins Thinking + Assistant into the full verbatim reply.
        assert_eq!(
            layout.assistant_text().as_deref(),
            Some("<think>r</think>a")
        );
        // The answer span itself is reasoning-free (answer-only tokens).
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 2));
    }

    /// **The reasoning span is the assistant body through `</think>`, measured
    /// from the decode's own boundary.**
    ///
    /// The seal records the generated-run index of the `</think>` it decoded
    /// (`DecodeState::think_close_at`) and passes `index + 1` as `think_len`.
    /// This pins that arithmetic against a real grid: the `Thinking` span must
    /// slice back exactly the reasoning tokens — including any preamble the
    /// model emitted before `<think>`, which is reasoning too — and the answer
    /// span must start on the first token after the marker.
    ///
    /// Asserted against the GRID, never against re-tokenised text. The whole
    /// point of recording the boundary is that a decode → text → encode round
    /// trip need not reproduce the token count it started from, so a test that
    /// re-derived the length from the prose would agree with the bug.
    #[test]
    fn thinking_span_covers_the_body_through_the_close_marker() {
        // Assistant body, as generated-run indices:
        //   0: "Hmm"  (preamble before the block — reasoning)
        //   1: <think>
        //   2: "why"
        //   3: </think>   ← think_close_at = 3
        //   4,5: the answer
        const THINK_OPEN: u32 = 7001;
        const THINK_CLOSE: u32 = 7002;
        // [user(2)][im_end(1)][a_start(1)][body(6)][im_end(1)] = 11
        let grid: Vec<u32> = vec![
            10,
            11,
            /*im_end*/ 90,
            /*a_start*/ 80,
            /*body*/ 500,
            THINK_OPEN,
            501,
            THINK_CLOSE,
            600,
            601,
            /*im_end*/ 200,
        ];
        let asst_start = 4u32;
        let think_close_at = 3u32; // index into the generated run

        let layout = TurnLayout::from_flat_grid_with_tail(
            0,
            2,
            asst_start,
            grid.len() as u32,
            1,
            1,
            1,
            "u".into(),
            Some("<think>why</think>answer".into()),
            false,
        )
        .with_thinking_split(
            "<think>why</think>".into(),
            think_close_at + 1, // exactly what the seal passes
            false,
        );
        assert_eq!(layout.validate_tiling(grid.len() as u32), Ok(()));

        let think = layout
            .segments
            .iter()
            .find_map(|s| match s {
                TurnSegment::Thinking { kv, .. } => *kv,
                _ => None,
            })
            .expect("a real thinking span");
        assert_eq!(
            think,
            KvSpan::new(asst_start, 4),
            "the span must start at the assistant body and run through `</think>`"
        );
        // The decisive assertion: the span slices back the reasoning tokens and
        // stops on the marker.
        assert_eq!(
            &grid[think.range()],
            &[500, THINK_OPEN, 501, THINK_CLOSE],
            "the reasoning span must cover the preamble, the block, and the \
             closing marker — and nothing of the answer"
        );
        // …and the answer begins on the very next token.
        assert_eq!(layout.assistant_span(), KvSpan::new(8, 2));
        assert_eq!(&grid[layout.assistant_span().range()], &[600, 601]);
    }

    /// A turn that decoded no `</think>` records no boundary, so the seal passes
    /// no split and the whole assistant body stays the answer. This is the
    /// prefilled-assistant-half case (repo_map, a `code_read` scope) and the
    /// no-single-token-tokenizer case, which must behave identically.
    #[test]
    fn no_recorded_boundary_leaves_the_body_whole() {
        let layout = TurnLayout::from_flat_grid_with_tail(
            0,
            2,
            4,
            11,
            1,
            1,
            1,
            "u".into(),
            Some("answer".into()),
            false,
        );
        assert_eq!(layout.validate_tiling(11), Ok(()));
        assert!(
            !layout
                .segments
                .iter()
                .any(|s| matches!(s, TurnSegment::Thinking { .. })),
            "no boundary was recorded, so there must be no Thinking segment"
        );
        assert_eq!(layout.assistant_span(), KvSpan::new(4, 6));
    }

    /// Dropping the reasoning K/V (ethereal) leaves the answer holding the whole
    /// region — still tiled.
    #[test]
    fn ethereal_thinking_split_preserves_tiling() {
        let layout = TurnLayout::from_flat_grid(
            0,
            3,
            8,
            20,
            2,
            3,
            "u".into(),
            Some("<think>r</think>a".into()),
            false,
        )
        .with_thinking_split("<think>r</think>".into(), 5, true);
        assert_eq!(layout.validate_tiling(20), Ok(()));
        // the thinking segment is ethereal (no K/V)
        assert!(layout
            .segments
            .iter()
            .any(|s| matches!(s, TurnSegment::Thinking { kv: None, .. })));
    }

    /// **A phase span covers the phase's real tokens and nothing else.**
    ///
    /// The span indexes a signature window directly, so an off-by-one here does
    /// not fail — it silently scores the wrong tokens and shifts every phase
    /// score by a token of glue.
    #[test]
    fn phase_spans_cover_exactly_their_own_segments() {
        // [user(0..3)][im_end(3..5)][a_start(5..8)][answer(8..12)]
        let layout =
            TurnLayout::from_flat_grid(0, 3, 8, 12, 2, 3, "hi".into(), Some("ok".into()), false);
        assert_eq!(phase_span_of(&layout.segments, Phase::User), Some(0..3));
        assert_eq!(
            phase_span_of(&layout.segments, Phase::Response),
            Some(8..12)
        );
        // No think block was split out, so there is no thinking phase at all —
        // distinct from "the thinking phase scored zero".
        assert_eq!(phase_span_of(&layout.segments, Phase::Thinking), None);
    }

    /// An ETHEREAL thinking segment (text kept, K/V dropped by the clean-turn
    /// re-prefill) has no tokens in the grid, so it must report no span rather
    /// than a zero-width one at the split point.
    #[test]
    fn an_ethereal_thinking_phase_has_no_span() {
        let layout =
            TurnLayout::from_flat_grid(0, 3, 8, 20, 2, 3, "hi".into(), Some("ok".into()), false)
                .with_thinking_split("<think>r</think>".into(), 5, true);
        assert_eq!(phase_span_of(&layout.segments, Phase::Thinking), None);
        // …while the phases that DO hold K/V are unaffected by the split.
        assert_eq!(phase_span_of(&layout.segments, Phase::User), Some(0..3));
    }

    /// A tool round-trip arrives as a second user segment mid-turn (the tool
    /// response), so the user phase is discontiguous. The span covers first to
    /// last: scoring the question alone would drop the tool's own output, which
    /// is the part naming the tool.
    #[test]
    fn a_split_phase_spans_first_to_last_segment() {
        let layout = TurnLayout::new(vec![
            TurnSegment::User {
                text: "q".into(),
                kv: KvSpan::new(0, 4),
            },
            TurnSegment::Assistant {
                text: Some("call".into()),
                kv: KvSpan::new(4, 3),
            },
            TurnSegment::User {
                text: "tool result".into(),
                kv: KvSpan::new(7, 5),
            },
            TurnSegment::Assistant {
                text: Some("answer".into()),
                kv: KvSpan::new(12, 2),
            },
        ]);
        assert_eq!(phase_span_of(&layout.segments, Phase::User), Some(0..12));
        assert_eq!(
            phase_span_of(&layout.segments, Phase::Response),
            Some(4..14)
        );
    }
}
