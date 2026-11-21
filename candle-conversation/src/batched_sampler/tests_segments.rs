//! Tests for the think-segment close override and the segment state transitions.
//!
//! Tests the closing script: a hard cap plays the script to the close token, and a graceful close waits for a line end.
//! Tests the rule for a call opened inside a segment, and that a segment boundary or a turn end cancels a stranded script.
//! Tests `update_segment_state`, where an open token enters a segment and a close token exits it.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
pub(super) mod tests {
    // Several tests pin a tuning constant inside a range. The assertion IS the
    // contract — it fires the moment someone retunes the constant out of band —
    // even though it is constant-valued at any given commit.
    #![allow(clippy::assertions_on_constants)]

    use std::sync::Arc;

    use super::*;

    use crate::config::SamplingConfig;

    use super::super::tests_helpers::tests::*;

    /// **A call being written is not cut by the answer's length budget.** Past
    /// both EOS thresholds a prose turn is ended; the same length inside a tool
    /// call keeps the model's own token, and the EOS ramp sees a length of 0.
    /// The call's own grammar and the turn's `max_tokens` bound it instead.
    #[test]
    pub(in super::super) fn writing_a_call_stands_the_length_budget_down() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.graceful_eos_after = 10;
        config.forced_eos_after = 20;
        let mut state = make_state();
        for _ in 0..25 {
            state.record_token(42, MAX_RECENT);
        }
        assert_eq!(
            sampler.resolve(7, &mut state, &config).0,
            EOS_TOKEN,
            "a prose turn past its budget is ended"
        );

        state.writing_call = true;
        assert_eq!(
            sampler.resolve(7, &mut state, &config).0,
            7,
            "a call past the same budget keeps its token"
        );
        let window = window_16();
        let rows = gather(&sampler, &[&state], &window, false);
        assert_eq!(rows.current_lens, vec![0], "the EOS ramp sees no length");

        // The degenerate-decode guard is a fault check, not a budget: it still
        // fires inside a call.
        for _ in 0..DEGENERATE_TOKEN_RUN {
            state.record_token(0, MAX_RECENT);
        }
        assert_eq!(sampler.resolve(7, &mut state, &config).0, EOS_TOKEN);
    }

    // ── Hard-cap closer script (segment_close_override tiers) ──────────

    /// Config with segment tracking on: close=90, graceful after 4 at a line
    /// end (token 7), a sentence end that is not a line end (token 8, `.`),
    /// hard cap at 8, closer phrase "A B C" (the sampler appends the close
    /// token 90 itself).
    pub(in super::super) fn closer_config() -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.segment_close_token_id = 90;
        c.segment_open_token_id = 89;
        c.graceful_segment_close_after = 4;
        c.force_segment_close_after = 8;
        c.sentence_end_token_ids = vec![8];
        c.line_end_token_ids = Arc::from([7]);
        c.segment_close_script = vec![100, 101, 102];
        c
    }

    /// A state `n` tokens into an open segment, last token `last`.
    pub(in super::super) fn in_segment_state(n: i32, last: u32) -> SequenceSamplingState {
        let mut s = make_state();
        s.record_token(last, MAX_RECENT);
        s.in_segment = true;
        s.segment_len = n;
        s
    }

    #[test]
    pub(in super::super) fn hard_cap_plays_the_closer_script_to_the_close_token() {
        let config = closer_config();
        let mut state = in_segment_state(8, 42); // past force, mid-sentence
        let mut played = Vec::new();
        for _ in 0..4 {
            played.push(segment_close_override(&config, &mut state).expect("override"));
        }
        assert_eq!(
            played,
            vec![100, 101, 102, 90],
            "phrase then the sampler-appended close, in order"
        );
        assert_eq!(
            state.close_script_pos, None,
            "script state cleared at the end"
        );
    }

    #[test]
    pub(in super::super) fn graceful_close_at_line_end_skips_the_script() {
        let config = closer_config();
        // Past graceful (not force), last token IS a line end.
        let mut state = in_segment_state(5, 7);
        assert_eq!(
            segment_close_override(&config, &mut state),
            Some(90),
            "soft cut closes bare — a completed line needs no rescue"
        );
        assert_eq!(state.close_script_pos, None);
    }

    /// **A `.` is not where a thought ends.** Past the graceful cap, a period
    /// that is not a line end — the one in `169.254` — leaves the block open;
    /// the close waits for the line to end.
    #[test]
    pub(in super::super) fn graceful_close_does_not_fire_at_a_period_mid_line() {
        let config = closer_config();
        let mut state = in_segment_state(5, 8);
        assert_eq!(segment_close_override(&config, &mut state), None);
    }

    /// The graceful EOS still accepts a sentence end: an answer that is one
    /// paragraph must not wait for a newline it will never write.
    #[test]
    pub(in super::super) fn the_answer_ends_at_a_sentence_or_a_line() {
        let config = closer_config();
        for last in [7, 8] {
            let mut state = make_state();
            state.record_token(last, MAX_RECENT);
            assert!(state.at_sentence_end(&config), "token {last}");
        }
        let mut mid = make_state();
        mid.record_token(42, MAX_RECENT);
        assert!(!mid.at_sentence_end(&config));
    }

    #[test]
    pub(in super::super) fn continuation_span_gets_the_bare_close_not_the_script() {
        let config = closer_config();
        let mut state = in_segment_state(8, 42);
        // A span that retires into more content: the steering drops the close
        // and decoding continues, so no terminal closing statement.
        state.close_would_continue = true;
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
        assert_eq!(state.close_script_pos, None, "no script started");
    }

    #[test]
    pub(in super::super) fn hard_cap_at_a_completed_sentence_closes_bare() {
        let config = closer_config();
        // Past force, but the last token IS a sentence end — the amputation
        // rescue is for dangling fragments only.
        let mut state = in_segment_state(8, 7);
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
        assert_eq!(state.close_script_pos, None, "no script started");
    }

    #[test]
    pub(in super::super) fn segment_close_override_is_inert_outside_a_segment_or_unconfigured() {
        let config = closer_config();
        let mut state = in_segment_state(8, 42);
        state.in_segment = false;
        assert_eq!(segment_close_override(&config, &mut state), None);

        let mut unconfigured = closer_config();
        unconfigured.segment_close_token_id = -1;
        let mut state = in_segment_state(8, 42);
        assert_eq!(segment_close_override(&unconfigured, &mut state), None);
    }

    #[test]
    pub(in super::super) fn below_both_caps_no_override() {
        let config = closer_config();
        let mut state = in_segment_state(3, 7);
        assert_eq!(segment_close_override(&config, &mut state), None);
    }

    /// A call opened inside a think block closes the block first — the opener
    /// is committed as the close — and nowhere else does the opener change.
    #[test]
    pub(in super::super) fn a_call_opened_inside_a_segment_closes_it_first() {
        let mut config = closer_config();
        config.tool_call_open_token_id = 70;
        let state = in_segment_state(3, 42);
        assert_eq!(close_before_call(&config, &state, 70), Some(90));
        assert_eq!(
            close_before_call(&config, &state, 71),
            None,
            "any other token stands"
        );
        let mut outside = in_segment_state(3, 42);
        outside.in_segment = false;
        assert_eq!(
            close_before_call(&config, &outside, 70),
            None,
            "a call outside the block stands"
        );
        config.tool_call_open_token_id = -1;
        assert_eq!(
            close_before_call(&config, &state, 70),
            None,
            "an unresolved opener changes nothing"
        );
    }

    /// Through the one authority both sampling paths resolve with: under the
    /// segment budget, a sampled opener inside the block commits as the close.
    #[test]
    pub(in super::super) fn the_committed_token_for_a_call_inside_a_segment_is_the_close() {
        let sampler = make_sampler();
        let mut config = closer_config();
        config.tool_call_open_token_id = 70;
        let mut state = in_segment_state(3, 42);
        assert_eq!(sampler.resolve(70, &mut state, &config).0, 90);
    }

    #[test]
    pub(in super::super) fn hard_cap_without_script_falls_back_to_bare_close() {
        let mut config = closer_config();
        config.segment_close_script = Vec::new();
        let mut state = in_segment_state(8, 42);
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
    }

    #[test]
    pub(in super::super) fn script_in_flight_overrides_graceful_and_force_conditions() {
        let config = closer_config();
        let mut state = in_segment_state(9, 7); // sentence end AND past force
        state.close_script_pos = Some(2);
        // Mid-script: the next scripted token wins over every other tier.
        assert_eq!(segment_close_override(&config, &mut state), Some(102));
        assert_eq!(segment_close_override(&config, &mut state), Some(90));
        assert_eq!(state.close_script_pos, None);
    }

    #[test]
    pub(in super::super) fn segment_close_wins_over_eos_failsafes_for_the_step() {
        let sampler = make_sampler();
        let mut config = closer_config();
        config.forced_eos_after = 5; // far exceeded — EOS wants to fire every step
        let mut state = in_segment_state(8, 42);
        for _ in 0..7 {
            state.record_token(42, MAX_RECENT);
        }
        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        // The closer script plays to completion; the EOS failsafe never
        // clobbers a scripted step or the close itself.
        let mut played = Vec::new();
        for _ in 0..4 {
            played.push(
                sampler
                    .sample_batch(&logits, &mut [&mut state], &[&config])
                    .expect("sample")[0],
            );
        }
        assert_eq!(played, vec![100, 101, 102, 90]);
        assert!(
            !state.in_segment,
            "the sampled close token exits the segment on the CPU path"
        );

        // With the segment closed, the deferred EOS failsafe fires next step.
        let next = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample")[0];
        assert_eq!(next, EOS_TOKEN);
    }

    #[test]
    pub(in super::super) fn segment_boundaries_and_turn_end_cancel_a_stranded_script() {
        let mut state = make_state();
        state.enter_segment();
        state.close_script_pos = Some(1);
        state.exit_segment();
        assert_eq!(state.close_script_pos, None, "exit cancels the script");

        state.enter_segment();
        state.close_script_pos = Some(2);
        state.enter_segment();
        assert_eq!(
            state.close_script_pos, None,
            "a fresh segment cannot inherit a script"
        );

        state.close_script_pos = Some(1);
        state.close_would_continue = true;
        state.end_turn(0);
        assert!(!state.in_segment, "turn end closes a dangling segment");
        assert_eq!(state.segment_len, 0);
        assert_eq!(state.close_script_pos, None);
        assert!(!state.close_would_continue);
    }

    // ── update_segment_state tests ────────────────────────────────────

    #[test]
    pub(in super::super) fn test_open_token_enters_segment() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        assert!(!state.in_segment);
        assert_eq!(state.segment_len, 0);

        state.update_segment_state(seg_open as u32, seg_open, seg_close);
        assert!(
            state.in_segment,
            "should enter the segment on the open token"
        );
        assert_eq!(state.segment_len, 0, "segment_len reset on enter");
    }

    #[test]
    pub(in super::super) fn test_close_token_exits_segment() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        // Open the segment
        state.update_segment_state(seg_open as u32, seg_open, seg_close);
        assert!(state.in_segment);

        // Generate some tokens inside the segment
        state.record_token(42, MAX_RECENT);
        state.record_token(43, MAX_RECENT);
        assert_eq!(state.segment_len, 2);

        // Close the segment
        state.update_segment_state(seg_close as u32, seg_open, seg_close);
        assert!(
            !state.in_segment,
            "should exit the segment on the close token"
        );
        assert_eq!(state.segment_len, 0, "segment_len reset on exit");
    }

    #[test]
    pub(in super::super) fn test_close_token_without_open_segment_is_noop() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        // The close token outside a segment should be a no-op
        state.update_segment_state(seg_close as u32, seg_open, seg_close);
        assert!(!state.in_segment, "should remain outside a segment");
    }

    #[test]
    pub(in super::super) fn test_segment_tracking_disabled_when_ids_negative() {
        let mut state = make_state();

        // -1 means not configured
        state.update_segment_state(10, -1, 11);
        assert!(
            !state.in_segment,
            "should not enter a segment when segment_open_id < 0"
        );

        state.update_segment_state(10, 10, -1);
        assert!(
            !state.in_segment,
            "should not enter a segment when segment_close_id < 0"
        );
    }

    #[test]
    pub(in super::super) fn test_segment_len_tracks_tokens() {
        let mut state = make_state();
        let seg_open = 10i32;
        let seg_close = 11i32;

        state.update_segment_state(seg_open as u32, seg_open, seg_close);

        for i in 0..5 {
            state.record_token(40 + i, MAX_RECENT);
        }
        assert_eq!(state.segment_len, 5);

        // Re-opening the segment resets the counter
        state.update_segment_state(seg_open as u32, seg_open, seg_close);
        assert_eq!(state.segment_len, 0);
    }
}
