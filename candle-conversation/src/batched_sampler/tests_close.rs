//! Tests for the guard rails that end a sequence or a segment: the EOS failsafes, the segment closer script, and the length budget.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
pub(super) mod tests {
    // Several tests pin a tuning constant inside a range. The assertion IS the
    // contract — it fires the moment someone retunes the constant out of band —
    // even though it is constant-valued at any given commit.
    #![allow(clippy::assertions_on_constants)]

    use super::*;

    use crate::config::SamplingConfig;

    use super::super::tests_support::tests::*;

    /// The degenerate-decode abort must live on the resolver BOTH sampling
    /// paths run through, not in one path's copy of the overrides.
    ///
    /// It was once written into the CPU sampler alone, while every
    /// unconstrained row went to the CUDA kernel whenever the device was CUDA —
    /// so on the only configuration production runs, the guard was dead: a
    /// forward emitting token 0 forever ran to the length cap instead of
    /// stopping at `DEGENERATE_TOKEN_RUN`, writing hundreds of `!` into the
    /// conversation and into the substrate, where the turn's signatures then
    /// polluted retrieval. Asserting on `resolve` is what keeps the guard
    /// device-independent: there is no second copy to be missing from.
    #[test]
    pub(in super::super) fn degenerate_run_forces_eos_on_the_resolver_both_paths_share() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax();
        let mut state = make_state();

        // One short of the bar: an ordinary sampled token is committed as-is.
        for _ in 0..DEGENERATE_TOKEN_RUN - 1 {
            state.record_token(0, MAX_RECENT);
        }
        assert_eq!(
            sampler.resolve(7, &mut state, &config).0,
            7,
            "below the bar the sampler's own token stands"
        );

        // The next consecutive zero crosses it, and the turn is cut short.
        state.record_token(0, MAX_RECENT);
        assert_eq!(
            sampler.resolve(7, &mut state, &config).0,
            EOS_TOKEN,
            "a degenerate run must force EOS whichever path sampled the row"
        );

        // A real token clears the run, and decoding resumes normally — the
        // guard fires on a consecutive run, never on token 0 being frequent.
        state.record_token(42, MAX_RECENT);
        assert_eq!(sampler.resolve(7, &mut state, &config).0, 7);
    }

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

    // ── EOS failsafe override tests ────────────────────────────────────

    #[test]
    pub(in super::super) fn test_forced_eos_after_overrides_token() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(0, 10);
        let mut state = make_state();

        // Simulate 10 tokens already generated
        for _ in 0..10 {
            state.record_token(42, MAX_RECENT);
        }
        assert_eq!(state.current_len, 10);

        // Build logits that strongly favor token 42
        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        // Should have been overridden to EOS despite logits favoring 42
        assert_eq!(
            tokens[0], EOS_TOKEN,
            "forced_eos_after should override to EOS"
        );
    }

    #[test]
    pub(in super::super) fn test_graceful_eos_after_overrides_token() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(5, 0);
        let mut state = make_state();

        // Generate 5 tokens
        for _ in 0..5 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        assert_eq!(
            tokens[0], EOS_TOKEN,
            "graceful_eos_after should override to EOS"
        );
    }

    #[test]
    pub(in super::super) fn test_eos_failsafe_disabled_when_zero() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax(); // both 0 = disabled
        let mut state = make_state();

        for _ in 0..1000 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        assert_eq!(tokens[0], 42, "failsafe disabled: should sample normally");
    }

    #[test]
    pub(in super::super) fn test_eos_failsafe_below_threshold_no_override() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(100, 200);
        let mut state = make_state();

        // Only 50 tokens — under both thresholds
        for _ in 0..50 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        assert_eq!(tokens[0], 42, "below threshold: should sample normally");
    }

    #[test]
    pub(in super::super) fn test_eos_failsafe_records_eos_token_in_state() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_eos_failsafe(5, 10);
        let mut state = make_state();

        for _ in 0..5 {
            state.record_token(42, MAX_RECENT);
        }

        let mut logits_data = vec![0.0f32; VOCAB_SIZE];
        logits_data[42] = 100.0;
        let logits = candle::Tensor::from_vec(logits_data, (1, VOCAB_SIZE), &candle::Device::Cpu)
            .expect("tensor");

        let _ = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");

        // State should have recorded the EOS token (not the original 42)
        assert_eq!(state.current_len, 6); // 5 + 1 for the overridden token
        assert_eq!(*state.recent_tokens.last().unwrap(), EOS_TOKEN as i32);
    }
}
