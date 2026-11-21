//! Tests for the degenerate-decode guard, the cross-turn window, and the EOS
//! failsafes.
//!
//! The guard counts only consecutive token-0 emissions, so interleaved zeros
//! never reach the bar and a real token clears the run. The run is per turn,
//! and once a turn has crossed the bar it stays marked until the turn ends,
//! even when a forced EOS or a rewound draft follows. The abort is tested on the
//! shared resolver, so the CPU and CUDA paths cannot disagree about it. The
//! cross-turn window tests check that a turn leaves with its tokens and that a
//! token used again stays counted until its last use leaves.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    // Several tests pin a tuning constant inside a range. The assertion IS the
    // contract — it fires the moment someone retunes the constant out of band —
    // even though it is constant-valued at any given commit.
    #![allow(clippy::assertions_on_constants)]

    use super::*;

    use crate::config::SamplingConfig;

    use super::super::tests_helpers::tests::*;

    /// A run of token 0 is counted, and any other token clears it — the guard
    /// must fire on a *consecutive* run, not on token 0 being frequent.
    #[test]
    fn degenerate_run_counts_consecutive_zero_tokens() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        assert_eq!(st.degenerate_run, 0);
        for expected in 1..=DEGENERATE_TOKEN_RUN {
            st.record_token(0, MAX_RECENT);
            assert_eq!(st.degenerate_run, expected);
        }
        // A real token breaks the run.
        st.record_token(7, MAX_RECENT);
        assert_eq!(st.degenerate_run, 0);
        // Interleaved zeros never accumulate to the bar.
        for _ in 0..50 {
            st.record_token(0, MAX_RECENT);
            st.record_token(7, MAX_RECENT);
        }
        assert_eq!(st.degenerate_run, 0);
    }

    /// The run is per-turn: a fresh turn must not inherit a previous turn's tail.
    #[test]
    fn degenerate_run_resets_at_turn_end() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for _ in 0..DEGENERATE_TOKEN_RUN {
            st.record_token(0, MAX_RECENT);
        }
        assert_eq!(st.degenerate_run, DEGENERATE_TOKEN_RUN);
        st.end_turn(0);
        assert_eq!(st.degenerate_run, 0);
    }

    /// **A turn that crossed the bar stays marked degenerate**, whatever is
    /// committed after: the guard's forced EOS resets the run, and a
    /// speculative step committing the run's last `!` and that EOS together
    /// left a turn of `!` reading as healthy by the end of the step — sealed
    /// as an answer in production. A draft rewound past the bar unmarks it;
    /// the next turn starts unmarked.
    #[test]
    fn crossing_the_degenerate_bar_marks_the_turn_until_it_ends() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for _ in 0..DEGENERATE_TOKEN_RUN - 1 {
            st.record_token(0, MAX_RECENT);
        }
        assert!(!st.degenerate, "below the bar");

        let before = st.mark();
        st.record_token(0, MAX_RECENT);
        st.record_token(7, MAX_RECENT);
        assert_eq!(st.degenerate_run, 0, "the EOS reset the run");
        assert!(st.degenerate, "but the turn crossed the bar");

        st.rewind(before, &[0, 7]);
        assert!(!st.degenerate, "a rewound draft never crossed it");
        assert_eq!(st.degenerate_run, DEGENERATE_TOKEN_RUN - 1);

        st.record_token(0, MAX_RECENT);
        assert!(st.degenerate);
        st.end_turn(0);
        assert!(!st.degenerate, "the next turn starts unmarked");
    }

    /// **The cross-turn penalty sees a window of turns, not the whole day.**
    ///
    /// It is flat — any count above zero costs the same — so without a bound it
    /// ends up on a character's entire working vocabulary, and a uniform shift
    /// tells no act from any other. With a window, a turn that leaves takes its
    /// tokens with it.
    #[test]
    fn the_cross_turn_window_forgets_turns_that_fall_out_of_it() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for tok in [3u32, 4, 5] {
            st.record_token(tok, MAX_RECENT);
            st.end_turn(2);
        }
        assert_eq!(
            st.cross_turn_counts[3], 0,
            "two turns on, the first is forgotten"
        );
        assert_eq!(st.cross_turn_counts[4], 1);
        assert_eq!(st.cross_turn_counts[5], 1);
    }

    /// A token used again is still penalised until its *last* use leaves.
    #[test]
    fn a_token_used_again_stays_counted_until_its_last_use_leaves() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for tok in [3u32, 3, 4] {
            st.record_token(tok, MAX_RECENT);
            st.end_turn(2);
        }
        assert_eq!(
            st.cross_turn_counts[3], 1,
            "its second use is still in the window"
        );
        st.record_token(5, MAX_RECENT);
        st.end_turn(2);
        assert_eq!(st.cross_turn_counts[3], 0, "and now it is not");
    }

    /// `0` keeps every turn, which is what every caller that sets no window —
    /// all of them with the penalty off — has always had.
    #[test]
    fn a_zero_window_remembers_every_turn() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        for tok in [3u32, 4, 5, 6] {
            st.record_token(tok, MAX_RECENT);
            st.end_turn(0);
        }
        assert!([3usize, 4, 5, 6]
            .iter()
            .all(|&t| st.cross_turn_counts[t] == 1));
        assert!(
            st.cross_turn_history.is_empty(),
            "nothing to forget, nothing kept"
        );
    }

    /// `end_turn` also runs at a conversation's first decode with nothing
    /// sampled yet. That must not use up a slot.
    #[test]
    fn a_turn_that_sampled_nothing_takes_no_slot() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        st.record_token(3, MAX_RECENT);
        st.end_turn(1);
        st.end_turn(1);
        assert_eq!(st.cross_turn_counts[3], 1);
    }

    /// The bar has to be low enough to stop a broken forward promptly, and high
    /// enough that ordinary text can never reach it.
    #[test]
    fn degenerate_run_bar_is_small_but_out_of_language_range() {
        assert!(
            DEGENERATE_TOKEN_RUN >= 4,
            "must tolerate a brief coincidence"
        );
        assert!(
            DEGENERATE_TOKEN_RUN <= 16,
            "must fire long before the length cap: the observed failure ran 1219 tokens",
        );
    }

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
    fn degenerate_run_forces_eos_on_the_resolver_both_paths_share() {
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

    // ── EOS failsafe override tests ────────────────────────────────────

    #[test]
    fn test_forced_eos_after_overrides_token() {
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
    fn test_graceful_eos_after_overrides_token() {
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
    fn test_eos_failsafe_disabled_when_zero() {
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
    fn test_eos_failsafe_below_threshold_no_override() {
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
    fn test_eos_failsafe_records_eos_token_in_state() {
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
