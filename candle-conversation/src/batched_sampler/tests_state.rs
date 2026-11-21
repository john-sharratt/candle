//! Tests for sequence state: the degenerate-run bar, the cross-turn token window, sparse indexes, and segment tracking.

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

    /// The per-row dials the kernel reads must carry each config's OWN
    /// segment-close token, not row 0's. A wave that co-batches a deliberating
    /// row (closes `</think>` = 90) with a narrator row (no segment) must give
    /// the deliberating row token 90 and the narrator row -1 — the exact bleed
    /// `SeqDials` exists to remove, and the one a scalar `configs[0]` reintroduced.
    #[test]
    pub(in super::super) fn seq_dials_carry_each_rows_own_segment_close_token() {
        let mut deliberating = SamplingConfig::argmax();
        deliberating.temperature = 0.7;
        deliberating.segment_close_boost = 3.0;
        deliberating.segment_close_token_id = 90;
        deliberating.segment_close_ramp_start = 2;
        deliberating.segment_close_ramp_len = 16;
        deliberating.segment_close_max_multiplier = 4.0;

        // A narrator: no segment close at all.
        let mut narrator = SamplingConfig::argmax();
        narrator.temperature = 0.9;
        narrator.segment_close_boost = 0.0;
        narrator.segment_close_token_id = -1;

        let d = SeqDials::from_config(&deliberating, 0);
        let n = SeqDials::from_config(&narrator, 3);
        assert_eq!(
            (d.logits_row, n.logits_row),
            (0, 3),
            "each names its own row"
        );

        assert_eq!(
            d.segment_close_token_id, 90,
            "deliberating row keeps its close token"
        );
        assert_eq!(d.segment_close_boost, 3.0);
        assert_eq!(d.segment_close_ramp_len, 16);
        assert_eq!(d.temperature, 0.7);

        assert_eq!(
            n.segment_close_token_id, -1,
            "narrator row's close path stays off"
        );
        assert_eq!(n.segment_close_boost, 0.0);
        assert_eq!(n.temperature, 0.9);
    }

    /// A config with the boost dialled up but NO token id is not "half on": the
    /// whole segment-close path is gated off (token id -1), so a co-batched row
    /// with it genuinely on is unaffected.
    #[test]
    pub(in super::super) fn seq_dials_gate_segment_close_off_when_token_missing() {
        let mut c = SamplingConfig::argmax();
        c.segment_close_boost = 5.0;
        c.segment_close_token_id = -1; // boost set, but no token to boost
        let d = SeqDials::from_config(&c, 0);
        assert_eq!(d.segment_close_token_id, -1);
        assert_eq!(
            d.segment_close_boost, 0.0,
            "boost neutralised without a token"
        );
    }

    /// A run of token 0 is counted, and any other token clears it — the guard
    /// must fire on a *consecutive* run, not on token 0 being frequent.
    #[test]
    pub(in super::super) fn degenerate_run_counts_consecutive_zero_tokens() {
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
    pub(in super::super) fn degenerate_run_resets_at_turn_end() {
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
    pub(in super::super) fn crossing_the_degenerate_bar_marks_the_turn_until_it_ends() {
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
    pub(in super::super) fn the_cross_turn_window_forgets_turns_that_fall_out_of_it() {
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
    pub(in super::super) fn a_token_used_again_stays_counted_until_its_last_use_leaves() {
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
    pub(in super::super) fn a_zero_window_remembers_every_turn() {
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
    pub(in super::super) fn a_turn_that_sampled_nothing_takes_no_slot() {
        let mut st = SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT);
        st.record_token(3, MAX_RECENT);
        st.end_turn(1);
        st.end_turn(1);
        assert_eq!(st.cross_turn_counts[3], 1);
    }

    /// The bar has to be low enough to stop a broken forward promptly, and high
    /// enough that ordinary text can never reach it.
    #[test]
    pub(in super::super) fn degenerate_run_bar_is_small_but_out_of_language_range() {
        assert!(
            DEGENERATE_TOKEN_RUN >= 4,
            "must tolerate a brief coincidence"
        );
        assert!(
            DEGENERATE_TOKEN_RUN <= 16,
            "must fire long before the length cap: the observed failure ran 1219 tokens",
        );
    }

    /// The sparse indexes name exactly the nonzero dense entries, through
    /// recording, a turn end and a windowed cross-turn eviction — they are what
    /// the device table is stamped from, so a token missing from them is a
    /// penalty silently not applied.
    #[test]
    pub(in super::super) fn the_sparse_indexes_track_the_nonzero_counts_across_turns() {
        fn nonzero(counts: &[i32]) -> Vec<u32> {
            (0..counts.len() as u32)
                .filter(|&t| counts[t as usize] > 0)
                .collect()
        }
        fn sorted(v: &[u32]) -> Vec<u32> {
            let mut v = v.to_vec();
            v.sort_unstable();
            v
        }
        let mut st = make_state();
        for t in [9, 3, 9, 4] {
            st.record_token(t, MAX_RECENT);
        }
        assert_eq!(st.counted, vec![9, 3, 4]);
        assert_eq!(sorted(&st.counted), nonzero(&st.token_counts));

        // Window of one: turn A enters the cross counts.
        st.end_turn(1);
        assert!(st.counted.is_empty());
        assert_eq!(nonzero(&st.token_counts), Vec::<u32>::new());
        assert_eq!(sorted(&st.cross_counted), vec![3, 4, 9]);
        assert_eq!(sorted(&st.cross_counted), nonzero(&st.cross_turn_counts));

        // Turn B evicts turn A: only B's tokens remain.
        for t in [4, 5] {
            st.record_token(t, MAX_RECENT);
        }
        st.end_turn(1);
        assert_eq!(sorted(&st.cross_counted), vec![4, 5]);
        assert_eq!(sorted(&st.cross_counted), nonzero(&st.cross_turn_counts));
        assert_eq!(st.cross_turn_counts[4], 1);

        st.clear();
        assert!(st.counted.is_empty() && st.cross_counted.is_empty());
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
