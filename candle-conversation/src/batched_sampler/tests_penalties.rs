//! Tests for the history each row is priced by: the penalty flags, DRY's window, and the cross-turn table.
//!
//! Tests that an unpenalised step reads no history for that step alone, and that DRY's window is one free-text value.
//! Tests that the repeat penalty flips with DRY, that the cross-turn table is gathered only when its penalty is on, and that the sparse indexes track the nonzero counts.
//! Tests the per-row dials the kernel reads, including each config's own segment-close token and the gating when that token is missing.

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

    use super::super::tests_helpers::tests::*;

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

    // ── Tool-call penalty suppression ──────────────────────────────────

    #[test]
    pub(in super::super) fn tool_call_row_penalty_state_is_zeroed_and_think_row_is_not() {
        let sampler = make_sampler();
        let mut in_call = make_state();
        let mut thinking = make_state();
        // Both rows generated the same tokens (e.g. digits reasoned in <think>).
        for _ in 0..5 {
            in_call.record_token(42, MAX_RECENT);
            thinking.record_token(42, MAX_RECENT);
        }
        in_call.in_tool_call = true;
        // A prior turn, so the cross-turn table has something to suppress.
        in_call.end_turn(0);
        thinking.end_turn(0);
        for _ in 0..5 {
            in_call.record_token(42, MAX_RECENT);
            thinking.record_token(42, MAX_RECENT);
        }

        let config = window_16();
        let rows = gather(&sampler, &[&in_call, &thinking], &config, true);

        // Row 0 (tool call) stamps nothing, so its table rows read zero — the
        // model is free to reproduce the query's tokens verbatim in the
        // arguments. Row 1 (think block) keeps full repetition control.
        let row1_42 = VOCAB_SIZE as u32 + 42;
        assert_eq!(
            rows.counts,
            SparseCounts {
                offsets: vec![row1_42],
                values: vec![5],
            }
        );
        assert_eq!(
            rows.cross,
            SparseCounts {
                offsets: vec![row1_42],
                values: vec![5],
            }
        );
        assert_eq!(rows.recent_lens, vec![0, 10]);
    }

    /// **An unpenalized step is priced by no history, for that step alone**:
    /// the same state, gathered under a config that marks the step and under
    /// one that does not, stamps nothing the first time and its counts, window
    /// and DRY span the second.
    #[test]
    pub(in super::super) fn an_unpenalized_step_reads_no_history_and_the_next_step_does() {
        let sampler = make_sampler();
        let mut state = make_state();
        // Five tokens of one free-text value: the first opens it.
        state.record_token(42, MAX_RECENT);
        state.set_free_text(true);
        for _ in 0..4 {
            state.record_token(42, MAX_RECENT);
        }
        let plain = window_16().with_dry_penalty(0.8, 1.75, 4, 512);
        let lifted = SamplingConfig {
            unpenalized_step: true,
            ..plain.clone()
        };
        let mut rows = KernelRows::new(sampler.vocab_size, sampler.max_recent_len, false);
        let spec = |r: usize| RowSpec {
            seq: r,
            logits_row: r,
            draft: None,
            ahead: &[],
        };
        rows.push(0, &state, &lifted, &spec(0), None, state.row_key());
        rows.push(1, &state, &plain, &spec(1), None, state.row_key());
        assert_eq!(
            rows.counts,
            SparseCounts {
                offsets: vec![VOCAB_SIZE as u32 + 42],
                values: vec![5],
            }
        );
        assert_eq!(rows.recent_lens, vec![0, 5]);
        assert_eq!(rows.dry_lens, vec![0, 5]);
    }

    /// **DRY's window is one free-text value.** Outside a value it is empty;
    /// entering one starts it at the token that opened it; it counts only that
    /// value's tokens; leaving empties it, and the next value starts afresh.
    #[test]
    pub(in super::super) fn dry_sees_only_the_value_being_written() {
        let sampler = make_sampler();
        let config = window_16().with_dry_penalty(0.8, 1.75, 4, 512);
        let mut state = make_state();
        // Structure and reasoning before the value: no window.
        for _ in 0..7 {
            state.record_token(42, MAX_RECENT);
        }
        state.set_free_text(false);
        assert_eq!(
            gather(&sampler, &[&state], &config, false).dry_lens,
            vec![0]
        );
        // The value's first token, then three more.
        state.record_token(5, MAX_RECENT);
        state.set_free_text(true);
        assert_eq!(
            gather(&sampler, &[&state], &config, false).dry_lens,
            vec![1]
        );
        for t in [6, 7, 8] {
            state.record_token(t, MAX_RECENT);
            state.set_free_text(true);
        }
        assert_eq!(
            gather(&sampler, &[&state], &config, false).dry_lens,
            vec![4]
        );
        // The value closes; the structure after it is outside any window.
        state.record_token(9, MAX_RECENT);
        state.set_free_text(false);
        state.record_token(10, MAX_RECENT);
        assert_eq!(
            gather(&sampler, &[&state], &config, false).dry_lens,
            vec![0]
        );
        // The next value starts at one, not where the last left off.
        state.record_token(11, MAX_RECENT);
        state.set_free_text(true);
        assert_eq!(
            gather(&sampler, &[&state], &config, false).dry_lens,
            vec![1]
        );
        // A tool call freed from the penalties is outside DRY even mid-value.
        state.in_tool_call = true;
        assert_eq!(
            gather(&sampler, &[&state], &config, false).dry_lens,
            vec![0]
        );
    }

    /// **The repeat penalty flips with DRY**: a row inside a free-text value
    /// samples on the config's DRY-time penalty, any other row on its own.
    #[test]
    pub(in super::super) fn the_repeat_penalty_a_row_samples_on_flips_with_dry() {
        let sampler = make_sampler();
        let config = SamplingConfig {
            repeat_penalty: 1.05,
            ..window_16()
        }
        .with_dry_penalty(0.8, 1.75, 2, 512)
        .with_dry_repeat_penalty(1.0);
        let outside = make_state();
        let mut inside = make_state();
        inside.record_token(5, MAX_RECENT);
        inside.set_free_text(true);
        let rows = gather(&sampler, &[&outside, &inside], &config, false);
        let repeat: Vec<f32> = rows.seq_dials.iter().map(|d| d.repeat_penalty).collect();
        assert_eq!(repeat, vec![1.05, 1.0]);

        // A config that cleared DRY keeps its own penalty inside a value too:
        // it has nothing standing in for the 1.05.
        let no_dry = SamplingConfig {
            dry: None,
            ..config
        };
        let rows = gather(&sampler, &[&inside], &no_dry, false);
        assert_eq!(rows.seq_dials[0].repeat_penalty, 1.05);
        assert_eq!(rows.dry_lens, vec![0]);
    }

    /// With the cross-turn penalty off, the cross table is never read, so
    /// nothing is gathered for it.
    #[test]
    pub(in super::super) fn the_cross_turn_table_is_gathered_only_when_its_penalty_is_on() {
        let sampler = make_sampler();
        let mut state = make_state();
        state.record_token(7, MAX_RECENT);
        state.end_turn(0);
        let config = window_16();
        let rows = gather(&sampler, &[&state], &config, false);
        assert_eq!(rows.cross, SparseCounts::default());
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
}
