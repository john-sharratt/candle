//! Tests for per-row budgets and penalty windows across a wave, and for kernel pricing.

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

    /// **Every row of a wave resolves against its OWN config.**
    ///
    /// The kernel takes one set of scalar params per launch, and the CUDA path
    /// reads them from `configs[0]`. That collapse must NOT reach the host-side
    /// per-row loop that applies the segment-close and EOS overrides: a wave
    /// mixes sequences with genuinely different budgets — a 200-token
    /// `ThinkMode::Off` ingest summary beside a dialogue turn — and each must
    /// get its own.
    ///
    /// Both orderings are asserted because either one alone passes under the
    /// wrong fix: reading `configs[0]` satisfies the first, reading the last
    /// row's config satisfies the second.
    ///
    /// Runs on CUDA, which is the only path that had the defect — the CPU
    /// sampler always read configs per row, so a CPU-only test could never have
    /// caught it.
    #[test]
    pub(in super::super) fn each_row_of_a_wave_honours_its_own_segment_budget() {
        let Ok(device) = candle::Device::new_cuda(0) else {
            return; // No CUDA device on this box; the CPU path is covered above.
        };
        let sampler = BatchedSampler::new(
            device.clone(),
            VOCAB_SIZE,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        );

        // Logits that make token 42 the argmax and the close token 90 the
        // least likely, so a row that emits 90 can only have been forced.
        let mut row = vec![0.0f32; VOCAB_SIZE];
        row[42] = 10.0;
        row[90] = -10.0;

        for (label, forces) in [("summary first", [1, 1536]), ("dialogue first", [1536, 1])] {
            let (cfg_a, mut st_a) = wave_row(forces[0]);
            let (cfg_b, mut st_b) = wave_row(forces[1]);
            let configs = [&cfg_a, &cfg_b];
            let mut states = [&mut st_a, &mut st_b];

            let logits = Tensor::from_vec(
                row.iter().chain(row.iter()).copied().collect::<Vec<f32>>(),
                (2, VOCAB_SIZE),
                &device,
            )
            .expect("logits");

            let tokens = sampler
                .sample_batch(&logits, &mut states, &configs)
                .expect("sample_batch");

            for (i, &force) in forces.iter().enumerate() {
                if force == 1 {
                    assert_eq!(
                        tokens[i], 90,
                        "{label}: row {i} is 5 tokens into its block with a hard cap of 1 — \
                         its close must be forced, whatever the other row's budget is"
                    );
                } else {
                    assert_ne!(
                        tokens[i], 90,
                        "{label}: row {i} has a cap of 1536 and is only 5 tokens in — \
                         it must NOT inherit the other row's forced close"
                    );
                }
            }
        }
    }

    /// The kernel reads its counts from the device-resident tables: each row
    /// is priced by its own history, a tool-call row by none, and the next
    /// dispatch starts from a clean table rather than the last one's counts.
    ///
    /// Token 42 leads token 43 by one logit, so a penalty of more than one on
    /// 42 flips the greedy pick — the pick says whether the count reached the
    /// kernel.
    #[test]
    pub(in super::super) fn the_kernel_prices_each_row_from_its_own_resident_counts() {
        let Ok(device) = candle::Device::new_cuda(0) else {
            return; // No CUDA device on this box.
        };
        let sampler = BatchedSampler::new(
            device.clone(),
            VOCAB_SIZE,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        );
        let mut row = vec![0.0f32; VOCAB_SIZE];
        row[42] = 10.0;
        row[43] = 9.0;
        let logits = |rows: usize| {
            Tensor::from_vec(row.repeat(rows), (rows, VOCAB_SIZE), &device).expect("logits")
        };

        // Frequency: 42 said three times costs it 3 × 2.0.
        let freq = SamplingConfig {
            frequency_penalty: 2.0,
            ..SamplingConfig::argmax()
        };
        let mut repeated = make_state();
        let mut fresh = make_state();
        let mut in_call = make_state();
        for _ in 0..3 {
            repeated.record_token(42, MAX_RECENT);
            in_call.record_token(42, MAX_RECENT);
        }
        in_call.in_tool_call = true;
        let tokens = sampler
            .sample_batch(
                &logits(3),
                &mut [&mut repeated, &mut fresh, &mut in_call],
                &[&freq, &freq, &freq],
            )
            .expect("sample_batch");
        assert_eq!(tokens, vec![43, 42, 42]);

        // The next dispatch puts a fresh row where the repeated one was: it must
        // read zero there, not the counts the last dispatch stamped.
        let mut after = make_state();
        let tokens = sampler
            .sample_batch(&logits(1), &mut [&mut after], &[&freq])
            .expect("sample_batch");
        assert_eq!(tokens, vec![42]);

        // Cross-turn: 42 said in a prior turn costs it 5.0 this turn.
        let cross = SamplingConfig {
            cross_turn_penalty: 5.0,
            ..SamplingConfig::argmax()
        };
        let mut said_before = make_state();
        said_before.record_token(42, MAX_RECENT);
        said_before.end_turn(0);
        let mut never = make_state();
        let tokens = sampler
            .sample_batch(
                &logits(2),
                &mut [&mut said_before, &mut never],
                &[&cross, &cross],
            )
            .expect("sample_batch");
        assert_eq!(tokens, vec![43, 42]);
    }
}
