//! Tests that every row of a mixed wave resolves against its own config.
//!
//! The kernel takes one set of scalar parameters per launch, and the host-side
//! post-kernel loop must not inherit them from the first row. Each row keeps its
//! own segment budget, its own EOS ramp and its own penalty history, whichever
//! order the rows come in. The structural think-close ban applies outside a
//! think block and not inside one, and a model with no think protocol is never
//! banned from anything. The next dispatch starts from clean device counts
//! rather than the last one's.

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

    // ── Structural think-close ban ─────────────────────────────────────

    /// **A `</think>` outside a think block is never sampleable.** The stencil
    /// owns the close inside a block; outside one there is nothing to close,
    /// and a stray close after the stencil's walk has finished is exactly what
    /// produced the doubled `</think>` that derailed a live turn — twice, on
    /// the same turn shape.
    #[test]
    pub(in super::super) fn a_think_close_outside_a_think_block_is_banned() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.segment_open_token_id = 89;
        config.segment_close_token_id = 90;
        let mut state = make_state();
        assert!(!state.in_segment, "a turn starts outside any think block");
        assert!(think_close_ban_active(&config, &state));

        let logits = logits_from_rows(&[&[(90, 100.0), (60, 50.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(
            tokens[0], 60,
            "the close token must be unreachable outside a block"
        );
    }

    // ── Per-row limits inside one shared launch ───────────────────────

    /// **A row's own segment-close budget applies inside a shared launch.**
    ///
    /// The host-side post-kernel loop resolves the segment-close budget (and the
    /// EOS failsafes) per row from `configs[i]`, never row 0. Both rows here go
    /// through ONE launch: row 0 is greedy over a decisive logit row, row 1 is
    /// mid-block with a hard cap of 1 and must still be forced closed on its own
    /// config, not row 0's.
    #[test]
    pub(in super::super) fn a_row_obeys_its_own_close_budget_in_a_shared_launch() {
        let sampler = make_sampler();
        let greedy = SamplingConfig::argmax();
        let mut forced = SamplingConfig::argmax();
        forced.segment_open_token_id = 89;
        forced.segment_close_token_id = 90;
        forced.force_segment_close_after = 1;

        let mut plain = make_state();
        let mut in_block = make_state();
        in_block.in_segment = true;
        in_block.segment_len = 5;

        let logits = logits_from_rows(&[&[(42, 100.0), (7, 1.0)], &[(42, 100.0), (7, 1.0)]]);
        let tokens = sampler
            .sample_batch(
                &logits,
                &mut [&mut plain, &mut in_block],
                &[&greedy, &forced],
            )
            .expect("sample");

        assert_eq!(tokens[0], 42, "the greedy row takes its own argmax");
        assert_eq!(
            tokens[1], 90,
            "the forced-close row closes on ITS config, not row 0's"
        );
    }

    /// Inside a block the ban is off: the model must stay free to emit the
    /// close (the steering intercepts it — `TokenClosedDrop` — or the hard-cap
    /// closer forces it; the sampler's job is only to not fight either).
    #[test]
    pub(in super::super) fn a_think_close_inside_a_think_block_is_free() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.segment_open_token_id = 89;
        config.segment_close_token_id = 90;
        let mut state = make_state();
        state.enter_segment();
        assert!(!think_close_ban_active(&config, &state));

        let logits = logits_from_rows(&[&[(90, 100.0), (60, 50.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 90, "inside a block the close samples normally");
    }

    /// Without segment tracking configured (close id < 0) the ban never
    /// activates — reference models with no think protocol are untouched.
    #[test]
    pub(in super::super) fn no_think_protocol_means_no_ban() {
        let config = SamplingConfig::argmax();
        let state = make_state();
        assert!(!think_close_ban_active(&config, &state));
    }

    // ── Per-row config in a mixed wave ────────────────────────────────────

    /// A row inside a think block, `n` tokens deep, with `force` as its hard
    /// segment-close cap. Close id 90, open id 80, no closer script and no
    /// sentence-end ids — so the ONLY thing that can emit 90 is the force cap.
    pub(in super::super) fn wave_row(force: i32) -> (SamplingConfig, SequenceSamplingState) {
        let mut config = SamplingConfig::top_k(1, 1.0);
        config.segment_close_token_id = 90;
        config.segment_open_token_id = 80;
        config.force_segment_close_after = force;
        config.graceful_segment_close_after = 0;
        let mut state = make_state();
        state.in_segment = true;
        state.segment_len = 5;
        (config, state)
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
