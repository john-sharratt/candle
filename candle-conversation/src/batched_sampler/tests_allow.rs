//! Tests for allow lists, stencils, bans, bitmaps and the think-close ban, on CPU and on CUDA.

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

    #[test]
    pub(in super::super) fn single_token_stencil_forces_that_token() {
        // Token 90 dominates, but the stencil allows only 33 — it must win.
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_stencil(vec![33]);
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(90, 100.0), (33, -10.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 33, "single-token stencil forces its token");
    }

    /// A row held off the span samples by name: a bitmap reads back as exactly
    /// its tokens.
    #[test]
    pub(in super::super) fn a_bitmap_reads_back_as_its_tokens() {
        assert_eq!(
            tokens_of(&[0x8000_0001, 0, 1 << 3]),
            vec![0, 31, 67],
            "bit `t % 32` of word `t / 32`"
        );
        assert_eq!(tokens_of(&bits_of(&[10, 20, 96])), vec![10, 20, 96]);
    }

    /// A row under a bitmap draws its best allowed token, on the host path.
    #[test]
    pub(in super::super) fn a_row_under_a_bitmap_draws_inside_it() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.allowed_bits = Some(bits_of(&[10, 20]));
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(50, 100.0), (20, 5.0), (10, 1.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 20);
    }

    #[test]
    pub(in super::super) fn stencil_picks_best_within_allow_list() {
        // Global best (50) is outside the stencil; best *inside* {10,20} is 20.
        let sampler = make_sampler();
        let config = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(50, 100.0), (20, 5.0), (10, 1.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 20);
    }

    /// **A batch that fails on one NaN row leaves every state as it found it**,
    /// so a clean row beside it, sampled again alone, draws exactly what it
    /// would have drawn alone from the start — what the scheduler's first
    /// sample relies on to fail only the bad row (`sample_first_tokens`). Two
    /// sampled allow-list rows: the clean one spread over its allowed tokens,
    /// the other NaN throughout, as a forward gone bad leaves it.
    #[test]
    pub(in super::super) fn a_batch_failed_by_a_nan_row_leaves_the_clean_rows_state_untouched() {
        let sampler = make_sampler();
        let config = sampled_config(7).with_stencil(vec![10, 20, 30]);
        let clean = logits_from_rows(&[&[(10, 2.0), (20, 2.2), (30, 1.9)]]);
        let mut data = clean.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        data.extend(vec![f32::NAN; VOCAB_SIZE]);
        let block = Tensor::from_vec(data, (2, VOCAB_SIZE), &Device::Cpu).unwrap();

        let mut state = make_state();
        let mut beside = make_state();
        let failed =
            sampler.sample_batch(&block, &mut [&mut state, &mut beside], &[&config, &config]);
        assert!(failed.is_err(), "a NaN row fails the batch it is in");
        let again = sampler
            .sample_batch(&clean, &mut [&mut state], &[&config])
            .expect("the clean row alone");

        let mut fresh = make_state();
        let alone = make_sampler()
            .sample_batch(&clean, &mut [&mut fresh], &[&config])
            .expect("the clean row from the start");
        assert_eq!(again, alone, "the failed batch moved the clean row's draw");
        assert_eq!(state.rng_offset, fresh.rng_offset);
    }

    /// Constrained rows are issued in the pass over the batch and finished
    /// after the unconstrained rows' kernel: each must land in its OWN row, and
    /// advance its own state, with an unconstrained row between them. Every row
    /// peaks at 50; the two allow-lists exclude it and pick their own best. On
    /// CUDA when a card is present — the path the deferral is for — and on the
    /// CPU path always.
    #[test]
    pub(in super::super) fn deferred_allow_list_rows_land_in_their_own_rows() {
        let mut devices = vec![Device::Cpu];
        devices.extend(Device::new_cuda(0).ok());
        let first = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        let free = SamplingConfig::argmax();
        let last = SamplingConfig::argmax().with_stencil(vec![30, 40]);
        for device in devices {
            let sampler = BatchedSampler::new(
                device.clone(),
                VOCAB_SIZE,
                VOCAB_SIZE,
                MAX_RECENT,
                vec![EOS_TOKEN].into(),
                None,
            );
            let (mut s0, mut s1, mut s2) = (make_state(), make_state(), make_state());
            let logits = logits_from_rows(&[
                &[(50, 100.0), (20, 5.0), (10, 1.0)],
                &[(50, 100.0), (20, 5.0)],
                &[(50, 100.0), (30, 7.0), (40, 9.0)],
            ])
            .to_device(&device)
            .expect("logits");
            let tokens = sampler
                .sample_batch(
                    &logits,
                    &mut [&mut s0, &mut s1, &mut s2],
                    &[&first, &free, &last],
                )
                .expect("sample");
            assert_eq!(tokens, vec![20, 50, 40], "{device:?}");
            assert_eq!((s0.rng_offset, s2.rng_offset), (1, 1), "{device:?}");
        }
    }

    /// **The kernel reads each free row in place, from its own row.** Rows 0
    /// and 2 are free and row 1 is constrained, so the kernel samples two rows
    /// out of a three-row block without a compacted copy of them: row 0 must
    /// come from row 0 (peak 70) and row 2 from row 2 (peak 90), not from the
    /// first two rows of the block (which would give 70 and row 1's peak, 60).
    /// Every row peaks at a different token so a misread row cannot pass.
    #[test]
    pub(in super::super) fn free_rows_beside_a_constrained_one_sample_their_own_logits() {
        let free = SamplingConfig::argmax();
        let constrained = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        for device in devices() {
            let sampler = BatchedSampler::new(
                device.clone(),
                VOCAB_SIZE,
                VOCAB_SIZE,
                MAX_RECENT,
                vec![EOS_TOKEN].into(),
                None,
            );
            let (mut s0, mut s1, mut s2) = (make_state(), make_state(), make_state());
            let logits = logits_from_rows(&[
                &[(70, 100.0)],
                &[(60, 100.0), (20, 5.0), (10, 1.0)],
                &[(90, 100.0)],
            ])
            .to_device(&device)
            .expect("logits");
            let tokens = sampler
                .sample_batch(
                    &logits,
                    &mut [&mut s0, &mut s1, &mut s2],
                    &[&free, &constrained, &free],
                )
                .expect("sample");
            assert_eq!(tokens, vec![70, 20, 90], "{device:?}");
        }
    }

    #[test]
    pub(in super::super) fn banned_token_excluded() {
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax();
        config.banned_tokens = vec![50];
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(50, 100.0), (60, 50.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 60, "banned best token → next best");
    }

    /// **A row's ban is its own inside a shared launch.** One row bans nothing;
    /// the other — the answer closing a stuck tool loop — bans 50. Both logit
    /// rows peak at 50: the free row takes it, the banning row its next best.
    /// Both orderings, because the kernel read the ban from `configs[0]`: with
    /// the banning row second its ban vanished, with it first the free row lost
    /// its token. On CUDA when a card is present — the path that had the defect
    /// — and on the CPU path always.
    #[test]
    pub(in super::super) fn a_row_s_ban_stays_in_its_row_in_a_shared_launch() {
        let mut devices = vec![Device::Cpu];
        devices.extend(Device::new_cuda(0).ok());
        let free = SamplingConfig::argmax();
        let mut closing = SamplingConfig::argmax();
        closing.banned_tokens = vec![50];
        for device in devices {
            let sampler = BatchedSampler::new(
                device.clone(),
                VOCAB_SIZE,
                VOCAB_SIZE,
                MAX_RECENT,
                vec![EOS_TOKEN].into(),
                None,
            );
            for (configs, expected) in [
                ([&free, &closing], vec![50, 60]),
                ([&closing, &free], vec![60, 50]),
            ] {
                let (mut a, mut b) = (make_state(), make_state());
                let logits =
                    logits_from_rows(&[&[(50, 100.0), (60, 50.0)], &[(50, 100.0), (60, 50.0)]])
                        .to_device(&device)
                        .expect("logits");
                let tokens = sampler
                    .sample_batch(&logits, &mut [&mut a, &mut b], &configs)
                    .expect("sample");
                assert_eq!(tokens, expected, "{device:?}");
            }
        }
    }

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

    #[test]
    pub(in super::super) fn stencil_and_banned_combine() {
        // Stencil {10,20,30}; 30 is best but banned → 20 wins.
        let sampler = make_sampler();
        let mut config = SamplingConfig::argmax().with_stencil(vec![10, 20, 30]);
        config.banned_tokens = vec![30];
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(30, 100.0), (20, 50.0), (10, 10.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 20);
    }

    #[test]
    pub(in super::super) fn empty_stencil_is_unconstrained() {
        let sampler = make_sampler();
        let config = SamplingConfig::argmax(); // no stencil, no bans
        let mut state = make_state();
        let logits = logits_from_rows(&[&[(77, 100.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut state], &[&config])
            .expect("sample");
        assert_eq!(tokens[0], 77);
    }

    #[test]
    pub(in super::super) fn per_row_stencils_are_independent() {
        // Two stenciled rows with disjoint allow-lists; the shared global best
        // (50) is outside both and must be masked in each.
        let sampler = make_sampler();
        let c0 = SamplingConfig::argmax().with_stencil(vec![10, 20]);
        let c1 = SamplingConfig::argmax().with_stencil(vec![30, 40]);
        let mut s0 = make_state();
        let mut s1 = make_state();
        let logits = logits_from_rows(&[
            &[(50, 100.0), (20, 5.0), (10, 1.0)],
            &[(50, 100.0), (30, 7.0), (40, 2.0)],
        ]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut s0, &mut s1], &[&c0, &c1])
            .expect("sample");
        assert_eq!(tokens, vec![20, 30]);
    }

    #[test]
    pub(in super::super) fn mixed_batch_stencils_only_its_own_row() {
        // THE per-row property the global-config kernel lacks: row 0 is forced
        // to token 10 while row 1 (free) keeps its global best 90 — the stencil
        // must not leak across rows.
        let sampler = make_sampler();
        let stenciled = SamplingConfig::argmax().with_stencil(vec![10]);
        let free = SamplingConfig::argmax();
        let mut s0 = make_state();
        let mut s1 = make_state();
        let logits = logits_from_rows(&[&[(90, 100.0), (10, 1.0)], &[(90, 100.0)]]);
        let tokens = sampler
            .sample_batch(&logits, &mut [&mut s0, &mut s1], &[&stenciled, &free])
            .expect("sample");
        assert_eq!(tokens, vec![10, 90], "stencil applies to row 0 only");
    }

    #[test]
    pub(in super::super) fn allow_list_gather_stays_in_set_under_temperature() {
        // Stochastic sampling (temperature > 0) over the gathered allow-list must
        // never escape it, across many seeds.
        let sampler = make_sampler();
        let allowed = [13usize, 41, 88];
        let logits = logits_from_rows(&[&[(13, 2.0), (41, 1.0), (88, 1.5), (90, 100.0)]]);
        for seed in 0..200u64 {
            let mut config = SamplingConfig::argmax().with_stencil(vec![13, 41, 88]);
            config.temperature = 1.0;
            config.top_p = 0.99;
            config.seed = seed;
            let mut state = make_state();
            let tokens = sampler
                .sample_batch(&logits, &mut [&mut state], &[&config])
                .expect("sample");
            assert!(
                allowed.contains(&(tokens[0] as usize)),
                "sampled {} escaped the allow-list at seed {seed}",
                tokens[0]
            );
        }
    }

    pub(in super::super) fn allow_row(allow: Vec<u32>, temperature: f32) -> AllowRow {
        AllowRow {
            logits_row: 0,
            banned: vec![false; allow.len()],
            allow,
            temperature,
            processor: LogitsProcessor::from_sampling(
                7,
                candle_transformers::generation::Sampling::All {
                    temperature: temperature as f64,
                },
            ),
        }
    }

    /// An argmax row reads back one word; a sampled row one per allowed token.
    #[test]
    pub(in super::super) fn an_allow_row_reads_back_its_index_or_its_probabilities() {
        assert_eq!(allow_result_words(&allow_row(vec![3, 4, 5], 0.0)), 1);
        assert_eq!(allow_result_words(&allow_row(vec![3, 4, 5], 0.8)), 3);
    }

    /// The host half maps the result back to the allowed token: the argmax's
    /// position, or the only position holding probability.
    #[test]
    pub(in super::super) fn an_allow_row_finishes_on_its_allowed_token() {
        assert_eq!(
            finish_allow_row(allow_row(vec![3, 4, 5], 0.0), AllowResult::Index(2)).unwrap(),
            5
        );
        assert_eq!(
            finish_allow_row(
                allow_row(vec![3, 4, 5], 1.0),
                AllowResult::Probs(vec![0.0, 1.0, 0.0])
            )
            .unwrap(),
            4
        );
        assert!(
            finish_allow_row(allow_row(vec![3, 4], 0.0), AllowResult::Index(2)).is_err(),
            "a position past the list is refused, not wrapped"
        );
    }

    /// **A bitmap binds its own row on the kernel path.** In one dispatch a row
    /// beside it samples its spike, and each bitmap row its own allowed best: a
    /// bitmap read from the wrong offset takes one row's mask to another.
    #[test]
    pub(in super::super) fn bitmaps_bind_their_own_rows_in_the_wave_on_cuda() {
        let Ok(device) = candle::Device::new_cuda(0) else {
            return;
        };
        let sampler = make_sampler_on(&device, VOCAB_SIZE);
        let mut data = vec![0.0f32; 3 * VOCAB_SIZE];
        for row in 0..3 {
            data[row * VOCAB_SIZE + 50] = 100.0;
            data[row * VOCAB_SIZE + 20] = 5.0;
            data[row * VOCAB_SIZE + 10] = 1.0;
        }
        let logits = Tensor::from_vec(data, (3, VOCAB_SIZE), &device).expect("logits");
        let free = SamplingConfig::argmax();
        let mut first = SamplingConfig::argmax();
        first.allowed_bits = Some(bits_of(&[10, 20]));
        let mut second = SamplingConfig::argmax();
        second.allowed_bits = Some(bits_of(&[10]));
        let (mut a, mut b, mut c) = (make_state(), make_state(), make_state());
        let tokens = sampler
            .sample_batch(
                &logits,
                &mut [&mut a, &mut b, &mut c],
                &[&first, &free, &second],
            )
            .expect("sample");
        assert_eq!(tokens, vec![20, 50, 10]);
    }

    /// What a step costs one row of a 248k vocabulary, free, under a bitmap and
    /// on the allow-list the bitmap replaces. Prints timings.
    #[test]
    #[ignore = "a timing on a quiet card"]
    pub(in super::super) fn a_bitmap_costs_a_step_this_much_on_cuda() {
        const VOCAB: usize = 248_077;
        let device = candle::Device::new_cuda(0).expect("a card");
        let sampler = BatchedSampler::new(device.clone(), VOCAB, VOCAB, 32, vec![2].into(), None);
        let mut data = vec![0.0f32; VOCAB];
        for (t, logit) in data.iter_mut().enumerate() {
            *logit = (t % 977) as f32 / 977.0;
        }
        let logits = Tensor::from_vec(data, (1, VOCAB), &device).expect("logits");
        let allowed: Vec<u32> = (0..VOCAB as u32).filter(|t| t % 256 != 7).collect();
        let set = crate::stencil::AllowedSet::from_tokens(allowed.clone());

        let mut free = SamplingConfig::argmax();
        free.temperature = 0.8;
        let mut masked = free.clone();
        masked.constrain_to(&set);
        let listed = free
            .clone()
            .with_stencil(allowed.iter().map(|&t| t as i32).collect());

        for (name, config) in [("free", &free), ("bitmap", &masked), ("list", &listed)] {
            let mut state = make_state();
            let started = std::time::Instant::now();
            for _ in 0..200 {
                sampler
                    .sample_batch(&logits, &mut [&mut state], &[config])
                    .expect("sample");
            }
            println!("{name}: {:?} a step", started.elapsed() / 200);
        }
    }

    /// On CUDA a sampled allow-list row draws inside its list, a banned
    /// position is never drawn however high its logit, and a row whose mass is
    /// nearly all on one token draws it.
    #[test]
    pub(in super::super) fn a_sampled_allow_list_row_draws_inside_its_list_on_cuda() {
        let Ok(device) = candle::Device::new_cuda(0) else {
            return;
        };
        let sampler = make_sampler_on(&device, VOCAB_SIZE);
        let mut data = vec![0.0f32; VOCAB_SIZE];
        data[10] = 9.0;
        data[20] = 0.0;
        data[30] = 50.0;
        let logits = Tensor::from_vec(data, (1, VOCAB_SIZE), &device).expect("logits");
        let mut config = SamplingConfig::argmax().with_stencil(vec![10, 20, 30]);
        config.temperature = 1.0;
        config.banned_tokens = vec![30];
        for seed in 0..8u64 {
            config.seed = seed;
            let mut state = make_state();
            let tokens = sampler
                .sample_batch(&logits, &mut [&mut state], &[&config])
                .expect("sample");
            assert_eq!(tokens[0], 10, "seed {seed}");
        }
    }
}
