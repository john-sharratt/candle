//! Tests for scoring a block in one dispatch and for the speculative verify rows.
//!
//! Tests that one dispatch over a whole block commits what a position-at-a-time walk commits, and that a rejected draft stops the block where the walk stops.
//! Tests that a position behind a forced token scores as unreachable, and that a draft over the typical bar is committed while a draft under it commits the sample.
//! Tests the typical-acceptance rule on raw probabilities and over the truncated distribution, and that greedy verification is unchanged.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
pub(super) mod tests {
    // Several tests pin a tuning constant inside a range. The assertion IS the
    // contract — it fires the moment someone retunes the constant out of band —
    // even though it is constant-valued at any given commit.
    #![allow(clippy::assertions_on_constants)]

    use std::collections::HashSet;

    use super::*;

    use crate::config::SamplingConfig;

    use super::super::tests_helpers::tests::*;

    /// The typical-acceptance rule on raw probabilities. `[0.5, 0.3, 0.2]` has
    /// entropy 1.0297, so `δ·e^(−H)` = 0.1071 and the bar is ε = 0.09: the 0.2
    /// draft clears it. `[0.9, 0.05, 0.05]` has entropy 0.3944, the bar is again
    /// 0.09, and a 0.05 draft does not.
    #[test]
    pub(in super::super) fn typical_rule_accepts_on_mass_against_the_entropy_bar() {
        let ln = |ps: &[f32]| ps.iter().map(|p| p.ln()).collect::<Vec<f32>>();
        let m = TypicalAcceptance::MEDUSA;
        let accepts =
            |ps: &[f32], t: f32, draft: u32| typical_accepts_row(&ln(ps), t, 0, 1.0, draft, m);
        assert!(accepts(&[0.5, 0.3, 0.2], 1.0, 2));
        assert!(!accepts(&[0.9, 0.05, 0.05], 1.0, 2));
        // Temperature sharpens the row: at 0.5, [0.5, 0.3, 0.2] becomes
        // [0.658, 0.237, 0.105] (entropy 0.8533, bar 0.09), and 0.105 still
        // clears it; [0.9, 0.05, 0.05] becomes [0.994, 0.003, 0.003].
        assert!(accepts(&[0.5, 0.3, 0.2], 0.5, 2));
        assert!(!accepts(&[0.9, 0.05, 0.05], 0.5, 1));
        // A draft past the row is not a token.
        assert!(!accepts(&[0.5, 0.5], 1.0, 7));
    }

    /// **The rule is measured over the distribution the kernel samples, not
    /// the whole row.** `[0.5, 0.3, 0.2]` accepts its 0.2 draft over the whole
    /// row, but a nucleus at 0.75 keeps `[0.5, 0.3]` (cumulative 0.8) and top-k
    /// 2 keeps the same two: the draft is outside either, has probability zero,
    /// and is not accepted — exactly as the kernel decides it.
    #[test]
    pub(in super::super) fn typical_rule_is_measured_over_the_truncated_distribution() {
        let row: Vec<f32> = [0.5f32, 0.3, 0.2].iter().map(|p| p.ln()).collect();
        let m = TypicalAcceptance::MEDUSA;
        assert!(typical_accepts_row(&row, 1.0, 0, 1.0, 2, m));
        assert!(!typical_accepts_row(&row, 1.0, 0, 0.75, 2, m));
        assert!(!typical_accepts_row(&row, 1.0, 2, 1.0, 2, m));
        // Renormalised over the nucleus `[0.625, 0.375]` (entropy 0.6616, bar
        // 0.09), the 0.375 draft clears it.
        assert!(typical_accepts_row(&row, 1.0, 0, 0.75, 1, m));
    }

    /// One logits row with `p(5) = 0.8` and `p(7) = 0.2`; every other token
    /// sits at logit 0, ~2e-9 each.
    pub(in super::super) fn verify_row_logits() -> Tensor {
        logits_from_rows(&[&[(5, 20.0), (7, 20.0 + 0.25f32.ln())]])
    }

    /// `n` rows that each spread their mass over tokens 40..45, ranked
    /// differently per row, so a sampled walk wanders and the penalties move
    /// its picks.
    pub(in super::super) fn walk_logits(n: usize) -> Tensor {
        let mut data = vec![-20.0f32; n * VOCAB_SIZE];
        for r in 0..n {
            for k in 0..5 {
                data[r * VOCAB_SIZE + 40 + k] = 5.0 - 0.3 * ((k + r) % 5) as f32;
            }
        }
        Tensor::from_vec(data, (n, VOCAB_SIZE), &Device::Cpu).expect("logits")
    }

    /// Sampled, with every history-reading penalty on.
    pub(in super::super) fn penalised(seed: u64) -> SamplingConfig {
        SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 1.0,
            repeat_penalty: 1.3,
            frequency_penalty: 0.4,
            presence_penalty: 0.5,
            seed,
            ..SamplingConfig::argmax()
        }
    }

    /// A sequence some way into its turn.
    pub(in super::super) fn walked_state() -> SequenceSamplingState {
        let mut s = make_state();
        for t in [40, 41, 40] {
            s.record_token(t, MAX_RECENT);
        }
        s
    }

    /// Row `p` of a block whose committed token is followed by `drafts`.
    pub(in super::super) fn block_specs(drafts: &[u32], rows: usize) -> Vec<RowSpec<'_>> {
        (0..rows)
            .map(|p| RowSpec {
                seq: 0,
                logits_row: p,
                draft: drafts.get(p).copied(),
                ahead: &drafts[..p],
            })
            .collect()
    }

    /// **One dispatch over a whole block commits exactly what scoring it a
    /// position at a time does.** The reference walks five rows one dispatch
    /// each, every row committed before the next is scored; its picks become
    /// the block's drafts. Scored together, against the history each position
    /// would be reached with, the same rows commit the same tokens and leave
    /// the same state — counts, window, RNG — with every penalty that reads
    /// history on and sampling at temperature 0.8. Scoring alone leaves the
    /// state untouched.
    #[test]
    pub(in super::super) fn a_block_scored_in_one_dispatch_commits_what_a_walk_does() {
        const ROWS: usize = 5;
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let block = walk_logits(ROWS).to_device(&device).expect("logits");
            for seed in 0..12 {
                let config = penalised(seed);
                let mut walked = walked_state();
                let mut picks = Vec::new();
                for p in 0..ROWS {
                    let row = block.narrow(0, p, 1).expect("row");
                    picks.push(
                        sampler
                            .sample_batch(&row, &mut [&mut walked], &[&config])
                            .expect("sample")[0],
                    );
                }

                let mut batched = walked_state();
                let before = (batched.row_key(), batched.token_counts.clone());
                let drafts = &picks[..ROWS - 1];
                let specs = block_specs(drafts, ROWS);
                let scored = sampler
                    .score_rows(
                        &block,
                        &mut [&mut batched],
                        &[&config],
                        &specs,
                        Some(TypicalAcceptance::MEDUSA),
                    )
                    .expect("score");
                assert_eq!(
                    (batched.row_key(), batched.token_counts.clone()),
                    before,
                    "{device:?} seed {seed}: scoring moved the state"
                );
                let committed: Vec<u32> = scored
                    .into_iter()
                    .enumerate()
                    .map(|(p, s)| {
                        sampler
                            .commit_row(&block, p, &mut batched, &config, s.expect("reached"))
                            .expect("commit")
                    })
                    .collect();
                assert_eq!(committed, picks, "{device:?} seed {seed}");
                assert_eq!(
                    batched.row_key(),
                    walked.row_key(),
                    "{device:?} seed {seed}"
                );
                assert_eq!(batched.token_counts, walked.token_counts);
                assert_eq!(batched.recent_tokens, walked.recent_tokens);
            }
        }
    }

    /// A rejected draft ends the walk where scoring a position at a time ends
    /// it: the row that tests it commits the model's own pick, and the state is
    /// the one that pick leaves.
    #[test]
    pub(in super::super) fn a_rejected_draft_stops_the_block_where_the_walk_stops() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let block = walk_logits(3).to_device(&device).expect("logits");
            let config = penalised(7);
            let mut walked = walked_state();
            let mut picks = Vec::new();
            for p in 0..2 {
                let row = block.narrow(0, p, 1).expect("row");
                picks.push(
                    sampler
                        .sample_batch(&row, &mut [&mut walked], &[&config])
                        .expect("sample")[0],
                );
            }
            // Token 9 sits at -20 on every row: never sampled, never accepted.
            let drafts = [picks[0], 9];
            let mut batched = walked_state();
            let specs = block_specs(&drafts, 3);
            let scored = sampler
                .score_rows(
                    &block,
                    &mut [&mut batched],
                    &[&config],
                    &specs,
                    Some(TypicalAcceptance::MEDUSA),
                )
                .expect("score");
            let mut committed = Vec::new();
            for (p, s) in scored.into_iter().enumerate().take(2) {
                committed.push(
                    sampler
                        .commit_row(&block, p, &mut batched, &config, s.expect("reached"))
                        .expect("commit"),
                );
            }
            assert_eq!(committed, picks, "{device:?}");
            assert_ne!(committed[1], 9, "{device:?}: the walk stops here");
            assert_eq!(batched.row_key(), walked.row_key(), "{device:?}");
        }
    }

    /// A position whose earlier draft a resolution would replace cannot be
    /// reached, and scores as `None`. With the hard EOS three tokens past the
    /// state's length, the third draft would be committed as EOS, so the rows
    /// after it are unreachable.
    #[test]
    pub(in super::super) fn a_position_behind_a_forced_token_scores_unreachable() {
        let sampler = make_sampler();
        let mut config = penalised(3);
        let mut state = walked_state();
        config.forced_eos_after = state.current_len + 2;
        let drafts = [40, 41, 42, 43];
        let block = walk_logits(5);
        let scored = sampler
            .score_rows(
                &block,
                &mut [&mut state],
                &[&config],
                &block_specs(&drafts, 5),
                Some(TypicalAcceptance::MEDUSA),
            )
            .expect("score");
        assert_eq!(
            scored.iter().map(Option::is_some).collect::<Vec<_>>(),
            vec![true, true, true, false, false]
        );
    }

    /// **A draft the distribution gives enough mass is committed whatever the
    /// sample.** `p(7) = 0.2` clears the 0.09 bar, so every seed commits 7 —
    /// including the ~80% whose sample lands on 5. The sample is still drawn,
    /// so the row's RNG advances exactly as a plain row's.
    #[test]
    pub(in super::super) fn a_draft_over_the_bar_is_committed_whatever_the_sample() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let logits = verify_row_logits().to_device(&device).expect("logits");
            for seed in 0..16 {
                let mut state = make_state();
                let token = verify_row(&sampler, &logits, &mut state, &sampled_config(seed), 7);
                assert_eq!(token, 7, "{device:?} seed {seed}");
                assert_eq!(state.rng_offset, 1, "{device:?} seed {seed}");
            }
        }
    }

    /// **A draft under the bar commits the sample.** Token 9 sits at logit 0,
    /// ~2e-9 of the row, so the row commits what plain sampling draws: 5 or 7.
    /// Across 32 seeds both turn up, which a correction pinned to the argmax
    /// could never produce.
    #[test]
    pub(in super::super) fn a_draft_under_the_bar_commits_the_sample() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let logits = verify_row_logits().to_device(&device).expect("logits");
            let mut seen = HashSet::new();
            for seed in 0..32 {
                let mut state = make_state();
                let token = verify_row(&sampler, &logits, &mut state, &sampled_config(seed), 9);
                assert!(token == 5 || token == 7, "{device:?} seed {seed}: {token}");
                seen.insert(token);
            }
            assert_eq!(seen.len(), 2, "{device:?}: corrections {seen:?}");
        }
    }

    /// **Greedy verification is unchanged.** At temperature zero the row is a
    /// point mass on its argmax, so a 0.2 draft is not accepted and the row
    /// commits 5.
    #[test]
    pub(in super::super) fn at_temperature_zero_only_the_argmax_is_committed() {
        for device in devices() {
            let sampler = make_sampler_on(&device, VOCAB_SIZE);
            let logits = verify_row_logits().to_device(&device).expect("logits");
            let mut state = make_state();
            let token = verify_row(&sampler, &logits, &mut state, &SamplingConfig::argmax(), 7);
            assert_eq!(token, 5, "{device:?}");
        }
    }

    /// One verify row testing `draft`, scored and committed as the accept walk
    /// does.
    pub(in super::super) fn verify_row(
        sampler: &BatchedSampler,
        logits: &Tensor,
        state: &mut SequenceSamplingState,
        config: &SamplingConfig,
        draft: u32,
    ) -> u32 {
        let spec = [RowSpec {
            seq: 0,
            logits_row: 0,
            draft: Some(draft),
            ahead: &[],
        }];
        let scored = sampler
            .score_rows(
                logits,
                &mut [&mut *state],
                &[config],
                &spec,
                Some(TypicalAcceptance::MEDUSA),
            )
            .expect("score");
        sampler
            .commit_row(logits, 0, state, config, scored[0].expect("reached"))
            .expect("commit")
    }
}
