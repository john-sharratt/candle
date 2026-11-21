//! Tests for the per-row allow-list and banned-token masking.
//!
//! A row's allow-list is its own: a stenciled row never leaks its mask into a
//! free row beside it, and two stenciled rows with disjoint lists each mask the
//! shared global best. A ban is also the row's own inside a shared launch, in
//! either order of the rows. Stochastic draws over a gathered allow-list never
//! leave the list, and a single allowed token is forced.

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
}
