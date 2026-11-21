//! Tests for batches that mix sampled rows, constrained rows and bad rows.
//!
//! A batch that fails on one NaN row must leave every state as it found it, so a
//! clean row sampled again alone draws what it would have drawn from the start.
//! Deferred allow-list rows must land in their own rows and advance their own
//! states, with an unconstrained row between them. Free rows beside a constrained
//! row must be read from their own rows of the block, and the padded tail of a
//! row must never be produced, even when a draft names it.

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

    use super::super::tests_scoring::tests::*;

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

    /// **The padded tail of a row is never a token.** The sampler's live
    /// vocabulary ends at 50; token 60, in the padding, carries the row's
    /// largest logit by far. Neither the argmax nor a sample may produce it,
    /// and a draft naming it is not accepted.
    #[test]
    pub(in super::super) fn the_padded_tail_of_a_row_is_never_produced() {
        for device in devices() {
            let sampler = make_sampler_on(&device, 50);
            let logits = logits_from_rows(&[&[(5, 20.0), (7, 20.0 + 0.25f32.ln()), (60, 90.0)]])
                .to_device(&device)
                .expect("logits");
            let mut state = make_state();
            let greedy = sampler
                .sample_batch(&logits, &mut [&mut state], &[&SamplingConfig::argmax()])
                .expect("sample");
            assert_eq!(greedy, vec![5], "{device:?}");
            for seed in 0..16 {
                let mut state = make_state();
                let token = verify_row(&sampler, &logits, &mut state, &sampled_config(seed), 60);
                assert!(token == 5 || token == 7, "{device:?} seed {seed}: {token}");
            }
        }
    }
}
