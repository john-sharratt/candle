//! Tests for the allow-list path, on the CUDA kernel where a card is present.
//!
//! They check that an argmax row reads back one word and a sampled row one per
//! allowed token, that the host half maps the result back to the right token,
//! and that each bitmap row in a shared dispatch takes its own allowed best. A
//! sampled allow-list row must draw only inside its list, even when a banned
//! position carries the highest logit. One test times a bitmap against the
//! allow-list it replaces and prints the figures.

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
