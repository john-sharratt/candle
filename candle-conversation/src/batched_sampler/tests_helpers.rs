//! Shared fixtures for the sampler tests: the vocabulary constants, the sampler and state builders, and the logits builders.
//!
//! `make_sampler` and `make_sampler_on` build a `BatchedSampler`, and `make_state` builds a `SequenceSamplingState`.
//! `logits_from_rows` builds a `[batch, VOCAB]` logits tensor from per-row spikes, and `gather` collects the kernel inputs for a set of rows.
//! `VOCAB_SIZE`, `MAX_RECENT` and `EOS_TOKEN` are the fixture constants, and `window_16`, `sampled_config` and `devices` build the shared configs and devices.

use super::*;

// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
pub(super) mod tests {
    // Several tests pin a tuning constant inside a range. The assertion IS the
    // contract — it fires the moment someone retunes the constant out of band —
    // even though it is constant-valued at any given commit.
    #![allow(clippy::assertions_on_constants)]

    use std::sync::Arc;

    use super::*;

    use crate::config::SamplingConfig;

    pub(in super::super) const VOCAB_SIZE: usize = 100;

    pub(in super::super) const MAX_RECENT: usize = 32;

    pub(in super::super) const EOS_TOKEN: u32 = 2;

    pub(in super::super) fn make_sampler() -> BatchedSampler {
        BatchedSampler::new(
            candle::Device::Cpu,
            VOCAB_SIZE,
            VOCAB_SIZE,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        )
    }

    pub(in super::super) fn make_state() -> SequenceSamplingState {
        SequenceSamplingState::new(VOCAB_SIZE, MAX_RECENT)
    }

    /// A config whose repeat window is 16 tokens.
    pub(in super::super) fn window_16() -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.repeat_last_n = 16;
        c
    }

    /// The kernel inputs `states` gather into at `config`, a row each.
    pub(in super::super) fn gather<'a>(
        sampler: &BatchedSampler,
        states: &[&SequenceSamplingState],
        config: &'a SamplingConfig,
        cross_turn: bool,
    ) -> KernelRows<'a> {
        let mut rows = KernelRows::new(sampler.vocab_size, sampler.max_recent_len, cross_turn);
        for (r, state) in states.iter().enumerate() {
            let spec = RowSpec {
                seq: r,
                logits_row: r,
                draft: None,
                ahead: &[],
            };
            rows.push(r, state, config, &spec, None, state.row_key());
        }
        rows
    }

    // ── Per-row stencil / banned-token masking ─────────────────────────

    /// Build a `[batch, VOCAB]` logits tensor from per-row spikes.
    pub(in super::super) fn logits_from_rows(rows: &[&[(usize, f32)]]) -> Tensor {
        let mut data = vec![0.0f32; rows.len() * VOCAB_SIZE];
        for (r, spikes) in rows.iter().enumerate() {
            for &(tok, val) in *spikes {
                data[r * VOCAB_SIZE + tok] = val;
            }
        }
        Tensor::from_vec(data, (rows.len(), VOCAB_SIZE), &Device::Cpu).expect("logits")
    }

    /// `tokens` as an allowed-token bitmap, as a wide set carries one.
    pub(in super::super) fn bits_of(tokens: &[usize]) -> Arc<[u32]> {
        let mut words = vec![0u32; VOCAB_SIZE.div_ceil(32)];
        for &t in tokens {
            words[t / 32] |= 1 << (t % 32);
        }
        words.into()
    }

    pub(in super::super) fn sampled_config(seed: u64) -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.temperature = 1.0;
        c.top_k = 0;
        c.top_p = 1.0;
        c.seed = seed;
        c
    }

    pub(in super::super) fn devices() -> Vec<Device> {
        let mut devices = vec![Device::Cpu];
        devices.extend(Device::new_cuda(0).ok());
        devices
    }

    pub(in super::super) fn make_sampler_on(device: &Device, live_vocab: usize) -> BatchedSampler {
        BatchedSampler::new(
            device.clone(),
            VOCAB_SIZE,
            live_vocab,
            MAX_RECENT,
            vec![EOS_TOKEN].into(),
            None,
        )
    }
}
