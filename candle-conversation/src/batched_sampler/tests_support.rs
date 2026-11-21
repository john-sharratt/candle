//! Shared test fixtures: the vocabulary and token constants, sampler and state builders, logit and row helpers, and the device and walk helpers the other test files use.

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

    // ── Hard-cap closer script (segment_close_override tiers) ──────────

    /// Config with segment tracking on: close=90, graceful after 4 at a line
    /// end (token 7), a sentence end that is not a line end (token 8, `.`),
    /// hard cap at 8, closer phrase "A B C" (the sampler appends the close
    /// token 90 itself).
    pub(in super::super) fn closer_config() -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.segment_close_token_id = 90;
        c.segment_open_token_id = 89;
        c.graceful_segment_close_after = 4;
        c.force_segment_close_after = 8;
        c.sentence_end_token_ids = vec![8];
        c.line_end_token_ids = Arc::from([7]);
        c.segment_close_script = vec![100, 101, 102];
        c
    }

    /// A state `n` tokens into an open segment, last token `last`.
    pub(in super::super) fn in_segment_state(n: i32, last: u32) -> SequenceSamplingState {
        let mut s = make_state();
        s.record_token(last, MAX_RECENT);
        s.in_segment = true;
        s.segment_len = n;
        s
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

    /// One logits row with `p(5) = 0.8` and `p(7) = 0.2`; every other token
    /// sits at logit 0, ~2e-9 each.
    pub(in super::super) fn verify_row_logits() -> Tensor {
        logits_from_rows(&[&[(5, 20.0), (7, 20.0 + 0.25f32.ln())]])
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
}
