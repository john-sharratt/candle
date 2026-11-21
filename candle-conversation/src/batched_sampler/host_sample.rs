//! The host sampler for unconstrained rows, with the logit masks it applies before it picks a token.
//!
//! `sample_row_cpu` picks a token for one full-vocabulary row on the host, against that row's logits and sequence state.
//! `apply_banned` sets a row's banned logits to negative infinity, and `apply_suppression` subtracts the penalty from segment-suppressed tokens.
//! `think_close_ban_active` says whether the structural think-close ban applies to a row right now.

use super::*;

impl BatchedSampler {
    /// The host sampler's pick for one unconstrained (full-vocab) row, against
    /// `state` as it stands — candle's built-in sampling, for a sampler that
    /// is not on a CUDA device. Committing it is [`Self::commit_row`]'s.
    pub(super) fn sample_row_cpu(
        &self,
        logits: &Tensor,
        row: &RowSpec<'_>,
        state: &SequenceSamplingState,
        config: &SamplingConfig,
        typical: Option<TypicalAcceptance>,
    ) -> candle::Result<u32> {
        // This sequence's row of the block.
        let seq_logits = logits.i(row.logits_row)?;

        // Apply this row's banned tokens (a small deny-list, e.g. a few EOS
        // ids) by setting just those values to `-inf`.  Cheap on CPU — only the
        // banned values change (apply_banned copies the row to host F32 to do
        // it); a no-op when the list is empty.
        let seq_logits = apply_banned(&seq_logits, config)?;

        // Structural: ban the think-close token while outside a think block —
        // there is nothing for it to close there, and a stray one derails the
        // turn (see `think_close_ban_active`).
        let seq_logits = if think_close_ban_active(config, state) {
            let dtype = seq_logits.dtype();
            let dims = seq_logits.dims().to_vec();
            let mut v: Vec<f32> = seq_logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            ban(&mut v, config.segment_close_token_id as u32);
            Tensor::from_vec(v, dims, seq_logits.device())?.to_dtype(dtype)?
        } else {
            seq_logits
        };

        // Token suppression: while inside a segment, subtract the per-turn
        // penalty from each suppress-token logit. Mirrors the kernel's
        // in-segment gate, so tokens outside the segment are never touched.
        let seq_logits = if state.in_segment
            && config.segment_suppress_penalty != 0.0
            && !config.segment_suppress_tokens.is_empty()
        {
            apply_suppression(&seq_logits, config)?
        } else {
            seq_logits
        };
        // The checkpoint's padded tail is not a token.
        let seq_logits = seq_logits.narrow(0, 0, self.live_vocab)?;

        // In-segment steering: while this sequence is inside a segment,
        // sample a touch hotter (temperature + segment_temp_boost).
        // Mirrors the kernel's per-seq gate so tokens outside the segment
        // stay at the base temperature.  DRY is GPU-only — the CPU
        // LogitsProcessor has no DRY path, so there is nothing to gate here for it.
        let temperature = if state.in_segment {
            config.temperature + config.segment_temp_boost
        } else {
            config.temperature
        };
        let sampling = if state.in_segment && config.segment_temp_boost != 0.0 {
            let mut boosted = config.clone();
            boosted.temperature += config.segment_temp_boost;
            config_to_sampling(&boosted)
        } else {
            config_to_sampling(config)
        };
        let seed = config.seed.wrapping_add(state.rng_offset);
        let mut processor = LogitsProcessor::from_sampling(seed, sampling);
        let mut sampled = processor.sample(&seq_logits).map_err(|e| {
            // Read only on a failure: which of the two ways a softmax goes
            // NaN this row took — non-finite logits, or everything at −inf.
            let (nan, inf, neg_inf) = seq_logits
                .to_dtype(DType::F32)
                .and_then(|l| l.to_vec1::<f32>())
                .map(|v| {
                    v.iter().fold((0, 0, 0), |(n, p, m), x| {
                        (
                            n + usize::from(x.is_nan()),
                            p + usize::from(*x == f32::INFINITY),
                            m + usize::from(*x == f32::NEG_INFINITY),
                        )
                    })
                })
                .unwrap_or((0, 0, 0));
            candle::Error::Msg(format!(
                "{e} — full row of {}: {nan} NaN, {inf} +inf, {neg_inf} −inf logits; \
                 {} banned",
                self.live_vocab,
                config.banned_tokens.len()
            ))
        })?;
        // A verify row's draft, accepted on the typical-acceptance rule over
        // the same top-k/nucleus distribution the kernel measures it on.
        if let (Some(draft), Some(typical)) = (row.draft, typical) {
            if temperature > 0.0 {
                let values: Vec<f32> = seq_logits.to_dtype(DType::F32)?.to_vec1()?;
                if typical_accepts_row(
                    &values,
                    temperature,
                    config.top_k,
                    config.top_p,
                    draft,
                    typical,
                ) {
                    sampled = draft;
                }
            }
        }
        Ok(sampled)
    }
}

/// Set this row's banned (deny-list) logits to `-inf`.  Modifies only the few
/// banned *values* (on an F32 host copy of the row), and is a no-op when the list
/// is empty, so unconstrained rows keep their exact logits.  Preserves the input
/// dtype.
pub(super) fn apply_banned(logits: &Tensor, config: &SamplingConfig) -> candle::Result<Tensor> {
    if config.banned_tokens.is_empty() && config.allowed_bits.is_none() {
        return Ok(logits.clone());
    }
    let dtype = logits.dtype();
    let dims = logits.dims().to_vec();
    let mut v: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    for &b in &config.banned_tokens {
        ban(&mut v, b as u32);
    }
    if let Some(bits) = &config.allowed_bits {
        for (t, logit) in v.iter_mut().enumerate() {
            let word = bits.get(t / 32).copied().unwrap_or(0);
            if word >> (t % 32) & 1 == 0 {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(v, dims, logits.device())?.to_dtype(dtype)
}

/// Whether the structural think-close ban applies to this row right now.
///
/// **A `</think>` outside a think block is never valid output.** It is a
/// structural token: inside a block the steering owns it (a sampled close is
/// intercepted and replaced by the stencil's canonical prefill), and outside a
/// block there is nothing for it to close. The window this guards is real and
/// was hit twice on the same turn shape: the stencil finishes its walk at the
/// prefilled close, steering ends, and the very next free-decoded tokens have
/// no owner for the close id — the model emitted a second `</think>`, and with
/// the segment state already cleared every think-scoped control (DRY,
/// suppression) was off, so nothing resisted it. The turn's answer then
/// derailed off the malformed transcript.
///
/// Gated on the sampler's own `in_segment` — deliberately not a new flag. This
/// state has desynced from its twins twice before, and the ban is shaped to
/// fail safe against both directions: wrongly *outside* (ban active in a think
/// block) cannot strand the block, because closing it is the stencil's job and
/// the hard-cap closer script forces the token rather than sampling it;
/// wrongly *inside* (ban off after a close) is exactly today's behaviour.
///
/// # The cost of the ban, and why it is still the right trade
///
/// The "cannot strand" argument above holds only for a block opened with the
/// open *token*. A model that spells `<think>` out as plain text never arms
/// `in_segment`, so on an unsteered turn — no stencil, and the hard cap keyed
/// off a flag that is clear — nothing can end it. Measured on a thinking-off
/// ingest: rather than stay stuck, the model closed with a spelling that is not
/// the banned id, and 2 of 22 summaries ended in a bare `</thinking>`.
///
/// Lifting the ban for suppressed turns was tried and is **wrong**: any
/// `</think>` sets `think_close_at` and cuts an index page at a reasoning
/// boundary (`scheduler::mod`), so a stray close outside a block would carve a
/// spurious page into the turn's K/V. A mis-cut index page is the worse of the
/// two, so the ban stays.
///
/// What the ban leaves behind is only partly cleaned up, and knowingly so.
/// `think_strip::strip_trailing_orphan_close` removes a leaked closer when it is
/// the LAST thing in a turn that opened no block — the shape actually measured.
/// A block the model spelled out in plain text mid-answer is NOT recovered: the
/// text is paired with its K/V span by offset, and editing the middle of it
/// would shift that pairing for every consumer that maps text onto tokens. The
/// leaked run is stored verbatim in that case.
///
pub(super) fn think_close_ban_active(
    config: &SamplingConfig,
    state: &SequenceSamplingState,
) -> bool {
    config.segment_close_token_id >= 0 && !state.in_segment
}

/// Subtract the suppression penalty from each `segment_suppress_tokens` logit.
/// The CPU mirror of the kernel's in-segment ceiling lever; the caller has
/// already confirmed the sequence is inside a segment and the penalty is
/// nonzero, so this is applied unconditionally here.
pub(super) fn apply_suppression(
    logits: &Tensor,
    config: &SamplingConfig,
) -> candle::Result<Tensor> {
    let dtype = logits.dtype();
    let dims = logits.dims().to_vec();
    let vocab = logits.elem_count();
    let mut v: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    for &t in &config.segment_suppress_tokens {
        if t >= 0 && (t as usize) < vocab {
            v[t as usize] -= config.segment_suppress_penalty;
        }
    }
    Tensor::from_vec(v, dims, logits.device())?.to_dtype(dtype)
}
