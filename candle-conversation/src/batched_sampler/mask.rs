//! Logit masks and closers: the banned-token and suppression masks, the think-close ban, and the segment closer script that forces a call or an answer to end.

use super::*;

pub(super) fn segment_close_override(
    config: &SamplingConfig,
    state: &mut SequenceSamplingState,
) -> Option<u32> {
    if config.segment_close_token_id < 0 || !state.in_segment {
        return None;
    }
    if let Some(pos) = state.close_script_pos {
        return Some(if pos < config.segment_close_script.len() {
            state.close_script_pos = Some(pos + 1);
            config.segment_close_script[pos]
        } else {
            state.close_script_pos = None;
            config.segment_close_token_id as u32
        });
    }
    let at_line_end = state.at_line_end(config);
    if config.graceful_segment_close_after > 0
        && state.segment_len >= config.graceful_segment_close_after
        && at_line_end
    {
        return Some(config.segment_close_token_id as u32);
    }
    if config.force_segment_close_after > 0 && state.segment_len >= config.force_segment_close_after
    {
        return Some(
            if config.segment_close_script.is_empty() || at_line_end || state.close_would_continue {
                config.segment_close_token_id as u32
            } else {
                state.close_script_pos = Some(1);
                config.segment_close_script[0]
            },
        );
    }
    None
}

/// A tool call opened inside an open segment closes the segment first: the
/// sampled `<tool_call>` is committed as `</think>`.
///
/// The template closes the reasoning block before the answer, and a call is an
/// answer. Written inside the block it is reasoning to every reader downstream,
/// so the client never receives it. Measured on a Cline turn: the block opened,
/// the call was written inside it, and the reply ended with the block still
/// open. Committing the close where the model chose to act ends the block
/// there; the call follows on the next step, outside it, where the tool-call
/// grammar takes it.
pub(super) fn close_before_call(
    config: &SamplingConfig,
    state: &SequenceSamplingState,
    sampled: u32,
) -> Option<u32> {
    (config.tool_call_open_token_id >= 0
        && config.segment_close_token_id >= 0
        && state.in_segment
        && sampled == config.tool_call_open_token_id as u32)
        .then_some(config.segment_close_token_id as u32)
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

/// Apply banned tokens within a gathered allow-list logit vector: any banned
/// token that also appears in `allow` is set to `-inf` at its local position.
/// A no-op when no banned token intersects the allow-list. Also returns how
/// many allowed tokens it banned, counted in the pass that masks them.
pub(super) fn apply_banned_local(
    gathered: &Tensor,
    allow: &[u32],
    config: &SamplingConfig,
) -> candle::Result<(Tensor, usize)> {
    if config.banned_tokens.is_empty() {
        return Ok((gathered.clone(), 0));
    }
    let banned: std::collections::HashSet<u32> =
        config.banned_tokens.iter().map(|&b| b as u32).collect();
    if !allow.iter().any(|t| banned.contains(t)) {
        return Ok((gathered.clone(), 0));
    }
    let dtype = gathered.dtype();
    let dims = gathered.dims().to_vec();
    let mut v: Vec<f32> = gathered.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut banned_n = 0;
    for (local, t) in allow.iter().enumerate() {
        if banned.contains(t) {
            v[local] = f32::NEG_INFINITY;
            banned_n += 1;
        }
    }
    let masked = Tensor::from_vec(v, dims, gathered.device())?.to_dtype(dtype)?;
    Ok((masked, banned_n))
}

/// Map a [`SamplingConfig`] to candle's `Sampling` strategy.
pub(super) fn config_to_sampling(
    config: &SamplingConfig,
) -> candle_transformers::generation::Sampling {
    use candle_transformers::generation::Sampling;
    if config.temperature <= 0.0 {
        Sampling::ArgMax
    } else if config.top_k > 0 && config.top_p < 1.0 {
        Sampling::TopKThenTopP {
            k: config.top_k as usize,
            p: config.top_p as f64,
            temperature: config.temperature as f64,
        }
    } else if config.top_k > 0 {
        Sampling::TopK {
            k: config.top_k as usize,
            temperature: config.temperature as f64,
        }
    } else if config.top_p < 1.0 {
        Sampling::TopP {
            p: config.top_p as f64,
            temperature: config.temperature as f64,
        }
    } else {
        Sampling::All {
            temperature: config.temperature as f64,
        }
    }
}
