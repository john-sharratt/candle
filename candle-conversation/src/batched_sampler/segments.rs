//! The segment-close override, the report of a degenerate logits row, and the
//! close that precedes a tool call.
//!
//! A reasoning segment is closed by the override when its budget runs out: a
//! closer script in flight plays to the end, a graceful cap closes at a line end,
//! and the hard cap starts the configured closer script, falling back to the bare
//! close where no script fits. When the override answers, the token it gives is
//! authoritative for the step. A row that looks degenerate is described once, in
//! the log, by whether its entries are non-finite or all equal, since those have
//! different causes. A tool call opened inside a segment closes the segment
//! first, so the call is written outside the reasoning block.

use super::*;

/// Summarise one `[batch, vocab]` logits row for the degenerate-decode report.
///
/// Names what the row actually is — how many entries are non-finite, its
/// min/max, and whether every entry is identical — so the fault is diagnosable
/// from the log alone. An all-equal row means the forward wrote nothing
/// (argmax lands on index 0, which is `!` in the Qwen vocabularies); a
/// non-finite row means the arithmetic blew up. Those have different causes and
/// the guard cannot distinguish them without looking.
pub(super) fn describe_logit_row(logits2d: &Tensor, row: usize) -> candle::Result<String> {
    let values = logits2d.i(row)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = values.len();
    let non_finite = values.iter().filter(|v| !v.is_finite()).count();
    let finite_min = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let finite_max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let first = values.first().copied().unwrap_or(0.0);
    let all_equal = values.iter().all(|v| *v == first);
    // The argmax names what the row actually wanted. A sampled token that is
    // not this one, at a logit far below it, is the sampler disagreeing with
    // the distribution rather than the forward producing a bad row — and those
    // two faults are indistinguishable from the emitted text alone.
    let argmax = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(format!(
        "logits row: vocab={n} non_finite={non_finite} finite_min={finite_min:.6} \
         finite_max={finite_max:.6} first={first:.6} argmax={argmax} all_equal={all_equal}"
    ))
}

/// Segment-close override for one sampled token, applied after sampling on
/// both the CPU and GPU paths.
///
/// Three tiers:
/// - a closer script in flight overrides everything: it plays the configured
///   phrase to its end, then emits the segment-close token itself (the close
///   is appended by this function, not stored in the script, so a played
///   script can never fail to close the segment);
/// - the GRACEFUL cap closes with the bare token at the end of a line — no
///   rescue needed;
/// - the HARD cap starts the configured closer script (a canned
///   self-interruption that turns the mid-sentence amputation into sensible
///   prose and primes the answer with an explicit commitment). It falls back
///   to the bare close token when no script is configured, when the sentence
///   happens to already be complete, or when the steering span would drop the
///   close and carry on decoding — a terminal closing statement does not belong
///   in the middle of a span that continues.
///
/// When this returns `Some`, the token is authoritative for the step: the EOS
/// failsafes must not replace it (they fire on a later step, once the segment
/// is closed and `in_segment` is false).
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
