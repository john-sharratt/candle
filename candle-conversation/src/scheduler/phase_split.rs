//! How one co-batched forward's wall-clock divides between the classes riding it.
//!
//! Decode, prefill and section rows share a single sweep when they are
//! co-batched, so the sweep's time has to be shared out rather than charged to
//! each — the phase timeline is wall-clock, and its bands must sum to the
//! window. The decode rows keep what a decode-only step takes; the creep beside
//! them takes the rest of the step, divided between its classes by rows
//! ([`cobatched_step_shares`]).

use candle::Tensor;

/// Rows the decode side put into a co-batched forward: one per plain decode
/// sequence, a whole block per verifying one. Each input is `[1, rows]`.
pub(super) fn cobatched_decode_rows(decode_inputs: &[Tensor], verify_inputs: &[Tensor]) -> usize {
    decode_inputs
        .iter()
        .chain(verify_inputs)
        .map(|t| t.dims().get(1).copied().unwrap_or(1))
        .sum()
}

/// The prefill and section shares, in microseconds, of a decode step of
/// `step_us` that carried `prefill_rows` and `section_rows` of creep beside
/// `decode_rows` decode rows. The decode share is the remainder, and stays
/// with the decode quantum the step ran in.
///
/// **The step's wall-clock is the measure, not the forward's launch.** A step
/// ends with its logits read back, so it holds the device's time; the host's
/// time around the creep's launch does not, and split by that, a 3,072-row
/// creep's device time surfaced at the readback inside decode's timer — one
/// decode row of three forwards drew 3.1 s of decode (BlackWorld, 2026-10-27).
/// The creep's part is what the step took past a decode-only step
/// (`decode_only_us`, which the scheduler learns from them), the cost it
/// prices a creep's debt by too, shared between prefill and section by their
/// rows. Before a decode-only step has been measured the rows share the whole
/// step; a step with no decode rows is the creep's entirely.
pub(super) fn cobatched_step_shares(
    step_us: u64,
    decode_only_us: Option<u64>,
    decode_rows: usize,
    prefill_rows: usize,
    section_rows: usize,
) -> (u64, u64) {
    let creep_rows = prefill_rows + section_rows;
    if creep_rows == 0 {
        return (0, 0);
    }
    let creep_us = match (decode_rows, decode_only_us) {
        (0, _) => step_us,
        (_, Some(alone)) => step_us - alone.min(step_us),
        (_, None) => {
            (step_us as u128 * creep_rows as u128 / (creep_rows + decode_rows) as u128) as u64
        }
    };
    let prefill = (creep_us as u128 * prefill_rows as u128 / creep_rows as u128) as u64;
    (prefill, creep_us - prefill)
}

/// The share of `tokens` a creep forward over layers `[start, end)` of `n`
/// prefilled, in whole-sweep tokens.
///
/// A creep rides a window of the layers per wave and is counted once per
/// window, so counting its whole token block each time multiplied it by the
/// number of windows while the time stayed one window's: a 1,885-token creep
/// read 54,000 tok/s off a 35 ms slice of a sweep the card takes ~2 s over.
/// Taken as the difference of two floors, the windows of a sweep sum to
/// exactly `tokens`.
pub(super) fn window_tokens(tokens: usize, start: usize, end: usize, n: usize) -> usize {
    if n == 0 || end <= start {
        return 0;
    }
    let upto = |layer: usize| (tokens as u128 * layer.min(n) as u128 / n as u128) as usize;
    upto(end) - upto(start)
}

/// What a forward over layers `[start, end)` of `n` that took `us` would take
/// over all `n` — the unit the planner's forward model is written in.
pub(super) fn full_sweep_us(us: u64, start: usize, end: usize, n: usize) -> u64 {
    let span = end.min(n).saturating_sub(start);
    if span == 0 {
        return 0;
    }
    (us as u128 * n as u128 / span as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    /// The measured creep: 1,885 tokens over a 48-layer model in windows of 3.
    /// Each window is a sixteenth of the tokens, and the sweep sums to exactly
    /// the block — never once per window.
    #[test]
    fn a_creeps_windows_sum_to_its_tokens_once() {
        assert_eq!(window_tokens(1_885, 0, 3, 48), 117);
        assert_eq!(window_tokens(1_885, 3, 6, 48), 118);
        let swept: usize = (0..16)
            .map(|w| window_tokens(1_885, w * 3, w * 3 + 3, 48))
            .sum();
        assert_eq!(swept, 1_885);
        assert_eq!(window_tokens(1_885, 0, 48, 48), 1_885);
        assert_eq!(window_tokens(1_885, 5, 5, 48), 0);
        assert_eq!(window_tokens(7, 0, 1, 0), 0);
    }

    /// A 35 ms window of 3 layers of 48 is a 560 ms sweep.
    #[test]
    fn a_windows_time_scales_to_the_whole_sweep() {
        assert_eq!(full_sweep_us(35_000, 0, 3, 48), 560_000);
        assert_eq!(full_sweep_us(35_000, 0, 48, 48), 35_000);
        assert_eq!(full_sweep_us(35_000, 12, 12, 48), 0);
    }

    /// **The creep takes what the step took past a decode-only step**: 3
    /// decode rows beside an 8,192-token prefill in a 4,645 ms step, a
    /// decode-only step being 40 ms, leave decode its 40 ms.
    #[test]
    fn the_creep_takes_the_step_past_a_decode_only_one() {
        assert_eq!(
            cobatched_step_shares(4_645_000, Some(40_000), 3, 8_192, 0),
            (4_605_000, 0)
        );
    }

    /// Before a decode-only step has been measured, the rows share the step.
    #[test]
    fn unmeasured_the_rows_share_the_step() {
        assert_eq!(
            cobatched_step_shares(4_645_000, None, 3, 8_192, 0),
            (4_643_299, 0)
        );
    }

    /// A step no longer than a decode-only one leaves the creep nothing.
    #[test]
    fn a_step_within_a_decode_only_one_is_decodes() {
        assert_eq!(
            cobatched_step_shares(30_000, Some(40_000), 3, 100, 0),
            (0, 0)
        );
    }

    /// The creep's part is shared between prefill and section by their rows,
    /// and the two never exceed it.
    #[test]
    fn prefill_and_section_share_the_creeps_part_by_rows() {
        assert_eq!(
            cobatched_step_shares(1_000_000, Some(400_000), 10, 30, 60),
            (200_000, 400_000)
        );
        assert_eq!(
            cobatched_step_shares(1_001, Some(0), 1, 5, 5),
            (500, 501),
            "floored, the rest to section"
        );
    }

    /// A decode-less wave driven from INSIDE the decode quantum (every decode
    /// excluded that wave) is all prefill's and section's — decode held it.
    #[test]
    fn a_decode_less_wave_inside_the_decode_quantum_moves_whole() {
        assert_eq!(
            cobatched_step_shares(8_000, Some(5_000), 0, 600, 200),
            (6_000, 2_000)
        );
    }

    /// Two plain decode rows and a five-row verify block are seven rows.
    #[test]
    fn decode_rows_count_a_verify_block_whole() {
        let dev = Device::Cpu;
        let one = Tensor::zeros((1, 1), DType::U32, &dev).unwrap();
        let block = Tensor::zeros((1, 5), DType::U32, &dev).unwrap();
        assert_eq!(
            cobatched_decode_rows(&[one.clone(), one], std::slice::from_ref(&block)),
            7
        );
        assert_eq!(cobatched_decode_rows(&[], &[]), 0);
    }

    #[test]
    fn a_step_with_only_decode_rows_or_none_shares_nothing() {
        assert_eq!(cobatched_step_shares(9_999, Some(1), 8, 0, 0), (0, 0));
        assert_eq!(cobatched_step_shares(9_999, None, 0, 0, 0), (0, 0));
    }
}
