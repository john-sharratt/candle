//! How one co-batched forward's wall-clock divides between the classes riding it.
//!
//! Decode, prefill and section rows share a single sweep when they are
//! co-batched, so the sweep's time has to be shared out rather than charged to
//! each — the phase timeline is wall-clock, and its bands must sum to the
//! window. Rows are the measure: the forward's cost is its rows' projections and
//! attention, and every row of every class goes through the same layers.

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

/// The prefill and section shares, in the forward's own unit, of a forward of
/// `fwd` that carried `decode_rows`, `prefill_rows` and `section_rows`. The
/// decode share is the remainder, and stays with the decode quantum the forward
/// ran in. A forward with no rows has nothing to share.
///
/// Only a forward that ran **inside the decode quantum** is shared out — the
/// caller says so (`in_decode_quantum`), because only there is the whole sweep
/// in decode's timer to be moved. The same wave step also runs from the
/// prefill quantum with no decode rows, its time already prefill's; sharing that
/// one too counted it twice — a 5.2 s window drew a 9.4 s prefill band.
pub(super) fn cobatched_shares(
    in_decode_quantum: bool,
    decode_rows: usize,
    prefill_rows: usize,
    section_rows: usize,
    fwd: u64,
) -> (u64, u64) {
    let total = (decode_rows + prefill_rows + section_rows) as u128;
    if !in_decode_quantum || total == 0 {
        return (0, 0);
    }
    let share = |rows: usize| (fwd as u128 * rows as u128 / total) as u64;
    (share(prefill_rows), share(section_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    /// The measured case: 3 decode rows beside an 8,192-token prefill in a
    /// 4,645 ms sweep. Prefill takes all but ~1.7 ms of it.
    #[test]
    fn a_long_prefill_takes_its_sweep_from_a_short_decode() {
        assert_eq!(
            cobatched_shares(true, 3, 8_192, 0, 4_645_000),
            (4_643_299, 0)
        );
    }

    #[test]
    fn shares_follow_rows_and_never_exceed_the_forward() {
        let (pf, sc) = cobatched_shares(true, 10, 30, 60, 1_000);
        assert_eq!((pf, sc), (300, 600));
        assert!(pf + sc <= 1_000);
        assert_eq!(
            cobatched_shares(true, 0, 5, 5, 1_001),
            (500, 500),
            "floored, never over"
        );
    }

    /// The measured window: a decode-less wave of 8,190 prefill rows ran from
    /// the prefill quantum, whose timer already holds it. Nothing moves.
    #[test]
    fn a_forward_outside_the_decode_quantum_shares_nothing() {
        assert_eq!(cobatched_shares(false, 0, 8_190, 0, 4_700_000), (0, 0));
        assert_eq!(cobatched_shares(false, 3, 8_190, 0, 4_700_000), (0, 0));
    }

    /// A decode-less wave driven from INSIDE the decode quantum (every decode
    /// excluded that wave) is all prefill's and section's — decode held it.
    #[test]
    fn a_decode_less_wave_inside_the_decode_quantum_moves_whole() {
        assert_eq!(cobatched_shares(true, 0, 600, 200, 8_000), (6_000, 2_000));
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
    fn a_forward_with_only_decode_rows_or_none_shares_nothing() {
        assert_eq!(cobatched_shares(true, 8, 0, 0, 9_999), (0, 0));
        assert_eq!(cobatched_shares(true, 0, 0, 0, 9_999), (0, 0));
    }
}
