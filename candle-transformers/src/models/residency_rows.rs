//! Which of a wave's rows the expert cache scores as decode.
//!
//! The routed experts' residency scoring weights a decode row's routing far
//! above a prompt row's: a decode sequence routes much the same experts from
//! one step to the next, a prompt sweeps the table roughly once
//! (`expert_lre::cache`). Two kinds of row outside the decode rows route like
//! decode, and are scored as it ([`DecodeRows`]):
//!
//! - **A verify segment** is a sequence's next tokens decoded several at once,
//!   laid out like a prompt. Scored as prompt rows, every expert a verify step
//!   missed arrived as the zone's cheapest victim and was evicted by the next
//!   refill: on Qwen3.8-Flash-Next (RTX 3090), whose decode runs entirely as
//!   verify waves, that was most of decode's misses.
//! - **A prompt's last token** is the one its decode continues from, so its
//!   experts are the best early guess at the first decode step's. Scored as a
//!   prompt row, the first decode step after a prompt found its experts evicted
//!   by the prompt itself — 357 misses against ~130 for a later step.
//!
//! A wave's rows are its decode rows, then one segment per prefill sequence in
//! wave order — the layout every hybrid wave forward builds.

use candle::quantized::decode_rows::DecodeRows;

/// The decode rows `[0, n_decode)`, every verify segment, and the last row of
/// every prompt segment. `pre` is each prefill segment's sequence and row
/// count, in wave order.
pub(crate) fn residency_decode_rows(
    n_decode: usize,
    pre: impl IntoIterator<Item = (usize, usize)>,
    is_verify: impl Fn(usize) -> bool,
) -> DecodeRows {
    let mut rows = DecodeRows::prefix(n_decode);
    let mut at = n_decode;
    for (seq, n) in pre {
        if is_verify(seq) {
            rows.push(at, at + n);
        } else if n > 0 {
            rows.push(at + n - 1, at + n);
        }
        at += n;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_segments_after_the_decode_rows_join_them() {
        let verify = |s: usize| s == 7 || s == 9;
        let r = residency_decode_rows(2, [(7, 5), (9, 4)], verify);
        assert_eq!(r.ranges(), (&[0u32][..], &[11u32][..]));
    }

    #[test]
    fn a_prompt_contributes_its_last_row_and_a_later_verify_segment_its_own() {
        let verify = |s: usize| s == 7 || s == 9;
        let r = residency_decode_rows(1, [(7, 5), (3, 40), (9, 4)], verify);
        // Decode 0, verify 1..6, prompt 6..46 → its last row 45, verify 46..50.
        assert_eq!(r.ranges(), (&[0u32, 45][..], &[6u32, 50][..]));
    }

    #[test]
    fn a_prompt_only_wave_scores_each_prompt_last_row() {
        let r = residency_decode_rows(0, [(3, 40), (4, 25)], |_| false);
        assert_eq!(r.ranges(), (&[39u32, 64][..], &[40u32, 65][..]));
        assert_eq!(residency_decode_rows(0, [], |_| true).ranges().0.len(), 0);
    }
}
