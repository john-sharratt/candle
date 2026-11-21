//! One sampling dispatch's small device arguments, packed into a single upload.
//!
//! The sampling kernel takes a dozen small per-dispatch arrays — the rows'
//! lengths, the recent-token window, the ban and suppression lists, the per-row
//! dials, the RNG offsets and the output slots. Uploading each one separately
//! cost an allocation and a pageable copy apiece, a dozen per accept-walk
//! position. Here they are laid end to end in one word buffer that is uploaded
//! once; each kernel argument is the buffer's base address plus its offset, and
//! the arrays the kernel writes — RNG offsets, outputs, then each row's
//! non-finite verdict — sit together at the end so one copy reads them back.

/// A word buffer and the offsets of the arguments laid into it.
#[derive(Debug, Default)]
pub(crate) struct ArgPack {
    words: Vec<u32>,
}

impl ArgPack {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `i32` values and return their word offset. An empty list takes a
    /// single `-1` word, so every argument has an address; the kernel is told
    /// such a list's length is zero and never reads it.
    pub fn push_i32(&mut self, values: &[i32]) -> usize {
        let at = self.words.len();
        if values.is_empty() {
            self.words.push(u32::MAX);
        } else {
            self.words.extend(values.iter().map(|&v| v as u32));
        }
        at
    }

    /// Append `f32` values as their bit patterns and return their word offset.
    /// An empty list takes a single zero word.
    pub fn push_f32(&mut self, values: &[f32]) -> usize {
        let at = self.words.len();
        if values.is_empty() {
            self.words.push(0);
        } else {
            self.words.extend(values.iter().map(|v| v.to_bits()));
        }
        at
    }

    /// Append `u64` values on an 8-byte boundary, low word first, and return
    /// their word offset. The buffer's base is allocation-aligned, so an even
    /// word offset is an aligned `u64` address.
    pub fn push_u64(&mut self, values: &[u64]) -> usize {
        if self.words.len() % 2 == 1 {
            self.words.push(0);
        }
        let at = self.words.len();
        for &v in values {
            self.words.push(v as u32);
            self.words.push((v >> 32) as u32);
        }
        at
    }

    /// Reserve `n` zeroed words for the kernel to write, and return their offset.
    pub fn reserve(&mut self, n: usize) -> usize {
        let at = self.words.len();
        self.words.resize(at + n, 0);
        at
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }
}

/// What one dispatch wrote, per row.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Readback {
    pub rng: Vec<u64>,
    pub outputs: Vec<u32>,
    /// 1 where the row's logits held a NaN or a +inf, else 0.
    pub nonfinite: Vec<u32>,
}

/// Split a read-back tail of `n` RNG offsets (two words each, low first),
/// then `n` output tokens, then `n` non-finite verdicts.
pub(crate) fn split_readback(tail: &[u32], n: usize) -> Readback {
    let rng = (0..n)
        .map(|i| u64::from(tail[2 * i]) | (u64::from(tail[2 * i + 1]) << 32))
        .collect();
    Readback {
        rng,
        outputs: tail[2 * n..3 * n].to_vec(),
        nonfinite: tail[3 * n..4 * n].to_vec(),
    }
}

/// What every non-finite refusal says, so the step that sees one can tell it
/// from any other sampling failure ([`is_nonfinite_refusal`]).
const NONFINITE_REFUSAL: &str = "sampling refused: the logits of row(s)";

/// Refuse a dispatch any of whose rows the kernel judged non-finite, naming
/// them. Such a row came from a forward gone bad — a NaN or a +inf in its
/// logits — and the token drawn from it means nothing, so the rows fail rather
/// than commit it. The error carries no stall marker: the forward is not run
/// again, its turns fail.
pub(crate) fn refuse_nonfinite(verdicts: &[u32]) -> candle::Result<()> {
    let bad: Vec<usize> = verdicts
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0)
        .map(|(row, _)| row)
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    candle::bail!(
        "{NONFINITE_REFUSAL} {bad:?} of {} held a NaN or a +inf — \
         the forward that produced them is broken, so no token is drawn from them",
        verdicts.len()
    )
}

/// Whether `e` is a [`refuse_nonfinite`] refusal: the forward produced a
/// non-finite logit, rather than the sampler failing on its own account.
pub(crate) fn is_nonfinite_refusal(e: &candle::Error) -> bool {
    e.to_string().contains(NONFINITE_REFUSAL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_lay_end_to_end_with_u64s_aligned() {
        let mut pack = ArgPack::new();
        assert_eq!(pack.push_i32(&[7, -2, 3]), 0);
        assert_eq!(pack.push_i32(&[]), 3);
        assert_eq!(pack.push_f32(&[1.0]), 4);
        // Five words so far: the u64s start on the next even word.
        assert_eq!(pack.push_u64(&[0x0000_0002_0000_0001, 9]), 6);
        assert_eq!(pack.reserve(2), 10);
        assert_eq!(
            pack.words(),
            &[
                7,
                (-2i32) as u32,
                3,
                u32::MAX,
                0x3f80_0000,
                0,
                1,
                2,
                9,
                0,
                0,
                0
            ]
        );
    }

    #[test]
    fn an_even_offset_needs_no_padding_before_a_u64() {
        let mut pack = ArgPack::new();
        pack.push_i32(&[1, 2]);
        assert_eq!(pack.push_u64(&[5]), 2);
        assert_eq!(pack.words(), &[1, 2, 5, 0]);
    }

    #[test]
    fn the_tail_splits_into_rng_offsets_then_outputs_then_verdicts() {
        let tail = [1, 2, 3, 0, 40, 41, 0, 1];
        assert_eq!(
            split_readback(&tail, 2),
            Readback {
                rng: vec![0x0000_0002_0000_0001, 3],
                outputs: vec![40, 41],
                nonfinite: vec![0, 1],
            }
        );
    }

    /// Clean rows pass; one flagged row refuses the dispatch, by its row.
    #[test]
    fn a_flagged_row_refuses_the_dispatch_and_is_named() {
        assert!(refuse_nonfinite(&[0, 0, 0]).is_ok());
        assert!(refuse_nonfinite(&[]).is_ok());
        let e = refuse_nonfinite(&[0, 1, 0, 1]).unwrap_err().to_string();
        assert!(e.contains("row(s) [1, 3] of 4"), "{e}");
    }

    /// A refusal is told apart from any other sampling failure.
    #[test]
    fn a_nonfinite_refusal_is_told_apart() {
        let refused = refuse_nonfinite(&[1]).unwrap_err();
        assert!(is_nonfinite_refusal(&refused));
        assert!(!is_nonfinite_refusal(&candle::Error::Msg(
            "a weight is negative".into()
        )));
    }
}
