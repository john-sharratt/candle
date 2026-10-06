//! Which tokens of a routed-expert launch the residency scoring counts as
//! decode.
//!
//! The expert cache weights a decode row's routing far above a prompt row's: a
//! decode sequence routes much the same experts from one step to the next, a
//! prompt sweeps the table roughly once. A wave's rows are not only a decode
//! prefix, though. A speculative verify segment is a sequence's next tokens
//! decoded several at once, laid out like a prompt; and the last token of a
//! prompt is the one its decode continues from, so its experts are the best
//! early guess at the first decode step's. Both are scored as decode, wherever
//! they sit — so the rows are a set of ranges, not a count.
//!
//! The ranges travel by value in `moe_bucketize`'s launch parameters, bounded
//! by [`MAX_DECODE_RANGES`]. Adjacent ranges merge; one that does not fit is
//! left out and its tokens score as prompt rows — the conservative side.

/// Mirrors `MAX_DECODE_RANGES` in `candle-kernels/src/simple/moe_bucketize.cu`.
pub const MAX_DECODE_RANGES: usize = 32;

/// Token ranges `[lo, hi)` scored as decode — see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeRows {
    n: usize,
    lo: [u32; MAX_DECODE_RANGES],
    hi: [u32; MAX_DECODE_RANGES],
}

impl Default for DecodeRows {
    fn default() -> Self {
        Self::none()
    }
}

impl DecodeRows {
    /// No token is a decode row.
    pub fn none() -> Self {
        Self {
            n: 0,
            lo: [0; MAX_DECODE_RANGES],
            hi: [0; MAX_DECODE_RANGES],
        }
    }

    /// Tokens `[0, n)` are decode rows.
    pub fn prefix(n: usize) -> Self {
        let mut rows = Self::none();
        rows.push(0, n);
        rows
    }

    /// Add `[lo, hi)`. Merges with the last range when they touch; returns
    /// false, leaving the tokens prompt-scored, when the bound is reached.
    pub fn push(&mut self, lo: usize, hi: usize) -> bool {
        if lo >= hi {
            return true;
        }
        // Saturating: the kernel counts tokens in 32 bits, so a bound past
        // that names no token either way.
        let (lo, hi) = (
            u32::try_from(lo).unwrap_or(u32::MAX),
            u32::try_from(hi).unwrap_or(u32::MAX),
        );
        if self.n > 0 && self.hi[self.n - 1] == lo {
            self.hi[self.n - 1] = hi;
            return true;
        }
        if self.n == MAX_DECODE_RANGES {
            return false;
        }
        self.lo[self.n] = lo;
        self.hi[self.n] = hi;
        self.n += 1;
        true
    }

    /// The ranges' starts and ends, for the launcher.
    pub fn ranges(&self) -> (&[u32], &[u32]) {
        (&self.lo[..self.n], &self.hi[..self.n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound the launcher enforces is this one.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_bound_mirrors_the_kernel() {
        assert_eq!(
            MAX_DECODE_RANGES,
            candle_kernels::simple::moe_bucketize::MAX_DECODE_RANGES
        );
    }

    #[test]
    fn a_prefix_is_one_range_from_zero() {
        let r = DecodeRows::prefix(3);
        assert_eq!(r.ranges(), (&[0u32][..], &[3u32][..]));
        assert_eq!(DecodeRows::prefix(0).ranges().0.len(), 0);
    }

    #[test]
    fn a_bound_past_32_bits_saturates() {
        let mut r = DecodeRows::none();
        assert!(r.push(u32::MAX as usize - 1, usize::MAX));
        assert_eq!(r.ranges(), (&[u32::MAX - 1][..], &[u32::MAX][..]));
    }

    #[test]
    fn touching_ranges_merge_and_gaps_stay_apart() {
        let mut r = DecodeRows::prefix(2);
        assert!(r.push(2, 7));
        assert!(r.push(19, 20));
        assert!(r.push(4, 4), "an empty range is nothing to add");
        assert_eq!(r.ranges(), (&[0u32, 19][..], &[7u32, 20][..]));
    }

    #[test]
    fn a_range_past_the_bound_is_left_out() {
        let mut r = DecodeRows::none();
        for i in 0..MAX_DECODE_RANGES {
            assert!(r.push(2 * i, 2 * i + 1));
        }
        assert!(!r.push(1000, 1001));
        assert_eq!(r.ranges().0.len(), MAX_DECODE_RANGES);
        assert!(!r.ranges().0.contains(&1000));
        assert!(
            r.push(2 * MAX_DECODE_RANGES - 1, 2 * MAX_DECODE_RANGES),
            "a merge still fits"
        );
    }
}
