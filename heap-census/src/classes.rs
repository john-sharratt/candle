//! Size classes: which power of two an allocation's requested size rounds up
//! to.
//!
//! Class `k` holds requests of `(2^(k−1), 2^k]` bytes: class 0 is the 0- and
//! 1-byte requests, class 15 is 16 KiB + 1 through 32 KiB. Requests above 2^47
//! all land in the top class. A class names a band of sizes, which is how a
//! growing class points at the structures that allocate in that band.

/// How many classes there are.
pub const CLASSES: usize = 49;

/// The class of a `size`-byte request.
#[inline(always)]
pub const fn class_of(size: usize) -> usize {
    if size <= 1 {
        return 0;
    }
    let k = (usize::BITS - (size - 1).leading_zeros()) as usize;
    if k < CLASSES {
        k
    } else {
        CLASSES - 1
    }
}

/// The largest request class `k` holds: `2^k`, or `u64::MAX` for the top
/// class, which holds everything above.
pub const fn upto_bytes(k: usize) -> u64 {
    if k + 1 >= CLASSES {
        u64::MAX
    } else {
        1u64 << k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each class holds `(2^(k−1), 2^k]`**, exactly, at both edges.
    #[test]
    fn a_class_holds_the_sizes_up_to_its_power_of_two() {
        let cases: [(usize, usize); 13] = [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (4096, 12),
            (4097, 13),
            (16_384, 14),
            (16_385, 15),
            (30_000, 15),
            (32_768, 15),
            (32_769, 16),
        ];
        for (size, class) in cases {
            assert_eq!(class_of(size), class, "size {size}");
        }
    }

    /// **The top class holds everything above 2^47**, so no size is out of
    /// range.
    #[test]
    fn the_top_class_holds_everything_above_it() {
        assert_eq!(class_of(1 << 47), 47);
        assert_eq!(class_of((1 << 47) + 1), 48);
        assert_eq!(class_of(usize::MAX), 48);
        assert_eq!(upto_bytes(15), 32_768);
        assert_eq!(upto_bytes(47), 1 << 47);
        assert_eq!(upto_bytes(48), u64::MAX);
    }
}
