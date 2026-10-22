//! Sign agreement between two layer-group signatures — `popcount(XNOR)` over
//! the group's words — at the widest popcount this CPU runs, chosen at runtime.
//!
//! The binary is built for a CPU level every cluster member has
//! (`.cargo/config.toml`), so an instruction past it is used only where the
//! CPU reports it. On a CPU with AVX-512 `VPOPCNTDQ` the locked 8-word group
//! (4 heads × 128 bits) is one 512-bit XOR and one vector popcount; elsewhere
//! it is the scalar loop. Both count the same bits, so a scan's scores are
//! identical whichever ran.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64, _mm512_xor_si512,
};
#[cfg(target_arch = "x86_64")]
use std::sync::LazyLock;

/// Words in the locked layer-group width the vector path covers.
const GROUP_WORDS: usize = 8;

/// Whether this CPU runs the 512-bit vector popcount — asked once.
#[cfg(target_arch = "x86_64")]
static VPOPCNT: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
});

/// `popcount(!(a ^ b))` over the words the two signatures share.
#[inline]
pub(super) fn group_agreement(a: &[u64], b: &[u64]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if a.len() == GROUP_WORDS && b.len() == GROUP_WORDS && *VPOPCNT {
        // SAFETY: the CPU reported both features, and both slices hold the
        // 8 words the unaligned loads read.
        return unsafe { agreement8_vpopcnt(a, b) };
    }
    agreement_scalar(a, b)
}

/// The portable loop: what every CPU runs, and what the vector path must
/// equal bit for bit.
#[inline]
fn agreement_scalar(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (!(x ^ y)).count_ones()).sum()
}

/// One 8-word group: the bits that differ, counted in one vector popcount and
/// subtracted from the group's 512.
///
/// # Safety
///
/// The CPU must support `avx512f` and `avx512vpopcntdq`, and `a` and `b` must
/// each hold [`GROUP_WORDS`] words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn agreement8_vpopcnt(a: &[u64], b: &[u64]) -> u32 {
    debug_assert!(a.len() == GROUP_WORDS && b.len() == GROUP_WORDS);
    let va = _mm512_loadu_si512(a.as_ptr().cast());
    let vb = _mm512_loadu_si512(b.as_ptr().cast());
    let differ = _mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
    (GROUP_WORDS as u32 * 64) - differ as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reproducible spread of words: all-zero, all-one, alternating and
    /// mixed patterns.
    fn words(seed: u64) -> Vec<u64> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..GROUP_WORDS)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect()
    }

    #[test]
    fn agreement_counts_the_bits_that_match() {
        assert_eq!(group_agreement(&[0; 8], &[0; 8]), 512);
        assert_eq!(group_agreement(&[0; 8], &[u64::MAX; 8]), 0);
        assert_eq!(group_agreement(&[0b1011], &[0b0001]), 62);
        // Widths other than the locked group take the scalar loop.
        assert_eq!(group_agreement(&[0; 3], &[0; 3]), 192);
    }

    /// Whichever path this CPU takes, it equals the scalar loop exactly. On a
    /// CPU without `VPOPCNTDQ` the vector path is not exercised, and the test
    /// says so rather than passing silently.
    #[test]
    fn the_vector_popcount_equals_the_scalar_loop() {
        #[cfg(target_arch = "x86_64")]
        if !*VPOPCNT {
            eprintln!("skipped: this CPU has no avx512vpopcntdq, the vector path is not checked");
            return;
        }
        for seed in 0..4096 {
            let (a, b) = (words(seed), words(seed ^ 0x5555));
            assert_eq!(
                group_agreement(&a, &b),
                agreement_scalar(&a, &b),
                "seed {seed}"
            );
        }
        let edge = [
            [0u64; 8],
            [u64::MAX; 8],
            [0xAAAA_AAAA_AAAA_AAAA; 8],
            [0x5555_5555_5555_5555; 8],
        ];
        for a in &edge {
            for b in &edge {
                assert_eq!(group_agreement(a, b), agreement_scalar(a, b));
            }
        }
    }
}
