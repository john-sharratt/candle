//! Fletcher-32 checksum — the KV-chunk "golden" integrity check.
//!
//! The golden is computed on-GPU over freshly-quantized arena bytes, before any
//! device→host copy (see candle-kernels `simple/fletcher32.cu`), and stored with
//! the chunk. This reference recomputes it on the CPU to verify a chunk once it
//! has landed in a warm/cold tier or been read back from disk: a mismatch means
//! the bytes were corrupted by the DtoH copy or on storage — corruption the old
//! host-side CRC, taken *after* the copy, could never detect.
//!
//! The GPU kernel and this reference produce identical values by construction:
//! bytes form little-endian 16-bit words, an odd trailing byte becomes a word
//! with a zero high byte, and both sums are taken mod 65535. The kernel reduces
//! only the final totals (u64 accumulators); this reduces every word. Modular
//! arithmetic makes the two bit-identical.

/// The Fletcher-32 modulus (2^16 − 1).
const MOD: u32 = 65535;

/// Fletcher-32 over `data`: `checksum = (sum2 << 16) | sum1`.
///
/// Empty input yields 0. Matches the GPU `run_fletcher32` kernel byte-for-byte.
pub fn fletcher32(data: &[u8]) -> u32 {
    // Reduced per BLOCK, not per word. Reducing every word costs two integer
    // divisions per two bytes, which showed up as real CPU on the persistence
    // path — every record is checksummed, and the records include multi-MB
    // snapshots. Deferring is the same arithmetic the GPU kernel already does
    // (see the module docs): modular addition is associative, so reducing at
    // block boundaries instead of each step yields identical sums.
    //
    // `u64` accumulators with a 65 536-word block cannot overflow: within a
    // block `sum1 < 65535 + 65536·65535 ≈ 4.3e9` and `sum2 < 65535 +
    // 65536·4.3e9 ≈ 2.8e14`, both far inside `u64::MAX`. Bounding the block is
    // what keeps a 63 MiB payload safe — accumulating one un-reduced `sum2`
    // across the whole input would overflow past ~47 MB.
    const BLOCK_WORDS: usize = 65_536;
    let modulus = MOD as u64;
    let mut sum1: u64 = 0;
    let mut sum2: u64 = 0;
    let words = data.chunks_exact(2);
    let remainder = words.remainder();
    let mut n = 0usize;
    for w in words {
        let word = (w[0] as u64) | ((w[1] as u64) << 8);
        sum1 += word;
        sum2 += sum1;
        n += 1;
        if n == BLOCK_WORDS {
            sum1 %= modulus;
            sum2 %= modulus;
            n = 0;
        }
    }
    if let [tail] = remainder {
        // An odd trailing byte becomes a word with a zero high byte.
        sum1 += *tail as u64;
        sum2 += sum1;
    }
    sum1 %= modulus;
    sum2 %= modulus;
    ((sum2 as u32) << 16) | (sum1 as u32)
}

#[cfg(test)]
mod tests {
    use super::fletcher32;

    /// The original per-word-reduction form, kept verbatim as the differential
    /// oracle. `fletcher32` defers reduction to block boundaries for speed; the
    /// two must agree on every input, because this value is a stored integrity
    /// golden and a drift would silently invalidate every chunk on disk.
    fn fletcher32_reduce_per_word(data: &[u8]) -> u32 {
        const MOD: u32 = 65535;
        let mut sum1: u32 = 0;
        let mut sum2: u32 = 0;
        let mut words = data.chunks_exact(2);
        for w in &mut words {
            let word = (w[0] as u32) | ((w[1] as u32) << 8);
            sum1 = (sum1 + word) % MOD;
            sum2 = (sum2 + sum1) % MOD;
        }
        if let [tail] = words.remainder() {
            let word = *tail as u32;
            sum1 = (sum1 + word) % MOD;
            sum2 = (sum2 + sum1) % MOD;
        }
        (sum2 << 16) | sum1
    }

    /// Bit-identity against the per-word oracle, including inputs that cross the
    /// 65 536-word block boundary the fast path reduces at — a deferred-reduction
    /// bug hides precisely there, not in short vectors.
    #[test]
    fn block_deferred_matches_per_word_reduction() {
        let mut lens: Vec<usize> = vec![0, 1, 2, 3, 8, 255, 4096];
        // Straddle the block boundary (65 536 words = 131 072 bytes).
        for d in [-3isize, -2, -1, 0, 1, 2, 3] {
            lens.push((131_072 + d) as usize);
            lens.push((262_144 + d) as usize);
        }
        for len in lens {
            // Deterministic non-uniform bytes: a constant fill would not
            // exercise carry behaviour in the accumulators.
            let data: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(31) ^ (i >> 5)) as u8)
                .collect();
            assert_eq!(
                fletcher32(&data),
                fletcher32_reduce_per_word(&data),
                "block-deferred fletcher32 diverged at len={len}",
            );
        }
    }

    #[test]
    fn known_vector_abcdefgh() {
        // The canonical Fletcher-32 test vector.
        assert_eq!(fletcher32(b"abcdefgh"), 0xEBE1_9591);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(fletcher32(b""), 0);
    }

    #[test]
    fn single_byte_matches_padded_word() {
        // One byte 'a' (0x61): sum1 = 0x61, sum2 = 0x61.
        assert_eq!(fletcher32(b"a"), (0x61 << 16) | 0x61);
    }

    #[test]
    fn odd_length_pads_trailing_byte_high_zero() {
        // "abc": words 0x6261 ('ab' LE), then tail 'c'=0x63 as a high-zero word.
        // sum1 = (0x6261 + 0x63) % 65535 = 0x62C4
        // sum2 = (0x6261 + 0x62C4) % 65535 = 0xC525
        let word1 = 0x6261u32;
        let mut s1 = word1 % 65535;
        let mut s2 = s1 % 65535;
        s1 = (s1 + 0x63) % 65535;
        s2 = (s2 + s1) % 65535;
        assert_eq!(fletcher32(b"abc"), (s2 << 16) | s1);
    }

    #[test]
    fn low_bytes_never_exceed_modulus() {
        // A long run of 0xFF bytes exercises the running reduction; the sums
        // must stay in [0, 65534] and the result must be reproducible.
        let data = vec![0xFFu8; 4096];
        let c = fletcher32(&data);
        assert_eq!(c & 0xFFFF, (c & 0xFFFF).min(65534));
        assert_eq!((c >> 16) & 0xFFFF, ((c >> 16) & 0xFFFF).min(65534));
        // Reference re-run: deterministic.
        assert_eq!(fletcher32(&data), c);
    }
}
