//! A fingerprint of the **repack formula itself**, so a pack built by a
//! different one is never reused.
//!
//! # The hole this closes
//!
//! The pack file is a cache of repacked expert weights, and everything else in
//! the header identifies the *input*: which checkpoint, which numeric mode,
//! which record geometry. Nothing identified the *function*. Change how the
//! repack lays bytes out — a different permutation, a moved rounding step in the
//! quantizer, a fixed bug — at unchanged sizes, offsets and dtypes, and every
//! one of those checks still passes. The stale pack is reused, and the model
//! serves subtly wrong weights for its entire expert set, silently, across
//! restarts and substrate wipes, until someone notices the outputs drifted.
//!
//! A version constant does not close it, because it relies on the person who
//! changed the formula remembering to bump the constant. This does not rely on
//! anyone: it *runs* the formula and hashes what comes out.
//!
//! # What it covers
//!
//! A fixed table of `(source dtype → target dtype)` pairs — every quantisation
//! an expert weight can arrive as, repacked every way the engine can repack it:
//! straight to the gemx K/128 layout, and to each of the KO twins the two int8
//! modes select. Each pair is fed a deterministic reference matrix, repacked,
//! and the output hashed.
//!
//! **The table is itself part of the hash.** Adding a dtype, removing one, or
//! changing which twin a mode selects moves the fingerprint even if every byte
//! of every repack is unchanged — so the "stable inventory of what is in the
//! repack" is checked alongside the bytes rather than trusted separately. A pair
//! the repack refuses hashes as a refusal, which means *gaining* support for a
//! type invalidates old packs too.
//!
//! Every pair is swept regardless of the mode this process runs in. The
//! fingerprint is a property of the build, not of the run, and a pack written by
//! a binary that repacks Q5_K differently is stale whether or not today's model
//! contains a Q5_K tensor.
//!
//! # Cost
//!
//! 36 repacks of a 32×256 matrix, once, at startup: tens of milliseconds against
//! a check that no amount of care replaces.

use candle::quantized::{repack_to_host, GgmlDType, Int8Mode};
use candle::CudaDevice;

/// Rows in the reference matrix. A multiple of 8, which the KO repack requires.
const REF_ROWS: usize = 32;

/// Columns in the reference matrix. A multiple of 256, which satisfies every
/// block size in play (32 or 256) and the KO repack's multiple-of-128.
const REF_COLS: usize = 256;

/// Every quantisation an expert weight can arrive as.
///
/// The order is part of the fingerprint — reordering this list is a change to
/// the inventory and invalidates existing packs, which is the intended
/// behaviour for anything that alters what the sweep covers.
const SOURCE_DTYPES: &[GgmlDType] = &[
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2_K,
    GgmlDType::Q3_K,
    GgmlDType::Q4_K,
    GgmlDType::Q5_K,
    GgmlDType::Q6_K,
    GgmlDType::Q8_K,
];

/// A reference matrix's bytes for `dtype`, deterministic and free of NaN.
///
/// The pattern is `i % 61`, so every byte is ≤ 60. That matters for more than
/// reproducibility: quantised blocks carry `f16` scales, and an `f16` whose high
/// byte is ≤ 0x3F can never have an all-ones exponent — so no scale is ever NaN
/// or infinite. A dequantise-requantise repack (the KO path) would otherwise be
/// hashing NaN payload bits, which are not guaranteed stable.
fn reference_bytes(dtype: GgmlDType) -> Vec<u8> {
    let blocks = REF_ROWS * REF_COLS / dtype.block_size();
    let len = blocks * dtype.type_size();
    (0..len).map(|i| (i % 61) as u8).collect()
}

/// FNV-1a over 64 bits. Written out rather than pulled from a crate because the
/// value has to be stable across dependency bumps — a hash that changed on its
/// own would invalidate every pack on the machine for no reason.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xCBF2_9CE4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }
}

/// The `(source, target)` pairs the sweep runs, in order.
///
/// Straight gemx first (target = source), then the KO twin each int8 mode
/// selects. A source with no twin in a mode contributes the pair and its
/// refusal, so the mapping itself is covered.
fn reference_pairs() -> Vec<(GgmlDType, Option<GgmlDType>)> {
    let mut pairs = Vec::with_capacity(SOURCE_DTYPES.len() * 3);
    for &src in SOURCE_DTYPES {
        pairs.push((src, Some(src)));
        for mode in [Int8Mode::Performance, Int8Mode::Precision] {
            pairs.push((src, src.to_ko(mode).ok()));
        }
    }
    pairs
}

/// Run the sweep and hash it.
///
/// Failures are folded into the hash rather than propagated: a dtype this build
/// cannot repack is a fact about the build, and the *next* build gaining support
/// for it should invalidate the pack exactly as a changed output would.
pub(crate) fn repack_fingerprint(device: &CudaDevice) -> u64 {
    let mut h = Fnv::new();
    h.write(b"expert-repack-sweep-v1");
    h.write_u32(REF_ROWS as u32);
    h.write_u32(REF_COLS as u32);
    for (src, target) in reference_pairs() {
        h.write_u32(src.to_u32());
        match target {
            None => h.write(b"-no-twin-"),
            Some(tgt) => {
                h.write_u32(tgt.to_u32());
                match repack_to_host(device, &reference_bytes(src), REF_ROWS, REF_COLS, src, tgt) {
                    Ok(out) => {
                        h.write_u32(out.len() as u32);
                        h.write(&out);
                    }
                    Err(_) => h.write(b"-unsupported-"),
                }
            }
        }
    }
    h.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reference matrix is a whole number of blocks, and every byte is
    /// low enough that an `f16` scale read out of it is finite.
    #[test]
    fn reference_bytes_are_block_sized_and_nan_free() {
        for &d in SOURCE_DTYPES {
            let bytes = reference_bytes(d);
            assert_eq!(
                REF_ROWS * REF_COLS % d.block_size(),
                0,
                "{d:?} block size {} does not divide the reference matrix",
                d.block_size()
            );
            assert_eq!(
                bytes.len(),
                REF_ROWS * REF_COLS / d.block_size() * d.type_size()
            );
            assert!(
                bytes.iter().all(|&b| b < 0x40),
                "{d:?} reference bytes can form an all-ones f16 exponent"
            );
        }
    }

    /// The sweep covers each source three ways — gemx, and the twin each int8
    /// mode picks — so a change to any one of those mappings moves the hash.
    #[test]
    fn the_sweep_covers_every_source_three_ways() {
        let pairs = reference_pairs();
        assert_eq!(pairs.len(), SOURCE_DTYPES.len() * 3);
        for &src in SOURCE_DTYPES {
            let mine: Vec<_> = pairs.iter().filter(|(s, _)| *s == src).collect();
            assert_eq!(mine.len(), 3, "{src:?}");
            assert_eq!(mine[0].1, Some(src), "{src:?} lacks its gemx pair");
        }
    }

    /// The two int8 modes are swept separately, because they do not select the
    /// same twin — Q4_K goes to Q4_KO under Performance and Q5_KO under
    /// Precision, and a fingerprint that swept only one would miss a change to
    /// the other.
    #[test]
    fn the_two_int8_modes_are_swept_separately() {
        let pairs = reference_pairs();
        let q4k: Vec<_> = pairs
            .iter()
            .filter(|(s, _)| *s == GgmlDType::Q4_K)
            .map(|(_, t)| *t)
            .collect();
        assert_eq!(
            q4k,
            vec![
                Some(GgmlDType::Q4_K),
                Some(GgmlDType::Q4_KO),
                Some(GgmlDType::Q5_KO)
            ]
        );
    }

    /// The hash is a pure function of what went into it: same input, same
    /// value, and any difference in the stream moves it.
    #[test]
    fn the_hash_is_stable_and_sensitive() {
        let mut a = Fnv::new();
        a.write(b"abc");
        let mut b = Fnv::new();
        b.write(b"abc");
        let mut c = Fnv::new();
        c.write(b"abd");
        assert_eq!(a.0, b.0);
        assert_ne!(a.0, c.0);
        // Field boundaries matter: two different splits of the same bytes must
        // not collide, or a reordered inventory could hash equal.
        let mut d = Fnv::new();
        d.write_u32(1);
        d.write_u32(2);
        let mut e = Fnv::new();
        e.write_u32(2);
        e.write_u32(1);
        assert_ne!(d.0, e.0);
    }
}
