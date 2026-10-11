//! A fingerprint of the **repack formula itself**, per format, so a pack built
//! by a different formula is never reused — and a change to one format's repack
//! invalidates only the packs that hold that format.
//!
//! Both repacked sections of a model pack carry a table of these — the expert
//! section (`expert_lre::pack`) and the layer section (`layer_stream::pack`) —
//! and each checks its table at open. The hashing and the check are plain
//! arithmetic, compiled everywhere so the section formats' own tests run on a
//! machine with no GPU; only producing a fingerprint from this build's repack
//! needs the device.
//!
//! # The hole this closes
//!
//! A pack is the repacked weights, and its geometry says where the bytes go:
//! which dtype, which offsets, which sizes. Nothing in the geometry identifies
//! the *function*. Change how the repack lays bytes out — a different
//! permutation, a moved rounding step in the quantizer, a fixed bug — at
//! unchanged sizes, offsets and dtypes, and every geometric check still passes.
//! The stale pack is reused, and the model serves subtly wrong weights,
//! silently, until someone notices the outputs drifted.
//!
//! A version constant does not close it, because it relies on the person who
//! changed the formula remembering to bump the constant. This does not rely on
//! anyone: it *runs* the formula and hashes what comes out.
//!
//! # Per pair, not per build
//!
//! Each `(source dtype → target dtype)` pair is fed a deterministic reference
//! matrix, repacked, and the output hashed on its own. A pack records the hash
//! of every pair its layers use — a set read off its own geometry (each
//! section's `pairs_in`), so it cannot omit a format it holds — and an open
//! recomputes exactly those. A change to the `Q6_K` repack moves the `Q6_K`
//! pairs' hashes and invalidates the packs holding `Q6_K` projections; a
//! `Q4_K_M` model's pack is untouched. A change to code every pair runs through
//! moves every hash, and invalidates every pack that uses any of them.
//!
//! The pair's identity and the reference geometry are part of each hash, and a
//! pair the repack refuses hashes as a refusal — so *gaining* support for a
//! type moves that pair too.
//!
//! # Cost
//!
//! One repack of a 32×256 matrix per pair a pack holds — two or three for a
//! typical model — at open.

#[cfg(feature = "cuda")]
use candle::quantized::repack_to_host;
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::CudaDevice;
use candle::Result;

/// Rows in the reference matrix. A multiple of 8, which the KO repack requires.
const REF_ROWS: usize = 32;

/// Columns in the reference matrix. A multiple of 256, which satisfies every
/// block size in play (32 or 256) and the KO repack's multiple-of-128.
const REF_COLS: usize = 256;

/// The version of the hashing scheme. Part of every hash, so a change to how a
/// pair is hashed is a change to every pair's value.
const SCHEME: &[u8] = b"repack-pair-v1";

/// What one repack pair produces, hashed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PairPrint {
    pub src_dtype: GgmlDType,
    pub dtype: GgmlDType,
    pub fp: u64,
}

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

/// The fingerprint of `src → dtype` under `repack`, which is handed the
/// reference matrix's bytes, rows and columns.
///
/// `repack` is a parameter so the hashing is testable without a device; the
/// production caller is `pair_fingerprint`.
pub(crate) fn pair_fingerprint_with(
    src: GgmlDType,
    dtype: GgmlDType,
    repack: impl FnOnce(&[u8], usize, usize) -> Result<Vec<u8>>,
) -> u64 {
    let mut h = Fnv::new();
    h.write(SCHEME);
    h.write_u32(REF_ROWS as u32);
    h.write_u32(REF_COLS as u32);
    h.write_u32(src.to_u32());
    h.write_u32(dtype.to_u32());
    // A refusal is a fact about the build, and the next build gaining support
    // for the pair should move its hash exactly as a changed output would.
    match repack(&reference_bytes(src), REF_ROWS, REF_COLS) {
        Ok(out) => {
            h.write_u32(out.len() as u32);
            h.write(&out);
        }
        Err(_) => h.write(b"-unsupported-"),
    }
    h.0
}

/// The fingerprint of `src → dtype` as this build's repack produces it.
#[cfg(feature = "cuda")]
pub(crate) fn pair_fingerprint(device: &CudaDevice, src: GgmlDType, dtype: GgmlDType) -> u64 {
    pair_fingerprint_with(src, dtype, |bytes, rows, cols| {
        repack_to_host(device, bytes, rows, cols, src, dtype)
    })
}

/// A fingerprint entry for every pair in `pairs`, in order.
#[cfg(feature = "cuda")]
pub(crate) fn prints_for(device: &CudaDevice, pairs: &[(GgmlDType, GgmlDType)]) -> Vec<PairPrint> {
    pairs
        .iter()
        .map(|&(src_dtype, dtype)| PairPrint {
            src_dtype,
            dtype,
            fp: pair_fingerprint(device, src_dtype, dtype),
        })
        .collect()
}

/// Check every recorded fingerprint against what `fingerprint` produces now,
/// naming the first pair that moved.
pub(crate) fn check_prints(
    recorded: &[PairPrint],
    fingerprint: impl Fn(GgmlDType, GgmlDType) -> u64,
) -> Result<()> {
    for p in recorded {
        let now = fingerprint(p.src_dtype, p.dtype);
        if now != p.fp {
            candle::bail!(
                "the {:?} → {:?} repack produces different bytes in this build \
                 (fingerprint {:#018x}, recorded {:#018x}) — the pack was written by another \
                 formula and must be rebuilt",
                p.src_dtype,
                p.dtype,
                now,
                p.fp
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reference matrix is a whole number of blocks, and every byte is
    /// low enough that an `f16` scale read out of it is finite.
    #[test]
    fn reference_bytes_are_block_sized_and_nan_free() {
        for d in [
            GgmlDType::Q4_0,
            GgmlDType::Q8_0,
            GgmlDType::Q2_K,
            GgmlDType::Q3_K,
            GgmlDType::Q4_K,
            GgmlDType::Q5_K,
            GgmlDType::Q6_K,
        ] {
            let bytes = reference_bytes(d);
            assert_eq!(REF_ROWS * REF_COLS % d.block_size(), 0, "{d:?}");
            assert_eq!(
                bytes.len(),
                REF_ROWS * REF_COLS / d.block_size() * d.type_size()
            );
            assert!(bytes.iter().all(|&b| b < 0x40), "{d:?}");
        }
    }

    /// A repack whose output is a fixed transform of its input, so tests can
    /// change one pair's output and leave the others alone.
    fn fake(bytes: &[u8], salt: u8) -> Result<Vec<u8>> {
        Ok(bytes.iter().map(|b| b.wrapping_add(salt)).collect())
    }

    /// Same pair, same repack, same value — and the value is pinned, so a
    /// change to the hashing scheme is seen here and not as a fleet-wide rebuild.
    #[test]
    fn a_pair_hashes_to_a_stable_value() {
        let a = pair_fingerprint_with(GgmlDType::Q4_K, GgmlDType::Q4_KO, |b, _, _| fake(b, 0));
        let b = pair_fingerprint_with(GgmlDType::Q4_K, GgmlDType::Q4_KO, |b, _, _| fake(b, 0));
        assert_eq!(a, b);
        assert_eq!(a, 0xBDFC_ABAF_32D7_B75F);
    }

    /// **The point of per-pair hashing.** A changed repack for one pair moves
    /// that pair's fingerprint and no other.
    #[test]
    fn a_changed_repack_moves_only_its_own_pair() {
        let q4 = (GgmlDType::Q4_K, GgmlDType::Q4_KO);
        let q6 = (GgmlDType::Q6_K, GgmlDType::Q6_KO);
        let before =
            |p: (GgmlDType, GgmlDType)| pair_fingerprint_with(p.0, p.1, |b, _, _| fake(b, 0));
        // The Q6_K repack changes; Q4_K's does not.
        let after = |p: (GgmlDType, GgmlDType)| {
            let salt = if p == q6 { 1 } else { 0 };
            pair_fingerprint_with(p.0, p.1, |b, _, _| fake(b, salt))
        };
        assert_eq!(before(q4), after(q4));
        assert_ne!(before(q6), after(q6));
    }

    /// The pair is part of its own hash: two pairs whose repacks happen to emit
    /// the same bytes still fingerprint differently.
    #[test]
    fn the_pair_names_are_hashed() {
        let a = pair_fingerprint_with(GgmlDType::Q4_K, GgmlDType::Q4_KO, |_, _, _| Ok(vec![1]));
        let b = pair_fingerprint_with(GgmlDType::Q4_K, GgmlDType::Q5_KO, |_, _, _| Ok(vec![1]));
        assert_ne!(a, b);
    }

    /// A refusal hashes as a refusal, distinct from any output.
    #[test]
    fn a_refused_pair_hashes_distinctly() {
        let refused = pair_fingerprint_with(GgmlDType::Q4_K, GgmlDType::Q4_KO, |_, _, _| {
            candle::bail!("no")
        });
        let empty = pair_fingerprint_with(GgmlDType::Q4_K, GgmlDType::Q4_KO, |_, _, _| Ok(vec![]));
        assert_ne!(refused, empty);
    }

    /// The check names the pair that moved and passes the ones that did not.
    #[test]
    fn the_check_names_the_moved_pair() {
        let recorded = [
            PairPrint {
                src_dtype: GgmlDType::Q4_K,
                dtype: GgmlDType::Q4_KO,
                fp: 1,
            },
            PairPrint {
                src_dtype: GgmlDType::Q6_K,
                dtype: GgmlDType::Q6_KO,
                fp: 2,
            },
        ];
        let same = |s: GgmlDType, _: GgmlDType| if s == GgmlDType::Q4_K { 1 } else { 2 };
        assert!(check_prints(&recorded, same).is_ok());
        let moved = |s: GgmlDType, _: GgmlDType| if s == GgmlDType::Q4_K { 1 } else { 3 };
        let e = check_prints(&recorded, moved).unwrap_err().to_string();
        assert!(e.contains("Q6_K"), "{e}");
        assert!(!e.contains("Q4_K →"), "{e}");
    }
}
