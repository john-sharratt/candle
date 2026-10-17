//! The AVX-512 VNNI tier's three pieces — a row's unpack, a tile's load and
//! their dot — on 64-byte vectors. The loop around them is every tier's own
//! (`simd`'s `matvec_tier`), so all of it but these runs, and is tested, on any
//! machine with a 256-bit tier.
//!
//! **The layouts are the 256-bit tiers'.** A row's 128 codes in unpack order
//! (`ko::Ko::perm`) are two `zmm`: Q3's bit pairs `s = 0, 1` in the first —
//! bytes `0..32` and `32..64`, `u = s·32 + b` — and `s = 2, 3` in the second;
//! Q4's low nibbles in the first, high in the second. A permuted activation
//! tile is the same 128 bytes, so it is two `zmm` loads: the weights stream in
//! place, `Permuted` is shared, and the eight-row fold stays in `ymm`.
//!
//! **What 64 bytes buy is the unpack.** Q3's crumb bytes are broadcast to both
//! halves of a vector and shifted by a different amount in each (`vpsrlvw`),
//! so one shift and one `and` yield two bit pairs; the hi bits come from one
//! `vptestmb` per vector — a mask register, set where the bit is — and a masked
//! add of 4. Against the 256-bit tiers' ~25 vector ops a row this is ~11, and
//! the dot is two `vpdpbusd` instead of four.
//!
//! **Exactness.** The dot's sixteen int32 lanes fold to eight by adding the
//! vector's halves — integer adds, so the order is free — and the shared loop
//! does the rest. The unpack is checked byte for byte two ways: by a model of
//! its steps on plain bytes, with the real tables, on every CPU; and by the
//! instructions themselves where the CPU has the tier.

use std::arch::x86_64::*;

use super::act::TILE;
use super::ko::{Ko, Q3_HI_BASE};
use super::simd::build_hi_index;

/// A 64-byte constant, loadable as a vector.
#[repr(align(64))]
struct V64([u8; 64]);

impl V64 {
    #[target_feature(enable = "avx512f")]
    fn load(&self) -> __m512i {
        // SAFETY: 64 aligned bytes.
        unsafe { _mm512_load_si512(self.0.as_ptr().cast()) }
    }
}

/// `a` in both halves.
const fn twice(a: [u8; 32]) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = a[i % 32];
        i += 1;
    }
    t
}

/// Crumb byte `b`'s hi byte, `b / 2`, per 128-bit lane: the 256-bit tiers'
/// index in both halves of a vector whose four lanes all hold the row's 16 hi
/// bytes.
static Q3_HI_INDEX: V64 = V64(twice(build_hi_index()));

/// Right shifts per 16-bit word, little-endian: `lo` for the vector's first
/// half, `hi` for its second — two bit pairs from one shift of the crumb bytes
/// broadcast to both.
const fn word_shifts(lo: u8, hi: u8) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut w = 0;
    while w < 32 {
        t[2 * w] = if w < 16 { lo } else { hi };
        w += 1;
    }
    t
}

/// Bit pairs `(0, 1)` for the first code vector, `(2, 3)` for the second.
static Q3_SHIFTS: [V64; 2] = [V64(word_shifts(0, 2)), V64(word_shifts(4, 6))];

/// The hi bit of crumb byte `b`'s pair `s` is bit `(b%2)·4 + s` of its hi
/// byte: `s_lo` for the vector's first half, `s_hi` for its second.
const fn hi_test(s_lo: usize, s_hi: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        let s = if i < 32 { s_lo } else { s_hi };
        t[i] = 1u8 << ((i % 2) * 4 + s);
        i += 1;
    }
    t
}

static Q3_HI_TEST: [V64; 2] = [V64(hi_test(0, 1)), V64(hi_test(2, 3))];

/// Row `r` of the chunk at `chunk`, in unpack order, as two vectors.
///
/// # Safety
/// AVX-512 F and BW; `chunk` is a whole chunk of `ko`, `r < 8`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub(super) unsafe fn unpack_row_512(ko: Ko, chunk: *const u8, r: usize) -> [__m512i; 2] {
    match ko {
        Ko::Q3 => unpack_q3(chunk.add(r * 32), chunk.add(Q3_HI_BASE + r * 16)),
        Ko::Q4 => unpack_q4(chunk.add(r * 64)),
    }
}

/// # Safety
/// AVX-512 F and BW; 32 readable bytes at `crumb`, 16 at `hi`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn unpack_q3(crumb: *const u8, hi: *const u8) -> [__m512i; 2] {
    let c = _mm512_broadcast_i64x4(_mm256_loadu_si256(crumb.cast()));
    let h = _mm512_broadcast_i32x4(_mm_loadu_si128(hi.cast()));
    let g = _mm512_shuffle_epi8(h, Q3_HI_INDEX.load());
    let three = _mm512_set1_epi8(3);
    let four = _mm512_set1_epi8(4);
    let mut out = [_mm512_setzero_si512(); 2];
    for (i, o) in out.iter_mut().enumerate() {
        // A word shift carries a byte's neighbour's bits into its top; the
        // `and` keeps the two this pair needs.
        let low = _mm512_and_si512(_mm512_srlv_epi16(c, Q3_SHIFTS[i].load()), three);
        let set = _mm512_test_epi8_mask(g, Q3_HI_TEST[i].load());
        *o = _mm512_mask_add_epi8(low, set, low, four);
    }
    out
}

/// # Safety
/// AVX-512 F and BW; 64 readable bytes at `ql`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn unpack_q4(ql: *const u8) -> [__m512i; 2] {
    let l = _mm512_loadu_si512(ql.cast());
    let f = _mm512_set1_epi8(0x0F);
    [
        _mm512_and_si512(l, f),
        _mm512_and_si512(_mm512_srli_epi16::<4>(l), f),
    ]
}

/// A permuted tile as two vectors.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn load_act_512(a: &[i8; TILE]) -> [__m512i; 2] {
    // SAFETY: 128 readable bytes.
    unsafe {
        [
            _mm512_loadu_si512(a.as_ptr().cast()),
            _mm512_loadu_si512(a.as_ptr().add(64).cast()),
        ]
    }
}

/// One row's dot with one token's tile, its sixteen int32 lanes folded to
/// the eight the shared loop reduces.
#[target_feature(enable = "avx2,avx512f,avx512vnni")]
#[inline]
pub(super) fn dot_512(w: &[__m512i; 2], a: &[__m512i; 2]) -> __m256i {
    let acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w[0], a[0]);
    let acc = _mm512_dpbusd_epi32(acc, w[1], a[1]);
    _mm256_add_epi32(
        _mm512_castsi512_si256(acc),
        _mm512_extracti64x4_epi64::<1>(acc),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::kernel::ko::{KoMatrix, RowCodes};
    use crate::models::expert_lre::cpu_pool::kernel::simd::Isa;
    use candle::quantized::ko_quant::quantize_ko;
    use candle::quantized::GgmlDType;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// The Q3 unpack's steps on plain bytes, each as its instruction defines
    /// it, with the real tables: the crumbs and hi bytes broadcast; each 16-bit
    /// word shifted right by its count (`vpsrlvw`) and the byte `and` 3; each
    /// byte's hi byte picked within its 128-bit lane (`vpshufb`: index bit 7
    /// zeroes, else its low four bits name the lane's byte); the mask set
    /// where `g & test` is non-zero (`vptestmb`), and 4 added there.
    fn model_q3(crumb: &[u8; 32], hi: &[u8; 16]) -> [u8; TILE] {
        let mut out = [0u8; TILE];
        for v in 0..2 {
            for i in 0..64 {
                let w = i / 2;
                let word =
                    u16::from(crumb[(2 * w) % 32]) | (u16::from(crumb[(2 * w + 1) % 32]) << 8);
                let count =
                    u16::from(Q3_SHIFTS[v].0[2 * w]) | (u16::from(Q3_SHIFTS[v].0[2 * w + 1]) << 8);
                let shifted = if count > 15 { 0 } else { word >> count };
                let byte = if i % 2 == 0 {
                    shifted as u8
                } else {
                    (shifted >> 8) as u8
                };
                let ix = Q3_HI_INDEX.0[i];
                let g = if ix & 0x80 != 0 {
                    0
                } else {
                    hi[usize::from(ix & 0x0F)]
                };
                let set = g & Q3_HI_TEST[v].0[i] != 0;
                out[v * 64 + i] = (byte & 3) + if set { 4 } else { 0 };
            }
        }
        out
    }

    /// The Q4 unpack's steps on plain bytes: the 64 bytes `and` 0x0F, then
    /// each 16-bit word shifted right by 4 (`vpsrlw`) and the byte `and` 0x0F.
    fn model_q4(ql: &[u8; 64]) -> [u8; TILE] {
        let mut out = [0u8; TILE];
        for i in 0..64 {
            out[i] = ql[i] & 0x0F;
            let w = i / 2;
            let word = (u16::from(ql[2 * w]) | (u16::from(ql[2 * w + 1]) << 8)) >> 4;
            let byte = if i % 2 == 0 {
                word as u8
            } else {
                (word >> 8) as u8
            };
            out[64 + i] = byte & 0x0F;
        }
        out
    }

    /// On every CPU: the modelled 512-bit unpack equals the scalar unpack
    /// byte for byte, for every bit pattern a row can hold — random bytes,
    /// all zeros and all ones — which checks the tables and the algorithm
    /// the instructions carry out.
    #[test]
    fn the_modelled_512_unpack_is_the_scalar_unpack() {
        let mut seed = 11;
        let mut rows: Vec<([u8; 32], [u8; 16], [u8; 64])> = (0..256)
            .map(|_| {
                (
                    std::array::from_fn(|_| lcg(&mut seed) as u8),
                    std::array::from_fn(|_| lcg(&mut seed) as u8),
                    std::array::from_fn(|_| lcg(&mut seed) as u8),
                )
            })
            .collect();
        rows.push(([0; 32], [0; 16], [0; 64]));
        rows.push(([0xFF; 32], [0xFF; 16], [0xFF; 64]));
        for (i, (crumb, hi, ql)) in rows.iter().enumerate() {
            let q3 = RowCodes::Q3 { crumb, hi }.unpack();
            assert_eq!(model_q3(crumb, hi), q3, "Q3 row {i}");
            let q4 = RowCodes::Q4 { ql }.unpack();
            assert_eq!(model_q4(ql), q4, "Q4 row {i}");
        }
    }

    /// The constant tables, raw: the index repeats the 256-bit one, the
    /// shifts are little-endian words split by half, and each test mask names
    /// bit `(b%2)·4 + s`.
    #[test]
    fn the_512_tables_are_the_256_layout_twice() {
        let idx = build_hi_index();
        assert_eq!(&Q3_HI_INDEX.0[..32], &idx);
        assert_eq!(&Q3_HI_INDEX.0[32..], &idx);
        assert_eq!(&Q3_SHIFTS[0].0[..4], &[0, 0, 0, 0]);
        assert_eq!(&Q3_SHIFTS[0].0[32..36], &[2, 0, 2, 0]);
        assert_eq!(&Q3_SHIFTS[1].0[..4], &[4, 0, 4, 0]);
        assert_eq!(&Q3_SHIFTS[1].0[62..], &[6, 0]);
        assert_eq!(&Q3_HI_TEST[0].0[..4], &[0x01, 0x10, 0x01, 0x10]);
        assert_eq!(&Q3_HI_TEST[0].0[32..34], &[0x02, 0x20]);
        assert_eq!(&Q3_HI_TEST[1].0[..2], &[0x04, 0x40]);
        assert_eq!(&Q3_HI_TEST[1].0[62..], &[0x08, 0x80]);
    }

    /// Where the CPU has the tier: the instructions' unpack equals the scalar
    /// unpack, byte for byte, for every row of random Q3_KO and Q4_KO
    /// projections.
    #[test]
    fn unpack_512_matches_scalar() {
        if !Isa::Avx512Vnni.available() {
            eprintln!("unpack_512_matches_scalar: skipped — this CPU has no AVX-512 VNNI");
            return;
        }
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let (nrows, ncols) = (32, 256);
            let mut seed = 3;
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|_| (lcg(&mut seed) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0)
                .collect();
            let bytes = quantize_ko(&w, nrows, ncols, dt);
            let m = KoMatrix::new(&bytes, nrows, ncols, ko);
            for row in 0..nrows {
                for kb in 0..m.k_blocks() {
                    let expect = m.row(row, kb).0.unpack();
                    let mut got = [0u8; TILE];
                    // SAFETY: the tier was detected; the chunk is whole and
                    // `got` holds two vectors.
                    unsafe {
                        let v = unpack_row_512(ko, m.chunk_ptr(row / 8, kb), row % 8);
                        _mm512_storeu_si512(got.as_mut_ptr().cast(), v[0]);
                        _mm512_storeu_si512(got.as_mut_ptr().add(64).cast(), v[1]);
                    }
                    assert_eq!(got, expect, "{ko:?} row {row} kb {kb}");
                }
            }
        }
    }
}
