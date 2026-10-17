//! The projection matvec, by instruction-set tier.
//!
//! Every tier computes exactly what [`matvec_scalar`](super::ko::matvec_scalar)
//! does: the per-block int32 sum is exact in any order, and the float fold is
//! the same expression in the same order (Rust never contracts it into an FMA
//! on its own), so the tiers agree with the reference byte for byte — which is
//! what their tests assert.
//!
//! | Tier | Dot | Machines |
//! |---|---|---|
//! | `Avx512Vnni` | `vpdpbusd` (EVEX, 512-bit; `simd512`) | Zen 4/5 (7970X, 9950X3D), Ice Lake+ Xeons |
//! | `Avx2` | `vpmaddubsw` + `vpmaddwd` | i7-10700K |
//! | `AvxVnni` | `vpdpbusd` (VEX, 256-bit) | Core Ultra 185H |
//! | `Scalar` | the reference | anything else |
//!
//! [`Isa::detect`] takes the widest a CPU has: AVX-512 VNNI first — a CPU
//! with it need not report the VEX `avxvnni` flag (Zen 4 does not), so the
//! 256-bit tiers would leave it on AVX2.
//!
//! Codes unpack to u8 0..15 and activations are s8, which is exactly the
//! operand pair both dot instructions take; `vpmaddubsw`'s i16 pairs cannot
//! saturate (15·127·2 = 3,810).

use std::ops::Range;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::act::{ActTiles, TILE};
#[cfg(test)]
use super::ko::RowCodes;
#[cfg(target_arch = "x86_64")]
use super::ko::Q3_HI_BASE;
use super::ko::{matvec_scalar, permute, Ko, KoMatrix, Permuted};
#[cfg(target_arch = "x86_64")]
use super::simd512::{dot_512, load_act_512, unpack_row_512};

/// The instruction set a matvec runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Avx2,
    AvxVnni,
    Avx512Vnni,
}

impl Isa {
    /// The fastest tier this CPU runs.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if Isa::Avx512Vnni.available() {
                return Isa::Avx512Vnni;
            }
            if Isa::AvxVnni.available() {
                return Isa::AvxVnni;
            }
            if Isa::Avx2.available() {
                return Isa::Avx2;
            }
        }
        Isa::Scalar
    }

    /// Whether this CPU can run `self`.
    pub fn available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c"),
            #[cfg(target_arch = "x86_64")]
            Isa::AvxVnni => {
                is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("f16c")
                    && is_x86_feature_detected!("avxvnni")
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => {
                is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("f16c")
                    && is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

/// The SIMD tiers this CPU runs, for the byte-equality test named `test`.
/// Each one it lacks is reported as skipped, so a pass on a machine without a
/// tier says which tiers it did not check.
#[cfg(test)]
pub(crate) fn tiers_under_test(test: &str) -> Vec<Isa> {
    [Isa::Avx2, Isa::AvxVnni, Isa::Avx512Vnni]
        .into_iter()
        .filter(|isa| {
            let runs = isa.available();
            if !runs {
                eprintln!("{test}: {isa:?} skipped — this CPU does not have it");
            }
            runs
        })
        .collect()
}

/// `out[tok · rows.len() + (row − rows.start)]` for every token and row, on
/// `isa`. `pa` is `act` permuted for `m`'s KO width ([`Permuted::new`]).
pub fn matvec(
    isa: Isa,
    m: &KoMatrix,
    act: &ActTiles,
    pa: &Permuted,
    rows: Range<usize>,
    out: &mut [f32],
) {
    assert_eq!(out.len(), act.tokens * rows.len());
    let stride = rows.len();
    // SAFETY: `out` holds `tokens · rows.len()` floats, exactly what a stride
    // of `rows.len()` addresses.
    unsafe { matvec_into(isa, m, act, pa, rows, out.as_mut_ptr(), stride) }
}

/// The matvec written to `out[tok · stride + (row − rows.start)]`, so a pool
/// task writes its row range of a buffer whose other ranges are other tasks'.
///
/// # Safety
/// `out` must be valid for writes at every `tok · stride + i`, `i <
/// rows.len()`, `tok < act.tokens`, and nothing else may access those floats
/// during the call.
pub unsafe fn matvec_into(
    isa: Isa,
    m: &KoMatrix,
    act: &ActTiles,
    pa: &Permuted,
    rows: Range<usize>,
    out: *mut f32,
    stride: usize,
) {
    assert!(isa.available(), "{isa:?} is not available on this CPU");
    assert_eq!(act.tiles, m.k_blocks());
    match isa {
        Isa::Scalar => {
            let mut tmp = vec![0f32; act.tokens * rows.len()];
            let n = rows.len();
            matvec_scalar(m, act, rows, &mut tmp);
            for tok in 0..act.tokens {
                for i in 0..n {
                    *out.add(tok * stride + i) = tmp[tok * n + i];
                }
            }
        }
        // SAFETY: the tier's features were checked just above.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => matvec_avx2(m, act, pa, rows, out, stride),
        #[cfg(target_arch = "x86_64")]
        Isa::AvxVnni => matvec_vnni(m, act, pa, rows, out, stride),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Vnni => matvec_avx512vnni(m, act, pa, rows, out, stride),
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!("checked by `available`"),
    }
}

/// A row's 128 codes for one block, in unpack order, as four 32-byte vectors
/// — the tests' way to check [`unpack_row`] against [`RowCodes::unpack`].
#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
fn unpack(codes: &RowCodes) -> [__m256i; 4] {
    // SAFETY: the arrays hold the bytes each reads.
    unsafe {
        match *codes {
            RowCodes::Q3 { crumb, hi } => unpack_q3(crumb.as_ptr(), hi.as_ptr()),
            RowCodes::Q4 { ql } => unpack_q4(ql.as_ptr()),
        }
    }
}

/// Row `r` of the chunk at `chunk`, unpacked as [`unpack`] does.
///
/// # Safety
/// `chunk` is a whole chunk of `ko` and `r < 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_row(ko: Ko, chunk: *const u8, r: usize) -> [__m256i; 4] {
    match ko {
        Ko::Q3 => unpack_q3(chunk.add(r * 32), chunk.add(Q3_HI_BASE + r * 16)),
        Ko::Q4 => unpack_q4(chunk.add(r * 64)),
    }
}

/// # Safety
/// 32 readable bytes at `crumb`, 16 at `hi`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_q3(crumb: *const u8, hi: *const u8) -> [__m256i; 4] {
    let c = _mm256_loadu_si256(crumb.cast());
    let h = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi.cast()));
    // Crumb byte b's hi bits live in hi byte (b/8)·4 + (b%8)/2.
    let g = _mm256_shuffle_epi8(h, Q3_HI_INDEX.load());
    // The same bits two places up, for bit pairs 0 and 1 (`Q3_HI_MASK`).
    let g2 = _mm256_slli_epi16::<2>(g);
    let three = _mm256_set1_epi8(3);
    let four = _mm256_set1_epi8(4);
    let mut out = [
        _mm256_and_si256(c, three),
        _mm256_and_si256(_mm256_srli_epi16::<2>(c), three),
        _mm256_and_si256(_mm256_srli_epi16::<4>(c), three),
        _mm256_and_si256(_mm256_srli_epi16::<6>(c), three),
    ];
    for (s, o) in out.iter_mut().enumerate() {
        let src = if s < 2 { g2 } else { g };
        // The masked bit is 4 or more when set, 0 when not: min with 4 is
        // the code's bit 2.
        let bit = _mm256_min_epu8(_mm256_and_si256(src, Q3_HI_MASK[s].load()), four);
        *o = _mm256_or_si256(*o, bit);
    }
    out
}

/// # Safety
/// 64 readable bytes at `ql`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_q4(ql: *const u8) -> [__m256i; 4] {
    let l0 = _mm256_loadu_si256(ql.cast());
    let l1 = _mm256_loadu_si256(ql.add(32).cast());
    let f = _mm256_set1_epi8(0x0F);
    [
        _mm256_and_si256(l0, f),
        _mm256_and_si256(l1, f),
        _mm256_and_si256(_mm256_srli_epi16::<4>(l0), f),
        _mm256_and_si256(_mm256_srli_epi16::<4>(l1), f),
    ]
}

/// Chunks ahead of the one being computed that are prefetched, in the order
/// the loop reads them: a row group's K blocks, then the next group's. A row
/// group's chunks are a whole row of chunks apart — 33 KB in a gate
/// projection, 133 KB in a down — so the hardware's streamers, which follow a
/// page, never see them coming; and a run of rows is a handful of groups, so
/// a lookahead that stopped at a group's end would leave every group to start
/// on misses.
#[cfg(target_arch = "x86_64")]
const PREFETCH_AHEAD: usize = 4;

/// The chunk `by` reads after `(g, kb)` among groups `..g_end`, if there is
/// one.
#[cfg(target_arch = "x86_64")]
#[inline]
fn ahead_of(g: usize, kb: usize, kbs: usize, g_end: usize, by: usize) -> Option<(usize, usize)> {
    // Subtraction rather than a divide per chunk: `by` is a few K blocks, so
    // this carries into the next group or two at most.
    let (mut g, mut kb) = (g, kb + by);
    while kb >= kbs {
        kb -= kbs;
        g += 1;
    }
    (g < g_end).then_some((g, kb))
}

/// Pull the `bytes` at `p` toward L1.
///
/// # Safety
/// None beyond `p..p + bytes` being one allocation's addresses: a prefetch
/// never faults.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn prefetch(p: *const u8, bytes: usize) {
    let mut o = 0;
    while o < bytes {
        _mm_prefetch::<_MM_HINT_T0>(p.add(o).cast());
        o += 64;
    }
    _mm_prefetch::<_MM_HINT_T0>(p.add(bytes - 1).cast());
}

/// One activation tile permuted into `ko`'s unpack order — [`permute`] with
/// `ko.perm()`, on AVX2 where this CPU has it.
///
/// [`permute`]: super::ko::permute
pub fn permute_tile(ko: Ko, q: &[i8]) -> [i8; TILE] {
    assert_eq!(q.len(), TILE);
    #[cfg(target_arch = "x86_64")]
    if Isa::Avx2.available() {
        // SAFETY: AVX2 just detected; `q` is a whole tile.
        return unsafe {
            match ko {
                Ko::Q3 => permute_q3_avx2(q.as_ptr()),
                Ko::Q4 => permute_q4_avx2(q.as_ptr()),
            }
        };
    }
    permute(q, ko.perm())
}

/// The 4×4 dword transpose inside each 128-bit lane: dword `d` of input `v`
/// becomes dword `v` of output `d`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_dwords(v: [__m256i; 4]) -> [__m256i; 4] {
    let t0 = _mm256_unpacklo_epi32(v[0], v[1]);
    let t1 = _mm256_unpackhi_epi32(v[0], v[1]);
    let t2 = _mm256_unpacklo_epi32(v[2], v[3]);
    let t3 = _mm256_unpackhi_epi32(v[2], v[3]);
    [
        _mm256_unpacklo_epi64(t0, t2),
        _mm256_unpackhi_epi64(t0, t2),
        _mm256_unpacklo_epi64(t1, t3),
        _mm256_unpackhi_epi64(t1, t3),
    ]
}

/// In each 128-bit lane, byte `4a + b` to byte `4b + a`.
#[cfg(target_arch = "x86_64")]
static TRANSPOSE_4X4_BYTES: V32 = V32(build_transpose_4x4());
#[cfg(target_arch = "x86_64")]
const fn build_transpose_4x4() -> [u8; 32] {
    let mut t = [0u8; 32];
    let mut p = 0;
    while p < 32 {
        let q = p % 16;
        t[p] = ((q % 4) * 4 + q / 4) as u8;
        p += 1;
    }
    t
}

/// Q3's permutation (`ko::Q3_PERM`). The tile `k = sub·32 + half·16 + q3·4 +
/// s` is read as `[sub][half][q3][s]` and written as `[s][q3][sub][half]`:
/// input vector `sub` holds `half` in its lane and `(q3, s)` in the lane's
/// bytes; a byte transpose makes that `(s, q3)`, a dword transpose gathers
/// each `s` from the four vectors as `[half][sub][q3]`, a second byte
/// transpose makes the lane `(q3, sub)`, and interleaving the two lanes' bytes
/// puts `half` last.
///
/// # Safety
/// AVX2, and 128 readable bytes at `q`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn permute_q3_avx2(q: *const i8) -> [i8; TILE] {
    let t44 = TRANSPOSE_4X4_BYTES.load();
    let x = transpose_dwords([
        _mm256_shuffle_epi8(_mm256_loadu_si256(q.cast()), t44),
        _mm256_shuffle_epi8(_mm256_loadu_si256(q.add(32).cast()), t44),
        _mm256_shuffle_epi8(_mm256_loadu_si256(q.add(64).cast()), t44),
        _mm256_shuffle_epi8(_mm256_loadu_si256(q.add(96).cast()), t44),
    ]);
    let mut out = [0i8; TILE];
    for (s, xs) in x.iter().enumerate() {
        let y = _mm256_shuffle_epi8(*xs, t44);
        // [A.lo, A.hi | B.lo, B.hi] → [A.lo, B.lo | A.hi, B.hi], then each
        // lane's two halves interleaved byte by byte.
        let z = _mm256_permute4x64_epi64::<0xD8>(y);
        let o = _mm256_unpacklo_epi8(z, _mm256_srli_si256::<8>(z));
        _mm256_storeu_si256(out.as_mut_ptr().add(s * 32).cast(), o);
    }
    out
}

/// Q4's permutation (`ko::Q4_PERM`): whole dwords, `[sub][high][q3]` to
/// `[high][q3][sub]` — a dword transpose inside the lanes, then the lanes
/// paired across vectors.
///
/// # Safety
/// AVX2, and 128 readable bytes at `q`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn permute_q4_avx2(q: *const i8) -> [i8; TILE] {
    let x = transpose_dwords([
        _mm256_loadu_si256(q.cast()),
        _mm256_loadu_si256(q.add(32).cast()),
        _mm256_loadu_si256(q.add(64).cast()),
        _mm256_loadu_si256(q.add(96).cast()),
    ]);
    let o = [
        _mm256_permute2x128_si256::<0x20>(x[0], x[1]),
        _mm256_permute2x128_si256::<0x20>(x[2], x[3]),
        _mm256_permute2x128_si256::<0x31>(x[0], x[1]),
        _mm256_permute2x128_si256::<0x31>(x[2], x[3]),
    ];
    let mut out = [0i8; TILE];
    for (w, ow) in o.iter().enumerate() {
        _mm256_storeu_si256(out.as_mut_ptr().add(w * 32).cast(), *ow);
    }
    out
}

/// A 32-byte constant, loadable as a vector.
#[cfg(target_arch = "x86_64")]
#[repr(align(32))]
struct V32([u8; 32]);

#[cfg(target_arch = "x86_64")]
impl V32 {
    #[target_feature(enable = "avx2")]
    fn load(&self) -> __m256i {
        // SAFETY: 32 aligned bytes.
        unsafe { _mm256_load_si256(self.0.as_ptr().cast()) }
    }
}

#[cfg(target_arch = "x86_64")]
static Q3_HI_INDEX: V32 = V32(build_hi_index());
#[cfg(target_arch = "x86_64")]
pub(super) const fn build_hi_index() -> [u8; 32] {
    let mut t = [0u8; 32];
    let mut b = 0;
    while b < 32 {
        t[b] = ((b / 8) * 4 + (b % 8) / 2) as u8;
        b += 1;
    }
    t
}

/// For bit pair `s`: crumb byte `b`'s hi bit is bit `(b%2)·4 + s` of its hi
/// byte. Pairs 0 and 1 read it from the hi bytes shifted two places up (an
/// `epi16` shift, which carries an even byte's top bits into its odd
/// neighbour's bits 0–1 — below every mask here), so that every masked bit
/// is worth at least 4.
#[cfg(target_arch = "x86_64")]
static Q3_HI_MASK: [V32; 4] = [
    V32(build_hi_mask(0, 2)),
    V32(build_hi_mask(1, 2)),
    V32(build_hi_mask(2, 0)),
    V32(build_hi_mask(3, 0)),
];
#[cfg(target_arch = "x86_64")]
const fn build_hi_mask(s: usize, up: usize) -> [u8; 32] {
    let mut t = [0u8; 32];
    let mut b = 0;
    while b < 32 {
        t[b] = 1u8 << ((b % 2) * 4 + s + up);
        b += 1;
    }
    t
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn load_act(a: &[i8; TILE]) -> [__m256i; 4] {
    // SAFETY: 128 readable bytes.
    unsafe {
        let p = a.as_ptr();
        [
            _mm256_loadu_si256(p.cast()),
            _mm256_loadu_si256(p.add(32).cast()),
            _mm256_loadu_si256(p.add(64).cast()),
            _mm256_loadu_si256(p.add(96).cast()),
        ]
    }
}

/// One row's dot with one token's tile, left in eight int32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn dot_avx2(w: &[__m256i; 4], a: &[__m256i; 4]) -> __m256i {
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    for s in 0..4 {
        let p = _mm256_maddubs_epi16(w[s], a[s]);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
fn dot_vnni(w: &[__m256i; 4], a: &[__m256i; 4]) -> __m256i {
    let mut acc = _mm256_setzero_si256();
    for s in 0..4 {
        acc = _mm256_dpbusd_avx_epi32(acc, w[s], a[s]);
    }
    acc
}

/// Eight rows' lane accumulators → one vector of the eight row sums, row `r`
/// in lane `r`. Integer adds, so the order is free.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn transpose_reduce(a: &[__m256i; 8]) -> __m256i {
    let t0 = _mm256_hadd_epi32(a[0], a[1]);
    let t1 = _mm256_hadd_epi32(a[2], a[3]);
    let t2 = _mm256_hadd_epi32(a[4], a[5]);
    let t3 = _mm256_hadd_epi32(a[6], a[7]);
    let u0 = _mm256_hadd_epi32(t0, t1);
    let u1 = _mm256_hadd_epi32(t2, t3);
    _mm256_add_epi32(
        _mm256_permute2x128_si256::<0x20>(u0, u1),
        _mm256_permute2x128_si256::<0x31>(u0, u1),
    )
}

/// A chunk's eight `(scale, min)` pairs as two vectors, row `r` in lane `r`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
fn load_dm(dm: &[u8; 32]) -> (__m256, __m256) {
    // SAFETY: 32 readable bytes.
    let (lo, hi) = unsafe {
        (
            _mm256_cvtph_ps(_mm_loadu_si128(dm.as_ptr().cast())),
            _mm256_cvtph_ps(_mm_loadu_si128(dm.as_ptr().add(16).cast())),
        )
    };
    // [s0 m0 s1 m1 s2 m2 s3 m3] → [s0 s1 s2 s3 m0 m1 m2 m3]
    let idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let (lo, hi) = (
        _mm256_permutevar8x32_ps(lo, idx),
        _mm256_permutevar8x32_ps(hi, idx),
    );
    (
        _mm256_permute2f128_ps::<0x20>(lo, hi),
        _mm256_permute2f128_ps::<0x31>(lo, hi),
    )
}

/// One K block's eight-row term: `(scale · s_a) · isum + min · Σx`, lane by
/// lane in the scalar reference's operation order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn fold_term(scale8: __m256, min8: __m256, s_a: f32, sum: f32, isum: __m256) -> __m256 {
    _mm256_add_ps(
        _mm256_mul_ps(_mm256_mul_ps(scale8, _mm256_set1_ps(s_a)), isum),
        _mm256_mul_ps(min8, _mm256_set1_ps(sum)),
    )
}

/// The tier loop, one body for every tier. Rows go eight at a time — one
/// chunk's rows, which share their `(scale, min)` block — and each token's
/// eight dots reduce in one transpose. Every row folds `out += term`, K blocks
/// ascending, from 0.
///
/// One token, decode's shape, dots each row straight from its unpack and
/// keeps the eight rows' sums in a register across the K blocks. More tokens
/// unpack a chunk once into an L1 buffer that every token's dots read.
///
/// A tier supplies only its vector width's three pieces: `$unpack` — a row of
/// a chunk as `$n` vectors in unpack order — `$load_act`, a permuted tile as
/// the same `$n` vectors, and `$dot`, their product reduced to eight int32
/// lanes. Everything else, prefetch, reduction and fold included, is shared,
/// so the tiers this machine runs test the loop every tier runs.
macro_rules! matvec_tier {
    (
        $name:ident,
        $features:literal,
        $zero:expr,
        $n:literal,
        $unpack:ident,
        $load_act:ident,
        $dot:ident
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $features)]
        unsafe fn $name(
            m: &KoMatrix,
            act: &ActTiles,
            pa: &Permuted,
            rows: Range<usize>,
            out: *mut f32,
            stride: usize,
        ) {
            assert!(
                rows.start.is_multiple_of(8) && rows.len().is_multiple_of(8) && rows.end <= m.nrows,
                "row ranges are whole chunks of the projection"
            );
            let (ko, kbs) = (m.ko, m.k_blocks());
            let (chunk_bytes, dm_at) = (ko.chunk_bytes(), ko.dm_base());
            let g_end = rows.end / 8;
            // The run's first chunks, all in flight at once rather than each
            // missing in turn.
            for by in 1..PREFETCH_AHEAD {
                if let Some((pg, pkb)) = ahead_of(rows.start / 8, 0, kbs, g_end, by) {
                    prefetch(m.chunk_ptr(pg, pkb), chunk_bytes);
                }
            }
            let mut buf = [[$zero; $n]; 8];
            for g0 in (rows.start..rows.end).step_by(8) {
                let (g, ri0) = (g0 / 8, g0 - rows.start);
                if act.tokens == 1 {
                    let mut acc = _mm256_setzero_ps();
                    for kb in 0..kbs {
                        if let Some((pg, pkb)) = ahead_of(g, kb, kbs, g_end, PREFETCH_AHEAD) {
                            prefetch(m.chunk_ptr(pg, pkb), chunk_bytes);
                        }
                        let p = m.chunk_ptr(g, kb);
                        let a = $load_act(pa.tile(0, kb));
                        let mut accs = [_mm256_setzero_si256(); 8];
                        for (r, slot) in accs.iter_mut().enumerate() {
                            *slot = $dot(&$unpack(ko, p, r), &a);
                        }
                        let isum = _mm256_cvtepi32_ps(transpose_reduce(&accs));
                        let (scale8, min8) = load_dm(&*p.add(dm_at).cast::<[u8; 32]>());
                        let (_, s_a, sum) = act.tile(0, kb);
                        acc = _mm256_add_ps(acc, fold_term(scale8, min8, s_a, sum, isum));
                    }
                    _mm256_storeu_ps(out.add(ri0), acc);
                    continue;
                }
                for tok in 0..act.tokens {
                    _mm256_storeu_ps(out.add(tok * stride + ri0), _mm256_setzero_ps());
                }
                for kb in 0..kbs {
                    if let Some((pg, pkb)) = ahead_of(g, kb, kbs, g_end, PREFETCH_AHEAD) {
                        prefetch(m.chunk_ptr(pg, pkb), chunk_bytes);
                    }
                    let p = m.chunk_ptr(g, kb);
                    for (r, w) in buf.iter_mut().enumerate() {
                        *w = $unpack(ko, p, r);
                    }
                    let (scale8, min8) = load_dm(&*p.add(dm_at).cast::<[u8; 32]>());
                    for tok in 0..act.tokens {
                        let a = $load_act(pa.tile(tok, kb));
                        let mut accs = [_mm256_setzero_si256(); 8];
                        for (acc, w) in accs.iter_mut().zip(&buf) {
                            *acc = $dot(w, &a);
                        }
                        let isum = _mm256_cvtepi32_ps(transpose_reduce(&accs));
                        let (_, s_a, sum) = act.tile(tok, kb);
                        let o = out.add(tok * stride + ri0);
                        let term = fold_term(scale8, min8, s_a, sum, isum);
                        _mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), term));
                    }
                }
            }
        }
    };
}

matvec_tier!(
    matvec_avx2,
    "avx2,f16c",
    _mm256_setzero_si256(),
    4,
    unpack_row,
    load_act,
    dot_avx2
);
matvec_tier!(
    matvec_vnni,
    "avx2,f16c,avxvnni",
    _mm256_setzero_si256(),
    4,
    unpack_row,
    load_act,
    dot_vnni
);
matvec_tier!(
    matvec_avx512vnni,
    "avx2,f16c,avx512f,avx512bw,avx512vnni",
    _mm512_setzero_si512(),
    2,
    unpack_row_512,
    load_act_512,
    dot_512
);

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::ko_quant::quantize_ko;
    use candle::quantized::GgmlDType;
    use half::f16;
    use std::array;
    use std::hint::black_box;
    use std::time::Instant;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// The SIMD unpack equals the scalar unpack, byte for byte, for every row
    /// of random Q3_KO and Q4_KO projections.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_unpack_matches_scalar() {
        if !Isa::Avx2.available() {
            eprintln!("simd_unpack_matches_scalar: skipped — this CPU has no AVX2");
            return;
        }
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let (nrows, ncols) = (32, 256);
            let mut seed = 3;
            let w: Vec<f32> = (0..nrows * ncols).map(|_| lcg(&mut seed)).collect();
            let bytes = quantize_ko(&w, nrows, ncols, dt);
            let m = KoMatrix::new(&bytes, nrows, ncols, ko);
            for row in 0..nrows {
                for kb in 0..m.k_blocks() {
                    let (codes, _, _) = m.row(row, kb);
                    let expect = codes.unpack();
                    // SAFETY: AVX2 checked above.
                    let v = unsafe { unpack(&codes) };
                    let mut got = [0u8; TILE];
                    for (s, vec) in v.iter().enumerate() {
                        // SAFETY: 32 writable bytes.
                        unsafe { _mm256_storeu_si256(got[s * 32..].as_mut_ptr().cast(), *vec) };
                    }
                    assert_eq!(got, expect, "{ko:?} row {row} kb {kb}");
                }
            }
        }
    }

    /// The bare matvec on one thread: a Flash-Next gate projection (640 ×
    /// 2560, Q3_KO, ~0.7 MB — L2/L3-hot after the first pass) at 1, 2 and 4
    /// tokens, best of many passes, as weights per second per core.
    ///
    /// `cargo test --release -p candle-transformers --features cuda --lib matvec_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement; read its output"]
    fn matvec_throughput() {
        let (nrows, ncols) = (640, 2560);
        let mut seed = 9;
        let w: Vec<f32> = (0..nrows * ncols).map(|_| lcg(&mut seed)).collect();
        let bytes = quantize_ko(&w, nrows, ncols, GgmlDType::Q3_KO);
        let m = KoMatrix::new(&bytes, nrows, ncols, Ko::Q3);
        let isa = Isa::detect();
        for tokens in [1usize, 2, 4] {
            let x: Vec<f32> = (0..tokens * ncols).map(|_| lcg(&mut seed)).collect();
            let act = ActTiles::quantize(&x, tokens, ncols);
            let pa = Permuted::new(&act, Ko::Q3);
            let mut out = vec![0f32; tokens * nrows];
            let mut best = f64::MAX;
            for _ in 0..200 {
                let t = Instant::now();
                matvec(isa, &m, &act, &pa, 0..nrows, &mut out);
                best = best.min(t.elapsed().as_secs_f64());
                black_box(&out);
            }
            println!(
                "{isa:?} tokens {tokens}: {:.1} µs, {:.1} G weights/s, {:.1} GB/s of weights",
                best * 1e6,
                (nrows * ncols) as f64 / best / 1e9,
                bytes.len() as f64 / best / 1e9
            );
        }
    }

    /// A NaN in one row's `(scale, min)` — at one K block — poisons that
    /// row's result and no other, on every tier as on the scalar reference,
    /// at one token and several: every other row stays bit for bit. (A NaN's
    /// payload is not compared; that it is NaN is.)
    #[test]
    fn a_nan_weight_scale_poisons_only_its_row() {
        let tiers = tiers_under_test("a_nan_weight_scale_poisons_only_its_row");
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let (nrows, ncols) = (32, 512);
            let mut seed = 17;
            let w: Vec<f32> = (0..nrows * ncols).map(|_| lcg(&mut seed)).collect();
            let mut bytes = quantize_ko(&w, nrows, ncols, dt);
            let nan = f16::NAN.to_le_bytes();
            // Row 11's scale at K block 2, row 20's min at K block 0.
            for (row, kb, field) in [(11usize, 2usize, 0usize), (20, 0, 2)] {
                let chunk = (kb * (nrows / 8) + row / 8) * ko.chunk_bytes();
                let at = chunk + ko.dm_base() + (row % 8) * 4 + field;
                bytes[at..at + 2].copy_from_slice(&nan);
            }
            let m = KoMatrix::new(&bytes, nrows, ncols, ko);
            for tokens in [1usize, 3] {
                let x: Vec<f32> = (0..tokens * ncols).map(|_| lcg(&mut seed)).collect();
                let act = ActTiles::quantize(&x, tokens, ncols);
                let pa = Permuted::new(&act, ko);
                let mut expect = vec![0f32; tokens * nrows];
                matvec_scalar(&m, &act, 0..nrows, &mut expect);
                for &isa in &tiers {
                    let mut got = vec![0f32; tokens * nrows];
                    matvec(isa, &m, &act, &pa, 0..nrows, &mut got);
                    for tok in 0..tokens {
                        for row in 0..nrows {
                            let (g, e) = (got[tok * nrows + row], expect[tok * nrows + row]);
                            if row == 11 || row == 20 {
                                assert!(g.is_nan() && e.is_nan(), "{ko:?} {isa:?} row {row}");
                            } else {
                                assert_eq!(g.to_bits(), e.to_bits(), "{ko:?} {isa:?} row {row}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The vector permutation of every tile equals the scalar definition,
    /// byte for byte, for both widths — over random tiles and the tile whose
    /// every byte is its own index, which names each output's source.
    #[test]
    fn permute_tile_is_the_scalar_permutation() {
        if !Isa::Avx2.available() {
            eprintln!(
                "permute_tile_is_the_scalar_permutation: skipped — this CPU has no AVX2, so \
                 `permute_tile` is the scalar permutation itself"
            );
            return;
        }
        let mut seed = 13u64;
        let mut tiles: Vec<[i8; TILE]> = (0..32)
            .map(|_| array::from_fn(|_| (lcg(&mut seed) * 127.0) as i8))
            .collect();
        tiles.push(array::from_fn(|i| i as i8));
        for ko in [Ko::Q3, Ko::Q4] {
            for t in &tiles {
                assert_eq!(permute_tile(ko, t), permute(t, ko.perm()), "{ko:?}");
            }
        }
    }

    /// The lookahead walks a group's K blocks, then the next group's, and
    /// stops at the run's last group — with a down projection's five K
    /// blocks it carries a whole group.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn prefetch_lookahead_carries_into_the_next_group() {
        assert_eq!(ahead_of(3, 0, 20, 9, 4), Some((3, 4)));
        assert_eq!(ahead_of(3, 17, 20, 9, 4), Some((4, 1)));
        assert_eq!(ahead_of(8, 17, 20, 9, 4), None);
        assert_eq!(ahead_of(0, 4, 5, 9, 4), Some((1, 3)));
        assert_eq!(ahead_of(0, 4, 2, 9, 4), Some((4, 0)));
        assert_eq!(ahead_of(7, 2, 5, 9, 9), None);
    }

    /// Every available tier's matvec equals the scalar reference byte for
    /// byte, over several token counts and a row range that is not the whole.
    #[test]
    fn tiers_match_scalar_bytes() {
        let tiers = tiers_under_test("tiers_match_scalar_bytes");
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let (nrows, ncols) = (64, 640);
            let mut seed = 5;
            let w: Vec<f32> = (0..nrows * ncols).map(|_| lcg(&mut seed)).collect();
            let bytes = quantize_ko(&w, nrows, ncols, dt);
            let m = KoMatrix::new(&bytes, nrows, ncols, ko);
            for tokens in [1usize, 2, 5, 9] {
                let x: Vec<f32> = (0..tokens * ncols).map(|_| lcg(&mut seed)).collect();
                let act = ActTiles::quantize(&x, tokens, ncols);
                let pa = Permuted::new(&act, ko);
                let rows = 8..56;
                let mut expect = vec![0f32; tokens * rows.len()];
                matvec_scalar(&m, &act, rows.clone(), &mut expect);
                for &isa in &tiers {
                    let mut got = vec![0f32; tokens * rows.len()];
                    matvec(isa, &m, &act, &pa, rows.clone(), &mut got);
                    let (gb, eb): (Vec<u32>, Vec<u32>) = (
                        got.iter().map(|v| v.to_bits()).collect(),
                        expect.iter().map(|v| v.to_bits()).collect(),
                    );
                    assert_eq!(gb, eb, "{ko:?} {isa:?} tokens {tokens}");
                }
            }
        }
    }
}
