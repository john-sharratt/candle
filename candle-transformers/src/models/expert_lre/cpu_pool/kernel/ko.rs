//! The weight side: one KO projection as the device stores it, read in place.
//!
//! A KO projection is `[K block][row group]` chunks of 8 rows × 128 K, each
//! chunk's planes lane-major (`candle::quantized::ko_quant`). Within a chunk a
//! row's codes are contiguous:
//!
//! | Kind | Row bytes | Layout |
//! |---|---|---|
//! | Q3_KO | 32 crumb + 16 hi | crumb byte `q3·8 + sub·2 + half`, 2 bits at `2i`; hi byte `q3·4 + sub`, nibble `half`, bit `i` |
//! | Q4_KO | 64 ql | byte `q3·16 + sub·4 + i`, low nibble `k`, high nibble `k + 16` |
//!
//! with `k = sub·32 + half·16 + q3·4 + i`. Each row of a K block then has an
//! affine `(scale, min)` in f16, and the value is `scale·code + min`.
//!
//! The SIMD unpack ([`super::simd`]) yields a row's 128 codes in a fixed
//! *unpack order* `u`, not in `k` order. Rather than unshuffle every weight,
//! the activation tile is permuted once into that order ([`Ko::perm`]) — the
//! int32 sum over a block does not depend on the order it is taken in.

use std::ops::Range;

use half::f16;

use super::act::{ActTiles, TILE};
use super::simd::permute_tile;

/// Where a Q3 chunk's hi bytes start: after its eight rows' 32 crumb bytes.
pub const Q3_HI_BASE: usize = 256;

/// The KO widths the expert path carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ko {
    Q3,
    Q4,
}

impl Ko {
    pub const fn chunk_bytes(self) -> usize {
        match self {
            Ko::Q3 => 256 + 128 + 32,
            Ko::Q4 => 512 + 32,
        }
    }

    /// Where a chunk's eight `(scale, min)` pairs start.
    pub const fn dm_base(self) -> usize {
        self.chunk_bytes() - 32
    }

    /// `perm[u]` is the `k` of unpack-order position `u`.
    pub fn perm(self) -> &'static [u8; TILE] {
        match self {
            Ko::Q3 => &Q3_PERM,
            Ko::Q4 => &Q4_PERM,
        }
    }
}

/// Q3: four 32-byte vectors, vector `s` holding bit pair `s` of each crumb
/// byte `b` — `u = s·32 + b`.
static Q3_PERM: [u8; TILE] = build_q3_perm();
const fn build_q3_perm() -> [u8; TILE] {
    let mut t = [0u8; TILE];
    let mut u = 0;
    while u < TILE {
        let (s, b) = (u / 32, u % 32);
        let (q3, sub, half) = (b / 8, (b % 8) / 2, b % 2);
        t[u] = (sub * 32 + half * 16 + q3 * 4 + s) as u8;
        u += 1;
    }
    t
}

/// Q4: low nibbles of bytes 0..64 (`u < 64`), then their high nibbles.
static Q4_PERM: [u8; TILE] = build_q4_perm();
const fn build_q4_perm() -> [u8; TILE] {
    let mut t = [0u8; TILE];
    let mut u = 0;
    while u < TILE {
        let (j, high) = (u % 64, u >= 64);
        let (q3, sub, i) = (j / 16, (j % 16) / 4, j % 4);
        t[u] = (sub * 32 + q3 * 4 + i + if high { 16 } else { 0 }) as u8;
        u += 1;
    }
    t
}

/// One projection `[nrows × ncols]` in place.
#[derive(Debug, Clone, Copy)]
pub struct KoMatrix<'a> {
    pub bytes: &'a [u8],
    pub nrows: usize,
    pub ncols: usize,
    pub ko: Ko,
}

impl<'a> KoMatrix<'a> {
    pub fn new(bytes: &'a [u8], nrows: usize, ncols: usize, ko: Ko) -> Self {
        // Empty is refused too: the tiers' prefetch walks K blocks ahead, and
        // with none there is nothing to walk.
        assert!(
            nrows > 0 && ncols > 0 && nrows.is_multiple_of(8) && ncols.is_multiple_of(TILE),
            "KO shape {nrows}×{ncols}"
        );
        assert!(bytes.len() >= (ncols / TILE) * (nrows / 8) * ko.chunk_bytes());
        Self {
            bytes,
            nrows,
            ncols,
            ko,
        }
    }

    pub fn k_blocks(&self) -> usize {
        self.ncols / TILE
    }

    /// Row group `g`'s chunk in K block `kb`, as its first byte: the SIMD
    /// tiers' way in, which reads a row's codes at its fixed offsets
    /// (`Q3_HI_BASE`, the module table) rather than slicing each row.
    ///
    /// # Safety
    /// `g < nrows / 8` and `kb < k_blocks()`.
    pub unsafe fn chunk_ptr(&self, g: usize, kb: usize) -> *const u8 {
        debug_assert!(g < self.nrows / 8 && kb < self.k_blocks());
        let base = (kb * (self.nrows / 8) + g) * self.ko.chunk_bytes();
        // SAFETY: in bounds by the caller's contract and `new`'s length check.
        unsafe { self.bytes.as_ptr().add(base) }
    }

    /// Row `row`'s code bytes in K block `kb` (48 for Q3, 64 for Q4, with Q3's
    /// 16 hi bytes following its 32 crumb bytes), and its `(scale, min)`.
    pub fn row(&self, row: usize, kb: usize) -> (RowCodes<'a>, f32, f32) {
        let (g, r) = (row / 8, row % 8);
        let base = (kb * (self.nrows / 8) + g) * self.ko.chunk_bytes();
        let chunk = &self.bytes[base..base + self.ko.chunk_bytes()];
        let d = self.ko.dm_base() + r * 4;
        let scale = f16::from_le_bytes([chunk[d], chunk[d + 1]]).to_f32();
        let min = f16::from_le_bytes([chunk[d + 2], chunk[d + 3]]).to_f32();
        let codes = match self.ko {
            Ko::Q3 => RowCodes::Q3 {
                crumb: chunk[r * 32..r * 32 + 32].try_into().unwrap(),
                hi: chunk[Q3_HI_BASE + r * 16..Q3_HI_BASE + r * 16 + 16]
                    .try_into()
                    .unwrap(),
            },
            Ko::Q4 => RowCodes::Q4 {
                ql: chunk[r * 64..r * 64 + 64].try_into().unwrap(),
            },
        };
        (codes, scale, min)
    }
}

/// One row's codes for one K block, still packed.
#[derive(Debug, Clone, Copy)]
pub enum RowCodes<'a> {
    Q3 {
        crumb: &'a [u8; 32],
        hi: &'a [u8; 16],
    },
    Q4 {
        ql: &'a [u8; 64],
    },
}

impl RowCodes<'_> {
    /// The 128 codes in unpack order `u` — the scalar definition the SIMD
    /// unpack is tested against.
    pub fn unpack(&self) -> [u8; TILE] {
        let mut out = [0u8; TILE];
        match *self {
            RowCodes::Q3 { crumb, hi } => {
                for (u, o) in out.iter_mut().enumerate() {
                    let (s, b) = (u / 32, u % 32);
                    let (q3, sub, half) = (b / 8, (b % 8) / 2, b % 2);
                    let low = (crumb[b] >> (2 * s)) & 3;
                    let high = (hi[q3 * 4 + sub] >> (half * 4 + s)) & 1;
                    *o = low | (high << 2);
                }
            }
            RowCodes::Q4 { ql } => {
                for (u, o) in out.iter_mut().enumerate() {
                    let b = ql[u % 64];
                    *o = if u < 64 { b & 0xF } else { b >> 4 };
                }
            }
        }
        out
    }
}

/// The activation tile permuted into unpack order: `out[u] = q[perm[u]]`.
pub fn permute(q: &[i8], perm: &[u8; TILE]) -> [i8; TILE] {
    let mut out = [0i8; TILE];
    for (o, &k) in out.iter_mut().zip(perm.iter()) {
        *o = q[k as usize];
    }
    out
}

/// The activations of every token, permuted for `ko`, tile by tile — computed
/// once per projection and shared by all of its rows.
pub struct Permuted {
    pub tiles: usize,
    pub tokens: usize,
    /// `[token][tile][128]`.
    pub qs: Vec<[i8; TILE]>,
}

impl Permuted {
    pub fn new(act: &ActTiles, ko: Ko) -> Self {
        let mut qs = Vec::with_capacity(act.tokens * act.tiles);
        for tok in 0..act.tokens {
            for t in 0..act.tiles {
                qs.push(permute_tile(ko, act.tile(tok, t).0));
            }
        }
        Self {
            tiles: act.tiles,
            tokens: act.tokens,
            qs,
        }
    }

    pub fn tile(&self, tok: usize, t: usize) -> &[i8; TILE] {
        &self.qs[tok * self.tiles + t]
    }
}

/// The scalar reference: `out[tok · rows.len() + (row − rows.start)]` for every
/// token and every row in `rows`, folding each K block as
/// `scale·s_a·Σ code·q + min·Σx`, blocks ascending. Built only from the scalar
/// definitions — [`RowCodes::unpack`] and [`permute`] — so a SIMD tier checked
/// against it shares none of its vector code.
pub fn matvec_scalar(m: &KoMatrix, act: &ActTiles, rows: Range<usize>, out: &mut [f32]) {
    assert_eq!(act.tiles, m.k_blocks());
    let n = rows.len();
    assert_eq!(out.len(), act.tokens * n);
    let perm = m.ko.perm();
    for (ri, row) in rows.enumerate() {
        for tok in 0..act.tokens {
            let mut acc = 0f32;
            for kb in 0..m.k_blocks() {
                let (codes, scale, min) = m.row(row, kb);
                let w = codes.unpack();
                let a = permute(act.tile(tok, kb).0, perm);
                let isum: i32 = w.iter().zip(&a).map(|(&c, &q)| c as i32 * q as i32).sum();
                let (_, s_a, sum) = act.tile(tok, kb);
                acc += scale * s_a * isum as f32 + min * sum;
            }
            out[tok * n + ri] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::ko_quant::{dequant_ko, quantize_ko};
    use candle::quantized::GgmlDType;

    pub(crate) fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Both permutations are permutations of 0..128.
    #[test]
    fn perms_are_bijections() {
        for ko in [Ko::Q3, Ko::Q4] {
            let mut seen = [false; TILE];
            for &k in ko.perm() {
                assert!(!seen[k as usize]);
                seen[k as usize] = true;
            }
        }
    }

    /// Unpacking a row and placing each code at `perm[u]` reproduces the codes
    /// `dequant_ko` implies, for every row of a random projection.
    #[test]
    fn unpack_matches_dequant_ko() {
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let (nrows, ncols) = (16, 256);
            let mut seed = 7;
            let w: Vec<f32> = (0..nrows * ncols).map(|_| lcg(&mut seed)).collect();
            let bytes = quantize_ko(&w, nrows, ncols, dt);
            let deq = dequant_ko(&bytes, nrows, ncols, dt);
            let m = KoMatrix::new(&bytes, nrows, ncols, ko);
            for row in 0..nrows {
                for kb in 0..m.k_blocks() {
                    let (codes, scale, min) = m.row(row, kb);
                    let u = codes.unpack();
                    for (ui, &k) in ko.perm().iter().enumerate() {
                        let expect = deq[row * ncols + kb * TILE + k as usize];
                        assert_eq!(
                            scale * u[ui] as f32 + min,
                            expect,
                            "{ko:?} r{row} kb{kb} u{ui}"
                        );
                    }
                }
            }
        }
    }

    /// The scalar matvec against the exact product of the dequantized weights
    /// and the *unquantized* activation: what remains is the activation's own
    /// int8 rounding, well under 1% relative L2.
    #[test]
    fn scalar_matvec_tracks_the_f32_product() {
        for (ko, dt) in [(Ko::Q3, GgmlDType::Q3_KO), (Ko::Q4, GgmlDType::Q4_KO)] {
            let (nrows, ncols, tokens) = (16, 384, 3);
            let mut seed = 11;
            let w: Vec<f32> = (0..nrows * ncols).map(|_| lcg(&mut seed)).collect();
            let x: Vec<f32> = (0..tokens * ncols).map(|_| lcg(&mut seed)).collect();
            let bytes = quantize_ko(&w, nrows, ncols, dt);
            let deq = dequant_ko(&bytes, nrows, ncols, dt);
            let act = ActTiles::quantize(&x, tokens, ncols);
            let m = KoMatrix::new(&bytes, nrows, ncols, ko);
            let mut out = vec![0f32; tokens * nrows];
            matvec_scalar(&m, &act, 0..nrows, &mut out);
            let (mut err, mut norm) = (0f64, 0f64);
            for tok in 0..tokens {
                for row in 0..nrows {
                    let expect: f64 = (0..ncols)
                        .map(|k| deq[row * ncols + k] as f64 * x[tok * ncols + k] as f64)
                        .sum();
                    let got = out[tok * nrows + row] as f64;
                    err += (got - expect).powi(2);
                    norm += expect.powi(2);
                }
            }
            let rel = (err / norm).sqrt();
            assert!(rel < 1e-2, "{ko:?}: rel L2 {rel}");
        }
    }
}
