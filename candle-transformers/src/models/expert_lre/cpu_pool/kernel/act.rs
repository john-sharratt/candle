//! The activation side of a CPU expert projection: q8 tiles of 128.
//!
//! A tile is 128 int8 codes, the scale they were quantized by (`amax / 127`)
//! and the tile's Σx — the term an affine weight's `min` multiplies. The pool
//! receives the gate/up input as the device produced it (`from_q8a1024`), and
//! quantizes its own SwiGLU output for the down projection (`quantize`).

use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Elements per tile.
pub const TILE: usize = 128;

/// Bytes in one q8a1024 super-block: 8 tiles' codes, then their 16-byte
/// `{scale, sum}` slots (`blocks.cuh`).
const Q8A1024_BLOCK: usize = 1152;
const Q8A1024_META: usize = 1024;

/// Σ of one 128-wide tile in the order the device's warp adds it: lane `l`
/// sums its elements `4l..4l+4` left to right, then the lanes combine in an
/// xor butterfly at offsets 16, 8, 4, 2, 1 (each pair's sum is the same either
/// way round, so every lane ends on this value).
fn warp_sum(tile: &[f32]) -> f32 {
    let mut lane: [f32; 32] = std::array::from_fn(|l| {
        let v = &tile[4 * l..4 * l + 4];
        ((v[0] + v[1]) + v[2]) + v[3]
    });
    for off in [16, 8, 4, 2, 1] {
        lane = std::array::from_fn(|l| lane[l] + lane[l ^ off]);
    }
    lane[0]
}

/// One tile's codes `rint(x · 127/amax)` into `qs`, and its amax — on AVX2
/// where this CPU has it, which computes exactly what
/// [`quantize_codes_scalar`] does.
fn quantize_codes(src: &[f32], qs: &mut [i8]) -> f32 {
    assert!(src.len() == TILE && qs.len() == TILE);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 just detected; both slices are whole tiles.
        return unsafe { quantize_codes_avx2(src.as_ptr(), qs.as_mut_ptr()) };
    }
    quantize_codes_scalar(src, qs)
}

/// The definition: amax over `|x|` (a NaN is passed over, as `f32::max`
/// passes it), then each code rounded half to even and clamped, a NaN code
/// being 0 (`as i8`).
fn quantize_codes_scalar(src: &[f32], qs: &mut [i8]) -> f32 {
    let amax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    for (q, &v) in qs.iter_mut().zip(src) {
        *q = (v * id).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    amax
}

/// # Safety
/// AVX2; `TILE` floats at `src`, `TILE` writable bytes at `qs`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_codes_avx2(src: *const f32, qs: *mut i8) -> f32 {
    let sign = _mm256_set1_ps(-0.0);
    let mut m = _mm256_setzero_ps();
    for i in 0..TILE / 8 {
        let a = _mm256_andnot_ps(sign, _mm256_loadu_ps(src.add(i * 8)));
        // `max_ps` returns its second operand when either is NaN: with the
        // running max second, a NaN element is passed over.
        m = _mm256_max_ps(a, m);
    }
    let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_max_ss(m, _mm_shuffle_ps::<1>(m, m));
    let amax = _mm_cvtss_f32(m);
    let id = _mm256_set1_ps(if amax != 0.0 { 127.0 / amax } else { 0.0 });
    // Four vectors of eight i32 → 32 i8 in order: the packs interleave by
    // lane, and the dword permute undoes it.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for g in 0..TILE / 32 {
        let at = src.add(g * 32);
        let p = _mm256_packs_epi32(codes8(at, id), codes8(at.add(8), id));
        let q = _mm256_packs_epi32(codes8(at.add(16), id), codes8(at.add(24), id));
        let b = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p, q), order);
        _mm256_storeu_si256(qs.add(g * 32).cast(), b);
    }
    amax
}

/// Eight codes `clamp(rint(x · id), ±127)` as i32, a NaN's code 0.
///
/// # Safety
/// AVX2; 8 floats at `src`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn codes8(src: *const f32, id: __m256) -> __m256i {
    let x = _mm256_mul_ps(_mm256_loadu_ps(src), id);
    let r = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(x);
    let clamped = _mm256_min_ps(
        _mm256_max_ps(r, _mm256_set1_ps(-127.0)),
        _mm256_set1_ps(127.0),
    );
    let c = _mm256_cvtps_epi32(clamped);
    // A NaN lane is 0, as the scalar cast makes it.
    let nan = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
    _mm256_andnot_si256(nan, c)
}

/// q8 tiles for `tokens` rows of `tiles` tiles each, token-major.
#[derive(Debug, Clone, PartialEq)]
pub struct ActTiles {
    pub tokens: usize,
    pub tiles: usize,
    /// `tokens · tiles · 128` codes.
    pub qs: Vec<i8>,
    /// Per tile: the dequantization scale.
    pub scale: Vec<f32>,
    /// Per tile: Σx of the values the tile holds.
    pub sum: Vec<f32>,
}

impl ActTiles {
    /// Quantizes `x` (`tokens × cols`, row-major, `cols` a multiple of 128)
    /// to exactly the tile the device's producers write
    /// (`quantize/q8a128_tile.cuh`): codes `rint(x · 127/amax)`, the scale
    /// `amax/127` rounded to f16, and Σx summed as the warp sums it — each of
    /// 32 lanes its 4 values in order, then an xor butterfly — rounded to f16.
    /// The down projection's int8 GEMM reads only those stored values, so a
    /// CPU tile that kept them in f32 would be a different operand than the
    /// GPU's for the same SwiGLU output.
    pub fn quantize(x: &[f32], tokens: usize, cols: usize) -> Self {
        assert_eq!(cols % TILE, 0, "cols must be a multiple of {TILE}");
        assert_eq!(x.len(), tokens * cols);
        let tiles = cols / TILE;
        let n = tokens * tiles;
        let mut out = Self {
            tokens,
            tiles,
            qs: vec![0; n * TILE],
            scale: vec![0.0; n],
            sum: vec![0.0; n],
        };
        for t in 0..n {
            let src = &x[t * TILE..(t + 1) * TILE];
            let amax = quantize_codes(src, &mut out.qs[t * TILE..(t + 1) * TILE]);
            out.scale[t] = f16::from_f32(amax / 127.0).to_f32();
            out.sum[t] = f16::from_f32(warp_sum(src)).to_f32();
        }
        out
    }

    /// Reads `tokens` rows of `tiles` tiles from a q8a1024 operand, token `i`
    /// being rows `rows[i]` of the operand. `sum_by_amax` is the operand's sum
    /// convention: the stored field is Σx/amax rather than Σx.
    pub fn from_q8a1024(bytes: &[u8], rows: &[usize], tiles: usize, sum_by_amax: bool) -> Self {
        let n = rows.len() * tiles;
        let mut out = Self {
            tokens: rows.len(),
            tiles,
            qs: vec![0; n * TILE],
            scale: vec![0.0; n],
            sum: vec![0.0; n],
        };
        for (i, &row) in rows.iter().enumerate() {
            for t in 0..tiles {
                let flat = row * tiles + t;
                let qs_off = (flat >> 3) * Q8A1024_BLOCK + (flat & 7) * TILE;
                let ds_off = (flat >> 3) * Q8A1024_BLOCK + Q8A1024_META + (flat & 7) * 16;
                let dst = (i * tiles + t) * TILE;
                for (q, &b) in out.qs[dst..dst + TILE]
                    .iter_mut()
                    .zip(&bytes[qs_off..qs_off + TILE])
                {
                    *q = b as i8;
                }
                let scale = f16::from_le_bytes([bytes[ds_off], bytes[ds_off + 1]]).to_f32();
                let field = f16::from_le_bytes([bytes[ds_off + 2], bytes[ds_off + 3]]).to_f32();
                out.scale[i * tiles + t] = scale;
                out.sum[i * tiles + t] = if sum_by_amax {
                    field * scale * 127.0
                } else {
                    field
                };
            }
        }
        out
    }

    /// Token `tok`'s tile `t`: its codes, scale and Σx.
    pub fn tile(&self, tok: usize, t: usize) -> (&[i8], f32, f32) {
        let i = tok * self.tiles + t;
        (
            &self.qs[i * TILE..(i + 1) * TILE],
            self.scale[i],
            self.sum[i],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Codes, scale and Σx of one tile, and the dead tile. The scale is the
    /// f16 the device stores: 8/127 rounds to 0x2c08. Σ of (i − 64)/8 over
    /// the tile is exactly −8.
    #[test]
    fn quantize_one_tile() {
        let mut x = vec![0f32; 2 * TILE];
        for (i, v) in x[..TILE].iter_mut().enumerate() {
            *v = (i as f32 - 64.0) / 8.0;
        }
        let a = ActTiles::quantize(&x, 2, TILE);
        let (q, scale, sum) = a.tile(0, 0);
        assert_eq!(f16::from_f32(scale).to_bits(), 0x2c08);
        assert_eq!(scale, f16::from_bits(0x2c08).to_f32());
        assert_eq!(q[0], -127);
        assert_eq!(q[64], 0);
        // 63/8 · 127/8 = 125.02 → 125; half-to-even is exercised below.
        assert_eq!(q[127], 125);
        assert_eq!(sum, -8.0);
        let (q, scale, sum) = a.tile(1, 0);
        assert!(q.iter().all(|&c| c == 0));
        assert_eq!((scale, sum), (0.0, 0.0));
    }

    /// The warp's order, not the sequential one: 1e8 in lane 0 and lane 16
    /// with opposite signs, and a 1.0 in lane 1. Sequentially the 1.0 is
    /// absorbed into 1e8 and lost (Σ = 0); the butterfly pairs lane 0 with
    /// lane 16 first, so the two cancel exactly and the 1.0 survives.
    #[test]
    fn the_sum_is_the_warps() {
        let mut x = vec![0f32; TILE];
        x[0] = 1e8;
        x[4] = 1.0;
        x[64] = -1e8;
        let seq: f32 = x.iter().sum();
        assert_eq!(seq, 0.0);
        assert_eq!(ActTiles::quantize(&x, 1, TILE).tile(0, 0).2, 1.0);
    }

    /// The tile quantizer this CPU runs equals the scalar definition, codes
    /// and amax bit for bit: random tiles, exact halves at unit scale, a NaN
    /// (passed over by amax, code 0), ±∞, −0 and a dead tile.
    #[test]
    fn tile_codes_are_the_scalar_definition() {
        #[cfg(target_arch = "x86_64")]
        let vector = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let vector = false;
        if !vector {
            eprintln!(
                "tile_codes_are_the_scalar_definition: skipped — this CPU has no AVX2, so the \
                 quantizer is the scalar definition itself"
            );
            return;
        }
        let mut seed = 5u64;
        let mut lcg = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut tiles: Vec<Vec<f32>> = (0..64)
            .map(|i| (0..TILE).map(|_| lcg() * (i as f32 + 1.0) * 0.37).collect())
            .collect();
        let mut halves: Vec<f32> = (0..TILE).map(|i| i as f32 - 63.5).collect();
        halves[0] = 127.0;
        tiles.push(halves);
        let mut odd: Vec<f32> = (0..TILE).map(|_| lcg()).collect();
        odd[3] = f32::NAN;
        odd[9] = -0.0;
        tiles.push(odd);
        let mut inf: Vec<f32> = (0..TILE).map(|_| lcg()).collect();
        inf[7] = f32::NEG_INFINITY;
        tiles.push(inf);
        tiles.push(vec![0.0; TILE]);
        for (i, t) in tiles.iter().enumerate() {
            let (mut a, mut b) = ([0i8; TILE], [0i8; TILE]);
            let am = quantize_codes(t, &mut a);
            let bm = quantize_codes_scalar(t, &mut b);
            assert_eq!(a, b, "tile {i} codes");
            assert_eq!(am.to_bits(), bm.to_bits(), "tile {i} amax");
        }
    }

    /// 0.5 and 1.5 after scaling round to even codes.
    #[test]
    fn quantize_rounds_half_to_even() {
        let mut x = vec![0f32; TILE];
        x[0] = 127.0;
        x[1] = 0.5;
        x[2] = 1.5;
        x[3] = -2.5;
        let a = ActTiles::quantize(&x, 1, TILE);
        let (q, _, _) = a.tile(0, 0);
        assert_eq!(&q[..4], &[127, 0, 2, -2]);
    }

    /// Reading the device's operand layout: a token whose tiles straddle a
    /// super-block, both sum conventions.
    #[test]
    fn reads_q8a1024_rows_across_super_blocks() {
        let tiles = 20usize;
        let rows_total = 3usize;
        let blocks = (rows_total * tiles).div_ceil(8);
        let mut bytes = vec![0u8; blocks * Q8A1024_BLOCK];
        for flat in 0..rows_total * tiles {
            let qs_off = (flat >> 3) * Q8A1024_BLOCK + (flat & 7) * TILE;
            for k in 0..TILE {
                bytes[qs_off + k] = ((flat + k) % 251) as u8;
            }
            let ds_off = (flat >> 3) * Q8A1024_BLOCK + Q8A1024_META + (flat & 7) * 16;
            bytes[ds_off..ds_off + 2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
            bytes[ds_off + 2..ds_off + 4]
                .copy_from_slice(&f16::from_f32(flat as f32).to_le_bytes());
        }
        let raw = ActTiles::from_q8a1024(&bytes, &[2, 1], tiles, false);
        let (q, scale, sum) = raw.tile(0, 3);
        let flat = 2 * tiles + 3;
        assert_eq!(q[0] as u8, (flat % 251) as u8);
        assert_eq!((scale, sum), (0.5, flat as f32));
        let normed = ActTiles::from_q8a1024(&bytes, &[1], tiles, true);
        let (_, scale, sum) = normed.tile(0, 4);
        assert_eq!(sum, (tiles + 4) as f32 * scale * 127.0);
    }
}
