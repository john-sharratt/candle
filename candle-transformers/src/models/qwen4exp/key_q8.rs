//! QSA index keys in Q8_0 (`docs/qsa_index_q8.md` §3).
//!
//! A completed index key is stored as Q8_0 along its channels: blocks of
//! [`BLOCK`] channels, an `int8` code a channel and one `f16` scale a block.
//!
//! ```text
//!   amax = max|x| over the block
//!   d    = f16( amax · (1/127) )
//!   id   = (d == 0) ? 0 : 127 / amax
//!   q[c] = int8( round_ties_even( x[c] · id ) ),  clamped to ±127
//!   x̂[c] = q[c] · f32(d)
//! ```
//!
//! A block whose scale rounds to `f16` zero stores all-zero codes, so one
//! stored form means one value. A width that is not a multiple of [`BLOCK`]
//! ends in a short block.
//!
//! This is the host definition: what a migration quantizes with, and what the
//! append kernel's keys are gated against (`candle-kernels/src/qsa_key_q8.cuh`
//! — within one code, since the kernel runs under fast math and nothing
//! reproduces its bytes).

use half::f16;

/// Channels one scale covers.
pub const BLOCK: usize = 32;

/// Scales a `dim`-wide key carries.
pub const fn blocks(dim: usize) -> usize {
    dim.div_ceil(BLOCK)
}

/// Bytes one `dim`-wide key occupies: its codes, then its scales.
pub const fn key_bytes(dim: usize) -> usize {
    dim + 2 * blocks(dim)
}

/// Rows of Q8_0 keys as two row-major planes: `codes` is `[rows, dim]` `int8`
/// held as bytes, `scales` is `[rows, blocks(dim)]` `f16` bits.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Q8Keys {
    pub codes: Vec<u8>,
    pub scales: Vec<u16>,
    pub dim: usize,
}

impl Q8Keys {
    /// No rows of `dim`-wide keys.
    pub fn empty(dim: usize) -> Self {
        Self {
            codes: Vec::new(),
            scales: Vec::new(),
            dim,
        }
    }

    /// The planes as given, refused unless they hold the same whole number of
    /// `dim`-wide rows.
    pub fn from_planes(codes: Vec<u8>, scales: Vec<u16>, dim: usize) -> Result<Self, String> {
        let nb = blocks(dim);
        let whole = match dim {
            0 => codes.is_empty() && scales.is_empty(),
            _ => codes.len().is_multiple_of(dim) && scales.len() == codes.len() / dim * nb,
        };
        if !whole {
            return Err(format!(
                "q8 keys: {} codes and {} scales are not rows of a {dim}-wide key",
                codes.len(),
                scales.len()
            ));
        }
        Ok(Self { codes, scales, dim })
    }

    /// `rows` (`[n, dim]` row-major) quantized key by key.
    pub fn quantize(rows: &[f32], dim: usize) -> Self {
        let mut out = Self::empty(dim);
        if dim == 0 {
            return out;
        }
        out.codes.reserve(rows.len());
        out.scales.reserve(rows.len() / dim * blocks(dim));
        for key in rows.chunks_exact(dim) {
            quantize_key(key, &mut out.codes, &mut out.scales);
        }
        out
    }

    /// The keys' values, `[n, dim]` row-major.
    pub fn dequantize(&self) -> Vec<f32> {
        let nb = blocks(self.dim);
        let mut out = Vec::with_capacity(self.codes.len());
        for (r, key) in self.codes.chunks_exact(self.dim.max(1)).enumerate() {
            for (c, &q) in key.iter().enumerate() {
                let d = f16::from_bits(self.scales[r * nb + c / BLOCK]).to_f32();
                out.push(q as i8 as f32 * d);
            }
        }
        out
    }

    pub fn n_rows(&self) -> usize {
        self.codes.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Rows `[first, n_rows)`.
    pub fn from_row(&self, first: usize) -> Self {
        let nb = blocks(self.dim);
        Self {
            codes: self.codes[first * self.dim..].to_vec(),
            scales: self.scales[first * nb..].to_vec(),
            dim: self.dim,
        }
    }

    /// Rows `[first, first + n)` as borrowed planes.
    pub fn rows(&self, first: usize, n: usize) -> (&[u8], &[u16]) {
        let nb = blocks(self.dim);
        (
            &self.codes[first * self.dim..(first + n) * self.dim],
            &self.scales[first * nb..(first + n) * nb],
        )
    }
}

/// Quantize one key, appending its codes and its scales.
pub fn quantize_key(key: &[f32], codes: &mut Vec<u8>, scales: &mut Vec<u16>) {
    for block in key.chunks(BLOCK) {
        let amax = block.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = f16::from_f32(amax * (1.0 / 127.0));
        let id = if d.to_bits() & 0x7fff != 0 {
            127.0 / amax
        } else {
            0.0
        };
        scales.push(d.to_bits());
        for &x in block {
            let q = (x * id).round_ties_even().clamp(-127.0, 127.0) as i8;
            codes.push(q as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bits(x: f32) -> u16 {
        f16::from_f32(x).to_bits()
    }

    /// The format, against the exact bytes: a block's largest magnitude is
    /// ±127, the rest scale to it, and the scale is `amax / 127` in `f16`.
    #[test]
    fn a_key_quantizes_to_the_declared_bytes() {
        let mut key = vec![0f32; 32];
        key[0] = 127.0;
        key[1] = -127.0;
        key[2] = 1.0;
        key[3] = -64.0;
        key[4] = 0.25;
        let q = Q8Keys::quantize(&key, 32);
        let mut want = vec![0u8; 32];
        want[0] = 127;
        want[1] = (-127i8) as u8;
        want[2] = 1;
        want[3] = (-64i8) as u8;
        assert_eq!(q.codes, want);
        assert_eq!(q.scales, vec![f16_bits(1.0)]);
        assert_eq!(q.scales, vec![0x3c00]);
    }

    /// Halves round to the even code, in both signs.
    #[test]
    fn ties_round_to_even() {
        let mut key = vec![0f32; 32];
        key[0] = 127.0;
        key[1] = 0.5;
        key[2] = 1.5;
        key[3] = 2.5;
        key[4] = -0.5;
        key[5] = -1.5;
        let q = Q8Keys::quantize(&key, 32);
        let got: Vec<i8> = q.codes[..6].iter().map(|&b| b as i8).collect();
        assert_eq!(got, vec![127, 0, 2, 2, 0, -2]);
    }

    /// A block of zeros, and one too small for an `f16` scale, store a zero
    /// scale and zero codes.
    #[test]
    fn a_block_with_no_scale_stores_zero_codes() {
        let q = Q8Keys::quantize(&[0f32; 32], 32);
        assert_eq!(q.codes, vec![0u8; 32]);
        assert_eq!(q.scales, vec![0]);

        let tiny = Q8Keys::quantize(&[1e-9f32; 32], 32);
        assert_eq!(tiny.codes, vec![0u8; 32]);
        assert_eq!(tiny.scales, vec![0]);
    }

    /// Every block takes its own scale, and a width that is not a whole number
    /// of blocks ends in a short one.
    #[test]
    fn blocks_scale_independently_and_the_last_may_be_short() {
        assert_eq!((blocks(128), key_bytes(128)), (4, 136));
        assert_eq!((blocks(16), key_bytes(16)), (1, 18));
        assert_eq!((blocks(40), key_bytes(40)), (2, 44));

        let mut key = vec![0f32; 40];
        key[0] = 2.0;
        key[31] = -1.0;
        key[32] = 0.5;
        key[39] = 0.25;
        let q = Q8Keys::quantize(&key, 40);
        assert_eq!(q.n_rows(), 1);
        assert_eq!(q.codes.len(), 40);
        assert_eq!(q.scales, vec![f16_bits(2.0 / 127.0), f16_bits(0.5 / 127.0)]);
        let c: Vec<i8> = q.codes.iter().map(|&b| b as i8).collect();
        assert_eq!((c[0], c[31], c[32], c[39]), (127, -64, 127, 64));
    }

    /// A dequantized key is within half a step of the key, block by block.
    #[test]
    fn a_dequantized_key_is_within_half_a_step() {
        let mut s = 7u64;
        let rows: Vec<f32> = (0..3 * 128)
            .map(|i| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (s >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0;
                // Blocks of very different magnitude.
                u * [0.01f32, 1.0, 30.0, 0.3][(i % 128) / 32]
            })
            .collect();
        let q = Q8Keys::quantize(&rows, 128);
        assert_eq!((q.n_rows(), q.scales.len()), (3, 12));
        let back = q.dequantize();
        for (i, (&x, &y)) in rows.iter().zip(&back).enumerate() {
            let d = f16::from_bits(q.scales[i / 128 * 4 + (i % 128) / 32]).to_f32();
            // Half a code, plus the scale's own f16 rounding over 127 codes.
            assert!(
                (x - y).abs() <= d * 0.5 + d * 127.0 * 6e-4,
                "channel {i}: {x} against {y} at step {d}"
            );
        }
    }

    /// Rows narrow and borrow by row, both planes together.
    #[test]
    fn rows_narrow_both_planes() {
        let rows: Vec<f32> = (0..3 * 40).map(|i| i as f32 - 60.0).collect();
        let q = Q8Keys::quantize(&rows, 40);
        let tail = q.from_row(1);
        assert_eq!(tail, Q8Keys::quantize(&rows[40..], 40));
        let (codes, scales) = q.rows(1, 1);
        assert_eq!(codes, &tail.codes[..40]);
        assert_eq!(scales, &tail.scales[..2]);
        assert!(Q8Keys::from_planes(vec![0; 40], vec![0; 2], 40).is_ok());
        assert!(Q8Keys::from_planes(vec![0; 40], vec![0; 1], 40).is_err());
        assert!(Q8Keys::from_planes(vec![0; 39], vec![0; 2], 40).is_err());
    }
}
