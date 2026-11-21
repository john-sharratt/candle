//! The Q4_KS and Q8_KS blocks, which give the attention-sink elements their own
//! sub-block scale.
//!
//! Elements 0 to 3 form sub-block A, the attention sinks, and the rest form
//! sub-block B. A shared header holds a coarse scale over the block and a fine
//! scale per sub-block. The quants divide by the unrounded coarse scale times the
//! fine one, as the kernels do, and `d` is rounded to F16 afterwards. Decoding
//! gives each sub-block a scale of `d` times its stored fine scale over 255.

use super::*;

/// Q4_KS: 4-bit with attention-sink sub-block scaling (elems 0-3 = sub-block A)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_KS {
    pub(crate) d: f16,
    pub(crate) sa: u8,
    pub(crate) sb: u8,
    pub(crate) qs: [u8; QK_Q4_KS / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_KS>() == 20);

/// Q8_KS: 8-bit signed with attention-sink sub-block scaling
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_KS {
    pub(crate) d: f16,
    pub(crate) sa: u8,
    pub(crate) sb: u8,
    pub(crate) qs: [i8; QK_Q8_KS],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_KS>() == 36);

/// The `_KS` decoders' sub-block scales, as `q4_ks_decode` / `q8_ks_decode`
/// form them: `d · (s · 1/255)` for sub-block A (elements 0–3) and B.
pub(super) fn ks_decode_scales(d: f16, sa: u8, sb: u8) -> (f32, f32) {
    let cd = d.to_f32();
    (cd * (sa as f32 * INV_255), cd * (sb as f32 * INV_255))
}

/// The `_KS` encoders' shared header: a coarse scale over the block and a
/// fine scale per sub-block — A is elements 0–3 (the attention sinks), B the
/// rest. The quants divide by the unrounded coarse scale times the fine one,
/// exactly as the kernels do; `d` is stored rounded to f16 afterwards.
pub(super) struct KsHeader {
    pub(super) coarse: f32,
    pub(super) sa: u8,
    pub(super) sb: u8,
}

impl KsHeader {
    /// `inv_levels` is the kernel's `1.0f / 7.0f` or `1.0f / 127.0f`.
    pub(super) fn of(xs: &[f32], inv_levels: f32) -> Self {
        let amax_a = xs[..4].iter().fold(0f32, |m, x| m.max(x.abs()));
        let amax_b = xs[4..].iter().fold(0f32, |m, x| m.max(x.abs()));
        let amax = amax_a.max(amax_b);
        if amax == 0.0 {
            return Self {
                coarse: 0.0,
                sa: 255,
                sb: 255,
            };
        }
        let fine = |a: f32| (a / amax * 255.0).round().clamp(1.0, 255.0) as u8;
        Self {
            coarse: amax * inv_levels,
            sa: fine(amax_a),
            sb: fine(amax_b),
        }
    }

    /// Element `j`'s signed quant, clamped to `±levels`.
    pub(super) fn quant(&self, xs: &[f32], j: usize, levels: f32) -> i32 {
        let s = if j < 4 { self.sa } else { self.sb };
        let actual_d = self.coarse * s as f32 * INV_255;
        if actual_d == 0.0 {
            return 0;
        }
        (xs[j] / actual_d).round().clamp(-levels, levels) as i32
    }
}

impl GgmlType for BlockQ4_KS {
    const DTYPE: GgmlDType = GgmlDType::Q4_KS;
    const BLCK_SIZE: usize = QK_Q4_KS;
    type VecDotType = BlockQ4_KS;

    /// `q4_ks_decode` (`block_q4_ks.cuh`): sub-block scale `d · (s · 1/255)`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_Q4_KS));
        let nb = k / QK_Q4_KS;
        for i in 0..nb {
            let (da, db) = ks_decode_scales(xs[i].d, xs[i].sa, xs[i].sb);
            for j in 0..QK_Q4_KS / 2 {
                let lo_nibble = (xs[i].qs[j] & 0x0F) as i32 - 8;
                let hi_nibble = (xs[i].qs[j] >> 4) as i32 - 8;
                let scale_lo = if j < 4 { da } else { db };
                ys[i * QK_Q4_KS + j] = scale_lo * lo_nibble as f32;
                ys[i * QK_Q4_KS + j + QK_Q4_KS / 2] = db * hi_nibble as f32;
            }
        }
    }

    /// `quantize_block_q4_ks_vec` (`quantize_q4_ks.cuh`), op for op.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let h = KsHeader::of(xs, 1.0 / 7.0);
            let q = |j: usize| -> u8 { (h.quant(xs, j, 7.0) + 8) as u8 };
            for j in 0..QK_Q4_KS / 2 {
                ys.qs[j] = q(j) | (q(j + QK_Q4_KS / 2) << 4);
            }
            ys.d = f16::from_f32(h.coarse);
            ys.sa = h.sa;
            ys.sb = h.sb;
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_Q4_KS));
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = x.d.to_f32();
            let dxa = dx * x.sa as f32 / 255.0;
            let dxb = dx * x.sb as f32 / 255.0;
            let dy = y.d.to_f32();
            let dya = dy * y.sa as f32 / 255.0;
            let dyb = dy * y.sb as f32 / 255.0;
            let mut s = 0f32;
            for j in 0..QK_Q4_KS / 2 {
                let sx_lo = if j < 2 { dxa } else { dxb };
                let sy_lo = if j < 2 { dya } else { dyb };
                let xlo = (x.qs[j] & 0x0F) as i32 - 8;
                let xhi = (x.qs[j] >> 4) as i32 - 8;
                let ylo = (y.qs[j] & 0x0F) as i32 - 8;
                let yhi = (y.qs[j] >> 4) as i32 - 8;
                s += (sx_lo * sy_lo) * (xlo * ylo) as f32;
                s += (dxb * dyb) * (xhi * yhi) as f32;
            }
            sumf += s;
        }
        sumf
    }
}

impl GgmlType for BlockQ8_KS {
    const DTYPE: GgmlDType = GgmlDType::Q8_KS;
    const BLCK_SIZE: usize = QK_Q8_KS;
    type VecDotType = BlockQ8_KS;

    /// `q8_ks_decode` (`block_q8_ks.cuh`): sub-block scale `d · (s · 1/255)`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_Q8_KS));
        let nb = k / QK_Q8_KS;
        for i in 0..nb {
            let (da, db) = ks_decode_scales(xs[i].d, xs[i].sa, xs[i].sb);
            for j in 0..QK_Q8_KS {
                let scale = if j < 4 { da } else { db };
                ys[i * QK_Q8_KS + j] = scale * xs[i].qs[j] as f32;
            }
        }
    }

    /// `quantize_block_q8_ks_vec` (`quantize_q8_ks.cuh`), op for op.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let h = KsHeader::of(xs, 1.0 / 127.0);
            for (j, q) in ys.qs.iter_mut().enumerate() {
                *q = h.quant(xs, j, 127.0) as i8;
            }
            ys.d = f16::from_f32(h.coarse);
            ys.sa = h.sa;
            ys.sb = h.sb;
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_Q8_KS));
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = x.d.to_f32();
            let dxa = dx * x.sa as f32 / 255.0;
            let dxb = dx * x.sb as f32 / 255.0;
            let dy = y.d.to_f32();
            let dya = dy * y.sa as f32 / 255.0;
            let dyb = dy * y.sb as f32 / 255.0;
            let mut s = 0f32;
            for j in 0..QK_Q8_KS {
                let sx = if j < 4 { dxa } else { dxb };
                let sy = if j < 4 { dya } else { dyb };
                s += sx * sy * x.qs[j] as f32 * y.qs[j] as f32;
            }
            sumf += s;
        }
        sumf
    }
}
