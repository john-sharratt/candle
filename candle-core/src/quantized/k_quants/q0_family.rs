//! The Q0 family of INT8-scaled blocks, from the single-centroid Q0 to the
//! asymmetric Q1_A.
//!
//! Q0 holds one INT8 centroid shared by all 32 lanes, the cheapest format at one
//! byte per 32 elements, and decodes it as the centroid over 127. The outer-scale
//! division is left to the BlockConverter, and the trait's `to_float` returns the
//! value before it. Q1_S, Q2_S and Q2_A encode their INT8 scale first and place
//! the quants against the decoded value. Q1_A keeps one amplitude per sign and a
//! 32-bit sign mask, and Q2_1 and Q3_1 use an F16 scale and minimum. Encoding
//! takes the block mean with a butterfly sum, rounds ties to even, and expects
//! inputs roughly in [-1, 1].

use super::*;

/// Q0: Single INT8 centroid shared by all 32 lanes.
///
/// Decode: every lane outputs `centroid as f32 / 127.0` (with outer-scale
/// division applied by the BlockConverter). Cheapest legitimate quant at
/// 1 byte / 32 elements (0.25 BPE) — useful for near-flat blocks.
///
/// Wire-compatible with `block_q0` in `candle-kernels/src/blocks.cuh` (i8
/// `centroid`).
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0 {
    pub(crate) centroid: i8,
}

const _: () = assert!(std::mem::size_of::<BlockQ0>() == 1);

/// Q1_S: 1-bit sign + INT8 scale (`block_q1_s`: `scale = rint(mean|x| · 127)`,
/// decode `±scale/127`).
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ1S {
    pub(crate) scale: i8,
    pub(crate) qs: [u8; QK1_S / 8],
}

const _: () = assert!(std::mem::size_of::<BlockQ1S>() == 5);

/// Q2_S: 2-bit symmetric + INT8 scale (`block_q2_s`: `d = scale/127`, decode
/// `d · (q − 1.5)`).
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2S {
    pub(crate) scale: i8,
    pub(crate) qs: [u8; QK2_S / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ2S>() == 9);

/// Q2_A: 2-bit asymmetric + INT8 scale + INT8 bias (`block_q2_a`: decode
/// `q · scale/127 + bias/127`).
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2A {
    pub(crate) scale: i8,
    pub(crate) bias: i8,
    pub(crate) qs: [u8; QK2_A / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ2A>() == 10);

/// Q2_1: 2-bit asymmetric + F16 scale + F16 min (packed as u32 dm)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2_1 {
    pub(crate) dm: u32, // f16 scale in low 16 bits, f16 min in high 16 bits
    pub(crate) qs: [u8; QK2_1 / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ2_1>() == 12);

/// Q3_1: 3-bit asymmetric + F16 scale + F16 min
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3_1 {
    pub(crate) dm: u32, // f16 scale in low 16 bits, f16 min in high 16 bits
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK3_1 / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ3_1>() == 16);

/// Q1_A: 1-bit asymmetric — separate INT8 amplitude per sign + 32 sign bits
/// (6 bytes per 32 elements, 1.50 BPE).
///
///   sign_bit = 1 → x = +scale_pos / 127
///   sign_bit = 0 → x = -scale_neg / 127
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ1A {
    pub(crate) scale_pos: i8,
    pub(crate) scale_neg: i8,
    pub(crate) qs: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ1A>() == 6);

impl GgmlType for BlockQ0 {
    const DTYPE: GgmlDType = GgmlDType::Q0;
    const BLCK_SIZE: usize = QK_Q0;
    type VecDotType = BlockQ8_0;

    /// Decode: every lane in the block expands to the same constant,
    /// `centroid · (1/127)` as `block_q0.cuh` forms it. The outer-scale division is applied by
    /// the BlockConverter on the GPU side; trait `to_float` returns the
    /// pre-outer-scaled value to match every other format's contract.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_Q0),
            "to_float Q0: {k} is not divisible by {QK_Q0}"
        );
        let nb = k / QK_Q0;
        for i in 0..nb {
            let v = xs[i].centroid as f32 * INV_127;
            for j in 0..QK_Q0 {
                ys[i * QK_Q0 + j] = v;
            }
        }
    }

    /// Encode: take the block mean, scale to i8 [-127, 127] space, as
    /// `quantize_block_q0` does — the warp's butterfly sum times `1/32`,
    /// rounded ties-to-even. The caller is expected to have applied outer
    /// scaling so input values lie roughly in [-1, +1].
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_Q0),
            "from_float Q0: {k} is not divisible by {QK_Q0}"
        );
        let nb = k / QK_Q0;
        for i in 0..nb {
            let block = &xs[i * QK_Q0..(i + 1) * QK_Q0];
            ys[i].centroid = encode_centroid(warp_sum(block) * (1.0 / 32.0));
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ1S {
    const DTYPE: GgmlDType = GgmlDType::Q1_S;
    const BLCK_SIZE: usize = QK1_S;
    type VecDotType = BlockQ8_0;

    /// `block_q1_s.cuh`: `±scale · (1/127)`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK1_S;
        for i in 0..nb {
            let scale = xs[i].scale as f32 * INV_127;
            for j in 0..QK1_S {
                let bit = (xs[i].qs[j / 8] >> (j % 8)) & 1;
                ys[i * QK1_S + j] = if bit == 1 { scale } else { -scale };
            }
        }
    }

    /// `quantize_block_q1_s_vec` (`quantize_q1_s.cuh`): the mean |x| is the
    /// eight lanes' four-element partials, butterfly-summed, over 32.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let partials: [f32; 8] = std::array::from_fn(|l| {
                let v = &block[4 * l..4 * l + 4];
                v[0].abs() + v[1].abs() + v[2].abs() + v[3].abs()
            });
            let mean_abs = lane8_sum(&partials) / 32.0;
            y.scale = rint((mean_abs * 127.0).min(127.0)) as i8;
            y.qs.fill(0);
            for (j, &b) in block.iter().enumerate().take(QK1_S) {
                if b >= 0.0 {
                    y.qs[j / 8] |= 1 << (j % 8);
                }
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ2S {
    const DTYPE: GgmlDType = GgmlDType::Q2_S;
    const BLCK_SIZE: usize = QK2_S;
    type VecDotType = BlockQ8_0;

    /// `block_q2_s.cuh`: `d · (q − 1.5)`, `d = scale · (1/127)`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK2_S;
        for i in 0..nb {
            let d = xs[i].scale as f32 * INV_127;
            for j in 0..QK2_S {
                let q = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as f32;
                ys[i * QK2_S + j] = d * (q - 1.5);
            }
        }
    }

    /// `quantize_block_q2_s_vec` (`quantize_q2_s.cuh`): the INT8 scale is
    /// encoded first and the quants divide by its decoded value.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax = block.iter().map(|x| x.abs()).fold(0f32, f32::max);
            y.scale = rint((amax * (1.0 / 1.5) * 127.0).min(127.0)) as i8;
            let d = y.scale as f32 * INV_127;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            y.qs.fill(0);
            for (j, &b) in block.iter().enumerate().take(QK2_S) {
                let q = (b * id + 1.5).round().clamp(0.0, 3.0) as u8;
                y.qs[j / 4] |= q << ((j % 4) * 2);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ2A {
    const DTYPE: GgmlDType = GgmlDType::Q2_A;
    const BLCK_SIZE: usize = QK2_A;
    type VecDotType = BlockQ8_0;

    /// `block_q2_a.cuh`: `d · q + m`, `d = scale · (1/127)`, `m = bias ·
    /// (1/127)`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK2_A;
        for i in 0..nb {
            let d = xs[i].scale as f32 * INV_127;
            let m = xs[i].bias as f32 * INV_127;
            for j in 0..QK2_A {
                let q = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as f32;
                ys[i * QK2_A + j] = d * q + m;
            }
        }
    }

    /// `quantize_block_q2_a_vec` (`quantize_q2_a.cuh`): scale and bias are
    /// encoded to INT8 first and the quants are placed against their decoded
    /// values.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let vmax = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let vmin = block.iter().cloned().fold(f32::INFINITY, f32::min);
            y.scale = rint(((vmax - vmin) * (1.0 / 3.0) * 127.0).min(127.0)) as i8;
            y.bias = rint((vmin * 127.0).clamp(-127.0, 127.0)) as i8;
            let d = y.scale as f32 * INV_127;
            let m = y.bias as f32 * INV_127;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            y.qs.fill(0);
            for (j, &b) in block.iter().enumerate().take(QK2_A) {
                let q = ((b - m) * id).round().clamp(0.0, 3.0) as u8;
                y.qs[j / 4] |= q << ((j % 4) * 2);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ2_1 {
    const DTYPE: GgmlDType = GgmlDType::Q2_1;
    const BLCK_SIZE: usize = QK2_1;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK2_1;
        for i in 0..nb {
            let d = f16::from_bits((xs[i].dm & 0xffff) as u16).to_f32();
            let m = f16::from_bits((xs[i].dm >> 16) as u16).to_f32();
            for j in 0..QK2_1 {
                let q = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as f32;
                ys[i * QK2_1 + j] = q * d + m;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let vmax = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let vmin = block.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = vmax - vmin;
            let d = range / 3.0;
            let d_bits = f16::from_f32(d).to_bits() as u32;
            let m_bits = f16::from_f32(vmin).to_bits() as u32;
            y.dm = d_bits | (m_bits << 16);
            y.qs.fill(0);
            if range == 0.0 {
                continue;
            }
            let id = 3.0 / range;
            for (j, &b) in block.iter().enumerate().take(QK2_1) {
                let q = ((b - vmin) * id).round().clamp(0.0, 3.0) as u8;
                y.qs[j / 4] |= q << ((j % 4) * 2);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ3_1 {
    const DTYPE: GgmlDType = GgmlDType::Q3_1;
    const BLCK_SIZE: usize = QK3_1;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK3_1;
        for i in 0..nb {
            let d = f16::from_bits((xs[i].dm & 0xffff) as u16).to_f32();
            let m = f16::from_bits((xs[i].dm >> 16) as u16).to_f32();
            for j in 0..QK3_1 {
                let lo = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as u32;
                let hi = ((xs[i].qh[j / 8] >> (j % 8)) & 1) as u32;
                let q = (hi << 2) | lo;
                ys[i * QK3_1 + j] = q as f32 * d + m;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let vmax = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let vmin = block.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = vmax - vmin;
            let d = range / 7.0;
            let d_bits = f16::from_f32(d).to_bits() as u32;
            let m_bits = f16::from_f32(vmin).to_bits() as u32;
            y.dm = d_bits | (m_bits << 16);
            y.qh.fill(0);
            y.qs.fill(0);
            if range == 0.0 {
                continue;
            }
            let id = 7.0 / range;
            for (j, &b) in block.iter().enumerate().take(QK3_1) {
                let q = ((b - vmin) * id).round().clamp(0.0, 7.0) as u8;
                y.qs[j / 4] |= (q & 3) << ((j % 4) * 2);
                y.qh[j / 8] |= ((q >> 2) & 1) << (j % 8);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ1A {
    const DTYPE: GgmlDType = GgmlDType::Q1_A;
    const BLCK_SIZE: usize = QK1_A;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            for j in 0..QK1_A {
                let sign_bit = (x.qs[j >> 3] >> (j & 7)) & 1;
                let scale = if sign_bit != 0 {
                    x.scale_pos
                } else {
                    x.scale_neg
                };
                let magnitude = (scale as f32) * (1.0 / 127.0);
                ys[i * QK1_A + j] = if sign_bit != 0 { magnitude } else { -magnitude };
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            let mut qmask: u32 = 0;
            let mut sum_pos: f32 = 0.0;
            let mut sum_neg: f32 = 0.0;
            let mut n_pos: u32 = 0;
            for (j, &x) in block.iter().enumerate() {
                if x >= 0.0 {
                    qmask |= 1u32 << j;
                    sum_pos += x;
                    n_pos += 1;
                } else {
                    sum_neg += -x;
                }
            }
            let n_neg = (Self::BLCK_SIZE as u32) - n_pos;
            let mean_pos = if n_pos > 0 {
                sum_pos / n_pos as f32
            } else {
                0.0
            };
            let mean_neg = if n_neg > 0 {
                sum_neg / n_neg as f32
            } else {
                0.0
            };

            y.scale_pos = (mean_pos * 127.0).round().clamp(0.0, 127.0) as i8;
            y.scale_neg = (mean_neg * 127.0).round().clamp(0.0, 127.0) as i8;
            y.qs[0] = (qmask & 0xFF) as u8;
            y.qs[1] = ((qmask >> 8) & 0xFF) as u8;
            y.qs[2] = ((qmask >> 16) & 0xFF) as u8;
            y.qs[3] = ((qmask >> 24) & 0xFF) as u8;
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}
