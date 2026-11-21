//! Candle's own small block formats: the 2- and 3-bit symmetric codes, the raw
//! F16 block with reserved capture space, and the palette index block.
//!
//! Q2_0 and Q3_0 are symmetric codes whose decode is `d` times the code minus its
//! midpoint. An all-zero block still encodes midpoint codes, as the seal writes
//! it. R16 stores 32 F16 values and reserves 32 `u16` for Q capture, and decode
//! reads only the F16 values. P2 is index metadata rather than quantized values,
//! so its decode returns the 2-bit indices as floats from 0.0 to 3.0.

use super::*;

/// Q2_0: 2-bit symmetric quantization (candle-specific)
/// 4 quants per byte, decode: d * (q - 1.5) where d = amax / 1.5
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK2_0 / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 10);

/// R16: Raw F16 with reserved Q-capture space (candle-specific)
/// 32×F16 primary values (64 bytes) + 32×u16 reserved Q space (64 bytes) = 128 bytes
/// Dequant reads the 32 F16 values; Q space is reserved for future Q capture.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockR16 {
    pub(crate) d: [f16; QK_R16],
    pub(crate) q: [u16; QK_R16],
}

const _: () = assert!(std::mem::size_of::<BlockR16>() == 128);

/// Q3_0: 3-bit symmetric quantization (candle-specific)
/// Low 2 bits in qs (4/byte), high bit in qh (8/byte), decode: d * (q - 3.5)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK3_0 / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ3_0>() == 14);

/// P2: 2-bit palette index — pure arena routing, not a quant.
/// Each byte packs 4 head_dim positions × 2-bit indices.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockP2 {
    pub(crate) packed: u8,
}

const _: () = assert!(std::mem::size_of::<BlockP2>() == 1);

impl GgmlType for BlockQ2_0 {
    const DTYPE: GgmlDType = GgmlDType::Q2_0;
    const BLCK_SIZE: usize = QK2_0;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK2_0));
        let nb = k / QK2_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK2_0 {
                let q = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as f32;
                ys[i * QK2_0 + j] = d * (q - 1.5);
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            // `quantize_block_q2_0_vec`: an all-zero block still encodes its
            // quants (every one the midpoint code 2), as the seal writes it.
            let amax = xs.iter().map(|x| x.abs()).fold(0f32, f32::max);
            ys.d = f16::from_f32(amax / 1.5);
            let id = if amax != 0.0 { 1.5 / amax } else { 0.0 };
            ys.qs.fill(0);
            for (j, &x) in xs.iter().enumerate().take(QK2_0) {
                let q = (x * id + 1.5).round().clamp(0.0, 3.0) as u8;
                ys.qs[j / 4] |= q << ((j % 4) * 2);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK2_0));
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = x.d.to_f32();
            let dy = y.d.to_f32();
            let mut s = 0f32;
            for j in 0..QK2_0 {
                let qx = ((x.qs[j / 4] >> ((j % 4) * 2)) & 3) as f32 - 1.5;
                let qy = y.qs[j] as f32;
                s += qx * qy;
            }
            sumf += dx * dy * s;
        }
        sumf
    }
}

impl GgmlType for BlockQ3_0 {
    const DTYPE: GgmlDType = GgmlDType::Q3_0;
    const BLCK_SIZE: usize = QK3_0;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK3_0));
        let nb = k / QK3_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK3_0 {
                let lo = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as u32;
                let hi = ((xs[i].qh[j / 8] >> (j % 8)) & 1) as u32;
                let q = (hi << 2) | lo;
                ys[i * QK3_0 + j] = d * (q as f32 - 3.5);
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            // `quantize_block_q3_0`: `d = amax · (1/3.5)`, and an all-zero
            // block still encodes its quants (every one the midpoint code 4).
            let amax = xs.iter().map(|x| x.abs()).fold(0f32, f32::max);
            ys.d = f16::from_f32(amax * (1.0 / 3.5));
            ys.qh.fill(0);
            ys.qs.fill(0);
            let id = if amax != 0.0 { 3.5 / amax } else { 0.0 };
            for (j, &x) in xs.iter().enumerate().take(QK3_0) {
                let q = (x * id + 3.5).round().clamp(0.0, 7.0) as u8;
                let lo = q & 3;
                let hi = (q >> 2) & 1;
                ys.qs[j / 4] |= lo << ((j % 4) * 2);
                ys.qh[j / 8] |= hi << (j % 8);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK3_0));
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = x.d.to_f32();
            let dy = y.d.to_f32();
            let mut s = 0f32;
            for j in 0..QK3_0 {
                let lo = ((x.qs[j / 4] >> ((j % 4) * 2)) & 3) as u32;
                let hi = ((x.qh[j / 8] >> (j % 8)) & 1) as u32;
                let qx = ((hi << 2) | lo) as f32 - 3.5;
                let qy = y.qs[j] as f32;
                s += qx * qy;
            }
            sumf += dx * dy * s;
        }
        sumf
    }
}

impl GgmlType for BlockR16 {
    const DTYPE: GgmlDType = GgmlDType::R16;
    const BLCK_SIZE: usize = QK_R16;
    type VecDotType = BlockR16;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_R16));
        let nb = k / QK_R16;
        for i in 0..nb {
            for j in 0..QK_R16 {
                ys[i * QK_R16 + j] = xs[i].d[j].to_f32();
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for (d, &x) in ys.d.iter_mut().zip(xs.iter()) {
                *d = f16::from_f32(x);
            }
            ys.q = [0u16; QK_R16]; // Zero-fill Q space
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_R16));
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            for j in 0..QK_R16 {
                sumf += x.d[j].to_f32() * y.d[j].to_f32();
            }
        }
        sumf
    }
}

impl GgmlType for BlockP2 {
    const DTYPE: GgmlDType = GgmlDType::P2;
    const BLCK_SIZE: usize = QK_P2;
    type VecDotType = BlockQ8_0; // unused — P2 is not a quant

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        // P2 is pure index metadata, not quantized values.
        // Dequant returns the 2-bit indices as floats 0.0..3.0.
        for (i, x) in xs.iter().enumerate() {
            for j in 0..QK_P2 {
                ys[i * QK_P2 + j] = ((x.packed >> (j * 2)) & 3) as f32;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let mut packed = 0u8;
            for j in 0..QK_P2 {
                let idx = xs[i * QK_P2 + j].round().clamp(0.0, 3.0) as u8;
                packed |= idx << (j * 2);
            }
            y.packed = packed;
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}
