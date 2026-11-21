//! The llama.cpp legacy block formats, Q4_0 through Q8_1, with their quantizers
//! and vec-dots.
//!
//! Most of these quantizers are generic Rust, and the comments cite the ggml
//! revisions they follow. Q8_1 is written as the `quantize_q8_1` kernel writes
//! it: `d` is `amax / 127` and the quants are `rint(x * 127 / amax)`. Its `s`
//! field is `d` times the sum of the quants, which the Q4_1 and Q5_1 vec-dot
//! folds against. The kernel stores the sum of the inputs there instead, and no
//! decoder of a KV band reads it. Dequantization is `qs * d` alone.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            k.is_multiple_of(qk),
            "dequantize_row_q4_0: {k} is not divisible by {qk}"
        );

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        debug_assert!(k.is_multiple_of(qk), "{k} is not divisible by {qk}");
        debug_assert_eq!(
            ys.len(),
            k / qk,
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if super::super::avx::available() {
            // SAFETY: the CPU reported AVX2 and FMA.
            return unsafe { super::super::avx::vec_dot_q4_0_q8_0(n, xs, ys) };
        }

        #[cfg(target_feature = "neon")]
        return super::super::neon::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::super::simd128::vec_dot_q4_0_q8_0(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK8_0),
            "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
        );
        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sum_i = 0;
            for j in 0..QK8_0 / 2 {
                let v0 = (xs.qs[j] & 0x0F) as i32 - 8;
                let v1 = (xs.qs[j] >> 4) as i32 - 8;
                sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + QK8_0 / 2] as i32
            }
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }
}

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        // ggml_vec_dot_q4_1_q8_1
        let qk = QK8_1;
        debug_assert!(
            n.is_multiple_of(qk),
            "vec_dot_q4_1_q8_1: {n} is not divisible by {qk}"
        );
        debug_assert!(
            (n / qk).is_multiple_of(2),
            "vec_dot_q4_1_q8_1: {n}, nb is not divisible by 2"
        );

        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sumi = 0i32;

            for j in 0..qk / 2 {
                let v0 = xs.qs[j] as i32 & 0x0F;
                let v1 = xs.qs[j] as i32 >> 4;
                sumi += (v0 * ys.qs[j] as i32) + (v1 * ys.qs[j + qk / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
                + f16::to_f32(xs.m) * f16::to_f32(ys.s)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q4_1
        let qk = Self::BLCK_SIZE;

        debug_assert_eq!(
            ys.len() * qk,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 4) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = u8::min(15, (x0 + 0.5) as u8);
                let xi1 = u8::min(15, (x1 + 0.5) as u8);

                *q = xi0 | (xi1 << 4);
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK4_1),
            "dequantize_row_q4_1: {k} is not divisible by {QK4_1}"
        );

        let nb = k / QK4_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();

            for j in 0..(QK4_1 / 2) {
                let x0 = xs[i].qs[j] & 0x0F;
                let x1 = xs[i].qs[j] >> 4;

                ys[i * QK4_1 + j] = (x0 as f32) * d + m;
                ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;
    type VecDotType = BlockQ8_0;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let qk = Self::BLCK_SIZE;

        debug_assert!(
            n.is_multiple_of(qk),
            "vec_dot_q5_0_q8_0: {n} is not divisible by {qk}"
        );
        debug_assert!(
            (n / qk).is_multiple_of(2),
            "vec_dot_q5_0_q8_0: {n}, nb is not divisible by 2"
        );
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(_n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let qh = LittleEndian::read_u32(&xs.qh);
            let mut sumi = 0i32;

            for j in 0..Self::BLCK_SIZE / 2 {
                let xh_0 = (((qh & (1u32 << j)) >> j) << 4) as u8;
                let xh_1 = ((qh & (1u32 << (j + 16))) >> (j + 12)) as u8;

                let x0 = ((xs.qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
                let x1 = ((xs.qs[j] >> 4) as i32 | xh_1 as i32) - 16;

                sumi += (x0 * ys.qs[j] as i32) + (x1 * ys.qs[j + Self::BLCK_SIZE / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q5_0
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            let mut amax = 0f32;
            let mut max = 0f32;
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -16.;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut qh = 0u32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let x0 = xs[j] * id;
                let x1 = xs[j + Self::BLCK_SIZE / 2] * id;
                let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
                let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
                ys.qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + Self::BLCK_SIZE / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh)
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_0),
            "dequantize_row_q5_0: {k} is not divisible by {QK5_0}"
        );
        let nb = k / QK5_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_0 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;

                ys[i * QK5_0 + j] = (x0 as f32) * d;
                ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
            }
        }
    }
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            n.is_multiple_of(qk),
            "vec_dot_q5_1_q8_1: {n} is not divisible by {qk}"
        );
        debug_assert!(
            (n / qk).is_multiple_of(2),
            "vec_dot_q5_1_q8_1: {n}, nb is not divisible by 2"
        );

        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let qh = LittleEndian::read_u32(&xs.qh);
            let mut sumi = 0i32;

            for j in 0..Self::BLCK_SIZE / 2 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;

                let x0 = (xs.qs[j] as i32 & 0xF) | xh_0 as i32;
                let x1 = (xs.qs[j] as i32 >> 4) | xh_1 as i32;

                sumi += (x0 * ys.qs[j] as i32) + (x1 * ys.qs[j + Self::BLCK_SIZE / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
                + f16::to_f32(xs.m) * f16::to_f32(ys.s)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q5_1
        let qk = Self::BLCK_SIZE;
        debug_assert_eq!(
            ys.len() * qk,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 5) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            let mut qh = 0u32;
            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = (x0 + 0.5) as u8;
                let xi1 = (x1 + 0.5) as u8;

                *q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                // get the 5-th bit and store it in qh at the right position
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + qk / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh);
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_1),
            "dequantize_row_q5_1: {k} is not divisible by {QK5_1}"
        );

        let nb = k / QK5_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_1 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
                let x1 = (xs[i].qs[j] >> 4) | xh_1;

                ys[i * QK5_1 + j] = (x0 as f32) * d + m;
                ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK8_0),
            "dequantize_row_q8_0: {k} is not divisible by {QK8_0}"
        );

        let nb = k / QK8_0;

        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q8_0
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(Self::BLCK_SIZE),
            "{k} is not divisible by {}",
            Self::BLCK_SIZE
        );
        debug_assert_eq!(
            ys.len(),
            k / Self::BLCK_SIZE,
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
        }
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if super::super::avx::available() {
            // SAFETY: the CPU reported AVX2 and FMA.
            return unsafe { super::super::avx::vec_dot_q8_0_q8_0(n, xs, ys) };
        }

        #[cfg(target_feature = "neon")]
        return super::super::neon::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::super::simd128::vec_dot_q8_0_q8_0(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK8_0),
            "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
        );

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q8_1;
    const BLCK_SIZE: usize = QK8_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK8_1),
            "vec_dot_q8_1_q8_1: {n} is not divisible by {QK8_1}"
        );

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }

    /// `d` and `qs` as `quantize_block_q8_1` (`quantize_q8_1.cuh`) writes
    /// them: `q = rint(x · 127/amax)`, `d = amax · (1/127)`. `s` is `d · Σq`,
    /// the sum of the quantized values the Q4_1/Q5_1 vec-dot folds its min
    /// against; the kernel stores `Σx` there instead, and no decoder of a KV
    /// band reads it.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax = xs.iter().fold(0f32, |m, x| m.max(x.abs()));
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            for (q, &x) in ys.qs.iter_mut().zip(xs) {
                *q = rint(x * id) as i8;
            }
            ys.d = f16::from_f32(amax * INV_127);
            let sum_q: i32 = ys.qs.iter().map(|&q| q as i32).sum();
            ys.s = f16::from_f32(sum_q as f32) * ys.d;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK8_1),
            "dequantize_row_q8_1: {k} is not divisible by {QK8_1}"
        );
        // Dequant is `qs * d`; the per-block sum `s` is only an auxiliary term for
        // the int8 dot product, not part of the reconstructed values.
        let nb = k / QK8_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK8_1 {
                ys[i * QK8_1 + j] = xs[i].qs[j] as f32 * d;
            }
        }
    }
}
