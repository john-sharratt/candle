//! The Q2_K, Q3_K and Q4_K block layouts and their quantizers, ported from
//! llama.cpp's k-quants.
//!
//! Each block follows the llama.cpp layout the GGML original writes, and each
//! quantizer cites the llama.cpp revision it follows. The AVX2 and FMA paths run
//! only after the CPU reports both features. Q3_K is written the way the
//! original is so the compiler can vectorize it: its scales are reconstructed
//! into an `i8` array, and dequantization walks the two 128-element halves, one
//! scale per 16 elements.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2_K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}

const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2_K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3_K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}

const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3_K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4_K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}

const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4_K>());

impl GgmlType for BlockQ2_K {
    const DTYPE: GgmlDType = GgmlDType::Q2_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if super::super::avx::available() {
            // SAFETY: the CPU reported AVX2 and FMA.
            return unsafe { super::super::avx::vec_dot_q2k_q8k(n, xs, ys) };
        }

        #[cfg(target_feature = "neon")]
        return super::super::neon::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::super::simd128::vec_dot_q2k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q2k_q8k: {n} is not divisible by {QK_K}"
        );

        let mut sumf = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q2: &[_] = &x.qs;
            let mut q8: &[_] = &y.qs;
            let sc = &x.scales;

            let mut summs = 0;
            for (bsum, scale) in y.bsums.iter().zip(sc) {
                summs += *bsum as i32 * ((scale >> 4) as i32);
            }

            let dall = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let mut isum = 0;
            let mut is = 0;
            for _ in 0..(QK_K / 128) {
                let mut shift = 0;
                for _ in 0..4 {
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    let mut isuml = 0;
                    for l in 0..16 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    isuml = 0;
                    for l in 16..32 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    shift += 2;
                    // adjust the indexing
                    q8 = &q8[32..];
                }
                // adjust the indexing
                q2 = &q2[32..];
            }
            sumf += dall * isum as f32 - dmin * summs as f32;
        }

        sumf
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L279
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        const Q4SCALE: f32 = 15.0;

        for (block, x) in group_for_quantization(xs, ys) {
            //calculate scales and mins
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];

            for (j, x_scale_slice) in x.chunks(16).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(3, 5, x_scale_slice);
            }
            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for (j, scale) in scales.iter().enumerate().take(QK_K / 16) {
                    block.scales[j] = nearest_int(iscale * scale) as u8;
                }
                block.d = f16::from_f32(max_scale / Q4SCALE);
            } else {
                block.scales[..QK_K / 16].fill(0);
                block.d = f16::from_f32(0.0);
            }

            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for (j, scale) in block.scales.iter_mut().enumerate() {
                    let l = nearest_int(iscale * mins[j]) as u8;
                    *scale |= l << 4;
                }
                block.dmin = f16::from_f32(max_min / Q4SCALE);
            } else {
                block.dmin = f16::from_f32(0.0);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();

            let mut is = 0;

            for (y_block, qs) in y
                .as_chunks_mut::<128>()
                .0
                .iter_mut()
                .zip(block.qs.as_chunks::<32>().0)
            {
                // Step by 32 over q.
                let mut shift = 0;
                let mut y_block_index = 0;
                for _j in 0..4 {
                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[..16] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[16..] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    shift += 2;
                }
            }
        }
    }
}

impl GgmlType for BlockQ3_K {
    const DTYPE: GgmlDType = GgmlDType::Q3_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if super::super::avx::available() {
            // SAFETY: the CPU reported AVX2 and FMA.
            return unsafe { super::super::avx::vec_dot_q3k_q8k(n, xs, ys) };
        }

        #[cfg(target_feature = "neon")]
        return super::super::neon::vec_dot_q3k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q3k_q8k: {n} is not divisible by {QK_K}"
        );

        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut auxs: [u32; 4] = [0; 4];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q3: &[u8] = &x.qs;
            let hmask: &[u8] = &x.hmask;
            let mut q8: &[i8] = &y.qs;

            aux32.fill(0);
            let mut a = &mut aux8[..];

            let mut m = 1;
            //Like the GGML original this is written this way to enable the compiler to vectorize it.
            for _ in 0..QK_K / 128 {
                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = (q3_val & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 2) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 4) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 6) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;
                q3 = &q3[32..];
            }

            a = &mut aux8[..];

            LittleEndian::read_u32_into(&x.scales, &mut auxs[0..3]);

            let tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            auxs[3] = ((auxs[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            auxs[0] = (auxs[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            auxs[1] = (auxs[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            for aux in auxs {
                for scale in aux.to_le_bytes() {
                    let scale = i8::from_be_bytes([scale]);
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += (scale as i32 - 32) * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];

                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += (scale as i32 - 32) * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
        }

        sums.iter().sum()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            for (j, x_scale_slice) in x.as_chunks::<16>().0.iter().enumerate() {
                scales[j] = make_q3_quants(x_scale_slice, 4, true);
            }

            // Get max scale by absolute value.
            let mut max_scale: f32 = 0.0;
            for &scale in scales.iter() {
                if scale.abs() > max_scale.abs() {
                    max_scale = scale;
                }
            }

            block.scales.fill(0);

            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                for (j, scale) in scales.iter().enumerate() {
                    let l_val = nearest_int(iscale * scale);
                    let l_val = l_val.clamp(-32, 31) + 32;
                    if j < 8 {
                        block.scales[j] = (l_val & 0xF) as u8;
                    } else {
                        block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                    }
                    let l_val = l_val >> 4;
                    block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
                }
                block.d = f16::from_f32(1.0 / iscale);
            } else {
                block.d = f16::from_f32(0.0);
            }

            let mut l: [i8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        for (block, y) in group_for_dequantization(xs, ys) {
            //Reconstruct the scales
            let mut aux = [0; 4];
            LittleEndian::read_u32_into(&block.scales, &mut aux[0..3]);

            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            //Transfer the scales into an i8 array
            let scales: &mut [i8] =
                unsafe { std::slice::from_raw_parts_mut(aux.as_mut_ptr() as *mut i8, 16) };

            let d_all = block.d.to_f32();
            let mut m = 1;
            let mut is = 0;

            // Dequantize both 128 long blocks
            // 32 qs values per 128 long block
            // Each 16 elements get a scale
            for (y, qs) in y
                .as_chunks_mut::<128>()
                .0
                .iter_mut()
                .zip(block.qs.as_chunks::<32>().0)
            {
                let mut shift = 0;
                for shift_scoped_y in y.as_chunks_mut::<32>().0.iter_mut() {
                    for (scale_index, scale_scoped_y) in shift_scoped_y
                        .as_chunks_mut::<16>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        for (i, inner_y) in scale_scoped_y.iter_mut().enumerate() {
                            let new_y = dl
                                * (((qs[i + 16 * scale_index] >> shift) & 3) as i8
                                    - if (block.hmask[i + 16 * scale_index] & m) == 0 {
                                        4
                                    } else {
                                        0
                                    }) as f32;
                            *inner_y = new_y;
                        }
                        // 16 block finished => advance scale index
                        is += 1;
                    }
                    // 32 block finished => increase shift and m
                    shift += 2;
                    m <<= 1;
                }
            }
        }
    }
}

impl GgmlType for BlockQ4_K {
    const DTYPE: GgmlDType = GgmlDType::Q4_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if super::super::avx::available() {
            // SAFETY: the CPU reported AVX2 and FMA.
            return unsafe { super::super::avx::vec_dot_q4k_q8k(n, xs, ys) };
        }

        #[cfg(target_feature = "neon")]
        return super::super::neon::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::super::simd128::vec_dot_q4k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
        );

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut sumf = 0.0;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q4 = &x.qs;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a = &mut aux8[..];
            let mut q4 = &q4[..];
            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q4[l] & 0xF) as i8;
                }
                a = &mut a[32..];
                for l in 0..32 {
                    a[l] = (q4[l] >> 4) as i8;
                }
                a = &mut a[32..];
                q4 = &q4[32..];
            }

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            //extract scales and mins
            LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
            LittleEndian::write_u32_into(&utmp[2..4], &mut mins);

            let mut sumi = 0;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a = &mut aux8[..];
            let mut q8 = &q8[..];

            for scale in scales {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = x.dmin.to_f32() * y.d;
            sumf -= dmin * sumi as f32;
        }
        sumf + sums.iter().sum::<f32>()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.as_chunks::<32>().0.iter().enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(15, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }

            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let q = &block.qs;
            let mut is = 0;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let q = &q[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for q in q {
                    y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                    ys_index += 1;
                }
                for q in q {
                    y[ys_index] = d2 * (q >> 4) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
            }
        }
    }
}
