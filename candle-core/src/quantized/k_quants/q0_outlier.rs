//! The Q0_X, Q0_M2 and Q0_M4 blocks: a flat anchor with one outlier escape, and
//! the two- and four-centroid variants chosen by a mask.
//!
//! Q0_X stores an INT8 bulk anchor and, in its second byte, the index of one
//! outlier with a signed 3-bit delta. Decode adds the scaled delta at that index
//! and clamps to plus or minus 127. Q0_M2 chooses one of two constants per
//! quartet by an 8-bit mask, and Q0_M4 one of four per pair by a 32-bit mask.
//! Each encoder performs the same operations as its CUDA kernel, with warp
//! butterfly sums and the same rounding rule, so the bytes agree.

use super::*;

/// Q0_X: Flat block + one outlier escape (2 bytes per 32 elements, 0.50 BPE)
///
/// Layout: byte 0 = INT8 bulk_anchor; byte 1 packs outlier_idx (low 5 bits)
/// and a signed 3-bit outlier_delta (high 3 bits, two's complement, [-4..3]).
/// Decode: v_i8 = clamp(bulk_anchor + (i==outlier_idx ? delta * S_OUTLIER : 0),
/// -127, 127); x = v_i8 / 127.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0X {
    pub(crate) bulk_anchor: i8,
    pub(crate) outlier_packed: u8,
}

const _: () = assert!(std::mem::size_of::<BlockQ0X>() == 2);

/// Outlier delta scale (in INT8 units). Must match Q0_X_S_OUTLIER in the
/// CUDA kernel header.
pub const Q0_X_S_OUTLIER: i32 = 32;

/// Q0_M2: Two INT8 constants + 8-bit quartet mask (3 bytes per 32 elements;
/// `block_q0_m2`, decode `centroid/127`)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0M2 {
    pub(crate) centroid: [i8; 2],
    pub(crate) qmask: u8,
}

const _: () = assert!(std::mem::size_of::<BlockQ0M2>() == 3);

/// Q0_M4: Four INT8 constants + 32-bit pair mask (8 bytes per 32 elements;
/// `block_q0_m4`, decode `centroid/127`)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0M4 {
    pub(crate) centroid: [i8; 4],
    pub(crate) qmask: u32,
}

const _: () = assert!(std::mem::size_of::<BlockQ0M4>() == 8);

impl GgmlType for BlockQ0X {
    const DTYPE: GgmlDType = GgmlDType::Q0_X;
    const BLCK_SIZE: usize = QK_Q0_X;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            let outlier_idx = (x.outlier_packed & 0x1F) as usize;
            let delta_u = ((x.outlier_packed >> 5) & 0x07) as i32;
            let delta = if delta_u < 4 { delta_u } else { delta_u - 8 };
            for j in 0..QK_Q0_X {
                let delta_scaled = if j == outlier_idx {
                    delta * Q0_X_S_OUTLIER
                } else {
                    0
                };
                let v = (x.bulk_anchor as i32 + delta_scaled).clamp(-127, 127);
                ys[i * QK_Q0_X + j] = v as f32 * INV_127;
            }
        }
    }

    /// `quantize_block_q0_x_core` (`quantize_q0.cuh`), op for op: the mean is
    /// the warp's butterfly sum times `1/32`, and every float-to-int rounding
    /// is `__float2int_rn` (ties to even).
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            // 1. bulk_anchor = INT8-encoded block mean
            let mean = warp_sum(block) * (1.0 / 32.0);
            let bulk_int = rint(mean * 127.0).clamp(-127, 127);
            let bulk_anchor = bulk_int as i8;

            // 2. argmax(|x_i_i8 - bulk|) → outlier_idx (the first on ties);
            //    track its residual
            let mut outlier_idx: usize = 0;
            let mut outlier_residual: i32 = 0;
            let mut best_abs: i32 = -1;
            for (j, &x) in block.iter().enumerate() {
                let xi = rint(x * 127.0).clamp(-127, 127);
                let r = xi - bulk_int;
                let ar = r.abs();
                if ar > best_abs {
                    best_abs = ar;
                    outlier_idx = j;
                    outlier_residual = r;
                }
            }

            // 3. Coarse delta in [-4, 3]
            let delta_raw = rint(outlier_residual as f32 / Q0_X_S_OUTLIER as f32);
            let outlier_delta = delta_raw.clamp(-4, 3);

            let packed_idx = (outlier_idx as u8) & 0x1F;
            let packed_delta = ((outlier_delta as u8) & 0x07) << 5;
            y.bulk_anchor = bulk_anchor;
            y.outlier_packed = packed_idx | packed_delta;
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ0M2 {
    const DTYPE: GgmlDType = GgmlDType::Q0_M2;
    const BLCK_SIZE: usize = QK_Q0_M2;
    type VecDotType = BlockQ8_0;

    /// `block_q0_m2.cuh`: each quartet's centroid times `1/127`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            for j in 0..QK_Q0_M2 {
                let c = x.centroid[((x.qmask >> (j / 4)) & 1) as usize];
                ys[i * QK_Q0_M2 + j] = c as f32 * INV_127;
            }
        }
    }

    /// `quantize_block_q0_m2_core` (`quantize_q0.cuh`), op for op: quartet
    /// means as the kernel's two xor rounds form them, four Lloyd iterations
    /// at quartet granularity whose sums are warp butterflies, ties to c0.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let qt_mean: [f32; 8] = std::array::from_fn(|q| {
                let v = &block[4 * q..4 * q + 4];
                ((v[0] + v[1]) + (v[2] + v[3])) * 0.25
            });
            let mut c0 = qt_mean.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut c1 = qt_mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let assign = |m: f32, c0: f32, c1: f32| usize::from((m - c1).abs() < (m - c0).abs());
            for _ in 0..4 {
                let a: [usize; 8] = std::array::from_fn(|q| assign(qt_mean[q], c0, c1));
                let masked = |k: usize, v: &dyn Fn(usize) -> f32| -> f32 {
                    let lanes: [f32; 32] =
                        std::array::from_fn(|l| if a[l / 4] == k { v(l) } else { 0.0 });
                    warp_sum(&lanes)
                };
                let (s0, n0) = (masked(0, &|l| block[l]), masked(0, &|_| 1.0));
                let (s1, n1) = (masked(1, &|l| block[l]), masked(1, &|_| 1.0));
                if n0 > 0.0 {
                    c0 = s0 / n0;
                }
                if n1 > 0.0 {
                    c1 = s1 / n1;
                }
            }
            y.qmask = (0..8).fold(0u8, |m, q| m | ((assign(qt_mean[q], c0, c1) as u8) << q));
            y.centroid = [encode_centroid(c0), encode_centroid(c1)];
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ0M4 {
    const DTYPE: GgmlDType = GgmlDType::Q0_M4;
    const BLCK_SIZE: usize = QK_Q0_M4;
    type VecDotType = BlockQ8_0;

    /// `block_q0_m4.cuh`: each pair's centroid times `1/127`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            for j in 0..QK_Q0_M4 {
                let k = ((x.qmask >> (2 * (j / 2))) & 3) as usize;
                ys[i * QK_Q0_M4 + j] = x.centroid[k] as f32 * INV_127;
            }
        }
    }

    /// `quantize_block_q0_m4_core` (`quantize_q0.cuh`), op for op: centroids
    /// start equally spaced over the pair means, five Lloyd iterations at pair
    /// granularity whose sums are warp butterflies, ties to the lower index.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let pair_mean: [f32; 16] =
                std::array::from_fn(|p| (block[p * 2] + block[p * 2 + 1]) * 0.5);
            let vmin = pair_mean.iter().cloned().fold(f32::INFINITY, f32::min);
            let vmax = pair_mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = (vmax - vmin) * (1.0 / 3.0);
            let mut c = [vmin, vmin + step, vmin + 2.0 * step, vmax];
            let nearest = |m: f32, c: &[f32; 4]| -> usize {
                let (mut best, mut best_d) = (0, (m - c[0]).abs());
                for (k, &ck) in c.iter().enumerate().skip(1) {
                    let d = (m - ck).abs();
                    if d < best_d {
                        (best, best_d) = (k, d);
                    }
                }
                best
            };
            for _ in 0..5 {
                let a: [usize; 16] = std::array::from_fn(|p| nearest(pair_mean[p], &c));
                for (k, ck) in c.iter_mut().enumerate() {
                    let sums: [f32; 32] =
                        std::array::from_fn(|l| if a[l / 2] == k { block[l] } else { 0.0 });
                    let counts: [f32; 32] =
                        std::array::from_fn(|l| if a[l / 2] == k { 1.0 } else { 0.0 });
                    let (sk, nk) = (warp_sum(&sums), warp_sum(&counts));
                    if nk > 0.0 {
                        *ck = sk / nk;
                    }
                }
            }
            y.qmask = (0..16).fold(0u32, |m, p| {
                m | ((nearest(pair_mean[p], &c) as u32) << (2 * p))
            });
            y.centroid = c.map(encode_centroid);
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}
