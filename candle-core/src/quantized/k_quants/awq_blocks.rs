//! The 4-bit AWQ block layouts and the `GgmlType` code that quantizes and dots
//! them.
//!
//! An AWQ block holds 128 weights as unsigned 4-bit codes packed eight to a
//! `u32`, with one F16 scale and zero, dequantized as `scale * (q - zero)`. The
//! `_G64` variant splits the block into two 64-element groups, each with its own
//! scale and zero. The format is built for the GPU. On the CPU both operands are
//! quantized to the same type, and the dot dequantizes each side before
//! multiplying.

use super::*;

// AWQ (ACTIVATION-AWARE WEIGHT QUANTIZATION) BLOCK STRUCTURES
// AWQ uses 4-bit asymmetric quantization with per-group scales and zeros.
// Dequant formula: w = scale * (q - zero)
// where q is 4-bit unsigned [0,15], scale and zero are FP16.
//
// Block size: 128 elements per block (matching K/128 GEMX kernels)
// Storage: 16 u32 (64 bytes packed nibbles) + scales/zeros + padding = 80 bytes

/// AWQ block size constant (128 elements)
pub const QK_AWQ: usize = 128;

/// BlockQAWQ: 4-bit AWQ with group size 128 (1 scale/zero per block)
///
/// Layout:
/// - qs[16]: uint32 containing packed 4-bit weights (8 weights per u32)
/// - scale: f16 scale factor
/// - zero: f16 zero point
/// - _pad: padding to 80 bytes (16-byte aligned)
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(16))]
pub struct BlockQAWQ {
    pub(crate) qs: [u32; 16],  // 128 × 4-bit = 64 bytes
    pub(crate) scale: f16,     // scale for entire block
    pub(crate) zero: f16,      // zero point for entire block
    pub(crate) _pad: [u32; 3], // padding to 80 bytes
}

const _: () = assert!(std::mem::size_of::<BlockQAWQ>() == 80);

/// BlockQAWQG64: 4-bit AWQ with group size 64 (2 scale/zero pairs per block)
///
/// Layout:
/// - qs[16]: uint32 containing packed 4-bit weights (8 weights per u32)
/// - scales[2]: f16 scale factors (one per 64 elements)
/// - zeros[2]: f16 zero points (one per 64 elements)
/// - _pad: padding to 80 bytes (16-byte aligned)
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(16))]
pub struct BlockQAWQ_G64 {
    pub(crate) qs: [u32; 16],    // 128 × 4-bit = 64 bytes
    pub(crate) scales: [f16; 2], // scale per 64-element group
    pub(crate) zeros: [f16; 2],  // zero per 64-element group
    pub(crate) _pad: u32,        // padding to 80 bytes
}

const _: () = assert!(std::mem::size_of::<BlockQAWQ_G64>() == 80);

// AWQ (ACTIVATION-AWARE WEIGHT QUANTIZATION) IMPLEMENTATIONS
// AWQ dequant formula: w = scale * (q - zero)
// where q is 4-bit unsigned [0,15], extracted from packed u32 nibbles.
//
// Nibble packing: each u32 contains 8 × 4-bit weights
//   qs[i] = n0 | (n1 << 4) | (n2 << 8) | ... | (n7 << 28)
//   Thread t extracts 8 nibbles from qs[t], handling elements t*8..(t+1)*8
//
// Note: AWQ is primarily a GPU-accelerated format. For CPU matmul, we use
// self as VecDotType (both inputs quantized the same) for simplicity.

impl GgmlType for BlockQAWQ {
    const DTYPE: GgmlDType = GgmlDType::QAWQ;
    const BLCK_SIZE: usize = QK_AWQ; // 128 elements per block
    type VecDotType = BlockQAWQ; // Self-dotting for CPU matmul

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_AWQ),
            "dequantize_row_qawq: {k} is not divisible by {QK_AWQ}"
        );
        for (i, x) in xs.iter().enumerate() {
            let scale = x.scale.to_f32();
            let zero = x.zero.to_f32();
            // Dequant: w = scale * (q - zero)
            for t in 0..16 {
                let packed = x.qs[t];
                for j in 0..8 {
                    let q = ((packed >> (j * 4)) & 0xF) as f32;
                    ys[i * QK_AWQ + t * 8 + j] = scale * (q - zero);
                }
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_AWQ),
            "quantize_row_qawq: {k} is not divisible by {QK_AWQ}"
        );
        for (i, y) in ys.iter_mut().enumerate() {
            let xs_block = &xs[i * QK_AWQ..(i + 1) * QK_AWQ];

            // Find min/max in block for optimal scale/zero
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &x in xs_block.iter() {
                min_val = min_val.min(x);
                max_val = max_val.max(x);
            }

            // AWQ asymmetric quantization to [0,15]
            // scale = (max - min) / 15, zero = -min / scale
            let range = max_val - min_val;
            let (scale, zero) = if range < 1e-10 {
                (0.0, 0.0)
            } else {
                let s = range / 15.0;
                let z = -min_val / s;
                (s, z)
            };

            y.scale = f16::from_f32(scale);
            y.zero = f16::from_f32(zero);
            y._pad = [0; 3];

            // Quantize: q = clamp((x / scale) + zero, 0, 15)
            let inv_scale = if scale.abs() > 1e-10 {
                1.0 / scale
            } else {
                0.0
            };
            for t in 0..16 {
                let mut packed = 0u32;
                for j in 0..8 {
                    let x = xs_block[t * 8 + j];
                    let q = ((x * inv_scale + zero).round().clamp(0.0, 15.0)) as u32;
                    packed |= q << (j * 4);
                }
                y.qs[t] = packed;
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_AWQ),
            "vec_dot_qawq: {n} is not divisible by {QK_AWQ}"
        );
        // Self-dotting: both xs and ys are BlockQAWQ
        // Dequantize both and compute dot product
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let x_scale = x.scale.to_f32();
            let x_zero = x.zero.to_f32();
            let y_scale = y.scale.to_f32();
            let y_zero = y.zero.to_f32();

            for t in 0..16 {
                let x_packed = x.qs[t];
                let y_packed = y.qs[t];
                for j in 0..8 {
                    let xq = ((x_packed >> (j * 4)) & 0xF) as f32;
                    let yq = ((y_packed >> (j * 4)) & 0xF) as f32;
                    let x_val = x_scale * (xq - x_zero);
                    let y_val = y_scale * (yq - y_zero);
                    sumf += x_val * y_val;
                }
            }
        }
        sumf
    }
}

impl GgmlType for BlockQAWQ_G64 {
    const DTYPE: GgmlDType = GgmlDType::QAWQ_G64;
    const BLCK_SIZE: usize = QK_AWQ; // 128 elements per block (2 groups of 64)
    type VecDotType = BlockQAWQ_G64; // Self-dotting for CPU matmul

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_AWQ),
            "dequantize_row_qawq_g64: {k} is not divisible by {QK_AWQ}"
        );
        for (i, x) in xs.iter().enumerate() {
            // Group 0: elements 0-63 (threads 0-7), Group 1: elements 64-127 (threads 8-15)
            for t in 0..16 {
                let group = t / 8; // 0 for threads 0-7, 1 for threads 8-15
                let scale = x.scales[group].to_f32();
                let zero = x.zeros[group].to_f32();

                let packed = x.qs[t];
                for j in 0..8 {
                    let q = ((packed >> (j * 4)) & 0xF) as f32;
                    ys[i * QK_AWQ + t * 8 + j] = scale * (q - zero);
                }
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_AWQ),
            "quantize_row_qawq_g64: {k} is not divisible by {QK_AWQ}"
        );
        for (i, y) in ys.iter_mut().enumerate() {
            let xs_block = &xs[i * QK_AWQ..(i + 1) * QK_AWQ];

            // Process each 64-element group separately
            for group in 0..2 {
                let group_start = group * 64;
                let group_slice = &xs_block[group_start..group_start + 64];

                // Find min/max in group
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                for &x in group_slice.iter() {
                    min_val = min_val.min(x);
                    max_val = max_val.max(x);
                }

                let range = max_val - min_val;
                let (scale, zero) = if range < 1e-10 {
                    (0.0, 0.0)
                } else {
                    let s = range / 15.0;
                    let z = -min_val / s;
                    (s, z)
                };

                y.scales[group] = f16::from_f32(scale);
                y.zeros[group] = f16::from_f32(zero);

                // Quantize group elements
                let inv_scale = if scale.abs() > 1e-10 {
                    1.0 / scale
                } else {
                    0.0
                };
                let thread_base = group * 8; // Threads 0-7 for group 0, 8-15 for group 1
                for t in 0..8 {
                    let mut packed = 0u32;
                    for j in 0..8 {
                        let x = group_slice[t * 8 + j];
                        let q = ((x * inv_scale + zero).round().clamp(0.0, 15.0)) as u32;
                        packed |= q << (j * 4);
                    }
                    y.qs[thread_base + t] = packed;
                }
            }
            y._pad = 0;
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_AWQ),
            "vec_dot_qawq_g64: {n} is not divisible by {QK_AWQ}"
        );
        // Self-dotting: both xs and ys are BlockQAWQG64
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            for t in 0..16 {
                let group = t / 8;
                let x_scale = x.scales[group].to_f32();
                let x_zero = x.zeros[group].to_f32();
                let y_scale = y.scales[group].to_f32();
                let y_zero = y.zeros[group].to_f32();

                let x_packed = x.qs[t];
                let y_packed = y.qs[t];
                for j in 0..8 {
                    let xq = ((x_packed >> (j * 4)) & 0xF) as f32;
                    let yq = ((y_packed >> (j * 4)) & 0xF) as f32;
                    let x_val = x_scale * (xq - x_zero);
                    let y_val = y_scale * (yq - y_zero);
                    sumf += x_val * y_val;
                }
            }
        }
        sumf
    }
}
