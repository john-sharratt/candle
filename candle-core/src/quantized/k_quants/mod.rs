//! Candle's quantized block formats for the CPU, each a `GgmlType` with encode,
//! decode and dot.
//!
//! The module defines the block layout of every quantized format the CPU path and
//! the GGUF loader handle, and the `GgmlType` trait that encodes, decodes and dots
//! them. Each format's block size is a `QK` constant, checked against its byte
//! size by `verify_block_sizes!`. The matmul hook takes a scratch buffer holding
//! the left operand quantized to the weight's vec-dot type, and f32, f16 and bf16
//! copy into it directly.
//!
//! - `primitive_impls`: the scalar types as `GgmlType`.
//! - `legacy_blocks`: the ggml Q4_0 to Q8_1 blocks.
//! - `kquants_q2_q4`: the Q2_K to Q4_K k-quant blocks.
//! - `kquants_q5_q8`: the Q5_K to Q8_K k-quant blocks.
//! - `ks_blocks`: the Q4_KS and Q8_KS blocks with sink sub-block scales.
//! - `cpu_ko_codecs`: CPU reference codecs for the KO layouts.
//! - `awq_blocks`: the 4-bit AWQ blocks.
//! - `candle_blocks`: Candle's Q2_0, Q3_0, R16 and P2 blocks.
//! - `mxfp4`: the MXFP4 block for the DeepSeek-V4 experts.
//! - `q0_family`: the INT8-scaled Q0 family.
//! - `q0_outlier`: Q0_X and the Q0_M2 and Q0_M4 mask blocks.
//! - `q0_v_codec`: the Q0_V pattern-indexed block.
//! - `tests_mxfp4`: tests for the MXFP4 block.

use super::utils::{
    get_scale_min_k4, group_for_dequantization, group_for_quantization, make_q3_quants,
    make_qkx1_quants, make_qx_quants, nearest_int,
};
use super::warp_mirror::{encode_centroid, lane8_sum, rint, warp_sum, INV_127, INV_255};
use super::GgmlDType;
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use rayon::prelude::*;

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;

pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;

pub const QK4_1: usize = 32;

pub const QK5_0: usize = 32;

pub const QK5_1: usize = 32;

pub const QK8_0: usize = 32;

pub const QK8_1: usize = 32;

pub const QK_Q4_KS: usize = 32;

pub const QK_Q8_KS: usize = 32;

pub const QK2_0: usize = 32;

pub const QK3_0: usize = 32;

pub const QK_MXFP4: usize = 32;

pub const QK_R16: usize = 32;

pub const QK_Q0: usize = 32;

pub const QK1_S: usize = 32;

pub const QK2_S: usize = 32;

pub const QK2_A: usize = 32;

pub const QK2_1: usize = 32;

pub const QK3_1: usize = 32;

pub const QK_P2: usize = 4;

pub const QK_Q0_V: usize = 32;

pub const QK1_A: usize = 32;

pub const QK_Q0_X: usize = 32;

pub const QK_Q0_M2: usize = 32;

pub const QK_Q0_M4: usize = 32;

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    const DIRECT_COPY: bool = false;
    type VecDotType: GgmlType;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]);
    fn from_float(xs: &[f32], ys: &mut [Self]);

    fn direct_copy(_xs: &[f32], _ys: &mut [Self]) {}

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32;

    /// Generic implementation of the dot product without simd optimizations.
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32;
}

// ==============================================================================
// Q0_V codebook tables — the reference codec's.
// ==============================================================================
/// Q0_V codebook tables. The CUDA side (q0_v_tables.cuh) holds the same scale
/// and centroid tables and, in place of the 128 curves, the four base curves
/// they are rotations of; the decode and encode oracle tests hold it to these.
/// K and V are calibrated separately; pick by IS_K at the call site.
#[allow(clippy::needless_range_loop)]
pub mod q0_v_tables;

// https://github.com/ggml-org/llama.cpp/blob/aa3ee0eb0b80efca126cedf9bcb4fb5864b46ce3/ggml/src/ggml-cpu/ggml-cpu.c#L1205
pub fn matmul<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(
        T::BLCK_SIZE,
        T::VecDotType::BLCK_SIZE,
        "Mismatched block sizes"
    );
    debug_assert_eq!(
        m * k,
        lhs.len(),
        "unexpected lhs length {} ({m},{k},{n})",
        lhs.len()
    );
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    // Scratch buffer holding the LHS pre-quantized to the weight's vec-dot type.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    // f32, f16, and bf16 support direct copy
    if T::DIRECT_COPY {
        T::VecDotType::direct_copy(lhs, &mut lhs_b);
    } else {
        for row_idx in 0..m {
            let lhs_b_mut = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
            T::VecDotType::from_float(lhs, lhs_b_mut)
        }
    }

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .for_each(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                *dst = T::vec_dot(k, rhs_col, lhs_row);
            });
    }
    Ok(())
}

macro_rules! verify_block_size {
    ( $block_type:ident ) => {
        const _: () =
            assert!($block_type::BLCK_SIZE == <$block_type as GgmlType>::VecDotType::BLCK_SIZE);
    };
}

macro_rules! verify_block_sizes {
    ( $( $block_type:ident ),* ) => {
        $(
            verify_block_size!($block_type);
        )*
    };
}

verify_block_sizes!(
    BlockQ4_0,
    BlockQ4_1,
    BlockQ5_0,
    BlockQ5_1,
    BlockQ8_0,
    BlockQ8_1,
    BlockQ4_KS,
    BlockQ8_KS,
    BlockQ2_0,
    BlockQ3_0,
    BlockR16,
    BlockQ2_K,
    BlockQ3_K,
    BlockQ4_K,
    BlockQ5_K,
    BlockQ6_K,
    BlockQ8_K,
    BlockQAWQ,
    BlockQAWQ_G64,
    f32,
    f16,
    bf16
);

mod awq_blocks;
mod candle_blocks;
mod cpu_ko_codecs;
mod kquants_q2_q4;
mod kquants_q5_q8;
mod ks_blocks;
mod legacy_blocks;
mod mxfp4;
mod primitive_impls;
mod q0_family;
mod q0_outlier;
mod q0_v_codec;
#[cfg(test)]
mod tests_mxfp4;

pub use awq_blocks::*;
pub use candle_blocks::*;
pub use cpu_ko_codecs::*;
pub use kquants_q2_q4::*;
pub use kquants_q5_q8::*;
pub use ks_blocks::*;
pub use legacy_blocks::*;
pub use mxfp4::*;
pub use q0_family::*;
pub use q0_outlier::*;
pub use q0_v_codec::*;
