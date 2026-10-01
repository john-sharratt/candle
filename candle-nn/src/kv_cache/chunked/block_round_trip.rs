//! One 32-element block through a KV format's host codec and back.
//!
//! The host codecs (`candle::quantized::k_quants`) reproduce the kernels'
//! block encoders and element decoders bit for bit (the KV block oracles in
//! `candle-core` pin that), so a host round trip is the round trip the GPU
//! selectors and the palette seal make. Both CPU selectors measure candidate
//! formats through this one function rather than through models of their own.
//!
//! Q0_V is calibrated per side, so the round trip takes `is_k`: a K block is
//! encoded and decoded against the K codebook.

use candle::quantized::k_quants::{
    decode_blocks_q0_v, encode_block_q0_v, BlockQ0, BlockQ0M2, BlockQ0M4, BlockQ0X, BlockQ1A,
    BlockQ1S, BlockQ2A, BlockQ2S, BlockQ2_0, BlockQ2_1, BlockQ3_0, BlockQ3_1, BlockQ4_0, BlockQ4_1,
    BlockQ4_KS, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, BlockQ8_KS, GgmlType,
};
use candle::quantized::GgmlDType;

use super::CHUNK_SIZE;

/// `block` encoded in `dtype` and decoded again, or `None` for a type that is
/// not a KV block format (floats, R16 — whose Q-capture half has no meaning
/// for a round trip — and the weight-only types).
pub(crate) fn block_round_trip(
    dtype: GgmlDType,
    block: &[f32; CHUNK_SIZE],
    is_k: bool,
) -> Option<[f32; CHUNK_SIZE]> {
    let mut out = [0.0f32; CHUNK_SIZE];
    match dtype {
        GgmlDType::Q4_0 => via::<BlockQ4_0>(block, &mut out),
        GgmlDType::Q4_1 => via::<BlockQ4_1>(block, &mut out),
        GgmlDType::Q5_0 => via::<BlockQ5_0>(block, &mut out),
        GgmlDType::Q5_1 => via::<BlockQ5_1>(block, &mut out),
        GgmlDType::Q8_0 => via::<BlockQ8_0>(block, &mut out),
        GgmlDType::Q8_1 => via::<BlockQ8_1>(block, &mut out),
        GgmlDType::Q4_KS => via::<BlockQ4_KS>(block, &mut out),
        GgmlDType::Q8_KS => via::<BlockQ8_KS>(block, &mut out),
        GgmlDType::Q2_0 => via::<BlockQ2_0>(block, &mut out),
        GgmlDType::Q3_0 => via::<BlockQ3_0>(block, &mut out),
        GgmlDType::Q2_1 => via::<BlockQ2_1>(block, &mut out),
        GgmlDType::Q3_1 => via::<BlockQ3_1>(block, &mut out),
        GgmlDType::Q0 => via::<BlockQ0>(block, &mut out),
        GgmlDType::Q1_S => via::<BlockQ1S>(block, &mut out),
        GgmlDType::Q2_S => via::<BlockQ2S>(block, &mut out),
        GgmlDType::Q2_A => via::<BlockQ2A>(block, &mut out),
        GgmlDType::Q1_A => via::<BlockQ1A>(block, &mut out),
        GgmlDType::Q0_X => via::<BlockQ0X>(block, &mut out),
        GgmlDType::Q0_M2 => via::<BlockQ0M2>(block, &mut out),
        GgmlDType::Q0_M4 => via::<BlockQ0M4>(block, &mut out),
        GgmlDType::Q0_V if is_k => {
            decode_blocks_q0_v::<true>(&[encode_block_q0_v::<true>(block)], &mut out)
        }
        GgmlDType::Q0_V => {
            decode_blocks_q0_v::<false>(&[encode_block_q0_v::<false>(block)], &mut out)
        }
        _ => return None,
    }
    Some(out)
}

fn via<B: GgmlType>(block: &[f32; CHUNK_SIZE], out: &mut [f32; CHUNK_SIZE]) {
    let mut blk = [B::zeros()];
    B::from_float(block, &mut blk);
    B::to_float(&blk, out);
}
