//! The MXFP4 block: the OCP micro-scaling FP4 format that DeepSeek-V4's routed
//! experts are trained in.
//!
//! A block holds 32 elements as one E8M0 power-of-two scale byte and 16 packed
//! nibbles indexing the E2M1 value table. The layout matches llama.cpp's
//! `block_mxfp4` on disk, so GGUF files decode to the same bits. The table is in
//! ggml's integer convention, and the half-factor that restores the real
//! magnitudes is folded into the scale. The dot reduces to an exact int32
//! accumulation scaled by the two block scales.

use super::*;

/// MXFP4: 4-bit micro-scaling FP4 (OCP MXFP4), the native trained format for the
/// DeepSeek-V4 routed experts. 32 elements per block: one E8M0 (`ue8m0`) power-of-two
/// scale byte + 16 packed nibbles indexing the E2M1 value table. Wire-identical to
/// llama.cpp `block_mxfp4` (ggml file code 39) — see `ggml-common.h` / `ggml-impl.h`.
///
/// Decode: `value = MXFP4_KVALUES[nibble] * e8m0_to_f32_half(e)`, where the ×0.5 that
/// turns the integer table `{0,1,2,3,4,6,8,12,…}` back into the real E2M1 magnitudes
/// `{0,.5,1,1.5,2,3,4,6,…}` is folded into the "half" scale.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockMXFP4 {
    pub(crate) e: u8,
    pub(crate) qs: [u8; QK_MXFP4 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockMXFP4>() == 17);

/// E2M1 value table in ggml's integer convention (2× the real FP4 magnitude; the ×0.5
/// lives in [`e8m0_to_f32_half`]). Mirrors `kvalues_mxfp4` in `ggml-common.h`.
pub(crate) const MXFP4_KVALUES: [i8; 16] =
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// ggml `ggml_e8m0_to_fp32_half`: decode a `ue8m0` scale byte to `2^(e-128)` (i.e.
/// `0.5·2^(e-127)`), reproducing ggml's subnormal bit patterns for `e < 2` exactly so
/// MXFP4 blocks decode bit-identically to llama.cpp.
#[inline]
pub(crate) fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e < 2 {
        // 0x00200000 = 2^-128, 0x00400000 = 2^-127
        0x0020_0000u32 << e
    } else {
        // 0.5·2^(e-127) = 2^(e-128) = normalized fp32 with biased exponent (e-1)
        (e as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

/// Nearest E2M1 index for `x` under scale `d`, by absolute error. Mirrors ggml's
/// `best_index_mxfp4` (first-wins ties, ascending index scan).
#[inline]
pub(super) fn best_index_mxfp4(x: f32, d: f32) -> u8 {
    let mut best = 0usize;
    let mut best_err = (MXFP4_KVALUES[0] as f32 * d - x).abs();
    for (i, &kv) in MXFP4_KVALUES.iter().enumerate().take(16).skip(1) {
        let err = (kv as f32 * d - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best as u8
}

impl GgmlType for BlockMXFP4 {
    const DTYPE: GgmlDType = GgmlDType::MXFP4;
    const BLCK_SIZE: usize = QK_MXFP4;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let qk = Self::BLCK_SIZE;
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(qk),
            "dequantize_row_mxfp4: {k} is not divisible by {qk}"
        );
        let nb = k / qk;
        for i in 0..nb {
            let d = e8m0_to_f32_half(xs[i].e);
            for j in 0..(qk / 2) {
                let x0 = MXFP4_KVALUES[(xs[i].qs[j] & 0x0F) as usize] as f32;
                let x1 = MXFP4_KVALUES[(xs[i].qs[j] >> 4) as usize] as f32;
                ys[i * qk + j] = x0 * d;
                ys[i * qk + j + qk / 2] = x1 * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        debug_assert!(k.is_multiple_of(qk), "{k} is not divisible by {qk}");
        debug_assert_eq!(ys.len(), k / qk, "size mismatch {} {} {}", k, ys.len(), qk);
        for (i, y) in ys.iter_mut().enumerate() {
            let xb = &xs[i * qk..(i + 1) * qk];
            let mut amax = 0f32;
            for &v in xb.iter() {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
            }
            // e = floor(log2(amax)) - 2 + 127, clamped like ggml's uint8 cast.
            let e: u8 = if amax > 0.0 {
                (amax.log2().floor() - 2.0 + 127.0) as u8
            } else {
                0
            };
            let d = e8m0_to_f32_half(e);
            y.e = e;
            for j in 0..(qk / 2) {
                let lo = best_index_mxfp4(xb[j], d);
                let hi = best_index_mxfp4(xb[qk / 2 + j], d);
                y.qs[j] = lo | (hi << 4);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_MXFP4),
            "vec_dot_mxfp4_q8_0: {n} is not divisible by {QK_MXFP4}"
        );
        // The E2M1 table is integer-valued, so the dot reduces to an exact int32
        // accumulation scaled by the two per-block scales (mirrors vec_dot_q4_0_q8_0).
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sum_i = 0i32;
            for j in 0..QK_MXFP4 / 2 {
                let v0 = MXFP4_KVALUES[(xs.qs[j] & 0x0F) as usize] as i32;
                let v1 = MXFP4_KVALUES[(xs.qs[j] >> 4) as usize] as i32;
                sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + QK_MXFP4 / 2] as i32
            }
            sumf += sum_i as f32 * e8m0_to_f32_half(xs.e) * f16::to_f32(ys.d)
        }
        sumf
    }
}
