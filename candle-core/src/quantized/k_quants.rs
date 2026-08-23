use super::utils::{
    get_scale_min_k4, group_for_dequantization, group_for_quantization, make_q3_quants,
    make_qkx1_quants, make_qx_quants, nearest_int,
};
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
/// 32Ã—F16 primary values (64 bytes) + 32Ã—u16 reserved Q space (64 bytes) = 128 bytes
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
fn best_index_mxfp4(x: f32, d: f32) -> u8 {
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

/// Q1_S: 1-bit sign + FP8 E4M3 scale
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ1S {
    pub(crate) scale: u8,
    pub(crate) qs: [u8; QK1_S / 8],
}
const _: () = assert!(std::mem::size_of::<BlockQ1S>() == 5);

/// Q2_S: 2-bit symmetric + FP8 E4M3 scale
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2S {
    pub(crate) scale: u8,
    pub(crate) qs: [u8; QK2_S / 4],
}
const _: () = assert!(std::mem::size_of::<BlockQ2S>() == 9);

/// Q2_A: 2-bit asymmetric + FP8 E4M3 scale + FP8 E4M3 min
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2A {
    pub(crate) scale: u8,
    pub(crate) bias: u8,
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

/// P2: 2-bit palette index â€” pure arena routing, not a quant.
/// Each byte packs 4 head_dim positions Ã— 2-bit indices.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockP2 {
    pub(crate) packed: u8,
}
const _: () = assert!(std::mem::size_of::<BlockP2>() == 1);

/// Q0_V: Pattern-indexed structured quantization (2 bytes per block,
/// 0.53 BPE effective with the per-group 4-byte header amortised).
///
/// Per-block layout (manually packed):
///   `deltas`  â€” bits[3:0]=mag_delta (4-bit), bits[7:4]=out_delta (4-bit)
///   `pattern` â€” bits[3:0]=sign_pattern (4-bit), bits[7:4]=shape (4-bit)
///
/// See `candle-kernels/src/quantize/q0_v_tables.cuh` for the constant
/// lookup tables.
///
/// Per-block layout (2 bytes / 16 bits total):
///   bits[6:0]   = curve_idx    (7-bit, 128 entries)
///   bits[11:7]  = scale_idx    (5-bit, 32 entries)
///   bits[15:12] = centroid_idx (4-bit, 16 entries per scale row)
///
/// Byte view:
///   lo: bits[6:0]=curve_idx,      bit[7]=scale_idx[0]
///   hi: bits[3:0]=scale_idx[4:1], bits[7:4]=centroid_idx
///
/// Reallocated one bit from `curve_idx` (was 8-bit/256 entries) to
/// `centroid_idx` (was 3-bit/8 entries). Picking more than ~4 useful
/// curves per side is empirically hard, but doubling the centroid
/// resolution gives noticeably better per-block magnitude/offset fits.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0V {
    pub(crate) lo: u8,
    pub(crate) hi: u8,
}
const _: () = assert!(std::mem::size_of::<BlockQ0V>() == 2);

impl BlockQ0V {
    pub(crate) fn pack(curve_idx: u8, scale_idx: u8, centroid_idx: u8) -> Self {
        let curve = (curve_idx & 0x7F) as u16;
        let scale = (scale_idx & 0x1F) as u16;
        let cent = (centroid_idx & 0x0F) as u16;
        let bits: u16 = curve | (scale << 7) | (cent << 12);
        Self {
            lo: (bits & 0xFF) as u8,
            hi: ((bits >> 8) & 0xFF) as u8,
        }
    }
    pub(crate) fn curve_idx(&self) -> usize {
        (self.lo as usize) & 0x7F
    }
    pub(crate) fn scale_idx(&self) -> usize {
        let bits: u16 = (self.lo as u16) | ((self.hi as u16) << 8);
        ((bits >> 7) & 0x1F) as usize
    }
    pub(crate) fn centroid_idx(&self) -> usize {
        ((self.hi as usize) >> 4) & 0x0F
    }

    /// The block's two raw bytes `[lo, hi]` — the exact storage layout, for
    /// external codecs (e.g. the latent band mirror) that author arena bytes.
    pub fn to_le_bytes(&self) -> [u8; 2] {
        [self.lo, self.hi]
    }

    /// Reconstruct a block from its two raw storage bytes.
    pub fn from_le_bytes(b: [u8; 2]) -> Self {
        Self { lo: b[0], hi: b[1] }
    }
}

/// Q1_A: 1-bit asymmetric â€” separate INT8 amplitude per sign + 32 sign bits
/// (6 bytes per 32 elements, 1.50 BPE).
///
///   sign_bit = 1 â†’ x = +scale_pos / 127
///   sign_bit = 0 â†’ x = -scale_neg / 127
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ1A {
    pub(crate) scale_pos: i8,
    pub(crate) scale_neg: i8,
    pub(crate) qs: [u8; 4],
}
const _: () = assert!(std::mem::size_of::<BlockQ1A>() == 6);

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

/// Q0_M2: Two E4M3 constants + 8-bit quartet mask (3 bytes per 32 elements)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0M2 {
    pub(crate) val_fp8: [u8; 2],
    pub(crate) qmask: u8,
}
const _: () = assert!(std::mem::size_of::<BlockQ0M2>() == 3);

/// Q0_M4: Four E4M3 constants + 32-bit pair mask (8 bytes per 32 elements)
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ0M4 {
    pub(crate) val_fp8: [u8; 4],
    pub(crate) qmask: u32,
}
const _: () = assert!(std::mem::size_of::<BlockQ0M4>() == 8);

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

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5_K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6_K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6_K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8_K>());

// =============================================================================
// AWQ (ACTIVATION-AWARE WEIGHT QUANTIZATION) BLOCK STRUCTURES
// =============================================================================
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
    pub(crate) qs: [u32; 16],  // 128 Ã— 4-bit = 64 bytes
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
    pub(crate) qs: [u32; 16],    // 128 Ã— 4-bit = 64 bytes
    pub(crate) scales: [f16; 2], // scale per 64-element group
    pub(crate) zeros: [f16; 2],  // zero per 64-element group
    pub(crate) _pad: u32,        // padding to 80 bytes
}
const _: () = assert!(std::mem::size_of::<BlockQAWQ_G64>() == 80);

/// BlockQ4_KO: the **CPU-side** per-32 codec — a byte-permuted twin of the Q4_K compact (K/128)
/// block. Holds the SAME data as Q4_K's compact block — 128 × 4-bit weights (interleaved nibble
/// order {n0,n4,n2,n6,n1,n5,n3,n7} within each int) plus four per-32 (scale, neg_min) f16 pairs —
/// reordered so the 16 qs ints occupy bytes 0-63 and the scale pairs group at the tail (bytes
/// 64-79). Within each 32-element sub the four qs ints are stored INTERLEAVED as [I0,I2,I1,I3];
/// element nibbles dequantize as `value = scale_s * q + neg_min_s`, sub `s = element/32`. Used only
/// for CPU reference / roundtrip tests, and for `type_size`/`block_size`.
///
/// IMPORTANT — this is NOT the layout the int8 matmul consumes. GPU-produced `Q4_KO` weights (the
/// matmul inputs, built by `ko_quant`/`QStorage::repack_ko`) use a distinct **per-128 affine,
/// re-quantized-from-F32** chunk layout (8 rows × 128 K per chunk, 68 B/row — see `ko_quant` and
/// `dequant_ko.cuh`). The two share the `GgmlDType::Q4_KO` discriminant but are NOT byte-compatible
/// and carry different granularity, so a GPU KO weight must not be dequantized through this codec.
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(16))]
pub struct BlockQ4_KO {
    pub(crate) qs: [u32; 16], // 128 × 4-bit, [I0,I2,I1,I3] per sub (bytes 0-63)
    pub(crate) dm: [f16; 8],  // 4 × (scale, neg_min) pairs (bytes 64-79)
}
const _: () = assert!(std::mem::size_of::<BlockQ4_KO>() == 80);

// Bit position of element j (0-7) inside a qs int: the Q4_K "LOP3-ready" nibble
// order {n0,n4,n2,n6,n1,n5,n3,n7}. Shared by every BlockQ4_KO codec method.
const Q4_KO_SHIFTS: [u32; 8] = [0, 16, 8, 24, 4, 20, 12, 28];
// Within a sub, struct slot k holds the original qs int [0,2,1,3][k] (the [I0,I2,I1,I3]
// interleave that makes a lane's qlo/qhi adjacent for the int2 load). Self-inverse.
const Q4_KO_PERM: [usize; 4] = [0, 2, 1, 3];

impl GgmlType for BlockQ4_KO {
    const DTYPE: GgmlDType = GgmlDType::Q4_KO;
    const BLCK_SIZE: usize = 128;
    type VecDotType = BlockQ4_KO; // self-dotting for CPU matmul

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(Self::BLCK_SIZE),
            "dequantize_row_q4_KO: {k} is not divisible by {}",
            Self::BLCK_SIZE
        );
        for (i, x) in xs.iter().enumerate() {
            for t in 0..16 {
                let sub = t / 4; // 4 qs ints per 32-element sub-block
                let orig_m = (t & !3) + Q4_KO_PERM[t & 3]; // de-interleave to element order
                let d = x.dm[2 * sub].to_f32();
                let m = x.dm[2 * sub + 1].to_f32();
                let packed = x.qs[t];
                for j in 0..8 {
                    let q = ((packed >> Q4_KO_SHIFTS[j]) & 0xF) as f32;
                    ys[i * Self::BLCK_SIZE + orig_m * 8 + j] = d * q + m;
                }
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(Self::BLCK_SIZE),
            "quantize_row_q4_KO: {k} is not divisible by {}",
            Self::BLCK_SIZE
        );
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            // Per-32 affine fit: value = scale * q + min, q in [0,15].
            for sub in 0..4 {
                let g = &block[sub * 32..(sub + 1) * 32];
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                for &v in g {
                    min_val = min_val.min(v);
                    max_val = max_val.max(v);
                }
                let scale = if (max_val - min_val) > 1e-10 {
                    (max_val - min_val) / 15.0
                } else {
                    0.0
                };
                y.dm[2 * sub] = f16::from_f32(scale);
                y.dm[2 * sub + 1] = f16::from_f32(min_val);
            }
            for t in 0..16 {
                let sub = t / 4;
                let orig_m = (t & !3) + Q4_KO_PERM[t & 3]; // de-interleave to element order
                let m = y.dm[2 * sub + 1].to_f32();
                let scale = y.dm[2 * sub].to_f32();
                let inv = if scale.abs() > 1e-10 {
                    1.0 / scale
                } else {
                    0.0
                };
                let mut packed = 0u32;
                for j in 0..8 {
                    let x = block[orig_m * 8 + j];
                    let q = (((x - m) * inv).round().clamp(0.0, 15.0)) as u32;
                    packed |= q << Q4_KO_SHIFTS[j];
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
            n.is_multiple_of(Self::BLCK_SIZE),
            "vec_dot_q4_KO: {n} is not divisible by {}",
            Self::BLCK_SIZE
        );
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            for t in 0..16 {
                let sub = t / 4;
                let (xd, xm) = (x.dm[2 * sub].to_f32(), x.dm[2 * sub + 1].to_f32());
                let (yd, ym) = (y.dm[2 * sub].to_f32(), y.dm[2 * sub + 1].to_f32());
                let xp = x.qs[t];
                let yp = y.qs[t];
                for &sh in &Q4_KO_SHIFTS {
                    let xq = ((xp >> sh) & 0xF) as f32;
                    let yq = ((yp >> sh) & 0xF) as f32;
                    sumf += (xd * xq + xm) * (yd * yq + ym);
                }
            }
        }
        sumf
    }
}

// ===========================================================================
// Q5_KO / Q6_KO / Q8_KO — CPU-side per-32 codecs (byte-permuted twins of the
// Q5_K/Q6_K/Q8_K compact blocks). As with BlockQ4_KO above, these are CPU
// reference/roundtrip codecs only; the int8 matmul consumes the distinct
// per-128 from-F32 `ko_quant` layout under the same GgmlDType discriminant.
// ===========================================================================

/// BlockQ5_KO: Q5_K twin — qs (low nibbles) contiguous, qh (5th bits) contiguous,
/// four per-32 (scale, neg_min) pairs at the tail. `value = scale_s·q5 + min_s`
/// for the 32-element sub `s`, with `q5 = nibble | (5th_bit << 4)` (0..31).
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(16))]
pub struct BlockQ5_KO {
    pub(crate) qs: [u32; 16],  // 0-63
    pub(crate) qh: [u32; 4],   // 64-79: one 5th-bit int per thread-quad
    pub(crate) dm: [f16; 8],   // 80-95: 4 × (scale, neg_min)
    pub(crate) _pad: [u32; 4], // 96-111
}
const _: () = assert!(std::mem::size_of::<BlockQ5_KO>() == 112);

impl GgmlType for BlockQ5_KO {
    const DTYPE: GgmlDType = GgmlDType::Q5_KO;
    const BLCK_SIZE: usize = 128;
    type VecDotType = BlockQ5_KO;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert!(ys.len().is_multiple_of(128));
        for (i, x) in xs.iter().enumerate() {
            for m in 0..16 {
                let sub = m / 4;
                let orig_m = (m & !3) + Q4_KO_PERM[m & 3]; // de-interleave qs slot
                let qs = x.qs[m];
                let qh = x.qh[sub];
                for j in 0..8 {
                    let nib = (qs >> Q4_KO_SHIFTS[j]) & 0xF;
                    let hb = (qh >> ((orig_m & 3) as u32 * 8 + j as u32)) & 1;
                    let q5 = (nib | (hb << 4)) as f32; // 0..31
                    ys[i * 128 + orig_m * 8 + j] =
                        x.dm[2 * sub].to_f32() * q5 + x.dm[2 * sub + 1].to_f32();
                }
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert!(xs.len().is_multiple_of(128));
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * 128..(i + 1) * 128];
            for s in 0..4 {
                let g = &block[s * 32..s * 32 + 32];
                let (mut mn, mut mx) = (f32::MAX, f32::MIN);
                for &v in g {
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = if (mx - mn) > 1e-10 {
                    (mx - mn) / 31.0
                } else {
                    0.0
                };
                y.dm[2 * s] = f16::from_f32(scale);
                y.dm[2 * s + 1] = f16::from_f32(mn);
            }
            y.qs = [0; 16];
            y.qh = [0; 4];
            y._pad = [0; 4];
            for m in 0..16 {
                let sub = m / 4;
                let orig_m = (m & !3) + Q4_KO_PERM[m & 3]; // de-interleave qs slot
                let scale = y.dm[2 * sub].to_f32();
                let mn = y.dm[2 * sub + 1].to_f32();
                let inv = if scale.abs() > 1e-10 {
                    1.0 / scale
                } else {
                    0.0
                };
                for j in 0..8 {
                    let q5 = (((block[orig_m * 8 + j] - mn) * inv)
                        .round()
                        .clamp(0.0, 31.0)) as u32;
                    y.qs[m] |= (q5 & 0xF) << Q4_KO_SHIFTS[j];
                    y.qh[sub] |= ((q5 >> 4) & 1) << ((orig_m & 3) as u32 * 8 + j as u32);
                }
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(128));
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        Self::to_float(xs, &mut a);
        Self::to_float(ys, &mut b);
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// BlockQ6_KO: Q6_K twin. Q6_K's compact block is already ordered (ql contiguous,
/// qh contiguous, per-16 scales at the tail), so this is byte-identical to it.
/// `value = scale_g·(q6 - 32)` for the 16-element group `g`, with
/// `q6 = nibble | (crumb << 4)` (0..63).
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(16))]
pub struct BlockQ6_KO {
    pub(crate) ql: [u32; 16],    // 0-63
    pub(crate) qh: [u32; 8],     // 64-95: one 2-bit-crumb int per thread-pair
    pub(crate) scales: [f16; 8], // 96-111: per-16 scale
}
const _: () = assert!(std::mem::size_of::<BlockQ6_KO>() == 112);

impl GgmlType for BlockQ6_KO {
    const DTYPE: GgmlDType = GgmlDType::Q6_KO;
    const BLCK_SIZE: usize = 128;
    type VecDotType = BlockQ6_KO;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert!(ys.len().is_multiple_of(128));
        for (i, x) in xs.iter().enumerate() {
            for m in 0..16 {
                let ql = x.ql[m];
                let qh16 = (x.qh[m >> 1] >> ((m & 1) * 16)) & 0xFFFF;
                for (j, &sh) in Q4_KO_SHIFTS.iter().enumerate() {
                    let nib = (ql >> sh) & 0xF;
                    let crumb = (qh16 >> (2 * j as u32)) & 3;
                    let q6 = (nib | (crumb << 4)) as i32; // 0..63
                    let idx = m * 8 + j;
                    ys[i * 128 + idx] = x.scales[idx / 16].to_f32() * (q6 - 32) as f32;
                }
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert!(xs.len().is_multiple_of(128));
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * 128..(i + 1) * 128];
            for g in 0..8 {
                let grp = &block[g * 16..g * 16 + 16];
                let amax = grp.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let scale = if amax > 1e-10 { amax / 32.0 } else { 0.0 };
                y.scales[g] = f16::from_f32(scale);
            }
            y.ql = [0; 16];
            y.qh = [0; 8];
            for m in 0..16 {
                for (j, _) in Q4_KO_SHIFTS.iter().enumerate() {
                    let idx = m * 8 + j;
                    let scale = y.scales[idx / 16].to_f32();
                    let inv = if scale.abs() > 1e-10 {
                        1.0 / scale
                    } else {
                        0.0
                    };
                    let q6 = ((block[idx] * inv).round() as i32 + 32).clamp(0, 63) as u32;
                    y.ql[m] |= (q6 & 0xF) << Q4_KO_SHIFTS[j];
                    y.qh[m >> 1] |= ((q6 >> 4) & 3) << ((m & 1) as u32 * 16 + 2 * j as u32);
                }
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(128));
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        Self::to_float(xs, &mut a);
        Self::to_float(ys, &mut b);
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// BlockQ8_KO: Q8_K twin — the 16 qs int2 made contiguous (natural element order),
/// the single block scale (replicated ×4) at the tail. `value = scale·q8`.
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(16))]
pub struct BlockQ8_KO {
    pub(crate) qs: [i8; 128],  // 0-127
    pub(crate) d: [f16; 8],    // 128-143: block scale at d[0]/d[2]/d[4]/d[6]
    pub(crate) _pad: [u32; 4], // 144-159
}
const _: () = assert!(std::mem::size_of::<BlockQ8_KO>() == 160);

impl GgmlType for BlockQ8_KO {
    const DTYPE: GgmlDType = GgmlDType::Q8_KO;
    const BLCK_SIZE: usize = 128;
    type VecDotType = BlockQ8_KO;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert!(ys.len().is_multiple_of(128));
        for (i, x) in xs.iter().enumerate() {
            let d = x.d[0].to_f32();
            for j in 0..128 {
                ys[i * 128 + j] = d * x.qs[j] as f32;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert!(xs.len().is_multiple_of(128));
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * 128..(i + 1) * 128];
            let amax = block.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let scale = if amax > 1e-10 { amax / 127.0 } else { 0.0 };
            let inv = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
            // Block scale replicated across the 4 read slots (d[0]/d[2]/d[4]/d[6]).
            y.d = [f16::from_f32(0.0); 8];
            for k in 0..4 {
                y.d[2 * k] = f16::from_f32(scale);
            }
            y._pad = [0; 4];
            for (q, &b) in y.qs.iter_mut().zip(block.iter()) {
                *q = (b * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(128));
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let xd = x.d[0].to_f32();
            let yd = y.d[0].to_f32();
            for j in 0..128 {
                sumf += (xd * x.qs[j] as f32) * (yd * y.qs[j] as f32);
            }
        }
        sumf
    }
}

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
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q4_0_q8_0(n, xs, ys);

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
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8_0_q8_0(n, xs, ys);

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

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q8_1
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
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
            let mut sum = 0i32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let v0 = xs[j] * id;
                let v1 = xs[j + Self::BLCK_SIZE / 2] * id;
                ys.qs[j] = f32::round(v0) as i8;
                ys.qs[j + Self::BLCK_SIZE / 2] = f32::round(v1) as i8;
                sum += ys.qs[j] as i32 + ys.qs[j + Self::BLCK_SIZE / 2] as i32;
            }
            ys.s = f16::from_f32(sum as f32) * ys.d;
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

impl GgmlType for BlockQ4_KS {
    const DTYPE: GgmlDType = GgmlDType::Q4_KS;
    const BLCK_SIZE: usize = QK_Q4_KS;
    type VecDotType = BlockQ4_KS;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_Q4_KS));
        let nb = k / QK_Q4_KS;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let da = d * xs[i].sa as f32 / 255.0;
            let db = d * xs[i].sb as f32 / 255.0;
            for j in 0..QK_Q4_KS / 2 {
                let lo_nibble = (xs[i].qs[j] & 0x0F) as i32 - 8;
                let hi_nibble = (xs[i].qs[j] >> 4) as i32 - 8;
                let scale_lo = if j < 2 { da } else { db }; // j<2 covers elems 0-3 (4 elems)
                ys[i * QK_Q4_KS + j] = scale_lo * lo_nibble as f32;
                ys[i * QK_Q4_KS + j + QK_Q4_KS / 2] = db * hi_nibble as f32;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax_a = xs[..4].iter().map(|x| x.abs()).fold(0f32, f32::max);
            let amax_b = xs[4..].iter().map(|x| x.abs()).fold(0f32, f32::max);
            let amax = amax_a.max(amax_b);
            let d = amax / 7.0;
            ys.d = f16::from_f32(d);
            if amax == 0.0 {
                ys.sa = 255;
                ys.sb = 255;
            } else {
                ys.sa = ((amax_a / amax * 255.0).round() as u8).clamp(1, 255);
                ys.sb = ((amax_b / amax * 255.0).round() as u8).clamp(1, 255);
            }
            let da = d * ys.sa as f32 / 255.0;
            let db = d * ys.sb as f32 / 255.0;
            for j in 0..QK_Q4_KS / 2 {
                let scale_lo = if j < 2 { da } else { db }; // j<2 -> elems 0-3
                let id_lo = if scale_lo != 0.0 { 1.0 / scale_lo } else { 0.0 };
                let id_hi = if db != 0.0 { 1.0 / db } else { 0.0 };
                let lo = (xs[j] * id_lo).round().clamp(-7.0, 7.0) as i8 + 8;
                let hi = (xs[j + QK_Q4_KS / 2] * id_hi).round().clamp(-7.0, 7.0) as i8 + 8;
                ys.qs[j] = (lo as u8) | ((hi as u8) << 4);
            }
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

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_Q8_KS));
        let nb = k / QK_Q8_KS;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let da = d * xs[i].sa as f32 / 255.0;
            let db = d * xs[i].sb as f32 / 255.0;
            for j in 0..QK_Q8_KS {
                let scale = if j < 4 { da } else { db };
                ys[i * QK_Q8_KS + j] = scale * xs[i].qs[j] as f32;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(k.is_multiple_of(Self::BLCK_SIZE));
        debug_assert_eq!(ys.len(), k / Self::BLCK_SIZE);
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax_a = xs[..4].iter().map(|x| x.abs()).fold(0f32, f32::max);
            let amax_b = xs[4..].iter().map(|x| x.abs()).fold(0f32, f32::max);
            let amax = amax_a.max(amax_b);
            let d = amax / 127.0;
            ys.d = f16::from_f32(d);
            if amax == 0.0 {
                ys.sa = 255;
                ys.sb = 255;
            } else {
                ys.sa = ((amax_a / amax * 255.0).round() as u8).clamp(1, 255);
                ys.sb = ((amax_b / amax * 255.0).round() as u8).clamp(1, 255);
            }
            let da = d * ys.sa as f32 / 255.0;
            let db = d * ys.sb as f32 / 255.0;
            for (j, (q, &x)) in ys.qs.iter_mut().zip(xs.iter()).enumerate() {
                let scale = if j < 4 { da } else { db };
                let id = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                *q = (x * id).round().clamp(-127.0, 127.0) as i8;
            }
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
            let amax = xs.iter().map(|x| x.abs()).fold(0f32, f32::max);
            if amax == 0.0 {
                ys.d = f16::from_f32(0.0);
                ys.qs.fill(0);
                continue;
            }
            ys.d = f16::from_f32(amax / 1.5);
            let id = 1.5 / amax;
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
            let amax = xs.iter().map(|x| x.abs()).fold(0f32, f32::max);
            if amax == 0.0 {
                ys.d = f16::from_f32(0.0);
                ys.qh.fill(0);
                ys.qs.fill(0);
                continue;
            }
            ys.d = f16::from_f32(amax / 3.5);
            ys.qh.fill(0);
            ys.qs.fill(0);
            let id = 3.5 / amax;
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

// â”€â”€ FP8 E4M3 helpers for CPU decode/encode â”€â”€
fn decode_e4m3(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = ((bits >> 3) & 0xf) as i32;
    let mantissa = (bits & 7) as f32;
    if exp == 0 {
        let val = mantissa / 8.0 * 2.0f32.powi(-6);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 15 && mantissa == 7.0 {
        // E4M3 encodes NaN with either sign bit; both decode to the same NaN.
        f32::NAN
    } else {
        let val = (1.0 + mantissa / 8.0) * 2.0f32.powi(exp - 7);
        if sign == 1 {
            -val
        } else {
            val
        }
    }
}

fn encode_e4m3(x: f32) -> u8 {
    if x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let ax = x.abs().min(448.0);
    let (frac, exp_raw) = {
        let bits = ax.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mantissa = (bits & 0x7fffff) as f32 / (1 << 23) as f32;
        (1.0 + mantissa, exp)
    };
    let biased = (exp_raw + 7).clamp(0, 15) as u8;
    let m = ((frac - 1.0) * 8.0).round().clamp(0.0, 7.0) as u8;
    (sign << 7) | (biased << 3) | m
}

impl GgmlType for BlockQ0 {
    const DTYPE: GgmlDType = GgmlDType::Q0;
    const BLCK_SIZE: usize = QK_Q0;
    type VecDotType = BlockQ8_0;

    /// Decode: every lane in the block expands to the same constant.
    /// `centroid as f32 / 127.0`. The outer-scale division is applied by
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
            let v = xs[i].centroid as f32 / 127.0;
            for j in 0..QK_Q0 {
                ys[i * QK_Q0 + j] = v;
            }
        }
    }

    /// Encode: take the block mean, scale to i8 [-127, 127] space.
    /// The caller is expected to have applied outer scaling so input
    /// values lie roughly in [-1, +1].
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_Q0),
            "from_float Q0: {k} is not divisible by {QK_Q0}"
        );
        let nb = k / QK_Q0;
        for i in 0..nb {
            let block = &xs[i * QK_Q0..(i + 1) * QK_Q0];
            let mean: f32 = block.iter().sum::<f32>() / QK_Q0 as f32;
            let q = (mean.clamp(-1.0, 1.0) * 127.0).round() as i32;
            ys[i].centroid = q.clamp(-127, 127) as i8;
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

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK1_S;
        for i in 0..nb {
            let scale = decode_e4m3(xs[i].scale);
            for j in 0..QK1_S {
                let bit = (xs[i].qs[j / 8] >> (j % 8)) & 1;
                ys[i * QK1_S + j] = if bit == 1 { scale } else { -scale };
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax = block.iter().map(|x| x.abs()).fold(0f32, f32::max);
            y.scale = encode_e4m3(amax);
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

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK2_S;
        for i in 0..nb {
            let d = decode_e4m3(xs[i].scale);
            for j in 0..QK2_S {
                let q = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as f32;
                ys[i * QK2_S + j] = d * (q - 1.5);
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let amax = block.iter().map(|x| x.abs()).fold(0f32, f32::max);
            if amax == 0.0 {
                y.scale = 0;
                y.qs.fill(0);
                continue;
            }
            let d = amax / 1.5;
            y.scale = encode_e4m3(d);
            y.qs.fill(0);
            let id = 1.5 / amax;
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

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let nb = ys.len() / QK2_A;
        for i in 0..nb {
            let d = decode_e4m3(xs[i].scale);
            let m = decode_e4m3(xs[i].bias);
            for j in 0..QK2_A {
                let q = ((xs[i].qs[j / 4] >> ((j % 4) * 2)) & 3) as f32;
                ys[i * QK2_A + j] = q * d + m;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let vmax = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let vmin = block.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = vmax - vmin;
            if range == 0.0 {
                y.scale = 0;
                y.bias = encode_e4m3(vmin);
                y.qs.fill(0);
                continue;
            }
            let d = range / 3.0;
            y.scale = encode_e4m3(d);
            y.bias = encode_e4m3(vmin);
            y.qs.fill(0);
            let id = 3.0 / range;
            for (j, &b) in block.iter().enumerate().take(QK2_A) {
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

impl GgmlType for BlockP2 {
    const DTYPE: GgmlDType = GgmlDType::P2;
    const BLCK_SIZE: usize = QK_P2;
    type VecDotType = BlockQ8_0; // unused â€” P2 is not a quant

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

/// IS_K-aware decode of a single Q0_V block element. Pure-Rust reference
/// for the `q0_v_elem<IS_K>` CUDA function in
/// `candle-kernels/src/convert/block_q0_v.cuh`.
#[inline]
pub fn q0_v_elem<const IS_K: bool>(block: &BlockQ0V, e: usize) -> f32 {
    let curve_idx = block.curve_idx();
    let scale_idx = block.scale_idx();
    let centroid_idx = block.centroid_idx();
    let (scale_bits, centroid_bits, curve_e) = if IS_K {
        (
            q0_v_tables::SCALE_TABLE_BITS_K[scale_idx],
            q0_v_tables::CENTROID_TABLE_BITS_K[scale_idx][centroid_idx],
            q0_v_tables::CURVE_TABLE_K[curve_idx][e] as f32,
        )
    } else {
        (
            q0_v_tables::SCALE_TABLE_BITS_V[scale_idx],
            q0_v_tables::CENTROID_TABLE_BITS_V[scale_idx][centroid_idx],
            q0_v_tables::CURVE_TABLE_V[curve_idx][e] as f32,
        )
    };
    let scale = half::f16::from_bits(scale_bits).to_f32();
    let centroid = half::f16::from_bits(centroid_bits).to_f32();
    scale.mul_add(curve_e, centroid)
}

/// IS_K-aware decode of a slice of Q0_V blocks. xs and ys cover N blocks.
pub fn decode_blocks_q0_v<const IS_K: bool>(xs: &[BlockQ0V], ys: &mut [f32]) {
    for (i, x) in xs.iter().enumerate() {
        for j in 0..QK_Q0_V {
            ys[i * QK_Q0_V + j] = q0_v_elem::<IS_K>(x, j);
        }
    }
}

impl GgmlType for BlockQ0V {
    const DTYPE: GgmlDType = GgmlDType::Q0_V;
    const BLCK_SIZE: usize = QK_Q0_V;
    type VecDotType = BlockQ8_0;

    /// Decode using V-side tables. Callers that need K-side decode (e.g. the
    /// K-pass of attention or paged-decode K) should use
    /// `decode_blocks_q0_v::<true>` directly. The trait method has no IS_K
    /// parameter so we default to V; matches the CUDA `BlockConverter`
    /// default-V specialisation.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        decode_blocks_q0_v::<false>(xs, ys);
    }

    /// Encode using V-side tables (default). K-side callers should use
    /// `encode_block_q0_v::<true>` directly.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            *y = encode_block_q0_v::<false>(block);
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

/// Per-block Q0_V encoder. Caller is expected to have already applied
/// `outer_scale` (1/head_amax) to the input so values are roughly in [-1, +1].
/// IS_K selects between K-side and V-side calibrated tables.
///
/// Pipeline matches the CUDA encoder in
/// `candle-kernels/src/quantize/quantize_q0_v.cuh`:
///   1. Compute actual (centroid, scale) = (mean(x), max|x − mean(x)|).
///   2. Pick scale_idx (32 entries) by argmin |actual_scale − scale_table[i]|.
///   3. Pick centroid_idx (8 entries within chosen scale row).
///   4. Normalise target[lane] = (x[lane] − chosen_centroid) / scale_baked,
///      where scale_baked = scale_norm / 127. After this, target is directly
///      comparable to the i8 curve_table values (no /127 needed).
///   5. Peak-bin curve search: find the lane of max |target|, then score
///      only the curves whose own peak lane is in {peak−1, peak, peak+1}
///      (cyclic). The 256-curve table is pre-sorted by peak lane and
///      indexed by `PEAK_CURVE_INDICES` / `PEAK_BIN_OFFSETS`.
///   6. Pack curve_idx (8 bits) + scale_idx (5 bits) + centroid_idx (3 bits).
pub fn encode_block_q0_v<const IS_K: bool>(block: &[f32]) -> BlockQ0V {
    debug_assert_eq!(block.len(), QK_Q0_V);

    // Per-side table references.
    let curve_table: &[[i8; 32]; 128] = if IS_K {
        &q0_v_tables::CURVE_TABLE_K
    } else {
        &q0_v_tables::CURVE_TABLE_V
    };
    let scale_table: &[u16; 32] = if IS_K {
        &q0_v_tables::SCALE_TABLE_BITS_K
    } else {
        &q0_v_tables::SCALE_TABLE_BITS_V
    };
    let centroid_table: &[[u16; 16]; 32] = if IS_K {
        &q0_v_tables::CENTROID_TABLE_BITS_K
    } else {
        &q0_v_tables::CENTROID_TABLE_BITS_V
    };

    // ── Step 1: actual (centroid, scale) of the block ──
    let mean: f32 = block.iter().sum::<f32>() / QK_Q0_V as f32;
    let mut max_dev = 0.0f32;
    for &x in block.iter() {
        let d = (x - mean).abs();
        if d > max_dev {
            max_dev = d;
        }
    }
    let actual_centroid = mean;
    let actual_scale = max_dev;

    // ── Step 2: pick scale_idx (32 entries) ──
    // Stored value = scale_norm / 127, so multiply by 127 for comparison.
    let mut best_scale_idx = 0u8;
    let mut best_scale_err = f32::INFINITY;
    for (i, &raw) in scale_table.iter().enumerate().take(32) {
        let scale_baked = half::f16::from_bits(raw).to_f32();
        let s = scale_baked * 127.0;
        let err = (actual_scale - s).abs();
        if err < best_scale_err {
            best_scale_err = err;
            best_scale_idx = i as u8;
        }
    }
    let chosen_scale_baked = half::f16::from_bits(scale_table[best_scale_idx as usize]).to_f32();

    // ── Step 3: pick centroid_idx within the chosen scale row (16 entries) ──
    let mut best_centroid_idx = 0u8;
    let mut best_centroid_err = f32::INFINITY;
    for (j, &raw) in centroid_table[best_scale_idx as usize].iter().enumerate() {
        let c = half::f16::from_bits(raw).to_f32();
        let err = (actual_centroid - c).abs();
        if err < best_centroid_err {
            best_centroid_err = err;
            best_centroid_idx = j as u8;
        }
    }
    let chosen_centroid =
        half::f16::from_bits(centroid_table[best_scale_idx as usize][best_centroid_idx as usize])
            .to_f32();

    // ── Step 4: target_scaled = (block − centroid) / scale_baked ──
    // After this, target_scaled values are in i8 [-127, +127] space and can
    // be compared directly against raw curve_table i8 values with no
    // additional /127 normalisation.
    let inv_scale = 1.0 / chosen_scale_baked;
    let target: [f32; QK_Q0_V] = std::array::from_fn(|i| (block[i] - chosen_centroid) * inv_scale);

    // ── Step 5: brute-force scan all 128 curves. The peak-bin shortcut from
    //           the 256-curve era is dropped — with half the curves the
    //           saving is small (128 vs ~12 candidates) and removing the
    //           peak tables simplifies bring-up of the new bit layout. CUDA
    //           kernel keeps a hierarchical search since GPU per-block cost
    //           is dominated by uploads/launch, not curve count.
    let score_curve = |curve: &[i8; 32]| -> f32 {
        let mut err = 0.0f32;
        for (k, &t) in target.iter().enumerate() {
            let cv = curve[k] as f32;
            let d = t - cv;
            err += d * d;
        }
        err
    };

    let mut best_curve_idx = 0u8;
    let mut best_curve_err = f32::INFINITY;
    for (c, curve) in curve_table.iter().enumerate().take(128) {
        let err = score_curve(curve);
        if err < best_curve_err {
            best_curve_err = err;
            best_curve_idx = c as u8;
        }
    }

    let _ = best_scale_err;
    let _ = best_centroid_err;

    BlockQ0V::pack(best_curve_idx, best_scale_idx, best_centroid_idx)
}

// ==============================================================================
// Q0_V constant tables — must match q0_v_tables.cuh byte-for-byte.
// ==============================================================================
/// Q0_V codebook tables, generated from q0_v_tables.cuh.
/// K and V are calibrated separately; pick by IS_K at the call site.
#[allow(clippy::needless_range_loop)]
pub mod q0_v_tables {

    const __CURVE_TABLE_K_RAW: [[i8; 32]; 128] = [
        /* slot   0  bucket 0 (A) phase  0 */
        [
            127, 126, 122, 116, 108, 98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1,
            -2, -4, -8, -13, -20, -29, -39, -50, -62, -74,
        ],
        /* slot   1  bucket 0 (A) phase  1 */
        [
            122, 116, 108, 98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8,
            -13, -20, -29, -39, -50, -62, -74, 127, 126,
        ],
        /* slot   2  bucket 0 (A) phase  2 */
        [
            108, 98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20,
            -29, -39, -50, -62, -74, 127, 126, 122, 116,
        ],
        /* slot   3  bucket 0 (A) phase  3 */
        [
            86, 75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29,
            -39, -50, -62, -74, 127, 126, 122, 116, 108, 98,
        ],
        /* slot   4  bucket 0 (A) phase  4 */
        [
            63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50,
            -62, -74, 127, 126, 122, 116, 108, 98, 86, 75,
        ],
        /* slot   5  bucket 0 (A) phase  5 */
        [
            40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50, -62, -74,
            127, 126, 122, 116, 108, 98, 86, 75, 63, 51,
        ],
        /* slot   6  bucket 0 (A) phase  6 */
        [
            21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50, -62, -74, 127,
            126, 122, 116, 108, 98, 86, 75, 63, 51, 40, 30,
        ],
        /* slot   7  bucket 0 (A) phase  7 */
        [
            8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50, -62, -74, 127, 126, 122,
            116, 108, 98, 86, 75, 63, 51, 40, 30, 21, 14,
        ],
        /* slot   8  bucket 0 (A) phase  8 */
        [
            2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50, -62, -74, 127, 126, 122, 116,
            108, 98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4,
        ],
        /* slot   9  bucket 0 (A) phase  9 */
        [
            0, 0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50, -62, -74, 127, 126, 122, 116, 108,
            98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1,
        ],
        /* slot  10  bucket 0 (A) phase 10 */
        [
            0, -1, -2, -4, -8, -13, -20, -29, -39, -50, -62, -74, 127, 126, 122, 116, 108, 98, 86,
            75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0,
        ],
        /* slot  11  bucket 0 (A) phase 11 */
        [
            -2, -4, -8, -13, -20, -29, -39, -50, -62, -74, 127, 126, 122, 116, 108, 98, 86, 75, 63,
            51, 40, 30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1,
        ],
        /* slot  12  bucket 0 (A) phase 12 */
        [
            -8, -13, -20, -29, -39, -50, -62, -74, 127, 126, 122, 116, 108, 98, 86, 75, 63, 51, 40,
            30, 21, 14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4,
        ],
        /* slot  13  bucket 0 (A) phase 13 */
        [
            -20, -29, -39, -50, -62, -74, 127, 126, 122, 116, 108, 98, 86, 75, 63, 51, 40, 30, 21,
            14, 8, 4, 2, 1, 0, 0, 0, -1, -2, -4, -8, -13,
        ],
        /* slot  14  bucket 0 (A) phase 14 */
        [
            -39, -50, -62, -74, 127, 126, 122, 116, 108, 98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4,
            2, 1, 0, 0, 0, -1, -2, -4, -8, -13, -20, -29,
        ],
        /* slot  15  bucket 0 (A) phase 15 */
        [
            -62, -74, 127, 126, 122, 116, 108, 98, 86, 75, 63, 51, 40, 30, 21, 14, 8, 4, 2, 1, 0,
            0, 0, -1, -2, -4, -8, -13, -20, -29, -39, -50,
        ],
        /* slot  16  bucket 1 (B) phase  0 */
        [
            -127, -54, -57, -52, -53, -52, -50, -43, 8, 8, 9, 10, 11, 10, 10, 7, -3, -4, -3, -5,
            -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41,
        ],
        /* slot  17  bucket 1 (B) phase  1 */
        [
            -57, -52, -53, -52, -50, -43, 8, 8, 9, 10, 11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7,
            6, 49, 51, 52, 53, 53, 53, 55, 41, -127, -54,
        ],
        /* slot  18  bucket 1 (B) phase  2 */
        [
            -53, -52, -50, -43, 8, 8, 9, 10, 11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51,
            52, 53, 53, 53, 55, 41, -127, -54, -57, -52,
        ],
        /* slot  19  bucket 1 (B) phase  3 */
        [
            -50, -43, 8, 8, 9, 10, 11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53,
            53, 53, 55, 41, -127, -54, -57, -52, -53, -52,
        ],
        /* slot  20  bucket 1 (B) phase  4 */
        [
            8, 8, 9, 10, 11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55,
            41, -127, -54, -57, -52, -53, -52, -50, -43,
        ],
        /* slot  21  bucket 1 (B) phase  5 */
        [
            9, 10, 11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41,
            -127, -54, -57, -52, -53, -52, -50, -43, 8, 8,
        ],
        /* slot  22  bucket 1 (B) phase  6 */
        [
            11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41, -127,
            -54, -57, -52, -53, -52, -50, -43, 8, 8, 9, 10,
        ],
        /* slot  23  bucket 1 (B) phase  7 */
        [
            10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41, -127, -54, -57,
            -52, -53, -52, -50, -43, 8, 8, 9, 10, 11, 10,
        ],
        /* slot  24  bucket 1 (B) phase  8 */
        [
            -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41, -127, -54, -57, -52,
            -53, -52, -50, -43, 8, 8, 9, 10, 11, 10, 10, 7,
        ],
        /* slot  25  bucket 1 (B) phase  9 */
        [
            -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41, -127, -54, -57, -52, -53, -52,
            -50, -43, 8, 8, 9, 10, 11, 10, 10, 7, -3, -4,
        ],
        /* slot  26  bucket 1 (B) phase 10 */
        [
            -6, -7, -7, 6, 49, 51, 52, 53, 53, 53, 55, 41, -127, -54, -57, -52, -53, -52, -50, -43,
            8, 8, 9, 10, 11, 10, 10, 7, -3, -4, -3, -5,
        ],
        /* slot  27  bucket 1 (B) phase 11 */
        [
            -7, 6, 49, 51, 52, 53, 53, 53, 55, 41, -127, -54, -57, -52, -53, -52, -50, -43, 8, 8,
            9, 10, 11, 10, 10, 7, -3, -4, -3, -5, -6, -7,
        ],
        /* slot  28  bucket 1 (B) phase 12 */
        [
            49, 51, 52, 53, 53, 53, 55, 41, -127, -54, -57, -52, -53, -52, -50, -43, 8, 8, 9, 10,
            11, 10, 10, 7, -3, -4, -3, -5, -6, -7, -7, 6,
        ],
        /* slot  29  bucket 1 (B) phase 13 */
        [
            52, 53, 53, 53, 55, 41, -127, -54, -57, -52, -53, -52, -50, -43, 8, 8, 9, 10, 11, 10,
            10, 7, -3, -4, -3, -5, -6, -7, -7, 6, 49, 51,
        ],
        /* slot  30  bucket 1 (B) phase 14 */
        [
            53, 53, 55, 41, -127, -54, -57, -52, -53, -52, -50, -43, 8, 8, 9, 10, 11, 10, 10, 7,
            -3, -4, -3, -5, -6, -7, -7, 6, 49, 51, 52, 53,
        ],
        /* slot  31  bucket 1 (B) phase 15 */
        [
            55, 41, -127, -54, -57, -52, -53, -52, -50, -43, 8, 8, 9, 10, 11, 10, 10, 7, -3, -4,
            -3, -5, -6, -7, -7, 6, 49, 51, 52, 53, 53, 53,
        ],
        /* slot  32  bucket 2 (C) phase  0 */
        [
            127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22, -20, -18,
            -18, -19, -20, -22, -22, -23, -24, -25, -27, -19, 21, 43,
        ],
        /* slot  33  bucket 2 (C) phase  1 */
        [
            33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22, -20, -18, -18, -19,
            -20, -22, -22, -23, -24, -25, -27, -19, 21, 43, 127, 37,
        ],
        /* slot  34  bucket 2 (C) phase  2 */
        [
            45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22, -20, -18, -18, -19, -20,
            -22, -22, -23, -24, -25, -27, -19, 21, 43, 127, 37, 33, 40,
        ],
        /* slot  35  bucket 2 (C) phase  3 */
        [
            9, -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22, -20, -18, -18, -19, -20, -22, -22,
            -23, -24, -25, -27, -19, 21, 43, 127, 37, 33, 40, 45, 34,
        ],
        /* slot  36  bucket 2 (C) phase  4 */
        [
            -4, -6, -8, -8, -6, -9, -17, -22, -23, -22, -20, -18, -18, -19, -20, -22, -22, -23,
            -24, -25, -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9, -2,
        ],
        /* slot  37  bucket 2 (C) phase  5 */
        [
            -8, -8, -6, -9, -17, -22, -23, -22, -20, -18, -18, -19, -20, -22, -22, -23, -24, -25,
            -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6,
        ],
        /* slot  38  bucket 2 (C) phase  6 */
        [
            -6, -9, -17, -22, -23, -22, -20, -18, -18, -19, -20, -22, -22, -23, -24, -25, -27, -19,
            21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8,
        ],
        /* slot  39  bucket 2 (C) phase  7 */
        [
            -17, -22, -23, -22, -20, -18, -18, -19, -20, -22, -22, -23, -24, -25, -27, -19, 21, 43,
            127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9,
        ],
        /* slot  40  bucket 2 (C) phase  8 */
        [
            -23, -22, -20, -18, -18, -19, -20, -22, -22, -23, -24, -25, -27, -19, 21, 43, 127, 37,
            33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22,
        ],
        /* slot  41  bucket 2 (C) phase  9 */
        [
            -20, -18, -18, -19, -20, -22, -22, -23, -24, -25, -27, -19, 21, 43, 127, 37, 33, 40,
            45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22,
        ],
        /* slot  42  bucket 2 (C) phase 10 */
        [
            -18, -19, -20, -22, -22, -23, -24, -25, -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9,
            -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22, -20, -18,
        ],
        /* slot  43  bucket 2 (C) phase 11 */
        [
            -20, -22, -22, -23, -24, -25, -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6,
            -8, -8, -6, -9, -17, -22, -23, -22, -20, -18, -18, -19,
        ],
        /* slot  44  bucket 2 (C) phase 12 */
        [
            -22, -23, -24, -25, -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8,
            -6, -9, -17, -22, -23, -22, -20, -18, -18, -19, -20, -22,
        ],
        /* slot  45  bucket 2 (C) phase 13 */
        [
            -24, -25, -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9,
            -17, -22, -23, -22, -20, -18, -18, -19, -20, -22, -22, -23,
        ],
        /* slot  46  bucket 2 (C) phase 14 */
        [
            -27, -19, 21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22,
            -23, -22, -20, -18, -18, -19, -20, -22, -22, -23, -24, -25,
        ],
        /* slot  47  bucket 2 (C) phase 15 */
        [
            21, 43, 127, 37, 33, 40, 45, 34, 9, -2, -4, -6, -8, -8, -6, -9, -17, -22, -23, -22,
            -20, -18, -18, -19, -20, -22, -22, -23, -24, -25, -27, -19,
        ],
        /* slot  48  bucket 3 (D) phase  0 */
        [
            -127, -84, -82, -83, 32, 33, 32, 32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27,
            23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83, -83,
        ],
        /* slot  49  bucket 3 (D) phase  1 */
        [
            -82, -83, 32, 33, 32, 32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23,
            24, 24, 25, 24, 25, -79, -81, -83, -83, -127, -84,
        ],
        /* slot  50  bucket 3 (D) phase  2 */
        [
            32, 33, 32, 32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25,
            24, 25, -79, -81, -83, -83, -127, -84, -82, -83,
        ],
        /* slot  51  bucket 3 (D) phase  3 */
        [
            32, 32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25,
            -79, -81, -83, -83, -127, -84, -82, -83, 32, 33,
        ],
        /* slot  52  bucket 3 (D) phase  4 */
        [
            31, 31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79,
            -81, -83, -83, -127, -84, -82, -83, 32, 33, 32, 32,
        ],
        /* slot  53  bucket 3 (D) phase  5 */
        [
            31, 32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83,
            -83, -127, -84, -82, -83, 32, 33, 32, 32, 31, 31,
        ],
        /* slot  54  bucket 3 (D) phase  6 */
        [
            27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83, -83,
            -127, -84, -82, -83, 32, 33, 32, 32, 31, 31, 31, 32,
        ],
        /* slot  55  bucket 3 (D) phase  7 */
        [
            28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83, -83, -127, -84,
            -82, -83, 32, 33, 32, 32, 31, 31, 31, 32, 27, 28,
        ],
        /* slot  56  bucket 3 (D) phase  8 */
        [
            28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83, -83, -127, -84, -82,
            -83, 32, 33, 32, 32, 31, 31, 31, 32, 27, 28, 28, 28,
        ],
        /* slot  57  bucket 3 (D) phase  9 */
        [
            28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83, -83, -127, -84, -82, -83, 32,
            33, 32, 32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28,
        ],
        /* slot  58  bucket 3 (D) phase 10 */
        [
            23, 23, 23, 24, 24, 25, 24, 25, -79, -81, -83, -83, -127, -84, -82, -83, 32, 33, 32,
            32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27,
        ],
        /* slot  59  bucket 3 (D) phase 11 */
        [
            23, 24, 24, 25, 24, 25, -79, -81, -83, -83, -127, -84, -82, -83, 32, 33, 32, 32, 31,
            31, 31, 32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23,
        ],
        /* slot  60  bucket 3 (D) phase 12 */
        [
            24, 25, 24, 25, -79, -81, -83, -83, -127, -84, -82, -83, 32, 33, 32, 32, 31, 31, 31,
            32, 27, 28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24,
        ],
        /* slot  61  bucket 3 (D) phase 13 */
        [
            24, 25, -79, -81, -83, -83, -127, -84, -82, -83, 32, 33, 32, 32, 31, 31, 31, 32, 27,
            28, 28, 28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25,
        ],
        /* slot  62  bucket 3 (D) phase 14 */
        [
            -79, -81, -83, -83, -127, -84, -82, -83, 32, 33, 32, 32, 31, 31, 31, 32, 27, 28, 28,
            28, 28, 28, 28, 27, 23, 23, 23, 24, 24, 25, 24, 25,
        ],
        /* slot  63  bucket 3 (D) phase 15 */
        [
            -83, -83, -127, -84, -82, -83, 32, 33, 32, 32, 31, 31, 31, 32, 27, 28, 28, 28, 28, 28,
            28, 27, 23, 23, 23, 24, 24, 25, 24, 25, -79, -81,
        ],
        /* slot  64  bucket 4 (A_neg) phase  0 */
        [
            -127, -126, -122, -116, -108, -98, -86, -75, -63, -51, -40, -30, -21, -14, -8, -4, -2,
            -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74,
        ],
        /* slot  65  bucket 4 (A_neg) phase  1 */
        [
            -122, -116, -108, -98, -86, -75, -63, -51, -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0,
            1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126,
        ],
        /* slot  66  bucket 4 (A_neg) phase  2 */
        [
            -108, -98, -86, -75, -63, -51, -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8,
            13, 20, 29, 39, 50, 62, 74, -127, -126, -122, -116,
        ],
        /* slot  67  bucket 4 (A_neg) phase  3 */
        [
            -86, -75, -63, -51, -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20,
            29, 39, 50, 62, 74, -127, -126, -122, -116, -108, -98,
        ],
        /* slot  68  bucket 4 (A_neg) phase  4 */
        [
            -63, -51, -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50,
            62, 74, -127, -126, -122, -116, -108, -98, -86, -75,
        ],
        /* slot  69  bucket 4 (A_neg) phase  5 */
        [
            -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74,
            -127, -126, -122, -116, -108, -98, -86, -75, -63, -51,
        ],
        /* slot  70  bucket 4 (A_neg) phase  6 */
        [
            -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126,
            -122, -116, -108, -98, -86, -75, -63, -51, -40, -30,
        ],
        /* slot  71  bucket 4 (A_neg) phase  7 */
        [
            -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126, -122,
            -116, -108, -98, -86, -75, -63, -51, -40, -30, -21, -14,
        ],
        /* slot  72  bucket 4 (A_neg) phase  8 */
        [
            -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126, -122, -116, -108,
            -98, -86, -75, -63, -51, -40, -30, -21, -14, -8, -4,
        ],
        /* slot  73  bucket 4 (A_neg) phase  9 */
        [
            0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126, -122, -116, -108, -98,
            -86, -75, -63, -51, -40, -30, -21, -14, -8, -4, -2, -1,
        ],
        /* slot  74  bucket 4 (A_neg) phase 10 */
        [
            0, 1, 2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126, -122, -116, -108, -98, -86, -75,
            -63, -51, -40, -30, -21, -14, -8, -4, -2, -1, 0, 0,
        ],
        /* slot  75  bucket 4 (A_neg) phase 11 */
        [
            2, 4, 8, 13, 20, 29, 39, 50, 62, 74, -127, -126, -122, -116, -108, -98, -86, -75, -63,
            -51, -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1,
        ],
        /* slot  76  bucket 4 (A_neg) phase 12 */
        [
            8, 13, 20, 29, 39, 50, 62, 74, -127, -126, -122, -116, -108, -98, -86, -75, -63, -51,
            -40, -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4,
        ],
        /* slot  77  bucket 4 (A_neg) phase 13 */
        [
            20, 29, 39, 50, 62, 74, -127, -126, -122, -116, -108, -98, -86, -75, -63, -51, -40,
            -30, -21, -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13,
        ],
        /* slot  78  bucket 4 (A_neg) phase 14 */
        [
            39, 50, 62, 74, -127, -126, -122, -116, -108, -98, -86, -75, -63, -51, -40, -30, -21,
            -14, -8, -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29,
        ],
        /* slot  79  bucket 4 (A_neg) phase 15 */
        [
            62, 74, -127, -126, -122, -116, -108, -98, -86, -75, -63, -51, -40, -30, -21, -14, -8,
            -4, -2, -1, 0, 0, 0, 1, 2, 4, 8, 13, 20, 29, 39, 50,
        ],
        /* slot  80  bucket 5 (B_neg) phase  0 */
        [
            127, 54, 57, 52, 53, 52, 50, 43, -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7,
            7, -6, -49, -51, -52, -53, -53, -53, -55, -41,
        ],
        /* slot  81  bucket 5 (B_neg) phase  1 */
        [
            57, 52, 53, 52, 50, 43, -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6,
            -49, -51, -52, -53, -53, -53, -55, -41, 127, 54,
        ],
        /* slot  82  bucket 5 (B_neg) phase  2 */
        [
            53, 52, 50, 43, -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51,
            -52, -53, -53, -53, -55, -41, 127, 54, 57, 52,
        ],
        /* slot  83  bucket 5 (B_neg) phase  3 */
        [
            50, 43, -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52,
            -53, -53, -53, -55, -41, 127, 54, 57, 52, 53, 52,
        ],
        /* slot  84  bucket 5 (B_neg) phase  4 */
        [
            -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52, -53, -53,
            -53, -55, -41, 127, 54, 57, 52, 53, 52, 50, 43,
        ],
        /* slot  85  bucket 5 (B_neg) phase  5 */
        [
            -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52, -53, -53, -53, -55,
            -41, 127, 54, 57, 52, 53, 52, 50, 43, -8, -8,
        ],
        /* slot  86  bucket 5 (B_neg) phase  6 */
        [
            -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52, -53, -53, -53, -55, -41,
            127, 54, 57, 52, 53, 52, 50, 43, -8, -8, -9, -10,
        ],
        /* slot  87  bucket 5 (B_neg) phase  7 */
        [
            -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52, -53, -53, -53, -55, -41, 127, 54, 57,
            52, 53, 52, 50, 43, -8, -8, -9, -10, -11, -10,
        ],
        /* slot  88  bucket 5 (B_neg) phase  8 */
        [
            3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52, -53, -53, -53, -55, -41, 127, 54, 57, 52, 53,
            52, 50, 43, -8, -8, -9, -10, -11, -10, -10, -7,
        ],
        /* slot  89  bucket 5 (B_neg) phase  9 */
        [
            3, 5, 6, 7, 7, -6, -49, -51, -52, -53, -53, -53, -55, -41, 127, 54, 57, 52, 53, 52, 50,
            43, -8, -8, -9, -10, -11, -10, -10, -7, 3, 4,
        ],
        /* slot  90  bucket 5 (B_neg) phase 10 */
        [
            6, 7, 7, -6, -49, -51, -52, -53, -53, -53, -55, -41, 127, 54, 57, 52, 53, 52, 50, 43,
            -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3, 5,
        ],
        /* slot  91  bucket 5 (B_neg) phase 11 */
        [
            7, -6, -49, -51, -52, -53, -53, -53, -55, -41, 127, 54, 57, 52, 53, 52, 50, 43, -8, -8,
            -9, -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7,
        ],
        /* slot  92  bucket 5 (B_neg) phase 12 */
        [
            -49, -51, -52, -53, -53, -53, -55, -41, 127, 54, 57, 52, 53, 52, 50, 43, -8, -8, -9,
            -10, -11, -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6,
        ],
        /* slot  93  bucket 5 (B_neg) phase 13 */
        [
            -52, -53, -53, -53, -55, -41, 127, 54, 57, 52, 53, 52, 50, 43, -8, -8, -9, -10, -11,
            -10, -10, -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51,
        ],
        /* slot  94  bucket 5 (B_neg) phase 14 */
        [
            -53, -53, -55, -41, 127, 54, 57, 52, 53, 52, 50, 43, -8, -8, -9, -10, -11, -10, -10,
            -7, 3, 4, 3, 5, 6, 7, 7, -6, -49, -51, -52, -53,
        ],
        /* slot  95  bucket 5 (B_neg) phase 15 */
        [
            -55, -41, 127, 54, 57, 52, 53, 52, 50, 43, -8, -8, -9, -10, -11, -10, -10, -7, 3, 4, 3,
            5, 6, 7, 7, -6, -49, -51, -52, -53, -53, -53,
        ],
        /* slot  96  bucket 6 (C_neg) phase  0 */
        [
            -127, -37, -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19,
            20, 22, 22, 23, 24, 25, 27, 19, -21, -43,
        ],
        /* slot  97  bucket 6 (C_neg) phase  1 */
        [
            -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22,
            22, 23, 24, 25, 27, 19, -21, -43, -127, -37,
        ],
        /* slot  98  bucket 6 (C_neg) phase  2 */
        [
            -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24,
            25, 27, 19, -21, -43, -127, -37, -33, -40,
        ],
        /* slot  99  bucket 6 (C_neg) phase  3 */
        [
            -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27,
            19, -21, -43, -127, -37, -33, -40, -45, -34,
        ],
        /* slot 100  bucket 6 (C_neg) phase  4 */
        [
            4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21,
            -43, -127, -37, -33, -40, -45, -34, -9, 2,
        ],
        /* slot 101  bucket 6 (C_neg) phase  5 */
        [
            8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21, -43,
            -127, -37, -33, -40, -45, -34, -9, 2, 4, 6,
        ],
        /* slot 102  bucket 6 (C_neg) phase  6 */
        [
            6, 9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21, -43, -127,
            -37, -33, -40, -45, -34, -9, 2, 4, 6, 8, 8,
        ],
        /* slot 103  bucket 6 (C_neg) phase  7 */
        [
            17, 22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21, -43, -127, -37,
            -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6, 9,
        ],
        /* slot 104  bucket 6 (C_neg) phase  8 */
        [
            23, 22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21, -43, -127, -37, -33, -40,
            -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22,
        ],
        /* slot 105  bucket 6 (C_neg) phase  9 */
        [
            20, 18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21, -43, -127, -37, -33, -40, -45,
            -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23, 22,
        ],
        /* slot 106  bucket 6 (C_neg) phase 10 */
        [
            18, 19, 20, 22, 22, 23, 24, 25, 27, 19, -21, -43, -127, -37, -33, -40, -45, -34, -9, 2,
            4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20, 18,
        ],
        /* slot 107  bucket 6 (C_neg) phase 11 */
        [
            20, 22, 22, 23, 24, 25, 27, 19, -21, -43, -127, -37, -33, -40, -45, -34, -9, 2, 4, 6,
            8, 8, 6, 9, 17, 22, 23, 22, 20, 18, 18, 19,
        ],
        /* slot 108  bucket 6 (C_neg) phase 12 */
        [
            22, 23, 24, 25, 27, 19, -21, -43, -127, -37, -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6,
            9, 17, 22, 23, 22, 20, 18, 18, 19, 20, 22,
        ],
        /* slot 109  bucket 6 (C_neg) phase 13 */
        [
            24, 25, 27, 19, -21, -43, -127, -37, -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17,
            22, 23, 22, 20, 18, 18, 19, 20, 22, 22, 23,
        ],
        /* slot 110  bucket 6 (C_neg) phase 14 */
        [
            27, 19, -21, -43, -127, -37, -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23,
            22, 20, 18, 18, 19, 20, 22, 22, 23, 24, 25,
        ],
        /* slot 111  bucket 6 (C_neg) phase 15 */
        [
            -21, -43, -127, -37, -33, -40, -45, -34, -9, 2, 4, 6, 8, 8, 6, 9, 17, 22, 23, 22, 20,
            18, 18, 19, 20, 22, 22, 23, 24, 25, 27, 19,
        ],
        /* slot 112  bucket 7 (D_neg) phase  0 */
        [
            127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27, -28, -28, -28, -28, -28,
            -28, -27, -23, -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83,
        ],
        /* slot 113  bucket 7 (D_neg) phase  1 */
        [
            82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27, -28, -28, -28, -28, -28, -28, -27,
            -23, -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84,
        ],
        /* slot 114  bucket 7 (D_neg) phase  2 */
        [
            -32, -33, -32, -32, -31, -31, -31, -32, -27, -28, -28, -28, -28, -28, -28, -27, -23,
            -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83,
        ],
        /* slot 115  bucket 7 (D_neg) phase  3 */
        [
            -32, -32, -31, -31, -31, -32, -27, -28, -28, -28, -28, -28, -28, -27, -23, -23, -23,
            -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33,
        ],
        /* slot 116  bucket 7 (D_neg) phase  4 */
        [
            -31, -31, -31, -32, -27, -28, -28, -28, -28, -28, -28, -27, -23, -23, -23, -24, -24,
            -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32, -32,
        ],
        /* slot 117  bucket 7 (D_neg) phase  5 */
        [
            -31, -32, -27, -28, -28, -28, -28, -28, -28, -27, -23, -23, -23, -24, -24, -25, -24,
            -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31, -31,
        ],
        /* slot 118  bucket 7 (D_neg) phase  6 */
        [
            -27, -28, -28, -28, -28, -28, -28, -27, -23, -23, -23, -24, -24, -25, -24, -25, 79, 81,
            83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31, -32,
        ],
        /* slot 119  bucket 7 (D_neg) phase  7 */
        [
            -28, -28, -28, -28, -28, -27, -23, -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83,
            127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27, -28,
        ],
        /* slot 120  bucket 7 (D_neg) phase  8 */
        [
            -28, -28, -28, -27, -23, -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84,
            82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27, -28, -28, -28,
        ],
        /* slot 121  bucket 7 (D_neg) phase  9 */
        [
            -28, -27, -23, -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32,
            -33, -32, -32, -31, -31, -31, -32, -27, -28, -28, -28, -28, -28,
        ],
        /* slot 122  bucket 7 (D_neg) phase 10 */
        [
            -23, -23, -23, -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32,
            -32, -31, -31, -31, -32, -27, -28, -28, -28, -28, -28, -28, -27,
        ],
        /* slot 123  bucket 7 (D_neg) phase 11 */
        [
            -23, -24, -24, -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31,
            -31, -31, -32, -27, -28, -28, -28, -28, -28, -28, -27, -23, -23,
        ],
        /* slot 124  bucket 7 (D_neg) phase 12 */
        [
            -24, -25, -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31,
            -32, -27, -28, -28, -28, -28, -28, -28, -27, -23, -23, -23, -24,
        ],
        /* slot 125  bucket 7 (D_neg) phase 13 */
        [
            -24, -25, 79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27,
            -28, -28, -28, -28, -28, -28, -27, -23, -23, -23, -24, -24, -25,
        ],
        /* slot 126  bucket 7 (D_neg) phase 14 */
        [
            79, 81, 83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27, -28, -28,
            -28, -28, -28, -28, -27, -23, -23, -23, -24, -24, -25, -24, -25,
        ],
        /* slot 127  bucket 7 (D_neg) phase 15 */
        [
            83, 83, 127, 84, 82, 83, -32, -33, -32, -32, -31, -31, -31, -32, -27, -28, -28, -28,
            -28, -28, -28, -27, -23, -23, -23, -24, -24, -25, -24, -25, 79, 81,
        ],
    ];

    const __CURVE_TABLE_V_RAW: [[i8; 32]; 128] = [
        /* slot   0  bucket 0 (A) phase  0 */
        [
            127, 28, 15, 5, 1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,
            -9, -10, -9, -7, -6, -3, 0, 4, 9, 15,
        ],
        /* slot   1  bucket 0 (A) phase  1 */
        [
            15, 5, 1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10,
            -9, -7, -6, -3, 0, 4, 9, 15, 127, 28,
        ],
        /* slot   2  bucket 0 (A) phase  2 */
        [
            1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -9, -7,
            -6, -3, 0, 4, 9, 15, 127, 28, 15, 5,
        ],
        /* slot   3  bucket 0 (A) phase  3 */
        [
            -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -9, -7, -6,
            -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2,
        ],
        /* slot   4  bucket 0 (A) phase  4 */
        [
            -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -9, -7, -6, -3, 0, 4,
            9, 15, 127, 28, 15, 5, 1, -2, -5, -7,
        ],
        /* slot   5  bucket 0 (A) phase  5 */
        [
            -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -9, -7, -6, -3, 0, 4, 9, 15,
            127, 28, 15, 5, 1, -2, -5, -7, -9, -9,
        ],
        /* slot   6  bucket 0 (A) phase  6 */
        [
            -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -9, -7, -6, -3, 0, 4, 9, 15, 127, 28,
            15, 5, 1, -2, -5, -7, -9, -9, -8, -9,
        ],
        /* slot   7  bucket 0 (A) phase  7 */
        [
            -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -9, -7, -6, -3, 0, 4, 9, 15, 127, 28, 15, 5,
            1, -2, -5, -7, -9, -9, -8, -9, -9, -9,
        ],
        /* slot   8  bucket 0 (A) phase  8 */
        [
            -9, -9, -9, -9, -9, -9, -9, -10, -9, -7, -6, -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2,
            -5, -7, -9, -9, -8, -9, -9, -9, -9, -9,
        ],
        /* slot   9  bucket 0 (A) phase  9 */
        [
            -9, -9, -9, -9, -9, -10, -9, -7, -6, -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2, -5, -7,
            -9, -9, -8, -9, -9, -9, -9, -9, -9, -9,
        ],
        /* slot  10  bucket 0 (A) phase 10 */
        [
            -9, -9, -9, -10, -9, -7, -6, -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2, -5, -7, -9, -9,
            -8, -9, -9, -9, -9, -9, -9, -9, -9, -9,
        ],
        /* slot  11  bucket 0 (A) phase 11 */
        [
            -9, -10, -9, -7, -6, -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2, -5, -7, -9, -9, -8, -9,
            -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,
        ],
        /* slot  12  bucket 0 (A) phase 12 */
        [
            -9, -7, -6, -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9,
            -9, -9, -9, -9, -9, -9, -9, -9, -10,
        ],
        /* slot  13  bucket 0 (A) phase 13 */
        [
            -6, -3, 0, 4, 9, 15, 127, 28, 15, 5, 1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9,
            -9, -9, -9, -9, -9, -9, -10, -9, -7,
        ],
        /* slot  14  bucket 0 (A) phase 14 */
        [
            0, 4, 9, 15, 127, 28, 15, 5, 1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9,
            -9, -9, -9, -9, -10, -9, -7, -6, -3,
        ],
        /* slot  15  bucket 0 (A) phase 15 */
        [
            9, 15, 127, 28, 15, 5, 1, -2, -5, -7, -9, -9, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9,
            -9, -9, -9, -10, -9, -7, -6, -3, 0, 4,
        ],
        /* slot  16  bucket 1 (B) phase  0 */
        [
            -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9,
            10, 11, 11, 9, 4,
        ],
        /* slot  17  bucket 1 (B) phase  1 */
        [
            -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11,
            9, 4, -127, -13,
        ],
        /* slot  18  bucket 1 (B) phase  2 */
        [
            8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4,
            -127, -13, -6, 6,
        ],
        /* slot  19  bucket 1 (B) phase  3 */
        [
            8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127,
            -13, -6, 6, 8, 8,
        ],
        /* slot  20  bucket 1 (B) phase  4 */
        [
            5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13,
            -6, 6, 8, 8, 8, 6,
        ],
        /* slot  21  bucket 1 (B) phase  5 */
        [
            3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6,
            8, 8, 8, 6, 5, 3,
        ],
        /* slot  22  bucket 1 (B) phase  6 */
        [
            2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8,
            8, 6, 5, 3, 3, 2,
        ],
        /* slot  23  bucket 1 (B) phase  7 */
        [
            1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6,
            5, 3, 3, 2, 2, 1,
        ],
        /* slot  24  bucket 1 (B) phase  8 */
        [
            0, 2, 1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3,
            3, 2, 2, 1, 1, 0,
        ],
        /* slot  25  bucket 1 (B) phase  9 */
        [
            1, 1, 2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2,
            2, 1, 1, 0, 0, 2,
        ],
        /* slot  26  bucket 1 (B) phase 10 */
        [
            2, 3, 4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1,
            1, 0, 0, 2, 1, 1,
        ],
        /* slot  27  bucket 1 (B) phase 11 */
        [
            4, 6, 8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0,
            0, 2, 1, 1, 2, 3,
        ],
        /* slot  28  bucket 1 (B) phase 12 */
        [
            8, 7, 9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2,
            1, 1, 2, 3, 4, 6,
        ],
        /* slot  29  bucket 1 (B) phase 13 */
        [
            9, 10, 11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1,
            2, 3, 4, 6, 8, 7,
        ],
        /* slot  30  bucket 1 (B) phase 14 */
        [
            11, 11, 9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3,
            4, 6, 8, 7, 9, 10,
        ],
        /* slot  31  bucket 1 (B) phase 15 */
        [
            9, 4, -127, -13, -6, 6, 8, 8, 8, 6, 5, 3, 3, 2, 2, 1, 1, 0, 0, 2, 1, 1, 2, 3, 4, 6, 8,
            7, 9, 10, 11, 11,
        ],
        /* slot  32  bucket 2 (C) phase  0 */
        [
            127, -42, -3, -10, -10, -5, -3, -2, 1, -2, 0, 1, -1, -1, -1, -2, -1, -2, -1, 0, 1, 1,
            4, 4, 4, 1, -1, -3, -8, -9, -12, -9,
        ],
        /* slot  33  bucket 2 (C) phase  1 */
        [
            -3, -10, -10, -5, -3, -2, 1, -2, 0, 1, -1, -1, -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1,
            -1, -3, -8, -9, -12, -9, 127, -42,
        ],
        /* slot  34  bucket 2 (C) phase  2 */
        [
            -10, -5, -3, -2, 1, -2, 0, 1, -1, -1, -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3,
            -8, -9, -12, -9, 127, -42, -3, -10,
        ],
        /* slot  35  bucket 2 (C) phase  3 */
        [
            -3, -2, 1, -2, 0, 1, -1, -1, -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9,
            -12, -9, 127, -42, -3, -10, -10, -5,
        ],
        /* slot  36  bucket 2 (C) phase  4 */
        [
            1, -2, 0, 1, -1, -1, -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9,
            127, -42, -3, -10, -10, -5, -3, -2,
        ],
        /* slot  37  bucket 2 (C) phase  5 */
        [
            0, 1, -1, -1, -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127,
            -42, -3, -10, -10, -5, -3, -2, 1, -2,
        ],
        /* slot  38  bucket 2 (C) phase  6 */
        [
            -1, -1, -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3,
            -10, -10, -5, -3, -2, 1, -2, 0, 1,
        ],
        /* slot  39  bucket 2 (C) phase  7 */
        [
            -1, -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3, -10,
            -10, -5, -3, -2, 1, -2, 0, 1, -1, -1,
        ],
        /* slot  40  bucket 2 (C) phase  8 */
        [
            -1, -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3, -10, -10, -5,
            -3, -2, 1, -2, 0, 1, -1, -1, -1, -2,
        ],
        /* slot  41  bucket 2 (C) phase  9 */
        [
            -1, 0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3, -10, -10, -5, -3, -2,
            1, -2, 0, 1, -1, -1, -1, -2, -1, -2,
        ],
        /* slot  42  bucket 2 (C) phase 10 */
        [
            1, 1, 4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3, -10, -10, -5, -3, -2, 1, -2,
            0, 1, -1, -1, -1, -2, -1, -2, -1, 0,
        ],
        /* slot  43  bucket 2 (C) phase 11 */
        [
            4, 4, 4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3, -10, -10, -5, -3, -2, 1, -2, 0, 1,
            -1, -1, -1, -2, -1, -2, -1, 0, 1, 1,
        ],
        /* slot  44  bucket 2 (C) phase 12 */
        [
            4, 1, -1, -3, -8, -9, -12, -9, 127, -42, -3, -10, -10, -5, -3, -2, 1, -2, 0, 1, -1, -1,
            -1, -2, -1, -2, -1, 0, 1, 1, 4, 4,
        ],
        /* slot  45  bucket 2 (C) phase 13 */
        [
            -1, -3, -8, -9, -12, -9, 127, -42, -3, -10, -10, -5, -3, -2, 1, -2, 0, 1, -1, -1, -1,
            -2, -1, -2, -1, 0, 1, 1, 4, 4, 4, 1,
        ],
        /* slot  46  bucket 2 (C) phase 14 */
        [
            -8, -9, -12, -9, 127, -42, -3, -10, -10, -5, -3, -2, 1, -2, 0, 1, -1, -1, -1, -2, -1,
            -2, -1, 0, 1, 1, 4, 4, 4, 1, -1, -3,
        ],
        /* slot  47  bucket 2 (C) phase 15 */
        [
            -12, -9, 127, -42, -3, -10, -10, -5, -3, -2, 1, -2, 0, 1, -1, -1, -1, -2, -1, -2, -1,
            0, 1, 1, 4, 4, 4, 1, -1, -3, -8, -9,
        ],
        /* slot  48  bucket 3 (D) phase  0 */
        [
            -127, -36, -13, -2, 7, 11, 16, 19, 23, 22, 21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10,
            6, 4, 0, -4, -7, -11, -15, -19, -24, -28, -31,
        ],
        /* slot  49  bucket 3 (D) phase  1 */
        [
            -13, -2, 7, 11, 16, 19, 23, 22, 21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0,
            -4, -7, -11, -15, -19, -24, -28, -31, -127, -36,
        ],
        /* slot  50  bucket 3 (D) phase  2 */
        [
            7, 11, 16, 19, 23, 22, 21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7,
            -11, -15, -19, -24, -28, -31, -127, -36, -13, -2,
        ],
        /* slot  51  bucket 3 (D) phase  3 */
        [
            16, 19, 23, 22, 21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15,
            -19, -24, -28, -31, -127, -36, -13, -2, 7, 11,
        ],
        /* slot  52  bucket 3 (D) phase  4 */
        [
            23, 22, 21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19,
            -24, -28, -31, -127, -36, -13, -2, 7, 11, 16, 19,
        ],
        /* slot  53  bucket 3 (D) phase  5 */
        [
            21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19, -24, -28,
            -31, -127, -36, -13, -2, 7, 11, 16, 19, 23, 22,
        ],
        /* slot  54  bucket 3 (D) phase  6 */
        [
            20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19, -24, -28, -31,
            -127, -36, -13, -2, 7, 11, 16, 19, 23, 22, 21, 22,
        ],
        /* slot  55  bucket 3 (D) phase  7 */
        [
            18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19, -24, -28, -31, -127, -36,
            -13, -2, 7, 11, 16, 19, 23, 22, 21, 22, 20, 20,
        ],
        /* slot  56  bucket 3 (D) phase  8 */
        [
            19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19, -24, -28, -31, -127, -36, -13, -2,
            7, 11, 16, 19, 23, 22, 21, 22, 20, 20, 18, 18,
        ],
        /* slot  57  bucket 3 (D) phase  9 */
        [
            14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19, -24, -28, -31, -127, -36, -13, -2, 7, 11,
            16, 19, 23, 22, 21, 22, 20, 20, 18, 18, 19, 16,
        ],
        /* slot  58  bucket 3 (D) phase 10 */
        [
            10, 6, 4, 0, -4, -7, -11, -15, -19, -24, -28, -31, -127, -36, -13, -2, 7, 11, 16, 19,
            23, 22, 21, 22, 20, 20, 18, 18, 19, 16, 14, 11,
        ],
        /* slot  59  bucket 3 (D) phase 11 */
        [
            4, 0, -4, -7, -11, -15, -19, -24, -28, -31, -127, -36, -13, -2, 7, 11, 16, 19, 23, 22,
            21, 22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6,
        ],
        /* slot  60  bucket 3 (D) phase 12 */
        [
            -4, -7, -11, -15, -19, -24, -28, -31, -127, -36, -13, -2, 7, 11, 16, 19, 23, 22, 21,
            22, 20, 20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0,
        ],
        /* slot  61  bucket 3 (D) phase 13 */
        [
            -11, -15, -19, -24, -28, -31, -127, -36, -13, -2, 7, 11, 16, 19, 23, 22, 21, 22, 20,
            20, 18, 18, 19, 16, 14, 11, 10, 6, 4, 0, -4, -7,
        ],
        /* slot  62  bucket 3 (D) phase 14 */
        [
            -19, -24, -28, -31, -127, -36, -13, -2, 7, 11, 16, 19, 23, 22, 21, 22, 20, 20, 18, 18,
            19, 16, 14, 11, 10, 6, 4, 0, -4, -7, -11, -15,
        ],
        /* slot  63  bucket 3 (D) phase 15 */
        [
            -28, -31, -127, -36, -13, -2, 7, 11, 16, 19, 23, 22, 21, 22, 20, 20, 18, 18, 19, 16,
            14, 11, 10, 6, 4, 0, -4, -7, -11, -15, -19, -24,
        ],
        /* slot  64  bucket 4 (A_neg) phase  0 */
        [
            -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7,
            6, 3, 0, -4, -9, -15,
        ],
        /* slot  65  bucket 4 (A_neg) phase  1 */
        [
            -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0,
            -4, -9, -15, -127, -28,
        ],
        /* slot  66  bucket 4 (A_neg) phase  2 */
        [
            -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9,
            -15, -127, -28, -15, -5,
        ],
        /* slot  67  bucket 4 (A_neg) phase  3 */
        [
            5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15,
            -127, -28, -15, -5, -1, 2,
        ],
        /* slot  68  bucket 4 (A_neg) phase  4 */
        [
            9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28,
            -15, -5, -1, 2, 5, 7,
        ],
        /* slot  69  bucket 4 (A_neg) phase  5 */
        [
            8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15,
            -5, -1, 2, 5, 7, 9, 9,
        ],
        /* slot  70  bucket 4 (A_neg) phase  6 */
        [
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5,
            -1, 2, 5, 7, 9, 9, 8, 9,
        ],
        /* slot  71  bucket 4 (A_neg) phase  7 */
        [
            9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2,
            5, 7, 9, 9, 8, 9, 9, 9,
        ],
        /* slot  72  bucket 4 (A_neg) phase  8 */
        [
            9, 9, 9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7,
            9, 9, 8, 9, 9, 9, 9, 9,
        ],
        /* slot  73  bucket 4 (A_neg) phase  9 */
        [
            9, 9, 9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9,
            8, 9, 9, 9, 9, 9, 9, 9,
        ],
        /* slot  74  bucket 4 (A_neg) phase 10 */
        [
            9, 9, 9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
        ],
        /* slot  75  bucket 4 (A_neg) phase 11 */
        [
            9, 10, 9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
        ],
        /* slot  76  bucket 4 (A_neg) phase 12 */
        [
            9, 7, 6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 10,
        ],
        /* slot  77  bucket 4 (A_neg) phase 13 */
        [
            6, 3, 0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 10, 9, 7,
        ],
        /* slot  78  bucket 4 (A_neg) phase 14 */
        [
            0, -4, -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 10, 9, 7, 6, 3,
        ],
        /* slot  79  bucket 4 (A_neg) phase 15 */
        [
            -9, -15, -127, -28, -15, -5, -1, 2, 5, 7, 9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
            10, 9, 7, 6, 3, 0, -4,
        ],
        /* slot  80  bucket 5 (B_neg) phase  0 */
        [
            127, 13, 6, -6, -8, -8, -8, -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3,
            -4, -6, -8, -7, -9, -10, -11, -11, -9, -4,
        ],
        /* slot  81  bucket 5 (B_neg) phase  1 */
        [
            6, -6, -8, -8, -8, -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6,
            -8, -7, -9, -10, -11, -11, -9, -4, 127, 13,
        ],
        /* slot  82  bucket 5 (B_neg) phase  2 */
        [
            -8, -8, -8, -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7,
            -9, -10, -11, -11, -9, -4, 127, 13, 6, -6,
        ],
        /* slot  83  bucket 5 (B_neg) phase  3 */
        [
            -8, -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7, -9, -10,
            -11, -11, -9, -4, 127, 13, 6, -6, -8, -8,
        ],
        /* slot  84  bucket 5 (B_neg) phase  4 */
        [
            -5, -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7, -9, -10, -11,
            -11, -9, -4, 127, 13, 6, -6, -8, -8, -8, -6,
        ],
        /* slot  85  bucket 5 (B_neg) phase  5 */
        [
            -3, -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7, -9, -10, -11, -11, -9,
            -4, 127, 13, 6, -6, -8, -8, -8, -6, -5, -3,
        ],
        /* slot  86  bucket 5 (B_neg) phase  6 */
        [
            -2, -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7, -9, -10, -11, -11, -9, -4, 127,
            13, 6, -6, -8, -8, -8, -6, -5, -3, -3, -2,
        ],
        /* slot  87  bucket 5 (B_neg) phase  7 */
        [
            -1, 0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7, -9, -10, -11, -11, -9, -4, 127, 13, 6,
            -6, -8, -8, -8, -6, -5, -3, -3, -2, -2, -1,
        ],
        /* slot  88  bucket 5 (B_neg) phase  8 */
        [
            0, -2, -1, -1, -2, -3, -4, -6, -8, -7, -9, -10, -11, -11, -9, -4, 127, 13, 6, -6, -8,
            -8, -8, -6, -5, -3, -3, -2, -2, -1, -1, 0,
        ],
        /* slot  89  bucket 5 (B_neg) phase  9 */
        [
            -1, -1, -2, -3, -4, -6, -8, -7, -9, -10, -11, -11, -9, -4, 127, 13, 6, -6, -8, -8, -8,
            -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2,
        ],
        /* slot  90  bucket 5 (B_neg) phase 10 */
        [
            -2, -3, -4, -6, -8, -7, -9, -10, -11, -11, -9, -4, 127, 13, 6, -6, -8, -8, -8, -6, -5,
            -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1,
        ],
        /* slot  91  bucket 5 (B_neg) phase 11 */
        [
            -4, -6, -8, -7, -9, -10, -11, -11, -9, -4, 127, 13, 6, -6, -8, -8, -8, -6, -5, -3, -3,
            -2, -2, -1, -1, 0, 0, -2, -1, -1, -2, -3,
        ],
        /* slot  92  bucket 5 (B_neg) phase 12 */
        [
            -8, -7, -9, -10, -11, -11, -9, -4, 127, 13, 6, -6, -8, -8, -8, -6, -5, -3, -3, -2, -2,
            -1, -1, 0, 0, -2, -1, -1, -2, -3, -4, -6,
        ],
        /* slot  93  bucket 5 (B_neg) phase 13 */
        [
            -9, -10, -11, -11, -9, -4, 127, 13, 6, -6, -8, -8, -8, -6, -5, -3, -3, -2, -2, -1, -1,
            0, 0, -2, -1, -1, -2, -3, -4, -6, -8, -7,
        ],
        /* slot  94  bucket 5 (B_neg) phase 14 */
        [
            -11, -11, -9, -4, 127, 13, 6, -6, -8, -8, -8, -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2,
            -1, -1, -2, -3, -4, -6, -8, -7, -9, -10,
        ],
        /* slot  95  bucket 5 (B_neg) phase 15 */
        [
            -9, -4, 127, 13, 6, -6, -8, -8, -8, -6, -5, -3, -3, -2, -2, -1, -1, 0, 0, -2, -1, -1,
            -2, -3, -4, -6, -8, -7, -9, -10, -11, -11,
        ],
        /* slot  96  bucket 6 (C_neg) phase  0 */
        [
            -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4,
            -1, 1, 3, 8, 9, 12, 9,
        ],
        /* slot  97  bucket 6 (C_neg) phase  1 */
        [
            3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3,
            8, 9, 12, 9, -127, 42,
        ],
        /* slot  98  bucket 6 (C_neg) phase  2 */
        [
            10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9,
            12, 9, -127, 42, 3, 10,
        ],
        /* slot  99  bucket 6 (C_neg) phase  3 */
        [
            3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9,
            -127, 42, 3, 10, 10, 5,
        ],
        /* slot 100  bucket 6 (C_neg) phase  4 */
        [
            -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127,
            42, 3, 10, 10, 5, 3, 2,
        ],
        /* slot 101  bucket 6 (C_neg) phase  5 */
        [
            0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3,
            10, 10, 5, 3, 2, -1, 2,
        ],
        /* slot 102  bucket 6 (C_neg) phase  6 */
        [
            1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10,
            5, 3, 2, -1, 2, 0, -1,
        ],
        /* slot 103  bucket 6 (C_neg) phase  7 */
        [
            1, 2, 1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3,
            2, -1, 2, 0, -1, 1, 1,
        ],
        /* slot 104  bucket 6 (C_neg) phase  8 */
        [
            1, 2, 1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2,
            -1, 2, 0, -1, 1, 1, 1, 2,
        ],
        /* slot 105  bucket 6 (C_neg) phase  9 */
        [
            1, 0, -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2,
            0, -1, 1, 1, 1, 2, 1, 2,
        ],
        /* slot 106  bucket 6 (C_neg) phase 10 */
        [
            -1, -1, -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1,
            1, 1, 1, 2, 1, 2, 1, 0,
        ],
        /* slot 107  bucket 6 (C_neg) phase 11 */
        [
            -4, -4, -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1,
            2, 1, 2, 1, 0, -1, -1,
        ],
        /* slot 108  bucket 6 (C_neg) phase 12 */
        [
            -4, -1, 1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1,
            2, 1, 0, -1, -1, -4, -4,
        ],
        /* slot 109  bucket 6 (C_neg) phase 13 */
        [
            1, 3, 8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0,
            -1, -1, -4, -4, -4, -1,
        ],
        /* slot 110  bucket 6 (C_neg) phase 14 */
        [
            8, 9, 12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1,
            -1, -4, -4, -4, -1, 1, 3,
        ],
        /* slot 111  bucket 6 (C_neg) phase 15 */
        [
            12, 9, -127, 42, 3, 10, 10, 5, 3, 2, -1, 2, 0, -1, 1, 1, 1, 2, 1, 2, 1, 0, -1, -1, -4,
            -4, -4, -1, 1, 3, 8, 9,
        ],
        /* slot 112  bucket 7 (D_neg) phase  0 */
        [
            127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21, -22, -20, -20, -18, -18, -19, -16,
            -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31,
        ],
        /* slot 113  bucket 7 (D_neg) phase  1 */
        [
            13, 2, -7, -11, -16, -19, -23, -22, -21, -22, -20, -20, -18, -18, -19, -16, -14, -11,
            -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36,
        ],
        /* slot 114  bucket 7 (D_neg) phase  2 */
        [
            -7, -11, -16, -19, -23, -22, -21, -22, -20, -20, -18, -18, -19, -16, -14, -11, -10, -6,
            -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2,
        ],
        /* slot 115  bucket 7 (D_neg) phase  3 */
        [
            -16, -19, -23, -22, -21, -22, -20, -20, -18, -18, -19, -16, -14, -11, -10, -6, -4, 0,
            4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11,
        ],
        /* slot 116  bucket 7 (D_neg) phase  4 */
        [
            -23, -22, -21, -22, -20, -20, -18, -18, -19, -16, -14, -11, -10, -6, -4, 0, 4, 7, 11,
            15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16, -19,
        ],
        /* slot 117  bucket 7 (D_neg) phase  5 */
        [
            -21, -22, -20, -20, -18, -18, -19, -16, -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24,
            28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23, -22,
        ],
        /* slot 118  bucket 7 (D_neg) phase  6 */
        [
            -20, -20, -18, -18, -19, -16, -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31,
            127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21, -22,
        ],
        /* slot 119  bucket 7 (D_neg) phase  7 */
        [
            -18, -18, -19, -16, -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36,
            13, 2, -7, -11, -16, -19, -23, -22, -21, -22, -20, -20,
        ],
        /* slot 120  bucket 7 (D_neg) phase  8 */
        [
            -19, -16, -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7,
            -11, -16, -19, -23, -22, -21, -22, -20, -20, -18, -18,
        ],
        /* slot 121  bucket 7 (D_neg) phase  9 */
        [
            -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16,
            -19, -23, -22, -21, -22, -20, -20, -18, -18, -19, -16,
        ],
        /* slot 122  bucket 7 (D_neg) phase 10 */
        [
            -10, -6, -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23,
            -22, -21, -22, -20, -20, -18, -18, -19, -16, -14, -11,
        ],
        /* slot 123  bucket 7 (D_neg) phase 11 */
        [
            -4, 0, 4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21,
            -22, -20, -20, -18, -18, -19, -16, -14, -11, -10, -6,
        ],
        /* slot 124  bucket 7 (D_neg) phase 12 */
        [
            4, 7, 11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21, -22,
            -20, -20, -18, -18, -19, -16, -14, -11, -10, -6, -4, 0,
        ],
        /* slot 125  bucket 7 (D_neg) phase 13 */
        [
            11, 15, 19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21, -22, -20,
            -20, -18, -18, -19, -16, -14, -11, -10, -6, -4, 0, 4, 7,
        ],
        /* slot 126  bucket 7 (D_neg) phase 14 */
        [
            19, 24, 28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21, -22, -20, -20, -18,
            -18, -19, -16, -14, -11, -10, -6, -4, 0, 4, 7, 11, 15,
        ],
        /* slot 127  bucket 7 (D_neg) phase 15 */
        [
            28, 31, 127, 36, 13, 2, -7, -11, -16, -19, -23, -22, -21, -22, -20, -20, -18, -18, -19,
            -16, -14, -11, -10, -6, -4, 0, 4, 7, 11, 15, 19, 24,
        ],
    ];

    /// Production curves regenerated from 7-bit calibration would replace
    /// this. For now we keep the original 256-entry calibration as a hidden
    /// `_FULL` array and expose the first 128 via const truncation. K-side
    /// inference paths that hit the static tables will use these 128 curves.
    #[rustfmt::skip]
    #[allow(dead_code)]
    const CURVE_TABLE_K_FULL: [[i8; 32]; 256] = [
    
      /* slot   0  bucket  0 phase  0  f= 0.18 ph=0.000 sharp */ [  127, 127, 126, 125, 123, 121, 118, 116, 112, 108, 104, 100,  96,  91,  86,  81,  76,  70,  65,  60,  55,  50,  45,  40,  36,  31,  27,  24,  20,  17,  14,  11 ],
      /* slot   1  bucket  0 phase  1  f= 0.18 ph=0.393 sharp */ [  100,  96,  91,  86,  81,  76,  70,  65,  60,  55,  50,  45,  40,  36,  31,  27,  24,  20,  17,  14,  11,   9,   7,   5,   4,   3,   2,   1,   1,   0,   0,   0 ],
      /* slot   2  bucket  0 phase  2  f= 0.18 ph=0.785 sharp */ [   45,  40,  36,  31,  27,  24,  20,  17,  14,  11,   9,   7,   5,   4,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -4 ],
      /* slot   3  bucket  0 phase  3  f= 0.18 ph=1.178 sharp */ [    7,   5,   4,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -4,  -5,  -7,  -9, -11, -14, -17, -20, -24, -27, -31, -36 ],
      /* slot   4  bucket  0 phase  4  f= 0.18 ph=1.571 sharp */ [    0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -4,  -5,  -7,  -9, -11, -14, -17, -20, -24, -27, -31, -36, -40, -45, -50, -55, -60, -65, -70, -76, -81, -86, -91 ],
      /* slot   5  bucket  0 phase  5  f= 0.18 ph=1.963 sharp */ [   -7,  -9, -11, -14, -17, -20, -24, -27, -31, -36, -40, -45, -50, -55, -60, -65, -70, -76, -81, -86, -91, -96,-100,-104,-108,-112,-116,-118,-121,-123,-125,-126 ],
      /* slot   6  bucket  0 phase  6  f= 0.18 ph=2.356 sharp */ [  -45, -50, -55, -60, -65, -70, -76, -81, -86, -91, -96,-100,-104,-108,-112,-116,-118,-121,-123,-125,-126,-127,-127,-127,-126,-125,-123,-121,-118,-116,-112,-108 ],
      /* slot   7  bucket  0 phase  7  f= 0.18 ph=2.749 sharp */ [ -100,-104,-108,-112,-116,-118,-121,-123,-125,-126,-127,-127,-127,-126,-125,-123,-121,-118,-116,-112,-108,-104,-100, -96, -91, -86, -81, -76, -70, -65, -60, -55 ],
      /* slot   8  bucket  0 phase  8  f= 0.18 ph=3.142 sharp */ [ -127,-127,-126,-125,-123,-121,-118,-116,-112,-108,-104,-100, -96, -91, -86, -81, -76, -70, -65, -60, -55, -50, -45, -40, -36, -31, -27, -24, -20, -17, -14, -11 ],
      /* slot   9  bucket  0 phase  9  f= 0.18 ph=3.534 sharp */ [ -100, -96, -91, -86, -81, -76, -70, -65, -60, -55, -50, -45, -40, -36, -31, -27, -24, -20, -17, -14, -11,  -9,  -7,  -5,  -4,  -3,  -2,  -1,  -1,   0,   0,   0 ],
      /* slot  10  bucket  0 phase 10  f= 0.18 ph=3.927 sharp */ [  -45, -40, -36, -31, -27, -24, -20, -17, -14, -11,  -9,  -7,  -5,  -4,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   4 ],
      /* slot  11  bucket  0 phase 11  f= 0.18 ph=4.320 sharp */ [   -7,  -5,  -4,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   4,   5,   7,   9,  11,  14,  17,  20,  24,  27,  31,  36 ],
      /* slot  12  bucket  0 phase 12  f= 0.18 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   1,   1,   2,   3,   4,   5,   7,   9,  11,  14,  17,  20,  24,  27,  31,  36,  40,  45,  50,  55,  60,  65,  70,  76,  81,  86,  91 ],
      /* slot  13  bucket  0 phase 13  f= 0.18 ph=5.105 sharp */ [    7,   9,  11,  14,  17,  20,  24,  27,  31,  36,  40,  45,  50,  55,  60,  65,  70,  76,  81,  86,  91,  96, 100, 104, 108, 112, 116, 118, 121, 123, 125, 126 ],
      /* slot  14  bucket  0 phase 14  f= 0.18 ph=5.498 sharp */ [   45,  50,  55,  60,  65,  70,  76,  81,  86,  91,  96, 100, 104, 108, 112, 116, 118, 121, 123, 125, 126, 127, 127, 127, 126, 125, 123, 121, 118, 116, 112, 108 ],
      /* slot  15  bucket  0 phase 15  f= 0.18 ph=5.890 sharp */ [  100, 104, 108, 112, 116, 118, 121, 123, 125, 126, 127, 127, 127, 126, 125, 123, 121, 118, 116, 112, 108, 104, 100,  96,  91,  86,  81,  76,  70,  65,  60,  55 ],
      /* slot  16  bucket  1 phase  0  f= 0.22 ph=0.000 sharp */ [  127, 127, 126, 124, 121, 118, 115, 110, 105, 100,  95,  89,  83,  76,  70,  64,  57,  51,  45,  39,  34,  29,  24,  20,  16,  13,  10,   7,   5,   4,   2,   1 ],
      /* slot  17  bucket  1 phase  1  f= 0.22 ph=0.393 sharp */ [  100,  95,  89,  83,  76,  70,  64,  57,  51,  45,  39,  34,  29,  24,  20,  16,  13,  10,   7,   5,   4,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,  -1 ],
      /* slot  18  bucket  1 phase  2  f= 0.22 ph=0.785 sharp */ [   45,  39,  34,  29,  24,  20,  16,  13,  10,   7,   5,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -5,  -7,  -9, -12, -16, -19 ],
      /* slot  19  bucket  1 phase  3  f= 0.22 ph=1.178 sharp */ [    7,   5,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -5,  -7,  -9, -12, -16, -19, -24, -28, -33, -39, -44, -50, -57, -63, -69 ],
      /* slot  20  bucket  1 phase  4  f= 0.22 ph=1.571 sharp */ [    0,   0,   0,   0,  -1,  -1,  -2,  -3,  -5,  -7, -10, -12, -16, -20, -24, -28, -33, -39, -45, -51, -57, -63, -69, -76, -82, -88, -94,-100,-105,-110,-114,-118 ],
      /* slot  21  bucket  1 phase  5  f= 0.22 ph=1.963 sharp */ [   -7, -10, -12, -16, -20, -24, -29, -34, -39, -45, -51, -57, -63, -70, -76, -82, -88, -94,-100,-105,-110,-114,-118,-121,-124,-125,-127,-127,-127,-126,-124,-122 ],
      /* slot  22  bucket  1 phase  6  f= 0.22 ph=2.356 sharp */ [  -45, -51, -57, -63, -70, -76, -82, -89, -94,-100,-105,-110,-114,-118,-121,-124,-125,-127,-127,-127,-126,-124,-121,-118,-115,-110,-106,-101, -95, -89, -83, -77 ],
      /* slot  23  bucket  1 phase  7  f= 0.22 ph=2.749 sharp */ [ -100,-105,-110,-114,-118,-121,-124,-126,-127,-127,-127,-126,-124,-121,-118,-115,-110,-106,-100, -95, -89, -83, -77, -70, -64, -58, -51, -45, -40, -34, -29, -24 ],
      /* slot  24  bucket  1 phase  8  f= 0.22 ph=3.142 sharp */ [ -127,-127,-126,-124,-121,-118,-115,-110,-105,-100, -95, -89, -83, -76, -70, -64, -57, -51, -45, -39, -34, -29, -24, -20, -16, -13, -10,  -7,  -5,  -4,  -2,  -1 ],
      /* slot  25  bucket  1 phase  9  f= 0.22 ph=3.534 sharp */ [ -100, -95, -89, -83, -76, -70, -64, -57, -51, -45, -39, -34, -29, -24, -20, -16, -13, -10,  -7,  -5,  -4,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   1 ],
      /* slot  26  bucket  1 phase 10  f= 0.22 ph=3.927 sharp */ [  -45, -39, -34, -29, -24, -20, -16, -13, -10,  -7,  -5,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   5,   7,   9,  12,  16,  19 ],
      /* slot  27  bucket  1 phase 11  f= 0.22 ph=4.320 sharp */ [   -7,  -5,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   5,   7,   9,  12,  16,  19,  24,  28,  33,  39,  44,  50,  57,  63,  69 ],
      /* slot  28  bucket  1 phase 12  f= 0.22 ph=4.712 sharp */ [    0,   0,   0,   0,   1,   1,   2,   3,   5,   7,  10,  12,  16,  20,  24,  28,  33,  39,  45,  51,  57,  63,  69,  76,  82,  88,  94, 100, 105, 110, 114, 118 ],
      /* slot  29  bucket  1 phase 13  f= 0.22 ph=5.105 sharp */ [    7,  10,  12,  16,  20,  24,  29,  34,  39,  45,  51,  57,  63,  70,  76,  82,  88,  94, 100, 105, 110, 114, 118, 121, 124, 125, 127, 127, 127, 126, 124, 122 ],
      /* slot  30  bucket  1 phase 14  f= 0.22 ph=5.498 sharp */ [   45,  51,  57,  63,  70,  76,  82,  89,  94, 100, 105, 110, 114, 118, 121, 124, 125, 127, 127, 127, 126, 124, 121, 118, 115, 110, 106, 101,  95,  89,  83,  77 ],
      /* slot  31  bucket  1 phase 15  f= 0.22 ph=5.890 sharp */ [  100, 105, 110, 114, 118, 121, 124, 126, 127, 127, 127, 126, 124, 121, 118, 115, 110, 106, 100,  95,  89,  83,  77,  70,  64,  58,  51,  45,  40,  34,  29,  24 ],
      /* slot  32  bucket  2 phase  0  f= 1.00 ph=0.000 sharp */ [  127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120 ],
      /* slot  33  bucket  2 phase  1  f= 1.00 ph=0.393 sharp */ [  100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120 ],
      /* slot  34  bucket  2 phase  2  f= 1.00 ph=0.785 sharp */ [   45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73 ],
      /* slot  35  bucket  2 phase  3  f= 1.00 ph=1.178 sharp */ [    7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22 ],
      /* slot  36  bucket  2 phase  4  f= 1.00 ph=1.571 sharp */ [    0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1 ],
      /* slot  37  bucket  2 phase  5  f= 1.00 ph=1.963 sharp */ [   -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1 ],
      /* slot  38  bucket  2 phase  6  f= 1.00 ph=2.356 sharp */ [  -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22 ],
      /* slot  39  bucket  2 phase  7  f= 1.00 ph=2.749 sharp */ [ -100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73 ],
      /* slot  40  bucket  2 phase  8  f= 1.00 ph=3.142 sharp */ [ -127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120 ],
      /* slot  41  bucket  2 phase  9  f= 1.00 ph=3.534 sharp */ [ -100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120 ],
      /* slot  42  bucket  2 phase 10  f= 1.00 ph=3.927 sharp */ [  -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73 ],
      /* slot  43  bucket  2 phase 11  f= 1.00 ph=4.320 sharp */ [   -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22 ],
      /* slot  44  bucket  2 phase 12  f= 1.00 ph=4.712 sharp */ [    0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1 ],
      /* slot  45  bucket  2 phase 13  f= 1.00 ph=5.105 sharp */ [    7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1 ],
      /* slot  46  bucket  2 phase 14  f= 1.00 ph=5.498 sharp */ [   45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22 ],
      /* slot  47  bucket  2 phase 15  f= 1.00 ph=5.890 sharp */ [  100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73 ],
      /* slot  48  bucket  3 phase  0  f= 0.14 ph=0.000 triangle */ [  127, 125, 122, 120, 118, 116, 113, 111, 109, 107, 104, 102, 100,  98,  95,  93,  91,  89,  86,  84,  82,  80,  77,  75,  73,  71,  68,  66,  64,  62,  59,  57 ],
      /* slot  49  bucket  3 phase  1  f= 0.14 ph=0.393 triangle */ [   95,  93,  91,  88,  86,  84,  82,  79,  77,  75,  73,  70,  68,  66,  64,  61,  59,  57,  55,  52,  50,  48,  46,  43,  41,  39,  37,  34,  32,  30,  28,  25 ],
      /* slot  50  bucket  3 phase  2  f= 0.14 ph=0.785 triangle */ [   64,  61,  59,  57,  54,  52,  50,  48,  45,  43,  41,  39,  36,  34,  32,  30,  27,  25,  23,  21,  18,  16,  14,  12,   9,   7,   5,   3,   0,  -2,  -4,  -6 ],
      /* slot  51  bucket  3 phase  3  f= 0.14 ph=1.178 triangle */ [   32,  29,  27,  25,  23,  20,  18,  16,  14,  11,   9,   7,   5,   2,   0,  -2,  -4,  -7,  -9, -11, -13, -16, -18, -20, -22, -25, -27, -29, -31, -34, -36, -38 ],
      /* slot  52  bucket  3 phase  4  f= 0.14 ph=1.571 triangle */ [    0,  -2,  -5,  -7,  -9, -11, -14, -16, -18, -20, -23, -25, -27, -29, -32, -34, -36, -38, -41, -43, -45, -47, -50, -52, -54, -56, -59, -61, -63, -65, -68, -70 ],
      /* slot  53  bucket  3 phase  5  f= 0.14 ph=1.963 triangle */ [  -32, -34, -36, -39, -41, -43, -45, -48, -50, -52, -54, -57, -59, -61, -63, -66, -68, -70, -72, -75, -77, -79, -81, -84, -86, -88, -90, -93, -95, -97, -99,-102 ],
      /* slot  54  bucket  3 phase  6  f= 0.14 ph=2.356 triangle */ [  -64, -66, -68, -70, -73, -75, -77, -79, -82, -84, -86, -88, -91, -93, -95, -97,-100,-102,-104,-106,-109,-111,-113,-115,-118,-120,-122,-124,-127,-125,-123,-121 ],
      /* slot  55  bucket  3 phase  7  f= 0.14 ph=2.749 triangle */ [  -95, -98,-100,-102,-104,-107,-109,-111,-113,-116,-118,-120,-122,-125,-127,-125,-123,-120,-118,-116,-114,-111,-109,-107,-105,-102,-100, -98, -96, -93, -91, -89 ],
      /* slot  56  bucket  3 phase  8  f= 0.14 ph=3.142 triangle */ [ -127,-125,-122,-120,-118,-116,-113,-111,-109,-107,-104,-102,-100, -98, -95, -93, -91, -89, -86, -84, -82, -80, -77, -75, -73, -71, -68, -66, -64, -62, -59, -57 ],
      /* slot  57  bucket  3 phase  9  f= 0.14 ph=3.534 triangle */ [  -95, -93, -91, -88, -86, -84, -82, -79, -77, -75, -73, -70, -68, -66, -64, -61, -59, -57, -55, -52, -50, -48, -46, -43, -41, -39, -37, -34, -32, -30, -28, -25 ],
      /* slot  58  bucket  3 phase 10  f= 0.14 ph=3.927 triangle */ [  -64, -61, -59, -57, -54, -52, -50, -48, -45, -43, -41, -39, -36, -34, -32, -30, -27, -25, -23, -21, -18, -16, -14, -12,  -9,  -7,  -5,  -3,   0,   2,   4,   6 ],
      /* slot  59  bucket  3 phase 11  f= 0.14 ph=4.320 triangle */ [  -32, -29, -27, -25, -23, -20, -18, -16, -14, -11,  -9,  -7,  -5,  -2,   0,   2,   4,   7,   9,  11,  13,  16,  18,  20,  22,  25,  27,  29,  31,  34,  36,  38 ],
      /* slot  60  bucket  3 phase 12  f= 0.14 ph=4.712 triangle */ [    0,   2,   5,   7,   9,  11,  14,  16,  18,  20,  23,  25,  27,  29,  32,  34,  36,  38,  41,  43,  45,  47,  50,  52,  54,  56,  59,  61,  63,  65,  68,  70 ],
      /* slot  61  bucket  3 phase 13  f= 0.14 ph=5.105 triangle */ [   32,  34,  36,  39,  41,  43,  45,  48,  50,  52,  54,  57,  59,  61,  63,  66,  68,  70,  72,  75,  77,  79,  81,  84,  86,  88,  90,  93,  95,  97,  99, 102 ],
      /* slot  62  bucket  3 phase 14  f= 0.14 ph=5.498 triangle */ [   64,  66,  68,  70,  73,  75,  77,  79,  82,  84,  86,  88,  91,  93,  95,  97, 100, 102, 104, 106, 109, 111, 113, 115, 118, 120, 122, 124, 127, 125, 123, 121 ],
      /* slot  63  bucket  3 phase 15  f= 0.14 ph=5.890 triangle */ [   95,  98, 100, 102, 104, 107, 109, 111, 113, 116, 118, 120, 122, 125, 127, 125, 123, 120, 118, 116, 114, 111, 109, 107, 105, 102, 100,  98,  96,  93,  91,  89 ],
      /* slot  64  bucket  4 phase  0  f= 0.30 ph=0.000 sharp */ [  127, 126, 124, 121, 117, 111, 105,  97,  90,  81,  73,  64,  55,  47,  39,  32,  25,  20,  14,  10,   7,   4,   2,   1,   0,   0,   0,   0,   0,   0,  -1,  -2 ],
      /* slot  65  bucket  4 phase  1  f= 0.30 ph=0.393 sharp */ [  100,  92,  84,  76,  67,  59,  50,  42,  35,  28,  22,  16,  12,   8,   5,   3,   2,   1,   0,   0,   0,   0,   0,  -1,  -2,  -3,  -6,  -9, -12, -17, -22, -29 ],
      /* slot  66  bucket  4 phase  2  f= 0.30 ph=0.785 sharp */ [   45,  37,  30,  24,  18,  13,   9,   6,   4,   2,   1,   0,   0,   0,   0,   0,  -1,  -1,  -3,  -5,  -7, -11, -15, -20, -26, -33, -40, -48, -57, -65, -74, -83 ],
      /* slot  67  bucket  4 phase  3  f= 0.30 ph=1.178 sharp */ [    7,   4,   3,   1,   0,   0,   0,   0,   0,   0,  -1,  -2,  -4,  -6, -10, -14, -18, -24, -31, -38, -46, -54, -62, -71, -79, -88, -96,-103,-110,-116,-120,-124 ],
      /* slot  68  bucket  4 phase  4  f= 0.30 ph=1.571 sharp */ [    0,   0,   0,  -1,  -2,  -3,  -5,  -8, -12, -17, -22, -28, -35, -43, -51, -59, -68, -76, -85, -93,-101,-108,-114,-119,-123,-125,-127,-127,-126,-123,-119,-114 ],
      /* slot  69  bucket  4 phase  5  f= 0.30 ph=1.963 sharp */ [   -7, -11, -15, -20, -26, -33, -40, -48, -56, -65, -73, -82, -90, -98,-105,-112,-117,-121,-125,-126,-127,-126,-124,-121,-116,-111,-104, -97, -89, -81, -72, -63 ],
      /* slot  70  bucket  4 phase  6  f= 0.30 ph=2.356 sharp */ [  -45, -53, -62, -70, -79, -87, -95,-103,-109,-115,-120,-124,-126,-127,-127,-125,-122,-118,-113,-107,-100, -92, -84, -75, -66, -58, -49, -41, -34, -27, -21, -16 ],
      /* slot  71  bucket  4 phase  7  f= 0.30 ph=2.749 sharp */ [ -100,-107,-113,-118,-122,-125,-127,-127,-126,-123,-120,-115,-109,-102, -95, -87, -78, -70, -61, -52, -44, -37, -30, -23, -18, -13,  -9,  -6,  -4,  -2,  -1,   0 ],
      /* slot  72  bucket  4 phase  8  f= 0.30 ph=3.142 sharp */ [ -127,-126,-124,-121,-117,-111,-105, -97, -90, -81, -73, -64, -55, -47, -39, -32, -25, -20, -14, -10,  -7,  -4,  -2,  -1,   0,   0,   0,   0,   0,   0,   1,   2 ],
      /* slot  73  bucket  4 phase  9  f= 0.30 ph=3.534 sharp */ [ -100, -92, -84, -76, -67, -59, -50, -42, -35, -28, -22, -16, -12,  -8,  -5,  -3,  -2,  -1,   0,   0,   0,   0,   0,   1,   2,   3,   6,   9,  12,  17,  22,  29 ],
      /* slot  74  bucket  4 phase 10  f= 0.30 ph=3.927 sharp */ [  -45, -37, -30, -24, -18, -13,  -9,  -6,  -4,  -2,  -1,   0,   0,   0,   0,   0,   1,   1,   3,   5,   7,  11,  15,  20,  26,  33,  40,  48,  57,  65,  74,  83 ],
      /* slot  75  bucket  4 phase 11  f= 0.30 ph=4.320 sharp */ [   -7,  -4,  -3,  -1,   0,   0,   0,   0,   0,   0,   1,   2,   4,   6,  10,  14,  18,  24,  31,  38,  46,  54,  62,  71,  79,  88,  96, 103, 110, 116, 120, 124 ],
      /* slot  76  bucket  4 phase 12  f= 0.30 ph=4.712 sharp */ [    0,   0,   0,   1,   2,   3,   5,   8,  12,  17,  22,  28,  35,  43,  51,  59,  68,  76,  85,  93, 101, 108, 114, 119, 123, 125, 127, 127, 126, 123, 119, 114 ],
      /* slot  77  bucket  4 phase 13  f= 0.30 ph=5.105 sharp */ [    7,  11,  15,  20,  26,  33,  40,  48,  56,  65,  73,  82,  90,  98, 105, 112, 117, 121, 125, 126, 127, 126, 124, 121, 116, 111, 104,  97,  89,  81,  72,  63 ],
      /* slot  78  bucket  4 phase 14  f= 0.30 ph=5.498 sharp */ [   45,  53,  62,  70,  79,  87,  95, 103, 109, 115, 120, 124, 126, 127, 127, 125, 122, 118, 113, 107, 100,  92,  84,  75,  66,  58,  49,  41,  34,  27,  21,  16 ],
      /* slot  79  bucket  4 phase 15  f= 0.30 ph=5.890 sharp */ [  100, 107, 113, 118, 122, 125, 127, 127, 126, 123, 120, 115, 109, 102,  95,  87,  78,  70,  61,  52,  44,  37,  30,  23,  18,  13,   9,   6,   4,   2,   1,   0 ],
      /* slot  80  bucket  5 phase  0  f= 0.26 ph=0.000 sharp */ [  127, 126, 125, 123, 119, 115, 110, 104,  98,  91,  84,  77,  69,  62,  54,  47,  40,  34,  28,  22,  18,  13,  10,   7,   5,   3,   2,   1,   0,   0,   0,   0 ],
      /* slot  81  bucket  5 phase  1  f= 0.26 ph=0.393 sharp */ [  100,  94,  86,  79,  72,  64,  57,  49,  43,  36,  30,  24,  19,  15,  11,   8,   5,   3,   2,   1,   0,   0,   0,   0,   0,   0,   0,  -1,  -2,  -4,  -6,  -8 ],
      /* slot  82  bucket  5 phase  2  f= 0.26 ph=0.785 sharp */ [   45,  38,  32,  26,  21,  16,  12,   9,   6,   4,   2,   1,   1,   0,   0,   0,   0,   0,   0,  -1,  -2,  -3,  -5,  -7, -10, -14, -18, -23, -28, -34, -41, -48 ],
      /* slot  83  bucket  5 phase  3  f= 0.26 ph=1.178 sharp */ [    7,   5,   3,   2,   1,   0,   0,   0,   0,   0,   0,  -1,  -1,  -3,  -4,  -6,  -9, -12, -16, -21, -26, -32, -38, -45, -52, -60, -67, -75, -82, -89, -96,-103 ],
      /* slot  84  bucket  5 phase  4  f= 0.26 ph=1.571 sharp */ [    0,   0,   0,   0,  -1,  -2,  -4,  -6,  -8, -11, -15, -19, -24, -30, -36, -43, -50, -57, -65, -72, -79, -87, -94,-100,-107,-112,-117,-121,-124,-126,-127,-127 ],
      /* slot  85  bucket  5 phase  5  f= 0.26 ph=1.963 sharp */ [   -7, -10, -14, -18, -23, -28, -34, -40, -47, -55, -62, -69, -77, -84, -91, -98,-104,-110,-115,-119,-123,-125,-127,-127,-126,-125,-122,-119,-115,-110,-104, -98 ],
      /* slot  86  bucket  5 phase  6  f= 0.26 ph=2.356 sharp */ [  -45, -52, -59, -67, -74, -82, -89, -96,-102,-108,-113,-118,-122,-124,-126,-127,-127,-126,-123,-120,-116,-112,-106,-100, -93, -86, -79, -71, -64, -56, -49, -42 ],
      /* slot  87  bucket  5 phase  7  f= 0.26 ph=2.749 sharp */ [ -100,-106,-112,-116,-120,-124,-126,-127,-127,-126,-124,-121,-118,-113,-108,-102, -96, -89, -81, -74, -66, -59, -52, -45, -38, -32, -26, -21, -16, -12,  -9,  -6 ],
      /* slot  88  bucket  5 phase  8  f= 0.26 ph=3.142 sharp */ [ -127,-126,-125,-123,-119,-115,-110,-104, -98, -91, -84, -77, -69, -62, -54, -47, -40, -34, -28, -22, -18, -13, -10,  -7,  -5,  -3,  -2,  -1,   0,   0,   0,   0 ],
      /* slot  89  bucket  5 phase  9  f= 0.26 ph=3.534 sharp */ [ -100, -94, -86, -79, -72, -64, -57, -49, -43, -36, -30, -24, -19, -15, -11,  -8,  -5,  -3,  -2,  -1,   0,   0,   0,   0,   0,   0,   0,   1,   2,   4,   6,   8 ],
      /* slot  90  bucket  5 phase 10  f= 0.26 ph=3.927 sharp */ [  -45, -38, -32, -26, -21, -16, -12,  -9,  -6,  -4,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   1,   2,   3,   5,   7,  10,  14,  18,  23,  28,  34,  41,  48 ],
      /* slot  91  bucket  5 phase 11  f= 0.26 ph=4.320 sharp */ [   -7,  -5,  -3,  -2,  -1,   0,   0,   0,   0,   0,   0,   1,   1,   3,   4,   6,   9,  12,  16,  21,  26,  32,  38,  45,  52,  60,  67,  75,  82,  89,  96, 103 ],
      /* slot  92  bucket  5 phase 12  f= 0.26 ph=4.712 sharp */ [    0,   0,   0,   0,   1,   2,   4,   6,   8,  11,  15,  19,  24,  30,  36,  43,  50,  57,  65,  72,  79,  87,  94, 100, 107, 112, 117, 121, 124, 126, 127, 127 ],
      /* slot  93  bucket  5 phase 13  f= 0.26 ph=5.105 sharp */ [    7,  10,  14,  18,  23,  28,  34,  40,  47,  55,  62,  69,  77,  84,  91,  98, 104, 110, 115, 119, 123, 125, 127, 127, 126, 125, 122, 119, 115, 110, 104,  98 ],
      /* slot  94  bucket  5 phase 14  f= 0.26 ph=5.498 sharp */ [   45,  52,  59,  67,  74,  82,  89,  96, 102, 108, 113, 118, 122, 124, 126, 127, 127, 126, 123, 120, 116, 112, 106, 100,  93,  86,  79,  71,  64,  56,  49,  42 ],
      /* slot  95  bucket  5 phase 15  f= 0.26 ph=5.890 sharp */ [  100, 106, 112, 116, 120, 124, 126, 127, 127, 126, 124, 121, 118, 113, 108, 102,  96,  89,  81,  74,  66,  59,  52,  45,  38,  32,  26,  21,  16,  12,   9,   6 ],
      /* slot  96  bucket  6 phase  0  f= 0.10 ph=0.000 sharp */ [  127, 127, 127, 126, 126, 125, 124, 123, 122, 121, 119, 118, 116, 115, 113, 111, 108, 106, 104, 102,  99,  96,  94,  91,  88,  86,  83,  80,  77,  74,  71,  68 ],
      /* slot  97  bucket  6 phase  1  f= 0.10 ph=0.393 sharp */ [  100,  98,  95,  92,  90,  87,  84,  81,  78,  75,  72,  69,  66,  64,  61,  58,  55,  52,  49,  46,  44,  41,  38,  36,  33,  31,  29,  27,  24,  22,  20,  19 ],
      /* slot  98  bucket  6 phase  2  f= 0.10 ph=0.785 sharp */ [   45,  42,  40,  37,  35,  32,  30,  28,  25,  23,  21,  19,  18,  16,  14,  13,  11,  10,   9,   8,   7,   6,   5,   4,   3,   3,   2,   2,   1,   1,   1,   1 ],
      /* slot  99  bucket  6 phase  3  f= 0.10 ph=1.178 sharp */ [    7,   6,   5,   4,   4,   3,   2,   2,   2,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -2 ],
      /* slot 100  bucket  6 phase  4  f= 0.10 ph=1.571 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -1,  -2,  -2,  -3,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10, -11, -13, -14, -16, -17, -19, -21, -23, -25 ],
      /* slot 101  bucket  6 phase  5  f= 0.10 ph=1.963 sharp */ [   -7,  -8,  -9, -11, -12, -13, -15, -17, -18, -20, -22, -24, -26, -29, -31, -33, -36, -38, -41, -43, -46, -49, -52, -55, -57, -60, -63, -66, -69, -72, -75, -78 ],
      /* slot 102  bucket  6 phase  6  f= 0.10 ph=2.356 sharp */ [  -45, -48, -50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -79, -82, -85, -88, -91, -94, -96, -99,-101,-104,-106,-108,-110,-112,-114,-116,-118,-119,-121,-122 ],
      /* slot 103  bucket  6 phase  7  f= 0.10 ph=2.749 sharp */ [ -100,-103,-105,-107,-109,-112,-113,-115,-117,-119,-120,-121,-123,-124,-125,-125,-126,-126,-127,-127,-127,-127,-127,-126,-125,-125,-124,-123,-122,-120,-119,-117 ],
      /* slot 104  bucket  6 phase  8  f= 0.10 ph=3.142 sharp */ [ -127,-127,-127,-126,-126,-125,-124,-123,-122,-121,-119,-118,-116,-115,-113,-111,-108,-106,-104,-102, -99, -96, -94, -91, -88, -86, -83, -80, -77, -74, -71, -68 ],
      /* slot 105  bucket  6 phase  9  f= 0.10 ph=3.534 sharp */ [ -100, -98, -95, -92, -90, -87, -84, -81, -78, -75, -72, -69, -66, -64, -61, -58, -55, -52, -49, -46, -44, -41, -38, -36, -33, -31, -29, -27, -24, -22, -20, -19 ],
      /* slot 106  bucket  6 phase 10  f= 0.10 ph=3.927 sharp */ [  -45, -42, -40, -37, -35, -32, -30, -28, -25, -23, -21, -19, -18, -16, -14, -13, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -3,  -2,  -2,  -1,  -1,  -1,  -1 ],
      /* slot 107  bucket  6 phase 11  f= 0.10 ph=4.320 sharp */ [   -7,  -6,  -5,  -4,  -4,  -3,  -2,  -2,  -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   2 ],
      /* slot 108  bucket  6 phase 12  f= 0.10 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   2,   2,   3,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,  14,  16,  17,  19,  21,  23,  25 ],
      /* slot 109  bucket  6 phase 13  f= 0.10 ph=5.105 sharp */ [    7,   8,   9,  11,  12,  13,  15,  17,  18,  20,  22,  24,  26,  29,  31,  33,  36,  38,  41,  43,  46,  49,  52,  55,  57,  60,  63,  66,  69,  72,  75,  78 ],
      /* slot 110  bucket  6 phase 14  f= 0.10 ph=5.498 sharp */ [   45,  48,  50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  79,  82,  85,  88,  91,  94,  96,  99, 101, 104, 106, 108, 110, 112, 114, 116, 118, 119, 121, 122 ],
      /* slot 111  bucket  6 phase 15  f= 0.10 ph=5.890 sharp */ [  100, 103, 105, 107, 109, 112, 113, 115, 117, 119, 120, 121, 123, 124, 125, 125, 126, 126, 127, 127, 127, 127, 127, 126, 125, 125, 124, 123, 122, 120, 119, 117 ],
      /* slot 112  bucket  7 phase  0  f= 0.34 ph=0.000 sharp */ [  127, 126, 124, 119, 114, 107,  99,  90,  81,  71,  61,  52,  43,  34,  26,  20,  14,   9,   6,   3,   2,   1,   0,   0,   0,   0,  -1,  -2,  -3,  -6, -10, -14 ],
      /* slot 113  bucket  7 phase  1  f= 0.34 ph=0.393 sharp */ [  100,  91,  82,  72,  63,  53,  44,  35,  27,  20,  15,  10,   6,   4,   2,   1,   0,   0,   0,   0,  -1,  -1,  -3,  -6,  -9, -14, -19, -26, -33, -42, -51, -61 ],
      /* slot 114  bucket  7 phase  2  f= 0.34 ph=0.785 sharp */ [   45,  36,  28,  21,  15,  10,   7,   4,   2,   1,   0,   0,   0,   0,   0,  -1,  -3,  -5,  -9, -13, -18, -25, -32, -41, -50, -59, -69, -79, -88, -97,-105,-113 ],
      /* slot 115  bucket  7 phase  3  f= 0.34 ph=1.178 sharp */ [    7,   4,   2,   1,   0,   0,   0,   0,   0,  -1,  -3,  -5,  -8, -12, -18, -24, -31, -40, -49, -58, -68, -78, -87, -96,-104,-112,-118,-122,-125,-127,-127,-125 ],
      /* slot 116  bucket  7 phase  4  f= 0.34 ph=1.571 sharp */ [    0,   0,   0,  -1,  -2,  -5,  -8, -12, -17, -23, -30, -38, -47, -57, -66, -76, -86, -95,-103,-111,-117,-122,-125,-127,-127,-125,-122,-117,-110,-103, -94, -85 ],
      /* slot 117  bucket  7 phase  5  f= 0.34 ph=1.963 sharp */ [   -7, -11, -16, -22, -29, -37, -46, -55, -65, -75, -85, -94,-102,-110,-116,-121,-125,-127,-127,-125,-122,-117,-111,-104, -96, -86, -77, -67, -57, -48, -39, -31 ],
      /* slot 118  bucket  7 phase  6  f= 0.34 ph=2.356 sharp */ [  -45, -54, -64, -74, -83, -93,-101,-109,-116,-121,-124,-127,-127,-126,-123,-118,-112,-105, -97, -88, -78, -68, -59, -49, -40, -32, -24, -18, -13,  -8,  -5,  -3 ],
      /* slot 119  bucket  7 phase  7  f= 0.34 ph=2.749 sharp */ [ -100,-108,-115,-120,-124,-126,-127,-126,-123,-119,-113,-106, -98, -89, -79, -70, -60, -50, -41, -33, -25, -19, -13,  -9,  -5,  -3,  -1,   0,   0,   0,   0,   0 ],
      /* slot 120  bucket  7 phase  8  f= 0.34 ph=3.142 sharp */ [ -127,-126,-124,-119,-114,-107, -99, -90, -81, -71, -61, -52, -43, -34, -26, -20, -14,  -9,  -6,  -3,  -2,  -1,   0,   0,   0,   0,   1,   2,   3,   6,  10,  14 ],
      /* slot 121  bucket  7 phase  9  f= 0.34 ph=3.534 sharp */ [ -100, -91, -82, -72, -63, -53, -44, -35, -27, -20, -15, -10,  -6,  -4,  -2,  -1,   0,   0,   0,   0,   1,   1,   3,   6,   9,  14,  19,  26,  33,  42,  51,  61 ],
      /* slot 122  bucket  7 phase 10  f= 0.34 ph=3.927 sharp */ [  -45, -36, -28, -21, -15, -10,  -7,  -4,  -2,  -1,   0,   0,   0,   0,   0,   1,   3,   5,   9,  13,  18,  25,  32,  41,  50,  59,  69,  79,  88,  97, 105, 113 ],
      /* slot 123  bucket  7 phase 11  f= 0.34 ph=4.320 sharp */ [   -7,  -4,  -2,  -1,   0,   0,   0,   0,   0,   1,   3,   5,   8,  12,  18,  24,  31,  40,  49,  58,  68,  78,  87,  96, 104, 112, 118, 122, 125, 127, 127, 125 ],
      /* slot 124  bucket  7 phase 12  f= 0.34 ph=4.712 sharp */ [    0,   0,   0,   1,   2,   5,   8,  12,  17,  23,  30,  38,  47,  57,  66,  76,  86,  95, 103, 111, 117, 122, 125, 127, 127, 125, 122, 117, 110, 103,  94,  85 ],
      /* slot 125  bucket  7 phase 13  f= 0.34 ph=5.105 sharp */ [    7,  11,  16,  22,  29,  37,  46,  55,  65,  75,  85,  94, 102, 110, 116, 121, 125, 127, 127, 125, 122, 117, 111, 104,  96,  86,  77,  67,  57,  48,  39,  31 ],
      /* slot 126  bucket  7 phase 14  f= 0.34 ph=5.498 sharp */ [   45,  54,  64,  74,  83,  93, 101, 109, 116, 121, 124, 127, 127, 126, 123, 118, 112, 105,  97,  88,  78,  68,  59,  49,  40,  32,  24,  18,  13,   8,   5,   3 ],
      /* slot 127  bucket  7 phase 15  f= 0.34 ph=5.890 sharp */ [  100, 108, 115, 120, 124, 126, 127, 126, 123, 119, 113, 106,  98,  89,  79,  70,  60,  50,  41,  33,  25,  19,  13,   9,   5,   3,   1,   0,   0,   0,   0,   0 ],
      /* slot 128  bucket  8 phase  0  f= 0.06 ph=0.000 sharp */ [  127, 127, 127, 127, 127, 126, 126, 126, 125, 125, 124, 124, 123, 122, 121, 121, 120, 119, 118, 117, 116, 115, 114, 113, 111, 110, 109, 107, 106, 105, 103, 102 ],
      /* slot 129  bucket  8 phase  1  f= 0.06 ph=0.393 sharp */ [  100,  99,  97,  95,  94,  92,  91,  89,  87,  85,  84,  82,  80,  78,  77,  75,  73,  71,  69,  68,  66,  64,  62,  60,  59,  57,  55,  53,  52,  50,  48,  47 ],
      /* slot 130  bucket  8 phase  2  f= 0.06 ph=0.785 sharp */ [   45,  43,  42,  40,  38,  37,  35,  34,  32,  31,  30,  28,  27,  26,  24,  23,  22,  21,  19,  18,  17,  16,  15,  14,  13,  12,  12,  11,  10,   9,   8,   8 ],
      /* slot 131  bucket  8 phase  3  f= 0.06 ph=1.178 sharp */ [    7,   7,   6,   5,   5,   4,   4,   3,   3,   3,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot 132  bucket  8 phase  4  f= 0.06 ph=1.571 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -2,  -2,  -2,  -2,  -3,  -3,  -3,  -4,  -4,  -5,  -5,  -6,  -7 ],
      /* slot 133  bucket  8 phase  5  f= 0.06 ph=1.963 sharp */ [   -7,  -8,  -8,  -9, -10, -11, -12, -12, -13, -14, -15, -16, -17, -18, -19, -21, -22, -23, -24, -26, -27, -28, -30, -31, -32, -34, -35, -37, -38, -40, -42, -43 ],
      /* slot 134  bucket  8 phase  6  f= 0.06 ph=2.356 sharp */ [  -45, -47, -48, -50, -52, -53, -55, -57, -59, -60, -62, -64, -66, -68, -69, -71, -73, -75, -77, -78, -80, -82, -84, -85, -87, -89, -91, -92, -94, -95, -97, -99 ],
      /* slot 135  bucket  8 phase  7  f= 0.06 ph=2.749 sharp */ [ -100,-102,-103,-105,-106,-107,-109,-110,-111,-113,-114,-115,-116,-117,-118,-119,-120,-121,-121,-122,-123,-124,-124,-125,-125,-126,-126,-126,-127,-127,-127,-127 ],
      /* slot 136  bucket  8 phase  8  f= 0.06 ph=3.142 sharp */ [ -127,-127,-127,-127,-127,-126,-126,-126,-125,-125,-124,-124,-123,-122,-121,-121,-120,-119,-118,-117,-116,-115,-114,-113,-111,-110,-109,-107,-106,-105,-103,-102 ],
      /* slot 137  bucket  8 phase  9  f= 0.06 ph=3.534 sharp */ [ -100, -99, -97, -95, -94, -92, -91, -89, -87, -85, -84, -82, -80, -78, -77, -75, -73, -71, -69, -68, -66, -64, -62, -60, -59, -57, -55, -53, -52, -50, -48, -47 ],
      /* slot 138  bucket  8 phase 10  f= 0.06 ph=3.927 sharp */ [  -45, -43, -42, -40, -38, -37, -35, -34, -32, -31, -30, -28, -27, -26, -24, -23, -22, -21, -19, -18, -17, -16, -15, -14, -13, -12, -12, -11, -10,  -9,  -8,  -8 ],
      /* slot 139  bucket  8 phase 11  f= 0.06 ph=4.320 sharp */ [   -7,  -7,  -6,  -5,  -5,  -4,  -4,  -3,  -3,  -3,  -2,  -2,  -2,  -2,  -1,  -1,  -1,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot 140  bucket  8 phase 12  f= 0.06 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   3,   3,   3,   4,   4,   5,   5,   6,   7 ],
      /* slot 141  bucket  8 phase 13  f= 0.06 ph=5.105 sharp */ [    7,   8,   8,   9,  10,  11,  12,  12,  13,  14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  26,  27,  28,  30,  31,  32,  34,  35,  37,  38,  40,  42,  43 ],
      /* slot 142  bucket  8 phase 14  f= 0.06 ph=5.498 sharp */ [   45,  47,  48,  50,  52,  53,  55,  57,  59,  60,  62,  64,  66,  68,  69,  71,  73,  75,  77,  78,  80,  82,  84,  85,  87,  89,  91,  92,  94,  95,  97,  99 ],
      /* slot 143  bucket  8 phase 15  f= 0.06 ph=5.890 sharp */ [  100, 102, 103, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 124, 124, 125, 125, 126, 126, 126, 127, 127, 127, 127 ],
      /* slot 144  bucket  9 phase  0  f= 0.14 ph=0.000 sharp */ [  127, 127, 126, 126, 125, 123, 122, 120, 118, 115, 113, 110, 107, 104, 100,  97,  93,  89,  86,  82,  78,  73,  69,  65,  61,  57,  53,  49,  46,  42,  38,  35 ],
      /* slot 145  bucket  9 phase  1  f= 0.14 ph=0.393 sharp */ [  100,  97,  93,  89,  85,  81,  77,  73,  69,  65,  61,  57,  53,  49,  45,  41,  38,  34,  31,  28,  25,  22,  19,  17,  15,  13,  11,   9,   7,   6,   5,   4 ],
      /* slot 146  bucket  9 phase  2  f= 0.14 ph=0.785 sharp */ [   45,  41,  38,  34,  31,  28,  25,  22,  19,  17,  14,  12,  10,   9,   7,   6,   5,   4,   3,   2,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot 147  bucket  9 phase  3  f= 0.14 ph=1.178 sharp */ [    7,   6,   5,   4,   3,   2,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -2,  -3,  -3,  -4,  -6,  -7,  -8, -10, -12 ],
      /* slot 148  bucket  9 phase  4  f= 0.14 ph=1.571 sharp */ [    0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -2,  -3,  -3,  -5,  -6,  -7,  -9, -10, -12, -14, -16, -19, -21, -24, -27, -30, -34, -37, -41, -44, -48, -52, -56 ],
      /* slot 149  bucket  9 phase  5  f= 0.14 ph=1.963 sharp */ [   -7,  -9, -10, -12, -14, -17, -19, -22, -24, -27, -31, -34, -37, -41, -45, -48, -52, -56, -60, -64, -68, -73, -77, -81, -85, -89, -92, -96,-100,-103,-106,-109 ],
      /* slot 150  bucket  9 phase  6  f= 0.14 ph=2.356 sharp */ [  -45, -49, -53, -57, -61, -65, -69, -73, -77, -81, -85, -89, -93, -96,-100,-103,-107,-110,-112,-115,-117,-120,-121,-123,-124,-126,-126,-127,-127,-127,-126,-126 ],
      /* slot 151  bucket  9 phase  7  f= 0.14 ph=2.749 sharp */ [ -100,-104,-107,-110,-113,-115,-118,-120,-122,-123,-125,-126,-126,-127,-127,-127,-126,-126,-125,-123,-122,-120,-118,-116,-113,-110,-107,-104,-101, -97, -94, -90 ],
      /* slot 152  bucket  9 phase  8  f= 0.14 ph=3.142 sharp */ [ -127,-127,-126,-126,-125,-123,-122,-120,-118,-115,-113,-110,-107,-104,-100, -97, -93, -89, -86, -82, -78, -73, -69, -65, -61, -57, -53, -49, -46, -42, -38, -35 ],
      /* slot 153  bucket  9 phase  9  f= 0.14 ph=3.534 sharp */ [ -100, -97, -93, -89, -85, -81, -77, -73, -69, -65, -61, -57, -53, -49, -45, -41, -38, -34, -31, -28, -25, -22, -19, -17, -15, -13, -11,  -9,  -7,  -6,  -5,  -4 ],
      /* slot 154  bucket  9 phase 10  f= 0.14 ph=3.927 sharp */ [  -45, -41, -38, -34, -31, -28, -25, -22, -19, -17, -14, -12, -10,  -9,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot 155  bucket  9 phase 11  f= 0.14 ph=4.320 sharp */ [   -7,  -6,  -5,  -4,  -3,  -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   2,   3,   3,   4,   6,   7,   8,  10,  12 ],
      /* slot 156  bucket  9 phase 12  f= 0.14 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   0,   1,   1,   1,   2,   3,   3,   5,   6,   7,   9,  10,  12,  14,  16,  19,  21,  24,  27,  30,  34,  37,  41,  44,  48,  52,  56 ],
      /* slot 157  bucket  9 phase 13  f= 0.14 ph=5.105 sharp */ [    7,   9,  10,  12,  14,  17,  19,  22,  24,  27,  31,  34,  37,  41,  45,  48,  52,  56,  60,  64,  68,  73,  77,  81,  85,  89,  92,  96, 100, 103, 106, 109 ],
      /* slot 158  bucket  9 phase 14  f= 0.14 ph=5.498 sharp */ [   45,  49,  53,  57,  61,  65,  69,  73,  77,  81,  85,  89,  93,  96, 100, 103, 107, 110, 112, 115, 117, 120, 121, 123, 124, 126, 126, 127, 127, 127, 126, 126 ],
      /* slot 159  bucket  9 phase 15  f= 0.14 ph=5.890 sharp */ [  100, 104, 107, 110, 113, 115, 118, 120, 122, 123, 125, 126, 126, 127, 127, 127, 126, 126, 125, 123, 122, 120, 118, 116, 113, 110, 107, 104, 101,  97,  94,  90 ],
      /* slot 160  bucket 10 phase  0  f= 0.26 ph=0.000 triangle */ [  127, 123, 119, 115, 110, 106, 102,  98,  94,  90,  86,  81,  77,  73,  69,  65,  61,  56,  52,  48,  44,  40,  36,  32,  27,  23,  19,  15,  11,   7,   3,  -2 ],
      /* slot 161  bucket 10 phase  1  f= 0.26 ph=0.393 triangle */ [   95,  91,  87,  83,  79,  75,  70,  66,  62,  58,  54,  50,  45,  41,  37,  33,  29,  25,  21,  16,  12,   8,   4,   0,  -4,  -8, -13, -17, -21, -25, -29, -33 ],
      /* slot 162  bucket 10 phase  2  f= 0.26 ph=0.785 triangle */ [   64,  59,  55,  51,  47,  43,  39,  34,  30,  26,  22,  18,  14,  10,   5,   1,  -3,  -7, -11, -15, -19, -24, -28, -32, -36, -40, -44, -49, -53, -57, -61, -65 ],
      /* slot 163  bucket 10 phase  3  f= 0.26 ph=1.178 triangle */ [   32,  28,  23,  19,  15,  11,   7,   3,  -1,  -6, -10, -14, -18, -22, -26, -30, -35, -39, -43, -47, -51, -55, -60, -64, -68, -72, -76, -80, -84, -89, -93, -97 ],
      /* slot 164  bucket 10 phase  4  f= 0.26 ph=1.571 triangle */ [    0,  -4,  -8, -12, -17, -21, -25, -29, -33, -37, -41, -46, -50, -54, -58, -62, -66, -71, -75, -79, -83, -87, -91, -95,-100,-104,-108,-112,-116,-120,-124,-125 ],
      /* slot 165  bucket 10 phase  5  f= 0.26 ph=1.963 triangle */ [  -32, -36, -40, -44, -48, -52, -57, -61, -65, -69, -73, -77, -82, -86, -90, -94, -98,-102,-106,-111,-115,-119,-123,-127,-123,-119,-114,-110,-106,-102, -98, -94 ],
      /* slot 166  bucket 10 phase  6  f= 0.26 ph=2.356 triangle */ [  -64, -68, -72, -76, -80, -84, -88, -93, -97,-101,-105,-109,-113,-117,-122,-126,-124,-120,-116,-112,-108,-103, -99, -95, -91, -87, -83, -78, -74, -70, -66, -62 ],
      /* slot 167  bucket 10 phase  7  f= 0.26 ph=2.749 triangle */ [  -95, -99,-104,-108,-112,-116,-120,-124,-126,-121,-117,-113,-109,-105,-101, -97, -92, -88, -84, -80, -76, -72, -67, -63, -59, -55, -51, -47, -43, -38, -34, -30 ],
      /* slot 168  bucket 10 phase  8  f= 0.26 ph=3.142 triangle */ [ -127,-123,-119,-115,-110,-106,-102, -98, -94, -90, -86, -81, -77, -73, -69, -65, -61, -56, -52, -48, -44, -40, -36, -32, -27, -23, -19, -15, -11,  -7,  -3,   2 ],
      /* slot 169  bucket 10 phase  9  f= 0.26 ph=3.534 triangle */ [  -95, -91, -87, -83, -79, -75, -70, -66, -62, -58, -54, -50, -45, -41, -37, -33, -29, -25, -21, -16, -12,  -8,  -4,   0,   4,   8,  13,  17,  21,  25,  29,  33 ],
      /* slot 170  bucket 10 phase 10  f= 0.26 ph=3.927 triangle */ [  -64, -59, -55, -51, -47, -43, -39, -34, -30, -26, -22, -18, -14, -10,  -5,  -1,   3,   7,  11,  15,  19,  24,  28,  32,  36,  40,  44,  49,  53,  57,  61,  65 ],
      /* slot 171  bucket 10 phase 11  f= 0.26 ph=4.320 triangle */ [  -32, -28, -23, -19, -15, -11,  -7,  -3,   1,   6,  10,  14,  18,  22,  26,  30,  35,  39,  43,  47,  51,  55,  60,  64,  68,  72,  76,  80,  84,  89,  93,  97 ],
      /* slot 172  bucket 10 phase 12  f= 0.26 ph=4.712 triangle */ [    0,   4,   8,  12,  17,  21,  25,  29,  33,  37,  41,  46,  50,  54,  58,  62,  66,  71,  75,  79,  83,  87,  91,  95, 100, 104, 108, 112, 116, 120, 124, 125 ],
      /* slot 173  bucket 10 phase 13  f= 0.26 ph=5.105 triangle */ [   32,  36,  40,  44,  48,  52,  57,  61,  65,  69,  73,  77,  82,  86,  90,  94,  98, 102, 106, 111, 115, 119, 123, 127, 123, 119, 114, 110, 106, 102,  98,  94 ],
      /* slot 174  bucket 10 phase 14  f= 0.26 ph=5.498 triangle */ [   64,  68,  72,  76,  80,  84,  88,  93,  97, 101, 105, 109, 113, 117, 122, 126, 124, 120, 116, 112, 108, 103,  99,  95,  91,  87,  83,  78,  74,  70,  66,  62 ],
      /* slot 175  bucket 10 phase 15  f= 0.26 ph=5.890 triangle */ [   95,  99, 104, 108, 112, 116, 120, 124, 126, 121, 117, 113, 109, 105, 101,  97,  92,  88,  84,  80,  76,  72,  67,  63,  59,  55,  51,  47,  43,  38,  34,  30 ],
      /* slot 176  bucket 11 phase  0  f= 0.14 ph=0.000 sine */ [  127, 127, 127, 127, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 117, 116, 115, 113, 111, 110, 108, 106, 104, 102, 100,  97,  95,  93,  90,  88,  85,  82 ],
      /* slot 177  bucket 11 phase  1  f= 0.14 ph=0.393 sine */ [  117, 116, 114, 113, 111, 109, 108, 106, 104, 102,  99,  97,  95,  92,  90,  87,  85,  82,  79,  77,  74,  71,  68,  65,  62,  59,  56,  52,  49,  46,  43,  39 ],
      /* slot 178  bucket 11 phase  2  f= 0.14 ph=0.785 sine */ [   90,  87,  85,  82,  79,  76,  74,  71,  68,  65,  62,  58,  55,  52,  49,  46,  42,  39,  36,  32,  29,  25,  22,  18,  15,  11,   8,   4,   1,  -3,  -7, -10 ],
      /* slot 179  bucket 11 phase  3  f= 0.14 ph=1.178 sine */ [   49,  45,  42,  39,  35,  32,  28,  25,  21,  18,  14,  11,   7,   4,   0,  -3,  -7, -10, -14, -17, -21, -24, -28, -31, -35, -38, -41, -45, -48, -51, -55, -58 ],
      /* slot 180  bucket 11 phase  4  f= 0.14 ph=1.571 sine */ [    0,  -4,  -7, -11, -14, -18, -21, -25, -28, -32, -35, -38, -42, -45, -48, -52, -55, -58, -61, -64, -67, -70, -73, -76, -79, -82, -84, -87, -89, -92, -94, -97 ],
      /* slot 181  bucket 11 phase  5  f= 0.14 ph=1.963 sine */ [  -49, -52, -55, -58, -61, -64, -67, -70, -73, -76, -79, -82, -84, -87, -90, -92, -94, -97, -99,-101,-103,-105,-107,-109,-111,-113,-114,-116,-117,-118,-120,-121 ],
      /* slot 182  bucket 11 phase  6  f= 0.14 ph=2.356 sine */ [  -90, -92, -95, -97, -99,-101,-104,-106,-107,-109,-111,-113,-114,-116,-117,-119,-120,-121,-122,-123,-124,-124,-125,-126,-126,-127,-127,-127,-127,-127,-127,-127 ],
      /* slot 183  bucket 11 phase  7  f= 0.14 ph=2.749 sine */ [ -117,-119,-120,-121,-122,-123,-124,-125,-125,-126,-126,-127,-127,-127,-127,-127,-127,-127,-126,-126,-125,-125,-124,-123,-122,-121,-120,-119,-118,-116,-115,-113 ],
      /* slot 184  bucket 11 phase  8  f= 0.14 ph=3.142 sine */ [ -127,-127,-127,-127,-126,-126,-125,-125,-124,-123,-122,-121,-120,-119,-117,-116,-115,-113,-111,-110,-108,-106,-104,-102,-100, -97, -95, -93, -90, -88, -85, -82 ],
      /* slot 185  bucket 11 phase  9  f= 0.14 ph=3.534 sine */ [ -117,-116,-114,-113,-111,-109,-108,-106,-104,-102, -99, -97, -95, -92, -90, -87, -85, -82, -79, -77, -74, -71, -68, -65, -62, -59, -56, -52, -49, -46, -43, -39 ],
      /* slot 186  bucket 11 phase 10  f= 0.14 ph=3.927 sine */ [  -90, -87, -85, -82, -79, -76, -74, -71, -68, -65, -62, -58, -55, -52, -49, -46, -42, -39, -36, -32, -29, -25, -22, -18, -15, -11,  -8,  -4,  -1,   3,   7,  10 ],
      /* slot 187  bucket 11 phase 11  f= 0.14 ph=4.320 sine */ [  -49, -45, -42, -39, -35, -32, -28, -25, -21, -18, -14, -11,  -7,  -4,   0,   3,   7,  10,  14,  17,  21,  24,  28,  31,  35,  38,  41,  45,  48,  51,  55,  58 ],
      /* slot 188  bucket 11 phase 12  f= 0.14 ph=4.712 sine */ [    0,   4,   7,  11,  14,  18,  21,  25,  28,  32,  35,  38,  42,  45,  48,  52,  55,  58,  61,  64,  67,  70,  73,  76,  79,  82,  84,  87,  89,  92,  94,  97 ],
      /* slot 189  bucket 11 phase 13  f= 0.14 ph=5.105 sine */ [   49,  52,  55,  58,  61,  64,  67,  70,  73,  76,  79,  82,  84,  87,  90,  92,  94,  97,  99, 101, 103, 105, 107, 109, 111, 113, 114, 116, 117, 118, 120, 121 ],
      /* slot 190  bucket 11 phase 14  f= 0.14 ph=5.498 sine */ [   90,  92,  95,  97,  99, 101, 104, 106, 107, 109, 111, 113, 114, 116, 117, 119, 120, 121, 122, 123, 124, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127 ],
      /* slot 191  bucket 11 phase 15  f= 0.14 ph=5.890 sine */ [  117, 119, 120, 121, 122, 123, 124, 125, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 118, 116, 115, 113 ],
      /* slot 192  bucket 12 phase  0  f= 1.41 ph=0.000 triangle */ [  127, 105,  82,  60,  37,  15,  -8, -30, -53, -75, -98,-120,-112, -89, -67, -44, -22,   1,  23,  46,  68,  90, 113, 119,  96,  74,  51,  29,   6, -16, -39, -61 ],
      /* slot 193  bucket 12 phase  1  f= 1.41 ph=0.393 triangle */ [   95,  73,  50,  28,   5, -17, -39, -62, -84,-107,-125,-102, -80, -57, -35, -12,  10,  32,  55,  77, 100, 122, 109,  87,  64,  42,  20,  -3, -25, -48, -70, -93 ],
      /* slot 194  bucket 12 phase  2  f= 1.41 ph=0.785 triangle */ [   64,  41,  19,  -4, -26, -49, -71, -94,-116,-115, -93, -71, -48, -26,  -3,  19,  42,  64,  87, 109, 122, 100,  78,  55,  33,  10, -12, -35, -57, -80,-102,-124 ],
      /* slot 195  bucket 12 phase  3  f= 1.41 ph=1.178 triangle */ [   32,   9, -13, -36, -58, -81,-103,-125,-106, -84, -61, -39, -16,   6,  29,  51,  73,  96, 118, 113,  91,  68,  46,  23,   1, -22, -44, -66, -89,-111,-120, -98 ],
      /* slot 196  bucket 12 phase  4  f= 1.41 ph=1.571 triangle */ [    0, -22, -45, -67, -90,-112,-119, -97, -74, -52, -29,  -7,  15,  38,  60,  83, 105, 126, 104,  81,  59,  37,  14,  -8, -31, -53, -76, -98,-121,-111, -88, -66 ],
      /* slot 197  bucket 12 phase  5  f= 1.41 ph=1.963 triangle */ [  -32, -54, -77, -99,-122,-110, -88, -65, -43, -20,   2,  25,  47,  70,  92, 115, 117,  95,  72,  50,  27,   5, -18, -40, -63, -85,-107,-124,-102, -79, -57, -34 ],
      /* slot 198  bucket 12 phase  6  f= 1.41 ph=2.356 triangle */ [  -64, -86,-108,-123,-101, -78, -56, -33, -11,  12,  34,  56,  79, 101, 124, 108,  85,  63,  40,  18,  -5, -27, -49, -72, -94,-117,-115, -92, -70, -47, -25,  -3 ],
      /* slot 199  bucket 12 phase  7  f= 1.41 ph=2.749 triangle */ [  -95,-118,-114, -91, -69, -46, -24,  -2,  21,  43,  66,  88, 111, 121,  98,  76,  54,  31,   9, -14, -36, -59, -81,-104,-126,-105, -83, -61, -38, -16,   7,  29 ],
      /* slot 200  bucket 12 phase  8  f= 1.41 ph=3.142 triangle */ [ -127,-105, -82, -60, -37, -15,   8,  30,  53,  75,  98, 120, 112,  89,  67,  44,  22,  -1, -23, -46, -68, -90,-113,-119, -96, -74, -51, -29,  -6,  16,  39,  61 ],
      /* slot 201  bucket 12 phase  9  f= 1.41 ph=3.534 triangle */ [  -95, -73, -50, -28,  -5,  17,  39,  62,  84, 107, 125, 102,  80,  57,  35,  12, -10, -32, -55, -77,-100,-122,-109, -87, -64, -42, -20,   3,  25,  48,  70,  93 ],
      /* slot 202  bucket 12 phase 10  f= 1.41 ph=3.927 triangle */ [  -64, -41, -19,   4,  26,  49,  71,  94, 116, 115,  93,  71,  48,  26,   3, -19, -42, -64, -87,-109,-122,-100, -78, -55, -33, -10,  12,  35,  57,  80, 102, 124 ],
      /* slot 203  bucket 12 phase 11  f= 1.41 ph=4.320 triangle */ [  -32,  -9,  13,  36,  58,  81, 103, 125, 106,  84,  61,  39,  16,  -6, -29, -51, -73, -96,-118,-113, -91, -68, -46, -23,  -1,  22,  44,  66,  89, 111, 120,  98 ],
      /* slot 204  bucket 12 phase 12  f= 1.41 ph=4.712 triangle */ [    0,  22,  45,  67,  90, 112, 119,  97,  74,  52,  29,   7, -15, -38, -60, -83,-105,-126,-104, -81, -59, -37, -14,   8,  31,  53,  76,  98, 121, 111,  88,  66 ],
      /* slot 205  bucket 12 phase 13  f= 1.41 ph=5.105 triangle */ [   32,  54,  77,  99, 122, 110,  88,  65,  43,  20,  -2, -25, -47, -70, -92,-115,-117, -95, -72, -50, -27,  -5,  18,  40,  63,  85, 107, 124, 102,  79,  57,  34 ],
      /* slot 206  bucket 12 phase 14  f= 1.41 ph=5.498 triangle */ [   64,  86, 108, 123, 101,  78,  56,  33,  11, -12, -34, -56, -79,-101,-124,-108, -85, -63, -40, -18,   5,  27,  49,  72,  94, 117, 115,  92,  70,  47,  25,   3 ],
      /* slot 207  bucket 12 phase 15  f= 1.41 ph=5.890 triangle */ [   95, 118, 114,  91,  69,  46,  24,   2, -21, -43, -66, -88,-111,-121, -98, -76, -54, -31,  -9,  14,  36,  59,  81, 104, 126, 105,  83,  61,  38,  16,  -7, -29 ],
      /* slot 208  bucket 13 phase  0  f= 1.00 ph=0.000 triangle */ [  127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111 ],
      /* slot 209  bucket 13 phase  1  f= 1.00 ph=0.393 triangle */ [   95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111 ],
      /* slot 210  bucket 13 phase  2  f= 1.00 ph=0.785 triangle */ [   64,  48,  32,  16,   0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  63,  79,  95, 111, 127, 111,  95,  79 ],
      /* slot 211  bucket 13 phase  3  f= 1.00 ph=1.178 triangle */ [   32,  16,   0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48 ],
      /* slot 212  bucket 13 phase  4  f= 1.00 ph=1.571 triangle */ [    0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16 ],
      /* slot 213  bucket 13 phase  5  f= 1.00 ph=1.963 triangle */ [  -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16 ],
      /* slot 214  bucket 13 phase  6  f= 1.00 ph=2.356 triangle */ [  -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48 ],
      /* slot 215  bucket 13 phase  7  f= 1.00 ph=2.749 triangle */ [  -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -63, -79 ],
      /* slot 216  bucket 13 phase  8  f= 1.00 ph=3.142 triangle */ [ -127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -64, -79, -95,-111 ],
      /* slot 217  bucket 13 phase  9  f= 1.00 ph=3.534 triangle */ [  -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -63, -79, -95,-111,-127,-111 ],
      /* slot 218  bucket 13 phase 10  f= 1.00 ph=3.927 triangle */ [  -64, -48, -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -63, -79, -95,-111,-127,-111, -95, -79 ],
      /* slot 219  bucket 13 phase 11  f= 1.00 ph=4.320 triangle */ [  -32, -16,   0,  16,  32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48 ],
      /* slot 220  bucket 13 phase 12  f= 1.00 ph=4.712 triangle */ [    0,  16,  32,  48,  63,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -63, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16 ],
      /* slot 221  bucket 13 phase 13  f= 1.00 ph=5.105 triangle */ [   32,  48,  64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -63, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16 ],
      /* slot 222  bucket 13 phase 14  f= 1.00 ph=5.498 triangle */ [   64,  79,  95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -64, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48 ],
      /* slot 223  bucket 13 phase 15  f= 1.00 ph=5.890 triangle */ [   95, 111, 127, 111,  95,  79,  64,  48,  32,  16,   0, -16, -32, -48, -63, -79, -95,-111,-127,-111, -95, -79, -64, -48, -32, -16,   0,  16,  32,  48,  64,  79 ],
      /* slot 224  bucket 14 phase  0  f= 1.41 ph=0.000 sharp */ [  127, 113,  78,  39,  11,   1,   0,  -6, -28, -65,-104,-126,-120, -90, -50, -18,  -2,   0,   3,  19,  53,  92, 121, 125, 102,  63,  26,   5,   0,  -1, -12, -41 ],
      /* slot 225  bucket 14 phase  1  f= 1.41 ph=0.393 sharp */ [  100,  61,  25,   5,   0,  -1, -13, -42, -82,-116,-127,-110, -74, -35,  -9,   0,   0,   8,  31,  69, 107, 126, 118,  86,  46,  16,   2,   0,  -4, -22, -57, -96 ],
      /* slot 226  bucket 14 phase  2  f= 1.41 ph=0.785 sharp */ [   45,  15,   2,   0,  -4, -23, -58, -98,-124,-123, -97, -57, -22,  -4,   0,   2,  15,  46,  86, 118, 126, 107,  70,  32,   8,   0,   0,  -9, -35, -73,-110,-127 ],
      /* slot 227  bucket 14 phase  3  f= 1.41 ph=1.178 sharp */ [    7,   0,  -1, -10, -36, -75,-111,-127,-115, -81, -41, -12,  -1,   0,   5,  26,  62, 101, 125, 122,  93,  53,  20,   3,   0,  -2, -18, -50, -90,-120,-126,-104 ],
      /* slot 228  bucket 14 phase  4  f= 1.41 ph=1.571 sharp */ [    0,  -3, -19, -51, -91,-121,-125,-103, -64, -27,  -6,   0,   1,  12,  40,  79, 114, 127, 112,  77,  38,  11,   1,   0,  -7, -29, -66,-105,-126,-120, -89, -49 ],
      /* slot 229  bucket 14 phase  5  f= 1.41 ph=1.963 sharp */ [   -7, -30, -68,-106,-126,-119, -88, -48, -16,  -2,   0,   3,  21,  55,  95, 123, 124,  99,  60,  24,   5,   0,  -1, -14, -43, -83,-116,-127,-109, -73, -34,  -9 ],
      /* slot 230  bucket 14 phase  6  f= 1.41 ph=2.356 sharp */ [  -45, -85,-117,-127,-108, -71, -33,  -8,   0,   0,   9,  34,  72, 109, 127, 117,  84,  44,  14,   1,   0,  -4, -24, -59, -99,-124,-123, -96, -56, -22,  -4,   0 ],
      /* slot 231  bucket 14 phase  7  f= 1.41 ph=2.749 sharp */ [ -100,-124,-122, -94, -54, -20,  -3,   0,   2,  17,  49,  89, 119, 126, 105,  67,  30,   7,   0,  -1, -10, -37, -76,-112,-127,-114, -80, -40, -12,  -1,   0,   6 ],
      /* slot 232  bucket 14 phase  8  f= 1.41 ph=3.142 sharp */ [ -127,-113, -78, -39, -11,  -1,   0,   6,  28,  65, 104, 126, 120,  90,  50,  18,   2,   0,  -3, -19, -53, -92,-121,-125,-102, -63, -26,  -5,   0,   1,  12,  41 ],
      /* slot 233  bucket 14 phase  9  f= 1.41 ph=3.534 sharp */ [ -100, -61, -25,  -5,   0,   1,  13,  42,  82, 116, 127, 110,  74,  35,   9,   0,   0,  -8, -31, -69,-107,-126,-118, -86, -46, -16,  -2,   0,   4,  22,  57,  96 ],
      /* slot 234  bucket 14 phase 10  f= 1.41 ph=3.927 sharp */ [  -45, -15,  -2,   0,   4,  23,  58,  98, 124, 123,  97,  57,  22,   4,   0,  -2, -15, -46, -86,-118,-126,-107, -70, -32,  -8,   0,   0,   9,  35,  73, 110, 127 ],
      /* slot 235  bucket 14 phase 11  f= 1.41 ph=4.320 sharp */ [   -7,   0,   1,  10,  36,  75, 111, 127, 115,  81,  41,  12,   1,   0,  -5, -26, -62,-101,-125,-122, -93, -53, -20,  -3,   0,   2,  18,  50,  90, 120, 126, 104 ],
      /* slot 236  bucket 14 phase 12  f= 1.41 ph=4.712 sharp */ [    0,   3,  19,  51,  91, 121, 125, 103,  64,  27,   6,   0,  -1, -12, -40, -79,-114,-127,-112, -77, -38, -11,  -1,   0,   7,  29,  66, 105, 126, 120,  89,  49 ],
      /* slot 237  bucket 14 phase 13  f= 1.41 ph=5.105 sharp */ [    7,  30,  68, 106, 126, 119,  88,  48,  16,   2,   0,  -3, -21, -55, -95,-123,-124, -99, -60, -24,  -5,   0,   1,  14,  43,  83, 116, 127, 109,  73,  34,   9 ],
      /* slot 238  bucket 14 phase 14  f= 1.41 ph=5.498 sharp */ [   45,  85, 117, 127, 108,  71,  33,   8,   0,   0,  -9, -34, -72,-109,-127,-117, -84, -44, -14,  -1,   0,   4,  24,  59,  99, 124, 123,  96,  56,  22,   4,   0 ],
      /* slot 239  bucket 14 phase 15  f= 1.41 ph=5.890 sharp */ [  100, 124, 122,  94,  54,  20,   3,   0,  -2, -17, -49, -89,-119,-126,-105, -67, -30,  -7,   0,   1,  10,  37,  76, 112, 127, 114,  80,  40,  12,   1,   0,  -6 ],
      /* slot 240  bucket 15 phase  0  f= 0.10 ph=0.000 triangle */ [  127, 125, 124, 122, 121, 119, 117, 116, 114, 112, 111, 109, 108, 106, 104, 103, 101,  99,  98,  96,  95,  93,  91,  90,  88,  86,  85,  83,  82,  80,  78,  77 ],
      /* slot 241  bucket 15 phase  1  f= 0.10 ph=0.393 triangle */ [   95,  94,  92,  90,  89,  87,  86,  84,  82,  81,  79,  77,  76,  74,  73,  71,  69,  68,  66,  64,  63,  61,  60,  58,  56,  55,  53,  51,  50,  48,  47,  45 ],
      /* slot 242  bucket 15 phase  2  f= 0.10 ph=0.785 triangle */ [   64,  62,  60,  59,  57,  55,  54,  52,  51,  49,  47,  46,  44,  42,  41,  39,  38,  36,  34,  33,  31,  29,  28,  26,  25,  23,  21,  20,  18,  16,  15,  13 ],
      /* slot 243  bucket 15 phase  3  f= 0.10 ph=1.178 triangle */ [   32,  30,  29,  27,  25,  24,  22,  20,  19,  17,  16,  14,  12,  11,   9,   7,   6,   4,   3,   1,  -1,  -2,  -4,  -6,  -7,  -9, -10, -12, -14, -15, -17, -19 ],
      /* slot 244  bucket 15 phase  4  f= 0.10 ph=1.571 triangle */ [    0,  -2,  -3,  -5,  -6,  -8, -10, -11, -13, -15, -16, -18, -19, -21, -23, -24, -26, -28, -29, -31, -32, -34, -36, -37, -39, -41, -42, -44, -45, -47, -49, -50 ],
      /* slot 245  bucket 15 phase  5  f= 0.10 ph=1.963 triangle */ [  -32, -33, -35, -37, -38, -40, -41, -43, -45, -46, -48, -50, -51, -53, -54, -56, -58, -59, -61, -63, -64, -66, -67, -69, -71, -72, -74, -76, -77, -79, -80, -82 ],
      /* slot 246  bucket 15 phase  6  f= 0.10 ph=2.356 triangle */ [  -64, -65, -67, -68, -70, -72, -73, -75, -76, -78, -80, -81, -83, -85, -86, -88, -89, -91, -93, -94, -96, -98, -99,-101,-102,-104,-106,-107,-109,-111,-112,-114 ],
      /* slot 247  bucket 15 phase  7  f= 0.10 ph=2.749 triangle */ [  -95, -97, -98,-100,-102,-103,-105,-107,-108,-110,-111,-113,-115,-116,-118,-120,-121,-123,-124,-126,-126,-125,-123,-121,-120,-118,-117,-115,-113,-112,-110,-108 ],
      /* slot 248  bucket 15 phase  8  f= 0.10 ph=3.142 triangle */ [ -127,-125,-124,-122,-121,-119,-117,-116,-114,-112,-111,-109,-108,-106,-104,-103,-101, -99, -98, -96, -95, -93, -91, -90, -88, -86, -85, -83, -82, -80, -78, -77 ],
      /* slot 249  bucket 15 phase  9  f= 0.10 ph=3.534 triangle */ [  -95, -94, -92, -90, -89, -87, -86, -84, -82, -81, -79, -77, -76, -74, -73, -71, -69, -68, -66, -64, -63, -61, -60, -58, -56, -55, -53, -51, -50, -48, -47, -45 ],
      /* slot 250  bucket 15 phase 10  f= 0.10 ph=3.927 triangle */ [  -64, -62, -60, -59, -57, -55, -54, -52, -51, -49, -47, -46, -44, -42, -41, -39, -38, -36, -34, -33, -31, -29, -28, -26, -25, -23, -21, -20, -18, -16, -15, -13 ],
      /* slot 251  bucket 15 phase 11  f= 0.10 ph=4.320 triangle */ [  -32, -30, -29, -27, -25, -24, -22, -20, -19, -17, -16, -14, -12, -11,  -9,  -7,  -6,  -4,  -3,  -1,   1,   2,   4,   6,   7,   9,  10,  12,  14,  15,  17,  19 ],
      /* slot 252  bucket 15 phase 12  f= 0.10 ph=4.712 triangle */ [    0,   2,   3,   5,   6,   8,  10,  11,  13,  15,  16,  18,  19,  21,  23,  24,  26,  28,  29,  31,  32,  34,  36,  37,  39,  41,  42,  44,  45,  47,  49,  50 ],
      /* slot 253  bucket 15 phase 13  f= 0.10 ph=5.105 triangle */ [   32,  33,  35,  37,  38,  40,  41,  43,  45,  46,  48,  50,  51,  53,  54,  56,  58,  59,  61,  63,  64,  66,  67,  69,  71,  72,  74,  76,  77,  79,  80,  82 ],
      /* slot 254  bucket 15 phase 14  f= 0.10 ph=5.498 triangle */ [   64,  65,  67,  68,  70,  72,  73,  75,  76,  78,  80,  81,  83,  85,  86,  88,  89,  91,  93,  94,  96,  98,  99, 101, 102, 104, 106, 107, 109, 111, 112, 114 ],
      /* slot 255  bucket 15 phase 15  f= 0.10 ph=5.890 triangle */ [   95,  97,  98, 100, 102, 103, 105, 107, 108, 110, 111, 113, 115, 116, 118, 120, 121, 123, 124, 126, 126, 125, 123, 121, 120, 118, 117, 115, 113, 112, 110, 108 ]

    ];

    /// CURVE_TABLE_K — built from the 4 picks of the iterative curve
    /// selection test (R1[0]:f=0.42 sharp + R2[0..2] data-derived) plus
    /// their negations. 8 buckets × 16 phases (cyclic stride-2 lane shifts)
    /// = 128 slots. Buckets 0..3 are the 4 unique shapes; buckets 4..7 are
    /// the same shapes negated (mirror polarity).
    pub const CURVE_TABLE_K: [[i8; 32]; 128] = __CURVE_TABLE_K_RAW;

    #[rustfmt::skip]
    pub const SCALE_TABLE_BITS_K: [u16; 32] = [
    
        0x01D0,0x0F3A,0x10B7,0x11BB,0x12BE,0x13B5,0x144C,0x14B4,
        0x1514,0x1570,0x15C8,0x161C,0x1670,0x16C4,0x1719,0x1770,
        0x17CB,0x1815,0x1846,0x187B,0x18B3,0x18F1,0x1935,0x1981,
        0x19D7,0x1A3B,0x1AB3,0x1B49,0x1C0D,0x1CAE,0x1DB6,0x2008
    
    ];

    #[rustfmt::skip]
    const CENTROID_TABLE_BITS_K_OLD: [[u16; 8]; 32] = [

      /* scale_idx =  0  (8253 src blocks) */ [ 0xA86D,0x9D30,0x99F0,0x9380,0x1400,0x1980,0x1CD0,0x26B2 ],
      /* scale_idx =  1  (44790 src blocks) */ [ 0xACA1,0xA2E4,0x9FE0,0x9B20,0x11C0,0x1D70,0x21D0,0x2CE8 ],
      /* scale_idx =  2  (37123 src blocks) */ [ 0xAE1E,0xA452,0xA114,0x9C40,0x1680,0x1FF0,0x23D0,0x2D1F ],
      /* scale_idx =  3  (35842 src blocks) */ [ 0xAF54,0xA4F6,0xA180,0x9C20,0x1930,0x20F8,0x24B0,0x2FB0 ],
      /* scale_idx =  4  (36114 src blocks) */ [ 0xB0C9,0xA5DC,0xA248,0x9C58,0x1AC0,0x21D0,0x25A6,0x3012 ],
      /* scale_idx =  5  (35808 src blocks) */ [ 0xB258,0xA698,0xA2DC,0x9C70,0x1C68,0x22E4,0x26A2,0x3028 ],
      /* scale_idx =  6  (36158 src blocks) */ [ 0xB152,0xA780,0xA3A4,0x9CA0,0x1D70,0x23E0,0x2796,0x3050 ],
      /* scale_idx =  7  (36233 src blocks) */ [ 0xB167,0xA830,0xA43A,0x9D10,0x1DC8,0x2464,0x2828,0x33A3 ],
      /* scale_idx =  8  (36056 src blocks) */ [ 0xB194,0xA874,0xA494,0x9DA0,0x1E70,0x24BA,0x2879,0x3283 ],
      /* scale_idx =  9  (36005 src blocks) */ [ 0xB15D,0xA8D2,0xA4EC,0x9DE8,0x1E98,0x2516,0x28D3,0x31D8 ],
      /* scale_idx = 10  (36228 src blocks) */ [ 0xB1F8,0xA90E,0xA544,0x9E80,0x1F18,0x2554,0x2910,0x31F4 ],
      /* scale_idx = 11  (36307 src blocks) */ [ 0xB220,0xA96C,0xA5AE,0x9F38,0x1F30,0x259E,0x295E,0x3220 ],
      /* scale_idx = 12  (35970 src blocks) */ [ 0xB2F0,0xA9B8,0xA610,0x9FD0,0x1FB8,0x2604,0x299D,0x3281 ],
      /* scale_idx = 13  (36282 src blocks) */ [ 0xB2EA,0xA9F4,0xA66A,0x9FF0,0x1FF0,0x2634,0x29D9,0x334A ],
      /* scale_idx = 14  (36018 src blocks) */ [ 0xB2CF,0xAA13,0xA682,0xA03C,0x200C,0x2652,0x2A07,0x333F ],
      /* scale_idx = 15  (35878 src blocks) */ [ 0xB441,0xAA6D,0xA6FE,0xA0A8,0x1FE0,0x2680,0x2A2A,0x33A0 ],
      /* scale_idx = 16  (36336 src blocks) */ [ 0xB452,0xAA9F,0xA724,0xA0B4,0x2044,0x26D6,0x2A82,0x3358 ],
      /* scale_idx = 17  (36077 src blocks) */ [ 0xB477,0xAAE4,0xA770,0xA10C,0x1FD0,0x26F6,0x2AB7,0x32C5 ],
      /* scale_idx = 18  (36093 src blocks) */ [ 0xB4D9,0xAB0F,0xA7AC,0xA138,0x2054,0x2740,0x2AEA,0x3368 ],
      /* scale_idx = 19  (36184 src blocks) */ [ 0xB532,0xAB65,0xA7FA,0xA134,0x20A4,0x2784,0x2B21,0x34AB ],
      /* scale_idx = 20  (36038 src blocks) */ [ 0xB4D7,0xABC2,0xA816,0xA160,0x20B4,0x27D0,0x2B7C,0x3495 ],
      /* scale_idx = 21  (36245 src blocks) */ [ 0xB5E6,0xAC04,0xA830,0xA158,0x212C,0x2826,0x2BDD,0x342C ],
      /* scale_idx = 22  (36260 src blocks) */ [ 0xB53A,0xAC2E,0xA86E,0xA1F4,0x2118,0x283B,0x2C13,0x34A4 ],
      /* scale_idx = 23  (35892 src blocks) */ [ 0xB510,0xAC6B,0xA8B6,0xA240,0x2170,0x2879,0x2C38,0x34C9 ],
      /* scale_idx = 24  (36229 src blocks) */ [ 0xB598,0xACA8,0xA8EA,0xA2B4,0x2198,0x28AF,0x2C7F,0x34ED ],
      /* scale_idx = 25  (36251 src blocks) */ [ 0xB593,0xAD19,0xA95A,0xA364,0x2198,0x28D8,0x2CB2,0x3601 ],
      /* scale_idx = 26  (36332 src blocks) */ [ 0xB591,0xAD7D,0xA9F2,0xA42E,0x2210,0x291F,0x2D0D,0x35E3 ],
      /* scale_idx = 27  (36591 src blocks) */ [ 0xB675,0xADD5,0xAA53,0xA494,0x21F8,0x297D,0x2D75,0x360A ],
      /* scale_idx = 28  (36833 src blocks) */ [ 0xB6DC,0xAE72,0xAAF1,0xA4CC,0x2364,0x2A71,0x2E3A,0x3639 ],
      /* scale_idx = 29  (35066 src blocks) */ [ 0xB722,0xB00D,0xAC2E,0xA54A,0x254C,0x2C43,0x3012,0x3746 ],
      /* scale_idx = 30  (47819 src blocks) */ [ 0xB6D2,0xB278,0xB0AF,0xA9D6,0x2ABE,0x3096,0x3263,0x3649 ],
      /* scale_idx = 31  (4897 src blocks) */ [ 0xB55E,0xAE9C,0xABF9,0xA2AC,0x28AA,0x2CBE,0x2F2C,0x33CC ]

    ];

    /// CENTROID_TABLE_BITS_K with 16 entries per scale row. Currently built by
    /// duplicating the calibrated 8-entry table into adjacent slot pairs
    /// (j → 2j, 2j+1); a full 16-entry recalibration is a separate measurement
    /// task. The kernel reads `[16]` rows so the type must match.
    pub const CENTROID_TABLE_BITS_K: [[u16; 16]; 32] = {
        let mut out = [[0u16; 16]; 32];
        let mut s = 0;
        while s < 32 {
            let mut j = 0;
            while j < 8 {
                out[s][j * 2] = CENTROID_TABLE_BITS_K_OLD[s][j];
                out[s][j * 2 + 1] = CENTROID_TABLE_BITS_K_OLD[s][j];
                j += 1;
            }
            s += 1;
        }
        out
    };

    #[rustfmt::skip]
    pub const PEAK_CURVE_INDICES_K: [u8; 256] = [
    
          0,  1,  2,  8,  9, 10, 16, 17, 18, 24, 25, 26, 32, 40, 48, 49,
         50, 56, 57, 58, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 98, 99,
        104,105,106,107,112,113,120,121,128,129,130,131,136,137,138,139,
        144,145,146,152,153,154,160,161,168,169,176,177,178,184,185,186,
        192,200,208,216,224,232,240,241,242,243,248,249,250,251, 39, 47,
        215,223,230,238, 38, 46,214,222, 37, 45, 71, 79,119,127,213,221,
         87, 95,195,203,227,235, 23, 31, 36, 44,167,175,212,220,  7, 15,
         35, 43,193,201,211,219,225,233,118,126,183,191, 34, 42,210,218,
         70, 78,151,159, 33, 41, 55, 63,198,206,209,217, 86, 94,166,174,
         22, 30,117,125,196,204,228,236,103,111,247,255, 69, 77,  6, 14,
         85, 93,116,124,165,173,199,207,231,239,182,190, 21, 29, 68, 76,
        150,158,197,205,229,237, 54, 62,135,143,115,123, 84, 92,  3,  4,
          5, 11, 12, 13, 19, 20, 27, 28, 51, 52, 53, 59, 60, 61, 66, 67,
         74, 75, 82, 83, 90, 91,100,101,102,108,109,110,114,122,132,133,
        134,140,141,142,147,148,149,155,156,157,162,163,164,170,171,172,
        179,180,181,187,188,189,194,202,226,234,244,245,246,252,253,254
    
    ];

    #[rustfmt::skip]
    pub const PEAK_BIN_OFFSETS_K: [u16; 33] = [
    
          0, 78, 78, 82, 84, 88, 88, 96,102,110,110,120,124,128,132,140,144,144,152,154,156,158,160,162,166,170,172,176,182,186,188,190,256
    
    ];

    #[rustfmt::skip]
    #[allow(dead_code)]
    const CURVE_TABLE_V_FULL: [[i8; 32]; 256] = [
    
      /* slot   0  bucket  0 phase  0  f= 0.18 ph=0.000 sharp */ [  127, 127, 126, 125, 123, 121, 118, 116, 112, 108, 104, 100,  96,  91,  86,  81,  76,  70,  65,  60,  55,  50,  45,  40,  36,  31,  27,  24,  20,  17,  14,  11 ],
      /* slot   1  bucket  0 phase  1  f= 0.18 ph=0.393 sharp */ [  100,  96,  91,  86,  81,  76,  70,  65,  60,  55,  50,  45,  40,  36,  31,  27,  24,  20,  17,  14,  11,   9,   7,   5,   4,   3,   2,   1,   1,   0,   0,   0 ],
      /* slot   2  bucket  0 phase  2  f= 0.18 ph=0.785 sharp */ [   45,  40,  36,  31,  27,  24,  20,  17,  14,  11,   9,   7,   5,   4,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -4 ],
      /* slot   3  bucket  0 phase  3  f= 0.18 ph=1.178 sharp */ [    7,   5,   4,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -4,  -5,  -7,  -9, -11, -14, -17, -20, -24, -27, -31, -36 ],
      /* slot   4  bucket  0 phase  4  f= 0.18 ph=1.571 sharp */ [    0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -4,  -5,  -7,  -9, -11, -14, -17, -20, -24, -27, -31, -36, -40, -45, -50, -55, -60, -65, -70, -76, -81, -86, -91 ],
      /* slot   5  bucket  0 phase  5  f= 0.18 ph=1.963 sharp */ [   -7,  -9, -11, -14, -17, -20, -24, -27, -31, -36, -40, -45, -50, -55, -60, -65, -70, -76, -81, -86, -91, -96,-100,-104,-108,-112,-116,-118,-121,-123,-125,-126 ],
      /* slot   6  bucket  0 phase  6  f= 0.18 ph=2.356 sharp */ [  -45, -50, -55, -60, -65, -70, -76, -81, -86, -91, -96,-100,-104,-108,-112,-116,-118,-121,-123,-125,-126,-127,-127,-127,-126,-125,-123,-121,-118,-116,-112,-108 ],
      /* slot   7  bucket  0 phase  7  f= 0.18 ph=2.749 sharp */ [ -100,-104,-108,-112,-116,-118,-121,-123,-125,-126,-127,-127,-127,-126,-125,-123,-121,-118,-116,-112,-108,-104,-100, -96, -91, -86, -81, -76, -70, -65, -60, -55 ],
      /* slot   8  bucket  0 phase  8  f= 0.18 ph=3.142 sharp */ [ -127,-127,-126,-125,-123,-121,-118,-116,-112,-108,-104,-100, -96, -91, -86, -81, -76, -70, -65, -60, -55, -50, -45, -40, -36, -31, -27, -24, -20, -17, -14, -11 ],
      /* slot   9  bucket  0 phase  9  f= 0.18 ph=3.534 sharp */ [ -100, -96, -91, -86, -81, -76, -70, -65, -60, -55, -50, -45, -40, -36, -31, -27, -24, -20, -17, -14, -11,  -9,  -7,  -5,  -4,  -3,  -2,  -1,  -1,   0,   0,   0 ],
      /* slot  10  bucket  0 phase 10  f= 0.18 ph=3.927 sharp */ [  -45, -40, -36, -31, -27, -24, -20, -17, -14, -11,  -9,  -7,  -5,  -4,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   4 ],
      /* slot  11  bucket  0 phase 11  f= 0.18 ph=4.320 sharp */ [   -7,  -5,  -4,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   4,   5,   7,   9,  11,  14,  17,  20,  24,  27,  31,  36 ],
      /* slot  12  bucket  0 phase 12  f= 0.18 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   1,   1,   2,   3,   4,   5,   7,   9,  11,  14,  17,  20,  24,  27,  31,  36,  40,  45,  50,  55,  60,  65,  70,  76,  81,  86,  91 ],
      /* slot  13  bucket  0 phase 13  f= 0.18 ph=5.105 sharp */ [    7,   9,  11,  14,  17,  20,  24,  27,  31,  36,  40,  45,  50,  55,  60,  65,  70,  76,  81,  86,  91,  96, 100, 104, 108, 112, 116, 118, 121, 123, 125, 126 ],
      /* slot  14  bucket  0 phase 14  f= 0.18 ph=5.498 sharp */ [   45,  50,  55,  60,  65,  70,  76,  81,  86,  91,  96, 100, 104, 108, 112, 116, 118, 121, 123, 125, 126, 127, 127, 127, 126, 125, 123, 121, 118, 116, 112, 108 ],
      /* slot  15  bucket  0 phase 15  f= 0.18 ph=5.890 sharp */ [  100, 104, 108, 112, 116, 118, 121, 123, 125, 126, 127, 127, 127, 126, 125, 123, 121, 118, 116, 112, 108, 104, 100,  96,  91,  86,  81,  76,  70,  65,  60,  55 ],
      /* slot  16  bucket  1 phase  0  f= 0.14 ph=0.000 sharp */ [  127, 127, 126, 126, 125, 123, 122, 120, 118, 115, 113, 110, 107, 104, 100,  97,  93,  89,  86,  82,  78,  73,  69,  65,  61,  57,  53,  49,  46,  42,  38,  35 ],
      /* slot  17  bucket  1 phase  1  f= 0.14 ph=0.393 sharp */ [  100,  97,  93,  89,  85,  81,  77,  73,  69,  65,  61,  57,  53,  49,  45,  41,  38,  34,  31,  28,  25,  22,  19,  17,  15,  13,  11,   9,   7,   6,   5,   4 ],
      /* slot  18  bucket  1 phase  2  f= 0.14 ph=0.785 sharp */ [   45,  41,  38,  34,  31,  28,  25,  22,  19,  17,  14,  12,  10,   9,   7,   6,   5,   4,   3,   2,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot  19  bucket  1 phase  3  f= 0.14 ph=1.178 sharp */ [    7,   6,   5,   4,   3,   2,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -2,  -3,  -3,  -4,  -6,  -7,  -8, -10, -12 ],
      /* slot  20  bucket  1 phase  4  f= 0.14 ph=1.571 sharp */ [    0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -2,  -3,  -3,  -5,  -6,  -7,  -9, -10, -12, -14, -16, -19, -21, -24, -27, -30, -34, -37, -41, -44, -48, -52, -56 ],
      /* slot  21  bucket  1 phase  5  f= 0.14 ph=1.963 sharp */ [   -7,  -9, -10, -12, -14, -17, -19, -22, -24, -27, -31, -34, -37, -41, -45, -48, -52, -56, -60, -64, -68, -73, -77, -81, -85, -89, -92, -96,-100,-103,-106,-109 ],
      /* slot  22  bucket  1 phase  6  f= 0.14 ph=2.356 sharp */ [  -45, -49, -53, -57, -61, -65, -69, -73, -77, -81, -85, -89, -93, -96,-100,-103,-107,-110,-112,-115,-117,-120,-121,-123,-124,-126,-126,-127,-127,-127,-126,-126 ],
      /* slot  23  bucket  1 phase  7  f= 0.14 ph=2.749 sharp */ [ -100,-104,-107,-110,-113,-115,-118,-120,-122,-123,-125,-126,-126,-127,-127,-127,-126,-126,-125,-123,-122,-120,-118,-116,-113,-110,-107,-104,-101, -97, -94, -90 ],
      /* slot  24  bucket  1 phase  8  f= 0.14 ph=3.142 sharp */ [ -127,-127,-126,-126,-125,-123,-122,-120,-118,-115,-113,-110,-107,-104,-100, -97, -93, -89, -86, -82, -78, -73, -69, -65, -61, -57, -53, -49, -46, -42, -38, -35 ],
      /* slot  25  bucket  1 phase  9  f= 0.14 ph=3.534 sharp */ [ -100, -97, -93, -89, -85, -81, -77, -73, -69, -65, -61, -57, -53, -49, -45, -41, -38, -34, -31, -28, -25, -22, -19, -17, -15, -13, -11,  -9,  -7,  -6,  -5,  -4 ],
      /* slot  26  bucket  1 phase 10  f= 0.14 ph=3.927 sharp */ [  -45, -41, -38, -34, -31, -28, -25, -22, -19, -17, -14, -12, -10,  -9,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot  27  bucket  1 phase 11  f= 0.14 ph=4.320 sharp */ [   -7,  -6,  -5,  -4,  -3,  -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   2,   3,   3,   4,   6,   7,   8,  10,  12 ],
      /* slot  28  bucket  1 phase 12  f= 0.14 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   0,   1,   1,   1,   2,   3,   3,   5,   6,   7,   9,  10,  12,  14,  16,  19,  21,  24,  27,  30,  34,  37,  41,  44,  48,  52,  56 ],
      /* slot  29  bucket  1 phase 13  f= 0.14 ph=5.105 sharp */ [    7,   9,  10,  12,  14,  17,  19,  22,  24,  27,  31,  34,  37,  41,  45,  48,  52,  56,  60,  64,  68,  73,  77,  81,  85,  89,  92,  96, 100, 103, 106, 109 ],
      /* slot  30  bucket  1 phase 14  f= 0.14 ph=5.498 sharp */ [   45,  49,  53,  57,  61,  65,  69,  73,  77,  81,  85,  89,  93,  96, 100, 103, 107, 110, 112, 115, 117, 120, 121, 123, 124, 126, 126, 127, 127, 127, 126, 126 ],
      /* slot  31  bucket  1 phase 15  f= 0.14 ph=5.890 sharp */ [  100, 104, 107, 110, 113, 115, 118, 120, 122, 123, 125, 126, 126, 127, 127, 127, 126, 126, 125, 123, 122, 120, 118, 116, 113, 110, 107, 104, 101,  97,  94,  90 ],
      /* slot  32  bucket  2 phase  0  f= 0.06 ph=0.000 sharp */ [  127, 127, 127, 127, 127, 126, 126, 126, 125, 125, 124, 124, 123, 122, 121, 121, 120, 119, 118, 117, 116, 115, 114, 113, 111, 110, 109, 107, 106, 105, 103, 102 ],
      /* slot  33  bucket  2 phase  1  f= 0.06 ph=0.393 sharp */ [  100,  99,  97,  95,  94,  92,  91,  89,  87,  85,  84,  82,  80,  78,  77,  75,  73,  71,  69,  68,  66,  64,  62,  60,  59,  57,  55,  53,  52,  50,  48,  47 ],
      /* slot  34  bucket  2 phase  2  f= 0.06 ph=0.785 sharp */ [   45,  43,  42,  40,  38,  37,  35,  34,  32,  31,  30,  28,  27,  26,  24,  23,  22,  21,  19,  18,  17,  16,  15,  14,  13,  12,  12,  11,  10,   9,   8,   8 ],
      /* slot  35  bucket  2 phase  3  f= 0.06 ph=1.178 sharp */ [    7,   7,   6,   5,   5,   4,   4,   3,   3,   3,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot  36  bucket  2 phase  4  f= 0.06 ph=1.571 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -2,  -2,  -2,  -2,  -3,  -3,  -3,  -4,  -4,  -5,  -5,  -6,  -7 ],
      /* slot  37  bucket  2 phase  5  f= 0.06 ph=1.963 sharp */ [   -7,  -8,  -8,  -9, -10, -11, -12, -12, -13, -14, -15, -16, -17, -18, -19, -21, -22, -23, -24, -26, -27, -28, -30, -31, -32, -34, -35, -37, -38, -40, -42, -43 ],
      /* slot  38  bucket  2 phase  6  f= 0.06 ph=2.356 sharp */ [  -45, -47, -48, -50, -52, -53, -55, -57, -59, -60, -62, -64, -66, -68, -69, -71, -73, -75, -77, -78, -80, -82, -84, -85, -87, -89, -91, -92, -94, -95, -97, -99 ],
      /* slot  39  bucket  2 phase  7  f= 0.06 ph=2.749 sharp */ [ -100,-102,-103,-105,-106,-107,-109,-110,-111,-113,-114,-115,-116,-117,-118,-119,-120,-121,-121,-122,-123,-124,-124,-125,-125,-126,-126,-126,-127,-127,-127,-127 ],
      /* slot  40  bucket  2 phase  8  f= 0.06 ph=3.142 sharp */ [ -127,-127,-127,-127,-127,-126,-126,-126,-125,-125,-124,-124,-123,-122,-121,-121,-120,-119,-118,-117,-116,-115,-114,-113,-111,-110,-109,-107,-106,-105,-103,-102 ],
      /* slot  41  bucket  2 phase  9  f= 0.06 ph=3.534 sharp */ [ -100, -99, -97, -95, -94, -92, -91, -89, -87, -85, -84, -82, -80, -78, -77, -75, -73, -71, -69, -68, -66, -64, -62, -60, -59, -57, -55, -53, -52, -50, -48, -47 ],
      /* slot  42  bucket  2 phase 10  f= 0.06 ph=3.927 sharp */ [  -45, -43, -42, -40, -38, -37, -35, -34, -32, -31, -30, -28, -27, -26, -24, -23, -22, -21, -19, -18, -17, -16, -15, -14, -13, -12, -12, -11, -10,  -9,  -8,  -8 ],
      /* slot  43  bucket  2 phase 11  f= 0.06 ph=4.320 sharp */ [   -7,  -7,  -6,  -5,  -5,  -4,  -4,  -3,  -3,  -3,  -2,  -2,  -2,  -2,  -1,  -1,  -1,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot  44  bucket  2 phase 12  f= 0.06 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   3,   3,   3,   4,   4,   5,   5,   6,   7 ],
      /* slot  45  bucket  2 phase 13  f= 0.06 ph=5.105 sharp */ [    7,   8,   8,   9,  10,  11,  12,  12,  13,  14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  26,  27,  28,  30,  31,  32,  34,  35,  37,  38,  40,  42,  43 ],
      /* slot  46  bucket  2 phase 14  f= 0.06 ph=5.498 sharp */ [   45,  47,  48,  50,  52,  53,  55,  57,  59,  60,  62,  64,  66,  68,  69,  71,  73,  75,  77,  78,  80,  82,  84,  85,  87,  89,  91,  92,  94,  95,  97,  99 ],
      /* slot  47  bucket  2 phase 15  f= 0.06 ph=5.890 sharp */ [  100, 102, 103, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 121, 122, 123, 124, 124, 125, 125, 126, 126, 126, 127, 127, 127, 127 ],
      /* slot  48  bucket  3 phase  0  f= 0.10 ph=0.000 sharp */ [  127, 127, 127, 126, 126, 125, 124, 123, 122, 121, 119, 118, 116, 115, 113, 111, 108, 106, 104, 102,  99,  96,  94,  91,  88,  86,  83,  80,  77,  74,  71,  68 ],
      /* slot  49  bucket  3 phase  1  f= 0.10 ph=0.393 sharp */ [  100,  98,  95,  92,  90,  87,  84,  81,  78,  75,  72,  69,  66,  64,  61,  58,  55,  52,  49,  46,  44,  41,  38,  36,  33,  31,  29,  27,  24,  22,  20,  19 ],
      /* slot  50  bucket  3 phase  2  f= 0.10 ph=0.785 sharp */ [   45,  42,  40,  37,  35,  32,  30,  28,  25,  23,  21,  19,  18,  16,  14,  13,  11,  10,   9,   8,   7,   6,   5,   4,   3,   3,   2,   2,   1,   1,   1,   1 ],
      /* slot  51  bucket  3 phase  3  f= 0.10 ph=1.178 sharp */ [    7,   6,   5,   4,   4,   3,   2,   2,   2,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -2 ],
      /* slot  52  bucket  3 phase  4  f= 0.10 ph=1.571 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -1,  -1,  -2,  -2,  -3,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10, -11, -13, -14, -16, -17, -19, -21, -23, -25 ],
      /* slot  53  bucket  3 phase  5  f= 0.10 ph=1.963 sharp */ [   -7,  -8,  -9, -11, -12, -13, -15, -17, -18, -20, -22, -24, -26, -29, -31, -33, -36, -38, -41, -43, -46, -49, -52, -55, -57, -60, -63, -66, -69, -72, -75, -78 ],
      /* slot  54  bucket  3 phase  6  f= 0.10 ph=2.356 sharp */ [  -45, -48, -50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -79, -82, -85, -88, -91, -94, -96, -99,-101,-104,-106,-108,-110,-112,-114,-116,-118,-119,-121,-122 ],
      /* slot  55  bucket  3 phase  7  f= 0.10 ph=2.749 sharp */ [ -100,-103,-105,-107,-109,-112,-113,-115,-117,-119,-120,-121,-123,-124,-125,-125,-126,-126,-127,-127,-127,-127,-127,-126,-125,-125,-124,-123,-122,-120,-119,-117 ],
      /* slot  56  bucket  3 phase  8  f= 0.10 ph=3.142 sharp */ [ -127,-127,-127,-126,-126,-125,-124,-123,-122,-121,-119,-118,-116,-115,-113,-111,-108,-106,-104,-102, -99, -96, -94, -91, -88, -86, -83, -80, -77, -74, -71, -68 ],
      /* slot  57  bucket  3 phase  9  f= 0.10 ph=3.534 sharp */ [ -100, -98, -95, -92, -90, -87, -84, -81, -78, -75, -72, -69, -66, -64, -61, -58, -55, -52, -49, -46, -44, -41, -38, -36, -33, -31, -29, -27, -24, -22, -20, -19 ],
      /* slot  58  bucket  3 phase 10  f= 0.10 ph=3.927 sharp */ [  -45, -42, -40, -37, -35, -32, -30, -28, -25, -23, -21, -19, -18, -16, -14, -13, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -3,  -2,  -2,  -1,  -1,  -1,  -1 ],
      /* slot  59  bucket  3 phase 11  f= 0.10 ph=4.320 sharp */ [   -7,  -6,  -5,  -4,  -4,  -3,  -2,  -2,  -2,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   2 ],
      /* slot  60  bucket  3 phase 12  f= 0.10 ph=4.712 sharp */ [    0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   2,   2,   3,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,  14,  16,  17,  19,  21,  23,  25 ],
      /* slot  61  bucket  3 phase 13  f= 0.10 ph=5.105 sharp */ [    7,   8,   9,  11,  12,  13,  15,  17,  18,  20,  22,  24,  26,  29,  31,  33,  36,  38,  41,  43,  46,  49,  52,  55,  57,  60,  63,  66,  69,  72,  75,  78 ],
      /* slot  62  bucket  3 phase 14  f= 0.10 ph=5.498 sharp */ [   45,  48,  50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  79,  82,  85,  88,  91,  94,  96,  99, 101, 104, 106, 108, 110, 112, 114, 116, 118, 119, 121, 122 ],
      /* slot  63  bucket  3 phase 15  f= 0.10 ph=5.890 sharp */ [  100, 103, 105, 107, 109, 112, 113, 115, 117, 119, 120, 121, 123, 124, 125, 125, 126, 126, 127, 127, 127, 127, 127, 126, 125, 125, 124, 123, 122, 120, 119, 117 ],
      /* slot  64  bucket  4 phase  0  f= 0.22 ph=0.000 sharp */ [  127, 127, 126, 124, 121, 118, 115, 110, 105, 100,  95,  89,  83,  76,  70,  64,  57,  51,  45,  39,  34,  29,  24,  20,  16,  13,  10,   7,   5,   4,   2,   1 ],
      /* slot  65  bucket  4 phase  1  f= 0.22 ph=0.393 sharp */ [  100,  95,  89,  83,  76,  70,  64,  57,  51,  45,  39,  34,  29,  24,  20,  16,  13,  10,   7,   5,   4,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,  -1 ],
      /* slot  66  bucket  4 phase  2  f= 0.22 ph=0.785 sharp */ [   45,  39,  34,  29,  24,  20,  16,  13,  10,   7,   5,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -5,  -7,  -9, -12, -16, -19 ],
      /* slot  67  bucket  4 phase  3  f= 0.22 ph=1.178 sharp */ [    7,   5,   3,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,  -2,  -3,  -5,  -7,  -9, -12, -16, -19, -24, -28, -33, -39, -44, -50, -57, -63, -69 ],
      /* slot  68  bucket  4 phase  4  f= 0.22 ph=1.571 sharp */ [    0,   0,   0,   0,  -1,  -1,  -2,  -3,  -5,  -7, -10, -12, -16, -20, -24, -28, -33, -39, -45, -51, -57, -63, -69, -76, -82, -88, -94,-100,-105,-110,-114,-118 ],
      /* slot  69  bucket  4 phase  5  f= 0.22 ph=1.963 sharp */ [   -7, -10, -12, -16, -20, -24, -29, -34, -39, -45, -51, -57, -63, -70, -76, -82, -88, -94,-100,-105,-110,-114,-118,-121,-124,-125,-127,-127,-127,-126,-124,-122 ],
      /* slot  70  bucket  4 phase  6  f= 0.22 ph=2.356 sharp */ [  -45, -51, -57, -63, -70, -76, -82, -89, -94,-100,-105,-110,-114,-118,-121,-124,-125,-127,-127,-127,-126,-124,-121,-118,-115,-110,-106,-101, -95, -89, -83, -77 ],
      /* slot  71  bucket  4 phase  7  f= 0.22 ph=2.749 sharp */ [ -100,-105,-110,-114,-118,-121,-124,-126,-127,-127,-127,-126,-124,-121,-118,-115,-110,-106,-100, -95, -89, -83, -77, -70, -64, -58, -51, -45, -40, -34, -29, -24 ],
      /* slot  72  bucket  4 phase  8  f= 0.22 ph=3.142 sharp */ [ -127,-127,-126,-124,-121,-118,-115,-110,-105,-100, -95, -89, -83, -76, -70, -64, -57, -51, -45, -39, -34, -29, -24, -20, -16, -13, -10,  -7,  -5,  -4,  -2,  -1 ],
      /* slot  73  bucket  4 phase  9  f= 0.22 ph=3.534 sharp */ [ -100, -95, -89, -83, -76, -70, -64, -57, -51, -45, -39, -34, -29, -24, -20, -16, -13, -10,  -7,  -5,  -4,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   1 ],
      /* slot  74  bucket  4 phase 10  f= 0.22 ph=3.927 sharp */ [  -45, -39, -34, -29, -24, -20, -16, -13, -10,  -7,  -5,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   5,   7,   9,  12,  16,  19 ],
      /* slot  75  bucket  4 phase 11  f= 0.22 ph=4.320 sharp */ [   -7,  -5,  -3,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   5,   7,   9,  12,  16,  19,  24,  28,  33,  39,  44,  50,  57,  63,  69 ],
      /* slot  76  bucket  4 phase 12  f= 0.22 ph=4.712 sharp */ [    0,   0,   0,   0,   1,   1,   2,   3,   5,   7,  10,  12,  16,  20,  24,  28,  33,  39,  45,  51,  57,  63,  69,  76,  82,  88,  94, 100, 105, 110, 114, 118 ],
      /* slot  77  bucket  4 phase 13  f= 0.22 ph=5.105 sharp */ [    7,  10,  12,  16,  20,  24,  29,  34,  39,  45,  51,  57,  63,  70,  76,  82,  88,  94, 100, 105, 110, 114, 118, 121, 124, 125, 127, 127, 127, 126, 124, 122 ],
      /* slot  78  bucket  4 phase 14  f= 0.22 ph=5.498 sharp */ [   45,  51,  57,  63,  70,  76,  82,  89,  94, 100, 105, 110, 114, 118, 121, 124, 125, 127, 127, 127, 126, 124, 121, 118, 115, 110, 106, 101,  95,  89,  83,  77 ],
      /* slot  79  bucket  4 phase 15  f= 0.22 ph=5.890 sharp */ [  100, 105, 110, 114, 118, 121, 124, 126, 127, 127, 127, 126, 124, 121, 118, 115, 110, 106, 100,  95,  89,  83,  77,  70,  64,  58,  51,  45,  40,  34,  29,  24 ],
      /* slot  80  bucket  5 phase  0  f= 0.14 ph=0.000 triangle */ [  127, 125, 122, 120, 118, 116, 113, 111, 109, 107, 104, 102, 100,  98,  95,  93,  91,  89,  86,  84,  82,  80,  77,  75,  73,  71,  68,  66,  64,  62,  59,  57 ],
      /* slot  81  bucket  5 phase  1  f= 0.14 ph=0.393 triangle */ [   95,  93,  91,  88,  86,  84,  82,  79,  77,  75,  73,  70,  68,  66,  64,  61,  59,  57,  55,  52,  50,  48,  46,  43,  41,  39,  37,  34,  32,  30,  28,  25 ],
      /* slot  82  bucket  5 phase  2  f= 0.14 ph=0.785 triangle */ [   64,  61,  59,  57,  54,  52,  50,  48,  45,  43,  41,  39,  36,  34,  32,  30,  27,  25,  23,  21,  18,  16,  14,  12,   9,   7,   5,   3,   0,  -2,  -4,  -6 ],
      /* slot  83  bucket  5 phase  3  f= 0.14 ph=1.178 triangle */ [   32,  29,  27,  25,  23,  20,  18,  16,  14,  11,   9,   7,   5,   2,   0,  -2,  -4,  -7,  -9, -11, -13, -16, -18, -20, -22, -25, -27, -29, -31, -34, -36, -38 ],
      /* slot  84  bucket  5 phase  4  f= 0.14 ph=1.571 triangle */ [    0,  -2,  -5,  -7,  -9, -11, -14, -16, -18, -20, -23, -25, -27, -29, -32, -34, -36, -38, -41, -43, -45, -47, -50, -52, -54, -56, -59, -61, -63, -65, -68, -70 ],
      /* slot  85  bucket  5 phase  5  f= 0.14 ph=1.963 triangle */ [  -32, -34, -36, -39, -41, -43, -45, -48, -50, -52, -54, -57, -59, -61, -63, -66, -68, -70, -72, -75, -77, -79, -81, -84, -86, -88, -90, -93, -95, -97, -99,-102 ],
      /* slot  86  bucket  5 phase  6  f= 0.14 ph=2.356 triangle */ [  -64, -66, -68, -70, -73, -75, -77, -79, -82, -84, -86, -88, -91, -93, -95, -97,-100,-102,-104,-106,-109,-111,-113,-115,-118,-120,-122,-124,-127,-125,-123,-121 ],
      /* slot  87  bucket  5 phase  7  f= 0.14 ph=2.749 triangle */ [  -95, -98,-100,-102,-104,-107,-109,-111,-113,-116,-118,-120,-122,-125,-127,-125,-123,-120,-118,-116,-114,-111,-109,-107,-105,-102,-100, -98, -96, -93, -91, -89 ],
      /* slot  88  bucket  5 phase  8  f= 0.14 ph=3.142 triangle */ [ -127,-125,-122,-120,-118,-116,-113,-111,-109,-107,-104,-102,-100, -98, -95, -93, -91, -89, -86, -84, -82, -80, -77, -75, -73, -71, -68, -66, -64, -62, -59, -57 ],
      /* slot  89  bucket  5 phase  9  f= 0.14 ph=3.534 triangle */ [  -95, -93, -91, -88, -86, -84, -82, -79, -77, -75, -73, -70, -68, -66, -64, -61, -59, -57, -55, -52, -50, -48, -46, -43, -41, -39, -37, -34, -32, -30, -28, -25 ],
      /* slot  90  bucket  5 phase 10  f= 0.14 ph=3.927 triangle */ [  -64, -61, -59, -57, -54, -52, -50, -48, -45, -43, -41, -39, -36, -34, -32, -30, -27, -25, -23, -21, -18, -16, -14, -12,  -9,  -7,  -5,  -3,   0,   2,   4,   6 ],
      /* slot  91  bucket  5 phase 11  f= 0.14 ph=4.320 triangle */ [  -32, -29, -27, -25, -23, -20, -18, -16, -14, -11,  -9,  -7,  -5,  -2,   0,   2,   4,   7,   9,  11,  13,  16,  18,  20,  22,  25,  27,  29,  31,  34,  36,  38 ],
      /* slot  92  bucket  5 phase 12  f= 0.14 ph=4.712 triangle */ [    0,   2,   5,   7,   9,  11,  14,  16,  18,  20,  23,  25,  27,  29,  32,  34,  36,  38,  41,  43,  45,  47,  50,  52,  54,  56,  59,  61,  63,  65,  68,  70 ],
      /* slot  93  bucket  5 phase 13  f= 0.14 ph=5.105 triangle */ [   32,  34,  36,  39,  41,  43,  45,  48,  50,  52,  54,  57,  59,  61,  63,  66,  68,  70,  72,  75,  77,  79,  81,  84,  86,  88,  90,  93,  95,  97,  99, 102 ],
      /* slot  94  bucket  5 phase 14  f= 0.14 ph=5.498 triangle */ [   64,  66,  68,  70,  73,  75,  77,  79,  82,  84,  86,  88,  91,  93,  95,  97, 100, 102, 104, 106, 109, 111, 113, 115, 118, 120, 122, 124, 127, 125, 123, 121 ],
      /* slot  95  bucket  5 phase 15  f= 0.14 ph=5.890 triangle */ [   95,  98, 100, 102, 104, 107, 109, 111, 113, 116, 118, 120, 122, 125, 127, 125, 123, 120, 118, 116, 114, 111, 109, 107, 105, 102, 100,  98,  96,  93,  91,  89 ],
      /* slot  96  bucket  6 phase  0  f= 0.26 ph=0.000 sharp */ [  127, 126, 125, 123, 119, 115, 110, 104,  98,  91,  84,  77,  69,  62,  54,  47,  40,  34,  28,  22,  18,  13,  10,   7,   5,   3,   2,   1,   0,   0,   0,   0 ],
      /* slot  97  bucket  6 phase  1  f= 0.26 ph=0.393 sharp */ [  100,  94,  86,  79,  72,  64,  57,  49,  43,  36,  30,  24,  19,  15,  11,   8,   5,   3,   2,   1,   0,   0,   0,   0,   0,   0,   0,  -1,  -2,  -4,  -6,  -8 ],
      /* slot  98  bucket  6 phase  2  f= 0.26 ph=0.785 sharp */ [   45,  38,  32,  26,  21,  16,  12,   9,   6,   4,   2,   1,   1,   0,   0,   0,   0,   0,   0,  -1,  -2,  -3,  -5,  -7, -10, -14, -18, -23, -28, -34, -41, -48 ],
      /* slot  99  bucket  6 phase  3  f= 0.26 ph=1.178 sharp */ [    7,   5,   3,   2,   1,   0,   0,   0,   0,   0,   0,  -1,  -1,  -3,  -4,  -6,  -9, -12, -16, -21, -26, -32, -38, -45, -52, -60, -67, -75, -82, -89, -96,-103 ],
      /* slot 100  bucket  6 phase  4  f= 0.26 ph=1.571 sharp */ [    0,   0,   0,   0,  -1,  -2,  -4,  -6,  -8, -11, -15, -19, -24, -30, -36, -43, -50, -57, -65, -72, -79, -87, -94,-100,-107,-112,-117,-121,-124,-126,-127,-127 ],
      /* slot 101  bucket  6 phase  5  f= 0.26 ph=1.963 sharp */ [   -7, -10, -14, -18, -23, -28, -34, -40, -47, -55, -62, -69, -77, -84, -91, -98,-104,-110,-115,-119,-123,-125,-127,-127,-126,-125,-122,-119,-115,-110,-104, -98 ],
      /* slot 102  bucket  6 phase  6  f= 0.26 ph=2.356 sharp */ [  -45, -52, -59, -67, -74, -82, -89, -96,-102,-108,-113,-118,-122,-124,-126,-127,-127,-126,-123,-120,-116,-112,-106,-100, -93, -86, -79, -71, -64, -56, -49, -42 ],
      /* slot 103  bucket  6 phase  7  f= 0.26 ph=2.749 sharp */ [ -100,-106,-112,-116,-120,-124,-126,-127,-127,-126,-124,-121,-118,-113,-108,-102, -96, -89, -81, -74, -66, -59, -52, -45, -38, -32, -26, -21, -16, -12,  -9,  -6 ],
      /* slot 104  bucket  6 phase  8  f= 0.26 ph=3.142 sharp */ [ -127,-126,-125,-123,-119,-115,-110,-104, -98, -91, -84, -77, -69, -62, -54, -47, -40, -34, -28, -22, -18, -13, -10,  -7,  -5,  -3,  -2,  -1,   0,   0,   0,   0 ],
      /* slot 105  bucket  6 phase  9  f= 0.26 ph=3.534 sharp */ [ -100, -94, -86, -79, -72, -64, -57, -49, -43, -36, -30, -24, -19, -15, -11,  -8,  -5,  -3,  -2,  -1,   0,   0,   0,   0,   0,   0,   0,   1,   2,   4,   6,   8 ],
      /* slot 106  bucket  6 phase 10  f= 0.26 ph=3.927 sharp */ [  -45, -38, -32, -26, -21, -16, -12,  -9,  -6,  -4,  -2,  -1,  -1,   0,   0,   0,   0,   0,   0,   1,   2,   3,   5,   7,  10,  14,  18,  23,  28,  34,  41,  48 ],
      /* slot 107  bucket  6 phase 11  f= 0.26 ph=4.320 sharp */ [   -7,  -5,  -3,  -2,  -1,   0,   0,   0,   0,   0,   0,   1,   1,   3,   4,   6,   9,  12,  16,  21,  26,  32,  38,  45,  52,  60,  67,  75,  82,  89,  96, 103 ],
      /* slot 108  bucket  6 phase 12  f= 0.26 ph=4.712 sharp */ [    0,   0,   0,   0,   1,   2,   4,   6,   8,  11,  15,  19,  24,  30,  36,  43,  50,  57,  65,  72,  79,  87,  94, 100, 107, 112, 117, 121, 124, 126, 127, 127 ],
      /* slot 109  bucket  6 phase 13  f= 0.26 ph=5.105 sharp */ [    7,  10,  14,  18,  23,  28,  34,  40,  47,  55,  62,  69,  77,  84,  91,  98, 104, 110, 115, 119, 123, 125, 127, 127, 126, 125, 122, 119, 115, 110, 104,  98 ],
      /* slot 110  bucket  6 phase 14  f= 0.26 ph=5.498 sharp */ [   45,  52,  59,  67,  74,  82,  89,  96, 102, 108, 113, 118, 122, 124, 126, 127, 127, 126, 123, 120, 116, 112, 106, 100,  93,  86,  79,  71,  64,  56,  49,  42 ],
      /* slot 111  bucket  6 phase 15  f= 0.26 ph=5.890 sharp */ [  100, 106, 112, 116, 120, 124, 126, 127, 127, 126, 124, 121, 118, 113, 108, 102,  96,  89,  81,  74,  66,  59,  52,  45,  38,  32,  26,  21,  16,  12,   9,   6 ],
      /* slot 112  bucket  7 phase  0  f= 0.10 ph=0.000 triangle */ [  127, 125, 124, 122, 121, 119, 117, 116, 114, 112, 111, 109, 108, 106, 104, 103, 101,  99,  98,  96,  95,  93,  91,  90,  88,  86,  85,  83,  82,  80,  78,  77 ],
      /* slot 113  bucket  7 phase  1  f= 0.10 ph=0.393 triangle */ [   95,  94,  92,  90,  89,  87,  86,  84,  82,  81,  79,  77,  76,  74,  73,  71,  69,  68,  66,  64,  63,  61,  60,  58,  56,  55,  53,  51,  50,  48,  47,  45 ],
      /* slot 114  bucket  7 phase  2  f= 0.10 ph=0.785 triangle */ [   64,  62,  60,  59,  57,  55,  54,  52,  51,  49,  47,  46,  44,  42,  41,  39,  38,  36,  34,  33,  31,  29,  28,  26,  25,  23,  21,  20,  18,  16,  15,  13 ],
      /* slot 115  bucket  7 phase  3  f= 0.10 ph=1.178 triangle */ [   32,  30,  29,  27,  25,  24,  22,  20,  19,  17,  16,  14,  12,  11,   9,   7,   6,   4,   3,   1,  -1,  -2,  -4,  -6,  -7,  -9, -10, -12, -14, -15, -17, -19 ],
      /* slot 116  bucket  7 phase  4  f= 0.10 ph=1.571 triangle */ [    0,  -2,  -3,  -5,  -6,  -8, -10, -11, -13, -15, -16, -18, -19, -21, -23, -24, -26, -28, -29, -31, -32, -34, -36, -37, -39, -41, -42, -44, -45, -47, -49, -50 ],
      /* slot 117  bucket  7 phase  5  f= 0.10 ph=1.963 triangle */ [  -32, -33, -35, -37, -38, -40, -41, -43, -45, -46, -48, -50, -51, -53, -54, -56, -58, -59, -61, -63, -64, -66, -67, -69, -71, -72, -74, -76, -77, -79, -80, -82 ],
      /* slot 118  bucket  7 phase  6  f= 0.10 ph=2.356 triangle */ [  -64, -65, -67, -68, -70, -72, -73, -75, -76, -78, -80, -81, -83, -85, -86, -88, -89, -91, -93, -94, -96, -98, -99,-101,-102,-104,-106,-107,-109,-111,-112,-114 ],
      /* slot 119  bucket  7 phase  7  f= 0.10 ph=2.749 triangle */ [  -95, -97, -98,-100,-102,-103,-105,-107,-108,-110,-111,-113,-115,-116,-118,-120,-121,-123,-124,-126,-126,-125,-123,-121,-120,-118,-117,-115,-113,-112,-110,-108 ],
      /* slot 120  bucket  7 phase  8  f= 0.10 ph=3.142 triangle */ [ -127,-125,-124,-122,-121,-119,-117,-116,-114,-112,-111,-109,-108,-106,-104,-103,-101, -99, -98, -96, -95, -93, -91, -90, -88, -86, -85, -83, -82, -80, -78, -77 ],
      /* slot 121  bucket  7 phase  9  f= 0.10 ph=3.534 triangle */ [  -95, -94, -92, -90, -89, -87, -86, -84, -82, -81, -79, -77, -76, -74, -73, -71, -69, -68, -66, -64, -63, -61, -60, -58, -56, -55, -53, -51, -50, -48, -47, -45 ],
      /* slot 122  bucket  7 phase 10  f= 0.10 ph=3.927 triangle */ [  -64, -62, -60, -59, -57, -55, -54, -52, -51, -49, -47, -46, -44, -42, -41, -39, -38, -36, -34, -33, -31, -29, -28, -26, -25, -23, -21, -20, -18, -16, -15, -13 ],
      /* slot 123  bucket  7 phase 11  f= 0.10 ph=4.320 triangle */ [  -32, -30, -29, -27, -25, -24, -22, -20, -19, -17, -16, -14, -12, -11,  -9,  -7,  -6,  -4,  -3,  -1,   1,   2,   4,   6,   7,   9,  10,  12,  14,  15,  17,  19 ],
      /* slot 124  bucket  7 phase 12  f= 0.10 ph=4.712 triangle */ [    0,   2,   3,   5,   6,   8,  10,  11,  13,  15,  16,  18,  19,  21,  23,  24,  26,  28,  29,  31,  32,  34,  36,  37,  39,  41,  42,  44,  45,  47,  49,  50 ],
      /* slot 125  bucket  7 phase 13  f= 0.10 ph=5.105 triangle */ [   32,  33,  35,  37,  38,  40,  41,  43,  45,  46,  48,  50,  51,  53,  54,  56,  58,  59,  61,  63,  64,  66,  67,  69,  71,  72,  74,  76,  77,  79,  80,  82 ],
      /* slot 126  bucket  7 phase 14  f= 0.10 ph=5.498 triangle */ [   64,  65,  67,  68,  70,  72,  73,  75,  76,  78,  80,  81,  83,  85,  86,  88,  89,  91,  93,  94,  96,  98,  99, 101, 102, 104, 106, 107, 109, 111, 112, 114 ],
      /* slot 127  bucket  7 phase 15  f= 0.10 ph=5.890 triangle */ [   95,  97,  98, 100, 102, 103, 105, 107, 108, 110, 111, 113, 115, 116, 118, 120, 121, 123, 124, 126, 126, 125, 123, 121, 120, 118, 117, 115, 113, 112, 110, 108 ],
      /* slot 128  bucket  8 phase  0  f= 0.14 ph=0.000 sine */ [  127, 127, 127, 127, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 117, 116, 115, 113, 111, 110, 108, 106, 104, 102, 100,  97,  95,  93,  90,  88,  85,  82 ],
      /* slot 129  bucket  8 phase  1  f= 0.14 ph=0.393 sine */ [  117, 116, 114, 113, 111, 109, 108, 106, 104, 102,  99,  97,  95,  92,  90,  87,  85,  82,  79,  77,  74,  71,  68,  65,  62,  59,  56,  52,  49,  46,  43,  39 ],
      /* slot 130  bucket  8 phase  2  f= 0.14 ph=0.785 sine */ [   90,  87,  85,  82,  79,  76,  74,  71,  68,  65,  62,  58,  55,  52,  49,  46,  42,  39,  36,  32,  29,  25,  22,  18,  15,  11,   8,   4,   1,  -3,  -7, -10 ],
      /* slot 131  bucket  8 phase  3  f= 0.14 ph=1.178 sine */ [   49,  45,  42,  39,  35,  32,  28,  25,  21,  18,  14,  11,   7,   4,   0,  -3,  -7, -10, -14, -17, -21, -24, -28, -31, -35, -38, -41, -45, -48, -51, -55, -58 ],
      /* slot 132  bucket  8 phase  4  f= 0.14 ph=1.571 sine */ [    0,  -4,  -7, -11, -14, -18, -21, -25, -28, -32, -35, -38, -42, -45, -48, -52, -55, -58, -61, -64, -67, -70, -73, -76, -79, -82, -84, -87, -89, -92, -94, -97 ],
      /* slot 133  bucket  8 phase  5  f= 0.14 ph=1.963 sine */ [  -49, -52, -55, -58, -61, -64, -67, -70, -73, -76, -79, -82, -84, -87, -90, -92, -94, -97, -99,-101,-103,-105,-107,-109,-111,-113,-114,-116,-117,-118,-120,-121 ],
      /* slot 134  bucket  8 phase  6  f= 0.14 ph=2.356 sine */ [  -90, -92, -95, -97, -99,-101,-104,-106,-107,-109,-111,-113,-114,-116,-117,-119,-120,-121,-122,-123,-124,-124,-125,-126,-126,-127,-127,-127,-127,-127,-127,-127 ],
      /* slot 135  bucket  8 phase  7  f= 0.14 ph=2.749 sine */ [ -117,-119,-120,-121,-122,-123,-124,-125,-125,-126,-126,-127,-127,-127,-127,-127,-127,-127,-126,-126,-125,-125,-124,-123,-122,-121,-120,-119,-118,-116,-115,-113 ],
      /* slot 136  bucket  8 phase  8  f= 0.14 ph=3.142 sine */ [ -127,-127,-127,-127,-126,-126,-125,-125,-124,-123,-122,-121,-120,-119,-117,-116,-115,-113,-111,-110,-108,-106,-104,-102,-100, -97, -95, -93, -90, -88, -85, -82 ],
      /* slot 137  bucket  8 phase  9  f= 0.14 ph=3.534 sine */ [ -117,-116,-114,-113,-111,-109,-108,-106,-104,-102, -99, -97, -95, -92, -90, -87, -85, -82, -79, -77, -74, -71, -68, -65, -62, -59, -56, -52, -49, -46, -43, -39 ],
      /* slot 138  bucket  8 phase 10  f= 0.14 ph=3.927 sine */ [  -90, -87, -85, -82, -79, -76, -74, -71, -68, -65, -62, -58, -55, -52, -49, -46, -42, -39, -36, -32, -29, -25, -22, -18, -15, -11,  -8,  -4,  -1,   3,   7,  10 ],
      /* slot 139  bucket  8 phase 11  f= 0.14 ph=4.320 sine */ [  -49, -45, -42, -39, -35, -32, -28, -25, -21, -18, -14, -11,  -7,  -4,   0,   3,   7,  10,  14,  17,  21,  24,  28,  31,  35,  38,  41,  45,  48,  51,  55,  58 ],
      /* slot 140  bucket  8 phase 12  f= 0.14 ph=4.712 sine */ [    0,   4,   7,  11,  14,  18,  21,  25,  28,  32,  35,  38,  42,  45,  48,  52,  55,  58,  61,  64,  67,  70,  73,  76,  79,  82,  84,  87,  89,  92,  94,  97 ],
      /* slot 141  bucket  8 phase 13  f= 0.14 ph=5.105 sine */ [   49,  52,  55,  58,  61,  64,  67,  70,  73,  76,  79,  82,  84,  87,  90,  92,  94,  97,  99, 101, 103, 105, 107, 109, 111, 113, 114, 116, 117, 118, 120, 121 ],
      /* slot 142  bucket  8 phase 14  f= 0.14 ph=5.498 sine */ [   90,  92,  95,  97,  99, 101, 104, 106, 107, 109, 111, 113, 114, 116, 117, 119, 120, 121, 122, 123, 124, 124, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127 ],
      /* slot 143  bucket  8 phase 15  f= 0.14 ph=5.890 sine */ [  117, 119, 120, 121, 122, 123, 124, 125, 125, 126, 126, 127, 127, 127, 127, 127, 127, 127, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 118, 116, 115, 113 ],
      /* slot 144  bucket  9 phase  0  f= 0.30 ph=0.000 sharp */ [  127, 126, 124, 121, 117, 111, 105,  97,  90,  81,  73,  64,  55,  47,  39,  32,  25,  20,  14,  10,   7,   4,   2,   1,   0,   0,   0,   0,   0,   0,  -1,  -2 ],
      /* slot 145  bucket  9 phase  1  f= 0.30 ph=0.393 sharp */ [  100,  92,  84,  76,  67,  59,  50,  42,  35,  28,  22,  16,  12,   8,   5,   3,   2,   1,   0,   0,   0,   0,   0,  -1,  -2,  -3,  -6,  -9, -12, -17, -22, -29 ],
      /* slot 146  bucket  9 phase  2  f= 0.30 ph=0.785 sharp */ [   45,  37,  30,  24,  18,  13,   9,   6,   4,   2,   1,   0,   0,   0,   0,   0,  -1,  -1,  -3,  -5,  -7, -11, -15, -20, -26, -33, -40, -48, -57, -65, -74, -83 ],
      /* slot 147  bucket  9 phase  3  f= 0.30 ph=1.178 sharp */ [    7,   4,   3,   1,   0,   0,   0,   0,   0,   0,  -1,  -2,  -4,  -6, -10, -14, -18, -24, -31, -38, -46, -54, -62, -71, -79, -88, -96,-103,-110,-116,-120,-124 ],
      /* slot 148  bucket  9 phase  4  f= 0.30 ph=1.571 sharp */ [    0,   0,   0,  -1,  -2,  -3,  -5,  -8, -12, -17, -22, -28, -35, -43, -51, -59, -68, -76, -85, -93,-101,-108,-114,-119,-123,-125,-127,-127,-126,-123,-119,-114 ],
      /* slot 149  bucket  9 phase  5  f= 0.30 ph=1.963 sharp */ [   -7, -11, -15, -20, -26, -33, -40, -48, -56, -65, -73, -82, -90, -98,-105,-112,-117,-121,-125,-126,-127,-126,-124,-121,-116,-111,-104, -97, -89, -81, -72, -63 ],
      /* slot 150  bucket  9 phase  6  f= 0.30 ph=2.356 sharp */ [  -45, -53, -62, -70, -79, -87, -95,-103,-109,-115,-120,-124,-126,-127,-127,-125,-122,-118,-113,-107,-100, -92, -84, -75, -66, -58, -49, -41, -34, -27, -21, -16 ],
      /* slot 151  bucket  9 phase  7  f= 0.30 ph=2.749 sharp */ [ -100,-107,-113,-118,-122,-125,-127,-127,-126,-123,-120,-115,-109,-102, -95, -87, -78, -70, -61, -52, -44, -37, -30, -23, -18, -13,  -9,  -6,  -4,  -2,  -1,   0 ],
      /* slot 152  bucket  9 phase  8  f= 0.30 ph=3.142 sharp */ [ -127,-126,-124,-121,-117,-111,-105, -97, -90, -81, -73, -64, -55, -47, -39, -32, -25, -20, -14, -10,  -7,  -4,  -2,  -1,   0,   0,   0,   0,   0,   0,   1,   2 ],
      /* slot 153  bucket  9 phase  9  f= 0.30 ph=3.534 sharp */ [ -100, -92, -84, -76, -67, -59, -50, -42, -35, -28, -22, -16, -12,  -8,  -5,  -3,  -2,  -1,   0,   0,   0,   0,   0,   1,   2,   3,   6,   9,  12,  17,  22,  29 ],
      /* slot 154  bucket  9 phase 10  f= 0.30 ph=3.927 sharp */ [  -45, -37, -30, -24, -18, -13,  -9,  -6,  -4,  -2,  -1,   0,   0,   0,   0,   0,   1,   1,   3,   5,   7,  11,  15,  20,  26,  33,  40,  48,  57,  65,  74,  83 ],
      /* slot 155  bucket  9 phase 11  f= 0.30 ph=4.320 sharp */ [   -7,  -4,  -3,  -1,   0,   0,   0,   0,   0,   0,   1,   2,   4,   6,  10,  14,  18,  24,  31,  38,  46,  54,  62,  71,  79,  88,  96, 103, 110, 116, 120, 124 ],
      /* slot 156  bucket  9 phase 12  f= 0.30 ph=4.712 sharp */ [    0,   0,   0,   1,   2,   3,   5,   8,  12,  17,  22,  28,  35,  43,  51,  59,  68,  76,  85,  93, 101, 108, 114, 119, 123, 125, 127, 127, 126, 123, 119, 114 ],
      /* slot 157  bucket  9 phase 13  f= 0.30 ph=5.105 sharp */ [    7,  11,  15,  20,  26,  33,  40,  48,  56,  65,  73,  82,  90,  98, 105, 112, 117, 121, 125, 126, 127, 126, 124, 121, 116, 111, 104,  97,  89,  81,  72,  63 ],
      /* slot 158  bucket  9 phase 14  f= 0.30 ph=5.498 sharp */ [   45,  53,  62,  70,  79,  87,  95, 103, 109, 115, 120, 124, 126, 127, 127, 125, 122, 118, 113, 107, 100,  92,  84,  75,  66,  58,  49,  41,  34,  27,  21,  16 ],
      /* slot 159  bucket  9 phase 15  f= 0.30 ph=5.890 sharp */ [  100, 107, 113, 118, 122, 125, 127, 127, 126, 123, 120, 115, 109, 102,  95,  87,  78,  70,  61,  52,  44,  37,  30,  23,  18,  13,   9,   6,   4,   2,   1,   0 ],
      /* slot 160  bucket 10 phase  0  f= 8.00 ph=0.000 sharp */ [  127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0 ],
      /* slot 161  bucket 10 phase  1  f= 8.00 ph=0.393 sharp */ [  100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7 ],
      /* slot 162  bucket 10 phase  2  f= 8.00 ph=0.785 sharp */ [   45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45 ],
      /* slot 163  bucket 10 phase  3  f= 8.00 ph=1.178 sharp */ [    7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100 ],
      /* slot 164  bucket 10 phase  4  f= 8.00 ph=1.571 sharp */ [    0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127 ],
      /* slot 165  bucket 10 phase  5  f= 8.00 ph=1.963 sharp */ [   -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100 ],
      /* slot 166  bucket 10 phase  6  f= 8.00 ph=2.356 sharp */ [  -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45 ],
      /* slot 167  bucket 10 phase  7  f= 8.00 ph=2.749 sharp */ [ -100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7 ],
      /* slot 168  bucket 10 phase  8  f= 8.00 ph=3.142 sharp */ [ -127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0 ],
      /* slot 169  bucket 10 phase  9  f= 8.00 ph=3.534 sharp */ [ -100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7 ],
      /* slot 170  bucket 10 phase 10  f= 8.00 ph=3.927 sharp */ [  -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45 ],
      /* slot 171  bucket 10 phase 11  f= 8.00 ph=4.320 sharp */ [   -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100 ],
      /* slot 172  bucket 10 phase 12  f= 8.00 ph=4.712 sharp */ [    0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127,   0, 127,   0,-127 ],
      /* slot 173  bucket 10 phase 13  f= 8.00 ph=5.105 sharp */ [    7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100,   7, 100,  -7,-100 ],
      /* slot 174  bucket 10 phase 14  f= 8.00 ph=5.498 sharp */ [   45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45,  45,  45, -45, -45 ],
      /* slot 175  bucket 10 phase 15  f= 8.00 ph=5.890 sharp */ [  100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7, 100,   7,-100,  -7 ],
      /* slot 176  bucket 11 phase  0  f= 0.26 ph=0.000 triangle */ [  127, 123, 119, 115, 110, 106, 102,  98,  94,  90,  86,  81,  77,  73,  69,  65,  61,  56,  52,  48,  44,  40,  36,  32,  27,  23,  19,  15,  11,   7,   3,  -2 ],
      /* slot 177  bucket 11 phase  1  f= 0.26 ph=0.393 triangle */ [   95,  91,  87,  83,  79,  75,  70,  66,  62,  58,  54,  50,  45,  41,  37,  33,  29,  25,  21,  16,  12,   8,   4,   0,  -4,  -8, -13, -17, -21, -25, -29, -33 ],
      /* slot 178  bucket 11 phase  2  f= 0.26 ph=0.785 triangle */ [   64,  59,  55,  51,  47,  43,  39,  34,  30,  26,  22,  18,  14,  10,   5,   1,  -3,  -7, -11, -15, -19, -24, -28, -32, -36, -40, -44, -49, -53, -57, -61, -65 ],
      /* slot 179  bucket 11 phase  3  f= 0.26 ph=1.178 triangle */ [   32,  28,  23,  19,  15,  11,   7,   3,  -1,  -6, -10, -14, -18, -22, -26, -30, -35, -39, -43, -47, -51, -55, -60, -64, -68, -72, -76, -80, -84, -89, -93, -97 ],
      /* slot 180  bucket 11 phase  4  f= 0.26 ph=1.571 triangle */ [    0,  -4,  -8, -12, -17, -21, -25, -29, -33, -37, -41, -46, -50, -54, -58, -62, -66, -71, -75, -79, -83, -87, -91, -95,-100,-104,-108,-112,-116,-120,-124,-125 ],
      /* slot 181  bucket 11 phase  5  f= 0.26 ph=1.963 triangle */ [  -32, -36, -40, -44, -48, -52, -57, -61, -65, -69, -73, -77, -82, -86, -90, -94, -98,-102,-106,-111,-115,-119,-123,-127,-123,-119,-114,-110,-106,-102, -98, -94 ],
      /* slot 182  bucket 11 phase  6  f= 0.26 ph=2.356 triangle */ [  -64, -68, -72, -76, -80, -84, -88, -93, -97,-101,-105,-109,-113,-117,-122,-126,-124,-120,-116,-112,-108,-103, -99, -95, -91, -87, -83, -78, -74, -70, -66, -62 ],
      /* slot 183  bucket 11 phase  7  f= 0.26 ph=2.749 triangle */ [  -95, -99,-104,-108,-112,-116,-120,-124,-126,-121,-117,-113,-109,-105,-101, -97, -92, -88, -84, -80, -76, -72, -67, -63, -59, -55, -51, -47, -43, -38, -34, -30 ],
      /* slot 184  bucket 11 phase  8  f= 0.26 ph=3.142 triangle */ [ -127,-123,-119,-115,-110,-106,-102, -98, -94, -90, -86, -81, -77, -73, -69, -65, -61, -56, -52, -48, -44, -40, -36, -32, -27, -23, -19, -15, -11,  -7,  -3,   2 ],
      /* slot 185  bucket 11 phase  9  f= 0.26 ph=3.534 triangle */ [  -95, -91, -87, -83, -79, -75, -70, -66, -62, -58, -54, -50, -45, -41, -37, -33, -29, -25, -21, -16, -12,  -8,  -4,   0,   4,   8,  13,  17,  21,  25,  29,  33 ],
      /* slot 186  bucket 11 phase 10  f= 0.26 ph=3.927 triangle */ [  -64, -59, -55, -51, -47, -43, -39, -34, -30, -26, -22, -18, -14, -10,  -5,  -1,   3,   7,  11,  15,  19,  24,  28,  32,  36,  40,  44,  49,  53,  57,  61,  65 ],
      /* slot 187  bucket 11 phase 11  f= 0.26 ph=4.320 triangle */ [  -32, -28, -23, -19, -15, -11,  -7,  -3,   1,   6,  10,  14,  18,  22,  26,  30,  35,  39,  43,  47,  51,  55,  60,  64,  68,  72,  76,  80,  84,  89,  93,  97 ],
      /* slot 188  bucket 11 phase 12  f= 0.26 ph=4.712 triangle */ [    0,   4,   8,  12,  17,  21,  25,  29,  33,  37,  41,  46,  50,  54,  58,  62,  66,  71,  75,  79,  83,  87,  91,  95, 100, 104, 108, 112, 116, 120, 124, 125 ],
      /* slot 189  bucket 11 phase 13  f= 0.26 ph=5.105 triangle */ [   32,  36,  40,  44,  48,  52,  57,  61,  65,  69,  73,  77,  82,  86,  90,  94,  98, 102, 106, 111, 115, 119, 123, 127, 123, 119, 114, 110, 106, 102,  98,  94 ],
      /* slot 190  bucket 11 phase 14  f= 0.26 ph=5.498 triangle */ [   64,  68,  72,  76,  80,  84,  88,  93,  97, 101, 105, 109, 113, 117, 122, 126, 124, 120, 116, 112, 108, 103,  99,  95,  91,  87,  83,  78,  74,  70,  66,  62 ],
      /* slot 191  bucket 11 phase 15  f= 0.26 ph=5.890 triangle */ [   95,  99, 104, 108, 112, 116, 120, 124, 126, 121, 117, 113, 109, 105, 101,  97,  92,  88,  84,  80,  76,  72,  67,  63,  59,  55,  51,  47,  43,  38,  34,  30 ],
      /* slot 192  bucket 12 phase  0  f= 1.00 ph=0.000 sharp */ [  127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120 ],
      /* slot 193  bucket 12 phase  1  f= 1.00 ph=0.393 sharp */ [  100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120 ],
      /* slot 194  bucket 12 phase  2  f= 1.00 ph=0.785 sharp */ [   45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73 ],
      /* slot 195  bucket 12 phase  3  f= 1.00 ph=1.178 sharp */ [    7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22 ],
      /* slot 196  bucket 12 phase  4  f= 1.00 ph=1.571 sharp */ [    0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1 ],
      /* slot 197  bucket 12 phase  5  f= 1.00 ph=1.963 sharp */ [   -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1 ],
      /* slot 198  bucket 12 phase  6  f= 1.00 ph=2.356 sharp */ [  -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22 ],
      /* slot 199  bucket 12 phase  7  f= 1.00 ph=2.749 sharp */ [ -100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73 ],
      /* slot 200  bucket 12 phase  8  f= 1.00 ph=3.142 sharp */ [ -127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120 ],
      /* slot 201  bucket 12 phase  9  f= 1.00 ph=3.534 sharp */ [ -100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120 ],
      /* slot 202  bucket 12 phase 10  f= 1.00 ph=3.927 sharp */ [  -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73 ],
      /* slot 203  bucket 12 phase 11  f= 1.00 ph=4.320 sharp */ [   -7,  -1,   0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22 ],
      /* slot 204  bucket 12 phase 12  f= 1.00 ph=4.712 sharp */ [    0,   1,   7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1 ],
      /* slot 205  bucket 12 phase 13  f= 1.00 ph=5.105 sharp */ [    7,  22,  45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1 ],
      /* slot 206  bucket 12 phase 14  f= 1.00 ph=5.498 sharp */ [   45,  73, 100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22 ],
      /* slot 207  bucket 12 phase 15  f= 1.00 ph=5.890 sharp */ [  100, 120, 127, 120, 100,  73,  45,  22,   7,   1,   0,  -1,  -7, -22, -45, -73,-100,-120,-127,-120,-100, -73, -45, -22,  -7,  -1,   0,   1,   7,  22,  45,  73 ],
      /* slot 208  bucket 13 phase  0  f= 0.34 ph=0.000 sharp */ [  127, 126, 124, 119, 114, 107,  99,  90,  81,  71,  61,  52,  43,  34,  26,  20,  14,   9,   6,   3,   2,   1,   0,   0,   0,   0,  -1,  -2,  -3,  -6, -10, -14 ],
      /* slot 209  bucket 13 phase  1  f= 0.34 ph=0.393 sharp */ [  100,  91,  82,  72,  63,  53,  44,  35,  27,  20,  15,  10,   6,   4,   2,   1,   0,   0,   0,   0,  -1,  -1,  -3,  -6,  -9, -14, -19, -26, -33, -42, -51, -61 ],
      /* slot 210  bucket 13 phase  2  f= 0.34 ph=0.785 sharp */ [   45,  36,  28,  21,  15,  10,   7,   4,   2,   1,   0,   0,   0,   0,   0,  -1,  -3,  -5,  -9, -13, -18, -25, -32, -41, -50, -59, -69, -79, -88, -97,-105,-113 ],
      /* slot 211  bucket 13 phase  3  f= 0.34 ph=1.178 sharp */ [    7,   4,   2,   1,   0,   0,   0,   0,   0,  -1,  -3,  -5,  -8, -12, -18, -24, -31, -40, -49, -58, -68, -78, -87, -96,-104,-112,-118,-122,-125,-127,-127,-125 ],
      /* slot 212  bucket 13 phase  4  f= 0.34 ph=1.571 sharp */ [    0,   0,   0,  -1,  -2,  -5,  -8, -12, -17, -23, -30, -38, -47, -57, -66, -76, -86, -95,-103,-111,-117,-122,-125,-127,-127,-125,-122,-117,-110,-103, -94, -85 ],
      /* slot 213  bucket 13 phase  5  f= 0.34 ph=1.963 sharp */ [   -7, -11, -16, -22, -29, -37, -46, -55, -65, -75, -85, -94,-102,-110,-116,-121,-125,-127,-127,-125,-122,-117,-111,-104, -96, -86, -77, -67, -57, -48, -39, -31 ],
      /* slot 214  bucket 13 phase  6  f= 0.34 ph=2.356 sharp */ [  -45, -54, -64, -74, -83, -93,-101,-109,-116,-121,-124,-127,-127,-126,-123,-118,-112,-105, -97, -88, -78, -68, -59, -49, -40, -32, -24, -18, -13,  -8,  -5,  -3 ],
      /* slot 215  bucket 13 phase  7  f= 0.34 ph=2.749 sharp */ [ -100,-108,-115,-120,-124,-126,-127,-126,-123,-119,-113,-106, -98, -89, -79, -70, -60, -50, -41, -33, -25, -19, -13,  -9,  -5,  -3,  -1,   0,   0,   0,   0,   0 ],
      /* slot 216  bucket 13 phase  8  f= 0.34 ph=3.142 sharp */ [ -127,-126,-124,-119,-114,-107, -99, -90, -81, -71, -61, -52, -43, -34, -26, -20, -14,  -9,  -6,  -3,  -2,  -1,   0,   0,   0,   0,   1,   2,   3,   6,  10,  14 ],
      /* slot 217  bucket 13 phase  9  f= 0.34 ph=3.534 sharp */ [ -100, -91, -82, -72, -63, -53, -44, -35, -27, -20, -15, -10,  -6,  -4,  -2,  -1,   0,   0,   0,   0,   1,   1,   3,   6,   9,  14,  19,  26,  33,  42,  51,  61 ],
      /* slot 218  bucket 13 phase 10  f= 0.34 ph=3.927 sharp */ [  -45, -36, -28, -21, -15, -10,  -7,  -4,  -2,  -1,   0,   0,   0,   0,   0,   1,   3,   5,   9,  13,  18,  25,  32,  41,  50,  59,  69,  79,  88,  97, 105, 113 ],
      /* slot 219  bucket 13 phase 11  f= 0.34 ph=4.320 sharp */ [   -7,  -4,  -2,  -1,   0,   0,   0,   0,   0,   1,   3,   5,   8,  12,  18,  24,  31,  40,  49,  58,  68,  78,  87,  96, 104, 112, 118, 122, 125, 127, 127, 125 ],
      /* slot 220  bucket 13 phase 12  f= 0.34 ph=4.712 sharp */ [    0,   0,   0,   1,   2,   5,   8,  12,  17,  23,  30,  38,  47,  57,  66,  76,  86,  95, 103, 111, 117, 122, 125, 127, 127, 125, 122, 117, 110, 103,  94,  85 ],
      /* slot 221  bucket 13 phase 13  f= 0.34 ph=5.105 sharp */ [    7,  11,  16,  22,  29,  37,  46,  55,  65,  75,  85,  94, 102, 110, 116, 121, 125, 127, 127, 125, 122, 117, 111, 104,  96,  86,  77,  67,  57,  48,  39,  31 ],
      /* slot 222  bucket 13 phase 14  f= 0.34 ph=5.498 sharp */ [   45,  54,  64,  74,  83,  93, 101, 109, 116, 121, 124, 127, 127, 126, 123, 118, 112, 105,  97,  88,  78,  68,  59,  49,  40,  32,  24,  18,  13,   8,   5,   3 ],
      /* slot 223  bucket 13 phase 15  f= 0.34 ph=5.890 sharp */ [  100, 108, 115, 120, 124, 126, 127, 126, 123, 119, 113, 106,  98,  89,  79,  70,  60,  50,  41,  33,  25,  19,  13,   9,   5,   3,   1,   0,   0,   0,   0,   0 ],
      /* slot 224  bucket 14 phase  0  f= 0.00 ph=0.000 sine */ [  127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 ],
      /* slot 225  bucket 14 phase  1  f= 0.00 ph=0.393 sine */ [  117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117 ],
      /* slot 226  bucket 14 phase  2  f= 0.00 ph=0.785 sine */ [   90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90 ],
      /* slot 227  bucket 14 phase  3  f= 0.00 ph=1.178 sine */ [   49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49 ],
      /* slot 228  bucket 14 phase  4  f= 0.00 ph=1.571 sine */ [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot 229  bucket 14 phase  5  f= 0.00 ph=1.963 sine */ [  -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49 ],
      /* slot 230  bucket 14 phase  6  f= 0.00 ph=2.356 sine */ [  -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90 ],
      /* slot 231  bucket 14 phase  7  f= 0.00 ph=2.749 sine */ [ -117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117 ],
      /* slot 232  bucket 14 phase  8  f= 0.00 ph=3.142 sine */ [ -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127 ],
      /* slot 233  bucket 14 phase  9  f= 0.00 ph=3.534 sine */ [ -117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117,-117 ],
      /* slot 234  bucket 14 phase 10  f= 0.00 ph=3.927 sine */ [  -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90, -90 ],
      /* slot 235  bucket 14 phase 11  f= 0.00 ph=4.320 sine */ [  -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49, -49 ],
      /* slot 236  bucket 14 phase 12  f= 0.00 ph=4.712 sine */ [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
      /* slot 237  bucket 14 phase 13  f= 0.00 ph=5.105 sine */ [   49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49 ],
      /* slot 238  bucket 14 phase 14  f= 0.00 ph=5.498 sine */ [   90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90 ],
      /* slot 239  bucket 14 phase 15  f= 0.00 ph=5.890 sine */ [  117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117 ],
      /* slot 240  bucket 15 phase  0  f= 1.41 ph=0.000 sharp */ [  127, 113,  78,  39,  11,   1,   0,  -6, -28, -65,-104,-126,-120, -90, -50, -18,  -2,   0,   3,  19,  53,  92, 121, 125, 102,  63,  26,   5,   0,  -1, -12, -41 ],
      /* slot 241  bucket 15 phase  1  f= 1.41 ph=0.393 sharp */ [  100,  61,  25,   5,   0,  -1, -13, -42, -82,-116,-127,-110, -74, -35,  -9,   0,   0,   8,  31,  69, 107, 126, 118,  86,  46,  16,   2,   0,  -4, -22, -57, -96 ],
      /* slot 242  bucket 15 phase  2  f= 1.41 ph=0.785 sharp */ [   45,  15,   2,   0,  -4, -23, -58, -98,-124,-123, -97, -57, -22,  -4,   0,   2,  15,  46,  86, 118, 126, 107,  70,  32,   8,   0,   0,  -9, -35, -73,-110,-127 ],
      /* slot 243  bucket 15 phase  3  f= 1.41 ph=1.178 sharp */ [    7,   0,  -1, -10, -36, -75,-111,-127,-115, -81, -41, -12,  -1,   0,   5,  26,  62, 101, 125, 122,  93,  53,  20,   3,   0,  -2, -18, -50, -90,-120,-126,-104 ],
      /* slot 244  bucket 15 phase  4  f= 1.41 ph=1.571 sharp */ [    0,  -3, -19, -51, -91,-121,-125,-103, -64, -27,  -6,   0,   1,  12,  40,  79, 114, 127, 112,  77,  38,  11,   1,   0,  -7, -29, -66,-105,-126,-120, -89, -49 ],
      /* slot 245  bucket 15 phase  5  f= 1.41 ph=1.963 sharp */ [   -7, -30, -68,-106,-126,-119, -88, -48, -16,  -2,   0,   3,  21,  55,  95, 123, 124,  99,  60,  24,   5,   0,  -1, -14, -43, -83,-116,-127,-109, -73, -34,  -9 ],
      /* slot 246  bucket 15 phase  6  f= 1.41 ph=2.356 sharp */ [  -45, -85,-117,-127,-108, -71, -33,  -8,   0,   0,   9,  34,  72, 109, 127, 117,  84,  44,  14,   1,   0,  -4, -24, -59, -99,-124,-123, -96, -56, -22,  -4,   0 ],
      /* slot 247  bucket 15 phase  7  f= 1.41 ph=2.749 sharp */ [ -100,-124,-122, -94, -54, -20,  -3,   0,   2,  17,  49,  89, 119, 126, 105,  67,  30,   7,   0,  -1, -10, -37, -76,-112,-127,-114, -80, -40, -12,  -1,   0,   6 ],
      /* slot 248  bucket 15 phase  8  f= 1.41 ph=3.142 sharp */ [ -127,-113, -78, -39, -11,  -1,   0,   6,  28,  65, 104, 126, 120,  90,  50,  18,   2,   0,  -3, -19, -53, -92,-121,-125,-102, -63, -26,  -5,   0,   1,  12,  41 ],
      /* slot 249  bucket 15 phase  9  f= 1.41 ph=3.534 sharp */ [ -100, -61, -25,  -5,   0,   1,  13,  42,  82, 116, 127, 110,  74,  35,   9,   0,   0,  -8, -31, -69,-107,-126,-118, -86, -46, -16,  -2,   0,   4,  22,  57,  96 ],
      /* slot 250  bucket 15 phase 10  f= 1.41 ph=3.927 sharp */ [  -45, -15,  -2,   0,   4,  23,  58,  98, 124, 123,  97,  57,  22,   4,   0,  -2, -15, -46, -86,-118,-126,-107, -70, -32,  -8,   0,   0,   9,  35,  73, 110, 127 ],
      /* slot 251  bucket 15 phase 11  f= 1.41 ph=4.320 sharp */ [   -7,   0,   1,  10,  36,  75, 111, 127, 115,  81,  41,  12,   1,   0,  -5, -26, -62,-101,-125,-122, -93, -53, -20,  -3,   0,   2,  18,  50,  90, 120, 126, 104 ],
      /* slot 252  bucket 15 phase 12  f= 1.41 ph=4.712 sharp */ [    0,   3,  19,  51,  91, 121, 125, 103,  64,  27,   6,   0,  -1, -12, -40, -79,-114,-127,-112, -77, -38, -11,  -1,   0,   7,  29,  66, 105, 126, 120,  89,  49 ],
      /* slot 253  bucket 15 phase 13  f= 1.41 ph=5.105 sharp */ [    7,  30,  68, 106, 126, 119,  88,  48,  16,   2,   0,  -3, -21, -55, -95,-123,-124, -99, -60, -24,  -5,   0,   1,  14,  43,  83, 116, 127, 109,  73,  34,   9 ],
      /* slot 254  bucket 15 phase 14  f= 1.41 ph=5.498 sharp */ [   45,  85, 117, 127, 108,  71,  33,   8,   0,   0,  -9, -34, -72,-109,-127,-117, -84, -44, -14,  -1,   0,   4,  24,  59,  99, 124, 123,  96,  56,  22,   4,   0 ],
      /* slot 255  bucket 15 phase 15  f= 1.41 ph=5.890 sharp */ [  100, 124, 122,  94,  54,  20,   3,   0,  -2, -17, -49, -89,-119,-126,-105, -67, -30,  -7,   0,   1,  10,  37,  76, 112, 127, 114,  80,  40,  12,   1,   0,  -6 ]

    ];

    /// CURVE_TABLE_V — built from the 4 R2 picks of the iterative curve
    /// selection test plus their negations. 8 buckets × 16 phases.
    pub const CURVE_TABLE_V: [[i8; 32]; 128] = __CURVE_TABLE_V_RAW;

    #[rustfmt::skip]
    pub const SCALE_TABLE_BITS_V: [u16; 32] = [
    
        0x0575,0x168E,0x17CE,0x1850,0x18A3,0x18EA,0x1929,0x1964,
        0x199B,0x19CF,0x1A02,0x1A34,0x1A65,0x1A95,0x1AC6,0x1AF7,
        0x1B29,0x1B5D,0x1B92,0x1BC8,0x1C01,0x1C1F,0x1C3F,0x1C62,
        0x1C88,0x1CB3,0x1CE4,0x1D1E,0x1D67,0x1DCA,0x1E69,0x2008
    
    ];

    #[rustfmt::skip]
    const CENTROID_TABLE_BITS_V_OLD: [[u16; 8]; 32] = [

      /* scale_idx =  0  (7635 src blocks) */ [ 0xA9F3,0x9FE8,0x9C08,0x9500,0x1340,0x1B10,0x1F40,0x2C53 ],
      /* scale_idx =  1  (43800 src blocks) */ [ 0xB09E,0xA5C2,0xA19C,0x9A40,0x1BF0,0x21F4,0x25E8,0x31D1 ],
      /* scale_idx =  2  (37297 src blocks) */ [ 0xB10F,0xA7B6,0xA3AC,0x9C88,0x1D50,0x2402,0x27C4,0x3220 ],
      /* scale_idx =  3  (36644 src blocks) */ [ 0xB41E,0xA833,0xA454,0x9D50,0x1D78,0x245E,0x2844,0x311D ],
      /* scale_idx =  4  (36307 src blocks) */ [ 0xB26F,0xA88A,0xA4B2,0x9E20,0x1DA0,0x2498,0x2871,0x319C ],
      /* scale_idx =  5  (36217 src blocks) */ [ 0xB33F,0xA8AD,0xA4BA,0x9DC8,0x1E68,0x24F2,0x28B4,0x3214 ],
      /* scale_idx =  6  (36480 src blocks) */ [ 0xB2D5,0xA8F9,0xA534,0x9EA8,0x1E40,0x2516,0x28EE,0x32A6 ],
      /* scale_idx =  7  (35974 src blocks) */ [ 0xB39B,0xA924,0xA55A,0x9EE0,0x1E88,0x2546,0x2934,0x3329 ],
      /* scale_idx =  8  (36106 src blocks) */ [ 0xB487,0xA948,0xA57A,0x9EE8,0x1ED0,0x2560,0x2933,0x3372 ],
      /* scale_idx =  9  (36212 src blocks) */ [ 0xB3FB,0xA974,0xA5B2,0x9F28,0x1EF0,0x2592,0x2967,0x328C ],
      /* scale_idx = 10  (35935 src blocks) */ [ 0xB460,0xA9A7,0xA5CA,0x9EF8,0x1F50,0x25D0,0x29A7,0x3350 ],
      /* scale_idx = 11  (36276 src blocks) */ [ 0xB3CC,0xA9B1,0xA5EA,0x9FA8,0x1F20,0x25DE,0x29B9,0x3482 ],
      /* scale_idx = 12  (35923 src blocks) */ [ 0xB473,0xA9E1,0xA624,0x9FE0,0x1F90,0x2604,0x29CF,0x345B ],
      /* scale_idx = 13  (36137 src blocks) */ [ 0xB359,0xAA09,0xA626,0x9FF0,0x1FB8,0x2628,0x29FC,0x347A ],
      /* scale_idx = 14  (36011 src blocks) */ [ 0xB4AA,0xAA25,0xA666,0xA034,0x1FC8,0x263A,0x2A16,0x345F ],
      /* scale_idx = 15  (36307 src blocks) */ [ 0xB4AB,0xAA47,0xA6A0,0xA054,0x1F68,0x2640,0x2A18,0x33E8 ],
      /* scale_idx = 16  (35915 src blocks) */ [ 0xB437,0xAA53,0xA69E,0xA064,0x2000,0x2676,0x2A38,0x34BF ],
      /* scale_idx = 17  (36203 src blocks) */ [ 0xB4DB,0xAA76,0xA6C8,0xA080,0x1F80,0x268C,0x2A7A,0x34E7 ],
      /* scale_idx = 18  (35969 src blocks) */ [ 0xB478,0xAAA2,0xA6D2,0xA054,0x2038,0x26D0,0x2A95,0x3564 ],
      /* scale_idx = 19  (36317 src blocks) */ [ 0xB4E4,0xAAC1,0xA704,0xA094,0x201C,0x26C0,0x2AA6,0x34E3 ],
      /* scale_idx = 20  (35929 src blocks) */ [ 0xB4AA,0xAAD5,0xA700,0xA074,0x2034,0x26EA,0x2AD4,0x35EA ],
      /* scale_idx = 21  (36227 src blocks) */ [ 0xB4C6,0xAAF8,0xA760,0xA0E0,0x2044,0x26E8,0x2ACB,0x3516 ],
      /* scale_idx = 22  (36215 src blocks) */ [ 0xB48E,0xAB21,0xA774,0xA0D8,0x2068,0x2732,0x2B00,0x34F1 ],
      /* scale_idx = 23  (36199 src blocks) */ [ 0xB51A,0xAB47,0xA786,0xA0BC,0x2084,0x2754,0x2B3F,0x34A4 ],
      /* scale_idx = 24  (36183 src blocks) */ [ 0xB4D3,0xAB64,0xA7A8,0xA108,0x2084,0x275C,0x2B4D,0x3484 ],
      /* scale_idx = 25  (36006 src blocks) */ [ 0xB4F6,0xAB82,0xA7A2,0xA0C4,0x20D0,0x27BC,0x2B7A,0x350C ],
      /* scale_idx = 26  (36250 src blocks) */ [ 0xB4BC,0xABB0,0xA808,0xA138,0x2094,0x27BA,0x2B93,0x3555 ],
      /* scale_idx = 27  (36361 src blocks) */ [ 0xB51A,0xAC10,0xA834,0xA15C,0x20E8,0x2812,0x2BEA,0x35C0 ],
      /* scale_idx = 28  (36642 src blocks) */ [ 0xB560,0xAC3C,0xA851,0xA1B4,0x2120,0x282C,0x2C27,0x3548 ],
      /* scale_idx = 29  (37210 src blocks) */ [ 0xB4D0,0xAC78,0xA886,0xA188,0x21C8,0x2883,0x2C6F,0x34C8 ],
      /* scale_idx = 30  (38828 src blocks) */ [ 0xB4DC,0xAD2F,0xA921,0xA27C,0x21D4,0x2902,0x2D19,0x3436 ],
      /* scale_idx = 31  (12493 src blocks) */ [ 0xB457,0xABCD,0xA844,0xA0F8,0x21E0,0x285B,0x2BD9,0x3435 ]

    ];

    /// CENTROID_TABLE_BITS_V — 16-entry table built by duplicating the 8-entry
    /// calibration. See the note on the K-side equivalent.
    pub const CENTROID_TABLE_BITS_V: [[u16; 16]; 32] = {
        let mut out = [[0u16; 16]; 32];
        let mut s = 0;
        while s < 32 {
            let mut j = 0;
            while j < 8 {
                out[s][j * 2] = CENTROID_TABLE_BITS_V_OLD[s][j];
                out[s][j * 2 + 1] = CENTROID_TABLE_BITS_V_OLD[s][j];
                j += 1;
            }
            s += 1;
        }
        out
    };

    #[rustfmt::skip]
    pub const PEAK_CURVE_INDICES_V: [u8; 256] = [
    
          0,  1,  2,  8,  9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34, 35,
         40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59, 64, 65, 66, 72,
         73, 74, 80, 81, 82, 88, 89, 90, 96, 97,104,105,112,113,114,115,
        120,121,122,123,128,129,130,136,137,138,144,145,152,153,160,161,
        162,166,167,168,169,170,174,175,176,177,184,185,192,200,208,209,
        216,217,224,225,226,227,228,229,230,231,232,233,234,235,236,237,
        238,239,240,248,163,164,165,171,172,173,199,207,246,254,198,206,
        151,159,197,205,215,223,103,111,243,251, 71, 79,183,191,196,204,
          7, 15,195,203,241,249,135,143,214,222,194,202, 23, 31,150,158,
         87, 95,193,201,102,110,182,190, 70, 78,213,221,244,252, 55, 63,
        119,127,149,157,  6, 14,101,109,181,189,212,220,247,255,134,142,
         69, 77,148,156, 22, 30,245,253, 39, 47, 86, 94,211,219,100,108,
          3,  4,  5, 11, 12, 13, 19, 20, 21, 27, 28, 29, 36, 37, 38, 44,
         45, 46, 52, 53, 54, 60, 61, 62, 67, 68, 75, 76, 83, 84, 85, 91,
         92, 93, 98, 99,106,107,116,117,118,124,125,126,131,132,133,139,
        140,141,146,147,154,155,178,179,180,186,187,188,210,218,242,250
    
    ];

    #[rustfmt::skip]
    pub const PEAK_BIN_OFFSETS_V: [u16; 33] = [
    
          0,100,106,108,110,112,112,118,122,128,128,134,138,140,144,148,152,152,158,160,162,164,166,168,172,174,176,180,184,188,190,192,256
    
    ];
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
                ys[i * QK_Q0_X + j] = v as f32 / 127.0;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            // 1. bulk_anchor = INT8-encoded block mean
            let mean: f32 = block.iter().sum::<f32>() / Self::BLCK_SIZE as f32;
            let bulk_int = (mean * 127.0).round().clamp(-127.0, 127.0) as i32;
            let bulk_anchor = bulk_int as i8;

            // 2. argmax(|x_i_i8 - bulk|) â†’ outlier_idx; track its residual
            let mut outlier_idx: usize = 0;
            let mut outlier_residual: i32 = 0;
            let mut best_abs: i32 = -1;
            for (j, &x) in block.iter().enumerate() {
                let xi = (x * 127.0).round().clamp(-127.0, 127.0) as i32;
                let r = xi - bulk_int;
                let ar = r.abs();
                if ar > best_abs {
                    best_abs = ar;
                    outlier_idx = j;
                    outlier_residual = r;
                }
            }

            // 3. Coarse delta in [-4, 3]
            let delta_raw = (outlier_residual as f32 / Q0_X_S_OUTLIER as f32).round() as i32;
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

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            let c0 = decode_e4m3(x.val_fp8[0]);
            let c1 = decode_e4m3(x.val_fp8[1]);
            for j in 0..QK_Q0_M2 {
                let val = if (x.qmask >> (j / 4)) & 1 == 0 {
                    c0
                } else {
                    c1
                };
                ys[i * QK_Q0_M2 + j] = val;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            let vmin = block.iter().cloned().fold(f32::INFINITY, f32::min);
            let vmax = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut c0 = vmin;
            let mut c1 = vmax;
            for _ in 0..4 {
                let mut s0 = 0.0f32;
                let mut n0 = 0.0f32;
                let mut s1 = 0.0f32;
                let mut n1 = 0.0f32;
                for &x in block {
                    if (x - c1).abs() < (x - c0).abs() {
                        s1 += x;
                        n1 += 1.0;
                    } else {
                        s0 += x;
                        n0 += 1.0;
                    }
                }
                if n0 > 0.0 {
                    c0 = s0 / n0;
                }
                if n1 > 0.0 {
                    c1 = s1 / n1;
                }
            }
            let mut qmask = 0u8;
            for q in 0..8usize {
                let mean: f32 = block[q * 4..(q + 1) * 4].iter().sum::<f32>() / 4.0;
                if (mean - c1).abs() < (mean - c0).abs() {
                    qmask |= 1 << q;
                }
            }
            y.val_fp8[0] = encode_e4m3(c0);
            y.val_fp8[1] = encode_e4m3(c1);
            y.qmask = qmask;
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

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            let c = [
                decode_e4m3(x.val_fp8[0]),
                decode_e4m3(x.val_fp8[1]),
                decode_e4m3(x.val_fp8[2]),
                decode_e4m3(x.val_fp8[3]),
            ];
            for j in 0..QK_Q0_M4 {
                let k = ((x.qmask >> (2 * (j / 2))) & 3) as usize;
                ys[i * QK_Q0_M4 + j] = c[k];
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let block = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            // Pair means drive the quantization at the format's actual
            // reconstruction granularity (2 elements share one centroid).
            let pair_means: [f32; 16] =
                std::array::from_fn(|p| (block[p * 2] + block[p * 2 + 1]) * 0.5);

            let vmin = pair_means.iter().cloned().fold(f32::INFINITY, f32::min);
            let vmax = pair_means.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = (vmax - vmin) / 3.0;
            let mut c = [vmin, vmin + step, vmin + 2.0 * step, vmax];

            // Lloyd at pair granularity. Centroid update uses the underlying
            // elements (sums all 32 contributions over assigned pairs);
            // equivalent to the mean of assigned pair_means.
            for _ in 0..5 {
                let mut sums = [0.0f32; 4];
                let mut counts = [0.0f32; 4];
                for p in 0..16usize {
                    let mean = pair_means[p];
                    let k = c
                        .iter()
                        .enumerate()
                        .min_by(|a, b| (mean - a.1).abs().partial_cmp(&(mean - b.1).abs()).unwrap())
                        .unwrap()
                        .0;
                    sums[k] += block[p * 2] + block[p * 2 + 1];
                    counts[k] += 2.0;
                }
                for k in 0..4 {
                    if counts[k] > 0.0 {
                        c[k] = sums[k] / counts[k];
                    }
                }
            }

            let mut qmask = 0u32;
            for (p, &mean) in pair_means.iter().enumerate().take(16) {
                let k = c
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (mean - a.1).abs().partial_cmp(&(mean - b.1).abs()).unwrap())
                    .unwrap()
                    .0 as u32;
                qmask |= k << (2 * p);
            }
            for (dst, &v) in y.val_fp8.iter_mut().zip(c.iter()) {
                *dst = encode_e4m3(v);
            }
            y.qmask = qmask;
        }
    }

    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        0.0
    }
}

impl GgmlType for BlockQ2_K {
    const DTYPE: GgmlDType = GgmlDType::Q2_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q2k_q8k(n, xs, ys);

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
                for j in 0..QK_K / 16 {
                    block.scales[j] = 0;
                }
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

            for (y_block, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
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
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q3k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q3k_q8k(n, xs, ys);

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
            for (j, x_scale_slice) in x.chunks_exact(16).enumerate() {
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
            for (y, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                let mut shift = 0;
                for shift_scoped_y in y.chunks_exact_mut(32) {
                    for (scale_index, scale_scoped_y) in
                        shift_scoped_y.chunks_exact_mut(16).enumerate()
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
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q4k_q8k(n, xs, ys);

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

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
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

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5_K {
    const DTYPE: GgmlDType = GgmlDType::Q5_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q5k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q5k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q5k_q8k: {n} is not divisible by {QK_K}"
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
            let q5 = &x.qs;
            let hm = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a = &mut aux8[..];
            let mut q5 = &q5[..];
            let mut m = 1u8;

            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q5[l] & 0xF) as i8;
                    a[l] += if hm[l] & m != 0 { 16 } else { 0 };
                }
                a = &mut a[32..];
                m <<= 1;
                for l in 0..32 {
                    a[l] = (q5[l] >> 4) as i8;
                    a[l] += if hm[l] & m != 0 { 16 } else { 0 };
                }
                a = &mut a[32..];
                m <<= 1;
                q5 = &q5[32..];
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

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L793
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(31, 5, x_scale_slice);
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
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let ll = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = ll.clamp(0, 31) as u8;
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0;
            let mut u1 = 1;
            let mut u2 = 2;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let ql = &ql[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u1 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d1 * ((ql & 0xF) as f32 + to_add) - m1;
                    ys_index += 1;
                }
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u2 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d2 * ((ql >> 4) as f32 + to_add) - m2;
                    ys_index += 1;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
}

impl GgmlType for BlockQ6_K {
    const DTYPE: GgmlDType = GgmlDType::Q6_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q6k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q6k_q8k: {n} is not divisible by {QK_K}"
        );

        let mut aux8 = [0i8; QK_K];
        let mut aux16 = [0i16; 8];
        let mut sums = [0f32; 8];
        let mut aux32 = [0f32; 8];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let q4 = &x.ql;
            let qh = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0f32);

            for j in (0..QK_K).step_by(128) {
                let aux8 = &mut aux8[j..];
                let q4 = &q4[j / 2..];
                let qh = &qh[j / 4..];
                for l in 0..32 {
                    aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 32] =
                        (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 96] =
                        (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                }
            }

            for (j, &scale) in x.scales.iter().enumerate() {
                let scale = scale as f32;
                let q8 = &q8[16 * j..];
                let aux8 = &aux8[16 * j..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
                let q8 = &q8[8..];
                let aux8 = &aux8[8..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
            }

            let d = x.d.to_f32() * y.d;
            for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
                *sum += a * d;
            }
        }
        sums.iter().sum()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len() * Self::BLCK_SIZE,
            "quantize_row_q6k: size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in ys.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale = make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1);
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q6k: {k} is not divisible by {QK_K}"
        );

        for (idx_x, x) in xs.iter().enumerate() {
            let d = x.d.to_f32();
            let ql = &x.ql;
            let qh = &x.qh;
            let sc = &x.scales;
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ys = &mut ys[idx_x * QK_K + n..];
                let sc = &sc[8 * idx..];
                let ql = &ql[64 * idx..];
                let qh = &qh[32 * idx..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    ys[l] = d * sc[is] as f32 * q1 as f32;
                    ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                    ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        }
    }
}

impl GgmlType for BlockQ8_K {
    const DTYPE: GgmlDType = GgmlDType::Q8_K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8_K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q8k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q8k_q8k: {n} is not divisible by {QK_K}"
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
            sumf += sum_i as f32 * xs.d * ys.d
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "quantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        for (i, y) in ys.iter_mut().enumerate() {
            let mut max = 0f32;
            let mut amax = 0f32;
            let xs = &xs[i * QK_K..(i + 1) * QK_K];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            if amax == 0f32 {
                y.d = 0f32;
                y.qs.fill(0)
            } else {
                let iscale = -128f32 / max;
                for (j, q) in y.qs.iter_mut().enumerate() {
                    // ggml uses nearest_int with bit magic here, maybe we want the same
                    // but we would have to test and benchmark it.
                    let v = (iscale * xs[j]).round();
                    *q = v.min(127.) as i8
                }
                for j in 0..QK_K / 16 {
                    let mut sum = 0i32;
                    for ii in 0..16 {
                        sum += y.qs[j * 16 + ii] as i32
                    }
                    y.bsums[j] = sum as i16
                }
                y.d = 1.0 / iscale
            }
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        for (i, x) in xs.iter().enumerate() {
            for (j, &q) in x.qs.iter().enumerate() {
                ys[i * QK_K + j] = x.d * q as f32
            }
        }
    }
}

// =============================================================================
// AWQ (ACTIVATION-AWARE WEIGHT QUANTIZATION) IMPLEMENTATIONS
// =============================================================================
// AWQ dequant formula: w = scale * (q - zero)
// where q is 4-bit unsigned [0,15], extracted from packed u32 nibbles.
//
// Nibble packing: each u32 contains 8 Ã— 4-bit weights
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

impl GgmlType for f64 {
    const DTYPE: GgmlDType = GgmlDType::F64;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f64;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs * *ys;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f64;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for bf16 {
    const DTYPE: GgmlDType = GgmlDType::BF16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = bf16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_bf16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u64 {
    const DTYPE: GgmlDType = GgmlDType::U64;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u64;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u64;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i64 {
    const DTYPE: GgmlDType = GgmlDType::I64;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i64;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i64;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u32 {
    const DTYPE: GgmlDType = GgmlDType::U32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u32;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i32 {
    const DTYPE: GgmlDType = GgmlDType::I32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i32;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u16 {
    const DTYPE: GgmlDType = GgmlDType::U16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u16;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i16 {
    const DTYPE: GgmlDType = GgmlDType::I16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i16;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for u8 {
    const DTYPE: GgmlDType = GgmlDType::U8;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = u8;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as u8;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
}

impl GgmlType for i8 {
    const DTYPE: GgmlDType = GgmlDType::I8;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = i8;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f64;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            res += *xs as f64 * *ys as f64;
        }
        res as f32
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as i8;
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (ys, xs) in ys.iter_mut().zip(xs.iter()) {
            *ys = *xs as f32;
        }
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }
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

#[cfg(test)]
mod mxfp4_tests {
    use super::{e8m0_to_f32_half, BlockMXFP4, BlockQ8_0, GgmlType, MXFP4_KVALUES, QK_MXFP4};
    use crate::quantized::GgmlDType;

    /// The 17-byte block layout `{ e: u8, qs: [u8; 16] }` is what we assert against —
    /// `#[repr(C)]` makes the field bytes the on-disk bytes.
    #[test]
    fn block_is_17_bytes() {
        assert_eq!(std::mem::size_of::<BlockMXFP4>(), 17);
        assert_eq!(GgmlDType::MXFP4.type_size(), 17);
        assert_eq!(GgmlDType::MXFP4.block_size(), 32);
    }

    /// `ggml_e8m0_to_fp32_half`: exact power-of-two decode incl. the subnormal `e<2` path.
    #[test]
    fn e8m0_half_exact() {
        assert_eq!(e8m0_to_f32_half(129), 2.0); // 2^(129-128)
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(125), 0.125); // 2^-3
        assert_eq!(e8m0_to_f32_half(1), f32::from_bits(0x0040_0000)); // 2^-127
        assert_eq!(e8m0_to_f32_half(0), f32::from_bits(0x0020_0000)); // 2^-128
    }

    /// GGUF on-disk file code for MXFP4 is 39 (matches llama.cpp `GGML_TYPE_MXFP4`), and
    /// it round-trips through our translation boundary.
    #[test]
    fn gguf_file_code_is_39() {
        assert_eq!(GgmlDType::MXFP4.to_gguf_file_code(), 39);
        assert_eq!(
            GgmlDType::from_gguf_file_code(39).unwrap(),
            GgmlDType::MXFP4
        );
        assert_eq!(GgmlDType::from_u32(49).unwrap(), GgmlDType::MXFP4);
    }

    /// Encode a block whose values are all exactly representable at `amax = 6.0`
    /// (⇒ `e = 127`, `d = 0.5`, real table `{0,.5,1,1.5,2,3,4,6}`), and assert the exact
    /// bytes produced, then that dequant recovers the inputs bit-for-bit.
    #[test]
    fn exact_bytes_and_roundtrip() {
        let mut xs = vec![0f32; QK_MXFP4];
        // Low nibbles come from positions 0..16, high nibbles from 16..32.
        xs[0] = 6.0; // idx 7  (kvalues 12 * 0.5)
        xs[1] = 3.0; // idx 5  (6 * 0.5)
        xs[2] = 1.5; // idx 3  (3 * 0.5)
        xs[3] = 0.5; // idx 1  (1 * 0.5)
        xs[16] = -6.0; // idx 15
        xs[17] = -3.0; // idx 13
        xs[18] = 4.0; // idx 6  (8 * 0.5)
        xs[19] = -0.5; // idx 9

        let mut blk = [BlockMXFP4::zeros()];
        BlockMXFP4::from_float(&xs, &mut blk);

        assert_eq!(blk[0].e, 127);
        assert_eq!(blk[0].qs[0], 7 | (15 << 4));
        assert_eq!(blk[0].qs[1], 5 | (13 << 4));
        assert_eq!(blk[0].qs[2], 3 | (6 << 4));
        assert_eq!(blk[0].qs[3], 1 | (9 << 4));
        for j in 4..16 {
            assert_eq!(blk[0].qs[j], 0, "qs[{j}] should be zero");
        }

        let mut ys = vec![0f32; QK_MXFP4];
        BlockMXFP4::to_float(&blk, &mut ys);
        assert_eq!(ys[0], 6.0);
        assert_eq!(ys[1], 3.0);
        assert_eq!(ys[2], 1.5);
        assert_eq!(ys[3], 0.5);
        assert_eq!(ys[16], -6.0);
        assert_eq!(ys[17], -3.0);
        assert_eq!(ys[18], 4.0);
        assert_eq!(ys[19], -0.5);
        for &y in ys.iter().take(16).skip(4) {
            assert_eq!(y, 0.0);
        }
    }

    /// Ties in `best_index_mxfp4` resolve to the lower index (ggml scans ascending and
    /// keeps strictly-smaller errors). At `d = 0.5`, `2.5` is equidistant from `2.0`
    /// (idx 4) and `3.0` (idx 5) ⇒ idx 4.
    #[test]
    fn quant_tie_breaks_low() {
        let mut xs = vec![0f32; QK_MXFP4];
        xs[0] = 6.0; // pin amax so e = 127, d = 0.5
        xs[1] = 2.5; // tie between idx 4 (2.0) and idx 5 (3.0)
        let mut blk = [BlockMXFP4::zeros()];
        BlockMXFP4::from_float(&xs, &mut blk);
        assert_eq!(blk[0].e, 127);
        assert_eq!(blk[0].qs[1] & 0x0F, 4);
    }

    /// Decode a hand-built raw block (bytes chosen directly) and check every lane,
    /// exercising the low/high nibble split and a non-0.5 scale (`e = 125 ⇒ d = 0.125`).
    #[test]
    fn decode_known_raw_block() {
        let mut blk = BlockMXFP4::zeros();
        blk.e = 125; // d = 0.125
        blk.qs[0] = 6 | (7 << 4); // low idx 6 -> 8, high idx 7 -> 12
        blk.qs[5] = 14 | (2 << 4); // low idx 14 -> -8, high idx 2 -> 2
        let mut ys = vec![0f32; QK_MXFP4];
        BlockMXFP4::to_float(std::slice::from_ref(&blk), &mut ys);
        assert_eq!(ys[0], 8.0 * 0.125); // 1.0
        assert_eq!(ys[16], 12.0 * 0.125); // 1.5
        assert_eq!(ys[5], -8.0 * 0.125); // -1.0
        assert_eq!(ys[21], 2.0 * 0.125); // 0.25
    }

    /// `vec_dot` against Q8_0 activations equals the exact integer-table dot scaled by the
    /// two per-block scales — verified against a from-scratch f32 reference on the
    /// dequantized operands.
    #[test]
    fn vec_dot_matches_dequant_reference() {
        let mut wq = vec![0f32; QK_MXFP4];
        let mut aq = vec![0f32; QK_MXFP4];
        for i in 0..QK_MXFP4 {
            // Values chosen to be exactly representable so the reference is unambiguous.
            wq[i] = MXFP4_KVALUES[i % 8] as f32 * 0.5;
            aq[i] = ((i as f32) - 16.0) * 0.03;
        }
        wq[0] = 6.0; // pin weight amax -> e = 127
        let mut wblk = [BlockMXFP4::zeros()];
        BlockMXFP4::from_float(&wq, &mut wblk);
        let mut ablk = [BlockQ8_0::zeros()];
        BlockQ8_0::from_float(&aq, &mut ablk);

        let got = BlockMXFP4::vec_dot(QK_MXFP4, &wblk, &ablk);

        // Reference: dequantize both operands and dot in f32.
        let mut wdq = vec![0f32; QK_MXFP4];
        let mut adq = vec![0f32; QK_MXFP4];
        BlockMXFP4::to_float(&wblk, &mut wdq);
        BlockQ8_0::to_float(&ablk, &mut adq);
        let want: f32 = wdq.iter().zip(adq.iter()).map(|(a, b)| a * b).sum();
        assert!((got - want).abs() < 1e-4, "got {got} want {want}");
    }
}
