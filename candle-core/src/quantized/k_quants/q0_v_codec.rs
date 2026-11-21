//! The Q0_V block: a two-byte codec that picks a curve, a scale and a centroid
//! from calibrated tables, with separate tables for K and V.
//!
//! The 16 bits hold a 7-bit curve index, a 5-bit scale index and a 4-bit centroid
//! index. Decode scales the curve entry and adds the centroid as one FMA, and the
//! CUDA read paths use the same arithmetic. The encoder takes the block's mean and
//! largest deviation, picks the nearest scale row and centroid, then the curve
//! with the least squared error. The reference encoder produces the same bytes as
//! the CUDA encoder, which an oracle test checks. `IS_K` selects the K or V
//! tables, and `to_le_bytes` exposes the two raw bytes for external codecs.

use super::*;

/// Q0_V: Pattern-indexed structured quantization (2 bytes per block,
/// 0.53 BPE effective with the per-group 4-byte header amortised).
///
/// Per-block layout (manually packed):
///   `deltas`  — bits[3:0]=mag_delta (4-bit), bits[7:4]=out_delta (4-bit)
///   `pattern` — bits[3:0]=sign_pattern (4-bit), bits[7:4]=shape (4-bit)
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

/// IS_K-aware decode of a single Q0_V block element. Pure-Rust reference
/// for the `q0_v_elem<IS_K>` CUDA function in
/// `candle-kernels/src/convert/block_q0_v.cuh`.
#[inline]
pub fn q0_v_elem<const IS_K: bool>(block: &BlockQ0V, e: usize) -> f32 {
    q0_v_elem_scaled::<IS_K>(block, e, 1.0)
}

/// A Q0_V element read at a palette scale, given its reciprocal `r`
/// (`1.0 / scale`, correctly rounded): `(scale_q · r) · curve[e] + centroid_q · r`
/// as one FMA, the two products rounded first. This is every GPU read path's
/// arithmetic (`q0_v_header` in block_q0_v.cuh); at `r = 1` it is the plain
/// decode.
#[inline]
pub fn q0_v_elem_scaled<const IS_K: bool>(block: &BlockQ0V, e: usize, r: f32) -> f32 {
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
    let scale = half::f16::from_bits(scale_bits).to_f32() * r;
    let centroid = half::f16::from_bits(centroid_bits).to_f32() * r;
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
/// This is the reference for the CUDA encoder in
/// `candle-kernels/src/quantize/quantize_q0_v.cuh`, which produces the same
/// bytes: every operation below is the one the kernel performs, rounded the
/// same way, in the same order (the encode oracle test compares them).
///   1. Compute actual (centroid, scale) = (mean(x), max|x − mean(x)|). The
///      sum is the warp's butterfly: five rounds of `x[i] + x[i ^ off]`.
///   2. Pick scale_idx (32 entries) by argmin |actual_scale − 127·scale[i]|.
///   3. Pick centroid_idx (16 entries within the chosen scale row).
///   4. Normalise target[e] = x[e] · (1/scale_baked) − centroid · (1/scale_baked),
///      where scale_baked = scale_norm / 127, so target is directly comparable
///      to the i8 curve_table values.
///   5. Score all 128 curves by E − 2R (E = Σ curve², R = Σ target · curve as
///      four FMA chains, element e into chain e mod 4 in element order, summed
///      (c0 + c1) + (c2 + c3)) — the squared error less the Σ target² every
///      curve shares — and keep the lowest; ties keep the lower index.
///   6. Pack curve_idx (7 bits) + scale_idx (5 bits) + centroid_idx (4 bits).
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
    // The kernel's warp sum: each round, lane i adds lane i ^ off's partial.
    let mut partial: [f32; QK_Q0_V] = std::array::from_fn(|i| block[i]);
    for off in [16, 8, 4, 2, 1] {
        partial = std::array::from_fn(|i| partial[i] + partial[i ^ off]);
    }
    let actual_centroid = partial[0] * (1.0 / 32.0);
    let actual_scale = block
        .iter()
        .fold(0.0f32, |m, &x| m.max((x - actual_centroid).abs()));

    // ── Step 2: pick scale_idx (32 entries) ──
    // Stored value = scale_norm / 127, so multiply by 127 for comparison.
    let mut best_scale_idx = 0u8;
    let mut best_scale_err = f32::INFINITY;
    for (i, &raw) in scale_table.iter().enumerate() {
        let scale_baked = half::f16::from_bits(raw).to_f32();
        let err = (actual_scale - scale_baked * 127.0).abs();
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

    // ── Step 4: target = (block − centroid) / scale_baked ──
    // In i8 [-127, +127] space, directly comparable to the raw curve values.
    let inv_scale = 1.0 / chosen_scale_baked;
    let neg_c_inv = -chosen_centroid * inv_scale;
    let target: [f32; QK_Q0_V] = std::array::from_fn(|i| block[i].mul_add(inv_scale, neg_c_inv));

    // ── Step 5: every curve, by E − 2R ──
    let mut best_curve_idx = 0u8;
    let mut best_curve_err = f32::INFINITY;
    for (c, curve) in curve_table.iter().enumerate() {
        let mut chain = [0.0f32; 4];
        let mut energy = 0i32;
        for (e, (&t, &v)) in target.iter().zip(curve.iter()).enumerate() {
            chain[e & 3] = t.mul_add(v as f32, chain[e & 3]);
            energy += v as i32 * v as i32;
        }
        let r = (chain[0] + chain[1]) + (chain[2] + chain[3]);
        let err = (-2.0f32).mul_add(r, energy as f32);
        if err < best_curve_err {
            best_curve_err = err;
            best_curve_idx = c as u8;
        }
    }

    BlockQ0V::pack(best_curve_idx, best_scale_idx, best_centroid_idx)
}
