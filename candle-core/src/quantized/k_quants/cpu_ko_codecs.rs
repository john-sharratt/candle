//! CPU reference codecs for the KO block layouts, which are byte-permuted twins
//! of the K-quant compact blocks.
//!
//! Each `BlockQ*_KO` holds the same data as its K-quant compact block, reordered
//! so the quant words come first and the scale pairs group at the tail. These
//! codecs serve CPU roundtrip tests and `type_size`/`block_size`. They do not
//! read the layout the int8 matmul consumes: GPU-produced `Q4_KO` weights use a
//! per-128 affine layout under the same `GgmlDType` discriminant, so a GPU KO
//! weight must not be dequantized through these codecs.

use super::*;

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
pub(super) const Q4_KO_SHIFTS: [u32; 8] = [0, 16, 8, 24, 4, 20, 12, 28];

// Within a sub, struct slot k holds the original qs int [0,2,1,3][k] (the [I0,I2,I1,I3]
// interleave that makes a lane's qlo/qhi adjacent for the int2 load). Self-inverse.
pub(super) const Q4_KO_PERM: [usize; 4] = [0, 2, 1, 3];

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

// Q5_KO / Q6_KO / Q8_KO — CPU-side per-32 codecs (byte-permuted twins of the
// Q5_K/Q6_K/Q8_K compact blocks). As with BlockQ4_KO above, these are CPU
// reference/roundtrip codecs only; the int8 matmul consumes the distinct
// per-128 from-F32 `ko_quant` layout under the same GgmlDType discriminant.

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
