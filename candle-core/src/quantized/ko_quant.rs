//! F32 → KO weight quantizers (and their inverse dequant) in the **lane-major per-128**
//! layout the int8 KO matmul kernels read.
//!
//! KO weights are NOT a byte-permute of a stored K-quant: they use a per-128 affine
//! `(scale, min)` (one per 128 K per row), which Q4_K/Q5_K/… don't carry — so they are
//! re-quantized straight from F32. This is an OFFLINE, one-shot weight prep (CPU is fine;
//! the hot path is the matmul, not this). The chunk layout matches the kernel's
//! `dequant_all_subs_int8` exactly: each lane's 4 sub-uint32s are contiguous so the kernel
//! pulls them in one wide LDS.
//!
//!   chunk = [ ql 512B | crumb (Q6) / hi (Q5) | dm (fp16 scale,min ×8) ]
//!   ql byte (lane L = r*4+q3, sub, i): L*16 + sub*4 + i = lo(q[k]) | hi(q[k+16])<<4
//!
//! Q8_KO is symmetric (min=0), 8-bit, with two int4 regions (b_frag[0]/b_frag[1]).

use crate::quantized::GgmlDType;
use half::f16;

/// (maxq, crumb_bytes, hi_bytes) for the affine KO formats. Q8_KO is handled separately.
fn ko_params(dtype: GgmlDType) -> Option<(i32, usize, usize)> {
    match dtype {
        GgmlDType::Q4_KO => Some((15, 0, 0)),
        GgmlDType::Q5_KO => Some((31, 0, 128)),
        GgmlDType::Q6_KO => Some((63, 256, 0)),
        _ => None,
    }
}

/// Bytes in one k1024 chunk (8 rows × 128 K) for a KO dtype.
pub fn ko_chunk_bytes(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_KO => 512 + 32,       // ql + dm
        GgmlDType::Q5_KO => 512 + 128 + 32, // ql + hi + dm
        GgmlDType::Q6_KO => 512 + 256 + 32, // ql + crumb + dm
        GgmlDType::Q8_KO => 1024 + 32,      // b0 + b1 + dm
        _ => 0,
    }
}

/// Quantize F32 weights `[nrows × ncols]` (row-major) → the lane-major KO chunk tensor the
/// int8 KO matmul reads. `nrows` (N, output rows) must be a multiple of 8, `ncols` (K) a
/// multiple of 128. Returns the packed chunk bytes (k_blocks × row_groups × chunk_bytes).
pub fn quantize_ko(w: &[f32], nrows: usize, ncols: usize, dtype: GgmlDType) -> Vec<u8> {
    assert_eq!(nrows % 8, 0, "KO quantize: nrows must be a multiple of 8");
    assert_eq!(
        ncols % 128,
        0,
        "KO quantize: ncols must be a multiple of 128"
    );
    assert_eq!(w.len(), nrows * ncols, "KO quantize: data length mismatch");
    if dtype == GgmlDType::Q8_KO {
        return quantize_q8_ko(w, nrows, ncols);
    }
    let (maxq, crumb_bytes, hi_bytes) =
        ko_params(dtype).unwrap_or_else(|| panic!("not a KO dtype: {dtype:?}"));
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let crumb_base = 512;
    let hi_base = 512 + crumb_bytes;
    let dm_base = 512 + crumb_bytes + hi_bytes;
    let chunk_bytes = dm_base + 32;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let wbase = row * ncols + k_blk * 128;
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for kk in 0..128 {
                    let v = w[wbase + kk];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / maxq as f32).max(1e-12);
                let mut q = [0u8; 128];
                for kk in 0..128 {
                    q[kk] = (((w[wbase + kk] - mn) / scale).round() as i32).clamp(0, maxq) as u8;
                }
                for sub in 0..4 {
                    for p in 0..16 {
                        let lo = q[sub * 32 + p] & 0xF;
                        let hi = q[sub * 32 + 16 + p] & 0xF;
                        ob[cbase + (r * 4 + p / 4) * 16 + sub * 4 + (p % 4)] = lo | (hi << 4);
                    }
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        if crumb_bytes > 0 {
                            let (mut cr0, mut cr1) = (0u8, 0u8);
                            for j in 0..4 {
                                cr0 |= (((q[sub * 32 + q3 * 4 + j] >> 4) & 0x3) << (2 * j)) as u8;
                                cr1 |=
                                    (((q[sub * 32 + 16 + q3 * 4 + j] >> 4) & 0x3) << (2 * j)) as u8;
                            }
                            ob[cbase + crumb_base + lane * 8 + sub * 2] = cr0;
                            ob[cbase + crumb_base + lane * 8 + sub * 2 + 1] = cr1;
                        }
                        if hi_bytes > 0 {
                            let (mut hb0, mut hb1) = (0u8, 0u8);
                            for j in 0..4 {
                                hb0 |= (((q[sub * 32 + q3 * 4 + j] >> 4) & 1) << j) as u8;
                                hb1 |= (((q[sub * 32 + 16 + q3 * 4 + j] >> 4) & 1) << j) as u8;
                            }
                            ob[cbase + hi_base + lane * 4 + sub] = hb0 | (hb1 << 4);
                        }
                    }
                }
                let d = cbase + dm_base + r * 4;
                ob[d..d + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[d + 2..d + 4].copy_from_slice(&f16::from_f32(mn).to_le_bytes());
            }
        }
    }
    ob
}

/// Q8_KO: symmetric 8-bit (min=0), lane-major split into b_frag[0] region [0,512) and
/// b_frag[1] region [512,1024), each `lane*16 + sub*4`. dm = (scale, 0) per row.
fn quantize_q8_ko(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let chunk_bytes = 1024 + 32;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let wbase = row * ncols + k_blk * 128;
                let mut amax = 0f32;
                for kk in 0..128 {
                    amax = amax.max(w[wbase + kk].abs());
                }
                let scale = (amax / 127.0).max(1e-12);
                let id = 1.0 / scale;
                for sub in 0..4 {
                    for q3 in 0..4 {
                        let base = (r * 4 + q3) * 16 + sub * 4;
                        for i in 0..4 {
                            let kc0 = sub * 32 + q3 * 4 + i;
                            let kc1 = sub * 32 + 16 + q3 * 4 + i;
                            let q0 = (w[wbase + kc0] * id).round().clamp(-127.0, 127.0) as i8;
                            let q1 = (w[wbase + kc1] * id).round().clamp(-127.0, 127.0) as i8;
                            ob[cbase + base + i] = q0 as u8;
                            ob[cbase + 512 + base + i] = q1 as u8;
                        }
                    }
                }
                let d = cbase + 1024 + r * 4;
                ob[d..d + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[d + 2..d + 4].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
            }
        }
    }
    ob
}

/// Inverse of [`quantize_ko`] — reconstruct F32 `[nrows × ncols]` from the lane-major KO
/// chunk tensor (the per-128 dequant `W = scale·q + min`). Used to validate the quantizer.
pub fn dequant_ko(chunk: &[u8], nrows: usize, ncols: usize, dtype: GgmlDType) -> Vec<f32> {
    if dtype == GgmlDType::Q8_KO {
        return dequant_q8_ko(chunk, nrows, ncols);
    }
    let (maxq, crumb_bytes, hi_bytes) =
        ko_params(dtype).unwrap_or_else(|| panic!("not a KO dtype: {dtype:?}"));
    let _ = maxq;
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let crumb_base = 512;
    let hi_base = 512 + crumb_bytes;
    let dm_base = 512 + crumb_bytes + hi_bytes;
    let chunk_bytes = dm_base + 32;
    let mut out = vec![0f32; nrows * ncols];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let d = cbase + dm_base + r * 4;
                let scale = f16::from_le_bytes([chunk[d], chunk[d + 1]]).to_f32();
                let mn = f16::from_le_bytes([chunk[d + 2], chunk[d + 3]]).to_f32();
                for sub in 0..4 {
                    for kk in 0..32 {
                        let half = kk / 16; // 0 = low nibble / cr0 / hb0, 1 = high
                        let p = kk % 16;
                        let q3 = p / 4;
                        let i = p % 4;
                        let lane = r * 4 + q3;
                        let qlb = chunk[cbase + lane * 16 + sub * 4 + i] as u32;
                        let mut qv = if half == 0 {
                            qlb & 0xF
                        } else {
                            (qlb >> 4) & 0xF
                        };
                        if crumb_bytes > 0 {
                            let off = cbase + crumb_base + lane * 8 + sub * 2;
                            let cr = if half == 0 {
                                chunk[off]
                            } else {
                                chunk[off + 1]
                            } as u32;
                            qv |= ((cr >> (2 * i)) & 0x3) << 4;
                        }
                        if hi_bytes > 0 {
                            let hb = chunk[cbase + hi_base + lane * 4 + sub] as u32;
                            let nib = if half == 0 { hb & 0xF } else { (hb >> 4) & 0xF };
                            qv |= ((nib >> i) & 1) << 4;
                        }
                        let k = sub * 32 + kk;
                        out[row * ncols + k_blk * 128 + k] = scale * qv as f32 + mn;
                    }
                }
            }
        }
    }
    out
}

fn dequant_q8_ko(chunk: &[u8], nrows: usize, ncols: usize) -> Vec<f32> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let chunk_bytes = 1024 + 32;
    let mut out = vec![0f32; nrows * ncols];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let d = cbase + 1024 + r * 4;
                let scale = f16::from_le_bytes([chunk[d], chunk[d + 1]]).to_f32();
                for sub in 0..4 {
                    for q3 in 0..4 {
                        let base = (r * 4 + q3) * 16 + sub * 4;
                        for i in 0..4 {
                            let q0 = chunk[cbase + base + i] as i8;
                            let q1 = chunk[cbase + 512 + base + i] as i8;
                            let kc0 = sub * 32 + q3 * 4 + i;
                            let kc1 = sub * 32 + 16 + q3 * 4 + i;
                            out[row * ncols + k_blk * 128 + kc0] = scale * q0 as f32;
                            out[row * ncols + k_blk * 128 + kc1] = scale * q1 as f32;
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..a.len() {
            let d = (a[i] - b[i]) as f64;
            num += d * d;
            den += (b[i] as f64).powi(2);
        }
        (num / den.max(1e-12)).sqrt()
    }

    // Deterministic pseudo-random f32 in [-1, 1] (no rng dependency in core tests).
    fn pseudo(n: usize) -> Vec<f32> {
        let mut s = 0x2545F491u32;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Quantize F32 → KO → dequant → compare. Each format's rel-L2 must be at its per-128
    /// affine floor (≈ 1/maxq / √3), confirming the lane-major pack ↔ dequant round-trips.
    #[test]
    fn ko_quantize_dequant_roundtrip() {
        let (nrows, ncols) = (64usize, 256usize);
        let w = pseudo(nrows * ncols);
        // (dtype, rel-L2 ceiling) — the per-128 affine quant error (≈1/maxq/√3), shrinking
        // with bit width; matches the dense bench's "vs f32 ground truth" column.
        for &(dtype, ceil) in &[
            (GgmlDType::Q4_KO, 0.080), // ~0.065
            (GgmlDType::Q5_KO, 0.050), // ~0.033
            (GgmlDType::Q6_KO, 0.025), // ~0.017
            (GgmlDType::Q8_KO, 0.007), // ~0.004
        ] {
            let chunk = quantize_ko(&w, nrows, ncols, dtype);
            assert_eq!(
                chunk.len() % ko_chunk_bytes(dtype),
                0,
                "{dtype:?} chunk size"
            );
            let de = dequant_ko(&chunk, nrows, ncols, dtype);
            let rel = rel_l2(&de, &w);
            assert!(rel < ceil, "{dtype:?} round-trip rel_l2 {rel:.5} ≥ {ceil}");
        }
    }
}
