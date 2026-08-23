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

use crate::quantized::k_quants::{e8m0_to_f32_half, BlockMXFP4, GgmlType, MXFP4_KVALUES};
use crate::quantized::GgmlDType;
use half::f16;
use rayon::prelude::*;

/// Bytes in one MXFP4_KO k1024 chunk (8 rows × 128 K): 512 B of nibbles (the reordered
/// MXFP4 `qs`) + 32 B of per-sub E8M0 scales (4 per row).
pub const MXFP4_KO_CHUNK_BYTES: usize = 512 + 32;

/// Repack an F32 weight `[nrows × ncols]` into the **MXFP4_KO** int8-matmul layout: MXFP4
/// per-32 quantization (E2M1 nibbles + E8M0 scale, symmetric) placed in the lane-major
/// `ql` layout the int8 KO kernel reads, with the four per-sub E8M0 scales stored in the
/// `dm` region (no min — the E2M1 codebook is centered). Unlike the affine KO twins this
/// is **exact** relative to plain MXFP4 (it only reorders + splits scales, no requant).
///
/// `nrows % 8 == 0`, `ncols % 128 == 0`. The int8 kernel maps each nibble → `MXFP4_KVALUES`
/// (an int8 in `[-12, 12]`) and folds the per-sub scale after the int8 MMA.
pub fn quantize_mxfp4_ko(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    assert_eq!(nrows % 8, 0, "MXFP4_KO: nrows must be a multiple of 8");
    assert_eq!(ncols % 128, 0, "MXFP4_KO: ncols must be a multiple of 128");
    assert_eq!(w.len(), nrows * ncols, "MXFP4_KO: data length mismatch");
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let chunk_bytes = MXFP4_KO_CHUNK_BYTES;
    let dm_base = 512;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    let mut blk = [BlockMXFP4::zeros()];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let wbase = row * ncols + k_blk * 128;
                for sub in 0..4 {
                    // Quantize this 32-element sub-block to one MXFP4 block.
                    BlockMXFP4::from_float(&w[wbase + sub * 32..wbase + sub * 32 + 32], &mut blk);
                    ob[cbase + dm_base + r * 4 + sub] = blk[0].e;
                    // The KO ql byte at lane (r,q3), sub, i holds MXFP4 qs[q3*4+i] verbatim:
                    //   low nibble = element (q3*4+i), high nibble = element 16+(q3*4+i).
                    for p in 0..16 {
                        let q3 = p / 4;
                        let i = p % 4;
                        ob[cbase + (r * 4 + q3) * 16 + sub * 4 + i] = blk[0].qs[p];
                    }
                }
            }
        }
    }
    ob
}

/// Bytes in one MXFP4_KO **GPU** k1024 chunk (matches `block_c_mxfp4_k1024`): the 544-byte
/// CPU chunk (`ql` + per-sub `e`) plus 32 B of the per-row collapsed scale `dm` (8 × `half2`).
pub const MXFP4_KO_GPU_CHUNK_BYTES: usize = MXFP4_KO_CHUNK_BYTES + 32;

/// Expand the CPU MXFP4_KO chunk (512 `ql` + 32 `e`, from [`quantize_mxfp4_ko`]) into the
/// on-GPU `block_c_mxfp4_k1024` layout by appending, per 8-row chunk, the per-row scale
/// `dm[8]`: `dm[row] = (2^(e_max-128), 0)` where `e_max` is the max of that row's four
/// per-sub E8M0 bytes. The int8 kernel's per-sub fold reads the `e` bytes directly and does
/// NOT read `dm` — it is baked only so the chunk layout (and the pack-file fingerprint over
/// it) stays fixed. `min = 0` (the E2M1 codebook is centred).
pub fn mxfp4_ko_to_gpu_chunk(cpu_chunk: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let n_chunks = (nrows / 8) * (ncols / 128);
    assert_eq!(
        cpu_chunk.len(),
        n_chunks * MXFP4_KO_CHUNK_BYTES,
        "mxfp4_ko_to_gpu_chunk: cpu_chunk length mismatch"
    );
    let dm_base = 512; // per-sub E8M0 bytes start here in both layouts
    let mut out = vec![0u8; n_chunks * MXFP4_KO_GPU_CHUNK_BYTES];
    for c in 0..n_chunks {
        let src = &cpu_chunk[c * MXFP4_KO_CHUNK_BYTES..(c + 1) * MXFP4_KO_CHUNK_BYTES];
        let dst = &mut out[c * MXFP4_KO_GPU_CHUNK_BYTES..(c + 1) * MXFP4_KO_GPU_CHUNK_BYTES];
        dst[..MXFP4_KO_CHUNK_BYTES].copy_from_slice(src); // ql + e verbatim
        for r in 0..8 {
            let e_max = (0..4).map(|sub| src[dm_base + r * 4 + sub]).max().unwrap();
            let scale = f16::from_f32(e8m0_to_f32_half(e_max));
            let base = MXFP4_KO_CHUNK_BYTES + r * 4; // dm[r] = (scale half, min half)
            dst[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            dst[base + 2..base + 4].copy_from_slice(&f16::ZERO.to_le_bytes());
        }
    }
    out
}

/// Reorder **native** MXFP4 GGUF bytes (`[nrows × ncols]`, row-major [`BlockMXFP4`]:
/// `e:u8` then `qs:[u8;16]`, 17 B / 32 elems) straight into the on-GPU
/// `block_c_mxfp4_k1024` layout — an **exact** byte permutation (nibbles + E8M0 bytes
/// copied verbatim) plus the per-row `dm` bake (unread by the kernel — kept for layout
/// stability, see [`mxfp4_ko_to_gpu_chunk`]). Unlike [`quantize_mxfp4_ko`] (which
/// re-quantizes from F32), this takes the already-quantized experts and only reorders
/// them, so it is lossless vs the source and keeps the weights 4-bit. `nrows % 8 == 0`,
/// `ncols % 128 == 0`. Returns the 576-B-per-1024 chunk tensor the int8 MXFP4_KO matmul
/// reads.
pub fn mxfp4_native_to_ko_gpu_chunk(native_bytes: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    assert_eq!(
        nrows % 8,
        0,
        "MXFP4_KO reorder: nrows must be a multiple of 8"
    );
    assert_eq!(
        ncols % 128,
        0,
        "MXFP4_KO reorder: ncols must be a multiple of 128"
    );
    const BLK: usize = 17; // sizeof(BlockMXFP4): 1 e byte + 16 qs bytes
    let blocks_per_row = ncols / 32;
    assert_eq!(
        native_bytes.len(),
        nrows * blocks_per_row * BLK,
        "MXFP4_KO reorder: native byte length mismatch"
    );
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let dm_base = 512;
    let mut ob = vec![0u8; k_blocks * row_groups * MXFP4_KO_GPU_CHUNK_BYTES];
    // Each 576-B chunk (8 rows × 128 K) writes a disjoint output region and reads disjoint
    // native blocks, so the reorder parallelizes across chunks with no coordination. Within a
    // chunk the 16 `qs` bytes of each 32-block land as FOUR contiguous 4-byte runs (one per q3:
    // dest (r*4+q3)*16 + sub*4 .. +4 ← native qs[q3*4 .. +4]), so we copy 4 bytes at a time
    // instead of 16 scalar stores. Chunk index c = k_blk*row_groups + g.
    ob.par_chunks_mut(MXFP4_KO_GPU_CHUNK_BYTES)
        .enumerate()
        .for_each(|(c, dst)| {
            let k_blk = c / row_groups;
            let g = c % row_groups;
            for r in 0..8 {
                let row = g * 8 + r;
                for sub in 0..4 {
                    // Native block covering this row's K-columns [k_blk*128+sub*32 .. +32].
                    let blk = (row * blocks_per_row + k_blk * 4 + sub) * BLK;
                    dst[dm_base + r * 4 + sub] = native_bytes[blk]; // e byte
                    for q3 in 0..4 {
                        let d = (r * 4 + q3) * 16 + sub * 4;
                        let s = blk + 1 + q3 * 4;
                        dst[d..d + 4].copy_from_slice(&native_bytes[s..s + 4]);
                    }
                }
                // Per-row collapsed scale dm[r] = (2^(e_max-128), 0).
                let e_max = (0..4).map(|sub| dst[dm_base + r * 4 + sub]).max().unwrap();
                let scale = f16::from_f32(e8m0_to_f32_half(e_max));
                let base = MXFP4_KO_CHUNK_BYTES + r * 4;
                dst[base..base + 2].copy_from_slice(&scale.to_le_bytes());
                dst[base + 2..base + 4].copy_from_slice(&f16::ZERO.to_le_bytes());
            }
        });
    ob
}

/// Reference **int8** matmul over the MXFP4_KO chunk — the exact CPU mirror of the in-kernel
/// per-sub fold. Activations `act` `[m, ncols]` are quantized to int8 per 128-K block
/// (symmetric, `amax/127`), the weight nibbles are mapped to their int8 `MXFP4_KVALUES`, and
/// each 32-K sub is accumulated in int32 and folded with **its own** per-sub E8M0 weight
/// scale × the per-128 activation scale (no min — symmetric). The FP order mirrors the
/// kernel operation-for-operation: the coefficient `wscale·ascale` is rounded once, then
/// FMA'd into the running output (`coeff.mul_add(isum, out)`), sub ascending within each
/// 128-K block, blocks ascending — the only divergence left against the GPU is the
/// activation scale's storage representation (q8a128 keeps it in f16). Returns `out[m, nrows]`.
pub fn mxfp4_ko_int8_matmul(
    chunk: &[u8],
    act: &[f32],
    nrows: usize,
    ncols: usize,
    m: usize,
) -> Vec<f32> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let dm_base = 512;
    let cb = MXFP4_KO_CHUNK_BYTES;

    // Quantize activations to int8 per (token, 128-K block): [m][k_blocks] -> (int8[128], scale).
    let mut a_i8 = vec![0i8; m * ncols];
    let mut a_scale = vec![0f32; m * k_blocks];
    for t in 0..m {
        for kb in 0..k_blocks {
            let base = t * ncols + kb * 128;
            let amax = (0..128).fold(0f32, |mx, i| mx.max(act[base + i].abs()));
            let scale = (amax / 127.0).max(1e-12);
            a_scale[t * k_blocks + kb] = scale;
            for i in 0..128 {
                a_i8[base + i] = (act[base + i] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    let mut out = vec![0f32; m * nrows];
    for kb in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (kb * row_groups + g) * cb;
            for r in 0..8 {
                let row = g * 8 + r;
                // Decode this row's 4 per-sub E8M0 scales.
                let mut wscale = [0f32; 4];
                for sub in 0..4 {
                    wscale[sub] = e8m0_to_f32_half(chunk[cbase + dm_base + r * 4 + sub]);
                }
                for t in 0..m {
                    let ascale = a_scale[t * k_blocks + kb];
                    for sub in 0..4 {
                        let mut isum = 0i32;
                        for p in 0..16 {
                            let q3 = p / 4;
                            let i = p % 4;
                            let qs = chunk[cbase + (r * 4 + q3) * 16 + sub * 4 + i];
                            let w0 = MXFP4_KVALUES[(qs & 0x0F) as usize] as i32;
                            let w1 = MXFP4_KVALUES[(qs >> 4) as usize] as i32;
                            let k0 = kb * 128 + sub * 32 + p;
                            let k1 = kb * 128 + sub * 32 + 16 + p;
                            isum += w0 * a_i8[t * ncols + k0] as i32;
                            isum += w1 * a_i8[t * ncols + k1] as i32;
                        }
                        // Coefficient rounded once, then one FMA — the kernel's exact op order.
                        let o = &mut out[t * nrows + row];
                        *o = (wscale[sub] * ascale).mul_add(isum as f32, *o);
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`quantize_mxfp4_ko`] — reconstruct F32 from the MXFP4_KO chunk
/// (`value = MXFP4_KVALUES[nibble] · 2^(e-128)`). Used as the exact CPU baseline the int8
/// kernel is compared against.
pub fn dequant_mxfp4_ko(chunk: &[u8], nrows: usize, ncols: usize) -> Vec<f32> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let chunk_bytes = MXFP4_KO_CHUNK_BYTES;
    let dm_base = 512;
    let mut out = vec![0f32; nrows * ncols];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                for sub in 0..4 {
                    let e = chunk[cbase + dm_base + r * 4 + sub];
                    let scale = e8m0_to_f32_half(e);
                    for p in 0..16 {
                        let q3 = p / 4;
                        let i = p % 4;
                        let qs = chunk[cbase + (r * 4 + q3) * 16 + sub * 4 + i];
                        let lo = MXFP4_KVALUES[(qs & 0x0F) as usize] as f32;
                        let hi = MXFP4_KVALUES[(qs >> 4) as usize] as f32;
                        let k0 = sub * 32 + p;
                        let k1 = sub * 32 + 16 + p;
                        out[row * ncols + k_blk * 128 + k0] = lo * scale;
                        out[row * ncols + k_blk * 128 + k1] = hi * scale;
                    }
                }
            }
        }
    }
    out
}

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
        GgmlDType::Q2_KO => 256 + 32,       // crumb (2-bit values) + dm
        GgmlDType::Q4_KO => 512 + 32,       // ql + dm
        GgmlDType::Q5_KO => 512 + 128 + 32, // ql + hi + dm
        GgmlDType::Q6_KO => 512 + 256 + 32, // ql + crumb + dm
        GgmlDType::Q8_KO => 1024 + 32,      // b0 + b1 + dm
        GgmlDType::MXFP4_KO => MXFP4_KO_GPU_CHUNK_BYTES, // 512 ql + 32 e + 32 dm
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
    if dtype == GgmlDType::Q2_KO {
        return quantize_q2_ko(w, nrows, ncols);
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
                                cr0 |= ((q[sub * 32 + q3 * 4 + j] >> 4) & 0x3) << (2 * j);
                                cr1 |= ((q[sub * 32 + 16 + q3 * 4 + j] >> 4) & 0x3) << (2 * j);
                            }
                            ob[cbase + crumb_base + lane * 8 + sub * 2] = cr0;
                            ob[cbase + crumb_base + lane * 8 + sub * 2 + 1] = cr1;
                        }
                        if hi_bytes > 0 {
                            let (mut hb0, mut hb1) = (0u8, 0u8);
                            for j in 0..4 {
                                hb0 |= ((q[sub * 32 + q3 * 4 + j] >> 4) & 1) << j;
                                hb1 |= ((q[sub * 32 + 16 + q3 * 4 + j] >> 4) & 1) << j;
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
    if dtype == GgmlDType::Q2_KO {
        return dequant_q2_ko(chunk, nrows, ncols);
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

/// Q2_KO: 2-bit affine KO twin — per-128 `(scale, min)`, value 0..3. Layout per 8-row × 128-K
/// chunk (288 B): a 256 B **crumb** region + 32 B `dm`. The crumb region is byte-identical to the
/// high-2-bit crumb region `Q6_KO` already carries — for `(lane = r*4 + q3, sub)`, `cr0` at
/// `lane*8 + sub*2` packs the 4 LOW-half values (`q[sub*32 + q3*4 + j]`, 2 bits at `2j`) and
/// `cr1` at `+1` the 4 HIGH-half values (`q[sub*32 + 16 + q3*4 + j]`) — but here the crumb IS the
/// whole value (Q6 stores only the high 2 bits of a 6-bit value). One row's 32 B = one
/// `block_c_q2_KO_k128`. `dm[r] = (scale, min)` f16 at `256 + r*4`.
fn quantize_q2_ko(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let dm_base = 256;
    let chunk_bytes = dm_base + 32; // 288
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
                let scale = ((mx - mn) / 3.0).max(1e-12);
                let mut q = [0u8; 128];
                for kk in 0..128 {
                    q[kk] = (((w[wbase + kk] - mn) / scale).round() as i32).clamp(0, 3) as u8;
                }
                for sub in 0..4 {
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        let (mut cr0, mut cr1) = (0u8, 0u8);
                        for j in 0..4 {
                            cr0 |= (q[sub * 32 + q3 * 4 + j] & 0x3) << (2 * j);
                            cr1 |= (q[sub * 32 + 16 + q3 * 4 + j] & 0x3) << (2 * j);
                        }
                        ob[cbase + lane * 8 + sub * 2] = cr0;
                        ob[cbase + lane * 8 + sub * 2 + 1] = cr1;
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

/// Inverse of [`quantize_q2_ko`] — reconstruct F32 `W = scale·q + min` from the crumb chunk.
fn dequant_q2_ko(chunk: &[u8], nrows: usize, ncols: usize) -> Vec<f32> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let dm_base = 256;
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
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        let cr0 = chunk[cbase + lane * 8 + sub * 2] as u32;
                        let cr1 = chunk[cbase + lane * 8 + sub * 2 + 1] as u32;
                        for j in 0..4 {
                            let v0 = (cr0 >> (2 * j)) & 0x3;
                            let v1 = (cr1 >> (2 * j)) & 0x3;
                            let k0 = sub * 32 + q3 * 4 + j;
                            let k1 = sub * 32 + 16 + q3 * 4 + j;
                            out[row * ncols + k_blk * 128 + k0] = scale * v0 as f32 + mn;
                            out[row * ncols + k_blk * 128 + k1] = scale * v1 as f32 + mn;
                        }
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

    /// The exact per-32 int8 matmul reference tracks the float baseline (MXFP4-dequant ×
    /// float activations) closely — its only error is the 8-bit activation quant, since the
    /// per-32 weight scales are applied exactly (no per-128 collapse). This is the
    /// "best-case int8" the per-128 optimized kernel is measured against.
    #[test]
    fn mxfp4_ko_int8_matmul_tracks_float() {
        let (nrows, ncols, m) = (16usize, 256usize, 8usize);
        let w = pseudo(nrows * ncols);
        let act: Vec<f32> = pseudo(m * ncols); // ~[-1, 1]
        let chunk = quantize_mxfp4_ko(&w, nrows, ncols);

        // Exact per-32 int8 matmul.
        let int8 = mxfp4_ko_int8_matmul(&chunk, &act, nrows, ncols, m);

        // Float baseline: dequantized MXFP4 weight × float activations.
        let wdq = dequant_mxfp4_ko(&chunk, nrows, ncols);
        let mut base = vec![0f32; m * nrows];
        for t in 0..m {
            for row in 0..nrows {
                let mut s = 0f32;
                for k in 0..ncols {
                    s += act[t * ncols + k] * wdq[row * ncols + k];
                }
                base[t * nrows + row] = s;
            }
        }
        let rel = rel_l2(&int8, &base);
        // Only the per-128 8-bit activation quant contributes; the weight side is exact.
        assert!(
            rel < 0.02,
            "exact per-32 int8 vs float baseline rel_l2 {rel:.5} too high"
        );
    }

    /// The per-sub int8 fold is exact on the weight side regardless of per-128 exponent spread —
    /// each 32-K sub carries its own E8M0 scale into the fold, so an outlier channel that lands
    /// the four subs on wildly different exponents costs NOTHING (the failure mode a shared
    /// per-128 scale would truncate on). Only the 8-bit activation quant contributes error.
    /// `--nocapture` to see the numbers.
    #[test]
    fn mxfp4_persub_int8_exact_under_exponent_spread() {
        let (nrows, ncols, m) = (16usize, 256usize, 8usize);
        // Induce EXPONENT SPREAD (what a shared per-128 scale would truncate on): scale each contiguous 32-block
        // by a per-block power of two spanning ~6 exponents, so the 4 subs of each 128-tile land on
        // different E8M0 scales — the adversarial case an outlier channel would create. (Uniform
        // random weights give spread 0 → a trivially lossless test that proves nothing.)
        let mut w = pseudo(nrows * ncols);
        let mut s = 0x9E3779B9u32;
        for blk in 0..(nrows * ncols / 32) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let e = (s % 7) as i32 - 3; // per-32 exponent offset in [-3, 3]
            let mul = 2f32.powi(e);
            for i in 0..32 {
                w[blk * 32 + i] *= mul;
            }
        }
        let act: Vec<f32> = pseudo(m * ncols);
        let chunk = quantize_mxfp4_ko(&w, nrows, ncols);

        let per32 = mxfp4_ko_int8_matmul(&chunk, &act, nrows, ncols, m);

        // Float baseline: exact MXFP4-dequant × float activations.
        let wdq = dequant_mxfp4_ko(&chunk, nrows, ncols);
        let mut base = vec![0f32; m * nrows];
        for t in 0..m {
            for row in 0..nrows {
                let mut s = 0f32;
                for k in 0..ncols {
                    s += act[t * ncols + k] * wdq[row * ncols + k];
                }
                base[t * nrows + row] = s;
            }
        }

        // Report the per-128 exponent-spread distribution (0 = all four subs share e_max → lossless).
        let mut spread_hist = [0usize; 8];
        let (row_groups, k_blocks, cb, dm_base) =
            (nrows / 8, ncols / 128, MXFP4_KO_CHUNK_BYTES, 512);
        for kb in 0..k_blocks {
            for g in 0..row_groups {
                let cbase = (kb * row_groups + g) * cb;
                for r in 0..8 {
                    let e: [u8; 4] = std::array::from_fn(|s| chunk[cbase + dm_base + r * 4 + s]);
                    let spread = (*e.iter().max().unwrap() - *e.iter().min().unwrap()) as usize;
                    spread_hist[spread.min(7)] += 1;
                }
            }
        }

        let rel_per32 = rel_l2(&per32, &base);
        println!("per-sub int8 vs float:       rel_l2 = {rel_per32:.5}");
        println!("per-128 exponent spread histogram (bins 0..7): {spread_hist:?}");

        // Spread must actually be present (else this test proves nothing) …
        assert!(
            spread_hist[1..].iter().sum::<usize>() > 0,
            "adversarial weight construction produced zero exponent spread"
        );
        // … and the per-sub fold must not care: activation-quant-level error only, same
        // bound the spread-free test holds.
        assert!(
            rel_per32 < 0.02,
            "per-sub int8 rel_l2 {rel_per32:.5} too high under exponent spread"
        );
    }

    /// MXFP4_KO is an **exact** repack of plain MXFP4: quantizing to MXFP4_KO and
    /// dequantizing must equal quantizing each contiguous 32-block to MXFP4 and decoding
    /// it — bit-for-bit (the KO layout only reorders nibbles and splits out the per-sub
    /// E8M0 scales; there is no requantization).
    #[test]
    fn mxfp4_ko_is_exact_repack_of_mxfp4() {
        let (nrows, ncols) = (16usize, 256usize);
        let w = pseudo(nrows * ncols);
        let chunk = quantize_mxfp4_ko(&w, nrows, ncols);
        assert_eq!(chunk.len() % MXFP4_KO_CHUNK_BYTES, 0);
        let de = dequant_mxfp4_ko(&chunk, nrows, ncols);

        // Reference: MXFP4-quantize each contiguous 32-block per row, then decode.
        let mut want = vec![0f32; nrows * ncols];
        let mut blk = [BlockMXFP4::zeros()];
        let mut ys = [0f32; 32];
        for row in 0..nrows {
            for b in 0..(ncols / 32) {
                let base = row * ncols + b * 32;
                BlockMXFP4::from_float(&w[base..base + 32], &mut blk);
                BlockMXFP4::to_float(&blk, &mut ys);
                want[base..base + 32].copy_from_slice(&ys);
            }
        }
        let maxdiff = de
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(maxdiff, 0.0, "MXFP4_KO repack must be bit-exact vs MXFP4");
    }

    /// The native-MXFP4→GPU reorder (`mxfp4_native_to_ko_gpu_chunk`, the engine's expert
    /// stage-in) must produce the EXACT same GPU chunk bytes as the from-F32 path
    /// (`quantize_mxfp4_ko` → `mxfp4_ko_to_gpu_chunk`). Both start from the same
    /// `BlockMXFP4::from_float`, so the reorder — which only permutes nibbles + copies the
    /// E8M0 bytes verbatim and bakes the per-row `dm` — must be byte-identical. This pins the
    /// engine's exact (no-requant) repack against the reference packer. Adversarial per-32
    /// exponent spread exercises non-trivial `dm` (e_max) values.
    #[test]
    fn mxfp4_native_reorder_matches_from_f32_bytes() {
        let (nrows, ncols) = (24usize, 384usize);
        let mut w = pseudo(nrows * ncols);
        let mut s = 0x1234_5678u32;
        for blk in 0..(nrows * ncols / 32) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let mul = 2f32.powi((s % 7) as i32 - 3);
            for i in 0..32 {
                w[blk * 32 + i] *= mul;
            }
        }

        // From-F32 reference chunk (544 CPU chunk → 576 GPU chunk with dm baked).
        let chunk544 = quantize_mxfp4_ko(&w, nrows, ncols);
        let from_f32 = mxfp4_ko_to_gpu_chunk(&chunk544, nrows, ncols);

        // Native GGUF bytes: per-32 MXFP4 blocks, row-major (e byte + 16 qs bytes each).
        let bpr = ncols / 32;
        let mut native = Vec::with_capacity(nrows * bpr * 17);
        let mut blk = [BlockMXFP4::zeros()];
        for row in 0..nrows {
            for kb in 0..bpr {
                let base = row * ncols + kb * 32;
                BlockMXFP4::from_float(&w[base..base + 32], &mut blk);
                native.push(blk[0].e);
                native.extend_from_slice(&blk[0].qs);
            }
        }
        let from_native = mxfp4_native_to_ko_gpu_chunk(&native, nrows, ncols);

        assert_eq!(
            from_native, from_f32,
            "native MXFP4 reorder diverged from the from-F32 GPU chunk bytes"
        );
        assert_eq!(
            from_native.len(),
            (nrows / 8) * (ncols / 128) * MXFP4_KO_GPU_CHUNK_BYTES
        );
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
            (GgmlDType::Q2_KO, 0.400), // ~0.325 (≈1/3, the 2-bit affine floor — 4 levels)
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

    /// Q2_KO RAW BYTES (per the codec rule — exact bytes, not a tolerance). One 8×128 chunk with
    /// `w[r][k] = (k % 4)` → per row min=0, max=3, scale=1, `q[k] = k % 4`. Each `(lane, sub)`
    /// crumb byte then packs values (0,1,2,3) at bit positions (0,2,4,6) = `0b11_10_01_00 = 0xE4`
    /// for BOTH cr0 (low half) and cr1 (high half), since `q` is periodic mod 4. dm = (1.0, 0.0)
    /// = f16 `0x3C00`,`0x0000`. Pins the crumb pack + dm layout the CUDA `block_c_q2_KO` loader reads.
    #[test]
    fn q2_ko_raw_bytes() {
        let (nrows, ncols) = (8usize, 128usize);
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|i| ((i % ncols) % 4) as f32)
            .collect();
        let chunk = quantize_ko(&w, nrows, ncols, GgmlDType::Q2_KO);
        assert_eq!(chunk.len(), 288, "one 8×128 chunk = 256 crumb + 32 dm");
        for (i, b) in chunk[..256].iter().enumerate() {
            assert_eq!(
                *b, 0xE4,
                "crumb byte {i} must be 0xE4 (values 0,1,2,3 packed)"
            );
        }
        for r in 0..8 {
            let d = 256 + r * 4;
            assert_eq!(
                &chunk[d..d + 4],
                &[0x00, 0x3C, 0x00, 0x00],
                "dm row {r} = (scale 1.0, min 0.0) f16"
            );
        }
        // On-grid input → dequant is EXACT (scale·q + min with q ∈ {0,1,2,3}, scale=1, min=0).
        let de = dequant_ko(&chunk, nrows, ncols, GgmlDType::Q2_KO);
        assert_eq!(de, w, "Q2_KO dequant must be exact on the 4-level grid");
    }

    /// Q2_KO crumb byte varies correctly with the quantized values: a row whose 128 values step
    /// 0,1,2,3,0,… but OFFSET so a specific (lane,sub) sees a distinct pattern pins the bit order
    /// (value j at bits 2j), not just the all-periodic 0xE4 case.
    #[test]
    fn q2_ko_crumb_bit_order() {
        // Row 0, sub 0, q3 0 → low-half K positions 0,1,2,3 with values 3,2,1,0 (reverse) →
        // cr0 = 3<<0 | 2<<2 | 1<<4 | 0<<6 = 3 | 8 | 16 = 0x1B. Build w so only that quad reverses.
        let (nrows, ncols) = (8usize, 128usize);
        let mut w: Vec<f32> = vec![0.0; nrows * ncols];
        // Give every row full range [0,3] (so scale=1, min=0) via K=4..7 = 0,1,2,3; the tested
        // quad K=0..3 = 3,2,1,0.
        for r in 0..nrows {
            let base = r * ncols;
            for (k, val) in [3.0f32, 2.0, 1.0, 0.0].iter().enumerate() {
                w[base + k] = *val;
            }
            for (k, val) in [0.0f32, 1.0, 2.0, 3.0].iter().enumerate() {
                w[base + 4 + k] = *val;
            }
        }
        let chunk = quantize_ko(&w, nrows, ncols, GgmlDType::Q2_KO);
        // lane = r*4 + q3 = 0 (r=0,q3=0), sub=0 → cr0 at byte 0.
        assert_eq!(chunk[0], 0x1B, "cr0 for K=0..3 values (3,2,1,0)");
        // q3=1 → K=4..7 values (0,1,2,3) → cr0 = 0xE4, at lane=1, sub=0 → byte 1*8 = 8.
        assert_eq!(chunk[8], 0xE4, "cr0 for K=4..7 values (0,1,2,3)");
        let de = dequant_ko(&chunk, nrows, ncols, GgmlDType::Q2_KO);
        assert_eq!(de[0], 3.0);
        assert_eq!(de[1], 2.0);
        assert_eq!(de[4], 0.0);
        assert_eq!(de[7], 3.0);
    }
}
