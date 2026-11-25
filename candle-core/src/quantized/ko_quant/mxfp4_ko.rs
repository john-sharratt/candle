//! Repacks MXFP4 weights into the lane-major layout the MXFP4_KO int8 kernel reads.
//!
//! MXFP4_KO keeps the MXFP4 nibbles and E8M0 scales exactly, only reordered, with
//! the four per-sub scales split out, so it is a lossless repack and not a
//! requantization. The engine stages experts from native MXFP4 GGUF bytes with
//! `mxfp4_native_to_ko_gpu_chunk`, a pure byte permutation, and that gives the same
//! GPU bytes as the from-F32 path `quantize_mxfp4_ko`. The per-row `dm` baked into
//! the GPU chunk is not read by the kernel; it is kept so the chunk layout and its
//! pack-file fingerprint stay fixed. `mxfp4_ko_int8_matmul` mirrors the kernel's
//! per-32 fold operation for operation, and `dequant_mxfp4_ko` is the exact
//! baseline it is measured against.

use super::*;

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
                    for (p, &byte) in blk[0].qs.iter().enumerate().take(16) {
                        let q3 = p / 4;
                        let i = p % 4;
                        ob[cbase + (r * 4 + q3) * 16 + sub * 4 + i] = byte;
                    }
                }
            }
        }
    }
    ob
}

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
