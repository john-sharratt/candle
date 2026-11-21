//! Host-side producers of the KO weight layouts, used as the references the GPU
//! kernels are tested against.
//!
//! Each helper builds one layout: a de-interleaved KO block with its scale
//! region after the quantized bytes, the sub-major chunk the int8 dequant reads
//! with conflict-free lanes, or the per-128 and per-32 affine producers that the
//! GPU quantizer is checked against. The Q5 and Q6 decoders mirror the device
//! dequant lane by lane, so a mismatch points at the layout and not at a
//! reference that shares its misreading.

use super::*;

/// Quant-region size (bytes/block) of a de-interleaved KO weight block — the scale
/// region is always a further 16 B/block.
#[cfg(feature = "cuda")]
pub(super) fn ko_quant_bytes(kod: GgmlDType) -> usize {
    match kod {
        GgmlDType::Q4_KO => 64,
        GgmlDType::Q5_KO => 80,
        GgmlDType::Q6_KO => 96,
        GgmlDType::Q8_KO => 128,
        other => unreachable!("not a KO format: {other:?}"),
    }
}

/// De-interleaved KO producer: run the K→KO byte permutation, then split each block
/// into `[quant region | 16 B scale region]` — quant blocks packed first, then one 16 B
/// scale block per quant block (same block index), matching is_scale_separate + the
/// dm-hoist in kernel.cuh. The perm writes an `in_stride`-byte interleaved KO block whose
/// first `quant_bytes` are quants and whose next 16 B are the per-block scales. For Q8_KO
/// (symmetric) the scale's high half (the min term) is zeroed — the de-interleaved fold
/// reads the raw half2 instead of sub_dm's explicit (d, 0).
#[cfg(feature = "cuda")]
pub(super) fn ko_deinterleave(
    kb: &[u8],
    in_stride: usize,
    quant_bytes: usize,
    zero_scale_hi: bool,
    perm: fn(&[u8], &mut [u8]),
) -> Vec<u8> {
    let n = kb.len() / in_stride;
    let mut ob = vec![0u8; n * (quant_bytes + 16)];
    let sbase = n * quant_bytes;
    let mut tmp = vec![0u8; in_stride];
    for i in 0..n {
        tmp.iter_mut().for_each(|b| *b = 0);
        perm(&kb[i * in_stride..(i + 1) * in_stride], &mut tmp);
        ob[i * quant_bytes..(i + 1) * quant_bytes].copy_from_slice(&tmp[0..quant_bytes]);
        let sc = sbase + i * 16;
        ob[sc..sc + 16].copy_from_slice(&tmp[quant_bytes..quant_bytes + 16]);
        if zero_scale_hi {
            for s in 0..4 {
                ob[sc + s * 4 + 2] = 0;
                ob[sc + s * 4 + 3] = 0;
            }
        }
    }
    ob
}

/// Decode the 32 unsigned 4-bit weights of one (compact Q4_K block, sub) into `K[0..32]`,
/// mirroring `q4k_dequant_to_b_frag_int8`: the sub's 4 ints J0..J3 (at `qs_base+{0,4,8,12}`)
/// map J0→K[0:8], J1→K[8:16], J2→K[16:24], J3→K[24:32]; within each int the low nibbles are
/// the first 4 K and the high nibbles the next 4, both in `byte_perm 0x3120` ([0,2,1,3]) order.
#[cfg(feature = "cuda")]
pub(super) fn q4k_decode_sub(block: &[u8], qs_base: usize) -> [u8; 32] {
    const PERM: [usize; 4] = [0, 2, 1, 3]; // byte_perm 0x3120: result[i] = src[PERM[i]]
    let mut k = [0u8; 32];
    for (j, &base) in [0usize, 8, 16, 24].iter().enumerate() {
        let jint = &block[qs_base + j * 4..qs_base + j * 4 + 4];
        for i in 0..4 {
            k[base + i] = jint[PERM[i]] & 0x0F;
            k[base + 4 + i] = (jint[PERM[i]] >> 4) & 0x0F;
        }
    }
    k
}

/// Build the SUB-MAJOR k1024 Q4_KO chunk tensor from compact Q4_K blocks (block index
/// `k_blk*nrows + row`, 80 B each). This is the wavefront-optimal weight layout the int8
/// dequant reads: chunk `[k_blk][row-group of 8]`, the 512 B quant region sub-major — sub s
/// at `[s*128, s*128+128)`, row r's 16 bytes packing the 32 K-nibbles as byte
/// `p = K[p] | (K[p+16]<<4)`. Lane `(row=lane>>2, q3=lane&3)` then reads one coalesced int at
/// `s*128 + lane*4` → bank `lane` (conflict-free, 1 wavefront/sub). The 128 B scale region
/// (`dm[row][sub]`, the same `(scale,min)` half2 as Q4_K) follows at offset 512.
#[cfg(feature = "cuda")]
pub(super) fn q4ko_submajor_from_compact(kb: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    const QS_BASE: [usize; 4] = [0, 24, 40, 64]; // compact sub qs-int bases
    const DM_OFF: [usize; 4] = [16, 20, 56, 60]; // compact sub (scale,min) half2 offsets
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let chunk_bytes = 512 + 128; // 8 rows × 64 B quant + 8 rows × 16 B scales
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let row = g * 8 + r;
                let blk = &kb[(k_blk * nrows + row) * 80..(k_blk * nrows + row) * 80 + 80];
                for sub in 0..4 {
                    let kk = q4k_decode_sub(blk, QS_BASE[sub]);
                    for p in 0..16 {
                        ob[cbase + sub * 128 + r * 16 + p] =
                            (kk[p] & 0xF) | ((kk[p + 16] & 0xF) << 4);
                    }
                    let d = cbase + 512 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(&blk[DM_OFF[sub]..DM_OFF[sub] + 4]);
                }
            }
        }
    }
    ob
}

/// Build the SUB-MAJOR k1024 Q8_KO chunk from the de-interleaved Q8_KO `[quants(128B) |
/// 16B scales]` tensor (block `k_blk*nrows+row`). The new int8 dequant reads `b_frag[0]` at
/// `sub*256 + lane*4` and `b_frag[1]` at `sub*256 + 128 + lane*4` (the two 128 B half-sub
/// blocks K[0:16]/K[16:32]); we place each lane's 4 bytes there by pulling them from the
/// proven row-major dequant offsets — so the values are correct by construction. dm
/// (`(d,0)` half2) region follows at offset 1024.
#[cfg(feature = "cuda")]
pub(super) fn q8ko_submajor_from_de(de: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 128;
    let chunk_bytes = 1024 + 128;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let i = k_blk * nrows + g * 8 + r;
                let q = &de[i * 128..i * 128 + 128];
                for sub in 0..4 {
                    for q3 in 0..4 {
                        let lane4 = (r * 4 + q3) * 4; // lane*4 within a half-sub block
                        let off0 = (sub * 4 + (q3 >> 1)) * 8 + (q3 & 1) * 4; // old b_frag[0]
                        let off1 = (sub * 4 + 2 + (q3 >> 1)) * 8 + (q3 & 1) * 4; // old b_frag[1]
                        ob[cbase + sub * 256 + lane4..cbase + sub * 256 + lane4 + 4]
                            .copy_from_slice(&q[off0..off0 + 4]);
                        ob[cbase + sub * 256 + 128 + lane4..cbase + sub * 256 + 128 + lane4 + 4]
                            .copy_from_slice(&q[off1..off1 + 4]);
                    }
                    let d = cbase + 1024 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(
                        &de[sbase + i * 16 + sub * 4..sbase + i * 16 + sub * 4 + 4],
                    );
                }
            }
        }
    }
    ob
}

/// Decode one Q5_KO block's `(sub, q3)` lane into its two 4-value b_frags (5-bit values
/// 0..31), mirroring the GPU `block_c_q5_KO` dequant: int2 over the sub's qs ints (byte_perm
/// 0x3120 + low/high nibble) plus the 5th-bit extract from the sub's qh int.
#[cfg(feature = "cuda")]
pub(super) fn q5ko_b_frags(blk: &[u8], sub: usize, q3: usize) -> ([u8; 4], [u8; 4]) {
    const PERM: [usize; 4] = [0, 2, 1, 3];
    let ri = |off: usize| u32::from_le_bytes([blk[off], blk[off + 1], blk[off + 2], blk[off + 3]]);
    let sh = (q3 & 1) * 4;
    // qs ints [I0,I2,I1,I3] at sub*16; q3<2 → {I0,I2}=(x,y at +0,+8), q3>=2 → {I1,I3}=(+8 base).
    let base = sub * 16 + (q3 >> 1) * 8;
    let (qx, qy) = (ri(base), ri(base + 4));
    let nib = |v: u32| -> [u8; 4] {
        let b = ((v >> sh) & 0x0F0F0F0F).to_le_bytes();
        [b[PERM[0]], b[PERM[1]], b[PERM[2]], b[PERM[3]]]
    };
    let qhw = ri(64 + sub * 4);
    let qh0 = (qhw >> ((q3 >> 1) * 8)) & 0xFF;
    let qh1 = (qhw >> (((q3 >> 1) + 2) * 8)) & 0xFF;
    let hb0 = (qh0 >> sh) & 0xF;
    let hb1 = (qh1 >> sh) & 0xF;
    let combine = |nb: [u8; 4], hb: u32| -> [u8; 4] {
        let mut o = [0u8; 4];
        for i in 0..4 {
            o[i] = nb[i] | ((((hb >> i) & 1) as u8) << 4);
        }
        o
    };
    (combine(nib(qx), hb0), combine(nib(qy), hb1))
}

/// Build the SUB-MAJOR k1024 Q5_KO chunk from a de-interleaved `[quants(80B) | 16B
/// (scale,min)]` tensor (block `k_blk*nrows+row`). Layout: 512 B sub-major ql + 128 B
/// sub-major 5th-bit stream (one byte/lane: lo nibble = b0 highs, hi nibble = b1 highs) +
/// 128 B dm. Lane `(row,q3)`: ql int at `sub*128+lane*4`, 5th-bit byte at `512+sub*32+lane`.
#[cfg(feature = "cuda")]
pub(super) fn q5ko_submajor_from_de(de: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 80;
    let chunk_bytes = 512 + 128 + 128;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let i = k_blk * nrows + g * 8 + r;
                let blk = &de[i * 80..i * 80 + 80];
                for sub in 0..4 {
                    let mut kk = [0u8; 32];
                    for q3 in 0..4 {
                        let (b0, b1) = q5ko_b_frags(blk, sub, q3);
                        for ii in 0..4 {
                            kk[q3 * 4 + ii] = b0[ii];
                            kk[q3 * 4 + 16 + ii] = b1[ii];
                        }
                    }
                    for p in 0..16 {
                        ob[cbase + sub * 128 + r * 16 + p] =
                            (kk[p] & 0xF) | ((kk[p + 16] & 0xF) << 4);
                    }
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        let (mut hb0, mut hb1) = (0u8, 0u8);
                        for j in 0..4 {
                            hb0 |= (((kk[q3 * 4 + j] >> 4) & 1) << j) as u8;
                            hb1 |= (((kk[q3 * 4 + 16 + j] >> 4) & 1) << j) as u8;
                        }
                        ob[cbase + 512 + sub * 32 + lane] = hb0 | (hb1 << 4);
                    }
                    let d = cbase + 640 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(
                        &de[sbase + i * 16 + sub * 4..sbase + i * 16 + sub * 4 + 4],
                    );
                }
            }
        }
    }
    ob
}

/// Decode one Q6_KO block's `(sub, q3)` lane into its two 4-value b_frags (6-bit values
/// 0..63), mirroring the GPU `block_c_q6_KO` dequant: int4 over the sub's ql ints (with the
/// `byte_perm 0x3120` [0,2,1,3] order + low/high nibble by `q3&1`), plus the qh crumb spread.
#[cfg(feature = "cuda")]
pub(super) fn q6ko_b_frags(blk: &[u8], sub: usize, q3: usize) -> ([u8; 4], [u8; 4]) {
    const PERM: [usize; 4] = [0, 2, 1, 3];
    let ri = |off: usize| u32::from_le_bytes([blk[off], blk[off + 1], blk[off + 2], blk[off + 3]]);
    let sh = (q3 & 1) * 4;
    let m0 = sub * 4 + (q3 >> 1);
    let m1 = m0 + 2;
    let (vx, vy, vz, vw) = (
        ri(sub * 16),
        ri(sub * 16 + 4),
        ri(sub * 16 + 8),
        ri(sub * 16 + 12),
    );
    let v0 = if q3 < 2 { vx } else { vy };
    let v1 = if q3 < 2 { vz } else { vw };
    let nib = |v: u32| -> [u8; 4] {
        let b = ((v >> sh) & 0x0F0F0F0F).to_le_bytes();
        [b[PERM[0]], b[PERM[1]], b[PERM[2]], b[PERM[3]]]
    };
    let qh = |m: usize| -> u32 {
        let o = 64 + (m >> 1) * 4 + (m & 1) * 2;
        u16::from_le_bytes([blk[o], blk[o + 1]]) as u32
    };
    let cr0 = (qh(m0) >> ((q3 & 1) * 8)) & 0xFF;
    let cr1 = (qh(m1) >> ((q3 & 1) * 8)) & 0xFF;
    let combine = |nb: [u8; 4], cr: u32| -> [u8; 4] {
        let mut o = [0u8; 4];
        for i in 0..4 {
            o[i] = nb[i] | ((((cr >> (2 * i)) & 0x3) as u8) << 4);
        }
        o
    };
    (combine(nib(v0), cr0), combine(nib(v1), cr1))
}

/// Build the SUB-MAJOR k1024 Q6_KO chunk from a `[quants(96B) | 16B (scale,min)]` tensor
/// (requant_q6_ko_affine output, block `k_blk*nrows+row`). Layout: 512 B sub-major ql +
/// 256 B sub-major qh crumb stream + 128 B dm. Lane `(row,q3)`: ql int at `sub*128+lane*4`
/// (low/high nibbles), qh uint16 at `512+sub*64+lane*2` (cr0 low byte, cr1 high byte).
#[cfg(feature = "cuda")]
pub(super) fn q6ko_submajor_from_de(de: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 96;
    let chunk_bytes = 512 + 256 + 128;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for r in 0..8 {
                let i = k_blk * nrows + g * 8 + r;
                let blk = &de[i * 96..i * 96 + 96];
                for sub in 0..4 {
                    let mut kk = [0u8; 32];
                    for q3 in 0..4 {
                        let (b0, b1) = q6ko_b_frags(blk, sub, q3);
                        for ii in 0..4 {
                            kk[q3 * 4 + ii] = b0[ii];
                            kk[q3 * 4 + 16 + ii] = b1[ii];
                        }
                    }
                    for p in 0..16 {
                        ob[cbase + sub * 128 + r * 16 + p] =
                            (kk[p] & 0xF) | ((kk[p + 16] & 0xF) << 4);
                    }
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        let (mut cr0, mut cr1) = (0u8, 0u8);
                        for j in 0..4 {
                            cr0 |= (((kk[q3 * 4 + j] >> 4) & 0x3) << (2 * j)) as u8;
                            cr1 |= (((kk[q3 * 4 + 16 + j] >> 4) & 0x3) << (2 * j)) as u8;
                        }
                        ob[cbase + 512 + sub * 64 + lane * 2] = cr0;
                        ob[cbase + 512 + sub * 64 + lane * 2 + 1] = cr1;
                    }
                    let d = cbase + 768 + r * 16 + sub * 4;
                    ob[d..d + 4].copy_from_slice(
                        &de[sbase + i * 16 + sub * 4..sbase + i * 16 + sub * 4 + 4],
                    );
                }
            }
        }
    }
    ob
}

/// Per-128 affine KO producer (Q4/Q5/Q6) — thin wrapper over the production
/// `ko_quant::quantize_ko`, so the GPU gates double as validation of the production quantizer.
#[cfg(feature = "cuda")]
pub(super) fn requant_ko_per128(
    w: &[f32],
    nrows: usize,
    ncols: usize,
    maxq: i32,
    crumb_bytes: usize,
    hi_bytes: usize,
) -> Vec<u8> {
    // The triple is a format *name* rather than a parameter set: `ko_quant::KoPlanes` owns the
    // plane widths and shifts, and this only maps back to the dtype it describes. Kept because
    // fifteen call sites spell their format this way; new ones should pass the dtype.
    let dtype = match (maxq, crumb_bytes, hi_bytes) {
        (3, 256, 0) => GgmlDType::Q2_KO,
        (7, 256, 128) => GgmlDType::Q3_KO,
        (15, 0, 0) => GgmlDType::Q4_KO,
        (31, 0, 128) => GgmlDType::Q5_KO,
        (63, 256, 0) => GgmlDType::Q6_KO,
        _ => panic!("unsupported KO params: maxq={maxq} crumb={crumb_bytes} hi={hi_bytes}"),
    };
    crate::quantized::ko_quant::quantize_ko(w, nrows, ncols, dtype)
}

/// Per-128 Q8_KO producer — thin wrapper over the production `ko_quant::quantize_ko`.
#[cfg(feature = "cuda")]
pub(super) fn requant_q8_ko_per128(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    crate::quantized::ko_quant::quantize_ko(w, nrows, ncols, GgmlDType::Q8_KO)
}

#[cfg(feature = "cuda")]
pub(super) fn requant_q8_ko_affine(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    use half::f16;
    let k_blocks = ncols / 128;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 128;
    let mut ob = vec![0u8; sbase + n_blocks * 16];
    for k_blk in 0..k_blocks {
        for row in 0..nrows {
            let block_idx = k_blk * nrows + row;
            for sub in 0..4 {
                let base = k_blk * 128 + sub * 32;
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for i in 0..32 {
                    let v = w[row * ncols + base + i];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / 255.0).max(1e-12);
                for i in 0..32 {
                    let v = w[row * ncols + base + i];
                    let q = (((v - mn) / scale).round() as i32).clamp(0, 255);
                    ob[block_idx * 128 + sub * 32 + i] = (q - 128) as i8 as u8;
                }
                let m_fold = 128.0 * scale + mn;
                let sc = sbase + block_idx * 16 + sub * 4;
                ob[sc..sc + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[sc + 2..sc + 4].copy_from_slice(&f16::from_f32(m_fold).to_le_bytes());
            }
        }
    }
    ob
}

/// Requantize F32 weights into Q6_KO with PER-32 affine (scale,min). 6-bit values stay
/// UNSIGNED (0..63 — the un-centered dequant). ql (low 4 bits) and qh (high 2 bits) land
/// at the positions the dequant reads: ql_int/qh use `sub*4 + k'/8`, byte = [0,2,1,3][k'%4]
/// (the dequant's byte_perm 0x3120), nibble/byte = (k'/4)&1, qh 2-bit = 2*(k'%4). Scale
/// region (tail): (scale, min) half2 per sub.
#[cfg(feature = "cuda")]
pub(super) fn requant_q6_ko_affine(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    use half::f16;
    const PERM: [usize; 4] = [0, 2, 1, 3];
    let k_blocks = ncols / 128;
    let n_blocks = k_blocks * nrows;
    let sbase = n_blocks * 96; // quant region 96B/block (ql 64 + qh 32)
    let mut ob = vec![0u8; sbase + n_blocks * 16];
    for k_blk in 0..k_blocks {
        for row in 0..nrows {
            let block_idx = k_blk * nrows + row;
            let qbase = block_idx * 96;
            for sub in 0..4 {
                let cbase = k_blk * 128 + sub * 32;
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for i in 0..32 {
                    let v = w[row * ncols + cbase + i];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / 63.0).max(1e-12);
                for kp in 0..32 {
                    let v = w[row * ncols + cbase + kp];
                    let q = (((v - mn) / scale).round() as i32).clamp(0, 63) as u32;
                    let unit = sub * 4 + kp / 8;
                    let nib = (kp / 4) & 1;
                    // ql low nibble
                    let pos = qbase + unit * 4 + PERM[kp % 4];
                    ob[pos] |= ((q & 0xF) as u8) << (nib * 4);
                    // qh high 2 bits
                    let qh_byte = 2 * unit + nib;
                    ob[qbase + 64 + qh_byte] |= (((q >> 4) & 0x3) as u8) << (2 * (kp % 4));
                }
                let sc = sbase + block_idx * 16 + sub * 4;
                ob[sc..sc + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[sc + 2..sc + 4].copy_from_slice(&f16::from_f32(mn).to_le_bytes());
            }
        }
    }
    ob
}
