//! Packs and unpacks the per-128 affine KO weight formats as lane-major chunks.
//!
//! Every affine width is a prefix of one plane stack: a 4-bit `ql` plane, then a
//! 2-bit crumb plane, then a 1-bit `hi` plane, stacked from the LSB. A value's bit
//! position is decided by which planes are present below it rather than by the
//! format, so one rule reproduces all five layouts and a new width is a row in the
//! table rather than a new codec. `quantize_ko` observes min-max per 128 columns,
//! while `pack_q4_ko` takes a source's own codes and scales so an imported 4-bit
//! checkpoint is kept bit-exact. `unpack_q4_ko` and `dequant_ko` read the bytes
//! back. Q8_KO is symmetric with a different lane split and has its own functions.

use super::*;

/// The plane set of an affine KO format: `(maxq, ql_bytes, crumb_bytes, hi_bytes)`.
///
/// **Every affine KO width is a prefix of one plane stack.** A chunk is 8 rows × 128 K = 1024
/// values, and a format spends its bits across at most three planes, stacked from the LSB in a
/// fixed order:
///
/// | plane | bits | bytes / chunk | address |
/// |---|---|---|---|
/// | `ql` | 4 | 512 | `lane*16 + sub*4 + i`, nibble per half |
/// | `crumb` | 2 | 256 | `lane*8 + sub*2 (+1)`, 2 bits at `2i` |
/// | `hi` | 1 | 128 | `lane*4 + sub`, nibble per half, bit `i` |
///
/// A value's bit position is therefore decided by which planes are *present* below it, not by
/// the format: `crumb` sits at bit 4 when `ql` is present and at bit 0 when it is not, and `hi`
/// sits above whichever of the two are there. That one rule reproduces all five layouts
/// bit-for-bit, which is why adding a width is a row in this table rather than a codec.
///
/// Q8_KO is not here: it is symmetric (min = 0) with a different lane split, and has its own
/// pair of functions.
pub(super) fn ko_params(dtype: GgmlDType) -> Option<(i32, usize, usize, usize)> {
    match dtype {
        //                        maxq   ql  crumb   hi     bits           bpw
        GgmlDType::Q2_KO => Some((3, 0, 256, 0)), //   crumb              2.25
        GgmlDType::Q3_KO => Some((7, 0, 256, 128)), // crumb + hi         3.25
        GgmlDType::Q4_KO => Some((15, 512, 0, 0)), //  ql                 4.25
        GgmlDType::Q5_KO => Some((31, 512, 0, 128)), // ql + hi           5.25
        GgmlDType::Q6_KO => Some((63, 512, 256, 0)), // ql + crumb        6.25
        _ => None,
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
    let p = KoPlanes::of(dtype);
    let (maxq, crumb_bytes, hi_bytes) = (p.maxq, p.crumb_bytes, p.hi_bytes);
    let (crumb_base, hi_base, dm_base, chunk_bytes) =
        (p.crumb_base, p.hi_base, p.dm_base, p.chunk_bytes);
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
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
                let (cs, hs) = (p.crumb_shift, p.hi_shift);
                for sub in 0..4 {
                    if p.ql_bytes > 0 {
                        for i in 0..16 {
                            let lo = q[sub * 32 + i] & 0xF;
                            let hi = q[sub * 32 + 16 + i] & 0xF;
                            ob[cbase + (r * 4 + i / 4) * 16 + sub * 4 + (i % 4)] = lo | (hi << 4);
                        }
                    }
                    for q3 in 0..4 {
                        let lane = r * 4 + q3;
                        if crumb_bytes > 0 {
                            let (mut cr0, mut cr1) = (0u8, 0u8);
                            for j in 0..4 {
                                cr0 |= ((q[sub * 32 + q3 * 4 + j] >> cs) & 0x3) << (2 * j);
                                cr1 |= ((q[sub * 32 + 16 + q3 * 4 + j] >> cs) & 0x3) << (2 * j);
                            }
                            ob[cbase + crumb_base + lane * 8 + sub * 2] = cr0;
                            ob[cbase + crumb_base + lane * 8 + sub * 2 + 1] = cr1;
                        }
                        if hi_bytes > 0 {
                            let (mut hb0, mut hb1) = (0u8, 0u8);
                            for j in 0..4 {
                                hb0 |= ((q[sub * 32 + q3 * 4 + j] >> hs) & 1) << j;
                                hb1 |= ((q[sub * 32 + 16 + q3 * 4 + j] >> hs) & 1) << j;
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
pub(super) fn quantize_q8_ko(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
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

/// Pack **precomputed** Q4 codes and per-group affine params into the lane-major
/// `Q4_KO` chunk layout — [`quantize_ko`] without its min-max observer.
///
/// The importer's entry point: a source that is *already* 4-bit affine per-128
/// (compressed-tensors W4A16 g128, AWQ/GPTQ) carries its own codes and scales,
/// and re-deriving them through the observer re-rounds whenever a group's
/// extremes don't span the grid. Packing the source's own `(code, scale, min)`
/// keeps the import **bit-exact**: `dequant_ko` reproduces `scale·q + min` with
/// the source's values.
///
/// `codes` are row-major `[nrows × ncols]`, each in `0..=15`. `dm` is row-major
/// `[nrows × ncols/128]` of `(scale, min)` per 128-column group. Scales and mins
/// are stored as f16 — the caller asserts f16-representability where exactness
/// is claimed (bf16-sourced scales in the f16 normal range are exact).
pub fn pack_q4_ko(codes: &[u8], dm: &[(f32, f32)], nrows: usize, ncols: usize) -> Vec<u8> {
    assert_eq!(nrows % 8, 0, "pack_q4_ko: nrows must be a multiple of 8");
    assert_eq!(
        ncols % 128,
        0,
        "pack_q4_ko: ncols must be a multiple of 128"
    );
    assert_eq!(codes.len(), nrows * ncols, "pack_q4_ko: codes length");
    let k_blocks = ncols / 128;
    assert_eq!(dm.len(), nrows * k_blocks, "pack_q4_ko: dm length");
    let p = KoPlanes::of(GgmlDType::Q4_KO);
    let chunk_bytes = p.chunk_bytes;
    let row_groups = nrows / 8;
    let mut ob = vec![0u8; k_blocks * row_groups * chunk_bytes];
    let mut cbuf = [0u8; 1024];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            // Gather the chunk's 8 × 128 codes contiguously, then place them
            // by the precomputed chunk permutation — a table walk instead of
            // per-element index arithmetic. The importer runs this 73k+ times
            // per checkpoint, which is what bought the table.
            for r in 0..8 {
                let row = g * 8 + r;
                let cb = row * ncols + k_blk * 128;
                cbuf[r * 128..(r + 1) * 128].copy_from_slice(&codes[cb..cb + 128]);
            }
            let out = &mut ob[cbase..cbase + 512];
            for (o, &(lo, hi)) in out.iter_mut().zip(Q4_KO_CHUNK_PERM.iter()) {
                *o = (cbuf[lo as usize] & 0xF) | ((cbuf[hi as usize] & 0xF) << 4);
            }
            for r in 0..8 {
                let row = g * 8 + r;
                let (scale, mn) = dm[row * k_blocks + k_blk];
                let d = cbase + p.dm_base + r * 4;
                ob[d..d + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                ob[d + 2..d + 4].copy_from_slice(&f16::from_f32(mn).to_le_bytes());
            }
        }
    }
    ob
}

/// The Q4_KO chunk permutation: byte `j` of a chunk's 512-byte `ql` plane
/// holds `(lo, hi)` — the chunk-local element indices (row-major `r·128 + k`)
/// whose codes land in its low and high nibble.
///
/// Derived once from the same arithmetic `quantize_ko`'s loop performs
/// (`lane = j/16`, `sub = (j%16)/4`, `r = lane/4`, `i = (lane%4)·4 + j%4`).
/// The byte-identity test against `quantize_ko` is what keeps the table
/// honest rather than merely self-consistent.
pub(super) static Q4_KO_CHUNK_PERM: [(u16, u16); 512] = build_q4_ko_chunk_perm();

pub(super) const fn build_q4_ko_chunk_perm() -> [(u16, u16); 512] {
    let mut t = [(0u16, 0u16); 512];
    let mut j = 0usize;
    while j < 512 {
        let lane = j / 16;
        let sub = (j % 16) / 4;
        let r = lane / 4;
        let i = (lane % 4) * 4 + j % 4;
        let lo = r * 128 + sub * 32 + i;
        t[j] = (lo as u16, (lo + 16) as u16);
        j += 1;
    }
    t
}

/// Inverse of [`pack_q4_ko`]: recover the codes and per-group `(scale, min)`
/// pairs (as the f16 values stored) from a lane-major Q4_KO chunk tensor.
///
/// The importer's fast verification path: comparing recovered codes and dm
/// against the source is byte-equivalent to comparing full dequantizations,
/// because `dequant_ko` is exactly the affine `scale·code + min` over these
/// bytes — pinned by the `pack_q4_ko` unit tests against the *untouched*
/// `dequant_ko`. That independence matters: this function shares
/// [`Q4_KO_CHUNK_PERM`] with the packer, so a wrong table would round-trip
/// cleanly here — and be caught there.
pub fn unpack_q4_ko(chunk: &[u8], nrows: usize, ncols: usize) -> (Vec<u8>, Vec<(f32, f32)>) {
    assert_eq!(nrows % 8, 0, "unpack_q4_ko: nrows must be a multiple of 8");
    assert_eq!(
        ncols % 128,
        0,
        "unpack_q4_ko: ncols must be a multiple of 128"
    );
    let p = KoPlanes::of(GgmlDType::Q4_KO);
    let chunk_bytes = p.chunk_bytes;
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
    assert_eq!(chunk.len(), k_blocks * row_groups * chunk_bytes);
    let mut codes = vec![0u8; nrows * ncols];
    let mut dm = vec![(0f32, 0f32); nrows * k_blocks];
    let mut cbuf = [0u8; 1024];
    for k_blk in 0..k_blocks {
        for g in 0..row_groups {
            let cbase = (k_blk * row_groups + g) * chunk_bytes;
            for (j, &(lo, hi)) in Q4_KO_CHUNK_PERM.iter().enumerate() {
                let b = chunk[cbase + j];
                cbuf[lo as usize] = b & 0xF;
                cbuf[hi as usize] = b >> 4;
            }
            for r in 0..8 {
                let row = g * 8 + r;
                let cb = row * ncols + k_blk * 128;
                codes[cb..cb + 128].copy_from_slice(&cbuf[r * 128..(r + 1) * 128]);
                let d = cbase + p.dm_base + r * 4;
                dm[row * k_blocks + k_blk] = (
                    f16::from_le_bytes([chunk[d], chunk[d + 1]]).to_f32(),
                    f16::from_le_bytes([chunk[d + 2], chunk[d + 3]]).to_f32(),
                );
            }
        }
    }
    (codes, dm)
}

/// Inverse of [`quantize_ko`] — reconstruct F32 `[nrows × ncols]` from the lane-major KO
/// chunk tensor (the per-128 dequant `W = scale·q + min`). Used to validate the quantizer.
pub fn dequant_ko(chunk: &[u8], nrows: usize, ncols: usize, dtype: GgmlDType) -> Vec<f32> {
    if dtype == GgmlDType::Q8_KO {
        return dequant_q8_ko(chunk, nrows, ncols);
    }
    let pl = KoPlanes::of(dtype);
    let (crumb_bytes, hi_bytes) = (pl.crumb_bytes, pl.hi_bytes);
    let (crumb_base, hi_base, dm_base, chunk_bytes) =
        (pl.crumb_base, pl.hi_base, pl.dm_base, pl.chunk_bytes);
    let k_blocks = ncols / 128;
    let row_groups = nrows / 8;
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
                        let mut qv = if pl.ql_bytes > 0 {
                            let qlb = chunk[cbase + lane * 16 + sub * 4 + i] as u32;
                            if half == 0 {
                                qlb & 0xF
                            } else {
                                (qlb >> 4) & 0xF
                            }
                        } else {
                            0
                        };
                        if crumb_bytes > 0 {
                            let off = cbase + crumb_base + lane * 8 + sub * 2;
                            let cr = if half == 0 {
                                chunk[off]
                            } else {
                                chunk[off + 1]
                            } as u32;
                            qv |= ((cr >> (2 * i)) & 0x3) << pl.crumb_shift;
                        }
                        if hi_bytes > 0 {
                            let hb = chunk[cbase + hi_base + lane * 4 + sub] as u32;
                            let nib = if half == 0 { hb & 0xF } else { (hb >> 4) & 0xF };
                            qv |= ((nib >> i) & 1) << pl.hi_shift;
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

pub(super) fn dequant_q8_ko(chunk: &[u8], nrows: usize, ncols: usize) -> Vec<f32> {
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
/// Q2_KO: 2-bit affine KO twin — per-128 `(scale, min)`, value 0..3. Layout per 8-row × 128-K
/// chunk (288 B): a 256 B **crumb** region + 32 B `dm`. The crumb region is byte-identical to the
/// high-2-bit crumb region `Q6_KO` already carries — for `(lane = r*4 + q3, sub)`, `cr0` at
/// `lane*8 + sub*2` packs the 4 LOW-half values (`q[sub*32 + q3*4 + j]`, 2 bits at `2j`) and
/// `cr1` at `+1` the 4 HIGH-half values (`q[sub*32 + 16 + q3*4 + j]`) — but here the crumb IS the
/// whole value (Q6 stores only the high 2 bits of a 6-bit value). One row's 32 B = one
/// `block_c_q2_KO_k128`. `dm[r] = (scale, min)` f16 at `256 + r*4`.
///
/// **Kept as an independent oracle, not as a code path.** This was the production Q2_KO
/// packer until [`KoPlanes`] generalised the plane stack and subsumed it. It survives because
/// it is a second, differently-written statement of the same layout: `the_plane_stack_
/// reproduces_the_hand_written_q2_ko` asserts the generic path equals it byte-for-byte, which
/// is the proof that generalising did not move a single bit of an existing format.
pub(super) fn quantize_q2_ko(w: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
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

#[cfg(test)]
/// Inverse of [`quantize_q2_ko`] — reconstruct F32 `W = scale·q + min` from the crumb chunk.
/// An oracle for the same reason its inverse is.
pub(super) fn dequant_q2_ko(chunk: &[u8], nrows: usize, ncols: usize) -> Vec<f32> {
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
