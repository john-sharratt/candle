//! Host codecs that author and mirror the band formats the latent attention
//! kernel reads, byte for byte.
//!
//! The kernel dequantises each band with the device's own converters, so a
//! synthetic arena written by the host has to hold bytes the device will read
//! back identically. Every format here repeats the device's operation order,
//! including where a rounding or a fused multiply happens, because the
//! bit-exact contract depends on it. A caller gets both the encoded bytes and
//! the exact per-token values the kernel will produce from them, so the mirror
//! and the arena are built from one codec.

use super::*;

// ─── FP8 E4M3 (e4m3fn: no inf, max ±448) host codec ─────────────────────────
// Used to author the synthetic band arenas and by the mirror. The harness only
// feeds exactly-representable values through it, so any correct rounding
// implementation agrees byte-for-byte with the device converter.

pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f32;
    if e == 0x0F && (b & 0x07) == 0x07 {
        return f32::NAN; // e4m3fn: only S.1111.111 is NaN
    }
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    if a >= 448.0 {
        return sign | 0x7E; // saturate to ±448 (e4m3fn has no inf)
    }
    // Round to the nearest representable magnitude, ties to even mantissa.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for cand in 0u8..0x7F {
        let cv = e4m3_to_f32(cand);
        if cv.is_nan() {
            continue;
        }
        let err = (cv - a).abs();
        if err < best_err || (err == best_err && cand & 1 == 0) {
            best_err = err;
            best = cand;
        }
    }
    sign | best
}

// ─── Synthetic slot construction (rung-2 harness) ────────────────────────────

/// A hand-built single-conversation window slot: the raw `SlotHeader` /
/// `TokenSlice` / `KvHead` bytes plus the FP8 band arenas, exactly as the
/// production cache lays them out — but authored from a plain
/// `[n_tokens, 512]` f32 window with no substrate involved.
#[cfg(feature = "cuda")]
/// Per-band storage spec for synthetic window authoring: the KvHead format
/// tag plus the outer (encoder-multiply / decoder-divide) scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandSpec {
    pub fmt: u8,
    pub outer: f32,
}

impl Default for BandSpec {
    fn default() -> Self {
        // F8E4M3 at unit scale — the writer-chunk format.
        Self {
            fmt: 34,
            outer: 1.0,
        }
    }
}

/// Lane-0 result of the GPU warp XOR-tree float sum (offsets 16, 8, 4, 2,
/// 1). The pairwise summation ORDER is part of the bit-exact contract — a
/// serial sum can round differently, flipping an INT8-encoded scale by one
/// code. Addition is commutative in IEEE, so every lane of a pair computes
/// identical bits and all 32 lanes converge to the same value.
pub(super) fn warp_tree_sum(v: &[f32; CHUNK]) -> f32 {
    let mut a = *v;
    for off in [16usize, 8, 4, 2, 1] {
        let p = a;
        for i in 0..CHUNK {
            a[i] = p[i] + p[i ^ off];
        }
    }
    a[0]
}

/// The GPU `q0_encode_centroid`: clamp the FLOAT to ±127 first, then
/// round-to-nearest-even (`__float2int_rn`).
// NOT `clamp`: this mirrors CUDA `fminf`/`fmaxf`, which return the non-NaN
// operand, so a NaN centroid saturates to ±127 here exactly as it does on the
// device. `f32::clamp` propagates the NaN instead and would break the byte
// equality this mirror exists to prove.
#[allow(clippy::manual_clamp)]
pub(super) fn q0_encode_centroid(val: f32) -> i8 {
    (val * 127.0).min(127.0).max(-127.0).round_ties_even() as i32 as i8
}

/// Band-chunk storage round-trip: encode `toks` (≤32 tokens × `SUB_DIM` dims,
/// zero-padded to the chunk) into the band's arena bytes for `spec`, and
/// return the exact per-token values the kernel's dequant produces (the
/// mirror's stored source — one codec for both sides). Float bands are
/// token-major rows; quant bands are token-oriented GGML blocks (block `d`
/// holds dim `d`'s 32 tokens; `CHUNK == 32` so blocks_per_dim is 1). Decode
/// arithmetic replicates the GPU converters op-for-op: Q8_0 is
/// `half2float(d) * qs / outer`, Q4_0 rounds `half2float(d) / outer` FIRST
/// and stores split nibbles (elements 0-15 low, 16-31 high).
///
/// For the INT8-scale and Q0-family formats the `spec.outer` is the searched
/// palette scale (the encoder sees `x * outer`, the decoder divides it back
/// out last — the exact `load_element` op order), and every warp-tree float
/// sum in the GPU encoder is replicated through [`warp_tree_sum`].
// Same `fminf`/`fmaxf` NaN semantics as [`q0_encode_centroid`] — see its note.
#[allow(clippy::manual_clamp)]
pub fn band_chunk_roundtrip(
    spec: BandSpec,
    toks: &[[f32; SUB_DIM]],
) -> Result<(Vec<u8>, Vec<[f32; SUB_DIM]>)> {
    let n = toks.len().min(CHUNK);
    let mut dec = vec![[0f32; SUB_DIM]; n];
    let bytes = match spec.fmt {
        34 => {
            // F8E4M3: token-major rows of SUB_DIM bytes.
            let mut bytes = vec![0u8; CHUNK * SUB_DIM];
            for (t, tok) in toks.iter().take(n).enumerate() {
                for (d, &v) in tok.iter().enumerate() {
                    let b = f32_to_e4m3(v * spec.outer);
                    bytes[t * SUB_DIM + d] = b;
                    dec[t][d] = e4m3_to_f32(b) / spec.outer;
                }
            }
            bytes
        }
        7 => {
            // Q8_0: 34-byte blocks `{f16 d; i8 qs[32]}` (GGML reference
            // quantizer: d = amax/127, qs = round(x/d)).
            let mut bytes = vec![0u8; SUB_DIM * 34];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let dh = half::f16::from_f32(amax * (1.0 / 127.0));
                let ds = dh.to_f32();
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 34..(d + 1) * 34];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                for (t, &v) in col.iter().enumerate() {
                    let q = (v * id).round().clamp(-127.0, 127.0) as i8;
                    blk[2 + t] = q as u8;
                    if t < n {
                        dec[t][d] = ds * q as f32 / spec.outer;
                    }
                }
            }
            bytes
        }
        15 => {
            // Q4_0: 18-byte blocks `{f16 d; u8 qs[16]}`, split nibbles (GGML
            // reference quantizer: d = signed_max/-8, q = x/d + 8.5 clamped).
            let mut bytes = vec![0u8; SUB_DIM * 18];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let mut smax = 0f32;
                for &v in &col {
                    if v.abs() > smax.abs() {
                        smax = v;
                    }
                }
                let dh = half::f16::from_f32(smax / -8.0);
                let ds = dh.to_f32();
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 18..(d + 1) * 18];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                let mut nib = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    nib[t] = ((v * id + 8.5) as i32).clamp(0, 15) as u8;
                }
                for j in 0..16 {
                    blk[2 + j] = nib[j] | (nib[j + 16] << 4);
                }
                // GPU converter rounds d/outer before the nibble multiply.
                let dso = ds / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = dso * (nib[t] as f32 - 8.0);
                }
            }
            bytes
        }
        16 => {
            // Q4_1: 20-byte blocks `{f16 d; f16 m; u8 qs[16]}`, split nibbles
            // (encoder: d = (max-min)/15, q = (x-min)/d + 0.5 truncated).
            let mut bytes = vec![0u8; SUB_DIM * 20];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let (mut vmax, mut vmin) = (col[0], col[0]);
                for &v in &col {
                    vmax = vmax.max(v);
                    vmin = vmin.min(v);
                }
                let dq = (vmax - vmin) * (1.0 / 15.0);
                let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                let dh = half::f16::from_f32(dq);
                let mh = half::f16::from_f32(vmin);
                let blk = &mut bytes[d * 20..(d + 1) * 20];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&mh.to_le_bytes());
                let mut nib = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    nib[t] = ((v - vmin) * id + 0.5).clamp(0.0, 15.0) as u8;
                }
                for j in 0..16 {
                    blk[4 + j] = nib[j] | (nib[j + 16] << 4);
                }
                // GPU decode: d and m each divided by outer first; the
                // `d*nibble + m` in BlockConverter<block_q4_1> CONTRACTS to an
                // fma in the latent TU — mirror with mul_add (verified by the
                // compressed-arena round-trip: plain mul+add differs by 1 ulp
                // on rare values).
                let d32 = dh.to_f32() / spec.outer;
                let m32 = mh.to_f32() / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = d32 * nib[t] as f32 + m32;
                }
            }
            bytes
        }
        10 => {
            // Q8_KS: 36-byte blocks `{f16 d; u8 sa; u8 sb; i8 qs[32]}` — the
            // attention-sink format: elements 0-3 (the chunk's first four
            // TOKENS in this token-oriented layout — the sink positions) get
            // their own sub-scale sa, elements 4-31 get sb.
            let mut bytes = vec![0u8; SUB_DIM * 36];
            for d in 0..SUB_DIM {
                let col: Vec<f32> = (0..CHUNK)
                    .map(|t| toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer))
                    .collect();
                let amax_a = col[..4].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax_b = col[4..].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax = amax_a.max(amax_b);
                let coarse = if amax != 0.0 {
                    amax * (1.0 / 127.0)
                } else {
                    0.0
                };
                let (sa, sb) = if amax == 0.0 {
                    (255u8, 255u8)
                } else {
                    (
                        (amax_a / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                        (amax_b / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                    )
                };
                let dh = half::f16::from_f32(coarse);
                let blk = &mut bytes[d * 36..(d + 1) * 36];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2] = sa;
                blk[3] = sb;
                // Encoder scales from the UNROUNDED coarse d ((coarse·s)/255,
                // mul then div — the GPU's op order).
                let act_a = coarse * sa as f32 * (1.0 / 255.0);
                let act_b = coarse * sb as f32 * (1.0 / 255.0);
                for (t, &v) in col.iter().enumerate() {
                    let act = if t < 4 { act_a } else { act_b };
                    let q = if act != 0.0 {
                        (v / act).round().clamp(-127.0, 127.0) as i8
                    } else {
                        0
                    };
                    blk[4 + t] = q as u8;
                }
                // Decoder scales from the f16-ROUNDED d (s/255 first, then
                // cd·that — the GPU's op order), then /outer.
                let cd = dh.to_f32();
                let da = cd * (sa as f32 * (1.0 / 255.0));
                let db = cd * (sb as f32 * (1.0 / 255.0));
                for (t, dt) in dec.iter_mut().enumerate() {
                    let q = blk[4 + t] as i8;
                    let sub = if t < 4 { da } else { db };
                    dt[d] = sub * q as f32 / spec.outer;
                }
            }
            bytes
        }
        8 => {
            // Q8_1: 36-byte blocks `{f16 d; f16 s; i8 qs[32]}` — d = amax/127,
            // s = the warp-tree sum (unused by decode), q = rne(x·127/amax)
            // via `__float2int_rn` (no clamp: |x| ≤ amax bounds it).
            let mut bytes = vec![0u8; SUB_DIM * 36];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let sum = warp_tree_sum(&col);
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                let dh = half::f16::from_f32(amax * (1.0 / 127.0));
                let sh = half::f16::from_f32(sum);
                let blk = &mut bytes[d * 36..(d + 1) * 36];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&sh.to_le_bytes());
                let ds = dh.to_f32();
                for (t, &v) in col.iter().enumerate() {
                    let q = (v * id).round_ties_even() as i32 as i8;
                    blk[4 + t] = q as u8;
                    if t < n {
                        dec[t][d] = ds * q as f32 / spec.outer;
                    }
                }
            }
            bytes
        }
        18 => {
            // Q4_KS: 20-byte blocks `{f16 d; u8 sa; u8 sb; u8 qs[16]}` — the
            // 4-bit sink format: elems 0-3 get sub-scale sa, 4-31 sb; biased
            // nibbles packed k | (k+16)<<4. Encoder scales from the UNROUNDED
            // coarse d; decoder from the f16-rounded d (GPU op orders).
            let mut bytes = vec![0u8; SUB_DIM * 20];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax_a = col[..4].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax_b = col[4..].iter().fold(0f32, |m, &v| m.max(v.abs()));
                let amax = amax_a.max(amax_b);
                let coarse = if amax != 0.0 { amax * (1.0 / 7.0) } else { 0.0 };
                let (sa, sb) = if amax == 0.0 {
                    (255u8, 255u8)
                } else {
                    (
                        (amax_a / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                        (amax_b / amax * 255.0).round().clamp(1.0, 255.0) as u8,
                    )
                };
                let dh = half::f16::from_f32(coarse);
                let blk = &mut bytes[d * 20..(d + 1) * 20];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2] = sa;
                blk[3] = sb;
                let act_a = coarse * sa as f32 * (1.0 / 255.0);
                let act_b = coarse * sb as f32 * (1.0 / 255.0);
                let mut qb = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    let act = if t < 4 { act_a } else { act_b };
                    let q = if act != 0.0 {
                        (v / act).round().clamp(-7.0, 7.0) as i32
                    } else {
                        0
                    };
                    qb[t] = (q + 8) as u8;
                }
                for k in 0..16 {
                    blk[4 + k] = qb[k] | (qb[k + 16] << 4);
                }
                let cd = dh.to_f32();
                let da = cd * (sa as f32 * (1.0 / 255.0));
                let db = cd * (sb as f32 * (1.0 / 255.0));
                for (t, dt) in dec.iter_mut().enumerate() {
                    let nib = if t < 16 {
                        (blk[4 + t] & 0xF) as i32 - 8
                    } else {
                        (blk[4 + t - 16] >> 4) as i32 - 8
                    };
                    let sc = if t < 4 { da } else { db };
                    dt[d] = sc * nib as f32 / spec.outer;
                }
            }
            bytes
        }
        19 => {
            // Q3_0: 14-byte blocks `{f16 d; u8 qh[4]; u8 qs[8]}` — 3-bit
            // symmetric centred at 3.5; low 2 bits in qs, high bit in qh.
            let mut bytes = vec![0u8; SUB_DIM * 14];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let dh = half::f16::from_f32(amax * (1.0 / 3.5));
                let id = if amax != 0.0 { 3.5 / amax } else { 0.0 };
                let blk = &mut bytes[d * 14..(d + 1) * 14];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = (v * id + 3.5).round().clamp(0.0, 7.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[2 + (t >> 3)] |= ((q >> 2) & 1) << (t & 7); // qh
                    blk[6 + (t >> 2)] |= (q & 3) << ((t & 3) * 2); // qs
                }
                let ds = dh.to_f32();
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = ds * (qv[t] as f32 - 3.5) / spec.outer;
                }
            }
            bytes
        }
        20 => {
            // Q3_1: 16-byte blocks `{f16 d; f16 m; u8 qh[4]; u8 qs[8]}` —
            // 3-bit affine (d = (max-min)/7, id from the UNROUNDED d). GPU
            // decode divides d and m by outer first, then PLAIN `d*q + m`
            // (no fma contraction in the kernel TU — measured: mul_add
            // shifts hundreds of cells by 1-2 bf16 codes at outer 1.5,
            // where d carries a full mantissa and the product rounds).
            let mut bytes = vec![0u8; SUB_DIM * 16];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let (mut vmax, mut vmin) = (col[0], col[0]);
                for &v in &col {
                    vmax = vmax.max(v);
                    vmin = vmin.min(v);
                }
                let dq = (vmax - vmin) * (1.0 / 7.0);
                let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                let dh = half::f16::from_f32(dq);
                let mh = half::f16::from_f32(vmin);
                let blk = &mut bytes[d * 16..(d + 1) * 16];
                blk[..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&mh.to_le_bytes());
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = ((v - vmin) * id).round().clamp(0.0, 7.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[4 + (t >> 3)] |= ((q >> 2) & 1) << (t & 7); // qh
                    blk[8 + (t >> 2)] |= (q & 3) << ((t & 3) * 2); // qs
                }
                let d32 = dh.to_f32() / spec.outer;
                let m32 = mh.to_f32() / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = d32 * qv[t] as f32 + m32;
                }
            }
            bytes
        }
        25 => {
            // Q2_S: 9-byte blocks `{i8 scale; u8 qs[8]}` — 2-bit symmetric
            // centred at 1.5 with an INT8 scale (d = amax/1.5, INT8-encoded
            // then decoded back for the round-trip-consistent quant step).
            let mut bytes = vec![0u8; SUB_DIM * 9];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let amax = col.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let scale =
                    ((amax * (1.0 / 1.5)) * 127.0).min(127.0).round_ties_even() as i32 as i8;
                let ds = scale as f32 * (1.0 / 127.0);
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 9..(d + 1) * 9];
                blk[0] = scale as u8;
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = (v * id + 1.5).round().clamp(0.0, 3.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[1 + (t >> 2)] |= (q & 3) << ((t & 3) * 2);
                }
                let dd = scale as f32 * (1.0 / 127.0) / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = dd * (qv[t] as f32 - 1.5);
                }
            }
            bytes
        }
        26 => {
            // Q2_A: 10-byte blocks `{i8 scale; i8 bias; u8 qs[8]}` — 2-bit
            // affine, both parameters INT8-encoded then decoded for the
            // quant step. Decode is PLAIN `d*q + m` (no fma contraction in
            // the kernel TU — same measured finding as Q3_1).
            let mut bytes = vec![0u8; SUB_DIM * 10];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let (mut vmax, mut vmin) = (col[0], col[0]);
                for &v in &col {
                    vmax = vmax.max(v);
                    vmin = vmin.min(v);
                }
                let scale = (((vmax - vmin) * (1.0 / 3.0)) * 127.0)
                    .min(127.0)
                    .round_ties_even() as i32 as i8;
                let bias = (vmin * 127.0).min(127.0).max(-127.0).round_ties_even() as i32 as i8;
                let ds = scale as f32 * (1.0 / 127.0);
                let m = bias as f32 * (1.0 / 127.0);
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                let blk = &mut bytes[d * 10..(d + 1) * 10];
                blk[0] = scale as u8;
                blk[1] = bias as u8;
                let mut qv = [0u8; CHUNK];
                for (t, &v) in col.iter().enumerate() {
                    qv[t] = ((v - m) * id).round().clamp(0.0, 3.0) as u8;
                }
                for (t, &q) in qv.iter().enumerate() {
                    blk[2 + (t >> 2)] |= (q & 3) << ((t & 3) * 2);
                }
                let dd = scale as f32 * (1.0 / 127.0) / spec.outer;
                let dm = bias as f32 * (1.0 / 127.0) / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = dd * qv[t] as f32 + dm;
                }
            }
            bytes
        }
        27 => {
            // Q1_S: 5-byte blocks `{i8 scale; u8 qs[4]}` — sign bits + the
            // INT8-encoded mean(|x|) magnitude (warp-tree sum order).
            let mut bytes = vec![0u8; SUB_DIM * 5];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let abs: [f32; CHUNK] = std::array::from_fn(|t| col[t].abs());
                let mean_abs = warp_tree_sum(&abs) / 32.0;
                let scale = (mean_abs * 127.0).min(127.0).round_ties_even() as i32 as i8;
                let blk = &mut bytes[d * 5..(d + 1) * 5];
                blk[0] = scale as u8;
                for (t, &v) in col.iter().enumerate() {
                    blk[1 + (t >> 3)] |= u8::from(v >= 0.0) << (t & 7);
                }
                let bs = scale as f32 * (1.0 / 127.0) / spec.outer;
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = if col[t] >= 0.0 { bs } else { -bs };
                }
            }
            bytes
        }
        29 => {
            // Q1_A: 6-byte blocks `{i8 scale_pos; i8 scale_neg; u8 qs[4]}` —
            // sign bits + a separate INT8 mean amplitude per sign class.
            let mut bytes = vec![0u8; SUB_DIM * 6];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let neg: [f32; CHUNK] =
                    std::array::from_fn(|t| if col[t] >= 0.0 { 0.0 } else { -col[t] });
                let pos_in: [f32; CHUNK] =
                    std::array::from_fn(|t| if col[t] >= 0.0 { col[t] } else { 0.0 });
                let n_pos = col.iter().filter(|&&v| v >= 0.0).count() as i32;
                let n_neg = 32 - n_pos;
                let sum_pos = warp_tree_sum(&pos_in);
                let sum_neg = warp_tree_sum(&neg);
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
                let sp = ((mean_pos * 127.0).round_ties_even() as i32).clamp(0, 127) as i8;
                let sn = ((mean_neg * 127.0).round_ties_even() as i32).clamp(0, 127) as i8;
                let blk = &mut bytes[d * 6..(d + 1) * 6];
                blk[0] = sp as u8;
                blk[1] = sn as u8;
                for (t, &v) in col.iter().enumerate() {
                    blk[2 + (t >> 3)] |= u8::from(v >= 0.0) << (t & 7);
                }
                let mp = sp as f32 * (1.0 / 127.0);
                let mn = sn as f32 * (1.0 / 127.0);
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = (if col[t] >= 0.0 { mp } else { -mn }) / spec.outer;
                }
            }
            bytes
        }
        28 => {
            // Q0_V: 2-byte pattern-indexed blocks. The Rust k_quants encoder
            // and decoder ARE the declared references for the CUDA pair —
            // K-side tables (the single latent is K≡V).
            let mut bytes = vec![0u8; SUB_DIM * 2];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let blk = k_quants::encode_block_q0_v::<true>(&col);
                bytes[d * 2..(d + 1) * 2].copy_from_slice(&blk.to_le_bytes());
                for (t, dt) in dec.iter_mut().enumerate() {
                    dt[d] = k_quants::q0_v_elem::<true>(&blk, t) / spec.outer;
                }
            }
            bytes
        }
        30 => {
            // Q0_X: 2-byte blocks `{i8 bulk_anchor; u8 outlier_packed}` —
            // INT8 block mean + one escaped outlier (first-lane tie-break,
            // 3-bit delta at stride 32). Anchor rounds THEN int-clamps.
            let mut bytes = vec![0u8; SUB_DIM * 2];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let mean = warp_tree_sum(&col) * (1.0 / 32.0);
                let anchor = ((mean * 127.0).round_ties_even() as i32).clamp(-127, 127);
                let residual: [i32; CHUNK] = std::array::from_fn(|t| {
                    ((col[t] * 127.0).round_ties_even() as i32).clamp(-127, 127) - anchor
                });
                let max_abs = residual.iter().map(|r| r.abs()).max().unwrap();
                let idx = residual.iter().position(|r| r.abs() == max_abs).unwrap();
                let delta = ((residual[idx] as f32 / 32.0).round_ties_even() as i32).clamp(-4, 3);
                let packed = (idx as u8 & 0x1F) | (((delta as u8) & 0x07) << 5);
                let blk = &mut bytes[d * 2..(d + 1) * 2];
                blk[0] = anchor as i8 as u8;
                blk[1] = packed;
                for (t, dt) in dec.iter_mut().enumerate() {
                    let ds = if t == idx { delta * 32 } else { 0 };
                    let v = (anchor + ds).clamp(-127, 127);
                    dt[d] = v as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        31 => {
            // Q0_M2: 3-byte blocks `{i8 centroid[2]; u8 qmask}` — Lloyd ×4
            // over quartet means (in-quartet XOR-tree sums, centroid updates
            // through the full warp tree).
            let mut bytes = vec![0u8; SUB_DIM * 3];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                // Quartet means: XOR-reduce offsets 1 then 2 within each 4.
                let mut qs = col;
                for off in [1usize, 2] {
                    let p = qs;
                    for i in 0..CHUNK {
                        qs[i] = p[i] + p[i ^ off];
                    }
                }
                let qt_mean: [f32; CHUNK] = std::array::from_fn(|i| qs[i] * 0.25);
                let mut c0 = qt_mean.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                let mut c1 = qt_mean.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                for _ in 0..4 {
                    let assign: [bool; CHUNK] =
                        std::array::from_fn(|i| (qt_mean[i] - c1).abs() < (qt_mean[i] - c0).abs());
                    let s0 =
                        warp_tree_sum(&std::array::from_fn(
                            |i| {
                                if assign[i] {
                                    0.0
                                } else {
                                    col[i]
                                }
                            },
                        ));
                    let n0 =
                        warp_tree_sum(&std::array::from_fn(|i| if assign[i] { 0.0 } else { 1.0 }));
                    let s1 =
                        warp_tree_sum(&std::array::from_fn(
                            |i| if assign[i] { col[i] } else { 0.0 },
                        ));
                    let n1 =
                        warp_tree_sum(&std::array::from_fn(|i| if assign[i] { 1.0 } else { 0.0 }));
                    if n0 > 0.0 {
                        c0 = s0 / n0;
                    }
                    if n1 > 0.0 {
                        c1 = s1 / n1;
                    }
                }
                let mut qmask = 0u8;
                for q in 0..8 {
                    let i = q * 4;
                    if (qt_mean[i] - c1).abs() < (qt_mean[i] - c0).abs() {
                        qmask |= 1 << q;
                    }
                }
                let blk = &mut bytes[d * 3..(d + 1) * 3];
                blk[0] = q0_encode_centroid(c0) as u8;
                blk[1] = q0_encode_centroid(c1) as u8;
                blk[2] = qmask;
                for (t, dt) in dec.iter_mut().enumerate() {
                    let c = blk[((qmask >> (t / 4)) & 1) as usize] as i8;
                    dt[d] = c as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        32 => {
            // Q0_M4: 8-byte blocks `{i8 centroid[4]; u32 qmask}` — Lloyd ×5
            // over pair means; centroids seeded equally-spaced vmin..vmax.
            let mut bytes = vec![0u8; SUB_DIM * 8];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let mut ps = col;
                {
                    let p = ps;
                    for i in 0..CHUNK {
                        ps[i] = p[i] + p[i ^ 1];
                    }
                }
                let pair_mean: [f32; CHUNK] = std::array::from_fn(|i| ps[i] * 0.5);
                let vmin = pair_mean.iter().fold(f32::INFINITY, |m, &v| m.min(v));
                let vmax = pair_mean.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let step = (vmax - vmin) * (1.0 / 3.0);
                let mut c = [vmin, vmin + step, vmin + 2.0 * step, vmax];
                let assign_of = |pm: f32, c: &[f32; 4]| -> usize {
                    let mut best = 0usize;
                    let mut best_d = (pm - c[0]).abs();
                    for (k, &ck) in c.iter().enumerate().skip(1) {
                        let dd = (pm - ck).abs();
                        if dd < best_d {
                            best_d = dd;
                            best = k;
                        }
                    }
                    best
                };
                for _ in 0..5 {
                    let assign: [usize; CHUNK] =
                        std::array::from_fn(|i| assign_of(pair_mean[i], &c));
                    for (k, ck) in c.iter_mut().enumerate() {
                        let sk = warp_tree_sum(&std::array::from_fn(|i| {
                            if assign[i] == k {
                                col[i]
                            } else {
                                0.0
                            }
                        }));
                        let nk = warp_tree_sum(&std::array::from_fn(|i| {
                            if assign[i] == k {
                                1.0
                            } else {
                                0.0
                            }
                        }));
                        if nk > 0.0 {
                            *ck = sk / nk;
                        }
                    }
                }
                let mut qmask = 0u32;
                for pair in 0..16 {
                    qmask |= (assign_of(pair_mean[pair * 2], &c) as u32) << (2 * pair);
                }
                let blk = &mut bytes[d * 8..(d + 1) * 8];
                for k in 0..4 {
                    blk[k] = q0_encode_centroid(c[k]) as u8;
                }
                blk[4..8].copy_from_slice(&qmask.to_le_bytes());
                for (t, dt) in dec.iter_mut().enumerate() {
                    let ci = ((qmask >> (2 * (t / 2))) & 3) as usize;
                    dt[d] = blk[ci] as i8 as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        33 => {
            // Q0: 1-byte blocks `{i8 centroid}` — the INT8-encoded block
            // mean, shared by all 32 elements.
            let mut bytes = vec![0u8; SUB_DIM];
            for d in 0..SUB_DIM {
                let mut col = [0f32; CHUNK];
                for (t, c) in col.iter_mut().enumerate() {
                    *c = toks.get(t).map_or(0.0, |tok| tok[d] * spec.outer);
                }
                let centroid = q0_encode_centroid(warp_tree_sum(&col) * (1.0 / 32.0));
                bytes[d] = centroid as u8;
                for dt in dec.iter_mut() {
                    dt[d] = centroid as f32 * (1.0 / 127.0) / spec.outer;
                }
            }
            bytes
        }
        f => candle::bail!("band_chunk_roundtrip: unsupported synthetic format {f}"),
    };
    Ok((bytes, dec))
}

/// The identity pal_map for the latent layout: dim d → band d/SUB_DIM.
///
/// One 4-bit id per dim (2 dims/byte) over the record's `[0,HD/2)` map region
/// (K≡V, so the old separate k_pal/v_pal halves are a single map). 4 bits names
/// all 16 bands, so the identity map is faithful.
pub fn identity_pal_map() -> [u8; HEAD_DIM / 2] {
    let mut m = [0u8; HEAD_DIM / 2];
    for d in 0..HEAD_DIM {
        m[d >> 1] |= ((d / SUB_DIM) as u8) << ((d & 1) * 4);
    }
    m
}

/// One round-tripped chunk: the per-band arena bytes, and the decoded token rows
/// with each dim back in its natural position.
pub type MappedChunk = ([Vec<u8>; N_BANDS], Vec<[f32; HEAD_DIM]>);

/// Band-chunk storage round-trip for the IDENTITY latent layout: band `p` holds
/// the natural dims `[p*SUB_DIM, p*SUB_DIM+SUB_DIM)`. Returns the per-band arena
/// bytes plus the decoded values in natural dim positions. Wraps
/// [`band_chunk_roundtrip`] (the proven per-band codec). There is no palette
/// regroup — the kernels read the identity layout only.
pub fn mapped_chunk_roundtrip(
    specs: &[BandSpec; N_BANDS],
    toks: &[[f32; HEAD_DIM]],
) -> Result<MappedChunk> {
    let n = toks.len().min(CHUNK);
    let mut dec = vec![[0f32; HEAD_DIM]; n];
    let mut bytes: [Vec<u8>; N_BANDS] = Default::default();
    for p in 0..N_BANDS {
        let band_toks: Vec<[f32; SUB_DIM]> = toks
            .iter()
            .take(n)
            .map(|tok| std::array::from_fn(|r| tok[p * SUB_DIM + r]))
            .collect();
        let (b, d) = band_chunk_roundtrip(specs[p], &band_toks)?;
        bytes[p] = b;
        for (t, row) in d.iter().enumerate() {
            for (r, &v) in row.iter().enumerate() {
                dec[t][p * SUB_DIM + r] = v;
            }
        }
    }
    Ok((bytes, dec))
}
