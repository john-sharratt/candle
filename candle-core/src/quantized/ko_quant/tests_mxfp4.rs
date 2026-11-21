//! Checks the MXFP4_KO int8 matmul and repack paths against float and byte
//! references.
//!
//! The inputs come from a deterministic pseudo-random generator rather than an RNG
//! dependency. The tests check that the per-32 weight scales are applied exactly,
//! so the only error left in the int8 matmul is the 8-bit activation quantization,
//! including when the four subs of a 128-column tile sit on very different
//! exponents. The native reorder must produce chunk bytes identical to the
//! from-F32 path, which pins the engine's no-requantize staging to the reference
//! packer. `mxfp4_ko_is_exact_repack_of_mxfp4` checks that the layout is a
//! bit-for-bit reordering of plain MXFP4.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..a.len() {
            let d = (a[i] - b[i]) as f64;
            num += d * d;
            den += (b[i] as f64).powi(2);
        }
        (num / den.max(1e-12)).sqrt()
    }

    // Deterministic pseudo-random f32 in [-1, 1] (no rng dependency in core tests).
    pub(in super::super) fn pseudo(n: usize) -> Vec<f32> {
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
}
