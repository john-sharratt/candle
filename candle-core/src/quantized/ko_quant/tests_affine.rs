//! Pins the affine KO codecs to raw bytes, round-trips, and the matmul tiling rule.
//!
//! The codec tests assert exact bytes on inputs whose dequantization is exact, so
//! a bit-order or plane-shift error fails on the byte it lands in rather than
//! inside a tolerance. The Q2_KO and Q3_KO cases use inputs chosen so the crumb and
//! hi planes disagree, which a shift bug in either one exposes.
//! `the_plane_stack_reproduces_the_hand_written_q2_ko` checks the generalised
//! packer against the older hand-written one, so a refactor cannot move bits
//! without a failure. The tiling test pins `ko_tileable` at the boundary between
//! the 8-row storage chunk and the 32-row matmul tile, since three call sites rely
//! on that one definition.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_mxfp4::tests::{pseudo, rel_l2};

    /// The matmul tiling rule, pinned at the boundary that separates it from the
    /// storage chunk. Three call sites share this predicate — `repack_ko`'s bail,
    /// `prepare::ko_target`, and `QMatMul::from_weights_with_mode`'s dense
    /// fallback — and they only stay in agreement while it has one definition.
    #[test]
    fn ko_tileable_is_the_matmul_bound_not_the_storage_bound() {
        // The storage chunk packs 8 rows; the matmul tiles 32. Every row count
        // between them packs and is still rejected — the exact gap that decides
        // whether a narrow projection loads int8 or dense.
        assert!(!ko_tileable(8, 128), "8 rows pack but the matmul tiles 32");
        assert!(
            !ko_tileable(16, 128),
            "Qwen3.5-0.8B w_alpha/w_beta [16, hidden]"
        );
        assert!(!ko_tileable(24, 128), "the mHC fn_w at mix_hc = 24");
        assert!(ko_tileable(32, 128));

        // K is bounded by the per-128 affine block, independently of N.
        assert!(!ko_tileable(32, 64));
        assert!(!ko_tileable(32, 129));
        assert!(ko_tileable(32, 256));

        // A production shape both ways round: tileable in N, and the same
        // numbers transposed are not necessarily tileable in K.
        assert!(ko_tileable(4096, 4096));
        assert!(!ko_tileable(4096, 100));
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
            (GgmlDType::Q3_KO, 0.180), // ~0.14  (≈1/7, the 3-bit affine floor — 8 levels)
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

    /// **The generalisation moved no bits.** [`KoPlanes`] replaced a hand-written Q2_KO packer
    /// with a plane-stack rule shared by all five affine widths; `quantize_q2_ko` /
    /// `dequant_q2_ko` survive as an independently written statement of the same layout, and
    /// this asserts the two agree byte-for-byte on input that exercises every crumb value and a
    /// non-trivial `dm`. A refactor of a codec that is only round-trip-tested proves nothing —
    /// both halves can move together.
    #[test]
    fn the_plane_stack_reproduces_the_hand_written_q2_ko() {
        let (nrows, ncols) = (24usize, 384usize);
        let w = pseudo(nrows * ncols);
        let generic = quantize_ko(&w, nrows, ncols, GgmlDType::Q2_KO);
        let oracle = quantize_q2_ko(&w, nrows, ncols);
        assert_eq!(
            generic, oracle,
            "KoPlanes must emit the hand-written Q2_KO bytes exactly"
        );
        assert_eq!(
            dequant_ko(&generic, nrows, ncols, GgmlDType::Q2_KO),
            dequant_q2_ko(&oracle, nrows, ncols),
            "and the generic dequant must read them the same way"
        );
    }

    /// Q3_KO RAW BYTES (per the codec rule — exact bytes, not a tolerance). One 8×128 chunk
    /// whose every row is `0,1,…,7,0,1,…` → scale 1, min 0, `q[k] = k % 8`.
    ///
    /// Each (lane, sub) quad covers 4 consecutive K, so the quads see values `(0,1,2,3)` then
    /// `(4,5,6,7)` alternating. The crumb plane holds bits 0-1 → `0,1,2,3` packs to `0xE4` and
    /// `4,5,6,7` packs to `0xE4` as well (`q & 3` is the same). The hi plane holds bit 2 → the
    /// first quad gives `0b0000` and the second `0b1111`. So the crumb region is uniformly
    /// `0xE4` and the hi region alternates by the quad's parity, which is exactly what pins the
    /// two planes apart: a bug that read hi at the wrong shift leaves the crumb bytes intact.
    #[test]
    fn q3_ko_raw_bytes() {
        let (nrows, ncols) = (8usize, 128usize);
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|i| ((i % ncols) % 8) as f32)
            .collect();
        let chunk = quantize_ko(&w, nrows, ncols, GgmlDType::Q3_KO);
        assert_eq!(
            chunk.len(),
            416,
            "one 8×128 chunk = 256 crumb + 128 hi + 32 dm"
        );
        assert_eq!(ko_chunk_bytes(GgmlDType::Q3_KO), 416);

        for (i, b) in chunk[..256].iter().enumerate() {
            assert_eq!(
                *b, 0xE4,
                "crumb byte {i}: bits 0-1 of 0,1,2,3 (and of 4,5,6,7)"
            );
        }
        // hi byte at `256 + lane*4 + sub` packs the low-half quad in its low nibble and the
        // high-half quad in its high nibble. For row r, lane = r*4 + q3, and the low half of
        // sub s covers K = s*32 + q3*4 + {0..3}; bit 2 of `K % 8` is 1 exactly when
        // `(K / 4)` is odd, i.e. when `q3` is odd (s*32 and 4 are both even multiples).
        for r in 0..8 {
            for q3 in 0..4 {
                let lane = r * 4 + q3;
                for sub in 0..4 {
                    let got = chunk[256 + lane * 4 + sub];
                    // low half: K = sub*32 + q3*4 + j  → bit2 set iff q3 odd
                    // high half: K = sub*32 + 16 + q3*4 + j → +16 keeps K/4 parity, so same
                    let want = if q3 % 2 == 1 { 0xFFu8 } else { 0x00u8 };
                    assert_eq!(got, want, "hi byte r{r} q3{q3} sub{sub}");
                }
            }
        }
        for r in 0..8 {
            let d = 384 + r * 4;
            assert_eq!(
                &chunk[d..d + 4],
                &[0x00, 0x3C, 0x00, 0x00],
                "dm row {r} = (scale 1.0, min 0.0) f16"
            );
        }
        // On-grid input → dequant is EXACT (scale·q + min with q ∈ 0..=7, scale=1, min=0).
        let de = dequant_ko(&chunk, nrows, ncols, GgmlDType::Q3_KO);
        assert_eq!(de, w, "Q3_KO dequant must be exact on the 8-level grid");
    }

    /// Q3_KO's two planes carry **different** bits of the value, and a shift bug in either one
    /// survives the uniform test above. Here every value in a quad is distinct mod 4 *and* mod
    /// 8, so crumb and hi disagree.
    #[test]
    fn q3_ko_splits_the_value_across_crumb_and_hi() {
        let (nrows, ncols) = (8usize, 128usize);
        let mut w: Vec<f32> = vec![0.0; nrows * ncols];
        for r in 0..nrows {
            let base = r * ncols;
            // K = 0..3 → 7,5,2,0: crumb bits (3,1,2,0) → 3|1<<2|2<<4|0<<6 = 0x27
            //                      hi bits    (1,1,0,0) → 0b0011 = 0x3
            for (k, v) in [7.0f32, 5.0, 2.0, 0.0].iter().enumerate() {
                w[base + k] = *v;
            }
            // give the row full range so scale = 1, min = 0
            for (k, v) in [0.0f32, 1.0, 2.0, 7.0].iter().enumerate() {
                w[base + 4 + k] = *v;
            }
        }
        let chunk = quantize_ko(&w, nrows, ncols, GgmlDType::Q3_KO);
        assert_eq!(chunk[0], 0x27, "cr0 for K=0..3 values (7,5,2,0), bits 0-1");
        // lane 0, sub 0 → hi byte at 256; low nibble = bit-2 of (7,5,2,0) = 1,1,0,0.
        assert_eq!(chunk[256] & 0x0F, 0x03, "hi nibble for K=0..3");
        let de = dequant_ko(&chunk, nrows, ncols, GgmlDType::Q3_KO);
        assert_eq!(&de[0..4], &[7.0, 5.0, 2.0, 0.0], "round trip on the grid");
    }

    /// Every affine KO width round-trips exactly on its own grid, and each is the byte size
    /// [`ko_chunk_bytes`] promises. One table, so a new width cannot be added without landing
    /// here — which is the property [`KoPlanes`] exists to give.
    #[test]
    fn every_affine_ko_width_round_trips_on_its_grid() {
        let (nrows, ncols) = (8usize, 128usize);
        for (dtype, maxq, bytes) in [
            (GgmlDType::Q2_KO, 3usize, 288usize),
            (GgmlDType::Q3_KO, 7, 416),
            (GgmlDType::Q4_KO, 15, 544),
            (GgmlDType::Q5_KO, 31, 672),
            (GgmlDType::Q6_KO, 63, 800),
        ] {
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|i| ((i % ncols) % (maxq + 1)) as f32)
                .collect();
            let chunk = quantize_ko(&w, nrows, ncols, dtype);
            assert_eq!(chunk.len(), bytes, "{dtype:?} chunk bytes");
            assert_eq!(ko_chunk_bytes(dtype), bytes, "{dtype:?} ko_chunk_bytes");
            let de = dequant_ko(&chunk, nrows, ncols, dtype);
            assert_eq!(de, w, "{dtype:?} must round-trip exactly on its own grid");
        }
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

    /// `pack_q4_ko` with supplied codes must byte-match `quantize_ko` whenever
    /// the observer would derive the same grid — i.e. on data whose every
    /// 128-group spans code 0 and code 15 exactly.
    #[test]
    fn pack_q4_ko_matches_quantize_ko_on_grid_spanning_data() {
        let (nrows, ncols) = (8usize, 256usize);
        let k_blocks = ncols / 128;
        let mut codes = vec![0u8; nrows * ncols];
        let mut lcg = 12345u64;
        for c in codes.iter_mut() {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            *c = ((lcg >> 33) % 16) as u8;
        }
        // Force each group to span the grid so the observer lands on it too.
        for row in 0..nrows {
            for kb in 0..k_blocks {
                codes[row * ncols + kb * 128] = 0;
                codes[row * ncols + kb * 128 + 1] = 15;
            }
        }
        let scale = 0.03125f32; // exact in f16
        let mn = -8.0 * scale;
        let dm = vec![(scale, mn); nrows * k_blocks];
        let w: Vec<f32> = codes.iter().map(|&c| scale * c as f32 + mn).collect();

        let packed = pack_q4_ko(&codes, &dm, nrows, ncols);
        let observed = quantize_ko(&w, nrows, ncols, GgmlDType::Q4_KO);
        assert_eq!(
            packed, observed,
            "packed bytes diverge from the observer path"
        );
    }

    /// The importer's exactness contract: dequant(pack(codes, s, −8s)) must
    /// reproduce `s·(code−8)` bit-for-bit — including groups whose codes do
    /// NOT span the grid, which is precisely where the observer re-rounds and
    /// this packer must not.
    #[test]
    fn pack_q4_ko_is_bit_exact_on_non_spanning_groups() {
        let (nrows, ncols) = (8usize, 128usize);
        // A narrow code range: the observer would re-derive scale=(mx−mn)/15
        // and re-round; the packer must keep the source grid untouched.
        let codes: Vec<u8> = (0..nrows * ncols).map(|i| 5 + (i % 4) as u8).collect();
        let scale = 0.0078125f32; // 2^-7, exact in f16
        let mn = -8.0 * scale;
        let dm = vec![(scale, mn); nrows];
        let packed = pack_q4_ko(&codes, &dm, nrows, ncols);
        let de = dequant_ko(&packed, nrows, ncols, GgmlDType::Q4_KO);
        for (i, (&code, &got)) in codes.iter().zip(de.iter()).enumerate() {
            let want = scale * code as f32 + mn; // == s·(q−8) exactly
            assert!(
                got == want,
                "elem {i}: {got} != {want} (code {code}) — the import re-rounded"
            );
        }
    }

    /// `unpack_q4_ko` recovers exactly what `pack_q4_ko` was given — the fast
    /// verify's contract. The dequant tests above are what anchor the shared
    /// permutation table to the real layout; this pins the inverse to it.
    #[test]
    fn unpack_q4_ko_recovers_codes_and_dm() {
        let (nrows, ncols) = (16usize, 256usize);
        let k_blocks = ncols / 128;
        let mut lcg = 7u64;
        let codes: Vec<u8> = (0..nrows * ncols)
            .map(|_| {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((lcg >> 33) % 16) as u8
            })
            .collect();
        let dm: Vec<(f32, f32)> = (0..nrows * k_blocks)
            .map(|i| {
                let s = 2f32.powi((i % 6) as i32 - 8);
                (s, -8.0 * s)
            })
            .collect();
        let packed = pack_q4_ko(&codes, &dm, nrows, ncols);
        let (codes2, dm2) = unpack_q4_ko(&packed, nrows, ncols);
        assert_eq!(codes, codes2, "codes did not round-trip");
        assert_eq!(dm, dm2, "dm pairs did not round-trip");
    }
}
