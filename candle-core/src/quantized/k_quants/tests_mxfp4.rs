//! Tests that pin the MXFP4 bytes, the GGUF file code, the E8M0 decode and the
//! dot product against llama.cpp's behaviour.
//!
//! The 17-byte layout is asserted directly, since `repr(C)` makes the field bytes
//! the on-disk bytes. Exact bytes are checked for a block representable at amax
//! 6.0, along with the subnormal E8M0 decode below exponent 2, the lower index
//! winning a tie in nearest-index selection, and GGUF file code 39. Hand-built raw
//! blocks decode lane by lane, and `vec_dot` is compared with a dequantized f32
//! reference.

use super::*;

#[cfg(test)]
pub(super) mod mxfp4_tests {
    use super::{e8m0_to_f32_half, BlockMXFP4, BlockQ8_0, GgmlType, MXFP4_KVALUES, QK_MXFP4};
    use crate::quantized::GgmlDType;

    /// The 17-byte block layout `{ e: u8, qs: [u8; 16] }` is what we assert against —
    /// `#[repr(C)]` makes the field bytes the on-disk bytes.
    #[test]
    fn block_is_17_bytes() {
        assert_eq!(std::mem::size_of::<BlockMXFP4>(), 17);
        assert_eq!(GgmlDType::MXFP4.type_size(), 17);
        assert_eq!(GgmlDType::MXFP4.block_size(), 32);
    }

    /// `ggml_e8m0_to_fp32_half`: exact power-of-two decode incl. the subnormal `e<2` path.
    #[test]
    fn e8m0_half_exact() {
        assert_eq!(e8m0_to_f32_half(129), 2.0); // 2^(129-128)
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(125), 0.125); // 2^-3
        assert_eq!(e8m0_to_f32_half(1), f32::from_bits(0x0040_0000)); // 2^-127
        assert_eq!(e8m0_to_f32_half(0), f32::from_bits(0x0020_0000)); // 2^-128
    }

    /// GGUF on-disk file code for MXFP4 is 39 (matches llama.cpp `GGML_TYPE_MXFP4`), and
    /// it round-trips through our translation boundary.
    #[test]
    fn gguf_file_code_is_39() {
        assert_eq!(GgmlDType::MXFP4.to_gguf_file_code(), 39);
        assert_eq!(
            GgmlDType::from_gguf_file_code(39).unwrap(),
            GgmlDType::MXFP4
        );
        assert_eq!(GgmlDType::from_u32(49).unwrap(), GgmlDType::MXFP4);
    }

    /// Encode a block whose values are all exactly representable at `amax = 6.0`
    /// (⇒ `e = 127`, `d = 0.5`, real table `{0,.5,1,1.5,2,3,4,6}`), and assert the exact
    /// bytes produced, then that dequant recovers the inputs bit-for-bit.
    #[test]
    fn exact_bytes_and_roundtrip() {
        let mut xs = vec![0f32; QK_MXFP4];
        // Low nibbles come from positions 0..16, high nibbles from 16..32.
        xs[0] = 6.0; // idx 7  (kvalues 12 * 0.5)
        xs[1] = 3.0; // idx 5  (6 * 0.5)
        xs[2] = 1.5; // idx 3  (3 * 0.5)
        xs[3] = 0.5; // idx 1  (1 * 0.5)
        xs[16] = -6.0; // idx 15
        xs[17] = -3.0; // idx 13
        xs[18] = 4.0; // idx 6  (8 * 0.5)
        xs[19] = -0.5; // idx 9

        let mut blk = [BlockMXFP4::zeros()];
        BlockMXFP4::from_float(&xs, &mut blk);

        assert_eq!(blk[0].e, 127);
        assert_eq!(blk[0].qs[0], 7 | (15 << 4));
        assert_eq!(blk[0].qs[1], 5 | (13 << 4));
        assert_eq!(blk[0].qs[2], 3 | (6 << 4));
        assert_eq!(blk[0].qs[3], 1 | (9 << 4));
        for (j, &byte) in blk[0].qs.iter().enumerate().take(16).skip(4) {
            assert_eq!(byte, 0, "qs[{j}] should be zero");
        }

        let mut ys = vec![0f32; QK_MXFP4];
        BlockMXFP4::to_float(&blk, &mut ys);
        assert_eq!(ys[0], 6.0);
        assert_eq!(ys[1], 3.0);
        assert_eq!(ys[2], 1.5);
        assert_eq!(ys[3], 0.5);
        assert_eq!(ys[16], -6.0);
        assert_eq!(ys[17], -3.0);
        assert_eq!(ys[18], 4.0);
        assert_eq!(ys[19], -0.5);
        for &y in ys.iter().take(16).skip(4) {
            assert_eq!(y, 0.0);
        }
    }

    /// Ties in `best_index_mxfp4` resolve to the lower index (ggml scans ascending and
    /// keeps strictly-smaller errors). At `d = 0.5`, `2.5` is equidistant from `2.0`
    /// (idx 4) and `3.0` (idx 5) ⇒ idx 4.
    #[test]
    fn quant_tie_breaks_low() {
        let mut xs = vec![0f32; QK_MXFP4];
        xs[0] = 6.0; // pin amax so e = 127, d = 0.5
        xs[1] = 2.5; // tie between idx 4 (2.0) and idx 5 (3.0)
        let mut blk = [BlockMXFP4::zeros()];
        BlockMXFP4::from_float(&xs, &mut blk);
        assert_eq!(blk[0].e, 127);
        assert_eq!(blk[0].qs[1] & 0x0F, 4);
    }

    /// Decode a hand-built raw block (bytes chosen directly) and check every lane,
    /// exercising the low/high nibble split and a non-0.5 scale (`e = 125 ⇒ d = 0.125`).
    #[test]
    fn decode_known_raw_block() {
        let mut blk = BlockMXFP4::zeros();
        blk.e = 125; // d = 0.125
        blk.qs[0] = 6 | (7 << 4); // low idx 6 -> 8, high idx 7 -> 12
        blk.qs[5] = 14 | (2 << 4); // low idx 14 -> -8, high idx 2 -> 2
        let mut ys = vec![0f32; QK_MXFP4];
        BlockMXFP4::to_float(std::slice::from_ref(&blk), &mut ys);
        assert_eq!(ys[0], 8.0 * 0.125); // 1.0
        assert_eq!(ys[16], 12.0 * 0.125); // 1.5
        assert_eq!(ys[5], -8.0 * 0.125); // -1.0
        assert_eq!(ys[21], 2.0 * 0.125); // 0.25
    }

    /// `vec_dot` against Q8_0 activations equals the exact integer-table dot scaled by the
    /// two per-block scales — verified against a from-scratch f32 reference on the
    /// dequantized operands.
    #[test]
    fn vec_dot_matches_dequant_reference() {
        let mut wq = vec![0f32; QK_MXFP4];
        let mut aq = vec![0f32; QK_MXFP4];
        for i in 0..QK_MXFP4 {
            // Values chosen to be exactly representable so the reference is unambiguous.
            wq[i] = MXFP4_KVALUES[i % 8] as f32 * 0.5;
            aq[i] = ((i as f32) - 16.0) * 0.03;
        }
        wq[0] = 6.0; // pin weight amax -> e = 127
        let mut wblk = [BlockMXFP4::zeros()];
        BlockMXFP4::from_float(&wq, &mut wblk);
        let mut ablk = [BlockQ8_0::zeros()];
        BlockQ8_0::from_float(&aq, &mut ablk);

        let got = BlockMXFP4::vec_dot(QK_MXFP4, &wblk, &ablk);

        // Reference: dequantize both operands and dot in f32.
        let mut wdq = vec![0f32; QK_MXFP4];
        let mut adq = vec![0f32; QK_MXFP4];
        BlockMXFP4::to_float(&wblk, &mut wdq);
        BlockQ8_0::to_float(&ablk, &mut adq);
        let want: f32 = wdq.iter().zip(adq.iter()).map(|(a, b)| a * b).sum();
        assert!((got - want).abs() < 1e-4, "got {got} want {want}");
    }
}
