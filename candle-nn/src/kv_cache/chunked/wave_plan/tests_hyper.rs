//! Checks the Gated Residual pre-mix, the attention mixer's dtype on such a
//! stack, and the PLE block, against their carves.
//!
//! The pre-mix is priced carve by carve as the cursor walks, with the KO padding
//! applied, and a float session's version is checked without operands. A stack
//! with parallel residual streams carves its attention mixer in the wave's dtype
//! while its residual stays F32, so pricing the mixer as F32 would double every
//! Q, gate and context buffer. The PLE block's four carves are checked, and the
//! float norms carve nothing under the pre-mix.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{hyper_moe, hyper_moe_full, moe};

    /// The pre-mix at 100 rows on an int8 session, carve by carve, with the KO
    /// projections' padding (rank 320 → 384, down rows 388 → 416): the grouped
    /// norm (100 × 10240 × 4 = 4,096,000), its q8a128 operand (8,000 tiles →
    /// 1,000 blocks × 1,152 = 1,152,000), the down projection
    /// (100 × 416 × 4 = 166,400), the SiLU (100 × 384 × 4 = 153,600), its
    /// operand (300 tiles → 38 blocks × 1,152 = 43,776), the raw gate
    /// (4,096,000) and the mix (1,024,000). Every carve starts on a 256
    /// boundary here, so the walk is their sum.
    #[test]
    fn the_hyper_prelude_prices_its_carves_as_the_cursor_walks() {
        let p = WavePlan::new(hyper_moe());
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(p.chain_bytes(Chain::HyperAttn, wide), 10_731_776);
        assert_eq!(p.chain_bytes(Chain::HyperFfn, wide), 10_731_776);
        // A float session has no operands: 4,096,000 + 166,400 + 153,600 +
        // 4,096,000 + 1,024,000.
        let float = WavePlan::new(ModelGeometry {
            packed_norm: false,
            ..hyper_moe()
        });
        assert_eq!(float.chain_bytes(Chain::HyperAttn, wide), 9_536_000);
        // Absent on every stack without hyper-connections.
        let plain = WavePlan::new(moe());
        assert_eq!(plain.chain_bytes(Chain::HyperAttn, wide), 0);
        assert_eq!(plain.chain_bytes(Chain::HyperFfn, wide), 0);
    }

    /// A hyper stack's attention mixer carves in the wave's dtype, its residual
    /// in F32. At 100 rows over q|k|v = (32 + 2·4) × 128 = 5,120 columns: the
    /// projection and the stacked split at two bytes (100 × 5,120 × 2 =
    /// 1,024,000 each), where an F32 mixer pays twice that; `o_proj`'s output
    /// is the residual's (100 × 2,560 × 4 = 1,024,000) under either; and the
    /// residual-side chains do not move.
    #[test]
    fn a_hyper_stack_prices_its_attention_mixer_in_the_wave_dtype() {
        let f32 = hyper_moe_full();
        let bf16 = ModelGeometry {
            hyper: f32.hyper.map(|h| HyperWidths {
                mixer_dtype: DType::BF16,
                ..h
            }),
            ..f32
        };
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(WaveBuffer::QkvProjection.bytes(&bf16, wide), 1_024_000);
        assert_eq!(WaveBuffer::QSplit.bytes(&bf16, wide), 1_024_000);
        assert_eq!(WaveBuffer::QkvProjection.bytes(&f32, wide), 2_048_000);
        assert_eq!(WaveBuffer::QSplit.bytes(&f32, wide), 2_048_000);
        assert_eq!(WaveBuffer::OProjOutput.bytes(&bf16, wide), 1_024_000);
        assert_eq!(WaveBuffer::OProjOutput.bytes(&f32, wide), 1_024_000);
        let (a, b) = (WavePlan::new(f32), WavePlan::new(bf16));
        for chain in [
            Chain::HyperAttn,
            Chain::DeltaNet,
            Chain::HeadPass,
            Chain::Ffn,
        ] {
            assert_eq!(
                a.chain_bytes(chain, wide),
                b.chain_bytes(chain, wide),
                "{chain:?}"
            );
        }
        assert!(b.chain_bytes(Chain::Attention, wide) < a.chain_bytes(Chain::Attention, wide));
    }

    /// The PLE block at 100 rows over one sequence: the table rows
    /// (1,024,000), the stacked key|value (100 × 12800 × 4 = 5,120,000), the
    /// normed conv input (4,096,000) and one span's four words (32).
    #[test]
    fn the_ple_block_prices_its_four_carves() {
        let p = WavePlan::new(hyper_moe_full());
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(p.chain_bytes(Chain::Ple, wide), 10_240_032);
        assert_eq!(WavePlan::new(hyper_moe()).chain_bytes(Chain::Ple, wide), 0);
    }

    /// Under the pre-mix a float session's norms carve nothing: the mix output
    /// IS the block input the mixer and the experts read.
    #[test]
    fn the_pre_mix_replaces_the_float_norms() {
        let g = ModelGeometry {
            packed_norm: false,
            ..hyper_moe()
        };
        let wide = WaveWidth::prefill(100, 1);
        for b in [
            WaveBuffer::AttnNorm,
            WaveBuffer::FfnNorm,
            WaveBuffer::DeltaNetNorm,
            WaveBuffer::DeltaNetBetaOperand,
            WaveBuffer::DeltaNetAlphaOperand,
        ] {
            assert_eq!(b.bytes(&g, wide), 0, "{b:?}");
        }
    }
}
