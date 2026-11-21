//! Pins the DeltaNet chains to their census, byte for byte.
//!
//! The pre-mix and live mixer are checked against an itemisation of a
//! hybrid layer at 2,100 rows over four spans, and the single-stream chain
//! against the 0.8B at 134 prefill rows. The census stopped before the norm-gate
//! and `w_out`, so those two are pinned from the chain's own arithmetic, and the
//! test says so. Each test walks the chain as the cursor walks it and asserts the
//! total.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{gated_partial_rotary, hyper_hybrid};

    /// Pinned to the `wave-census-labels` itemisation of a DeltaNet layer's
    /// attention-phase generation on Qwen3.8-Flash-Next at 2100 rows over four
    /// prefill spans: the pre-mix's full-width carves, then the live mixer's,
    /// byte for byte, through the scan's four transients. The pre-mix's two
    /// projection outputs are pinned by
    /// [`the_hyper_prelude_prices_its_carves_as_the_cursor_walks`], whose widths
    /// the KO padding sets.
    ///
    /// The census stopped at `g_cs`: the span it measured was priced without
    /// `w` and `kq`, so it filled there and the norm-gate and `w_out` fell back
    /// to the pool. Those three are the chain's own arithmetic — `value_dim`
    /// F32, its q8a128 operand (12,902,400 elements → 12,600 blocks × 1,152),
    /// and `hidden` in the F32 activation dtype.
    #[test]
    fn the_pre_mix_and_live_delta_net_match_the_census() {
        let g = hyper_hybrid();
        let w = WaveWidth::prefill(2100, 4);
        let measured = [
            (WaveBuffer::HyperAttnNorm, 86_016_000),
            (WaveBuffer::HyperAttnGateRaw, 86_016_000),
            (WaveBuffer::HyperAttnMixed, 21_504_000),
            (WaveBuffer::DeltaNetLiveOperand, 6_048_000),
            (WaveBuffer::DeltaNetLiveProjection, 138_432_000),
            (WaveBuffer::DeltaNetLiveSplit, 138_432_000),
            (WaveBuffer::DeltaNetLiveSpanPtrs, 128),
            (WaveBuffer::DeltaNetLiveSpanExtents, 32),
            (WaveBuffer::DeltaNetLiveConv, 86_016_000),
            (WaveBuffer::DeltaNetLiveScanOut, 51_609_600),
            (WaveBuffer::DeltaNetLiveScanU, 51_609_600),
            (WaveBuffer::DeltaNetLiveScanW, 51_609_600),
            (WaveBuffer::DeltaNetLiveScanKq, 25_804_800),
            (WaveBuffer::DeltaNetLiveDecay, 403_200),
            (WaveBuffer::DeltaNetLiveNormGate, 51_609_600),
            (WaveBuffer::DeltaNetLiveOutOperand, 14_515_200),
            (WaveBuffer::DeltaNetLiveOut, 21_504_000),
        ];
        for (b, bytes) in measured {
            assert_eq!(b.bytes(&g, w), bytes, "{b:?}");
        }
    }

    /// **The single-stream DeltaNet chain against its measured carves.**
    ///
    /// The `KV_WAVE_CENSUS=labels` itemisation of the 0.8B's DeltaNet layer at
    /// 134 prefill rows over one span, carve for carve up to the conv: the
    /// norm, the `[q|k|v]` and `z` projections each behind its q8a128 operand,
    /// `beta` and `alpha` each behind its F32 upcast, the two span tables, the
    /// conv. The layer norm roots the chain on the phase and every projection
    /// inherits it, so the whole mixer is on the span.
    #[test]
    fn the_delta_net_chain_prices_its_measured_generation() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        let rows = WaveWidth::prefill(134, 1);
        for (b, want) in [
            (WaveBuffer::DeltaNetNorm, 274_432),
            (WaveBuffer::DeltaNetQkvOperand, 154_368),
            (WaveBuffer::DeltaNetQkvProj, 3_293_184),
            (WaveBuffer::DeltaNetZOperand, 154_368),
            (WaveBuffer::DeltaNetZProj, 1_097_728),
            (WaveBuffer::DeltaNetBetaOperand, 548_864),
            (WaveBuffer::DeltaNetBetaProj, 8_576),
            (WaveBuffer::DeltaNetAlphaOperand, 548_864),
            (WaveBuffer::DeltaNetAlphaProj, 8_576),
            (WaveBuffer::DeltaNetLiveSpanPtrs, 32),
            (WaveBuffer::DeltaNetLiveSpanExtents, 8),
            (WaveBuffer::DeltaNetLiveConv, 3_293_184),
        ] {
            assert_eq!(b.bytes(&g, rows), want, "{b:?} against its measured carve");
        }
        // The pre-mix stack's stacked projection is not this stack's.
        for b in [
            WaveBuffer::DeltaNetLiveOperand,
            WaveBuffer::DeltaNetLiveProjection,
            WaveBuffer::DeltaNetLiveSplit,
        ] {
            assert_eq!(b.bytes(&g, rows), 0, "{b:?}");
        }
        // The chain is the walk of every buffer it prices, nothing more.
        let walked = WaveBuffer::iter()
            .filter(|b| b.chain() == Chain::DeltaNet)
            .map(|b| b.bytes(&g, rows))
            .filter(|&len| len > 0)
            .fold(0usize, |cursor, len| {
                cursor.div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT + len
            });
        assert_eq!(plan.chain_bytes(Chain::DeltaNet, rows), walked);
    }
}
