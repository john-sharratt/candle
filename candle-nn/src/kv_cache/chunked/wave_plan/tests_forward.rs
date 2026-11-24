//! Checks what the forward phase and the wave's setup tables cost, with and
//! without a draft head.
//!
//! The forward span carries the residual for every forward, and the head's
//! scored rows only where the head runs, so a window that stops short still pays
//! for the residual it hands on. The head's cost scales with sequences rather
//! than tokens, which the tests pin against the measured carves and against the
//! constant it replaced, which covered only 33 scored rows. A wave that runs no
//! head reserves only the setup its layers read.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{gated_partial_rotary, hyper_moe, hyper_moe_full};

    /// The wide residual rides the forward span for every forward, and the
    /// rows' embeddings where the head runs: at 100 rows, `100 × 10240 × 4` and
    /// `100 × 2560 × 4`. A window that stops short of the head hands its
    /// residual on from the span, so it is charged there too, and gathers no
    /// bare rows; a stack without a draft head gathers none either.
    #[test]
    fn the_forward_span_carries_the_residual_for_every_forward() {
        let g = hyper_moe_full();
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(WaveBuffer::HyperResidual.bytes(&g, wide), 4_096_000);
        assert_eq!(WaveBuffer::HeadRowEmbeds.bytes(&g, wide), 1_024_000);
        // One scored row: its gathered residual and its index.
        assert_eq!(WaveBuffer::HeadScoredResidual.bytes(&g, wide), 40_960);
        assert_eq!(WaveBuffer::HeadScoredRows.bytes(&g, wide), 4);
        let window = WaveWidth {
            prefill_rows: 100,
            ..WaveWidth::default()
        };
        assert_eq!(WaveBuffer::HyperResidual.bytes(&g, window), 4_096_000);
        assert_eq!(WaveBuffer::HeadRowEmbeds.bytes(&g, window), 0);
        let headless = hyper_moe();
        assert_eq!(WaveBuffer::HyperResidual.bytes(&headless, wide), 4_096_000);
        assert_eq!(WaveBuffer::HeadRowEmbeds.bytes(&headless, wide), 0);
    }

    /// A single-stream stack embeds onto the forward span in every forward,
    /// the window that stops short included — it copies the residual off the
    /// span as it hands it on — so the residual is charged either way: at 100
    /// rows of 1024 BF16, `100 × 1024 × 2`.
    #[test]
    fn a_single_stream_residual_is_on_the_span_with_or_without_the_head() {
        let g = gated_partial_rotary();
        let window = WaveWidth {
            prefill_rows: 100,
            ..WaveWidth::default()
        };
        assert_eq!(WaveBuffer::ForwardResidual.bytes(&g, window), 204_800);
        assert_eq!(
            WaveBuffer::ForwardResidual.bytes(&g, WaveWidth::prefill(100, 1)),
            204_800
        );
        assert_eq!(WaveBuffer::ForwardResidual.bytes(&hyper_moe(), window), 0);
    }

    /// The multi-stream stack's per-forward setup tables are on its forward
    /// span whether or not the wave reaches the head — every layer reads them —
    /// so a window that stops short still prices them: one prefill span's three
    /// offset tables (2, 1 and 1 words) and two words per row of token ids,
    /// with an alignment between its two carves.
    #[test]
    fn the_forward_setup_tables_are_priced_with_or_without_the_head() {
        let g = hyper_moe_full();
        let window = WaveWidth {
            prefill_rows: 100,
            prefill_spans: 1,
            ..WaveWidth::default()
        };
        for (b, want) in [
            (WaveBuffer::PrefillCuSeqlens, 8),
            (WaveBuffer::PrefillQLens, 4),
            (WaveBuffer::PrefillKvLens, 4),
        ] {
            assert_eq!(b.bytes(&g, window), want, "{b:?}");
        }
        assert_eq!(WaveBuffer::GlueWriteTables.bytes(&g, window), 0);
        // A single-stream stack builds the prefill group's set and the glue
        // group's, each table two carves an alignment apart, and the glue
        // group's three per-row scatter tables.
        let single = gated_partial_rotary();
        for (b, want) in [
            (WaveBuffer::PrefillCuSeqlens, 12 + BUMP_ALIGNMENT),
            (WaveBuffer::PrefillQLens, 4 + BUMP_ALIGNMENT),
            (WaveBuffer::PrefillKvLens, 4 + BUMP_ALIGNMENT),
            (WaveBuffer::GlueWriteTables, 1200 + 2 * BUMP_ALIGNMENT),
        ] {
            assert_eq!(b.bytes(&single, window), want, "{b:?}");
        }
        // The ids are every stack's: two words a row where they are read back,
        // one where the device inputs are only laid on the span.
        assert_eq!(
            WaveBuffer::RowTokenIds.bytes(&g, window),
            800 + BUMP_ALIGNMENT
        );
        assert_eq!(
            WaveBuffer::RowTokenIds.bytes(&gated_partial_rotary(), window),
            400
        );
        assert!(
            WavePlan::new(g).phase_bytes(LayerPhase::Forward, window)
                >= 3 * BUMP_ALIGNMENT + 800 + BUMP_ALIGNMENT
        );
    }

    /// A wave's image rows ride the forward span as the gather's second table,
    /// `rows × hidden` F32, and their turn as a row id and `head_dim` table
    /// words a row in three carves; a wave with no image prices neither.
    #[test]
    fn a_waves_images_are_priced_on_its_forward_span() {
        let g = hyper_moe_full();
        let images = WaveWidth {
            prefill_rows: 100,
            image_rows: 10,
            ..WaveWidth::default()
        };
        assert_eq!(WaveBuffer::ImageRows.bytes(&g, images), 10 * g.hidden * 4);
        assert_eq!(
            WaveBuffer::ImageTurns.bytes(&g, images),
            10 * (1 + g.head_dim) * 4 + 3 * BUMP_ALIGNMENT
        );
        let text = WaveWidth {
            image_rows: 0,
            ..images
        };
        assert_eq!(WaveBuffer::ImageRows.bytes(&g, text), 0);
        assert_eq!(WaveBuffer::ImageTurns.bytes(&g, text), 0);
    }

    /// **The forward phase against its measured generation, and against the
    /// constant it replaces.**
    ///
    /// Two carves on the 0.8B, `0 B lost to alignment`: the packed head norm at
    /// 1,152 B a scored row and the logits at 496,640 B (248,320 vocab × BF16) —
    /// 1,991,168 B at four scored rows, and 497,920 B at one. Ahead of them,
    /// with the residual on the span, the gather of the scored rows: 2,048 B a
    /// row in BF16 and a `u32` index, 8,208 B at four. A wave whose caller reads
    /// the logits in place also holds the accept walk's selections from them —
    /// at most the scored rows again, 496,640 B a row.
    ///
    /// The second half is the reason this matters beyond slack.
    /// `WAVE_FORWARD_BYTES` was 16 MiB, which covers 33 scored rows — and this
    /// engine composes waves of 64 sessions. Past that the phase needed more
    /// than its reservation, and the span would have exhausted *after* every
    /// layer had launched, as a refusal with nothing in it naming the head.
    #[test]
    fn the_forward_phase_prices_its_measured_generation_and_outgrows_the_old_constant() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        let w4 = WaveWidth::prefill(2100, 4);
        assert_eq!(
            plan.chain_bytes(Chain::Forward, w4),
            1_991_168 + 8_208,
            "the measured generation at four scored rows and their gather, to the byte"
        );
        assert_eq!(
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(2100, 1)),
            497_920 + 2_048 + 4,
            "and at one"
        );
        // A caller walking the logits in place adds its selections, one logits
        // row per scored row.
        let walked = WaveWidth {
            accept_rows: 4,
            ..w4
        };
        assert_eq!(WaveBuffer::AcceptRows.bytes(&g, walked), 4 * 496_640);
        // Each row under a grammar's bitmap holds one bit per token, rounded up
        // to whole words: 7,760 words at 248,320 tokens.
        let masked = WaveWidth {
            stencil_rows: 3,
            ..w4
        };
        assert_eq!(WaveBuffer::StencilBitmaps.bytes(&g, masked), 3 * 31_040);
        assert_eq!(WaveBuffer::StencilBitmaps.bytes(&g, w4), 0);
        assert_eq!(
            plan.chain_bytes(Chain::Forward, walked),
            1_991_168 + 4 * 496_640 + 8_208,
            "the head, the walk and the gather, to the byte"
        );
        // It scales with scored rows and not with tokens — the whole reason the
        // head is sized separately from the layer phases. (The setup prelude
        // ahead of it is per row: `a_wave_that_runs_no_head_prices_no_forward_phase`.)
        assert_eq!(
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(8192, 4)),
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(2100, 4)),
        );
        // The constant this replaces, and the width at which it stopped being a
        // reservation and started being a ceiling.
        const OLD_CONSTANT: usize = 16 << 20;
        let at_33 = plan.chain_bytes(Chain::Forward, WaveWidth::prefill(1, 33));
        let at_64 = plan.chain_bytes(Chain::Forward, WaveWidth::prefill(1, 64));
        assert!(at_33 <= OLD_CONSTANT, "33 sessions fitted the constant");
        assert!(
            at_64 > OLD_CONSTANT,
            "64 sessions need {at_64} B against the {OLD_CONSTANT} B the constant reserved"
        );
    }

    /// **One carve each for both span tables.** A wave carrying a verify block
    /// (a short span: three pointers) beside a prefill chunk (a long span: four)
    /// uploads the two tables' pointers as one carve and their extents as
    /// another, priced at four pointers and two extents a multi-row span — 64 B
    /// and 16 B for two spans, which hold the 56 B and 16 B the two tables take.
    #[test]
    fn the_span_tables_price_one_carve_for_short_and_long_spans() {
        let g = ModelGeometry {
            delta_net: Some(DeltaNetWidths {
                conv_dim: 8192,
                value_dim: 4096,
                n_v_heads: 32,
                layers: 24,
            }),
            ..gated_partial_rotary()
        };
        let w = WaveWidth::prefill(20, 2);
        assert_eq!(w.prefill_spans, 2);
        assert_eq!(WaveBuffer::DeltaNetLiveSpanPtrs.bytes(&g, w), 64);
        assert_eq!(WaveBuffer::DeltaNetLiveSpanExtents.bytes(&g, w), 16);
        assert!((3 + 4) * 8 <= WaveBuffer::DeltaNetLiveSpanPtrs.bytes(&g, w));
    }

    /// **A wave that stops short of the last layer runs no head**, so it
    /// reserves none of the head's forward phase — only the setup its layers
    /// read: the token ids the embedding gathers and the ragged prefill tables.
    /// A segmented sweep prices one window per layer range, and charging the
    /// head's full width to each was 17.4 MiB a window on a 60-session Qwen2.
    #[test]
    fn a_wave_that_runs_no_head_prices_no_forward_phase() {
        let plan = WavePlan::new(gated_partial_rotary());
        let no_head = WaveWidth {
            prefill_rows: 2100,
            ..WaveWidth::default()
        };
        assert_eq!(plan.chain_bytes(Chain::Forward, no_head), 0);
        let setup = plan.chain_bytes(Chain::ForwardSetup, no_head);
        assert!(setup >= 2100 * 4, "the ids alone are a word a row");
        assert_eq!(plan.phase_bytes(LayerPhase::Forward, no_head), setup);
        // The layer phases are untouched — the window still runs its layers.
        assert!(plan.phase_bytes(LayerPhase::Attention, no_head) > 0);
        assert!(plan.phase_bytes(LayerPhase::Ffn, no_head) > 0);
    }
}
