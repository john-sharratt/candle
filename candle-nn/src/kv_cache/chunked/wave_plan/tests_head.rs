//! Checks the draft head's pass against its measured carves and against the
//! attention phase it runs in.
//!
//! The head's input assembly is priced carve by carve on an int8 session, and a
//! single-stream draft head's pass is checked the same way. One test reproduces
//! the failure where the head pass set the attention phase's price and the
//! projection's operand was missing from the plan, and asserts that the assembly
//! fits the phase it prices. The head is two different chains depending on
//! whether its session is int8 or float, and each is checked against its own
//! measurement.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::prefill_kv_stage::PrefillLaunchBounds;

    use super::*;

    use super::super::tests_fixtures::tests::{
        gated_partial_rotary, hyper_moe, hyper_moe_full, qwen36_35b_a3b_with_head,
    };

    /// The draft head's input assembly at 100 rows on an int8 session, carve by
    /// carve. On its attention phase: the shift, the hidden norm and the sum
    /// (each 100 × 10240 × 4 = 4,096,000), the hidden operand (one row per
    /// stream: 8,000 tiles → 1,000 blocks × 1,152 = 1,152,000) and the hidden
    /// projection (400 × 2560 × 4 = 4,096,000) — 17,536,000 — and, on the same
    /// phase, the embedding norm and its projection (each 100 × 2560 × 4 =
    /// 1,024,000) and the embedding operand (2,000 tiles → 250 blocks × 1,152 =
    /// 288,000). Every carve starts on a 256 boundary, so the walk is the sum.
    #[test]
    fn the_head_pass_prices_its_assembly() {
        let g = hyper_moe_full();
        let p = WavePlan::new(g);
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(p.chain_bytes(Chain::HeadPass, wide), 19_872_000);
        for b in [
            WaveBuffer::HeadEmbedNorm,
            WaveBuffer::HeadEmbedOperand,
            WaveBuffer::HeadEmbedProj,
        ] {
            assert_eq!(b.phase(), LayerPhase::Attention, "{b:?}");
        }
        assert_eq!(
            WavePlan::new(hyper_moe()).chain_bytes(Chain::HeadPass, wide),
            0
        );
        let embed: usize = [
            WaveBuffer::HeadEmbedNorm,
            WaveBuffer::HeadEmbedOperand,
            WaveBuffer::HeadEmbedProj,
        ]
        .iter()
        .map(|b| b.bytes(&g, wide))
        .sum();
        assert_eq!(embed, 2_336_000);
        // Only where the head runs.
        let window = WaveWidth {
            prefill_rows: 100,
            ..WaveWidth::default()
        };
        assert_eq!(WaveBuffer::HeadEmbedNorm.bytes(&g, window), 0);
    }

    /// The head's block runs behind its assembly: the pass is the assembly,
    /// then the pre-mix, then the attention chain — an alternative the phase
    /// takes the max with, never a sum over the trunk's mixers.
    #[test]
    fn the_head_pass_walks_its_assembly_then_the_trunk_attention_layer() {
        let p = WavePlan::new(hyper_moe_full());
        let wide = WaveWidth::prefill(100, 1);
        let head = p.walk_chain(
            p.walk_chain(19_872_000, Chain::HyperAttn, wide),
            Chain::Attention,
            wide,
        );
        let trunk = p
            .walk_chain(10_731_776, Chain::Attention, wide)
            .max(p.walk_chain(10_731_776, Chain::DeltaNet, wide));
        assert_eq!(p.phase_bytes(LayerPhase::Attention, wide), head.max(trunk));
        assert!(head > p.walk_chain(10_731_776, Chain::Attention, wide));
    }

    /// A single-stream NextN head's pass over 100 rows of 1024 BF16: five
    /// `100 × 1024 × 2` buffers, the `2 · hidden` concatenation and — on this
    /// int8 session — its q8a128 operand for `eh_proj` (`100 × 2048` is 1,600
    /// tiles, 200 super-blocks) on the head's attention phase, the row
    /// embeddings on the forward phase — and none of it on a wave that stops
    /// short of the head, or on a stack without one. A float session's
    /// `eh_proj` reads the concatenation as it stands and carves no operand.
    #[test]
    fn a_single_stream_draft_head_prices_its_pass() {
        let g = ModelGeometry {
            mtp_head: true,
            ..gated_partial_rotary()
        };
        let wide = WaveWidth::prefill(100, 1);
        assert_eq!(WaveBuffer::MtpRowEmbeds.bytes(&g, wide), 204_800);
        assert_eq!(WaveBuffer::MtpInputCat.bytes(&g, wide), 409_600);
        assert_eq!(WaveBuffer::MtpInputOperand.bytes(&g, wide), 230_400);
        let p = WavePlan::new(g);
        assert_eq!(
            p.chain_bytes(Chain::HeadPass, wide),
            5 * 204_800 + 409_600 + 230_400
        );
        let float = ModelGeometry {
            packed_norm: false,
            ..g
        };
        assert_eq!(WaveBuffer::MtpInputOperand.bytes(&float, wide), 0);
        let window = WaveWidth {
            prefill_rows: 100,
            ..WaveWidth::default()
        };
        assert_eq!(p.chain_bytes(Chain::HeadPass, window), 0);
        assert_eq!(WaveBuffer::MtpRowEmbeds.bytes(&g, window), 0);
        let headless = WavePlan::new(gated_partial_rotary());
        assert_eq!(headless.chain_bytes(Chain::HeadPass, wide), 0);
    }

    /// **The head pass, walked as it carves, fits the attention phase it runs
    /// in — including where it is the chain that sets that phase's price.**
    ///
    /// The daemon's failure, reproduced in the plan: two deep ingests of 511
    /// and 510 rows at 100,000 positions each, so each stages a full key chunk
    /// and the prefill launch's stage and carry make the head's attention block
    /// out-price every DeltaNet layer. The head pass then IS the attention
    /// phase's price, and every byte it carves beyond its declarations comes
    /// out of the last carves' room. `eh_proj`'s q8a128 operand was such a
    /// byte-for-byte omission — `[1021, 4096]`, the very size the pass's
    /// `o_proj` operand asks for second-to-last — so the live span ran out on
    /// exactly that request: `4704768 B asked … Asked by: QMatMul::forward_live_as
    /// <- output_projection <- forward_attn_batched <- head_wave_pass`.
    ///
    /// The assembly is listed here in `head_wave_pass`'s carve order from its
    /// own arithmetic, not read back from the plan: the trunk's final norm over
    /// every row, the shift, `enorm`, `hnorm`, the concatenation, the operand
    /// `eh_proj` quantizes it into, and `eh_proj`'s result. The block behind it
    /// is a trunk attention layer, whose chain the census tests pin.
    #[test]
    fn the_head_pass_fits_the_attention_phase_it_prices() {
        let g = qwen36_35b_a3b_with_head();
        let p = WavePlan::new(g);
        let q_lens = [511usize, 510];
        let offsets = [100_000usize, 100_000];
        let w = WaveWidth {
            kv_stage: PrefillKvStageLayout::new(
                &q_lens,
                &offsets,
                g.n_head,
                g.n_kv_head,
                g.head_dim,
                PrefillLaunchBounds::PRODUCTION,
            ),
            ..WaveWidth::prefill(1021, 2)
        };

        // `1021 × 4096` is 32,672 tiles, 4,084 super-blocks of 1,152 B: the
        // daemon's ask, to the byte.
        assert_eq!(WaveBuffer::MtpInputOperand.bytes(&g, w), 4_704_768);

        let hidden_row = 1021 * 2048 * 2;
        let assembly = [
            hidden_row,     // final_norm over every row
            hidden_row,     // the shift
            hidden_row,     // enorm
            hidden_row,     // hnorm
            2 * hidden_row, // [enorm ; hnorm]
            4_704_768,      // its q8a128 operand for eh_proj
            hidden_row,     // eh_proj
        ];
        let assembled = assembly.iter().fold(0usize, |cursor, &len| {
            cursor.div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT + len
        });
        assert_eq!(assembled, 33_978_880);
        assert_eq!(
            p.chain_bytes(Chain::HeadPass, w),
            assembled,
            "the plan's head-pass assembly is not the one head_wave_pass carves"
        );

        let head_pass = p.walk_chain(assembled, Chain::Attention, w);
        let delta_net = p.chain_bytes(Chain::DeltaNet, w);
        assert!(
            head_pass > delta_net,
            "this width must make the head pass the phase's binding chain \
             ({head_pass} B against the DeltaNet layer's {delta_net} B), or the \
             assertion below is answered by the DeltaNet layer's slack"
        );
        assert_eq!(
            p.phase_bytes(LayerPhase::Attention, w),
            head_pass,
            "the attention phase is not priced at what the head pass carves"
        );
    }

    /// **The head is two different chains, and the session's mode picks one.**
    ///
    /// Both measured. An **int8** session carves a packed norm and its logits:
    /// Qwen3.5-0.8B at 4 scored rows, `4 × 1,152 + 4 × 248,320 × 2`. A **float**
    /// session carves a dense norm and its F32 working copy — Qwen2 at 60
    /// scored rows, hidden 896: `60 × 896 × 2 + 60 × 896 × 4` = 322,560 B — and
    /// its logits beside them: the dequantized-weight path's `to_owned_tensor`
    /// inherits the operand's arena, and Qwen2's gate ran out of span one
    /// `151,936 × 4` row into them before they were priced.
    #[test]
    fn the_head_is_priced_for_the_session_it_runs_in() {
        let int8 = gated_partial_rotary();
        assert!(int8.packed_norm);
        let plan = WavePlan::new(int8);
        let w = WaveWidth::prefill(2100, 4);
        assert_eq!(WaveBuffer::HeadNormF32.bytes(&int8, w), 0);
        assert_eq!(WaveBuffer::HeadLogits.bytes(&int8, w), 4 * 248_320 * 2);
        assert_eq!(WaveBuffer::HeadLogitsF32.bytes(&int8, w), 0);
        // The head's two carves and the gather of its four scored rows.
        assert_eq!(plan.chain_bytes(Chain::Forward, w), 1_991_168 + 8_208);
        // An in-place accept walk's selections sit beside the logits, the same
        // width again — only for a caller that reads them there.
        assert_eq!(WaveBuffer::AcceptRows.bytes(&int8, w), 0);
        let walked = WaveWidth {
            accept_rows: 4,
            ..w
        };
        assert_eq!(WaveBuffer::AcceptRows.bytes(&int8, walked), 4 * 248_320 * 2);

        // Qwen2's shapes, on the float path its gate actually runs — and note
        // `packed_norm` stays **true**: its layers are packed and its head is
        // not, so the head follows `packed_head` alone.
        let float = ModelGeometry {
            packed_head: false,
            hidden: 896,
            vocab: 151_936,
            act_dtype: DType::F16,
            ..int8
        };
        assert!(float.packed_norm, "the layers stay packed");
        let w60 = WaveWidth::prefill(4096, 60);
        assert_eq!(
            WaveBuffer::HeadLogits.bytes(&float, w60),
            60 * 151_936 * 2,
            "a float session's logits are on the span too"
        );
        assert_eq!(
            WaveBuffer::AcceptRows.bytes(
                &float,
                WaveWidth {
                    accept_rows: 60,
                    ..w60
                }
            ),
            60 * 151_936 * 2,
            "and so are the walk's selections from them"
        );
        assert_eq!(WaveBuffer::HeadNorm.bytes(&float, w60), 60 * 896 * 2);
        assert_eq!(WaveBuffer::HeadNormF32.bytes(&float, w60), 60 * 896 * 4);
        // The matmul writes F32 and the result is narrowed to F16 after it.
        assert_eq!(
            WaveBuffer::HeadLogitsF32.bytes(&float, w60),
            60 * 151_936 * 4
        );
        assert_eq!(
            WavePlan::new(float).chain_bytes(Chain::Forward, w60),
            322_560 + 60 * 151_936 * (2 + 4) + 60 * 896 * 2 + 60 * 4,
            "the norm's two carves, the logits in both widths and the scored-row \
             gather, to the byte"
        );
        // An F32 session's matmul result is its logits: no second copy.
        let float32 = ModelGeometry {
            act_dtype: DType::F32,
            ..float
        };
        assert_eq!(WaveBuffer::HeadLogitsF32.bytes(&float32, w60), 0);
    }
}
