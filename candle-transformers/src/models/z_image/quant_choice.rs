//! Which Z-Image checkpoint a card runs, and where the text encoder comes from.
//!
//! The single place the quantisation is decided, so the file fetched and the
//! file opened cannot disagree.
//!
//! # Why every card runs a quantised transformer
//!
//! Not because the unquantised release does not fit — on a 72 GB card it would
//! fit several times over — but because on this fork the quantised weight is the
//! *faster* one. Each projection is repacked at load into its KO twin and run
//! through the q8a128 int8 tensor-core matmul (see
//! [`crate::models::quantized_matmul::QMatMul`]), and `Int8Mode::Precision` maps
//! Q8_0 → Q8_KO and Q6_K → Q6_KO — same width, no step down. So the choice is
//! not "quality against memory": at Q8 it is a near-lossless weight running on a
//! kernel the bf16 release cannot reach. There is no rung above Q8 worth having.
//!
//! # Why a ladder at all, then
//!
//! Z-Image's transformer is dense: all ~6B parameters are read on every one of
//! the eight steps, so the whole thing has to be resident and the file size *is*
//! the requirement. That is the opposite of a MoE language model, where the
//! resident set is the dense weights plus whatever experts fit and a bigger card
//! buys speed rather than feasibility. A 16 GB card takes Q6 for that reason
//! alone.
//!
//! # What else is on the card
//!
//! The transformer is not the only tenant, but it is the only one that shares
//! the card with itself: the Qwen3-4B text encoder runs first and is **dropped**
//! before the transformer is built, so the peak is the larger of the two phases
//! rather than their sum. What sits beside the transformer is the autoencoder
//! (~0.3 GB) and a 1024×1024 pass's activations.

use candle::quantized::GgmlDType;

/// A published Z-Image-Turbo transformer checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZQuant {
    /// ~5.5 GiB. The rung for a 16 GB card.
    Q6K,
    /// ~6.7 GiB. Everything from 18 GB up.
    Q8_0,
}

/// At or above this much total VRAM, Q8; below it, Q6.
pub const Q8_MIN_TOTAL_VRAM_BYTES: u64 = 18 * (1 << 30);

/// The prompt encoder — Qwen3-4B, the same weights as the release's
/// `text_encoder/`, at Q8_0.
///
/// One rung rather than a ladder. It is ~4 GB, it is loaded and dropped before
/// the transformer exists, and so it is never what a card runs out of room for;
/// a second rung would be a choice with nothing to trade.
///
/// A *different* publisher to the transformer, because there is no Z-Image text
/// encoder in [`ZQuant::repo`]. That costs nothing here: the "one conversion"
/// argument on [`ZQuant::repo`] is about two rungs of one model being comparable
/// to each other, and this model has one rung and nothing to compare it to.
pub const TEXT_ENCODER_REPO: &str = "gguf-org/z-image-gguf";

/// The file within [`TEXT_ENCODER_REPO`].
///
/// It carries *transformers* tensor names (`model.layers.0.self_attn.q_proj`),
/// not llama.cpp's (`blk.0.attn_q`) — which is why it loads through
/// [`super::text_encoder`] rather than [`crate::models::quantized_qwen3`].
pub const TEXT_ENCODER_FILE: &str = "qwen3_4b_f32-q8_0.gguf";

/// Its published length, for a downloader's progress total.
pub const TEXT_ENCODER_BYTES: u64 = 4_274_478_528;

impl ZQuant {
    /// The rung for a card with `total_vram_bytes`.
    ///
    /// Total rather than free: this decides which checkpoint a deployment runs,
    /// and free memory is a reading of the moment that would have the same
    /// machine pick a different model depending on what happened to be resident
    /// when it started.
    pub fn for_vram(total_vram_bytes: u64) -> Self {
        if total_vram_bytes >= Q8_MIN_TOTAL_VRAM_BYTES {
            Self::Q8_0
        } else {
            Self::Q6K
        }
    }

    /// The repository publishing it.
    ///
    /// Both rungs come from **one** publisher on purpose. Several repos carry
    /// Z-Image GGUFs and their conversions differ, so taking Q6 from one and Q8
    /// from another would make the ladder a change of *conversion* as well as of
    /// precision, and a quality difference between rungs would have two possible
    /// causes.
    pub fn repo(&self) -> &'static str {
        "jayn7/Z-Image-Turbo-GGUF"
    }

    /// The file within it.
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Q6K => "z_image_turbo-Q6_K.gguf",
            Self::Q8_0 => "z_image_turbo-Q8_0.gguf",
        }
    }

    /// The published length, for a downloader's progress total.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Q6K => 5_910_505_536,
            Self::Q8_0 => 7_224_707_136,
        }
    }

    /// The format the file's projections are stored in.
    pub fn dtype(&self) -> GgmlDType {
        match self {
            Self::Q6K => GgmlDType::Q6_K,
            Self::Q8_0 => GgmlDType::Q8_0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::Int8Mode;

    /// The one threshold, from both sides and exactly on it.
    #[test]
    fn the_ladder_turns_at_eighteen() {
        let gb = |n: u64| n * (1 << 30);
        // The fleet, by the numbers in CLAUDE.md.
        assert_eq!(ZQuant::for_vram(gb(16)), ZQuant::Q6K, "4090 Mobile");
        assert_eq!(ZQuant::for_vram(gb(24)), ZQuant::Q8_0, "3090");
        assert_eq!(ZQuant::for_vram(gb(72)), ZQuant::Q8_0, "PRO 5000");

        assert_eq!(ZQuant::for_vram(Q8_MIN_TOTAL_VRAM_BYTES), ZQuant::Q8_0);
        assert_eq!(ZQuant::for_vram(Q8_MIN_TOTAL_VRAM_BYTES - 1), ZQuant::Q6K);
    }

    /// **A rung must leave room for the rest of the pass.** Holding the file is
    /// not enough: the autoencoder and a 1024×1024 pass's activations sit beside
    /// it, and that margin is what makes the threshold a threshold rather than a
    /// restatement of a file size.
    #[test]
    fn every_rung_fits_its_smallest_card() {
        // The autoencoder plus a 1024×1024 pass — 4,096 latent patches and a
        // caption through 34 blocks.
        const REST_OF_THE_PASS: u64 = 4 * (1 << 30);
        for (q, floor) in [
            (ZQuant::Q6K, 16 * (1u64 << 30)),
            (ZQuant::Q8_0, Q8_MIN_TOTAL_VRAM_BYTES),
        ] {
            assert_eq!(ZQuant::for_vram(floor), q, "the ladder moved");
            assert!(
                q.bytes() + REST_OF_THE_PASS < floor,
                "{q:?} needs {:.1} GiB resident against a {:.1} GiB floor",
                q.bytes() as f64 / (1u64 << 30) as f64,
                floor as f64 / (1u64 << 30) as f64,
            );
        }
    }

    /// The text encoder runs before the transformer exists, so what has to fit
    /// beside it is nothing — but it still has to fit the smallest card on its
    /// own, and that is the claim the "one rung" comment rests on.
    #[test]
    fn the_text_encoder_fits_the_smallest_card_alone() {
        // The Q8_0 weights, plus the embedding table dequantised to f32 for the
        // lookup (151,936 × 2,560), plus the prompt's own activations.
        const EMBED_F32: u64 = 151_936 * 2_560 * 4;
        const RESIDENT: u64 = TEXT_ENCODER_BYTES + EMBED_F32 + (1 << 30);
        const { assert!(RESIDENT < 16 * (1u64 << 30)) };
    }

    /// **Both rungs keep their width under the int8 twin.** This is the claim
    /// the module header makes — that quantising is not a concession here — and
    /// it holds only because Q6_K and Q8_0 are at the top of `to_ko`'s ladder.
    /// A rung whose twin stepped *down* would be trading quality for the kernel.
    #[test]
    fn neither_rung_loses_width_to_its_ko_twin() {
        for (q, twin) in [
            (ZQuant::Q6K, GgmlDType::Q6_KO),
            (ZQuant::Q8_0, GgmlDType::Q8_KO),
        ] {
            for mode in [Int8Mode::Performance, Int8Mode::Precision] {
                assert_eq!(q.dtype().to_ko(mode).unwrap(), twin, "{q:?} under {mode:?}");
            }
        }
    }
}
